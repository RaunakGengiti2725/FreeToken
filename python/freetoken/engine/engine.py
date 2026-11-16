from __future__ import annotations

import collections
import contextlib
import copy
import errno
import gc
import math
import os
from datetime import timedelta
from typing import Any, Dict, Iterable, NamedTuple, Tuple

import torch
from freetoken.attention import AttnType, attention_backend_info, create_attention_backend
from freetoken.attention.linear import ChunkCaptureLinearBackend, LinearAttnBackend, WideVerifyLinearBackend, gdn_slot
from freetoken.core import DEFAULT_MAX_OUTPUT_TOKENS, Batch, Context, Req, set_global_ctx
from freetoken.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from freetoken.gpu_select import gpu_identity
from freetoken.checkpoint.ftw import ftw_tensor_names, is_ftw_checkpoint
from freetoken.layers import OffloadMoELayer, ParallelLMHead, VocabParallelEmbedding, iter_moe_layers, iter_ops, set_rope_device
from freetoken.layers.quantization import LayerKind, QuantBackend, QuantKind, finalize_quant, set_quant_backend
from freetoken.layers.quantization.names import name_set
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.mm.config import ENCODER_SECTIONS
from freetoken.models import create_model, load_weight
from freetoken.models.weight import ftw_lacks_vision
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.expert_banks import attach_resident_banks, load_expert_banks, pin_probe_enabled
from freetoken.moe.host_banks import PinFailed
from freetoken.moe.locked_fetch import attach_locked_fetch, check_locked_fetch_config, locked_pool_tags, settle_banks, staging_reserve
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache, bank_layouts
from freetoken.moe.pool_policy import group_layers, make_pools, pool_of_layer, resolve_profile, split_slots
from freetoken.utils import align_ceil, init_logger, is_sm90_family, is_sm100_family, mem_GB, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, _determine_cuda_graph_bs, get_free_memory
from .sample import BatchSamplingArgs, Sampler
from .spec_protocol import SpecRuntime
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.format import NONE, KVFormat
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.cache_status import _supports_swa_ratio
from freetoken.kvcache.linear_state_pool import (
    _linear_local_dims, _linear_pool_min_slots, _linear_pool_num_slots, state_pool_bytes,
)

logger = init_logger(__name__)

def _disable_unsafe_cpu_moe_graphs(config: EngineConfig, executor, dtype, tp_cpu_group) -> None:
    safe = torch.tensor([executor.graph_capture_safe(dtype)], dtype=torch.int32)
    if config.tp_info.size > 1:
        torch.distributed.all_reduce(safe, op=torch.distributed.ReduceOp.MIN, group=tp_cpu_group)
    if not safe.item():
        logger.warning_rank0("CPU MoE IO/handshake is not graph-safe on every TP rank for this dtype; disabling decode graphs")
        # Clear explicit sizes too; GraphRunner gives them priority over max_bs, including on rebuild.
        object.__setattr__(config, "cuda_graph_bs", [])
        object.__setattr__(config, "cuda_graph_max_bs", 0)


def _clamp_decode_batch_to_pools(config: EngineConfig, pool_sizes: list[int]) -> None:
    """Clamp the decode batch so every GPU-decoded layer's routed experts fit in its pool."""
    num_experts = config.model_config.num_experts
    small = [s for s in pool_sizes if s < num_experts]
    if not small:
        return
    # a decode step routes every row of every request, decode_rows_per_req rows each
    limit = min(small) // (config.model_config.num_experts_per_tok * config.decode_rows_per_req)
    if config.max_running_req > limit:
        logger.warning_rank0(
            f"the smallest MoE slot pool has {min(small)} slots; lowering max_running_req "
            f"{config.max_running_req} -> {limit} so a decode step fits in it"
        )
        object.__setattr__(config, "max_running_req", limit)
    if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs > limit:
        object.__setattr__(config, "cuda_graph_max_bs", limit)
    if config.cuda_graph_bs is not None:
        object.__setattr__(config, "cuda_graph_bs", [bs for bs in config.cuda_graph_bs if bs <= limit])


def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
    """The offload MoE cache needs at least one slot per expert per layer. A too-small size
    (e.g. a bare offload run with moe_cache_size unset and auto disabled) must fail loudly."""
    if cache_size < num_experts:
        raise ValueError(
            f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
            f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
            f"--moe-strategy cpu always sizes its own fixed two-layer buffer and ignores "
            f"cache-sizing flags)."
        )


def _flashinfer_available() -> bool:
    from freetoken.kernel.backend import is_flashinfer_installed

    return is_flashinfer_installed()


def _sgl_flash_attn_available() -> bool:
    from freetoken.kernel.backend import is_rocm

    if is_rocm():
        return False
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        logger.warning_rank0(
            "sgl_kernel.flash_attn is unavailable; auto attention backend falls back to fi "
            f"({type(exc).__name__}: {detail})"
        )
        return False
    return True


def _startup_kv_budget(memory_ratio: float, init_free_memory: int, new_free_memory: int) -> int:
    """Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus
    what the resident model consumed. Kept as a pure function so the composition with the
    pool families' ``solve_num_pages`` stays CPU-testable."""
    return int(memory_ratio * init_free_memory) - (init_free_memory - new_free_memory)


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    """Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the
    highest column touched is ``align_ceil(max_seq_len, page_size) - 1`` -- which the 32-alignment
    alone does not cover once page_size > 32 (an unaligned --max-seq-len-override on DSV4's P=128
    or trtllm's forced 64 would index past the row)."""
    return align_ceil(align_ceil(max_seq_len, page_size), 32)


def _seq_limits(config, num_tokens: int) -> tuple[int, int]:
    """(longest request, page-table width). A decode step may widen a request by decode_rows_per_req - 1 provisional rows past its length, so requests get that much less and the table still covers the widened tail."""
    limit = min(config.max_seq_len, num_tokens)
    return limit - (config.decode_rows_per_req - 1), _page_table_width(limit, config.page_size)


def _required_attn_types(model_config) -> frozenset[AttnType]:
    """Backend-driving attention types of this model, from the group-spec walk
    (single source shared with the pool factory and the KV cost model). getattr
    fallbacks: duck-typed test configs may not implement the spec walk; for those,
    dsv4_args marks DSV4 (the real config declares a DSV4 attention group)."""
    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            return frozenset({AttnType.DSV4})
        return frozenset({AttnType.FULL})
    types = frozenset(
        spec.attn_type for spec in specs_fn() if spec.attn_type.backend_driven
    )
    return types or frozenset({AttnType.FULL})


def _backend_parts_serve(name: str, required: frozenset[AttnType], kv_format: KVFormat = NONE) -> bool:
    return all(
        required <= attention_backend_info(part).supported_types and kv_format.name in attention_backend_info(part).kv_formats
        for part in name.split(",")
    )


def _backend_requirements_met(name: str) -> bool:
    # flashinfer first across ALL parts: the sgl probe logs a "falls back to fi" warning,
    # which would mislead when the candidate is about to fail on flashinfer anyway.
    infos = [attention_backend_info(part) for part in name.split(",")]
    if any(i.requires_flashinfer for i in infos) and not _flashinfer_available():
        return False
    if any(i.requires_sgl_kernel for i in infos) and not _sgl_flash_attn_available():
        return False
    if any(i.requires_sm100 for i in infos) and not is_sm100_family():
        return False
    return True


def _resolve_auto_attention_backend(required: frozenset[AttnType], kv_format: KVFormat = NONE) -> str:
    """First candidate (in per-type priority order) whose arch condition holds,
    whose packages are installed, and whose every comma part serves ALL required
    types. Reproduces the historical hardware tree for FULL-only models:
    sm_100 -> trtllm, sm_90+sgl_kernel -> "fa,fi", flashinfer -> fi, else triton."""
    candidates: list[tuple[str, bool]] = []
    if AttnType.DSV4 in required:
        candidates.append(("dsv4_sparse", True))
    if AttnType.DSV41 in required:
        candidates.append(("dsv41_sparse", True))
    if required & {AttnType.MLA, AttnType.DSA}:
        candidates.append(("dsa", True))
    if AttnType.BSA in required:
        candidates.append(("m3_sparse", True))
    if AttnType.QSA in required:
        candidates.append(("qsa_sparse", True))
    if AttnType.SWA in required:
        candidates.append(("triton", True))
    if AttnType.FULL in required:
        candidates += [
            ("trtllm", is_sm100_family()),
            ("fa,fi", is_sm90_family()),
            ("fi", True),
            ("triton", True),
        ]
    for name, arch_ok in candidates:
        if not arch_ok:
            continue
        if not _backend_parts_serve(name, required, kv_format):
            continue
        if not _backend_requirements_met(name):
            continue
        return name
    raise RuntimeError(
        "No attention backend can serve attention types "
        f"{sorted(t.value for t in required)} with KV format {kv_format.name} on this machine."
    )


def _spec_runner():
    from freetoken.engine.spec_protocol import SPEC_RUNNER
    from freetoken.models.register import _load_attr

    return _load_attr(*SPEC_RUNNER.split(":"))


def _adjust_spec_config(config, override, windowed_kv: bool) -> None:
    """Resolve k and the method's own limits, then check what every speculative method needs of the target."""
    runner = _spec_runner()
    runner.adjust_config(config, override)
    override("spec_slot_states", runner.slot_states(config))
    if windowed_kv and not runner.target_may_be_windowed(config):
        raise ValueError(
            "speculative decoding does not support targets with sliding-window or DSV4 window-tier KV yet: "
            "their window frees would release slots a verify step still reads"
        )
    # at config time: the state pool that holds the handlers is built only after the weights load
    if unverified := [s.name for s in config.model_config.slot_states if s.verify is None]:
        raise ValueError(f"speculative decoding does not support per-request state {unverified} yet: it declares no verify handler")
    parts = [p.strip() for p in config.attention_backend.split(",")]
    graph = attention_backend_info(parts[-1]).uniform_extend_graph
    if graph == "host_plan" and len(parts) == 2 and attention_backend_info(parts[0]).uniform_extend_graph == "device":
        # verify steps run on the decode part; a prefill part that captures on device serves them better
        logger.info_rank0(f"speculative decoding: attention backend {config.attention_backend} -> {parts[0]}")
        override("attention_backend", parts[0])
    elif graph == "none":
        raise ValueError(
            f"speculative decoding needs a decode attention backend that captures multi-row graphs "
            f"(fa, triton or fi), got {config.attention_backend!r}"
        )


def _drop_preset_fields(config, *names: str) -> None:
    object.__setattr__(config, "preset_fields", tuple(name for name in config.preset_fields if name not in names))


# --preset single-fast: its prefill graphs took 2.1 GiB on the one checkpoint and card they were measured on (70 sizes,
# Qwen3.6-35B-A3B NVFP4, RTX 5090); the margin on top is unmeasured
PRESET_CAPTURE_BYTES = 2304 << 20


def _preset_capture_bytes(config) -> int:
    """GPU bytes a preset leaves free for the prefill graphs it asked for; a --prefill-graph-chunks from the command line gets none."""
    return PRESET_CAPTURE_BYTES if "prefill_graph_chunks" in getattr(config, "preset_fields", ()) else 0


def _drop_preset_captures(config, reason: str) -> None:
    """Serve a preset without the prefill graphs it asked for, with one log line: what is left is --preset single."""
    filled = [name for name in ("prefill_graph_tokens", "prefill_graph_chunks") if name in config.preset_fields]
    for name in filled:
        object.__setattr__(config, name, getattr(EngineConfig, name))
    _drop_preset_fields(config, *filled)
    logger.info_rank0(f"--preset {config.preset}: serving as --preset single, without the captured prefill forwards ({reason})")


# --preset: the drafts it turns on are served only where an offload expert cache holds this share of all experts or
# more. Chosen from four decode cells of Qwen3.6-35B-A3B NVFP4 on RTX 5090s, the preset with 3 drafts against a
# server with no flags: under it at 512 and 701 of 10240 slots (5.0% and 6.8%), over it at 2048 and 2991 (20.0%
# and 29.2%). No cell lies between 6.8% and 20%, and the share is the lowest at which the drafts were seen to win.
# unmeasured: the answer a preset's KV pool still holds after a prompt as long as a prefill chunk, where the pool is
# lowered to make room for the chunk's working memory. A pool of exactly the chunk refuses a prompt of that length.
PRESET_CHUNK_ANSWER_TOKENS = 1024
PRESET_DRAFT_SHARE = 0.20
# the log line of a start that drops them; moe/calibrate.py reads it from the server of its static candidate
PRESET_NO_DRAFTS = "serving without the preset's draft tokens"


def _drop_preset_demand(config, reason: str) -> None:
    """Serve a preset without the short prefill chunks it loads on demand, with one log line: they stream as longer ones do."""
    if "moe_prefill_demand_tokens" not in getattr(config, "preset_fields", ()):
        return
    object.__setattr__(config, "moe_prefill_demand_tokens", EngineConfig.moe_prefill_demand_tokens)
    _drop_preset_fields(config, "moe_prefill_demand_tokens")
    logger.info_rank0(f"--preset {config.preset}: no prefill chunk loads its experts on demand ({reason})")


def _settle_preset_demand(config, decode_target: str, pool_sizes: list[int], cache=None) -> None:
    """A preset's short prefill chunks load their experts on demand only through a GPU-decoded expert cache that cannot
    hold every expert. Called before the cache is built, which refuses them on another decode target, and with the cache."""
    if decode_target != "gpu":
        _drop_preset_demand(config, f"the expert cache decodes on {decode_target}")
    elif min(pool_sizes) < config.model_config.num_experts:
        _drop_preset_demand(config, f"a slot pool of {min(pool_sizes)} slots cannot hold the experts a chunk routes to in one layer")
    elif cache is not None and cache.keeps_streamed_experts and cache.holds_every_expert:
        # a prefill reads the slots once they are filled, and until then it fills them by streaming
        _drop_preset_demand(config, "the expert cache has a slot for every expert")
    if cache is not None:
        cache.prefill_demand_tokens = config.moe_prefill_demand_tokens


def _preset_kv_target(config) -> int:
    """The KV reserve a preset asks for, at most the model's context (no request holds more) and at least the default reserve."""
    return max(min(config.kv_reserve_tokens, config.max_seq_len), EngineConfig.kv_reserve_tokens)


def _prefill_working_bytes(config) -> tuple[int, int]:
    """(tokens, bytes): the longest prefill chunk (--max-prefill-length, at most the context) and the GPU memory its
    temporaries hold at their peak on ``config``'s model, per rank (cache_budget.prefill_working_bytes). The bytes
    are 0 for a model whose prefill is not counted there, which is any model without GatedDeltaNet layers."""
    from freetoken.engine.cache_budget import prefill_working_bytes

    model = config.model_config
    tokens = min(config.max_forward_len, config.max_seq_len)
    group = model.linear_attention_group()
    if group is None or group.variant != "gdn":
        return tokens, 0
    from freetoken.kernel.fla.chunk import CHUNK_SIZE

    _, _, value_heads, key_heads = _linear_local_dims(group, config.tp_info.size)
    return tokens, prefill_working_bytes(
        tokens, hidden=model.hidden_size, key_dim=key_heads * group.key_head_dim,
        value_dim=value_heads * group.value_head_dim, value_heads=value_heads, head_dim=group.value_head_dim,
        chunk=CHUNK_SIZE, itemsize=config.dtype.itemsize,
    )


def _checkpoint_bytes(path: str) -> int | None:
    """Bytes of a checkpoint's weight files on disk; None where ``path`` is neither a weight file nor a folder that holds some."""
    from freetoken.moe.decode_profile import _WEIGHT_FILES

    path = os.path.realpath(os.path.expanduser(path))
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if not os.path.isdir(path):
            return None
        files = [os.path.join(path, name) for name in os.listdir(path) if name.endswith(_WEIGHT_FILES)]
        return sum(os.path.getsize(f) for f in files if os.path.isfile(f)) or None
    except OSError:
        return None


def _preset_resident_need(config) -> int | None:
    """GPU bytes a preset's server needs with every expert resident ('fused'): the checkpoint's weight files, the
    preset's KV reserve with the draft's share, the linear states and one prefill chunk's working memory. None where
    one of them cannot be counted before the load, or the command line asks for an expert cache."""
    model = config.model_config
    sized = config.moe_cache_size > 0 or config.moe_cache_rate or config.moe_cache_mem is not None or config.moe_cache_auto
    if sized or config.moe_cpu_layers or config.tp_info.size > 1 or not _fused_resident_ok(model):
        return None
    weights = _checkpoint_bytes(config.model_path)
    working = _prefill_working_bytes(config)[1]
    if weights is None or not working:
        return None
    if config.speculative_draft_model_path:
        draft = _checkpoint_bytes(config.speculative_draft_model_path)
        if draft is None:
            return None
        weights += draft
    page_bytes, fixed, page_tokens, _ = resolve_pool_class(model).kv_cost(config)
    if config.speculative_algorithm is not None:
        spec_fixed, draft_token = _spec_runner().budget(config)
        fixed += spec_fixed
        page_bytes += draft_token * page_tokens
    pages = -(-_preset_kv_target(config) // page_tokens)
    # a KV pool the command line sized is allocated at that size, whatever is free
    if config.num_token_override is not None:
        pages = -(-config.num_token_override // page_tokens)
    elif config.num_page_override is not None:
        pages = config.num_page_override
    return weights + pages * page_bytes + fixed + state_pool_bytes(config) + working


def _preset_takes_fused(config) -> bool:
    """Whether a preset the command line left the strategy to serves every expert resident: what that needs
    (_preset_resident_need) is within the memory ratio's share of the free GPU memory. A wrong yes is an
    out-of-memory error while the weights load, so what cannot be counted is a no."""
    if "moe_strategy" not in getattr(config, "preset_fields", ()):
        return False
    need, free = _preset_resident_need(config), _gpu_free_bytes()
    if need is None or free is None:
        return False
    allowed, captures = int(config.memory_ratio * free), _preset_capture_bytes(config)
    if captures and need <= allowed < need + captures:
        _drop_preset_captures(config, f"the experts fit on the GPU, but not beside the {mem_GB(captures)} the prefill graphs take")
        captures = 0
    fits = need + captures <= allowed
    logger.info_rank0(
        f"--preset {config.preset}: every expert resident needs {mem_GB(need + captures)} of the {mem_GB(allowed)} "
        f"the memory ratio allows: {'fused' if fits else 'the expert cache'}"
    )
    return fits


def _adjust_preset_spec_config(config, override, windowed_kv: bool) -> None:
    """Speculative decoding a preset turned on: kept where an explicit start's checks pass and the draft's source is
    on disk to be checked, dropped with one log line elsewhere."""
    path = config.speculative_draft_model_path or config.model_path
    if not os.path.exists(path):
        reason = f"{path} is not on disk, so its draft layers cannot be checked before the load"
    else:
        try:
            _adjust_spec_config(config, override, windowed_kv)
            return
        except (ValueError, KeyError, OSError) as exc:
            reason = str(exc)
    logger.info_rank0(f"--preset {config.preset}: no speculative decoding ({reason})")
    override("speculative_algorithm", None)
    if "speculative_num_steps" in config.preset_fields:
        override("speculative_num_steps", 0)
    override("spec_slot_states", ())
    _drop_preset_fields(config, "speculative_algorithm", "speculative_num_steps")


def _validate_kv_format(config, model_config) -> None:
    kv_format = getattr(config, "kv_format", NONE)
    if not kv_format.quantized:
        return
    pool_cls = resolve_pool_class(model_config)
    if kv_format.name not in pool_cls.kv_formats:
        raise ValueError(f"--kv-cache-dtype {kv_format.name}: {pool_cls.__name__} only stores {sorted(pool_cls.kv_formats)}")
    if config.dtype != torch.bfloat16:
        raise ValueError(f"--kv-cache-dtype {kv_format.name} needs --dtype bfloat16, got {config.dtype}")
    if config.tp_info.size != 1:
        raise ValueError(f"--kv-cache-dtype {kv_format.name} does not support tensor parallelism yet")


def _validate_attention_backend_choice(config, override, required: frozenset[AttnType]) -> None:
    """Config-time type x backend capability check for the resolved (or explicit)
    backend string: every comma part must serve every required type and have its
    packages/arch available. Replaces the per-model gates; in particular this is
    where a DSV4 or MLA checkpoint rejects a generic backend before weights load,
    and where a generic model rejects dsa/dsv4_sparse."""
    from freetoken.attention import validate_attn_backend
    from freetoken.kernel.backend import is_rocm

    # Name membership first (ArgumentTypeError listing the supported names): the CLI already
    # ran this, but the programmatic EngineConfig path reaches here unvalidated and would
    # otherwise die on a bare KeyError from the info lookup below.
    validate_attn_backend(config.attention_backend, allow_auto=False)

    model_config = config.model_config
    backend_parts = [p.strip() for p in config.attention_backend.split(",")]
    for part in backend_parts:
        info = attention_backend_info(part)
        missing = required - info.supported_types
        if missing:
            valid = [
                name
                for name in (
                    "fa", "fi", "trtllm", "triton", "dsa", "dsv4_sparse", "m3_sparse",
                    "qsa_sparse",
                )
                if required <= attention_backend_info(name).supported_types
            ]
            missing_names = "/".join(sorted(t.value for t in missing))
            raise ValueError(
                f"{getattr(model_config, 'model_type', 'model')} uses {missing_names} "
                f"attention, which backend {part!r} does not support; valid backends: "
                f"{', '.join(valid)} (or auto), got {config.attention_backend!r}."
            )
        kv_format = getattr(config, "kv_format", NONE)
        if kv_format.name not in info.kv_formats:
            raise ValueError(
                f"--kv-cache-dtype {kv_format.name}: backend {part!r} cannot read that format, "
                f"got {config.attention_backend!r}."
            )
        if AttnType.SWA in required and not info.consumes_attn_spec:
            # SWA models drive window/sinks/sm_scale through the per-call AttentionSpec;
            # a backend that drops it would attend with the wrong window silently.
            raise ValueError(
                f"backend {part!r} does not consume the per-call AttentionSpec that "
                f"SWA models require, got {config.attention_backend!r}."
            )

    # An explicitly-selected backend may require a package that isn't installed. Auto
    # never resolves to one of these when its package is missing, so this only fires for
    # explicit --attention-backend choices.
    for part in backend_parts:
        info = attention_backend_info(part)
        if info.requires_flashinfer and not _flashinfer_available():
            if is_rocm():
                raise RuntimeError(
                    f"Attention backend {part!r} uses CUDA-only FlashInfer kernels "
                    "unavailable on ROCm; use --attention-backend auto or triton."
                )
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires flashinfer, which is "
                "not installed. Install it with `pip install 'freetoken[fi]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sgl_kernel and not _sgl_flash_attn_available():
            if is_rocm():
                raise RuntimeError(
                    f"Attention backend {part!r} uses CUDA-only sgl_kernel kernels "
                    "unavailable on ROCm; use --attention-backend auto or triton."
                )
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires sgl_kernel, which is "
                "not installed. Install it with `pip install 'freetoken[sgl]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sm100 and not is_sm100_family():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires a compute capability "
                "10.x GPU: flashinfer's trtllm-gen kernels ship sm_100a/103a cubins only. "
                "Use --attention-backend fi (or triton) instead."
            )

    if required & {AttnType.MLA, AttnType.DSA}:
        # Plain MLA/DSA runs on page_size 1; the kpool indexer layout needs 64.
        _kpool_ratio = max(
            (s.index_ratio for s in model_config.kv_cache_group_specs() if s.mla),
            default=1,
        )
        want_page = 64 if _kpool_ratio > 1 else 1
        if config.page_size != want_page:
            logger.warning_rank0(
                f"Page size {config.page_size} is auto-adjusted to {want_page} "
                f"for latent-KV attention."
            )
            override("page_size", want_page)

    for part in backend_parts:
        info = attention_backend_info(part)
        if info.page_sizes is not None and config.page_size not in info.page_sizes:
            override("page_size", info.page_sizes[-1])
            logger.warning_rank0(
                f"Page size is overridden to {info.page_sizes[-1]} for the {part} backend"
            )


def _host_embedding_keys(model, config: EngineConfig) -> frozenset[str]:
    """State-dict keys of the tables --embed-weights host keeps in pinned host RAM."""
    if config.embed_weights != "host":
        return frozenset()
    # a tied LM head reads its whole table every step, so that table stays on the GPU
    tied = {id(op.tied_embedding) for op in iter_ops(model) if isinstance(op, ParallelLMHead) and op.tied_embedding is not None}
    tables = {
        id(op.weight) for op in iter_ops(model) if type(op) is VocabParallelEmbedding and id(op) not in tied and not op.quant_method.online
    }
    keys = frozenset(key for key, t in model.state_dict().items() if id(t) in tables)
    if not keys and tied:
        raise ValueError(
            "--embed-weights host: this model's embedding table is tied to its LM head, which reads the whole table "
            "every step; serve it with --embed-weights gpu"
        )
    if not keys:
        raise ValueError(
            f"--embed-weights host: {type(model).__name__} has no plain embedding table to move (an --online-quant "
            "table stays on the GPU); serve it with --embed-weights gpu"
        )
    return keys


def _online_quant_keys(model, config: EngineConfig) -> frozenset[str]:
    """State-dict keys of the bf16 weights an online quant method converts in finalize; they load to host RAM so the GPU holds one of them at a time."""
    if not config.online_quant:
        return frozenset()
    to_checkpoint = config.model_config.quant.name_map.to_checkpoint
    ops = [op for op in iter_ops(model) if getattr(op, "quant_method", None) is not None and op.quant_method.online]
    names = [name for op in ops for name in to_checkpoint(op.prefix)]
    missed = [p for p in config.online_quant_modules if not any(map(name_set((p,)), names))]
    if missed:
        example = to_checkpoint(next(op.prefix for op in iter_ops(model) if getattr(op, "prefix", "")))[0]
        raise ValueError(
            f"--online-quant-modules: no layer --online-quant {config.online_quant} converts matches {', '.join(missed)}; "
            f"give the checkpoint's module names, e.g. {example!r}"
        )
    if not ops:
        raise ValueError(
            f"--online-quant {config.online_quant}: {config.model_path} has no layer it converts; the checkpoint quantizes "
            "its Linear layers and tables, or they must stay as stored"
        )
    weights = {id(op.weight) for op in ops}
    return frozenset(key for key, t in model.state_dict().items() if id(t) in weights)


def _pinned_copy(key: str, t: torch.Tensor) -> torch.Tensor:
    from freetoken.kernel.pinned import alloc_pinned_tensor

    try:
        host = alloc_pinned_tensor(*t.shape, dtype=t.dtype)
    except RuntimeError as exc:
        raise PinFailed(
            f"cudaHostAlloc refused {t.nbytes / 2**20:.0f} MiB for {key} (--embed-weights host); "
            "free host RAM or serve with --embed-weights gpu"
        ) from exc
    host.copy_(t)
    return host


def _make_dummy_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    host_keys: frozenset[str] = frozenset(),
    staged_keys: frozenset[str] = frozenset(),
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for key, param in model_state.items():
        if param.dtype in fp8_dtypes:
            # torch.randn is not implemented for fp8; fill via a uint8 view with small
            # codes (avoid NaN/inf fp8 encodings). Lets dummy-weight startup work for
            # block-fp8 models (the dense fp8 linears are fp8 regardless of moe_strategy).
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).random_(0, 16)
            state_dict[key] = t
        elif param.dtype.is_floating_point or param.dtype.is_complex:
            state_dict[key] = torch.randn(param.shape, dtype=param.dtype, device=device)
        elif param.dtype == torch.uint8 and key.endswith("weight_scale_inv"):
            # MXFP8 e8m0 exponent codes: 127 encodes scale 1.0; zeros would collapse
            # every scale to 2^-127 and zero the model. Scoped BY NAME: other uint8
            # buffers are packed payloads whose bytes mean something else entirely
            # (GGUF blocks embed fp16 scales -- 0x7F7F is fp16 NaN), so they
            # keep the benign all-zeros fill below.
            state_dict[key] = torch.full(param.shape, 127, dtype=param.dtype, device=device)
        else:
            state_dict[key] = torch.zeros(param.shape, dtype=param.dtype, device=device)
        if key in host_keys:
            state_dict[key] = _pinned_copy(key, state_dict[key])
        elif key in staged_keys:
            state_dict[key] = state_dict[key].cpu()
    return state_dict


class WeightLoadError(RuntimeError):
    """The checkpoint itself could not be read. Resource and config failures keep their own type."""


def _is_resource_failure(exc: Exception) -> bool:
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError, PinFailed)):
        return True
    # an anonymous mmap that does not fit raises ENOMEM, not MemoryError
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return True
    # torch has no type for a failed CPU allocation
    return isinstance(exc, RuntimeError) and "DefaultCPUAllocator" in str(exc)


@contextlib.contextmanager
def _weight_load_context():
    """Wrap a checkpoint read so the startup failure reason (log and /health) starts with WeightLoadError."""
    try:
        yield
    except Exception as exc:
        if _is_resource_failure(exc):
            raise
        raise WeightLoadError(f"{type(exc).__name__}: {exc}") from exc


def _materialize_loaded_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    device: torch.device,
    host_keys: frozenset[str] = frozenset(),
    staged_keys: frozenset[str] = frozenset(),
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for key, weight in weights:
        expected = model_state.get(key)
        if expected is None:
            # NOTE: the quant scheme may declare no input_scale for a layer whose FTW still stores one
            if key.endswith(".input_scale"):
                continue
            state_dict[key] = weight.to(device=device)
        elif key in host_keys:
            state_dict[key] = _pinned_copy(key, weight.to(dtype=expected.dtype))
        elif key in staged_keys:
            state_dict[key] = weight.to(device="cpu", dtype=expected.dtype)
        else:
            state_dict[key] = weight.to(device=device, dtype=expected.dtype)
    return state_dict


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        set_quant_backend(_adjust_ftw_quant_backend(config.model_path, QuantBackend.parse(config.quant_backend)))
        _ensure_expandable_segments()  # before the first CUDA allocation below

        from freetoken.gpu_select import bind_assigned_gpu

        self.device = bind_assigned_gpu(config.tp_info.rank)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config  # retained for runtime cache rebuild (rebuild_runtime_cache)
        # KV pool family fixed at construction from the model config: its classmethods own the
        # page-token geometry and cost arithmetic the engine needs BEFORE the pool exists
        # (num_pages sizing, --moe-cache-auto); the instance owns rebuild/validation after.
        self._pool_cls = resolve_pool_class(config.model_config)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        free_min, free_max = self._sync_get_memory()
        init_free_memory = free_max  # startup KV sizing keeps cross-rank MAX (unchanged)
        self._baseline_free = free_min  # rebuild baseline: cross-rank MIN, deterministic across ranks
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self._load_weights(config)
        if config.active_encoders:
            from freetoken.models.blocks import SupportsMultimodal

            if not isinstance(self.model, SupportsMultimodal):
                raise TypeError(
                    f"{type(self.model).__name__} has encoders registered but lacks the SupportsMultimodal hooks; "
                    "run with --text-model-only"
                )
            # before the residency snapshot, so streamed blocks are not charged as resident weights
            self.model.place_encoder_weights(config.mm.encoder_weights)
        # draft weights load before any pool is sized, so every free-memory read below counts them
        self.spec: SpecRuntime | None = _spec_runner()(config, self) if config.speculative_algorithm is not None else None
        if _preset_capture_bytes(config) and not (self.spec is not None and self.spec.can_prefill_as_verify):
            _drop_preset_captures(config, f"{config.speculative_algorithm} drafts cannot run a prefill chunk beside their verify step")
        post_weights_free = self._sync_get_memory()[0]
        self._weights_bytes = self._baseline_free - post_weights_free
        # Pool-budget baseline for the desktop cache sliders: free VRAM after the weights are
        # resident but before ANY runtime cache pool (MoE expert cache below, KV pages, GDN
        # state) is allocated. This is the stable "if all free VRAM went to one pool" budget —
        # unlike a query-time mem_get_info it doesn't drift with allocator caching, CUDA
        # graphs, or other processes. Cross-rank MIN, deterministic across ranks.
        self._post_weights_free = post_weights_free
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        # before the host tables: a model sizes its own for the rows of a decode step, which the drafts decide
        self._settle_preset_memory_ratio(config)
        if is_offload_moe_strategy(config.moe_strategy):
            self._settle_preset_drafts(config)
        # Host-side auxiliary stores (--embed-weights host tables, qwen4_exp's pinned PLE table): after the weights so a
        # load failure is not masked, before the MoE offload cache so the bank residency
        # planning sees the pin quota the table already spent.
        host_embeds = [
            op for op in iter_ops(self.model)
            if type(op) is VocabParallelEmbedding and not op.quant_method.online and not op.weight.is_cuda
        ]
        self._host_tables_bytes = sum(op.weight.nbytes for op in host_embeds)
        if host_embeds:
            logger.info_rank0(f"Embedding tables in pinned host RAM (--embed-weights host): {mem_GB(self._host_tables_bytes)}")
        if hasattr(self.model, "load_host_tables"):
            with _weight_load_context():
                self._host_tables_bytes += int(self.model.load_host_tables(config) or 0)
        if is_offload_moe_strategy(config.moe_strategy):
            self._init_offload_moe_cache(config)
        if hasattr(self.model, "prepare_for_runtime"):
            self.model.prepare_for_runtime()
        self.encoder_cache = None
        self.mm_processor = None
        if config.active_encoders:
            from freetoken.mm.encoder_cache import EncoderCache
            from freetoken.mm.processor import get_mm_processor

            self.mm_processor = get_mm_processor(config.model_path, config.mm)
            self.encoder_cache = EncoderCache(storage=config.mm.embed_cache_device)
            logger.info_rank0(
                f"Multimodal enabled: {type(self.mm_processor).__name__}, encoders "
                f"{[e.kind for e in config.active_encoders]} on {config.mm.encoder_weights}, serving {sorted(config.served_modalities)}"
            )
            self._warmup_encoders()
        elif any(getattr(config.hf_config, key, None) is not None for key in ENCODER_SECTIONS):
            logger.info_rank0(
                "Multimodal disabled: --text-model-only"
                if config.mm.text_model_only
                else "Multimodal disabled: --mm-disable"
                if config.mm.disabled_encoders
                else "Multimodal disabled: no encoder registered for this architecture"
            )

        # ======================= KV cache initialization ========================
        new_free = self._sync_get_memory()[1]
        # The engine measures the budget and settles the sibling GDN state pool's bytes
        # off it; the KV pool family owns every geometry-specific formula behind the rest.
        available_memory = _startup_kv_budget(config.memory_ratio, init_free_memory, new_free)
        available_memory -= state_pool_bytes(config) + self._preset_prefill_reserve(config) + _preset_capture_bytes(config)
        if self.spec is not None:
            # the pool solves pages from its own cost; hand it the share its pages get next to the draft's
            spec_fixed, draft_page = self._spec_budget(config)
            page_bytes, fixed_bytes = self._pool_cls.kv_cost(config)[:2]
            available_memory = fixed_bytes + (available_memory - spec_fixed - fixed_bytes) * page_bytes // (page_bytes + draft_page)
        self.num_pages = self._pool_cls.solve_num_pages(config, available_memory)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kv_pool(
            config, self.num_pages, device=self.device, dtype=self.dtype
        )
        if config.preset is not None:
            self._log_preset_plan(config, num_tokens)

        # ======================= Linear (GatedDeltaNet) state initialization ========================
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            self.linear_state_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
                slot_states=config.slot_states,
            )
            self.ctx.linear_state_pool = self.linear_state_pool
        else:
            self.linear_state_pool = None

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len, aligned_max_seq_len = _seq_limits(config, num_tokens)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # Pools routed by the shared table but deriving reads through their own mappings (DSV4)
        # re-point here (and again on any table realloc). The graph-input snapshot that reads
        # through them belongs to the attention backend, built later in init_capture_graph.
        self.kv_cache.attach_page_table(self.page_table)

        # ======================= Attention backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        self.linear_backend = LinearAttnBackend(self.device, self.linear_state_pool) if self.linear_state_pool is not None else None

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        # padded rows replay this request, so it carries one token per decode row
        self.dummy_req = Req(
            input_ids=torch.zeros(config.decode_rows_per_req, dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # padded/dummy rows index the GDN padding slot (0) so gather/scatter hits scratch.
        # Naive GDN keys state by table_idx, where slot 0 is a live request and the dummy's own row is the scratch.
        if self.linear_state_pool is not None and config.cache_type == "hybrid_radix":
            self.dummy_req.linear_slot_idx = self.linear_state_pool.padding_slot
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page

        # ======================= Verify state initialization ========================
        if config.decode_rows_per_req > 1 and self.linear_state_pool is not None:
            self.linear_state_pool.attach_verify_scratch(config.max_running_req, config.decode_rows_per_req)
        if self.spec is not None:
            self.spec.attach(self)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.graph_runner = self._make_graph_runner(config.cuda_graph_bs, init_free_memory, aligned_max_seq_len)
        backend = config.attention_backend.split(",")[0]
        # the warmup addresses K/V through the page table row alone, which these backends do too
        keeps = backend in ("triton", "fi", "fa", "trtllm") and self.moe_offload_cache is not None and self.moe_offload_cache.warm_prefill_keeps()
        if backend == "triton" or keeps:
            # Prefill runs on the first comma part; warm its autotune cache. With a cache that keeps what a prefill
            # streams, the warmup also loads the experts.
            self._warmup_prefill(autotune=backend == "triton", keeps=keeps)
        if self.graph_runner.rows_per_req > 1:
            # the warmup is what makes every expert resident, which the prefill graphs of an offloaded MoE are captured
            # on: the sizes the first pass had to leave are captured now
            self.chunk_runners.update(self._make_chunk_runners(self.graph_runner, init_free_memory, aligned_max_seq_len, have=self.chunk_runners))
            if os.environ.get("FREETOKEN_PREFILL_GRAPH_CHECK") == "selftest":
                self._selftest_chunk_graphs()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weights(self, config: EngineConfig) -> None:
        if config.active_encoders and not config.use_dummy_weight and ftw_lacks_vision(config.model_path):
            raise ValueError(
                f"{config.model_path} holds no vision encoder tensors: it was converted by a build before this "
                "family served images. Reconvert it with `ft checkpoint`, add the encoder in place with "
                "scripts/ftw_hotfix.py (docs/ftw-hotfix.md), or start with --text-model-only"
            )
        resident = [layer for layer in iter_moe_layers(self.model) if not isinstance(layer, OffloadMoELayer)]
        if resident and not config.use_dummy_weight and is_ftw_checkpoint(config.model_path) and not ftw_tensor_names(config.model_path, "experts_bank"):
            raise ValueError(f"{config.model_path} keeps its routed experts as dense weights (an older `ft checkpoint --moe-backend fused`); reconvert it with `ft checkpoint`")
        try:
            host_keys = _host_embedding_keys(self.model, config)
        except ValueError as exc:
            if "embed_weights" not in config.preset_fields:
                raise
            logger.info_rank0(f"--preset {config.preset}: embeddings stay on the GPU ({exc})")
            object.__setattr__(config, "embed_weights", "gpu")
            _drop_preset_fields(config, "embed_weights")
            host_keys = frozenset()
        staged_keys = _online_quant_keys(self.model, config)
        if config.online_quant:
            logger.info_rank0(f"--online-quant {config.online_quant}: converting {len(staged_keys)} weights at load")
        with _weight_load_context():
            self.model.load_state_dict(self._load_weight_state_dict(config, host_keys, staged_keys))
        if config.speculative_algorithm is not None:
            _spec_runner().prepare_target(config, self.model)
        finalize_quant(self.model)
        if resident:
            self._load_resident_experts(config, resident)

    def _load_weight_state_dict(
        self, config: EngineConfig, host_keys: frozenset[str], staged_keys: frozenset[str]
    ) -> Dict[str, torch.Tensor]:
        model_state = self.model.state_dict()
        if config.use_dummy_weight:
            return _make_dummy_weight_state_dict(model_state, device=self.device, host_keys=host_keys, staged_keys=staged_keys)
        # _materialize casts each loaded tensor to its model-param dtype (model_state), so
        # models declaring per-tensor dtypes (e.g. DSV4's mixed fp8/fp32/bf16) are preserved;
        # MoE models exclude routed experts (they load as expert banks, not dense weights).
        return _materialize_loaded_weight_state_dict(
            model_state,
            load_weight(
                config.model_path,
                self.device,
                include_moe_experts=not getattr(config.model_config, "is_moe", False),
                include_vision=bool(config.active_encoders),
            ),
            device=self.device,
            host_keys=host_keys,
            staged_keys=staged_keys,
        )

    def _load_resident_experts(self, config: EngineConfig, layers) -> None:
        with _weight_load_context():
            banks = load_expert_banks(
                config.model_path,
                config.model_config,
                method=layers[0].quant_method,
                device=self.device,
                dummy=config.use_dummy_weight,
                parallel={"serial": False, "parallel": True}.get(config.expert_load, None),
                resident=True,
                layer_methods=[layer.quant_method for layer in sorted(layers, key=lambda layer: layer.layer_id)],
            )
        attach_resident_banks(layers, banks)

    @torch.inference_mode()
    def _warmup_encoders(self) -> None:
        for item in self.mm_processor.dummy_items(self.dtype, self.device):
            if item.modality in self.config.served_modalities:
                self.model.encode(item)
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _run_mm_encoder(self, batch: Batch) -> None:
        """Encode the chunk's cache-miss items and gather its embedding rows into batch.mm_embeds, right before the LM forward."""
        cache = self.encoder_cache
        jobs = batch.mm_encoder_jobs or ()
        # a job whose rows are not gathered this chunk would be encoded and freed unread
        planned = {row[1] for row in batch.mm_gather_plan}
        orphans = [item.hash for item in jobs if item.hash not in planned]
        assert not orphans, f"encoder jobs without gather rows: {orphans}"
        for item in jobs:
            if not cache.has(item.hash):
                if item.precomputed_embeddings is not None:
                    emb = item.precomputed_embeddings.to(self.device, non_blocking=True)
                else:
                    emb = self.model.encode(item)
                cache.put(item.hash, emb)
            # cached now; free the ~MiB feature buffer
            item.feature = None
            item.precomputed_embeddings = None
        parts = []
        for uid, item_hash, lo, hi, _, _ in batch.mm_gather_plan:
            parts.append(cache.get_slice(item_hash, lo, hi, self.device))
            cache.consume(item_hash, uid, hi - lo)
        if parts:
            batch.mm_embeds = torch.cat([p.to(self.dtype) for p in parts], dim=0)

    def _resolve_auto_moe_cache_size(self, config: EngineConfig, banks, method=None, max_slots=None) -> tuple[int, int, bool]:
        """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

        Pure glue over the Phase-1 budget policy; isolated here so it is unit-testable
        without a GPU. Reused by the Phase-2 runtime rebuild.
        """
        from freetoken.engine.cache_budget import CacheBudgetTooSmall, budget_advice, resolve_moe_cache_auto

        try:
            return resolve_moe_cache_auto(**self._auto_cache_terms(config, banks, method, max_slots))
        except CacheBudgetTooSmall as exc:
            _, kv_fixed, _, min_reserve = self._pool_cls.kv_cost(config)
            raise ValueError(
                f"{exc}; budget = {config.memory_ratio} x {mem_GB(self._baseline_free)} free - {mem_GB(self._weights_bytes)} "
                f"weights - {mem_GB(kv_fixed)} fixed KV - {mem_GB(state_pool_bytes(config))} linear state - "
                f"{mem_GB(self._spec_budget(config)[0])} speculative buffers; "
                f"{budget_advice(self._budget_stages(config, banks, method, max_slots), min_reserve)}; "
                "docs/performance/decode.md (Getting more slots) lists what each flag costs"
            ) from exc

    def _auto_cache_terms(self, config: EngineConfig, banks, method=None, max_slots=None, per_expert_bytes: int | None = None) -> dict:
        """resolve_moe_cache_auto's arguments for ``config``. ``per_expert_bytes`` stands in for the banks' slot size
        before they are loaded."""
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        spec_fixed, draft_page = self._spec_budget(config)
        fixed_cache_size += spec_fixed + _preset_capture_bytes(config)
        cache_per_page += draft_page
        num_experts = config.model_config.num_experts
        total_experts = config.model_config.num_moe_layers * num_experts
        return dict(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=per_expert_bytes or expert_bytes_per_slot(banks.sources),
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            max_slots=max_slots if max_slots is not None or method is None else method.slot_limit(),
            # the planner takes it from KV pages above the reserve first, then from expert slots
            reserve_bytes=self._preset_prefill_reserve(config),
        )

    def _budget_stages(self, config: EngineConfig, banks, method, max_slots) -> list[tuple[str, dict]]:
        """What budget_advice tries, as (flags, resolver arguments): the current settings, then each
        flag that frees budget on top of those before it, least harmful first."""
        terms = self._auto_cache_terms(config, banks, method, max_slots)
        stages = [("", terms)]

        def stage(flag: str, **changed) -> None:
            flags, last = stages[-1]
            stages.append((f"{flags} {flag}".strip(), {**last, **changed}))

        if config.max_running_req > 1:
            one = copy.copy(config)
            object.__setattr__(one, "max_running_req", 1)
            stage("--max-running-requests 1", **self._auto_cache_terms(one, banks, method, max_slots))
        if config.embed_weights != "host":
            host = copy.copy(config)
            object.__setattr__(host, "embed_weights", "host")
            try:
                state = self.model.state_dict()
                moved = sum(state[key].nbytes for key in _host_embedding_keys(self.model, host))
            except ValueError:  # the flag is refused for this model
                moved = 0
            if moved:
                stage("--embed-weights host", weights_bytes=terms["weights_bytes"] - moved)
        if config.memory_ratio < 0.95:
            stage("--memory-ratio 0.95", memory_ratio=0.95)
        return stages

    def _settle_preset_memory_ratio(self, config: EngineConfig) -> None:
        """A preset's memory ratio stands only where the working memory of a prefill is known, so that the caches can leave
        it free (_preset_prefill_reserve): for any other model the default ratio is back, with one log line."""
        if "memory_ratio" not in config.preset_fields or _prefill_working_bytes(config)[1] > 0:
            return
        logger.info_rank0(
            f"--preset {config.preset}: the memory ratio stays {EngineConfig.memory_ratio}, not {config.memory_ratio}: "
            "the working memory of this model's prefill is not known, and the default headroom is what it has run in"
        )
        object.__setattr__(config, "memory_ratio", EngineConfig.memory_ratio)
        _drop_preset_fields(config, "memory_ratio")

    def _preset_draft_slots(self, config: EngineConfig) -> tuple[int, str] | None:
        """(expert slots, what was counted) of the cache a preset's server would serve its drafts from, counted before the
        expert banks load: the size the command line gave, or the automatic plan at the preset's KV reserve as asked with a
        slot's bytes taken from the bank layout (0 where that plan does not fit). The plan that is served can hold more:
        the KV reserve may still be lowered. None where the bank layout's bytes are not known."""
        from freetoken.engine.cache_budget import _plan_or_short
        from freetoken.moe.expert_banks import is_mixed

        if config.moe_cache_size > 0:
            return config.moe_cache_size, "the expert cache the command line sized has"
        method, layer_methods = shared_offload_method(self.model), offload_layer_methods(self.model)
        bank_bytes = _bank_bytes(config, method, layer_methods if is_mixed(layer_methods) else None)
        if not bank_bytes:
            return None
        total = config.model_config.num_moe_layers * config.model_config.num_experts
        per_slot = -(-bank_bytes // total)
        if config.moe_cache_mem is not None:
            return config.moe_cache_mem // per_slot, "--moe-cache-mem holds, at a slot's bytes from the bank layout,"
        terms = self._auto_cache_terms(config, None, method, per_expert_bytes=per_slot)
        if "kv_reserve_tokens" in config.preset_fields:
            terms["kv_reserve_tokens"] = min(terms["kv_reserve_tokens"], _preset_kv_target(config))
        plan, _ = _plan_or_short(terms)
        what = f"the automatic plan with them, at a KV reserve of {terms['kv_reserve_tokens']} tokens and a slot's bytes from the bank layout, has"
        return (plan[0], what) if plan is not None else (0, what.replace(", has", ", does not fit:"))

    def _settle_preset_drafts(self, config: EngineConfig) -> None:
        """A preset's drafts stay only where the offload expert cache holds PRESET_DRAFT_SHARE of all experts or more:
        under that, the draft's verify rows miss the cache more often than their accepted tokens pay back, and its weights
        take slots. Called once the draft has loaded and before the expert banks do; a strategy other than offload, CPU
        layers, and a draft depth or method from the command line are left alone."""
        drafts = {"speculative_algorithm", "speculative_num_steps"} <= set(getattr(config, "preset_fields", ()))
        if self.spec is None or not drafts or config.moe_strategy != "offload" or config.moe_cpu_layers:
            return
        total = config.model_config.num_moe_layers * config.model_config.num_experts
        counted = self._preset_draft_slots(config)
        if counted is None or counted[0] >= PRESET_DRAFT_SHARE * total:
            return
        slots, what = counted
        before = self._sync_get_memory()[0]
        self.spec = None
        for name in ("speculative_algorithm", "speculative_num_steps", "spec_slot_states"):
            object.__setattr__(config, name, getattr(EngineConfig, name))
        _drop_preset_fields(config, "speculative_algorithm", "speculative_num_steps")
        gc.collect()
        torch.cuda.empty_cache()
        # the draft's weights were counted with the target's: what the driver got back is free for the caches again
        freed = max(self._sync_get_memory()[0] - before, 0)
        self._weights_bytes -= freed
        self._post_weights_free += freed
        logger.info_rank0(
            f"--preset {config.preset}: {PRESET_NO_DRAFTS}: {what} {slots} expert slots for {total} experts, "
            f"under {PRESET_DRAFT_SHARE:.0%}, where offload decodes faster without ({mem_GB(freed)} of draft weights freed)"
        )
        if _preset_capture_bytes(config):
            _drop_preset_captures(config, "they are captured beside the verify step of speculative decoding, which is off")
        if getattr(config, "moe_decode_key", None) is not None:
            # the key was made with the drafts on: `ft bench decode` reads the last one a server logs
            object.__setattr__(config, "moe_decode_key", _decode_decision_key(config))
            logger.info_rank0(f"MoE decode setup: {config.moe_decode_key}; {self._baseline_free} bytes of GPU memory free")

    def _preset_prefill_reserve(self, config: EngineConfig) -> int:
        """GPU bytes the caches leave free under a preset's memory ratio, on top of its headroom: the working memory of
        one prefill chunk of the longest length (--max-prefill-length). The ratio's headroom alone is smaller than that
        on a 16 GB card, and a prompt it does not hold ends the server with a CUDA out-of-memory error. An automatic
        expert cache takes it in plan_cache_budget's order: KV pages above the KV reserve first, expert slots after."""
        if "memory_ratio" not in getattr(config, "preset_fields", ()):  # duck-typed test configs omit it
            return 0
        return _prefill_working_bytes(config)[1]

    def _preset_staging_slots(self, config: EngineConfig, decode_target: str, pools: list) -> int:
        """Expert slots a preset's automatic cache takes beyond one an expert where the plan has room for them
        (cache_budget.plan_resident_staging): the slots the two prefill staging buffers lie over. 0 where a prefill
        cannot read the slots anyway (no hit path, CPU-computed experts, or several pools, each capped at its experts),
        and where the command line sized the KV pool: the plan's pages, which pay for the slots, are then not used."""
        if config.preset is None or not (config.moe_prefill_hit_d2d and config.moe_prefill_overlap):
            return 0
        if decode_target != "gpu" or len(pools) != 1 or config.num_page_override is not None:
            return 0
        return 2 * config.model_config.num_experts

    def _settle_preset_captures(self, config: EngineConfig, banks, method, max_slots, staging: int) -> None:
        """A preset keeps its prefill graphs under an automatic expert cache only where the plan, with their memory
        left free, still has the KV reserve the preset asks for and every expert in a slot outside the staging
        buffers: the graphs read the slots, and are not captured on a cache that streams."""
        from freetoken.engine.cache_budget import _plan_or_short, plan_resident_staging

        if not _preset_capture_bytes(config):
            return
        terms = self._auto_cache_terms(config, banks, method, max_slots)
        if "kv_reserve_tokens" in config.preset_fields:
            terms["kv_reserve_tokens"] = min(terms["kv_reserve_tokens"], _preset_kv_target(config))
        plan, _ = _plan_or_short(terms)
        total = terms["total_experts"]
        if plan is None or not staging or plan_resident_staging(plan, terms, staging, self._moe_slot_limit)[0] < total + staging:
            held = f"{plan[0]} of {total} experts" if plan is not None else "no plan"
            _drop_preset_captures(config, f"with {mem_GB(_preset_capture_bytes(config))} left free for them the expert cache holds {held}, and they need every expert resident")

    def _settle_preset_kv_reserve(self, config: EngineConfig, banks, method, max_slots) -> None:
        """A preset's KV reserve is at most the model's context, and stays only while the plan keeps the expert slots
        of a server without the preset: where it would not, the reserve is the largest that does, the default reserve
        at the least, with one log line. Where a server without the preset has no plan at all, the reserve stays where
        the expert cache's minimum fits beside it, and elsewhere the default comes back with one warning."""
        from freetoken.engine.cache_budget import _plan_or_short, plan_kv_reserve, plan_kv_target

        asked, default = config.kv_reserve_tokens, EngineConfig.kv_reserve_tokens
        target = _preset_kv_target(config)
        plain, _ = _plan_or_short(self._preset_stages(config, banks, method, max_slots)[0][1])
        object.__setattr__(config, "kv_reserve_tokens", default)
        terms = self._auto_cache_terms(config, banks, method, max_slots)
        if plain is not None:
            reserve, plan = plan_kv_reserve(terms, target, plain[0])
            object.__setattr__(config, "kv_reserve_tokens", reserve)
            if reserve <= default:
                _drop_preset_fields(config, "kv_reserve_tokens")
            what = f"--preset {config.preset}: the KV reserve is {reserve} tokens, not {asked}" + (f", at {plan[0]} expert slots" if plan is not None else "")
            if reserve < target:
                logger.warning_rank0(
                    f"{what}: a larger one leaves fewer expert slots than the {plain[0]} a server without the preset plans. "
                    "A longer prompt and answer end with finish_reason length"
                )
            elif reserve < asked:
                logger.info_rank0(f"{what}: the model's context is {config.max_seq_len} tokens")
            return
        plan, kept, most = plan_kv_target(terms, target)
        if plan is not None:
            object.__setattr__(config, "kv_reserve_tokens", target)
            if target < asked:
                logger.info_rank0(f"--preset {config.preset}: the KV reserve is {target} tokens, not {asked}: the model's context is {config.max_seq_len} tokens")
            return
        _drop_preset_fields(config, "kv_reserve_tokens")
        if kept is None:
            return  # the default reserve does not fit either: the resolver's own error says what does
        page = terms["page_size"]
        if most[0] > kept[1] * page:
            more = f"--kv-reserve-tokens {most[0]} is the most that fits, at {most[1]} expert slots"
        else:
            more = "no larger --kv-reserve-tokens fits: the expert cache is at its minimum"
        logger.warning_rank0(
            f"--preset {config.preset}: a KV reserve of {target} tokens (the default max output tokens and a prompt) does not fit "
            f"beside the expert cache's minimum; keeping the default plan of {kept[0]} expert slots and {kept[1] * page} KV tokens, "
            f"where a longer prompt and answer end with finish_reason length. {more}"
        )

    def _prefill_tokens_within(self, config: EngineConfig, room: int) -> int:
        """The longest prefill chunk whose working memory (_prefill_working_bytes) is at most ``room`` bytes."""
        probe = copy.copy(config)

        def fits(tokens: int) -> bool:
            object.__setattr__(probe, "max_forward_len", tokens)
            return _prefill_working_bytes(probe)[1] <= room

        low, high = 0, _prefill_working_bytes(config)[0]
        while low < high:
            mid = (low + high + 1) // 2
            low, high = (mid, high) if fits(mid) else (low, mid - 1)
        return low

    def _settle_preset_chunk_reserve(self, config: EngineConfig, banks, method, max_slots) -> None:
        """A prefill chunk's working memory comes before a preset's raised KV reserve: where the plan cannot leave the
        whole chunk free, the reserve is lowered to the largest that can, at the least to a chunk-long prompt and
        PRESET_CHUNK_ANSWER_TOKENS of answer (and never under the default). What the plan then leaves free is kept for
        the plan line, and where that is still under the chunk's working memory one warning names the prompt length
        it holds."""
        from freetoken.engine.cache_budget import kv_reserve_keeping, reserve_kept

        need = self._preset_prefill_reserve(config)
        if not need:
            return
        chunk = _prefill_working_bytes(config)[0]
        terms = self._auto_cache_terms(config, banks, method, max_slots)
        if reserve_kept(terms) < need and "kv_reserve_tokens" in config.preset_fields:
            page, default = terms["page_size"], EngineConfig.kv_reserve_tokens
            floor = max(default, min(_preset_kv_target(config), align_ceil(chunk + PRESET_CHUNK_ANSWER_TOKENS, page)))
            lowered = max(floor, min(config.kv_reserve_tokens, kv_reserve_keeping(terms)))
            if lowered < config.kv_reserve_tokens:
                logger.info_rank0(
                    f"--preset {config.preset}: the KV reserve is {lowered} tokens, not {config.kv_reserve_tokens}: the working "
                    f"memory of a prefill chunk of {chunk} tokens comes before it"
                )
                object.__setattr__(config, "kv_reserve_tokens", lowered)
                if lowered <= default:
                    _drop_preset_fields(config, "kv_reserve_tokens")
                terms = self._auto_cache_terms(config, banks, method, max_slots)
        self._preset_reserve_kept = kept = reserve_kept(terms)
        if kept < need:
            longest = self._prefill_tokens_within(config, kept)
            if longest > 0:
                risk = (
                    f"a prompt longer than {longest} tokens can end the server with a CUDA out-of-memory error. "
                    f"--max-prefill-length {longest} or less keeps a chunk inside it"
                )
            else:
                risk = (
                    "no prefill chunk fits in that, so a prompt of any length can end the server with a CUDA out-of-memory "
                    "error. Both caches are at their minimum: only more free GPU memory helps"
                )
            logger.warning_rank0(
                f"--preset {config.preset}: the plan leaves {mem_GB(kept)} free for a prefill, and a chunk of {chunk} tokens works in "
                f"about {mem_GB(need)}: {risk}"
            )

    def _preset_left_free(self, config: EngineConfig) -> str:
        """What a preset's plan leaves free for a prefill chunk, for a log line: the bytes kept, and of how many where the plan is short."""
        need, chunk = self._preset_prefill_reserve(config), _prefill_working_bytes(config)[0]
        kept = getattr(self, "_preset_reserve_kept", need)
        short = f" of the {mem_GB(need)} it works in" if kept < need else ""
        return f"{mem_GB(kept)}{short} left free for a prefill chunk of {chunk} tokens"

    def _preset_stages(self, config: EngineConfig, banks, method, max_slots) -> list[tuple[str, dict]]:
        """plan_changes' stages for the budget settings a preset filled: the plan without any of them, then with each in turn."""
        filled = config.preset_fields
        steps = []
        if "max_running_req" in filled:
            steps.append((f"--max-running-requests {config.max_running_req}", ("max_running_req",)))
        if "embed_weights" in filled:
            steps.append((f"--embed-weights {config.embed_weights}", ("embed_weights",)))
        if "memory_ratio" in filled:
            steps.append((f"--memory-ratio {config.memory_ratio}", ("memory_ratio",)))
        if "speculative_algorithm" in filled:
            label = f"{config.speculative_algorithm} with {config.speculative_num_steps} draft tokens (its weights not counted)"
            steps.append((label, ("speculative_algorithm", "spec_slot_states")))
        if "kv_reserve_tokens" in filled:
            steps.append((f"--kv-reserve-tokens {config.kv_reserve_tokens}", ("kv_reserve_tokens",)))
        reserve, captures = self._preset_prefill_reserve(config), _preset_capture_bytes(config)
        if reserve or captures:
            # both are on wherever the staged config's preset_fields say so: right after the ratio's own step, where there is one
            at = next((i + 1 for i, (_, names) in enumerate(steps) if names == ("memory_ratio",)), len(steps))
            left = [self._preset_left_free(config)] if reserve else []
            left += [f"{mem_GB(captures)} left free for the prefill graphs"] if captures else []
            steps.insert(at, (" and ".join(left), ("preset_fields",)))
        staged = copy.copy(config)
        for _, names in steps:
            for name in names:
                object.__setattr__(staged, name, getattr(EngineConfig, name))
        moved = 0
        if "embed_weights" in filled:
            state = self.model.state_dict()
            moved = sum(state[key].nbytes for key in _host_embedding_keys(self.model, config))

        def terms() -> dict:
            out = self._auto_cache_terms(staged, banks, method, max_slots)
            if moved and staged.embed_weights != config.embed_weights:
                out["weights_bytes"] += moved  # the tables the measured weights no longer hold
            return out

        stages = [("", terms())]
        for label, names in steps:
            for name in names:
                object.__setattr__(staged, name, getattr(config, name))
            stages.append((label, terms()))
        return stages

    def _log_preset_plan(self, config: EngineConfig, num_tokens: int) -> None:
        """One line with what a --preset server ended up with: expert slots, KV tokens, running requests, speculative decoding."""
        parts = []
        if self.moe_offload_cache is not None:
            total = config.model_config.num_moe_layers * config.model_config.num_experts
            parts.append(f"{config.moe_cache_size} of {total} expert slots ({config.moe_strategy})")
            if self.moe_offload_cache.keeps_streamed_experts and self.moe_offload_cache.holds_every_expert:
                parts.append("a prefill reads the slots once every expert is loaded")
            elif config.moe_prefill_hit_d2d:
                parts.append("a prefill streams the experts that are not resident (--moe-prefill-hit-d2d)")
            if demand := self.moe_offload_cache.prefill_demand_tokens:
                parts.append(f"a prefill chunk of at most {demand} tokens loads only the experts it routes to")
        elif getattr(config.model_config, "is_moe", False):
            parts.append(f"every expert resident ({config.moe_strategy})")
        parts.append(f"{num_tokens} KV tokens")
        if self._preset_prefill_reserve(config):
            parts.append(self._preset_left_free(config))
        if captures := _preset_capture_bytes(config):
            parts.append(f"{mem_GB(captures)} left free for the prefill graphs of chunks up to {config.prefill_graph_tokens} tokens")
        parts.append(f"{config.max_running_req} running request" + ("s" if config.max_running_req != 1 else ""))
        if config.speculative_algorithm is not None:
            parts.append(f"{config.speculative_algorithm} speculative decoding with {config.speculative_num_steps} draft tokens")
        else:
            parts.append("no speculative decoding")
        line = f"--preset {config.preset} plan: " + ", ".join(parts)
        if "kv_reserve_tokens" in config.preset_fields and num_tokens < _preset_kv_target(config):
            # only where the expert cache was not sized by --moe-cache-auto, which keeps the reserve
            logger.warning_rank0(
                f"{line}; fewer KV tokens than the default max output tokens and a prompt ({_preset_kv_target(config)}): "
                "a longer prompt and answer end with finish_reason length"
            )
        else:
            logger.info_rank0(line)

    def _init_offload_moe_cache(self, config: EngineConfig) -> OffloadMoeCache:
        from freetoken.moe.expert_banks import is_mixed

        method = shared_offload_method(self.model)
        num_moe_layers = config.model_config.num_moe_layers
        # experts of several formats: one method per MoE layer, one slot pool per format
        layer_methods = offload_layer_methods(self.model)
        if not is_mixed(layer_methods):
            layer_methods = None
        elif config.moe_strategy == "cpu" or config.moe_cpu_layers:
            raise ValueError(
                "this model's experts come in several formats, which are served from per-format GPU slot pools; "
                "they do not combine with --moe-strategy cpu or --moe-cpu-layers (use offload or hybrid)"
            )
        # Fast path: an FTW checkpoint loads its repacked banks directly.
        # Slow path: load_expert_banks auto-picks parallel vs serial baseline by
        # expert-tensor granularity. Both pin-after-fill.
        # --expert-load: serial/parallel force the read; auto (None) lets load_expert_banks
        # pick (parallel for scattered experts, with a low-RAM fallback to serial).
        expert_parallel = {"serial": False, "parallel": True}.get(config.expert_load, None)
        check_locked_fetch_config(config)
        # --moe-cpu-layers auto without a manual pin budget: load unpinned, then pin layers until the driver refuses (settle_banks)
        # a model the CPU executor cannot serve takes the estimate path, which rejects it before the load when layers must lock
        probe = pin_probe_enabled(config, mixed_formats=layer_methods is not None) and _cpu_moe_executor_viable(config.model_config)
        if probe:
            from freetoken.moe.host_banks import HostResidency

            requested_residency = [HostResidency.DEFERRED.value] * num_moe_layers
            cpu_layer_ids, decode_target, split_residency = frozenset(), None, False
        else:
            reserved = self._host_tables_bytes + staging_reserve(config, lambda: _bank_bytes(config, method, layer_methods))
            cpu_layer_ids = _resolve_cpu_layers(config, num_moe_layers, reserved=reserved, method=method)
            _check_pin_budget(config, reserved=reserved, method=method, layer_methods=layer_methods)
            decode_target, split_residency = self._residency_split(config, cpu_layer_ids, method, capped=_pin_budget_bytes(reserved) is not None)
            requested_residency = None
            if split_residency:
                from freetoken.moe.host_banks import HostResidency

                requested_residency = [
                    HostResidency.LOCKED.value if i in cpu_layer_ids
                    else HostResidency.PINNED.value
                    for i in range(config.model_config.num_moe_layers)
                ]
        # read before the expert load, so a bad profile fails in seconds
        expert_counts = resolve_profile(
            config.moe_cache_split, config.model_path,
            config.model_config.num_moe_layers, config.model_config.num_experts,
        )
        try:
            with _weight_load_context():
                banks = load_expert_banks(
                    config.model_path,
                    config.model_config,
                    method=method,
                    device=self.device,
                    dummy=config.use_dummy_weight,
                    parallel=expert_parallel,
                    layer_residency=requested_residency,
                    layer_methods=layer_methods,
                )
        except PinFailed as exc:
            raise RuntimeError(f"{exc}; {_pin_hint(self._host_tables_bytes)}") from exc
        banks, probed, staging = settle_banks(config, banks, probe=probe, has_locked=split_residency)
        if probe:
            cpu_layer_ids = probed
            decode_target, split_residency = self._residency_split(config, cpu_layer_ids, method, capped=True)
        from freetoken.engine.cache_budget import expert_bytes_per_slot, kv_capacity_warning, layer_slot_bytes

        self._moe_layer_slot_bytes = layer_slot_bytes(banks.sources)
        # hybrid: locked layers keep GPU slots of their own pools and fetch part of their misses through the pinned staging (moe/locked_fetch.py)
        locked_fetch = staging is not None
        layer_tags = locked_pool_tags(banks.layer_tags, cpu_layer_ids, num_moe_layers) if locked_fetch else banks.layer_tags
        self._moe_pool_groups = group_layers(config.moe_cache_scope, bank_layouts(banks.sources, layer_tags), self._moe_layer_slot_bytes)
        if locked_fetch:
            # a locked refill costs a staged copy: place the locked pools after the pinned ones, which hold the prefill staging when they are big enough
            self._moe_pool_groups.sort(key=lambda layers: layers[0] in cpu_layer_ids)
        if len(self._moe_pool_groups) > 1 and cpu_layer_ids and not locked_fetch:
            raise ValueError(
                "several MoE slot pools (--moe-cache-scope per-layer, or experts in several quant formats) need every MoE "
                "layer to decode from the GPU slot cache; they combine with --moe-cpu-layers only for locked layers under "
                "--moe-strategy hybrid unless --moe-locked-fetch is 0"
            )
        self._moe_expert_counts = expert_counts if len(self._moe_pool_groups) > 1 else None
        if len(self._moe_pool_groups) == 1 and config.moe_cache_split != "equal":
            logger.warning_rank0(
                f"--moe-cache-split {config.moe_cache_split} is ignored: all MoE layers share one slot pool "
                f"(--moe-cache-scope per-layer gives each layer its own)"
            )
        self._moe_slot_limit = _slot_limit(layer_methods) if layer_methods is not None else (method.slot_limit() if method is not None else None)
        # with GPU decode, a pool below num_experts slots bounds the batch; the startup split keeps room for one request's decode step
        pools = self._moe_pools(config.decode_rows_per_req if decode_target == "gpu" else 1)
        size_cap = sum(p.max_num_slots for p in pools)
        if decode_target == "hybrid" and config.moe_decode_row is not None:
            from freetoken.moe import decode_profile

            # auto took hybrid on a row measured at one cache size: more slots than that were never compared with offload
            size_cap = decode_profile.slot_cap(config.moe_decode_row, size_cap)
        spare = self._preset_staging_slots(config, decode_target, pools)
        if config.moe_cache_auto:
            self._settle_preset_captures(config, banks, method, size_cap, spare)
            if "kv_reserve_tokens" in config.preset_fields:
                self._settle_preset_kv_reserve(config, banks, method, size_cap)
            self._settle_preset_chunk_reserve(config, banks, method, size_cap)
            size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks, method, max_slots=size_cap)
            if spare:
                from freetoken.engine.cache_budget import plan_resident_staging

                terms = self._auto_cache_terms(config, banks, method, size_cap)
                plan = plan_resident_staging((size, pages, overlap), terms, spare, self._moe_slot_limit)
                if plan[0] > size:
                    logger.info_rank0(
                        f"--preset {config.preset}: {spare} expert slots more than the {size} experts, for "
                        f"{(pages - plan[1]) * config.page_size} KV tokens: every expert then has a slot outside the prefill "
                        "staging buffers, and a prefill reads the slots as decode does"
                    )
                size, pages, overlap = plan
            object.__setattr__(config, "moe_cache_size", size)
            object.__setattr__(config, "moe_prefill_overlap", overlap)
            if config.num_page_override is None:
                # Honor the plan's KV half too: MoE slots and KV pages were solved
                # against ONE budget (ratio x baseline - weights), so both must come
                # from it. Re-solving pages later from a fresh free-memory reading
                # double-counts everything allocated since the weights measurement
                # (this expert cache, the CPU-executor GPU buffers, allocator
                # slack) and goes negative whenever the expert fill is exact --
                # a greedy fill leaves no headroom for the measurement delta.
                object.__setattr__(config, "num_page_override", pages)
                warning = kv_capacity_warning(
                    pages * config.page_size,
                    # a request is capped at the model's context too, so a pool that holds it is enough
                    min(getattr(config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS, config.max_seq_len),
                    size,
                    self._auto_cache_terms(config, banks, method, size_cap),
                )
                if warning:
                    logger.warning_rank0(warning)
            logger.info_rank0(
                f"--moe-cache-auto resolved moe_cache_size={size} "
                f"num_pages={pages} (prefill_overlap={overlap})"
            )
            if config.preset is not None:
                from freetoken.engine.cache_budget import plan_changes

                if changes := plan_changes(self._preset_stages(config, banks, method, size_cap)):
                    logger.info_rank0(f"--preset {config.preset}, by the cache planner: " + "; ".join(changes))
        elif config.moe_cache_mem is not None:
            per_slot = expert_bytes_per_slot(banks.sources)
            size = min(config.moe_cache_mem // per_slot, size_cap)
            # per_slot is a mean over layers, so with pools of different slot sizes the split can exceed the budget: shrink until it fits
            while size > 0 and self._moe_pool_bytes(split_slots(size, pools, self._moe_expert_counts)) > config.moe_cache_mem:
                size -= 1
            overlap = config.moe_prefill_overlap and size >= 2 * config.model_config.num_experts
            object.__setattr__(config, "moe_cache_size", size)
            object.__setattr__(config, "moe_prefill_overlap", overlap)
            logger.info_rank0(
                f"--moe-cache-mem {mem_GB(config.moe_cache_mem)} resolved moe_cache_size={size} "
                f"({per_slot / 2**20:.2f} MiB per slot, prefill_overlap={overlap})"
            )
        _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
        pool_sizes = split_slots(config.moe_cache_size, pools, self._moe_expert_counts)
        if len(pools) > 1:
            logger.info_rank0(
                f"MoE slot pools: {config.moe_cache_scope}, {len(pools)} pools of {min(pool_sizes)}-{max(pool_sizes)} "
                f"slots ({config.moe_cache_split})"
            )
            if decode_target == "gpu":
                _clamp_decode_batch_to_pools(config, pool_sizes)
        layout = layer_layouts = max_slots = None
        if layer_methods is not None:
            _check_layer_formats(banks, layer_methods)
            layer_layouts = [m.layout() for m in layer_methods]
            max_slots = self._moe_slot_limit
        elif method is not None:
            if banks.layer_formats is not None:
                raise ValueError(
                    f"expert banks hold experts of several formats ({sorted(set(banks.layer_formats))}) but the model "
                    f"binds {method.kind} / {method.kernel.name} on every MoE layer; reconvert the checkpoint"
                )
            if banks.kind is not None and (banks.kind, banks.kernel) != (method.kind, method.kernel.name):
                raise ValueError(
                    f"expert banks were packed for {banks.kind} / {banks.kernel} but the model "
                    f"binds {method.kind} / {method.kernel.name}; reconvert or select that kernel"
                )
            # GGUF layers keep their own ggml types, so the cache reads each layer's layout off its banks
            layout = None if method.kind is QuantKind.GGUF else method.layout()
            max_slots = method.slot_limit()
        _settle_preset_demand(config, decode_target, pool_sizes)
        cache = OffloadMoeCache(
            # Models with leading dense layers (GLM-4) only have experts on the MoE
            # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
            num_layers=config.model_config.num_moe_layers,
            num_experts=config.model_config.num_experts,
            cache_size=config.moe_cache_size,
            device=self.device,
            cache_policy=config.moe_cache_policy,
            prefill_demand_tokens=config.moe_prefill_demand_tokens,
            prefill_overlap=config.moe_prefill_overlap,
            prefill_hit_d2d=config.moe_prefill_hit_d2d,
            quant_format=banks.quant_format,
            decode_target=decode_target,
            hybrid_max_fetch=config.moe_hybrid_max_fetch,
            layout=layout,
            layer_layouts=layer_layouts,
            max_slots=max_slots,
            pool_of_layer=pool_of_layer(pools, config.model_config.num_moe_layers),
            pool_sizes=pool_sizes,
        )
        # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set
        cache.cpu_layer_ids = cpu_layer_ids
        if locked_fetch:
            cache.locked_hybrid = cpu_layer_ids
        cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency, layer_tags=layer_tags, row_tail_bytes=banks.row_tail_bytes)
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        _settle_preset_demand(config, decode_target, pool_sizes, cache)
        # Must be set before CUDA graph capture so the (device-side) accumulation ops are
        # captured and re-run on every decode replay.
        cache.collect_stats = config.moe_collect_stats
        layers = attach_offload_moe_cache(self.model, cache)
        assert len(layers) == config.model_config.num_moe_layers
        if cache.decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(config, cache, layers)
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(config, cache)
        if locked_fetch:
            attach_locked_fetch(cache, config, staging, self._profile_layer_rows)
        if _preset_capture_bytes(config) and not (cache.keeps_streamed_experts and cache.holds_every_expert):
            # a cache the command line sized, or one whose prefills cannot keep what they stream
            _drop_preset_captures(config, f"the expert cache of {config.moe_cache_size} slots cannot hold every expert outside its prefill staging buffers, which they need")
        self.moe_offload_cache = cache
        return cache

    def _residency_split(self, config: EngineConfig, cpu_layer_ids, method, *, capped: bool):
        """(decode_target, split_residency) for a CPU-layer set; turns prefill overlap off when some layers stay unpinned."""
        # the kernels were picked for model_config.decode_target; --moe-cpu-layers auto may still find that every bank fits the pin budget
        decode_target = config.model_config.decode_target
        if decode_target == "cpu" and not cpu_layer_ids:
            decode_target = "gpu"
        # split residency: where pinning is quota-capped (_pin_budget_bytes), pin only the GPU layers' banks and mlock the CPU layers'
        # uncapped hosts keep every bank pinned (CPU decode reads them the same; overlap prefill stays on)
        # not applied to plain --moe-strategy cpu; all-locked under a cap = --moe-strategy offload --moe-cpu-layers 1.0
        split_residency = (
            bool(cpu_layer_ids)
            and config.moe_strategy in ("offload", "hybrid")
            and capped
        )
        if config.moe_strategy == "cpu" and not split_residency:
            # cpu mode pins every bank for the prefill double buffer; over the pin cap that dies in cudaHostRegister, so lock everything instead
            from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

            budget = _pin_budget_bytes(self._host_tables_bytes)
            bank_bytes = None
            if budget is not None:
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config, method)
            if bank_bytes and bank_bytes > budget:
                split_residency = True
                logger.info_rank0(
                    f"--moe-strategy cpu: banks {bank_bytes / 2**30:.2f} GiB exceed the "
                    f"pin budget; OS-locking all layers instead of pinning"
                )
        if split_residency and config.moe_prefill_overlap:
            # locked (unregistered) layers cannot feed the async pinned H2D double buffer; their prefill is a synchronous pageable copy via materialize
            logger.info_rank0(
                "--moe-cpu-layers split residency: disabling MoE prefill overlap "
                "(locked layers prefill via synchronous pageable copies)"
            )
            object.__setattr__(config, "moe_prefill_overlap", False)
        return decode_target, split_residency

    def _moe_pool_bytes(self, pool_sizes: list[int]) -> int:
        return sum(s * self._moe_layer_slot_bytes[g[0]] for g, s in zip(self._moe_pool_groups, pool_sizes))

    def _moe_pools(self, batch: int) -> list:
        """The slot pools, each with room for the routed experts of ``batch`` decode tokens."""
        model_config = self.config.model_config
        num_experts = model_config.num_experts
        min_num_slots = min(num_experts, model_config.num_experts_per_tok * batch)
        return make_pools(self._moe_pool_groups, num_experts, min_num_slots, self._moe_slot_limit)

    def _resolve_hybrid_fetch(self, config: EngineConfig, cache) -> None:
        """Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction.

        Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw :
        (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode
        step's misses -- both sides then finish together instead of one idling. The
        achieved bandwidths come from the cached `ft bench bw` profile, from the row of each
        MoE layer's kind (format, options, expert shape, top_k, CPU worker threads); a layer
        without a row keeps the fixed cap of 1.
        """
        if config.moe_hybrid_max_fetch >= 0:
            return  # explicit fixed cap
        from freetoken.moe.bench_profile import fetch_fraction

        cache.hybrid_max_fetch = 1
        layer_rows = self._profile_layer_rows()
        if layer_rows is None:
            logger.warning_rank0("--moe-hybrid-max-fetch auto: the MoE layers bind no expert method to look up in the `ft bench bw` profile; using a fixed fetch cap of 1")
            return
        executor = self.cpu_moe_executor
        keys, labels, rows = layer_rows
        fractions = [fetch_fraction(rows.get(key)) for key in keys]
        cache.hybrid_fetch_fraction = [f or 0.0 for f in fractions]
        # locked layers fetch the fetch_plan share instead; their own log line lists it
        groups = collections.Counter((labels[l], fractions[l]) for l in range(len(keys)) if l not in cache.locked_hybrid)
        fetched = [f"{f:.1%} on {n} {label} layers" for (label, f), n in groups.items() if f is not None]
        if fetched:
            logger.info_rank0(
                "--moe-hybrid-max-fetch auto: each decode step fetches this share of a layer's expert misses over PCIe "
                f"(benched PCIe/CPU bandwidth ratio) and computes the rest on the CPU: {'; '.join(fetched)}"
            )
        if missing := sorted({label for label, f in groups if f is None}):
            capped = sum(n for (_, f), n in groups.items() if f is None)
            logger.warning_rank0(
                f"--moe-hybrid-max-fetch auto: no `ft bench bw` row for {capped} MoE layers ({', '.join(missing)}) at "
                f"{executor.num_threads} CPU threads; using a fixed fetch cap of 1 there. `ft bench bw --model <checkpoint>` benches each layer kind"
            )
        benched_isas = {rows[key].get("cpu_moe_isa") for key, f in zip(keys, fractions) if f is not None} - {None}
        if other := sorted(benched_isas - set(executor.isa.split(","))):
            logger.warning_rank0(
                f"--moe-hybrid-max-fetch auto: some `ft bench bw` rows were benched on CPU kernels {other}, but this executor runs "
                f"{executor.isa}; using them anyway"
            )

    def _profile_layer_rows(self):
        """(row key, label) of each MoE layer's kind in the `ft bench bw` profile, and the profile's rows; None without expert methods."""
        from freetoken.moe.bench_profile import layer_kind_key, load_layer_kinds, nearest_kind
        from freetoken.moe.cpu_executor import _format_label

        methods = offload_layer_methods(self.model)
        if methods is None:
            return None
        gpu_name, gpu_uuid = _profile_gpu(self.device.index)
        rows = load_layer_kinds(gpu_name=gpu_name, gpu_uuid=gpu_uuid) or {}
        keys, labels, stand_ins = [], [], {}
        for m in methods:
            fmt, opts = m.cpu_spec()
            c = m.cfg
            key = layer_kind_key(fmt, opts, c.hidden, c.local_intermediate, c.top_k, self.cpu_moe_executor.num_threads)
            if key not in rows and (near := nearest_kind(rows, key)) is not None:
                # a benched row of the same format and top_k, close in expert size and thread count, stands in for the missing one
                stand_ins[key] = near
                key = near
            keys.append(key)
            labels.append(f"{_format_label(fmt, tuple(sorted(opts.items())))} {c.hidden}x{c.local_intermediate}")
        if stand_ins:
            logger.info_rank0(
                "`ft bench bw` profile: no row for " + "; ".join(f"{want} (using the row of {got})" for want, got in sorted(stand_ins.items()))
                + ". `ft bench bw --model <checkpoint>` benches the exact layer kinds"
            )
        return keys, labels, rows

    def _init_cpu_moe_executor(self, config: EngineConfig, cache, layers) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Must run before CUDA graph capture: the worker pool has to be live for the
        eager warmup forward, and the pinned IO block must be stable for the captured
        nodes.
        """
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        sample = layers[0]
        required = ("top_k", "activation", "apply_router_weight_on_input")
        if not all(hasattr(sample, attr) for attr in required):
            raise NotImplementedError(
                "CPU MoE backend is not yet supported for this model architecture "
                f"(MoE layer {type(sample).__name__} is missing {required})."
            )
        # Decode batches never exceed max_running_req, but CUDA-graph padding can
        # round a batch up to the largest captured size; cover both.
        max_tokens = max(config.max_running_req, config.cuda_graph_max_bs or 0, *(config.cuda_graph_bs or []), 1)
        max_tokens *= config.decode_rows_per_req
        # Match GraphRunner's explicit-size precedence, including an empty list.
        if config.cuda_graph_bs is not None:
            graph_enabled = bool(config.cuda_graph_bs)
        else:
            graph_enabled = config.cuda_graph_max_bs is None or config.cuda_graph_max_bs > 0
        by_id = sorted(layers, key=lambda layer: layer.layer_id)
        layer_formats = [layer.quant_method.cpu_spec() for layer in by_id]
        executor = CpuMoeExecutor(
            cache,
            top_k=sample.top_k,
            activation=sample.activation,
            apply_router_weight_on_input=sample.apply_router_weight_on_input,
            num_threads=config.moe_cpu_threads,
            max_tokens=max_tokens,
            device=self.device,
            activation_dtype=self.dtype,
            graph_enabled=graph_enabled,
            swiglu_alpha=float(sample.alpha),
            swiglu_limit=sample.limit,
            layer_formats=layer_formats,
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor
        if graph_enabled:
            _disable_unsafe_cpu_moe_graphs(config, executor, self.dtype, self.tp_cpu_group)

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _target_moe_and_expert_bytes(self, moe_cache_size: int | None, pool_sizes: list[int] | None) -> tuple[int, int]:
        cache = self.moe_offload_cache
        if cache is None:
            return moe_cache_size or 0, 0
        if moe_cache_size is None:
            target_moe = cache.cache_size
            nbytes = sum(c.numel() * c.element_size() for _, c in cache.banks)
        else:
            target_moe = moe_cache_size
            nbytes = self._moe_pool_bytes(pool_sizes)
        # pools of several bank layouts have different slot sizes; round up so target * bytes covers them
        return target_moe, -(-nbytes // target_moe)

    def _resize_kv_pool(self, config, num_pages: int, num_swa_pages: int | None) -> None:
        # IN-PLACE, identity-preserving: the CacheManager's swa_pool reference, ctx.kv_cache and
        # the model's per-access pool property all keep pointing at THIS pool, which frees its old
        # buffers before allocating the new ones. mark_for_rebind re-binds per-bind scratch on the
        # next forward (graph re-capture); the prefix tree + page bookkeeping reset is the
        # scheduler's generic cache_manager.rebuild.
        if self.kv_cache.needs_rebind_on_rebuild:
            self.model.mark_for_rebind()
        self.kv_cache.rebuild_from_config(config, num_pages, num_swa_pages=num_swa_pages)
        self.num_pages = num_pages

    def _refresh_seq_state(self, config) -> None:
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len, aligned_max_seq_len = _seq_limits(config, num_tokens)
        if aligned_max_seq_len != self.page_table.shape[1]:
            # max_seq_len changed (e.g. KV grew past the startup token budget); the page table
            # columns must track it or new requests would index out of bounds. The scheduler
            # re-points its managers to engine.page_table on a num_pages rebuild.
            self.ctx.page_table = self.page_table = torch.zeros(
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.kv_cache.attach_page_table(self.page_table)

    @torch.inference_mode()
    def rebuild_runtime_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
        and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
        re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
        guarantee no in-flight prefill/decode.
        """
        config = self.config
        if (moe_cache_size is None and num_pages is None and num_mamba_slots is None
                and num_swa_pages is None):
            return

        # 0a. Geometry prevalidation BEFORE any destructive free. An invalid target (moe
        #     slots on a model with no offload cache, moe below num_experts / above the
        #     marlin cap, non-positive pages, or too few GDN slots to run) must reject
        #     recoverably with the old cache intact -- NOT after teardown, which would
        #     leave the server unable to serve. These checks are model-agnostic.
        if moe_cache_size is not None:
            if self.moe_offload_cache is None:
                raise CacheRebuildRejected(
                    "moe_cache_size requested but this model has no MoE offload cache"
                )
            try:
                # a rebuild does not lower max_running_req again, so with GPU decode every pool must hold a full batch; hybrid caps its fetches instead
                batch = config.max_running_req * config.decode_rows_per_req if self.moe_offload_cache.decode_target == "gpu" else 1
                moe_pool_sizes = split_slots(moe_cache_size, self._moe_pools(batch), self._moe_expert_counts)
                self.moe_offload_cache.validate_rebuild(moe_cache_size, moe_pool_sizes)
            except ValueError as e:
                raise CacheRebuildRejected(str(e)) from e
            if self.moe_offload_cache.decode_target == "hybrid" and config.moe_decode_row is not None:
                from freetoken.moe import decode_profile

                if note := decode_profile.resize_note(config.moe_decode_row, moe_cache_size):
                    logger.warning_rank0(f"--moe-strategy auto: {note}")
        if num_pages is not None and num_pages <= 0:
            raise CacheRebuildRejected(f"num_pages must be positive, got {num_pages}")
        if num_mamba_slots is not None:
            if self.linear_state_pool is None:
                raise CacheRebuildRejected(
                    "num_mamba_slots requested but this model has no GDN state pool"
                )
            # num_mamba_slots is the USABLE slot count (what the user sets and the status bar
            # shows); the pool also reserves a padding sink (slot 0), so the physical pool is
            # num_mamba_slots + 1. _linear_pool_min_slots is the physical floor -> usable - 1.
            min_usable = _linear_pool_min_slots(config) - 1
            if num_mamba_slots < min_usable:
                raise CacheRebuildRejected(
                    f"num_mamba_slots {num_mamba_slots} is below the minimum {min_usable} "
                    f"(non-evictable working set for max_running_req={config.max_running_req}) "
                    f"needed to run; admission would deadlock"
                )
        if num_swa_pages is not None:
            # An absolute window pin for the radix-SWA window pool (Gemma) or the DSV4 window tier;
            # meaningless for dense/MHA models and the naive SWA path (concurrency x window).
            if not _supports_swa_ratio(config):
                raise CacheRebuildRejected(
                    "num_swa_pages requested but this model has no window pool "
                    "(needs DSV4 or a sliding-window model with --cache-type radix)"
                )
            if num_swa_pages <= 0:
                raise CacheRebuildRejected(
                    f"num_swa_pages must be positive, got {num_swa_pages}"
                )

        # 0b. Pool-family budget fit-check BEFORE any destructive free: an unfit geometry
        #     must reject (recoverable) so the old caches stay intact and serving continues,
        #     rather than freeing and then OOMing into permanent failure. The engine supplies
        #     the memory account; the pool answers whether its target geometry fits.
        target_moe, per_expert_bytes = self._target_moe_and_expert_bytes(
            moe_cache_size, moe_pool_sizes if moe_cache_size is not None else None
        )
        # Price the sibling GDN state pool at ITS target (physical slots = usable + padding
        # sink) and hand the bytes in -- the KV pool only budgets its own tiers.
        target_mamba = (
            num_mamba_slots + 1
            if num_mamba_slots is not None
            else (self.linear_state_pool.num_slots if self.linear_state_pool is not None else None)
        )
        spec_fixed, draft_page = self._spec_budget(config)
        prefill_reserve = self._preset_prefill_reserve(config)
        self.kv_cache.validate_rebuild(
            config, num_pages=num_pages,
            num_swa_pages=num_swa_pages, target_moe=target_moe,
            per_expert_bytes=per_expert_bytes, baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes, current_num_pages=self.num_pages,
            extra_fixed_bytes=(state_pool_bytes(config, target_mamba) if target_mamba is not None else 0) + spec_fixed + prefill_reserve,
            extra_page_bytes=draft_page,
            extra_note=(
                f", mamba={target_mamba - 1} slots" if target_mamba is not None else ""
            ),
        )

        torch.cuda.synchronize(self.device)
        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
        # off free memory, which is far smaller now that the caches are resident (post-cache
        # free << startup pre-load free), so re-deriving it here would silently drop large
        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
        prior_graph_bs = self.graph_runner.graph_bs_list
        # Point of no return for the scheduler's rollback logic: from here the live graphs and
        # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
        # untouched (no rollback needed); after it, only a rebuild restores service.
        self.rebuild_teardown_started = True
        # 1. Tear down CUDA graphs + backend capture scratch (free-before-alloc).
        self.attn_backend.reset_capture()
        if self.linear_backend is not None:
            self.linear_backend.reset_capture()
        self.graph_runner.destroy_cuda_graphs()
        if self.prefill_runner is not None:
            self.prefill_runner.destroy_cuda_graphs()
            self.prefill_runner = None
        for chunk_runner in self.chunk_runners.values():
            chunk_runner.destroy_cuda_graphs()
        self.chunk_runners = {}
        if self.spec is not None:
            # the draft KV is keyed by the target's slots, so it can stay only when the target KV pool does
            self.spec.detach(keep_kv=num_pages is None and num_swa_pages is None)
        # 2. Resize caches in place (each frees its old GPU tensors before allocating).
        # Pin the new window first (validated above) so any KV-pool rebuild below sizes the window
        # to it (_dsv4_pool_sizes / _swa_paged_num_tokens read config.swa_num_pages_override).
        # frozen EngineConfig — mutate in place like the moe_cache_size path; `config.x = y` raises
        # FrozenInstanceError, which here aborts the rebuild after the CUDA graphs are gone (→ 503).
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        if moe_cache_size is not None:
            assert self.moe_offload_cache is not None, "no MoE offload cache to resize"
            self.moe_offload_cache.rebuild(moe_cache_size, moe_pool_sizes)
        if num_pages is not None:
            # sets self.num_pages (rebuilds KV + window)
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_swa_pages is not None:
            # Window-only change: no page-count change, but re-derive the window pool at the new
            # pin against the CURRENT page count. This re-allocs the same-size full pool and
            # the resized window, both inside the pool's own rebuild_from_config.
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            # Reallocate the GDN state pool (frees old tensors first). Must sit between graph
            # teardown and re-capture so the recaptured graphs bind the new state tensors.
            # +1 for the reserved padding sink: num_mamba_slots is the usable count.
            self.linear_state_pool.rebuild(num_mamba_slots + 1)
        # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
        self._refresh_seq_state(config)
        aligned_max_seq_len = self.page_table.shape[1]
        # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
        #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
        gc.collect()
        free_min = self._sync_get_memory()[0]
        if self.spec is not None:
            self.spec.attach(self)
        # reuse the startup-resolved bs set (see above)
        self.graph_runner = self._make_graph_runner(prior_graph_bs, free_min, aligned_max_seq_len)

    def _make_graph_runner(self, cuda_graph_bs, free_memory: int, max_seq_len: int) -> GraphRunner:
        """The target's graphs: the decode family, or with speculative decoding the verify family in its place."""
        config = self.config
        rows = config.decode_rows_per_req
        verify = {}
        if rows > 1:
            # every decode step is a verify; the verify scratch has max_running_req rows
            cuda_graph_bs = [
                bs for bs in _determine_cuda_graph_bs(cuda_graph_bs, config.cuda_graph_max_bs, free_memory)
                if bs <= config.max_running_req
            ]
            verify = dict(rows_per_req=rows, phase="verify", batch_hook=self._prepare_verify_batch)
        runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            forward=self.model.forward,
            attn_backend=self.attn_backend,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=free_memory,
            max_seq_len=max_seq_len,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
            mrope=config.model_config.model_is_mrope,
            out_dtype=torch.float32,
            linear_backend=self.linear_backend,
            **verify,
        )
        if self.spec is not None:
            self.spec.capture_graphs(runner.graph_bs_list, runner.graph_pool)
        self.prefill_runner = self._make_prefill_runner(runner, free_memory, max_seq_len) if rows > 1 else None
        self.chunk_runners = self._make_chunk_runners(runner, free_memory, max_seq_len) if rows > 1 else {}
        return runner

    def _capture_attn_backend(self):
        """One more attention backend for a capture of another row count; it borrows the target backend's workspaces, which only one forward uses at a time."""
        attn = create_attention_backend(self.config.attention_backend, self.config.model_config)
        for name in ("float_workspace_buffer", "int_workspace_buffer", "prefill_wrapper", "decode_wrappers"):
            if hasattr(attn, name) and hasattr(self.attn_backend, name):
                setattr(attn, name, getattr(self.attn_backend, name))
        return attn

    def _make_chunk_runners(self, target: GraphRunner, free_memory: int, max_seq_len: int, have=()) -> Dict[int, GraphRunner]:
        """Prefill graphs of --prefill-graph-chunks token counts of one request (SpecRunner.prefill_plan cuts a short
        chunk into replays of them). Each has its own attention capture and linear capture; one that cannot be
        captured is left out, with a warning."""
        config, pool, cache = self.config, self.linear_state_pool, self.moe_offload_cache
        from freetoken.kernel.fla.chunk import CHUNK_SIZE as CHUNK

        sizes = set()
        for part in config.prefill_graph_chunks.split(","):
            # a count, a range "1-64", or a range with a step "128-448:64"
            span, _, step = part.strip().partition(":")
            low, _, high = span.partition("-")
            if not part.strip():
                continue
            try:
                low, high, step = int(low), int(high or low), int(step or 1)
            except ValueError:
                low = 0
            if not (1 <= low <= high and step >= 1):
                raise ValueError(f"--prefill-graph-chunks: '{part.strip()}' is not a token count, a range a-b or a range a-b:step of positive counts")
            sizes.update(range(low, high + 1, step))
        # the plan replays a tail of 1 to 64 tokens and a head that is a multiple of 64 within the limit; nothing else
        usable = {t for t in sizes if t <= CHUNK or (t % CHUNK == 0 and t <= (config.prefill_graph_tokens - 1) // CHUNK * CHUNK)}
        if sizes - usable:
            logger.info_rank0(f"--prefill-graph-chunks: not captured, no plan replays them: {sorted(sizes - usable)}")
        if len(usable) > 256:
            raise ValueError(f"--prefill-graph-chunks names {len(usable)} sizes; at most 256")
        sizes = sorted(usable, reverse=True)
        if config.prefill_graph_tokens <= 0 or not sizes or 1 not in target.graph_map or not self.spec.can_prefill_as_verify:
            return {}
        runners: Dict[int, GraphRunner] = {}
        resident = cache is not None and cache.all_resident
        self._chunks_need_resident = resident or (bool(have) and self._chunks_need_resident)
        for tokens in sizes:
            if tokens in have:
                continue
            if cache is not None and not resident and not cache.loads_prefill_on_demand(tokens):
                logger.info_rank0(f"--prefill-graph-chunks {tokens}: not captured yet, the expert cache stages chunks of that length through host-side steps (not every expert is resident and --moe-prefill-demand-tokens is below it)")
                continue
            before = get_free_memory(self.device)
            if before < 1 << 30:
                logger.warning_rank0(f"--prefill-graph-chunks {tokens}: not captured, {before / 2**20:.0f} MiB of GPU memory are free; chunks that need it run eagerly")
                continue
            try:
                dummy = Req(
                    input_ids=torch.zeros(tokens, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
                    sampling_params=None, cache_handle=None,  # type: ignore[arg-type]
                )
                dummy.linear_slot_idx = self.dummy_req.linear_slot_idx
                attn = self._capture_attn_backend()
                attn.capture_prefill = True
                linear = ChunkCaptureLinearBackend(self.device, pool, tokens) if pool is not None else None
                sink = self.spec.chunk_rows_sink(tokens)

                def hook(batch: Batch, sink=sink) -> None:
                    batch.features = sink

                runners[tokens] = GraphRunner(
                    stream=self.stream, device=self.device, forward=self.model.forward, attn_backend=attn,
                    cuda_graph_bs=[1], cuda_graph_max_bs=1, free_memory=free_memory, max_seq_len=max_seq_len,
                    # with every expert resident the forward reads the slots and changes nothing in the cache: no reset
                    dummy_req=dummy, moe_offload_cache=None if resident else cache, mrope=config.model_config.model_is_mrope,
                    rows_per_req=tokens, phase="prefill", ctx_override={"attn_backend": attn}, batch_hook=hook,
                    pool=target.graph_pool, out_dtype=torch.float32, linear_backend=linear,
                )
                torch.cuda.synchronize(self.device)
                logger.info_rank0(f"prefill graph of {tokens} tokens: {(before - get_free_memory(self.device)) / 2**20:.0f} MiB")
            except Exception as exc:  # noqa: BLE001 -- optional graphs: a failure leaves that size to the other steps
                logger.warning_rank0(f"--prefill-graph-chunks {tokens}: the prefill graph was not captured ({type(exc).__name__}: {exc})")
        return runners

    @torch.inference_mode()
    def _selftest_chunk_graphs(self) -> None:
        """Diagnostic (FREETOKEN_PREFILL_GRAPH_CHECK=selftest): every captured prefill forward against an eager prefill
        forward of the same tokens, once from an empty state and once after a 96-token prefix, then three two-replay
        plans with their snapshot, on the padding request's row with seeded token ids. SpecRunner.prefill_as_verify
        prints one PREFILL_SPLIT_CHECK line a captured step; a line here says whether each snapshot is the state at its cut."""
        from types import SimpleNamespace

        from freetoken.kernel.fla.chunk import CHUNK_SIZE as CHUNK

        if not self.chunk_runners or not self.chunk_graphs_usable():
            return
        row, pool = self.page_table[self.dummy_req.table_idx], self.linear_state_pool
        saved, prefix = row.clone(), 96
        generator = torch.Generator().manual_seed(0)
        vocab = min(self.config.model_config.vocab_size, 100_000)
        live = self.dummy_req.linear_slot_idx

        def run(ids: torch.Tensor, start: int, plan: list, track: int | None = None) -> None:
            req = SimpleNamespace(
                table_idx=self.dummy_req.table_idx, cached_len=start, device_len=ids.numel(), extend_len=ids.numel() - start,
                linear_slot_idx=live, mrope_delta=0, mrope_positions_full=None,
            )
            batch = SimpleNamespace(reqs=[req], input_ids=ids[start:].to(self.device), out_loc=row[start : ids.numel()], prefill_plan=plan)
            if track is not None:
                batch.prefill_track_dst = torch.tensor([track], dtype=torch.int64, device=self.device)
            self.spec.prefill_as_verify(batch, check=True)

        def tokens(n: int) -> torch.Tensor:
            return torch.randint(1000, vocab, (n,), generator=generator, dtype=torch.int32)

        try:
            for size in sorted(self.chunk_runners):
                if prefix + size > self.max_seq_len:
                    continue
                # free pages, as the warmup takes them
                row[: prefix + size] = torch.arange(prefix + size, dtype=row.dtype, device=row.device)
                ids = tokens(prefix + size)
                run(ids[:size], 0, [("chunk", size)])
                run(ids[:prefix], 0, [("eager", prefix)])
                run(ids, prefix, [("chunk", size)])
            # the plans as requests run them: a head, the snapshot, a tail over what the head's replay stored
            spare = pool.alloc(1)[0] if pool is not None and pool.num_free_slots > 0 else None
            for head, tail in ((64, 64), (128, 37), (max((s for s in self.chunk_runners if s % CHUNK == 0), default=CHUNK), 1)):
                if spare is None or head not in self.chunk_runners or tail not in self.chunk_runners or prefix + head + tail > self.max_seq_len:
                    continue
                row[: prefix + head + tail] = torch.arange(prefix + head + tail, dtype=row.dtype, device=row.device)
                ids = tokens(prefix + head + tail)
                run(ids[:prefix], 0, [("eager", prefix)])
                run(ids[: prefix + head], prefix, [("eager", head)])
                at_cut = [pool.recurrent_states[:, live].clone(), pool.conv_states[:, live].clone()]
                run(ids[:prefix], 0, [("eager", prefix)])
                run(ids, prefix, [("chunk", head), ("track", 0), ("chunk", tail)], track=spare)
                same = torch.equal(pool.recurrent_states[:, spare], at_cut[0]) and torch.equal(pool.conv_states[:, spare], at_cut[1])
                print(f"PREFILL_SPLIT_CHECK plan={head}+{tail} start={prefix} snapshot_is_the_eager_state_at_the_cut={same}", flush=True)
            if spare is not None:
                pool.free([spare])
        finally:
            row.copy_(saved)
            self.spec.proposer._prefill_sink = None
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset_stats()
        torch.cuda.synchronize(self.device)
        logger.info_rank0(f"prefill graph self-test done for {len(self.chunk_runners)} sizes; {get_free_memory(self.device) / 2**20:.0f} MiB of GPU memory free")

    def chunk_graphs_usable(self) -> bool:
        """Whether the prefill graphs may run now: those captured on an all-resident expert cache read its slots as they are."""
        cache = self.moe_offload_cache
        return cache is None or not self._chunks_need_resident or cache.all_resident

    def _make_prefill_runner(self, target: GraphRunner, free_memory: int, max_seq_len: int) -> GraphRunner | None:
        """A verify graph of --prefill-graph-rows rows of one request, for short prefill chunks longer than a decode
        step (SpecRunner.prefill_as_verify). The attention and linear captures are tied to one row count, so it has
        its own of each, its own verify scratch and its own padding request; None when it is not asked for or cannot
        be captured, and the chunks then take the decode step's graph."""
        config, rows, pool = self.config, self.config.prefill_graph_rows, self.linear_state_pool
        if config.prefill_graph_tokens <= 0 or rows <= target.rows_per_req or 1 not in target.graph_map:
            return None
        if not self.spec.can_prefill_as_verify or (pool is not None and pool._verify_handlers):
            return None
        try:
            dummy = Req(
                input_ids=torch.zeros(rows, dtype=torch.int32, device="cpu"),
                table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None,  # type: ignore[arg-type]
            )
            dummy.linear_slot_idx = self.dummy_req.linear_slot_idx
            attn = self._capture_attn_backend()
            linear = WideVerifyLinearBackend(self.device, pool, rows) if pool is not None else None
            sink = self.spec.prefill_rows_sink(rows)

            def hook(batch: Batch) -> None:
                batch.features = sink

            return GraphRunner(
                stream=self.stream, device=self.device, forward=self.model.forward, attn_backend=attn,
                cuda_graph_bs=[1], cuda_graph_max_bs=1, free_memory=free_memory, max_seq_len=max_seq_len,
                dummy_req=dummy, moe_offload_cache=self.moe_offload_cache, mrope=config.model_config.model_is_mrope,
                rows_per_req=rows, phase="verify", ctx_override={"attn_backend": attn}, batch_hook=hook,
                pool=target.graph_pool, out_dtype=torch.float32, linear_backend=linear,
            )
        except Exception as exc:  # noqa: BLE001 -- an optional graph: any failure leaves the decode step's graph in use
            logger.warning_rank0(f"--prefill-graph-rows {rows}: the wider verify graph was not captured ({type(exc).__name__}: {exc}); short prefill chunks use the decode step's graph")
            return None

    def _prepare_verify_batch(self, batch: Batch) -> None:
        batch.features = self.spec.feature_sink(batch)

    def commit_verify(self, reqs, counts, out: torch.Tensor) -> None:
        """Commit each verified request's first ``counts`` tokens (row i of ``out`` is reqs[i]) to the target's per-request state."""
        if self.linear_state_pool is not None and self.linear_state_pool.verify_scratch is not None:
            self.linear_state_pool.commit_verify([gdn_slot(r) for r in reqs], counts)
        for req, n, tokens in zip(reqs, counts, out):
            if req.output_token_counts is not None:
                req.output_token_counts.scatter_add_(0, tokens[:n], req.output_token_counts.new_ones(n))
            if req.context_token_seen is not None:
                req.context_token_seen.index_fill_(0, tokens[:n], True)

    def _spec_budget(self, config) -> tuple[int, int]:
        """(fixed bytes, draft KV bytes per page) speculative decoding adds next to the KV pool."""
        if getattr(self, "spec", None) is None or getattr(config, "speculative_algorithm", None) is None:
            return 0, 0
        fixed, per_token = self.spec.budget(config)
        return fixed, per_token * config.page_size

    def forward_logits(self, batch: Batch) -> torch.Tensor:
        """The batch's logits from its captured graph, or eagerly when none fits."""
        use_graph = self.graph_runner.can_use_cuda_graph(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph):
            logits = self.graph_runner.replay(batch) if use_graph else self.model.forward()
        if self.cpu_moe_executor is not None:
            # One pinned read: surfaces a fired flag-handshake watchdog (dead coordinator
            # -> stale expert outputs) as a loud error instead of silent corruption.
            self.cpu_moe_executor.raise_if_unhealthy()
        return logits

    def _prefill_runs_as_verify(self, batch: Batch) -> bool:
        """Whether this prefill chunk runs as the steps of a plan (SpecRunner.prefill_plan): one request, text only, and a plan exists for it; without prefill graphs, at most --prefill-graph-tokens tokens."""
        limit, runner = self.config.prefill_graph_tokens, self.graph_runner
        if limit <= 0 or batch.size != 1 or runner.phase != "verify" or batch.mm_gather_plan:
            return False
        if not self.spec.can_prefill_as_verify or (not self.chunk_runners and batch.reqs[0].extend_len > limit):
            return False
        batch.prefill_plan = self.spec.prefill_plan(batch)
        return batch.prefill_plan is not None

    def _check_prefill_as_verify(self, batch: Batch) -> torch.Tensor:
        """Diagnostic: run the chunk eagerly, put the linear state back, run it as verify steps, and log how far apart the two are."""
        pool, req = self.linear_state_pool, batch.reqs[0]
        slot = gdn_slot(req)
        rec0, conv0 = pool.recurrent_states[:, slot].clone(), pool.conv_states[:, slot].clone()
        loc = batch.out_loc.long()
        batch.features = self.spec.feature_sink(batch)
        eager = self.forward_logits(batch)[:1].float().clone()
        hid_e = batch.features.hidden.float().clone()
        rec_e, conv_e = pool.recurrent_states[:, slot].clone(), pool.conv_states[:, slot].clone()
        kv_e = []
        for layer in range(64):
            try:
                kv_e.append((self.kv_cache.k_cache(layer)[loc].float().clone(), self.kv_cache.v_cache(layer)[loc].float().clone()))
            except Exception:  # noqa: BLE001 -- past the last attention layer
                break
        pool.recurrent_states[:, slot] = rec0
        pool.conv_states[:, slot] = conv0
        graph = self.spec.prefill_as_verify(batch)
        hid_g = self.spec.proposer._prefill_sink.hidden.float()
        rel = lambda a, b: float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
        top = eager[0].topk(2)
        kv = [max(rel(self.kv_cache.k_cache(i)[loc], k), rel(self.kv_cache.v_cache(i)[loc], v)) for i, (k, v) in enumerate(kv_e)]
        logger.info_rank0(
            f"PREFILL_GRAPH_CHECK n={req.extend_len} cached={req.cached_len} argmax eager={int(eager[0].argmax())} graph={int(graph[0].argmax())} "
            f"equal={bool(eager[0].argmax() == graph[0].argmax())} eager_top2_gap={float(top.values[0] - top.values[1]):.4f} "
            f"logits_max_abs_diff={float((eager - graph.float()).abs().max()):.4f} logits_scale={float(eager.abs().max()):.2f} "
            f"hidden_rel={rel(hid_g, hid_e):.5f} rec_state_rel={rel(pool.recurrent_states[:, slot], rec_e):.5f} "
            f"conv_state_rel={rel(pool.conv_states[:, slot], conv_e):.5f} kv_rel_max={max(kv) if kv else -1:.5f} kv_layers={len(kv)} "
            f"kv_rel_first={kv[0] if kv else -1:.5f} kv_rel_last={kv[-1] if kv else -1:.5f} "
            f"rec_by_layer={[round(rel(pool.recurrent_states[i, slot], rec_e[i]), 5) for i in range(0, rec_e.shape[0], max(1, rec_e.shape[0] // 8))]} "
            f"conv_by_layer={[round(rel(pool.conv_states[i, slot], conv_e[i]), 5) for i in range(0, conv_e.shape[0], max(1, conv_e.shape[0] // 8))]} "
            f"hidden_by_row={[round(rel(hid_g[i], hid_e[i]), 5) for i in range(hid_e.shape[0])]} "
            f"hidden_cos_last={float(torch.nn.functional.cosine_similarity(hid_g[-1], hid_e[-1], dim=0)):.6f}"
        )
        return graph

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        if batch.mm_gather_plan:
            self._run_mm_encoder(batch)
        spec_prefill = self.spec is not None and batch.is_prefill
        if spec_prefill and self._prefill_runs_as_verify(batch):
            if os.environ.get("FREETOKEN_PREFILL_GRAPH_CHECK") == "split":
                logits = self.spec.prefill_as_verify(batch, check=True)
            elif os.environ.get("FREETOKEN_PREFILL_GRAPH_CHECK") not in (None, "", "selftest"):
                logits = self._check_prefill_as_verify(batch)
            else:
                logits = self.spec.prefill_as_verify(batch)
        else:
            if spec_prefill:
                batch.features = self.spec.feature_sink(batch)
            logits = self.forward_logits(batch)

        for req in batch.reqs:
            req.complete_one()

        batch_logits = logits[: batch.size]
        next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        if spec_prefill:
            # after the D2H is queued: the drain waits for the copy only, while the draft's work overlaps it
            self.spec.on_prefill(batch, next_tokens_gpu)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    @torch.inference_mode()
    def _warmup_prefill(self, autotune: bool = True, keeps: bool = False) -> None:
        """Compile the Triton prefill path before the first real request.

        Decode CUDA graph capture warms the decode path, but the first prefill
        can still pay Triton/cublas setup costs. Use the dummy request row and
        restore it afterwards so padded decode graph replay keeps using the
        dedicated dummy KV slot.
        """
        if self.max_seq_len < 2:
            return

        warmup_lens = [min(80, self.max_seq_len)]
        if self.max_seq_len >= 128 and autotune:
            # the second length is for the autotune cache; one forward streams every expert
            warmup_lens.append(128)
        warmup_lens = sorted({length for length in warmup_lens if length >= 2})
        if not warmup_lens:
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            for length in warmup_lens:
                dummy_row[:length] = torch.arange(
                    length, dtype=torch.int32, device=self.device
                )
                warm_req = Req(
                    input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=None,  # type: ignore[arg-type]
                    cache_handle=None,  # type: ignore[arg-type]
                )
                warm_req.linear_slot_idx = self.dummy_req.linear_slot_idx  # the padding state slot, not a free one
                batch = Batch(reqs=[warm_req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
                batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
                if self.config.model_config.model_is_mrope:
                    batch.mrope_positions = (
                        batch.positions.unsqueeze(0).expand(3, -1).contiguous()
                    )
                batch.out_loc = dummy_row[:length]
                self.attn_backend.prepare_metadata(batch)
                with self.ctx.forward_batch(batch):
                    self.model.forward()
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                if keeps:
                    # the warmup streamed every expert once and the cache kept what it had room for: the first request
                    # finds them resident. One more snapshot tells the cache whether that is every expert.
                    self.moe_offload_cache.begin_prefill()
                    self.moe_offload_cache.reset_stats()
                else:
                    self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"Prefill warmup complete for lengths {warmup_lens} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _profile_gpu(index: "int | None" = None) -> Tuple[str | None, str | None]:
    """(name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    ident = gpu_identity(torch.cuda.current_device() if index is None else index)
    return ident["name"], ident["uuid"]


def _is_unified_memory_gpu(index: "int | None" = None) -> bool:
    """True when the GPU has no separate device memory (cudaDevAttrIntegrated): host
    banks and the GPU slot cache are the same DRAM, so the offload family's pinned
    staging + slot gather are DRAM-to-DRAM copies with no PCIe link to hide behind.

    The attribute is reliable on true-UMA parts (Jetson, GB10/DGX Spark) but not on
    C2C-linked discrete-HBM parts (GH200 reports integrated=0 despite coherent CPU
    memory), and it has only been verified on GB10 so far --
    FREETOKEN_UNIFIED_MEMORY=0/1 overrides the probe where the attribute lies."""
    env = os.environ.get("FREETOKEN_UNIFIED_MEMORY")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    if not torch.cuda.is_available():
        return False
    try:
        dev = torch.cuda.current_device() if index is None else index
        return bool(torch.cuda.get_device_properties(dev).is_integrated)
    except Exception:
        return False


_RESIDENT_EXPERT_QUANTS = ("none", "fp8_block", "nvfp4", "gguf")


def _fused_resident_ok(model_config) -> bool:
    """Whether the resident ('fused') MoE path can hold this model's experts; DeepSeek-V4 experts stay offload."""
    if getattr(model_config, "expert_quant", "none") not in _RESIDENT_EXPERT_QUANTS:
        return False
    return getattr(model_config, "moe_weight_format", None) in (None, "bf16", "mxfp4")


def _ensure_expandable_segments() -> None:
    """Default the CUDA allocator to expandable segments.

    The motivating case is the offload prefill, which repeatedly dequantizes
    variable-sized NVFP4 expert blocks to BF16 (a different size per layer as the
    active-expert count varies). Under that alloc/free churn the default caching
    allocator fragments badly -- reserved memory can balloon far past the actual peak
    allocation (observed ~78GiB reserved for a <30GiB working set).
    ``expandable_segments`` lets freed regions of any size be reused, keeping
    reserved ~= allocated, so it is applied to every run, not just offload ones.

    Env vars are parsed once at import and ignored if set afterwards, so we apply the
    setting via the runtime API instead. Must run before the first CUDA allocation (the
    caller guarantees CUDA is not yet initialized). Any user-provided allocator config
    is respected and left untouched.
    """
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - depends on torch build
        logger.info_rank0(f"Could not enable expandable_segments ({exc}); continuing")
        return
    logger.info_rank0("Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)")


def _resolve_cache_type(has_linear_attention: bool, requested: str) -> str:
    # Hybrid GDN models default to the HybridRadixCache (snapshots GDN state at chunk
    # boundaries -> cross-request prefix reuse). An explicit ``--cache-type naive`` opts out
    # to the old no-reuse path (debugging / parity baseline / lower GDN-state memory).
    if has_linear_attention:
        return "naive" if requested == "naive" else "hybrid_radix"
    return requested


def _adjust_dsv41_config(config: EngineConfig, override) -> None:
    """DeepSeek-V4.1 config reconciliation, the DSV4 policy with the replay knob: sync the
    runtime into ``dsv41_args``, page_size = the window page P, radix -> swa_radix, decode graph
    batches <= max_running_req. Unlike DSV4 the prefill chunk keeps ``max_extend_tokens`` (whole
    window pages): at a 1M ceiling the window pool alone would admit ~40K-token chunks whose
    activations (64 x 512 latent queries per token) do not fit next to the expert cache."""
    model_config = config.model_config
    args = model_config.dsv41_args
    args.max_seq_len = config.max_seq_len
    args.max_batch_size = config.max_running_req + 1  # +1 dummy
    args.swa_decoder_replay = config.swa_decoder_replay
    P = args.window_size
    override("page_size", P)
    logger.info_rank0(
        f"DSV41 KV pages are {P}-token window pages; page_size set to {P}; SWA decoder replay: {args.swa_decoder_replay}"
        + ("" if args.swa_decoder_replay == "exact" else f" (prefix hits stop {P} tokens before the prompt end)")
    )
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    current = getattr(config, "max_extend_tokens", None)  # a SchedulerConfig field
    if current is not None and (extend := max(P, current // P * P)) != current:
        override("max_extend_tokens", extend)
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds max_running_req {mr}; clamping (larger decode batches never occur).")
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            logger.warning_rank0(f"dropping cuda_graph_bs entries above max_running_req {mr}: {[bs for bs in config.cuda_graph_bs if bs > mr]}")
            override("cuda_graph_bs", kept)


def _adjust_dsv4_config(config: EngineConfig, override) -> None:
    """DSV4 engine-config reconciliation at config-resolution time (before the pool exists).
    Syncs the resolved runtime config into the opaque ``dsv4_args`` payload, sets
    page_size to the window page P, forces single-chunk prefill, and clamps cuda_graph_bs/max_bs to
    the DSV4 decode batch size.
    """
    model_config = config.model_config
    model_config.dsv4_args.max_seq_len = config.max_seq_len
    model_config.dsv4_args.max_batch_size = config.max_running_req + 1  # +1 dummy
    # config.swa_full_tokens_ratio is the DSV4 window/full ratio directly (default sizing);
    # a runtime rebuild pins an absolute window via swa_num_pages_override instead.
    # DSV4's KV page IS the P-token window page (window == radix reuse granularity == lcm of
    # the compress ratios), so max_num_tokens = num_pages * page_size holds like every model.
    P = model_config.dsv4_args.window_size
    override("page_size", P)
    logger.info_rank0(f"DSV4 KV pages are {P}-token window pages; page_size set to {P}")
    # The generic CacheManager materializes DSV4 'radix' as the shared SWARadixCache (is_swa);
    # 'naive' stays naive with the pool's swa currency riding swa_paged.
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    # 'radix' (SWARadixCache on the full-loc currency, carry-aware re-prefill) is the default and is
    # honored, as is an explicit 'naive'. Don't let max_extend_tokens force a second chunk within
    # one prompt (the pool's prefill_chunk_budget still chunks prompts larger than the window
    # pool); prefill batches ragged (bs>=1), each segment resuming from its own cached_len.
    if getattr(config, "max_extend_tokens", 0) < config.max_seq_len:
        override("max_extend_tokens", config.max_seq_len)

    # DSV4 decode batches at most max_running_req rows; its full-loc snapshot is sized to that,
    # so a graph bs above it would exceed the backend's captured snapshot rows. Clamp any
    # oversized explicit list / max_bs here (before GraphRunner ever sees it).
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(
            f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds DSV4 max_running_req {mr}; "
            "clamping to max_running_req (larger decode batches never occur)."
        )
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            dropped = [bs for bs in config.cuda_graph_bs if bs > mr]
            logger.warning_rank0(
                f"dropping cuda_graph_bs entries {dropped} above DSV4 max_running_req {mr} "
                "(larger decode batches never occur)."
            )
            override("cuda_graph_bs", kept)


def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count
    (``"8"`` -> 8 layers evenly strided across depth), or a fraction (``"0.5"``). Ids are
    indices into the MoE layers, ``[0, num_moe_layers)``."""
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"--moe-cpu-layers id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"--moe-cpu-layers fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(f"--moe-cpu-layers count {k} must be in [0, {num_moe_layers}]")
    # k layers spread evenly across depth (frozenset dedups any rounding collisions;
    # k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int, *, reserved: int = 0, method=None) -> frozenset[int]:
    """MoE layer ids whose decode runs on the CPU executor.

    ``--moe-strategy cpu`` -> every layer. ``--moe-strategy offload`` + ``--moe-cpu-layers``
    -> the parsed subset, or the pin-budget pick for ``auto`` (the rest stay on the GPU offload/PCIe path). Otherwise none.
    """
    if config.moe_strategy == "cpu":
        return frozenset(range(num_moe_layers))
    spec = config.moe_cpu_layers
    if not spec or not is_offload_moe_strategy(config.moe_strategy):
        return frozenset()
    if spec.strip() == "auto":
        return _auto_cpu_layers(config, num_moe_layers, reserved=reserved, method=method)
    return _parse_cpu_layers_spec(spec, num_moe_layers)


def _decode_target(config: EngineConfig) -> str:
    """Where routed experts decode, from the flags alone: hybrid co-compute, the CPU executor for some or all layers, or the GPU slot cache."""
    if config.moe_strategy == "hybrid":
        return "hybrid"
    if config.moe_strategy == "cpu" or (config.moe_cpu_layers and is_offload_moe_strategy(config.moe_strategy)):
        return "cpu"
    return "gpu"


# settings the decode-profile key renders itself, or that the row decides
_DECODE_KEY_OWN = frozenset((
    "model_path", "tp_info", "moe_strategy", "moe_backend", "moe_cache_size", "moe_cache_rate", "moe_cache_auto", "moe_cache_mem",
    "memory_ratio", "moe_cpu_threads", "max_running_req", "speculative_algorithm", "speculative_num_steps", "spec_slot_states",
    "moe_decode_key", "moe_decode_row", "preset", "preset_fields",
))


def _moe_cache_is_auto(config: EngineConfig) -> bool:
    # no sizing flag becomes --moe-cache-auto a few lines after auto resolves, so it counts as auto before that too.
    # A --moe-cache-rate is a size, also before _adjust_config has made it one.
    return config.moe_cache_auto or (config.moe_cache_size <= 0 and config.moe_cache_mem is None and not config.moe_cache_rate)


def _decode_decision_key(config: EngineConfig) -> str:
    """This setup's key in the decode profile (moe/decode_profile.py): the checkpoint, the cache, the CPUs, speculative decoding, and every other engine setting that is not at its default."""
    from freetoken.moe import decode_profile

    spec = f"{config.speculative_algorithm}:{config.speculative_num_steps}" if config.speculative_algorithm is not None else "off"
    return decode_profile.decision_key(
        model=decode_profile.model_fingerprint(config.model_path),
        cache=decode_profile.cache_spec(
            size=config.moe_cache_size, mem=config.moe_cache_mem, auto=_moe_cache_is_auto(config), memory_ratio=config.memory_ratio,
            rate=config.moe_cache_rate,
        ),
        cpu=decode_profile.cpu_spec(config.moe_cpu_threads),
        spec=spec,
        tp_size=config.tp_info.size,
        max_running_req=config.max_running_req,
        extra=decode_profile.non_default_settings(config, EngineConfig, skip=_DECODE_KEY_OWN),
        env=decode_profile.perf_env(),
    )


def _gpu_free_bytes() -> int | None:
    """Free memory of the bound GPU when the decode-profile key is made: before anything is loaded."""
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.mem_get_info(torch.cuda.current_device())[0])


def _measured_decode_strategy(config: EngineConfig) -> tuple[str, dict | None]:
    """``(strategy, row)`` for --moe-strategy auto on a model that would take offload: hybrid only on a decode row measured for this setup, on this machine and release (moe/decode_profile.py); offload otherwise, as before."""
    from freetoken.moe import decode_profile

    def offload(reason: str) -> tuple[str, None]:
        logger.info_rank0(f"--moe-strategy auto: offload ({reason}); `ft bench decode` measures whether hybrid is faster for this setup")
        return "offload", None

    if config.tp_info.size > 1:
        # each rank binds its own GPU and would read that GPU's rows: the ranks could disagree
        return offload("tensor parallel runs are not picked from decode rows")
    if not _cpu_moe_executor_viable(config.model_config):
        return offload("the CPU MoE executor cannot serve this model")
    gpu_name, gpu_uuid = _profile_gpu()
    row, reason = decode_profile.lookup(
        config.moe_decode_key, gpu_name=gpu_name, gpu_uuid=gpu_uuid, now=lambda: decode_profile.stamp(gpu_name, gpu_uuid)
    )
    floor = config.model_config.num_experts
    if row is not None and decode_profile.slot_cap(row, floor) < floor:
        # a hand-edited row below the cache's floor would cap the cache under it and fail the start
        row, reason = None, f"the decode row's hybrid cache is below this model's floor of {floor} slots"
    if row is None:
        return offload(reason)
    # the rounding mode this server will run (moe/offload_kernels.py reads the same variable)
    round_mode = (os.getenv("FREETOKEN_HYBRID_FETCH_ROUND") or "min-wait").strip().lower()
    verdict = decode_profile.verdict_for(row, round_mode)
    if verdict["strategy"] != "hybrid":
        other = decode_profile.verdict_for(row, None)
        also = f"; hybrid with {other['round']} rounding qualifies (FREETOKEN_HYBRID_FETCH_ROUND={other['round']})" if other["strategy"] == "hybrid" else ""
        logger.info_rank0(f"--moe-strategy auto: {decode_profile.explain(row, verdict)}{also}")
        return "offload", None
    if _moe_cache_is_auto(config):
        if objection := decode_profile.auto_size_objection(row, _gpu_free_bytes(), config.memory_ratio):
            return offload(objection)
    logger.info_rank0(f"--moe-strategy auto: {decode_profile.explain(row, verdict)}")
    return "hybrid", row


# expert activations the CPU MoE executor supports (csrc ActKind)
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
    "swiglu_clamp",
)


def _cpu_moe_executor_viable(model_config) -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in CpuMoeExecutor after the whole load; explicit cpu/hybrid/--moe-cpu-layers picks still fail loudly."""
    from freetoken.moe.cpu_executor import compiled_extension_supports

    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        return False
    act = getattr(model_config, "hidden_act", "silu")
    moe_wfmt = getattr(model_config, "moe_weight_format", None)
    if act not in _CPU_MOE_ACTS and moe_wfmt != "mxfp4":
        return False
    if moe_wfmt != "mxfp4" and not compiled_extension_supports(act):
        return False
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
    if fmt == "gguf":
        from freetoken.kernel import ggml_cpu

        return fmt in _cpu_moe.weight_formats() and ggml_cpu.select() is not None
    return fmt == "mxfp4" or fmt in _cpu_moe.weight_formats()


def _pin_budget_bytes(reserved: int = 0) -> int | None:
    """Bytes this process can still safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux).

    WSL's WDDM-backed CUDA caps pinning near half of RAM, shared across processes -- budget 40%. FREETOKEN_PIN_BUDGET_GB overrides anywhere. ``reserved`` subtracts host bytes already pinned outside the expert banks (qwen4_exp's PLE table)."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        cap = int(float(env) * 2**30)
    elif not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():  # WSL kernel tag
        return None
    else:
        cap = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)
    return max(0, cap - reserved)


def _bank_bytes(config: EngineConfig, method=None, layer_methods=None) -> int | None:
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    return ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config, method, layer_methods)


def _pin_hint(reserved: int) -> str:
    if _pin_budget_bytes(reserved) is None:
        return "the expert banks need more page-locked host RAM than this host has; free host RAM or serve a smaller model"
    return (
        "pass --moe-cpu-layers auto to lock the layers over the pin budget for CPU decode, "
        "or --moe-cpu-layers <count|fraction|ids> to choose them yourself"
    )


def _check_pin_budget(config: EngineConfig, *, reserved: int, method=None, layer_methods=None) -> None:
    """Stop a plain offload boot whose banks exceed a known pin budget before any bank is read."""
    if config.moe_cpu_layers or config.moe_strategy not in ("offload", "hybrid"):
        return
    budget = _pin_budget_bytes(reserved)
    bank_bytes = _bank_bytes(config, method, layer_methods) if budget is not None else None
    if bank_bytes and bank_bytes > budget:
        raise ValueError(
            f"expert banks need {bank_bytes / 2**30:.1f} GiB of pinned host RAM but the pin budget is "
            f"{budget / 2**30:.1f} GiB (WSL caps CUDA pinning; FREETOKEN_PIN_BUDGET_GB overrides); {_pin_hint(reserved)}"
        )


def _auto_cpu_layers(config: EngineConfig, num_moe_layers: int, *, reserved: int = 0, method=None) -> frozenset[int]:
    """Pick CPU (locked) MoE layers for ``--moe-cpu-layers auto``: none while the banks fit the pin budget.

    Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the cheapest to move off the slot cache."""
    bank_bytes = _bank_bytes(config, method)
    if not bank_bytes:
        return frozenset()
    budget = _pin_budget_bytes(reserved)
    if budget is None or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable(config.model_config):
        raise ValueError(
            f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB exceed the "
            f"pin budget {budget / 2**30:.2f} GiB, but the CPU MoE executor cannot "
            f"serve this model (see --moe-strategy cpu requirements)"
        )
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(range(num_moe_layers - (n - head), num_moe_layers))
    logger.info_rank0(
        f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB > pin budget "
        f"{budget / 2**30:.2f} GiB; locking {n} head+tail MoE layers for CPU decode "
        f"({sorted(ids)})"
    )
    return ids


# MoE-only knobs and the value each resolves to on a dense model. moe_strategy is handled
# separately (its dense value is 'fused', but 'auto' resolves there without a warning).
_DENSE_MOE_SETTINGS = {
    "moe_cache_size": 0,
    "moe_cache_rate": None,
    "moe_cache_auto": False,
    "moe_cache_mem": None,
    "moe_cache_scope": "unified",
    "moe_cache_split": "equal",
    "moe_cpu_layers": None,
    "moe_cpu_threads": 0,
    "moe_hybrid_max_fetch": -1,
    "moe_locked_fetch": -1.0,
    "moe_locked_max_fetch": 8,
    "moe_prefill_demand_tokens": 0,
    "moe_prefill_overlap": True,
    "moe_prefill_hit_d2d": False,
    "expert_load": "auto",
}


def _adjust_ftw_quant_backend(model_path: str, quant_backend: QuantBackend) -> QuantBackend:
    """--quant-backend with an FTW checkpoint's packed expert kernel filled in where the flag leaves that table automatic.

    An entry that names another kernel is refused here, before any bank is read."""
    from freetoken.checkpoint.ftw import ftw_quant_format
    from freetoken.moe.legacy_format import kind_kernel_for

    with _weight_load_context():
        fmt = ftw_quant_format(model_path) if model_path else None
    if fmt is None:
        return quant_backend
    try:
        kind, kernel = kind_kernel_for(fmt)
    except KeyError:
        return quant_backend
    requested = quant_backend.select(LayerKind.MOE, kind)
    if requested == kernel:
        return quant_backend
    if requested != "auto":
        raise ValueError(
            f"the FTW checkpoint's expert banks were packed for {kind} / {kernel} but --quant-backend asks for "
            f"{requested}; drop the entry, or reconvert with ft checkpoint --quant-backend moe.{kind}={requested}"
        )
    return QuantBackend(quant_backend.items + (((LayerKind.MOE, kind), kernel),))


def offload_layer_methods(model) -> list | None:
    """The expert method of each offload MoE layer of ``model``, by MoE-layer id; None for models whose MoE layers carry none (GGUF)."""
    layers = [l for l in iter_offload_moe_layers(model) if getattr(l, "quant_method", None) is not None]
    if not layers:
        return None
    layers.sort(key=lambda layer: layer.layer_id)
    assert [layer.layer_id for layer in layers] == list(range(len(layers))), [layer.layer_id for layer in layers]
    return [layer.quant_method for layer in layers]


def shared_offload_method(model):
    """The expert method the offload MoE layers of ``model`` use, or None for models whose MoE layers carry none (GGUF).

    With experts of several formats (a checkpoint that keeps some layers unquantized) this is the format of the most
    layers; the cache then takes every layer's method (``offload_layer_methods``) and keeps one slot pool per format."""
    from freetoken.moe.expert_banks import method_groups

    methods = offload_layer_methods(model)
    if methods is None:
        return None
    groups = method_groups(methods)
    found = ", ".join(f"{m.kind} via {m.kernel.name}" + (f" on {len(ids)} layers" if len(groups) > 1 else "") for m, ids in groups)
    logger.info_rank0(f"MoE experts: {found}")
    return groups[0][0]


def _slot_limit(layer_methods) -> int | None:
    """The tightest slot cap of the layers' expert kernels (a pool never mixes kernels, but the limit is checked per pool against the largest)."""
    limits = [limit for limit in (m.slot_limit() for m in layer_methods) if limit is not None]
    return min(limits) if limits else None


def _check_layer_formats(banks, layer_methods) -> None:
    from freetoken.moe.legacy_format import legacy_format_for

    want = [legacy_format_for(m.kind, m.kernel.name) for m in layer_methods]
    have = banks.layer_formats or [banks.quant_format] * len(want)
    bad = [layer for layer, (w, h) in enumerate(zip(want, have)) if w != h]
    if bad:
        raise ValueError(
            f"expert banks of MoE layers {bad[:8]} were packed as {[have[l] for l in bad[:8]]} but the model binds "
            f"{[want[l] for l in bad[:8]]}; reconvert the checkpoint or select those kernels"
        )


def _meta_model(config: EngineConfig, decode_target: str):
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    set_quant_backend(_adjust_ftw_quant_backend(config.model_path, QuantBackend.parse(config.quant_backend)))
    object.__setattr__(config.model_config, "moe_strategy", config.moe_strategy)
    object.__setattr__(config.model_config, "decode_target", decode_target)
    with torch.device("meta"), torch_dtype(config.dtype):
        return create_model(config.model_config)


def offload_expert_methods(config: EngineConfig) -> tuple:
    """``(method, layer_methods)`` of ``config``'s model, from a meta-device build: the most common expert method and,
    for experts of several formats, one per MoE layer (else None).

    For tools that load expert banks without an engine (the FTW converter); they pack for GPU decode."""
    from freetoken.moe.expert_banks import is_mixed

    model = _meta_model(config, "gpu")
    layer_methods = offload_layer_methods(model)
    return shared_offload_method(model), (layer_methods if is_mixed(layer_methods) else None)


def meta_offload_layer_methods(config: EngineConfig, decode_target: str) -> list | None:
    """The expert method of each offload MoE layer of ``config``'s model, from a meta-device build for ``decode_target``; see ``offload_layer_methods``."""
    return offload_layer_methods(_meta_model(config, decode_target))


def offload_expert_method(config: EngineConfig):
    """The (most common) offload expert method of ``config``'s model; see ``offload_expert_methods``."""
    return offload_expert_methods(config)[0]


def _check_reproducible_seeds(config: EngineConfig, is_moe: bool) -> None:
    """Refuse the settings under which a seeded request's logits follow what the server did before it.

    --reproducible-seeds says the text follows from the flags and the request alone, so a setting this cannot hold for
    stops the start instead of serving text that only sometimes repeats."""
    if not getattr(config, "reproducible_seeds", False):  # duck-typed test configs omit it
        return
    if config.max_running_req != 1:
        raise ValueError(
            f"--reproducible-seeds needs --max-running-requests 1 (got {config.max_running_req}): the logits of a step "
            "change in the last bit with the rows the step holds, so a seeded request has to run alone"
        )
    if config.speculative_algorithm == "dflash":
        raise ValueError(
            "--reproducible-seeds does not work with --speculative-algorithm dflash: its sampled drafts and their "
            "accept test draw from the stream every request shares; use mtp or no speculative decoding"
        )
    if is_moe and _decode_target(config) != "gpu":
        raise ValueError(
            "--reproducible-seeds needs every expert computed on the GPU (--moe-strategy offload or fused, without "
            "--moe-cpu-layers): hybrid computes on the CPU the experts that are not resident, which follows earlier "
            "requests, and the CPU kernels round differently from the GPU's"
        )


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    model_config = config.model_config
    single_stream_only = getattr(model_config, "single_stream_only", False)
    is_dsv4 = getattr(model_config, "dsv4_args", None) is not None
    is_dsv41 = getattr(model_config, "dsv41_args", None) is not None
    has_swa_attention = getattr(model_config, "has_swa_attention", False)
    has_linear_attention = getattr(model_config, "has_linear_attention", False)
    is_moe = getattr(model_config, "is_moe", False)
    expert_quant = getattr(model_config, "expert_quant", "none")
    preset_fields = getattr(config, "preset_fields", ())  # duck-typed test configs omit it

    if not is_moe:
        # A dense model has no routed experts: the MoE knobs are inert, and the offload family
        # is worse than inert -- engine init would build an expert cache for a model that has
        # none and abort startup (weights already resident) on an unrelated expert-source
        # error. Drop them at this one choke point, which the CLI and the programmatic
        # LLM(...) path both pass through. 'auto'/'fused' is the silent dense resolution;
        # anything else was asked for explicitly, so report what is being ignored.
        dropped = [
            f"{name}={getattr(config, name)!r}"
            for name, dense_value in _DENSE_MOE_SETTINGS.items()
            if getattr(config, name, dense_value) != dense_value and name not in preset_fields
        ]
        if config.moe_strategy not in ("auto", "fused"):
            dropped.insert(0, f"moe_strategy={config.moe_strategy!r}")
        override("moe_strategy", "fused")
        for name, dense_value in _DENSE_MOE_SETTINGS.items():
            override(name, dense_value)
        if dropped:
            logger.warning_rank0(
                f"{getattr(model_config, 'model_type', 'model')} is a dense model (no routed "
                f"experts); ignoring MoE settings: {', '.join(dropped)}"
            )

    if single_stream_only:
        # The model runs one sequence at a time: it collapses the batch to one row and the
        # decode CUDA graph is captured at bs=1. Force the runtime knobs so the KV pool, page
        # table and graph capture all stay bs=1.
        if config.max_running_req != 1:
            override("max_running_req", 1)
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs >= 1:
            override("cuda_graph_bs", [1])
            override("cuda_graph_max_bs", 1)

    if config.cuda_graph_max_bs is None:
        override("cuda_graph_max_bs", config.max_running_req)

    if is_dsv4:
        _adjust_dsv4_config(config, override)
    if is_dsv41:
        _adjust_dsv41_config(config, override)

    if has_swa_attention:
        # Both SWA cache paths use the global-paged swa pool (page_size==1 only for now).
        if config.page_size != 1:
            raise ValueError(
                f"SWA models currently support only page_size=1, got {config.page_size}."
            )
        # naive keeps cache_type='naive' (NaivePrefixCache, no reuse) on the paged pool (==
        # sglang SWAChunkCache); radix materializes as swa_radix (SWARadixCache, cross-request
        # reuse == sglang SWARadixCache). Both allocate from the same swa pool + free out-of-window.
        if getattr(config, "cache_type", "radix") != "naive":
            if not 0.0 < config.swa_full_tokens_ratio <= 1.0:
                raise ValueError(
                    f"swa_full_tokens_ratio must be in (0, 1], got {config.swa_full_tokens_ratio}"
                )
            override("cache_type", "swa_radix")

    if has_linear_attention:
        override(
            "cache_type",
            _resolve_cache_type(True, getattr(config, "cache_type", "radix")),
        )

    # Type x backend capability matrix: resolve auto from the per-type priority
    # lists, then validate whatever is now selected (explicit or auto) -- every
    # comma part must serve every required type, with packages/arch available.
    required_attn_types = _required_attn_types(model_config)
    _dtype = getattr(config, "dtype", None)  # duck-typed test configs omit it
    if (
        required_attn_types & {AttnType.BSA, AttnType.QSA}
        and _dtype is not None
        and _dtype.itemsize != 2
    ):
        # Reject at config time: the BSA/QSA pool's own assert only fires after the
        # model is resident (and not at all under `python -O`).
        raise ValueError(
            f"--dtype {config.dtype}: block-sparse attention serves 16-bit "
            "compute only (the index slab budgets 2 bytes/token); use bfloat16 "
            "or float16."
        )
    if _dtype == torch.float16 and "mxfp8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
    ):
        # The MXFP8 GEMV folds the pow2-descaled fp8 weight into the activation
        # dtype; fp16's narrow exponent can overflow/flush what bf16 represents
        # exactly, and the combination was never numerically validated.
        raise ValueError(
            "--dtype float16 with MXFP8 resident weights is unsupported (the "
            "W8A16 fold is only validated exact in bfloat16); use bfloat16."
        )
    if _dtype == torch.float16 and getattr(getattr(model_config, "quant", None), "weight_scales", False):
        # GGUF NVFP4 results are stored in the activation dtype before weight_scale_2 (~1e-4) multiplies them, which overflows fp16.
        raise ValueError("--dtype float16 with NVFP4 GGUF weight scales is unsupported (the unscaled results overflow float16); use bfloat16.")
    _validate_kv_format(config, model_config)
    if config.attention_backend == "auto":
        override(
            "attention_backend",
            _resolve_auto_attention_backend(required_attn_types, getattr(config, "kv_format", NONE)),
        )
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    _validate_attention_backend_choice(config, override, required_attn_types)
    if getattr(config, "speculative_algorithm", None) is not None:
        adjust_spec = _adjust_preset_spec_config if "speculative_algorithm" in preset_fields else _adjust_spec_config
        adjust_spec(config, override, is_dsv4 or is_dsv41 or has_swa_attention)
    if _preset_capture_bytes(config) and getattr(config, "speculative_algorithm", None) is None:
        _drop_preset_captures(config, "they are captured beside the verify step of speculative decoding, which is off")

    if config.moe_cache_rate is not None:
        total_experts = config.model_config.num_moe_layers * config.model_config.num_experts
        override("moe_cache_size", math.ceil(total_experts * config.moe_cache_rate))

    # The CPU MoE executor supports the silu/gelu family plus the clamped
    # swigluoai (csrc ActKind; "gpt_oss_swiglu" rides inside the mxfp4 kernel and
    # swigluoai the generic GEMV epilogue). A model with any other expert
    # activation cannot decode on the CPU: reject an explicit cpu/hybrid pick at
    # config time.
    # hidden_act (the dense activation) stands proxy for the expert activation --
    # true for every in-tree model. mxfp4 experts pass regardless: their act runs
    # inside the mxfp4 kernel, not the generic epilogue.
    _cpu_moe_act_ok = getattr(model_config, "hidden_act", "silu") in _CPU_MOE_ACTS or (
        getattr(model_config, "moe_weight_format", None) == "mxfp4"
    )
    if (
        is_moe
        and not _cpu_moe_act_ok
        and (config.moe_strategy in ("cpu", "hybrid") or config.moe_cpu_layers)
    ):
        asked = (
            f"--moe-cpu-layers={config.moe_cpu_layers!r}"
            if config.moe_strategy not in ("cpu", "hybrid")
            else f"--moe-strategy {config.moe_strategy!r}"
        )
        raise ValueError(
            f"{asked}: the CPU MoE executor does not support this model's expert "
            f"activation {getattr(model_config, 'hidden_act', None)!r}; drop the flag "
            "and let every layer decode on the GPU offload path instead."
        )

    bench_server = config.moe_strategy in ("offload", "hybrid") and os.getenv("FREETOKEN_BENCH_DECODE_KEY") == "1"
    if is_moe and (config.moe_strategy == "auto" or bench_server) and isinstance(config, EngineConfig):
        # the key auto looks its decode row up under; `ft bench decode` asks its offload and hybrid servers for the same key
        override("moe_decode_key", _decode_decision_key(config))
        free = _gpu_free_bytes()
        logger.info_rank0(f"MoE decode setup: {config.moe_decode_key}" + (f"; {free} bytes of GPU memory free" if free is not None else ""))

    if is_moe and config.moe_strategy == "auto":
        # A MoE model always defaults to the offload family: experts stream from pinned host
        # banks into an auto-sized GPU slot cache, which is the only default that serves a model
        # bigger than the GPU. The resident 'fused' path (every expert format but
        # DeepSeek-V4's) is still reachable, but only when asked for explicitly
        # -- auto never picks it, because nothing here knows whether the experts would fit in
        # HBM and a wrong guess is a weight-load OOM rather than a slower-but-working run.
        default_backend = "offload"
        # Unified memory (GB10/DGX Spark, Jetson): there is no host/device boundary, so
        # the offload family stages and gathers between two names for the same DRAM (on
        # GB10 this added a measured ~130 s stall to every request, #369). Resident
        # experts are the safe default here, not the risky one: the model and its banks
        # page from the same pool, so the "wrong guess = weight-load OOM" rationale above
        # does not apply. Only formats the resident path
        # can actually hold take this branch; the rest stay on offload as before.
        unified_memory = _is_unified_memory_gpu()
        if unified_memory and _fused_resident_ok(model_config):
            default_backend = "fused"
            logger.info_rank0(
                "Unified-memory GPU detected; auto-selecting 'fused' MoE strategy "
                "(resident experts) instead of offload"
            )
            # the cache sizing flags get their own warning in the 'fused' block below
            inert = [flag for flag, on in (
                (f"--moe-cpu-layers={config.moe_cpu_layers}", config.moe_cpu_layers),
                ("--moe-prefill-hit-d2d", config.moe_prefill_hit_d2d and "moe_prefill_hit_d2d" not in preset_fields),
                ("--disable-moe-prefill-overlap", not config.moe_prefill_overlap),
            ) if on]
            if inert:
                logger.warning_rank0(
                    f"Unified-memory GPU serves MoE experts resident; ignoring {', '.join(inert)} "
                    "(use --moe-strategy offload to keep the offload flags)"
                )
                override("moe_cpu_layers", None)
        if default_backend == "offload" and _preset_takes_fused(config):
            # the one case where it is known that the experts fit: no slot cache, and a prefill reads them in place
            default_backend = "fused"
        if default_backend == "offload" and getattr(config, "moe_decode_key", None) is not None:
            # hybrid only on a decode row measured for this setup, on this machine and release; offload without one
            default_backend, row = _measured_decode_strategy(config)
            override("moe_decode_row", row)
        override("moe_strategy", default_backend)
        logger.info_rank0(f"Auto-selected MoE strategy: {config.moe_strategy}")

        if (
            is_offload_moe_strategy(config.moe_strategy)
            and config.moe_cache_size <= 0
            and config.moe_cache_rate is None
            and config.moe_cache_mem is None
            and not getattr(config, "moe_cache_auto", False)
        ):
            # args.py's "no sizing flag -> default --moe-cache-auto" only fires when the
            # backend is already offload-family at *parse* time. A bare `ft serve <FTW MoE
            # checkpoint>` (no --moe-strategy, no cache flags) still has moe_strategy=="auto" at
            # parse time -- the auto -> offload/cpu/hybrid resolution above is the first point
            # the concrete backend is known, so mirror the same default here: no sizing flag
            # was given, so let the scheduler resolve the cache size from free VRAM instead of
            # failing the _require_offload_cache_size guard with size=0.
            override("moe_cache_auto", True)
            logger.info_rank0(
                "No MoE cache sizing flag given; defaulting to --moe-cache-auto for "
                f"auto-selected strategy {config.moe_strategy!r}"
            )

    if (
        is_moe
        and config.moe_strategy == "offload"
        and _is_unified_memory_gpu()
        and _fused_resident_ok(model_config)
    ):
        # An explicit offload pick is honored, but on unified memory the user is paying
        # for copies between two names for the same DRAM; say so once at config time.
        logger.warning_rank0(
            "--moe-strategy offload on a unified-memory GPU: expert 'streaming' copies "
            "DRAM to DRAM (there is no PCIe link to overlap it with). If the model fits, "
            "--moe-strategy fused avoids the slot-cache machinery entirely."
        )

    if is_moe and config.moe_strategy == "fused":
        _drop_preset_demand(config, "every expert is resident")
        # An explicit 'fused' keeps the experts resident, so there is no slot cache to size. The
        # sizing flags no longer redirect the backend, so ignore them here and say so -- the
        # geometry the user asked for is what runs. Report the flag actually passed: --moe-cache-
        # rate was already folded into moe_cache_size above, and the sizing flags are mutually exclusive.
        if config.moe_cache_rate:
            inert = f"--moe-cache-rate={config.moe_cache_rate}"
        elif config.moe_cache_size:
            inert = f"--moe-cache-size={config.moe_cache_size}"
        elif config.moe_cache_mem is not None:
            inert = f"--moe-cache-mem={config.moe_cache_mem}"
        elif getattr(config, "moe_cache_auto", False):
            inert = "--moe-cache-auto"
        else:
            inert = None
        if inert:
            logger.warning_rank0(
                f"MoE backend 'fused' keeps its experts resident; ignoring {inert} "
                "(use --moe-strategy offload to serve experts from a slot cache)"
            )
            override("moe_cache_size", 0)
            override("moe_cache_rate", None)
            override("moe_cache_mem", None)
            override("moe_cache_auto", False)

    if is_moe and config.moe_strategy == "cpu":
        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
        num_experts = config.model_config.num_experts
        if getattr(config, "moe_cache_auto", False):
            override("moe_cache_auto", False)
        override("moe_cache_mem", None)
        override("moe_cache_size", 2 * num_experts)
        override("moe_prefill_overlap", True)
        logger.info_rank0(
            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
        )

    if (
        is_moe
        and expert_quant not in _RESIDENT_EXPERT_QUANTS
        and not is_offload_moe_strategy(config.moe_strategy)
    ):
        raise ValueError(
            f"{expert_quant} experts require --moe-strategy offload or cpu, "
            f"got {config.moe_strategy!r}"
        )

    if is_moe and config.moe_cpu_layers and config.moe_strategy not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-strategy offload or hybrid (got "
            f"{config.moe_strategy!r}); use --moe-strategy cpu to run all layers on CPU"
        )

    if is_moe and config.moe_cpu_layers and config.moe_cpu_layers.strip() != "auto":
        if not _parse_cpu_layers_spec(config.moe_cpu_layers, model_config.num_moe_layers):
            override("moe_cpu_layers", None)

    if is_moe:
        object.__setattr__(model_config, "moe_strategy", config.moe_strategy)
        object.__setattr__(model_config, "decode_target", _decode_target(config))

    # Must stay LAST: page_size is only final here (_adjust_dsv4_config sets P=128, the
    # TRTLLM block sets 64). Also covers the programmatic LLM(...) path that bypasses parse_args.
    if config.num_token_override is not None:
        if config.num_page_override is not None:
            raise ValueError("--num-tokens and --num-pages are mutually exclusive")
        if config.num_token_override % config.page_size != 0:
            raise ValueError(
                f"--num-tokens {config.num_token_override} is not a multiple of the resolved "
                f"page size {config.page_size}; nearest valid values: "
                f"{config.num_token_override // config.page_size * config.page_size} or "
                f"{(config.num_token_override // config.page_size + 1) * config.page_size}"
            )
        override("num_page_override", config.num_token_override // config.page_size)

    # The rope cos/sin table covers rotary_config.table_positions, and neither rope kernel
    # bounds-checks the position it gathers with -- a longer ceiling reads past the table.
    # DSV4 is exempt: it sizes its own table from the resolved max_seq_len (_adjust_dsv4_config).
    rotary = getattr(model_config, "rotary_config", None)
    seq_override = getattr(config, "max_seq_len_override", None)
    if seq_override is not None and rotary is not None and not (is_dsv4 or is_dsv41):
        if seq_override > rotary.table_positions:
            raise ValueError(
                f"--max-seq-len-override {seq_override} exceeds the model's "
                f"rope table ({rotary.table_positions} positions). Serving past it would read "
                "out of bounds; extend the rope with --hf-overrides (YaRN rope_parameters) instead."
            )

    _check_reproducible_seeds(config, is_moe)

    # The startup ServerArgs dump is the *requested* config, printed in the frontend process
    # before any of the resolution above ran -- so "moe_strategy='auto'" is all it can say. This
    # is the one line that reports what actually runs, for every path (explicit backends never
    # hit an "Auto-selected ..." log at all).
    resolved = [
        f"attention_backend={config.attention_backend!r}",
        f"cache_type={getattr(config, 'cache_type', 'radix')!r}",
        f"page_size={config.page_size}",
    ]
    if is_moe:
        resolved.insert(0, f"moe_strategy={config.moe_strategy!r}")
    if getattr(config, "reproducible_seeds", False):
        resolved.append("reproducible_seeds=True")
    logger.info_rank0(f"Resolved config: {', '.join(resolved)}")
