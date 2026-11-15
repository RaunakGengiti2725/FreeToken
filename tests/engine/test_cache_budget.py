from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import os

from freetoken.engine.cache_budget import expert_bytes_per_slot, plan_cache_budget, resolve_moe_cache_auto
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1,
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_caps_slots_at_the_kernel_limit():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, max_slots=992,
    )
    assert size == 992


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton" if torch.version.hip is not None else "fi"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16,
    )
    assert (size, pages, overlap) == expected

    class StubMethod:
        def slot_limit(self):
            return 5

    size, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks(), StubMethod())
    assert size == 5


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_strategy="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    overrides.setdefault("model_path", "/tmp/freetoken-test-model")
    overrides.setdefault("tp_info", DistributedInfo(rank=0, size=1))
    config = EngineConfig(
        dtype=torch.bfloat16,
        attention_backend="triton" if torch.version.hip is not None else "fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",
            moe_strategy="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend(monkeypatch):
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_strategy="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    # a unified-memory host (GB10) resolves auto to fused instead
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)
    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim. Assert the family, not the member.
    assert is_offload_moe_strategy(config.moe_strategy)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


# ---- --moe-strategy auto and the decode rows `ft bench decode` files (moe/decode_profile.py), through the real _adjust_config ----

_ROW_GPU = ("FAKE GPU", "GPU-fake")
_ROW_FREE = 15 << 30  # GPU memory free when the row was measured


def _passes(*columns):
    """Three passes a column, 1 tok/s apart around the given tok/s: the shape of a row's ``tok_s``."""
    return [[v - 1.0, v, v + 1.0] for v in columns]


_HYBRID_FASTER = {"offload": _passes(100.0, 101.0), "offload#2": _passes(100.5, 100.0), "hybrid:min-wait": _passes(131.0, 128.0)}
_HYBRID_SLOWER = {"offload": _passes(84.0, 83.0), "offload#2": _passes(84.5, 83.2), "hybrid:min-wait": _passes(27.0, 28.0)}
# a workload column and a guard column (output that barely misses the cache): min-wait leads by more on the first and is 17% behind on the second
_TWO_ARMS = {"offload": _passes(100.0, 230.0), "offload#2": _passes(100.4, 231.0), "hybrid:min-wait": _passes(133.0, 191.0),
             "hybrid:stochastic": _passes(130.0, 219.0)}


@pytest.fixture
def decode_rows(tmp_path, monkeypatch):
    """A checkpoint folder, an empty decode profile, and stand-ins for what only a Linux GPU box can answer."""
    import freetoken.engine.engine as engine
    from freetoken.moe import decode_profile

    model = tmp_path / "ckpt"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "fake"}')
    (model / "model-00001.safetensors").write_bytes(b"x" * 100)
    monkeypatch.setenv("FREETOKEN_BENCHDECODE_PATH", str(tmp_path / "benchdecode.json"))
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", str(tmp_path / "benchbw.json"))
    for name in [n for n in os.environ if n.startswith("FREETOKEN_") and n not in ("FREETOKEN_BENCHDECODE_PATH", "FREETOKEN_BENCHBW_PATH")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(engine, "_is_unified_memory_gpu", lambda index=None: False)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: True)
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: _ROW_GPU)
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE)
    monkeypatch.setattr("freetoken.version.__version__", "9.9.9+gabc")
    proc = {"/proc/meminfo": "65536000 kB", "/proc/cpuinfo": "Fake CPU", "/sys/fs/cgroup/cpu.max": "max 100000"}
    monkeypatch.setattr(decode_profile, "_first_line", lambda path, prefix="": proc.get(path))
    return SimpleNamespace(model=model, engine=engine, dp=decode_profile, tmp=tmp_path, patch=monkeypatch)


def _resolved(rows, **flags):
    config = _offload_engine_config(model_path=str(rows.model), **flags)
    rows.engine._adjust_config(config)
    return config


def _file_row(rows, tok_s, *, slots=40, asked="offload", guard=(), **flags):
    """File a row the way the bench does: under the key one of its servers, started with an explicit strategy, logs."""
    with rows.patch.context() as m:
        m.setenv("FREETOKEN_BENCH_DECODE_KEY", "1")
        key = _resolved(rows, moe_strategy=asked, **flags).moe_decode_key
    row = {"tok_s": tok_s, "columns": ["aime", "repeat"], "guard": list(guard), "slots": {"offload": slots, "hybrid": slots},
           "total_slots": 80, "slot_bytes": 1 << 20, "gpu_free_bytes": _ROW_FREE, "measured_at": "2026-11-12T10:00:00",
           "stamp": rows.dp.stamp(*_ROW_GPU)}
    rows.dp.store(key, row, gpu={"name": _ROW_GPU[0], "uuid": _ROW_GPU[1]})
    return key


def test_auto_stays_on_offload_without_a_decode_row(decode_rows):
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and config.moe_decode_row is None


@pytest.mark.parametrize("asked", ["offload", "hybrid"])
@pytest.mark.parametrize("flags", [{}, {"moe_cache_size": 40}, {"moe_cache_rate": 0.5}, {"moe_cache_mem": 4 << 30}, {"memory_ratio": 0.5}])
def test_auto_takes_hybrid_on_the_row_an_explicit_start_filed(decode_rows, asked, flags):
    key = _file_row(decode_rows, _HYBRID_FASTER, asked=asked, **flags)
    config = _resolved(decode_rows, **flags)
    assert config.moe_decode_key == key
    assert config.moe_strategy == "hybrid" and config.moe_decode_row["tok_s"] == _HYBRID_FASTER
    if not flags:
        assert config.moe_cache_auto is True  # a bare start still gets the auto-sized cache


def test_auto_stays_on_offload_when_the_row_says_hybrid_is_slower(decode_rows):
    _file_row(decode_rows, _HYBRID_SLOWER)
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and config.moe_decode_row is None


@pytest.mark.parametrize(
    "served, differs",
    [
        ({"max_running_req": 2}, "req"),
        ({"moe_cache_size": 48}, "cache"),
        ({"memory_ratio": 0.8}, "cache"),
        ({"moe_cpu_threads": 6}, "cpu"),
        ({"speculative_algorithm": "mtp", "speculative_num_steps": 3}, "spec"),
        ({"moe_hybrid_max_fetch": 2}, "moe_hybrid_max_fetch"),
        ({"moe_cache_scope": "per-layer"}, "moe_cache_scope"),
        ({"moe_cpu_layers": "2"}, "moe_cpu_layers"),
        ({"kv_reserve_tokens": 2048}, "kv_reserve_tokens"),
        ({"cuda_graph_max_bs": 1}, "cuda_graph_max_bs"),
        ({"moe_collect_stats": True}, "moe_collect_stats"),
        ({"hf_overrides": {"rope_scaling": {"factor": 2}}}, "hf_overrides"),
    ],
)
def test_auto_does_not_use_a_row_measured_with_other_settings(decode_rows, served, differs, monkeypatch):
    # speculative decoding resolves through the proposer; keep that out of this test
    monkeypatch.setattr(decode_rows.engine, "_adjust_spec_config", lambda *a, **k: None)
    key = _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"  # the row is good for its own settings
    config = _resolved(decode_rows, **served)
    assert config.moe_strategy == "offload"
    # and the log can name the setting: that is what tells a user why auto did not take the row
    assert differs in decode_rows.dp.key_diff(key, config.moe_decode_key)


def test_auto_tells_speculative_methods_and_environment_switches_apart(decode_rows, monkeypatch):
    monkeypatch.setattr(decode_rows.engine, "_adjust_spec_config", lambda *a, **k: None)
    _file_row(decode_rows, _HYBRID_FASTER, speculative_algorithm="mtp", speculative_num_steps=3)
    assert _resolved(decode_rows, speculative_algorithm="mtp", speculative_num_steps=3).moe_strategy == "hybrid"
    assert _resolved(decode_rows, speculative_algorithm="dflash", speculative_num_steps=3).moe_strategy == "offload"
    _file_row(decode_rows, _HYBRID_FASTER)
    monkeypatch.setenv("FREETOKEN_API_KEY", "not a setup")  # ignored, and never written into a key
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setenv("FREETOKEN_CPU_MOE_SPIN", "0")
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and "not a setup" not in config.moe_decode_key
    monkeypatch.delenv("FREETOKEN_CPU_MOE_SPIN")
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setenv("DSV4_SWITCH_NOBODY_BENCHED", "1")  # the kernels' own switches do not start with FREETOKEN_
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_environment_switches_the_source_reads_are_in_the_decode_key_or_named_as_left_out():
    """A switch that changes decode speed and is not in the key lets auto apply a row measured without it."""
    import re
    from pathlib import Path

    import freetoken
    from freetoken.moe import decode_profile

    literal = re.compile(r"""["']([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["']""")
    unknown = set()
    for path in Path(freetoken.__file__).parent.rglob("*"):
        if path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h"):
            for line in path.read_text(errors="replace").splitlines():
                if "environ" in line or "getenv" in line:
                    unknown.update(name for name in literal.findall(line) if decode_profile.env_role(name) is None)
    assert not unknown, f"say in moe/decode_profile.py whether these are part of the decode key: {sorted(unknown)}"


def test_auto_does_not_use_a_row_from_another_release_machine_gpu_or_checkpoint(decode_rows, monkeypatch):
    dp, engine = decode_rows.dp, decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    now = dp.stamp(*_ROW_GPU)
    for name, other in (("torch", "0.0"), ("triton", "0.0"), ("nvidia", "another driver"), ("cpu", "another CPU"), ("ram_gib", 1),
                        ("cpu_quota", "200000 100000")):
        with monkeypatch.context() as m:
            m.setattr(dp, "stamp", lambda gpu_name=None, gpu_uuid=None, name=name, other=other: {**now, name: other})
            assert _resolved(decode_rows).moe_strategy == "offload", name
    with monkeypatch.context() as m:
        m.setattr("freetoken.version.__version__", "9.9.9+gdef")  # another commit of the same release: the row holds
        assert _resolved(decode_rows).moe_strategy == "hybrid"
        m.setattr("freetoken.version.__version__", "9.9.10")  # another release: the row is stale
        assert _resolved(decode_rows).moe_strategy == "offload"
    with monkeypatch.context() as m:
        m.setattr(dp, "_first_line", lambda path, prefix="": None)  # no /proc to read the machine from, as on anything but Linux
        assert _resolved(decode_rows).moe_strategy == "offload"
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: ("FAKE GPU", "GPU-other"))
    assert _resolved(decode_rows).moe_strategy == "offload"
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: _ROW_GPU)
    (decode_rows.model / "README.md").write_text("not a weight file")
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    (decode_rows.model / "model-00001.safetensors").write_bytes(b"x" * 101)  # the checkpoint changed on disk
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_auto_drops_the_row_when_the_bandwidth_profile_or_the_cpu_set_changes(decode_rows, monkeypatch):
    import json

    from freetoken.moe.bench_profile import PROFILE_VERSION

    def bench_bw(cpu_gbs):
        profile = {"version": PROFILE_VERSION, "gpu": {"name": _ROW_GPU[0]}, "layer_kinds": {"nvfp4:2048x512:top8:t12": {"cpu_moe_overlap_gbs": cpu_gbs}}}
        (decode_rows.tmp / "benchbw.json").write_text(json.dumps(profile))

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    bench_bw(170.0)
    _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    bench_bw(90.0)  # hybrid's fetch split changes with it
    assert _resolved(decode_rows).moe_strategy == "offload"
    bench_bw(170.0)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})  # as many CPUs, but other ones
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_auto_never_takes_hybrid_where_it_cannot_run_or_is_not_the_default(decode_rows, monkeypatch):
    from freetoken.distributed import DistributedInfo

    engine = decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)
    with monkeypatch.context() as m:
        m.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: False)
        assert _resolved(decode_rows).moe_strategy == "offload"
    with monkeypatch.context() as m:  # unified memory keeps its own default
        m.setattr(engine, "_is_unified_memory_gpu", lambda index=None: True)
        m.setattr(engine, "_fused_resident_ok", lambda model_config: True)
        assert _resolved(decode_rows).moe_strategy == "fused"
    two_ranks = DistributedInfo(rank=0, size=2)  # each rank would read the rows of its own GPU
    _file_row(decode_rows, _HYBRID_FASTER, tp_info=two_ranks)
    assert _resolved(decode_rows, tp_info=two_ranks).moe_strategy == "offload"


def test_auto_judges_the_arm_of_the_rounding_mode_the_server_runs(decode_rows, monkeypatch):
    _file_row(decode_rows, _TWO_ARMS, guard=["repeat"])
    assert _resolved(decode_rows).moe_strategy == "offload"  # min-wait is what runs by default, and it is outside the guard limit
    monkeypatch.setenv("FREETOKEN_HYBRID_FETCH_ROUND", "stochastic")
    assert _resolved(decode_rows).moe_strategy == "hybrid"


def test_explicit_strategy_never_reads_the_decode_profile(decode_rows, monkeypatch):
    _file_row(decode_rows, _HYBRID_SLOWER, moe_cache_size=40)
    monkeypatch.setattr(decode_rows.dp, "lookup", lambda *a, **k: pytest.fail("an explicit strategy read the decode profile"))
    for asked in ("hybrid", "offload"):
        config = _resolved(decode_rows, moe_strategy=asked, moe_cache_size=40)
        assert config.moe_strategy == asked and config.moe_decode_key is None  # nor is the key made for it


def test_auto_does_not_take_a_row_whose_hybrid_cache_is_below_the_models_floor(decode_rows):
    # the fake model has 8 experts a layer: a row edited to 4 slots would cap the cache under the floor and fail the start
    _file_row(decode_rows, _HYBRID_FASTER, slots=4)
    assert _resolved(decode_rows).moe_strategy == "offload"
    _file_row(decode_rows, _HYBRID_FASTER, slots=8)
    assert _resolved(decode_rows).moe_strategy == "hybrid"


def test_auto_sized_cache_takes_the_row_only_with_no_more_memory_than_measured(decode_rows, monkeypatch):
    engine = decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)  # 40 of 80 slots at 1 MiB each, with _ROW_FREE bytes free
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE - (8 << 30))
    assert _resolved(decode_rows).moe_strategy == "hybrid"   # less memory: a smaller cache, where hybrid leads by more
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (1 << 20))
    assert _resolved(decode_rows).moe_strategy == "hybrid"   # one slot more is inside the tolerance
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (8 << 20))
    assert _resolved(decode_rows).moe_strategy == "offload"  # offload would get 7 slots more than hybrid was compared at
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: None)
    assert _resolved(decode_rows).moe_strategy == "offload"  # unknown is not a yes
    # an explicit size is not sized from free memory, so free memory does not matter
    _file_row(decode_rows, _HYBRID_FASTER, moe_cache_size=40)
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (8 << 30))
    assert _resolved(decode_rows, moe_cache_size=40).moe_strategy == "hybrid"


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override, scaling=None):
    from types import SimpleNamespace

    from freetoken.models.config import RotaryConfig

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=RotaryConfig(
            head_dim=128, rotary_dim=128, max_position=max_position, base=1e4, scaling=scaling
        ),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


YARN_4X = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}


def test_adjust_config_allows_override_to_the_yarn_extended_table():
    # A YaRN override that leaves max_position_embeddings at the trained length (Qwen3.8's 1M
    # recipe) serves original * factor positions; the gate must follow the table, not the config.
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=4096, scaling=YARN_4X))
    with pytest.raises(ValueError, match=r"rope table \(4096 positions\)"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=4097, scaling=YARN_4X))


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None


def test_moe_slot_pool_split(tmp_path):
    import json

    from freetoken.moe.pool_policy import group_layers, load_profile, make_pools, split_slots

    pools = make_pools(group_layers("per-layer", [0] * 4, [1] * 4), num_experts=16, min_num_slots=8, slot_limit=None)
    assert split_slots(41, pools) == [11, 10, 10, 10]
    with pytest.raises(ValueError, match="need"):
        split_slots(31, pools)
    with pytest.raises(ValueError, match="hold"):
        split_slots(65, pools)
    # one pool takes the whole cache, as before pools existed
    assert split_slots(100, make_pools(group_layers("unified", [0] * 4, [1] * 4), 16, 8, None)) == [100]
    # unified pools one bank layout each, widest slots first
    assert group_layers("unified", ["a", "b", "a", "b"], [1, 3, 1, 3]) == [[1, 3], [0, 2]]

    # layer 0 uses two experts, layer 1 has a hot half and a cold half, layers 2 and 3 use all 16 evenly
    counts = [[5] * 2 + [0] * 14, [3] * 8 + [1] * 8, [1] * 16, [1] * 16]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"model": "toy", "expert_counts": counts}))
    assert split_slots(48, pools, load_profile(path, 4, 16)) == [8, 8, 16, 16]
    with pytest.raises(ValueError, match="needs"):
        load_profile(path, 4, 32)



def test_moe_pool_flags_and_decode_batch_clamp():
    from unittest.mock import patch

    from freetoken.engine.engine import _clamp_decode_batch_to_pools
    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--moe-cache-mem", "1.5GiB",
            "--moe-cache-scope", "per-layer", "--moe-cache-split", "profile:/p.json",
        ])
        assert (args.moe_cache_mem, args.moe_cache_scope, args.moe_cache_split) == (3 << 29, "per-layer", "profile:/p.json")
        for bad in (["--moe-cache-mem", "12"], ["--moe-cache-mem", "1GiB", "--moe-cache-size", "512"]):
            with pytest.raises(SystemExit):
                parse_args(["--model", "/models/anon", *bad])

    # GPU decode needs batch * top_k distinct experts to fit the smallest pool below num_experts
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=16, num_experts_per_tok=4),
        max_running_req=4, cuda_graph_max_bs=None, cuda_graph_bs=[1, 2, 4], decode_rows_per_req=1,
    )
    _clamp_decode_batch_to_pools(config, [12, 16, 9])
    assert (config.max_running_req, config.cuda_graph_max_bs, config.cuda_graph_bs) == (2, 2, [1, 2])
    # two rows per request route twice the experts: 9 // (4 * 2) leaves one request
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=16, num_experts_per_tok=4),
        max_running_req=4, cuda_graph_max_bs=None, cuda_graph_bs=[1, 2, 4], decode_rows_per_req=2,
    )
    _clamp_decode_batch_to_pools(config, [12, 16, 9])
    assert (config.max_running_req, config.cuda_graph_max_bs, config.cuda_graph_bs) == (1, 1, [1])
