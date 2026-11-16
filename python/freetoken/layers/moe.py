import dataclasses
import os
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, RoutedExperts, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

# Once every expert is resident, a prefill groups its routes over the layer's experts where the kernel can, not over the
# pool's slots. Set FREETOKEN_PREFILL_EXPERT_ALIGN=0 for the grouping over the slots -- a measurement-only escape hatch.
_PREFILL_EXPERT_ALIGN = os.getenv("FREETOKEN_PREFILL_EXPERT_ALIGN", "1") != "0"


class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        self.quant_method = quant_method_for(quant_config, self, prefix)
        if allocate_experts:
            self.quant_method.create_weights(self)

    def finalize(self) -> None:
        self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sum_routes: bool = True,
    ) -> torch.Tensor:
        view = self.quant_method.resident_view(self)
        if not sum_routes:
            view = dataclasses.replace(view, sum_routes=False)
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, view,
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        sum_routes: bool = True,
    ):
        """``sum_routes=False`` asks for each route's weighted output [M, top_k, hidden], unsummed.
        Only a decode kernel that has it gives it; every other path returns the sum [M, hidden]."""
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        # ranks reduce the summed output
        sum_routes = sum_routes or self.tp_size > 1
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids, sum_routes)
        return self._maybe_all_reduce(out)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    @property
    def skips_inactive_slots(self) -> bool:
        return self.quant_method.kernel.supports_inactive_slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        sum_routes: bool = True,
    ):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(
                hidden_states, router_logits, sum_routes or self.tp_size > 1
            )
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        sum_routes: bool = True,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._decode_routed(hidden_states, topk_weights, topk_ids, sum_routes)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand, and so does a short enough prefill chunk; a longer one
    # streams whole layers, double-buffered when overlap is enabled. The
    # kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sum_routes: bool = True,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here.

        ``sum_routes=False`` reaches the GPU kernel only: the CPU and hybrid paths return the sum."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=self._slot_alphas(cache),
            is_prefill=False,
            sum_routes=sum_routes,
        )

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split runs inside the hybrid ensure kernel and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- CPU-assigned routes reach the GPU as slot -1 with
        weight 0 (a kernel with ``supports_inactive_slots`` skips them without a read; any other reads
        slot 0, whose storage is a payload or the bank's zero fill) and the CPU ids are -1 for
        GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        # the kernel reads every route's slot: give the inactive ones slot 0, which holds a
        # complete payload or the bank's zero fill (never uninitialized storage), times weight 0
        # with several quants this holds only because the cache zeroes its staging area after prefill
        clamp = not self.skips_inactive_slots
        gpu_slots, gpu_w, cpu_ids = cache.ensure_experts_hybrid(self.layer_id, topk_ids, topk_weights, clamp=clamp)
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=self._slot_alphas(cache),
            is_prefill=False,
        )
        if not _HYBRID_OVERLAP:
            return gpu_routed + cpu_routed_early
        return executor.decode_sync(pending, add_to=gpu_routed)

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement. A chunk within ``prefill_demand_tokens``, or a longer one the cache
        chose at its first layer (``choose_prefill_path``), loads its routed experts
        into the slot cache as decode does and runs the same grouped GEMM over the slots; so does
        any chunk once every expert is resident, with nothing to load. Otherwise a chunk streams
        whole layers -- double-buffered behind the previous layer's GEMMs when ``prefill_overlap``
        is on, else a synchronous ``materialize_layer`` -- where position == expert id, so the
        routing ids pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        demand = cache.loads_prefill_on_demand(hidden_states.shape[0])
        if self.layer_id == 0 and not demand:
            cache.begin_prefill(num_routes=topk_ids.numel(), num_tokens=hidden_states.shape[0])
            cache.choose_prefill_path(self.layer_id, topk_ids)
        demand = demand or cache.prefill_chunk_demand
        if cache.all_resident or demand:
            slot_of = None
            by_expert = _PREFILL_EXPERT_ALIGN and self.quant_method.kernel.groups_prefill_by_expert
            if not cache.all_resident and by_expert:
                # the admission gave each routed expert a slot: the grid spans the layer's experts, not the pool's slots
                cache.admit_prefill_experts(self.layer_id, topk_ids)
                routes, slot_of = topk_ids, cache.slot_for_id[self.layer_id]
            elif not cache.all_resident:
                routes = cache.load_prefill_experts(self.layer_id, topk_ids)
            elif by_expert:
                # a GEMM grid sized for a padded block per slot of the pool is mostly empty programs: the kernel groups
                # the expert ids and looks up one slot per block
                routes, slot_of = topk_ids, cache.resident_slot_of(self.layer_id)
            else:
                # every expert has a slot (the host knows it from a prefill's snapshot): the slots are read as they are
                routes = cache.resident_slots(self.layer_id, topk_ids)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                routes,
                views=cache.pool_bank_views(self.layer_id),
                n=cache.pool_slots(self.layer_id),
                alphas=self._slot_alphas(cache),
                routed=RoutedExperts(self.num_experts, cache.alphas_for_layer(self.layer_id), slot_of),
                is_prefill=True,
            )
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache, topk_ids)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.stage_views(self.layer_id, 0),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )
        cache.release_prefill_layer(self.layer_id)
        return out

    def _slot_alphas(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The cache's alphas by slot, for a kernel that reads them from a slot view."""
        return cache.alphas_for_slots(self.layer_id) if self.quant_method.kernel.reads_slot_alphas else None

    def _wait_prefill_overlap(self, cache: OffloadMoeCache, topk_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id, topk_ids)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows, unless ``routed.slot_of`` maps them.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
        sum_routes: bool = True,
        routed: RoutedExperts | None = None,
    ) -> torch.Tensor:
        from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

        view = ExpertView(
            {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
            slots=None if n is not None else topk_ids, n=n, alphas=alphas, sum_routes=sum_routes, routed=routed,
        )
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
        )


def iter_moe_layers(model) -> Iterator[MoELayer]:
    if isinstance(model, MoELayer):
        yield model
    if not isinstance(model, BaseOP):
        return
    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_moe_layers(item)


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
        layer_id=layer_id,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
