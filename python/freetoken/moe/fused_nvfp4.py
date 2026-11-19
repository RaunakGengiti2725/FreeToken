"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import torch

from freetoken.kernel import moe_sum_reduce_triton, nvfp4_moe
from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    sum_routes: bool = True,
    shared: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm`), both with the same calling convention.
    ``sum_routes=False`` skips the sum-reduce and returns each route's weighted output [M, top_k, H].

    ``shared`` holds one expert outside the bank by the bank's roles (``gate_up``, ``gate_up_scale``, ...), each with
    a slot's shape and strides: the last column of ``topk_ids`` / ``topk_weights`` is then that expert's route, its id
    unread (marlin GEMV only)."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    if nvfp4_moe._DECODE_DISTINCT:
        nvfp4_moe.count_distinct_experts(topk_ids if shared is None else topk_ids[:, :-1])
    route = {}
    if shared is not None:
        assert gemm_fn is nvfp4_moe.decode_gemm_marlin, "only the marlin decode GEMV takes an expert outside the bank"
        route = dict(gate_up=dict(shared=(shared["gate_up"], shared["gate_up_scale"], shared["gate_up_global"])),
                     down=dict(shared=(shared["down"], shared["down_scale"], shared["down_global"])))

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False, **route.get("gate_up", {}),
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True, **route.get("down", {}),
    )
    if not sum_routes:
        return ic3
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    sum_routes: bool = True,
    shared: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV; ``shared`` as in
    :func:`_fused_experts_decode_nvfp4`."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit, sum_routes, shared,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def _align_routes(
    topk_ids: torch.Tensor, block_m: int, num_experts: int, tile_experts: int, slot_of: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``moe_align_block_size`` over the ``num_experts`` bank rows ``topk_ids`` index; or, with ``slot_of``, over the
    layer's ``tile_experts`` expert ids, each M block then naming the bank row (slot) of its expert.

    Aligning over a pool of slots sizes ``sorted_ids``, and with it the GEMM grid, for a padded block per slot of the
    pool, though a layer's routes reach ``tile_experts`` of them at most."""
    if slot_of is None:
        return moe_align_block_size(topk_ids, block_m, num_experts)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, block_m, tile_experts)
    # the blocks past ntpp are never run but hold uninitialized ids, which may name an expert without a slot (-1)
    # after a demand admission: those name row 0. A block that runs and names no slot is a routed expert the
    # admission did not load, and stops the forward instead of reading a row before the banks.
    slots = slot_of.index_select(0, expert_ids.clamp_(0, tile_experts - 1))
    runs = torch.arange(slots.numel(), device=slots.device) * block_m < ntpp.to(slots.device)
    torch._assert_async(((slots >= 0) | ~runs).all())
    return sorted_ids, torch.where(runs, slots, 0), ntpp


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    routed_experts: int | None = None,
    slot_of: torch.Tensor | None = None,
    sum_routes: bool = True,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids; or the
    cache slots a demand prefill chunk routes to, whose layer's ``routed_experts``
    size the tiles as the stream's banks would. With ``slot_of`` they are the expert
    ids of that layer, and ``slot_of[expert]`` the bank row of each.
    ``sum_routes=False`` skips the sum-reduce and returns each route's weighted output [M, top_k, H], as decode does."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    tile_experts = num_experts if routed_experts is None else routed_experts
    cfg = nvfp4_moe.prefill_config(M, top_k, tile_experts, hidden_states.device.index)

    sorted_ids, expert_ids, ntpp = _align_routes(topk_ids, cfg["BLOCK_SIZE_M"], num_experts, tile_experts, slot_of)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    nvfp4_moe.prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    if not sum_routes:
        return ic3
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_nvfp4_w4a4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    routed_experts: int | None = None,
    slot_of: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill NVFP4 MoE with NVFP4 activations (W4A4) where the GPU and the batch favour it, else :func:`fused_experts_nvfp4`.

    ``*_input_scale`` are the per-expert static activation scales of the experts the routes are drawn from: the
    ``num_experts`` banks, or a demand prefill chunk's layer of ``routed_experts`` where the banks are cache slots.
    Each GEMM input is quantized under their max, as vLLM does. The down GEMM stores each route's row and the routes are
    summed in route order, as the W4A16 prefill does, so the output repeats bit for bit; under
    FREETOKEN_NVFP4_W4A4_ATOMIC_SUM it adds them into the output with atomics instead.
    """
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter = gate_up_packed.shape[1] // 2
    tile_experts = num_experts if routed_experts is None else routed_experts
    cfg = nvfp4_moe.w4a4_prefill_config(M, top_k, tile_experts, H, inter, activation, hidden_states.dtype, hidden_states.device.index)
    if cfg is None:
        return fused_experts_nvfp4(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, down_packed, down_scale, down_global,
            topk_weights, topk_ids, num_experts, activation, False, act_alpha, act_limit, routed_experts, slot_of,
        )
    dev = hidden_states.device
    g = torch.stack((gate_up_input_scale[:tile_experts], down_input_scale[:tile_experts])).float().amax(dim=1)
    sorted_ids, expert_ids, ntpp = _align_routes(topk_ids, cfg["BLOCK_SIZE_M"], num_experts, tile_experts, slot_of)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    xq, xs = quantize_nvfp4(hidden_states, g[0:1])
    q = torch.empty((num_valid, inter // 2), device=dev, dtype=torch.uint8)
    qs = torch.empty((num_valid, inter // 16), device=dev, dtype=torch.float8_e4m3fn)
    nvfp4_moe.w4a4_gate_up_gemm(
        xq, xs, g[0:1], gate_up_packed, gate_up_scale, gate_up_global, q, qs, g[1:2], sorted_ids, expert_ids, ntpp,
        num_valid, top_k, activation, act_alpha, act_limit, cfg["gate_up"],
    )
    route_out = cfg["down"]["ROUTE_OUT"]
    out = torch.empty_like(hidden_states) if route_out else torch.zeros_like(hidden_states)
    dst = torch.empty((num_valid, H), device=dev, dtype=out.dtype) if route_out else out
    nvfp4_moe.w4a4_down_gemm(
        q, qs, g[1:2], down_packed, down_scale, down_global, dst, tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        cfg["down"],
    )
    if route_out:
        moe_sum_reduce_triton(dst.view(M, top_k, H), out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
    "fused_experts_nvfp4_w4a4",
]
