# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (kernels/ops/moe/moe_fused_gate.py)
"""Fused top-k MoE routers: softmax (bias-free, ungrouped) and sigmoid with a selection-only bias and optional group-limited top-k.

The ``num_token_non_padded`` row mask reads the device tensor, so it survives CUDA-graph capture.
For decode-sized batches ``router_gemv_topk_softmax`` also computes the softmax router's bf16 logits in its launch.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


@triton.jit
def _keep_top_groups(ranked, offs_n, N_GROUP: tl.constexpr, TOPK_GROUP: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_G: tl.constexpr):
    # DeepSeek-V3 noaux_tc: a group scores the sum of its top-2 entries; experts outside the TOPK_GROUP best groups drop to -inf.
    offs_g = tl.arange(0, BLOCK_G)
    group_of_n = offs_n // GROUP_SIZE
    group_score = tl.full([ranked.shape[0], BLOCK_G], -float("inf"), dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        vals = tl.where(group_of_n[None, :] == g, ranked, -float("inf"))
        top1 = tl.max(vals, axis=1)[:, None]
        top1_lane = tl.min(tl.where(vals == top1, offs_n[None, :], offs_n.shape[0]), axis=1)[:, None]
        top2 = tl.max(tl.where(offs_n[None, :] == top1_lane, -float("inf"), vals), axis=1)[:, None]
        group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

    keep = tl.zeros(ranked.shape, dtype=tl.int1)
    for _ in tl.static_range(TOPK_GROUP):
        best = tl.max(group_score, axis=1)[:, None]
        win_g = tl.min(tl.where(group_score == best, offs_g[None, :], BLOCK_G), axis=1)[:, None]
        keep = keep | (group_of_n[None, :] == win_g)
        group_score = tl.where(offs_g[None, :] == win_g, -float("inf"), group_score)
    return tl.where(keep, ranked, -float("inf"))


@triton.jit
def _route_rows(
    logits,
    offs_m,
    mask_m,
    offs_n,
    mask_n,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    num_token_non_padded_ptr,
    routed_scale,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SIGMOID: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_TOKEN_LIMIT: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """The router past its logits: score the [BLOCK_M, BLOCK_N] fp32 ``logits`` of rows ``offs_m``, take the top-k, store it."""
    if SIGMOID:
        # libdevice exp and round-to-nearest division match torch.sigmoid bit for bit.
        activated = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        ranked = tl.where(mask_n[None, :], activated + bias[None, :], -float("inf"))
    else:
        ranked = tl.where(mask_n[None, :], logits, -float("inf"))
        row_max = tl.max(ranked, axis=1)[:, None]
        exp_row = tl.where(mask_n[None, :], tl.exp(ranked - row_max), 0.0)
        activated = exp_row / tl.sum(exp_row, axis=1)[:, None]

    # Map NaN -> a finite floor
    ranked = tl.where(ranked == ranked, ranked, -1e30)

    if N_GROUP > 1:
        ranked = _keep_top_groups(ranked, offs_n, N_GROUP, TOPK_GROUP, N // N_GROUP, BLOCK_G)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = ranked
    for k in tl.static_range(K):
        max_val = tl.max(cur, axis=1)[:, None]
        lane_id = tl.where(cur == max_val, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)
        win_activated = tl.sum(
            tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
        )[:, None]
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]
        if SIGMOID:
            selected_vals = tl.div_rn(selected_vals, routed_sum + 1e-20)
        else:
            selected_vals = selected_vals / tl.where(routed_sum > 0.0, routed_sum, 1.0)
    if SIGMOID:
        selected_vals = selected_vals * routed_scale

    if HAS_TOKEN_LIMIT:
        limit = tl.load(num_token_non_padded_ptr)
        selected_idx = tl.where(offs_m[:, None] < limit, selected_idx, -1)

    out_w_ptr = out_weights_ptr + offs_m[:, None].to(tl.int64) * stride_wm + offs_k[None, :] * stride_wk
    out_i_ptr = out_indices_ptr + offs_m[:, None].to(tl.int64) * stride_im + offs_k[None, :] * stride_ik
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


@triton.jit
def _router_triton_kernel(
    scores_ptr,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    num_token_non_padded_ptr,
    M,
    routed_scale,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SIGMOID: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_TOKEN_LIMIT: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # offs_m * stride can overflow int32 for large token counts.
    row_ptr = scores_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    logits = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    _route_rows(
        logits, offs_m, mask_m, offs_n, mask_n, bias_ptr, out_weights_ptr, out_indices_ptr, num_token_non_padded_ptr,
        routed_scale, stride_wm, stride_wk, stride_im, stride_ik,
        N, K, BLOCK_M, BLOCK_K, SIGMOID, N_GROUP, TOPK_GROUP, BLOCK_G, RENORMALIZE, HAS_TOKEN_LIMIT, launch_pdl,
    )


@triton.jit
def _router_gemv_topk_kernel(
    x_ptr,        # [M, H] bf16 hidden states, unit inner stride
    w_ptr,        # [N, H] bf16 router weight, unit inner stride
    part_ptr,     # [M, SPLIT_K, N] fp32 partial logits
    counter_ptr,  # [M] int32 arrival counters, zeroed (self-resetting)
    logits_ptr,   # [M, N] bf16 logits
    out_weights_ptr,
    out_indices_ptr,
    M,
    tiles_per,
    stride_xm,
    stride_wn,
    stride_qm,
    stride_qk,
    stride_lm,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLIT_K: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    """``(row, expert tile, H slice)``: one slice of BLOCK_E logits of a row. The last program to land
    a slice of the row sums the slices in a fixed order, stores the row's logits in bf16 as F.linear
    does, and runs the softmax router's own one-row program on them."""
    row = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    mask_e = offs_e < N
    offs_h = pid_k * tiles_per * BLOCK_H + tl.arange(0, BLOCK_H)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x_row = x_ptr + row.to(tl.int64) * stride_xm
    acc = tl.zeros([BLOCK_E, BLOCK_H], dtype=tl.float32)
    for t in range(tiles_per):
        h = offs_h + t * BLOCK_H
        xv = tl.load(x_row + h).to(tl.float32)
        wv = tl.load(w_ptr + offs_e[:, None] * stride_wn + h[None, :], mask=mask_e[:, None], other=0.0).to(tl.float32)
        acc += wv * xv[None, :]
    q_row = part_ptr + row.to(tl.int64) * stride_qm
    tl.store(q_row + pid_k * stride_qk + offs_e, tl.sum(acc, axis=1), mask=mask_e)

    # every thread's store lands before the one atomic that publishes them
    tl.debug_barrier()
    c_ptr = counter_ptr + row
    cnt = tl.atomic_add(c_ptr, 1)  # default acq_rel/gpu
    if cnt == N_TILES * SPLIT_K - 1:
        # the router's one-row tile, so the program past the logits is its program
        offs_m = row + tl.arange(0, 1)
        offs_n = tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask2d = mask_m[:, None] & mask_n[None, :]
        q_ptr = part_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_n[None, :]
        total = tl.zeros([1, BLOCK_N], dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):  # fixed order -> replay-stable logits
            total += tl.load(q_ptr + k * stride_qk, mask=mask2d, other=0.0, cache_modifier=".cg")
        l_ptr = logits_ptr + offs_m[:, None].to(tl.int64) * stride_lm + offs_n[None, :]
        tl.store(l_ptr, total.to(tl.bfloat16), mask=mask2d)
        # read back as the router reads them: the load's layout sets the order of its reductions
        tl.debug_barrier()
        logits = tl.load(l_ptr, mask=mask2d, other=0.0).to(tl.float32)
        _route_rows(
            logits, offs_m, mask_m, offs_n, mask_n, w_ptr, out_weights_ptr, out_indices_ptr, c_ptr,
            1.0, stride_wm, stride_wk, stride_im, stride_ik,
            N, K, 1, BLOCK_K, False, 1, 1, 1, RENORMALIZE, False, launch_pdl,
        )
        tl.store(c_ptr, 0)  # leave zeroed for the next launch/replay


def _launch_router(
    gating_output: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scale: float,
    n_group: int,
    topk_group: int,
    num_token_non_padded: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gating_output.ndim == 2, "gating_output must be 2D"
    M, N = gating_output.shape
    weights = torch.empty((M, topk), dtype=torch.float32, device=gating_output.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=gating_output.device)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(topk)
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200; larger tiles / more warps regress
    # (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows the K sequential argmax passes dominate and benefit from more
    # warps despite the cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4

    _router_triton_kernel[(triton.cdiv(M, BLOCK_M),)](
        gating_output,
        bias,
        weights,
        indices,
        num_token_non_padded,
        M,
        routed_scale,
        gating_output.stride(0),
        gating_output.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        N=N,
        K=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SIGMOID=bias is not None,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        BLOCK_G=triton.next_power_of_2(n_group),
        RENORMALIZE=renormalize,
        HAS_TOKEN_LIMIT=num_token_non_padded is not None,
        launch_pdl=_pdl_supported(),
        num_warps=num_warps,
    )
    return weights, indices


def fused_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, top-k, then renormalize; ties keep the lowest expert id.

    ``num_token_non_padded`` is a device scalar; rows at or past it get expert id -1.
    """
    return _launch_router(gating_output, None, topk, renormalize, 1.0, 1, 1, num_token_non_padded)


def fused_topk_sigmoid(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scale: float = 1.0,
    n_group: int = 1,
    topk_group: int = 1,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid scores; ``scores + bias`` picks the top-k (within the ``topk_group`` best of ``n_group`` groups when ``n_group > 1``).

    The weights are the unbiased scores, renormalized to sum to 1 when ``renormalize``, then times ``routed_scale``.
    Ties keep the lowest expert id; rows at or past ``num_token_non_padded`` get expert id -1.
    """
    num_experts = gating_output.shape[1]
    assert bias.shape == (num_experts,) and bias.stride(0) == 1
    assert n_group == 1 or (num_experts % n_group == 0 and num_experts // n_group >= 2 and 1 <= topk_group <= n_group)
    return _launch_router(gating_output, bias, topk, renormalize, float(routed_scale), n_group, topk_group, num_token_non_padded)


# The fold's GEMV tiles: one-warp programs of BLOCK_E experts by one H slice of SPLIT_K, each slice
# _FOLD_BLOCK_H wide, so the whole weight is read once a row in a single wave of small programs.
_FOLD_BLOCK_E = 8
_FOLD_BLOCK_H = 256
_FOLD_MAX_SPLIT_K = 8
# Each row re-reads the whole weight; past this many rows the cuBLAS GEMM's reuse wins.
_FOLD_MAX_ROWS = 16
_FOLD_COUNTERS: dict = {}


def _fold_counters(device: torch.device) -> torch.Tensor:
    c = _FOLD_COUNTERS.get(device)
    if c is None:
        c = torch.zeros(_FOLD_MAX_ROWS, dtype=torch.int32, device=device)
        _FOLD_COUNTERS[device] = c
    return c


def _fold_split_k(hidden_size: int) -> int:
    tiles = hidden_size // _FOLD_BLOCK_H
    return next(s for s in (8, 4, 2, 1) if s <= _FOLD_MAX_SPLIT_K and tiles % s == 0)


def router_gemv_topk_folds(rows: int, num_experts: int, hidden_size: int, topk: int) -> bool:
    """Whether :func:`router_gemv_topk_softmax` takes a step of ``rows`` rows: a decode-sized batch,
    129 to 512 experts, which :func:`fused_topk_softmax` routes in its one-row, one-warp program,
    and a hidden size of whole GEMV slices. Other steps keep the linear and the router apart."""
    lanes = 1 << max(num_experts - 1, 0).bit_length()
    return (
        1 <= rows <= _FOLD_MAX_ROWS
        and 256 <= lanes <= 512
        and 1 <= topk <= num_experts
        and hidden_size > 0
        and hidden_size % _FOLD_BLOCK_H == 0
    )


def router_gemv_topk_softmax(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk: int,
    renormalize: bool,
    logits_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``fused_topk_softmax(F.linear(x, weight), topk, renormalize)`` in one launch, for ``x`` [M, H]
    and ``weight`` [N, H] bf16 where :func:`router_gemv_topk_folds` holds. CUDA-graph safe.

    The logits are rounded to bf16 as F.linear stores them and routed by the same program, but their
    dot is summed in this kernel's order, not cuBLAS's: a near-tie between two experts may route the
    other way. ``logits_out`` [M, N] bf16, when given, receives the logits.
    """
    M, H = x.shape
    N = weight.shape[0]
    assert router_gemv_topk_folds(M, N, H, topk), "this step keeps the linear and the router apart"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert weight.shape == (N, H) and x.stride(1) == 1 and weight.stride(1) == 1
    weights = torch.empty((M, topk), dtype=torch.float32, device=x.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=x.device)
    if logits_out is None:
        logits_out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    assert logits_out.shape == (M, N) and logits_out.dtype == torch.bfloat16 and logits_out.stride(1) == 1
    split_k = _fold_split_k(H)
    n_tiles = triton.cdiv(N, _FOLD_BLOCK_E)
    part = torch.empty((M, split_k, N), dtype=torch.float32, device=x.device)
    _router_gemv_topk_kernel[(M, n_tiles, split_k)](
        x,
        weight,
        part,
        _fold_counters(x.device),
        logits_out,
        weights,
        indices,
        M,
        H // (_FOLD_BLOCK_H * split_k),
        x.stride(0),
        weight.stride(0),
        part.stride(0),
        part.stride(1),
        logits_out.stride(0),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        N=N,
        K=topk,
        BLOCK_E=_FOLD_BLOCK_E,
        BLOCK_H=_FOLD_BLOCK_H,
        SPLIT_K=split_k,
        N_TILES=n_tiles,
        BLOCK_N=triton.next_power_of_2(N),
        BLOCK_K=triton.next_power_of_2(topk),
        RENORMALIZE=renormalize,
        launch_pdl=_pdl_supported(),
        num_warps=1,
    )
    return weights, indices


__all__ = ["fused_topk_sigmoid", "fused_topk_softmax", "router_gemv_topk_folds", "router_gemv_topk_softmax"]
