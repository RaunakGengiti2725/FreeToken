"""The gated shared expert of a decode step on resident NVFP4 weights,

    out = routed + sigmoid(gate) * down(silu_and_mul(gate_up(x)))

with the silu-mul folded into the kernel that reduces gate_up, and the sigmoid, the scale and the
add into the kernel that reduces down. No reduction is reordered and every value is rounded where
the unfused kernels store it, so the result is theirs bit for bit. CUDA-graph safe.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.kernel.backend import is_flashinfer_installed
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.nvfp4_linear import (
    _GEMM_MAX_INKERNEL_M,
    _GEMV_SPLITK_BLOCK_KW,
    _GEMV_WARPS,
    _gemm_inkernel_plan,
    _gemm_launch,
    _gemv_splitk_plan,
    _nvfp4_pair_f32,
    _split8,
    _splitk_counters,
)

# the tiles of _nvfp4_gemm_splitk_reduce_kernel at the shared expert's widths
_REDUCE_BLOCK = 128
_REDUCE_WARPS = 2


@triton.jit
def _round_bf16(x):
    """fp32 onto the bf16 grid, round to nearest even, in integer arithmetic: the compiler folds
    a cast to bf16 and back away, and the mul and add around it would then round once."""
    b = x.to(tl.uint32, bitcast=True)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    # an inf or a NaN passes through: the carry out of a NaN's low bits would leave -0
    return tl.where((b & 0x7F800000) == 0x7F800000, b, r).to(tl.float32, bitcast=True)


@triton.jit
def _silu_mul(gate, up):
    """flashinfer's silu_and_mul, ``gate / (1 + __expf(-gate)) * up`` in fp32, as the PTX nvcc
    emits for it under -use_fast_math: an approximate divide, and denormals taken as zero."""
    # gate * -log2(e)
    t = tl.inline_asm_elementwise("mul.ftz.f32 $0, $1, 0fBFB8AA3B;", "=f,f", [gate], dtype=tl.float32, is_pure=True, pack=1)
    t = tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [t], dtype=tl.float32, is_pure=True, pack=1)
    t = tl.inline_asm_elementwise("add.ftz.f32 $0, $1, 0f3F800000;", "=f,f", [t], dtype=tl.float32, is_pure=True, pack=1)
    t = tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, $2;", "=f,f,f", [gate, t], dtype=tl.float32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise("mul.ftz.f32 $0, $1, $2;", "=f,f,f", [t, up], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _gate_scale_add(down, gate, routed):
    """``routed + down * sigmoid(gate)`` in fp32 from bf16 values, the sigmoid and the product
    rounded to bf16 as torch's kernels round them; the caller's store rounds the add."""
    # torch's sigmoid of a bf16 tensor is the fp32 formula rounded once
    scale = _round_bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-gate)))
    return routed + _round_bf16(down * scale)


@triton.jit
def _gemv_partial(
    a_ptr,        # [K] one activation row
    packed_ptr,   # logical [N, K // 8] int32
    scale_ptr,    # logical [N, K // 16] fp8-e4m3
    offs_n, n_mask, tile_start, tiles_per, K, K_WORDS,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, EVEN_K: tl.constexpr,
):
    """One program's partial sums of ``offs_n`` over ``tiles_per`` word tiles of K. The loop is
    ``_nvfp4_gemv_splitk_kernel``'s, statement for statement: keep the two in step."""
    offs_kw = tl.arange(0, BLOCK_KW)
    p_base = packed_ptr + offs_n[None, :] * stride_pn
    s_base = scale_ptr + offs_n[None, :] * stride_sn
    partial = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
    for t in range(tiles_per):
        widx = (tile_start + t) * BLOCK_KW + offs_kw
        offs_a = (tile_start + t) * BLOCK_KW * 8 + tl.arange(0, BLOCK_KW * 8)
        if EVEN_K:
            word = tl.load(p_base + widx[:, None] * stride_pkw, mask=n_mask[None, :], other=0)
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=n_mask[None, :], other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a).to(tl.float32)
        else:
            w_mask = widx < K_WORDS
            word = tl.load(
                p_base + widx[:, None] * stride_pkw,
                mask=w_mask[:, None] & n_mask[None, :], other=0,
            )
            s_ptrs = s_base + (widx[:, None] // 2) * stride_sblk
            s_mask = w_mask[:, None] & n_mask[None, :]
            if e4m3_native_cx():
                scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32) * 16384.0
            else:
                scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0)) * 16384.0
            a_vec = tl.load(a_ptr + offs_a, mask=offs_a < K, other=0.0).to(tl.float32)

        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_KW)
        acc_w = tl.zeros((BLOCK_KW, BLOCK_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc_w += a0[:, None] * b_lo
        acc_w += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc_w += a1[:, None] * b_lo
        acc_w += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc_w += a2[:, None] * b_lo
        acc_w += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc_w += a3[:, None] * b_lo
        acc_w += a7[:, None] * b_hi
        partial += acc_w * scale
    return tl.sum(partial, axis=0)


@triton.jit
def _shared_gate_up_kernel(
    a_ptr,        # [M, K] hidden states, unit inner stride
    packed_ptr,   # logical [2 * HALF, K // 8] int32: the gate rows, then the up rows
    scale_ptr,    # logical [2 * HALF, K // 16] fp8-e4m3
    gscale_ptr,   # [2 * HALF] fp32 per-output-row global scale
    part_ptr,     # [M, SPLIT_K, 2 * HALF] fp32 partial sums
    counter_ptr,  # [M * N_TILES] int32 arrival counters, zeroed (self-resetting)
    act_ptr,      # [M, HALF] silu(gate) * up
    HALF, K, K_WORDS, tiles_per, N_TILES,
    stride_am,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_qm, stride_qk, stride_qn,
    stride_om,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, SPLIT_K: tl.constexpr,
    OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    """``(row, pid_n, pid_k)``: one K slice of BLOCK_N gate outputs and of the up outputs they
    multiply. The last program to land a slice reduces both and writes silu(gate) * up."""
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < HALF
    a_row = a_ptr + row * stride_am
    tile_start = pid_k * tiles_per

    acc_g = _gemv_partial(
        a_row, packed_ptr, scale_ptr, offs_n, n_mask, tile_start, tiles_per, K, K_WORDS,
        stride_pn, stride_pkw, stride_sn, stride_sblk, BLOCK_N, BLOCK_KW, EVEN_K,
    )
    acc_u = _gemv_partial(
        a_row, packed_ptr, scale_ptr, offs_n + HALF, n_mask, tile_start, tiles_per, K, K_WORDS,
        stride_pn, stride_pkw, stride_sn, stride_sblk, BLOCK_N, BLOCK_KW, EVEN_K,
    )
    q_row = part_ptr + row * stride_qm
    tl.store(q_row + pid_k * stride_qk + offs_n * stride_qn, acc_g, mask=n_mask)
    tl.store(q_row + pid_k * stride_qk + (offs_n + HALF) * stride_qn, acc_u, mask=n_mask)

    c_ptr = counter_ptr + row * N_TILES + pid_n
    cnt = tl.atomic_add(c_ptr, 1)  # default acq_rel/gpu: publishes the stores
    if cnt == SPLIT_K - 1:
        tot_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
        tot_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):  # fixed order -> replay-stable reduction
            tot_g += tl.load(q_row + k * stride_qk + offs_n * stride_qn, mask=n_mask, other=0.0)
            tot_u += tl.load(
                q_row + k * stride_qk + (offs_n + HALF) * stride_qn, mask=n_mask, other=0.0
            )
        g_g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        g_u = tl.load(gscale_ptr + offs_n + HALF, mask=n_mask, other=0.0).to(tl.float32)
        # rounded to OUT here because the GEMV stores OUT and the silu-mul reads it back
        gate = (tot_g * g_g).to(OUT).to(tl.float32)
        up = (tot_u * g_u).to(OUT).to(tl.float32)
        tl.store(act_ptr + row * stride_om + offs_n, _silu_mul(gate, up).to(OUT), mask=n_mask)
        tl.store(c_ptr, 0)  # leave zeroed for the next launch/replay


@triton.jit
def _shared_down_add_kernel(
    a_ptr,        # [M, K] silu(gate) * up, unit inner stride
    packed_ptr,   # logical [N, K // 8] int32
    scale_ptr,    # logical [N, K // 16] fp8-e4m3
    gscale_ptr,   # [N] fp32 per-output-row global scale
    gate_ptr,     # [M] the shared expert gate's logit in the activation dtype
    routed_ptr,   # [M, N] routed experts' output, unit inner stride
    part_ptr,     # [M, SPLIT_K, N] fp32 partial sums
    counter_ptr,  # [M * N_TILES] int32 arrival counters, zeroed (self-resetting)
    out_ptr,      # [M, N]
    N, K, K_WORDS, tiles_per, N_TILES,
    stride_am,
    stride_pn, stride_pkw,
    stride_sn, stride_sblk,
    stride_qm, stride_qk, stride_qn,
    stride_rm, stride_om,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr, SPLIT_K: tl.constexpr,
    OUT: tl.constexpr, EVEN_K: tl.constexpr,
):
    """``(row, pid_n, pid_k)``: one K slice of BLOCK_N down outputs. The last program to land a
    slice reduces them, scales by the sigmoid of the row's gate and adds the routed output."""
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = _gemv_partial(
        a_ptr + row * stride_am, packed_ptr, scale_ptr, offs_n, n_mask, pid_k * tiles_per,
        tiles_per, K, K_WORDS, stride_pn, stride_pkw, stride_sn, stride_sblk,
        BLOCK_N, BLOCK_KW, EVEN_K,
    )
    q_row = part_ptr + row * stride_qm
    tl.store(q_row + pid_k * stride_qk + offs_n * stride_qn, acc, mask=n_mask)

    c_ptr = counter_ptr + row * N_TILES + pid_n
    cnt = tl.atomic_add(c_ptr, 1)  # default acq_rel/gpu: publishes the store
    if cnt == SPLIT_K - 1:
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):  # fixed order -> replay-stable reduction
            total += tl.load(q_row + k * stride_qk + offs_n * stride_qn, mask=n_mask, other=0.0)
        g = tl.load(gscale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        down = _round_bf16(total * g)
        gate = tl.load(gate_ptr + row).to(tl.float32)
        routed = tl.load(routed_ptr + row * stride_rm + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out = _gate_scale_add(down, gate, routed)
        tl.store(out_ptr + row * stride_om + offs_n, out.to(OUT), mask=n_mask)
        tl.store(c_ptr, 0)  # leave zeroed for the next launch/replay


@triton.jit
def _shared_gate_up_reduce_kernel(
    part_ptr,     # [SPLIT_K, M, 2 * HALF] fp32 partial sums of the dot GEMM: the gate columns, then the up columns
    gscale_ptr,   # [2 * HALF] fp32 per-output-row global scale
    act_ptr,      # [M, HALF] silu(gate) * up
    HALF,
    stride_qk, stride_qm, stride_qn,
    stride_om,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """``_nvfp4_gemm_splitk_reduce_kernel`` for BLOCK gate outputs of a row and the up outputs
    they multiply, with the silu-mul in place of its store: keep the two in step."""
    pid_m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HALF
    q_row = part_ptr + pid_m * stride_qm
    acc_g = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc_g += tl.load(q_row + k * stride_qk + offs * stride_qn, mask=mask, other=0.0)
        acc_u += tl.load(q_row + k * stride_qk + (offs + HALF) * stride_qn, mask=mask, other=0.0)
    g_g = tl.load(gscale_ptr + offs, mask=mask, other=0.0).to(tl.float32) * 128.0
    g_u = tl.load(gscale_ptr + offs + HALF, mask=mask, other=0.0).to(tl.float32) * 128.0
    # rounded to bf16 here because the reduce stores bf16 and the silu-mul reads it back
    gate = _round_bf16(acc_g * g_g)
    up = _round_bf16(acc_u * g_u)
    tl.store(act_ptr + pid_m * stride_om + offs, _silu_mul(gate, up).to(OUT), mask=mask)


@triton.jit
def _shared_down_add_reduce_kernel(
    part_ptr,     # [SPLIT_K, M, N] fp32 partial sums of the dot GEMM
    gscale_ptr,   # [N] fp32 per-output-row global scale
    gate_ptr,     # [M] the shared expert gate's logit in the activation dtype
    routed_ptr,   # [M, N] routed experts' output, unit inner stride
    out_ptr,      # [M, N]
    N,
    stride_qk, stride_qm, stride_qn,
    stride_rm, stride_om,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """``_nvfp4_gemm_splitk_reduce_kernel`` for BLOCK down outputs of a row, with the scale by the
    gate and the add of the routed output in place of its store: keep the two in step."""
    pid_m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_qk + pid_m * stride_qm + offs * stride_qn, mask=mask, other=0.0)
    g = tl.load(gscale_ptr + offs, mask=mask, other=0.0).to(tl.float32) * 128.0
    down = _round_bf16(acc * g)
    gate = tl.load(gate_ptr + pid_m).to(tl.float32)
    routed = tl.load(routed_ptr + pid_m * stride_rm + offs, mask=mask, other=0.0).to(tl.float32)
    out = _gate_scale_add(down, gate, routed)
    tl.store(out_ptr + pid_m * stride_om + offs, out.to(OUT), mask=mask)


def shared_expert_fusable(hidden_size: int, intermediate_size: int) -> bool:
    """The gate of this module. The fused silu is NVIDIA PTX that copies flashinfer's kernel, so
    flashinfer must be what the unfused path runs; both projections must take the split-K GEMV."""
    if not is_flashinfer_installed():
        return False
    gate_up = _gemv_splitk_plan(2 * intermediate_size, hidden_size, True)
    down = _gemv_splitk_plan(hidden_size, intermediate_size, True)
    return gate_up[1] > 1 and down[1] > 1


@functools.cache
def shared_expert_fuses_rows(M: int, hidden_size: int, intermediate_size: int, device: torch.device) -> bool:
    """Whether a step of M rows reduces both projections in a kernel the fused ones follow: the
    split-K GEMV for one row, the dot GEMM with a reduce launch for 2 to 64. Others stay unfused."""
    if M <= 1:
        return True
    if M > _GEMM_MAX_INKERNEL_M:
        return False
    gate_up = _gemm_inkernel_plan(M, 2 * intermediate_size, hidden_size, device)
    down = _gemm_inkernel_plan(M, hidden_size, intermediate_size, device)
    return gate_up[1] > 1 and down[1] > 1


def _gate_up_rows(
    x: torch.Tensor, weight_t: torch.Tensor, scale_t: torch.Tensor, weight_global: torch.Tensor,
) -> torch.Tensor:
    """Every row through the one-row GEMV's statements, one program a (row, tile, K slice)."""
    M, K = x.shape
    packed = weight_t.t()
    scale = e4m3_kernel_view(scale_t.t())
    N = packed.shape[0]
    half = N // 2
    block_n, split_k, tiles_per, even = _gemv_splitk_plan(N, K, True)
    assert split_k > 1, "the single-pass GEMV shape is not fused"

    act = torch.empty((M, half), dtype=x.dtype, device=x.device)
    if M == 0:
        return act
    n_tiles = triton.cdiv(half, block_n)
    assert M * split_k * N < 2**31, "the kernel indexes rows in int32"
    part = torch.empty((M, split_k, N), dtype=torch.float32, device=x.device)
    counters = _splitk_counters(M * n_tiles, x.device)
    _shared_gate_up_kernel[(M, n_tiles, split_k)](
        x, packed, scale, weight_global, part, counters, act,
        half, K, K // 8, tiles_per, n_tiles,
        x.stride(0),
        packed.stride(0), packed.stride(1), scale.stride(0), scale.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        act.stride(0),
        BLOCK_N=block_n, BLOCK_KW=_GEMV_SPLITK_BLOCK_KW, SPLIT_K=split_k,
        OUT=tl.bfloat16, EVEN_K=even,
        num_warps=_GEMV_WARPS,
    )
    return act


def shared_expert_gate_up(
    x: torch.Tensor, weight_t: torch.Tensor, scale_t: torch.Tensor, weight_global: torch.Tensor,
) -> torch.Tensor:
    """``silu_and_mul(gate_up(x))`` for ``x`` [M, K] bf16 as [M, N // 2] bf16, where
    :func:`shared_expert_fuses_rows` holds. The weight is the fused gate|up projection in the
    K-major resident layout of :func:`nvfp4_transpose_resident`."""
    M, K = x.shape
    N = weight_t.shape[1]
    half = N // 2
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert weight_t.shape[0] * 8 == K and N == 2 * half
    if M <= 1:
        return _gate_up_rows(x, weight_t, scale_t, weight_global)
    plan = _gemm_inkernel_plan(M, N, K, x.device)
    split_k = plan[1]
    assert M <= _GEMM_MAX_INKERNEL_M and split_k > 1, "this batch has no reduce launch to fuse into"

    act = torch.empty((M, half), dtype=x.dtype, device=x.device)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    # act stands in for the output the GEMM does not write when a reduce launch follows
    _gemm_launch(x, weight_t.t(), e4m3_kernel_view(scale_t.t()), weight_global, act, part, plan)
    _shared_gate_up_reduce_kernel[(M, triton.cdiv(half, _REDUCE_BLOCK))](
        part, weight_global, act, half,
        part.stride(0), part.stride(1), part.stride(2),
        act.stride(0),
        SPLIT_K=split_k, BLOCK=_REDUCE_BLOCK, OUT=tl.bfloat16, num_warps=_REDUCE_WARPS,
    )
    return act


def _down_add_rows(
    act: torch.Tensor, weight_t: torch.Tensor, scale_t: torch.Tensor, weight_global: torch.Tensor,
    gate: torch.Tensor, routed: torch.Tensor,
) -> torch.Tensor:
    """Every row through the one-row GEMV's statements, one program a (row, tile, K slice)."""
    M, K = act.shape
    packed = weight_t.t()
    scale = e4m3_kernel_view(scale_t.t())
    N = packed.shape[0]
    block_n, split_k, tiles_per, even = _gemv_splitk_plan(N, K, True)
    assert split_k > 1, "the single-pass GEMV shape is not fused"

    out = torch.empty((M, N), dtype=act.dtype, device=act.device)
    if M == 0:
        return out
    n_tiles = triton.cdiv(N, block_n)
    assert M * split_k * N < 2**31 and M * K < 2**31, "the kernel indexes rows in int32"
    part = torch.empty((M, split_k, N), dtype=torch.float32, device=act.device)
    counters = _splitk_counters(M * n_tiles, act.device)
    _shared_down_add_kernel[(M, n_tiles, split_k)](
        act, packed, scale, weight_global, gate, routed, part, counters, out,
        N, K, K // 8, tiles_per, n_tiles,
        act.stride(0),
        packed.stride(0), packed.stride(1), scale.stride(0), scale.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        routed.stride(0), out.stride(0),
        BLOCK_N=block_n, BLOCK_KW=_GEMV_SPLITK_BLOCK_KW, SPLIT_K=split_k,
        OUT=tl.bfloat16, EVEN_K=even,
        num_warps=_GEMV_WARPS,
    )
    return out


def shared_expert_down_add(
    act: torch.Tensor, weight_t: torch.Tensor, scale_t: torch.Tensor, weight_global: torch.Tensor,
    gate: torch.Tensor, routed: torch.Tensor,
) -> torch.Tensor:
    """``routed + down(act) * sigmoid(gate)[:, None]`` into a fresh [M, N] bf16 tensor, where
    :func:`shared_expert_fuses_rows` holds. ``act`` [M, K], ``gate`` [M] (the logit) and ``routed``
    [M, N] are bf16; the weight is the down projection in the K-major resident layout."""
    M, K = act.shape
    N = weight_t.shape[1]
    assert act.dtype == torch.bfloat16 and act.stride(1) == 1
    assert weight_t.shape[0] * 8 == K
    assert routed.shape == (M, N) and routed.dtype == act.dtype and routed.stride(1) == 1
    assert gate.shape == (M,) and gate.dtype == act.dtype and gate.is_contiguous()
    if M <= 1:
        return _down_add_rows(act, weight_t, scale_t, weight_global, gate, routed)
    plan = _gemm_inkernel_plan(M, N, K, act.device)
    split_k = plan[1]
    assert M <= _GEMM_MAX_INKERNEL_M and split_k > 1, "this batch has no reduce launch to fuse into"

    out = torch.empty((M, N), dtype=act.dtype, device=act.device)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=act.device)
    _gemm_launch(act, weight_t.t(), e4m3_kernel_view(scale_t.t()), weight_global, out, part, plan)
    _shared_down_add_reduce_kernel[(M, triton.cdiv(N, _REDUCE_BLOCK))](
        part, weight_global, gate, routed, out, N,
        part.stride(0), part.stride(1), part.stride(2),
        routed.stride(0), out.stride(0),
        SPLIT_K=split_k, BLOCK=_REDUCE_BLOCK, OUT=tl.bfloat16, num_warps=_REDUCE_WARPS,
    )
    return out


__all__ = ["shared_expert_down_add", "shared_expert_fusable", "shared_expert_fuses_rows", "shared_expert_gate_up"]
