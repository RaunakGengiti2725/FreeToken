"""The W4A4 NVFP4 linears: cublas (our activation quant with 128x4-swizzled scales feeding torch's NVFP4 scaled_mm) and b12x (flashinfer only),
and the sm_120 W4A16 GEMV / skinny GEMM both run for small batches on the same row-major weight.
Also the triton W4A16 linear's decode kernels, which run on any CUDA GPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sm100 = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0), reason="needs NVFP4 tensor cores (sm_100+)"
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to the even code, saturating at 6 (cvt.rn.satfinite)."""
    grid = _E2M1[:8].to(v.device)
    a = v.abs().clamp(max=6.0)
    hi = torch.searchsorted(grid, a.contiguous()).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    code = torch.where(d_lo < d_hi, lo, torch.where(d_hi < d_lo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return torch.copysign(grid[code], v)


def _qdq(x: torch.Tensor, g: float) -> torch.Tensor:
    xb = x.float().view(x.shape[0], -1, 16)
    g = torch.tensor(g, dtype=torch.float32, device=x.device)
    sf = ((xb.abs().amax(-1) / (6.0 * g)).to(torch.float8_e4m3fn).float() * g).unsqueeze(-1)
    inv = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))
    return (_round_e2m1(xb * inv) * sf).view_as(x)


def _dequant(packed: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    return _E2M1.to(packed.device)[codes] * scale.float().repeat_interleave(16, dim=1) * row_global.float().unsqueeze(1)


def _small_batch_route(N, K, M, max_tokens=16):
    if torch.cuda.get_device_capability() != (12, 0):
        return None
    from freetoken.kernel import gluon

    if not gluon.available():
        return None
    from freetoken.kernel.gluon.nvfp4.linear.sm120_gemv import gemv_config

    return gemv_config(N, K, M, max_tokens)


@sm100
@pytest.mark.parametrize("rows,K", [(1, 2048), (7, 2080), (300, 5376)])
def test_swizzled_scales_are_the_cublas_128x4_layout(rows, K):
    """Same codes either way; the swizzled scales are the linear ones in cuBLAS's [row tile, k tile, 32, 4, 4] order, zero padded."""
    from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
    from freetoken.layers.quantization.linear.nvfp4 import _swizzle_128x4

    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    g = (x.float().abs().max() / 2688).reshape(1)
    q, s = quantize_nvfp4(x, g)
    q_swz, s_swz = quantize_nvfp4(x, g, swizzled=True)
    assert torch.equal(q, q_swz)
    assert torch.equal(_swizzle_128x4(s).view(torch.uint8), s_swz.view(torch.uint8))


@sm100
@pytest.mark.parametrize("kernel_name", ["cublas", "b12x"])
@pytest.mark.parametrize("M", [1, 2, 7, 16, 300])
@pytest.mark.parametrize("N,K", [(512, 1024), (528, 2080), (12, 2048)])
@pytest.mark.parametrize("parts_differ", [False, True])
def test_w4a4_linear_matches_nvfp4_activation_reference(kernel_name, M, N, K, parts_differ):
    """finalize + apply against a torch emulation: W4A4, or W4A16 where the sm_120 small-batch kernels take the batch (M <= 16 for cublas,
    M <= 8 for b12x, at 512 x 1024; 528 x 2080 fits neither and keeps the GEMM; N = 12 must be refused, b12x returns garbage there);
    parts_differ gives the halves of a fused weight different globals, and the larger one applies to both."""
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.nvfp4 import Nvfp4LinearMethod
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    kernel = next(c for c in Nvfp4LinearMethod.candidates if c.name == kernel_name)()
    reason = kernel.unusable_reason(LinearConfig(K, N, (N // 2, N // 2), nvfp4_scheme(input_scale=True)))
    if reason:
        pytest.skip(reason)
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = (torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn)
    row_global = torch.full((N,), 1e-3, dtype=torch.float32, device="cuda")
    if parts_differ:
        row_global[N // 2:] = 4e-3
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    # off amax / 2688 by an irrational factor: with amax / 2688 many values sit exactly on an e2m1 midpoint, where the kernel's
    # approximate fp32 reciprocal and torch's exact one round apart
    g_in = float(x.float().abs().max()) / 2688 * math.pi / 3
    # one tensor-wise global: the largest part's, applied to every row
    w_ref = _dequant(weight, scale, torch.full_like(row_global, float(row_global.max())))
    small = _small_batch_route(N, K, M, 16 if kernel_name == "cublas" else 8) is not None
    ref = (x.float() if small else _qdq(x, g_in)) @ w_ref.T + bias.float()

    layer = SimpleNamespace(weight=weight.clone(), weight_scale=scale.clone(), weight_global=row_global.clone(),
                            input_scale=torch.tensor(g_in, device="cuda"), bias=bias)
    kernel.finalize(layer)
    out = kernel.apply(layer, x)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


@sm100
@pytest.mark.parametrize("N,K,T", [
    (16384, 1024, 1),  # fma, 4 rows per warp
    (16384, 768, 2),  # fma, one word per lane
    (640, 2048, 1),  # fma, four words per lane
    (640, 17408, 1),  # long K: mma
    (640, 4224, 12),  # mma bm32, masked last K block
    (256, 16512, 4),  # mma, masked last K block
    (640, 2048, 5),  # mma bm16, padded tokens
])
def test_sm120_gemv_tiles_match_dequant_reference(N, K, T):
    """Each tile gemv_config hands out, against x @ dequant(W)^T on the flat 128x4 scales."""
    route = _small_batch_route(N, K, T)
    if route is None:
        pytest.skip("needs sm_120 with Gluon")
    from freetoken.layers.quantization.linear.nvfp4 import _swizzle_128x4

    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = (torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn)
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, generator=g)
    g_w = torch.tensor(1e-3, device="cuda")
    ref = x.float() @ _dequant(weight, scale, torch.full((N,), 1e-3, device="cuda")).T
    kernel, tile = route
    out = kernel(x, weight, _swizzle_128x4(scale), g_w, **tile)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


@sm100
@pytest.mark.parametrize("kernel_name", ["cublas", "b12x"])
@pytest.mark.parametrize("M", [0, 4, 64])
def test_w4a4_linear_takes_row_strided_and_empty_inputs(kernel_name, M):
    """A row-strided slice and an empty batch give the same output as a contiguous input, on both the small-batch and the GEMM route."""
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.nvfp4 import Nvfp4LinearMethod
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    N, K = 512, 1024
    kernel = next(c for c in Nvfp4LinearMethod.candidates if c.name == kernel_name)()
    reason = kernel.unusable_reason(LinearConfig(K, N, (N,), nvfp4_scheme(input_scale=True)))
    if reason:
        pytest.skip(reason)
    g = torch.Generator(device="cuda").manual_seed(0)
    layer = SimpleNamespace(weight=torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g),
                            weight_scale=(torch.rand(N, K // 16, device="cuda", generator=g) + 0.25).to(torch.float8_e4m3fn),
                            weight_global=torch.full((N,), 1e-3, dtype=torch.float32, device="cuda"),
                            input_scale=torch.tensor(0.01, device="cuda"), bias=None)
    kernel.finalize(layer)
    buf = torch.randn(M, K + 64, device="cuda", dtype=torch.bfloat16, generator=g)
    out = kernel.apply(layer, buf[:, :K])
    assert out.shape == (M, N)
    torch.testing.assert_close(out, kernel.apply(layer, buf[:, :K].contiguous()), rtol=0, atol=0)


def _dense_linear_before_the_split(x, packed, scale, row_global, transposed):
    """nvfp4_linear's host code for 1 to 64 rows as it stood at 4b71d4d, before the plan and the launch of its GEMV
    and dot GEMM became functions of their own, over the same kernels. ``packed`` and ``scale`` are the logical views."""
    import triton
    import triton.language as tl

    from freetoken.kernel.triton import nvfp4_linear as nl

    M, K = x.shape
    N, words = packed.shape[0], K // 8
    scale = nl.e4m3_kernel_view(scale)
    strides = (packed.stride(0), packed.stride(1), scale.stride(0), scale.stride(1))
    tiles = triton.cdiv(words, 16)
    if M == 1:
        out = torch.empty(N, dtype=x.dtype, device=x.device)
        block_n = 32 if transposed else 16
        n_blocks = triton.cdiv(N, block_n)
        split_k = max(1, min(1024 // n_blocks, tiles))
        split_k = 1 << (split_k.bit_length() - 1)
        if split_k == 1:
            block_kw = 16 if transposed else 32
            nl._nvfp4_gemv_kernel[(triton.cdiv(N, 32),)](
                x[0], packed, scale, row_global, out, N, K, *strides,
                BLOCK_N=32, BLOCK_KW=block_kw, OUT=tl.bfloat16, EVEN_K=K % (block_kw * 8) == 0, num_warps=4,
            )
            return out[None]
        tiles_per = triton.cdiv(tiles, split_k)
        part = torch.empty((split_k, N), dtype=torch.float32, device=x.device)
        nl._nvfp4_gemv_splitk_kernel[(n_blocks, split_k)](
            x[0], packed, scale, row_global, part, nl._splitk_counters(n_blocks, x.device), out, N, K, words, tiles_per,
            *strides, part.stride(0), part.stride(1),
            BLOCK_N=block_n, BLOCK_KW=16, SPLIT_K=split_k, OUT=tl.bfloat16,
            EVEN_K=K % 128 == 0 and tiles == tiles_per * split_k, num_warps=4,
        )
        return out[None]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    block_m = 16 if M <= 16 else (32 if M <= 32 else 64)
    num_mn = triton.cdiv(M, block_m) * triton.cdiv(N, 128)
    if block_m == 16:
        split_k = nl._pick_split_k_bm16(num_mn, tiles, nl._gemm_wave(x.device))
    else:
        split_k = max(1, min(1024 // num_mn, tiles))
        split_k = 1 << (split_k.bit_length() - 1)
        even_k = split_k
        while even_k > 1 and tiles % even_k != 0:
            even_k //= 2
        if even_k * 4 >= split_k:
            split_k = even_k
    tiles_per = triton.cdiv(tiles, split_k)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device) if split_k > 1 else out
    nl._nvfp4_gemm_kernel[(num_mn, split_k)](
        x, packed, scale, row_global, out, part, M, N, K, words, tiles_per,
        x.stride(0), x.stride(1), *strides, out.stride(0), out.stride(1), *(part.stride() if split_k > 1 else (0, 0, 0)),
        BLOCK_M=block_m, BLOCK_N=128, BLOCK_KW=16, SPLIT_K=split_k, OUT=tl.bfloat16,
        EVEN_K=K % 128 == 0 and tiles == tiles_per * split_k, num_warps=8 if block_m == 64 else 4, num_stages=3,
    )
    if split_k > 1:
        block, warps = (128, 2) if N <= 8192 else (512, 8)
        nl._nvfp4_gemm_splitk_reduce_kernel[(M, triton.cdiv(N, block))](
            part, row_global, out, M, N, split_k, *part.stride(), out.stride(0), out.stride(1),
            BLOCK=block, OUT=tl.bfloat16, num_warps=warps,
        )
    return out


@pytest.mark.parametrize("transposed", [True, False])
# Qwen3.6-35B-A3B's shared expert gate_up and down, a partial last tile along N and K, and an N wide enough for the single-pass GEMV
@pytest.mark.parametrize("N,K", [(1024, 2048), (2048, 512), (2064, 528), (49152, 256)])
def test_w4a16_decode_linear_keeps_its_bits_across_the_plan_and_launch_split(N, K, transposed):
    """One row (the GEMV) and 2 to 64 rows (the dot GEMM) through the W4A16 linear, bit for bit against its host code
    from before the split. The plan fixes the reduction order, so a plan that drifts shows here as a changed bit."""
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear, nvfp4_dense_linear_t, nvfp4_transpose_resident

    gen = torch.Generator(device="cuda").manual_seed(N + K)
    weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=gen)
    scale = (torch.rand(N, K // 16, device="cuda", generator=gen) + 0.25).to(torch.float8_e4m3fn)
    row_global = torch.rand(N, device="cuda", generator=gen) * 0.05 + 0.01
    weight_t, scale_t = nvfp4_transpose_resident(weight, scale)
    logical = (weight_t.t(), scale_t.t()) if transposed else (weight.view(torch.int32), scale)
    for M in range(1, 65):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=gen)
        if transposed:
            out = nvfp4_dense_linear_t(x, weight_t, scale_t, row_global)
        else:
            out = nvfp4_dense_linear(x, weight, scale, row_global)
        ref = _dense_linear_before_the_split(x, *logical, row_global, transposed)
        differ = out.view(torch.int16) != ref.view(torch.int16)
        assert not bool(differ.any()), f"{M} rows: {int(differ.sum())} of {differ.numel()} elements differ"
