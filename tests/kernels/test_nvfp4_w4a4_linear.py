"""The W4A4 NVFP4 linears: cublas (our activation quant with 128x4-swizzled scales feeding torch's NVFP4 scaled_mm) and b12x (flashinfer only),
and the sm_120 W4A16 GEMV / skinny GEMM both run for small batches on the same row-major weight.
Also the triton W4A16 linear's decode kernels, which run on any CUDA GPU, and the fused decode kernels of the gated
shared expert and of the routed experts' sum against the projection, sum and elementwise kernels they replace."""

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


# 17 and 33 rows take the dot GEMM's wider row tiles; 64 is the most it takes
_ROWS = [1, 2, 3, 4, 8, 17, 33, 64]
# Qwen3.6-35B-A3B's shared expert, and one with a partial last tile along both N and K
_SHARED = [(2048, 512), (2064, 528)]
_KINDS = ["decode", "cancel", "large", "denormal", "ties"]


def _skip_unless_the_shared_expert_fuses():
    from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_fusable

    if not shared_expert_fusable(2048, 512):
        pytest.skip("the shared expert fusion is off here: its silu copies flashinfer's, which is not installed")


def _resident(n, k, kind, row_global, gen):
    """An NVFP4 projection in the K-major resident layout. "random" codes; "paired" gives inputs 2j and 2j + 1 one code;
    "tap" makes output n the input n % k times one, so the row global alone sets what the projection rounds."""
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_transpose_resident

    if kind == "tap":
        rows = torch.arange(n, device="cuda")
        col = rows % k
        weight = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda")
        weight[rows, col // 2] = torch.where(col % 2 == 0, 0x02, 0x20).to(torch.uint8)  # e2m1 code of 1.0
        scale = torch.ones(n, k // 16, device="cuda").to(torch.float8_e4m3fn)
        return *nvfp4_transpose_resident(weight, scale), row_global
    if kind == "paired":
        weight = (torch.randint(0, 16, (n, k // 2), device="cuda", generator=gen) * 17).to(torch.uint8)
    else:
        weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    scale = (torch.rand(n, k // 16, device="cuda", generator=gen) + 0.25).to(torch.float8_e4m3fn)
    return *nvfp4_transpose_resident(weight, scale), row_global


def _bf16_grid(shape, lo, hi, gen):
    """Random bf16 values of either sign with an exponent in [lo, hi)."""
    mantissa = torch.randint(128, 256, shape, device="cuda", generator=gen).float() / 128
    sign = torch.randint(0, 2, shape, device="cuda", generator=gen).float() * 2 - 1
    return torch.ldexp(sign * mantissa, torch.randint(lo, hi, shape, device="cuda", generator=gen)).to(torch.bfloat16)


def _bf16_denormals(shape, gen):
    bits = torch.randint(1, 128, shape, device="cuda", generator=gen) - 32768 * torch.randint(0, 2, shape, device="cuda", generator=gen)
    return bits.to(torch.int16).view(torch.bfloat16)


def _mirrored(M, k, gen):
    """Inputs 2j and 2j + 1 are v and -v, every third pair one bf16 step apart in magnitude: under paired weights a
    row's products cancel to a residue that depends on the order they are summed in."""
    base = torch.randn(M, k // 2, device="cuda", generator=gen).to(torch.bfloat16)
    mirror = (-base).view(torch.int16).clone()
    mirror[:, ::3] += 1
    return torch.stack([base, mirror.view(torch.bfloat16)], dim=-1).view(M, k).contiguous()


def _cycle(n, values):
    return torch.tensor(values, dtype=torch.float32, device="cuda")[torch.arange(n, device="cuda") % len(values)]


def _shared_case(kind, M, hidden, inter, gen):
    """A shared expert and one step of M rows through it. ``act`` is the down projection's input where the case sets
    it (the silu-mul's output otherwise); ``routed(scaled)`` is the routed output for a gated shared output."""
    weights = {"decode": "random", "cancel": "paired"}.get(kind, "tap")
    gate_up_global = torch.full((2 * inter,), 0.012, dtype=torch.float32, device="cuda")
    gate_up_global[inter:] = 0.02  # the halves of gate_up under different globals
    down_global = torch.full((hidden,), 0.03, dtype=torch.float32, device="cuda")
    x = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
    fixed = torch.randn(M, hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
    logit = (torch.randn(M, device="cuda", generator=gen) * 2).to(torch.bfloat16)
    act = None
    routed = lambda scaled: fixed  # noqa: E731
    rows = torch.arange(M, device="cuda")

    if kind == "cancel":
        # products that cancel to near zero in both projections and in the add of the routed output
        x, act = _mirrored(M, hidden, gen), _mirrored(M, inter, gen)

        def routed(scaled):
            mirror = (-scaled).view(torch.int16).clone()
            mirror.view(-1)[::3] += 1
            return mirror.view(torch.bfloat16)
    elif kind == "large":
        # outputs up to and past the largest bf16 value; inputs stay under fp16's 65504, which the dot GEMM converts them to
        x, act = _bf16_grid((M, hidden), 13, 15, gen), _bf16_grid((M, inter), 13, 15, gen)
        steps = [2.0**112, 2.0**112 * (1 + 2.0**-8), 2.0**-8, 2.0**-20, 2.0**113]
        gate_up_global, down_global = _cycle(2 * inter, steps), _cycle(hidden, steps)
        # silu gates of -87.5, -88 and -88.5 (columns under the 2^-8 global): flashinfer's approximate divide gives zero there
        x[:, 2], x[:, 7], x[:, 12] = -87.5 * 2.0**8, -88.0 * 2.0**8, -88.5 * 2.0**8
        big, inf = torch.finfo(torch.bfloat16).max, float("inf")
        pool = [big, -big, 88.5, -88.5, 100.0, -100.0, 16.5, -16.5, 0.0, -0.0, inf, -inf, 1e-30, -1e-30, 744.0, -744.0]
        logit = torch.tensor(pool, device="cuda").to(torch.bfloat16)[(rows * 5 + M) % len(pool)]
        fixed = _bf16_grid((M, hidden), 110, 127, gen)
    elif kind == "denormal":
        # outputs that land among the bf16 denormals, which flashinfer's silu-mul takes as zero
        x, act = _bf16_grid((M, hidden), -12, -4, gen), _bf16_grid((M, inter), -12, -4, gen)
        steps = [2.0**-121, 2.0**-118, 2.0**-125, 2.0**-121 * (1 + 2.0**-8), 2.0**-110]
        gate_up_global, down_global = _cycle(2 * inter, steps), _cycle(hidden, steps)
        # the first 32 gate, up and down outputs read denormal inputs under a global that makes them normal again
        x[:, :32], x[:, inter:inter + 32], act[:, :32] = (_bf16_denormals((M, 32), gen) for _ in range(3))
        gate_up_global[:32], gate_up_global[inter:inter + 32], down_global[:32] = 2.0**100, 2.0**100, 2.0**100
        # denormal and barely normal gates against large ups, then gates near one against denormal ups
        gate_up_global[inter + 32:inter + 96], gate_up_global[96:160] = 2.0**120, 2.0**8
        fixed = _bf16_grid((M, hidden), -133, -118, gen)
    elif kind == "ties":
        # powers of two; every fourth column 1 + 2^-7 times one, which times a sigmoid of 0.75 is a tie
        def inputs(k):
            v = _bf16_grid((M, k), -8, 9, gen).float()
            v = torch.ldexp(torch.sign(v), torch.frexp(v).exponent)
            v[:, 3::4] *= 1 + 2.0**-7
            return v.to(torch.bfloat16)

        x, act = inputs(hidden), inputs(inter)
        gate_up_global = _cycle(2 * inter, [1 + 2.0**-8, 1 + 3 * 2.0**-8, 1.0, 1.5])
        down_global = _cycle(hidden, [1.0, 1 + 2.0**-8, 1 + 3 * 2.0**-8, 1.0])
        # the bf16 value nearest ln 3, whose sigmoid rounds to 0.75, and 0, whose sigmoid is 0.5
        logit = torch.tensor([1.1015625, 0.0], device="cuda").to(torch.bfloat16)[rows % 2]

        def routed(scaled):
            # half a bf16 step of each value, or one and a half: the sum is a tie either way
            half = torch.ldexp(torch.ones_like(scaled, dtype=torch.float32), torch.frexp(scaled.float()).exponent - 9)
            half[:, 1::2] *= 3
            return torch.copysign(half, scaled.float()).to(torch.bfloat16)

    return SimpleNamespace(
        gate_up=_resident(2 * inter, hidden, weights, gate_up_global, gen),
        down=_resident(hidden, inter, weights, down_global, gen),
        x=x, act=act, logit=logit, routed=routed,
    )


def _assert_same_bits(out, ref, what):
    """Equal bf16 bits; a NaN matches a NaN of any payload."""
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16, f"{what}: {out.shape} {out.dtype} against {ref.shape} {ref.dtype}"
    differ = (out.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)) & ~(out.isnan() & ref.isnan())
    if bool(differ.any()):
        first = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(differ.sum())} of {out.numel()} elements differ, first at {first}: "
            f"fused {float(out[first])!r}, unfused {float(ref[first])!r}"
        )


def _ties(v):
    """How many fp32 values lie exactly halfway between two bf16 values."""
    return int(((v.contiguous().view(torch.int32) & 0xFFFF) == 0x8000).sum())


@pytest.mark.parametrize("M", _ROWS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("hidden,inter", _SHARED)
def test_fused_shared_expert_is_the_unfused_kernels_bit_for_bit(hidden, inter, kind, M):
    """A decode step of M rows through the two fused kernels, against the projections, flashinfer's silu-mul and
    torch's sigmoid, mul and add. Each kind asserts that it reaches the rounding edge it is built for."""
    _skip_unless_the_shared_expert_fuses()
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t
    from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_down_add, shared_expert_fuses_rows, shared_expert_gate_up
    from freetoken.layers import silu_and_mul

    if not shared_expert_fuses_rows(M, hidden, inter, torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("the dot GEMM reduces this shape in one K slice at this row count, so the step runs unfused")
    gen = torch.Generator(device="cuda").manual_seed(hidden + 131 * M)
    for _ in range(2):  # a launch must leave the arrival counters zeroed for the next
        case = _shared_case(kind, M, hidden, inter, gen)
        gate_up_out = nvfp4_dense_linear_t(case.x, *case.gate_up)
        act_ref = silu_and_mul(gate_up_out)
        _assert_same_bits(shared_expert_gate_up(case.x, *case.gate_up), act_ref, f"silu, {M} rows")
        silu_gate, silu_up = gate_up_out[:, :inter].float(), gate_up_out[:, inter:].float()
        if kind == "ties":
            assert _ties(case.x[:, : 2 * inter].float() * case.gate_up[2]), "no projection output is a tie"
        elif kind == "large":
            divisor = 1 + torch.exp(-silu_gate.double())
            assert ((divisor > 2.0**126) & (divisor < 2.0**128) & silu_up.isfinite() & (silu_up != 0)).any(), \
                "no gate is in the approximate divide's zero range"
        elif kind == "denormal":
            tiny = torch.finfo(torch.float32).tiny
            assert ((silu_gate != 0) & (silu_gate.abs() < tiny) & (silu_up.abs() > 1)).any(), "no denormal gate meets a large up"
            assert ((silu_up != 0) & (silu_up.abs() < tiny) & (silu_gate.abs() > 2.0**-5)).any(), "no denormal up meets a gate near one"

        act = act_ref if case.act is None else case.act
        shared = nvfp4_dense_linear_t(act, *case.down)
        sigmoid = torch.sigmoid(case.logit)
        scaled = shared * sigmoid[:, None]
        routed = case.routed(scaled)
        if kind == "ties":
            assert _ties(shared.float() * sigmoid.float()[:, None]), "no gate product is a tie"
            assert _ties(routed.float() + scaled.float()), "no sum is a tie"
        out = shared_expert_down_add(act, *case.down, case.logit, routed)
        _assert_same_bits(out, routed + scaled, f"down, {M} rows")


@pytest.mark.parametrize("route", ["one row", "reduce"])
def test_fused_shared_expert_sigmoid_is_torch_sigmoid_for_every_bf16_logit(route):
    """The sigmoid inside each fused down kernel against torch.sigmoid on every bf16 value but the NaNs, one a row.
    A row's down output is exactly 1.0 in column 0 and its routed output is zero, so column 0 is the sigmoid itself."""
    _skip_unless_the_shared_expert_fuses()
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_transpose_resident
    from freetoken.kernel.triton.nvfp4_shared_expert import _down_add_rows, shared_expert_down_add

    logit = torch.arange(-32768, 32768, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    logit = logit[~logit.isnan()].contiguous()
    M, N, K = logit.numel(), 32, 256
    weight = torch.zeros(N, K // 2, dtype=torch.uint8, device="cuda")
    weight[0, 0] = 0x02  # e2m1 code of 1.0 at row 0, k = 0
    scale = torch.ones(N, K // 16, device="cuda").to(torch.float8_e4m3fn)
    down = (*nvfp4_transpose_resident(weight, scale), torch.ones(N, dtype=torch.float32, device="cuda"))
    act = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    act[:, 0] = 1.0
    routed = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    if route == "one row":
        # the kernel of a one-row step over every row at once
        out = _down_add_rows(act, *down, logit, routed)
    else:
        # 64 rows a call: the most the dot GEMM and its reduce take
        out = torch.cat([
            shared_expert_down_add(act[r:r + 64], *down, logit[r:r + 64], routed[r:r + 64]) for r in range(0, M, 64)
        ])
    _assert_same_bits(out[:, 0], torch.sigmoid(logit), "sigmoid")
    assert not out[:, 1:].any()


@pytest.mark.parametrize("route", ["one row", "reduce"])
def test_fused_shared_expert_silu_is_flashinfers_for_every_gate_at_decode_magnitudes(route):
    """The silu-mul inside each fused gate_up kernel against flashinfer's, on every bf16 gate from 2^-8 to 2^8 in
    magnitude, each against 4096 ups: a divide that is not flashinfer's own shows about once in 2^15 elements.
    The projection reads each input once at a weight of one, so its outputs are the inputs themselves."""
    _skip_unless_the_shared_expert_fuses()
    from freetoken.kernel.triton.nvfp4_shared_expert import _gate_up_rows, shared_expert_gate_up
    from freetoken.layers import silu_and_mul

    half = 256
    gate_up = _resident(2 * half, 2 * half, "tap", torch.ones(2 * half, device="cuda"), None)
    gates = torch.arange(-32768, 32768, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    gates = gates[(gates.abs() >= 2.0**-8) & (gates.abs() < 2.0**8)]
    assert gates.numel() == 4096
    rows = gates.numel() * 4096 // half
    x = torch.empty(rows, 2 * half, dtype=torch.bfloat16, device="cuda")
    x[:, :half] = gates.repeat(4096).view(rows, half)
    x[:, half:] = _bf16_grid((rows, half), -2, 3, torch.Generator(device="cuda").manual_seed(0))

    # the kernel of a one-row step over 4096 rows at once; 64 rows a call are the most the dot GEMM and its reduce take
    fused, step = (_gate_up_rows, 4096) if route == "one row" else (shared_expert_gate_up, 64)
    for start in range(0, rows, step):
        chunk = x[start:start + step]
        _assert_same_bits(fused(chunk, *gate_up), silu_and_mul(chunk), f"silu of rows {start} to {start + step}")


_TOP_K = 8  # Qwen3.6-35B-A3B routes a token to 8 experts
_ROUTE_KINDS = ["decode", "cancel", "large", "denormal", "ties", "zeros"]


def _route_outputs(kind, scaled, gen):
    """Each route's output [M, _TOP_K, N] bf16, for a step whose gated shared output is ``scaled`` [M, N]."""
    M, N = scaled.shape
    shape = (M, _TOP_K, N)
    if kind == "decode":
        return (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    if kind == "zeros":
        # the sum kernel adds onto +0, so routes of -0 sum to +0; a sum that began with its first route would give -0
        return torch.full(shape, -0.0, dtype=torch.bfloat16, device="cuda")
    if kind == "cancel":
        # pairs of large routes that cancel around small ones, and a last route that cancels the shared output
        routes = _bf16_grid(shape, -22, -14, gen)
        big = _bf16_grid((M, _TOP_K // 4, N), 8, 12, gen)
        mirror = (-big).view(torch.int16).clone()
        mirror[..., ::3] += 1
        routes[:, 0::4], routes[:, 2::4] = big, mirror.view(torch.bfloat16)
        last = (-scaled).view(torch.int16).clone()
        last.view(-1)[::3] += 1
        routes[:, -1] = last.view(torch.bfloat16)
        return routes
    if kind == "large":
        routes = _bf16_grid(shape, 124, 128, gen)
        big, inf = torch.finfo(torch.bfloat16).max, float("inf")
        head = torch.zeros(_TOP_K, 6, device="cuda")
        # by column: +inf; inf - inf; the largest value plus half a step, and plus less; its mirror; a sum past fp32 that comes back
        head[0] = torch.tensor([inf, inf, big, big, -big, big])
        head[1] = torch.tensor([1.0, -inf, 2.0**119, 2.0**118, -(2.0**119), big])
        head[2, 5] = -big
        routes[:, :, :6] = head.to(torch.bfloat16)
        return routes
    if kind == "denormal":
        routes = _bf16_denormals(shape, gen)
        routes[:, ::2, 1::2] = _bf16_grid((M, _TOP_K // 2, N // 2), -126, -120, gen)
        return routes
    routes = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    v = _bf16_grid((M, N), -8, 9, gen)
    # even columns: a value and half a bf16 step of it, or one and a half
    half = torch.ldexp(torch.ones(M, N, device="cuda"), torch.frexp(v.float()).exponent - 9)
    half[:, 2::4] *= 3
    routes[:, 0], routes[:, 3] = v, torch.copysign(half, v.float()).to(torch.bfloat16)
    # odd columns: the routes sum to half a bf16 step of the shared output, in two parts
    half = torch.ldexp(torch.ones(M, N, device="cuda"), torch.frexp(scaled.float()).exponent - 10)
    half = torch.copysign(half, scaled.float()).to(torch.bfloat16)
    routes[:, :, 1::2] = 0
    routes[:, 2, 1::2], routes[:, 5, 1::2] = half[:, 1::2], half[:, 1::2]
    return routes


@pytest.mark.parametrize("M", _ROWS)
@pytest.mark.parametrize("kind", _ROUTE_KINDS)
@pytest.mark.parametrize("hidden,inter", _SHARED)
def test_fused_route_sum_is_the_sum_kernel_bit_for_bit(hidden, inter, kind, M):
    """Each route's output handed to the fused down kernel, against moe_sum_reduce_kernel and the unfused kernels
    after it: one row sums in the GEMV kernel, 2 to 64 rows in the reduce kernel."""
    _skip_unless_the_shared_expert_fuses()
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t
    from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_down_add, shared_expert_fuses_rows
    from freetoken.layers import silu_and_mul

    if not shared_expert_fuses_rows(M, hidden, inter, torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("the dot GEMM reduces this shape in one K slice at this row count, so the step runs unfused")
    gen = torch.Generator(device="cuda").manual_seed(7 * hidden + 131 * M)
    case = _shared_case("decode", M, hidden, inter, gen)
    # a sigmoid of zero leaves a gated shared output of -0 wherever the shared output is negative
    logit = torch.full_like(case.logit, float("-inf")) if kind == "zeros" else case.logit
    act = silu_and_mul(nvfp4_dense_linear_t(case.x, *case.gate_up))
    scaled = nvfp4_dense_linear_t(act, *case.down) * torch.sigmoid(logit)[:, None]
    routes = _route_outputs(kind, scaled, gen)
    routed = torch.empty_like(scaled)
    moe_sum_reduce_triton(routes, routed)
    # the fp32 partial sums in route order, to check that the kind reaches its edge
    acc = routes.float().cumsum(dim=1)
    if kind == "ties":
        assert _ties(acc[:, -1]), "no sum of routes is a tie"
        assert _ties(routed.float() + scaled.float()), "no add of the shared output is a tie"
    elif kind == "large":
        assert bool(routed.isinf().any()) and bool(routed.isnan().any()) and not bool(acc.isfinite().all())
    elif kind == "zeros":
        assert bool((scaled.view(torch.int16) == -32768).any()), "no gated shared output is -0"

    out = shared_expert_down_add(act, *case.down, logit, routes)
    _assert_same_bits(out, routed + scaled, f"sum, {kind}, {M} rows")
