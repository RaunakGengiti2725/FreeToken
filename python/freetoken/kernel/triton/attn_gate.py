"""The output gate of gated attention on a decode step, ``attn * sigmoid(gate)``, in one kernel
that reads the gate where the fused q|gate projection wrote it, in place of a copy that gathers
it, torch's sigmoid and torch's mul. Rounded to bf16 where torch's two kernels round, so the
result is theirs bit for bit. CUDA-graph safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from freetoken.kernel.backend import is_rocm

_WARPS = 2


@triton.jit
def _round_bf16(x):
    """fp32 onto the bf16 grid, round to nearest even, in integer arithmetic: the compiler folds
    a cast to bf16 and back away, and the mul after it would then round once."""
    b = x.to(tl.uint32, bitcast=True)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    # an inf or a NaN passes through: the carry out of a NaN's low bits would leave -0
    return tl.where((b & 0x7F800000) == 0x7F800000, b, r).to(tl.float32, bitcast=True)


@triton.jit
def _attn_gate_kernel(
    attn_ptr,   # [M, HEADS * D] attention output, unit inner stride
    qg_ptr,     # [M, HEADS, 2 * D] each head's q|gate projection: q, then the gate; unit inner stride
    out_ptr,    # [M, HEADS * D]
    D,
    stride_am, stride_gm, stride_gh, stride_om,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    """``(row, head)``: one head's gated output."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    gate = tl.load(qg_ptr + row * stride_gm + head * stride_gh + D + offs, mask=mask, other=0.0).to(tl.float32)
    attn = tl.load(attn_ptr + row * stride_am + head * D + offs, mask=mask, other=0.0).to(tl.float32)
    # torch's sigmoid of a bf16 tensor is the fp32 formula rounded once; the store rounds the product
    scale = _round_bf16(tl.div_rn(1.0, 1.0 + libdevice.exp(-gate)))
    tl.store(out_ptr + row * stride_om + head * D + offs, (attn * scale).to(OUT), mask=mask)


def attn_gate_fusable() -> bool:
    """The gate of this module: the kernel copies the rounding of torch's CUDA sigmoid and mul,
    which is checked on NVIDIA only."""
    return not is_rocm()


def attn_gate_mul(attn: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
    """``attn * sigmoid(qg[..., D:])`` into a fresh [M, HEADS * D] bf16 tensor. ``attn`` [M, HEADS * D]
    bf16; ``qg`` [M, HEADS, 2 * D] bf16 at any row and head strides, read in place."""
    M, heads, width = qg.shape
    d = width // 2
    assert width == 2 * d and attn.shape == (M, heads * d)
    assert attn.dtype == torch.bfloat16 and qg.dtype == torch.bfloat16
    assert attn.stride(1) == 1 and qg.stride(2) == 1
    out = torch.empty((M, heads * d), dtype=attn.dtype, device=attn.device)
    if M == 0:
        return out
    assert M * max(qg.stride(0), attn.stride(0)) < 2**31, "the kernel indexes rows in int32"
    _attn_gate_kernel[(M, heads)](
        attn, qg, out, d,
        attn.stride(0), qg.stride(0), qg.stride(1), out.stride(0),
        BLOCK=triton.next_power_of_2(d), OUT=tl.bfloat16,
        num_warps=_WARPS,
    )
    return out


__all__ = ["attn_gate_fusable", "attn_gate_mul"]
