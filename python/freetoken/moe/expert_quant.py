"""Load-time quantization of bf16 routed experts into a format an expert kernel already serves.

For experts the checkpoint stores unquantized and the engine keeps resident (an MTP draft layer). The
bf16 pieces become the pieces a quantized checkpoint would deliver, so the kernel's own pack fills the
resident banks and its own forward reads them: no bank layout or kernel is specific to this path.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch

from freetoken.layers.quantization import KernelSelectionError, LayerKind, MoEConfig, QuantKind, method_class
from freetoken.layers.quantization.moe.base import fused_piece
from freetoken.layers.quantization.scheme import FP8_BLOCK, NVFP4_GROUP, fp8_block_scheme, nvfp4_scheme

from .expert_banks import _method_expert_bytes

FP8 = torch.float8_e4m3fn
# --speculative-draft-quant value -> the expert kind it quantizes to
EXPERT_QUANTS = {"fp8": QuantKind.FP8_BLOCK, "nvfp4": QuantKind.NVFP4}
_GROUPS = {QuantKind.FP8_BLOCK: FP8_BLOCK, QuantKind.NVFP4: NVFP4_GROUP}
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
# between neighbouring e2m1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# multiplied by, never divided by: CUDA evaluates tensor / python scalar as a multiply by the reciprocal, the CPU
# as a division, and the two round apart
_FP8_SCALE = 1.0 / _E4M3_MAX
_NVFP4_GLOBAL = 1.0 / (_E2M1_MAX * _E4M3_MAX)
_TINY = torch.finfo(torch.float32).tiny
# experts quantized per step: bounds the work buffers on the load device
_CHUNK = 8


def quantize_fp8_block(weight: torch.Tensor, block: int = FP8_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., N, K]`` float -> e4m3 codes ``[..., N, K]`` and bf16 scales ``[..., N / block, K / block]``, ``weight ~= code * scale``.

    Each block's largest magnitude lands on the top of the e4m3 range. A block too small to scale (all zeros, or
    under fp32's normal range) keeps scale 1 and zero codes."""
    *lead, n, k = weight.shape
    if n % block or k % block:
        raise ValueError(f"fp8 block quantization needs multiples of {block}, got a {n}x{k} weight")
    w = weight.to(torch.float32).reshape(*lead, n // block, block, k // block, block)
    amax = w.abs().amax(dim=(-3, -1))
    scale = (amax * _FP8_SCALE).to(torch.bfloat16)
    # a denormal scale is too coarse to divide by, and one rounded to zero turns a zero weight into NaN
    scale = torch.where(scale >= _TINY, scale, torch.ones_like(scale))
    # the codes are taken against the scale as stored, not the ideal one
    codes = (w / scale.to(torch.float32)[..., :, None, :, None]).to(FP8)
    return codes.reshape(weight.shape), scale


def _e2m1_codes(w: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """The e2m1 codes (bit 3 is the sign) of fp32 ``w`` over the float64 ``step``: the nearest value, a tie to the
    even code, saturated at 6. A zero step keeps code 0, its block decodes to zero under any code.

    ``w`` is compared with midpoint * step, which float64 holds exactly: no rounded quotient decides a code, so
    every device writes the same one."""
    a = w.abs().to(torch.float64)
    step = torch.where(step > 0, step, torch.inf)
    mag = torch.zeros(a.shape, dtype=torch.uint8, device=a.device)
    for upper, mid in enumerate(_E2M1_MIDPOINTS, 1):
        edge = mid * step
        # on the midpoint the even code of the two wins
        mag += (a > edge) if upper % 2 else (a >= edge)
    return mag | (torch.signbit(w).to(torch.uint8) << 3)


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[..., N, K]`` float -> packed e2m1 pairs ``[..., N, K / 2]`` uint8 (low nibble = even k), e4m3 block
    scales ``[..., N, K / 16]`` and fp32 row globals ``[..., N]``, ``weight ~= e2m1 * scale * global``.

    The row global puts the row's largest block scale on the top of the e4m3 range. A row too small to scale (all
    zeros, or under fp32's normal range) keeps global 1 and zero scales."""
    *lead, n, k = weight.shape
    if k % NVFP4_GROUP:
        raise ValueError(f"nvfp4 quantization needs a multiple of {NVFP4_GROUP} columns, got a {n}x{k} weight")
    w = weight.to(torch.float32).reshape(*lead, n, k // NVFP4_GROUP, NVFP4_GROUP)
    amax = w.abs().amax(dim=-1)
    glob = amax.amax(dim=-1, keepdim=True) * _NVFP4_GLOBAL
    # a denormal global is too coarse to scale with, and one rounded to zero turns a zero block's scale into NaN
    glob = torch.where(glob >= _TINY, glob, torch.ones_like(glob))
    scale = (amax / (_E2M1_MAX * glob)).to(FP8)
    # the codes are taken against the scale as stored, not the ideal one; float64 holds scale * global exactly
    step = scale.to(torch.float32).to(torch.float64) * glob
    codes = _e2m1_codes(w, step.unsqueeze(-1)).reshape(*lead, n, k // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scale, glob.squeeze(-1)


# kind -> (quantizer, the piece suffixes its outputs take, in order)
_QUANTIZERS = {
    QuantKind.FP8_BLOCK: (quantize_fp8_block, ("", "_scale")),
    QuantKind.NVFP4: (quantize_nvfp4, ("", "_scale", "_global")),
}


def quantized_pieces(pieces: Iterable, kind: QuantKind, device: torch.device) -> Iterator:
    """bf16 expert pieces as the pieces of a ``kind`` checkpoint, quantized on ``device`` a few experts at a time."""
    quantize, suffixes = _QUANTIZERS[kind]
    for layer_id, e0, e1, piece in pieces:
        if extra := set(piece) - {"gate_up", "gate", "up", "down"}:
            raise ValueError(f"experts with {sorted(extra)} tensors are already quantized")
        for start in range(e0, e1, _CHUNK):
            stop = min(start + _CHUNK, e1)
            part = {role: tensor[start - e0 : stop - e0].to(device) for role, tensor in piece.items()}
            out = {}
            for proj, weight in (("gate_up", fused_piece(part, "gate_up")), ("down", part["down"])):
                out.update(zip((proj + s for s in suffixes), quantize(weight)))
            yield layer_id, start, stop, out
    if device.type == "cuda":
        # the cache budget reads free memory: return the work buffers before a later allocation lands in their blocks
        torch.cuda.empty_cache()


def check_expert_quant(quant: str, hidden: int, intermediate: int) -> QuantKind:
    """The expert kind ``quant`` names; raises ValueError for an unknown name or experts its blocks do not tile."""
    if quant not in EXPERT_QUANTS:
        raise ValueError(f"--speculative-draft-quant must be one of none, {', '.join(EXPERT_QUANTS)}, got {quant!r}")
    kind = EXPERT_QUANTS[quant]
    group = _GROUPS[kind]
    if hidden % group or intermediate % group:
        raise ValueError(
            f"--speculative-draft-quant {quant} needs expert dims that are multiples of {group}, "
            f"got hidden {hidden} and intermediate {intermediate}"
        )
    return kind


def expert_quant_method(layer, quant: str):
    """The expert method that serves ``layer``'s bf16 experts as ``quant``, on the Triton kernel of that kind."""
    stored = layer.quant_method.kind
    if stored is not QuantKind.NONE:
        raise ValueError(f"--speculative-draft-quant {quant}: the experts of {layer.prefix} are already {stored} in the checkpoint")
    kind = check_expert_quant(quant, layer.hidden_size, layer.intermediate_size)
    scheme = fp8_block_scheme("float") if kind is QuantKind.FP8_BLOCK else nvfp4_scheme(input_scale=False)
    try:
        return method_class(kind, LayerKind.MOE)(MoEConfig.from_layer(layer, scheme), "triton")
    except KernelSelectionError as exc:
        raise ValueError(f"--speculative-draft-quant {quant}: {exc}") from exc


def slot_bytes(method) -> int:
    """Bytes of one expert in ``method``'s banks."""
    return _method_expert_bytes(method) // method.cfg.num_experts


__all__ = [
    "EXPERT_QUANTS",
    "check_expert_quant",
    "expert_quant_method",
    "quantize_fp8_block",
    "quantize_nvfp4",
    "quantized_pieces",
    "slot_bytes",
]
