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
# experts quantized per step: bounds the fp32 work buffers on the load device
_CHUNK = 8


def quantize_fp8_block(weight: torch.Tensor, block: int = FP8_BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., N, K]`` float -> e4m3 codes ``[..., N, K]`` and bf16 scales ``[..., N / block, K / block]``, ``weight ~= code * scale``.

    Each block's largest magnitude lands on the top of the e4m3 range."""
    *lead, n, k = weight.shape
    if n % block or k % block:
        raise ValueError(f"fp8 block quantization needs multiples of {block}, got a {n}x{k} weight")
    w = weight.to(torch.float32).reshape(*lead, n // block, block, k // block, block)
    amax = w.abs().amax(dim=(-3, -1))
    scale = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax)).to(torch.bfloat16)
    # the codes are taken against the scale as stored, not the ideal one
    codes = (w / scale.to(torch.float32)[..., :, None, :, None]).to(FP8)
    return codes.reshape(weight.shape), scale


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 codes (bit 3 is the sign), rounded to nearest even and saturated at 6."""
    a = x.abs()
    # the grid steps by 0.5 below 2, by 1 below 4 and by 2 above
    mag = torch.where(a < 2, torch.round(a * 2), torch.where(a < 4, torch.round(a - 2) + 4, torch.round((a - 4) * 0.5) + 6))
    return mag.clamp_(max=7).to(torch.uint8) | (torch.signbit(x).to(torch.uint8) << 3)


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[..., N, K]`` float -> packed e2m1 pairs ``[..., N, K / 2]`` uint8 (low nibble = even k), e4m3 block
    scales ``[..., N, K / 16]`` and fp32 row globals ``[..., N]``, ``weight ~= e2m1 * scale * global``.

    The row global puts the row's largest block scale on the top of the e4m3 range."""
    *lead, n, k = weight.shape
    if k % NVFP4_GROUP:
        raise ValueError(f"nvfp4 quantization needs a multiple of {NVFP4_GROUP} columns, got a {n}x{k} weight")
    w = weight.to(torch.float32).reshape(*lead, n, k // NVFP4_GROUP, NVFP4_GROUP)
    amax = w.abs().amax(dim=-1)
    row_max = amax.amax(dim=-1, keepdim=True)
    glob = torch.where(row_max > 0, row_max / (_E2M1_MAX * _E4M3_MAX), torch.ones_like(row_max))
    scale = (amax / (_E2M1_MAX * glob)).to(FP8)
    # the codes are taken against the scale as stored, not the ideal one
    step = scale.to(torch.float32) * glob
    inv = torch.where(step > 0, 1.0 / step, torch.zeros_like(step))
    codes = _e2m1_codes(w * inv.unsqueeze(-1)).reshape(*lead, n, k // 2, 2)
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
