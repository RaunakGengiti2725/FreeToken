"""Load-time quantization of bf16 routed experts (--speculative-draft-quant): the quantizers against brute-force
nearest-value searches on their grids, and the resident layer they fill against a torch reference that decodes the
same banks.

The CPU tests pin the numbers the quantizers write. The CUDA tests run the kernels the draft layer runs on those
banks: the decode GEMV (chain and verify rows, eager and replayed from a CUDA graph) and the prefill GEMM.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.layers.quantization import QuantKind
from freetoken.moe import expert_quant
from freetoken.moe.expert_quant import quantize_fp8_block, quantize_nvfp4, quantized_pieces

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn
# every finite non-negative e4m3 value, by code
E4M3_GRID = torch.arange(127, dtype=torch.uint8).view(FP8).float()
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = ((E2M1_GRID[:-1] + E2M1_GRID[1:]) / 2).tolist()


def _weights(experts: int, rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """bf16 weights with the spread a trained projection has: rows and columns of different scale, one outlier, one all-zero row and one all-zero block."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(experts, rows, cols, generator=g) * 0.02
    w = w * torch.exp(torch.randn(experts, rows, 1, generator=g)) * torch.exp(0.5 * torch.randn(experts, 1, cols, generator=g))
    w[0, 3, 5] *= 50
    w[0, 1] = 0
    w[-1, 2, 16:32] = 0
    return w.to(torch.bfloat16)


def _nearest_gap(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Distance from each ``|x|`` to the closest grid value."""
    return (x.abs().unsqueeze(-1) - grid).abs().amin(dim=-1)


def _nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 pairs -> one code per weight (low nibble = even k)."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)


def _nvfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    """e2m1 codes x e4m3 block scale x row global, the product the NVFP4 kernels compute."""
    codes = _nibbles(packed).long()
    values = E2M1_GRID.to(packed.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return values * (scale.float() * glob.unsqueeze(-1)).repeat_interleave(16, dim=-1)


def _fp8_dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 codes x the bf16 scale of their 128x128 block."""
    return codes.float() * scale.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)


def _first_difference(weight: torch.Tensor, a, b, names: tuple[str, str]) -> str:
    """For an assertion message, where two quantizations ``(codes, scale[, global])`` of ``weight`` part: the differing bytes of each
    tensor, and behind the first differing code its weight, block scale and weight / step."""
    a, b = ([t.cpu() for t in q] for q in (a, b))
    nvfp4 = len(a) == 3
    lines = []
    for what, x, y in zip(("codes", "scales", "globals"), a, b):
        x, y = x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
        where = (x != y).nonzero()
        if len(where):
            i = tuple(where[0].tolist())
            lines.append(f"{what}: {len(where)} of {x.numel()} bytes differ, first at {i}: {int(x[i])} {names[0]}, {int(y[i])} {names[1]}")
    codes = [_nibbles(q[0]) if nvfp4 else q[0].view(torch.uint8) for q in (a, b)]
    where = (codes[0] != codes[1]).nonzero()
    if len(where):
        i = tuple(where[0].tolist())
        w = float(weight.cpu()[i])
        lines.append(f"first differing code at {i}: weight {w!r} ({w.hex()})")
        for name, q, c in zip(names, (a, b), codes):
            if nvfp4:
                scale, glob = float(q[1][i[:-1] + (i[-1] // 16,)].float()), float(q[2][i[:-1]])
                step, detail = scale * glob, f"block scale {scale!r}, row global {glob!r} ({glob.hex()})"
            else:
                step = float(q[1][i[:-2] + (i[-2] // 128, i[-1] // 128)].float())
                detail = f"block scale {step!r}"
            lines.append(f"  {name}: code {int(c[i])}, {detail}, weight / step {w / step if step else float('nan')!r}")
    return "\n".join(lines) or "no byte differs"


def test_e2m1_rounding_takes_ties_to_the_even_code():
    """Under a step of 1 the grid's midpoints go to the even code, 6 is the top, a zero keeps its sign and a denormal is a zero. Without a step every code is 0."""
    w = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, 0.26, 4.99, -0.25, -5.0, -9.0, 0.0, -0.0, 1e-40, -1e-40, float("inf")])
    one = torch.ones((), dtype=torch.float64)
    assert expert_quant._e2m1_codes(w, one).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 7, 1, 6, 8, 14, 15, 0, 8, 0, 8, 7]
    assert expert_quant._e2m1_codes(torch.tensor([3.0, -3.0, 0.0]), 0 * one).tolist() == [0, 8, 0]


# 0.1 and 448 * 1.7e-5 fill fp32's 24 bits, so no fp32 weight sits on their midpoints; the last step is an fp32 denormal
@pytest.mark.parametrize("step", [1.0, 0.75, 448 * 2.0**-20, 0.1, 448 * 1.7e-5, 1.5 * 2.0**-130], ids=["1", "0.75", "448*2^-20", "0.1", "448*1.7e-5", "1.5*2^-130"])
def test_e2m1_codes_are_exact_at_and_beside_every_midpoint(step):
    """A weight on midpoint * step takes the even code of the two, and the fp32 values either side of that product take the side they are on.

    The comparison is exact, so no code depends on how a device rounds a quotient."""
    step = torch.tensor(step, dtype=torch.float32).double()
    zero, inf = torch.zeros((), dtype=torch.float32), torch.tensor(float("inf"))
    for lower, mid in enumerate(E2M1_MIDPOINTS):
        # exact in float64
        edge = mid * step
        w = edge.float()
        below = w if w < edge else torch.nextafter(w, zero)
        above = w if w > edge else torch.nextafter(w, inf)
        assert below < edge < above
        cases = [(below, lower), (above, lower + 1)] + ([(w, lower + lower % 2)] if w == edge else [])
        for value, code in cases:
            assert expert_quant._e2m1_codes(torch.stack([value, -value]), step).tolist() == [code, code | 8], (mid, float(value))


def test_nvfp4_scales_and_codes_are_the_nearest_on_their_grids():
    w = _weights(3, 64, 256)
    packed, scale, glob = quantize_nvfp4(w)
    assert (packed.shape, packed.dtype) == ((3, 64, 128), torch.uint8)
    assert (scale.shape, scale.dtype) == ((3, 64, 16), FP8)
    assert (glob.shape, glob.dtype) == ((3, 64), torch.float32)

    wf = w.float()
    amax = wf.reshape(3, 64, 16, 16).abs().amax(dim=-1)
    s = scale.float()
    # the row global puts the row's largest block on the top e4m3 value; an all-zero row keeps a finite global and zero scales
    zero_rows = amax.amax(dim=-1) == 0
    assert zero_rows.sum() == 1
    # the row max times the fp32 reciprocal of 2688, as CUDA divides by a Python scalar: the CPU's quotient differs on a fifth of the rows
    assert torch.equal(glob[~zero_rows], (amax.amax(dim=-1) * (1.0 / 2688.0))[~zero_rows])
    assert torch.equal(s.amax(dim=-1)[~zero_rows], torch.full((int((~zero_rows).sum()),), 448.0))
    assert torch.equal(glob[zero_rows], torch.ones(1)) and not s[zero_rows].any() and not packed[zero_rows].any()
    # each block scale is the e4m3 value nearest to amax / (6 * global)
    ideal = amax / (6.0 * glob.unsqueeze(-1))
    assert ((ideal - s).abs() <= _nearest_gap(ideal, E4M3_GRID) * (1 + 1e-6)).all()

    # each code is the e2m1 value nearest to the weight over the stored step, to the last bit (float64 holds grid * step), with the weight's sign
    step = (s.double() * glob.double().unsqueeze(-1)).repeat_interleave(16, dim=-1)
    live = step > 0
    x = wf.double()[live]
    codes = _nibbles(packed)[live].long()
    gap = (x.abs().unsqueeze(-1) - E2M1_GRID.double() * step[live].unsqueeze(-1)).abs()
    assert torch.equal(gap.gather(-1, (codes & 7).unsqueeze(-1)).squeeze(-1), gap.amin(dim=-1))
    assert torch.equal((codes & 8 > 0)[x != 0], (x < 0)[x != 0])
    # a block of zeros decodes to zeros
    deq = _nvfp4_dequant(packed, scale, glob)
    assert not deq[~live].any() and not wf[~live].any()
    assert (deq - wf).norm() / wf.norm() < 0.12


def test_zeros_and_denormals_quantize_to_zeros():
    """Zeros of either sign, and weights whose scale would fall under fp32's normal range, keep global 1 (nvfp4) or scale 1 (fp8) and decode to zeros.

    No scale or code is NaN, and inside a live row a zero block and a denormal leave their neighbours alone."""
    w = torch.zeros(1, 6, 32)
    w[0, 1] = -0.0
    # a global of max / 2688 rounds to zero here, next to a zero block
    w[0, 2, :16] = torch.linspace(-1e-42, 1e-42, 16)
    w[0, 3] = torch.linspace(-1.0, 1.0, 32) * 2.0**-120
    w[0, 4, :8] = torch.tensor([3.0, -3.0, 1e-40, -1e-40, 0.0, -0.0, 1.5, 0.5])
    w[0, 5] = torch.linspace(-1.0, 1.0, 32) * 1e-30
    packed, scale, glob = quantize_nvfp4(w)
    assert torch.isfinite(scale.float()).all()
    assert torch.equal(glob[0, :4], torch.ones(4)) and not scale[0, :4].float().any() and not (_nibbles(packed)[0, :4] & 7).any()
    assert not _nvfp4_dequant(packed, scale, glob)[0, :4].any()
    assert _nibbles(packed)[0, 4].tolist() == [7, 15, 0, 8, 0, 8, 5, 2] + [0] * 24
    assert scale[0, 4].float().tolist() == [448.0, 0.0]
    # a small row that still scales is quantized like any other; fp32 underflows its norm
    assert (_nvfp4_dequant(packed, scale, glob)[0, 5].double() - w[0, 5].double()).norm() / w[0, 5].double().norm() < 0.12

    w = torch.zeros(1, 128, 384)
    # a bf16 scale of max / 448 rounds to zero here, next to a zero weight
    w[0, :, :128] = torch.linspace(-1e-38, 1e-38, 128)
    w[0, 0, 0] = 0
    w[0, :, 128:256] = -0.0
    w[0, :, 256:] = torch.linspace(-1.0, 1.0, 128)
    w[0, 0, 256:259] = torch.tensor([1e-40, -1e-40, 0.0])
    codes, scale = quantize_fp8_block(w)
    assert torch.isfinite(codes.float()).all()
    assert scale[0, 0].tolist() == [1.0, 1.0, float(torch.tensor(1 / 448.0).to(torch.bfloat16))]
    assert not codes[0, :, :256].float().any() and not codes[0, 0, 256:259].float().any()
    assert (_fp8_dequant(codes, scale)[0, :, 256:] - w[0, :, 256:]).norm() / w[0, :, 256:].norm() < 0.035


def test_fp8_block_codes_are_the_nearest_under_the_stored_scale():
    from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

    w = _weights(2, 128, 256, seed=1)
    w[1, :, 128:] = 0
    codes, scale = quantize_fp8_block(w)
    assert (codes.shape, codes.dtype) == ((2, 128, 256), FP8)
    assert (scale.shape, scale.dtype) == ((2, 1, 2), torch.bfloat16)

    wf = w.float()
    amax = wf.reshape(2, 1, 128, 2, 128).abs().amax(dim=(-3, -1))
    # a block's largest magnitude lands on 448; an all-zero block keeps scale 1 and zero codes
    assert torch.equal(scale[amax > 0], (amax[amax > 0] / 448.0).to(torch.bfloat16))
    assert scale[1, 0, 1] == 1 and not codes[1, :, 128:].float().any()
    x = wf / scale.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    assert ((x.abs() - codes.float().abs()).abs() <= _nearest_gap(x, E4M3_GRID) * (1 + 1e-6)).all()
    assert torch.equal(codes.float().sign()[codes.float() != 0], x.sign()[codes.float() != 0])

    # the tree's own block-fp8 unpacker reads the pair back as the weight, to e4m3's 3 mantissa bits
    for e in range(2):
        back = dequant_block_fp8(codes[e], scale[e]).float()
        assert ((back - wf[e]).abs() <= wf[e].abs() * 0.071 + 3e-6 * wf[e].abs().max()).all()
    assert (_fp8_dequant(codes, scale) - wf).norm() / wf.norm() < 0.035


@pytest.mark.parametrize("kind", [QuantKind.FP8_BLOCK, QuantKind.NVFP4])
def test_quantized_pieces_are_the_same_for_any_chunking_and_piece_form(monkeypatch, kind):
    """One stacked piece split over chunks, and per-expert gate / up pieces, quantize to the bytes each expert gets alone: no value depends on a neighbour, and gate rows come before up rows."""
    E, I, H = 5, 128, 256
    gate_up, down = _weights(E, 2 * I, H, seed=2), _weights(E, H, I, seed=3)
    monkeypatch.setattr(expert_quant, "_CHUNK", 2)
    cpu = torch.device("cpu")
    stacked = list(quantized_pieces(iter([(0, 0, E, {"gate_up": gate_up, "down": down})]), kind, cpu))
    assert [p[:3] for p in stacked] == [(0, 0, 2), (0, 2, 4), (0, 4, 5)]
    single = list(quantized_pieces(
        ((0, e, e + 1, {"gate": gate_up[e : e + 1, :I], "up": gate_up[e : e + 1, I:], "down": down[e : e + 1]}) for e in range(E)), kind, cpu,
    ))
    again = list(quantized_pieces(iter([(0, 0, E, {"gate_up": gate_up, "down": down})]), kind, cpu))
    for role in stacked[0][3]:
        whole = torch.cat([p[3][role] for p in stacked])
        assert whole.shape[0] == E
        assert torch.equal(whole.view(torch.uint8), torch.cat([p[3][role] for p in single]).view(torch.uint8)), role
        assert torch.equal(whole.view(torch.uint8), torch.cat([p[3][role] for p in again]).view(torch.uint8)), role
    with pytest.raises(ValueError, match="already quantized"):
        list(quantized_pieces(iter([(0, 0, E, {"gate_up": gate_up, "gate_up_scale": down, "down": down})]), kind, cpu))


# --------------------------------------------------------------------------- the resident layer on CUDA

E, TOPK, H, I = 16, 4, 256, 128


def _resident_layer(quant: str | None, gate_up: torch.Tensor, down: torch.Tensor, device: torch.device):
    """A resident MoE layer over the bf16 experts, as the MTP draft builds it, with its banks quantized to ``quant`` at load."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import MoELayer
    from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks
    from freetoken.moe.expert_quant import expert_quant_method
    from freetoken.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    with torch_dtype(torch.bfloat16):
        layer = MoELayer(E, TOPK, H, I, layer_id=0, prefix="mtp.layers.0.mlp.experts")
    pieces = iter([(0, 0, E, {"gate_up": gate_up, "down": down})])
    if quant is not None:
        layer.quant_method = expert_quant_method(layer, quant)
        pieces = quantized_pieces(pieces, layer.quant_method.kind, device)
    attach_resident_banks([layer], build_expert_banks(layer.quant_method, 1, pieces, device=device, resident=True))
    return layer


def _apply(layer, x, weights, ids, *, is_prefill: bool) -> torch.Tensor:
    # clones: the kernels may write into the input and the ids
    return layer.quant_method.apply(x.clone(), weights, ids.clone(), layer.quant_method.resident_view(layer), layer=layer, is_prefill=is_prefill)


def _dequantized(layer) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 gate_up and down of the layer's quantized banks."""
    b = layer._expert_banks
    if layer.quant_method.kind is QuantKind.NVFP4:
        return _nvfp4_dequant(b["gate_up"], b["gate_up_scale"], b["gate_up_global"]), _nvfp4_dequant(b["down"], b["down_scale"], b["down_global"])
    # the scale banks carry padding past the last block column
    return _fp8_dequant(b["gate_up"], b["gate_up_scale"][..., : H // 128]), _fp8_dequant(b["down"], b["down_scale"][..., : I // 128])


def _act_fp8(x: torch.Tensor) -> torch.Tensor:
    """The W8A8 prefill's activation rounding: e4m3 per row and 128-wide group, the group's largest magnitude on 448."""
    g = x.float().reshape(x.shape[0], -1, 128)
    s = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0
    return ((g / s).clamp(-448.0, 448.0).to(FP8).float() * s).reshape(x.shape)


def _reference(gate_up: torch.Tensor, down: torch.Tensor, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, *, act_fp8: bool) -> torch.Tensor:
    """Dense gated-silu MoE over fp32 weights, rounding to bf16 where the kernels store bf16: after each GEMM, the activation and the sum."""
    bf16 = lambda t: t.to(torch.bfloat16).float()
    a = _act_fp8(x) if act_fp8 else x.float()
    h = bf16(torch.einsum("mknh,mh->mkn", gate_up[ids.long()], a))
    act = bf16(F.silu(h[..., :I]) * h[..., I:])
    if act_fp8:
        act = _act_fp8(act.reshape(-1, I)).reshape(act.shape)
    routed = bf16(torch.einsum("mkhn,mkn->mkh", down[ids.long()], act) * weights.unsqueeze(-1))
    return bf16(routed.sum(dim=1))


def _routing(m: int, device: torch.device, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, H, generator=g) / 4).to(device=device, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)]).to(device=device, dtype=torch.int32)
    weights = torch.rand(m, TOPK, generator=g).to(device)
    return x, weights / weights.sum(dim=-1, keepdim=True), ids


def _experts(device: torch.device):
    return _weights(E, 2 * I, H, seed=4).to(device) * 4, _weights(E, H, I, seed=5).to(device) * 4


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).norm() / ref.float().norm())


@cuda
@pytest.mark.parametrize("quant", ["fp8", "nvfp4"])
@pytest.mark.parametrize("rows, is_prefill", [(1, False), (16, False), (64, True)], ids=["chain", "verify", "prefill"])
def test_quantized_resident_layer_matches_its_dequantized_banks(quant, rows, is_prefill):
    """The draft layer over quantized banks computes what a dense MoE over the same banks, decoded in torch, computes; against the bf16 experts it differs by the format's rounding."""
    device = torch.device("cuda")
    gate_up, down = _experts(device)
    layer = _resident_layer(quant, gate_up, down, device)
    x, weights, ids = _routing(rows, device, seed=rows)
    out = _apply(layer, x, weights, ids, is_prefill=is_prefill)
    assert torch.isfinite(out.float()).all()

    ref = _reference(*_dequantized(layer), x, weights, ids, act_fp8=quant == "fp8" and is_prefill)
    assert _rel(out, ref) < 1e-2

    stored = _apply(_resident_layer(None, gate_up, down, device), x, weights, ids, is_prefill=is_prefill)
    drift = _rel(out, stored)
    print(f"\ndraft quant {quant}, {rows} rows, {'prefill' if is_prefill else 'decode'} kernel: relative error against the bf16 experts {drift:.4f}")
    assert drift < (0.2 if quant == "fp8" else 0.5)


@cuda
@pytest.mark.parametrize("quant", ["fp8", "nvfp4"])
@pytest.mark.parametrize("rows", [1, 4])
def test_quantized_resident_layer_replays_from_a_cuda_graph(quant, rows):
    """The draft's chain and verify steps run from CUDA graphs: a replay over new inputs in the captured buffers gives the eager result."""
    device = torch.device("cuda")
    layer = _resident_layer(quant, *_experts(device), device)
    x, weights, ids = _routing(rows, device, seed=7)
    out = torch.empty_like(x)

    def run():
        out.copy_(_apply(layer, x, weights, ids, is_prefill=False))

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (8, 9):
        nx, nw, nids = _routing(rows, device, seed=seed)
        x.copy_(nx)
        weights.copy_(nw)
        ids.copy_(nids)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        # not bitwise: the decode GEMV picks its activation load width from the buffer's alignment
        assert _rel(replayed, _apply(layer, nx, nw, nids, is_prefill=False)) < 1e-3
        assert _rel(replayed, _reference(*_dequantized(layer), nx, nw, nids, act_fp8=False)) < 1e-2


@cuda
def test_nvfp4_banks_decode_the_same_under_the_dequant_kernel():
    """The tree's NVFP4 unpacker reads the quantized banks as the torch decode of this file does."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    device = torch.device("cuda")
    gate_up, down = _experts(device)
    b = _resident_layer("nvfp4", gate_up, down, device)._expert_banks
    slots = torch.arange(E, dtype=torch.int32, device=device)
    for proj, stored in (("gate_up", gate_up), ("down", down)):
        got = dequant_nvfp4(b[proj], b[f"{proj}_scale"], b[f"{proj}_global"], slots, dtype=torch.float32)
        want = _nvfp4_dequant(b[proj], b[f"{proj}_scale"], b[f"{proj}_global"])
        torch.testing.assert_close(got, want, rtol=1e-6, atol=0)
        assert _rel(got, stored) < 0.12


@cuda
def test_nvfp4_quantizer_matches_the_gpu_e2m1_conversion():
    """Row by row under the row's global, the torch quantizer writes the scales and codes quantize_nvfp4 (the GPU's own e2m1 / e4m3 conversions)
    writes, except on a rounding midpoint, where the two may take either neighbour.

    The kernel divides with Triton's approximate fp32 division (div.full, within 2 ulps) and the torch quantizer compares exactly. A row
    global of max / 2688 puts a weight of every few rows on an e2m1 midpoint to the last bit, where the kernel's quotient falls on either side."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("the e2m1 conversion instruction needs sm_100 or newer")
    from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4 as quantize_rows

    device = torch.device("cuda")
    w = _weights(1, 64, 256, seed=6).to(device)[0].float()
    packed, scale, glob = quantize_nvfp4(w)
    q, s = (torch.cat(t) for t in zip(*(quantize_rows(w[row : row + 1].contiguous(), glob[row : row + 1].contiguous()) for row in range(w.shape[0]))))
    report = _first_difference(w, (q, s, glob), (packed, scale, glob), ("kernel", "torch"))
    # the kernel's 2-ulp quotient, then one rounding of its product
    near = 4 * torch.finfo(torch.float32).eps

    # a scale differs only between the two e4m3 values around a midpoint the ideal scale sits on
    kernel_scale, torch_scale = s.view(torch.uint8).int(), scale.view(torch.uint8).int()
    scale_off = kernel_scale != torch_scale
    ideal = w.reshape(64, 16, 16).abs().amax(dim=-1).double() / (6.0 * glob).double().unsqueeze(-1)
    assert ((kernel_scale - torch_scale).abs() == 1)[scale_off].all(), report
    assert (((s.float().double() + scale.float().double()) / 2 - ideal).abs() <= near * ideal)[scale_off].all(), report

    # under the same scale, a code differs only between the two e2m1 values around a midpoint the weight sits on
    kernel_code, torch_code = _nibbles(q).long(), _nibbles(packed).long()
    code_off = (kernel_code != torch_code) & ~scale_off.repeat_interleave(16, dim=-1)
    step = (scale.float().double() * glob.double().unsqueeze(-1)).repeat_interleave(16, dim=-1)
    grid = E2M1_GRID.double().to(device)
    edge = (grid[kernel_code & 7] + grid[torch_code & 7]) / 2 * step
    assert (((kernel_code ^ torch_code) < 8) & ((kernel_code - torch_code).abs() == 1))[code_off].all(), report
    assert ((step > 0) & ((edge - w.double().abs()).abs() <= near * edge))[code_off].all(), report


@cuda
@pytest.mark.parametrize("kind", [QuantKind.FP8_BLOCK, QuantKind.NVFP4])
def test_quantizing_on_the_gpu_writes_the_cpu_bytes(kind):
    """The load quantizes on the GPU; the banks do not depend on the device."""
    gate_up, down = _weights(E, 2 * I, H, seed=4), _weights(E, H, I, seed=5)
    on = {}
    for device in (torch.device("cpu"), torch.device("cuda")):
        pieces = list(quantized_pieces(iter([(0, 0, E, {"gate_up": gate_up, "down": down})]), kind, device))
        on[device.type] = {role: torch.cat([p[3][role] for p in pieces]).cpu() for role in pieces[0][3]}
    for proj, weight in (("gate_up", gate_up), ("down", down)):
        cpu, gpu = ([tensor for role, tensor in on[name].items() if role.startswith(proj)] for name in ("cpu", "cuda"))
        assert len(cpu) == len(gpu) == (2 if kind is QuantKind.FP8_BLOCK else 3)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(cpu, gpu)), _first_difference(weight, cpu, gpu, ("cpu", "cuda"))
