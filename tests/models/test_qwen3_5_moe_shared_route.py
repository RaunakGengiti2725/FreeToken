"""The shared expert of Qwen3.5 / 3.6 MoE as one more route of the routed experts' decode kernels.

Which steps take it (decode steps of 1 to 6 rows whose router runs in the top-k launch, on resident experts under the
Triton NVFP4 kernels, one rank), what it is built from, and on CUDA the MoE block's output against the shared expert's
own kernels at the 35B-A3B shapes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.layers.moe as moe_layers
import freetoken.models.qwen3_5_moe.moe as qmoe
from freetoken.layers.moe import MoELayer, OffloadMoELayer
from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
from freetoken.layers.quantization.moe.base import ExpertView
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

E, TOPK, H, MI = 256, 8, 2048, 512
ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
FP8 = torch.float8_e4m3fn


def _ctx(prefill: bool):
    return SimpleNamespace(batch=SimpleNamespace(is_prefill=prefill))


def _row_major(n, k, gen, device="cpu", glob=1.0):
    """One NVFP4 projection [n, k] in the bank's row-major layout: codes, block scales, per-row globals."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    scale = (torch.rand(n, k // 16, generator=gen) * 1.5 + 0.25).to(FP8)
    row_global = (torch.rand(n, generator=gen) + 0.5) * glob
    return packed.to(device), scale.to(device), row_global.to(device)


def _resident_linear(packed, scale, row_global, out_features, in_features):
    """The projection as TritonNvfp4LinearKernel keeps it: K-major words and scales (nvfp4_transpose_resident)."""
    return SimpleNamespace(weight=packed.view(torch.int32).t().contiguous(), weight_scale=scale.t().contiguous(),
                           weight_global=row_global, out_features=out_features, in_features=in_features, _tp_size=1)


def _experts(banks, *, cls=MoELayer, kernel=None, tp_size=1):
    layer = object.__new__(cls)
    kernel = kernel or TritonNvfp4MoEKernel()
    layer.quant_method = SimpleNamespace(
        kernel=kernel,
        resident_view=lambda lyr: ExpertView(banks, n=E),
        apply=lambda x, w, i, view, *, layer, is_prefill: kernel.apply(layer, x, w, i, view, is_prefill=is_prefill),
    )
    layer._expert_banks = banks
    layer.num_experts, layer.top_k, layer.renormalize = E, TOPK, True
    layer.tp_size, layer.activation, layer.apply_router_weight_on_input = tp_size, "silu", False
    layer.alpha, layer.limit = 1.0, None
    return layer


def _moe(device="cpu", seed=0, glob=1.0):
    """A Qwen3_5MoE with resident NVFP4 experts and shared expert, a bf16 router and gate, built without a model."""
    gen = torch.Generator().manual_seed(seed)
    bank_gate_up = [_row_major(2 * MI, H, gen, device, glob) for _ in range(E)] if device != "cpu" else None
    if bank_gate_up is None:   # the CPU checks look only at shapes
        banks = {"gate_up": torch.zeros(E, 2 * MI, H // 2, dtype=torch.uint8), "gate_up_scale": torch.zeros(E, 2 * MI, H // 16, dtype=FP8),
                 "gate_up_global": torch.zeros(E, 2 * MI), "down": torch.zeros(E, H, MI // 2, dtype=torch.uint8),
                 "down_scale": torch.zeros(E, H, MI // 16, dtype=FP8), "down_global": torch.zeros(E, H)}
    else:
        bank_down = [_row_major(H, MI, gen, device, glob) for _ in range(E)]
        banks = {role: torch.stack([p[i] for p in parts]) for parts, names in ((bank_gate_up, ROLES[:3]), (bank_down, ROLES[3:]))
                 for i, role in enumerate(names)}
    moe = object.__new__(qmoe.Qwen3_5MoE)
    moe.experts = _experts(banks)
    router = (torch.randn(E, H, generator=gen) * 0.02).to(device, torch.bfloat16)
    gate_w = (torch.randn(1, H, generator=gen) * 0.05).to(device, torch.bfloat16)
    moe.gate = SimpleNamespace(weight=router, bias=None, quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
    moe.shared_expert_gate = SimpleNamespace(weight=gate_w, bias=None, quant_method=SimpleNamespace(kernel=TorchLinearKernel()),
                                             forward=lambda x: torch.nn.functional.linear(x, gate_w))
    moe.shared_expert = SimpleNamespace(gate_up_proj=_resident_linear(*_row_major(2 * MI, H, gen, device, glob), 2 * MI, H),
                                        down_proj=_resident_linear(*_row_major(H, MI, gen, device, glob), H, MI))
    moe._fuse_shared = True
    moe._router_weight = False
    moe._shared_route = False
    return moe


@pytest.fixture
def decode(monkeypatch):
    state = {"prefill": False}
    monkeypatch.setattr(qmoe, "get_global_ctx", lambda: _ctx(state["prefill"]))
    monkeypatch.setattr(moe_layers, "get_global_ctx", lambda: _ctx(state["prefill"]))
    return state


# --------------------------------------------------------------------------- which steps take it (no GPU)


@pytest.mark.parametrize("rows,takes", [(0, False), (1, True), (2, True), (6, True), (7, False), (8, False), (16, False)])
def test_shared_route_takes_decode_steps_of_one_to_six_rows(decode, rows, takes):
    moe = _moe()
    moe._router_weight = moe.gate.weight   # the fold's weight, decided
    moe._shared_route = {"gate_up": torch.empty(0)}
    x = torch.zeros(rows, H, dtype=torch.bfloat16)
    assert (moe._shared_route_for(x) is moe._shared_route) is takes


@pytest.mark.parametrize("rows", [7, 8])
def test_shared_route_follows_the_router_folds_row_cap(decode, monkeypatch, rows):
    """The gate rides the fold's launch, so a step the fold does not take keeps today's kernels even where the
    route's own row limit would let it through."""
    moe = _moe()
    moe._router_weight = moe.gate.weight
    moe._shared_route = {"gate_up": torch.empty(0)}
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE_MAX_ROWS", 8)
    assert moe._shared_route_for(torch.zeros(rows, H, dtype=torch.bfloat16)) is None
    assert moe._shared_route_for(torch.zeros(6, H, dtype=torch.bfloat16)) is moe._shared_route


def test_shared_route_leaves_prefill_other_dtypes_the_knob_and_the_unfolded_router_alone(decode, monkeypatch):
    moe = _moe()
    moe._router_weight = moe.gate.weight
    moe._shared_route = route = {"gate_up": torch.empty(0)}
    x = torch.zeros(6, H, dtype=torch.bfloat16)
    assert moe._shared_route_for(x) is route
    decode["prefill"] = True
    assert moe._shared_route_for(x) is None
    decode["prefill"] = False
    assert moe._shared_route_for(x.half()) is None
    assert moe._shared_route_for(torch.zeros(6, 2 * H, dtype=torch.bfloat16)[:, ::2]) is None
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", False)
    assert moe._shared_route_for(x) is None
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)
    moe._router_weight = None   # the router's linear stays its own launch: no row for the gate to join
    assert moe._shared_route_for(x) is None
    moe._router_weight, moe._shared_route = moe.gate.weight, None   # built and refused once: never again
    assert moe._shared_route_for(x) is None


def test_shared_route_leaves_the_grouped_kernels_row_counts_alone(decode, monkeypatch):
    from freetoken.kernel import nvfp4_moe

    moe = _moe()
    moe._router_weight = moe.gate.weight
    moe._shared_route = route = {"gate_up": torch.empty(0)}
    monkeypatch.setattr(nvfp4_moe, "_DECODE_GROUPED_ROWS", 6)
    assert moe._shared_route_for(torch.zeros(2, H, dtype=torch.bfloat16)) is route
    assert moe._shared_route_for(torch.zeros(6, H, dtype=torch.bfloat16)) is None


def test_shared_route_is_built_for_resident_triton_nvfp4_experts_only(monkeypatch):
    moe = _moe()
    route = moe._build_shared_route()
    assert set(route) == set(ROLES)
    for role in ROLES:
        assert route[role].shape == moe.experts._expert_banks[role].shape[1:]
    loaded = _moe()
    loaded.prepare_decode_weights()   # the engine's load makes it, before the KV pool is sized
    assert set(loaded._shared_route) == set(ROLES)
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", False)
    off = _moe()
    off.prepare_decode_weights()   # the knob off: no copy
    assert off._shared_route is False
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)

    def refused(change):
        m = _moe()
        change(m)
        return m._build_shared_route() is None

    assert refused(lambda m: setattr(m, "_fuse_shared", False))   # the shared expert's linears are not the triton NVFP4 kernel
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, cls=OffloadMoELayer)))
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, kernel=SimpleNamespace(takes_shared_route=False))))
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, tp_size=2)))
    assert refused(lambda m: setattr(m.experts, "activation", "gelu"))
    assert refused(lambda m: setattr(m.shared_expert_gate, "bias", torch.zeros(1, dtype=torch.bfloat16)))
    assert refused(lambda m: setattr(m.experts, "_expert_banks", None))
    # a bank of another width: the shared expert is not one of its slots
    assert refused(lambda m: m.experts._expert_banks.update(down=torch.zeros(E, H, MI // 4, dtype=torch.uint8)))


def test_shared_route_says_once_a_row_count_that_a_served_step_took_it(decode, monkeypatch):
    """One line a row count a process, from the blocks the engine prepared (the served model's, not a draft's)."""
    from freetoken.kernel.triton import moe_router

    said, seen = [], []
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE_ROWS_SAID", set())
    monkeypatch.setattr(qmoe.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)
    monkeypatch.setattr(moe_router, "router_gemv_topk_softmax", lambda x, w, k, renorm, shared_gate=None: (None, None))

    def block(prepared):
        moe = _moe()
        moe._router_weight = moe.gate.weight
        if prepared:
            moe._decode_prepared = True
        moe.experts.routed_forward = lambda x, w, i, shared=None: seen.append(x.shape[0]) or x
        return moe

    served, draft = block(True), block(False)
    route = {"gate_up": torch.empty(0)}
    for moe, rows in ((draft, 6), (served, 6), (served, 6), (draft, 1), (served, 1), (served, 6)):
        moe._forward_shared_route(torch.zeros(rows, H, dtype=torch.bfloat16), route)
    assert seen == [6, 6, 6, 1, 1, 6]
    assert said == ["shared expert as a ninth route on decode: taken at 6 rows",
                    "shared expert as a ninth route on decode: taken at 1 row"]


def test_shared_expert_bank_slot_undoes_the_resident_transpose():
    gen = torch.Generator().manual_seed(3)
    gate_up, down = _row_major(2 * MI, H, gen), _row_major(H, MI, gen)
    route = qmoe.shared_expert_bank_slot(_resident_linear(*gate_up, 2 * MI, H), _resident_linear(*down, H, MI))
    for role, want in zip(ROLES, (*gate_up, *down)):
        assert route[role].dtype == want.dtype and route[role].is_contiguous()
        assert torch.equal(route[role].view(torch.uint8), want.contiguous().view(torch.uint8)), role


# --------------------------------------------------------------------------- the MoE block on CUDA


def _bound_check(on, off, routed, shared):
    """The pass bound of the ninth route against the shared expert's own kernels: the two differ only in rounding (the
    shared projections summed in the marlin kernel's order, the nine routes summed once in fp32), so every element is
    within 2^-6 of the larger of the routed and shared magnitudes of the step. Returns (max |on - off|, the bound)."""
    diff = (on.float() - off.float()).abs().max().item()
    bound = 2.0**-6 * (routed.float().abs().max().item() + shared.float().abs().max().item())
    return diff, bound


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("rows", [1, 2, 6])
def test_shared_route_block_output_keeps_the_shared_experts_own_kernels_values(decode, monkeypatch, rows):
    """The MoE block of a decode step through the ninth route against the same block through the router fold and the
    shared expert's fused kernels (the path it replaces): the eight routed ids and weights unchanged, the output inside
    the rounding bound, finite, and not the routed experts alone."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax
    from freetoken.kernel.triton.nvfp4_shared_expert import shared_expert_down_add, shared_expert_fuses_rows, shared_expert_gate_up

    device = torch.device("cuda")
    if not shared_expert_fuses_rows(rows, H, MI, device):
        pytest.skip("the shared expert's own kernels do not fuse this row count here")
    moe = _moe(device, seed=rows, glob=1.0 / 64)
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    x = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)

    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", False)
    off = moe.forward(x.clone())
    assert moe._shared_route is False   # the knob kept the route from being built
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)
    on = moe.forward(x.clone())
    assert isinstance(moe._shared_route, dict), "the step did not take the ninth route"

    w8, i8 = router_gemv_topk_softmax(x, moe.gate.weight, TOPK, True)
    w9, i9 = router_gemv_topk_softmax(x, moe.gate.weight, TOPK, True, shared_gate=moe.shared_expert_gate.weight)
    assert torch.equal(i9[:, :TOPK], i8) and torch.equal(w9[:, :TOPK].contiguous().view(torch.int32), w8.view(torch.int32))

    routed = moe.experts.routed_forward(x.clone(), w8, i8.clone())
    gu, dn = moe.shared_expert.gate_up_proj, moe.shared_expert.down_proj
    act = shared_expert_gate_up(x, gu.weight, gu.weight_scale, gu.weight_global)
    shared = shared_expert_down_add(act, dn.weight, dn.weight_scale, dn.weight_global,
                                    moe.shared_expert_gate.forward(x).view(-1), torch.zeros_like(routed))
    assert on.isfinite().all() and off.isfinite().all()
    diff, bound = _bound_check(on, off, routed, shared)
    assert diff <= bound, (diff, bound)
    # the shared expert is in the output: the routed experts alone are far from it
    assert (on.float() - routed.float()).abs().max().item() > 4 * bound


@cuda
@pytest.mark.parametrize("rows", [1, 6])
def test_shared_route_block_replays_in_a_cuda_graph(decode, rows):
    """Built at the eager step before the capture, as the graph runner runs one; every replay routes its own input."""
    moe = _moe(torch.device("cuda"), seed=40 + rows, glob=1.0 / 64)
    gen = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        moe.forward(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert isinstance(moe._shared_route, dict)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = moe.forward(x)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=gen, device="cuda").to(torch.bfloat16))
        graph.replay()
        ref = moe.forward(x.clone())
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
