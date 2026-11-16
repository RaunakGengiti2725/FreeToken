"""GraphRunner capture: the forwards run on the dummy request's page, and the MoE offload cache stays warm from the eager run into the capture. A graph with a tail is captured the same way, its tail outside the forward's batch. The startup warmup prefill runs on the same request's row and state slot."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import freetoken.core as core
from freetoken.core import Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import GraphRunner


def test_capture_runs_on_the_dummy_page_with_a_warm_offload_cache(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    # row 0 is a live request holding slot 0, row 1 is the dummy request on the dummy page
    ctx.page_table = torch.tensor([[0], [9]], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    events = []

    @contextmanager
    def cuda_graph(graph, pool=None, stream=None):
        events.append("capture")
        yield
        events.append("captured")

    def forward():
        batch = get_global_ctx().batch
        events.append(batch.out_loc.tolist())
        return torch.zeros(batch.size, 3)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("torch.cuda.CUDAGraph", lambda: SimpleNamespace(pool=lambda: "pool"))
    monkeypatch.setattr("torch.cuda.graph", cuda_graph)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(f"torch.cuda.{name}", noop)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 0)
    monkeypatch.setattr("freetoken.engine.graph._upload_graph", noop)

    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    runner = GraphRunner(stream=None, device=torch.device("cpu"), forward=forward,
                         attn_backend=SimpleNamespace(init_capture_graph=noop, prepare_for_capture=noop),
                         cuda_graph_bs=[2], cuda_graph_max_bs=None, free_memory=0, max_seq_len=1, dummy_req=dummy,
                         moe_offload_cache=SimpleNamespace(reset=lambda: events.append("reset")))

    assert events == ["reset", [9, 9], "capture", [9, 9], "captured", "reset", "reset"]

    events.clear()
    graphs = runner.capture_with_tail(lambda bs, out: events.append(("tail", bs, get_global_ctx()._batch is None)))
    assert list(graphs) == [2]
    assert events == [[9, 9], ("tail", 2, True), "capture", [9, 9], ("tail", 2, True), "captured", "reset"]


def test_warmup_prefill_runs_for_resident_experts_on_the_dummy_request_and_restores_its_row(monkeypatch):
    from freetoken.engine.engine import Engine, _prefill_warmup_args

    keeps = SimpleNamespace(warm_prefill_keeps=lambda: True)
    streams = SimpleNamespace(warm_prefill_keeps=lambda: False)
    one, two = dict(autotune=False, keeps=False), dict(autotune=True, keeps=False)
    # experts held as weights: warmed on every backend the warmup can address, with the forwards that backend had
    assert [_prefill_warmup_args(b, None, True) for b in ("fi", "fa", "trtllm", "triton", "dsa")] == [one, one, one, two, None]
    # a dense model and an expert cache that streams start as before
    assert [_prefill_warmup_args(b, None, False) for b in ("fi", "triton")] == [None, two]
    assert [_prefill_warmup_args(b, streams, False) for b in ("fi", "triton")] == [None, two]
    assert [_prefill_warmup_args(b, keeps, False) for b in ("fi", "dsa")] == [dict(autotune=False, keeps=True), None]

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    # row 0 is a live request, row 1 is the dummy request on the dummy page
    ctx.page_table = torch.tensor([[5] * 96, [200] * 96], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    seen = []

    def forward():
        batch = get_global_ctx().batch
        req = batch.reqs[0]
        seen.append((batch.phase, req.table_idx, req.linear_slot_idx, req.cached_len, batch.out_loc.tolist()))

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: SimpleNamespace(record=noop, elapsed_time=lambda end: 0.0))
    monkeypatch.setattr("torch.cuda.synchronize", noop)
    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    dummy.linear_slot_idx = 7
    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine.max_seq_len, engine.page_table, engine.dummy_req, engine.ctx = 96, ctx.page_table, dummy, ctx
    engine.device, engine.stream, engine.moe_offload_cache = torch.device("cpu"), None, None
    engine.config = SimpleNamespace(model_config=SimpleNamespace(model_is_mrope=False))
    engine.attn_backend = SimpleNamespace(prepare_metadata=noop)
    engine.model = SimpleNamespace(forward=forward)

    engine._warmup_prefill(**_prefill_warmup_args("fi", None, True))

    assert seen == [("prefill", 1, 7, 0, list(range(80)))]
    assert ctx.page_table.tolist() == [[5] * 96, [200] * 96]
