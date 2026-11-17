"""The scheduler side of a speculative verify step: what the drain commits, what it gives back, and in which order.

The real unbound Scheduler methods run against CPU-built hybrid managers; the engine is a recorder, so the target state commit is checked by its arguments and its order against the finish frees.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

K = 3
PROMPT = 12


def _setup():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=16, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(128, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=page_table)
    dm = DecodeManager(page_size=1, token_headroom=K)
    pm = PrefillManager(cm, tm, dm, token_headroom=K)

    events = []
    engine = SimpleNamespace(
        commit_verify=lambda reqs, counts, out: events.append(("commit", [r.uid for r in reqs], list(counts))),
    )

    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm, prefill_manager=pm, spec=None, engine=engine,
        config=SimpleNamespace(decode_rows_per_req=K + 1), device=torch.device("cpu"),
        token_pool=torch.zeros(4, 64, dtype=torch.int32), eos_token_ids=set(), toolcall_anchor_id=None,
        _match_stop_str=lambda _req: None, _reasoning={},
    )

    def free(req):
        events.append(("free", req.uid))
        Scheduler._free_req_resources(stub, req)

    stub._free_req_resources = free
    stub._commit_tokens = lambda req, tokens: Scheduler._commit_tokens(stub, req, tokens)
    return pool, cm, tm, dm, stub, events


def _decoding_req(pool, cm, tm, uid: int, max_tokens: int, timings_per_token: bool = False) -> Req:
    """A request whose prefill has drained: KV for the prompt, its first output token pending at PROMPT."""
    prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32) + 100 * uid
    match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
    req = Req(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=max_tokens, uid=uid,
              sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True,
                                             timings_per_token=timings_per_token),
              cache_handle=match.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(match.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    return req


def test_drain_commits_the_accepted_tokens_and_returns_the_rejected_slots():
    pool, cm, tm, dm, stub, events = _setup()
    # one accepted draft; the output limit cut inside the step; every draft accepted
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 3), _decoding_req(pool, cm, tm, 3, 16)]
    dm.filter_reqs(reqs)
    # the verify launch: k provisional slots past each pending token
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    free_before = cm.available_size
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([1, 3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()
    tables = [r.table_idx for r in reqs]

    committed = Scheduler._drain_verify(stub, batch, output, reply, finished)

    assert committed == 2 + 2 + 4
    assert [m.next_tokens for m in reply] == [[11, 12], [21, 22], [31, 32, 33, 34]]
    assert [m.finished for m in reply] == [False, True, False]
    # the target state is committed once, before the finished request gives its GDN slot back
    assert events == [("commit", [1, 2, 3], [2, 2, 4]), ("free", 2)]
    assert finished == {reqs[1]}
    a, c = reqs[0], reqs[2]
    assert (a.cached_len, a.device_len) == (PROMPT + 2, PROMPT + 3)
    assert (c.cached_len, c.device_len) == (PROMPT + 4, PROMPT + 5)
    # the last committed token is pending at cached_len for the next step
    assert int(stub.token_pool[tables[0], PROMPT + 2]) == 12
    assert int(stub.token_pool[tables[2], PROMPT + 4]) == 34
    # slots from cached_len on go back (allocate_paged allocates from cached_len again next step); b gave everything back
    assert cm.available_size - free_before == (K + 1 - 2) + (K + 1 - 4) + (PROMPT + K + 1)
    assert dm.running_reqs == {a, c}
    # no slot was freed twice: with every request gone the pool adds up
    for req in (a, c):
        stub._free_req_resources(req)
    cm.check_integrity()


def test_drain_counts_drafts_and_attaches_timings():
    """The verify drain feeds ReqStats: k drafts per step, accepted = min(n, emitted)
    (a stop landing on a draft makes every committed token an accepted draft), and the
    terminal reply carries the finished request's spans."""
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 3)]
    dm.filter_reqs(reqs)
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    # spans the prefill drain would have stamped: enqueued -> first chunk -> first token
    for req in reqs:
        req.stats.enqueued_at = 1.0
        req.stats.first_chunk_at = 2.0
        req.stats.first_token_at = 3.0
        req.stats.prompt_n = PROMPT
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()

    Scheduler._drain_verify(stub, batch, output, reply, finished)

    # req 1: every draft accepted, n = emitted + 1 -> 3 accepted drafts
    assert reqs[0].stats.draft_n == K and reqs[0].stats.draft_n_accepted == 3
    # req 2: all 3 drafts accepted but the length budget cuts the step to n = 2, and
    # both committed tokens ARE accepted drafts (the target's own never landed): 2
    assert reqs[1].stats.draft_n == K and reqs[1].stats.draft_n_accepted == 2
    # only the terminal reply carries timings
    assert reply[0].timings is None
    t = reply[1].timings
    assert t["queue_ms"] == 1000.0  # enqueued -> first chunk
    assert t["prompt_ms"] == 1000.0  # first chunk -> first token
    assert t["prompt_n"] == PROMPT
    assert t["predicted_n"] == 3  # first token + the two committed here
    assert t["predicted_ms"] > 0.0  # first -> last committed token
    assert t["draft_n"] == K and t["draft_n_accepted"] == 2


def test_drain_stamps_intermediate_timings_when_requested():
    """A request whose sampling_params ask for per-token timings get the ReqStats
    snapshot on every reply, not only the terminal one; the non-asking table above
    keeps the old None-on-intermediate rule."""
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16, timings_per_token=True),
            _decoding_req(pool, cm, tm, 2, 3, timings_per_token=True)]
    dm.filter_reqs(reqs)
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    for req in reqs:
        req.stats.enqueued_at = 1.0
        req.stats.first_chunk_at = 2.0
        req.stats.first_token_at = 3.0
        req.stats.prompt_n = PROMPT
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()

    Scheduler._drain_verify(stub, batch, output, reply, finished)

    # req 1's step is not terminal, yet its reply already carries the growing snapshot
    assert not reply[0].finished and reply[0].timings is not None
    assert reply[0].timings["predicted_n"] == 5  # first token + the four committed here
    assert reply[0].timings["prompt_n"] == PROMPT
    assert reply[0].timings["draft_n"] == K and reply[0].timings["draft_n_accepted"] == 3
    # the terminal reply of the same drain keeps the unchanged terminal snapshot
    assert reply[1].finished and reply[1].timings["predicted_n"] == 3


def test_a_step_reads_the_table_rows_of_the_batch_in_front_of_it():
    """The scheduler repeats the device copy of a decode batch's row mapping while its requests repeat: a reorder, a request leaving, a table row handed to another request, a different row count and a prefill in between must each reach the rows the step gathers its tokens and KV slots with."""
    stub = SimpleNamespace(device=torch.device("cpu"), _sent_rows=None)

    def rows(reqs, phase="verify"):
        batch = Batch(reqs=reqs, phase=phase, rows_per_req=None if phase == "prefill" else reqs[0].extend_len)
        batch.padded_reqs = reqs
        batch.positions = torch.zeros(sum(r.extend_len for r in reqs), dtype=torch.int32)
        return Scheduler._input_rows(stub, batch).tolist()

    a, b = SimpleNamespace(table_idx=1, extend_len=K + 1), SimpleNamespace(table_idx=2, extend_len=K + 1)
    for reqs in ([a, b], [a, b], [b, a], [a], [b]):
        assert rows(reqs) == [r.table_idx for r in reqs for _ in range(K + 1)]
    b.table_idx = 3
    assert rows([b]) == [3] * (K + 1)
    b.extend_len = 1
    assert rows([b], "decode") == [3]
    assert rows([SimpleNamespace(table_idx=0, extend_len=5)], "prefill") == [0] * 5
    assert rows([b], "decode") == [3]


def test_drain_commits_a_forced_token_as_the_end_of_the_accepted_run():
    """A request at its reasoning budget inside a verify step: the host plans the close tag for the row where the budget ends, the verify cuts the accepted drafts there, and the drain commits, rolls back and leaves pending exactly that prefix."""
    from freetoken.scheduler.reasoning import ReasoningTracker, ThinkTokens

    close = 99
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 16)]
    dm.filter_reqs(reqs)
    tags = ThinkTokens(open=(98,), close=(close,), exits=frozenset(), warning=())
    # request 1 starts inside thinking with its first output token (7) observed: one more fits the budget of 2
    tracker = ReasoningTracker(tags, [98], budget=2)
    tracker.observe([7])
    stub._reasoning = {1: tracker}
    stub._think_void = torch.tensor(sorted(tags.void))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    forced = Scheduler._plan_forced(stub, batch)
    assert forced.index.tolist() == [1, -1] and forced.token[0] == close

    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    free_before = cm.available_size
    # the verify accepted every draft of both requests
    out, emitted = forced.cut(torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]]), torch.tensor([3, 3]))
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=emitted)
    reply, finished = [], set()
    tables = [r.table_idx for r in reqs]

    Scheduler._drain_verify(stub, batch, output, reply, finished)

    assert [m.next_tokens for m in reply] == [[11, close], [21, 22, 23, 24]]
    # the target state and the page table take two tokens of request 1, and its close tag is the pending token
    assert events == [("commit", [1, 2], [2, 4])]
    assert (reqs[0].cached_len, reqs[0].device_len) == (PROMPT + 2, PROMPT + 3)
    assert int(stub.token_pool[tables[0], PROMPT + 2]) == close
    assert cm.available_size - free_before == (K + 1 - 2) + (K + 1 - 4)
    # one accepted draft: the forced token is not one
    assert reqs[0].stats.draft_n_accepted == 1 and reqs[1].stats.draft_n_accepted == 3
    assert not tracker.thinking and tracker.usage() == {"reasoning_tokens": 2, "closed_by": "budget"}
    # the span is closed: the next step forces nothing
    assert Scheduler._plan_forced(stub, Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)) is None
    for req in reqs:
        stub._free_req_resources(req)
    cm.check_integrity()


def test_a_step_commits_up_to_its_first_finishing_token():
    """A multi-token step stops at the first token that ends the request: the reply, the host ids, the stop match and the tool-call anchor see the committed tokens and none after them."""
    step = [11, 12, 13, 14]
    # (eos ids, stop strings, anchor id, max_tokens) -> (committed, finish reason, matched stop, anchor length)
    cases = [
        ((set(), [], None, 16), ([11, 12, 13, 14], None, None, None)),
        (({13}, [], None, 16), ([11, 12, 13], "stop", None, None)),
        # matched on the ids up to the token under test: with the step's later tokens already on the host it would match early
        ((set(), ["13 14"], None, 16), ([11, 12, 13, 14], "stop", "13 14", None)),
        ((set(), ["12 13"], None, 16), ([11, 12, 13], "stop", "12 13", None)),
        ((set(), [], 12, 16), ([11, 12, 13, 14], None, None, PROMPT + 3)),
        ((set(), [], None, 3), ([11, 12], "length", None, None)),
        # EOS wins over the length limit reached on the same token, and a finishing token sets no anchor
        (({12}, [], 12, 3), ([11, 12], "stop", None, None)),
    ]
    for (eos, stops, anchor, max_tokens), (committed, reason, matched, anchor_len) in cases:
        pool, cm, tm, dm, stub, events = _setup()
        stub.eos_token_ids, stub.toolcall_anchor_id = eos, anchor
        stub.tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
        stub._match_stop_str = lambda req: Scheduler._match_stop_str(stub, req)
        req = _decoding_req(pool, cm, tm, 1, max_tokens)
        req.sampling_params.ignore_eos = False
        req.sampling_params.stop_strs = stops
        before = req.input_ids.tolist()

        msg, n = Scheduler._commit_tokens(stub, req, torch.tensor(step))

        case = (eos, stops, anchor, max_tokens)
        assert (msg.next_tokens, n) == (committed, len(committed)), case
        assert (msg.finished, msg.finish_reason, msg.matched_stop) == (reason is not None, reason, matched), case
        assert req.input_ids.tolist() == before + committed, case
        assert req.toolcall_anchor_len == anchor_len, case


def _prefill_drain(with_draft: bool, max_tokens: int):
    """Drain the final prefill chunk of two requests (uid 1 finishes on its first token when max_tokens is 1); returns
    the recorded events and the replies."""
    pool, cm, tm, dm, stub, events = _setup()
    reqs = []
    for uid in (1, 2):
        prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32) + 100 * uid
        match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
        req = Req(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=max_tokens if uid == 1 else 16,
                  uid=uid, sampling_params=SamplingParams(max_tokens=max_tokens if uid == 1 else 16, ignore_eos=True),
                  cache_handle=match.cuda_handle)
        req.linear_slot_idx = pool.alloc(1)[0]
        req.mamba_ping_pong = tuple(pool.alloc(2))
        req.mamba_next_track_idx = 1
        cm.lock(match.cuda_handle)
        cm.allocate_paged([req])
        req.complete_one()
        reqs.append(req)
    dm.filter_reqs(reqs)
    batch = Batch(reqs=reqs, phase="prefill")
    cache_req = cm.cache_req

    def cached(req, finished):
        events.append(("cache", req.uid))
        cache_req(req, finished=finished)

    cm.cache_req = cached
    sent = []
    stub.send_result = lambda reply: (events.append(("send", [m.uid for m in reply])), sent.extend(reply))
    stub.finished_reqs = set()
    stub.status_reporter = SimpleNamespace(report_batch=lambda *a, **k: events.append(("report",)))
    stub.prefill_manager = SimpleNamespace(pending_list=[])
    stub.config.page_size = 1
    for name in ("_stamp_usage", "_finish_drained", "_kv_usage_pages", "_mamba_slot_usage", "_swa_token_usage"):
        setattr(stub, name, getattr(Scheduler, name).__get__(stub))
    stub._gpu_mem_bytes = lambda: 0
    tokens = torch.tensor([31, 32], dtype=torch.int32)
    draft = (lambda: events.append(("draft",))) if with_draft else None
    output = engine_output(tokens, draft)
    Scheduler._process_last_data(stub, (SimpleNamespace(batch=batch), output))
    return events, [(m.uid, m.next_tokens, m.finished, m.finish_reason) for m in sent]


class _Output(NamedTuple):
    """engine.ForwardOutput's fields, without importing the engine's kernels."""

    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: object
    draft_prefill: object


def engine_output(tokens, draft):
    return _Output(tokens, tokens, SimpleNamespace(synchronize=lambda: None), draft)


def test_a_prefill_drain_sends_the_first_tokens_before_the_draft_pass_and_frees_after_it():
    """The first sampled tokens leave before the draft's prompt pass, and every free and prefix insert of the batch
    waits for the pass, which reads and writes the requests' pages through the page table. The replies are those of
    a drain without a draft pass, whose page operations run in their places as before."""
    for max_tokens in (1, 16):
        events, replies = _prefill_drain(True, max_tokens)
        plain_events, plain_replies = _prefill_drain(False, max_tokens)
        assert replies == plain_replies == [(1, [31], max_tokens == 1, "length" if max_tokens == 1 else None),
                                            (2, [32], False, None)]
        pages = [e for e in events if e[0] in ("free", "cache")]
        assert events[: 2] == [("send", [1, 2]), ("draft",)] and events[2 : 2 + len(pages)] == pages
        # a finishing request is freed (its free commits the prefix), the other's prefix is inserted
        assert pages == ([("free", 1), ("cache", 1)] if max_tokens == 1 else [("cache", 1)]) + [("cache", 2)]
        # without a draft pass the order is the drain's as before: page operations first, then the report and the send
        assert plain_events == pages + [("report",), ("send", [1, 2])]


def test_a_warm_request_left_queued_or_running_is_dropped_with_what_it_holds():
    """The first-request warmup takes out a warm request the pools could not finish: one still queued would stand at
    the head of the queue for good. Queued, it leaves the queue; running, it leaves the running set and gives back its
    pages, table row and state slots; a uid it does not hold is left alone."""
    from freetoken.message import UserMsg

    pool, cm, tm, dm, stub, events = _setup()
    free_pages, free_rows, free_slots = cm.available_size, len(tm._free_slots), pool.num_free_slots
    stub._drop_unserved_warm = lambda uid: Scheduler._drop_unserved_warm(stub, uid)
    stub.prefill_manager.add_one_req(UserMsg(uid=-1, input_ids=torch.arange(1, 9, dtype=torch.int32),
                                             sampling_params=SamplingParams(max_tokens=2)))
    running = _decoding_req(pool, cm, tm, -2, 16)
    # as in the warmup: seeded, under reproducible seeds, so its finish commits nothing to the tree
    running.sampling_params.seed, cm.reproducible_seeds = 0, True
    dm.filter_reqs([running])
    assert stub.prefill_manager.runnable and dm.runnable

    assert not Scheduler._drop_unserved_warm(stub, -3)
    assert Scheduler._drop_unserved_warm(stub, -1)
    assert not stub.prefill_manager.runnable and dm.runnable
    assert Scheduler._drop_unserved_warm(stub, -2)
    assert not dm.runnable and events == [("free", -2)]
    assert (cm.available_size, len(tm._free_slots), pool.num_free_slots) == (free_pages, free_rows, free_slots)
    cm.check_integrity()
