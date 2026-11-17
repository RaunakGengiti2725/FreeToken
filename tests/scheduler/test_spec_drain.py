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


class _FullHitOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: object
    draft_prefill: object
    prefill_logits: object


def _full_hit_setup(monkeypatch):
    """CPU hybrid managers keyed one token ahead (MTP's radix), a store, and a stub scheduler with the real drain,
    admission and full-hit step; the sampler is greedy argmax."""
    from freetoken.scheduler.full_hit import FullHitStore

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=16, dtype=torch.float32, device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(128, 1, page_table, "hybrid_radix", linear_state_pool=pool, radix_lookahead=True)
    tm = TableManager(max_running_reqs=4, page_table=page_table)
    dm = DecodeManager(page_size=1, token_headroom=K)
    pm = PrefillManager(cm, tm, dm, token_headroom=K)
    store = FullHitStore(pool.free)
    cm.on_evict = store.clear
    sampler = SimpleNamespace(prepare=lambda batch: None, sample=lambda logits, args: logits.argmax(dim=-1))
    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm, prefill_manager=pm, spec=None, full_hits=store,
        engine=SimpleNamespace(sampler=sampler, stream=None, linear_state_pool=pool), device=torch.device("cpu"),
        config=SimpleNamespace(decode_rows_per_req=K + 1, page_size=1), token_pool=tm.token_pool, eos_token_ids=set(),
        toolcall_anchor_id=None, _match_stop_str=lambda _req: None, _reasoning={}, finished_reqs=set(),
        status_reporter=SimpleNamespace(report_batch=lambda *a, **k: None), send_result=lambda reply: None,
        _gpu_mem_bytes=lambda: 0,
    )
    for name in ("_stamp_usage", "_finish_drained", "_kv_usage_pages", "_mamba_slot_usage", "_swa_token_usage",
                 "_free_req_resources", "_commit_tokens", "_keep_full_hit", "_full_hit_batch", "_forward_full_hit",
                 "_restore_linear_states"):
        setattr(stub, name, getattr(Scheduler, name).__get__(stub))
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None, synchronize=lambda: None))
    import freetoken.kernel as kernel

    def compare(x, y):
        # the radix walk's compiled key compare, in torch: the length of the common start
        n = min(len(x), len(y))
        differ = (x[:n] != y[:n]).nonzero()
        return int(differ[0]) if len(differ) else n

    monkeypatch.setattr(kernel, "fast_compare_key", compare)
    return pool, cm, tm, dm, pm, store, stub


def test_an_identical_resend_takes_the_first_sends_state_logits_and_pages_with_no_forward(monkeypatch):
    """FREETOKEN_FULL_HIT: the first send's final prefill drain keeps its live GDN state (a copy), its last-row logits and
    its pages; once it has finished, an identical resend takes the tree's pages for the whole prompt, the copy of the
    state and the same first token, and reports the whole prompt cached. A prompt with another last token, another
    first-token sampling or an unseeded sampled request prefills as before, and an eviction drops the entry."""
    from freetoken.message import UserMsg

    pool, cm, tm, dm, pm, store, stub = _full_hit_setup(monkeypatch)
    vocab = 50
    prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32)
    greedy = SamplingParams(max_tokens=8, ignore_eos=True)

    # the first send, its prompt prefilled: live state, logits whose argmax is 31
    match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
    first = Req(input_ids=prompt.clone(), table_idx=tm.allocate(), cached_len=0, output_len=8, uid=1,
                sampling_params=greedy, cache_handle=match.cuda_handle)
    first.linear_slot_idx, first.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    cm.lock(match.cuda_handle)
    cm.allocate_paged([first])
    first.complete_one()
    dm.filter_reqs([first])
    torch.manual_seed(5)
    pool.recurrent_states[:, first.linear_slot_idx] = torch.randn_like(pool.recurrent_states[:, first.linear_slot_idx])
    pool.conv_states[:, first.linear_slot_idx] = torch.randn_like(pool.conv_states[:, first.linear_slot_idx])
    logits = torch.zeros(1, vocab)
    logits[0, 31] = 1.0
    out = _FullHitOutput(torch.tensor([31], dtype=torch.int32), torch.tensor([31], dtype=torch.int32),
                         SimpleNamespace(synchronize=lambda: None), None, logits.clone())
    Scheduler._process_last_data(stub, (SimpleNamespace(batch=Batch(reqs=[first], phase="prefill")), out))
    (entry,) = store.entries
    assert entry.first_token == 31 and torch.equal(entry.input_ids, prompt) and torch.equal(entry.logits, logits)
    assert entry.state_slot != first.linear_slot_idx
    assert torch.equal(pool.recurrent_states[:, entry.state_slot], pool.recurrent_states[:, first.linear_slot_idx])
    assert torch.equal(entry.kv_indices, cm.page_table[first.table_idx, :PROMPT])
    state = pool.recurrent_states[:, entry.state_slot].clone(), pool.conv_states[:, entry.state_slot].clone()

    # while the first send decodes, its prompt is not in the tree yet: a resend prefills
    pm.add_one_req(UserMsg(uid=2, input_ids=prompt.clone(), sampling_params=greedy))
    assert Scheduler._full_hit_batch(stub) is None and len(pm.pending_list) == 1 and len(store) == 1
    pm.pending_list.clear()
    # it decodes and finishes with three tokens, the last one pending: its prompt and answer go into the tree
    for token in (7, 8):
        cm.allocate_paged([first])  # the slot of the token this step reads
        first.append_host(torch.tensor([token], dtype=torch.int32))
        first.commit_pending(1)
    with cm.lazy_free_region():
        stub._finish_drained(first)
    free_pages = len(cm.free_slots)

    def resend(uid, ids, params):
        pm.add_one_req(UserMsg(uid=uid, input_ids=ids.clone(), sampling_params=params))
        return Scheduler._full_hit_batch(stub)

    # another last token, another first-token sampling, an unseeded sampled request: no hit, the request stays queued
    other = prompt.clone()
    other[-1] += 1
    assert resend(3, other, greedy) is None
    assert resend(4, prompt, SamplingParams(max_tokens=8, ignore_eos=True, repetition_penalty=1.2)) is None
    assert resend(5, prompt, SamplingParams(temperature=0.7, max_tokens=8, ignore_eos=True)) is None
    assert [p.uid for p in pm.pending_list] == [3, 4, 5] and len(store) == 1
    pm.pending_list.clear()

    batch = resend(6, prompt, greedy)
    assert batch is not None and batch.full_hit is entry and not pm.pending_list
    (req,) = batch.reqs
    assert (req.cached_len, req.device_len, req.cache_handle.cached_len) == (PROMPT - 1, PROMPT, PROMPT)
    assert torch.equal(cm.page_table[req.table_idx, :PROMPT], entry.kv_indices)
    assert req.mamba_restore_src == entry.state_slot and batch.prompt_admissions == [(6, PROMPT, PROMPT)]
    assert len(cm.free_slots) == free_pages, "the resend allocated nothing for its prompt"

    stub._restore_linear_states(batch)
    write = (torch.tensor([req.table_idx]), torch.tensor([PROMPT]))
    output = stub._forward_full_hit((batch, None, None, write))
    assert output.next_tokens_cpu.tolist() == [31] and int(stub.token_pool[req.table_idx, PROMPT]) == 31
    assert torch.equal(pool.recurrent_states[:, req.linear_slot_idx], state[0])
    assert torch.equal(pool.conv_states[:, req.linear_slot_idx], state[1])
    assert (req.cached_len, req.device_len) == (PROMPT, PROMPT + 1) and req in dm.running_reqs
    sent = []
    stub.send_result = sent.extend
    Scheduler._process_last_data(stub, (SimpleNamespace(batch=batch), output))
    assert [(m.uid, m.next_tokens, m.finished) for m in sent] == [(6, [31], False)]
    assert len(store) == 1, "a full hit stores nothing again"

    # the resend finishes; then an eviction drops the entry and gives its state slot back
    with cm.lazy_free_region():
        stub._finish_drained(req)
    free_slots = pool.num_free_slots
    cm.flush()
    assert len(store) == 0 and pool.num_free_slots > free_slots
    cm.check_integrity()


def test_the_first_token_key_names_what_decides_a_deterministic_first_token():
    from freetoken.scheduler.full_hit import first_token_key

    greedy = first_token_key(SamplingParams(max_tokens=3))
    assert greedy == first_token_key(SamplingParams(max_tokens=99, ignore_eos=True, stop_strs=["x"]))
    assert first_token_key(SamplingParams(temperature=0.7)) is None
    seeded = first_token_key(SamplingParams(temperature=0.7, top_p=0.9, seed=3))
    assert seeded is not None and seeded != first_token_key(SamplingParams(temperature=0.7, top_p=0.9, seed=4))
    assert greedy != first_token_key(SamplingParams(presence_penalty=1.5))
