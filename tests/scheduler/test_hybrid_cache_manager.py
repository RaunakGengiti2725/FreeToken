"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_a_final_chunk_commits_the_snapshot_of_the_chunk_before_it():
    """A prompt longer than one chunk whose last chunk crosses no boundary: the snapshot the chunk before wrote is
    committed with the final chunk, so the same prompt sent again has a prefix to reuse."""
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(512, 1, pt, "hybrid_radix", linear_state_pool=pool)
    pm = PrefillManager(cm, TableManager(max_running_reqs=3, page_table=pt), DecodeManager(page_size=1))
    prompt = torch.arange(1, 131, dtype=torch.int32)  # 128 tokens in the first chunk, 2 in the last
    pm.pending_list = [PendingReq(0, prompt, SamplingParams(max_tokens=1))]

    first = pm.schedule_next_batch(128).reqs[0]
    assert isinstance(first, ChunkedReq) and first.extend_len == 128
    cm.allocate_paged([first])
    # what the chunk's forward does (attention/linear.py: _build_track_metadata): a snapshot at the deepest boundary
    # strictly inside the chunk, written to the ping-pong slot it then flips away from
    frozen = first.mamba_ping_pong[first.mamba_next_track_idx]
    first.mamba_last_track_seqlen, first.mamba_next_track_idx = 64, 1 - first.mamba_next_track_idx
    first.complete_one()

    last = pm.schedule_next_batch(128).reqs[0]
    assert not isinstance(last, ChunkedReq) and last.extend_len == 2
    cm.allocate_paged([last])
    last.complete_one()
    cm.cache_req(last, finished=False)

    again = cm.match_req(_pend(prompt.tolist()))
    assert again.cuda_handle.cached_len == 64 and again.mamba_value == frozen


def test_turn_anchor_is_where_the_next_turn_parts_from_the_prompt():
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.utils.hf import load_turn_anchor_tails

    class Tok:
        """One id a character; the generation prompt ends with a tag only when thinking is off, and a rendered reply drops it."""

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "".join(f"<{m['role'][0]}>{m['content']}" for m in msgs) + "<a>" + ("" if enable_thinking else "TT")

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    tails = load_turn_anchor_tails(Tok())
    assert tails == ((ord("T"), ord("T")),)  # with thinking on the generation prompt survives whole: no tail

    pt = torch.zeros(4, 64, dtype=torch.int32)
    prompt = torch.tensor([1, 2, 3, 4, ord("T"), ord("T")], dtype=torch.int32)
    for lookahead in (False, True):
        cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=_pool(), radix_lookahead=lookahead)
        pm = PrefillManager(cm, TableManager(max_running_reqs=3, page_table=pt), DecodeManager(page_size=1))
        pm.turn_anchor_tails = tails
        # a node's last key holds the token after it under lookahead, so the anchor stops one token earlier
        assert pm._turn_anchor(prompt) == (3 if lookahead else 4)
        assert pm._turn_anchor(prompt[:4]) is None


def test_prefill_plan_covers_the_chunk_and_snapshots_where_the_next_request_matches():
    """SpecRunner.prefill_plan with prefill graphs: the steps cover the chunk's tokens exactly, there is one snapshot, at
    the turn anchor when the chunk crosses it and else one token before the end, and the request's mark and slot agree with it."""
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.spec.runner import SpecRunner

    sizes, limit, pair = [1 << i for i in range(11)], 2048, (7, 8)
    runner = object.__new__(SpecRunner)
    runner.k, runner.device = 3, torch.device("cpu")
    runner.engine = SimpleNamespace(
        chunk_runners=dict.fromkeys(sizes), chunk_graphs_usable=lambda: True, page_table=torch.zeros(1, 1 << 16),
        config=SimpleNamespace(prefill_graph_tokens=limit), prefill_runner=None,
    )

    def plan(start, n, anchor=None, chunked=False):
        tracked = (n - 1) // 64 >= 1  # what attention/linear.py's own tracking does for this extend
        req = object.__new__(ChunkedReq) if chunked else SimpleNamespace()
        req.cached_len, req.turn_anchor_len, req.mamba_ping_pong = start, anchor, pair
        req.mamba_next_track_idx, req.mamba_last_track_seqlen = (1, start + (n - 1) // 64 * 64) if tracked else (0, None)
        if chunked:
            req.device_len = start + n
        else:
            req.extend_len = n
        fla = SimpleNamespace(track_dst=torch.tensor([pair[0]]) if tracked else None)
        batch = SimpleNamespace(reqs=[req], fla_metadata=fla)
        return runner.prefill_plan(batch), req, batch, tracked

    for start in (0, 100):
        for n in [*range(1, 200), 1023, 1024, 1025, 2047, 2048, 2049, 5000, 8192]:
            for anchor in (None, start + n - 5, start + 3, start):
                for chunked in (False, True):
                    steps, req, batch, tracked = plan(start, n, anchor, chunked)
                    done, at = 0, None
                    for i, (kind, size) in enumerate(steps):
                        if kind == "track":
                            assert at is None and 0 < done < n
                            at = done
                            continue
                        # an eager step only opens a cut longer than the limit; nothing is left over, since size 1 is captured
                        opens = i == 0 or steps[i - 1][0] == "track"
                        assert kind == "chunk" and size in sizes or kind == "eager" and opens and size % 64 == 0 and n > limit, (n, steps)
                        done += size
                    assert done == n, (start, n, anchor, chunked, steps)
                    if chunked:
                        want = (n - 1) // 64 * 64 if tracked else None
                    elif anchor is not None and start < anchor < start + n:
                        want = anchor - start
                    else:
                        want = n - 1 if n >= 2 else None
                    assert at == want, (start, n, anchor, chunked, steps)
                    if at is None:
                        assert req.mamba_last_track_seqlen is None and req.mamba_next_track_idx == 0
                        continue
                    assert req.mamba_last_track_seqlen == start + at
                    # the slot the commit donates is the one the plan's snapshot goes to
                    assert batch.prefill_track_dst.tolist() == [pair[1 - req.mamba_next_track_idx]]


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
