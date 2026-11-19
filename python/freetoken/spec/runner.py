"""SpecRunner: the target side of speculative decoding and the order of one step; the Proposer it holds is the draft side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from freetoken.engine.sample import make_device_tensor
from freetoken.engine.spec_protocol import SpecForwardOutput

from .proposer import DraftContext, Proposer, SpecProposal
from .rejection import reject, reject_with_draft_probs

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.blocks import FeatureSink
    from freetoken.models.config import SlotStateSpec


@dataclass
class _PrefillRows:
    """The request fields the backends read, for one verify step over positions [cached_len, device_len) of a prefill chunk."""

    table_idx: int
    cached_len: int
    device_len: int
    linear_slot_idx: int | None
    mrope_delta: int = 0
    mrope_positions_full: torch.Tensor | None = None
    mamba_ping_pong: tuple | None = None  # a step never tracks: the plan places the chunk's snapshot

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


def _mrope_rows(reqs) -> torch.Tensor:
    """[3, n] rope rows of the requests' [cached_len, device_len): a prompt position's precomputed column, else position + delta."""
    cols = []
    for r in reqs:
        pos = torch.arange(r.cached_len, r.device_len, dtype=torch.int32)
        col = (pos + r.mrope_delta).unsqueeze(0).expand(3, -1).clone()
        full = r.mrope_positions_full
        if full is not None:
            inside = pos < full.shape[1]
            col[:, inside] = full[:, pos[inside].long()]
        cols.append(col)
    return torch.cat(cols, dim=1)


def _layer_kv(kv_cache, layer: int, loc: torch.Tensor) -> list[torch.Tensor]:
    """A layer's K and V rows at ``loc``; none for a layer without paged K/V (linear attention)."""
    try:
        return [kv_cache.k_cache(layer)[loc], kv_cache.v_cache(layer)[loc]]
    except (KeyError, IndexError):
        return []


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| over max |b|: nan where either holds a nan (torch's max keeps it)."""
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))


def _worst(values: list[float]) -> float:
    """The largest of ``values``, nan if any is nan (Python's max keeps a nan only when it comes first); -1 for none."""
    if any(v != v for v in values):
        return float("nan")
    return max(values, default=-1.0)


class _RowsSink:
    """Copies a verify forward's final hidden rows into a buffer the runner owns, so a captured graph writes it too."""

    wanted_layers = frozenset()

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buffer[: x.shape[0]].copy_(x)

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        pass


import os as _os

_STEP_SYNC = bool(_os.environ.get("FREETOKEN_PREFILL_GRAPH_SYNC"))  # diagnosis only: a sync and a line around every step


def proposer_class(config) -> type[Proposer]:
    from freetoken.models.register import _load_attr
    from freetoken.spec import DRAFT_PROPOSERS, PROPOSERS

    algorithm = config.speculative_algorithm
    if algorithm == "mtp":
        # a family's draft method picks the proposer behind the mtp CLI name; one with no draft of the
        # right method falls through to MTPProposer, whose adjust_config refuses draft-less families
        drafts = dict(config.model_spec.spec_drafts)
        for method, ref in DRAFT_PROPOSERS.items():
            if method in drafts:
                return _load_attr(*ref.split(":"))
    module, attr = PROPOSERS[algorithm].split(":")
    return _load_attr(module, attr)


class SpecRunner:
    """Implements engine.spec_protocol.SpecRuntime; the engine loads it by name and never imports freetoken.spec."""

    def __init__(self, config, engine) -> None:
        self.config = config
        self.device = engine.device
        self.k = config.speculative_num_steps
        self.proposer = proposer_class(config)(config, engine.model)
        self.radix_lookahead = self.proposer.radix_lookahead
        self.engine = None
        # the step's drafts, from propose_into to the verify_step of the same batch
        self._proposal: SpecProposal | None = None

    @classmethod
    def adjust_config(cls, config, override) -> None:
        proposer_cls = proposer_class(config)
        proposer_cls.adjust_config(config, override)
        default_k, max_k = proposer_cls.num_steps(config)
        if config.speculative_num_steps == 0:
            override("speculative_num_steps", default_k)
        if not 1 <= config.speculative_num_steps <= max_k:
            raise ValueError(
                f"--speculative-num-steps must be in [1, {max_k}] for "
                f"{config.speculative_algorithm}, got {config.speculative_num_steps}"
            )

    @classmethod
    def prepare_target(cls, config, model) -> None:
        proposer_class(config).prepare_target(config, model)

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        return proposer_class(config).budget(config)

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]:
        return proposer_class(config).slot_states(config)

    @classmethod
    def target_may_be_windowed(cls, config) -> bool:
        return proposer_class(config).target_may_be_windowed(config)

    def attach(self, engine) -> None:
        """Second construction phase, after the pools and before graph capture."""
        self.engine = engine
        self._host = torch.zeros(self.config.max_running_req + 1, self.k + 2, dtype=torch.int64, pin_memory=True)
        pool = engine.linear_state_pool
        states = {s.name: pool.slot_state(s.name) for s in self.config.spec_slot_states} if pool is not None else {}
        self.proposer.attach(DraftContext(
            page_table=engine.page_table, num_pages=engine.num_pages, device=engine.device,
            stream=engine.stream, dummy_req=engine.dummy_req, slot_states=states, kv_cache=engine.kv_cache,
        ))

    def detach(self, *, keep_kv: bool) -> None:
        self.proposer.detach(keep_kv=keep_kv)

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        self.proposer.capture_graphs(bs_list, pool)

    def feature_sink(self, batch: Batch) -> FeatureSink | None:
        return self.proposer.feature_sink(batch)

    @property
    def can_prefill_as_verify(self) -> bool:
        return type(self.proposer).adopt_verify_rows is not Proposer.adopt_verify_rows

    def prefill_rows_sink(self, rows: int) -> _RowsSink:
        """The sink of the wider verify graph: its rows' final hidden, in a buffer shaped like the proposer's."""
        like = self.proposer.adopt_verify_rows(1)
        self._wide_hidden = torch.zeros(rows, like.shape[1], dtype=like.dtype, device=like.device)
        return _RowsSink(self._wide_hidden)

    def chunk_rows_sink(self, tokens: int) -> _RowsSink:
        """The sink of a captured prefill forward of ``tokens`` tokens."""
        like = self.proposer.adopt_verify_rows(1)
        self._chunk_hidden = getattr(self, "_chunk_hidden", {})
        self._chunk_hidden[tokens] = torch.zeros(tokens, like.shape[1], dtype=like.dtype, device=like.device)
        return _RowsSink(self._chunk_hidden[tokens])

    def prefill_step_rows(self, n: int) -> int:
        """Rows of the verify step that takes ``n`` tokens of a prefill chunk: the decode step's, or the wider graph's when more than one decode step is left."""
        wide = self.engine.prefill_runner
        return wide.rows_per_req if wide is not None and n > self.k + 1 else self.k + 1

    def prefill_plan(self, batch: Batch) -> List[tuple[str, int]] | None:
        """How a short prefill chunk of one request is cut into steps, or None when it must run as one eager forward.

        ("chunk", b): a replay of the prefill graph of b tokens. ("bucket", b): a replay of the prefill graph of b
        tokens for the chunk's tokens left, at most b, whose rows past them are padding; the replay writes the chunk's
        snapshot itself. ("eager", m): an eager prefill forward of m tokens (no
        plan made here has one; the self-test builds a prefix with it). ("verify", w): a verify step of w rows, whose rows past the chunk are padding. ("track", 0): the linear state
        as it is there is the chunk's snapshot.

        With prefill graphs (--prefill-graph-chunks) a chunk of at most --prefill-graph-tokens tokens runs as one or
        two replays of them, cut at the last 64-token boundary before its end, where its snapshot goes anyway: the
        result is that of eager prefill forwards cut at that token. A chunk no such pair covers runs as one eager
        forward. Without prefill graphs the chunk runs as verify steps, on the decode kernels.
        """
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        engine, req, fla = self.engine, batch.reqs[0], batch.fla_metadata
        sizes = sorted(engine.chunk_runners, reverse=True)
        if not sizes and engine.config.prefill_graph_chunks:
            return None  # prefill graphs were asked for and are not there (a rebuild): eager, never verify steps in their place
        steps: List[tuple[str, int]] = []
        left = req.extend_len

        if sizes:
            # One or two replays, or nothing: a replay of a captured prefill forward has a fixed cost of several
            # milliseconds, so a chunk cut into many of them is slower than one eager forward. The cut is where the
            # eager forward's own tracking puts the chunk's snapshot (the last 64-token boundary before its end), so
            # the snapshot costs no step of its own and the request's mark and slot are the ones the forward's
            # metadata already set.
            if left > engine.config.prefill_graph_tokens or not engine.chunk_graphs_usable():
                return None
            if getattr(engine, "chunk_pad", False) and left > CHUNK_SIZE:
                # padded: one replay of the smallest captured multiple of 64 that holds the chunk
                bucket = min((b for b in sizes if b % CHUNK_SIZE == 0 and b >= left), default=None)
                if bucket is not None:
                    if fla is not None and fla.track_dst is not None:
                        batch.prefill_track_dst = fla.track_dst
                    return [("bucket", bucket)]
            head = (left - 1) // CHUNK_SIZE * CHUNK_SIZE
            tail = left - head
            if tail not in engine.chunk_runners or (head and head not in engine.chunk_runners):
                return None
            if head:
                steps.append(("chunk", head))
                if fla is not None and fla.track_dst is not None:
                    batch.prefill_track_dst = fla.track_dst
                    steps.append(("track", 0))
            steps.append(("chunk", tail))
            return steps
        if fla is not None and fla.track_dst is not None:
            return None  # verify steps cannot stop at the tracked boundary
        if left > 2 * CHUNK_SIZE:
            return None  # too much for verify steps, which cost about as much a row as a decode step does
        end = req.cached_len
        while left > 0:
            width = self.prefill_step_rows(left)
            if end + width > engine.page_table.shape[1]:
                return None
            steps.append(("verify", width))
            end += min(width, left)
            left -= min(width, left)
        return steps

    def prefill_as_verify(self, batch: Batch, check: bool = False) -> torch.Tensor:
        """Run a short prefill chunk of one request as the steps of its plan and return its last token's logits, ``[1, vocab]``.

        An eager prefill forward of a few tokens spends tens of milliseconds reaching the GPU a few dozen times a layer;
        a replayed graph does not. A prefill graph is the prefill forward itself for a fixed token count, continuing
        the request's linear state; a request with no cached prefix starts from a zeroed state slot, as an eager
        prefill does. A verify step is k + 1 (or --prefill-graph-rows) consecutive rows of one request on the decode
        kernels: every row is a prompt token, so each step's rows are committed as accepted; a step with fewer tokens
        than rows repeats its last token in the rows past the chunk and stores them to the padding page. Verify steps
        give the same weights' logits, not always the same bits. The draft reads the rows' hidden states as it does
        after a prefill.

        ``check``: every prefill graph step is also run as an eager prefill forward of the same tokens from the same
        state, and a line says whether the two are equal bit for bit (logits, hidden rows, K/V rows, linear state) and
        how far apart they are (max abs difference over the eager forward's max abs value; nan where either is not finite).
        ``batch.pad_loc``, where set, is the page the padding rows store to in place of the padding request's first page
        (the self-test lays its check request on that request's row).
        """
        from freetoken.attention.linear import gdn_slot
        from freetoken.core import Batch as _Batch

        engine, req = self.engine, batch.reqs[0]
        start, n = req.cached_len, req.extend_len
        pad_loc = getattr(batch, "pad_loc", None)
        if pad_loc is None:
            pad_loc = engine.page_table[engine.dummy_req.table_idx, 0]
        pool, slot = engine.linear_state_pool, gdn_slot(req)
        mrope = engine.config.model_config.model_is_mrope
        states = [] if pool is None else [pool.recurrent_states, pool.conv_states, *pool.slot_states.values()]
        if start == 0:
            for state in states:
                state[:, slot].zero_()

        def make(kind: str, size: int, done: int, m: int):
            # a padded replay's attention and linear metadata see the chunk's own rows only
            end = start + done + (m if kind == "bucket" else size)
            rows = _PrefillRows(req.table_idx, start + done, end, req.linear_slot_idx, req.mrope_delta, req.mrope_positions_full)
            step = _Batch(reqs=[rows], phase="verify" if kind == "verify" else "prefill", rows_per_req=size if kind != "eager" else None)
            ids, out_loc = batch.input_ids[done : done + m], batch.out_loc[done : done + m]
            if m < size:
                ids = torch.cat([ids, ids[-1:].expand(size - m)])
                out_loc = torch.cat([out_loc, pad_loc.expand(size - m).to(out_loc.dtype)])
            return step, ids, out_loc

        def place(step, ids, out_loc) -> None:
            if step.padded_size > 1:
                # the padding requests' rows, as a verify batch of the scheduler has them
                pad = (step.padded_size - 1) * step.rows_per_req
                ids = torch.cat([ids, ids.new_zeros(pad)])
                out_loc = torch.cat([out_loc, pad_loc.expand(pad).to(out_loc.dtype)])
            step.input_ids, step.out_loc = ids, out_loc
            positions = torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in step.padded_reqs])
            # the pad rows of a bucket repeat the last position, so that no rope row past the context is read
            pad = ids.numel() - positions.numel()
            step.positions = torch.cat([positions, positions[-1:].expand(pad)]).to(self.device, non_blocking=True)
            if mrope:
                cols = _mrope_rows(step.padded_reqs)
                step.mrope_positions = torch.cat([cols, cols[:, -1:].expand(-1, pad)], dim=1).to(self.device, non_blocking=True)

        def eager(size: int, done: int) -> tuple[torch.Tensor, torch.Tensor]:
            """One eager prefill forward of the chunk's tokens [done, done + size): its last row's logits and its rows' hidden."""
            step, ids, out_loc = make("eager", size, done, size)
            engine.graph_runner.pad_batch(step)
            place(step, ids, out_loc)
            if engine.graph_runner.linear_backend is not None:
                engine.graph_runner.linear_backend.prepare_metadata(step)
            engine.graph_runner.attn_backend.prepare_metadata(step)
            step.features = self.feature_sink(step)
            out = engine.forward_logits(step)[:1]
            return out, step.features.hidden

        def kv_rows(done: int, size: int) -> list[torch.Tensor]:
            loc = batch.out_loc[done : done + size].long()
            return [t.clone() for layer in range(engine.config.model_config.num_layers) for t in _layer_kv(engine.kv_cache, layer, loc)]

        def kv_wipe(done: int, size: int) -> None:
            loc = batch.out_loc[done : done + size].long()
            for layer in range(engine.config.model_config.num_layers):
                try:
                    engine.kv_cache.k_cache(layer)[loc] = 0
                    engine.kv_cache.v_cache(layer)[loc] = 0
                except (KeyError, IndexError):
                    continue

        hidden, done, logits = [], 0, None
        for kind, size in batch.prefill_plan:
            if _STEP_SYNC:
                torch.cuda.synchronize()
                print(f"PREFILL_STEP begin {kind} {size} done={done}/{n} start={start}", flush=True)
            if kind == "track":
                # the live slot holds the state at this boundary: that is the snapshot
                dst, src = batch.prefill_track_dst, slice(slot, slot + 1)
                for state in states:
                    state.index_copy_(1, dst, state[:, src].expand(-1, dst.numel(), *state.shape[2:]))
                continue
            if kind == "eager":
                logits, rows_hidden = eager(size, done)
                hidden.append(rows_hidden.clone())
                done += size
                continue
            bucket = kind == "bucket"
            chunk = kind == "chunk" or bucket
            runner = engine.chunk_runners[size] if chunk else (engine.prefill_runner if size != self.k + 1 else engine.graph_runner)
            m = size if kind == "chunk" else min(size, n - done)
            if chunk and check:
                before = [state[:, slot].clone() for state in states]
                want_logits, want_hidden = eager(m, done)
                want = [want_logits.clone(), want_hidden.clone(), *kv_rows(done, m), *[state[:, slot].clone() for state in states]]
                for state, saved in zip(states, before):
                    state[:, slot] = saved
                # the eager forward's K/V rows are wiped, so that a replay that stored nothing, or elsewhere, shows
                kv_wipe(done, m)
                # the pages past the chunk, which a pad row must not write
                past = engine.page_table[req.table_idx, start + done + m : start + done + size].long()
                past_kv = [t.clone() for layer in range(engine.config.model_config.num_layers) for t in _layer_kv(engine.kv_cache, layer, past)]
            step, ids, out_loc = make(kind, size, done, m)
            runner.pad_batch(step)
            place(step, ids, out_loc)
            if bucket:
                step.prefill_track_dst = getattr(batch, "prefill_track_dst", None)
            if getattr(runner, "pad_rows", None) is not None:
                runner.pad_rows[:m].fill_(False)
                runner.pad_rows[m:].fill_(True)
            if runner.linear_backend is not None:
                runner.linear_backend.prepare_metadata(step)
            runner.attn_backend.prepare_metadata(step)
            if runner is engine.graph_runner:
                step.features = self.feature_sink(step)
                logits = engine.forward_logits(step)[m - 1 : m]
                if pool is not None and pool.verify_scratch is not None:
                    pool.commit_verify([slot], [m])
                hidden.append(self.proposer.adopt_verify_rows(m))
            else:
                with engine.ctx.forward_batch(step, **runner.ctx_override), engine.model.forward_host_ctx(step, True):
                    out = runner.replay(step)
                if chunk:
                    logits = out[:1]
                    hidden.append(self._chunk_hidden[size][:m].clone())
                else:
                    logits = out[m - 1 : m]
                    if runner.linear_backend is not None:
                        runner.linear_backend.scratch.commit(pool, [slot], [m])
                    hidden.append(self._wide_hidden[:m].clone())
            if chunk and check:
                got = [logits, hidden[-1], *kv_rows(done, m), *[state[:, slot] for state in states]]
                same = [torch.equal(a, b) for a, b in zip(got, want, strict=True)]
                kv = same[2 : len(same) - len(states)]
                far = float((got[0].float() - want[0].float()).abs().max())
                top = want[0][0].float().topk(2).values
                kv_rel = [_rel(a, b) for a, b in zip(got[2 : len(got) - len(states)], want[2 : len(want) - len(states)])]
                line = (
                    f"size={size} start={start + done} logits_equal={same[0]} logits_max_abs_diff={far:.6g} "
                    f"hidden_equal={same[1]} kv_equal={all(kv)} kv_tensors={len(kv)} kv_first_unequal={kv.index(False) if not all(kv) else -1} "
                    f"state_equal={same[len(same) - len(states):]} argmax_equal={int(got[0].argmax()) == int(want[0].argmax())} "
                    f"eager_top2_gap={float(top[0] - top[1]):.6g} logits_rel={_rel(got[0], want[0]):.3g} hidden_rel={_rel(got[1], want[1]):.3g} "
                    f"kv_rel_max={_worst(kv_rel):.3g} "
                    f"state_rel={[round(_rel(a, b), 6) for a, b in zip(got[len(got) - len(states):], want[len(want) - len(states):])]}"
                )
                if bucket:
                    untouched = all(torch.equal(a, b) for a, b in zip(
                        past_kv, [t for layer in range(engine.config.model_config.num_layers) for t in _layer_kv(engine.kv_cache, layer, past)]))
                    print(f"PREFILL_PAD_CHECK rows={m} {line} pad_kv_untouched={untouched}", flush=True)
                else:
                    print(f"PREFILL_SPLIT_CHECK {line}", flush=True)
            done += m
            if _STEP_SYNC:
                torch.cuda.synchronize()
                print(f"PREFILL_STEP ok {kind} {size} m={m} done={done}/{n} start={start}", flush=True)
        self.proposer.prefill_sink_from(batch, torch.cat(hidden))
        return logits

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        self.proposer.on_prefill(batch, next_tokens_gpu)

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None:
        """Draft k tokens per request into token_pool after its pending token at cached_len."""
        # both index rows in one host-to-device copy
        rows, cols = make_device_tensor([[r.table_idx for r in reqs], [r.cached_len for r in reqs]], torch.int64, self.device).unbind(0)
        self._proposal = self.proposer.propose(reqs, rows, token_pool[rows, cols])
        cols = cols.unsqueeze(1) + 1 + torch.arange(self.k, device=self.device)
        token_pool[rows.unsqueeze(1), cols] = self._proposal.draft_token_ids.to(token_pool.dtype)

    def verify_step(self, batch: Batch, sample_args: BatchSamplingArgs) -> SpecForwardOutput:
        engine = self.engine
        bs, width = batch.size, self.k + 1
        # each request's rows read its pending token then its drafts
        drafts = batch.input_ids[: bs * width].view(bs, width)[:, 1:].to(torch.int64)
        batch.features = self.feature_sink(batch)
        logits = engine.forward_logits(batch)[: bs * width]
        proposal, self._proposal = self._proposal, None
        if proposal.draft_support_probs is not None:
            out, emitted = reject_with_draft_probs(
                logits, drafts, proposal.draft_support_ids, proposal.draft_support_probs, sample_args, engine.sampler,
            )
        else:
            out, emitted = reject(logits, drafts, sample_args, engine.sampler)
        if sample_args.forced is not None:
            # before the draft advances and the host copy: both must see the step the request commits
            out, emitted = sample_args.forced.cut(out, emitted)
        host = self._host[:bs]
        host.copy_(torch.cat([out, emitted.unsqueeze(1)], dim=1), non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(engine.stream)
        # after the D2H is queued: the drain waits for the copy only, while the draft's work overlaps it
        self.proposer.post_verify(batch, out, emitted)
        return SpecForwardOutput(out, host[:, :width], copy_done, host[:, width])

    def release(self, req: Req) -> None:
        self.proposer.release(req)

    @property
    def keeps_full_hits(self) -> bool:
        """Whether the proposer can hand over and take back its per-row state after a prompt (scheduler/full_hit.py)."""
        return hasattr(self.proposer, "full_hit_rows")

    def full_hit_rows(self, table_idx: int):
        return self.proposer.full_hit_rows(table_idx)

    def restore_full_hit_rows(self, table_idx: int, rows) -> None:
        self.proposer.restore_full_hit_rows(table_idx, rows)
