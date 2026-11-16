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


class _RowsSink:
    """Copies a verify forward's final hidden rows into a buffer the runner owns, so a captured graph writes it too."""

    wanted_layers = frozenset()

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buffer[: x.shape[0]].copy_(x)

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        pass


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
        """How a short prefill chunk of one request is cut into captured steps, or None when it must run eagerly.

        ("chunk", b): a replay of the prefill graph of b tokens. ("verify", w): a verify step of w rows, whose rows past
        the chunk are padding. ("track", 0): the linear state as it is there is the chunk's track snapshot. Prefill graphs
        come first, largest first, so that verify steps only take what no graph fits. A chunk that owes a track snapshot
        (65 tokens or more) needs its first tokens, up to the tracked boundary, to be cut exactly by graphs of whole
        64-token multiples.
        """
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        engine, req, fla = self.engine, batch.reqs[0], batch.fla_metadata
        sizes = sorted(engine.chunk_runners, reverse=True)
        steps: List[tuple[str, int]] = []
        left = req.extend_len
        if fla is not None and fla.track_dst is not None:
            aligned = (left - 1) // CHUNK_SIZE * CHUNK_SIZE
            for b in sizes:
                if b % CHUNK_SIZE == 0:
                    while aligned >= b:
                        steps.append(("chunk", b))
                        aligned -= b
                        left -= b
            if aligned:
                return None
            steps.append(("track", 0))
        for b in sizes:
            while left >= b:
                steps.append(("chunk", b))
                left -= b
        if left > 2 * CHUNK_SIZE:
            return None  # too much for verify steps, which cost about as much a row as a decode step does
        end = req.cached_len + req.extend_len - left
        while left > 0:
            width = self.prefill_step_rows(left)
            if end + width > engine.page_table.shape[1]:
                return None
            steps.append(("verify", width))
            end += min(width, left)
            left -= min(width, left)
        return steps

    def prefill_as_verify(self, batch: Batch) -> torch.Tensor:
        """Run a short prefill chunk of one request as the captured steps of its plan and return its last token's logits, ``[1, vocab]``.

        An eager prefill forward of a few tokens spends tens of milliseconds reaching the GPU a few dozen times a layer;
        a replayed graph does not. A prefill graph is the prefill forward itself for a fixed token count. A verify step
        is k + 1 (or --prefill-graph-rows) consecutive rows of one request on the decode kernels: here every row is a
        prompt token, so each step's rows are committed as accepted, which leaves the K/V and the linear state where a
        prefill of those tokens leaves them; a step with fewer tokens than rows repeats its last token in the rows past
        the chunk and stores them to the padding page. A request with no cached prefix starts from a zeroed state slot,
        as an eager prefill does. The draft then reads the rows' hidden states as it does after a prefill.
        Steps on the decode kernels give the same weights' logits, not always the same bits.
        """
        from freetoken.attention.linear import gdn_slot
        from freetoken.core import Batch as _Batch

        engine, req, fla = self.engine, batch.reqs[0], batch.fla_metadata
        start, n = req.cached_len, req.extend_len
        pad_loc = engine.page_table[engine.dummy_req.table_idx, 0]
        pool, slot = engine.linear_state_pool, gdn_slot(req)
        mrope = engine.config.model_config.model_is_mrope
        if start == 0 and pool is not None:
            pool.recurrent_states[:, slot].zero_()
            pool.conv_states[:, slot].zero_()
            for state in pool.slot_states.values():
                state[:, slot].zero_()
        hidden, done, logits = [], 0, None
        for kind, size in batch.prefill_plan:
            if kind == "track":
                # the live slot holds the state at the tracked boundary: that is the snapshot an eager chunk writes
                src = slice(slot, slot + 1)
                pool.conv_states.index_copy_(1, fla.track_dst, pool.conv_states[:, src].expand(-1, fla.track_dst.numel(), -1, -1))
                rec = pool.recurrent_states
                rec.index_copy_(1, fla.track_dst, rec[:, src].expand(-1, fla.track_dst.numel(), *rec.shape[2:]))
                for state in pool.slot_states.values():
                    state.index_copy_(1, fla.track_dst, state[:, src].expand(-1, fla.track_dst.numel(), *state.shape[2:]))
                continue
            chunk = kind == "chunk"
            runner = engine.chunk_runners[size] if chunk else (engine.prefill_runner if size != self.k + 1 else engine.graph_runner)
            m = size if chunk else min(size, n - done)
            rows = _PrefillRows(req.table_idx, start + done, start + done + size, req.linear_slot_idx, req.mrope_delta, req.mrope_positions_full)
            step = _Batch(reqs=[rows], phase="prefill" if chunk else "verify", rows_per_req=size)
            runner.pad_batch(step)
            ids, out_loc = batch.input_ids[done : done + m], batch.out_loc[done : done + m]
            if m < size:
                ids = torch.cat([ids, ids[-1:].expand(size - m)])
                out_loc = torch.cat([out_loc, pad_loc.expand(size - m).to(out_loc.dtype)])
            if step.padded_size > 1:
                # the padding requests' rows, as a verify batch of the scheduler has them
                pad = (step.padded_size - 1) * size
                ids = torch.cat([ids, ids.new_zeros(pad)])
                out_loc = torch.cat([out_loc, pad_loc.expand(pad).to(out_loc.dtype)])
            step.input_ids, step.out_loc = ids, out_loc
            positions = [torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in step.padded_reqs]
            step.positions = torch.cat(positions).to(self.device, non_blocking=True)
            if mrope:
                step.mrope_positions = _mrope_rows(step.padded_reqs).to(self.device, non_blocking=True)
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
                    hidden.append(self._chunk_hidden[size].clone())
                else:
                    logits = out[m - 1 : m]
                    if runner.linear_backend is not None:
                        runner.linear_backend.scratch.commit(pool, [slot], [m])
                    hidden.append(self._wide_hidden[:m].clone())
            done += m
        self.proposer.prefill_sink_from(batch, torch.cat(hidden))
        return logits

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        self.proposer.on_prefill(batch, next_tokens_gpu)

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None:
        """Draft k tokens per request into token_pool after its pending token at cached_len."""
        rows = make_device_tensor([r.table_idx for r in reqs], torch.int64, self.device)
        cols = make_device_tensor([r.cached_len for r in reqs], torch.int64, self.device)
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
