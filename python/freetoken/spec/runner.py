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

    def prefill_as_verify(self, batch: Batch) -> torch.Tensor:
        """Run a short prefill chunk of one request as verify steps and return its last token's logits, ``[1, vocab]``.

        A verify step is a forward of k + 1 consecutive rows of one request on the decode kernels, captured in a
        CUDA graph; an eager prefill forward of two tokens costs several times one. Here every row is a prompt
        token, so none is drafted or rejected: each step's rows are committed as accepted (commit_verify), which
        leaves the K/V and the linear state where a prefill of those tokens leaves them. A last step with fewer
        tokens than rows repeats its last token in the rows past the chunk and stores them to the padding page;
        nothing of those rows is kept. The draft then reads the rows' hidden states as it does after a prefill.
        The logits come from the decode kernels, not the prefill ones: the same weights, not always the same bits.
        """
        from freetoken.attention.linear import gdn_slot
        from freetoken.core import Batch as _Batch

        engine, req, width = self.engine, batch.reqs[0], self.k + 1
        start, n = req.cached_len, req.extend_len
        pad_loc = engine.page_table[engine.dummy_req.table_idx, 0]
        pool = engine.linear_state_pool
        hidden = []
        for done in range(0, n, width):
            m = min(width, n - done)
            rows = _PrefillRows(
                req.table_idx, start + done, start + done + width, req.linear_slot_idx,
                req.mrope_delta, req.mrope_positions_full,
            )
            step = _Batch(reqs=[rows], phase="verify", rows_per_req=width)
            engine.graph_runner.pad_batch(step)
            ids, out_loc = batch.input_ids[done : done + m], batch.out_loc[done : done + m]
            if m < width:
                ids = torch.cat([ids, ids[-1:].expand(width - m)])
                out_loc = torch.cat([out_loc, pad_loc.expand(width - m).to(out_loc.dtype)])
            if step.padded_size > 1:
                # the padding requests' rows, as a verify batch of the scheduler has them
                pad = (step.padded_size - 1) * width
                ids = torch.cat([ids, ids.new_zeros(pad)])
                out_loc = torch.cat([out_loc, pad_loc.expand(pad).to(out_loc.dtype)])
            step.input_ids, step.out_loc = ids, out_loc
            positions = [torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in step.padded_reqs]
            step.positions = torch.cat(positions).to(self.device, non_blocking=True)
            if engine.config.model_config.model_is_mrope:
                step.mrope_positions = _mrope_rows(step.padded_reqs).to(self.device, non_blocking=True)
            if engine.linear_backend is not None:
                engine.linear_backend.prepare_metadata(step)
            engine.attn_backend.prepare_metadata(step)
            step.features = self.feature_sink(step)
            logits = engine.forward_logits(step)
            if pool is not None and pool.verify_scratch is not None:
                pool.commit_verify([gdn_slot(req)], [m])
            hidden.append(self.proposer.adopt_verify_rows(m))
        self.proposer.prefill_sink_from(batch, torch.cat(hidden))
        return logits[m - 1 : m]

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
