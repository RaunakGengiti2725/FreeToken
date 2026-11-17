from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


class ForcedTokens(NamedTuple):
    """Tokens the host picked in place of sampled ones: request i commits ``token[i]`` as token ``index[i]`` of its step, -1 for none."""

    index: torch.Tensor  # [bs] int64
    token: torch.Tensor  # [bs] int64
    void: torch.Tensor   # [n] int64: one of these among a request's earlier tokens of the step cancels its forced token

    def replace(self, tokens: torch.Tensor) -> torch.Tensor:
        """A decode step's tokens [bs] with the forced ones in place."""
        return torch.where(self.index == 0, self.token.to(tokens.dtype), tokens)

    def cut(self, out: torch.Tensor, emitted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A verify step's (out [bs, k+1], emitted [bs]) with each forced token ending its request's accepted run, as a draft rejected at that row does."""
        cols = torch.arange(out.shape[1], device=out.device)
        index = self.index.unsqueeze(1)
        voided = (torch.isin(out, self.void) & (cols < index)).any(dim=1)
        # a row past the first rejected draft was computed on a token that is not committed
        hit = (self.index >= 0) & (self.index <= emitted) & ~voided
        out = torch.where(hit.unsqueeze(1) & (cols == index), self.token.unsqueeze(1).to(out.dtype), out)
        return out, torch.where(hit, self.index.to(emitted.dtype), emitted)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    penalties: list[tuple[int, torch.Tensor, float, float]] = field(default_factory=list)
    min_p: torch.Tensor | None = None
    # (row, the request's [V] bool mask of the tokens in its context, penalty)
    repetition: list[tuple[int, torch.Tensor, float]] = field(default_factory=list)
    # set by the scheduler between steps; None on every batch without a forced token
    forced: ForcedTokens | None = None
    # draw_key per row for the rows of a sampled request with a seed, None for the other rows; None when no row has one
    keys: list[int | None] | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sampling_module():
    """flashinfer.sampling when installed, else the triton kernels with the same API."""
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling
    return sampling


def sampling_streams() -> tuple:
    """The state of every stream an unseeded draw reads: torch's CUDA and CPU generators and, once the Triton sampling
    kernels are loaded, their own unseeded generators."""
    triton_sampling = sys.modules.get("freetoken.kernel.triton.sampling")
    cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    kernels = {} if triton_sampling is None else {slot: g.get_state() for slot, g in triton_sampling._UGEN.items()}
    return cuda, torch.get_rng_state(), kernels


def restore_sampling_streams(saved: tuple) -> None:
    """Put back what sampling_streams read; a Triton generator made since then is dropped, so it starts afresh."""
    cuda, cpu, kernels = saved
    for index, state in enumerate(cuda):
        torch.cuda.set_rng_state(state, index)
    torch.set_rng_state(cpu)
    triton_sampling = sys.modules.get("freetoken.kernel.triton.sampling")
    if triton_sampling is not None:
        for slot in list(triton_sampling._UGEN):
            if slot in kernels:
                triton_sampling._UGEN[slot].set_state(kernels[slot])
            else:
                del triton_sampling._UGEN[slot]


def min_p_renorm_probs(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """``probs`` [B, V] with every token under ``min_p`` [B] times its row's top probability zeroed, renormalized."""
    kept = torch.where(probs >= min_p.unsqueeze(1) * probs.amax(dim=1, keepdim=True), probs, 0.0)
    return kept / kept.sum(dim=1, keepdim=True)


def repetition_penalize(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """``logits`` with the tokens of the bool mask ``seen`` made less likely: a positive logit divided by ``penalty``, a negative one multiplied."""
    return torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)


def draw_key(seed: int, position: int) -> int:
    """Seed of the draw that samples the token at ``position`` of a request with ``seed``.

    It reads the token's position, not a step count, so the draw is the same in whichever step and batch the token is sampled.
    The mix is the one kernel.triton.sampling applies to (seed, offset); 63 bits fit both backends' seed argument.
    """
    return (seed * 0x9E3779B97F4A7C15 + position) & 0x7FFFFFFFFFFFFFFF


def key_uniform(key: int) -> float:
    """The uniform in [0, 1) a draw_key gives: splitmix64's finalizer of the key, its top 24 bits, which a float32 holds exactly."""
    z = (key + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return ((z ^ (z >> 31)) >> 40) * 2.0**-24


def keyed_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None,
) -> torch.Tensor:
    """[B, V] fp32: softmax(T), then top-k, then top-p, then min-p, as sample_impl applies them, but the min-p cut keeps
    its tokens' probabilities unscaled: keyed_draw scales by the row's total, and a division by a sum would bring in
    last bits that depend on how many rows torch's reduction covered."""
    sampling = sampling_module()
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is not None:
        probs = sampling.top_k_renorm_probs(probs, top_k)
    if top_p is not None:
        probs = sampling.top_p_renorm_probs(probs, top_p)
    if min_p is not None:
        probs = torch.where(probs >= min_p.unsqueeze(1) * probs.amax(dim=1, keepdim=True), probs, 0.0)
    return probs


def keyed_draw(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """One token per row of ``probs`` [n, V], by the inverse CDF at ``uniforms`` [n] in [0, 1) times the row's total:
    the first token whose running sum passes it, which always has a probability above 0. The sums run over at least
    two rows, as torch scans a lone row with another kernel than the rows of a matrix."""
    n = probs.shape[0]
    if n == 1:
        probs = probs.expand(2, -1)
    cdf = probs.cumsum(dim=-1)[:n]
    target = (uniforms.to(cdf.dtype) * cdf[:, -1]).unsqueeze(1)
    tokens = torch.searchsorted(cdf, target, right=True).squeeze(1)
    # a target that rounds up to the total: the last token with a probability above 0
    positive = probs[:n] > 0
    last = probs.shape[1] - 1 - positive.flip(1).to(torch.uint8).argmax(dim=1)
    return torch.minimum(tokens, last)


# A/B switch: FREETOKEN_SEEDED_DRAW=batched draws every seeded row of a step in one pass (keyed_probs, keyed_draw), for
# its speed. It is not batch-independent on CUDA: a row's top-k / top-p probabilities in a batch of keyed rows are not
# the bits it gets alone, so a seeded request's text can follow the requests beside it. The default draws each seeded
# row in a call of its own. Its tokens are batch-independent only when every keyed row of the step has the same
# filters: on an RTX 5090, a request with top-k 20 and top-p 0.95 among requests with other temperatures and top-p drew
# other tokens than alone (box B, 1c946ea). One running request is unaffected.
_SEEDED_DRAW_PER_ROW = os.getenv("FREETOKEN_SEEDED_DRAW", "per-row") != "batched"


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    min_p: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    sampling = sampling_module()
    rng = {} if seed is None else {"seed": seed, "offset": 0}
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if min_p is not None:
        # min-p runs last: its cut is relative to the top token, which top-k and top-p always keep
        if top_k is not None:
            probs = sampling.top_k_renorm_probs(probs, top_k)
        if top_p is not None:
            probs = sampling.top_p_renorm_probs(probs, top_p)
        return sampling.min_p_sampling_from_probs(probs, min_p, **rng)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs, **rng)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k, **rng)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p, **rng)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p, **rng)


def sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None,
    keys: list[int | None] | None,
) -> torch.Tensor:
    """sample_impl over the batch; a row with a key is then drawn again from that key, in a call of its own. Under
    FREETOKEN_SEEDED_DRAW=batched the keyed rows are drawn in one pass instead (keyed_draw), each from the uniform its
    own key gives; not batch-independent on CUDA (_SEEDED_DRAW_PER_ROW). Either way a seeded row's token is the one it
    draws alone only where every keyed row of the step has the same filters."""
    tokens = sample_impl(logits, temperatures, top_k, top_p, min_p)
    if not keys:
        return tokens
    if _SEEDED_DRAW_PER_ROW:
        for row, key in enumerate(keys):
            if key is None:
                continue
            # One row per call: a batched call takes one seed, and flashinfer mixes the row index into it. Alone, a row's
            # softmax and draw also run the same arithmetic whatever else the step holds.
            one = slice(row, row + 1)
            tokens[one] = sample_impl(
                logits[one], temperatures[one], None if top_k is None else top_k[one], None if top_p is None else top_p[one],
                None if min_p is None else min_p[one], key,
            )
        return tokens
    rows = [row for row, key in enumerate(keys) if key is not None]
    if not rows:
        return tokens
    at = make_device_tensor(rows, torch.int64, logits.device)
    uniforms = make_device_tensor([key_uniform(keys[row]) for row in rows], torch.float32, logits.device)

    def pick(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.index_select(0, at)

    probs = keyed_probs(logits.index_select(0, at), pick(temperatures), pick(top_k), pick(top_p), pick(min_p))
    tokens[at] = keyed_draw(probs, uniforms).to(tokens.dtype)
    return tokens


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    # the sampling values of the last batch that had a sampled request and their device tensors: decode steps repeat them, so only a change is copied again
    _sent: tuple | None = field(default=None, init=False, repr=False)

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        penalties = []
        repetition = []
        for row, req in enumerate(batch.reqs):
            p = req.sampling_params
            if p.repetition_penalty != 1.0 and req.can_decode:
                if req.context_token_seen is None:
                    # the prompt; sample() and commit_verify add every token committed after it
                    ids = req.input_ids[req.input_ids < self.vocab_size]
                    seen = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
                    seen[ids.to(self.device).long()] = True
                    req.context_token_seen = seen
                repetition.append((row, req.context_token_seen, p.repetition_penalty))
            if not (p.presence_penalty or p.frequency_penalty) or not req.can_decode:
                continue
            if req.output_token_counts is None:
                req.output_token_counts = torch.zeros(
                    self.vocab_size, dtype=torch.int32, device=self.device
                )
            penalties.append(
                (row, req.output_token_counts, p.presence_penalty, p.frequency_penalty)
            )
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None, penalties=penalties, repetition=repetition)
        # called before a verify batch is widened: device_len is where row 0's token goes, row j's goes j further
        width = 1 if batch.is_prefill else batch.rows_per_req
        keys = [
            None if g or p.seed is None or not r.can_decode else draw_key(p.seed, r.device_len + j)
            for r, p, g in zip(batch.reqs, params, is_greedy)
            for j in range(width)
        ]

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            min(p.top_k, self.vocab_size) if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        min_ps = [0.0 if g else p.min_p for p, g in zip(params, is_greedy)]
        values = (ts, top_ks, top_ps, min_ps, is_greedy)
        if self._sent is None or self._sent[0] != values:
            temperatures = make_device_tensor(ts, torch.float32, self.device)
            top_k, top_p, min_p = None, None, None
            if any(k != self.vocab_size for k in top_ks):
                top_k = make_device_tensor(top_ks, torch.int32, self.device)
            if any(p < 1.0 for p in top_ps):
                top_p = make_device_tensor(top_ps, torch.float32, self.device)
            if any(m > 0.0 for m in min_ps):
                min_p = make_device_tensor(min_ps, torch.float32, self.device)
            greedy_mask = (
                make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
            )
            self._sent = (values, (temperatures, top_k, top_p, min_p, greedy_mask))
        temperatures, top_k, top_p, min_p, greedy_mask = self._sent[1]
        return BatchSamplingArgs(
            temperatures, top_k=top_k, top_p=top_p, greedy_mask=greedy_mask, penalties=penalties,
            min_p=min_p, repetition=repetition,
            keys=keys if any(k is not None for k in keys) else None,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.penalties or args.repetition:
                logits = logits.float().clone()
                # repetition scales the raw logit, so it goes before the additive penalties
                for row, seen, penalty in args.repetition:
                    logits[row] = repetition_penalize(logits[row], seen, penalty)
                for row, counts, presence, frequency in args.penalties:
                    logits[row] -= frequency * counts + presence * (counts > 0)
            if args.temperatures is None:  # greedy sampling
                tokens = torch.argmax(logits, dim=-1)
            else:
                tokens = sample_rows(logits.float(), args.temperatures, args.top_k, args.top_p, args.min_p, args.keys)
                if args.greedy_mask is not None:
                    # Mixed batches still run probability sampling for all rows, but
                    # greedy rows must follow argmax's deterministic tie-breaking.
                    greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                    tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            if args.forced is not None:
                tokens = args.forced.replace(tokens)
            # Update on the sampling stream: overlapped scheduling can prepare the next
            # batch before the previous token reaches Req.input_ids on the CPU.
            for row, counts, _, _ in args.penalties:
                counts.scatter_add_(
                    0, tokens[row : row + 1].long(), counts.new_ones(1)
                )
            for row, seen, _ in args.repetition:
                seen.index_fill_(0, tokens[row : row + 1].long(), True)
            return tokens

    def sample_verify(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor:
        """One token per verify row as decode would sample it, ``drafts`` [bs, k] giving k+1 rows per request.

        Row j of a request is penalized as if its drafts d1..dj were already committed; the penalty counts and the context mask are not updated.
        """
        bs, k = drafts.shape
        width = k + 1
        if args.penalties or args.repetition:
            logits = self._penalize_rows(logits, args, drafts)
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)

        def rows(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.repeat_interleave(width)

        tokens = sample_rows(logits.float(), rows(args.temperatures), rows(args.top_k), rows(args.top_p), rows(args.min_p), args.keys)
        if args.greedy_mask is not None:
            greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
            tokens = torch.where(rows(args.greedy_mask), greedy_tokens, tokens)
        return tokens

    def probs(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor | None:
        """[bs*(k+1), V] fp32 distribution each verify row samples from (softmax(T), then top-k, then top-p, then min-p, as sample_impl applies them), penalized like sample_verify and one-hot at the argmax on greedy rows; None when every request is greedy."""
        if args.temperatures is None:
            return None
        sampling = sampling_module()
        width = drafts.shape[1] + 1
        if args.penalties or args.repetition:
            logits = self._penalize_rows(logits, args, drafts)
        top = None if args.greedy_mask is None else torch.argmax(logits, dim=-1, keepdim=True)
        probs = sampling.softmax(logits.float(), args.temperatures.repeat_interleave(width), enable_pdl=is_sm90_supported())
        # frees the penalized copy before the renorms allocate theirs
        del logits
        if args.top_k is not None:
            probs = sampling.top_k_renorm_probs(probs, args.top_k.repeat_interleave(width))
        if args.top_p is not None:
            probs = sampling.top_p_renorm_probs(probs, args.top_p.repeat_interleave(width))
        if args.min_p is not None:
            probs = min_p_renorm_probs(probs, args.min_p.repeat_interleave(width))
        if top is not None:
            greedy = args.greedy_mask.repeat_interleave(width).unsqueeze(1)
            hot = torch.where(greedy, 1.0, probs.gather(1, top))
            probs.masked_fill_(greedy, 0.0).scatter_(1, top, hot)
        return probs

    def _penalize_rows(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor:
        k = drafts.shape[1]
        width = k + 1
        logits = logits.float().clone()
        # (row j, draft i) pairs with i < j: row j has seen drafts 0..j-1
        pair_rows, pair_drafts = torch.tril_indices(width, k, offset=-1, device=logits.device)
        for req_row, seen, penalty in args.repetition:
            block = logits[req_row * width : (req_row + 1) * width]
            in_context = seen.repeat(width, 1)
            in_context[pair_rows, drafts[req_row].long()[pair_drafts]] = True
            block.copy_(repetition_penalize(block, in_context, penalty))
        earlier = torch.ones(k, k, dtype=torch.bool, device=logits.device).tril(-1)
        for req_row, counts, presence, frequency in args.penalties:
            block = logits[req_row * width : (req_row + 1) * width]
            block -= frequency * counts + presence * (counts > 0)
            if k == 0:
                continue
            d = drafts[req_row].long()
            # presence applies once per token: only to a draft that is new to both the counts and the earlier drafts
            first = ~((d.unsqueeze(1) == d.unsqueeze(0)) & earlier).any(dim=1)
            extra = frequency + presence * ((counts[d] == 0) & first).float()
            block.index_put_((pair_rows, d[pair_drafts]), -extra[pair_drafts], accumulate=True)
        return logits
