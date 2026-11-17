"""The sampler's min-p filter and repetition penalty: on a plain decode step, behind a graph replay and on the rows of a speculative verify.

With CUDA the real sampling kernels run; without it a plain-torch copy of the calls the sampler makes stands in, so the filter order, the per-request routing and the verify statistics are checked on any machine. The kernels have their own test in tests/kernels.
"""

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine import sample as S
from freetoken.engine.sample import Sampler

VOCAB = 24


class TorchKernels:
    """The calls the sampler makes on the sampling kernel module, in plain torch."""

    @staticmethod
    def softmax(logits, temperature, enable_pdl=None):
        return torch.softmax(logits.float() / temperature.unsqueeze(1), dim=-1)

    @staticmethod
    def _renorm(probs, thr):
        kept = torch.where(probs >= thr, probs, 0.0)
        return kept / kept.sum(dim=1, keepdim=True)

    @classmethod
    def top_k_renorm_probs(cls, probs, top_k):
        ranked = probs.sort(dim=1, descending=True).values
        return cls._renorm(probs, ranked.gather(1, top_k.long().unsqueeze(1) - 1))

    @classmethod
    def top_p_renorm_probs(cls, probs, top_p):
        ranked = probs.sort(dim=1, descending=True).values
        inside = ranked.cumsum(dim=1) - ranked < top_p.unsqueeze(1)
        return cls._renorm(probs, torch.where(inside, ranked, 2.0).amin(dim=1, keepdim=True))

    @staticmethod
    def sampling_from_probs(probs):
        return torch.multinomial(probs, 1).squeeze(1).int()

    @classmethod
    def top_k_sampling_from_probs(cls, probs, top_k):
        return cls.sampling_from_probs(cls.top_k_renorm_probs(probs, top_k))

    @classmethod
    def top_p_sampling_from_probs(cls, probs, top_p):
        return cls.sampling_from_probs(cls.top_p_renorm_probs(probs, top_p))

    @classmethod
    def top_k_top_p_sampling_from_probs(cls, probs, top_k, top_p):
        return cls.top_p_sampling_from_probs(cls.top_k_renorm_probs(probs, top_k), top_p)

    @classmethod
    def min_p_sampling_from_probs(cls, probs, min_p):
        return cls.sampling_from_probs(cls._renorm(probs, min_p.unsqueeze(1) * probs.amax(dim=1, keepdim=True)))


def _backends() -> list[str]:
    if not torch.cuda.is_available():
        return ["torch"]
    from freetoken.kernel.backend import is_flashinfer_installed

    return ["triton", "flashinfer"] if is_flashinfer_installed() else ["triton"]


@pytest.fixture(params=_backends())
def device(request, monkeypatch):
    if request.param == "torch":
        monkeypatch.setattr(S, "sampling_module", TorchKernels)
        # a CPU build of torch has no NVTX
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
        return torch.device("cpu")
    from freetoken.kernel import backend

    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: request.param == "flashinfer")
    return torch.device("cuda")


def reference_probs(logits, seen=(), penalty=1.0, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """One row's sampling distribution the way transformers chains its processors: the repetition penalty on the raw logits, then temperature, top-k, top-p and min-p, each one masking logits."""
    x = logits.detach().double().cpu().clone()
    for token in set(seen):
        x[token] = x[token] / penalty if x[token] > 0 else x[token] * penalty
    x /= temperature
    if top_k:
        x[x < x.topk(top_k).values[-1]] = -math.inf
    if top_p < 1.0:
        p, order = torch.softmax(x, dim=0).sort(descending=True)
        x[order[p.cumsum(0) - p >= top_p]] = -math.inf
    if min_p:
        p = torch.softmax(x, dim=0)
        x[p < min_p * p.max()] = -math.inf
    return torch.softmax(x, dim=0).float()


def _requests(params: list[SamplingParams], prompt: list[int]) -> list[Req]:
    return [Req(input_ids=torch.tensor(prompt, dtype=torch.int32), table_idx=i, cached_len=0, output_len=64, uid=i,
                sampling_params=p, cache_handle=None) for i, p in enumerate(params)]


def _context(req: Req) -> list[int]:
    return req.context_token_seen.nonzero().flatten().tolist()


@pytest.mark.parametrize("filters", [
    dict(min_p=0.2),
    dict(min_p=0.2, temperature=1.7),
    dict(min_p=0.05, temperature=0.8, top_k=8, top_p=0.9),
    # each of top-p and min-p being the one that cuts deeper
    dict(min_p=0.01, top_p=0.5),
    dict(min_p=0.5, top_k=12, top_p=0.99),
    # top-p reads the distribution min-p has not cut yet: after it, this nucleus would be smaller
    dict(min_p=0.3, top_p=0.8),
])
def test_min_p_filters_after_temperature_top_k_and_top_p(device, filters):
    """A request's min_p keeps the tokens with at least that share of the top token's probability, after its temperature, top-k and top-p: the distribution the verify reads matches the reference, and decode draws from it."""
    torch.manual_seed(0)
    distinct, copies, calls = 6, 500, 8
    # drawn on the CPU so every machine tests the same rows
    logits = (2.0 * torch.randn(distinct, VOCAB)).to(device)
    want = torch.stack([reference_probs(row, **filters) for row in logits]).to(device)
    # top-k of the whole vocabulary and top-p of 1 leave the request's fields at their off values
    params = SamplingParams(temperature=filters.get("temperature", 1.0), top_k=filters.get("top_k", -1),
                            top_p=filters.get("top_p", 1.0), min_p=filters["min_p"])
    sampler = Sampler(device, VOCAB)
    args = sampler.prepare(Batch(reqs=_requests([params] * distinct * copies, [1]), phase="decode"))
    logits = logits.repeat(copies, 1)

    probs = sampler.probs(logits, args, torch.empty(distinct * copies, 0, dtype=torch.int64, device=device))
    torch.testing.assert_close(probs[:distinct], want, atol=1e-6, rtol=1e-4)

    counts = torch.zeros(distinct, VOCAB, device=device)
    for _ in range(calls):
        tokens = sampler.sample(logits, args).long().view(copies, distinct)
        counts.scatter_add_(1, tokens.T, torch.ones(distinct, copies, device=device))
    n = copies * calls
    assert not counts[want == 0].any()
    # 5 sigma of a frequency over n draws
    assert (counts / n - want).abs().max() < 5 * (0.25 / n) ** 0.5


def test_prepare_follows_every_change_of_the_sampling_values(device):
    """One sampler prepares batch after batch and repeats the device tensors of a batch whose sampling values did not change: every field that does change, and a change of the batch itself, must reach the tensors the step samples with."""
    params = dict(temperature=0.7, top_k=8, top_p=0.9, min_p=0.1)
    steps = [[dict(params)], [dict(params)]]
    # one field at a time, the others as the step before left them; the last three turn each filter off, which drops its tensor
    for change in (dict(temperature=1.3), dict(top_k=5), dict(top_p=0.5), dict(min_p=0.3), dict(top_k=-1), dict(top_p=1.0), dict(min_p=0.0)):
        params.update(change)
        steps.append([dict(params)])
    plain, greedy = dict(temperature=1.0), dict(temperature=0.0)
    # a second request whose values are a greedy row's without and with the greedy flag, then the batch reordered and shrunk
    steps += [[params, plain], [params, greedy], [greedy, params], [greedy], [params]]
    sampler = Sampler(device, VOCAB)
    for step in steps:
        batch = Batch(reqs=_requests([SamplingParams(**p) for p in step], [1]), phase="decode")
        got, want = sampler.prepare(batch), Sampler(device, VOCAB).prepare(batch)
        for name in ("temperatures", "top_k", "top_p", "min_p", "greedy_mask"):
            a, b = getattr(got, name), getattr(want, name)
            assert (a is None) == (b is None) and (a is None or torch.equal(a, b)), (name, step)


def test_repetition_penalty_scales_every_context_token_once(device):
    """The prompt's tokens, a repeated one once, and each token sampled since are penalized: a positive logit divided, a negative one multiplied. A request without the penalty in the same batch keeps its logits and gets no mask."""
    penalty, prompt = 1.8, [23, 0, 0, 7, 23]
    logits = torch.linspace(-3.0, 3.0, VOCAB, device=device).repeat(3, 1)
    reqs = _requests([SamplingParams(temperature=1.0, repetition_penalty=penalty), SamplingParams(temperature=1.0),
                      SamplingParams(temperature=0.0, repetition_penalty=penalty)], prompt)
    batch = Batch(reqs=reqs, phase="decode")
    sampler = Sampler(device, VOCAB)
    no_drafts = torch.empty(3, 0, dtype=torch.int64, device=device)
    sampled, greedy = list(prompt), list(prompt)
    for _ in range(3):
        args = sampler.prepare(batch)
        assert [row for row, _, _ in args.repetition] == [0, 2] and args.min_p is None
        probs = sampler.probs(logits, args, no_drafts).cpu()
        torch.testing.assert_close(probs[0], reference_probs(logits[0], sampled, penalty), atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(probs[1], reference_probs(logits[1]), atol=1e-6, rtol=1e-4)
        tokens = sampler.sample(logits, args).tolist()
        # the top token is in the prompt, then each greedy pick joins it: the argmax walks down
        assert tokens[2] == int(reference_probs(logits[2], greedy, penalty).argmax())
        sampled.append(tokens[0])
        greedy.append(tokens[2])
    assert greedy[len(prompt):] == [22, 21, 20]
    assert _context(reqs[0]) == sorted(set(sampled)) and _context(reqs[2]) == sorted(set(greedy))
    assert reqs[1].context_token_seen is None


def test_each_verify_row_filters_as_the_decode_step_it_stands_for(device):
    """Row j of a verify is penalized for the context plus drafts d1..dj and then cut by min-p, on the sampled draws, on the distribution the draft-probs verify reads and on a greedy request's argmax; the context mask stays as it was."""
    torch.manual_seed(2)
    penalty, min_p, prompt, k = 1.6, 0.15, [3, 7, 7], 3
    drafts = torch.tensor([[5, 9, 5], [2, 2, 4]], device=device)
    reqs = _requests([SamplingParams(temperature=1.0, min_p=min_p, repetition_penalty=penalty),
                      SamplingParams(temperature=0.0, repetition_penalty=penalty)], prompt)
    sampler = Sampler(device, VOCAB)
    args = sampler.prepare(Batch(reqs=reqs, phase="verify", rows_per_req=k + 1))
    logits = (2.0 * torch.randn(2 * (k + 1), VOCAB)).to(device)

    probs = sampler.probs(logits, args, drafts).cpu()
    draws = torch.stack([sampler.sample_verify(logits, args, drafts) for _ in range(200)]).long().cpu()
    for j in range(k + 1):
        want = reference_probs(logits[j], prompt + drafts[0, :j].tolist(), penalty, min_p=min_p)
        torch.testing.assert_close(probs[j], want, atol=1e-6, rtol=1e-4, msg=f"row {j}")
        assert (want[draws[:, j]] > 0).all(), f"row {j}"
        top = int(reference_probs(logits[k + 1 + j], prompt + drafts[1, :j].tolist(), penalty).argmax())
        assert (draws[:, k + 1 + j] == top).all() and probs[k + 1 + j, top] == 1.0, f"greedy row {j}"
    assert _context(reqs[0]) == _context(reqs[1]) == sorted(set(prompt))


@pytest.mark.parametrize("route", ["eager", "graph", "verify"])
def test_every_decode_route_samples_with_the_requests_filters(device, monkeypatch, route):
    """A request's min_p and repetition_penalty reach the sampler from a plain forward, from a graph replay and from a speculative verify step. With min_p = 1 only the top token of the penalized logits survives, so every route must commit it; the context grows with what was committed."""
    from freetoken.engine.engine import Engine
    from freetoken.spec.proposer import SpecProposal
    from freetoken.spec.runner import SpecRunner

    prompt, k = [23, 0], 2
    logits = torch.linspace(-3.0, 3.0, VOCAB, device=device)
    reqs = _requests([SamplingParams(temperature=1.0, min_p=1.0, repetition_penalty=1.8)] * 3, prompt)
    sampler = Sampler(device, VOCAB)
    forwards = []
    if device.type == "cuda":
        stream = torch.cuda.current_stream()
    else:
        stream = object()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
        monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None))
    engine = SimpleNamespace(
        stream=stream, spec=None, cpu_moe_executor=None, linear_state_pool=None, sampler=sampler,
        ctx=SimpleNamespace(forward_batch=lambda batch: nullcontext()),
        graph_runner=SimpleNamespace(can_use_cuda_graph=lambda batch: route == "graph"),
        model=SimpleNamespace(forward_host_ctx=lambda batch, use_graph: nullcontext()),
    )
    engine.forward_logits = lambda batch: Engine.forward_logits(engine, batch)

    def forward(kind, rows):
        forwards.append(kind)
        return logits.repeat(rows, 1)

    if route != "verify":
        batch = Batch(reqs=reqs, phase="decode")
        engine.graph_runner.replay = lambda batch: forward("graph", 3)
        engine.model.forward = lambda: forward("eager", 3)
        out = Engine.forward_batch(engine, batch, sampler.prepare(batch))
        assert forwards == [route]
        # 23 is in the prompt, so its penalized logit falls under token 22's
        assert out.next_tokens_gpu.tolist() == [22, 22, 22]
        assert all(_context(req) == [0, 22, 23] for req in reqs)
        return

    # every draft right, the second wrong, the first wrong
    drafts = torch.tensor([[22, 21], [22, 5], [23, 22]], device=device)
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=k + 1)
    batch.input_ids = torch.cat([torch.full((3, 1), prompt[-1], device=device), drafts], dim=1).flatten()
    engine.model.forward = lambda: forward("eager", 3 * (k + 1))
    runner = SimpleNamespace(
        engine=engine, k=k, feature_sink=lambda batch: None, _proposal=SpecProposal(drafts),
        _host=torch.zeros(4, k + 2, dtype=torch.int64), proposer=SimpleNamespace(post_verify=lambda batch, out, emitted: None),
    )
    out = SpecRunner.verify_step(runner, batch, sampler.prepare(batch))
    emitted = out.emitted_cpu.tolist()
    assert emitted == [2, 1, 0]
    committed = [row[: n + 1] for row, n in zip(out.next_tokens_cpu.tolist(), emitted)]
    assert committed == [[22, 21, 20], [22, 21], [22]]
    Engine.commit_verify(engine, reqs, [n + 1 for n in emitted], out.next_tokens_gpu)
    assert [_context(req) for req in reqs] == [[0, 20, 21, 22, 23], [0, 21, 22, 23], [0, 22, 23]]


def _homogeneity_p_value(a: torch.Tensor, b: torch.Tensor) -> float:
    """p-value of Pearson's chi-square test that two samples of equal size share a distribution; cells under 10 counts are pooled."""
    small = a + b < 10
    a = torch.cat([a[~small], a[small].sum().view(1)])
    b = torch.cat([b[~small], b[small].sum().view(1)])
    cells = a + b > 0
    df = int(cells.sum()) - 1
    if df < 1:
        return 1.0
    stat = ((a - b) ** 2 / (a + b))[cells].sum()
    return float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64), stat / 2.0))


@pytest.mark.parametrize("drafts_from", ["argmax", "sampled"])
def test_speculative_decoding_commits_the_distribution_plain_decode_samples(device, drafts_from):
    """At temperature 1 with min_p and repetition_penalty set, the first three tokens a request commits follow the same distribution with speculative decoding on and off.

    The target is a table of next-token logits per last token, drawn on the CPU so every machine tests the same one. Off, three decode steps sample it. On, three verify steps commit accepted drafts and the target's own token, with argmax drafts (MTP) or drafts sampled from a q over four candidates (DFlash), from a draft that knows neither filter.
    Each arm's per-position token frequencies must sit within 5 sigma of the exact marginals and off their support; the two arms must pass a chi-square homogeneity test at p > 1e-6.
    """
    from freetoken.engine.engine import Engine
    from freetoken.spec.rejection import reject, reject_with_draft_probs

    torch.manual_seed(3)
    k, steps, bs, rounds = 2, 3, 128, 125
    penalty, min_p, prompt = 1.7, 0.1, [1, 4, 4, 9, 13, 20]
    # the diagonal makes repeating the last token likely, so a row that misses a draft in its context shows
    target = (2.0 * torch.randn(VOCAB, VOCAB) + 2.5 * torch.eye(VOCAB)).to(device)
    draft = target + torch.randn(VOCAB, VOCAB).to(device)
    params = SamplingParams(temperature=1.0, min_p=min_p, repetition_penalty=penalty)
    sampler = Sampler(device, VOCAB)
    engine = SimpleNamespace(linear_state_pool=None)
    slots = torch.arange(k + 1, device=device)

    def admit():
        reqs = _requests([params] * bs, prompt)
        return reqs, sampler.prepare(Batch(reqs=reqs, phase="decode")), torch.full((bs,), prompt[-1], device=device)

    def decode():
        _, args, last = admit()
        tokens = []
        for _ in range(steps):
            last = sampler.sample(target[last], args).long()
            tokens.append(last)
        return torch.stack(tokens, dim=1)

    def speculate():
        reqs, args, last = admit()
        committed = torch.zeros(bs, steps * (k + 1), dtype=torch.int64, device=device)
        filled = torch.zeros(bs, dtype=torch.int64, device=device)
        for _ in range(steps):
            drafts, support, q, prev = [], [], [], last
            for _ in range(k):
                if drafts_from == "argmax":
                    prev = draft[prev].argmax(dim=-1)
                else:
                    top = draft[prev].topk(4, dim=-1)
                    q.append(torch.softmax(top.values, dim=-1))
                    support.append(top.indices)
                    prev = top.indices.gather(1, torch.multinomial(q[-1], 1)).squeeze(1)
                drafts.append(prev)
            drafts = torch.stack(drafts, dim=1)
            logits = target[torch.cat([last.unsqueeze(1), drafts], dim=1)].view(bs * (k + 1), VOCAB)
            if drafts_from == "argmax":
                out, emitted = reject(logits, drafts, args, sampler)
            else:
                out, emitted = reject_with_draft_probs(logits, drafts, torch.stack(support, dim=1), torch.stack(q, dim=1), args, sampler)
            Engine.commit_verify(engine, reqs, (emitted + 1).tolist(), out)
            keep = slots <= emitted.unsqueeze(1)
            committed[keep.nonzero()[:, 0], (filled.unsqueeze(1) + slots)[keep]] = out[keep]
            filled += emitted + 1
            last = out.gather(1, emitted.unsqueeze(1)).squeeze(1)
        return committed[:, :steps]

    def histogram(arm):
        tokens = torch.cat([arm() for _ in range(rounds)])
        return torch.stack([torch.bincount(tokens[:, j], minlength=VOCAB) for j in range(steps)]).double().cpu()

    exact = torch.zeros(steps, VOCAB, dtype=torch.float64)

    def walk(last, seen, weight, position):
        if position == steps:
            return
        p = reference_probs(target[last], seen, penalty, min_p=min_p).double()
        exact[position] += weight * p
        for token in p.nonzero().flatten().tolist():
            walk(token, seen + [token], weight * float(p[token]), position + 1)

    walk(prompt[-1], prompt, 1.0, 0)
    off, on = histogram(decode), histogram(speculate)
    n = bs * rounds
    for name, counts in (("off", off), ("on", on)):
        assert not counts[exact == 0].any(), name
        assert (counts / n - exact).abs().max() < 5 * (0.25 / n) ** 0.5, name
    for position in range(steps):
        assert _homogeneity_p_value(off[position], on[position]) > 1e-6, f"position {position}"


def test_the_unseeded_streams_are_put_back_as_they_were(monkeypatch):
    """sampling_streams / restore_sampling_streams (the first-request warmup's guard): the draws after a restore are
    the draws after the save, for torch's generators and the Triton kernels' own unseeded one, and a Triton generator
    made in between is dropped."""
    triton_sampling = pytest.importorskip("freetoken.kernel.triton.sampling")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cpu = torch.device("cpu")
    triton_sampling._gen_u(4, cpu, None, None)
    saved = S.sampling_streams()
    want = (torch.rand(3), triton_sampling._gen_u(4, cpu, None, None))
    # a generator first made after the save, as for another device
    triton_sampling._UGEN[("other", False)] = torch.Generator()
    S.restore_sampling_streams(saved)
    got = (torch.rand(3), triton_sampling._gen_u(4, cpu, None, None))
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    assert ("other", False) not in triton_sampling._UGEN
