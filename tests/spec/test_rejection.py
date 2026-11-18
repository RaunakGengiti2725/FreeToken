"""Rejection sampler and verify-row sampling: greedy prefix match, per-row sampling params and per-row penalties."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 8


def _sampler(device, vocab=VOCAB):
    from freetoken.engine.sample import Sampler

    return Sampler(device=device, vocab_size=vocab)


@cuda
def test_greedy_prefix_match_exact():
    from freetoken.spec.rejection import reject
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    k = 3
    # Target argmax sequence: [3, 1, 2, 0] over 4 positions.
    logits = torch.full((k + 1, VOCAB), -10.0, device=device)
    for pos, t in enumerate([3, 1, 2, 0]):
        logits[pos, t] = 10.0
    args = BatchSamplingArgs(temperatures=None)
    sampler = _sampler(device)

    cases = [
        ([3, 1, 2], 3, [3, 1, 2, 0]),   # all accepted + bonus
        ([3, 1, 7], 2, [3, 1, 2]),      # mismatch at pos 2 -> recovery = target argmax
        ([7, 1, 2], 0, [3]),            # first draft rejected
    ]
    for draft, want_emitted, want_tokens in cases:
        d = torch.tensor([draft], device=device)
        out, emitted = reject(logits, d, args, sampler)
        assert int(emitted[0]) == want_emitted
        assert out[0, : want_emitted + 1].tolist() == want_tokens


def test_a_forced_token_ends_the_accepted_run_at_its_row():
    """A forced token is committed like the target's own token at a rejected row: the rows before it stay, the drafts after it are dropped, and it is skipped when its row was computed on a rejected draft or a void token precedes it in the step."""
    from freetoken.engine.sample import BatchSamplingArgs, ForcedTokens
    from freetoken.spec.rejection import reject

    device = torch.device("cpu")
    k, forced, void = 3, 7, 6
    # the target's argmax after the pending token and after each draft
    target = [3, 1, 2, 0]
    logits = torch.full((k + 1, VOCAB), -10.0)
    for pos, t in enumerate(target):
        logits[pos, t] = 10.0
    cases = [
        # drafts, forced index, committed tokens
        ([3, 1, 2], 0, [forced]),            # every draft right, cut at the first row
        ([3, 1, 2], 2, [3, 1, forced]),
        ([3, 1, 2], 3, [3, 1, 2, forced]),   # the bonus row
        ([3, 5, 2], 1, [3, forced]),         # at the first rejected row: replaces the recovered token
        ([3, 5, 2], 2, [3, 1]),              # past it: that row read the rejected draft
        ([3, 1, 2], -1, [3, 1, 2, 0]),       # no forced token
    ]
    for drafts, index, want in cases:
        out, emitted = reject(logits, torch.tensor([drafts]), BatchSamplingArgs(temperatures=None), _sampler(device))
        plan = ForcedTokens(torch.tensor([index]), torch.tensor([forced]), torch.tensor([void]))
        out, emitted = plan.cut(out, emitted)
        assert out[0, : int(emitted) + 1].tolist() == want, (drafts, index)

    # a void token (the model's own close tag) before the forced row keeps the step as sampled
    logits[1] = -10.0
    logits[1, void] = 10.0
    out, emitted = reject(logits, torch.tensor([[3, void, 2]]), BatchSamplingArgs(temperatures=None), _sampler(device))
    for index, want in ((2, [3, void, 2, 0]), (1, [3, forced])):
        cut_out, cut_emitted = ForcedTokens(torch.tensor([index]), torch.tensor([forced]), torch.tensor([void])).cut(out, emitted)
        assert cut_out[0, : int(cut_emitted) + 1].tolist() == want, index


@cuda
def test_each_verify_row_counts_the_drafts_before_it_as_committed():
    """Row j penalizes the committed counts plus drafts d1..dj, frequency per occurrence and presence once per token; the counts stay as they were."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(1)
    vocab, presence, frequency = 32, 0.7, 0.4
    drafts = torch.tensor([[5, 9, 5]], device=device)  # d3 repeats d1
    counts = torch.zeros(vocab, dtype=torch.int32, device=device)
    counts[9] = 2  # already committed, so presence must not apply to it again
    logits = torch.randn(drafts.shape[1] + 1, vocab, device=device)
    args = BatchSamplingArgs(temperatures=None, penalties=[(0, counts, presence, frequency)])

    sampler = _sampler(device, vocab)
    tokens = sampler.sample_verify(logits, args, drafts)
    sampled = BatchSamplingArgs(temperatures=torch.ones(1, device=device), penalties=args.penalties)
    probs = sampler.probs(logits, sampled, drafts)

    for j in range(drafts.shape[1] + 1):
        seen = counts.clone()
        for d in drafts[0, :j].tolist():
            seen[d] += 1
        expected = logits[j] - frequency * seen - presence * (seen > 0)
        assert int(tokens[j]) == int(expected.argmax()), f"row {j}"
        # the verify with draft probs reads the same penalized rows
        assert torch.allclose(probs[j], torch.softmax(expected, dim=-1), atol=1e-6), f"row {j}"
    assert counts[9] == 2 and counts.sum() == 2


@cuda
def test_each_verify_row_samples_with_its_requests_params():
    """Request i's sampling params cover its k+1 verify rows: a top_k=1 request and a greedy one give the argmax on every row, a sampled one does not; the draft-probs verify reads each row's softmax at its request's temperature, renormalized over its top-k and then its top-p."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, k, trials, bs = 32, 2, 16, 4
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.3, 1.0, 0.7], device=device),
        top_k=torch.tensor([1, vocab, vocab, 6], dtype=torch.int32, device=device),
        top_p=torch.tensor([1.0, 0.9, 1.0, 0.8], device=device),
        greedy_mask=torch.tensor([False, False, True, False], device=device),
    )
    drafts = torch.zeros(bs, k, dtype=torch.int64, device=device)
    off_argmax = []
    for _ in range(trials):
        logits = 0.1 * torch.randn(bs * (k + 1), vocab, device=device)
        tokens = _sampler(device, vocab).sample_verify(logits, args, drafts)
        off_argmax.append((tokens != logits.argmax(dim=-1)).view(bs, k + 1))
    off_argmax = torch.stack(off_argmax)
    assert not off_argmax[:, 0].any() and not off_argmax[:, 2].any()
    assert off_argmax[:, 1].any(dim=0).all()

    logits = torch.randn(bs * (k + 1), vocab, device=device)
    probs = _sampler(device, vocab).probs(logits, args, drafts)
    for row, i in enumerate(torch.arange(bs).repeat_interleave(k + 1).tolist()):
        want = torch.softmax(logits[row] / args.temperatures[i], dim=-1)
        want = torch.where(want >= want.topk(int(args.top_k[i])).values[-1], want, 0.0)
        # top-p keeps the smallest best-first set whose mass reaches top_p
        p_sorted, order = (want / want.sum()).sort(descending=True)
        p_sorted = torch.where(p_sorted.cumsum(0) - p_sorted < args.top_p[i], p_sorted, 0.0)
        want = torch.zeros_like(want).scatter(0, order, p_sorted / p_sorted.sum())
        if args.greedy_mask[i]:
            want = torch.nn.functional.one_hot(logits[row].argmax(), vocab).float()
        torch.testing.assert_close(probs[row], want, atol=1e-6, rtol=1e-5, msg=f"row {row}")


@cuda
def test_reject_with_draft_probs_commits_the_target_distribution():
    """Drafts drawn from a q over K candidates per slot, in a batch that mixes rows trusting q, rows whose q is unusable and greedy rows: the token committed at each position still follows the target p on both sampled kinds, and a greedy row commits the argmax prefix (Monte Carlo, k=2, V=6, K=3)."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.spec.rejection import reject_with_draft_probs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, K, k, n = 6, 3, 2, 6_000_000
    p = torch.softmax(torch.randn(k + 1, vocab, device=device) * 1.5, dim=-1)
    cand = torch.tensor([[0, 2, 5], [1, 2, 3]], device=device)
    q = torch.softmax(torch.randn(k, K, device=device), dim=-1)
    picks = torch.stack([torch.multinomial(q[j], n, replacement=True) for j in range(k)], dim=1)
    drafts = cand.gather(1, picks.T).T
    # rows cycle through: q trusted, q unusable (a NaN or a total of 2), greedy
    kind = torch.arange(n, device=device) % 3
    q_rows = q.expand(n, k, K).clone()
    unusable = (kind == 1).nonzero().squeeze(1)
    q_rows[unusable[::2], 0, 0] = float("nan")
    q_rows[unusable[1::2], 0] *= 2
    # softmax(log p) at T=1 hands the verify exactly p on every row
    logits = p.log().repeat(n, 1)
    args = BatchSamplingArgs(temperatures=torch.ones(n, device=device), greedy_mask=kind == 2)
    out, emitted = reject_with_draft_probs(logits, drafts, cand.expand(n, k, K).contiguous(), q_rows, args, _sampler(device, vocab))

    def marginal(tokens):
        return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()

    for rows in (kind == 0, kind == 1):
        rows_out, rows_emitted = out[rows], emitted[rows]
        accepted_first = rows_emitted >= 1
        assert int(accepted_first.sum()) > 100_000
        assert (marginal(rows_out[:, 0]) - p[0]).abs().max() < 5e-3
        assert (marginal(rows_out[accepted_first, 1]) - p[1]).abs().max() < 5e-3
    # both kinds draw the bonus token from p itself; a row that does not trust q rarely accepts every draft
    accepted_all = (kind < 2) & (emitted == k)
    assert int(accepted_all.sum()) > 100_000
    assert (marginal(out[accepted_all, k]) - p[k]).abs().max() < 5e-3
    greedy = kind == 2
    argmax = logits[: k + 1].argmax(dim=-1)
    assert torch.equal(emitted[greedy], (drafts[greedy] == argmax[:k]).cumprod(dim=1).sum(dim=1))
    committed = torch.arange(k + 1, device=device) <= emitted[greedy].unsqueeze(1)
    assert torch.equal(out[greedy][committed], argmax.expand(int(greedy.sum()), -1)[committed])


def test_a_seeded_draw_is_keyed_by_the_token_position_not_by_the_step():
    """Positions 5..10 of one request reached three ways: a decode step each, verify steps of 4 rows, one verify step of 6 rows. Each position must get one key whatever the split, the batch around it or the phase, and a key of its own."""
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq

    sampler = _sampler(torch.device("cpu"))

    def req(device_len, cls=Req, **params):
        return cls(input_ids=torch.zeros(device_len, dtype=torch.int32), table_idx=0, cached_len=device_len - 1, output_len=64,
                   uid=0, sampling_params=SamplingParams(**{"temperature": 1.0, **params}), cache_handle=None)

    def keys(reqs, phase, rows=None):
        return sampler.prepare(Batch(reqs=reqs, phase=phase, rows_per_req=rows)).keys

    decode = [keys([req(n, seed=7)], "decode")[0] for n in range(5, 11)]
    assert len(set(decode)) == 6
    assert keys([req(5, seed=7)], "verify", 4) + keys([req(9, seed=7)], "verify", 4)[:2] == decode
    assert keys([req(5, seed=7)], "verify", 6) == decode
    # a prompt of 5 tokens draws its first token at position 5 too
    assert keys([req(5, seed=7)], "prefill") == decode[:1]
    # beside an unseeded request the rows keep their keys, and the neighbor gets none
    assert keys([req(3), req(5, seed=7)], "verify", 4) == [None] * 4 + decode[:4]
    assert not set(keys([req(5, seed=8)], "verify", 6)) & set(decode)
    # no seed, a greedy request, and a prompt chunk whose token nobody keeps: the batch draws as it does without the field
    for other in (req(5), req(5, seed=7, temperature=0.0), req(5, ChunkedReq, seed=7)):
        assert keys([other, req(5)], "prefill") is None


def test_a_seeded_request_draws_the_same_tokens_whatever_the_server_drew_before(monkeypatch):
    """Positions 5..10 of a seeded request on servers whose shared stream stands elsewhere, in decode steps, in verify steps of 4 and of 6 rows, and beside another request. The kernels are a stand-in that reads seed and offset as both backends do (one seed a call; the shared stream without one), so this checks which draws the sampler asks for; the CUDA test below checks the kernels."""
    from types import SimpleNamespace

    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.engine import sample

    def sampling_from_probs(probs, seed=None, offset=None):
        generator = None if seed is None else torch.Generator().manual_seed(seed + offset)
        uniforms = torch.rand(probs.shape[0], 1, generator=generator)
        return torch.searchsorted(probs.cumsum(-1), uniforms).squeeze(1).clamp_(max=probs.shape[1] - 1)

    kernels = SimpleNamespace(
        softmax=lambda logits, temperatures, enable_pdl=False: torch.softmax(logits / temperatures.unsqueeze(1), dim=-1),
        sampling_from_probs=sampling_from_probs,
    )
    monkeypatch.setattr(sample, "sampling_module", lambda: kernels)
    monkeypatch.setattr(sample, "is_sm90_supported", lambda: False)
    vocab = 64
    sampler = _sampler(torch.device("cpu"), vocab)
    table = torch.randn(16, vocab, generator=torch.Generator().manual_seed(3))

    def req(device_len, seed=None):
        return Req(input_ids=torch.zeros(device_len, dtype=torch.int32), table_idx=0, cached_len=device_len - 1, output_len=64,
                   uid=0, sampling_params=SamplingParams(temperature=1.0, seed=seed), cache_handle=None)

    def lifetime(history, width, seed=7, neighbor=False):
        """The request's tokens at positions 5..10 on a server whose shared stream starts at ``history``, ``width`` rows a step (1: decode steps)."""
        torch.manual_seed(history)
        tokens = []
        for start in range(5, 11, width):
            reqs = [req(3), req(start, seed)] if neighbor else [req(start, seed)]
            logits = torch.cat([table[r.device_len : r.device_len + width] for r in reqs])
            batch = Batch(reqs=reqs, phase="decode") if width == 1 else Batch(reqs=reqs, phase="verify", rows_per_req=width)
            # a decode step goes through sample_verify with no drafts: Sampler.sample opens an NVTX range, which a CPU build of torch lacks
            out = sampler.sample_verify(logits, sampler.prepare(batch), torch.zeros(len(reqs), width - 1, dtype=torch.int64))
            tokens += out[-width:].tolist()
        return tokens[:6]

    first = lifetime(0, 1)
    assert lifetime(1, 1) == first
    assert lifetime(2, 4) == first
    assert lifetime(3, 6, neighbor=True) == first
    assert lifetime(0, 1, seed=8) != first
    # without a seed the same request follows the shared stream, as it does today
    assert lifetime(0, 1, seed=None) != lifetime(1, 1, seed=None)


@cuda
@pytest.mark.parametrize("params", [{}, {"top_k": 8, "top_p": 0.9}, {"min_p": 0.05}])
def test_a_seeded_request_draws_the_same_token_at_a_position_in_any_step(params):
    """The tokens a seed gives at positions 5..10, drawn a decode step at a time, must come back from verify steps of 4 and of 6 rows and from a step shared with another request; row n of ``table`` stands for the logits at position n."""
    from freetoken.core import Batch, Req, SamplingParams

    device = torch.device("cuda")
    torch.manual_seed(3)
    vocab = 64
    sampler = _sampler(device, vocab)
    table = torch.randn(16, vocab, device=device)

    def req(device_len, seed=None):
        return Req(input_ids=torch.zeros(device_len, dtype=torch.int32), table_idx=0, cached_len=device_len - 1, output_len=64,
                   uid=0, sampling_params=SamplingParams(temperature=1.0, seed=seed, **params), cache_handle=None)

    def verify(start, width, seed=7, neighbor=False, batch=1, at=0):
        """The seeded request's ``width`` rows from one verify step at ``start``, after an unseeded request's rows when
        ``neighbor``; or as request ``at`` of ``batch`` requests, the others seeded and unseeded in turn."""
        reqs = [req(3), req(start, seed)] if neighbor else [req(start, seed)]
        if batch > 1:
            others = [req(3 + i % 4, None if i % 2 else 100 + i) for i in range(batch - 1)]
            reqs = others[:at] + [req(start, seed)] + others[at:]
        args = sampler.prepare(Batch(reqs=reqs, phase="verify", rows_per_req=width))
        logits = torch.cat([table[r.device_len : r.device_len + width] for r in reqs])
        drafts = torch.zeros(len(reqs), width - 1, dtype=torch.int64, device=device)
        mine = at if batch > 1 else len(reqs) - 1
        return sampler.sample_verify(logits, args, drafts)[mine * width : (mine + 1) * width].tolist()

    decode = []
    for n in range(5, 11):
        args = sampler.prepare(Batch(reqs=[req(n, seed=7)], phase="decode"))
        decode.append(int(sampler.sample(table[n : n + 1], args)))
    assert verify(5, 4) + verify(9, 4)[:2] == decode
    assert verify(5, 6) == decode
    assert verify(5, 6, neighbor=True) == decode
    assert verify(5, 6, seed=8) != decode
    for batch in (2, 8, 16):
        for at in sorted({0, batch // 2, batch - 1}):
            assert verify(5, 6, batch=batch, at=at) == decode, (batch, at)


def test_the_keyed_draw_samples_the_rows_distribution_and_never_a_token_it_cut():
    """keyed_draw's inverse CDF: over many uniforms each row's tokens come up as often as its probabilities say, a
    token at probability 0 (the tail top-k and top-p cut) never does, and the ends of [0, 1) land on tokens it keeps."""
    from freetoken.engine.sample import key_uniform, keyed_draw

    g = torch.Generator().manual_seed(5)
    probs = torch.rand(3, 32, generator=g)
    probs[0, 20:] = 0.0  # a cut tail
    probs[1, :5] = 0.0
    probs[2, ::2] = 0.0
    probs /= probs.sum(dim=1, keepdim=True)
    n = 200_000
    uniforms = torch.tensor([key_uniform(key) for key in range(3 * n)], dtype=torch.float32).view(n, 3)
    tokens = torch.stack([keyed_draw(probs, u) for u in uniforms[:2000]])
    tokens = torch.cat([tokens, keyed_draw(probs.repeat(n - 2000, 1), uniforms[2000:].reshape(-1)).view(-1, 3)])
    for row in range(3):
        seen = torch.bincount(tokens[:, row], minlength=32).float() / n
        assert (seen - probs[row]).abs().max() < 4e-3
        assert not seen[probs[row] == 0].any()
    # the uniforms' ends; a total that is not 1 (min-p keeps its tokens unscaled) is scaled into
    ends = torch.tensor([0.0, 1.0 - 2.0**-24, 1.0 - 2.0**-24])
    assert keyed_draw(probs, ends).tolist() == [0, 31, 31] and keyed_draw(probs * 0.3, ends).tolist() == [0, 31, 31]
    assert keyed_draw(probs[:1], torch.tensor([1.0 - 2.0**-24])).tolist() == [19]
    assert keyed_draw(probs[1:2], torch.tensor([0.0])).tolist() == [5]


@pytest.mark.parametrize("per_row,neighbors", [
    (True, {}), (False, {}),
    # the requests beside it turn a filter on that it leaves off
    (True, {"top_k": 20}), (True, {"top_p": 0.9}), (True, {"min_p": 0.05}),
    (True, {"top_k": 20, "top_p": 0.9, "min_p": 0.05}),
], ids=["per-row", "batched", "per-row-top-k", "per-row-top-p", "per-row-min-p", "per-row-all"])
def test_a_seeded_row_draws_alone_what_it_draws_in_a_batch_and_leaves_the_unseeded_rows_alone(monkeypatch, per_row, neighbors):
    """One seeded request alone and as request 0, a middle one and the last of 2, 8 and 16, beside seeded and unseeded
    requests, in a decode step and a verify step of 3 rows: the same tokens, by the default per-row draw and by
    FREETOKEN_SEEDED_DRAW=batched, also when the requests beside it turn on filters it leaves off. The unseeded rows
    draw from the shared stream exactly as in the same step without a seed, and the stream stands where it would.
    Stand-in kernels as above, which run a row alone as in a batch; each draw entry point mixes its own salt into the
    seed, as the real kernels run another algorithm per entry point. The CUDA test checks the arithmetic of the kernels."""
    from types import SimpleNamespace

    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.engine import sample

    assert sample._SEEDED_DRAW_PER_ROW  # the default
    monkeypatch.setattr(sample, "_SEEDED_DRAW_PER_ROW", per_row)

    def renorm(probs, thr):
        kept = torch.where(probs >= thr, probs, 0.0)
        return kept / kept.sum(dim=1, keepdim=True)

    def top_k_renorm_probs(probs, top_k):
        return renorm(probs, probs.sort(dim=1, descending=True).values.gather(1, top_k.long().unsqueeze(1) - 1))

    def top_p_renorm_probs(probs, top_p):
        ranked = probs.sort(dim=1, descending=True).values
        inside = ranked.cumsum(dim=1) - ranked < top_p.unsqueeze(1)
        return renorm(probs, torch.where(inside, ranked, 2.0).amin(dim=1, keepdim=True))

    def draw(salt):
        def sampling(probs, seed=None, offset=None):
            generator = None if seed is None else torch.Generator().manual_seed(seed + offset + salt)
            uniforms = torch.rand(probs.shape[0], 1, generator=generator)
            return torch.searchsorted(probs.cumsum(-1), uniforms).squeeze(1).clamp_(max=probs.shape[1] - 1).to(torch.int32)
        return sampling

    kernels = SimpleNamespace(
        softmax=lambda logits, temperatures, enable_pdl=False: torch.softmax(logits / temperatures.unsqueeze(1), dim=-1),
        top_k_renorm_probs=top_k_renorm_probs,
        top_p_renorm_probs=top_p_renorm_probs,
        sampling_from_probs=draw(0),
        top_k_sampling_from_probs=lambda probs, k, **rng: draw(1)(top_k_renorm_probs(probs, k), **rng),
        top_p_sampling_from_probs=lambda probs, p, **rng: draw(2)(top_p_renorm_probs(probs, p), **rng),
        top_k_top_p_sampling_from_probs=lambda probs, k, p, **rng: draw(3)(top_p_renorm_probs(top_k_renorm_probs(probs, k), p), **rng),
        min_p_sampling_from_probs=lambda probs, m, **rng: draw(4)(renorm(probs, m.unsqueeze(1) * probs.amax(1, keepdim=True)), **rng),
    )
    monkeypatch.setattr(sample, "sampling_module", lambda: kernels)
    monkeypatch.setattr(sample, "is_sm90_supported", lambda: False)
    vocab = 512
    sampler = sample.Sampler(device=torch.device("cpu"), vocab_size=vocab)
    g = torch.Generator().manual_seed(11)
    mine, others = torch.randn(3, vocab, generator=g), torch.randn(15, 3, vocab, generator=g)

    def req(seed, **params):
        return Req(input_ids=torch.zeros(40, dtype=torch.int32), table_idx=0, cached_len=39, output_len=64, uid=0,
                   sampling_params=SamplingParams(temperature=0.8, seed=seed, **params), cache_handle=None)

    def step(batch, at, width, seeded=True):
        reqs = [req(None if i % 2 or not seeded else 100 + i, **neighbors) for i in range(batch - 1)]
        reqs = reqs[:at] + [req(7 if seeded else None)] + reqs[at:]
        logits = torch.cat([*others[:at, :width], mine[:width], *others[at : batch - 1, :width]])
        if width == 1:
            args = sampler.prepare(Batch(reqs=reqs, phase="decode"))
        else:
            args = sampler.prepare(Batch(reqs=reqs, phase="verify", rows_per_req=width))
        torch.manual_seed(1)
        # a decode step goes through sample_verify with no drafts: Sampler.sample opens an NVTX range, which a CPU build of torch lacks
        tokens = sampler.sample_verify(logits, args, torch.zeros(batch, width - 1, dtype=torch.int64)).view(batch, width)
        return tokens, torch.rand(1), [i for i, r in enumerate(reqs) if r.sampling_params.seed is None]

    for width in (1, 3):
        alone = step(1, 0, width)[0][0]
        for batch in (2, 8, 16):
            for at in sorted({0, batch // 2, batch - 1}):
                tokens, after, unseeded = step(batch, at, width)
                assert torch.equal(tokens[at], alone), (width, batch, at)
                plain, plain_after, _ = step(batch, at, width, seeded=False)
                assert torch.equal(tokens[unseeded], plain[unseeded]) and torch.equal(after, plain_after)


@cuda
@pytest.mark.xfail(reason=(
    "FREETOKEN_SEEDED_DRAW=batched is not batch-independent: on an RTX 5090 a row's top-k probabilities in a batch of "
    "keyed rows were not its bits alone (box B, 90507c0); the default per-row draw is not this path"
), strict=False)
def test_a_seeded_rows_probabilities_and_token_have_the_same_bits_alone_and_in_a_batch():
    """At Qwen3.6's vocabulary on the GPU: a seeded row's filtered probabilities, their running sums and its token,
    alone and as a row of 2, 8 and 16 rows, for each filter keyed_probs applies (FREETOKEN_SEEDED_DRAW=batched)."""
    from freetoken.engine.sample import key_uniform, keyed_draw, keyed_probs

    device = torch.device("cuda")
    vocab = 248320
    g = torch.Generator(device=device).manual_seed(4)
    logits = torch.randn(16, vocab, device=device, generator=g) * 4
    u = torch.tensor([key_uniform(k) for k in range(16)], dtype=torch.float32, device=device)
    filters = [
        dict(top_k=None, top_p=None, min_p=None),
        dict(top_k=torch.full((16,), 20, dtype=torch.int32, device=device), top_p=torch.full((16,), 0.95, device=device), min_p=None),
        dict(top_k=None, top_p=None, min_p=torch.full((16,), 0.05, device=device)),
    ]
    for f in filters:
        def probs(rows):
            return keyed_probs(logits[rows], torch.full((len(rows),), 0.7, device=device),
                               *(None if f[n] is None else f[n][rows] for n in ("top_k", "top_p", "min_p")))

        for row in (0, 7, 15):
            p1 = probs([row])
            t1 = keyed_draw(p1, u[row : row + 1])
            for batch in (2, 8, 16):
                rows = sorted({row, *range(batch)})[:batch] if row < batch else [*range(batch - 1), row]
                i = rows.index(row)
                pb = probs(rows)
                assert torch.equal(pb[i], p1[0]), (f, row, batch)
                assert torch.equal(pb.cumsum(-1)[i], p1.expand(2, -1).cumsum(-1)[0]), (f, row, batch)
                assert torch.equal(keyed_draw(pb, u[rows])[i], t1[0]), (f, row, batch)


@cuda
@pytest.mark.xfail(reason=(
    "seeded tokens are batch-independent only when every keyed row shares the same filters: on an RTX 5090 with the "
    "default per-row draw, a request with top-k 20 / top-p 0.95 drew 176152 in the batch against 201795 alone, and one "
    "with temperature 1.3 / top-p 0.9 differed from position 0 (box B, 1c946ea)"
), strict=False)
@pytest.mark.parametrize("seeded,width", [(8, 6), (16, 6), (32, 4)])
def test_a_seeded_requests_verify_tokens_are_the_same_alone_and_among_many_keyed_rows(seeded, width):
    """48, 96 and 128 keyed rows at Qwen3.6's vocabulary through Sampler: seeded requests with and without top-k,
    top-p and min-p, at their own temperatures and positions, beside a greedy and an unseeded request, in a shuffled
    order. Each seeded request's verify tokens must be the ones it draws in a batch of its own."""
    from freetoken.core import Batch, Req, SamplingParams

    device = torch.device("cuda")
    vocab = 248320
    sampler = _sampler(device, vocab)
    g = torch.Generator(device=device).manual_seed(9)
    filters = [{}, {"top_k": 20, "top_p": 0.95}, {"top_p": 0.9}, {"top_k": 50}, {"min_p": 0.05}]

    def req(i, seed, **params):
        n = 40 + 7 * i
        return Req(input_ids=torch.zeros(n, dtype=torch.int32), table_idx=0, cached_len=n - 1, output_len=64, uid=i,
                   sampling_params=SamplingParams(seed=seed, **params), cache_handle=None)

    reqs = [req(i, 1000 + i, temperature=(0.7, 1.0, 1.3)[i % 3], **filters[i % len(filters)]) for i in range(seeded)]
    reqs += [req(seeded, None, temperature=0.0), req(seeded + 1, None, temperature=1.0, top_k=40)]
    logits = {r.uid: torch.randn(width, vocab, device=device, generator=g) * 4 for r in reqs}

    def step(batch_reqs):
        args = sampler.prepare(Batch(reqs=batch_reqs, phase="verify", rows_per_req=width))
        drafts = torch.zeros(len(batch_reqs), width - 1, dtype=torch.int64, device=device)
        out = sampler.sample_verify(torch.cat([logits[r.uid] for r in batch_reqs]), args, drafts).view(len(batch_reqs), width)
        return {r.uid: out[i].tolist() for i, r in enumerate(batch_reqs)}

    alone = {r.uid: step([r])[r.uid] for r in reqs[:seeded]}
    order = torch.randperm(len(reqs), generator=torch.Generator().manual_seed(seeded)).tolist()
    shuffled = [reqs[i] for i in order]
    together = step(shuffled)
    for r in reqs[:seeded]:
        assert together[r.uid] == alone[r.uid], (r.uid, r.sampling_params)
    greedy = reqs[seeded]
    assert together[greedy.uid] == logits[greedy.uid].argmax(dim=-1).tolist()
    # and the same again in another order: the keyed rows' places move
    together = step(shuffled[::-1])
    assert all(together[r.uid] == alone[r.uid] for r in reqs[:seeded])
