"""The reasoning budget, the answer reserve and their notice: what the tracker plans on the host, and what the device applies.

The model here is a function of the tokens so far, so a run can be replayed with any step width and any number of accepted drafts: each step applies the tracker's plan with the same ForcedTokens ops the sampler and the verify use, and the tracker reads back only the tokens the step committed.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from freetoken.engine.sample import ForcedTokens
from freetoken.scheduler import reasoning as R
from freetoken.scheduler.reasoning import ReasoningTracker, ThinkTokens

OPEN, CLOSE, TOOL = 1, 2, 3
NOTICE = (50, 51, 52)
ONE = ThinkTokens(open=(OPEN,), close=(CLOSE,), exits=frozenset({TOOL}), warning=NOTICE)
# a close tag the tokenizer splits in three; its last token also occurs alone
MULTI = ThinkTokens(open=(OPEN,), close=(7, 8, 9), exits=frozenset(), warning=NOTICE)
OPEN_PROMPT = [40, 41, OPEN, 42]


def _closed(history, tokens) -> bool:
    n = len(tokens.close)
    return any(tuple(history[i : i + n]) == tokens.close for i in range(len(history)))


def _model(tokens, close_after=None, think=lambda i: 100 + i):
    """Thinks one token per position, writes the close tag itself after ``close_after`` tokens when set, then answers; it answers after a close tag wherever that came from."""

    def model(history):
        i = len(history)
        if _closed(history, tokens):
            return 900 + i
        if close_after is not None and i >= close_after:
            return tokens.close[i - close_after]
        return think(i)

    return model


def _generate(tracker, model, steps, k=0, accepts=None):
    """Run ``steps`` steps of k drafts each (k=0: plain decode); ``accepts`` gives the drafts accepted per step, all of them when None."""
    history = []
    void = torch.tensor(sorted(tracker.tokens.void)) if tracker is not None else None
    for _ in range(steps):
        plan = tracker.plan(k + 1) if tracker is not None else None
        # row j is the target's token after the pending token and j drafts; the drafts are the model's own continuation
        target = []
        for _ in range(k + 1):
            target.append(model(history + target))
        out = torch.tensor([target])
        emitted = torch.tensor([k if accepts is None else next(accepts)])
        if plan is not None:
            forced = ForcedTokens(torch.tensor([plan[0]]), torch.tensor([plan[1]]), void)
            if k == 0:
                out = forced.replace(out[:, 0]).unsqueeze(1)
            else:
                out, emitted = forced.cut(out, emitted)
        committed = out[0, : int(emitted) + 1].tolist()
        if tracker is not None:
            tracker.observe(committed)
        history += committed
    return history


ACCEPTS = {"all": None, "none": (0,), "mixed": (2, 0, 3, 1), "one": (1,)}


@pytest.mark.parametrize("tokens", [ONE, MULTI], ids=["one-token-close", "three-token-close"])
@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("accepts", list(ACCEPTS))
@pytest.mark.parametrize("budget", [1, 4, 5, 8, 11])
def test_the_close_tag_follows_exactly_budget_thinking_tokens(tokens, k, accepts, budget):
    """Whatever the step width and wherever the drafts are rejected - a budget that ends on a step's last row included - the committed stream is budget thinking tokens, the whole close tag, then the answer sampled as usual."""
    pattern = ACCEPTS[accepts]
    cycle = None if pattern is None else itertools.cycle(min(a, k) for a in pattern)
    tracker = ReasoningTracker(tokens, OPEN_PROMPT, budget)
    history = _generate(tracker, _model(tokens), steps=40, k=k, accepts=cycle)

    end = budget + len(tokens.close)
    assert history[:budget] == [100 + i for i in range(budget)]
    assert tuple(history[budget:end]) == tokens.close
    assert history[end:] == [900 + i for i in range(end, len(history))] and len(history) > end
    assert tracker.usage() == {"reasoning_tokens": budget, "closed_by": "budget"}


@pytest.mark.parametrize("tokens", [ONE, MULTI], ids=["one-token-close", "three-token-close"])
@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("close_after", [0, 3, 5])
def test_a_model_that_closes_within_the_budget_is_left_alone(tokens, k, close_after):
    """A model that writes its own close tag before the budget, in the very step that planned a forced one too, generates what it does with no budget."""
    model = _model(tokens, close_after=close_after)
    tracker = ReasoningTracker(tokens, OPEN_PROMPT, 6)
    assert _generate(tracker, model, steps=12, k=k) == _generate(None, model, steps=12, k=k)
    assert tracker.usage() == {"reasoning_tokens": close_after}


def test_a_token_after_the_model_closed_is_not_taken_for_the_forced_one():
    """The model closes ahead of the planned row of a verify step, so the device forces nothing; the planned token then sampled at that row is the model's own."""
    tracker = ReasoningTracker(ONE, OPEN_PROMPT, 2)
    assert tracker.plan(4) == (2, CLOSE)
    tracker.observe([CLOSE, 900, CLOSE, 901])
    assert tracker.usage() == {"reasoning_tokens": 0}


@pytest.mark.parametrize("k", [0, 3])
def test_the_notice_is_forced_once_at_80_percent_and_counts_as_thinking(k):
    tracker = ReasoningTracker(ONE, OPEN_PROMPT, 100)
    history = _generate(tracker, _model(ONE), steps=120, k=k)
    assert tuple(history[80:83]) == NOTICE
    assert history.index(CLOSE) == 100 and history.count(NOTICE[0]) == 1
    assert history[83:100] == [100 + i for i in range(83, 100)]
    assert tracker.usage() == {"reasoning_tokens": 100, "closed_by": "budget"}


@pytest.mark.parametrize(
    "prompt_tail, model_opens, expect_forced",
    [
        ([40, OPEN, 41], False, True),          # the template opened thinking
        ([40, OPEN, 41, CLOSE, 42], False, False),  # the template opened and closed it: thinking is off
        ([40, 41], False, False),                # no tag: the model answers directly
        ([40, 41], True, True),                  # the model opens thinking itself
        ([OPEN] + [40] * R.OPEN_TAIL, False, False),  # an open tag before the prompt's last OPEN_TAIL tokens is not the template's
    ],
)
def test_the_span_is_read_from_the_prompt_tail_and_the_generated_tags(prompt_tail, model_opens, expect_forced):
    tracker = ReasoningTracker(ONE, prompt_tail, 4)
    script = ([OPEN] if model_opens else []) + [100 + i for i in range(20)]
    history = _generate(tracker, lambda h: 900 + len(h) if CLOSE in h else script[len(h)], steps=12)
    start = 1 if model_opens else 0
    if expect_forced:
        assert history[start + 4] == CLOSE and history.count(CLOSE) == 1
        assert tracker.usage() == {"reasoning_tokens": 4, "closed_by": "budget"}
    else:
        assert CLOSE not in history and tracker.usage() == {"reasoning_tokens": 0}


def test_a_tool_call_opener_ends_the_span_without_a_close_tag():
    """Qwen sometimes goes from thinking straight to <tool_call>; a close tag forced into the call would break it."""
    script = [100, 101, TOOL] + [200 + i for i in range(20)]
    tracker = ReasoningTracker(ONE, OPEN_PROMPT, 5)
    assert _generate(tracker, lambda h: script[len(h)], steps=15) == script[:15]
    assert tracker.usage() == {"reasoning_tokens": 2}


@pytest.mark.parametrize("tokens", [ONE, MULTI], ids=["one-token-close", "three-token-close"])
@pytest.mark.parametrize("budget", [1, 2, 5, 100])
def test_a_step_may_launch_before_the_last_token_drains_unless_the_tracker_is_armed(tokens, budget):
    """Under overlap a decode step launches while the token before it is still in flight, so its plan is made without that token. armed() names the steps that must wait for it: launching every other step early commits the stream a run that always waits commits."""
    model = _model(tokens)
    history, early = _overlapped(ReasoningTracker(tokens, OPEN_PROMPT, budget), model, budget + 20)
    assert history == _generate(ReasoningTracker(tokens, OPEN_PROMPT, budget), model, steps=budget + 20)
    # the answer after the close tag overlaps again
    assert early >= 10


def _overlapped(tracker, model, steps):
    """Plain decode where each step launches before the one before it drains unless the tracker is armed; (history, steps launched early)."""
    void = torch.tensor(sorted(tracker.tokens.void))
    history, in_flight, early = [], None, 0
    for _ in range(steps):
        if in_flight is not None and tracker.armed():
            tracker.observe(in_flight)
            in_flight = None
        early += in_flight is not None
        plan = tracker.plan(1)
        token = torch.tensor([model(history)])
        if plan is not None:
            token = ForcedTokens(torch.tensor([plan[0]]), torch.tensor([plan[1]]), void).replace(token)
        if in_flight is not None:
            # the drain of the step before runs after this launch
            tracker.observe(in_flight)
        in_flight = token.tolist()
        history += in_flight
    return history, early


def _reserve(tokens, max_tokens, reserve, budget=0, prompt=OPEN_PROMPT):
    return ReasoningTracker(tokens, prompt, budget, max_tokens=max_tokens, reserve=reserve)


@pytest.mark.parametrize("tokens", [ONE, MULTI], ids=["one-token-close", "three-token-close"])
@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("accepts", list(ACCEPTS))
def test_the_reserve_forces_the_notice_reserve_tokens_before_max_tokens_and_no_close(tokens, k, accepts):
    """With a reserve and no budget the notice starts at token max_tokens - reserve, once, and the model then thinks on until it closes by itself."""
    pattern = ACCEPTS[accepts]
    cycle = None if pattern is None else itertools.cycle(min(a, k) for a in pattern)
    tracker = _reserve(tokens, max_tokens=100, reserve=20)
    history = _generate(tracker, _model(tokens), steps=110, k=k, accepts=cycle)[:110]
    assert history[:80] == [100 + i for i in range(80)]
    assert tuple(history[80:83]) == NOTICE and history.count(NOTICE[0]) == 1
    assert history[83:] == [100 + i for i in range(83, len(history))] and len(history) >= 100
    assert not set(tokens.close) & set(history)
    assert "closed_by" not in tracker.usage()


@pytest.mark.parametrize("k", [0, 3])
def test_the_reserve_leaves_a_model_that_closes_first_alone(k):
    model = _model(ONE, close_after=30)
    tracker = _reserve(ONE, max_tokens=100, reserve=20)
    assert _generate(tracker, model, steps=60, k=k) == _generate(None, model, steps=60, k=k)
    assert tracker.usage() == {"reasoning_tokens": 30}


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize(
    "budget, max_tokens, reserve, notice_at",
    [
        (100, 60, 20, 40),    # the reserve's notice comes first
        (100, 200, 50, 80),   # the budget's comes first
        (100, 100, 20, 80),   # both at 80
    ],
)
def test_with_a_budget_the_earlier_notice_is_forced_once_and_the_budget_still_closes(k, budget, max_tokens, reserve, notice_at):
    tracker = _reserve(ONE, max_tokens=max_tokens, reserve=reserve, budget=budget)
    history = _generate(tracker, _model(ONE), steps=130, k=k)
    assert history[:notice_at] == [100 + i for i in range(notice_at)]
    assert tuple(history[notice_at : notice_at + 3]) == NOTICE and history.count(NOTICE[0]) == 1
    assert history.index(CLOSE) == budget
    assert tracker.usage() == {"reasoning_tokens": budget, "closed_by": "budget"}


@pytest.mark.parametrize(
    "max_tokens, reserve",
    [
        (100, 0),     # off
        (100, 5),     # less than twice the notice's length after it
        (100, 100),   # the reply is never longer than the reserve
        (100, 150),
    ],
)
def test_a_reserve_without_room_or_point_forces_nothing(max_tokens, reserve):
    model = _model(ONE)
    tracker = _reserve(ONE, max_tokens=max_tokens, reserve=reserve)
    assert _generate(tracker, model, steps=120) == _generate(None, model, steps=120)
    assert tracker.usage() == {"reasoning_tokens": 120}


@pytest.mark.parametrize(
    "budget,reserve,max_tokens,want",
    [
        (None, None, 100, False),
        (0, 0, 100, False),
        (None, 100, 100, False),   # the reply is never longer than the reserve
        (None, 150, 100, False),
        (None, 20, 100, True),
        (50, None, 100, True),
        (50, 150, 100, True),      # the budget is tracked whatever the reserve
    ],
)
def test_only_a_budget_or_a_reserve_shorter_than_max_tokens_is_tracked(budget, reserve, max_tokens, want):
    assert R.tracked(budget, reserve, max_tokens) is want


@pytest.mark.parametrize(
    "budget,reserve,kept",
    [(0, 20, True), (0, 5, False), (50, 5, True), (50, 0, True)],
    ids=["reserve", "reserve-without-room", "budget-and-short-reserve", "budget"],
)
def test_the_scheduler_keeps_no_tracker_for_a_reserve_that_forces_nothing(budget, reserve, kept):
    """A reserve shorter than twice the notice, with no budget, leaves the request untracked, so it reports no reasoning tokens."""
    from types import SimpleNamespace

    from freetoken.core import SamplingParams
    from freetoken.scheduler.scheduler import Scheduler

    sched = SimpleNamespace(_think_tokens=ONE, _reasoning={})
    params = SamplingParams(max_tokens=100)
    params.reasoning_budget, params.reasoning_reserve = budget, reserve
    msg = SimpleNamespace(uid=7, input_ids=torch.tensor(OPEN_PROMPT), sampling_params=params)
    assert Scheduler._track_reasoning(sched, msg) is None
    assert (7 in sched._reasoning) is kept


def test_the_reserve_counts_every_generated_token_when_the_model_opens_thinking_itself():
    """max_tokens counts the reply's tokens, so the open tag the model writes counts toward the notice's place too."""
    script = [OPEN] + [100 + i for i in range(60)]
    tracker = _reserve(ONE, max_tokens=40, reserve=10, prompt=[40, 41])
    history = _generate(tracker, lambda h: script[len(h)] if len(h) < len(script) else 900, steps=40)
    assert history[:30] == script[:30] and tuple(history[30:33]) == NOTICE


def test_a_reserve_of_2048_at_32768_places_the_notice_where_a_budget_of_38400_does():
    """At max_tokens 32,768 the 38,400 budget's notice (80%) falls at 30,720 tokens, 2,048 before the cap: the reserve gives that place without a budget, and no close."""
    assert ReasoningTracker(ONE, OPEN_PROMPT, 38400).warn_at == 30720
    assert _reserve(ONE, max_tokens=32768, reserve=2048).reserve_at == 30720
    # the same at a small scale, step by step: up to the cap the two streams are one
    model = _model(ONE)
    assert _generate(ReasoningTracker(ONE, OPEN_PROMPT, 100), model, steps=90) == _generate(_reserve(ONE, 90, 10), model, steps=90)


@pytest.mark.parametrize("tokens", [ONE, MULTI], ids=["one-token-close", "three-token-close"])
@pytest.mark.parametrize("reserve", [6, 20, 99])
def test_a_reserve_step_may_launch_before_the_last_token_drains_unless_the_tracker_is_armed(tokens, reserve):
    model = _model(tokens, close_after=110)
    history, early = _overlapped(_reserve(tokens, 100, reserve), model, 130)
    assert history == _generate(_reserve(tokens, 100, reserve), model, steps=130)
    assert tuple(history[100 - reserve : 103 - reserve]) == NOTICE
    assert early >= 100
