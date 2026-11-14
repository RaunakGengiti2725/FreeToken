"""A token budget on a response's thinking span, with a notice before it runs out.

The scheduler keeps one ReasoningTracker per request with a budget. The tracker reads the committed tokens on the host and, between steps, names the token the next step must commit in place of a sampled one: the notice, then the close tag.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# forced once into the thinking span when WARN_AT of the budget is used
WARNING_TEXT = "\n(Reasoning budget almost used; conclude.)\n"
WARN_AT = 0.8
# the notice is skipped unless this many times its length is left after it
WARN_ROOM = 2
# a template that opens thinking itself does so within the prompt's last tokens
OPEN_TAIL = 8


@dataclass(frozen=True)
class ThinkTokens:
    """The token ids of a model's thinking span, resolved once per server."""

    open: Tuple[int, ...]
    close: Tuple[int, ...]
    # single tokens that end thinking without a close tag (the tool-call opener)
    exits: frozenset
    warning: Tuple[int, ...]

    @classmethod
    def resolve(cls, tokenizer, tags: Tuple[str, str, str | None] | None) -> ThinkTokens:
        """From the reasoning parser's (open tag, close tag, tool marker); raises ValueError when there is no tag pair to track."""
        if tags is None:
            raise ValueError("a reasoning budget needs a reasoning parser with an open and a close tag")

        def encode(text: str) -> Tuple[int, ...]:
            return tuple(int(t) for t in tokenizer.encode(text, add_special_tokens=False))

        open_tag, close_tag, tool_marker = tags
        open_ids, close_ids = encode(open_tag), encode(close_tag)
        if not open_ids or not close_ids:
            raise ValueError(f"the tokenizer gives no tokens for the thinking tags {open_tag!r} / {close_tag!r}")
        marker = encode(tool_marker) if tool_marker else ()
        return cls(open_ids, close_ids, frozenset(marker) if len(marker) == 1 else frozenset(), encode(WARNING_TEXT))

    @property
    def void(self) -> frozenset:
        """Tokens that may close the span or start its close tag: a forced token planned for later in the same step is dropped after one."""
        return self.exits | frozenset(self.close)


def _ends_with(tokens: Sequence[int], tail: Tuple[int, ...]) -> bool:
    n = len(tail)
    return len(tokens) >= n and tuple(tokens)[-n:] == tail


def starts_open(prompt_tail: Sequence[int], tokens: ThinkTokens) -> bool:
    """Whether the prompt leaves thinking open: its last tokens hold the open tag with no close tag after it."""
    tail = list(prompt_tail)[-OPEN_TAIL:]
    for end in range(len(tail), 0, -1):
        if _ends_with(tail[:end], tokens.close):
            return False
        if _ends_with(tail[:end], tokens.open):
            return True
    return False


class ReasoningTracker:
    """One request's thinking span as the committed tokens show it.

    ``plan`` is called before a step with every earlier token observed, ``observe`` after it with the tokens it committed.
    """

    def __init__(self, tokens: ThinkTokens, prompt_tail: Sequence[int], budget: int) -> None:
        self.tokens = tokens
        self._void = tokens.void
        self.budget = budget
        warn_at = int(budget * WARN_AT)
        fits = budget - warn_at >= WARN_ROOM * len(tokens.warning)
        self.warn_at = warn_at if tokens.warning and fits else 0
        self.warned = False
        self.thinking = starts_open(prompt_tail, tokens)
        # tokens generated inside the thinking span, the forced notice included, the close tag not
        self.count = 0
        # whether the budget closed the span, not the model
        self.closed = False
        self._recent: deque = deque(maxlen=max(len(tokens.open), len(tokens.close)))
        # the rest of the notice or close tag being forced, one token per step
        self._queue: Tuple[int, ...] = ()
        self._plan: Tuple[int, int, str, Tuple[int, ...]] | None = None

    def _close_started(self) -> int:
        """How many tokens of a close tag the stream ends in, short of the whole tag."""
        close, recent = self.tokens.close, tuple(self._recent)
        return next((n for n in range(len(close) - 1, 0, -1) if recent[-n:] == close[:n]), 0)

    def _next(self) -> Tuple[int, int, str, Tuple[int, ...]] | None:
        """(tokens the model still samples first, the forced token, why, the forced tokens after it) of the next forced token."""
        if self._queue:
            return 0, self._queue[0], "rest", self._queue[1:]
        if not self.thinking:
            return None
        tags = self.tokens
        # a forced close tag continues the one the model has started
        started = self._close_started()
        used = self.count - started
        if self.warn_at and not self.warned and not started and used + len(tags.warning) < self.budget:
            return max(self.warn_at - used, 0), tags.warning[0], "warning", tags.warning[1:]
        free = max(self.budget - used, 0)
        if started and free:
            # it may be closing by itself; what to force is known once the tag completes or breaks off
            return None
        return free, tags.close[started], "close", tags.close[started + 1 :]

    def armed(self) -> bool:
        """Whether the token after one still in flight may be forced: the step that samples it must then wait for the one in flight."""
        nxt = self._next()
        return nxt is not None and nxt[0] <= 1

    def plan(self, width: int) -> Tuple[int, int] | None:
        """(index, token): the next step, which commits up to ``width`` tokens, must commit ``token`` as its ``index``-th; None when it forces nothing."""
        nxt = self._next()
        self._plan = nxt if nxt is not None and nxt[0] < width else None
        return self._plan[:2] if self._plan is not None else None

    def observe(self, tokens: List[int]) -> None:
        """Take the tokens a step committed, in order."""
        plan, self._plan = self._plan, None
        forced_at = -1
        if plan is not None:
            index, token = plan[:2]
            # the rule ForcedTokens applies on the device, read back from the tokens it produced
            if len(tokens) > index and tokens[index] == token and self._void.isdisjoint(tokens[:index]):
                forced_at = index
        for j, token in enumerate(tokens):
            if j == forced_at:
                self._forced(plan[2], plan[3])
            self._step(token)

    def _forced(self, why: str, rest: Tuple[int, ...]) -> None:
        self._queue = rest
        if why == "warning":
            self.warned = True
        elif why == "close":
            self.closed = True

    def _step(self, token: int) -> None:
        self._recent.append(token)
        tags = self.tokens
        if not self.thinking:
            self.thinking = _ends_with(self._recent, tags.open)
        elif _ends_with(self._recent, tags.close):
            self.thinking = False
            # the tag's earlier tokens were counted before the tag was complete
            self.count = max(self.count - (len(tags.close) - 1), 0)
        elif token in tags.exits:
            self.thinking = False
        else:
            self.count += 1

    def usage(self) -> dict:
        """What the terminal reply reports."""
        usage: dict = {"reasoning_tokens": self.count}
        if self.closed:
            usage["closed_by"] = "budget"
        return usage
