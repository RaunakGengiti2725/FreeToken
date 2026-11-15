"""The rule that reads a decode A/B of offload against hybrid (``ft bench decode``, moe/benchdecode.py)."""

from __future__ import annotations

import math
import statistics

# Seed noise measured on one desktop: the SD of tok/s over passes was 1.0% to 3.8% on the workload columns.
DEFAULT_MARGIN = 0.015  # unmeasured: the lead hybrid needs on its worst workload column, besides a lead clear of that noise
MAX_SPREAD = 0.05       # unmeasured: the largest timing spread allowed between two starts of one arm on the same text
# unmeasured: about twice the 0.25 ms CPU round trip on a 4.2 ms all-hit step (docs/performance/decode.md:47 and 86-87)
GUARD_LOSS = 0.10
MIN_PASSES = 3          # fewer passes of an arm give no interval, and hybrid cannot qualify without one

ROUND_MODES = ("min-wait", "stochastic", "prefer-fetch")  # FREETOKEN_HYBRID_FETCH_ROUND (moe/offload_kernels.py)


# ---- the rule ----

def t95(df: float) -> float:
    """One-sided 95% quantile of Student's t at ``df`` degrees of freedom; under Welch ``df`` is not a whole number."""
    df = max(df, 1.0)
    scale = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def area(x: float, n: int = 128) -> float:
        # the density from 0 to x, by Simpson's rule
        h = x / n
        f = [(1 + (i * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(n + 1)]
        return scale * h / 3 * (f[0] + f[-1] + 4 * sum(f[1:-1:2]) + 2 * sum(f[2:-1:2]))

    lo, hi = 1.6, 6.4  # between the normal's 1.645 and the 6.314 of one degree of freedom
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if area(mid) < 0.45 else (lo, mid)
    return 0.5 * (lo + hi)


def _starts(arms: dict[str, list[list[float]]]) -> dict[str, list[list[list[float]]]]:
    """``offload``, ``offload#2``, ``offload#3`` are three server starts of one arm. Kept: the starts with as many
    columns as the first offload start and a positive reading for every pass."""
    out: dict[str, list[list[list[float]]]] = {}
    for name, cols in arms.items():
        if cols and all(col and min(col) > 0 for col in cols):
            out.setdefault(name.split("#", 1)[0], []).append([list(col) for col in cols])
    width = len(out["offload"][0]) if "offload" in out else 0
    return {name: kept for name, starts in out.items() if (kept := [s for s in starts if len(s) == width])}


def _passes(starts: list[list[list[float]]], c: int) -> list[float]:
    """An arm's samples on column ``c``. Pass k decodes the same text in every start, so the starts that made it
    are timing replicates of one sample: their mean."""
    return [statistics.fmean(s[c][k] for s in starts if k < len(s[c])) for k in range(max(len(s[c]) for s in starts))]


def arm_means(arms: dict[str, list[list[float]]]) -> dict[str, list[float]]:
    """Per arm and column, the mean over its passes as the rule takes it."""
    return {name: [statistics.fmean(_passes(starts, c)) for c in range(len(starts[0]))] for name, starts in _starts(arms).items()}


def _lead(hybrid: list[float], offload: list[float]) -> tuple[float, float | None]:
    """``(lead, low)`` on one column: hybrid's mean over offload's minus one, and the one-sided 95% lower bound of
    that (Welch, on the relative standard errors). ``low`` is None when an arm has fewer than MIN_PASSES passes."""
    mh, mo = statistics.fmean(hybrid), statistics.fmean(offload)
    lead = mh / mo - 1.0
    if min(len(hybrid), len(offload)) < MIN_PASSES:
        return lead, None
    a = statistics.variance(hybrid) / len(hybrid) / mh**2
    b = statistics.variance(offload) / len(offload) / mo**2
    if a + b == 0.0:
        return lead, lead  # every pass read the same: there is no noise to allow for
    df = (a + b) ** 2 / (a * a / (len(hybrid) - 1) + b * b / (len(offload) - 1))
    return lead, lead - t95(df) * math.sqrt(a + b)


def _verdict(strategy, mode, why, *, again=False, lead=None, low=None, leads=(), spread=None, guard=None) -> dict:
    return {"strategy": strategy, "round": mode, "lead": lead, "low": low, "leads": [list(x) for x in leads], "spread": spread,
            "why": why, "again": again, "guard": guard}


def decide(
    arms: dict[str, list[list[float]]],
    *,
    margin: float = DEFAULT_MARGIN,
    max_spread: float = MAX_SPREAD,
    guard: tuple[int, ...] = (),
    guard_loss: float = GUARD_LOSS,
    names: list[str] | None = None,
) -> dict:
    """Measured tok/s -> verdict. ``arms`` maps a server start (``offload``, ``offload#2``, ``hybrid:min-wait``, ...)
    to its columns (a probe prompt, or the round of concurrent streams), in the same order everywhere; a column holds
    the tok/s of each pass. ``guard`` holds the indexes of the guard columns; the others are workload columns.
    ``names`` are the columns' names, for the text of the verdict.

    The samples are the passes. Pass k decodes the same text in every start of an arm, so the starts are averaged
    pass by pass first. On each workload column lead = mean(hybrid) / mean(offload) - 1 and low = lead - t * se: se is
    the relative Welch standard error of the two arms' passes, t the one-sided 95% Student quantile at the Welch
    degrees of freedom. A hybrid arm qualifies if low > 0 on every workload column, lead >= ``margin`` on its worst
    workload column, and its mean is not more than ``guard_loss`` behind on any guard column. The qualifying arm with
    the largest worst-column lead wins; offload otherwise. An arm with fewer than MIN_PASSES passes has no interval
    and cannot qualify. If two starts of one arm differ by more than ``max_spread`` over the same passes, the machine
    was not steady and nothing is decided. ``again`` asks the bench for more passes: the worst-column lead reaches
    the margin, and some low is at or below zero."""
    runs = _starts(arms)
    if "offload" not in runs:
        return _verdict("offload", None, "no offload measurement")
    columns = len(runs["offload"][0])
    work = [c for c in range(columns) if c not in guard]

    def label(c: int) -> str:
        return names[c] if names and c < len(names) else f"column {c + 1}"

    spread = where = None
    for name, starts in runs.items():
        for c in range(columns) if len(starts) >= 2 else ():
            shared = min(len(s[c]) for s in starts)
            means = [statistics.fmean(s[c][:shared]) for s in starts]
            s = (max(means) - min(means)) / statistics.fmean(means)
            if spread is None or s > spread:
                spread, where = s, f"{name} differ by {s:.1%} on {label(c)}"
    if spread is not None and spread > max_spread:
        return _verdict("offload", None, f"two starts of {where} over the same passes (limit {max_spread:.0%}): "
                        "the machine was not steady, so nothing is decided", spread=spread)
    passes = {name: [_passes(starts, c) for c in range(columns)] for name, starts in runs.items()}
    mean = {name: [statistics.fmean(col) for col in cols] for name, cols in passes.items()}
    best = None
    for name in sorted(passes):  # a fixed order, so an exact tie goes to the same arm however the arms are listed
        if not name.startswith("hybrid") or not work:
            continue
        leads = [(c, *_lead(passes[name][c], passes["offload"][c])) for c in work]
        worst = min(leads, key=lambda x: x[1])
        blind = next((x for x in leads if x[2] is None), None)
        unclear = min((x for x in leads if x[2] is not None and x[2] <= 0.0), key=lambda x: x[2], default=None)
        kept = min(((mean[name][c] / mean["offload"][c], c) for c in guard if c < columns), default=None)
        if worst[1] < margin:
            reason = f"lead {worst[1]:+.1%} on {label(worst[0])} is below the margin of {margin:.1%}"
        elif blind is not None:
            reason = f"lead {blind[1]:+.1%} on {label(blind[0])} has no interval: fewer than {MIN_PASSES} passes"
        elif unclear is not None:
            reason = f"lead {unclear[1]:+.1%} on {label(unclear[0])} is not clear of zero: at least {unclear[2]:+.1%}"
        elif kept is not None and kept[0] < 1.0 - guard_loss:
            reason = f"{kept[0] - 1.0:+.1%} on {label(kept[1])}, output that barely misses the cache, is past the limit of {-guard_loss:+.1%}"
        else:
            reason = None
        if best is None or (reason is None, worst[1]) > best[:2]:
            best = (reason is None, worst[1], name, leads, kept, reason, worst[1] >= margin and unclear is not None)
    if best is None:
        return _verdict("offload", None, "no hybrid measurement", spread=spread)
    ok, lead, name, leads, kept, reason, again = best
    mode = name.split(":", 1)[1] if ":" in name else "min-wait"
    shown = ", ".join(f"{label(c)} {x:+.1%} " + (f"(at least {low:+.1%})" if low is not None else "(no interval)") for c, x, low in leads)
    if kept is not None:
        shown += f"; {kept[0] - 1.0:+.1%} on its worst guard column, limit {-guard_loss:+.1%}"
    why = f"hybrid ({mode}) against offload: {shown}" if ok else f"hybrid ({mode}) does not qualify: {reason}. Against offload: {shown}"
    lows = [low for _, _, low in leads]
    return _verdict("hybrid" if ok else "offload", mode if ok else None, why, again=again, lead=lead, low=None if None in lows else min(lows),
                    leads=leads, spread=spread, guard=kept[0] if kept is not None else None)


def guard_columns(row: dict) -> tuple[int, ...]:
    """Indexes of the row's guard columns (the row names them in ``guard``)."""
    names = row.get("guard") or ()
    return tuple(i for i, name in enumerate(row.get("columns") or ()) if name in names)


def verdict_for(row: dict, round_mode: str | None = "min-wait", *, margin: float = DEFAULT_MARGIN) -> dict:
    """The rule applied to the row's offload starts and to the one hybrid arm a server with ``round_mode`` runs
    (None: to every hybrid arm of the row)."""
    tok = row.get("tok_s") or {}
    arms = {name: vals for name, vals in tok.items()
            if not name.startswith("hybrid") or round_mode is None or name.split("#", 1)[0] == f"hybrid:{round_mode}"}
    return decide(arms, margin=margin, guard=guard_columns(row), names=row.get("columns"))
