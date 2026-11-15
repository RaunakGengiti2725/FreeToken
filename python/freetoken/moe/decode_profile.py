"""The decode A/B of offload against hybrid: the rule that reads it, and the rows ``--moe-strategy auto`` reads.

``ft bench decode`` (moe/benchdecode.py) measures both strategies through the real serving path and files one
row per setup. A row belongs to one setup (its key) on one machine and release (its stamp). ``auto`` takes
hybrid only on a row whose key and stamp both match and whose numbers say hybrid is faster; otherwise it
stays on offload, as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Mapping

from freetoken.moe.bench_profile import load_layer_kinds

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


# ---- the key: the setup a row was measured for ----

DECODE_PROFILE_VERSION = 3  # 3: a row holds the tok/s of every pass, not one number per column

# 5% more slots is worth about DEFAULT_MARGIN to offload: 86 -> 131 tok/s from a 5% to a 20% cache (docs/performance/decode.md:48 and 93-94)
SLOT_GROWTH = 0.05

_WEIGHT_FILES = (".safetensors", ".gguf", ".ftw", ".bin")


def model_fingerprint(model_path: str) -> str:
    """``<name>@<8 hex>`` from the names and sizes of the checkpoint's weight files and the bytes of its ``config.json``.
    Other files in the folder (a README, a tokenizer, a download cache) do not make another checkpoint."""
    path = os.path.realpath(os.path.expanduser(model_path))
    h = hashlib.sha1()
    try:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if name.endswith(_WEIGHT_FILES) and os.path.isfile(full):
                    h.update(f"{name}:{os.path.getsize(full)};".encode())
            cfg = os.path.join(path, "config.json")
            if os.path.isfile(cfg):
                with open(cfg, "rb") as f:
                    h.update(f.read())
        elif os.path.isfile(path):
            h.update(f"{os.path.basename(path)}:{os.path.getsize(path)};".encode())
        else:
            h.update(model_path.encode())  # a Hub id that is not on disk: the id as given, not a path under the current directory
    except OSError:
        h.update(b"unreadable")  # the load reports the real problem; this only must not stop a start
    return f"{os.path.basename(path.rstrip('/'))}@{h.hexdigest()[:8]}"


def cache_spec(*, size: int = 0, mem: int | None = None, auto: bool = False, memory_ratio: float = 0.9) -> str:
    """How the expert cache was asked for: ``auto:<memory ratio>``, ``mem:<bytes>`` or ``size:<slots>``.

    auto and mem come first: the engine writes their resolved slot count into moe_cache_size only after the model is loaded."""
    if auto:
        return f"auto:{memory_ratio:g}"
    if mem is not None:
        return f"mem:{int(mem)}"
    return f"size:{int(size)}"


def cpu_spec(threads: int, cpus=None) -> str:
    """``<--moe-cpu-threads>/<CPUs allowed>@<hash of that CPU set>``: the same count on other cores is another setup."""
    if cpus is None:
        cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    ids = sorted(int(c) for c in cpus)
    return f"{int(threads)}/{len(ids)}@{hashlib.sha1(','.join(map(str, ids)).encode()).hexdigest()[:6]}"


# Switches that are not `ft serve` flags arrive through the environment. Keyed: this code base's own prefix, those of
# the kernels and model code it carries, and those of the libraries under it.
_ENV_KEYED_PREFIXES = (
    "FREETOKEN_",
    "DSV4_", "FLA_", "GDN_", "GGML_", "SGLANG_GDN_",
    "CUDA_", "PYTORCH_", "TORCH_", "TRITON_", "OMP_", "MKL_",
)
# Left out: cannot change decode speed, holds a secret, names a place, or is accounted for elsewhere (the GPU by the
# per-GPU row file, the bandwidth profile by the stamp, the rounding mode by the row's arms).
_ENV_IGNORED = frozenset((
    "FREETOKEN_HOST", "FREETOKEN_BENCH_PROGRESS", "FREETOKEN_BENCHBW_PATH", "FREETOKEN_BENCHDECODE_PATH", "FREETOKEN_BENCH_DECODE_KEY",
    "FREETOKEN_CONVERT_PROGRESS", "FREETOKEN_FORWARD_UNKNOWN_TOOLS", "FREETOKEN_HYBRID_FETCH_ROUND",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_HOME", "CUDA_PATH", "CUDA_CACHE_PATH", "TORCH_HOME", "TRITON_CACHE_DIR",
))
_ENV_IGNORED_PREFIXES = ("FREETOKEN_API_", "FREETOKEN_DAEMON_", "FREETOKEN_AIME25_", "FREETOKEN_KERNEL_CACHE_")
# Read somewhere in this code base and not a switch of decode; named so that a test can tell them from a new variable.
_ENV_NOT_DECODE = frozenset((
    "DSH_HOME", "LIGHTLLM_ENABLE_TOOL_NAME_CHECK", "LOG_LEVEL", "LOG_PID", "ROCM_HOME", "SGLANG_FORWARD_UNKNOWN_TOOLS", "XDG_CACHE_HOME",
))


def env_role(name: str) -> str | None:
    """``"key"`` for a variable that is part of the key, ``"ignored"`` for one that is deliberately not, None for one nobody has looked at."""
    if name in _ENV_IGNORED or name in _ENV_NOT_DECODE or name.startswith(_ENV_IGNORED_PREFIXES):
        return "ignored"
    return "key" if name.startswith(_ENV_KEYED_PREFIXES) else None


def perf_env(environ=None) -> dict[str, str]:
    """Every keyed variable, as name -> short hash of its value: an unknown switch is another setup, and no value is
    written to a log or a file."""
    environ = os.environ if environ is None else environ
    return {name: hashlib.sha1(value.encode()).hexdigest()[:6] for name, value in sorted(environ.items()) if env_role(name) == "key"}


def _plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


_KEY_TEXT = re.compile(r"[\w.,:=/+\-\[\]]{0,40}")


def non_default_settings(config, cls, skip=()) -> dict[str, str]:
    """Every field of dataclass ``cls`` that is not at its default on ``config``, as short text (a long or unusual
    value as a hash). A setting added to the config later lands here without a change to this module: a setting
    the row was measured without makes another key, and the row is not used."""
    out = {}
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        else:
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else dataclasses.MISSING
        value = _plain(getattr(config, f.name))
        if default is dataclasses.MISSING or value != _plain(default):
            text = str(value).removeprefix("torch.")
            short = (value is None or isinstance(value, (bool, int, float, str))) and _KEY_TEXT.fullmatch(text)
            out[f.name] = text if short else "#" + hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]
    return out


def decision_key(
    *,
    model: str,
    cache: str,
    cpu: str,
    spec: str = "off",
    tp_size: int = 1,
    max_running_req: int = 4,
    extra: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """One readable line, e.g. ``Qwen3.6-35B-A3B-NVFP4@3f2a9c1e|cache=auto:0.9|cpu=0/32@1f0c2a|spec=off|tp=1|req=4``.

    ``extra`` is :func:`non_default_settings`, ``env`` is :func:`perf_env`; both are appended sorted."""
    parts = [model, f"cache={cache}", f"cpu={cpu}", f"spec={spec}", f"tp={int(tp_size)}", f"req={int(max_running_req)}"]
    parts += [f"{k}={v}" for k, v in sorted((extra or {}).items())]
    parts += [f"env:{k}#{v}" for k, v in sorted((env or {}).items())]
    return "|".join(parts)


def key_diff(a: str, b: str) -> list[str]:
    """The parts in which two keys differ, by name: what a log line needs to say why a row does not apply."""
    def split(key: str) -> dict[str, str]:
        head, *rest = key.split("|")
        out = {"checkpoint": head}
        for part in rest:
            name, sep, value = part.partition("#" if part.startswith("env:") else "=")
            out[name] = value if sep else ""
        return out

    pa, pb = split(a), split(b)
    return [name for name in sorted(set(pa) | set(pb)) if pa.get(name) != pb.get(name)]


# ---- the stamp: the machine and the release a row was measured on ----

def freetoken_build() -> dict:
    """The release and the decode profile version: a new release makes every row stale, a commit between two
    releases does not (a nightly wheel's ``+g<sha>`` is left out)."""
    from freetoken.version import __version__

    return {"version": __version__.split("+", 1)[0], "decode_profile": DECODE_PROFILE_VERSION}


def _dist_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _first_line(path: str, prefix: str = "") -> str | None:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip() if prefix else line.strip()
    except OSError:
        pass
    return None


def bw_profile_digest(gpu_name: str | None, gpu_uuid: str | None) -> str | None:
    """A hash of the ``ft bench bw`` rows hybrid takes its fetch split from; None without a usable profile."""
    rows = load_layer_kinds(gpu_name=gpu_name, gpu_uuid=gpu_uuid)
    return None if rows is None else hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:10]


def stamp(gpu_name: str | None = None, gpu_uuid: str | None = None) -> dict | None:
    """What must be unchanged for a row to hold: the release, torch and triton, the GPU driver, the CPU, its memory,
    the container's CPU quota, and the bandwidth profile hybrid's fetch split comes from.

    None where /proc does not describe the machine (anything but Linux): no row is filed or used there."""
    mem_kb = _first_line("/proc/meminfo", "MemTotal")
    if mem_kb is None:
        return None
    return {
        "freetoken": freetoken_build(),
        "torch": _dist_version("torch"),
        "triton": _dist_version("triton"),
        "nvidia": _first_line("/proc/driver/nvidia/version", "NVRM version"),  # not there under WSL: a driver change is not seen
        "cpu": _first_line("/proc/cpuinfo", "model name"),
        "ram_gib": round(int(mem_kb.split()[0]) / 2**20),
        # the cgroup mounted at the root: a container's own quota, not one set on a nested cgroup
        "cpu_quota": _first_line("/sys/fs/cgroup/cpu.max") or _first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "benchbw": bw_profile_digest(gpu_name, gpu_uuid),
    }


def stamp_diff(row_stamp: dict | None, now: dict | None) -> list[str]:
    """The names of the stamp entries that changed since the row was measured."""
    if not isinstance(row_stamp, dict) or not isinstance(now, dict):
        return ["stamp"]
    return [name for name in sorted(set(row_stamp) | set(now)) if row_stamp.get(name) != now.get(name)]


# ---- rows on disk ----

def default_path(gpu_uuid: str | None = None) -> str:
    """``$XDG_CACHE_HOME/freetoken/benchdecode/<gpu-uuid>.json``: one file per GPU, next to the ``ft bench bw`` profiles."""
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken", "benchdecode", f"{gpu_uuid}.json" if gpu_uuid else "default.json")


def _path(path: str | None, gpu_uuid: str | None) -> str:
    return path or os.environ.get("FREETOKEN_BENCHDECODE_PATH") or default_path(gpu_uuid)


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _well_formed(row: dict) -> bool:
    """What every reader of a row relies on: per start and column the tok/s of its passes, and whole numbers of slots
    and bytes. The file can be edited by hand, and a row that breaks a reader must not stop a server start."""
    tok = row.get("tok_s")
    if not isinstance(tok, dict) or not isinstance(tok.get("offload"), list) or not tok["offload"]:
        return False
    width = len(tok["offload"])
    if not all(isinstance(cols, list) and len(cols) == width and all(_measured_passes(col) for col in cols) for cols in tok.values()):
        return False
    names, guard, slots = row.get("columns"), row.get("guard"), row.get("slots")
    if not (names is None or (isinstance(names, list) and len(names) == width)) or not (guard is None or isinstance(guard, list)):
        return False
    if not (slots is None or (isinstance(slots, dict) and all(_whole(v) and v > 0 for v in slots.values()))):
        return False
    return all(row.get(name) is None or _whole(row[name]) for name in ("total_slots", "slot_bytes", "gpu_free_bytes"))


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _measured_passes(col) -> bool:
    # json reads Infinity and NaN, and True is an int: none of them is a tok/s the bench wrote
    return isinstance(col, list) and bool(col) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in col)


def lookup(key: str, *, gpu_name: str | None, gpu_uuid: str | None, now, path: str | None = None) -> tuple[dict | None, str]:
    """``(row, reason)``. A row comes back only for this GPU, exactly this key and an unchanged stamp. ``now`` is
    the current stamp or a function that makes it; it is only called once a row with this key is found.
    Otherwise the reason names what differs or is stale, for the log line that says why auto stays on offload."""
    src = _path(path, gpu_uuid)
    prof = _load(src)
    if prof is None:
        return None, f"no decode profile at {src}"
    if prof.get("version") != DECODE_PROFILE_VERSION:
        return None, f"decode profile {src} has version {prof.get('version')}, this build reads {DECODE_PROFILE_VERSION}"
    measured_on = prof.get("gpu") if isinstance(prof.get("gpu"), dict) else {}
    if (gpu_uuid and measured_on.get("uuid") and measured_on["uuid"] != gpu_uuid) or (gpu_name and measured_on.get("name") and measured_on["name"] != gpu_name):
        return None, f"decode profile {src} was measured on another GPU ({measured_on.get('name')!r})"
    rows = prof.get("decisions") if isinstance(prof.get("decisions"), dict) else {}
    row = rows.get(key)
    if not isinstance(row, dict):
        head = key.split("|", 1)[0]
        near = min((k for k in rows if k.split("|", 1)[0] == head), key=lambda k: len(key_diff(k, key)), default=None)
        if near is None:
            return None, "no decode row for this checkpoint"
        return None, "no decode row for these settings; the nearest row differs in " + ", ".join(key_diff(near, key))
    if not _well_formed(row):
        return None, "the decode row holds no usable measurement"
    current = now() if callable(now) else now
    if current is None:
        return None, "decode rows are read on Linux only: this machine's memory cannot be read from /proc"
    changed = stamp_diff(row.get("stamp"), current)
    if changed:
        return None, "the decode row is stale: it was measured before a change of " + ", ".join(changed)
    return row, "ok"


def store(key: str, row: dict, *, gpu: dict, path: str | None = None) -> str:
    """Write one row. Rows of the same file, GPU and version are kept if their stamp equals this row's; a row with
    another stamp can no longer be used on this machine and release, so it is dropped. Atomic."""
    dst = _path(path, gpu.get("uuid"))
    prof = _load(dst)
    same = (
        prof is not None
        and prof.get("version") == DECODE_PROFILE_VERSION
        and isinstance(prof.get("gpu"), dict)
        and prof["gpu"].get("uuid") == gpu.get("uuid")
        and isinstance(prof.get("decisions"), dict)
    )
    rows = {k: r for k, r in prof["decisions"].items() if isinstance(r, dict) and not stamp_diff(r.get("stamp"), row.get("stamp"))} if same else {}
    rows[key] = row
    out = {"version": DECODE_PROFILE_VERSION, "gpu": {"name": gpu.get("name"), "uuid": gpu.get("uuid")}, "decisions": rows}
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    os.replace(tmp, dst)
    return dst


# ---- what a row says ----

def _measured(row: dict, verdict: dict) -> str:
    """When the row was measured, at what cache size, each arm's tok/s on the first workload column, and the rule's reading."""
    names = row.get("columns") or []
    first = next((c for c in range(len(names)) if c not in guard_columns(row)), 0)  # the first workload column: one stream
    parts = [f"{name} {means[first]:.1f}" for name, means in arm_means(row.get("tok_s") or {}).items() if len(means) > first]
    slots = _slots(row, "offload")
    at = f" at {slots} cache slots" if slots else ""
    on = f" on {names[first]}" if names else ""
    return f"decode A/B of {row.get('measured_at') or 'unknown time'}{at}, tok/s{on}: " + ", ".join(parts) + f"; {verdict.get('why', '')}"


def explain(row: dict, verdict: dict | None = None) -> str:
    """The log line that says what auto took, what was measured and why."""
    verdict = verdict or row.get("verdict") or {}
    mode = f" ({verdict['round']} rounding)" if verdict.get("round") else ""
    return f"{verdict.get('strategy', 'offload')}{mode}: {_measured(row, verdict)}"


def _slots(row: dict | None, arm: str) -> int | None:
    """Expert cache slots the arm ran with: offload and hybrid can be given different counts from the same memory."""
    slots = (row or {}).get("slots")
    return int(slots[arm]) if isinstance(slots, dict) and slots.get(arm) else None


# ---- an automatically sized cache: the row was measured at one size ----

def auto_size_objection(row: dict, gpu_free_bytes: int | None, memory_ratio: float = 0.9, growth: float = SLOT_GROWTH) -> str | None:
    """Before the model is built: why a row measured under --moe-cache-auto should not be used now, or None.

    More free GPU memory than at bench time means offload would get a larger cache than the one hybrid was
    compared against. Less is accepted: fewer slots mean more misses, and hybrid wins when misses are many
    (docs/performance/decode.md:89)."""
    slots, slot_bytes, free_then = _slots(row, "offload"), row.get("slot_bytes"), row.get("gpu_free_bytes")
    if not slots or not slot_bytes or not free_then or not gpu_free_bytes:
        return "the decode row does not say how much GPU memory was free when it was measured"
    if row.get("total_slots") and slots >= row["total_slots"]:
        return None  # every expert was cached already: more memory cannot grow the cache
    more = (gpu_free_bytes - free_then) * memory_ratio / slot_bytes
    if more > growth * slots:
        return (f"{(gpu_free_bytes - free_then) / 2**30:.1f} GiB more GPU memory is free than when the decode row was measured "
                f"({slots} cache slots): offload would now get about {slots + int(more)}")
    return None


def slot_cap(row: dict | None, size_cap: int) -> int:
    """When the cache is sized: hybrid was measured at this many slots, so do not run it with more."""
    slots = _slots(row, "hybrid")
    return min(size_cap, slots) if slots else size_cap


def resize_note(row: dict | None, new_slots: int, growth: float = SLOT_GROWTH) -> str | None:
    """When a running server's cache is resized: a warning if hybrid now runs with more slots than it was measured at."""
    slots = _slots(row, "hybrid")
    if not slots or new_slots <= slots * (1.0 + growth):
        return None
    return (f"hybrid was taken on a decode row measured at {slots} expert cache slots; at {new_slots} slots offload "
            "may be faster. Restart with --moe-strategy offload, or run `ft bench decode` at this size")
