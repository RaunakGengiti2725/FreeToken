"""The decode A/B of offload against hybrid: the rule that reads it, and the rows ``--moe-strategy auto`` reads.

``ft bench decode`` (moe/benchdecode.py) measures both strategies through the real serving path and files one
row per setup. A row belongs to one setup (its key) on one machine and build (its stamp). ``auto`` takes
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
import subprocess
from collections.abc import Mapping

from freetoken.moe.bench_profile import load_layer_kinds

# What --moe-strategy auto does with a row on which hybrid qualifies. True: it takes hybrid. False: it stays on
# offload and logs the measured lead with the flags that run hybrid (see hint).
AUTO_TAKES_HYBRID = True

DEFAULT_MARGIN = 0.015  # hybrid must beat offload by this much on its worst workload column
MAX_SPREAD = 0.05       # starts of one arm further apart than this: too unsettled to decide
AGAIN_BAND = 0.03       # a lead this close to the line is worth another round of starts
GUARD_BAND = 0.01       # the same for the guard column, whose reading moves less between starts
# Hybrid pays a CPU round trip per MoE layer even when nothing misses: 0.25 ms on top of a 4.2 ms all-hit step in
# docs/performance/decode.md, 5.6%. The guard column may lose about twice that; a cache that never fills lost 18% there.
GUARD_LOSS = 0.10

ROUND_MODES = ("min-wait", "stochastic", "prefer-fetch")  # FREETOKEN_HYBRID_FETCH_ROUND (moe/offload_kernels.py)


# ---- the rule ----

def _starts(arms: dict[str, list[float]]) -> dict[str, list[list[float]]]:
    """``offload``, ``offload#2``, ``offload#3`` are three server starts of one arm."""
    out: dict[str, list[list[float]]] = {}
    for name, vals in arms.items():
        if vals and min(vals) > 0:
            out.setdefault(name.split("#", 1)[0], []).append(list(vals))
    return out


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _verdict(strategy, mode, ratio, need, spread, why, again, guard=None) -> dict:
    return {"strategy": strategy, "round": mode, "ratio": ratio, "need": need, "spread": spread, "why": why, "again": again, "guard": guard}


def decide(
    arms: dict[str, list[float]],
    *,
    margin: float = DEFAULT_MARGIN,
    max_spread: float = MAX_SPREAD,
    guard: tuple[int, ...] = (),
    guard_loss: float = GUARD_LOSS,
) -> dict:
    """Measured tok/s -> verdict. ``arms`` maps a server start (``offload``, ``offload#2``, ``hybrid:min-wait``, ...)
    to one number per column (a probe prompt, or the round of concurrent streams), in the same order everywhere.
    ``guard`` holds the indexes of the guard columns; the others are workload columns.

    Per arm and column take the median over its starts. spread is the largest (max - min) / median among the starts
    of one arm. A hybrid arm qualifies if, on its worst workload column, it beats offload by max(margin, spread), and
    on its worst guard column it is not more than ``guard_loss`` behind. The qualifying arm with the largest lead wins;
    offload otherwise. ``again`` asks the bench for another round: the spread was too large, the lead is within
    AGAIN_BAND of its line, or the guard reading is within GUARD_BAND of its limit."""
    runs = _starts(arms)
    off = runs.get("offload")
    if not off:
        return _verdict("offload", None, None, margin, None, "no offload measurement", False)
    columns = max(len(r) for r in off)
    work = [c for c in range(columns) if c not in guard]
    usable = {name: [r for r in rs if len(r) == columns] for name, rs in runs.items()}
    med = {name: [_median([r[c] for r in rs]) for c in range(columns)] for name, rs in usable.items() if rs}
    spread = None
    for name, rs in usable.items():
        if len(rs) >= 2:
            s = max((max(r[c] for r in rs) - min(r[c] for r in rs)) / med[name][c] for c in range(columns))
            spread = s if spread is None else max(spread, s)
    if spread is not None and spread > max_spread:
        return _verdict("offload", None, None, margin, spread,
                        f"starts of the same arm differ by {spread:.1%} (limit {max_spread:.0%}): too unsettled to decide", True)
    need = max(margin, spread or 0.0)
    best = None
    for name in sorted(med):  # a fixed order, so an exact tie goes to the same arm however the arms are listed
        if not name.startswith("hybrid") or not work:
            continue
        ratio = min(med[name][c] / med["offload"][c] for c in work)
        kept = min((med[name][c] / med["offload"][c] for c in guard if c < columns), default=None)
        ok = ratio >= 1.0 + need and (kept is None or kept >= 1.0 - guard_loss)
        if best is None or (ok, ratio) > best[:2]:
            best = (ok, ratio, kept, name)
    if best is None:
        return _verdict("offload", None, None, need, spread, "no hybrid measurement", False)
    ok, ratio, kept, name = best
    mode = name.split(":", 1)[1] if ":" in name else "min-wait"
    why = f"hybrid ({mode}) is {ratio - 1.0:+.1%} on its worst workload column, needs {need:+.1%}"
    if kept is not None:
        why += f", and {kept - 1.0:+.1%} on output that barely misses the cache, limit {-guard_loss:+.1%}"
    again = abs(ratio - (1.0 + need)) < AGAIN_BAND or (kept is not None and ratio >= 1.0 + need and abs(kept - (1.0 - guard_loss)) < GUARD_BAND)
    if ok:
        return _verdict("hybrid", mode, ratio, need, spread, why, again, kept)
    return _verdict("offload", None, ratio, need, spread, why, again, kept)


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
    return decide(arms, margin=margin, guard=guard_columns(row))


# ---- the key: the setup a row was measured for ----

DECODE_PROFILE_VERSION = 2

# offload gains about 0.3% per 1% more cache slots (docs/performance/decode.md: 86 -> 131 tok/s from a 5% to a 20% cache),
# so 5% more slots than the row was measured at is worth about DEFAULT_MARGIN
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
    "FREETOKEN_HOST", "FREETOKEN_BENCH_PROGRESS", "FREETOKEN_BENCHBW_PATH", "FREETOKEN_BENCHDECODE_PATH", "FREETOKEN_CONVERT_PROGRESS",
    "FREETOKEN_FORWARD_UNKNOWN_TOOLS", "FREETOKEN_HYBRID_FETCH_ROUND",
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


# ---- the stamp: the machine and the build a row was measured on ----

def freetoken_build() -> dict:
    """FreeToken's version and, in a git checkout, its commit (as moe/benchbw.py stamps its profile)."""
    from freetoken.version import __version__

    try:
        commit = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"version": __version__, "commit": commit}


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


def stamp(gpu_name: str | None = None, gpu_uuid: str | None = None) -> dict:
    """What must be unchanged for a row to hold: the build, torch and triton, the GPU driver, the CPU, its memory,
    the container's CPU quota, and the bandwidth profile hybrid's fetch split comes from."""
    mem_kb = _first_line("/proc/meminfo", "MemTotal")
    return {
        "freetoken": freetoken_build(),
        "torch": _dist_version("torch"),
        "triton": _dist_version("triton"),
        "nvidia": _first_line("/proc/driver/nvidia/version", "NVRM version"),  # not there under WSL: a driver change is not seen
        "cpu": _first_line("/proc/cpuinfo", "model name"),
        "ram_gib": round(int(mem_kb.split()[0]) / 2**20) if mem_kb else None,
        "cpu_quota": _first_line("/sys/fs/cgroup/cpu.max") or _first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "benchbw": bw_profile_digest(gpu_name, gpu_uuid),
    }


def same_build(a: dict | None, b: dict | None) -> bool:
    """Same version, and the same commit when both sides know theirs."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("version") != b.get("version"):
        return False
    ca, cb = a.get("commit"), b.get("commit")
    return ca == cb if (ca and cb) else True


def stamp_diff(row_stamp: dict | None, now: dict | None) -> list[str]:
    """The names of the stamp entries that changed since the row was measured."""
    if not isinstance(row_stamp, dict) or not isinstance(now, dict):
        return ["stamp"]
    out = [] if same_build(row_stamp.get("freetoken"), now.get("freetoken")) else ["freetoken"]
    return out + [name for name in sorted(set(row_stamp) | set(now)) if name != "freetoken" and row_stamp.get(name) != now.get(name)]


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
    """What every reader of a row relies on: per start one number per column, and whole numbers of slots and bytes.
    The file can be edited by hand, and a row that breaks a reader must not stop a server start."""
    tok = row.get("tok_s")
    if not isinstance(tok, dict) or not isinstance(tok.get("offload"), list) or not tok["offload"]:
        return False
    # auto hands the rounding mode of the arm it takes to the kernel, so an arm must name a mode the kernel has
    if any(name.split("#", 1)[0] not in ("offload", "hybrid", *(f"hybrid:{m}" for m in ROUND_MODES)) for name in tok):
        return False
    width = len(tok["offload"])
    if not all(isinstance(vals, list) and len(vals) == width and all(_measured_number(v) for v in vals) for vals in tok.values()):
        return False
    names, guard, slots = row.get("columns"), row.get("guard"), row.get("slots")
    if not (names is None or (isinstance(names, list) and len(names) == width)) or not (guard is None or isinstance(guard, list)):
        return False
    if not (slots is None or (isinstance(slots, dict) and all(_whole(v) and v > 0 for v in slots.values()))):
        return False
    return all(row.get(name) is None or _whole(row[name]) for name in ("total_slots", "slot_bytes", "gpu_free_bytes"))


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _measured_number(v) -> bool:
    # json reads Infinity and NaN, and True is an int: none of them is a tok/s the bench wrote
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def lookup(key: str, *, gpu_name: str | None, gpu_uuid: str | None, now, path: str | None = None) -> tuple[dict | None, str]:
    """``(row, reason)``. A row comes back only for this GPU, exactly this key and an unchanged stamp. ``now`` is
    the current stamp or a function that makes it; it is only called once a row with this key is found.
    Otherwise the reason names what differs, for the log line that says why auto stays on offload."""
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
    changed = stamp_diff(row.get("stamp"), now() if callable(now) else now)
    if changed:
        return None, "the decode row was measured before a change of " + ", ".join(changed)
    return row, "ok"


def store(key: str, row: dict, *, gpu: dict, path: str | None = None) -> str:
    """Write one row. Rows of the same file, GPU and version are kept if their stamp equals this row's; a row with
    another stamp can no longer be used on this machine and build, so it is dropped. Atomic."""
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
    parts = [f"{name} {_median([r[first] for r in rs if len(r) > first]):.1f}" for name, rs in _starts(row.get("tok_s") or {}).items()]
    slots = _slots(row, "offload")
    at = f" at {slots} cache slots" if slots else ""
    on = f" on {names[first]}" if names else ""
    return f"decode A/B of {row.get('measured_at') or 'unknown time'}{at}, tok/s{on}: " + ", ".join(parts) + f"; {verdict.get('why', '')}"


def explain(row: dict, verdict: dict | None = None) -> str:
    """The log line that says what auto took, what was measured and why."""
    verdict = verdict or row.get("verdict") or {}
    mode = f" ({verdict['round']} rounding)" if verdict.get("round") else ""
    return f"{verdict.get('strategy', 'offload')}{mode}: {_measured(row, verdict)}"


def hint(row: dict, verdict: dict, running_mode: str = "min-wait") -> str:
    """The log line of a build whose auto does not take hybrid (AUTO_TAKES_HYBRID is False): it stays on offload and
    says what the row measured and the flags that run hybrid. ``running_mode`` is the rounding mode this server's
    environment selects; the variable is named only when the arm that qualified is another one."""
    mode = verdict.get("round") or running_mode
    flags = "`--moe-strategy hybrid`" + ("" if mode == running_mode else f" with FREETOKEN_HYBRID_FETCH_ROUND={mode}")
    return f"offload. The decode bench measured hybrid ({mode} rounding) faster for this setup: {_measured(row, verdict)}. {flags} runs it"


def _slots(row: dict | None, arm: str) -> int | None:
    """Expert cache slots the arm ran with: offload and hybrid can be given different counts from the same memory."""
    slots = (row or {}).get("slots")
    return int(slots[arm]) if isinstance(slots, dict) and slots.get(arm) else None


# ---- an automatically sized cache: the row was measured at one size ----

def auto_size_objection(row: dict, gpu_free_bytes: int | None, memory_ratio: float = 0.9, growth: float = SLOT_GROWTH) -> str | None:
    """Before the model is built: why a row measured under --moe-cache-auto should not be used now, or None.

    More free GPU memory than at bench time means offload would get a larger cache than the one hybrid was
    compared against. Less is accepted: with fewer slots there are more misses, and hybrid's lead grows with them."""
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
