"""--preset single-measured: time what the static preset cannot choose, once, and keep the choice.

The rule: every candidate (offload and hybrid, each with no drafts, 3 and 5 MTP draft tokens) is one server start
that decodes DECODE tokens after WARM warm-up tokens on each of the two built-in prompts. The candidate whose
smaller lead over the static preset's choice across the two prompts is largest runs, if that lead is above MARGIN;
otherwise the static choice runs.

A dimension the command line fixed (--moe-strategy, --speculative-algorithm) is not a candidate. The timings are a
row of the decode profile (moe/decode_profile.py), under the key and the stamp `--moe-strategy auto` reads its rows
by, so a later start with the same checkpoint, flags, machine and release measures nothing. A candidate that does
not start is not chosen; when the static choice itself cannot be timed, or anything else fails, the static preset
runs and one log line says why. The servers are started and timed by `ft bench decode`'s own code (moe/benchdecode.py).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from freetoken.moe import decode_profile
from freetoken.utils import init_logger

logger = init_logger(__name__)

PRESET = "single-measured"
WARM, DECODE = 256, 512   # unmeasured: tokens that fill the expert cache, then tokens that are timed, per prompt
# unmeasured: about twice the noise of a lead between two single passes, whose tok/s had an SD of up to 3.8% (decode_profile.py)
MARGIN = 0.10
READY_S = 900             # the longest a candidate's server may take to load
PROMPTS = ("math", "code")
DEPTHS = (3, 0, 5)        # the static preset's first
_VERSION = 1

# removed from the command line of a candidate's server: the bench names these itself, and the preset must not measure again
_DROPPED = ("--model-path", "--model", "--host", "--port", "--api-key", "--moe-strategy", "--moe-backend", "--preset")


def candidates(args) -> list[tuple[str, str, int | None]]:
    """(name, strategy, depth) of every candidate, the static preset's choice first. A depth of None leaves
    speculative decoding as the command line set it."""
    left = args.preset_fields
    strategies = ["offload", "hybrid"] if "moe_strategy" in left else [args.moe_strategy]
    depths = DEPTHS if "speculative_algorithm" in left and "speculative_num_steps" in left else (None,)
    return [(f"{s}@{'set' if d is None else d}", s, d) for d in depths for s in strategies]


def choose(tok_s: dict, names: list[str], margin: float = MARGIN) -> tuple[str, float]:
    """``(candidate, lead)``: of ``names`` (the static choice first), the one whose smaller lead over the static
    choice across the prompts is largest, if that lead is above ``margin``; else the static choice, with lead 0.
    ``tok_s`` maps a candidate to its tok/s per prompt; one that is missing did not run and is not chosen. A tie
    goes to the candidate named first."""
    base = tok_s[names[0]]
    best, best_lead = names[0], 0.0
    for name in names[1:]:
        if name not in tok_s:
            continue
        lead = min(c / b for c, b in zip(tok_s[name], base)) - 1.0
        if lead > margin and lead > best_lead:
            best, best_lead = name, lead
    return best, best_lead


def _usable(row: dict) -> bool:
    """What every reader of a calibration row relies on; the file can be edited by hand."""
    tok = row.get("tok_s")
    if not isinstance(tok, dict) or not tok or not isinstance(row.get("failed", {}), dict):
        return False
    width = len(PROMPTS)
    return all(
        isinstance(cols, list) and len(cols) == width
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in cols)
        for cols in tok.values()
    )


def _probe_gpu(args) -> tuple[str | None, str | None, int | None]:
    """(name, uuid, free bytes) of the GPU the server will bind. Read in a process of its own: this one must hold
    no CUDA context while the candidates' servers run."""
    target = (args.gpu_assigned or args.gpu or ("0",))[0]
    code = (
        "import json, sys, torch\n"
        "from freetoken.gpu_select import bind_assigned_gpu, gpu_identity, set_assigned_gpu\n"
        "set_assigned_gpu(sys.argv[1])\n"
        "device = bind_assigned_gpu()\n"
        "ident = gpu_identity(device.index)\n"
        "print(json.dumps([ident['name'], ident['uuid'], int(torch.cuda.mem_get_info(device)[0])]))\n"
    )
    done = subprocess.run([sys.executable, "-c", code, target], capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise RuntimeError(f"the GPU could not be read: {(done.stderr.strip().splitlines() or ['no error text'])[-1][:200]}")
    name, uuid, free = json.loads(done.stdout.strip().splitlines()[-1])
    return name, uuid, free


def child_flags(argv: list[str], depth: int | None) -> list[str]:
    """The flags of a candidate's server: the command line without what the bench sets, as the static preset, at ``depth``."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in _DROPPED:
            skip = True
        elif arg.split("=", 1)[0] in _DROPPED or arg == "--shell-mode":
            pass
        else:
            out.append(arg)
    out += ["--preset", "single"]
    if depth == 0:
        out += ["--speculative-algorithm", "off"]
    elif depth is not None and depth != DEPTHS[0]:
        out += ["--speculative-num-steps", str(depth)]
    return out


def _time_candidate(args, argv: list[str], strategy: str, depth: int | None, floors: list[float] | None = None) -> dict:
    """One server start of a candidate through the bench's own code: its tok/s per prompt and what it ran with.
    Raises RuntimeError when the server does not start or does not answer, or when a request starts below its
    prompt's tok/s in ``floors`` (the bench then stops it: such a candidate cannot be the one chosen)."""
    from freetoken.moe import benchdecode

    arm = strategy
    if strategy == "hybrid":
        # the rounding mode the real server will run (moe/offload_kernels.py reads the same variable)
        arm = f"hybrid:{(os.getenv('FREETOKEN_HYBRID_FETCH_ROUND') or 'min-wait').strip().lower()}"
    bench_argv = ["--model", args.model_path, "--seeds", "1", "--warm", str(WARM), "--decode", str(DECODE), "--streams", "1",
                  "--no-write", "--server-timeout", str(READY_S), "--", *child_flags(argv, depth)]
    try:
        ns, extra = benchdecode.parse_args(bench_argv, "ft serve --preset " + PRESET)
    except SystemExit as exc:
        raise RuntimeError(f"the bench refused the command line (exit {exc.code})") from None
    builtin = dict(benchdecode.BUILTIN_PROMPTS)
    sampling, _ = benchdecode.resolve_sampling(ns.model, ns.greedy)
    ns.eos = benchdecode.eos_strings(ns.model)
    r = benchdecode.run_arm(ns, extra, arm, None, [(name, builtin[name]) for name in PROMPTS], [], sampling, floors)
    if any(x.get("gave_up") for x in r["results"]):
        raise RuntimeError("stopped early: under half the static choice's speed")
    return {
        "tok_s": [float(v) for v in r["tok_s"]], "drafts": sum(int(x.get("draft_n") or 0) for x in r["results"]),
        "early": [x.get("early") for x in r["results"]],
        "cache_slots": r["cache_slots"], "total_slots": r["total_slots"], "slot_bytes": r["slot_bytes"],
        "gpu_free_bytes": r["gpu_free_bytes"], "ready_s": r["ready_s"],
    }


def measure(args, argv: list[str], todo: list[tuple[str, str, int | None]]) -> dict:
    """Time ``todo`` (the static choice first) and return the row. Raises RuntimeError when the static choice cannot be timed."""
    t0 = time.monotonic()
    tok_s, failed = {}, {}
    first = None
    for name, strategy, depth in todo:
        if first is not None and not first["drafts"] and depth != todo[0][2]:
            # the static choice drafted nothing: this checkpoint has no MTP layers, and every depth is the same server
            failed[name] = "not timed: the checkpoint drafts nothing"
            continue
        logger.info(f"--preset {PRESET}: timing {name} ({len(tok_s) + len(failed) + 1} of {len(todo)}) ...")
        try:
            # a candidate that starts at under half the static choice's speed is stopped after its first tokens
            floors = [0.5 * e for e in first["early"]] if first is not None and all(first.get("early") or [None]) else None
            r = _time_candidate(args, argv, strategy, depth, floors)
        except RuntimeError as exc:
            if first is None:
                raise
            failed[name] = str(exc).splitlines()[0][:300]
            continue
        first = first or r
        tok_s[name] = r["tok_s"]
    return {
        "tok_s": tok_s, "failed": failed, "columns": list(PROMPTS), "warm": WARM, "decode": DECODE,
        "cache_slots": first["cache_slots"], "total_slots": first["total_slots"], "slot_bytes": first["slot_bytes"],
        "gpu_free_bytes": first["gpu_free_bytes"], "seconds": round(time.monotonic() - t0, 1),
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def _stale(row: dict, names: list[str], free: int | None, args) -> str | None:
    """Why a stored row does not answer for this start, or None."""
    from freetoken.engine.engine import _moe_cache_is_auto

    if names[0] not in row["tok_s"]:
        return f"it has no timing of {names[0]}"
    if missing := [n for n in names if n not in row["tok_s"] and n not in row.get("failed", {})]:
        return f"it has no timing of {', '.join(missing)}"
    if _moe_cache_is_auto(args) and free and all(row.get(k) for k in ("cache_slots", "slot_bytes", "gpu_free_bytes")):
        # the same check auto makes of a decode row: more free memory now means a larger cache than the one timed
        return decode_profile.auto_size_objection({**row, "slots": {"offload": row["cache_slots"]}}, free, args.memory_ratio)
    return None


def _overrides(args, strategy: str, depth: int | None) -> dict:
    """The fields of ``args`` that make it run the candidate."""
    from freetoken.moe import is_offload_moe_strategy

    out: dict = {}
    if "moe_strategy" in args.preset_fields:
        out["moe_strategy"] = strategy
        sized = args.moe_cache_size or args.moe_cache_auto or args.moe_cache_mem is not None or args.moe_cache_rate
        if is_offload_moe_strategy(strategy) and not sized:
            out["moe_cache_auto"] = True  # as server/args.py does for a strategy typed on the command line
    if depth == 0:
        out.update(speculative_algorithm=None, speculative_num_steps=0)
        out["preset_fields"] = tuple(n for n in args.preset_fields if n not in ("speculative_algorithm", "speculative_num_steps"))
    elif depth is not None:
        out["speculative_num_steps"] = depth
    return out


def measured_choice(args, argv: list[str]) -> dict:
    """The fields to replace on ``args`` for --preset single-measured: the stored choice, or a calibration's, or
    none (the static preset) when it cannot be measured. ``argv`` is the command line ``args`` was parsed from."""
    from freetoken.engine.engine import _decode_decision_key

    def static(reason: str) -> dict:
        logger.info(f"--preset {PRESET}: running the static preset ({reason})")
        return {}

    todo = candidates(args)
    names = [name for name, _, _ in todo]
    if len(todo) == 1:
        return static("the command line sets the strategy and the speculative decoding: nothing to measure")
    if args.tp_info.size > 1 or args.shell_mode:
        return static("no calibration with tensor parallelism or under the shell")
    if decode_profile.stamp() is None:
        return static("calibrations are kept on Linux only: this machine's memory cannot be read from /proc")
    try:
        gpu_name, gpu_uuid, free = _probe_gpu(args)
        # the candidates are part of the setup: a command line that fixes one of them is timed for itself
        key = f"{_decode_decision_key(args)}|calibration={_VERSION}:{','.join(names)}"
        stamp = decode_profile.stamp(gpu_name, gpu_uuid)
        row, reason = decode_profile.lookup(key, gpu_name=gpu_name, gpu_uuid=gpu_uuid, now=stamp, well_formed=_usable)
        if row is not None and (stale := _stale(row, names, free, args)):
            row, reason = None, f"the stored calibration does not apply: {stale}"
        if row is None:
            logger.info(f"--preset {PRESET}: calibrating, {len(todo)} server starts of {WARM} + {DECODE} tokens on {len(PROMPTS)} prompts ({reason})")
            row = {**measure(args, argv, todo), "stamp": stamp}
            decode_profile.store(key, row, gpu={"name": gpu_name, "uuid": gpu_uuid})
            took = f"calibration took {row['seconds']:.0f} s"
        else:
            took = f"read the calibration of {row.get('measured_at')}, which took {row.get('seconds')} s"
        name, lead = choose(row["tok_s"], names)
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return static(f"the calibration failed: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")
    _, strategy, depth = todo[names.index(name)]
    timed = ", ".join(f"{n} {'/'.join(f'{v:.0f}' for v in row['tok_s'][n])}" for n in names if n in row["tok_s"])
    why = f"its smaller lead over {names[0]} is {lead:+.1%}" if name != names[0] else f"no candidate leads it by more than {MARGIN:.0%} on both prompts"
    logger.info(f"--preset {PRESET}: {took}; tok/s on {'/'.join(PROMPTS)}: {timed}; taking {name}: {why}")
    return _overrides(args, strategy, depth)
