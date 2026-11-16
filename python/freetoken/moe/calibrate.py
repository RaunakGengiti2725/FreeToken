"""--preset single-measured: time what the static preset cannot choose, once, and keep the choice.

The rule: every candidate (offload and hybrid, each with no drafts, 3 and 5 MTP draft tokens) is one server start
that decodes DECODE tokens after WARM warm-up tokens on each of the two built-in prompts, PASSES times over. A
candidate's lead over the static preset's choice is taken per prompt from the means over the passes, and is the
smaller of the prompts'; a candidate behind the static choice in any single pass has none. The candidate with the
largest lead runs, if that lead is above MARGIN; otherwise the static choice runs. (The lead was first the smallest
single pass's: on a 16 GB card that kept offload over a hybrid that led by 25% to 36% in three passes and 9.7% in one.)

A dimension the command line fixed (--moe-strategy, --speculative-algorithm) is not a candidate. Where the static
preset serves every expert resident (fused: they fit on the GPU), fused is the only strategy and the depths are what
is timed: hybrid then has nothing to compute on the CPU, and timing it against offload would judge it against a
strategy the static preset does not run. The timings are a
row of the decode profile (moe/decode_profile.py), under the key and the stamp `--moe-strategy auto` reads its rows
by, so a later start with the same checkpoint, flags, machine and release measures nothing; with an automatically
sized cache, only while about as much GPU memory is free as when they were taken. A candidate that does
not start is not chosen, and the next start times the candidates again (once: a candidate that fails twice in a
row is left out from then on); when the static choice itself cannot be timed, or anything else fails, the static preset
runs and one log line says why. The servers are started and timed by `ft bench decode`'s own code (moe/benchdecode.py).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from freetoken.moe import decode_profile
from freetoken.utils import init_logger

logger = init_logger(__name__)

PRESET = "single-measured"
# unmeasured: tokens that fill the expert cache, then tokens that are timed, per prompt and pass. The warm-up is the
# bench's own default: after fewer, a cold cache was timed, and a candidate that misses less from cold looked faster.
WARM, DECODE = 1024, 512
PASSES = 2                # per prompt, in one server start: the second pass runs on the cache the first one left
# unmeasured: about twice the noise of a lead between two single passes, whose tok/s had an SD of up to 3.8% (decode_profile.py); asked of a prompt's mean over its passes
MARGIN = 0.10
READY_S = 900             # the longest a candidate's server may take to load
PROMPTS = ("math", "code")
DEPTHS = (3, 0, 5)        # the static preset's first
_VERSION = 2

# removed from the command line of a candidate's server: the bench names these itself, and the preset must not measure again
_DROPPED = ("--model-path", "--model", "--host", "--port", "--api-key", "--moe-strategy", "--moe-backend", "--preset")


def candidates(args, fused: bool = False) -> list[tuple[str, str, int | None]]:
    """(name, strategy, depth) of every candidate, the static preset's choice first. A depth of None leaves
    speculative decoding as the command line set it. ``fused``: the static preset takes fused (_static_strategy)."""
    left = args.preset_fields
    if "moe_strategy" not in left:
        strategies = [args.moe_strategy]
    else:
        strategies = ["fused"] if fused else ["offload", "hybrid"]
    depths = DEPTHS if "speculative_algorithm" in left and "speculative_num_steps" in left else (None,)
    return [(f"{s}@{'set' if d is None else d}", s, d) for d in depths for s in strategies]


def choose(tok_s: dict, names: list[str], margin: float = MARGIN) -> tuple[str, float]:
    """``(candidate, lead)``: of ``names`` (the static choice first), the one whose lead over the static choice is
    largest, if that lead is above ``margin``; else the static choice, with lead 0. A candidate's lead is, per prompt,
    its mean tok/s over the passes against the static choice's, and the smaller of the prompts'; one that is behind
    in any single pass has no lead. ``tok_s`` maps a candidate to its tok/s per prompt and pass; one that is missing
    did not run and is not chosen. A tie goes to the candidate named first."""
    base = tok_s[names[0]]
    best, best_lead = names[0], 0.0
    for name in names[1:]:
        if name not in tok_s:
            continue
        # pass k against pass k: both servers had then decoded the same number of tokens
        if any(c < b for cs, bs in zip(tok_s[name], base) for c, b in zip(cs, bs)):
            continue
        # a single pass's tok/s moves by several percent: the margin is asked of a prompt's mean, not of its worst pass
        lead = min(sum(cs) / sum(bs) for cs, bs in zip(tok_s[name], base)) - 1.0
        if lead > margin and lead > best_lead:
            best, best_lead = name, lead
    return best, best_lead


def _usable(row: dict) -> bool:
    """What every reader of a calibration row relies on; the file can be edited by hand."""
    tok = row.get("tok_s")
    if not isinstance(tok, dict) or not tok or not all(isinstance(row.get(k, {}), dict) for k in ("failed", "skipped")):
        return False
    width = len(PROMPTS)
    if not all(row.get(k) is None or (isinstance(row[k], int) and not isinstance(row[k], bool)) for k in ("cache_slots", "total_slots", "slot_bytes", "gpu_free_bytes")):
        return False
    return all(
        isinstance(cols, list) and len(cols) == width
        and all(isinstance(col, list) and len(col) == PASSES for col in cols)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for col in cols for v in col)
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


def _static_strategy(args, argv: list[str]) -> str | None:
    """The strategy the static preset resolves to for this command line on the GPU the server will bind, from the
    engine's own _adjust_config in a process of its own (see _probe_gpu); None where that cannot be run."""
    target = (args.gpu_assigned or args.gpu or ("0",))[0]
    code = (
        "import json, sys\n"
        "from freetoken.server.args import parse_args\n"
        "args, _ = parse_args(sys.argv[2:])\n"
        "from freetoken.distributed import set_tp_info\n"
        "from freetoken.engine import engine\n"
        "from freetoken.gpu_select import bind_assigned_gpu, set_assigned_gpu\n"
        "set_tp_info(rank=args.tp_info.rank, size=args.tp_info.size)\n"
        "engine.set_quant_backend(engine._adjust_ftw_quant_backend(args.model_path, engine.QuantBackend.parse(args.quant_backend)))\n"
        "set_assigned_gpu(sys.argv[1])\n"
        "bind_assigned_gpu()\n"
        "engine._adjust_config(args)\n"
        "print(json.dumps(args.moe_strategy))\n"
    )
    try:
        done = subprocess.run([sys.executable, "-c", code, target, *argv], capture_output=True, text=True, timeout=300)
        return json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


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


class _Slower(RuntimeError):
    """The bench stopped a candidate that started under half the static choice's speed: a result, not a failure."""


def _time_candidate(args, argv: list[str], strategy: str, depth: int | None, floors: list[float] | None = None) -> dict:
    """One server start of a candidate through the bench's own code: its tok/s per prompt and pass, and what it ran with.
    Raises RuntimeError when the server does not start or does not answer, or when a request starts below its
    prompt's tok/s in ``floors`` (the bench then stops it: such a candidate cannot be the one chosen)."""
    from freetoken.moe import benchdecode

    arm = strategy
    if strategy == "hybrid":
        # the rounding mode the real server will run (moe/offload_kernels.py reads the same variable)
        arm = f"hybrid:{(os.getenv('FREETOKEN_HYBRID_FETCH_ROUND') or 'min-wait').strip().lower()}"
    bench_argv = ["--model", args.model_path, "--seeds", str(PASSES), "--warm", str(WARM), "--decode", str(DECODE), "--streams", "1",
                  "--no-write", "--server-timeout", str(READY_S), "--", *child_flags(argv, depth)]
    try:
        ns, extra = benchdecode.parse_args(bench_argv, "ft serve --preset " + PRESET)
    except SystemExit as exc:
        if _signalled:
            raise  # the launcher's own exit on a signal, not the parser's
        raise RuntimeError(f"the bench refused the command line (exit {exc.code})") from None
    builtin = dict(benchdecode.BUILTIN_PROMPTS)
    sampling, _ = benchdecode.resolve_sampling(ns.model, ns.greedy)
    ns.eos = benchdecode.eos_strings(ns.model)
    r = benchdecode.run_arm(ns, extra, arm, None, [(name, builtin[name]) for name in PROMPTS], [], sampling, floors)
    if any(x.get("gave_up") for x in r["results"]):
        raise _Slower("stopped early: under half the static choice's speed")
    return {
        "tok_s": [[float(run["tok_s"]) for run in benchdecode.runs_of(x)] for x in r["results"]],
        "drafts": sum(int(x.get("draft_n") or 0) for x in r["results"]),
        "early": [x.get("early") for x in r["results"]],
        "cache_slots": r["cache_slots"], "total_slots": r["total_slots"], "slot_bytes": r["slot_bytes"],
        "gpu_free_bytes": r["gpu_free_bytes"], "ready_s": r["ready_s"],
    }


# the signal that is ending the launcher, while candidates are timed
_signalled: list[int] = []


@contextlib.contextmanager
def _stopping_on_signals():
    """While candidates are timed, SIGTERM, SIGHUP and SIGINT end the launcher with exit code 128 + the signal.

    A candidate's server runs in a session of its own, so a signal to the launcher does not reach it, and the default
    action would end the launcher with the server still holding the GPU. The exit raised here passes through the
    bench's ``finally``, which stops that server's process group. The handlers found are put back afterwards."""
    if threading.current_thread() is not threading.main_thread():
        yield  # handlers can only be set from the main thread
        return
    seen = _signalled
    del seen[:]

    def stop(signum, _frame):
        if seen:
            return  # a second signal must not cut short the stop of the candidate's server
        seen.append(signum)
        logger.info(f"--preset {PRESET}: {signal.Signals(signum).name} during the calibration, stopping")
        raise SystemExit(128 + signum)

    # a signal the launcher ignores (SIGHUP under nohup) stays ignored
    found = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        if signal.getsignal(sig) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for sig, handler in found.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


def measure(args, argv: list[str], todo: list[tuple[str, str, int | None]]) -> dict:
    """Time ``todo`` (the static choice first) and return the row. Raises RuntimeError when the static choice cannot be timed."""
    with _stopping_on_signals():
        return _measure(args, argv, todo)


def _measure(args, argv: list[str], todo: list[tuple[str, str, int | None]]) -> dict:
    t0 = time.monotonic()
    # failed: did not start or answer, timed again on the next start. skipped: left out for a reason that holds then too
    tok_s, failed, skipped = {}, {}, {}
    first = None
    for name, strategy, depth in todo:
        if first is not None and not first["drafts"] and depth != todo[0][2]:
            # the static choice drafted nothing: this checkpoint has no MTP layers, and every depth is the same server
            skipped[name] = "the checkpoint drafts nothing"
            continue
        logger.info(f"--preset {PRESET}: timing {name} ({len(tok_s) + len(failed) + len(skipped) + 1} of {len(todo)}) ...")
        try:
            # a candidate that starts at under half the static choice's speed is stopped after its first tokens
            floors = [0.5 * e for e in first["early"]] if first is not None and all(first.get("early") or [None]) else None
            r = _time_candidate(args, argv, strategy, depth, floors)
        except RuntimeError as exc:
            if first is None:
                raise
            (skipped if isinstance(exc, _Slower) else failed)[name] = str(exc).splitlines()[0][:300]
            continue
        first = first or r
        tok_s[name] = r["tok_s"]
    return {
        "tok_s": tok_s, "failed": failed, "skipped": skipped, "columns": list(PROMPTS), "warm": WARM, "decode": DECODE, "passes": PASSES,
        "cache_slots": first["cache_slots"], "total_slots": first["total_slots"], "slot_bytes": first["slot_bytes"],
        "gpu_free_bytes": first["gpu_free_bytes"], "seconds": round(time.monotonic() - t0, 1),
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def _stale(row: dict, names: list[str], free: int | None, args) -> str | None:
    """Why a stored row does not answer for this start, or None."""
    from freetoken.engine.engine import _moe_cache_is_auto

    if names[0] not in row["tok_s"]:
        return f"it has no timing of {names[0]}"
    if missing := [n for n in names if n not in row["tok_s"] and n not in row.get("skipped", {})]:
        failed = row.get("failed", {})
        return "it has no timing of " + ", ".join(f"{n} (failed: {failed[n]})" if n in failed else n for n in missing)
    if _moe_cache_is_auto(args):
        return _memory_objection(row, free, args.memory_ratio)
    return None


def _memory_objection(row: dict, free: int | None, memory_ratio: float) -> str | None:
    """Why timings taken with an automatically sized cache do not answer at ``free`` bytes of GPU memory, or None.

    A decode row is accepted at less memory than it was measured with (decode_profile.auto_size_objection). These
    timings are not: they also chose the draft depth, whose buffers and the cache share that memory. The tolerance
    is the same SLOT_GROWTH of the timed cache, in both directions."""
    then, slots, slot_bytes = row.get("gpu_free_bytes"), row.get("cache_slots"), row.get("slot_bytes")
    if not then or not free:
        return "it does not say how much GPU memory was free when it was timed"
    if slots and slot_bytes:
        if free > then and row.get("total_slots") and slots >= row["total_slots"]:
            return None  # every expert was cached already: more memory cannot grow the cache
        moved = abs(free - then) * memory_ratio / slot_bytes > decode_profile.SLOT_GROWTH * slots
    else:
        moved = abs(free - then) > decode_profile.SLOT_GROWTH * then
    if moved:
        return f"{abs(free - then) / 2**30:.1f} GiB {'more' if free > then else 'less'} GPU memory is free than when it was timed ({slots or 'unknown'} cache slots)"
    return None


def _overrides(args, strategy: str, depth: int | None, static: bool = False) -> dict:
    """The fields of ``args`` that make it run the candidate. The ``static`` choice leaves the strategy as the
    command line did: --moe-strategy auto then resolves as it does for --preset single (a filed decode row, a
    unified-memory GPU), which an explicit offload would bypass."""
    from freetoken.moe import is_offload_moe_strategy

    out: dict = {}
    if "moe_strategy" in args.preset_fields and not static:
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
        if "moe_strategy" in args.preset_fields and _static_strategy(args, argv) == "fused":
            todo = candidates(args, fused=True)
            names = [name for name, _, _ in todo]
            if len(todo) == 1:
                return static("every expert fits on the GPU and the command line sets the speculative decoding: nothing to measure")
        # the candidates are part of the setup: a command line that fixes one of them is timed for itself
        key = f"{_decode_decision_key(args)}|calibration={_VERSION}:{','.join(names)}"
        stamp = decode_profile.stamp(gpu_name, gpu_uuid)
        row, reason = decode_profile.lookup(key, gpu_name=gpu_name, gpu_uuid=gpu_uuid, now=stamp, well_formed=_usable)
        failed_before = set(row.get("failed", {})) if row is not None else set()
        if row is not None and (stale := _stale(row, names, free, args)):
            row, reason = None, f"the stored calibration does not apply: {stale}"
        if row is None:
            logger.info(
                f"--preset {PRESET}: calibrating, {len(todo)} server starts of {PASSES} passes of {WARM} + {DECODE} tokens on "
                f"{len(PROMPTS)} prompts ({reason})"
            )
            # the launcher's own reading of the free memory: a later start compares its reading with this one
            row = {**measure(args, argv, todo), "stamp": stamp, "gpu_free_bytes": free}
            for again in failed_before & set(row["failed"]):
                # a second failure in a row is not a passing one: later starts do not time for it again
                row["skipped"][again] = "failed twice: " + row["failed"].pop(again)
            if row["failed"]:
                logger.warning(
                    f"--preset {PRESET}: not timed, and so not chosen: " + "; ".join(f"{n} ({why})" for n, why in row["failed"].items())
                    + ". The next start times the candidates again"
                )
            took = f"calibration took {row['seconds']:.0f} s"
            try:
                decode_profile.store(key, row, gpu={"name": gpu_name, "uuid": gpu_uuid})
            except (OSError, ValueError, TypeError) as exc:
                # the timings are in hand: serve their choice, and the next start times again
                logger.warning(f"--preset {PRESET}: the calibration could not be kept ({type(exc).__name__}: {exc}); the next start times again")
        else:
            took = f"read the calibration of {row.get('measured_at')}, which took {row.get('seconds')} s"
        name, lead = choose(row["tok_s"], names)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        return static(f"the calibration failed: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")
    _, strategy, depth = todo[names.index(name)]
    timed = ", ".join(f"{n} {'/'.join(f'{sum(col) / len(col):.0f}' for col in row['tok_s'][n])}" for n in names if n in row["tok_s"])
    if name != names[0]:
        why = f"its lead over {names[0]} on its worse prompt (the passes' mean) is {lead:+.1%}"
    else:
        why = f"no candidate leads it by more than {MARGIN:.0%} on both prompts without falling behind in a pass"
        if "moe_strategy" in args.preset_fields:
            why += f"; --moe-strategy stays {args.moe_strategy}"
    left = {**row.get("skipped", {}), **row.get("failed", {})}
    if untimed := "; ".join(f"{n} ({left[n]})" for n in names if n in left):
        timed += f"; not timed: {untimed}"
    logger.info(f"--preset {PRESET}: {took}; mean tok/s on {'/'.join(PROMPTS)}: {timed}; taking {name}: {why}")
    return _overrides(args, strategy, depth, static=name == names[0])
