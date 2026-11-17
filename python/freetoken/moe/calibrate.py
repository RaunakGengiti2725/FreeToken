"""--preset single-measured: time what the static preset cannot choose, once, and keep the choice.

The rule: every candidate (offload and hybrid, each with 3, 4 and 5 MTP draft tokens and with none) is one server
start that decodes each of the two built-in prompts PASSES times over, DECODE timed tokens after a warm-up: WARM
tokens in the first pass, which fills the expert cache, and WARM_AGAIN in the later ones, which find it filled.
With t(c, p, k) the tok/s of candidate c on prompt p in pass k, and s the static preset's choice:

    lead(c) = min over the prompts p of [ mean over k of t(c, p, k) / mean over k of t(s, p, k) ] - 1
    lead(c) is none where t(c, p, k) < t(s, p, k) for any prompt p and pass k

The candidate with the largest lead runs, if that lead is above MARGIN; otherwise the static choice runs; of two equal
leads the candidate that comes first in candidates() has it. The strategy and the draft depth are chosen together by this one rule: a
depth is not chosen per prompt. (The lead was first the smallest single pass's: on a 16 GB card that kept offload
over a hybrid that led by 25% to 36% in three passes and 9.7% in one.)

A draft token map is one more candidate where a file for this checkpoint is in the store (token_map_path, written
by scripts/make_token_map.py --store with a manifest that names its corpus): MAP_DEPTH draft tokens with the map,
per strategy. A --speculative-token-map on the command line is not a candidate: every server then has that map.

The candidates are timed in a fixed order, inside BUDGET_S seconds, the static choice first. Where it serves every
expert from the GPU (fused, or a cache with a slot for each): 5 drafts with the map (5 without, where there is no
map), 4 drafts, the other strategy at the static depth, then the rest. On full cards 5 drafts with a 32K map was
the one setting seen ahead of the static choice, on two boxes, and the strategies decode alike there. Where the
cache holds part of the experts: the other strategy at the static choice's depth, then the other settings (5 with
the map, 4, 5, none) of the strategy that is ahead of the two, then those of the other strategy; there the strategy
decided most on every box that was read. One strategy is ahead of the other by the rule above with a margin of 0. A
candidate is started only if the time used so far plus SLACK times the longest candidate so far is within the
budget, and its server must be ready while the longest time a candidate so far took after its load still fits. The
budget's end is also a deadline for every candidate, the static choice included: a load or a request still running
then is ended, and the server is gone GRACE_S seconds later at the latest, whatever it does. The candidates not
started, not ready or not done in time are stored as not timed and are not chosen. A later start does not time
them; --calibrate-again does, keeping the stored timings and spending a budget of its own on the candidates not
timed yet (once none is left, it times them all again). Where the static choice itself is not done in time, the
static preset runs untimed and the log says so. The draft depth cannot be changed in a running server (its
verify graphs and buffers are built for one depth), so every candidate is a model load, and on a box that loads in
80 s the budget holds two or three of them.

A dimension the command line fixed (--moe-strategy, --speculative-algorithm) is not a candidate. Where the static
preset serves every expert resident (fused: they fit on the GPU), fused is the only strategy and the depths are what
is timed: hybrid then has nothing to compute on the CPU, and timing it against offload would judge it against a
strategy the static preset does not run. The timings are a
row of the decode profile (moe/decode_profile.py), under the key and the stamp `--moe-strategy auto` reads its rows
by, so a later start with the same checkpoint, flags, machine and release measures nothing; with an automatically
sized cache, only while about as much GPU memory is free as when they were taken. A candidate that does
not start, or whose server ran without the drafts it was started with (a token map the engine refuses), is not
chosen, and the next start times the candidates again (once: a candidate that fails twice in a
row is left out from then on); when the static choice itself cannot be timed, or anything else fails, the static preset
runs and one log line says why. The servers are started and timed by `ft bench decode`'s own code (moe/benchdecode.py).
"""

from __future__ import annotations

import contextlib
import hashlib
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
# unmeasured: tokens that fill the expert cache, then tokens that are timed, per prompt and pass. The first warm-up is
# the bench's own default: after fewer, a cold cache was timed, and a candidate that misses less from cold looked
# faster. The later passes start on the cache the first one left, and warm up only as long as they are timed.
WARM, WARM_AGAIN, DECODE = 1024, 256, 256
PASSES = 2                # per prompt, in one server start
# seconds a calibration may take; a candidate is started only while SLACK times the longest one so far still fits
BUDGET_S, SLACK = 300, 1.1
OUT_OF_TIME = f"not timed: the {BUDGET_S} s of a calibration were used"
SETTLE_S = 3.0            # the wait after a stopped server where the GPU's free memory cannot be read
# unmeasured: after a candidate's server is gone, the longest wait for the driver to give its memory back, and how
# much may still be missing of what was free before the candidates
SETTLE_WAIT_S, SETTLE_MARGIN = 20.0, 64 << 20
GRACE_S = 30.0            # a candidate's server is gone this long after the budget's end at the latest (benchdecode.run_arm)
# unmeasured: about twice the noise of a lead between two single passes, whose tok/s had an SD of up to 3.8% (decode_profile.py); asked of a prompt's mean over its passes
MARGIN = 0.10
READY_S = 900             # the longest a candidate's server may take to load
PROMPTS = ("math", "code")
DEPTHS = (3, 4, 5, 0)     # the static preset's first
MAP_DEPTH = 5             # draft tokens of the candidate with a draft token map
_VERSION = 5
_now, _sleep = time.monotonic, time.sleep

# removed from the command line of a candidate's server: the bench names these itself, and the preset must not measure again
_DROPPED = ("--model-path", "--model", "--host", "--port", "--api-key", "--moe-strategy", "--moe-backend", "--preset")


def token_map_path(model_path: str) -> str:
    """Where the calibration looks for a draft token map of this checkpoint: in the directory of the decode profile,
    under the checkpoint's fingerprint. Not in the checkpoint's folder, where a .bin file would change that fingerprint."""
    store = os.path.dirname(decode_profile._path(None, None))
    return os.path.join(store, "token_maps", decode_profile.model_fingerprint(model_path) + ".bin")


def store_token_map(model_path: str) -> tuple[str, str | None, str | None]:
    """(path, sha256 of the file, sha256 its manifest <path>.json records) of the store's draft token map for this
    checkpoint; both sha256 are None where the map or its manifest cannot be read. --preset single serves the map
    where the two are equal, as this calibration times it."""
    path = token_map_path(model_path)
    try:
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        with open(path + ".json") as f:
            recorded = json.load(f).get("map_sha256")
    except (OSError, ValueError, AttributeError):
        return path, None, None
    return path, sha, recorded


def stored_token_map(args) -> tuple[str, str] | None:
    """(path, sha256) of the draft token map to time as a candidate, or None: the command line names a map itself, the
    store holds none for this checkpoint, or the file is not the one its manifest (<path>.json) describes."""
    if args.speculative_token_map:
        return None
    path, sha, recorded = store_token_map(args.model_path)
    if sha is None:
        return None
    if recorded != sha:
        logger.warning(f"--preset {PRESET}: {path} is not the map its manifest describes (sha256 {sha[:12]}, recorded {str(recorded)[:12]}): not timed")
        return None
    return path, sha


def candidates(args, fused: bool = False, token_map: str | None = None) -> list[tuple[str, str, int | None, str | None]]:
    """(name, strategy, depth, token map) of every candidate, the static preset's choice first. A depth of None leaves
    speculative decoding as the command line set it. ``fused``: the static preset takes fused (_static_strategy).
    ``token_map``: the store's map for this checkpoint (stored_token_map), timed at MAP_DEPTH draft tokens."""
    left = args.preset_fields
    if "moe_strategy" not in left:
        strategies = [args.moe_strategy]
    else:
        strategies = ["fused"] if fused else ["offload", "hybrid"]
    depths = DEPTHS if "speculative_algorithm" in left and "speculative_num_steps" in left else (None,)
    # the order they are timed in when the first strategy stays ahead (_measure): the strategies at the static depth
    # first; then the map, the one setting seen ahead of the static choice before (5 drafts and a 32K map, on a full card)
    out = [(f"{s}@{'set' if d is None else d}", s, d, None) for s in strategies for d in depths[:1]]
    for s in strategies:
        if token_map and len(depths) > 1:
            out.append((f"{s}@{MAP_DEPTH}map", s, MAP_DEPTH, token_map))
        out += [(f"{s}@{d}", s, d, None) for d in depths[1:]]
    return out


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


def child_flags(argv: list[str], depth: int | None, token_map: str | None = None) -> list[str]:
    """The flags of a candidate's server: the command line without what the bench sets, as the static preset, at
    ``depth``, with ``token_map`` as its draft token map."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in _DROPPED:
            skip = True
        elif arg.split("=", 1)[0] in _DROPPED or arg in ("--shell-mode", "--calibrate-again"):
            pass
        else:
            out.append(arg)
    out += ["--preset", "single"]
    if depth == 0:
        out += ["--speculative-algorithm", "off"]
    elif depth is not None and (depth != DEPTHS[0] or token_map):
        out += ["--speculative-num-steps", str(depth)]
    if token_map:
        out += ["--speculative-token-map", token_map]
    return out


def _dropped_drafts(log_path: str | None) -> bool:
    """Whether a candidate's server dropped the preset's drafts for the size of its expert cache (engine.PRESET_NO_DRAFTS in its log)."""
    from freetoken.engine.engine import PRESET_NO_DRAFTS

    try:
        with open(log_path, errors="replace") as f:
            return any(PRESET_NO_DRAFTS in line for line in f)
    except (OSError, TypeError):
        return False


class _Slower(RuntimeError):
    """The bench stopped a candidate that started under half the static choice's speed: a result, not a failure."""


def _time_candidate(args, argv: list[str], strategy: str, depth: int | None, floors: list[float] | None = None, ready_s: float = READY_S,
                    token_map: str | None = None, left_s: float | None = None) -> dict:
    """One server start of a candidate through the bench's own code: its tok/s per prompt and pass, and what it ran with.
    Raises RuntimeError when the server does not start within ``ready_s`` seconds or does not answer, when a request
    starts below its prompt's tok/s in ``floors`` (the bench then stops it: such a candidate cannot be the one chosen),
    or when it is not done ``left_s`` seconds from now: its load and every request end there, whatever the server does."""
    from freetoken.moe import benchdecode

    arm = strategy
    if strategy == "hybrid":
        # the rounding mode the real server will run (moe/offload_kernels.py reads the same variable)
        arm = f"hybrid:{(os.getenv('FREETOKEN_HYBRID_FETCH_ROUND') or 'min-wait').strip().lower()}"
    bench_argv = ["--model", args.model_path, "--seeds", str(PASSES), "--warm", str(WARM), "--decode", str(DECODE), "--streams", "1",
                  "--no-write", "--server-timeout", str(int(ready_s)), "--settle", "0", "--", *child_flags(argv, depth, token_map)]
    try:
        ns, extra = benchdecode.parse_args(bench_argv, "ft serve --preset " + PRESET)
    except SystemExit as exc:
        if _signalled:
            raise  # the launcher's own exit on a signal, not the parser's
        raise RuntimeError(f"the bench refused the command line (exit {exc.code})") from None
    # a candidate is timed on decode: the first-request warmup would only spend its seconds to ready
    ns.server_env = {"FREETOKEN_WARM_FIRST_REQUEST": "0"}
    builtin = dict(benchdecode.BUILTIN_PROMPTS)
    sampling, _ = benchdecode.resolve_sampling(ns.model, ns.greedy)
    ns.eos = benchdecode.eos_strings(ns.model)
    r = benchdecode.run_arm(ns, extra, arm, None, [(name, builtin[name]) for name in PROMPTS], [], sampling, floors, warm_after=WARM_AGAIN,
                            deadline=None if left_s is None else time.monotonic() + left_s, grace=GRACE_S)
    if any(x.get("gave_up") for x in r["results"]):
        raise _Slower("stopped early: under half the static choice's speed")
    return {
        "tok_s": [[float(run["tok_s"]) for run in benchdecode.runs_of(x)] for x in r["results"]],
        "drafts": sum(int(x.get("draft_n") or 0) for x in r["results"]),
        "no_drafts": _dropped_drafts(r.get("log")),
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


def _await_memory(args, before: int | None, name: str, final: bool) -> None:
    """After the server of candidate ``name`` was stopped or killed: wait until the GPU has again what was free before
    the candidates (``before`` bytes, within SETTLE_MARGIN), SETTLE_WAIT_S at most. Memory the driver gives back late
    leaves the next server's plan fewer expert slots, or none."""
    if not before:
        _sleep(SETTLE_S)
        return
    began = _now()
    while True:
        try:
            free = _probe_gpu(args)[2]
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError):
            free = None
        if not free:
            _sleep(SETTLE_S)
            return
        if before - free <= SETTLE_MARGIN:
            return
        if _now() - began >= SETTLE_WAIT_S:
            break
        _sleep(1.0)
    when = "before the server starts, outside the calibration's time" if final else "before the next candidate, inside the calibration's time"
    logger.warning(
        f"--preset {PRESET}: {before - free} bytes of GPU memory are still not free {SETTLE_WAIT_S:.0f} s after the server of "
        f"{name} was stopped ({when}); going on, and the next plan may be smaller for it"
    )


def measure(args, argv: list[str], todo: list[tuple[str, str, int | None, str | None]], known: dict | None = None,
            free: int | None = None) -> dict:
    """Time ``todo`` (the static choice first) and return the row. ``known``: a stored row whose timings are kept, so
    that only the candidates it had no time for are timed. ``free``: the GPU's free bytes before any candidate, which
    every stopped server is waited for to give back. Raises RuntimeError when the static choice cannot be timed."""
    stopped: list[str] = []  # the candidate whose server was stopped last and has not been waited for
    with _stopping_on_signals():
        try:
            return _measure(args, argv, todo, known, free, stopped)
        finally:
            # also where the static choice was cut: the server that starts next must not find the memory still held
            if stopped and not _signalled:
                _await_memory(args, free, stopped[0], final=True)


def _resident(static: tuple, first: dict) -> bool:
    """Whether the static choice serves every expert from the GPU: fused, or an expert cache with a slot for each."""
    return static[1] == "fused" or (bool(first.get("total_slots")) and (first.get("cache_slots") or 0) >= first["total_slots"])


def _measure(args, argv: list[str], todo: list[tuple[str, str, int | None, str | None]], known: dict | None, free: int | None,
             stopped: list[str]) -> dict:
    t0 = _now()
    static, names = todo[0], [c[0] for c in todo]
    # failed: did not start or answer, timed again on the next start. skipped: left out for a reason that holds then too
    tok_s, failed, skipped, wall = {}, {}, {}, {}
    first, tail = None, 0.0  # tail: the longest a candidate took after its server was ready (its requests and its stop)
    if known is not None:
        tok_s = {n: v for n, v in known["tok_s"].items() if n in names}
        skipped = {n: why for n, why in known.get("skipped", {}).items() if n in names and why != OUT_OF_TIME}
        wall = {n: float(v) for n, v in (known.get("wall_s") or {}).items() if n in tok_s} or {static[0]: 0.0}
        first = {**known["static"], **{k: known.get(k) for k in ("cache_slots", "total_slots", "slot_bytes", "gpu_free_bytes")}}
    # the candidates this run started: the budget is theirs, so a stored static choice that took most of an earlier
    # budget does not keep the ones it left untimed from ever starting
    ran: list[str] = []
    heads = [c for c in todo[1:] if c[2] == static[2] and c[3] is None]
    rest = [c for c in todo[1:] if c not in heads]
    queue, stage = [static], 0
    while queue or stage < 2:
        if not queue and stage == 0:
            stage = 1
            queue = list(heads)
            if _resident(static, first):
                # every expert on the GPU: a start is short and the strategies decode alike, so the static strategy's
                # deeper drafts come first (with the map: the one setting seen ahead of the static choice on full
                # cards), then 4 drafts, then the other strategy and what is left
                own = [c for c in rest if c[1] == static[1]]
                deep = next((c for c in own if c[3]), None) or next((c for c in own if c[2] == MAP_DEPTH), None)
                near = next((c for c in own if c[2] == DEPTHS[1] and c[3] is None), None)
                lead_in = [c for c in (deep, near) if c is not None]
                queue, stage = lead_in + heads + [c for c in rest if c not in lead_in], 2
            continue
        if not queue:
            # a cache that holds part of the experts: the strategy decides most, so the other one came first; then the
            # other settings, those of the strategy that is ahead first
            stage = 2
            ahead = choose(tok_s, [static[0]] + [c[0] for c in heads], margin=0.0)[0]
            leader = next(c[1] for c in todo if c[0] == ahead)
            queue = sorted(rest, key=lambda c: c[1] != leader)
            continue
        name, strategy, depth, token_map = queue.pop(0)
        if name in tok_s or name in skipped:
            continue  # kept from the stored row
        if first is not None and not first["drafts"] and (depth != static[2] or token_map) and not first.get("no_drafts"):
            # the static choice drafted nothing: this checkpoint has no MTP layers, and every depth is the same server
            skipped[name] = "the checkpoint drafts nothing"
            continue
        if first is not None and first.get("no_drafts") and strategy == static[1]:
            # the static preset dropped its drafts for this cache (engine._settle_preset_drafts): no drafts is the static
            # choice itself, and its rule already says the other depths of this strategy lose here
            skipped[name] = "the static preset serves this cache size without drafts"
            continue
        if ran and _now() - t0 + SLACK * max(wall[n] for n in ran) > BUDGET_S:
            skipped[name] = OUT_OF_TIME
            continue
        if stopped:
            _await_memory(args, free, stopped[0], final=False)
            del stopped[:]
        began = _now()
        stopped[:] = [name]
        ran.append(name)
        logger.info(f"--preset {PRESET}: timing {name} ({len(tok_s) + len(failed) + len(skipped) + 1} of {len(todo)}) ...")
        try:
            # a candidate that starts at under half the static choice's speed is stopped after its first tokens
            floors = [0.5 * e for e in first["early"]] if first is not None and all(first.get("early") or [None]) else None
            # the static choice may take as long as any server to load; the others must be ready with time left for their requests
            left = BUDGET_S - (began - t0)
            ready_s = min(READY_S, left) if len(ran) == 1 else max(1.0, min(READY_S, left - tail))
            r = _time_candidate(args, argv, strategy, depth, floors, ready_s=ready_s, left_s=left, **({"token_map": token_map} if token_map else {}))
        except RuntimeError as exc:
            wall[name] = _now() - began
            if first is None:
                if wall[name] >= left:
                    raise RuntimeError(f"the static choice was not timed inside the {BUDGET_S} s of a calibration, so it runs untimed") from exc
                raise
            if wall[name] >= min(ready_s, left):
                skipped[name] = OUT_OF_TIME  # still loading when its time was up: not a server that cannot start
            else:
                (skipped if isinstance(exc, _Slower) else failed)[name] = str(exc).splitlines()[0][:300]
            continue
        wall[name] = _now() - began
        tail = max(tail, wall[name] - float(r.get("ready_s") or 0.0))
        if first is not None and depth and not r["drafts"] and (first["drafts"] or first.get("no_drafts")):
            # the engine served it without the drafts it was started with (a token map it refuses, a draft that does
            # not load): what was timed is another candidate than the one it is named for
            failed[name] = "its server ran without drafts"
            continue
        first = first or r
        tok_s[name] = r["tok_s"]
    return {
        "tok_s": tok_s, "failed": failed, "skipped": skipped, "columns": list(PROMPTS), "warm": WARM, "warm_again": WARM_AGAIN,
        "decode": DECODE, "passes": PASSES, "budget_s": BUDGET_S, "wall_s": {n: round(v, 1) for n, v in wall.items()}, "tail_s": round(tail, 1),
        "static": {"early": first.get("early"), "drafts": first["drafts"], "no_drafts": bool(first.get("no_drafts"))},
        "cache_slots": first["cache_slots"], "total_slots": first["total_slots"], "slot_bytes": first["slot_bytes"],
        "gpu_free_bytes": first["gpu_free_bytes"], "seconds": round(_now() - t0, 1),
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


def _overrides(args, strategy: str, depth: int | None, static: bool = False, token_map: str | None = None) -> dict:
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
        if depth != DEPTHS[0] and not static:
            # timed with the depth on its command line, which the static preset's rule for small caches leaves alone
            out["preset_fields"] = tuple(n for n in args.preset_fields if n != "speculative_num_steps")
    if token_map:
        out["speculative_token_map"] = token_map
    return out


def measured_choice(args, argv: list[str]) -> dict:
    """The fields to replace on ``args`` for --preset single-measured: the stored choice, or a calibration's, or
    none (the static preset) when it cannot be measured. ``argv`` is the command line ``args`` was parsed from."""
    from freetoken.engine.engine import _decode_decision_key

    def static(reason: str) -> dict:
        logger.info(f"--preset {PRESET}: running the static preset ({reason})")
        return {}

    stored = stored_token_map(args)
    token_map, map_sha = stored or (None, None)
    todo = candidates(args, token_map=token_map)
    names = [c[0] for c in todo]
    if len(todo) == 1:
        return static("the command line sets the strategy and the speculative decoding: nothing to measure")
    if args.tp_info.size > 1 or args.shell_mode:
        return static("no calibration with tensor parallelism or under the shell")
    if decode_profile.stamp() is None:
        return static("calibrations are kept on Linux only: this machine's memory cannot be read from /proc")
    try:
        gpu_name, gpu_uuid, free = _probe_gpu(args)
        if "moe_strategy" in args.preset_fields and _static_strategy(args, argv) == "fused":
            todo = candidates(args, fused=True, token_map=token_map)
            names = [c[0] for c in todo]
            if len(todo) == 1:
                return static("every expert fits on the GPU and the command line sets the speculative decoding: nothing to measure")
        # the candidates are part of the setup: a command line that fixes one of them is timed for itself
        key = f"{_decode_decision_key(args)}|calibration={_VERSION}:{','.join(names)}"
        if any(c[3] for c in todo):
            key += f"|map={map_sha[:12]}"  # another map is another candidate
            logger.info(f"--preset {PRESET}: {MAP_DEPTH} draft tokens with the draft token map {token_map} (sha256 {map_sha[:12]}) are a candidate")
        stamp = decode_profile.stamp(gpu_name, gpu_uuid)
        row, reason = decode_profile.lookup(key, gpu_name=gpu_name, gpu_uuid=gpu_uuid, now=stamp, well_formed=_usable)
        failed_before = set(row.get("failed", {})) if row is not None else set()
        if row is not None and (stale := _stale(row, names, free, args)):
            row, reason = None, f"the stored calibration does not apply: {stale}"
        known = None
        if row is not None and getattr(args, "calibrate_again", False):
            reason = "--calibrate-again"
            static_kept = row.get("static")
            if OUT_OF_TIME in row.get("skipped", {}).values() and isinstance(static_kept, dict) and "drafts" in static_kept:
                # the last calibration ran out of time: its timings stay, and this one times what it did not reach
                known, reason = row, "--calibrate-again: timing what the stored calibration had no time for"
            row = None
        if row is None:
            logger.info(
                f"--preset {PRESET}: calibrating, up to {len(todo)} server starts within {BUDGET_S} s, each {PASSES} passes on "
                f"{len(PROMPTS)} prompts of {DECODE} timed tokens after {WARM} warm-up tokens, {WARM_AGAIN} from the second pass on ({reason})"
            )
            # the launcher's own reading of the free memory: a later start compares its reading with this one
            row = {**measure(args, argv, todo, known, free), "stamp": stamp, "gpu_free_bytes": free}
            if any(c[3] for c in todo):
                row["token_map"] = {"path": token_map, "sha256": map_sha}
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
    _, strategy, depth, token_map = todo[names.index(name)]
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
    return _overrides(args, strategy, depth, static=name == names[0], token_map=token_map)
