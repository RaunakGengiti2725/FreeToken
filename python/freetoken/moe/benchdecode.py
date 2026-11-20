"""Decode A/B of offload against hybrid through the real serving path, and the row ``--moe-strategy auto`` reads.

The crossover between the two strategies moves with the machine, the model and the cache size
(docs/performance/decode.md), so this command measures it where it will be used. The method follows
``benchmarks/bench_decode_moe.py``: one ``ft serve`` per arm, streamed chat requests with thinking on and the
checkpoint's sampling, numbers from the server's own timings.

Server starts of the first round, in this order by default:

    offload               ft serve --moe-strategy offload
    hybrid:min-wait       ft serve --moe-strategy hybrid   (FREETOKEN_HYBRID_FETCH_ROUND=min-wait, the default mode)
    hybrid:stochastic     ft serve --moe-strategy hybrid   (FREETOKEN_HYBRID_FETCH_ROUND=stochastic)
    offload#2             offload again, last: the offload starts bracket the hybrid ones and show the spread

Each start decodes every probe prompt once, alone: ``--warm`` tokens that only fill the expert cache, then
``--decode`` tokens that are timed. Then it decodes ``--streams`` prompts at once and takes the sum of their
speeds, because a server that allows several running requests is also used that way. Each of these is one
column of the result. The prompts are an AIME-25 problem and a coding task (workload columns) and two prompts
that each make the model repeat one sentence (the guard columns, marked ``*``): output that barely misses the
cache, where hybrid gains nothing and still pays its CPU round trip per layer. Two, because how much hybrid
loses there differs from one repeated text to another.

The verdict is ``decode_profile.decide``: hybrid only if it beat offload by the margin on every workload
column and lost no more than the guard limit on either guard column. When the result is close to a line, or the
starts of one arm disagree, the bench adds a round (each hybrid arm again, then offload again), up to
``--rounds``. With ``--cache-rates`` the whole A/B runs once per rate, which gives the two curves and where
they cross.

    ft bench decode --model /path/to/ckpt
    ft bench decode --model /path/to/ckpt --cache-rates 0.05,0.1,0.2,0.35,0.5 --csv ab.csv
    ft bench decode --model /path/to/ckpt -- --speculative-algorithm mtp --moe-cpu-threads 8

Everything after ``--`` goes to every ``ft serve`` unchanged, so the row is filed for the flags you will serve with.
Run ``ft bench bw --model <checkpoint>`` first: hybrid takes its fetch split from that profile.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

from freetoken.moe import decode_profile

DEFAULT_ARMS = "offload,hybrid:min-wait,hybrid:stochastic"

# benchmarks/bench_decode_moe.py: the same dataset, answer instruction and sampling fallback
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# Used when AIME-25 cannot be fetched, and for the code column. Written for this tool.
BUILTIN_PROMPTS = [
    (
        "math",
        "A sequence starts with a_1 = 7 and follows a_(n+1) = (3 * a_n + 4) mod 101. Find the smallest n > 1 with "
        "a_n = 7, and explain each step of your reasoning before you give the final number.\n" + BOXED_INSTRUCTION,
    ),
    (
        "code",
        "Write a Python module that parses a crontab line into a schedule object and returns its next five run "
        "times after a given datetime. Handle lists, ranges, steps and the names of months and weekdays, raise a "
        "clear error for an invalid field, and add unit tests. Explain each function before you write it.",
    ),
    (
        "essay",
        "Explain to a new engineer how a least-recently-used cache works, when it performs badly, and what the "
        "alternatives are. Use one worked example with eight requests and a cache of three entries.",
    ),
    (
        "plan",
        "You maintain a web service that has become slow. Describe, step by step, how you would find out whether "
        "the database, the network or the application code is responsible, and what you would measure at each step.",
    ),
]

# The guard columns: hybrid's worst case is output whose experts all fit in the cache. Two prompts, because the
# loss there differs from one repeated text to another.
GUARD_PROMPTS = [
    (
        "repeat",
        "Write the sentence below 400 times, one copy per line, and write nothing else.\n\n"
        "The quick brown fox jumps over the lazy dog near the river bank.",
    ),
    (
        "repeat-2",
        "Copy the line below 300 times, one copy per line, and write nothing else.\n\n"
        "All work and no play makes for a very predictable sequence of tokens.",
    ),
]

# engine.py logs these while it resolves the config
KEY_LINE = re.compile(r"MoE decode setup: (?P<key>.+?); (?P<free>\d+) bytes of GPU memory free")
AUTO_LINE = re.compile(r"Auto-selected MoE strategy: (?P<strategy>\w+)")

EARLY = 64    # tokens after which a start is compared with offload's first tokens, to stop a collapsed arm

# what a server that died or hung up in the middle of an arm raises, besides the RuntimeErrors made here
SERVER_GONE = (OSError, http.client.HTTPException, ValueError)


def parse_args(argv: list[str] | None, prog: str) -> tuple[argparse.Namespace, list[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir, .gguf or .ftw")
    p.add_argument("--arms", default=DEFAULT_ARMS,
                   help=f"comma list (default {DEFAULT_ARMS}); a closing offload start is added to every round. `auto` adds a server "
                        "started with --moe-strategy auto, to compare what auto resolves to with the explicit strategies (use it with --no-write)")
    p.add_argument("--rounds", type=int, default=3, help="most rounds of server starts per cache rate (a second and third only when the result is close)")
    p.add_argument("--min-rounds", type=int, default=1, help="rounds to run even when the first is clear (2 or more shows the spread of every arm)")
    p.add_argument("--cache-rates", default=None,
                   help="comma list of --moe-cache-rate values: one A/B per rate. Without it the cache is whatever "
                        "the flags after -- say (--moe-cache-auto when they say nothing)")
    p.add_argument("--warm", type=int, default=1024,
                   help="tokens decoded before the timed window, per prompt. Hybrid caches only the experts it fetches, so its "
                        "cache fills slowly; a short warm-up reads hybrid too low at mid and large caches")
    p.add_argument("--decode", type=int, default=256, help="tokens in the timed window, per prompt")
    p.add_argument("--streams", type=int, default=None,
                   help="prompts decoded at once in the last column (default: the --max-running-requests after --, else 4; 1 = no such column)")
    p.add_argument("--aime", default=None, help=f"local AIME-25 jsonl (default: {AIME_REPO} from the Hub, the built-in prompts if that fails)")
    p.add_argument("--prompts", default=None,
                   help='jsonl of {"name":..., "prompt":..., "guard": false}: the single-stream columns, instead of AIME-25 problem 0, '
                        "the code prompt and the two repeat prompts. A line with \"guard\": true is a guard column. "
                        "The concurrent column cycles the lines that are not guards")
    p.add_argument("--greedy", action="store_true", help="temperature 0 instead of the checkpoint's sampling (steadier, but greedy text can loop and miss less)")
    p.add_argument("--gpu", default=None, help="GPU for the servers: a UUID or nvidia-smi index (as ft serve --gpu)")
    p.add_argument("--no-write", action="store_true", help="print the verdict, do not touch the decode profile")
    p.add_argument("--profile", default=None,
                   help="decode profile path (default: per GPU under ~/.cache/freetoken/benchdecode; a server reads another path from FREETOKEN_BENCHDECODE_PATH)")
    p.add_argument("--csv", default=None, help="append one line per cache rate, server start and column")
    p.add_argument("--json", default=None, help="write everything that was measured")
    p.add_argument("--server-timeout", type=float, default=1800, help="seconds to wait for a server to become ready")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to wait after a server stops, for the driver to give its memory back")
    p.add_argument("--server-cmd", default=None,
                   help="replace `python -m freetoken.cli` (to test this tool against a stand-in server)")
    ns = p.parse_args(argv)
    ns.arms = [a.strip() for a in ns.arms.split(",") if a.strip()]
    bad = [a for a in ns.arms if a not in ("offload", "auto") and not (a.startswith("hybrid:") and a.split(":", 1)[1] in decode_profile.ROUND_MODES)]
    if bad or "offload" not in ns.arms or len(ns.arms) < 2:
        p.error(f"arms must be offload plus one or more of hybrid:<{'|'.join(decode_profile.ROUND_MODES)}> and auto; got {ns.arms}")
    if not 1 <= ns.min_rounds <= ns.rounds:
        p.error("--min-rounds must be between 1 and --rounds")
    ns.cache_rates = [float(x) for x in ns.cache_rates.split(",")] if ns.cache_rates else [None]
    # --api-key: the bench talks to its own servers on 127.0.0.1, and a key must not end up in a results file
    for flag in ("--moe-strategy", "--moe-backend", "--port", "--host", "--model", "--api-key"):
        if any(f == flag or f.startswith(flag + "=") for f in extra):
            p.error(f"{flag} is set by the bench; drop it from the flags after --")
    if ns.cache_rates != [None] and any(f.startswith("--moe-cache-") for f in extra):
        p.error("--cache-rates sets the cache; drop the --moe-cache-* flag after --")
    if ns.streams is None:
        ns.streams = int(_flag_value(extra, "--max-running-requests", 4))
    return ns, extra


def _flag_value(flags: list[str], name: str, default):
    for i, f in enumerate(flags):
        if f == name and i + 1 < len(flags):
            return flags[i + 1]
        if f.startswith(name + "="):
            return f.split("=", 1)[1]
    return default


def load_aime(path: str | None) -> list[str] | None:
    """The AIME-25 problems, each with the answer instruction; None when they cannot be had."""
    if not path:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as e:  # offline, rate-limited, repo moved
            print(f"[bench] could not fetch {AIME_REPO}/{AIME_FILE} ({type(e).__name__}); using the built-in prompts. --aime takes a local jsonl", flush=True)
            return None
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            text = row.get("problem") or row["prompt"]
            out.append(text if "boxed" in text else f"{text}\n{BOXED_INSTRUCTION}")
    return out


def load_prompts(ns: argparse.Namespace) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """(single-stream columns, prompts of the concurrent column, names of the guard columns)."""
    aime = load_aime(ns.aime) if not ns.prompts else None
    builtin = dict(BUILTIN_PROMPTS)
    if ns.prompts:
        rows = [json.loads(line) for line in Path(ns.prompts).read_text().splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"--prompts {ns.prompts} has no prompts")
        single = [(str(r.get("name", i)), r["prompt"]) for i, r in enumerate(rows)]
        guard = [str(r.get("name", i)) for i, r in enumerate(rows) if r.get("guard")]
    else:
        single = [("aime", aime[0]) if aime else ("math", builtin["math"]), ("code", builtin["code"]), *GUARD_PROMPTS]
        guard = [name for name, _ in GUARD_PROMPTS]
    if ns.streams <= 1:
        return single, [], guard
    if ns.prompts:   # the prompts the file names, so a held-out file holds out every column; its guard lines apart
        pool = [s for s, r in zip(single, rows) if not r.get("guard")] or single
    else:
        pool = [(f"aime-{i}", text) for i, text in enumerate(aime)] if aime else list(BUILTIN_PROMPTS)
    return single, [pool[i % len(pool)] for i in range(ns.streams)], guard


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """The checkpoint's recommended sampling with a per-field fallback, as benchmarks/bench_decode_moe.py resolves it:
    the server's own defaults for unspecified fields are greedy, and greedy text routes to fewer experts."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    return params, f"checkpoint{sorted(recommended)} + fallback" if recommended else "fallback (no generation_config)"


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(ns: argparse.Namespace, extra: list[str], arm: str, rate: float | None, port: int) -> tuple[list[str], dict]:
    """The server of one arm. Only the strategy and, for hybrid, the rounding mode differ between arms."""
    strategy = arm.split("#")[0].split(":")[0]
    launcher = ns.server_cmd.split() if ns.server_cmd else [sys.executable, "-m", "freetoken.cli"]
    cmd = launcher + [
        "serve", "--model", ns.model, "--host", "127.0.0.1", "--port", str(port),
        "--moe-strategy", strategy,
        "--enable-timings-report",   # per-request timings measured in the scheduler
    ]
    if rate is not None:
        cmd += ["--moe-cache-rate", str(rate)]
    if ns.gpu:
        cmd += ["--gpu", ns.gpu]
    env = dict(os.environ)
    env.pop("FREETOKEN_API_KEY", None)  # a local server on a private port, stopped when the arm is done
    if env.get("LOG_LEVEL", "").strip().upper() not in ("", "INFO", "DEBUG"):
        env["LOG_LEVEL"] = "INFO"  # the bench reads what a server says of its setup from its INFO lines
    env.pop("FREETOKEN_HYBRID_FETCH_ROUND", None)
    if ":" in arm:
        env["FREETOKEN_HYBRID_FETCH_ROUND"] = arm.split(":", 1)[1].split("#")[0]
    return cmd + extra, env


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during startup; log: {log_path}\n{tail(log_path)}")
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except SERVER_GONE:
            time.sleep(0.2)
            continue
        if health.get("status") == "error":
            raise RuntimeError(f"server reported a startup error: {health}; log: {log_path}")
        if health.get("maintenance") == "serving":
            return
        time.sleep(0.5)
    raise RuntimeError(f"server not ready after {timeout:.0f} s; log: {log_path}")


def tail(log_path: str, n: int = 25) -> str:
    try:
        return "".join(Path(log_path).read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend and workers), then SIGKILL; as benchmarks/bench_decode_moe.py does."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        os.killpg(proc.pid, signal.SIGKILL)   # the frontend can exit on SIGTERM while a worker of its group still holds the GPU
    except ProcessLookupError:
        pass


def timed_request(origin: str, model_id: str, prompt: str, sampling: dict, warm: int, decode: int, floor: float | None = None) -> dict:
    """One streamed chat completion of warm + decode tokens, thinking on. Returns tok/s over the last ``decode``
    tokens and ``early``, the tok/s of the first EARLY tokens. With ``floor``, a request whose first EARLY tokens
    ran slower than that is abandoned: a collapsed hybrid arm would otherwise take hours to say the same."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": warm + decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,          # a timings object on every chunk
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    marks: list[tuple[int, float]] = []   # (tokens so far, decode milliseconds so far), from the server
    stamps: list[float] = []
    pieces: list[str] = []
    usage = None
    drafts = (0, 0)
    early = ttft_ms = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"request failed: HTTP {e.code}: {e.read()[:300]!r}") from e
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            t = chunk.get("timings")
            if t and t.get("predicted_n"):
                marks.append((int(t["predicted_n"]), float(t["predicted_ms"])))
                drafts = (int(t.get("draft_n", 0)), int(t.get("draft_n_accepted", 0)))
                if ttft_ms is None:
                    ttft_ms = float(t.get("queue_ms", 0.0)) + float(t.get("prompt_ms", 0.0))
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(time.perf_counter())
                    pieces.append(text)
            if early is None:
                if marks and marks[-1][0] >= EARLY and marks[-1][1] > 0:
                    early = (marks[-1][0] - 1) / marks[-1][1] * 1e3
                elif not marks and len(stamps) >= EARLY:
                    early = (len(stamps) - 1) / (stamps[-1] - stamps[0])   # no timings: one text event per decode step
                if early is not None and floor is not None and early < floor:
                    # leaving the with-block closes the stream; the server cancels a request whose client left
                    return {"tok_s": early, "early": early, "gave_up": True, "window": f"stopped after {EARLY} tokens", "ttft_ms": ttft_ms,
                            "halves": None, "repeat": None, "completion_tokens": None, "sha1": "-", "draft_n": drafts[0], "draft_n_accepted": drafts[1]}
    if usage is None:
        raise RuntimeError("stream ended without a usage chunk; is this a FreeToken server?")
    text = "".join(pieces)
    timed_text = "".join(pieces[-decode:]).encode()
    out = {"completion_tokens": usage.get("completion_tokens"), "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
           "draft_n": drafts[0], "draft_n_accepted": drafts[1], "early": early, "gave_up": False,
           "ttft_ms": ttft_ms if ttft_ms is not None else ((stamps[0] - t0) * 1e3 if stamps else None),
           # how well the timed text compresses: looping text is far below ordinary text, and it misses fewer experts
           "repeat": len(zlib.compress(timed_text)) / len(timed_text) if timed_text else None, "halves": None}
    after = [m for m in marks if m[0] > warm]
    if len(after) >= 2 and after[-1][1] > after[0][1]:
        (n0, ms0), (n1, ms1) = after[0], after[-1]
        out.update(tok_s=(n1 - n0) / (ms1 - ms0) * 1e3, window=f"tokens {n0}-{n1}, server timings")
        nm, msm = after[len(after) // 2]
        if ms0 < msm < ms1:
            out["halves"] = ((nm - n0) / (msm - ms0) * 1e3, (n1 - nm) / (ms1 - msm) * 1e3)
    elif len(stamps) >= 2 and usage.get("completion_tokens", 0) > 1:
        # no per-chunk timings: the whole request by arrival time, warm-up included (bench_decode_moe.py's method)
        out.update(tok_s=(usage["completion_tokens"] - 1) / (stamps[-1] - stamps[0]), window="whole request, arrival times")
    else:
        raise RuntimeError("the response carried neither timings nor two text events")
    return out


def concurrent_round(origin: str, model_id: str, prompts, sampling: dict, warm: int, decode: int) -> dict:
    """All ``prompts`` at once. tok/s is the sum over the streams, each over its own timed window."""
    results: list = [None] * len(prompts)

    def one(i: int) -> None:
        try:
            results[i] = timed_request(origin, model_id, prompts[i][1], sampling, warm, decode)
        except Exception as e:  # reported by the caller, in the main thread
            results[i] = e

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(prompts))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for r in results:
        if isinstance(r, Exception):
            raise r
    return {"tok_s": sum(r["tok_s"] for r in results), "early": None, "gave_up": False, "streams": results,
            "window": results[0]["window"], "ttft_ms": statistics.median([r["ttft_ms"] or 0.0 for r in results]),
            "halves": None, "repeat": min((r["repeat"] for r in results if r["repeat"] is not None), default=None),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in results),
            "sha1": hashlib.sha1("".join(r["sha1"] for r in results).encode()).hexdigest()[:12],
            "draft_n": sum(r["draft_n"] for r in results), "draft_n_accepted": sum(r["draft_n_accepted"] for r in results)}


def run_arm(ns, extra, arm: str, rate: float | None, single, together, sampling: dict, floors: list[float] | None = None) -> dict:
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-decode-{arm.replace(':', '-').replace('#', '-')}-", suffix=".log")
    cmd, env = serve_cmd(ns, extra, arm, rate, port)
    t_start = time.monotonic()
    columns = len(single) + (1 if together else 0)
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            wait_ready(origin, proc, log_path, ns.server_timeout)
            ready_s = time.monotonic() - t_start
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            gpus = get_json(f"{origin}/v1/stats").get("gpus") or [{}]
            geometry = get_json(f"{origin}/v1/cache/status").get("geometry") or {}
            results = []
            for i, (_, text) in enumerate(single):
                results.append(timed_request(origin, model_id, text, sampling, ns.warm, ns.decode, floors[i] if floors else None))
                if results[-1]["gave_up"]:
                    break
            if results[-1]["gave_up"]:
                # a collapsed arm: the other columns would only take longer to say the same
                results += [dict(results[-1]) for _ in range(columns - len(results))]
            elif together:
                results.append(concurrent_round(origin, model_id, together, sampling, ns.warm, ns.decode))
        except SERVER_GONE as e:
            raise RuntimeError(f"the server stopped answering ({type(e).__name__}: {e}); log: {log_path}") from e
        finally:
            stop_server(proc)
    time.sleep(ns.settle)  # let the driver reclaim VRAM before the next server
    key = free = resolved = None
    capped = False
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if m := KEY_LINE.search(line):
            key, free = m["key"].strip(), int(m["free"])
        if m := AUTO_LINE.search(line):
            resolved = m["strategy"]
        # a hybrid layer without an `ft bench bw` row fetches at most one expert per step
        capped |= "using a fixed fetch cap of 1" in line
    return {"arm": arm, "tok_s": [r["tok_s"] for r in results], "results": results, "key": key, "gpu_free_bytes": free,
            "cache_slots": int(geometry.get("moe_cache_size") or 0),
            "total_slots": int(geometry.get("num_experts") or 0) * int(geometry.get("num_moe_layers") or 0),
            "slot_bytes": int((geometry.get("unit_bytes") or {}).get("moe_per_expert") or 0),
            "capped": capped, "resolved": resolved, "gpu": gpus[0], "ready_s": ready_s, "total_s": time.monotonic() - t_start, "log": log_path, "cmd": cmd}


def run_ab(ns, extra, rate: float | None, single, together, guard: list[str], sampling: dict) -> dict:
    label = f"cache rate {rate}" if rate is not None else "cache from the serve flags"
    names = [n for n, _ in single] + ([f"x{len(together)}"] if together else [])
    arms: dict[str, dict] = {}
    lost: dict[str, str] = {}
    floors: list[float] | None = None

    def start(arm: str) -> None:
        nonlocal floors
        print(f"[bench] {label}: {arm} ...", flush=True)
        try:
            r = run_arm(ns, extra, arm, rate, single, together, sampling, None if arm.startswith("offload") else floors)
        except RuntimeError as e:
            if arm.startswith("offload"):
                raise  # nothing to compare with
            # a server that cannot start or answer with these flags is not the faster one with them, in any round
            name = arm.split("#")[0]
            lost[name] = str(e).splitlines()[0]
            for a in [a for a in arms if a.split("#")[0] == name]:
                del arms[a]
            print(f"[bench]   did not run: {e}", flush=True)
            return
        arms[arm] = r
        if arm == "offload" and all(x["early"] for x in r["results"][: len(single)]):
            floors = [0.5 * x["early"] for x in r["results"][: len(single)]]     # an arm at under half of offload's early speed is stopped
        print(f"[bench]   ready in {r['ready_s']:.0f} s, " + ", ".join(f"{n} {v:.1f} tok/s" for n, v in zip(names, r["tok_s"]))
              + f", {r['cache_slots']} slots, {r['results'][0]['window']}", flush=True)

    hybrids = [a for a in ns.arms if a.startswith("hybrid:")]
    start("offload")
    for h in hybrids:
        start(h)
    if "auto" in ns.arms:
        start("auto")
    start("offload#2")

    def close_call() -> str | None:
        """Why another round is worth it: some hybrid arm is near the line, or the starts of an arm disagree."""
        tok = {"tok_s": {a: r["tok_s"] for a, r in arms.items()}, "columns": names, "guard": guard}
        for h in hybrids:
            v = decode_profile.verdict_for(tok, h.split(":", 1)[1])
            if v["again"]:
                return v["why"]
        return None

    rounds = 1
    while rounds < ns.rounds and (rounds < ns.min_rounds or close_call()):
        rounds += 1
        print(f"[bench] {label}: {close_call() or 'another round was asked for'}; round {rounds}", flush=True)
        for h in hybrids:
            if h not in lost:
                start(f"{h}#{rounds}")
        start(f"offload#{rounds + 1}")
    verdict = decode_profile.verdict_for({"tok_s": {a: r["tok_s"] for a, r in arms.items() if a != "auto"}, "columns": names, "guard": guard}, None)
    keys = {r["key"] for r in arms.values()}
    key = next(iter(keys)) if len(keys) == 1 else None
    return {"rate": rate, "arms": arms, "verdict": verdict, "key": key, "keys_seen": sorted(k for k in keys if k), "rounds": rounds, "columns": names, "guard": guard, "lost": lost}


def slots_of(arms: dict[str, dict]) -> tuple[dict[str, int], str | None]:
    """Cache slots per strategy, and a complaint when starts of one strategy did not get the same cache."""
    seen: dict[str, set[int]] = {}
    for a, r in arms.items():
        name = a.split("#")[0].split(":")[0]
        if name in ("offload", "hybrid") and r["cache_slots"]:
            seen.setdefault(name, set()).add(r["cache_slots"])
    for name, counts in seen.items():
        if max(counts) > 1.01 * min(counts):
            return {}, (f"{name} starts ran with {min(counts)} to {max(counts)} cache slots: something else was using GPU "
                        "memory, or a stopped server had not released its own")
    return {name: min(counts) for name, counts in seen.items()}, None


def notes_for(ab: dict) -> list[str]:
    arms = ab["arms"]
    out = [f"{arm} did not run ({why}); it counts as not faster." for arm, why in ab["lost"].items()]
    if any(r["capped"] for r in arms.values()):
        out.append("hybrid ran with a fixed fetch cap of 1 on some layers, because the `ft bench bw` profile has no row for them. "
                   "Run `ft bench bw --model <checkpoint>` with the same --moe-cpu-threads, then bench again: this result is hybrid at its worst.")
    looped = sorted({ab["columns"][i] for r in arms.values() for i, x in enumerate(r["results"])
                     if x["repeat"] is not None and x["repeat"] < 0.12 and ab["columns"][i] not in ab["guard"]})
    if looped:
        out.append(f"the timed text of {', '.join(looped)} repeats itself; looping text uses fewer experts than real text, so offload reads better here than it would in use.")
    moved = [a for a, r in arms.items() for x in r["results"] if x["halves"] and abs(x["halves"][1] / x["halves"][0] - 1.0) > 0.15]
    if moved:
        out.append(f"the speed moved by more than 15% inside the timed window of {', '.join(sorted(set(moved)))}; a longer --decode gives a steadier number.")
    return out


def column_medians(arms: dict[str, dict], name: str) -> list[float]:
    """Per column, the median over the starts of one arm (``offload`` covers offload, offload#2, ...)."""
    starts = [r["tok_s"] for a, r in arms.items() if a.split("#")[0] == name]
    return [statistics.median(col) for col in zip(*starts)]


def worst_vs_offload(ab: dict, tok_s: list[float]) -> float:
    """The number the rule reads first: over the workload columns, the smallest ratio to offload's median, minus one."""
    off = column_medians(ab["arms"], "offload")
    return min((v / o for n, v, o in zip(ab["columns"], tok_s, off) if n not in ab["guard"]), default=float("nan")) - 1.0   # nan: every column is a guard


def report(ab: dict) -> None:
    arms = ab["arms"]
    names = ab["columns"]
    slots, _ = slots_of(arms)
    print(f"\n==== decode A/B, {'cache rate ' + str(ab['rate']) if ab['rate'] is not None else 'cache from the serve flags'}"
          f", cache slots {' / '.join(f'{k} {v}' for k, v in slots.items()) or '?'}, {ab['rounds']} round(s) ====")
    print(f"  {'server start':22s}" + "".join(f"{n + '*' if n in ab['guard'] else n:>10s}" for n in names)
          + "   worst workload column vs offload   first token   accepted drafts   output sha1 (per column)")
    for a, r in arms.items():
        if r["results"][0]["gave_up"]:
            print(f"  {a:22s}" + f"  stopped after {EARLY} tokens at {r['tok_s'][0]:.1f} tok/s, under half of offload's speed over the same tokens")
            continue
        dn = sum(x["draft_n"] for x in r["results"])
        acc = f"{sum(x['draft_n_accepted'] for x in r['results']) / dn:14.0%}" if dn else f"{'-':>14s}"
        ttft = [x["ttft_ms"] for x in r["results"] if x["ttft_ms"] is not None]
        first = f"{statistics.median(ttft):8.0f} ms" if ttft else f"{'-':>11s}"
        print(f"  {a:22s}" + "".join(f"{v:10.1f}" for v in r["tok_s"]) + f"   {worst_vs_offload(ab, r['tok_s']):+31.1%}   {first}   {acc}   "
              + " ".join(x["sha1"] for x in r["results"]))
    for note in notes_for(ab):
        print(f"  note: {note}")
    v = ab["verdict"]
    print(f"  best arm: {v['strategy']}" + (f" ({v['round']} rounding)" if v["round"] else "") + f". {v['why']}."
          + (f" Spread between starts of one arm: {v['spread']:.1%}." if v["spread"] is not None else ""))
    tok = {"tok_s": {a: r["tok_s"] for a, r in arms.items()}, "columns": names, "guard": ab["guard"]}
    for mode in dict.fromkeys(a.split("#")[0].split(":", 1)[1] for a in arms if a.startswith("hybrid:")):
        w = decode_profile.verdict_for(tok, mode)
        flag = "--moe-strategy hybrid" + ("" if mode == "min-wait" else f" with FREETOKEN_HYBRID_FETCH_ROUND={mode}")
        print(f"  {mode} rounding: {'hybrid qualifies, ' + flag if w['strategy'] == 'hybrid' else 'hybrid does not qualify'} ({w['why']})")
    if "auto" in arms:
        # the check of the whole idea: what the auto server did against the faster of the strategies it chooses from
        others = {a.split("#")[0] for a in arms} - {"auto"}
        best = [max(column_medians(arms, name)[c] for name in others) for c in range(len(names))]
        gap = min((v / b for n, v, b in zip(names, arms["auto"]["tok_s"], best) if n not in ab["guard"]), default=float("nan")) - 1
        off = column_medians(arms, "offload")
        kept = min((v / o for n, v, o in zip(names, arms["auto"]["tok_s"], off) if n in ab["guard"]), default=None)
        print(f"  the auto server resolved to {arms['auto']['resolved'] or 'an unknown strategy'}; on its worst workload column it is {gap:+.1%} "
              "from the faster of the other arms" + (f", and on its worst guard column {kept - 1:+.1%} from offload" if kept is not None else ""))


def file_row(ns, ab: dict, sampling_source: str) -> None:
    """File the row under the key the servers logged, or say why not."""
    arms = ab["arms"]
    any_arm = next(iter(arms.values()))
    slots, complaint = slots_of(arms)
    free = [r["gpu_free_bytes"] for a, r in arms.items() if a.split("#")[0] == "offload" and r["gpu_free_bytes"]]
    no_hybrid = not any(a.startswith("hybrid") for a in arms)
    if no_hybrid:
        print("  no hybrid server ran with these flags, so there is nothing for auto to take")
    if ab["key"] is None:
        print("  not filed: " + ("the servers logged different setups: " + " | ".join(ab["keys_seen"]) if len(ab["keys_seen"]) > 1
                                 else "the servers logged no `MoE decode setup` line (a build whose auto does not read decode rows)"))
        return
    if complaint:
        print(f"  not filed: {complaint}")
        return
    if ns.no_write:
        print(f"  not filed (--no-write). Setup: {ab['key']}")
        return
    row = {
        "tok_s": {a: r["tok_s"] for a, r in arms.items() if a != "auto"},
        "columns": ab["columns"],
        "guard": ab["guard"],
        "slots": slots,
        "total_slots": any_arm["total_slots"],
        "slot_bytes": any_arm["slot_bytes"],
        "gpu_free_bytes": min(free) if free else None,
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "stamp": decode_profile.stamp(any_arm["gpu"].get("name"), any_arm["gpu"].get("uuid")),
        "verdict": ab["verdict"],
        "bench": {"warm": ns.warm, "decode": ns.decode, "rounds": ab["rounds"], "sampling": sampling_source, "notes": notes_for(ab)},
    }
    # filed without a hybrid arm too: an earlier row of this setup on which hybrid qualified must not outlive a bench where it did not run
    dst = decode_profile.store(ab["key"], row, gpu=any_arm["gpu"], path=ns.profile)
    print(f"  filed for {ab['key']}\n  in {dst}")
    if no_hybrid:
        print("  --moe-strategy auto stays on offload for this setup on this GPU and build.")
        return
    modes = [m for m in decode_profile.ROUND_MODES if decode_profile.verdict_for(row, m)["strategy"] == "hybrid"]
    taken = decode_profile.verdict_for(row, None)   # what a server without FREETOKEN_HYBRID_FETCH_ROUND reads
    if taken["strategy"] == "hybrid" and decode_profile.AUTO_TAKES_HYBRID:
        print(f"  --moe-strategy auto now takes hybrid with {taken['round']} rounding for this setup on this GPU and build (FREETOKEN_HYBRID_FETCH_ROUND\n"
              f"  forces a mode; qualifying among those measured: {', '.join(modes)}). Hybrid computes some experts on the CPU: where two tokens\n"
              "  are nearly tied its output can differ from offload's. --moe-strategy offload keeps offload.")
    elif modes and decode_profile.AUTO_TAKES_HYBRID:
        print(f"  --moe-strategy auto stays on offload for this setup: {taken['why']}. On its own arm alone, {', '.join(modes)} rounding qualifies;\n"
              "  a server started with FREETOKEN_HYBRID_FETCH_ROUND set to it is judged on that arm and takes hybrid.")
    elif modes:
        print("  --moe-strategy auto stays on offload for this setup on this GPU and build, and logs this measurement with the flags that run hybrid.")


def main(argv: list[str] | None = None, prog: str = "ft bench decode") -> int:
    ns, extra = parse_args(argv, prog)
    single, together, guard = load_prompts(ns)
    sampling, sampling_source = resolve_sampling(ns.model, ns.greedy)
    print(f"[bench] model {ns.model}; arms {ns.arms}; columns {[n for n, _ in single]}" + (f" + {len(together)} streams at once" if together else "") + f"; guard {guard}"
          + f"; {ns.warm} warm + {ns.decode} timed tokens each; sampling: {sampling_source}; extra serve flags: {' '.join(extra) or 'none'}", flush=True)
    t0 = time.monotonic()
    summary = []
    for rate in ns.cache_rates:
        ab = run_ab(ns, extra, rate, single, together, guard, sampling)
        report(ab)
        file_row(ns, ab, sampling_source)
        slots, _ = slots_of(ab["arms"])
        if ns.csv:
            new = not os.path.exists(ns.csv)
            with open(ns.csv, "a") as f:
                if new:
                    f.write("cache_rate,cache_slots,arm,prompt,tok_s,verdict,sha1,ttft_ms\n")
                for a, r in ab["arms"].items():
                    for n, x in zip(ab["columns"], r["results"]):
                        ttft = "" if x["ttft_ms"] is None else f"{x['ttft_ms']:.1f}"
                        f.write(f"{'' if rate is None else rate},{r['cache_slots']},{a},{n},{x['tok_s']:.3f},{ab['verdict']['strategy']},{x['sha1']},{ttft}\n")
        summary.append((rate, slots.get("offload", 0), ab))
    if len(summary) > 1:
        first_col = summary[0][2]["columns"][0]
        print(f"\n==== by cache rate: offload's median tok/s on {first_col}, and each other arm's worst workload column against offload ====")
        names = list(dict.fromkeys(a.split("#")[0] for _, _, ab in summary for a in ab["arms"]))
        modes = [n.split(":", 1)[1] for n in names if n.startswith("hybrid:")]
        print(f"  {'rate':>6s} {'slots':>7s} {'offload':>9s}" + "".join(f"{n:>19s}" for n in names if n != "offload") + "   rounds   "
              + ("auto server took" if "auto" in names else "what qualifies, by hybrid rounding mode"))
        for rate, slots_n, ab in summary:
            arms = ab["arms"]
            tok = {"tok_s": {a: r["tok_s"] for a, r in arms.items()}, "columns": ab["columns"], "guard": ab["guard"]}
            ran = {a.split("#")[0] for a in arms}
            worst = {n: f"{worst_vs_offload(ab, column_medians(arms, n)):+19.1%}" if n in ran else f"{'did not run':>19s}" for n in names if n != "offload"}
            if "auto" in names:
                takes = arms.get("auto", {}).get("resolved") or "?"
            else:
                takes = ", ".join(f"{m}: {decode_profile.verdict_for(tok, m)['strategy']}" for m in modes)
            print(f"  {rate:6.2f} {slots_n:7d} {column_medians(arms, 'offload')[0]:9.1f}" + "".join(worst.values()) + f"   {ab['rounds']:6d}   {takes}")
    if ns.json:
        with open(ns.json, "w") as f:
            json.dump([{"cache_rate": rate, **{k: ab[k] for k in ("columns", "guard", "rounds", "verdict", "key", "arms", "lost")}} for rate, _, ab in summary], f, indent=1)
    servers = sum(len(ab["arms"]) for _, _, ab in summary)
    print(f"\n[bench] {servers} server starts, {time.monotonic() - t0:.0f} s in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
