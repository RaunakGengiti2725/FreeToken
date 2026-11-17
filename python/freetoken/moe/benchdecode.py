"""Decode A/B of offload against hybrid through the real serving path, and the row ``--moe-strategy auto`` reads.

The crossover between the two strategies moves with the machine, the model and the cache size
(docs/performance/decode.md), so this command measures it where it will be used. The method follows
``benchmarks/bench_decode_moe.py``: one ``ft serve`` per arm, streamed chat requests with thinking on and the
checkpoint's sampling, numbers from the server's own timings.

Server starts of the first round, in this order by default:

    offload               ft serve --moe-strategy offload
    hybrid:min-wait       ft serve --moe-strategy hybrid   (FREETOKEN_HYBRID_FETCH_ROUND=min-wait, the default mode)
    hybrid:stochastic     ft serve --moe-strategy hybrid   (FREETOKEN_HYBRID_FETCH_ROUND=stochastic)
    offload#2             offload again, last: the offload starts bracket the hybrid ones and show the timing spread

Each start decodes every probe prompt alone: ``--warm`` tokens that only fill the expert cache, then ``--decode``
tokens that are timed. It does so ``--seeds`` times over, one pass over the prompts after the other. Then it
decodes ``--streams`` prompts at once, as many times, and takes the sum of their speeds, because a server that
allows several running requests is also used that way. Each of these is one column of the result: the mean over
its passes, with the 95% interval.

A server seeds its sampler once, at start, and never between requests, so the passes of one start decode
different text and pass k is the same draws in every start: the k-th pass after a fresh start has had the same
sampling calls before it. Two starts of one arm therefore agree by construction; what differs between texts is
in the interval over the passes. Both arms draw the same uniform at the same step as long as every step draws
once, that is without speculative decoding. With it a verify step draws a row per draft and the number of steps
follows what was accepted, so the arms share their draws only until their texts first differ. The concurrent
passes come last: their batches form by arrival, so their draws do not repeat between starts.

Everything reported of a request is of its timed window, cut by token count: tok/s, decode steps per second and
tokens per step (the first moves with the share of drafts a speculative server accepts, the second does not),
that share, and the rate and the gaps at which tokens reached the client. ``--warm 0,256,1024,4096`` also prints
tok/s after each of those warm-ups; the verdict reads the longest. ``--context 32768,65536`` adds, per start, one
turn after a prompt of about that many tokens: prefill seconds, seconds to the first token and decode tok/s there.

The prompts are an AIME-25 problem and a coding task (workload columns) and two prompts that each make the model
repeat one sentence (the guard columns, marked ``*``): output that barely misses the cache, where hybrid gains
nothing and still pays its CPU round trip per layer. Two, so that the guard does not rest on one text.

The verdict is ``decode_profile.decide``. Its samples are the passes, with the starts of one arm averaged pass by
pass. Hybrid qualifies only if the one-sided 95% lower bound of its lead over offload is above zero on every
workload column, the lead reaches the margin on its worst one, and it lost no more than the guard limit on either
guard column. When the lead reaches the margin and a lower bound does not clear zero, the bench adds a round, up
to ``--rounds``: each hybrid arm again, then offload again, each decoding the earlier passes and ``--seeds`` new
ones. With ``--cache-rates`` the whole A/B runs once per rate, which gives the two curves and where they cross.

    ft bench decode --model /path/to/ckpt
    ft bench decode --model /path/to/ckpt --cache-rates 0.05,0.1,0.2,0.35,0.5 --csv ab.csv
    ft bench decode --model /path/to/ckpt -- --speculative-algorithm mtp --moe-cpu-threads 8
    ft bench decode --model /path/to/ckpt --arms offload,hybrid:min-wait --cache-rates 0.2 --seeds 3 --no-write -- --speculative-algorithm mtp --speculative-num-steps 3

Everything after ``--`` goes to every ``ft serve`` unchanged, so the row is filed for the flags you will serve with.
Run ``ft bench bw --model <checkpoint>`` first: hybrid takes its fetch split from that profile.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import http.client
import json
import math
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

# The guard columns: output that barely misses the cache, where hybrid only pays its CPU round trip. Two prompts,
# so that the guard does not rest on one text.
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

THREADS_LINE = re.compile(r"CPU MoE executor ready: threads=(\d+)")   # cpu_executor.py, once the pool is up

EARLY = 64    # tokens after which a start is compared with offload's first tokens, to stop a collapsed arm
CONTEXT_TURN = 300   # tokens decoded after a --context prompt
CPUFREQ = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"

# two-sided 95% Student t by degrees of freedom, 1 to 30
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)

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
    p.add_argument("--rounds", type=int, default=3, help="most rounds of server starts per cache rate. A second and third only when the lead is not clear of zero: "
                                                             "each decodes the passes of the round before and --seeds new ones")
    p.add_argument("--min-rounds", type=int, default=1, help="rounds to run even when the first is clear (2 or more shows the timing spread of every arm)")
    p.add_argument("--cache-rates", default=None,
                   help="comma list of --moe-cache-rate values: one A/B per rate. Without it the cache is whatever "
                        "the flags after -- say (--moe-cache-auto when they say nothing)")
    p.add_argument("--warm", default="1024",
                   help="tokens decoded before the timed window, per prompt: they fill the expert cache. Hybrid does not cache the "
                        "experts it computes on the CPU, so its cache fills more slowly than offload's. A comma list "
                        "(0,256,1024,4096) also reports tok/s after each warm-up; the verdict reads the longest")
    p.add_argument("--decode", type=int, default=2048, help="tokens in the timed window, per prompt")
    p.add_argument("--seeds", type=int, default=5,
                   help="passes over the prompts per start: a column is the mean over them, with a 95%% interval. The sampler's "
                        "stream runs on from one request to the next, so pass k decodes its own text, the same in every start. "
                        "With --greedy every pass decodes the same text")
    p.add_argument("--context", default=None,
                   help="comma list of prompt lengths in tokens (32768,65536,100000): per start and length, one prompt of about "
                        f"that many tokens and a {CONTEXT_TURN}-token turn after it. Off by default; not part of the verdict")
    p.add_argument("--streams", type=int, default=None,
                   help="prompts decoded at once in the last column (default: the --max-running-requests after --, else 4; 1 = no such column)")
    p.add_argument("--aime", default=None, help=f"local AIME-25 jsonl (default: {AIME_REPO} from the Hub, the built-in prompts if that fails)")
    p.add_argument("--prompts", default=None,
                   help='jsonl of {"name":..., "prompt":..., "guard": false}: the single-stream columns, instead of AIME-25 problem 0, '
                        "the code prompt and the two repeat prompts. A line with \"guard\": true is a guard column. "
                        "The concurrent column cycles the lines that are not guards")
    p.add_argument("--greedy", action="store_true", help="temperature 0 instead of the checkpoint's sampling: every pass then decodes the same text")
    p.add_argument("--gpu", default=None, help="GPU for the servers: a UUID or nvidia-smi index (as ft serve --gpu)")
    p.add_argument("--no-write", action="store_true", help="print the verdict, do not touch the decode profile")
    p.add_argument("--profile", default=None,
                   help="decode profile path (default: per GPU under ~/.cache/freetoken/benchdecode; a server reads another path from FREETOKEN_BENCHDECODE_PATH)")
    p.add_argument("--csv", default=None, help="append one line per cache rate, server start and column")
    p.add_argument("--json", default=None, help="write everything that was measured")
    p.add_argument("--server-timeout", type=float, default=1800, help="seconds to wait for a server to become ready")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to wait after a server stops, for the driver to give its memory back")
    p.add_argument("--server-cmd", default=None,
                   help="replace `python -m freetoken.cli`: the `ft` of another checkout, or a stand-in server to test this tool against")
    ns = p.parse_args(argv)
    ns.arms = [a.strip() for a in ns.arms.split(",") if a.strip()]
    bad = [a for a in ns.arms if a not in ("offload", "auto") and not (a.startswith("hybrid:") and a.split(":", 1)[1] in decode_profile.ROUND_MODES)]
    if bad or "offload" not in ns.arms or len(ns.arms) < 2:
        p.error(f"arms must be offload plus one or more of hybrid:<{'|'.join(decode_profile.ROUND_MODES)}> and auto; got {ns.arms}")
    if not 1 <= ns.min_rounds <= ns.rounds:
        p.error("--min-rounds must be between 1 and --rounds")
    try:
        ns.warms = int_list(ns.warm)
        ns.context = int_list(ns.context) if ns.context else []
    except ValueError as e:
        p.error(f"--warm and --context take comma lists of token counts: {e}")
    ns.warm = ns.warms[-1]
    if ns.seeds < 1 or ns.decode < 2 or (ns.context and ns.context[0] < 1):
        p.error("--seeds and --context need at least 1, --decode at least 2")
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


def int_list(text: str) -> list[int]:
    """``"1024,0,256"`` -> [0, 256, 1024]: token counts, each once, in rising order."""
    vals = sorted({int(x) for x in text.split(",") if x.strip()})
    if not vals or vals[0] < 0:
        raise ValueError(f"{text!r} names no count, or a negative one")
    return vals


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
    the server's own defaults for unspecified fields are greedy."""
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
    env["FREETOKEN_BENCH_DECODE_KEY"] = "1"  # an offload or hybrid server then logs the setup key auto looks its row up under
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


def stop_server(proc: subprocess.Popen, term_s: float = 90) -> None:
    """SIGTERM the whole session (frontend and workers), then SIGKILL after ``term_s`` seconds; as benchmarks/bench_decode_moe.py does."""
    try:
        for sig, wait_s in ((signal.SIGTERM, term_s), (signal.SIGKILL, 30)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        # also when a signal's exit ends the wait. The frontend can exit on SIGTERM while a worker of its group still holds the GPU
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def mean_ci(vals: list[float]) -> tuple[float, float | None]:
    """(mean, half-width of the two-sided 95% Student interval of the mean); no interval from fewer than two values."""
    n = len(vals)
    mean = statistics.fmean(vals)
    if n < 2:
        return mean, None
    return mean, (T95[n - 2] if n - 2 < len(T95) else 1.96) * statistics.stdev(vals) / math.sqrt(n)


def window_stats(marks: list[tuple], lo: int, hi: int) -> dict | None:
    """The stretch of one request after its first ``lo`` tokens, up to token ``hi``.

    ``marks`` holds one tuple per chunk: (tokens so far, server decode ms so far, arrival time at the client in
    seconds, drafted tokens so far, accepted drafts so far). A speculative step commits several tokens in one chunk,
    so the window is cut where the token count says, never by counting chunks: it starts at the last mark at or
    before ``lo``. Each verify step commits its accepted drafts and one token of the target's own, so the steps of
    the window are its tokens minus its accepted drafts; without drafts every token is a step."""
    inside = [i for i, m in enumerate(marks) if m[0] <= hi]
    if not inside:
        return None
    end = inside[-1]
    base = max((i for i in inside if marks[i][0] <= lo), default=0)
    (n0, ms0, t0, dn0, da0), (n1, ms1, t1, dn1, da1) = marks[base], marks[end]
    if n1 <= n0 or ms1 <= ms0:
        return None
    tokens, accepted = n1 - n0, da1 - da0
    steps = max(tokens - accepted, 1)
    # the gap before each token: a chunk's first token waited for the chunk, the others came with it
    gaps = sorted(g for i in range(base + 1, end + 1)
                  for g in [marks[i][2] - marks[i - 1][2]] + [0.0] * (marks[i][0] - marks[i - 1][0] - 1))
    out = {"from": n0, "to": n1, "tokens": tokens, "steps": steps, "draft_n": dn1 - dn0, "draft_n_accepted": accepted,
           "tok_s": tokens / (ms1 - ms0) * 1e3, "steps_s": steps / (ms1 - ms0) * 1e3,
           "client_tok_s": tokens / (t1 - t0) if t1 > t0 else None,
           "gap_p95_ms": gaps[math.ceil(0.95 * len(gaps)) - 1] * 1e3, "halves": None}
    nm, msm = marks[(base + end) // 2][:2]
    if ms0 < msm < ms1:
        out["halves"] = ((nm - n0) / (msm - ms0) * 1e3, (n1 - nm) / (ms1 - msm) * 1e3)
    return out


def window_text(pieces: list[tuple[int, str]], lo: int, hi: int) -> str:
    """The text of tokens ``lo + 1`` to ``hi``. A piece is (tokens so far when it arrived, its text)."""
    return "".join(text for n, text in pieces if lo < n <= hi)


def eos_strings(model_path: str) -> tuple[str, ...]:
    """The texts of the checkpoint's end-of-sequence tokens, from its tokenizer_config.json and generation_config.json;
    empty when they cannot be read (a .gguf or .ftw file). The server prints them like any other token."""
    root = Path(model_path)
    try:
        tok = json.loads((root / "tokenizer_config.json").read_text())
    except (OSError, ValueError):
        return ()
    found = {tok.get("eos_token")["content"] if isinstance(tok.get("eos_token"), dict) else tok.get("eos_token")}
    try:
        ids = json.loads((root / "generation_config.json").read_text()).get("eos_token_id")
    except (OSError, ValueError):
        ids = None
    added = tok.get("added_tokens_decoder") or {}
    for i in ids if isinstance(ids, list) else [ids]:
        found.add((added.get(str(i)) or {}).get("content"))
    return tuple(sorted(x for x in found if isinstance(x, str) and x))


def stream_chat(origin: str, body: dict, floor: float | None = None, deadline: float | None = None) -> dict:
    """One streamed chat completion, as it arrived: ``marks`` (see window_stats), ``pieces`` (tokens so far, text),
    the usage, the last timings object and the finish reason. With ``floor``, a request whose first EARLY tokens
    ran slower than that is abandoned (``gave_up``): a collapsed hybrid arm would otherwise take hours to say the same.
    With ``deadline`` (a time.monotonic() value) the request ends there with an error, whether the server streams or is silent."""
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    marks: list[tuple] = []
    pieces: list[tuple[int, str]] = []
    out = {"marks": marks, "pieces": pieces, "usage": None, "timings": None, "finish_reason": None, "early": None, "gave_up": False,
           "clock": "server timings", "first_s": None}
    t0 = time.perf_counter()
    left = 3600 if deadline is None else deadline - time.monotonic()
    if left <= 0:
        raise RuntimeError("the request was not sent: its deadline had passed")
    try:
        resp = urllib.request.urlopen(req, timeout=left)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"request failed: HTTP {e.code}: {e.read()[:300]!r}") from e
    with resp:
        while True:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError("the request ran past its deadline")
                # urlopen's timeout counts from every read, so a server that streams and then goes silent would get
                # it all over again: what is left of the deadline goes on the socket before each read
                sock = getattr(getattr(resp.fp, "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(left)
            raw = resp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            now = time.perf_counter()
            if chunk.get("error"):
                raise RuntimeError(f"the server refused the request: {chunk['error'].get('message')}")
            if chunk.get("usage"):
                out["usage"] = chunk["usage"]
            t = chunk.get("timings")
            if t and t.get("predicted_n"):
                out["timings"] = t
                if not marks or t["predicted_n"] > marks[-1][0]:
                    marks.append((int(t["predicted_n"]), float(t["predicted_ms"]), now, int(t.get("draft_n", 0)), int(t.get("draft_n_accepted", 0))))
            for choice in chunk.get("choices", []):
                out["finish_reason"] = choice.get("finish_reason") or out["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if not text:
                    continue
                if out["first_s"] is None:
                    out["first_s"] = now - t0
                if not out["timings"]:
                    # no timings: one text event per decode step, on the client's clock
                    out["clock"] = "arrival times"
                    marks.append((len(pieces) + 1, (now - t0) * 1e3, now, 0, 0))
                pieces.append((marks[-1][0], text))
            if out["early"] is None and marks and marks[-1][0] >= EARLY and marks[-1][1] > marks[0][1]:
                out["early"] = (marks[-1][0] - marks[0][0]) / (marks[-1][1] - marks[0][1]) * 1e3
                if floor is not None and out["early"] < floor:
                    out["gave_up"] = True   # leaving the with-block closes the stream; the server cancels a request whose client left
                    return out
    if out["usage"] is None:
        raise RuntimeError("stream ended without a usage chunk; is this a FreeToken server?")
    return out


def chat_body(model_id: str, prompt: str, sampling: dict, max_tokens: int) -> dict:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,                 # a window of a fixed length; measure() marks where the answer ended
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,          # a timings object on every chunk
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }


def first_token_ms(raw: dict) -> float | None:
    t = raw["timings"]
    if t:
        return float(t.get("queue_ms", 0.0)) + float(t.get("prompt_ms", 0.0))
    return raw["first_s"] * 1e3 if raw["first_s"] is not None else None


def measure(raw: dict, warm: int, decode: int, warms: list[int], eos: tuple[str, ...] = ()) -> dict:
    """What one stream says of its timed window, tokens ``warm + 1`` to ``warm + decode``, and tok/s of the window after each of ``warms``."""
    marks, pieces = raw["marks"], raw["pieces"]
    if raw["gave_up"]:
        return {"tok_s": raw["early"], "early": raw["early"], "gave_up": True, "window": f"stopped after {EARLY} tokens", "ttft_ms": first_token_ms(raw),
                "halves": None, "repeat": None, "completion_tokens": None, "sha1": "-", "draft_n": 0, "draft_n_accepted": 0}
    stats = window_stats(marks, warm, warm + decode)
    if stats is None:
        raise RuntimeError(f"the response had no tokens to time after its first {warm}")
    timed = window_text(pieces, stats["from"], stats["to"])
    data = timed.encode()
    return {
        **stats, "gave_up": False, "early": raw["early"], "ttft_ms": first_token_ms(raw),
        "window": f"tokens {stats['from']}-{stats['to']}, {raw['clock']}",
        "completion_tokens": raw["usage"].get("completion_tokens"), "finish_reason": raw["finish_reason"],
        "sha1": hashlib.sha1("".join(text for _, text in pieces).encode()).hexdigest()[:12],
        # how well the timed text compresses: text that loops compresses far better than ordinary text
        "repeat": len(zlib.compress(data)) / len(data) if data else None,
        # ignore_eos decodes past the end of the answer; this is the token count at which an end-of-sequence text first arrived
        "eos_at": next((n for n, text in pieces if n <= stats["to"] and any(e in text for e in eos)), None),
        "by_warm": {str(w): (window_stats(marks, w, w + decode) or {}).get("tok_s") for w in warms},
        "text": timed,
    }


def timed_request(origin: str, model_id: str, prompt: str, sampling: dict, warm: int, decode: int, floor: float | None = None,
                  warms: list[int] | None = None, eos: tuple[str, ...] = (), deadline: float | None = None) -> dict:
    """One streamed chat completion of warm + decode tokens, thinking on, measured over its last ``decode`` tokens."""
    raw = stream_chat(origin, chat_body(model_id, prompt, sampling, warm + decode), floor, deadline)
    return measure(raw, warm, decode, warms or [warm], eos)


def concurrent_round(origin: str, model_id: str, prompts, sampling: dict, warm: int, decode: int,
                     warms: list[int] | None = None, eos: tuple[str, ...] = ()) -> dict:
    """All ``prompts`` at once. The rates are sums over the streams, each over its own timed window."""
    results: list = [None] * len(prompts)

    def one(i: int) -> None:
        try:
            results[i] = timed_request(origin, model_id, prompts[i][1], sampling, warm, decode, None, warms, eos)
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

    def total(key):
        return sum(r[key] or 0 for r in results)

    ended = [r["eos_at"] for r in results if r["eos_at"] is not None]
    repeats = [r["repeat"] for r in results if r["repeat"] is not None]
    return {"tok_s": total("tok_s"), "steps_s": total("steps_s"), "client_tok_s": total("client_tok_s"), "early": None, "gave_up": False,
            "streams": results, "window": results[0]["window"], "ttft_ms": statistics.median([r["ttft_ms"] or 0.0 for r in results]),
            "halves": None, "repeat": statistics.fmean(repeats) if repeats else None, "gap_p95_ms": max(r["gap_p95_ms"] for r in results),
            "tokens": total("tokens"), "steps": total("steps"), "draft_n": total("draft_n"), "draft_n_accepted": total("draft_n_accepted"),
            "completion_tokens": total("completion_tokens"), "finish_reason": ",".join(sorted({r["finish_reason"] for r in results if r["finish_reason"]})) or None,
            "eos_at": min(ended) if ended else None,
            "by_warm": {w: sum(r["by_warm"][w] or 0 for r in results) for w in results[0]["by_warm"]},
            "sha1": hashlib.sha1("".join(r["sha1"] for r in results).encode()).hexdigest()[:12]}


def runs_of(result: dict) -> list[dict]:
    """The results of the passes behind one column of a start."""
    return result.get("seeds") or [result]


def over_seeds(runs: list[dict]) -> dict:
    """One column of a start from the results of its passes: rates are means over the passes, counts are sums over
    them, and ``ci95`` is the half-width of the 95% interval of tok/s."""
    if runs[-1]["gave_up"]:
        return {**runs[-1], "seeds": runs}
    mean, half = mean_ci([r["tok_s"] for r in runs])

    def avg(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    def total(key):
        return sum(r[key] or 0 for r in runs)

    ttft = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
    # the passes after the first may have another warm-up (run_arm's warm_after): a window is averaged over the passes that have it
    by_warm = {}
    for w in dict.fromkeys(w for r in runs for w in r["by_warm"]):
        vals = [r["by_warm"][w] for r in runs if r["by_warm"].get(w)]
        by_warm[w] = mean_ci(vals) if vals else (None, None)
    return {"tok_s": mean, "ci95": half, "steps_s": avg("steps_s"), "client_tok_s": avg("client_tok_s"), "gap_p95_ms": avg("gap_p95_ms"),
            "tok_per_step": total("tokens") / total("steps"), "accept": total("draft_n_accepted") / total("draft_n") if total("draft_n") else None,
            "tokens": total("tokens"), "steps": total("steps"), "draft_n": total("draft_n"), "draft_n_accepted": total("draft_n_accepted"),
            "early": runs[0]["early"], "gave_up": False, "window": runs[0]["window"], "ttft_ms": statistics.median(ttft) if ttft else None,
            "halves": None, "repeat": avg("repeat"), "completion_tokens": total("completion_tokens"), "by_warm": by_warm,
            "sha1": runs[0]["sha1"] if len(runs) == 1 else hashlib.sha1("".join(r["sha1"] for r in runs).encode()).hexdigest()[:12],
            "seeds": runs}


def cpu_khz(pattern: str = CPUFREQ) -> dict | None:
    """Mean and minimum of the cores' current frequency in kHz; None where the kernel does not show it (most containers)."""
    vals = []
    for path in glob.glob(pattern):
        try:
            vals.append(int(Path(path).read_text()))
        except (OSError, ValueError):
            continue
    return {"mean": statistics.fmean(vals), "min": min(vals), "cores": len(vals)} if vals else None


def long_text(words: int, seed: int) -> str:
    """About ``words`` words of service-log text, the same on every machine for the same arguments. Two seeds differ
    from the first line on, so one prompt is never a cached prefix of another."""
    parts = ("scheduler", "cache manager", "tokenizer", "prefill queue", "page table", "expert cache", "detokenizer", "rate limiter",
             "health probe", "request router", "KV pool", "stream writer")
    verbs = ("admitted", "evicted", "retried", "flushed", "rejected", "resized", "reported", "queued", "released", "merged")
    things = ("request", "page", "batch", "slot", "prefix", "chunk", "stream", "token window", "expert", "snapshot")
    tails = ("after a timeout", "under memory pressure", "on the second attempt", "without an error", "while another batch was in flight",
             "because the budget was spent", "before the next step", "and logged a warning", "at the caller's request", "as planned")
    x = (seed * 2 + 1) & (2**64 - 1)
    lines, n, i = [], 0, 0
    while n < words:
        picks = []
        for _ in range(7):
            x = (x * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
            picks.append(x >> 33)
        i += 1
        line = (f"{i:06d} {picks[0] % 24:02d}:{picks[1] % 60:02d}:{picks[2] % 60:02d} the {parts[picks[3] % len(parts)]} {verbs[picks[4] % len(verbs)]} "
                f"{things[picks[5] % len(things)]} {picks[6] % 100000} {tails[(picks[6] >> 20) % len(tails)]}.")
        lines.append(line)
        n += len(line.split())
    return "\n".join(lines)


def context_turns(origin: str, model_id: str, sizes: list[int], sampling: dict) -> list[dict]:
    """Per size, one prompt of about that many tokens and CONTEXT_TURN tokens after it: prefill, first token and decode speed."""
    out = []
    per_token = None
    for tokens in sizes:
        try:
            if per_token is None:
                # a short probe sizes the prompts in this tokenizer's words per token; its text is no prefix of theirs
                probe = stream_chat(origin, chat_body(model_id, long_text(1500, 0), sampling, 1))
                per_token = 1500 / max(probe["usage"].get("prompt_tokens") or 1, 1)
            prompt = (long_text(int(tokens * per_token), tokens) + "\n\nThe lines above are a service log. Which component appears most often, "
                      "and what does it do most? Explain how you counted before you answer.")
            raw = stream_chat(origin, chat_body(model_id, prompt, sampling, CONTEXT_TURN))
        except RuntimeError as e:   # a prompt past the server's context or KV budget; the other columns stand
            out.append({"context": tokens, "error": str(e).splitlines()[0]})
            continue
        stats = window_stats(raw["marks"], 0, CONTEXT_TURN) or {}
        t = raw["timings"] or {}
        out.append({"context": tokens, "prompt_tokens": raw["usage"].get("prompt_tokens"), "cached_tokens": t.get("cache_n"),
                    "prefill_s": t["prompt_ms"] / 1e3 if "prompt_ms" in t else None, "first_token_s": raw["first_s"],
                    "tok_s": stats.get("tok_s"), "tokens": stats.get("tokens"), "finish_reason": raw["finish_reason"]})
    return out


def run_arm(ns, extra, arm: str, rate: float | None, single, together, sampling: dict, floors: list[float] | None = None,
            passes: int | None = None, warm_after: int | None = None, deadline: float | None = None, grace: float = 30.0) -> dict:
    """``warm_after``: the warm-up tokens of the passes after the first, which find the expert cache as the pass before left it.
    ``deadline`` (a time.monotonic() value): the server must be ready and every request done by then, and the server is
    gone ``grace`` seconds after it at the latest (SIGKILL where SIGTERM has not ended it 10 s before that)."""

    def left() -> float:
        return float("inf") if deadline is None else deadline - time.monotonic()

    passes = passes or ns.seeds
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-decode-{arm.replace(':', '-').replace('#', '-')}-", suffix=".log")
    cmd, env = serve_cmd(ns, extra, arm, rate, port)
    t_start = time.monotonic()
    columns = len(single) + (1 if together else 0)
    context: list[dict] = []
    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            wait_ready(origin, proc, log_path, min(ns.server_timeout, left()))
            ready_s = time.monotonic() - t_start
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            gpus = get_json(f"{origin}/v1/stats").get("gpus") or [{}]
            geometry = get_json(f"{origin}/v1/cache/status").get("geometry") or {}
            khz = {"before": cpu_khz()}
            runs: list[list[dict]] = [[] for _ in range(columns)]
            gave_up = False
            # pass after pass, not prompt after prompt: pass k is then the same requests in the same order however many passes there are
            for rep in range(1, passes + 1):
                warm, warms = (ns.warm, ns.warms) if rep == 1 or warm_after is None else (warm_after, [warm_after])
                for i, (_, text) in enumerate(single):
                    r = timed_request(origin, model_id, text, sampling, warm, ns.decode, floors[i] if floors else None, warms, ns.eos, deadline)
                    runs[i].append({**r, "rep": rep})
                    if r["gave_up"]:
                        gave_up = True
                        break
                if gave_up:
                    break
            if gave_up:
                # a collapsed arm: the other columns would only take longer to say the same
                last = next(r for col in runs for r in col if r["gave_up"])
                results = [over_seeds([last])] * columns
            else:
                for rep in range(1, passes + 1 if together else 1):
                    r = concurrent_round(origin, model_id, together, sampling, ns.warm, ns.decode, ns.warms, ns.eos)
                    runs[-1].append({**r, "rep": rep})
                results = [over_seeds(col) for col in runs]
                context = context_turns(origin, model_id, ns.context, sampling)
            khz["after"] = cpu_khz()
        except SERVER_GONE as e:
            raise RuntimeError(f"the server stopped answering ({type(e).__name__}: {e}); log: {log_path}") from e
        finally:
            stop_server(proc, max(0.0, min(90.0, left() + grace - 10.0)))
    time.sleep(ns.settle)  # let the driver reclaim VRAM before the next server
    key = free = resolved = threads = None
    capped = False
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if m := KEY_LINE.search(line):
            key, free = m["key"].strip(), int(m["free"])
        if m := AUTO_LINE.search(line):
            resolved = m["strategy"]
        if m := THREADS_LINE.search(line):
            threads = int(m[1])
        # a hybrid layer without an `ft bench bw` row fetches at most one expert per step
        capped |= "using a fixed fetch cap of 1" in line
    return {"arm": arm, "tok_s": [r["tok_s"] for r in results], "results": results, "key": key, "gpu_free_bytes": free,
            "cache_slots": int(geometry.get("moe_cache_size") or 0),
            "total_slots": int(geometry.get("num_experts") or 0) * int(geometry.get("num_moe_layers") or 0),
            "slot_bytes": int((geometry.get("unit_bytes") or {}).get("moe_per_expert") or 0),
            "capped": capped, "resolved": resolved, "gpu": gpus[0], "ready_s": ready_s, "total_s": time.monotonic() - t_start, "log": log_path, "cmd": cmd,
            "cpu_khz": khz, "cpu_threads": threads, "context": context}


def rule_input(arms: dict[str, dict], columns: list[str], guard: list[str]) -> dict:
    """What the rule reads: per start and column the tok/s of each pass, and which columns are guards."""
    return {"tok_s": {a: [[run["tok_s"] for run in runs_of(x)] for x in r["results"]] for a, r in arms.items()}, "columns": columns, "guard": guard}


def run_ab(ns, extra, rate: float | None, single, together, guard: list[str], sampling: dict) -> dict:
    label = f"cache rate {rate}" if rate is not None else "cache from the serve flags"
    names = [n for n, _ in single] + ([f"x{len(together)}"] if together else [])
    arms: dict[str, dict] = {}
    lost: dict[str, str] = {}
    floors: list[float] | None = None

    def start(arm: str, passes: int) -> None:
        nonlocal floors
        print(f"[bench] {label}: {arm} ...", flush=True)
        try:
            r = run_arm(ns, extra, arm, rate, single, together, sampling, None if arm.startswith("offload") else floors, passes)
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
              + f", {r['cache_slots']} slots, {r['results'][0]['window']}" + machine_line(r), flush=True)

    hybrids = [a for a in ns.arms if a.startswith("hybrid:")]
    start("offload", ns.seeds)
    for h in hybrids:
        start(h, ns.seeds)
    if "auto" in ns.arms:
        start("auto", ns.seeds)
    start("offload#2", ns.seeds)

    def close_call() -> str | None:
        """Why another round is worth it: some hybrid arm's lead reaches the margin and is not clear of zero."""
        tok = rule_input(arms, names, guard)
        for h in hybrids:
            v = decode_profile.verdict_for(tok, h.split(":", 1)[1])
            if v["again"]:
                return v["why"]
        return None

    rounds = 1
    while rounds < ns.rounds and (rounds < ns.min_rounds or close_call()):
        rounds += 1
        print(f"[bench] {label}: {close_call() or 'another round was asked for'}; round {rounds}", flush=True)
        # a new start replays the passes of the rounds before it, so only a longer one adds samples
        for h in hybrids:
            if h not in lost:
                start(f"{h}#{rounds}", ns.seeds * rounds)
        start(f"offload#{rounds + 1}", ns.seeds * rounds)
    verdict = decode_profile.verdict_for(rule_input({a: r for a, r in arms.items() if a != "auto"}, names, guard), None)
    keys = {r["key"] for r in arms.values()}
    key = next(iter(keys)) if len(keys) == 1 else None
    return {"rate": rate, "arms": arms, "verdict": verdict, "key": key, "keys_seen": sorted(k for k in keys if k), "rounds": rounds, "columns": names, "guard": guard, "lost": lost}


def machine_line(r: dict) -> str:
    """What a start says of the CPU it ran on: the worker threads the server took, the cores' frequency before and after."""
    out = f", {r['cpu_threads']} CPU MoE threads" if r.get("cpu_threads") else ""
    khz = r.get("cpu_khz") or {}
    if khz.get("before") and khz.get("after"):
        out += ", CPU GHz mean/min " + " -> ".join(f"{khz[k]['mean'] / 1e6:.2f}/{khz[k]['min'] / 1e6:.2f}" for k in ("before", "after"))
    return out


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
    cols = ab["columns"]
    each = [(a, cols[i], x, run) for a, r in arms.items() for i, x in enumerate(r["results"]) for run in runs_of(x)]
    looped = sorted({c for _, c, _, run in each if run["repeat"] is not None and run["repeat"] < 0.12 and c not in ab["guard"]})
    if looped:
        out.append(f"the timed text of {', '.join(looped)} repeats itself, like the guard prompts: it is not the workload this column stands for.")
    moved = [a for a, _, _, run in each if run["halves"] and abs(run["halves"][1] / run["halves"][0] - 1.0) > 0.15]
    if moved:
        out.append(f"the speed moved by more than 15% inside the timed window of {', '.join(sorted(set(moved)))}; a longer --decode gives a steadier number.")
    by_arm: dict[str, list[list[float | None]]] = {}
    for a, r in arms.items():
        by_arm.setdefault(a.split("#")[0], []).append([x.get("repeat") for x in r["results"]])
    out += compress_notes(cols, {name: [statistics.fmean(v) if (v := [x for x in col if x is not None]) else None for col in zip(*starts)]
                                 for name, starts in by_arm.items()})
    ended = sorted({f"{c} ({a}, pass {run.get('rep')}, token {run['eos_at']})" for a, c, _, run in each if run.get("eos_at") is not None})
    if ended:
        out.append(f"the model ended its answer inside or before the timed window of {', '.join(ended)}; the bench decodes on past it (ignore_eos), and what follows is not an answer.")
    cut = sorted({f"{c} ({a}: {run['finish_reason']})" for a, c, _, run in each if run.get("finish_reason") not in (None, "length")})
    if cut:
        out.append(f"a request did not run to its token limit: {', '.join(cut)}.")
    same = sorted({c for _, c, x, _ in each if len({run["sha1"] for run in runs_of(x)}) < len(runs_of(x))})
    if same:
        out.append(f"passes of {', '.join(same)} decoded the same text (greedy sampling, or a server that restarts its sampler per request): their interval is of the timing alone.")
    for name in by_arm:
        hashes = [[[run["sha1"] for run in runs_of(x)] for x in r["results"]] for a, r in arms.items() if a.split("#")[0] == name]
        # the last column, when it is the concurrent one, never repeats: its batches form by arrival
        single = cols[:-1] if re.fullmatch(r"x\d+", cols[-1]) else cols
        # a start of a later round makes more passes: compared are the passes every start made
        shared = min(len(col) for h in hashes for col in h)
        differ = [c for i, c in enumerate(single) if len({tuple(h[i][:shared]) for h in hashes if i < len(h)}) > 1]
        if differ:
            out.append(f"the starts of {name} did not decode the same text in {', '.join(differ)}: pass k was not the same draws in each of them.")
    threads = {a: r.get("cpu_threads") for a, r in arms.items() if r.get("cpu_threads")}
    if len(set(threads.values())) > 1:
        out.append("the servers ran with different numbers of CPU MoE threads: " + ", ".join(f"{a} {n}" for a, n in threads.items()) + ".")
    return out


def compress_notes(columns: list[str], by_arm: dict[str, list[float | None]], limit: float = 0.10) -> list[str]:
    """A note per column on which the timed text of two arms compresses more than ``limit`` apart: text that compresses
    differently is a different kind of text, and its speed is not the arm's alone. ``by_arm`` holds one ratio per column."""
    out = []
    for i, name in enumerate(columns):
        vals = {arm: v[i] for arm, v in by_arm.items() if i < len(v) and v[i]}
        if len(vals) >= 2 and max(vals.values()) / min(vals.values()) - 1.0 > limit:
            lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
            out.append(f"the timed text of {name} compresses to {vals[lo]:.2f} of its size under {lo} and {vals[hi]:.2f} under {hi}, "
                       f"more than {limit:.0%} apart: the arms decoded different kinds of text there.")
    return out


def column_means(arms: dict[str, dict], name: str) -> list[float]:
    """Per column, the mean over the passes of one arm as the rule takes it (``offload`` covers offload, offload#2, ...)."""
    return decode_profile.arm_means(rule_input(arms, [], [])["tok_s"])[name]


def worst_vs_offload(ab: dict, tok_s: list[float]) -> float:
    """Over the workload columns, the smallest ratio to offload's mean, minus one."""
    off = column_means(ab["arms"], "offload")
    return min((v / o for n, v, o in zip(ab["columns"], tok_s, off) if n not in ab["guard"]), default=float("nan")) - 1.0   # nan: every column is a guard


def report(ab: dict) -> None:
    arms = ab["arms"]
    names = ab["columns"]
    slots, _ = slots_of(arms)
    print(f"\n==== decode A/B, {'cache rate ' + str(ab['rate']) if ab['rate'] is not None else 'cache from the serve flags'}"
          f", cache slots {' / '.join(f'{k} {v}' for k, v in slots.items()) or '?'}, {ab['rounds']} round(s) ====")
    print(f"  {'server start':22s}" + "".join(f"{n + '*' if n in ab['guard'] else n:>10s}" for n in names)
          + "   worst workload column vs offload   first token   output sha1 (per column)")
    for a, r in arms.items():
        if r["results"][0]["gave_up"]:
            print(f"  {a:22s}" + f"  stopped after {EARLY} tokens at {r['tok_s'][0]:.1f} tok/s, under half of offload's speed over the same tokens")
            continue
        ttft = [x["ttft_ms"] for x in r["results"] if x["ttft_ms"] is not None]
        first = f"{statistics.median(ttft):8.0f} ms" if ttft else f"{'-':>11s}"
        print(f"  {a:22s}" + "".join(f"{v:10.1f}" for v in r["tok_s"]) + f"   {worst_vs_offload(ab, r['tok_s']):+31.1%}   {first}   "
              + " ".join(x["sha1"] for x in r["results"]))
    report_windows(ab)
    for note in notes_for(ab):
        print(f"  note: {note}")
    v = ab["verdict"]
    print(f"  best arm: {v['strategy']}" + (f" ({v['round']} rounding)" if v["round"] else "") + f". {v['why']}."
          + (f" Largest timing spread between two starts of one arm: {v['spread']:.1%}." if v["spread"] is not None else ""))
    # per hybrid arm: the lead on each workload column with its one-sided 95% lower bound, and why it does not qualify
    tok = rule_input(arms, names, ab["guard"])
    for mode in dict.fromkeys(a.split("#")[0].split(":", 1)[1] for a in arms if a.startswith("hybrid:")):
        w = decode_profile.verdict_for(tok, mode)
        flag = "--moe-strategy hybrid" + ("" if mode == "min-wait" else f" with FREETOKEN_HYBRID_FETCH_ROUND={mode}")
        print(f"  {mode} rounding: {w['why']}" + (f". It qualifies: {flag}" if w["strategy"] == "hybrid" else ""))
    if "auto" in arms:
        # what the auto server did against the faster of the explicit strategies
        others = {a.split("#")[0] for a in arms} - {"auto"}
        best = [max(column_means(arms, name)[c] for name in others) for c in range(len(names))]
        gap = min((v / b for n, v, b in zip(names, arms["auto"]["tok_s"], best) if n not in ab["guard"]), default=float("nan")) - 1
        off = column_means(arms, "offload")
        kept = min((v / o for n, v, o in zip(names, arms["auto"]["tok_s"], off) if n in ab["guard"]), default=None)
        print(f"  the auto server resolved to {arms['auto']['resolved'] or 'an unknown strategy'}; on its worst workload column it is {gap:+.1%} "
              "from the faster of the other arms" + (f", and on its worst guard column {kept - 1:+.1%} from offload" if kept is not None else ""))


def _num(v, spec: str, width: int) -> str:
    return f"{v:{width}{spec}}" if v is not None else f"{'-':>{width}s}"


def report_windows(ab: dict) -> None:
    """Per start and column, the timed window alone: the mean over the passes, then tok/s by warm-up and the long prompts."""
    live = {a: r for a, r in ab["arms"].items() if not r["results"][0]["gave_up"] and "tokens" in r["results"][0]}
    if not live:
        return
    counts = sorted({len(runs_of(x)) for r in live.values() for x in r["results"]})
    passes = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
    print(f"  the timed window by column, mean over {passes} pass(es): tok/s and its 95% interval, verify or decode steps per second, tokens per step,\n"
          "  accepted share of the drafts, tok/s by arrival at the client and the 95th percentile of the gap before a token")
    print(f"  {'server start':22s}{'column':>10s}{'tok/s':>9s}{'+-95%':>8s}{'steps/s':>9s}{'tok/step':>9s}{'accepted':>9s}{'client':>9s}{'p95 gap':>10s}{'compress':>9s}")
    for a, r in live.items():
        for n, x in zip(ab["columns"], r["results"]):
            print(f"  {a:22s}{n:>10s}{x['tok_s']:9.1f}{_num(x['ci95'], '.1f', 8)}{_num(x['steps_s'], '.1f', 9)}{x['tok_per_step']:9.2f}"
                  f"{_num(x['accept'], '.0%', 9)}{_num(x['client_tok_s'], '.1f', 9)}{_num(x['gap_p95_ms'], '.1f', 7)} ms{_num(x['repeat'], '.2f', 9)}")
    warms = list(next(iter(live.values()))["results"][0]["by_warm"])
    if len(warms) > 1:
        print("  tok/s over the window after each warm-up, mean over the passes:")
        print(f"  {'server start':22s}{'column':>10s}" + "".join(f"{'warm ' + w:>12s}" for w in warms))
        for a, r in live.items():
            for n, x in zip(ab["columns"], r["results"]):
                print(f"  {a:22s}{n:>10s}" + "".join(_num(x["by_warm"][w][0], ".1f", 12) for w in warms))
    if any(r.get("context") for r in live.values()):
        print(f"  one {CONTEXT_TURN}-token turn after a long prompt:")
        print(f"  {'server start':22s}{'asked':>9s}{'prompt tokens':>15s}{'prefill s':>11s}{'first token s':>15s}{'decode tok/s':>14s}")
        for a, r in live.items():
            for c in r.get("context") or []:
                if "error" in c:
                    print(f"  {a:22s}{c['context']:>9d}   did not run: {c['error']}")
                else:
                    print(f"  {a:22s}{c['context']:>9d}{_num(c['prompt_tokens'], 'd', 15)}{_num(c['prefill_s'], '.2f', 11)}{_num(c['first_token_s'], '.2f', 15)}{_num(c['tok_s'], '.1f', 14)}")


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
    stamp = decode_profile.stamp(any_arm["gpu"].get("name"), any_arm["gpu"].get("uuid"))
    if stamp is None:
        print("  not filed: decode rows are filed and read on Linux only (this machine's memory cannot be read from /proc)")
        return
    row = {
        **rule_input({a: r for a, r in arms.items() if a != "auto"}, ab["columns"], ab["guard"]),   # tok_s: every pass of every start
        "slots": slots,
        "total_slots": any_arm["total_slots"],
        "slot_bytes": any_arm["slot_bytes"],
        "gpu_free_bytes": min(free) if free else None,
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "stamp": stamp,
        "verdict": ab["verdict"],
        "bench": {"warm": ns.warm, "decode": ns.decode, "seeds": ns.seeds, "rounds": ab["rounds"], "sampling": sampling_source, "notes": notes_for(ab),
                  "ci95": {a: [x.get("ci95") for x in r["results"]] for a, r in arms.items() if a != "auto"}},
    }
    # filed without a hybrid arm too: an earlier row of this setup on which hybrid qualified must not outlive a bench where it did not run
    dst = decode_profile.store(ab["key"], row, gpu=any_arm["gpu"], path=ns.profile)
    print(f"  filed for {ab['key']}\n  in {dst}")
    if no_hybrid:
        print("  --moe-strategy auto stays on offload for this setup on this GPU and release.")
        return
    modes = [m for m in decode_profile.ROUND_MODES if decode_profile.verdict_for(row, m)["strategy"] == "hybrid"]
    if modes:
        print(f"  --moe-strategy auto now takes hybrid for this setup on this GPU and release when the server runs {' or '.join(modes)} rounding\n"
              "  (FREETOKEN_HYBRID_FETCH_ROUND; min-wait without it). Hybrid computes some experts on the CPU: where two tokens are nearly\n"
              "  tied its output can differ from offload's. --moe-strategy offload keeps offload.")


def main(argv: list[str] | None = None, prog: str = "ft bench decode") -> int:
    ns, extra = parse_args(argv, prog)
    single, together, guard = load_prompts(ns)
    sampling, sampling_source = resolve_sampling(ns.model, ns.greedy)
    ns.eos = eos_strings(ns.model)
    print(f"[bench] model {ns.model}; arms {ns.arms}; columns {[n for n, _ in single]}" + (f" + {len(together)} streams at once" if together else "") + f"; guard {guard}"
          + f"; {ns.warm} warm + {ns.decode} timed tokens each, {ns.seeds} pass(es)" + (f"; tok/s also after {ns.warms[:-1]} warm tokens" if len(ns.warms) > 1 else "")
          + (f"; prompts of {ns.context} tokens" if ns.context else "") + f"; sampling: {sampling_source}; extra serve flags: {' '.join(extra) or 'none'}", flush=True)
    if not ns.eos:
        print("[bench] no end-of-sequence text found for this checkpoint: an answer that ends inside a timed window will not be marked", flush=True)
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
                    f.write("cache_rate,cache_slots,arm,prompt,tok_s,verdict,sha1,ttft_ms,ci95,steps_s,tok_per_step,accepted,client_tok_s,gap_p95_ms\n")
                for a, r in ab["arms"].items():
                    for n, x in zip(ab["columns"], r["results"]):
                        ttft = "" if x["ttft_ms"] is None else f"{x['ttft_ms']:.1f}"
                        more = ",".join("" if x.get(k) is None else f"{x[k]:.4g}" for k in ("ci95", "steps_s", "tok_per_step", "accept", "client_tok_s", "gap_p95_ms"))
                        f.write(f"{'' if rate is None else rate},{r['cache_slots']},{a},{n},{x['tok_s']:.3f},{ab['verdict']['strategy']},{x['sha1']},{ttft},{more}\n")
        summary.append((rate, slots.get("offload", 0), ab))
    if len(summary) > 1:
        first_col = summary[0][2]["columns"][0]
        print(f"\n==== by cache rate: offload's mean tok/s on {first_col}, and each other arm's worst workload column against offload ====")
        names = list(dict.fromkeys(a.split("#")[0] for _, _, ab in summary for a in ab["arms"]))
        modes = [n.split(":", 1)[1] for n in names if n.startswith("hybrid:")]
        print(f"  {'rate':>6s} {'slots':>7s} {'offload':>9s}" + "".join(f"{n:>19s}" for n in names if n != "offload") + "   rounds   "
              + ("auto server took" if "auto" in names else "what qualifies, by hybrid rounding mode"))
        for rate, slots_n, ab in summary:
            arms = ab["arms"]
            tok = rule_input(arms, ab["columns"], ab["guard"])
            ran = {a.split("#")[0] for a in arms}
            worst = {n: f"{worst_vs_offload(ab, column_means(arms, n)):+19.1%}" if n in ran else f"{'did not run':>19s}" for n in names if n != "offload"}
            if "auto" in names:
                takes = arms.get("auto", {}).get("resolved") or "?"
            else:
                takes = ", ".join(f"{m}: {decode_profile.verdict_for(tok, m)['strategy']}" for m in modes)
            print(f"  {rate:6.2f} {slots_n:7d} {column_means(arms, 'offload')[0]:9.1f}" + "".join(worst.values()) + f"   {ab['rounds']:6d}   {takes}")
    if ns.json:
        with open(ns.json, "w") as f:
            json.dump([{"cache_rate": rate, "warm": ns.warms, "decode": ns.decode, "seeds": ns.seeds,
                        **{k: ab[k] for k in ("columns", "guard", "rounds", "verdict", "key", "arms", "lost")}} for rate, _, ab in summary], f, indent=1)
    servers = sum(len(ab["arms"]) for _, _, ab in summary)
    print(f"\n[bench] {servers} server starts, {time.monotonic() - t0:.0f} s in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
