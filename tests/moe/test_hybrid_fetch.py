"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns the `ft bench bw`
row of each MoE layer kind into a fetch fraction, and `ft bench decode`: the rule that
reads its numbers, the timed window they come from, and the rows it files.
"""

import json
import os

import pytest

from freetoken.moe.bench_profile import PROFILE_VERSION, default_profile_path, fetch_fraction, layer_kind_key, load_layer_kinds, locked_fetch_fraction


def test_layer_kind_rows_give_fetch_fractions(tmp_path):
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    gguf = layer_kind_key("gguf", {"gate_up_type": 12, "down_type": 14}, 2048, 768, 8, 12)
    assert gguf == "gguf[down_type=14,gate_up_type=12]:2048x768:top8:t12"
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            gguf: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0, "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert fetch_fraction(rows[gguf]) == pytest.approx(0.25)
    # a locked layer's fetch adds a memcpy at the CPU MoE rate before the gather: p / (2p + c), from the same pair
    assert locked_fetch_fraction(rows[bf16]) == pytest.approx(40 / 180)
    assert locked_fetch_fraction(rows[gguf]) == pytest.approx(30 / 150)
    # another thread count is another kind
    assert layer_kind_key("bf16", {}, 2048, 768, 8, 6) not in rows
    # a profile from different hardware is ignored
    assert load_layer_kinds(gpu_name="OTHER", path=str(path)) is None
    # so is one of another version
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_layer_kinds(path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "layer_kinds": {"k": row}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "legacy")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "legacy"}
    assert load_layer_kinds(gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "card")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "card"}


def test_decode_ab_rule_takes_the_passes_as_samples_and_needs_a_lead_clear_of_zero():
    import statistics

    from freetoken.moe.decode_profile import decide, t95, verdict_for

    # textbook one-sided 95% Student quantiles, and the normal's where the degrees of freedom are many
    for df, t in ((4, 2.132), (7, 1.895), (10, 1.812), (30, 1.697), (1e6, 1.645)):
        assert t95(df) == pytest.approx(t, abs=5e-4)

    def passes(mean, sd, n=5):
        """n pass values with exactly this mean and this relative SD."""
        z = [k - (n - 1) / 2 for k in range(n)]
        return [mean * (1 + sd * x / statistics.stdev(z)) for x in z]

    # thresholds of the test's own, so that it holds whatever the defaults are set to
    rule = {"margin": 0.02, "max_spread": 0.06, "guard_loss": 0.08}
    offload = {"offload": [passes(100.0, 0.02), passes(210.0, 0.02)]}
    # +10% and +8% at 2% noise: clear. By hand, equal SDs over 5 passes give 8 degrees of freedom, t = 1.860
    verdict = decide({**offload, "hybrid:min-wait": [passes(110.0, 0.02), passes(226.8, 0.02)]}, **rule)
    assert verdict["strategy"] == "hybrid" and not verdict["again"] and verdict["lead"] == pytest.approx(0.08)
    assert verdict["low"] == pytest.approx(0.08 - 1.860 * 0.02 * (2 / 5) ** 0.5, abs=2e-5)
    # one column whose +3% is over the margin and inside the noise is enough to stay on offload, and worth more passes
    verdict = decide({**offload, "hybrid:min-wait": [passes(110.0, 0.02), passes(216.3, 0.04)]}, **rule)
    assert verdict["strategy"] == "offload" and verdict["lead"] == pytest.approx(0.03) and verdict["low"] < 0 and verdict["again"]
    # +1% is under the margin: not taken however steady it is, and no more passes are asked for, steady or not
    for sd in (0.001, 0.04):
        verdict = decide({"offload": [passes(100.0, 0.02), passes(210.0, sd)], "hybrid:min-wait": [passes(110.0, 0.02), passes(212.1, sd)]}, **rule)
        assert verdict["strategy"] == "offload" and (verdict["low"] > 0) == (sd == 0.001) and not verdict["again"]
    # two passes an arm give no interval: no lead qualifies on them
    verdict = decide({"offload": [[100.0, 101.0]], "hybrid:min-wait": [[150.0, 151.0]]}, **rule)
    assert verdict["strategy"] == "offload" and verdict["low"] is None and not verdict["again"]
    # two arms that measured the same: the same one is taken whichever order the row lists them in
    tied = {"hybrid:stochastic": [passes(110.0, 0.02), passes(226.8, 0.02)], "hybrid:min-wait": [passes(110.0, 0.02), passes(226.8, 0.02)]}
    assert decide({**offload, **tied}, **rule)["round"] == decide({**offload, **dict(reversed(list(tied.items())))}, **rule)["round"]

    # two starts of an arm decode the same texts, so they are one set of samples: each pass is its mean over the starts
    first, second, hybrid = [100.0, 110.0, 120.0], [102.0, 108.0, 121.0], {"hybrid:min-wait": [[118.0, 131.0, 140.0]]}
    averaged = decide({"offload": [[101.0, 109.0, 120.5]], **hybrid}, **rule)
    assert decide({"offload": [first], "offload#2": [second], **hybrid}, **rule)["low"] == pytest.approx(averaged["low"])
    # a start that made more passes adds those, and replays the others
    longer = decide({"offload": [first], "offload#2": [second + [130.0]], **hybrid}, **rule)
    assert longer["low"] == pytest.approx(decide({"offload": [[101.0, 109.0, 120.5, 130.0]], **hybrid}, **rule)["low"])
    # the same passes 7% slower in a second start is timing, over the limit: nothing is decided, whatever hybrid read
    slow = {"offload#2": [[v * 0.93 for v in col] for col in offload["offload"]]}
    verdict = decide({**offload, **slow, "hybrid:min-wait": [passes(150.0, 0.02), passes(300.0, 0.02)]}, **rule)
    assert verdict["strategy"] == "offload" and verdict["lead"] is None and verdict["spread"] == pytest.approx(7.0 / 96.5) and not verdict["again"]

    # column 0 is a workload, column 1 a guard: 5% behind there is inside the limit, 17% is not, whatever the lead on column 0
    offload = {"offload": [passes(100.0, 0.02), passes(230.0, 0.02)]}
    stochastic, min_wait = {"hybrid:stochastic": [passes(130.0, 0.02), passes(219.0, 0.02)]}, {"hybrid:min-wait": [passes(131.0, 0.02), passes(191.0, 0.02)]}
    assert decide({**offload, **stochastic}, guard=(1,), **rule)["strategy"] == "hybrid"
    verdict = decide({**offload, **min_wait}, guard=(1,), **rule)
    assert verdict["strategy"] == "offload" and verdict["guard"] == pytest.approx(191.0 / 230.0)
    # with both arms measured, the one inside the limit is taken, though the other leads by more
    assert decide({**offload, **min_wait, **stochastic}, guard=(1,), **rule)["round"] == "stochastic"
    # two guard columns: the worse one counts, whichever comes first
    two = {"offload": [passes(100.0, 0.02), passes(230.0, 0.02), passes(240.0, 0.02)]}
    for cols, want, worst in (((130.0, 219.0, 232.0), "hybrid", 219.0 / 230.0), ((130.0, 219.0, 211.0), "offload", 211.0 / 240.0),
                              ((130.0, 201.0, 232.0), "offload", 201.0 / 230.0)):
        verdict = decide({**two, "hybrid:stochastic": [passes(v, 0.02) for v in cols]}, guard=(1, 2), **rule)
        assert verdict["strategy"] == want and verdict["guard"] == pytest.approx(worst)

    # a row is read for the arm of one rounding mode, or for all of them: here one arm is far behind offload and one far ahead
    row = {"tok_s": {"offload": [passes(100.0, 0.02)], "hybrid:min-wait": [passes(80.0, 0.02)], "hybrid:stochastic": [passes(150.0, 0.02)]},
           "columns": ["aime"], "guard": []}
    assert verdict_for(row, "min-wait")["strategy"] == "offload"
    assert verdict_for(row, "stochastic")["strategy"] == "hybrid" and verdict_for(row, None)["round"] == "stochastic"

    # two cells measured on one desktop (mean and SD over 8 and over 4 passes), at the thresholds they were read with
    read = {"margin": 0.015, "max_spread": 0.05, "guard_loss": 0.10}
    verdict = decide({"offload": [passes(132.4, 0.014, 8), passes(149.5, 0.038, 8)],
                      "hybrid:min-wait": [passes(144.1, 0.022, 8), passes(157.2, 0.034, 8)]}, **read)
    assert verdict["strategy"] == "hybrid" and [low for _, _, low in verdict["leads"]] == pytest.approx([0.07, 0.02], abs=0.005)
    verdict = decide({"offload": [passes(113.1, 0.010, 4), passes(133.9, 0.027, 4)],
                      "hybrid:min-wait": [passes(123.3, 0.025, 4), passes(136.2, 0.011, 4)]}, **read)
    assert verdict["strategy"] == "offload" and 0.017 < verdict["lead"] < 0.018 and verdict["leads"][1][2] < 0 < verdict["leads"][0][2]


def _speculative_stream(warm_steps=11, timed_steps=5, k=3, step_ms=10.0):
    """A stream as a server with k drafts per step sends it: the first token, then ``warm_steps`` steps that accept
    nothing, then ``timed_steps`` steps that accept every draft. One chunk per step, 10 ms apart on both clocks."""
    marks, pieces = [(1, 0.0, 100.0, 0, 0)], [(1, "w1 ")]
    n, drafted, accepted = 1, 0, 0
    for step in range(1, warm_steps + timed_steps + 1):
        took = 0 if step <= warm_steps else k
        drafted, accepted = drafted + k, accepted + took
        text = "".join(f"w{i} " for i in range(n + 1, n + took + 2))
        n += took + 1
        marks.append((n, step * step_ms, 100.0 + step * step_ms / 1e3, drafted, accepted))
        pieces.append((n, text))
    return {"marks": marks, "pieces": pieces, "usage": {"completion_tokens": n}, "timings": {"queue_ms": 1.0, "prompt_ms": 2.0},
            "finish_reason": "length", "early": None, "gave_up": False, "clock": "server timings", "first_s": 0.003}


def test_bench_reads_the_timed_window_by_token_count_and_its_drafts_alone():
    from freetoken.moe import benchdecode

    raw = _speculative_stream()
    # 12 warm tokens in 12 chunks, then 20 timed tokens in 5 chunks of 4: 50 ms, 5 verify steps, all 15 drafts accepted
    out = benchdecode.measure(raw, warm=12, decode=20, warms=[0, 12])
    assert (out["from"], out["to"], out["tokens"], out["steps"]) == (12, 32, 20, 5)
    assert out["tok_s"] == pytest.approx(400.0) and out["steps_s"] == pytest.approx(100.0)
    # over the whole request 15 of 48 drafts were accepted; in the window all of them
    assert (out["draft_n"], out["draft_n_accepted"]) == (15, 15)
    # the last 20 chunks would reach back to token 2; the window's text is tokens 13 to 32 and nothing else
    assert out["text"].split() == [f"w{i}" for i in range(13, 33)]
    # by arrival the 20 tokens came in 5 bursts 10 ms apart: the same rate, and a gap that the mean does not show
    assert out["client_tok_s"] == pytest.approx(400.0) and out["gap_p95_ms"] == pytest.approx(10.0)
    # without a warm-up the window is tokens 2 to 20: 11 slow steps and 2 fast ones
    assert out["by_warm"]["12"] == pytest.approx(400.0) and out["by_warm"]["0"] == pytest.approx(19 / 0.130)
    # a window that starts inside a step starts at that step's first token
    assert benchdecode.window_stats(raw["marks"], 14, 32)["from"] == 12
    # an end-of-sequence text is reported with the token count of the chunk that carried it
    raw["pieces"][14] = (raw["pieces"][14][0], "w21 <|im_end|> w23 w24 ")
    assert benchdecode.measure(raw, 12, 20, [12], eos=("<|im_end|>",))["eos_at"] == 24
    assert benchdecode.measure(raw, 12, 20, [12], eos=("</s>",))["eos_at"] is None


def test_bench_column_is_the_mean_over_its_passes_with_a_student_interval():
    from freetoken.moe import benchdecode

    # textbook values: t(0.975, 4) = 2.776 and s = sqrt(10), t(0.975, 1) = 12.706 and s = sqrt(50)
    mean, half = benchdecode.mean_ci([10.0, 12.0, 14.0, 16.0, 18.0])
    assert mean == pytest.approx(14.0) and half == pytest.approx(2.776 * 10 ** 0.5 / 5 ** 0.5, rel=1e-4)
    assert benchdecode.mean_ci([100.0, 110.0])[1] == pytest.approx(63.53, rel=1e-3)
    assert benchdecode.mean_ci([100.0]) == (100.0, None)
    # the accepted share and the tokens per step of a column pool its passes; they are not means of per-pass ratios
    a = benchdecode.measure(_speculative_stream(), 12, 20, [12])
    b = {**a, "steps": 20, "draft_n": 60, "draft_n_accepted": 0}   # the same 20 tokens with every draft rejected: 20 steps of 3 drafts
    col = benchdecode.over_seeds([a, b])
    assert col["accept"] == pytest.approx(15 / 75) and col["tok_per_step"] == pytest.approx(40 / 25)
    assert col["tok_s"] == pytest.approx((a["tok_s"] + b["tok_s"]) / 2) and len(col["seeds"]) == 2


def test_bench_stops_the_whole_server_group_even_when_the_frontend_exits_at_once(monkeypatch):
    import signal
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    sent = []
    monkeypatch.setattr(benchdecode.os, "killpg", lambda pid, sig: sent.append(sig))
    benchdecode.stop_server(SimpleNamespace(pid=7, wait=lambda timeout: 0))
    assert sent == [signal.SIGTERM, signal.SIGKILL]

    # an exit raised inside the wait (a signal to the launcher) still kills the group
    def interrupted(timeout):
        raise SystemExit(143)

    del sent[:]
    with pytest.raises(SystemExit):
        benchdecode.stop_server(SimpleNamespace(pid=7, wait=interrupted))
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_bench_counts_an_arm_lost_in_a_later_round_as_not_faster(monkeypatch):
    """hybrid runs 30% faster in round 1, then its round-2 server stops answering: the verdict must not take it on the one start."""
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    asked = {}

    def run_arm(ns, extra, arm, rate, single, together, sampling, floors=None, passes=None):
        asked[arm] = passes
        if arm == "hybrid:min-wait#2":
            raise RuntimeError("the server stopped answering (RemoteDisconnected); log: /tmp/x.log")
        tok = 130.0 if arm.startswith("hybrid") else 100.0
        return {"arm": arm, "tok_s": [tok, 230.0], "results": [{"early": 80.0, "window": "w", "tok_s": tok, "repeat": None, "halves": None}] * 2,
                "key": "k", "cache_slots": 9, "capped": False, "ready_s": 1.0, "log": "/tmp/l"}

    monkeypatch.setattr(benchdecode, "run_arm", run_arm)
    ns = SimpleNamespace(arms=["offload", "hybrid:min-wait"], rounds=2, min_rounds=2, seeds=3)
    ab = benchdecode.run_ab(ns, [], 0.2, [("aime", "text"), ("repeat", "r")], None, ["repeat"], {})
    assert list(ab["lost"]) == ["hybrid:min-wait"] and not any(a.startswith("hybrid") for a in ab["arms"])
    assert ab["verdict"]["strategy"] == "offload"
    # a new start replays the texts of the round before, so a second round decodes as many passes again: only those are new samples
    assert (asked["offload"], asked["offload#2"], asked["hybrid:min-wait#2"], asked["offload#3"]) == (3, 3, 6, 6)


def test_bench_replaces_an_older_row_when_hybrid_did_not_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from freetoken.moe import benchdecode, decode_profile

    gpu = {"name": "GPU", "uuid": "GPU-1"}
    path = str(tmp_path / "rows.json")
    ns = SimpleNamespace(no_write=False, profile=path, warm=8, decode=8, seeds=3)
    monkeypatch.setattr(decode_profile, "stamp", lambda gpu_name=None, gpu_uuid=None: {"cpu": "the machine of this test"})

    def arm(tok):
        # three passes a column, 1 tok/s apart
        results = [{"tok_s": v, "sha1": "t", "repeat": None, "halves": None, "seeds": [{"tok_s": v + d, "sha1": "t", "repeat": None, "halves": None} for d in (-1.0, 0.0, 1.0)]}
                   for v in tok]
        return {"tok_s": tok, "results": results, "key": "k", "gpu_free_bytes": 1 << 30, "cache_slots": 40,
                "total_slots": 80, "slot_bytes": 1 << 20, "gpu": gpu, "capped": False}

    def ab(arms, lost=()):
        return {"arms": arms, "columns": ["aime", "repeat"], "guard": ["repeat"], "key": "k", "keys_seen": ["k"], "rounds": 1,
                "lost": {a: "the server did not start" for a in lost},
                "verdict": decode_profile.verdict_for(benchdecode.rule_input(arms, ["aime", "repeat"], ["repeat"]), None)}

    def stored():
        row, _ = decode_profile.lookup("k", gpu_name="GPU", gpu_uuid="GPU-1", now=decode_profile.stamp, path=path)
        return row, decode_profile.verdict_for(row, None)

    offload = {"offload": arm([100.0, 230.0]), "offload#2": arm([100.2, 230.5]), "offload#3": arm([100.1, 230.2])}
    benchdecode.file_row(ns, ab({**offload, "hybrid:min-wait": arm([131.0, 228.0])}), "test")
    # the row keeps every pass of every start, and the rule reads from the file what it read at the bench
    assert stored()[0]["tok_s"]["hybrid:min-wait"] == [[130.0, 131.0, 132.0], [227.0, 228.0, 229.0]]
    assert stored()[1]["strategy"] == "hybrid" and stored()[1]["low"] == pytest.approx(stored()[0]["verdict"]["low"])
    # the same setup benched again, and this time the hybrid server does not start: the row on which it qualified must not stay
    benchdecode.file_row(ns, ab(offload, lost=["hybrid:min-wait"]), "test")
    assert stored()[1]["strategy"] == "offload"
    # --no-write files nothing, and neither does a machine whose stamp cannot be read
    dry = SimpleNamespace(no_write=True, profile=str(tmp_path / "none.json"), warm=8, decode=8, seeds=3)
    benchdecode.file_row(dry, ab(offload, lost=["hybrid:min-wait"]), "test")
    monkeypatch.setattr(decode_profile, "stamp", lambda gpu_name=None, gpu_uuid=None: None)
    benchdecode.file_row(SimpleNamespace(**{**vars(dry), "no_write": False}), ab(offload), "test")
    assert not (tmp_path / "none.json").exists()


def test_decode_profile_store_keeps_the_rows_that_can_still_be_used(tmp_path):
    from freetoken.moe import decode_profile as dp

    path = str(tmp_path / "p.json")
    gpu = {"name": "A", "uuid": "GPU-a"}
    build = {"version": "1.0", "decode_profile": dp.DECODE_PROFILE_VERSION}
    old, new = {"freetoken": build, "torch": "2.13"}, {"freetoken": build, "torch": "2.14"}
    dp.store("k1", {"tok_s": {"offload": [[1.0]]}, "stamp": old}, gpu=gpu, path=path)
    dp.store("k2", {"tok_s": {"offload": [[2.0]]}, "stamp": old}, gpu=gpu, path=path)
    assert set(json.load(open(path))["decisions"]) == {"k1", "k2"}
    dp.store("k3", {"tok_s": {"offload": [[3.0]]}, "stamp": new}, gpu=gpu, path=path)  # rows of the old stamp can no longer match
    assert set(json.load(open(path))["decisions"]) == {"k3"}
    dp.store("k4", {"tok_s": {"offload": [[4.0, 4.5]]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert set(json.load(open(path))["decisions"]) == {"k4"}  # a file is for one GPU
    row, reason = dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=lambda: {"freetoken": build, "torch": "2.15"}, path=path)
    assert row is None and "torch" in reason
    assert dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0]["tok_s"] == {"offload": [[4.0, 4.5]]}
    # with no row for the key the stamp is never made: a start without a profile costs one failed open
    assert dp.lookup("k9", gpu_name="B", gpu_uuid="GPU-b", now=lambda: pytest.fail("stamp made without a row"), path=path)[0] is None
    # a row edited by hand into a shape a reader would trip over is not handed out: here an arm with a column missing
    dp.store("k5", {"tok_s": {"offload": [[5.0], [6.0]], "hybrid:min-wait": [[7.0]]}, "columns": ["aime", "code"], "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k5", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # nor one that holds a number per column where the passes belong, as rows did before the rule read passes
    dp.store("k5", {"tok_s": {"offload": [5.0], "hybrid:min-wait": [7.0]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k5", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # nor is one with a number no bench writes: json reads Infinity, and an infinite hybrid reading would win every rule
    dp.store("k7", {"tok_s": {"offload": [[5.0]], "hybrid:min-wait": [[float("inf")]]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k7", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # a damaged file reads as no row, and the next store replaces it
    with open(path, "w") as f:
        json.dump({"version": dp.DECODE_PROFILE_VERSION, "gpu": ["B"], "decisions": {}}, f)
    assert dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    dp.store("k8", {"tok_s": {"offload": [[8.0]]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k8", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0]["tok_s"] == {"offload": [[8.0]]}
def _resolve_with_profile(tmp_path, monkeypatch, layer_shapes, benched, threads=12):
    """Engine._resolve_hybrid_fetch on a stand-in engine: the layers have these (format, hidden, intermediate) kinds, the profile these rows."""
    from types import SimpleNamespace

    import freetoken.engine.engine as engine

    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps({"version": PROFILE_VERSION, "gpu": {"name": "FAKE GPU"}, "layer_kinds": benched}))
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", str(path))
    methods = [
        SimpleNamespace(cpu_spec=lambda fmt=fmt: (fmt, {}), cfg=SimpleNamespace(hidden=h, local_intermediate=i, top_k=8))
        for fmt, h, i in layer_shapes
    ]
    monkeypatch.setattr(engine, "offload_layer_methods", lambda model: methods)
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: ("FAKE GPU", None))
    eng = SimpleNamespace(model=None, device=SimpleNamespace(index=0), cpu_moe_executor=SimpleNamespace(num_threads=threads, isa="avx512"))
    eng._profile_layer_rows = lambda: engine.Engine._profile_layer_rows(eng)
    cache = SimpleNamespace(hybrid_max_fetch=None, hybrid_fetch_fraction=None, locked_hybrid=frozenset())
    engine.Engine._resolve_hybrid_fetch(eng, SimpleNamespace(moe_hybrid_max_fetch=-1), cache)
    return cache


def test_engine_takes_the_nearest_benched_shape_but_not_a_far_one(tmp_path, monkeypatch):
    def row(cpu, pcie):
        return {"cpu_moe_overlap_gbs": cpu, "pcie_gather_overlap_gbs": pcie}

    benched = {
        layer_kind_key("nvfp4", {}, 3072, 1536, 8, 12): row(90.0, 30.0),   # 4.5x the bytes of 2048x512: 25% fetched
        layer_kind_key("nvfp4", {}, 2048, 768, 8, 12): row(60.0, 30.0),    # 1.5x: 33.3%
        layer_kind_key("bf16", {}, 2048, 512, 8, 12): row(30.0, 30.0),     # right shape, wrong format: 50%
        layer_kind_key("nvfp4", {}, 2048, 512, 8, 6): row(10.0, 30.0),     # right shape, half the threads: 75%
    }
    cache = _resolve_with_profile(tmp_path, monkeypatch, [("nvfp4", 2048, 512), ("nvfp4", 2816, 704), ("mxfp4", 2048, 512)], benched)
    # 2048x512 and 2816x704 (1.9x and 1.26x from 2048x768) take the 2048x768 row; mxfp4 has no row of its format and keeps cap 1
    assert cache.hybrid_fetch_fraction == [pytest.approx(1 / 3), pytest.approx(1 / 3), 0.0]
    assert cache.hybrid_max_fetch == 1
    # with only the far shape benched, a 2048x512 layer keeps the cap instead of the split of a 4.5x larger expert
    far = {layer_kind_key("nvfp4", {}, 3072, 1536, 8, 12): row(90.0, 30.0)}
    assert _resolve_with_profile(tmp_path, monkeypatch, [("nvfp4", 2048, 512)], far).hybrid_fetch_fraction == [0.0]
    # a row of the right shape benched at half the threads is not a stand-in: the split moves with the thread count
    half = {layer_kind_key("nvfp4", {}, 2048, 512, 8, 6): row(10.0, 30.0)}
    assert _resolve_with_profile(tmp_path, monkeypatch, [("nvfp4", 2048, 512)], half).hybrid_fetch_fraction == [0.0]
    # an explicit --moe-cpu-threads 13 finds the row benched at the default 12 workers
    cache = _resolve_with_profile(tmp_path, monkeypatch, [("nvfp4", 2048, 768)], benched, threads=13)
    assert cache.hybrid_fetch_fraction == [pytest.approx(1 / 3)]
    # an exact row always wins over a nearer-looking neighbour
    exact = dict(benched)
    exact[layer_kind_key("nvfp4", {}, 2048, 512, 8, 12)] = row(20.0, 30.0)
    assert _resolve_with_profile(tmp_path, monkeypatch, [("nvfp4", 2048, 512)], exact).hybrid_fetch_fraction == [pytest.approx(0.6)]
