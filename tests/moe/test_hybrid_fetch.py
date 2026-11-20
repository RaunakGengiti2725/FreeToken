"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns the `ft bench bw`
row of each MoE layer kind into a fetch fraction.
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
    # with only the far shape benched, a 2048x512 layer keeps the cap instead of the split of a 4.5x larger expert (#822)
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


def test_decode_ab_rule_needs_a_margin_on_the_worst_column():
    from freetoken.moe.decode_profile import decide

    offload = {"offload": [100.0, 100.0, 210.0], "offload#2": [100.4, 100.2, 211.0]}
    assert decide({**offload, "hybrid:min-wait": [131.0, 128.0, 260.0]})["strategy"] == "hybrid"
    # +1% on every column is inside the margin
    assert decide({**offload, "hybrid:min-wait": [101.2, 101.3, 213.0]})["strategy"] == "offload"
    # one slower column is enough to stay on offload: here the round of concurrent streams
    verdict = decide({**offload, "hybrid:min-wait": [140.0, 138.0, 200.0]})
    assert verdict["strategy"] == "offload" and verdict["ratio"] < 1.0
    # #278: hybrid at 27 tok/s where offload gave 84
    assert decide({"offload": [84.0], "offload#2": [83.5], "hybrid:min-wait": [27.0]})["strategy"] == "offload"
    # two arms that measured the same: the same one is taken whichever order the row lists them in
    tied = {"hybrid:stochastic": [131.0, 128.0, 260.0], "hybrid:min-wait": [131.0, 128.0, 260.0]}
    assert decide({**offload, **tied})["round"] == decide({**offload, **dict(reversed(list(tied.items())))})["round"]


def test_decode_ab_rule_does_not_decide_on_unsettled_starts():
    from freetoken.moe.decode_profile import decide

    # 7% between two offload starts: nothing is decided, and the bench is asked for another round
    verdict = decide({"offload": [100.0, 100.0], "offload#2": [93.0, 100.0], "hybrid:min-wait": [150.0, 150.0]})
    assert verdict["strategy"] == "offload" and verdict["again"]
    # 3% between starts raises the bar to that spread
    verdict = decide({"offload": [100.0], "offload#2": [103.0], "hybrid:min-wait": [104.0]})
    assert verdict["strategy"] == "offload" and verdict["need"] == pytest.approx(3.0 / 101.5)
    # a spread exactly at the limit is not above it: the bar is the spread; one hair over and nothing is decided
    at_limit = {"offload": [100.0], "offload#2": [110.0], "hybrid:min-wait": [120.0]}
    settled = decide(at_limit, max_spread=10.0 / 105.0)
    assert settled["strategy"] == "hybrid" and settled["need"] == pytest.approx(10.0 / 105.0)
    over = decide(at_limit, max_spread=10.0 / 105.0 - 1e-9)
    assert over["strategy"] == "offload" and over["again"] and "too unsettled" in over["why"]


def test_decode_ab_rule_bounds_the_loss_on_a_guard_column():
    from freetoken.moe.decode_profile import decide, verdict_for

    # column 0 is a workload, column 1 is output that barely misses the cache
    offload = {"offload": [100.0, 230.0], "offload#2": [100.4, 231.0]}
    stochastic, min_wait = {"hybrid:stochastic": [130.0, 219.0]}, {"hybrid:min-wait": [131.0, 191.0]}
    # hybrid pays its CPU round trip there and gains nothing: 5% behind is accepted
    assert decide({**offload, **stochastic}, guard=(1,))["strategy"] == "hybrid"
    # 17% behind is a cache that never fills, and no lead on the workload column buys that
    verdict = decide({**offload, **min_wait}, guard=(1,))
    assert verdict["strategy"] == "offload" and verdict["guard"] < 0.92
    # with both arms measured, the one inside the bound is taken, though the other leads by more
    row = {"tok_s": {**offload, **min_wait, **stochastic}, "columns": ["aime", "repeat"], "guard": ["repeat"]}
    assert verdict_for(row, None)["round"] == "stochastic"
    # and a server is judged on the arm of the rounding mode it runs
    assert verdict_for(row, "min-wait")["strategy"] == "offload"
    assert verdict_for(row, "stochastic")["strategy"] == "hybrid"

    # two guard columns: the worse one counts, whichever comes first
    two = {"offload": [100.0, 230.0, 240.0], "offload#2": [100.4, 231.0, 240.5]}
    for hybrid, want, worst in (([130.0, 219.0, 232.0], "hybrid", 219.0 / 230.5), ([130.0, 219.0, 211.0], "offload", 211.0 / 240.25),
                                ([130.0, 201.0, 232.0], "offload", 201.0 / 230.5)):
        verdict = verdict_for({"tok_s": {**two, "hybrid:stochastic": hybrid}, "columns": ["aime", "repeat", "repeat-2"], "guard": ["repeat", "repeat-2"]}, None)
        assert verdict["strategy"] == want and verdict["guard"] == pytest.approx(worst)


def test_bench_reads_two_repeat_prompts_as_guard_columns_by_default(monkeypatch):
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    monkeypatch.setattr(benchdecode, "load_aime", lambda path: [f"problem {i}" for i in range(6)])
    single, together, guard = benchdecode.load_prompts(SimpleNamespace(prompts=None, aime=None, streams=4))
    assert [name for name, _ in single] == ["aime", "code", "repeat", "repeat-2"] and guard == ["repeat", "repeat-2"]
    # two different texts to repeat: one repeated sentence does not stand for another
    texts = dict(single)
    assert texts["repeat"] != texts["repeat-2"] and texts["repeat"].split("\n\n")[1] != texts["repeat-2"].split("\n\n")[1]
    # the concurrent column decodes workload prompts only
    assert len(together) == 4 and not {text for _, text in together} & {texts["repeat"], texts["repeat-2"]}
    # without AIME the built-in math prompt takes its place, and the guards stay
    monkeypatch.setattr(benchdecode, "load_aime", lambda path: None)
    single, _, guard = benchdecode.load_prompts(SimpleNamespace(prompts=None, aime=None, streams=1))
    assert [name for name, _ in single] == ["math", "code", "repeat", "repeat-2"] and guard == ["repeat", "repeat-2"]


def test_bench_servers_log_at_info_and_a_row_of_guard_columns_only_is_reported(monkeypatch):
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    # the bench reads a server's setup from its INFO lines: a quieter level in the caller's shell must not hide them
    ns = SimpleNamespace(server_cmd=None, model="/m", gpu=None)
    for level, want in (("WARNING", "INFO"), ("error", "INFO"), ("DEBUG", "DEBUG"), ("INFO", "INFO")):
        monkeypatch.setenv("LOG_LEVEL", level)
        assert benchdecode.serve_cmd(ns, [], "offload", 0.2, 1234)[1]["LOG_LEVEL"] == want
    monkeypatch.delenv("LOG_LEVEL")
    assert "LOG_LEVEL" not in benchdecode.serve_cmd(ns, [], "offload", 0.2, 1234)[1]
    # a prompts file with guard columns only leaves no workload column: the table still prints
    ab = {"arms": {"offload": {"tok_s": [230.0]}, "offload#2": {"tok_s": [231.0]}}, "columns": ["repeat"], "guard": ["repeat"]}
    assert benchdecode.worst_vs_offload(ab, [220.0]) != benchdecode.worst_vs_offload(ab, [220.0])   # nan: nothing to compare


def test_bench_refuses_a_prompts_file_with_no_prompts(tmp_path):
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    p = tmp_path / "p.jsonl"
    p.write_text("\n  \n")
    with pytest.raises(SystemExit, match="has no prompts"):
        benchdecode.load_prompts(SimpleNamespace(prompts=str(p), aime=None, streams=4))


def test_bench_decodes_a_prompts_file_in_the_concurrent_column_too(tmp_path, monkeypatch):
    """--prompts names prompts the bench never used: the concurrent column cycles its lines that are not guards, not AIME."""
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    monkeypatch.setattr(benchdecode, "load_aime", lambda path: pytest.fail("AIME loaded although --prompts names the prompts"))
    p = tmp_path / "p.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ({"name": "a", "prompt": "pa"}, {"name": "g", "prompt": "pg", "guard": True}, {"name": "b", "prompt": "pb"})))
    single, together, guard = benchdecode.load_prompts(SimpleNamespace(prompts=str(p), aime=None, streams=5))
    assert [n for n, _ in single] == ["a", "g", "b"] and guard == ["g"]
    assert [t for _, t in together] == ["pa", "pb", "pa", "pb", "pa"]


def test_bench_stops_the_whole_server_group_even_when_the_frontend_exits_at_once(monkeypatch):
    import signal
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    sent = []
    monkeypatch.setattr(benchdecode.os, "killpg", lambda pid, sig: sent.append(sig))
    benchdecode.stop_server(SimpleNamespace(pid=7, wait=lambda timeout: 0))
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_bench_counts_an_arm_lost_in_a_later_round_as_not_faster(monkeypatch):
    """hybrid runs 30% faster in round 1, then its round-2 server stops answering: the verdict must not take it on the one start."""
    from types import SimpleNamespace

    from freetoken.moe import benchdecode

    def run_arm(ns, extra, arm, rate, single, together, sampling, floors=None):
        if arm == "hybrid:min-wait#2":
            raise RuntimeError("the server stopped answering (RemoteDisconnected); log: /tmp/x.log")
        tok = 130.0 if arm.startswith("hybrid") else 100.0
        return {"arm": arm, "tok_s": [tok, 230.0], "results": [{"early": 80.0, "window": "w", "tok_s": tok, "repeat": None, "halves": None}] * 2,
                "key": "k", "cache_slots": 9, "capped": False, "ready_s": 1.0, "log": "/tmp/l", "off": []}

    monkeypatch.setattr(benchdecode, "run_arm", run_arm)
    ns = SimpleNamespace(arms=["offload", "hybrid:min-wait"], asked_arms=True, rounds=2, min_rounds=2)
    ab = benchdecode.run_ab(ns, [], 0.2, [("aime", "text"), ("repeat", "r")], None, ["repeat"], {})
    assert list(ab["lost"]) == ["hybrid:min-wait"] and not any(a.startswith("hybrid") for a in ab["arms"])
    assert ab["verdict"]["strategy"] == "offload"


def test_bench_replaces_an_older_row_when_hybrid_did_not_run_and_says_what_auto_reads(tmp_path, capsys):
    from types import SimpleNamespace

    from freetoken.moe import benchdecode, decode_profile

    gpu = {"name": "GPU", "uuid": "GPU-1"}
    path = str(tmp_path / "rows.json")
    ns = SimpleNamespace(no_write=False, profile=path, warm=8, decode=8)

    def arm(tok):
        return {"tok_s": tok, "results": [{"sha1": "t", "repeat": None, "halves": None}] * len(tok), "key": "k", "gpu_free_bytes": 1 << 30, "cache_slots": 40,
                "total_slots": 80, "slot_bytes": 1 << 20, "gpu": gpu, "capped": False}

    def ab(arms, lost=()):
        tok = {"tok_s": {a: r["tok_s"] for a, r in arms.items()}, "columns": ["aime", "repeat"], "guard": ["repeat"]}
        return {"arms": arms, "columns": ["aime", "repeat"], "guard": ["repeat"], "key": "k", "keys_seen": ["k"], "rounds": 1,
                "lost": {a: "the server did not start" for a in lost}, "verdict": decode_profile.verdict_for(tok, None)}

    def stored():
        row, _ = decode_profile.lookup("k", gpu_name="GPU", gpu_uuid="GPU-1", now=lambda: decode_profile.stamp("GPU", "GPU-1"), path=path)
        return decode_profile.verdict_for(row, None)

    offload = {"offload": arm([100.0, 230.0]), "offload#2": arm([100.2, 230.5]), "offload#3": arm([100.1, 230.2])}
    benchdecode.file_row(ns, ab({**offload, "hybrid:min-wait": arm([131.0, 228.0])}), "test")
    assert stored()["strategy"] == "hybrid" and "now takes hybrid with min-wait rounding" in capsys.readouterr().out
    # the same setup benched again, and this time the hybrid server does not start: the row on which it qualified must not stay
    benchdecode.file_row(ns, ab(offload, lost=["hybrid:min-wait"]), "test")
    out = capsys.readouterr().out
    assert stored()["strategy"] == "offload" and "stays on offload" in out and "no hybrid server ran" in out
    # one steady arm that qualifies alone and one noisy arm: a server without a forced mode reads both and stays on offload, and the bench says so
    two = {**offload, "hybrid:min-wait": arm([110.0, 228.0]), "hybrid:min-wait#2": arm([110.2, 228.2]),
           "hybrid:stochastic": arm([90.0, 228.0]), "hybrid:stochastic#2": arm([97.5, 228.1])}
    benchdecode.file_row(ns, ab(two), "test")
    out = capsys.readouterr().out
    assert stored()["strategy"] == "offload" and "None rounding" not in out and "stays on offload" in out and "min-wait rounding qualifies" in out
    # --no-write files nothing, with or without a hybrid arm
    dry = SimpleNamespace(no_write=True, profile=str(tmp_path / "none.json"), warm=8, decode=8)
    benchdecode.file_row(dry, ab(offload, lost=["hybrid:min-wait"]), "test")
    assert not (tmp_path / "none.json").exists()


def test_decode_row_caps_the_cache_at_the_measured_size_and_notes_a_larger_resize():
    from freetoken.moe import decode_profile as dp

    row = {"slots": {"offload": 1900, "hybrid": 2000}}
    assert dp.slot_cap(row, 10240) == 2000 and dp.slot_cap(row, 1500) == 1500 and dp.slot_cap(None, 10240) == 10240
    assert dp.resize_note(row, 2000) is None and dp.resize_note(row, 2100) is None and dp.resize_note(row, 1000) is None
    assert "2400" in dp.resize_note(row, 2400) and dp.resize_note(None, 2400) is None
    free = 15 << 30
    full = {"slots": {"offload": 80, "hybrid": 80}, "total_slots": 80, "slot_bytes": 1 << 20, "gpu_free_bytes": free}
    assert dp.auto_size_objection(full, free + (8 << 30)) is None  # every expert was cached: more memory changes nothing


def test_decode_profile_store_keeps_the_rows_that_can_still_be_used(tmp_path):
    from freetoken.moe import decode_profile as dp

    path = str(tmp_path / "p.json")
    gpu = {"name": "A", "uuid": "GPU-a"}
    build = {"version": "1.0", "commit": None}
    old, new = {"freetoken": build, "torch": "2.13"}, {"freetoken": build, "torch": "2.14"}
    dp.store("k1", {"tok_s": {"offload": [1.0]}, "stamp": old}, gpu=gpu, path=path)
    dp.store("k2", {"tok_s": {"offload": [2.0]}, "stamp": old}, gpu=gpu, path=path)
    assert set(json.load(open(path))["decisions"]) == {"k1", "k2"}
    dp.store("k3", {"tok_s": {"offload": [3.0]}, "stamp": new}, gpu=gpu, path=path)  # rows of the old stamp can no longer match
    assert set(json.load(open(path))["decisions"]) == {"k3"}
    dp.store("k4", {"tok_s": {"offload": [4.0]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert set(json.load(open(path))["decisions"]) == {"k4"}  # a file is for one GPU
    row, reason = dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=lambda: {"freetoken": build, "torch": "2.15"}, path=path)
    assert row is None and "torch" in reason
    assert dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0]["tok_s"] == {"offload": [4.0]}
    # with no row for the key the stamp is never made: a start without a profile costs one failed open
    assert dp.lookup("k9", gpu_name="B", gpu_uuid="GPU-b", now=lambda: pytest.fail("stamp made without a row"), path=path)[0] is None
    # a row edited by hand into a shape a reader would trip over is not handed out: here an arm with a column missing
    dp.store("k5", {"tok_s": {"offload": [5.0, 6.0], "hybrid:min-wait": [7.0]}, "columns": ["aime", "code"], "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k5", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # nor is one whose arm names a rounding mode the kernel does not have: auto would hand that name on
    dp.store("k6", {"tok_s": {"offload": [5.0], "hybrid:fastest": [7.0]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k6", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # nor is one with a number no bench writes: json reads Infinity, and an infinite hybrid reading would win every rule
    dp.store("k7", {"tok_s": {"offload": [5.0], "hybrid:min-wait": [float("inf")]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k7", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    # a damaged file reads as no row, and the next store replaces it
    with open(path, "w") as f:
        json.dump({"version": dp.DECODE_PROFILE_VERSION, "gpu": ["B"], "decisions": {}}, f)
    assert dp.lookup("k4", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0] is None
    dp.store("k8", {"tok_s": {"offload": [8.0]}, "stamp": new}, gpu={"name": "B", "uuid": "GPU-b"}, path=path)
    assert dp.lookup("k8", gpu_name="B", gpu_uuid="GPU-b", now=new, path=path)[0]["tok_s"] == {"offload": [8.0]}
