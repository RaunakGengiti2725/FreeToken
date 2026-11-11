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
