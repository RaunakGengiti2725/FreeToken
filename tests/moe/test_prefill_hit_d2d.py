"""Prefill hit-D2D split: resident experts must be gathered device-side, misses
H2D'd via cudaMemcpyBatchAsync, and the buffer must end up byte-identical to the
full-layer copy for every mix -- including experts resident in the volatile
buffer slots (< 2 * num_experts), which must be re-fetched over PCIe."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)

BATCH_API = pytest.mark.skipif(
    torch.version.hip is None and not _cuda_at_least(13, 0),
    reason="the cudaMemcpyBatchAsync binding needs CUDA >= 13.0",
)

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # hit region = slots [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(cache, sources, layer_id: int, expert_id: int, slot: int) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_roundtrip():
    from freetoken.kernel.batch_memcpy import batch_memcpy_jit

    rows, feat = 16, 1024
    src = torch.randint(0, 256, (rows, feat), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(rows, feat, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(rows)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in range(rows)], dtype=torch.int64)
    src_ptrs = torch.tensor([src[p].data_ptr() for p in perm.tolist()], dtype=torch.int64)
    sizes = torch.full((rows,), feat, dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batch_memcpy_jit(dst_ptrs, src_ptrs, sizes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(dst.cpu(), src[perm])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_probe_waits_for_current_stream():
    from freetoken.kernel.batch_memcpy import load_batch_memcpy

    # Compile and initialize the binding before queuing work so JIT cannot drain it.
    load_batch_memcpy()
    # Warm probe-sized allocations and the stream pool to avoid first-use syncs.
    for _ in range(4):
        block = torch.zeros(16, dtype=torch.uint8, device="cuda")
        del block
        pinned = torch.arange(16, dtype=torch.uint8).pin_memory()
        del pinned
    streams = [torch.cuda.Stream() for _ in range(40)]
    torch.cuda.synchronize()
    del streams

    try:
        torch.cuda._sleep(1_500_000_000)
        assert not torch.cuda.current_stream().query(), "backlog drained before the probe ran"
        load_batch_memcpy()
    finally:
        torch.cuda.synchronize()


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_matches_sources():
    cache, sources = _make_cache()
    # layer 1 residents: two real hits in the hit region, one expert stuck in a
    # buffer slot (< 2E) whose cache bytes are poisoned -- it must come back via
    # H2D, so poison must never surface in the buffer.
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    # prefetch(0) had no hits; prefetch(1) served experts 1 and 3 from the cache.
    assert cache.prefill_hit_rows == 2
    assert cache.prefill_total_rows == 2 * E
    # the buffer-slot resident was invalidated (its slot belongs to buffer 0)...
    assert int(cache.slot_for_id[1, 0].item()) == -1
    # ...while true hits keep their cache residency.
    assert int(cache.slot_for_id[1, 1].item()) == 17
    assert int(cache.slot_for_id[1, 3].item()) == 20


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, E])
def test_prefill_hit_d2d_pure_extremes(nhit):
    cache, sources = _make_cache()
    for e in range(nhit):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name
    assert cache.prefill_hit_rows == nhit


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_resident_layer_stays_on_device():
    cache, sources = _make_cache()
    for e in range(E):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    # the host rows change after the experts were loaded: a buffer equal to the loaded rows did not cross PCIe
    loaded = {name: per_layer[2].clone() for name, per_layer in sources.items()}
    for per_layer in sources.values():
        per_layer[2].add_(1)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, name in zip(views, sources):
        assert torch.equal(view.cpu(), loaded[name]), name


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_short_chunk_gathers_routed_experts_only():
    cache, sources = _make_cache()
    for e in range(E):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    routed = torch.tensor([[1, 3]], device="cuda")
    cache.begin_prefill(num_routes=routed.numel())
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2, routed)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view[[1, 3]].cpu(), per_layer[2][[1, 3]]), name
        # a resident expert outside the routes is not copied
        assert not torch.equal(view[0].cpu(), per_layer[2][0]), name


@CUDA
@JIT
@BATCH_API
def test_prefill_chunk_chosen_for_demand_reads_the_rows_the_stream_stages():
    """A chunk above the token threshold that the cache chooses to load on demand (choose_prefill_path) reads, expert
    for expert, the bytes the stream stages for the same routes; a cache with empty hit slots keeps streaming."""
    cache, sources = _make_cache()
    cache.prefill_demand_tokens, cache.prefill_demand_by_misses = 1, True
    routed = torch.tensor([[1, 5], [5, 2]], dtype=torch.int32, device="cuda")
    cache.begin_prefill(num_routes=routed.numel())
    cache.choose_prefill_path(2, routed)
    assert not cache.prefill_chunk_demand, "the hit slots are empty: the stream fills them"

    cache, sources = _make_cache()
    cache.prefill_demand_tokens, cache.prefill_demand_by_misses = 1, True
    # full hit slots: experts 0..3 of layers 2 and 1
    for e in range(4):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
        _seed_resident(cache, sources, layer_id=1, expert_id=e, slot=2 * E + 4 + e)
    rows = routed.view(-1).long()
    cache.begin_prefill(num_routes=routed.numel())
    cache.prefetch_prefill_layer(2)
    staged = [view[rows].clone() for view in cache.wait_prefill_layer(2, routed)]
    cache.release_prefill_layer(2)

    cache.begin_prefill(num_routes=routed.numel())
    cache.choose_prefill_path(2, routed)
    assert cache.prefill_chunk_demand
    slots = cache.load_prefill_experts(2, routed)
    torch.cuda.synchronize()
    assert not torch.equal(slots, routed), "the admission must have mapped expert ids to slots"
    for view, stream_rows, (name, per_layer) in zip(cache.pool_bank_views(2), staged, sources.items()):
        loaded = view[slots.view(-1).long()]
        assert torch.equal(loaded, stream_rows), name
        assert torch.equal(loaded.cpu(), per_layer[2][rows.cpu()]), name


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_keeps_streamed_experts_in_empty_slots():
    cache, sources = _make_cache()
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    cache.wait_prefill_layer(2)
    cache.release_prefill_layer(2)
    torch.cuda.synchronize()
    # the E empty slots outside the staging buffers now hold layer 2
    slots = cache.slot_for_id[2].cpu().long()
    assert sorted(slots.tolist()) == list(range(2 * E, 3 * E))
    assert cache.id_of_slot[slots].cpu().tolist() == [2 * E + e for e in range(E)]
    assert int(cache.usage[slots].max()) == 0  # the first to be evicted, and no LRU step's stamp
    for name, per_layer in sources.items():
        assert torch.equal(cache.bank_caches[name][slots].cpu(), per_layer[2]), name
    # so the next prefill finds every expert of the layer resident
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    assert cache.prefill_hit_rows == E
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name


@CUDA
@JIT
@BATCH_API
def test_prefill_reads_slots_once_every_expert_is_resident():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    # a slot for every expert outside the two staging buffers
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=(2 + NUM_LAYERS) * E, device=dev, prefill_overlap=True, prefill_hit_d2d=True
    )
    cache.set_bank_sources(sources)

    def stream(layers):
        cache.begin_prefill()
        for layer in layers:
            cache.prefetch_prefill_layer(layer)
            cache.wait_prefill_layer(layer)
            cache.release_prefill_layer(layer)

    stream(range(NUM_LAYERS - 1))
    cache.begin_prefill()
    assert not cache.all_resident  # the last layer has not been streamed
    stream([NUM_LAYERS - 1])
    assert not cache.all_resident  # known only from the next prefill's snapshot
    cache.begin_prefill()
    assert cache.all_resident
    ids = torch.arange(E, device=dev).flip(0)
    for layer in range(NUM_LAYERS):
        slots = cache.resident_slots(layer, ids).long()
        assert int(slots.min()) >= 2 * E  # none in a staging buffer
        for view, (name, per_layer) in zip(cache.pool_bank_views(layer), sources.items()):
            assert torch.equal(view[slots].cpu(), per_layer[layer].flip(0)), name
    # a prefill no longer stages anything
    before = cache.id_of_slot.clone()
    cache.begin_prefill()
    assert torch.equal(cache.id_of_slot, before)
    cache.reset()
    assert not cache.all_resident


@CUDA
def test_prefill_hit_d2d_noop_without_spare_slots():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,  # the cpu backend's pinned geometry: no hit region
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    views = cache.wait_prefill_layer(0)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[0]), name


@pytest.mark.parametrize("seed", range(4))
def test_prefill_routed_copy_stages_the_routed_misses_as_the_stream_does(seed):
    """The batch memcpy entries of a chunk that copies only its routed misses, on host memory standing in for the
    device: the small banks' whole-layer entries ahead of the layer and the big banks' runs of routed misses after it
    leave every routed row, and every small-bank row, as the stream's entries do, and only entries of whole big rows."""
    import numpy as np

    from freetoken.moe.offload_cache import _SMALL_BANK_FEAT_BYTES, routed_misses, stage_copy_entries

    rng = np.random.default_rng(seed)
    experts = 16
    rows = [_SMALL_BANK_FEAT_BYTES, 64, 0, _SMALL_BANK_FEAT_BYTES + 32]  # big, small, a role the layer lacks, big
    sizes = [experts * feat for feat in rows]
    host = rng.integers(0, 256, sum(sizes), dtype=np.uint8)
    src = list(np.cumsum([0] + sizes[:-1]))
    stage = [base + len(host) for base in src]

    def staged(*batches):
        memory = np.concatenate([host, np.zeros(len(host), dtype=np.uint8)])
        for dst, source, nbytes in batches:
            for d, s, n in zip(dst, source, nbytes):
                memory[d : d + n] = memory[s : s + n]
        return [memory[st : st + size].reshape(experts, max(feat, 1)) if feat else None for st, size, feat in zip(stage, sizes, rows)]

    miss = np.sort(rng.choice(experts, int(rng.integers(1, experts)), replace=False))
    routed = rng.random(experts) < 0.5
    stream = staged(stage_copy_entries(rows, stage, src, experts, miss, small=True, large=True))
    ahead = stage_copy_entries(rows, stage, src, experts, miss, small=True, large=False)
    after = stage_copy_entries(rows, stage, src, experts, routed_misses(miss, routed), small=False, large=True)
    routed_copy = staged(ahead, after)

    copied = miss[routed[miss]]
    for feat, by_stream, by_routes in zip(rows, stream, routed_copy):
        if not feat:
            continue
        if feat < _SMALL_BANK_FEAT_BYTES:
            assert np.array_equal(by_routes, by_stream)
        else:
            assert np.array_equal(by_routes[copied], by_stream[copied])
            assert not by_routes[np.setdiff1d(miss, copied)].any(), "an unrouted miss crossed"
    # no entry under the driver's async floor in the batch issued after the layer's routes
    assert all(n >= _SMALL_BANK_FEAT_BYTES for n in after[2])
    assert sum(after[2]) == copied.size * (rows[0] + rows[3])


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("decode_target", ["gpu", "hybrid"])
def test_prefill_routed_copy_matches_the_stream_layer_for_layer(monkeypatch, decode_target):
    """A chunk that copies only the absent experts it routes to, after each layer's routes, stages every routed row
    bit for bit as the stream that copies every absent expert ahead of the layer, layer for layer, and leaves the
    slot map as the stream does (hybrid decode reads residency). gate_up counts as a big bank here, down as a small
    one; the hit slots are full, so the stream keeps nothing either."""
    from freetoken.moe import offload_cache

    monkeypatch.setattr(offload_cache, "_SMALL_BANK_FEAT_BYTES", 300)  # gate_up rows are 512 bytes, down rows 256
    routes = torch.tensor([[1, 5], [5, 2], [0, 7]], dtype=torch.int32, device="cuda")
    rows = routes.view(-1).unique().long()

    def run(routed_copy_tokens):
        torch.manual_seed(0)
        cache, sources = _make_cache()
        cache.decode_target = decode_target
        cache.prefill_routed_copy_tokens = routed_copy_tokens
        # every hit slot full: experts 1 and 3 of each layer, and others; expert 0 of layer 1 sits in a staging slot, poisoned
        for layer_id in range(NUM_LAYERS):
            for k, e in enumerate((1, 3) if layer_id < 2 else (1, 3, 4, 6)):
                _seed_resident(cache, sources, layer_id, e, 2 * E + 2 * layer_id + k if layer_id < 2 else 2 * E + 4 + k)
        _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
        for name in sources:
            cache.bank_caches[name][2].fill_(float("nan"))
        cache.begin_prefill(num_routes=routes.numel(), num_tokens=routes.shape[0])
        assert cache._prefill_routed_copy == bool(routed_copy_tokens)
        staged = []
        for layer_id in range(NUM_LAYERS):
            cache.prefetch_prefill_layer(layer_id)
            cache.prefetch_prefill_layer(layer_id + 1)
            views = cache.wait_prefill_layer(layer_id, routes)
            torch.cuda.synchronize()
            staged.append([view[rows].clone() for view in views])
            for view, (name, per_layer) in zip(views, sources.items()):
                assert torch.equal(view[rows].cpu(), per_layer[layer_id][rows.cpu()]), (layer_id, name)
            cache.release_prefill_layer(layer_id)
        torch.cuda.synchronize()
        return staged, cache.slot_for_id.clone(), cache.id_of_slot.clone(), cache.prefill_hit_rows

    stream, routed = run(0), run(64)
    for layer_id, (by_stream, by_routes) in enumerate(zip(stream[0], routed[0])):
        for a, b in zip(by_stream, by_routes):
            assert torch.equal(a, b), layer_id
    assert torch.equal(stream[1], routed[1]) and torch.equal(stream[2], routed[2]) and stream[3] == routed[3]
