from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import json
import os
import time

from freetoken.engine.cache_budget import (
    CacheBudgetTooSmall, budget_advice, expert_bytes_per_slot, kv_capacity_warning, plan_cache_budget, plan_changes, plan_kv_target,
    resolve_moe_cache_auto,
)
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(ValueError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_budget_advice_names_flags_the_resolver_accepts():
    # budget int(0.9 * 1005) - 200 = 704 B; the floor of 4 slots (400 B) leaves 30 of the 64 reserved pages (10 B each)
    terms = dict(
        baseline_free=1005, weights_bytes=200, memory_ratio=0.9, cache_per_page=10, fixed_cache_size=0,
        per_expert_bytes=100, num_experts=4, total_experts=40, prefill_overlap=False,
        kv_reserve_tokens=256, page_size=4,
    )

    def plan(tokens):
        try:
            return resolve_moe_cache_auto(**{**terms, "kv_reserve_tokens": tokens})
        except ValueError:
            return None

    # checked against the resolver itself: the reserve named is the largest one it accepts, with the plan it returns
    largest = max(tokens for tokens in range(8, 257) if plan(tokens))
    slots, pages, _ = plan(largest)
    advice = budget_advice([("", terms)], min_reserve_tokens=0)
    assert advice.startswith(f"fits with --kv-reserve-tokens {largest}: {slots} expert slots, {pages * 4} KV tokens")

    # weights above the budget: int(0.95 * 1005) - 1200 = -246 B. The pool's own floor of 200 tokens (50 pages, 500 B)
    # is kept, so the last try is 400 + 500 + 246 = 1146 B short.
    over = {**terms, "weights_bytes": 1200}
    advice = budget_advice([("", over), ("--memory-ratio 0.95", {**over, "memory_ratio": 0.95})], min_reserve_tokens=200)
    assert advice.startswith("no fit") and "--kv-reserve-tokens 200 " in advice and "1146 B" in advice


def test_kv_target_is_planned_only_where_the_slot_floor_fits_beside_it():
    # 4-token pages of 10 B, slots of 100 B with a floor of 4; the reserve to fall back to is 64 tokens, the target 256
    def terms(budget, reserve=64):
        return dict(
            baseline_free=budget, weights_bytes=0, memory_ratio=1.0, cache_per_page=10, fixed_cache_size=0,
            per_expert_bytes=100, num_experts=4, total_experts=40, prefill_overlap=False,
            kv_reserve_tokens=reserve, page_size=4,
        )

    # room for both: the target's plan, 5 slots below the plan of the default reserve
    plan, kept, most = plan_kv_target(terms(2000), 256)
    assert plan == resolve_moe_cache_auto(**terms(2000, 256)) == (13, 70, False)
    assert kept == resolve_moe_cache_auto(**terms(2000)) == (18, 20, False) and most == (256, 13)
    # 400 B of slot floor + 640 B of target is 140 B over 900 B: no plan, and 200 tokens is the largest reserve the resolver takes
    plan, kept, most = plan_kv_target(terms(900), 256)
    assert plan is None and kept == resolve_moe_cache_auto(**terms(900)) == (7, 20, False)
    assert most == (200, resolve_moe_cache_auto(**terms(900, 200))[0]) == (200, 4)
    with pytest.raises(ValueError, match="budget too small"):
        resolve_moe_cache_auto(**terms(900, 204))
    # not even the default reserve fits
    assert plan_kv_target(terms(500), 256) == (None, None, (0, 0))

    # what each stage did to the plan, from the resolver's own plans: none at 500 B, (7, 20) at 900 B, (4, 50) with the reserve at 200
    stages = [("", terms(500)), ("more memory", terms(900)), ("--kv-reserve-tokens 200", terms(900, 200)), ("again", terms(900, 200))]
    assert plan_changes(stages) == [
        "more memory: the plan fits: 7 expert slots, 80 KV tokens",
        "--kv-reserve-tokens 200: -3 expert slots, +120 KV tokens",
        "again: no change",
    ]
    assert plan_changes([("", terms(900)), ("less memory", terms(500))])[0].startswith("less memory: the minimum plan is ")


@pytest.mark.parametrize(
    "kv_token, gain, free, plain, reserve, slots",
    [
        # 20 KB of KV a token; one request and host embeddings give back more than the MTP head takes (+170 slots): the target stays
        (20_000, 170, 8, 1572, 36864, 1661),
        (20_000, 170, 16, 5916, 36864, 6246),
        # tied embeddings stay on the GPU (-405 slots): under the plain plan already, so the default reserve
        (20_000, -405, 8, 1572, 8192, 1409),
        (20_000, -405, 16, 5916, 15104, 5916),
        # 100 KB of KV a token, +575 slots: the target costs more slots than the preset gave, or does not fit at all
        (100_000, 575, 12, 3376, 24864, 3376),
        (100_000, 575, 8, 1204, 22720, 1204),
        (100_000, 575, 24, 9891, 31312, 9891),
    ],
)
def test_preset_kv_reserve_is_the_largest_that_keeps_the_expert_slots_of_a_server_without_the_preset(kv_token, gain, free, plain, reserve, slots):
    from freetoken.engine.cache_budget import plan_kv_reserve

    slot, weights = 1_780_000, int(4.44 * 2**30)  # docs/performance/decode.md: Qwen3.6-35B-A3B NVFP4 slots; 5589 default slots at 15.4 GiB free

    def terms(ratio, gain_slots, tokens=8192):
        return dict(
            baseline_free=free << 30, weights_bytes=weights - gain_slots * slot, memory_ratio=ratio, cache_per_page=16 * kv_token,
            fixed_cache_size=0, per_expert_bytes=slot, num_experts=256, total_experts=10240, prefill_overlap=True,
            kv_reserve_tokens=tokens, page_size=16,
        )

    assert resolve_moe_cache_auto(**terms(0.9, 0))[0] == plain
    preset = terms(0.95, gain)
    got, plan = plan_kv_reserve(preset, 36864, plain)
    assert (got, plan[0]) == (reserve, slots) and plan == resolve_moe_cache_auto(**terms(0.95, gain, got))
    if 8192 < got < 36864:
        assert slots >= plain > resolve_moe_cache_auto(**terms(0.95, gain, got + 16))[0]  # one page more is under the plain plan


@pytest.mark.parametrize("free", [8, 12, 16])
def test_preset_plan_leaves_a_prefill_chunk_of_working_memory_free_and_the_plain_plan_alone(free):
    from freetoken.engine.cache_budget import prefill_working_bytes, required_bytes

    # Qwen3.6-35B-A3B in bf16: hidden 2048, 16 key and 32 value heads of 128, the chunk kernel's 64-token chunks
    shapes = dict(hidden=2048, key_dim=2048, value_dim=4096, value_heads=32, head_dim=128, chunk=64, itemsize=2)
    need = prefill_working_bytes(8192, **shapes)
    # 116 KB of temporaries a token and 1 MiB of state per kernel chunk: 0.46 GiB free did not hold 3785 tokens, 1.20 GiB held 8192
    assert need == 8192 * 118_784 + 128 * 2**20
    assert 0.46 * 2**30 < prefill_working_bytes(3785, **shapes) and need < 1.20 * 2**30
    slot, weights = 1_780_000, int(4.44 * 2**30)

    def terms(ratio, gain_slots, reserve):
        return dict(
            baseline_free=free << 30, weights_bytes=weights - gain_slots * slot, memory_ratio=ratio, cache_per_page=16 * 20_000,
            fixed_cache_size=0, per_expert_bytes=slot, num_experts=256, total_experts=10240, prefill_overlap=True,
            kv_reserve_tokens=8192, page_size=16, reserve_bytes=reserve,
        )

    def left_free(plan, t):
        return t["baseline_free"] - t["weights_bytes"] - required_bytes(plan[0], plan[1], slot, t["cache_per_page"])

    plain, preset = terms(0.9, 0, 0), terms(0.95, 575, need)
    plan = resolve_moe_cache_auto(**preset)
    # not every expert fits at these sizes, so KV holds its reserve and less than a slot's bytes more: the slots pay for the chunk
    assert (plan[1] - 8192 // 16) * 16 * 20_000 < slot
    assert resolve_moe_cache_auto(**terms(0.95, 575, 0))[0] - plan[0] in (need // slot, need // slot + 1)
    # the ratio's own headroom and the whole chunk on top of it; without the reserve the same ratio leaves less than the chunk needs
    assert left_free(plan, preset) >= need + int(0.05 * (free << 30))
    unreserved = terms(0.95, 575, 0)
    assert left_free(resolve_moe_cache_auto(**unreserved), unreserved) < need
    assert plan[0] >= resolve_moe_cache_auto(**plain)[0]  # host embeddings and one request pay for the reserve


@pytest.mark.parametrize(
    "reserve, slots, pages",
    [
        (0, 10, 100),  # every expert cached, KV holds 90 pages above its floor of 10
        (300, 10, 70),  # the pages above the floor cover the reserve: the slots are untouched
        (900, 10, 10),  # exactly the pages above the floor
        (1200, 7, 10),  # 300 B more than those pages: three slots pay the rest
        (5000, 4, 10),  # more than the plan can give: both caches at their floors, and the plan still stands
    ],
)
def test_reserve_comes_out_of_kv_above_its_floor_before_expert_slots(reserve, slots, pages):
    plan = dict(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10, num_experts=2, total_experts=10,
        prefill_overlap=True, kv_reserve_pages=10, max_slots=10,
    )
    assert plan_cache_budget(**plan, reserve_bytes=reserve)[:2] == (slots, pages)
    # what is left out of the caches is the reserve, or all the floors (4 slots, 10 pages) leave of it
    assert 2000 - slots * 100 - pages * 10 >= min(reserve, 2000 - 4 * 100 - 10 * 10)
    # a budget the floors do not fit fails as it does without a reserve, the same bytes short
    with pytest.raises(CacheBudgetTooSmall) as short:
        plan_cache_budget(**{**plan, "budget_bytes": 480}, reserve_bytes=reserve)
    assert short.value.short == 20


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1,
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_caps_slots_at_the_kernel_limit():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, max_slots=992,
    )
    assert size == 992


def test_kv_capacity_warning_prices_the_floor_it_names():
    # budget 1000 B, 50 B a slot, 10 B a one-token page: 30 more KV tokens are 300 B, 6 slots.
    terms = dict(
        baseline_free=1000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=40, prefill_overlap=False,
        kv_reserve_tokens=8, page_size=1,
    )
    slots, kv_tokens, _ = resolve_moe_cache_auto(**terms)
    assert (slots, kv_tokens) == (18, 10)
    assert "--kv-reserve-tokens 40 costs 6 of the 18 expert" in kv_capacity_warning(kv_tokens, 40, slots, terms)

    # a server started with the flag the warning names has exactly that many fewer slots, and no warning
    at_floor = {**terms, "kv_reserve_tokens": 40}
    floor_slots, floor_tokens, _ = resolve_moe_cache_auto(**at_floor)
    assert floor_slots == slots - 6
    assert kv_capacity_warning(floor_tokens, 40, floor_slots, at_floor) is None


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_strategy == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton" if torch.version.hip is not None else "fi"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs(monkeypatch):
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16,
    )
    assert (size, pages, overlap) == expected

    class StubMethod:
        def slot_limit(self):
            return 5

    size, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks(), StubMethod())
    assert size == 5

    # 25 linear states (4 requests) leave less than 4 slots + the 2-page reserve, 9 states (1 request) leave room:
    # the error names that flag, and the plan it prints is the plan the flag gives.
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(1, 3), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    config = StubConfig()
    config.model_config.linear_attention_group = lambda: group
    config.cache_type, config.linear_state_cache_ratio, config.embed_weights = "hybrid_radix", 2.0, "host"
    config.kv_reserve_tokens, config.moe_prefill_overlap = 32, False
    engine._baseline_free, engine._weights_bytes = 1_500_000, 962_448
    with pytest.raises(ValueError, match="budget too small") as refused:
        engine._resolve_auto_moe_cache_size(config, StubBanks())
    config.max_running_req = 1
    size, pages, _ = engine._resolve_auto_moe_cache_size(config, StubBanks())
    assert f"fits with --max-running-requests 1: {size} expert slots, {pages * 16} KV tokens" in str(refused.value)

    # --preset's KV reserve, with 8 slots (6144 B) and 98304 B per 16-token page: kept where the slot floor fits beside it;
    # where not, the default reserve is back and the field is no longer the preset's
    from freetoken.engine.config import EngineConfig

    default = EngineConfig.kv_reserve_tokens
    for free, context, kept in ((200 << 20, 1 << 20, 16384), (200 << 20, 12288, 12288), (200 << 20, 4096, default), (70 << 20, 1 << 20, None)):
        engine._baseline_free, engine._weights_bytes, config.max_seq_len = free, 0, context
        config.preset, config.preset_fields, config.kv_reserve_tokens = "single", ("max_running_req", "kv_reserve_tokens"), 16384
        plain = resolve_moe_cache_auto(**engine._preset_stages(config, StubBanks(), None, None)[0][1])  # a server without the preset
        engine._settle_preset_kv_reserve(config, StubBanks(), None, None)
        size, pages, _ = engine._resolve_auto_moe_cache_size(config, StubBanks())
        if kept is None:
            # 16384 tokens do not fit in 70 MiB: the largest reserve that keeps the slots of a server without the preset
            assert default < config.kv_reserve_tokens < 16384 and size >= plain[0]
            config.kv_reserve_tokens += 16
            with pytest.raises(ValueError, match="budget too small"):
                engine._resolve_auto_moe_cache_size(config, StubBanks())
            config.kv_reserve_tokens -= 16
        else:
            assert config.kv_reserve_tokens == kept  # never more than the model's context, never less than the default
        assert ("kv_reserve_tokens" in config.preset_fields) == (config.kv_reserve_tokens != default)
        assert pages * 16 >= config.kv_reserve_tokens
    # the planner's account starts from the plan without the preset's fields and ends at the plan that runs
    config.preset_fields, config.kv_reserve_tokens = ("max_running_req", "kv_reserve_tokens"), 16384
    stages = engine._preset_stages(config, StubBanks(), None, None)
    assert [label for label, _ in stages] == ["", "--max-running-requests 1", "--kv-reserve-tokens 16384"]
    assert stages[-1][1] == engine._auto_cache_terms(config, StubBanks())
    assert stages[0][1]["kv_reserve_tokens"] == stages[1][1]["kv_reserve_tokens"] == default
    assert stages[0][1]["fixed_cache_size"] > stages[1][1]["fixed_cache_size"]  # the linear states of 4 requests, then of 1
    # a preset's memory ratio comes with the working memory of one prefill chunk left free, as a step of its own in the account;
    # a typed ratio gets no reserve, and a model whose prefill is not accounted for keeps the default ratio
    from freetoken.engine.cache_budget import prefill_working_bytes

    config.dtype, config.max_forward_len, config.model_config.hidden_size, config.memory_ratio = torch.bfloat16, 256, 64, 0.95
    need = prefill_working_bytes(256, hidden=64, key_dim=32, value_dim=64, value_heads=4, head_dim=16, chunk=64, itemsize=2)
    typed = engine._auto_cache_terms(config, StubBanks())
    assert engine._preset_prefill_reserve(config) == 0 == typed["reserve_bytes"]
    config.preset_fields = ("max_running_req", "memory_ratio", "kv_reserve_tokens")
    assert engine._preset_prefill_reserve(config) == need > 0
    # the planner gets it as bytes neither cache takes, not as a fixed cost: KV pages above the reserve pay first
    assert engine._auto_cache_terms(config, StubBanks()) == {**typed, "reserve_bytes": need}
    stages = engine._preset_stages(config, StubBanks(), None, None)
    assert [label.split(" left free")[0] for label, _ in stages[2:4]] == ["--memory-ratio 0.95", f"{need / 2**30:.2f} GiB"]
    assert (stages[2][1]["reserve_bytes"], stages[3][1]["reserve_bytes"]) == (0, need) and stages[0][1]["memory_ratio"] == 0.9
    # every expert has a slot here, so the chunk costs KV tokens and no slot
    with_chunk, without = resolve_moe_cache_auto(**stages[3][1]), resolve_moe_cache_auto(**stages[2][1])
    assert with_chunk[0] == without[0] == 8 and 0 < (without[1] - with_chunk[1]) * stages[3][1]["cache_per_page"] - need < stages[3][1]["cache_per_page"]
    assert stages[-1][1] == engine._auto_cache_terms(config, StubBanks())
    engine._settle_preset_memory_ratio(config)
    assert config.memory_ratio == 0.95
    # a forward of one request cuts the GatedDeltaNet chunk rule into calls (kvcache.linear_state_pool.gdn_rule_tokens):
    # what is alive only inside a call is counted for a call's tokens. Of the 1280 elements a token, 704 belong to a call,
    # here of 64 tokens and the 63 a rest can add; 4 kernel chunks of state become 2, and the gathered output adds 64
    shapes = dict(hidden=64, key_dim=32, value_dim=64, value_heads=4, head_dim=16, chunk=64, itemsize=2)
    assert need == (256 * (576 + 704) + 4 * 1024) * 2
    in_calls = prefill_working_bytes(256, **shapes, rule_tokens=64)
    assert in_calls == (256 * (576 + 64) + 127 * 704 + 2 * 1024) * 2
    # a long prefill in short calls: the layer's tensors after the rule, with its normed and projected output, are the peak
    assert prefill_working_bytes(4096, **shapes, rule_tokens=64) == 4096 * (576 + 64 + 64 + 64) * 2
    assert prefill_working_bytes(256, **shapes, rule_tokens=256) == prefill_working_bytes(256, **shapes, rule_tokens=0) == need
    with monkeypatch.context() as m:
        m.setattr("freetoken.kvcache.linear_state_pool.gdn_rule_tokens", lambda: 64)
        assert engine._preset_prefill_reserve(config) == in_calls
        m.setattr(config, "max_running_req", 4)  # a forward of several requests runs the rule as one call
        assert engine._preset_prefill_reserve(config) == need
    # the chunk's working memory comes before the preset's raised KV reserve: a budget of the slot floor, some KV pages and
    # some bytes more. The reserve gives way page by page, to a chunk-long prompt and 1024 tokens of answer at the least.
    # The chunk is counted as the layer runs it for one request, in calls of 2048 tokens: the planner's reserve, the
    # lowered KV reserve and the warning's prompt length all come from that count
    from unittest.mock import patch

    from freetoken.engine import engine as eng
    from freetoken.engine.cache_budget import reserve_kept

    config.max_forward_len = 8192
    need, fields = engine._preset_prefill_reserve(config), config.preset_fields
    cut = dict(shapes, rule_tokens=2048)
    assert need == prefill_working_bytes(8192, **cut) < prefill_working_bytes(8192, **shapes) and eng.PRESET_CHUNK_ANSWER_TOKENS == 1024
    assert engine._auto_cache_terms(config, StubBanks())["reserve_bytes"] == need
    t = engine._auto_cache_terms(config, StubBanks())
    page_bytes, engine._weights_bytes = t["cache_per_page"], 0

    def settled(kv_pages, more, preset=fields):
        config.kv_reserve_tokens, config.preset_fields = 16384, preset
        budget = t["fixed_cache_size"] + 4 * t["per_expert_bytes"] + kv_pages * page_bytes + more
        engine._baseline_free = -(-budget * 20 // 19) + 1  # the ratio is 0.95
        warned = []
        with patch.object(eng.logger, "warning_rank0", warned.append), patch.object(eng.logger, "info_rank0", lambda line: None):
            engine._settle_preset_chunk_reserve(config, StubBanks(), None, None)
        kept = reserve_kept(engine._auto_cache_terms(config, StubBanks()))
        assert engine._resolve_auto_moe_cache_size(config, StubBanks())[1] * 16 >= config.kv_reserve_tokens  # the plan stands
        return config.kv_reserve_tokens, kept, warned

    assert settled(1024, need) == (16384, need, [])  # room for both
    assert settled(1024, need - 3 * page_bytes) == (16384 - 3 * 16, need, [])  # three pages short: three pages of reserve go
    assert settled(576, need) == (9216, need, [])
    reserve, kept, warned = settled(576, need // 2)  # not even at 8192 + 1024 tokens: the reserve stops there, and the log says what is kept
    assert reserve == 9216 and need // 2 <= kept < need // 2 + page_bytes and len(warned) == 1
    longest = int(warned[0].split("a prompt longer than ")[1].split(" tokens")[0])
    assert prefill_working_bytes(longest, **cut) <= kept < prefill_working_bytes(longest + 1, **cut) and 2048 < longest < 8192
    assert " of the " in engine._preset_left_free(config) and f"--max-prefill-length {longest} or less" in warned[0]
    # nothing kept at all: the advice is not a chunk length of 0
    reserve, kept, warned = settled(576, 0)
    assert reserve == 9216 and kept < prefill_working_bytes(1, **cut) and "no prefill chunk fits" in warned[0] and "length 0" not in warned[0]
    # a typed --kv-reserve-tokens is not the preset's to lower: the chunk is shorted and the warning says so
    reserve, kept, warned = settled(1024, need // 2, preset=("max_running_req", "memory_ratio"))
    assert reserve == 16384 and kept < need and len(warned) == 1
    assert settled(1024, 0, preset=()) == (16384, 0, [])  # without the preset: no reserve, nothing lowered, nothing said
    config.preset_fields, config.max_forward_len = fields, 256
    del engine._preset_reserve_kept
    config.model_config.linear_attention_group = lambda: None
    engine._settle_preset_memory_ratio(config)
    assert config.memory_ratio == 0.9 and "memory_ratio" not in config.preset_fields and engine._preset_prefill_reserve(config) == 0


def test_prefill_working_bytes_counts_gdn_models_only():
    from dataclasses import replace

    from freetoken.engine.cache_budget import prefill_working_bytes
    from freetoken.engine.engine import _prefill_working_bytes
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    # Qwen3.6-35B-A3B in bf16: hidden 2048, 16 key and 32 value heads of 128, the chunk kernel's 64-token chunks
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1, 2), num_key_heads=16, num_value_heads=32, key_head_dim=128, value_head_dim=128,
        conv_kernel_dim=4, output_gate="silu",
    )
    shapes = dict(hidden=2048, key_dim=2048, value_dim=4096, value_heads=32, head_dim=128, chunk=64, itemsize=2)

    def config(group, chunk=8192, context=262144, requests=4):
        model = SimpleNamespace(hidden_size=2048, linear_attention_group=lambda: group)
        return SimpleNamespace(
            model_config=model, max_forward_len=chunk, max_seq_len=context, dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1),
            max_running_req=requests,
        )

    assert _prefill_working_bytes(config(group)) == (8192, prefill_working_bytes(8192, **shapes))
    # one running request: the layer runs the chunk rule in calls of 2048 tokens, and the count is a call's (0.65 GiB, not 1.03)
    one = _prefill_working_bytes(config(group, requests=1))
    assert one == (8192, prefill_working_bytes(8192, **shapes, rule_tokens=2048)) and round(one[1] / 2**30, 2) == 0.65
    assert round(prefill_working_bytes(8192, **shapes) / 2**30, 2) == 1.03
    # no chunk is longer than the context
    assert _prefill_working_bytes(config(group, context=4096)) == (4096, prefill_working_bytes(4096, **shapes))
    # two ranks hold half the heads each
    halved = {**shapes, "key_dim": 1024, "value_dim": 2048, "value_heads": 16}
    two_ranks = config(group)
    two_ranks.tp_info.size = 2
    assert _prefill_working_bytes(two_ranks)[1] == prefill_working_bytes(8192, **halved)
    # not counted: no linear attention layers, or the KDA kernels
    assert _prefill_working_bytes(config(None))[1] == 0
    assert _prefill_working_bytes(config(replace(group, variant="kda")))[1] == 0


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_strategy="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    overrides.setdefault("model_path", "/tmp/freetoken-test-model")
    overrides.setdefault("tp_info", DistributedInfo(rank=0, size=1))
    config = EngineConfig(
        dtype=torch.bfloat16,
        attention_backend="triton" if torch.version.hip is not None else "fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",
            moe_strategy="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend(monkeypatch):
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_strategy="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    # a unified-memory host (GB10) resolves auto to fused instead
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)
    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim. Assert the family, not the member.
    assert is_offload_moe_strategy(config.moe_strategy)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


# ---- --moe-strategy auto and the decode rows `ft bench decode` files (moe/decode_profile.py), through the real _adjust_config ----

_ROW_GPU = ("FAKE GPU", "GPU-fake")
_ROW_FREE = 15 << 30  # GPU memory free when the row was measured


def _passes(*columns):
    """Three passes a column, 1 tok/s apart around the given tok/s: the shape of a row's ``tok_s``."""
    return [[v - 1.0, v, v + 1.0] for v in columns]


_HYBRID_FASTER = {"offload": _passes(100.0, 101.0), "offload#2": _passes(100.5, 100.0), "hybrid:min-wait": _passes(131.0, 128.0)}
_HYBRID_SLOWER = {"offload": _passes(84.0, 83.0), "offload#2": _passes(84.5, 83.2), "hybrid:min-wait": _passes(27.0, 28.0)}
# a workload column and a guard column (output that barely misses the cache): min-wait leads by more on the first and is 17% behind on the second
_TWO_ARMS = {"offload": _passes(100.0, 230.0), "offload#2": _passes(100.4, 231.0), "hybrid:min-wait": _passes(133.0, 191.0),
             "hybrid:stochastic": _passes(130.0, 219.0)}


@pytest.fixture
def decode_rows(tmp_path, monkeypatch):
    """A checkpoint folder, an empty decode profile, and stand-ins for what only a Linux GPU box can answer."""
    import freetoken.engine.engine as engine
    from freetoken.moe import decode_profile

    model = tmp_path / "ckpt"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "fake"}')
    (model / "model-00001.safetensors").write_bytes(b"x" * 100)
    monkeypatch.setenv("FREETOKEN_BENCHDECODE_PATH", str(tmp_path / "benchdecode.json"))
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", str(tmp_path / "benchbw.json"))
    for name in [n for n in os.environ if n.startswith("FREETOKEN_") and n not in ("FREETOKEN_BENCHDECODE_PATH", "FREETOKEN_BENCHBW_PATH")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(engine, "_is_unified_memory_gpu", lambda index=None: False)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: True)
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: _ROW_GPU)
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE)
    monkeypatch.setattr("freetoken.version.__version__", "9.9.9+gabc")
    proc = {"/proc/meminfo": "65536000 kB", "/proc/cpuinfo": "Fake CPU", "/sys/fs/cgroup/cpu.max": "max 100000"}
    monkeypatch.setattr(decode_profile, "_first_line", lambda path, prefix="": proc.get(path))
    return SimpleNamespace(model=model, engine=engine, dp=decode_profile, tmp=tmp_path, patch=monkeypatch)


def _resolved(rows, **flags):
    config = _offload_engine_config(model_path=str(rows.model), **flags)
    rows.engine._adjust_config(config)
    return config


def _file_row(rows, tok_s, *, slots=40, asked="offload", guard=(), **flags):
    """File a row the way the bench does: under the key one of its servers, started with an explicit strategy, logs."""
    with rows.patch.context() as m:
        m.setenv("FREETOKEN_BENCH_DECODE_KEY", "1")
        key = _resolved(rows, moe_strategy=asked, **flags).moe_decode_key
    row = {"tok_s": tok_s, "columns": ["aime", "repeat"], "guard": list(guard), "slots": {"offload": slots, "hybrid": slots},
           "total_slots": 80, "slot_bytes": 1 << 20, "gpu_free_bytes": _ROW_FREE, "measured_at": "2026-11-12T10:00:00",
           "stamp": rows.dp.stamp(*_ROW_GPU)}
    rows.dp.store(key, row, gpu={"name": _ROW_GPU[0], "uuid": _ROW_GPU[1]})
    return key


def test_auto_stays_on_offload_without_a_decode_row(decode_rows):
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and config.moe_decode_row is None


@pytest.mark.parametrize("asked", ["offload", "hybrid"])
@pytest.mark.parametrize("flags", [{}, {"moe_cache_size": 40}, {"moe_cache_rate": 0.5}, {"moe_cache_mem": 4 << 30}, {"memory_ratio": 0.5}])
def test_auto_takes_hybrid_on_the_row_an_explicit_start_filed(decode_rows, asked, flags):
    key = _file_row(decode_rows, _HYBRID_FASTER, asked=asked, **flags)
    config = _resolved(decode_rows, **flags)
    assert config.moe_decode_key == key
    assert config.moe_strategy == "hybrid" and config.moe_decode_row["tok_s"] == _HYBRID_FASTER
    if not flags:
        assert config.moe_cache_auto is True  # a bare start still gets the auto-sized cache


def test_auto_stays_on_offload_when_the_row_says_hybrid_is_slower(decode_rows):
    _file_row(decode_rows, _HYBRID_SLOWER)
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and config.moe_decode_row is None


@pytest.mark.parametrize(
    "served, differs",
    [
        ({"max_running_req": 2}, "req"),
        ({"moe_cache_size": 48}, "cache"),
        ({"memory_ratio": 0.8}, "cache"),
        ({"moe_cpu_threads": 6}, "cpu"),
        ({"speculative_algorithm": "mtp", "speculative_num_steps": 3}, "spec"),
        ({"moe_hybrid_max_fetch": 2}, "moe_hybrid_max_fetch"),
        ({"moe_cache_scope": "per-layer"}, "moe_cache_scope"),
        ({"moe_cpu_layers": "2"}, "moe_cpu_layers"),
        ({"kv_reserve_tokens": 2048}, "kv_reserve_tokens"),
        ({"cuda_graph_max_bs": 1}, "cuda_graph_max_bs"),
        ({"moe_collect_stats": True}, "moe_collect_stats"),
        ({"hf_overrides": {"rope_scaling": {"factor": 2}}}, "hf_overrides"),
    ],
)
def test_auto_does_not_use_a_row_measured_with_other_settings(decode_rows, served, differs, monkeypatch):
    # speculative decoding resolves through the proposer; keep that out of this test
    monkeypatch.setattr(decode_rows.engine, "_adjust_spec_config", lambda *a, **k: None)
    key = _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"  # the row is good for its own settings
    config = _resolved(decode_rows, **served)
    assert config.moe_strategy == "offload"
    # and the log can name the setting: that is what tells a user why auto did not take the row
    assert differs in decode_rows.dp.key_diff(key, config.moe_decode_key)


# the keys these command lines got before the preset set a prefill demand length (7cad44b), with the stand-ins below, and
# the draft expert quant the preset fills in since: a row measured with the draft's bf16 experts is another setup
_KEY_HEAD = "ckpt@bf41ffe2|cache=auto:0.95|cpu=0/4@e5b47b|spec=mtp:3|tp=1|req=1|dtype=bfloat16|embed_weights=host|kv_reserve_tokens=36864|moe_prefill_hit_d2d=True"
_KEY_SINGLE = _KEY_HEAD + "|speculative_draft_quant=fp8"
_KEY_SINGLE_FAST = _KEY_HEAD + "|prefill_graph_chunks=1-64,128-384:64|prefill_graph_tokens=448|speculative_draft_quant=fp8"


@pytest.mark.parametrize(
    "flags, key",
    [
        (("--preset", "single"), _KEY_SINGLE),
        (("--preset", "single-measured"), _KEY_SINGLE),
        (("--preset", "single-fast"), _KEY_SINGLE_FAST),
        # a demand length changes a prefill only: typed, it does not make another setup either, nor did the preset's going from 96 to 512
        (("--preset", "single", "--moe-prefill-demand-tokens", "32"), _KEY_SINGLE),
        (("--preset", "single", "--moe-prefill-demand-tokens", "96"), _KEY_SINGLE),
        (("--moe-prefill-demand-tokens", "32"), "ckpt@bf41ffe2|cache=auto:0.9|cpu=0/4@e5b47b|spec=off|tp=1|req=4|dtype=bfloat16"),
    ],
)
def test_a_preset_start_finds_the_rows_filed_before_its_prefill_demand_length(decode_rows, monkeypatch, flags, key):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(decode_rows.dp, "perf_env", lambda environ=None: {})
    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args = parse_args(["--model", str(decode_rows.model), *flags])[0]
    assert decode_rows.engine._decode_decision_key(args) == key


def test_auto_tells_speculative_methods_and_environment_switches_apart(decode_rows, monkeypatch):
    monkeypatch.setattr(decode_rows.engine, "_adjust_spec_config", lambda *a, **k: None)
    _file_row(decode_rows, _HYBRID_FASTER, speculative_algorithm="mtp", speculative_num_steps=3)
    assert _resolved(decode_rows, speculative_algorithm="mtp", speculative_num_steps=3).moe_strategy == "hybrid"
    assert _resolved(decode_rows, speculative_algorithm="dflash", speculative_num_steps=3).moe_strategy == "offload"
    _file_row(decode_rows, _HYBRID_FASTER)
    monkeypatch.setenv("FREETOKEN_API_KEY", "not a setup")  # ignored, and never written into a key
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setenv("FREETOKEN_CPU_MOE_SPIN", "0")
    config = _resolved(decode_rows)
    assert config.moe_strategy == "offload" and "not a setup" not in config.moe_decode_key
    monkeypatch.delenv("FREETOKEN_CPU_MOE_SPIN")
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setenv("DSV4_SWITCH_NOBODY_BENCHED", "1")  # the kernels' own switches do not start with FREETOKEN_
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_environment_switches_the_source_reads_are_in_the_decode_key_or_named_as_left_out():
    """A switch that changes decode speed and is not in the key lets auto apply a row measured without it."""
    import re
    from pathlib import Path

    import freetoken
    from freetoken.moe import decode_profile

    literal = re.compile(r"""["']([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["']""")
    unknown = set()
    for path in Path(freetoken.__file__).parent.rglob("*"):
        if path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h"):
            for line in path.read_text(errors="replace").splitlines():
                if "environ" in line or "getenv" in line:
                    unknown.update(name for name in literal.findall(line) if decode_profile.env_role(name) is None)
    assert not unknown, f"say in moe/decode_profile.py whether these are part of the decode key: {sorted(unknown)}"


def test_auto_does_not_use_a_row_from_another_release_machine_gpu_or_checkpoint(decode_rows, monkeypatch):
    dp, engine = decode_rows.dp, decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    now = dp.stamp(*_ROW_GPU)
    for name, other in (("torch", "0.0"), ("triton", "0.0"), ("nvidia", "another driver"), ("cpu", "another CPU"), ("ram_gib", 1),
                        ("cpu_quota", "200000 100000")):
        with monkeypatch.context() as m:
            m.setattr(dp, "stamp", lambda gpu_name=None, gpu_uuid=None, name=name, other=other: {**now, name: other})
            assert _resolved(decode_rows).moe_strategy == "offload", name
    with monkeypatch.context() as m:
        m.setattr("freetoken.version.__version__", "9.9.9+gdef")  # another commit of the same release: the row holds
        assert _resolved(decode_rows).moe_strategy == "hybrid"
        m.setattr("freetoken.version.__version__", "9.9.10")  # another release: the row is stale
        assert _resolved(decode_rows).moe_strategy == "offload"
    with monkeypatch.context() as m:
        m.setattr(dp, "_first_line", lambda path, prefix="": None)  # no /proc to read the machine from, as on anything but Linux
        assert _resolved(decode_rows).moe_strategy == "offload"
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: ("FAKE GPU", "GPU-other"))
    assert _resolved(decode_rows).moe_strategy == "offload"
    monkeypatch.setattr(engine, "_profile_gpu", lambda index=None: _ROW_GPU)
    (decode_rows.model / "README.md").write_text("not a weight file")
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    (decode_rows.model / "model-00001.safetensors").write_bytes(b"x" * 101)  # the checkpoint changed on disk
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_auto_drops_the_row_when_the_bandwidth_profile_or_the_cpu_set_changes(decode_rows, monkeypatch):
    import json

    from freetoken.moe.bench_profile import PROFILE_VERSION

    def bench_bw(cpu_gbs):
        profile = {"version": PROFILE_VERSION, "gpu": {"name": _ROW_GPU[0]}, "layer_kinds": {"nvfp4:2048x512:top8:t12": {"cpu_moe_overlap_gbs": cpu_gbs}}}
        (decode_rows.tmp / "benchbw.json").write_text(json.dumps(profile))

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    bench_bw(170.0)
    _file_row(decode_rows, _HYBRID_FASTER)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    bench_bw(90.0)  # hybrid's fetch split changes with it
    assert _resolved(decode_rows).moe_strategy == "offload"
    bench_bw(170.0)
    assert _resolved(decode_rows).moe_strategy == "hybrid"
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})  # as many CPUs, but other ones
    assert _resolved(decode_rows).moe_strategy == "offload"


def test_auto_never_takes_hybrid_where_it_cannot_run_or_is_not_the_default(decode_rows, monkeypatch):
    from freetoken.distributed import DistributedInfo

    engine = decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)
    with monkeypatch.context() as m:
        m.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: False)
        assert _resolved(decode_rows).moe_strategy == "offload"
    with monkeypatch.context() as m:  # unified memory keeps its own default
        m.setattr(engine, "_is_unified_memory_gpu", lambda index=None: True)
        m.setattr(engine, "_fused_resident_ok", lambda model_config: True)
        assert _resolved(decode_rows).moe_strategy == "fused"
    two_ranks = DistributedInfo(rank=0, size=2)  # each rank would read the rows of its own GPU
    _file_row(decode_rows, _HYBRID_FASTER, tp_info=two_ranks)
    assert _resolved(decode_rows, tp_info=two_ranks).moe_strategy == "offload"


def test_auto_judges_the_arm_of_the_rounding_mode_the_server_runs(decode_rows, monkeypatch):
    _file_row(decode_rows, _TWO_ARMS, guard=["repeat"])
    assert _resolved(decode_rows).moe_strategy == "offload"  # min-wait is what runs by default, and it is outside the guard limit
    monkeypatch.setenv("FREETOKEN_HYBRID_FETCH_ROUND", "stochastic")
    assert _resolved(decode_rows).moe_strategy == "hybrid"


def test_explicit_strategy_never_reads_the_decode_profile(decode_rows, monkeypatch):
    _file_row(decode_rows, _HYBRID_SLOWER, moe_cache_size=40)
    monkeypatch.setattr(decode_rows.dp, "lookup", lambda *a, **k: pytest.fail("an explicit strategy read the decode profile"))
    for asked in ("hybrid", "offload"):
        config = _resolved(decode_rows, moe_strategy=asked, moe_cache_size=40)
        assert config.moe_strategy == asked and config.moe_decode_key is None  # nor is the key made for it


@pytest.mark.parametrize("draft_quant", [False, True])  # the fields of --preset single, with and without its draft quant
def test_preset_speculative_decoding_is_dropped_where_an_explicit_start_is_refused(decode_rows, monkeypatch, draft_quant):
    def resolved(preset: bool, model_path=str(decode_rows.model)):
        fields = ("speculative_algorithm", "speculative_num_steps") if preset else ()
        quant = {}
        if preset and draft_quant:
            fields, quant = (*fields, "speculative_draft_quant"), {"speculative_draft_quant": "fp8"}
        config = _offload_engine_config(
            model_path=model_path, speculative_algorithm="mtp", speculative_num_steps=3, preset="single" if preset else None, preset_fields=fields,
            **quant,
        )
        # a checkpoint of a family that registers no MTP draft
        object.__setattr__(config, "hf_config", SimpleNamespace(architectures=["FakeForCausalLM"]))
        object.__setattr__(config, "model_spec", SimpleNamespace(spec_drafts=()))
        decode_rows.engine._adjust_config(config)
        return config

    with pytest.raises(ValueError, match="MTP is not supported"):
        resolved(preset=False)
    config = resolved(preset=True)
    assert (config.speculative_algorithm, config.speculative_num_steps, config.preset_fields) == (None, 0, ())
    assert config.speculative_draft_quant == "none"
    assert "|spec=off|" in config.moe_decode_key and "speculative_draft_quant" not in config.moe_decode_key
    # where the checks pass it stays, and a checkpoint that is not on disk cannot be checked
    monkeypatch.setattr(decode_rows.engine, "_adjust_spec_config", lambda config, override, windowed_kv: None)
    config = resolved(preset=True)
    assert config.speculative_algorithm == "mtp" and "speculative_algorithm" in config.preset_fields
    gone = resolved(preset=True, model_path=str(decode_rows.tmp / "not-downloaded"))
    assert (gone.speculative_algorithm, gone.speculative_draft_quant) == (None, "none")


def test_preset_draft_quant_is_fp8_with_every_expert_resident_and_nvfp4_on_an_expert_cache(decode_rows, monkeypatch):
    engine = decode_rows.engine
    refused = set()  # the draft quants the stand-in for the drafts' checks refuses

    def check(config, override, windowed_kv):
        if config.speculative_draft_quant in refused:
            raise ValueError(f"the draft's experts are already {config.speculative_draft_quant}")

    monkeypatch.setattr(engine, "_adjust_spec_config", check)
    fitting = []
    monkeypatch.setattr(engine, "_preset_takes_fused", lambda config: bool(fitting))
    preset = ("speculative_algorithm", "speculative_num_steps", "speculative_draft_quant", "moe_strategy")

    def resolved(fields=preset, quant="fp8", fits=False, **flags):
        flags = {"speculative_algorithm": "mtp", "speculative_num_steps": 3, **flags}
        fitting[:] = [True] if fits else []
        config = _offload_engine_config(
            model_path=str(decode_rows.model), preset="single", preset_fields=fields, speculative_draft_quant=quant, **flags,
        )
        engine._adjust_config(config)
        return config

    assert (resolved(fits=True).moe_strategy, resolved(fits=True).speculative_draft_quant) == ("fused", "fp8")
    config = resolved()
    assert (config.moe_strategy, config.speculative_draft_quant) == ("offload", "nvfp4")
    assert "speculative_draft_quant=nvfp4" in config.moe_decode_key
    assert resolved(moe_strategy="fused").speculative_draft_quant == "fp8"
    assert resolved(moe_strategy="hybrid").speculative_draft_quant == "nvfp4"
    # `ft bench decode`'s servers type the strategy: their rows are filed under the key the preset's auto looks up
    _file_row(decode_rows, _HYBRID_FASTER, asked="hybrid", preset="single", preset_fields=preset, speculative_algorithm="mtp",
              speculative_num_steps=3, speculative_draft_quant="fp8")
    assert resolved().moe_strategy == "hybrid"
    # a quant the command line typed, "none" included, is served as typed
    no_quant = tuple(n for n in preset if n != "speculative_draft_quant")
    assert resolved(fields=no_quant, quant="nvfp4", fits=True).speculative_draft_quant == "nvfp4"
    assert resolved(fields=no_quant, quant="none").speculative_draft_quant == "none"
    # a draft that refuses the quant keeps its drafts, with its experts as stored
    refused.add("fp8")
    config = resolved(fits=True)
    assert (config.speculative_algorithm, config.speculative_draft_quant) == ("mtp", "none")
    assert "speculative_draft_quant" not in config.preset_fields
    refused.clear()
    # no speculative decoding: the preset's quant goes with it, where a typed one is refused
    config = resolved(speculative_algorithm=None, speculative_num_steps=0)
    assert config.speculative_draft_quant == "none" and "speculative_draft_quant" not in config.preset_fields
    with pytest.raises(ValueError, match="needs --speculative-algorithm mtp"):
        resolved(fields=no_quant, speculative_algorithm=None, speculative_num_steps=0)


def test_preset_serves_every_expert_resident_where_the_checkpoint_fits_and_the_strategy_was_left_alone(decode_rows, monkeypatch):
    engine = decode_rows.engine
    # stand-ins for what the checkpoint's own shapes give: 1000 B of KV a token in 16-token pages and 500 B fixed,
    # 300 B of linear state, 7000 B for a prefill chunk; the weight file on disk is 100 B
    monkeypatch.setattr(engine, "resolve_pool_class", lambda model: SimpleNamespace(kv_cost=lambda config: (16_000, 500, 16, 0)))
    monkeypatch.setattr(engine, "state_pool_bytes", lambda config: 300)
    monkeypatch.setattr(engine, "_prefill_working_bytes", lambda config: (8192, 7000))
    need = 100 + 8192 * 1000 + 500 + 300 + 7000

    def strategy(free, fields=("moe_strategy",), preset="single", **flags):
        monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: free)
        config = _resolved(decode_rows, preset=preset, preset_fields=fields, max_seq_len_override=40960, **flags)
        return config.moe_strategy

    enough = -(-need * 20 // 19)  # the preset's memory ratio is the user's here: the default 0.9 would need more
    assert strategy(enough, memory_ratio=0.95) == "fused" and strategy(enough - 64, memory_ratio=0.95) == "offload"
    assert strategy(None, memory_ratio=0.95) == "offload"  # the free memory cannot be read
    # the larger reserve of the preset counts: 36864 tokens do not fit where 8192 did; so does a KV pool sized on the command line
    assert strategy(enough, memory_ratio=0.95, kv_reserve_tokens=36864) == "offload"
    assert strategy(enough, memory_ratio=0.95, num_token_override=8192) == "fused"
    assert strategy(enough, memory_ratio=0.95, num_token_override=8193) == "offload"
    assert strategy(enough, memory_ratio=0.95, num_page_override=8193) == "offload"
    # a reserve above the context: 'single' counts one context of it, 'multi' the whole of it up to a context a running request
    context = -(-(need + (40960 - 8192) * 1000) * 20 // 19)
    whole = -(-(need + (65536 - 8192) * 1000) * 20 // 19)
    assert strategy(context, memory_ratio=0.95, kv_reserve_tokens=65536, max_running_req=2) == "fused"
    multi = dict(preset="multi", memory_ratio=0.95, kv_reserve_tokens=65536)
    assert strategy(context, max_running_req=2, **multi) == "offload" and strategy(whole, max_running_req=2, **multi) == "fused"
    assert strategy(context, max_running_req=1, **multi) == "fused"
    # the decode graphs of the batch sizes a preset names count: 8 sizes of one row (no drafts here), and of 4 rows with 3 drafts
    graphs = 8 * engine.PRESET_DECODE_GRAPH_ROW_BYTES
    sized = dict(memory_ratio=0.95, max_running_req=8, cuda_graph_bs=list(range(1, 9)))
    assert strategy(enough, **sized) == "offload" and strategy(-(-(need + graphs) * 20 // 19), **sized) == "fused"
    assert engine._preset_decode_graph_bytes(SimpleNamespace(preset="multi", cuda_graph_bs=[1, 2, 8], speculative_algorithm="mtp", speculative_num_steps=3)) == 4 * graphs
    assert engine._preset_decode_graph_bytes(SimpleNamespace(preset="single", cuda_graph_bs=None, speculative_algorithm="mtp", speculative_num_steps=3)) == 0
    assert engine._preset_decode_graph_bytes(SimpleNamespace(preset=None, cuda_graph_bs=[1, 2], speculative_algorithm=None, speculative_num_steps=0)) == 0
    # a strategy from the command line, typed auto included, is never replaced; nor is one without the preset
    assert strategy(1 << 40, fields=()) == "offload" and strategy(1 << 40, fields=(), moe_strategy="hybrid") == "hybrid"
    assert _resolved(decode_rows, max_seq_len_override=40960).moe_strategy == "offload"
    # an expert cache the command line asks for is a cache; a checkpoint that is not on disk cannot be counted
    assert strategy(1 << 40, moe_cache_rate=0.5) == "offload" and strategy(1 << 40, moe_cache_auto=True) == "offload"
    monkeypatch.setattr(engine, "_checkpoint_bytes", lambda path: None)
    assert strategy(1 << 40) == "offload"


def test_preset_single_fast_serves_as_single_where_its_captures_do_not_fit(decode_rows, monkeypatch):
    from freetoken.engine.engine import PRESET_CAPTURE_BYTES, Engine

    engine = decode_rows.engine
    captures = ("prefill_graph_tokens", "prefill_graph_chunks")
    monkeypatch.setattr(engine, "resolve_pool_class", lambda model: SimpleNamespace(kv_cost=lambda config: (16_000, 500, 16, 0)))
    monkeypatch.setattr(engine, "state_pool_bytes", lambda config: 300)
    monkeypatch.setattr(engine, "_prefill_working_bytes", lambda config: (8192, 7000))
    monkeypatch.setattr(engine, "_adjust_spec_config", lambda config, override, windowed_kv: None)
    monkeypatch.setattr(engine, "_spec_runner", lambda: SimpleNamespace(budget=lambda config: (0, 0)))
    need = 100 + 8192 * 1000 + 500 + 300 + 7000

    def resolved(free, **flags):
        monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: free)
        flags = {"speculative_algorithm": "mtp", "speculative_num_steps": 3, "memory_ratio": 1.0, "preset_fields": ("moe_strategy", *captures), **flags}
        return _resolved(
            decode_rows, preset="single-fast", max_seq_len_override=40960, prefill_graph_tokens=448, prefill_graph_chunks="1-64,128-384:64", **flags,
        )

    def kept(config):
        return (config.prefill_graph_tokens, config.prefill_graph_chunks) == (448, "1-64,128-384:64") and set(captures) <= set(config.preset_fields)

    def single(config):
        return (config.prefill_graph_tokens, config.prefill_graph_chunks) == (0, "") and not set(captures) & set(config.preset_fields)

    # resident experts: the captures' memory counts; where only it does not fit the server is --preset single's, fused
    config = resolved(need + PRESET_CAPTURE_BYTES)
    assert config.moe_strategy == "fused" and kept(config)
    config = resolved(need + PRESET_CAPTURE_BYTES - 1)
    assert config.moe_strategy == "fused" and single(config)
    # not even the experts fit: an expert cache, and its plan decides (below)
    config = resolved(need - 1)
    assert config.moe_strategy == "offload" and kept(config)
    # no speculative decoding, no verify step to capture them beside
    assert single(resolved(1 << 40, speculative_algorithm=None, speculative_num_steps=0))

    # an automatic expert cache, Qwen3.6-35B-A3B's sizes: the captures stay where the plan, with their memory left free,
    # has every expert outside the staging buffers and the preset's KV reserve; one byte less and they go
    slot, weights, page = 1_780_000, int(4.44 * 2**30), 16 * 20_000

    def planned(free, staging=512, **flags):
        config = resolved(1 << 40, moe_strategy="offload", kv_reserve_tokens=36864, **flags)
        terms = dict(
            baseline_free=free, weights_bytes=weights, memory_ratio=1.0, cache_per_page=page, fixed_cache_size=PRESET_CAPTURE_BYTES,
            per_expert_bytes=slot, num_experts=256, total_experts=10240, prefill_overlap=True, kv_reserve_tokens=36864, page_size=16,
        )
        stub = SimpleNamespace(_auto_cache_terms=lambda *a: dict(terms), _moe_slot_limit=None)
        Engine._settle_preset_captures(stub, config, None, None, None, staging)
        return config

    fits = weights + PRESET_CAPTURE_BYTES + (10240 + 512) * slot + (36864 // 16) * page
    assert kept(planned(fits)) and single(planned(fits - 1))
    assert single(planned(16 << 30)) and single(planned(1 << 40, staging=0))  # a small card; several pools or CPU experts
    # typed chunks are not the preset's to drop
    typed = planned(16 << 30, preset_fields=("moe_strategy", "prefill_graph_tokens"))
    assert typed.prefill_graph_chunks == "1-64,128-384:64" and typed.prefill_graph_tokens == 448


def test_preset_serves_the_stores_draft_token_map_only_where_every_expert_is_resident(decode_rows, monkeypatch):
    import hashlib

    from freetoken.moe import calibrate

    engine = decode_rows.engine
    monkeypatch.setattr(engine, "resolve_pool_class", lambda model: SimpleNamespace(kv_cost=lambda config: (16_000, 500, 16, 0)))
    monkeypatch.setattr(engine, "state_pool_bytes", lambda config: 300)
    monkeypatch.setattr(engine, "_prefill_working_bytes", lambda config: (8192, 7000))
    monkeypatch.setattr(engine, "_spec_runner", lambda: SimpleNamespace(budget=lambda config: (0, 0)))
    checked = []

    def adjust_spec(config, override, windowed_kv):
        checked.append((config.speculative_num_steps, config.speculative_token_map))
        if config.speculative_token_map and "refused" in config.speculative_token_map:
            raise ValueError("--speculative-token-map needs an LM head with its own weights")
        override("spec_slot_states", ("draft state", config.speculative_num_steps))

    monkeypatch.setattr(engine, "_adjust_spec_config", adjust_spec)
    lines = []
    monkeypatch.setattr(engine.logger, "info_rank0", lines.append)
    drafts = ("speculative_algorithm", "speculative_num_steps")
    partial = 100 + 8192 * 1000 + 500 + 300 + 7000 - 1  # one byte short of every expert resident
    last = [None]

    def resolved(free=1 << 40, preset="single", fields=("moe_strategy", *drafts), **flags):
        lines.clear()
        monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: free)
        flags = {"speculative_algorithm": "mtp", "speculative_num_steps": 3, "memory_ratio": 1.0, **flags}
        last[0] = config = _resolved(decode_rows, preset=preset, preset_fields=fields, max_seq_len_override=40960, **flags)
        said = [line for line in lines if line.startswith(f"--preset {preset} with every expert resident")]
        assert len(said) <= 1
        return (config.moe_strategy, config.speculative_num_steps, config.speculative_token_map), (said or [""])[0]

    # no map in the store: the preset's 3 drafts, without one
    served, said = resolved()
    assert served == ("fused", 3, None) and "3 MTP draft tokens without a draft token map: the store holds none" in said
    path = calibrate.token_map_path(str(decode_rows.model))
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(b"\x00\x00\x00\x00\x01\x00\x00\x00")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()

    def manifest(**fields):
        with open(path + ".json", "w") as f:
            json.dump(fields, f)

    manifest(map_sha256=sha)
    served, said = resolved()
    assert served == ("fused", 5, path) and said == f"--preset single with every expert resident: 5 MTP draft tokens with the store's draft token map {path} (sha256 {sha[:12]}, as its manifest records)"
    # checked as a typed map is, and the setup's key is the one that runs
    assert checked[-1] == (5, path) and "|spec=mtp:5|" in last[0].moe_decode_key and last[0].spec_slot_states == ("draft state", 5)
    # single-fast follows; single-measured is its own calibration's (moe/calibrate.py)
    assert resolved(preset="single-fast")[0] == ("fused", 5, path)
    assert resolved(preset="single-measured") == (("fused", 3, None), "")
    # a file that is not the one its manifest describes, a manifest that records none, a map the checks refuse
    manifest(map_sha256="0" * 64)
    served, said = resolved()
    assert served == ("fused", 3, None) and f"has sha256 {sha[:12]}, its manifest records 000000000000" in said
    manifest(sha256=sha)  # the store reads "map_sha256" only (scripts/make_token_map.py --install writes it)
    assert resolved()[0] == ("fused", 3, None)
    manifest(map_sha256=sha)
    monkeypatch.setattr(calibrate, "token_map_path", lambda model_path: path + ".refused")
    with open(path + ".refused", "wb") as f:
        f.write(open(path, "rb").read())
    with open(path + ".refused.json", "w") as f:
        json.dump({"map_sha256": sha}, f)
    served, said = resolved()
    assert served == ("fused", 3, None) and "is refused (--speculative-token-map needs an LM head" in said
    assert last[0].spec_slot_states == ("draft state", 3) and "|spec=mtp:3|" in last[0].moe_decode_key
    monkeypatch.setattr(calibrate, "token_map_path", lambda model_path: path)
    # drafts from the command line are served as typed
    served, said = resolved(fields=("moe_strategy", "speculative_algorithm"), speculative_num_steps=4)
    assert served == ("fused", 4, None) and said.endswith("4 mtp draft tokens as --speculative-num-steps set them")
    assert resolved(fields=("moe_strategy",))[0] == ("fused", 3, None)
    assert resolved(speculative_token_map="/maps/typed.bin")[0] == ("fused", 3, "/maps/typed.bin")
    assert resolved(fields=("moe_strategy",), speculative_algorithm=None, speculative_num_steps=0) == (("fused", 0, None), "")
    # a partial expert cache changes nothing, and neither does a typed fused: a calibration candidate's server has one
    assert resolved(free=partial) == (("offload", 3, None), "")
    assert resolved(fields=drafts, moe_strategy="fused") == (("fused", 3, None), "")


def test_preset_keeps_its_drafts_only_where_the_offload_cache_holds_a_fifth_of_the_experts(monkeypatch):
    from freetoken.engine import engine as eng
    from freetoken.engine.engine import PRESET_DRAFT_SHARE, PRESET_NO_DRAFTS, Engine

    assert PRESET_DRAFT_SHARE == 0.20
    lines = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(eng.logger, "info_rank0", lines.append)
    monkeypatch.setattr(eng, "shared_offload_method", lambda model: None)
    monkeypatch.setattr(eng, "offload_layer_methods", lambda model: None)
    bank_bytes = [10240 * 1_780_000 - 5]
    monkeypatch.setattr(eng, "_bank_bytes", lambda config, method=None, layer_methods=None: bank_bytes[0])
    drafts = ("speculative_algorithm", "speculative_num_steps")
    # Qwen3.6-35B-A3B's sizes: 10240 experts of 1.78 MB, 20 KB of KV a token
    slot, weights, page = 1_780_000, int(4.44 * 2**30), 16 * 20_000

    def served(slots=0, mem=None, fields=drafts, strategy="offload", cpu_layers=None, free=None):
        config = SimpleNamespace(
            model_config=SimpleNamespace(num_moe_layers=40, num_experts=256), model=None, moe_cache_size=slots, moe_cache_mem=mem,
            moe_strategy=strategy, moe_cpu_layers=cpu_layers, preset="single", preset_fields=fields, speculative_algorithm="mtp",
            speculative_num_steps=3, spec_slot_states=("draft state",), moe_decode_key=None, kv_reserve_tokens=36864, max_seq_len=262144,
        )
        engine = Engine.__new__(Engine)
        engine.model = None
        reads = iter((1000, 1700))
        engine._sync_get_memory = lambda: (next(reads), 0)
        engine.spec, engine._weights_bytes, engine._post_weights_free = object(), 5000, 3000

        def terms(config, banks, method=None, max_slots=None, per_expert_bytes=None):
            assert banks is None and per_expert_bytes == slot  # a slot's bytes from the bank layout, rounded up
            return dict(
                baseline_free=free, weights_bytes=weights, memory_ratio=1.0, cache_per_page=page, fixed_cache_size=0, per_expert_bytes=slot,
                num_experts=256, total_experts=10240, prefill_overlap=True, kv_reserve_tokens=config.kv_reserve_tokens, page_size=16,
            )

        engine._auto_cache_terms = terms
        engine._settle_preset_drafts(config)
        dropped = engine.spec is None
        if dropped:
            # as a server started without them, and the draft's weights are free for the caches again
            assert (config.speculative_algorithm, config.speculative_num_steps, config.spec_slot_states, config.preset_fields) == (None, 0, (), ())
            assert (engine._weights_bytes, engine._post_weights_free) == (4300, 3700)
        else:
            assert config.speculative_algorithm == "mtp" and config.preset_fields == fields and engine._weights_bytes == 5000
        return dropped

    # a cache the command line sized (--moe-cache-rate 0.05 and 0.2 are 512 and 2048 slots), in slots or in bytes
    assert served(slots=512) and served(slots=2047) and not served(slots=2048)
    assert served(mem=2047 * slot) and not served(mem=2048 * slot)
    assert any(PRESET_NO_DRAFTS in line and "sized has 512 expert slots for 10240 experts" in line for line in lines)
    # an automatic cache: the plan with the drafts at the preset's KV reserve, a slot more or less
    at_share = weights + 2048 * slot + (36864 // 16) * page
    assert not served(free=at_share) and served(free=at_share - 1)
    # the line says what was counted: the plan before the KV reserve is settled, not the cache that is served
    assert "the automatic plan with them, at a KV reserve of 36864 tokens and a slot's bytes from the bank layout, has 2047 expert slots" in lines[-1]
    assert served(free=weights + 100 * slot)  # no plan at all with them: not served with them
    # only offload's own drafts at the preset's own depth: hybrid, CPU layers and a typed depth keep them
    assert not served(slots=512, strategy="hybrid") and not served(slots=512, cpu_layers="2")
    assert not served(slots=512, fields=("speculative_algorithm",))
    bank_bytes[0] = None  # a format whose bank size is not known before the load: not counted, so kept
    assert not served(free=at_share - 1) and served(slots=512)


def test_preset_plan_takes_the_staging_slots_only_where_every_expert_and_the_kv_reserve_still_fit():
    from freetoken.engine.cache_budget import plan_resident_staging

    def terms(free, reserve=64):
        return dict(
            baseline_free=free, weights_bytes=0, memory_ratio=1.0, cache_per_page=160, fixed_cache_size=0, per_expert_bytes=50,
            num_experts=4, total_experts=40, prefill_overlap=True, kv_reserve_tokens=reserve, page_size=16,
        )

    def staged(free, limit=None, **kw):
        t = terms(free, **kw)
        plan = resolve_moe_cache_auto(**t)
        return plan, plan_resident_staging(plan, t, 8, limit)

    # 40 experts of 50 B and 4 pages of reserve are 2640 B; 8 slots more cost 400 B, 3 pages
    plan, with_staging = staged(2640 + 480)
    assert plan == (40, 7, True) and with_staging == (48, 4, True)
    assert staged(2640 + 479)[1] == (40, 6, True)  # the reserve would lose a page
    assert staged(2640 - 50)[1][0] == 39  # not every expert fits: the plan as it is
    capped = {**terms(1 << 20), "max_slots": 30}  # nor under a kernel's slot cap, whatever is left for KV
    assert plan_resident_staging(resolve_moe_cache_auto(**capped), capped, 8)[0] == 30
    assert staged(2640 + 480, limit=47)[1] == plan  # the kernel's slot limit
    assert plan_resident_staging((40, 7, False), terms(2640 + 480), 8) == (40, 7, False)  # no prefill overlap, no staging buffers
    # what the engine then builds: a cache of one slot an expert can never hold them all outside the staging buffers
    from freetoken.moe.offload_cache import OffloadMoeCache

    def cache(extra):
        c = OffloadMoeCache(num_layers=10, num_experts=4, cache_size=40 + extra, device=torch.device("cpu"), prefill_overlap=True, prefill_hit_d2d=True)
        c.set_bank_sources({"gate_up": [torch.randn(4, 32, 8) for _ in range(10)], "down": [torch.randn(4, 8, 16) for _ in range(10)]})
        return c

    assert cache(8).holds_every_expert and not cache(7).holds_every_expert and not cache(0).holds_every_expert


def test_preset_loads_short_prefill_chunks_on_demand_only_into_a_gpu_decoded_cache_that_cannot_hold_every_expert(decode_rows, monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    engine = decode_rows.engine
    monkeypatch.setattr(OffloadMoeCache, "keeps_streamed_experts", property(lambda self: self.prefill_hit_d2d))

    def settled(extra, decode_target="gpu", pool_sizes=None, fields=("moe_prefill_demand_tokens",), tokens=512, by_misses=True):
        monkeypatch.setattr(engine, "_PREFILL_DEMAND_BY_MISSES", by_misses)
        config = SimpleNamespace(
            preset="single", preset_fields=fields, moe_prefill_demand_tokens=tokens, model_config=SimpleNamespace(num_experts=4),
        )
        pool_sizes = pool_sizes or [40 + extra]
        engine._settle_preset_demand(config, decode_target, pool_sizes)  # before the cache: it refuses demand chunks off the GPU
        cache = OffloadMoeCache(
            num_layers=10, num_experts=4, cache_size=40 + extra, device=torch.device("cpu"), prefill_overlap=True, prefill_hit_d2d=True,
            prefill_demand_tokens=config.moe_prefill_demand_tokens, decode_target=decode_target,
        )
        cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8) for _ in range(10)], "down": [torch.randn(4, 8, 16) for _ in range(10)]})
        engine._settle_preset_demand(config, decode_target, pool_sizes, cache)
        assert cache.prefill_demand_tokens == config.moe_prefill_demand_tokens
        # only the preset's chunks may also go by the bytes each way copies, and FREETOKEN_PREFILL_DEMAND_BY_MISSES=0 stops that
        assert cache.prefill_demand_by_misses == (by_misses and "moe_prefill_demand_tokens" in config.preset_fields)
        assert ("moe_prefill_demand_tokens" in config.preset_fields) == (bool(fields) and config.moe_prefill_demand_tokens == 512)
        return config.moe_prefill_demand_tokens

    # 40 experts, 8 staging slots: a slot for every expert outside them, or one short of it
    assert settled(7) == 512 and settled(0) == 512 and settled(8) == 0 and settled(7, by_misses=False) == 512
    assert settled(0, decode_target="hybrid") == 0 and settled(0, decode_target="cpu") == 0
    assert settled(0, pool_sizes=[3, 37]) == 0  # a pool smaller than a layer: a chunk can route to every expert of one
    # typed tokens are the command line's: kept on any cache, and refused by it as without the preset
    assert settled(8, fields=()) == 512 and settled(8, fields=(), tokens=32) == 32
    with pytest.raises(ValueError, match="needs GPU decode"):
        settled(0, decode_target="hybrid", fields=())
    # resident experts have no cache to load into
    fields = ("max_running_req", "moe_prefill_demand_tokens")
    fused = _resolved(decode_rows, preset="single", preset_fields=fields, moe_strategy="fused", moe_prefill_demand_tokens=512)
    assert (fused.moe_prefill_demand_tokens, fused.preset_fields) == (0, ("max_running_req",))
    offload = _resolved(decode_rows, preset="single", preset_fields=fields, moe_prefill_demand_tokens=512)
    assert (offload.moe_strategy, offload.moe_prefill_demand_tokens, offload.preset_fields) == ("offload", 512, fields)
    assert _resolved(decode_rows, moe_strategy="fused", moe_prefill_demand_tokens=512).moe_prefill_demand_tokens == 512


# ---- --preset single-measured (moe/calibrate.py): its rule, the timings it keeps in the decode profile, its fallback ----


def test_calibration_takes_the_largest_worst_lead_above_the_margin_and_times_two_passes_on_a_filled_cache(monkeypatch):
    from freetoken.moe import benchdecode, calibrate
    from freetoken.moe.calibrate import MARGIN, choose

    def both(*prompts):
        return [[v, v] for v in prompts]  # the same tok/s in each pass

    names = ["offload@3", "hybrid@3", "offload@0", "hybrid@0"]
    static = {"offload@3": both(100.0, 200.0)}
    assert MARGIN == 0.10
    assert choose(static, names) == ("offload@3", 0.0)  # no other candidate ran: a failed one has no timing and is not chosen
    assert choose({**static, "hybrid@3": both(150.0, 205.0)}, names)[0] == "offload@3"  # +50% on one prompt, +2.5% on the other
    assert choose({**static, "hybrid@3": both(109.0, 240.0)}, names)[0] == "offload@3"  # +9% on its worse prompt
    assert choose({**static, "hybrid@3": both(120.0, 250.0)}, names) == ("hybrid@3", pytest.approx(0.2))
    assert choose({**static, "hybrid@3": both(120.0, 240.0), "hybrid@0": both(120.0, 240.0)}, names)[0] == "hybrid@3"  # a tie: the one named first
    assert choose({**static, "hybrid@3": both(120.0, 240.0), "hybrid@0": both(126.0, 250.0)}, names)[0] == "hybrid@0"
    assert choose({**static, "offload@0": both(60.0, 90.0), "hybrid@0": both(95.0, 230.0)}, names)[0] == "offload@3"  # slower ones change nothing
    # a pass behind the static choice is no lead, whatever the other pass says
    assert choose({**static, "hybrid@3": [[150.0, 95.0], [300.0, 260.0]]}, names)[0] == "offload@3"
    # the margin is asked of a prompt's mean over its passes: one pass at +5% beside one at +50% is +27.5%
    assert choose({**static, "hybrid@3": [[150.0, 105.0], [300.0, 260.0]]}, names) == ("hybrid@3", pytest.approx(0.275))
    # the depth is chosen by the same rule: 5 drafts lead one prompt by 30% and the other by 4%, 4 drafts lead both by 12%
    depths = ["hybrid@3", "hybrid@4", "hybrid@5"]
    timed = {"hybrid@3": both(100.0, 200.0), "hybrid@4": both(112.0, 224.0), "hybrid@5": both(130.0, 208.0)}
    assert choose(timed, depths) == ("hybrid@4", pytest.approx(0.12))
    # the timings of a 16 GB card: +25% to +36% in three passes, +9.7% in the fourth
    card = {"offload@3": [[145.4, 143.8], [143.9, 151.3]], "hybrid@3": [[181.9, 195.7], [193.9, 165.9]]}
    assert choose(card, names) == ("hybrid@3", pytest.approx(0.219, abs=0.001))

    # a candidate's server, through the bench's own code: a warm-up that fills the cache, then two or more timed passes a prompt
    seen = {}

    def run_arm(ns, extra, arm, rate, single, together, sampling, floors=None, passes=None, warm_after=None, deadline=None, grace=30.0):
        seen.update(passes=ns.seeds, warm=ns.warm, warm_after=warm_after, timeout=ns.server_timeout, extra=extra,
                    prompts=[name for name, _ in single], together=together)
        cols = [{"tok_s": (a + b) / 2, "seeds": [{"tok_s": a}, {"tok_s": b}], "draft_n": 4, "early": a, "gave_up": False} for a, b in ((90.0, 110.0), (180.0, 220.0))]
        return {"tok_s": [c["tok_s"] for c in cols], "results": cols, "cache_slots": 40, "total_slots": 80, "slot_bytes": 1 << 20, "gpu_free_bytes": None, "ready_s": 1.0}

    monkeypatch.setattr(benchdecode, "run_arm", run_arm)
    monkeypatch.setattr(benchdecode, "eos_strings", lambda model: ())
    argv = ["--preset", "single-measured", "--calibrate-again", "--memory-ratio", "0.8"]
    timed = calibrate._time_candidate(SimpleNamespace(model_path="/models/anon"), argv, "offload", 3, ready_s=123.9)
    assert timed["tok_s"] == [[90.0, 110.0], [180.0, 220.0]] and timed["drafts"] == 8
    assert seen["passes"] == calibrate.PASSES >= 2 and seen["warm"] == calibrate.WARM >= 1024
    # the later passes find the cache filled and warm up for less; the server has the seconds it was given to load, and
    # is the static preset with the rest of the command line, never one that calibrates again
    assert seen["warm_after"] == calibrate.WARM_AGAIN < calibrate.WARM and seen["timeout"] == 123
    assert seen["extra"] == ["--memory-ratio", "0.8", "--preset", "single"]
    assert seen["prompts"] == list(calibrate.PROMPTS) and not seen["together"]


def test_measured_preset_times_its_candidates_once_and_falls_back_to_the_static_preset(decode_rows, monkeypatch):
    from unittest.mock import patch

    from freetoken.moe import calibrate
    from freetoken.server.args import parse_args

    def parsed(*flags):
        argv = ["--model", str(decode_rows.model), "--preset", "single-measured", *flags]
        hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
        with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
            return parse_args(argv)[0], argv

    speeds = {("offload", 3): [100.0, 200.0], ("hybrid", 3): [130.0, 250.0], ("offload", 0): [60.0, 90.0],
              ("hybrid", 0): [70.0, 120.0], ("offload", 5): [105.0, 190.0], ("hybrid", 5): [110.0, 195.0],
              ("offload", 4): [102.0, 195.0], ("hybrid", 4): [100.0, 190.0]}
    started = []

    def time_candidate(args, argv, strategy, depth, floors=None, ready_s=None, token_map=None, left_s=None):
        depth = f"{depth}map" if token_map else depth
        started.append((strategy, depth))
        if (strategy, depth) not in speeds:
            raise RuntimeError("server exited with code 1 during startup")
        return {"tok_s": [[v, v] for v in speeds[strategy, depth]], "drafts": 100 if depth else 0, "early": speeds[strategy, depth], "cache_slots": 40,
                "total_slots": 80, "slot_bytes": 1 << 20, "gpu_free_bytes": _ROW_FREE, "ready_s": 1.0}

    def starts(args, argv):
        before = len(started)
        return calibrate.measured_choice(args, argv), len(started) - before

    def free_now(free):
        monkeypatch.setattr(calibrate, "_probe_gpu", lambda args: (*_ROW_GPU, free))

    static_strategy = ["offload"]
    monkeypatch.setattr(calibrate, "_time_candidate", time_candidate)
    monkeypatch.setattr(calibrate, "_static_strategy", lambda args, argv: static_strategy[0])
    monkeypatch.setattr(calibrate, "_sleep", lambda seconds: None)
    free_now(_ROW_FREE)
    args, argv = parsed()
    hybrid_3 = {"moe_strategy": "hybrid", "moe_cache_auto": True, "speculative_num_steps": 3}
    # the faster candidate does not start, once: the static choice runs, and the next start times again and takes it
    hybrid = speeds.pop(("hybrid", 3))
    assert starts(args, argv) == ({"speculative_num_steps": 3}, 8)
    speeds["hybrid", 3] = hybrid
    assert starts(args, argv) == (hybrid_3, 8) and starts(args, argv) == (hybrid_3, 0)
    # a candidate that fails on two starts in a row is left out from then on, not timed at every start
    del speeds["hybrid", 5]
    again, again_argv = parsed("--memory-ratio", "0.6")
    assert [starts(again, again_argv)[1] for _ in range(3)] == [8, 8, 0]
    speeds["hybrid", 5] = [110.0, 195.0]
    args, argv = parsed("--memory-ratio", "0.65")
    assert starts(args, argv) == (hybrid_3, 8)
    # the same checkpoint, flags, machine and release: the stored timings answer and no server is started
    assert starts(args, argv) == (hybrid_3, 0)
    # a strategy from the command line is not a candidate: the four depths of that strategy are another setup
    assert starts(*parsed("--moe-strategy", "offload")) == ({"speculative_num_steps": 3}, 4)
    # a cache rate or size from the command line is its own setup, not the automatic size: each is timed for itself
    for flags in (("--moe-cache-rate", "0.2"), ("--moe-cache-rate", "0.05"), ("--moe-cache-size", "512")):
        sized = {"moe_strategy": "hybrid", "speculative_num_steps": 3}
        assert starts(*parsed(*flags)) == (sized, 8), flags
        assert starts(*parsed(*flags)) == (sized, 0)
    # an automatically sized cache follows the free memory (40 slots of 1 MiB were timed): two slots more or less are the
    # same setup; clearly more, clearly less, or a row that does not say how much was free, is timed again
    for free, timed in ((_ROW_FREE - (1 << 20), 0), (_ROW_FREE + (1 << 20), 0), (_ROW_FREE - (8 << 30), 8), (_ROW_FREE - (8 << 30) + (1 << 20), 0),
                        (_ROW_FREE + (8 << 30), 8)):
        free_now(free)
        assert starts(args, argv) == (hybrid_3, timed), free
    profile = decode_rows.tmp / "benchdecode.json"
    kept = json.loads(profile.read_text())
    assert all(row.pop("gpu_free_bytes") for row in kept["decisions"].values())
    profile.write_text(json.dumps(kept))
    assert starts(args, argv) == (hybrid_3, 8) and starts(args, argv) == (hybrid_3, 0)
    assert starts(*parsed("--moe-cache-rate", "0.2")) == (sized, 0)  # a cache of a fixed size is the same at any free memory
    # another release: timed again
    monkeypatch.setattr("freetoken.version.__version__", "9.9.10")
    assert starts(args, argv) == (hybrid_3, 8)
    # a checkpoint that drafts nothing: the depths are one server, so only the strategies are timed
    speeds.update({("offload", 3): [60.0, 90.0], ("hybrid", 3): [80.0, 120.0]})
    monkeypatch.setattr(calibrate, "_time_candidate", lambda args, argv, s, d, floors=None, ready_s=None, left_s=None: {**time_candidate(args, argv, s, d), "drafts": 0})
    assert starts(*parsed("--memory-ratio", "0.9")) == (hybrid_3, 2)
    # the static preset dropped its drafts for the size of the cache: that is the static choice, the other depths of its
    # strategy are not timed against it, and hybrid's are
    def no_drafts(args, argv, s, d, floors=None, ready_s=None, left_s=None):
        return {**time_candidate(args, argv, s, d), "drafts": 0 if (s, d) == ("offload", 3) else 100, "no_drafts": (s, d) == ("offload", 3)}

    monkeypatch.setattr(calibrate, "_time_candidate", no_drafts)
    choice, timed = starts(*parsed("--memory-ratio", "0.91"))
    assert timed == 5 and started[-5:] == [("offload", 3), ("hybrid", 3), ("hybrid", 4), ("hybrid", 5), ("hybrid", 0)]
    assert (choice["moe_strategy"], choice["speculative_num_steps"]) == ("hybrid", 5) and "speculative_num_steps" not in choice["preset_fields"]
    # the static choice cannot be timed, or the GPU cannot be read: the static preset, and nothing is kept
    monkeypatch.setattr(calibrate, "_time_candidate", time_candidate)
    del speeds["offload", 3]
    third, third_argv = parsed("--memory-ratio", "0.8")
    assert starts(third, third_argv) == ({}, 1)
    # hybrid with 3 drafts is now the slower one: the static choice, measured. The strategy stays as the command line
    # left it (auto), so that auto still resolves it as it does for --preset single
    speeds["offload", 3] = [100.0, 200.0]
    assert third.moe_strategy == "auto" and starts(third, third_argv) == ({"speculative_num_steps": 3}, 8)
    # the timings cannot be written (a full or read-only disk): their choice is served all the same, and the next start times again
    free_now(_ROW_FREE)
    fourth, fourth_argv = parsed("--memory-ratio", "0.7")
    with monkeypatch.context() as m:
        m.setattr(calibrate.decode_profile, "store", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left on device")))
        assert starts(fourth, fourth_argv) == ({"speculative_num_steps": 3}, 8)
    assert starts(fourth, fourth_argv) == ({"speculative_num_steps": 3}, 8) and starts(fourth, fourth_argv) == ({"speculative_num_steps": 3}, 0)
    # every expert fits and the static preset serves them resident: fused is the strategy of every candidate, no hybrid
    # is timed against it, and the timings are a row of their own. A deeper draft that leads keeps fused
    static_strategy[0] = "fused"
    speeds.update({("fused", 3): [100.0, 200.0], ("fused", 0): [60.0, 90.0], ("fused", 5): [120.0, 230.0], ("fused", 4): [101.0, 201.0]})
    fifth, fifth_argv = parsed("--memory-ratio", "0.75")
    measured_depth = tuple(n for n in fifth.preset_fields if n != "speculative_num_steps")  # as its server was started: the depth typed
    assert starts(fifth, fifth_argv) == ({"moe_strategy": "fused", "speculative_num_steps": 5, "preset_fields": measured_depth}, 4)
    # every expert on the GPU: the deeper drafts of the static strategy come right after it
    assert started[-4:] == [("fused", 3), ("fused", 5), ("fused", 4), ("fused", 0)] and starts(fifth, fifth_argv)[1] == 0
    speeds["fused", 5] = [105.0, 190.0]
    assert starts(*parsed("--memory-ratio", "0.72")) == ({"speculative_num_steps": 3}, 4)  # the static choice: the strategy stays auto
    # a typed strategy is not asked about and stays, and typed drafts leave nothing to time
    assert starts(*parsed("--moe-strategy", "hybrid", "--memory-ratio", "0.75"))[1] == 4 and started[-1] == ("hybrid", 0)
    assert starts(*parsed("--speculative-algorithm", "off", "--memory-ratio", "0.75")) == ({}, 0)
    # a draft token map for this checkpoint in the store, with the manifest its builder wrote: 5 drafts with it are a
    # candidate, timed right after the static choice, and the choice carries the map
    import hashlib

    map_path = calibrate.token_map_path(str(decode_rows.model))
    assert os.path.dirname(os.path.dirname(map_path)) == str(decode_rows.tmp) and not map_path.startswith(str(decode_rows.model))
    os.makedirs(os.path.dirname(map_path))

    def store_map(data, recorded=None):
        with open(map_path, "wb") as f:
            f.write(data)
        with open(map_path + ".json", "w") as f:
            json.dump({"map_sha256": recorded or hashlib.sha256(data).hexdigest()}, f)

    store_map(b"\x01\x00\x00\x00\x02\x00\x00\x00")
    speeds["fused", "5map"] = [140.0, 240.0]
    mapped, mapped_argv = parsed("--memory-ratio", "0.76")
    with_map = {"moe_strategy": "fused", "speculative_num_steps": 5, "preset_fields": measured_depth, "speculative_token_map": map_path}
    assert starts(mapped, mapped_argv) == (with_map, 5)
    assert started[-5:] == [("fused", 3), ("fused", "5map"), ("fused", 4), ("fused", 5), ("fused", 0)]
    assert starts(mapped, mapped_argv) == (with_map, 0)
    flags = calibrate.child_flags(mapped_argv, 5, map_path)
    assert flags[-6:] == ["--preset", "single", "--speculative-num-steps", "5", "--speculative-token-map", map_path]
    # another map is another candidate, timed again; a file that is not the one its manifest describes is not timed,
    # and a map on the command line is every server's, not a candidate
    store_map(b"\x01\x00\x00\x00\x03\x00\x00\x00")
    assert starts(mapped, mapped_argv) == (with_map, 5)
    store_map(b"\x01\x00\x00\x00\x04\x00\x00\x00", recorded="0" * 64)
    assert starts(*parsed("--memory-ratio", "0.77"))[1] == 4 and ("fused", "5map") not in started[-4:]
    store_map(b"\x01\x00\x00\x00\x04\x00\x00\x00")
    assert starts(*parsed("--memory-ratio", "0.78", "--speculative-token-map", map_path))[1] == 4
    # a map the engine refuses leaves that server without drafts: it is not the candidate it is named for, and however
    # fast it was it is not chosen
    def refused(args, argv, s, d, floors=None, ready_s=None, token_map=None, left_s=None):
        return {**time_candidate(args, argv, s, d, token_map=token_map), **({"drafts": 0} if token_map else {})}

    monkeypatch.setattr(calibrate, "_time_candidate", refused)
    speeds["fused", "5map"] = [400.0, 800.0]
    assert starts(*parsed("--memory-ratio", "0.79")) == ({"speculative_num_steps": 3}, 5)
    monkeypatch.setattr(calibrate, "_time_candidate", time_candidate)
    os.rename(map_path, map_path + ".off")  # no map from here on
    static_strategy[0] = "offload"
    # a calibration has 300 s. A candidate is started only while 1.1 times the longest one so far still fits: at 100 s a
    # candidate that is two, the static choice and the other strategy; what was not started is not timed on a later start
    clock = [0.0]
    monkeypatch.setattr(calibrate, "_now", lambda: clock[0])

    lefts = []

    def taking(seconds, late=(), hung=()):
        def timed(args, argv, s, d, floors=None, ready_s=None, left_s=None):
            lefts.append(left_s)
            if (s, d) in late:  # still loading when its time is up
                started.append((s, d))
                clock[0] += ready_s
                raise RuntimeError("server not ready")
            if (s, d) in hung:  # a request that never ends: the bench ends it at the deadline it was given
                started.append((s, d))
                clock[0] += left_s
                raise RuntimeError("the request ran past its deadline")
            clock[0] += seconds
            return time_candidate(args, argv, s, d)
        return timed

    assert calibrate.BUDGET_S == 300 and calibrate.SLACK == 1.1
    speeds.update({("offload", 3): [100.0, 200.0], ("hybrid", 3): [130.0, 250.0], ("hybrid", 4): [140.0, 260.0], ("offload", 4): [102.0, 195.0]})
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0))
    boxed, boxed_argv = parsed("--memory-ratio", "0.85")
    assert starts(boxed, boxed_argv) == (hybrid_3, 2) and started[-2:] == [("offload", 3), ("hybrid", 3)]
    assert starts(boxed, boxed_argv) == (hybrid_3, 0)
    # --calibrate-again keeps those timings and spends a budget of its own on what was not reached: the other depths of
    # the strategy that was ahead. A start without the flag reads the larger row
    again = parsed("--memory-ratio", "0.85", "--calibrate-again")
    choice, timed = starts(*again)
    assert timed == 2 and started[-2:] == [("hybrid", 4), ("hybrid", 5)] and (choice["moe_strategy"], choice["speculative_num_steps"]) == ("hybrid", 4)
    assert starts(boxed, boxed_argv) == (choice, 0)
    assert [starts(*again)[1] for _ in range(2)] == [2, 2] and started[-4:] == [("hybrid", 0), ("offload", 4), ("offload", 5), ("offload", 0)]
    # nothing is left untimed: the flag times every candidate again, from the static choice
    assert starts(*again)[1] == 2 and started[-2:] == [("offload", 3), ("hybrid", 3)]
    # an expert cache with a slot for every expert is ordered as fused is: 5 drafts, 4 drafts, then the other strategy
    whole = lambda args, argv, s, d, floors=None, ready_s=None, left_s=None: {**time_candidate(args, argv, s, d), "cache_slots": 80}
    monkeypatch.setattr(calibrate, "_time_candidate", whole)
    assert starts(*parsed("--memory-ratio", "0.83"))[1] == 8
    assert started[-8:-4] == [("offload", 3), ("offload", 5), ("offload", 4), ("hybrid", 3)]
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0))
    # at 60 s a candidate that is four: then the other depths of the strategy that is ahead of the two, nearest first
    monkeypatch.setattr(calibrate, "_time_candidate", taking(60.0))
    choice, timed = starts(*parsed("--memory-ratio", "0.86"))
    assert timed == 4 and started[-4:] == [("offload", 3), ("hybrid", 3), ("hybrid", 4), ("hybrid", 5)]
    assert (choice["moe_strategy"], choice["speculative_num_steps"]) == ("hybrid", 4)
    speeds["hybrid", 3] = [90.0, 250.0]  # behind the static choice on one prompt: offload's depths come first
    assert starts(*parsed("--memory-ratio", "0.87"))[1] == 4 and started[-2:] == [("offload", 4), ("offload", 5)]
    # a server still loading when the budget ends was not given the time, which is not a server that cannot start:
    # the next start does not time for it again
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0, late={("hybrid", 3)}))
    late, late_argv = parsed("--memory-ratio", "0.88")
    assert starts(late, late_argv) == ({"speculative_num_steps": 3}, 2) and starts(late, late_argv)[1] == 0
    # every candidate is given the seconds the budget has left, and is ended there: one that hangs in a request is not
    # chosen, and nothing after it is started
    del lefts[:]
    clock[0] = 0.0
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0, hung={("hybrid", 3)}))
    assert starts(*parsed("--memory-ratio", "0.89")) == ({"speculative_num_steps": 3}, 2) and lefts == [300.0, 200.0]
    # the static choice itself hangs: it is ended at the budget, the static preset runs untimed, and nothing is stored
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0, hung={("offload", 3)}))
    stuck, stuck_argv = parsed("--memory-ratio", "0.84")
    before = clock[0]
    assert starts(stuck, stuck_argv) == ({}, 1) and clock[0] - before == calibrate.BUDGET_S
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0))
    assert starts(stuck, stuck_argv)[1] == 2
    # a stopped server's memory is waited for before the next start: here the driver reads 1 GiB short twice, then whole
    short, probes, warned, naps = [0], [0], [], []
    speeds["hybrid", 3] = [130.0, 250.0]

    def probe(args):
        probes[0] += 1
        if probes[0] > 1 and short[0] > 0:  # the first reading is the one before any candidate
            short[0] -= 1
            return (*_ROW_GPU, _ROW_FREE - (1 << 30))
        return (*_ROW_GPU, _ROW_FREE)

    def nap(seconds):
        naps.append(seconds)
        clock[0] += seconds

    def waited(ratio, shorts, *flags):
        short[0], probes[0] = shorts, 0
        del warned[:], naps[:]
        before = clock[0]
        return starts(*parsed("--memory-ratio", ratio, *flags)), clock[0] - before

    monkeypatch.setattr(calibrate, "_probe_gpu", probe)
    monkeypatch.setattr(calibrate, "_sleep", nap)
    monkeypatch.setattr(calibrate.logger, "warning", warned.append)
    assert waited("0.81", 2) == ((hybrid_3, 2), 202.0) and naps == [1.0, 1.0] and not warned
    # it never comes back: 20 s after each stop the calibration goes on and says how much is missing, between two
    # candidates inside its time and before the server that serves outside it
    result, took = waited("0.82", 10**6)
    assert result == (hybrid_3, 2) and took == 240.0 and len(warned) == 2
    assert all(f"{1 << 30} bytes of GPU memory are still not free" in line for line in warned)
    assert "inside the calibration's time" in warned[0] and "outside the calibration's time" in warned[1]
    # a static choice that was cut is waited for too, before the static preset's own server starts
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0, hung={("offload", 3)}))
    result, took = waited("0.801", 3)
    assert result == ({}, 1) and took == calibrate.BUDGET_S + 3.0 and naps == [1.0, 1.0, 1.0]
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0))
    monkeypatch.setattr(calibrate, "_sleep", lambda seconds: None)
    free_now(_ROW_FREE)
    # a static choice that took 280 s left every other candidate untimed. --calibrate-again does not time it again, so
    # its seconds are not this run's: the first candidate not timed always starts, and the next ones are gated by the
    # candidates of this run
    monkeypatch.setattr(calibrate, "_time_candidate", taking(280.0))
    slow, slow_argv = parsed("--memory-ratio", "0.802")
    assert starts(slow, slow_argv) == ({"speculative_num_steps": 3}, 1)
    monkeypatch.setattr(calibrate, "_time_candidate", taking(100.0))
    choice, timed = starts(*parsed("--memory-ratio", "0.802", "--calibrate-again"))
    assert timed == 2 and started[-2:] == [("hybrid", 3), ("hybrid", 4)] and choice["moe_strategy"] == "hybrid"
    monkeypatch.setattr(calibrate, "_now", time.monotonic)
    monkeypatch.setattr(calibrate, "_probe_gpu", lambda args: (_ for _ in ()).throw(RuntimeError("the GPU could not be read")))
    assert starts(third, third_argv) == ({}, 0)


def test_token_map_builder_refuses_benchmark_text_and_records_what_it_counted(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path

    import numpy as np

    from freetoken.moe import calibrate

    spec = importlib.util.spec_from_file_location("make_token_map", Path(__file__).parents[2] / "scripts" / "make_token_map.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    for path, text, marked in (
        ("notes/gsm8k_train.txt", "", "gsm8k"), ("data/test.jsonl", "", "named test"), ("src/math.py", "", "math"),
        ("benchmarks/prompts.txt", "", "benchmarks"), ("a.txt", "From AIME 2025, problem 3", "AIME"), ("a.txt", "the MMLU score", "MMLU"),
        ("a.txt", "so she has 42 left\n#### 42\n", "answer mark"), ("a.txt", "see cais/mmlu", "mmlu"),
        # words that only contain a name, test code, and the built-in prompt's kind are not marks
        ("tests/test_plan.py", "we claimed it and aimed high", None), ("src/mathutils.py", "#### a heading", None),
        ("answers.jsonl", '{"kind": "math", "content": "x"}', None),
    ):
        why = tool.benchmark_mark(path, text)
        assert (why is None) if marked is None else (marked in why), (path, text, why)

    class Tokenizer:  # one token a character, 64 ids, three of them special
        all_special_ids = [0, 1]

        def __len__(self):
            return 64

        def get_added_vocab(self):
            return {"<think>": 2}

        def encode(self, text, add_special_tokens=False):
            return [3 + ord(c) % 61 for c in text]

    monkeypatch.setattr(tool, "load_tokenizer", lambda model: Tokenizer())
    model = tmp_path / "model"
    model.mkdir()
    answers, prose, corpus = tmp_path / "answers.jsonl", tmp_path / "prose.txt", tmp_path / "corpus"
    answers.write_text('{"name": "p1", "content": "zz", "reasoning_content": "z"}\n{"name": "p2", "error": "timeout"}\n')
    prose.write_text("a" * 50)
    (corpus / "pkg").mkdir(parents=True)
    (corpus / "pkg" / "clean.py").write_text("bbb")
    (corpus / "pkg" / "aime.py").write_text("ccccccc")  # a benchmark's by its name: left out and listed
    (corpus / "pkg" / "notes.md").write_text("as in GSM8K")  # and by its text
    out = tmp_path / "map.bin"
    run = [str(model), "--strategy", "topk", "--size", "7", "--answers", str(answers), "--text", str(prose), "--corpus", str(corpus)]
    assert tool.main([*run, "-o", str(out)]) == 0
    ids = np.fromfile(out, dtype="<i4").tolist()
    tok = Tokenizer().encode
    # the special ids, then the tokens of the checkpoint's own replies (z, and the line break between its two strings)
    # before any other text, however much of that there is
    assert ids == [0, 1, 2, tok("z")[0], tok("\n")[0], tok("a")[0], tok("b")[0]]
    manifest = json.loads((tmp_path / "map.bin.json").read_text())
    assert manifest["map_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and manifest["size"] == 7
    assert [(Path(f["path"]).name, f["role"], f["tokens"]) for f in manifest["corpus"]] == [("answers.jsonl", "answers", 4), ("prose.txt", "text", 50), ("clean.py", "corpus", 3)]
    assert all(f["sha256"] == hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() for f in manifest["corpus"])
    assert sorted(Path(r["path"]).name for r in manifest["refused"]) == ["aime.py", "notes.md"] and manifest["checks"] == tool.CHECKS
    before = manifest["corpus_sha256"]
    prose.write_text("a" * 51)
    tool.main([*run, "-o", str(out)])
    assert json.loads((tmp_path / "map.bin.json").read_text())["corpus_sha256"] != before  # another corpus, another hash
    # a file named on the command line that is a benchmark's ends the run, and no map is written
    marked = tmp_path / "gsm8k_sample.txt"
    marked.write_text("Natalia sold clips")
    with pytest.raises(SystemExit):
        tool.main([str(model), "--strategy", "topk", "--size", "6", "--text", str(marked), "-o", str(tmp_path / "bad.bin")])
    assert not (tmp_path / "bad.bin").exists()
    # --store writes where the calibration looks, with the manifest it asks for
    monkeypatch.setenv("FREETOKEN_BENCHDECODE_PATH", str(tmp_path / "store" / "profile.json"))
    args = SimpleNamespace(speculative_token_map=None, model_path=str(model))
    assert calibrate.stored_token_map(args) is None
    assert tool.main([*run, "--store"]) == 0
    stored = calibrate.token_map_path(str(model))
    assert calibrate.stored_token_map(args) == (stored, hashlib.sha256(Path(stored).read_bytes()).hexdigest())
    assert Path(stored).parent == tmp_path / "store" / "token_maps"
    # --install copies a map built elsewhere, whose manifest may record its sha256 as "sha256", and writes the key the store reads
    built = tmp_path / "built.bin"
    built.write_bytes(np.arange(5, dtype="<i4").tobytes())
    sha = hashlib.sha256(built.read_bytes()).hexdigest()
    for fields in ({"map": "built.bin", "sha256": sha, "size": 5}, {"map": "built.bin", "map_sha256": sha}):
        Path(str(built) + ".json").write_text(json.dumps(fields))
        assert tool.main([str(model), "--install", str(built)]) == 0
        assert Path(stored).read_bytes() == built.read_bytes() and calibrate.stored_token_map(args) == (stored, sha)
        assert json.loads(Path(stored + ".json").read_text())["map_sha256"] == sha
    # a file that is not the one its manifest records, or a manifest that records none, is not installed
    for ids, fields in ((6, {"sha256": sha}), (5, {"map_sha256": sha, "sha256": "0" * 64}), (5, {"size": 5})):
        built.write_bytes(np.arange(ids, dtype="<i4").tobytes())
        Path(str(built) + ".json").write_text(json.dumps(fields))
        with pytest.raises(SystemExit):
            tool.main([str(model), "--install", str(built)])
        assert calibrate.stored_token_map(args) == (stored, sha)


def test_a_request_and_a_server_end_at_their_deadline_whatever_the_server_does():
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from freetoken.moe import benchdecode

    done = threading.Event()

    class Server(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                began = time.monotonic()
                while not done.wait(0.05):
                    # tokens keep coming forever, or for a second and then never again, or never at all
                    if self.path.endswith("trickle") or (self.path.endswith("then-silent") and time.monotonic() - began < 1.0):
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except OSError:
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    real = benchdecode.urllib.request.Request
    try:
        # then-silent: the deadline is 0.3 s after the last token. A timeout that counts from each read ends it a full
        # 1.3 s after that token, at 2.3 s
        for kind, after, at_most in (("silent", 0.5, 1.2), ("trickle", 0.5, 1.2), ("then-silent", 1.3, 2.0)):
            with pytest.MonkeyPatch.context() as m:
                # the bench posts to /v1/chat/completions; this server tells its failures apart by the path
                m.setattr(benchdecode.urllib.request, "Request", lambda url, **kw: real(f"http://127.0.0.1:{httpd.server_port}/{kind}", **kw))
                t0 = time.monotonic()
                with pytest.raises((RuntimeError, OSError)):
                    benchdecode.stream_chat("http://unused", {"stream": True}, deadline=t0 + after)
                assert after - 0.1 < time.monotonic() - t0 < at_most, kind
        with pytest.raises(RuntimeError, match="deadline had passed"):
            benchdecode.stream_chat("http://unused", {}, deadline=time.monotonic() - 1.0)
    finally:
        done.set()
        httpd.shutdown()
        httpd.server_close()
    # a server that ignores SIGTERM is killed once its time to stop is up, not after the bench's 90 s
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(600)"],
        stdout=subprocess.PIPE, start_new_session=True,
    )
    try:
        assert proc.stdout.readline().strip() == b"up"
        t0 = time.monotonic()
        benchdecode.stop_server(proc, term_s=0.3)
        assert proc.poll() is not None and time.monotonic() - t0 < 10.0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_signal_during_calibration_stops_the_candidates_server_and_ends_the_launcher():
    import signal
    import subprocess

    from freetoken.moe import benchdecode, calibrate

    servers, outer = [], []

    def time_candidate(args, argv, strategy, depth, floors=None, ready_s=None, left_s=None):
        # as benchdecode.run_arm runs a candidate: a session of its own, which a signal to the launcher does not reach
        proc = subprocess.Popen(["sleep", "600"], start_new_session=True)
        servers.append(proc)
        try:
            signal.raise_signal(sig)
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        finally:
            signal.raise_signal(sig)  # a second one must not cut the stop short
            benchdecode.stop_server(proc)
        return {"tok_s": [[100.0, 100.0], [200.0, 200.0]], "drafts": 300, "early": [100.0, 200.0], "cache_slots": 40, "total_slots": 80,
                "slot_bytes": 1 << 20, "gpu_free_bytes": _ROW_FREE, "ready_s": 1.0}

    def theirs(signum, frame):
        outer.append(signum)

    todo = [("offload@3", "offload", 3, None), ("hybrid@3", "hybrid", 3, None)]
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        del servers[:]
        before = signal.signal(sig, theirs)  # what the launcher had: it is back afterwards, and pytest outlives a launcher without the fix
        try:
            with pytest.MonkeyPatch.context() as m:
                m.setattr(calibrate, "_time_candidate", time_candidate)
                m.setattr(calibrate, "_sleep", lambda seconds: None)
                with pytest.raises(SystemExit) as stopped:
                    calibrate.measure(SimpleNamespace(), [], todo)
            assert stopped.value.code == 128 + sig
            assert len(servers) == 1 and servers[0].poll() is not None  # the candidate is gone, and the next one was not started
            assert signal.getsignal(sig) is theirs and not outer
        finally:
            signal.signal(sig, before)
            for proc in servers:
                if proc.poll() is None:
                    proc.kill()
    # a signal that lands in the bench's argument parser ends the launcher too: it is not taken for a refused command line
    from freetoken.moe import benchdecode as bench

    with pytest.MonkeyPatch.context() as m:
        m.setattr(bench, "parse_args", lambda argv, prog: signal.raise_signal(signal.SIGTERM))
        with pytest.raises(SystemExit) as stopped:
            calibrate.measure(SimpleNamespace(model_path="/models/anon"), ["--preset", "single-measured"], todo)
    assert stopped.value.code == 128 + signal.SIGTERM
    # a signal the launcher ignores (SIGHUP under nohup) stays ignored while the candidates are timed
    row = {"tok_s": [[100.0, 100.0], [200.0, 200.0]], "drafts": 300, "early": [100.0, 200.0], "cache_slots": 40, "total_slots": 80,
           "slot_bytes": 1 << 20, "gpu_free_bytes": _ROW_FREE, "ready_s": 1.0}
    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(calibrate, "_sleep", lambda seconds: None)
            m.setattr(calibrate, "_time_candidate", lambda *a, **k: (signal.raise_signal(signal.SIGHUP), row)[1])
            assert set(calibrate.measure(SimpleNamespace(), [], todo)["tok_s"]) == {"offload@3", "hybrid@3"}
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, before)


def test_auto_does_not_take_a_row_whose_hybrid_cache_is_below_the_models_floor(decode_rows):
    # the fake model has 8 experts a layer: a row edited to 4 slots would cap the cache under the floor and fail the start
    _file_row(decode_rows, _HYBRID_FASTER, slots=4)
    assert _resolved(decode_rows).moe_strategy == "offload"
    _file_row(decode_rows, _HYBRID_FASTER, slots=8)
    assert _resolved(decode_rows).moe_strategy == "hybrid"


def test_auto_sized_cache_takes_the_row_only_with_no_more_memory_than_measured(decode_rows, monkeypatch):
    engine = decode_rows.engine
    _file_row(decode_rows, _HYBRID_FASTER)  # 40 of 80 slots at 1 MiB each, with _ROW_FREE bytes free
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE - (8 << 30))
    assert _resolved(decode_rows).moe_strategy == "hybrid"   # less memory: a smaller cache, where hybrid leads by more
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (1 << 20))
    assert _resolved(decode_rows).moe_strategy == "hybrid"   # one slot more is inside the tolerance
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (8 << 20))
    assert _resolved(decode_rows).moe_strategy == "offload"  # offload would get 7 slots more than hybrid was compared at
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: None)
    assert _resolved(decode_rows).moe_strategy == "offload"  # unknown is not a yes
    # an explicit size is not sized from free memory, so free memory does not matter
    _file_row(decode_rows, _HYBRID_FASTER, moe_cache_size=40)
    monkeypatch.setattr(engine, "_gpu_free_bytes", lambda: _ROW_FREE + (8 << 30))
    assert _resolved(decode_rows, moe_cache_size=40).moe_strategy == "hybrid"


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override, scaling=None):
    from types import SimpleNamespace

    from freetoken.models.config import RotaryConfig

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=RotaryConfig(
            head_dim=128, rotary_dim=128, max_position=max_position, base=1e4, scaling=scaling
        ),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


YARN_4X = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024}


def test_adjust_config_allows_override_to_the_yarn_extended_table():
    # A YaRN override that leaves max_position_embeddings at the trained length (Qwen3.8's 1M
    # recipe) serves original * factor positions; the gate must follow the table, not the config.
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=4096, scaling=YARN_4X))
    with pytest.raises(ValueError, match=r"rope table \(4096 positions\)"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=4097, scaling=YARN_4X))


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None


def test_moe_slot_pool_split(tmp_path):
    import json

    from freetoken.moe.pool_policy import group_layers, load_profile, make_pools, split_slots

    pools = make_pools(group_layers("per-layer", [0] * 4, [1] * 4), num_experts=16, min_num_slots=8, slot_limit=None)
    assert split_slots(41, pools) == [11, 10, 10, 10]
    with pytest.raises(ValueError, match="need"):
        split_slots(31, pools)
    with pytest.raises(ValueError, match="hold"):
        split_slots(65, pools)
    # one pool takes the whole cache, as before pools existed
    assert split_slots(100, make_pools(group_layers("unified", [0] * 4, [1] * 4), 16, 8, None)) == [100]
    # unified pools one bank layout each, widest slots first
    assert group_layers("unified", ["a", "b", "a", "b"], [1, 3, 1, 3]) == [[1, 3], [0, 2]]

    # layer 0 uses two experts, layer 1 has a hot half and a cold half, layers 2 and 3 use all 16 evenly
    counts = [[5] * 2 + [0] * 14, [3] * 8 + [1] * 8, [1] * 16, [1] * 16]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"model": "toy", "expert_counts": counts}))
    assert split_slots(48, pools, load_profile(path, 4, 16)) == [8, 8, 16, 16]
    with pytest.raises(ValueError, match="needs"):
        load_profile(path, 4, 32)



def test_moe_pool_flags_and_decode_batch_clamp():
    from unittest.mock import patch

    from freetoken.engine.engine import _clamp_decode_batch_to_pools
    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--moe-cache-mem", "1.5GiB",
            "--moe-cache-scope", "per-layer", "--moe-cache-split", "profile:/p.json",
        ])
        assert (args.moe_cache_mem, args.moe_cache_scope, args.moe_cache_split) == (3 << 29, "per-layer", "profile:/p.json")
        for bad in (["--moe-cache-mem", "12"], ["--moe-cache-mem", "1GiB", "--moe-cache-size", "512"]):
            with pytest.raises(SystemExit):
                parse_args(["--model", "/models/anon", *bad])

    # GPU decode needs batch * top_k distinct experts to fit the smallest pool below num_experts
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=16, num_experts_per_tok=4),
        max_running_req=4, cuda_graph_max_bs=None, cuda_graph_bs=[1, 2, 4], decode_rows_per_req=1,
    )
    _clamp_decode_batch_to_pools(config, [12, 16, 9])
    assert (config.max_running_req, config.cuda_graph_max_bs, config.cuda_graph_bs) == (2, 2, [1, 2])
    # two rows per request route twice the experts: 9 // (4 * 2) leaves one request
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=16, num_experts_per_tok=4),
        max_running_req=4, cuda_graph_max_bs=None, cuda_graph_bs=[1, 2, 4], decode_rows_per_req=2,
    )
    _clamp_decode_batch_to_pools(config, [12, 16, 9])
    assert (config.max_running_req, config.cuda_graph_max_bs, config.cuda_graph_bs) == (1, 1, [1])
