"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil, mem_GB

if TYPE_CHECKING:
    import torch


class CacheBudgetTooSmall(ValueError):
    """The minimum --moe-cache-auto plan (slot floor + KV reserve) is ``short`` bytes over the budget."""

    def __init__(self, message: str, short: int):
        super().__init__(message)
        self.short = short


def layer_slot_bytes(sources: dict[str, "list[torch.Tensor]"]) -> list[int]:
    """Bytes one expert slot of each layer occupies on GPU: its row bytes summed over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4/v4_pool.py.
    num_layers = len(next(iter(sources.values())))
    return [sum(t[l][0].numel() * t[l].element_size() for t in sources.values()) for l in range(num_layers)]


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU, rounded up to the mean over layers when their bank layouts differ."""
    per_layer = layer_slot_bytes(sources)
    return -(-sum(per_layer) // len(per_layer))


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def prefill_working_bytes(
    tokens: int, *, hidden: int, key_dim: int, value_dim: int, value_heads: int, head_dim: int, chunk: int, itemsize: int
) -> tuple[int, int]:
    """GPU bytes the temporaries of a prefill of ``tokens`` tokens hold at their peak, and the largest one of them.

    The peak of a model with GatedDeltaNet layers is in one of those, where the chunk kernel allocates its output
    (layers/linear_attn/gdn.py, kernel/fla/chunk.py): every term below is a tensor alive there, in elements per token.
    """
    conv = 2 * key_dim + value_dim
    per_token = (
        4 * hidden  # the layer's input and residual, and the normed input and summed residual made from them
        + conv + value_dim  # in_proj: q, k, v before the conv, and the output gate z
        + conv  # the conv's output
        + conv  # q, k, v copied out of it head by head
        + 2 * key_dim  # q and k l2-normalised
        + value_heads * chunk  # A, the solved chunk-local matrix
        + value_heads * head_dim + value_dim  # w and u
        + 2 * value_dim  # v_new and the output o
    )
    states = div_ceil(tokens, chunk) * value_heads * head_dim * head_dim  # h: the recurrent state entering each kernel chunk
    return (tokens * per_token + states) * itemsize, tokens * (conv + value_dim) * itemsize


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    if total > budget_bytes:
        raise CacheBudgetTooSmall(
            f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
            f"kv={num_pages} pages) needs {total} B ({mem_GB(total)}) > budget {budget_bytes} B "
            f"({mem_GB(budget_bytes)}), {total - budget_bytes} B ({mem_GB(total - budget_bytes)}) short",
            total - budget_bytes,
        )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )


def _plan_or_short(terms: dict) -> tuple[tuple[int, int, bool] | None, int]:
    """resolve_moe_cache_auto's plan for ``terms``, or None and the bytes its minimum plan is over the budget."""
    try:
        return resolve_moe_cache_auto(**terms), 0
    except CacheBudgetTooSmall as exc:
        return None, exc.short


def plan_kv_target(terms: dict, target_tokens: int) -> tuple[tuple[int, int, bool] | None, tuple[int, int, bool] | None, tuple[int, int]]:
    """resolve_moe_cache_auto's ``terms`` with the KV reserve raised to ``target_tokens``: (that plan, the plan of ``terms``,
    (the largest reserve that fits, its expert slots)).

    A plan is None where its minimum (the slot floor and the reserve) is over the budget. The largest reserve is the
    target where that fits, and (0, 0) where not even the reserve of ``terms`` does.
    """
    kept, _ = _plan_or_short(terms)
    target_tokens = max(target_tokens, terms["kv_reserve_tokens"])
    plan, short = _plan_or_short({**terms, "kv_reserve_tokens": target_tokens})
    if plan is not None:
        return plan, kept, (target_tokens, plan[0])
    if kept is None:
        return None, None, (0, 0)
    page = terms["page_size"]
    # ``short`` bytes over at the target: the largest reserve that fits holds that many bytes of pages less
    most = (div_ceil(target_tokens, page) - div_ceil(short, terms["cache_per_page"])) * page
    at_most, _ = _plan_or_short({**terms, "kv_reserve_tokens": most})
    return None, kept, (most, at_most[0])


def plan_kv_reserve(terms: dict, target_tokens: int, min_slots: int) -> tuple[int, tuple[int, int, bool] | None]:
    """The largest KV reserve, from that of ``terms`` up to ``target_tokens``, whose plan has ``min_slots`` expert slots or more:
    (the reserve, its plan).

    Where no reserve in that range has, the reserve of ``terms`` comes back with its own plan (None where its minimum is over
    the budget). A larger reserve never plans more slots, so the search is a bisection over whole pages.
    """
    page, low = terms["page_size"], terms["kv_reserve_tokens"]

    def plan(tokens: int) -> tuple[int, int, bool] | None:
        return _plan_or_short({**terms, "kv_reserve_tokens": tokens})[0]

    def enough(tokens: int) -> bool:
        at = plan(tokens)
        return at is not None and at[0] >= min_slots

    high = max(target_tokens, low)
    if enough(high) or not enough(low):
        tokens = high if enough(high) else low
        return tokens, plan(tokens)
    fits, over = div_ceil(low, page), div_ceil(high, page)
    while over - fits > 1:
        mid = (fits + over) // 2
        if enough(mid * page):
            fits = mid
        else:
            over = mid
    tokens = max(fits * page, low)
    return tokens, plan(tokens)


def plan_changes(stages: list[tuple[str, dict]]) -> list[str]:
    """What each stage after the first did to the --moe-cache-auto plan, e.g. ``--memory-ratio 0.95: +140 expert slots``.

    ``stages`` are (label, resolve_moe_cache_auto arguments) pairs, each adding to the one before, as in budget_advice.
    """
    out = []
    before, _ = _plan_or_short(stages[0][1])
    for label, terms in stages[1:]:
        after, short = _plan_or_short(terms)
        if after is None:
            text = f"the minimum plan is {mem_GB(short)} short"
        elif before is None:
            text = f"the plan fits: {after[0]} expert slots, {after[1] * terms['page_size']} KV tokens"
        else:
            slots, tokens = after[0] - before[0], (after[1] - before[1]) * terms["page_size"]
            text = ", ".join(([f"{slots:+d} expert slots"] if slots else []) + ([f"{tokens:+d} KV tokens"] if tokens else [])) or "no change"
        out.append(f"{label}: {text}")
        before = after
    return out


def budget_advice(stages: list[tuple[str, dict]], min_reserve_tokens: int) -> str:
    """The command-line flags that make a refused --moe-cache-auto plan fit, or how far off the last try is.

    ``stages`` are (flags, resolve_moe_cache_auto arguments) pairs, each adding to the one before; the first holds the current settings.
    A stage that does not fit is tried again at the largest KV reserve that does, and every plan named is the resolver's own.
    """
    for flags, terms in stages:
        page, reserve = terms["page_size"], terms["kv_reserve_tokens"]
        plan, short = _plan_or_short(terms)
        if plan is None:
            # ``short`` bytes over at this reserve: the largest reserve that fits holds that many bytes of pages less.
            # The planner refuses a KV of one page, so the reserve stays at two or more.
            fewer = div_ceil(short, terms["cache_per_page"])
            tokens = max(min_reserve_tokens, 2 * page, (div_ceil(reserve, page) - fewer) * page)
            if tokens < reserve:
                flags, terms = f"--kv-reserve-tokens {tokens} {flags}".rstrip(), {**terms, "kv_reserve_tokens": tokens}
                plan, short = _plan_or_short(terms)
        if plan is not None:
            return f"fits with {flags}: {plan[0]} expert slots, {plan[1] * page} KV tokens (the longest prompt + output)"
    return f"no fit{' even with ' + flags if flags else ''}: the minimum plan is still {short} B ({mem_GB(short)}) short"
