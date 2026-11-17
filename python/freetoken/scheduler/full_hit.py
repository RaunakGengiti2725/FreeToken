"""FREETOKEN_FULL_HIT=1: an identical resend samples the first send's stored last-row logits, with no forward.

At the drain of a prompt's final prefill chunk the scheduler keeps what the first send held there: its live GDN state
(copied to a slot of its own), the logits its first token was sampled from, the draft's relay rows and the first token.
A later request whose prompt has the same ids and whose first-token sampling is the same and deterministic (greedy, or
seeded with the same seed) then takes the tree's pages for the whole prompt, restores that state, and samples those
logits with its own sampling arguments: it gets the first send's first token and state bit for bit.

The tree's pages are checked against the ones the first send held, and any eviction drops every entry: a page the tree
gave back may be filled again by another computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import torch

from freetoken.core import SamplingParams

# entries kept, most recent first; each holds one GDN state slot
CAPACITY = 2


def first_token_key(params: SamplingParams) -> tuple | None:
    """What decides a request's first token besides the logits, or None where that token is not deterministic (an
    unseeded sampled request draws from the shared stream)."""
    if not params.is_greedy and params.seed is None:
        return None
    sampled = () if params.is_greedy else (params.temperature, params.top_k, params.top_p, params.min_p, params.seed)
    return (params.is_greedy, *sampled, params.repetition_penalty, params.presence_penalty, params.frequency_penalty)


@dataclass
class FullHitEntry:
    input_ids: torch.Tensor  # [P] int32, CPU: the prompt
    kv_indices: torch.Tensor  # [P]: the pages the first send held for it
    state_slot: int  # a GDN state slot holding the first send's live state after the prompt
    logits: torch.Tensor  # [1, vocab]: what its first token was sampled from
    draft_rows: Any  # the draft's per-row state after the prompt (SpecRuntime.full_hit_rows), or None
    first_token: int
    key: tuple


class FullHitStore:
    def __init__(self, free_slot: Callable[[int], None], capacity: int = CAPACITY) -> None:
        self._free_slot = free_slot
        self.capacity = capacity
        self.entries: List[FullHitEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: FullHitEntry) -> None:
        """Keep ``entry``; one for the same prompt and key is replaced, and the oldest past the capacity is dropped."""
        for old in [e for e in self.entries if self._same(e, entry.input_ids, entry.key)]:
            self._drop(old)
        self.entries.insert(0, entry)
        while len(self.entries) > self.capacity:
            self._drop(self.entries[-1])

    def find(self, input_ids: torch.Tensor, params: SamplingParams) -> FullHitEntry | None:
        key = first_token_key(params)
        if key is None:
            return None
        return next((e for e in self.entries if self._same(e, input_ids, key)), None)

    def drop(self, entry: FullHitEntry) -> None:
        if entry in self.entries:
            self._drop(entry)

    def clear(self) -> bool:
        """Drop every entry; returns whether there was one (its state slot went back to the pool)."""
        had = bool(self.entries)
        for entry in list(self.entries):
            self._drop(entry)
        return had

    @staticmethod
    def _same(entry: FullHitEntry, input_ids: torch.Tensor, key: tuple) -> bool:
        return entry.key == key and entry.input_ids.numel() == input_ids.numel() and torch.equal(entry.input_ids, input_ids)

    def _drop(self, entry: FullHitEntry) -> None:
        self.entries.remove(entry)
        self._free_slot(entry.state_slot)


def tree_pages(cache_manager, entry: FullHitEntry):
    """(handle, pages) of the tree's node that ends at the entry's whole prompt, when its pages are the first send's;
    None while the tree does not hold the whole prompt (the first send has not finished), and False where it holds
    other pages for it (they were filled by another computation): the entry is of no use then. The keys carry the
    first token where the radix looks one token ahead."""
    from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

    tree = cache_manager.prefix_cache
    n = entry.input_ids.numel()
    ids = torch.cat([entry.input_ids, torch.tensor([entry.first_token], dtype=entry.input_ids.dtype)])
    keys = cache_manager._keys(ids, n)
    if keys.numel() != n:
        return None
    node, matched = tree._walk(keys)
    if matched != n:
        return None
    pages = tree._collect_kv(node)
    if pages.numel() != n or not torch.equal(pages.to(entry.kv_indices.device), entry.kv_indices):
        return False
    return HybridCacheHandle(n, node, pages), pages
