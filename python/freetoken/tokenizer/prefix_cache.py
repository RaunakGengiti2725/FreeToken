from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, comparing each character once."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a.startswith(b[lo:mid], lo):
            lo = mid
        else:
            hi = mid - 1
    return lo


def cut_tokens(tokenizer: PreTrainedTokenizerBase) -> dict[int, str]:
    """The added tokens a text can be cut in front of without changing any token id, as ``{id: text}``.

    A fast tokenizer splits the raw text at its added tokens (leftmost, then longest) and encodes each piece between them on its own.
    A token qualifies when only its own text decides its match: not normalized, no lstrip, rstrip or single_word, and inside no other added token.
    """
    if not getattr(tokenizer, "is_fast", False):
        return {}
    added = tokenizer.added_tokens_decoder
    # one search per token over every added token's text: its own is the one match allowed
    texts = "\0".join(t.content for t in added.values())
    return {
        i: t.content
        for i, t in added.items()
        if not (t.normalized or t.lstrip or t.rstrip or t.single_word) and texts.count(t.content) == 1
    }


@dataclass
class _Prompt:
    text: str
    ids: torch.Tensor  # int32 [n]
    # per cut token, in order: where its text ends, where it starts, its index in ids
    cuts: list[tuple[int, int, int]]


class TokenPrefixCache:
    """Token ids of the last ``entries`` prompts: a prompt that repeats the start of one is encoded from a cut token on.

    The reused ids end in front of the last cut token that lies wholly in the shared text; the rest, starting with that token, is encoded alone.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, entries: int) -> None:
        self.tokenizer = tokenizer
        self.entries = entries
        self.cut_text = cut_tokens(tokenizer)
        self.cut_ids = torch.tensor(sorted(self.cut_text), dtype=torch.int32)
        # a prompt is never changed and the list is replaced whole, so encodes on several threads need no lock
        self.prompts: list[_Prompt] = []
        self.reused_tokens = 0  # ids the last encode took from a cached prompt

    def _encode(self, text: str, text_base: int = 0, ids_base: int = 0) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor(out["input_ids"], dtype=torch.int32)
        cuts = []
        for k in torch.isin(ids, self.cut_ids).nonzero().flatten().tolist():
            start, end = out["offset_mapping"][k]
            # only where the reported span is the token's own text
            if text[start:end] == self.cut_text[int(ids[k])]:
                cuts.append((text_base + end, text_base + start, ids_base + k))
        return ids, cuts

    def encode(self, text: str) -> torch.Tensor:
        """``tokenizer.encode(text, add_special_tokens=False)`` as an int32 tensor."""
        if not self.cut_text:
            return self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).view(-1).to(torch.int32)
        best, kept = None, []
        for prompt in self.prompts:
            common = common_prefix_len(prompt.text, text)
            if common < len(prompt.text):
                # a cached prompt the new one starts with has nothing left to offer
                kept.append(prompt)
            n = bisect_right(prompt.cuts, common, key=lambda cut: cut[0]) - 1
            if n >= 0 and prompt.cuts[n][2] > (best[2] if best else 0):
                best = (prompt, n, prompt.cuts[n][2])
        ids = None
        if best is not None:
            prompt, n, k = best
            start = prompt.cuts[n][1]
            tail, cuts = self._encode(text[start:], start, k)
            # the text was cut in front of this token, so the rest must begin with it
            if tail.numel() and tail[0] == prompt.ids[k]:
                ids, cuts = torch.cat([prompt.ids[:k], tail]), prompt.cuts[:n] + cuts
        self.reused_tokens = 0 if ids is None else k
        if ids is None:
            ids, cuts = self._encode(text)
        self.prompts = ([_Prompt(text, ids, cuts)] + kept)[: self.entries]
        # the caller owns what it gets; the cached ids must not change under it
        return ids.clone()
