from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

# The split patterns of the Qwen2 and Qwen3.6 byte-level BPE tokenizers. Under either, a match never runs from a
# newline into the character after it, nor from a letter into a space that a letter follows, so the text can be cut
# there and its pieces encoded on their own (``piece_cuts``).
_PIECE_PATTERNS = frozenset({
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
})
# after a newline, before printable ASCII; or at a space between two ASCII letters
_PIECE_CUT = re.compile(r"(?<=\n)(?=[!-~])|(?<=[A-Za-z])(?= [A-Za-z])")
# a text is encoded in pieces of about this many characters, at most _MAX_PIECES of them, in parallel
_PIECE_CHARS = 2048
_MAX_PIECES = 32


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


def piece_encoder(tokenizer: PreTrainedTokenizerBase):
    """The tokenizer's Rust backend when a text cut at ``_PIECE_CUT`` encodes to the ids of the whole, else None.

    Pieces are encoded alone, so no rule may look across a cut: a split pattern of ``_PIECE_PATTERNS`` then byte-level
    without its own regex, a normalizer that is NFC or none, a BPE model, and added tokens that hold no whitespace and
    strip none (none can then span a cut).
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or getattr(tokenizer, "split_special_tokens", False):
        return None
    try:
        spec = json.loads(backend.to_str())
    except Exception:  # noqa: BLE001 -- a backend that does not serialize keeps the whole-text encode
        return None
    pre = spec.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not steps or len(steps) != 2:
        return None
    split, byte_level = steps
    if not (
        split.get("type") == "Split"
        and (split.get("pattern") or {}).get("Regex") in _PIECE_PATTERNS
        and split.get("behavior") == "Isolated"
        and not split.get("invert")
        and byte_level.get("type") == "ByteLevel"
        and not byte_level.get("use_regex")
        and not byte_level.get("add_prefix_space")
        and spec.get("normalizer") in (None, {"type": "NFC"})
        and (spec.get("model") or {}).get("type") == "BPE"
    ):
        return None
    for t in spec.get("added_tokens", []):
        if t.get("lstrip") or t.get("rstrip") or t.get("single_word") or any(c.isspace() for c in t["content"]):
            return None
    return backend


def piece_cuts(text: str, pieces: int) -> list[int]:
    """Up to ``pieces - 1`` cut positions, each the first ``_PIECE_CUT`` from an even share of the text on."""
    cuts: list[int] = []
    for i in range(1, pieces):
        at = max(len(text) * i // pieces, cuts[-1] + 1 if cuts else 1)
        m = _PIECE_CUT.search(text, at)
        if m is None:
            break
        cuts.append(m.start())
    return cuts


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
        self._cut_ids_np = self.cut_ids.numpy()
        self.pieces = piece_encoder(tokenizer) if self.cut_text else None
        # a prompt is never changed and the list is replaced whole, so encodes on several threads need no lock
        self.prompts: list[_Prompt] = []
        self.reused_tokens = 0  # ids the last encode took from a cached prompt

    def warm(self) -> None:
        """Start the backend's thread pool, so that the first long prompt does not."""
        if self.pieces is not None:
            self.pieces.encode_batch(["warm", "up"], add_special_tokens=False)

    def _encode(self, text: str, text_base: int = 0, ids_base: int = 0) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        if self.pieces is not None:
            return self._encode_pieces(text, text_base, ids_base)
        out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor(out["input_ids"], dtype=torch.int32)
        cuts = []
        for k in torch.isin(ids, self.cut_ids).nonzero().flatten().tolist():
            start, end = out["offset_mapping"][k]
            # only where the reported span is the token's own text
            if text[start:end] == self.cut_text[int(ids[k])]:
                cuts.append((text_base + end, text_base + start, ids_base + k))
        return ids, cuts

    def _encode_pieces(self, text: str, text_base: int, ids_base: int) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        """``_encode`` from the backend: the text cut at ``piece_cuts`` and the pieces encoded in parallel, with only
        the cut tokens' spans read back."""
        bounds = [0, *piece_cuts(text, min(_MAX_PIECES, len(text) // _PIECE_CHARS)), len(text)]
        if len(bounds) == 2:
            encodings = [self.pieces.encode(text, add_special_tokens=False)]
        else:
            encodings = self.pieces.encode_batch([text[a:b] for a, b in zip(bounds, bounds[1:])], add_special_tokens=False)
        parts, cuts, n = [], [], 0
        for start, enc in zip(bounds, encodings):
            ids = np.asarray(enc.ids, dtype=np.int32)
            for k in np.flatnonzero(np.isin(ids, self._cut_ids_np)).tolist():
                a, b = enc.token_to_chars(k)
                a, b = a + start, b + start
                if text[a:b] == self.cut_text[int(ids[k])]:
                    cuts.append((text_base + b, text_base + a, ids_base + n + k))
            parts.append(ids)
            n += len(ids)
        return torch.from_numpy(np.concatenate(parts)), cuts

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
