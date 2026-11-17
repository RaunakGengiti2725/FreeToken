"""Write a --speculative-token-map file: the token ids the MTP draft head keeps.

    python scripts/make_token_map.py MODEL [--strategy low-id|topk] [--text FILE ...] [--answers FILE.jsonl ...]
        [--corpus DIR ...] [--size 106496] [-o token_map.bin | --store]
    python scripts/make_token_map.py MODEL --install MAP

MODEL is the target checkpoint, an HF directory or a GGUF file. Only its tokenizer is read, so every model that shares the tokenizer can use the same file.
The file holds all special tokens plus the strategy's first ids, --size ids in all, as little-endian int32.

Strategies:
- low-id: lowest ids first. BPE tokenizers mostly number tokens in merge order, so earlier tokens are usually the more common ones. Some tokenizers do not follow this (Gemma 4 puts single characters last); use topk for them.
- topk: the tokens that occur most often in the --answers files, then those that occur most often in the --text and --corpus files, tokenized with MODEL's tokenizer. Tokens neither uses follow in id order.

Text to count tokens in:
- --text: UTF-8 text files.
- --answers: JSON lines of the checkpoint's own replies, one object a line; its "content" and "reasoning_content" (or "reasoning") strings are counted. A line with an "error" key is skipped. These rank first: they are what the draft will be asked to propose.
- --corpus: every .py .md .txt .rst .c .h .cpp .toml .sh file under a directory.

With any of them, both strategies also print how much of that text the map covers.

No benchmark's text may shape the map: a map built from a test set's own tokens raises the draft's acceptance on that test set and nowhere else. A file is refused when its path or its content marks it as one:
- path: one of its parts, split at every character that is not a letter or a digit, is a name in BENCHMARK_NAMES (gsm8k, aime, mmlu, math, humaneval, ...), or the file is a data file (.jsonl .json .parquet .csv .tsv .arrow) named "test".
- content: it names a test set (BENCHMARK_TEXT: GSM8K, AIME, MMLU, Hendrycks, competition_math, MATH-500, HumanEval, MBPP, SWE-bench, GPQA, HellaSwag, TruthfulQA), or has a line that is GSM8K's answer mark ("#### 42").
A --text or --answers file that is refused ends the run with an error. Under --corpus a refused file is left out and listed. The check reads names and text only: it cannot tell a copy of a test set that carries no such mark.

<out>.json is written beside the map: its sha256, size and strategy, every file that was counted (path, role, bytes, sha256, tokens), one sha256 over those, the files refused and why, and the checks above. --store writes the map where `ft serve --preset single-measured` looks for this checkpoint's map and times it as a candidate, and where `--preset single` and `single-fast` serve it when every expert is resident. --install MAP copies a map built before and its MAP.json there, if the sha256 its manifest records is the file's, and prints where it went.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

from freetoken.utils.hf import load_tokenizer

# each strategy ranks every token id from its counts in the secondary text and in the answers, most wanted first
STRATEGIES = {
    "low-id": lambda counts, first: np.arange(counts.size),
    "topk": lambda counts, first: np.lexsort((np.arange(counts.size), -counts, -first)),
}

CORPUS_SUFFIXES = (".py", ".md", ".txt", ".rst", ".c", ".h", ".cpp", ".toml", ".sh")
DATA_SUFFIXES = (".jsonl", ".json", ".parquet", ".csv", ".tsv", ".arrow")
BENCHMARK_NAMES = frozenset((
    "gsm8k", "gsm", "aime", "aime24", "aime25", "aime2024", "aime2025", "mmlu", "math", "math500", "hendrycks", "humaneval",
    "mbpp", "swebench", "gpqa", "hellaswag", "truthfulqa", "livebench", "benchmark", "benchmarks", "eval", "evals", "testset",
))
BENCHMARK_TEXT = re.compile(
    r"(?<![a-z])(gsm-?8k|aime|mmlu|hendrycks|competition_math|math-?500|humaneval|mbpp|swe-?bench|gpqa|hellaswag|truthfulqa)(?![a-z])",
    re.IGNORECASE,
)
GSM8K_ANSWER = re.compile(r"^#### -?[\d,]+(\.\d+)?\s*$", re.MULTILINE)
CHECKS = [
    "path: a part (split at non-alphanumerics, lower case) in " + " ".join(sorted(BENCHMARK_NAMES)),
    "path: a data file (" + " ".join(DATA_SUFFIXES) + ") named test",
    "content: " + BENCHMARK_TEXT.pattern + " (any case)",
    "content: a line matching " + GSM8K_ANSWER.pattern,
]


def benchmark_mark(path: str, text: str) -> str | None:
    """Why ``path`` with content ``text`` is taken for a benchmark's file, or None."""
    parts = [p for p in re.split(r"[^a-z0-9]+", path.lower()) if p]
    if named := sorted(BENCHMARK_NAMES.intersection(parts)):
        return f"its path names {named[0]}"
    stem, suffix = os.path.splitext(os.path.basename(path).lower())
    if suffix in DATA_SUFFIXES and "test" in re.split(r"[^a-z0-9]+", stem):
        return "it is a data file named test"
    if found := BENCHMARK_TEXT.search(text):
        return f"its text names {found.group(0)} (line {text.count(chr(10), 0, found.start()) + 1})"
    if found := GSM8K_ANSWER.search(text):
        return f"line {text.count(chr(10), 0, found.start()) + 1} is GSM8K's answer mark"
    return None


def answers_text(raw: str, path: str) -> str:
    """The replies in a JSON-lines file: content and reasoning of every line that is not an error."""
    out = []
    for n, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{n}: not a JSON line ({exc})") from None
        if not isinstance(row, dict) or "error" in row:
            continue
        out += [row[k] for k in ("reasoning_content", "reasoning", "content") if isinstance(row.get(k), str)]
    return "\n".join(out)


def install(p: argparse.ArgumentParser, model: str, src: str) -> int:
    """Copy the map ``src`` and its manifest into the store for ``model``, where the sha256 the manifest records
    ("map_sha256", or "sha256" in manifests written by other tools) is the file's. The manifest is written with
    "map_sha256", the key the store is read by."""
    from freetoken.moe.calibrate import token_map_path

    try:
        with open(src, "rb") as f:
            data = f.read()
        with open(src + ".json") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        p.error(f"--install {src}: {exc}")
    if not isinstance(manifest, dict):
        p.error(f"--install {src}: {src}.json is not a JSON object")
    sha = hashlib.sha256(data).hexdigest()
    recorded = {manifest[k] for k in ("map_sha256", "sha256") if k in manifest}
    if recorded != {sha}:
        p.error(f"--install {src}: the file's sha256 is {sha}, its manifest records {', '.join(map(str, sorted(recorded, key=str))) or 'none'}; nothing installed")
    out = token_map_path(model)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    with open(out + ".json", "w") as f:
        json.dump({**manifest, "map_sha256": sha, "installed_from": os.path.abspath(src)}, f, indent=1)
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="target checkpoint: an HF directory or a GGUF file")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="low-id", help="how to rank the token ids (default: low-id)")
    p.add_argument("--text", nargs="+", default=[], metavar="FILE", help="UTF-8 text files to count tokens in")
    p.add_argument("--answers", nargs="+", default=[], metavar="FILE", help="JSON-lines files of the checkpoint's own replies; they rank before --text and --corpus")
    p.add_argument("--corpus", nargs="+", default=[], metavar="DIR", help="directories whose code and prose files are counted like --text")
    p.add_argument("--size", type=int, default=106496, help="ids in the file, special tokens included (default: 106496, a multiple of 128 so the draft head needs no padding rows)")
    p.add_argument("-o", "--out", default="token_map.bin", help="output file (default: token_map.bin)")
    p.add_argument("--store", action="store_true", help="write the map where `ft serve --preset single-measured` looks for this checkpoint's map, instead of --out")
    p.add_argument("--install", metavar="MAP", help="copy a map built before, and its MAP.json manifest, where --store writes them; nothing is counted")
    args = p.parse_args(argv)
    if args.install:
        if args.text or args.answers or args.corpus:
            p.error("--install copies a map built before: --text, --answers and --corpus count nothing there")
        return install(p, args.model, args.install)
    if args.strategy == "topk" and not (args.text or args.answers or args.corpus):
        p.error("--strategy topk needs --text, --answers or --corpus")

    tokenizer = load_tokenizer(args.model)
    vocab = len(tokenizer)
    special = sorted(set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids))
    if not len(special) <= args.size <= vocab:
        p.error(f"--size must be between the {len(special)} special tokens and the vocabulary size {vocab}")

    counts = {"answers": np.zeros(vocab, dtype=np.int64), "text": np.zeros(vocab, dtype=np.int64)}
    files, refused = [], []

    def count(path: str, role: str, shown: str) -> None:
        with open(path, "rb") as f:
            data = f.read()
        raw = data.decode("utf-8", errors="strict" if role != "corpus" else "ignore")
        if why := benchmark_mark(shown, raw):
            if role != "corpus":
                p.error(f"{path} is refused: {why}. No benchmark's text may shape a draft token map")
            refused.append({"path": path, "why": why})
            return
        text = io.StringIO(answers_text(raw, path) if role == "answers" else raw, newline=None)
        into = counts["answers" if role == "answers" else "text"]
        before = int(into.sum())
        # small chunks stay under the tokenizer's max length, which it warns about
        while lines := text.readlines(1 << 16):
            into += np.bincount(tokenizer.encode("".join(lines), add_special_tokens=False), minlength=vocab)
        files.append({"path": path, "role": role, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "tokens": int(into.sum()) - before})

    for path in args.answers:
        count(path, "answers", path)
    for path in args.text:
        count(path, "text", path)
    for root in args.corpus:
        for folder, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in sorted(names):
                if name.endswith(CORPUS_SUFFIXES):
                    full = os.path.join(folder, name)
                    # the path inside the corpus: where the directory itself lies says nothing of its files
                    count(full, "corpus", os.path.relpath(full, root))

    first, second = counts["answers"], counts["text"]
    taken = set(special)
    ranked = (int(i) for i in STRATEGIES[args.strategy](second, first) if i not in taken)
    ids = np.array(special + [next(ranked) for _ in range(args.size - len(special))], dtype="<i4")
    out = args.out
    if args.store:
        from freetoken.moe.calibrate import token_map_path

        out = token_map_path(args.model)
        os.makedirs(os.path.dirname(out), exist_ok=True)
    ids.tofile(out)
    print(f"{out}: {ids.size} of {vocab} token ids ({100 * ids.size / vocab:.1f}%), strategy {args.strategy}")
    coverage = {}
    for role, flag in (("answers", "--answers"), ("text", "--text and --corpus" if args.corpus else "--text")):
        if counts[role].any():
            coverage[role] = float(counts[role][ids].sum() / counts[role].sum())
            print(f"covers {100 * coverage[role]:.2f}% of the {counts[role].sum()} tokens in {flag}")
    if refused:
        print(f"left out of --corpus as a benchmark's: {len(refused)} files (listed in {out}.json)")
    corpus_sha = hashlib.sha256("".join(sorted(f"{f['role']} {f['sha256']} {f['bytes']}\n" for f in files)).encode()).hexdigest()
    manifest = {
        "map": os.path.basename(out), "map_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "size": int(ids.size), "vocab": vocab,
        "special": len(special), "strategy": args.strategy, "model": os.path.basename(os.path.normpath(args.model)),
        "corpus_sha256": corpus_sha, "corpus": files, "refused": refused, "checks": CHECKS, "coverage": coverage,
    }
    with open(out + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{out}.json: {len(files)} files counted, corpus sha256 {corpus_sha[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
