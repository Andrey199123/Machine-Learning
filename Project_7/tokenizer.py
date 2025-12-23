#!/usr/bin/env python3
"""
Tokenize WikiExtractor output with your custom Byte-Level BPE tokenizer
and save token IDs, using as much CPU parallelism as the tokenizers
library will give us.

- Reads from:  extracted/
- Skips:       <doc ...> and </doc> lines
- Keeps:       title and article text between <doc> tags
- Writes:      BPE_tokens.txt  (one line per article, space-separated token IDs)

Requires:
    pip install tokenizers
"""

import os
from pathlib import Path
from typing import Iterable, List

# ---------------------------------------------------------
# 0. Enable maximum parallelism for HF tokenizers (Rust/Rayon)
# ---------------------------------------------------------

# Tell the tokenizers lib that parallelism is allowed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Rayon (Rust thread pool used under the hood) respects this env var
cpu_count = os.cpu_count() or 1
os.environ.setdefault("RAYON_NUM_THREADS", str(cpu_count))

from tokenizers import Tokenizer  # noqa: E402 (import after env vars)


# If you run this from ~/Project_7, these paths will be correct:
EXTRACTED_DIR = Path("extracted")

# Your trained tokenizer JSON
TOKENIZER_PATH = Path.home() / "Project_7" / "tokenizer_bpe" / "tokenizer.json"

OUTPUT_TOKENS = Path("BPE_tokens.txt")

# How many docs to encode per batch (you can tweak this)
DEFAULT_BATCH_SIZE = 1024


# ---------------------------------------------------------
# 1. WikiExtractor doc iterator
# ---------------------------------------------------------

def iter_wiki_docs(root_dir: Path) -> Iterable[str]:
    """
    Yield plain-text docs from WikiExtractor output, skipping <doc> tags.
    Each yielded item is a full article as a single string.
    """
    # WikiExtractor typically names files like wiki_00, wiki_01, etc.
    for path in sorted(root_dir.rglob("wiki_*")):
        with path.open("r", encoding="utf-8") as f:
            in_doc = False
            buffer: List[str] = []

            for line in f:
                # Start of a document
                if line.startswith("<doc "):
                    in_doc = True
                    buffer = []
                    continue

                # End of a document
                if line.startswith("</doc>"):
                    if buffer:
                        text = "\n".join(buffer).strip()
                        if text:
                            yield text
                    in_doc = False
                    buffer = []
                    continue

                # Lines inside a document
                if in_doc:
                    buffer.append(line.rstrip("\n"))


# ---------------------------------------------------------
# 2. Simple batching utility
# ---------------------------------------------------------

def batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """
    Yield lists of up to `batch_size` items from `iterable`.
    """
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main(batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    if not EXTRACTED_DIR.exists():
        raise SystemExit(f"Folder not found: {EXTRACTED_DIR}")

    if not TOKENIZER_PATH.exists():
        raise SystemExit(f"Tokenizer file not found: {TOKENIZER_PATH}")

    print(f"CPU cores detected: {cpu_count}")
    print(f"Using tokenizers parallelism (TOKENIZERS_PARALLELISM={os.environ['TOKENIZERS_PARALLELISM']}, "
          f"RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']})")

    print(f"Loading custom BPE tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print("Opening output token file:", OUTPUT_TOKENS)
    OUTPUT_TOKENS.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    token_count = 0

    print(f"Starting tokenization from: {EXTRACTED_DIR}")
    print(f"Batch size: {batch_size}")

    with OUTPUT_TOKENS.open("w", encoding="utf-8") as out_f:
        for docs_batch in batched(iter_wiki_docs(EXTRACTED_DIR), batch_size):
            # Parallel batch encoding happens inside encode_batch
            encodings = tokenizer.encode_batch(docs_batch)

            for enc in encodings:
                ids = enc.ids
                out_f.write(" ".join(str(i) for i in ids) + "\n")
                doc_count += 1
                token_count += len(ids)

            if doc_count % 100_000 == 0:
                print(f"Encoded {doc_count} docs, {token_count} tokens so far")

    print(f"Done. Total docs: {doc_count}, total tokens: {token_count}")
    print(f"Tokens saved to: {OUTPUT_TOKENS}")


if __name__ == "__main__":
    # If you want to tweak the batch size, you can hardcode it here
    # or modify the script to read from CLI args.
    main(batch_size=DEFAULT_BATCH_SIZE)