#!/usr/bin/env python3
"""
Train a Byte-Level BPE tokenizer on the MediaWiki dataset.

- Scans all wiki_* or *.txt files under:  ~/Project_7/extracted  (recursively)
- Trains a BPE tokenizer using HuggingFace `tokenizers`
- Saves tokenizer JSON to:      <output-dir>/tokenizer.json

Example usage:

    python train_tokenizer.py \
        --data-root  ~/Project_7/extracted \
        --output-dir ~/Project_7/tokenizer_bpe \
        --vocab-size 50257 \
        --min-frequency 2
"""

import argparse
import os
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


# ---------------------------------------------------------
# 1. Find text files (recursively)
# ---------------------------------------------------------

def find_text_files(root: Path) -> List[Path]:
    """
    Recursively scan `root` for text shards.

    Priority:
      1) Files named like 'wiki_*' (typical WikiExtractor output)
      2) If none, any '*.txt' files
      3) If still none, all regular files under `root`
    """
    # 1) Try wiki_* first (most likely for WikiExtractor dumps)
    files = [p for p in root.rglob("wiki_*") if p.is_file()]

    # 2) If that found nothing, try *.txt
    if not files:
        files = [p for p in root.rglob("*.txt") if p.is_file()]

    # 3) If still nothing, fall back to "all files"
    if not files:
        files = [p for p in root.rglob("*") if p.is_file()]

    if not files:
        raise RuntimeError(
            f"No candidate text files found under {root}. "
            "Check the directory structure or filenames."
        )

    files.sort()
    print(f"Found {len(files)} text shards under (recursively): {root}")
    return files


# ---------------------------------------------------------
# 2. Train BPE tokenizer
# ---------------------------------------------------------

def train_bpe_tokenizer(
    files: List[Path],
    vocab_size: int,
    min_frequency: int,
    save_path: Path,
) -> None:
    """
    Train a Byte-Level BPE tokenizer on the given text files.
    """
    # Byte-level BPE (like GPT-2)
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<|bos|>",
            "<|eos|>",
            "<|pad|>",
            "<|unk|>",
        ],
    )

    # Convert Path -> str for the `tokenizers` library
    file_paths = [str(p) for p in files]

    print("Training Byte-Level BPE tokenizer...")
    print(f"  Vocab size:     {vocab_size}")
    print(f"  Min frequency:  {min_frequency}")
    print(f"  # training files: {len(file_paths)}")
    print(f"  Output path:    {save_path}")

    tokenizer.train(files=file_paths, trainer=trainer)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"✓ Saved tokenizer JSON to: {save_path}")


# ---------------------------------------------------------
# 3. CLI args
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    home = Path(os.path.expanduser("~"))
    default_data_root = home / "Project_7" / "extracted"
    default_output_dir = home / "Project_7" / "tokenizer_bpe"

    parser = argparse.ArgumentParser(
        description="Train a Byte-Level BPE tokenizer on MediaWiki text files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(default_data_root),
        help=f"Root directory containing text shards (default: {default_data_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Directory where tokenizer.json will be saved (default: {default_output_dir})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50257,
        help="Target vocabulary size (default: 50257 ~ GPT-2 style)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency to be included in vocab (default: 2)",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help=(
            "Optional: limit the number of text files used for training "
            "(useful for quick experiments)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Scanning for text shards under: {data_root}")
    text_files = find_text_files(data_root)

    if args.limit_files is not None and args.limit_files > 0:
        print(f"Limiting to first {args.limit_files} files for training.")
        text_files = text_files[: args.limit_files]

    save_path = output_dir / "tokenizer.json"

    train_bpe_tokenizer(
        files=text_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()