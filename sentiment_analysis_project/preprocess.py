"""
Data Preprocessing for Amazon Reviews Sentiment Analysis
=========================================================
Two implementations:
  1. Pandas (sequential baseline)
  2. Dask  (parallel – scales across CPU cores)

Both produce identical outputs:  train_processed.csv  /  test_processed.csv
with columns:  label  |  clean_text  |  token_ids

Usage:
    python preprocess.py                # runs both, prints timing comparison
    python preprocess.py --mode pandas  # sequential only
    python preprocess.py --mode dask    # parallel  only
"""

import argparse
import os
import pickle
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

import config


# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────

def parse_line(line: str):
    """Parse a fastText-formatted line into (label, text).
    Format:  __label__1 review text here ...
    Returns: (0 or 1, cleaned text string)
    """
    line = line.strip()
    if line.startswith("__label__1"):
        label = 0  # negative
        text = line[len("__label__1 "):]
    elif line.startswith("__label__2"):
        label = 1  # positive
        text = line[len("__label__2 "):]
    else:
        return None, None
    return label, text


def clean_text(text: str) -> str:
    """Remove special characters and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)        # strip HTML tags
    text = re.sub(r"http\S+", " ", text)         # strip URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # keep alphanumeric + spaces
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def build_vocab(texts, max_size: int = config.MAX_VOCAB_SIZE):
    """Build word→index mapping from a list/series of cleaned text."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # Reserve 0 for <PAD>, 1 for <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict, max_len: int = config.MAX_SEQ_LENGTH):
    """Convert cleaned text to a fixed-length list of integer token ids."""
    tokens = text.split()
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens[:max_len]]
    # Pad
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


# ──────────────────────────────────────────────────────────────────
# 1.  Pandas (sequential) pipeline
# ──────────────────────────────────────────────────────────────────

def preprocess_pandas(input_path: str):
    """Read raw fastText file, clean, tokenize – pure Pandas."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        label, text = parse_line(line)
        if label is not None:
            records.append({"label": label, "text": text})

    df = pd.DataFrame(records)
    df["clean_text"] = df["text"].apply(clean_text)
    df.drop(columns=["text"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────
# 2.  Dask (parallel) pipeline
# ──────────────────────────────────────────────────────────────────

def preprocess_dask(input_path: str, n_partitions: int = config.DASK_NUM_PARTITIONS):
    """Read raw fastText file, clean, tokenize – parallelized with Dask."""
    import dask.dataframe as dd

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        label, text = parse_line(line)
        if label is not None:
            records.append({"label": label, "text": text})

    pdf = pd.DataFrame(records)
    ddf = dd.from_pandas(pdf, npartitions=n_partitions)

    # Parallel text cleaning
    ddf["clean_text"] = ddf["text"].apply(clean_text, meta=("clean_text", "str"))
    ddf = ddf.drop("text", axis=1)

    # Compute back to Pandas
    df = ddf.compute()
    return df


# ──────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────

def run_pipeline(mode: str = "both"):
    """Run the preprocessing pipeline and save processed CSVs."""

    results = {}

    for split_name, raw_path in [("train", config.TRAIN_RAW), ("test", config.TEST_RAW)]:
        if not os.path.exists(raw_path):
            print(f"[!] {raw_path} not found – download the dataset from Kaggle first.")
            return

    # ── Pandas baseline ───────────────────────────────────────────
    if mode in ("pandas", "both"):
        print("\n{'='*60}")
        print("  PANDAS (sequential) preprocessing")
        print(f"{'='*60}")

        t0 = time.time()
        train_df = preprocess_pandas(config.TRAIN_RAW)
        t1 = time.time()
        test_df = preprocess_pandas(config.TEST_RAW)
        t2 = time.time()

        pandas_train_time = t1 - t0
        pandas_test_time = t2 - t1
        pandas_total = t2 - t0

        print(f"  Train cleaning : {pandas_train_time:.2f}s  ({len(train_df):,} rows)")
        print(f"  Test  cleaning : {pandas_test_time:.2f}s  ({len(test_df):,} rows)")
        print(f"  Total          : {pandas_total:.2f}s")

        results["pandas"] = pandas_total

        if mode == "pandas":
            # Build vocab & encode
            vocab = build_vocab(train_df["clean_text"])
            train_df["token_ids"] = train_df["clean_text"].apply(
                lambda t: encode_text(t, vocab)
            )
            test_df["token_ids"] = test_df["clean_text"].apply(
                lambda t: encode_text(t, vocab)
            )
            train_df.to_csv(config.TRAIN_PROCESSED, index=False)
            test_df.to_csv(config.TEST_PROCESSED, index=False)
            with open(config.VOCAB_PATH, "wb") as f:
                pickle.dump(vocab, f)
            print(f"\n  Vocab size: {len(vocab):,}")
            print(f"  Saved → {config.TRAIN_PROCESSED}")
            print(f"  Saved → {config.TEST_PROCESSED}")

    # ── Dask parallel ─────────────────────────────────────────────
    if mode in ("dask", "both"):
        print(f"\n{'='*60}")
        print(f"  DASK (parallel, {config.DASK_NUM_PARTITIONS} partitions) preprocessing")
        print(f"{'='*60}")

        t0 = time.time()
        train_df = preprocess_dask(config.TRAIN_RAW)
        t1 = time.time()
        test_df = preprocess_dask(config.TEST_RAW)
        t2 = time.time()

        dask_train_time = t1 - t0
        dask_test_time = t2 - t1
        dask_total = t2 - t0

        print(f"  Train cleaning : {dask_train_time:.2f}s  ({len(train_df):,} rows)")
        print(f"  Test  cleaning : {dask_test_time:.2f}s  ({len(test_df):,} rows)")
        print(f"  Total          : {dask_total:.2f}s")

        results["dask"] = dask_total

    # ── Always save the last computed version ─────────────────────
    print(f"\n{'='*60}")
    print("  Building vocabulary & encoding tokens")
    print(f"{'='*60}")

    t0 = time.time()
    vocab = build_vocab(train_df["clean_text"])
    train_df["token_ids"] = train_df["clean_text"].apply(
        lambda t: encode_text(t, vocab)
    )
    test_df["token_ids"] = test_df["clean_text"].apply(
        lambda t: encode_text(t, vocab)
    )
    t1 = time.time()
    print(f"  Vocab size     : {len(vocab):,}")
    print(f"  Encoding time  : {t1 - t0:.2f}s")

    train_df.to_csv(config.TRAIN_PROCESSED, index=False)
    test_df.to_csv(config.TEST_PROCESSED, index=False)
    with open(config.VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    print(f"\n  Saved → {config.TRAIN_PROCESSED}")
    print(f"  Saved → {config.TEST_PROCESSED}")
    print(f"  Saved → {config.VOCAB_PATH}")

    # ── Comparison ────────────────────────────────────────────────
    if "pandas" in results and "dask" in results:
        speedup = results["pandas"] / results["dask"]
        print(f"\n{'='*60}")
        print(f"  SPEEDUP COMPARISON")
        print(f"{'='*60}")
        print(f"  Pandas total : {results['pandas']:.2f}s")
        print(f"  Dask total   : {results['dask']:.2f}s")
        print(f"  Speedup      : {speedup:.2f}×")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Amazon reviews")
    parser.add_argument(
        "--mode",
        choices=["pandas", "dask", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    args = parser.parse_args()
    run_pipeline(mode=args.mode)
