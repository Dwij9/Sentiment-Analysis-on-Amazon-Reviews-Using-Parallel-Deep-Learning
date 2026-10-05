"""
Full Experiment Pipeline
========================
Runs all stages end-to-end and prints a summary table.
Designed for a single machine; DDP is launched as a subprocess.

Usage:
    python run_experiments.py
    python run_experiments.py --skip_preprocess   # if data is already preprocessed
"""

import argparse
import json
import os
import subprocess
import sys
import time

import config


def banner(msg: str):
    print(f"\n{'='*64}")
    print(f"  {msg}")
    print(f"{'='*64}\n")


def run_cmd(cmd: list[str], description: str):
    """Run a shell command, stream output, return elapsed time."""
    banner(description)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[!] Command failed with return code {result.returncode}")
        sys.exit(1)
    print(f"\n  ⏱  {description}: {elapsed:.1f}s")
    return elapsed


def main(args):
    timings = {}

    # ── 1. Preprocessing ──────────────────────────────────────
    if not args.skip_preprocess:
        t = run_cmd(
            [sys.executable, "preprocess.py", "--mode", "both"],
            "Data Preprocessing (Pandas vs Dask comparison)",
        )
        timings["preprocess"] = t

    # Verify preprocessed files exist
    for path in [config.TRAIN_PROCESSED, config.TEST_PROCESSED, config.VOCAB_PATH]:
        if not os.path.exists(path):
            print(f"[!] Missing: {path}  — run preprocessing first.")
            sys.exit(1)

    # ── 2. Sequential training ────────────────────────────────
    t = run_cmd(
        [sys.executable, "train_sequential.py",
         "--epochs", str(args.epochs),
         "--batch_size", str(args.batch_size)],
        "Sequential Training (baseline)",
    )
    timings["train_sequential"] = t

    # ── 3. DDP training ───────────────────────────────────────
    n_gpus = config.NUM_GPUS
    if n_gpus >= 2:
        t = run_cmd(
            ["torchrun", f"--nproc_per_node={n_gpus}", "train_ddp.py",
             "--epochs", str(args.epochs),
             "--batch_size", str(args.batch_size)],
            f"DDP Training ({n_gpus} GPUs)",
        )
        timings[f"train_ddp_{n_gpus}gpu"] = t
    else:
        # Run DDP with 2 CPU workers for demonstration
        t = run_cmd(
            ["torchrun", "--nproc_per_node=2", "train_ddp.py",
             "--epochs", str(args.epochs),
             "--batch_size", str(args.batch_size)],
            "DDP Training (2 CPU workers – demo mode)",
        )
        timings["train_ddp_2cpu"] = t

    # ── 4. Evaluation ─────────────────────────────────────────
    t = run_cmd(
        [sys.executable, "evaluate.py"],
        "Evaluation & Visualization",
    )
    timings["evaluate"] = t

    # ── Summary ───────────────────────────────────────────────
    banner("EXPERIMENT SUMMARY")
    for stage, t in timings.items():
        print(f"  {stage:<30s}  {t:>8.1f}s")

    # Speedup
    seq = timings.get("train_sequential", 0)
    for key, t in timings.items():
        if key.startswith("train_ddp") and seq > 0 and t > 0:
            print(f"\n  Speedup ({key} vs sequential): {seq / t:.2f}×")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--skip_preprocess", action="store_true")
    args = parser.parse_args()
    main(args)
