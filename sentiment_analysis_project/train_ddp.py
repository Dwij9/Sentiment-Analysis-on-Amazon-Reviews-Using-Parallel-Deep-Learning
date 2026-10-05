"""
Distributed Data Parallel (DDP) training for the LSTM sentiment classifier.
Uses PyTorch's DistributedDataParallel to distribute training across GPUs.

Usage (single machine):
    torchrun --nproc_per_node=2 train_ddp.py
    torchrun --nproc_per_node=4 train_ddp.py --epochs 10

Usage (multi-machine – example with 2 nodes, 2 GPUs each):
    # Node 0
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \\
             --master_addr=<IP> --master_port=29500 train_ddp.py
    # Node 1
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \\
             --master_addr=<IP> --master_port=29500 train_ddp.py
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import config
from dataset import AmazonReviewDataset
from model import SentimentLSTM, count_parameters


# ──────────────────────────────────────────────────────────────
# Setup / Teardown
# ──────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize the distributed process group.
    `torchrun` sets the env variables RANK, LOCAL_RANK, WORLD_SIZE.
    """
    dist.init_process_group(backend=config.DDP_BACKEND)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    iterator = tqdm(loader, desc="  Train", leave=False) if is_main(dist.get_rank()) else loader

    for token_ids, labels in iterator:
        token_ids = token_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    iterator = tqdm(loader, desc="  Eval ", leave=False) if is_main(rank) else loader

    for token_ids, labels in iterator:
        token_ids = token_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(token_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    rank, local_rank, world_size, device = setup_ddp()

    if is_main(rank):
        print(f"DDP initialized – world_size={world_size}, backend={config.DDP_BACKEND}")
        print(f"Device: {device}")

    # ── Vocab ──
    with open(config.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # ── Datasets & Samplers ──
    train_ds = AmazonReviewDataset(config.TRAIN_PROCESSED)
    test_ds = AmazonReviewDataset(config.TEST_PROCESSED)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )

    if is_main(rank):
        print(f"Vocab size: {vocab_size:,}")
        print(f"Train samples: {len(train_ds):,}  |  Test samples: {len(test_ds):,}")
        print(f"Batches per rank: train={len(train_loader):,}  test={len(test_loader):,}")

    # ── Model ──
    model = SentimentLSTM(vocab_size=vocab_size).to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    if is_main(rank):
        print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )

    # ── Training loop ──
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": [],
               "epoch_time": []}

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # IMPORTANT: set epoch on sampler so each epoch sees a different shuffle
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        if is_main(rank):
            print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.GRAD_CLIP
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, test_loader, criterion, device, rank
        )

        epoch_time = time.time() - epoch_start

        # ── Aggregate metrics across ranks (average) ──
        metrics = torch.tensor(
            [train_loss, train_acc, train_f1, val_loss, val_acc, val_f1],
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size

        if is_main(rank):
            tl, ta, tf, vl, va, vf = metrics.cpu().tolist()
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["train_f1"].append(tf)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            history["val_f1"].append(vf)
            history["epoch_time"].append(epoch_time)

            print(f"  Train  loss={tl:.4f}  acc={ta:.4f}  f1={tf:.4f}")
            print(f"  Val    loss={vl:.4f}  acc={va:.4f}  f1={vf:.4f}")
            print(f"  Time   {epoch_time:.1f}s")

    total_time = time.time() - total_start

    if is_main(rank):
        history["total_time"] = total_time
        history["world_size"] = world_size
        print(f"\n{'='*50}")
        print(f"  Total DDP training time: {total_time:.1f}s  (world_size={world_size})")
        print(f"{'='*50}")

        # Save model (only rank 0)
        ckpt_path = os.path.join(config.MODEL_DIR, f"lstm_ddp_ws{world_size}.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"Model saved → {ckpt_path}")

        hist_path = os.path.join(config.RESULTS_DIR, f"history_ddp_ws{world_size}.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved → {hist_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    main(args)
