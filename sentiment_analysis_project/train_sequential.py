"""
Sequential (single-device) training script for the LSTM sentiment classifier.
Serves as the baseline for speedup comparison against DDP training.

Usage:
    python train_sequential.py
    python train_sequential.py --epochs 10 --batch_size 256
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config
from dataset import AmazonReviewDataset, get_dataloaders
from model import SentimentLSTM, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for token_ids, labels in tqdm(loader, desc="  Train", leave=False):
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for token_ids, labels in tqdm(loader, desc="  Eval ", leave=False):
        token_ids = token_ids.to(device)
        labels = labels.to(device)

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


def main(args):
    device = config.DEVICE
    print(f"Device: {device}")

    # ── Load vocab size ──
    with open(config.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size:,}")

    # ── DataLoaders ──
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=2
    )
    print(f"Train batches: {len(train_loader):,}  |  Test batches: {len(test_loader):,}")

    # ── Model ──
    model = SentimentLSTM(vocab_size=vocab_size).to(device)
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
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.GRAD_CLIP
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, test_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epoch_time"].append(epoch_time)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")
        print(f"  Time   {epoch_time:.1f}s")

    total_time = time.time() - total_start
    history["total_time"] = total_time
    print(f"\n{'='*50}")
    print(f"  Total training time: {total_time:.1f}s")
    print(f"{'='*50}")

    # ── Save ──
    ckpt_path = os.path.join(config.MODEL_DIR, "lstm_sequential.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved → {ckpt_path}")

    hist_path = os.path.join(config.RESULTS_DIR, "history_sequential.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    main(args)
