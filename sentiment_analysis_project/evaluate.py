"""
Evaluation & Visualization for the Sentiment Analysis project.
Loads a saved model checkpoint, evaluates on the test set, and produces:
  - Classification report (precision, recall, F1)
  - Confusion matrix
  - Loss / accuracy curves (from training history JSON)
  - Speedup comparison chart

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/lstm_sequential.pt
    python evaluate.py --history results/history_sequential.json
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

import config
from dataset import AmazonReviewDataset, get_dataloaders
from model import SentimentLSTM


# ──────────────────────────────────────────────────────────────
# Evaluate a checkpoint on the test set
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, device=None):
    device = device or config.DEVICE

    # Load vocab
    with open(config.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = SentimentLSTM(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=2)

    all_preds, all_labels = [], []
    for token_ids, labels in tqdm(test_loader, desc="Evaluating"):
        token_ids = token_ids.to(device)
        logits = model(token_ids)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"{'='*60}")

    target_names = ["Negative (1-2★)", "Positive (4-5★)"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=target_names, yticklabels=target_names,
        xlabel="Predicted", ylabel="True",
        title=f"Confusion Matrix\nAcc={acc:.4f}  F1={f1:.4f}",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved → {cm_path}")

    return acc, f1


# ──────────────────────────────────────────────────────────────
# Plot training curves from a history JSON
# ──────────────────────────────────────────────────────────────

def plot_training_curves(history_path: str, tag: str = ""):
    with open(history_path, "r") as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(epochs, h["train_loss"], "o-", label="Train")
    axes[0].plot(epochs, h["val_loss"], "s-", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, h["train_acc"], "o-", label="Train")
    axes[1].plot(epochs, h["val_acc"], "s-", label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, h["train_f1"], "o-", label="Train")
    axes[2].plot(epochs, h["val_f1"], "s-", label="Validation")
    axes[2].set(xlabel="Epoch", ylabel="F1 Score", title="F1 Score Curve")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Training History {tag}", fontsize=14, y=1.02)
    fig.tight_layout()

    save_name = f"training_curves{'_' + tag if tag else ''}.png"
    save_path = os.path.join(config.RESULTS_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# Speedup comparison
# ──────────────────────────────────────────────────────────────

def plot_speedup():
    """Compare total training times from sequential and DDP history files."""
    results_dir = config.RESULTS_DIR
    seq_path = os.path.join(results_dir, "history_sequential.json")

    if not os.path.exists(seq_path):
        print("No sequential history found – skipping speedup comparison.")
        return

    with open(seq_path, "r") as f:
        seq = json.load(f)

    seq_time = seq["total_time"]
    labels = ["Sequential"]
    times = [seq_time]
    speedups = [1.0]

    # Find all DDP history files
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("history_ddp_ws") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r") as f:
                ddp = json.load(f)
            ws = ddp.get("world_size", "?")
            t = ddp["total_time"]
            labels.append(f"DDP (×{ws})")
            times.append(t)
            speedups.append(seq_time / t)

    if len(labels) < 2:
        print("No DDP history found – skipping speedup chart.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(labels)))

    ax1.bar(labels, times, color=colors)
    ax1.set(ylabel="Seconds", title="Total Training Time")
    ax1.grid(True, alpha=0.3, axis="y")
    for i, t in enumerate(times):
        ax1.text(i, t + max(times) * 0.02, f"{t:.0f}s", ha="center", fontsize=10)

    ax2.bar(labels, speedups, color=colors)
    ax2.set(ylabel="Speedup (×)", title="Speedup vs. Sequential")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, s in enumerate(speedups):
        ax2.text(i, s + max(speedups) * 0.02, f"{s:.2f}×", ha="center", fontsize=10)

    fig.suptitle("Parallelization Speedup Comparison", fontsize=14)
    fig.tight_layout()

    path = os.path.join(results_dir, "speedup_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Speedup chart saved → {path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.MODEL_DIR, "lstm_sequential.pt"),
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to a history JSON to plot. If omitted, plots all found histories.",
    )
    args = parser.parse_args()

    # Evaluate checkpoint
    if os.path.exists(args.checkpoint):
        evaluate_checkpoint(args.checkpoint)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")

    # Plot training curves
    if args.history and os.path.exists(args.history):
        plot_training_curves(args.history)
    else:
        # Auto-discover history files
        for fname in sorted(os.listdir(config.RESULTS_DIR)):
            if fname.startswith("history_") and fname.endswith(".json"):
                tag = fname.replace("history_", "").replace(".json", "")
                plot_training_curves(os.path.join(config.RESULTS_DIR, fname), tag=tag)

    # Speedup comparison
    plot_speedup()
