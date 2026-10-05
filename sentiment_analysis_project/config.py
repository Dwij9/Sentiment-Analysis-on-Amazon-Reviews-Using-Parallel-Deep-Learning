"""
Central configuration for the Sentiment Analysis project.
Adjust these parameters to experiment with different settings.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_RAW = os.path.join(DATA_DIR, "train.ft.txt")
TEST_RAW = os.path.join(DATA_DIR, "test.ft.txt")
TRAIN_PROCESSED = os.path.join(DATA_DIR, "train_processed.csv")
TEST_PROCESSED = os.path.join(DATA_DIR, "test_processed.csv")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
MAX_VOCAB_SIZE = 50_000          # top-N words to keep
MAX_SEQ_LENGTH = 200             # pad / truncate reviews to this length
DASK_NUM_PARTITIONS = 8          # number of Dask partitions (≈ CPU cores)

# ──────────────────────────────────────────────
# Model (LSTM)
# ──────────────────────────────────────────────
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
NUM_CLASSES = 2                  # negative / positive

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ──────────────────────────────────────────────
# DDP
# ──────────────────────────────────────────────
DDP_BACKEND = "nccl" if torch.cuda.is_available() else "gloo"

# ──────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
