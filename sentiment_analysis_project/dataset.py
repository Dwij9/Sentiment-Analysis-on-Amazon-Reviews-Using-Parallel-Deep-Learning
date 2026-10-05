"""
PyTorch Dataset for preprocessed Amazon Reviews.
Reads the CSV produced by preprocess.py and serves (token_ids, label) pairs.
"""

import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


class AmazonReviewDataset(Dataset):
    """
    Reads a preprocessed CSV with columns: label, clean_text, token_ids.
    `token_ids` is stored as a string representation of a Python list, e.g. "[1, 42, 7, ...]"
    """

    def __init__(self, csv_path: str, max_len: int = config.MAX_SEQ_LENGTH):
        self.max_len = max_len
        df = pd.read_csv(csv_path)

        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

        # Parse the stringified token-id lists
        self.token_ids = []
        for row in df["token_ids"]:
            ids = ast.literal_eval(row) if isinstance(row, str) else row
            ids = list(ids)[:max_len]
            # Pad if shorter
            ids += [0] * (max_len - len(ids))
            self.token_ids.append(ids)

        self.token_ids = torch.tensor(self.token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler_train=None,
    sampler_test=None,
):
    """
    Convenience function to build train / test DataLoaders.
    When using DDP, pass DistributedSampler instances as sampler_train / sampler_test.
    """
    from torch.utils.data import DataLoader

    train_ds = AmazonReviewDataset(config.TRAIN_PROCESSED)
    test_ds = AmazonReviewDataset(config.TEST_PROCESSED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
