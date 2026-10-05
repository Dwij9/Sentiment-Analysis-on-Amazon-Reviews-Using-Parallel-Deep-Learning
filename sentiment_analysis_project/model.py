"""
LSTM-based Sentiment Classifier for Amazon Reviews.
Supports bidirectional LSTM with dropout and configurable depth.
"""

import torch
import torch.nn as nn

import config


class SentimentLSTM(nn.Module):
    """
    Architecture
    ────────────
    Embedding  →  Bidirectional LSTM  →  Dropout  →  FC  →  Softmax

    The final hidden states from both directions are concatenated and
    passed through a fully connected layer for binary classification.
    """

    def __init__(
        self,
        vocab_size: int = config.MAX_VOCAB_SIZE,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        bidirectional: bool = config.BIDIRECTIONAL,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len) – integer token ids

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        # x: (batch, seq_len)
        embedded = self.embedding(x)       # (batch, seq_len, embed_dim)

        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        # h_n:      (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Grab the last hidden state from each direction
        if self.bidirectional:
            # h_n[-2] = last layer forward, h_n[-1] = last layer backward
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim*2)
        else:
            hidden = h_n[-1]  # (batch, hidden_dim)

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # (batch, num_classes)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = SentimentLSTM()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Forward pass with dummy input
    dummy = torch.randint(0, config.MAX_VOCAB_SIZE, (4, config.MAX_SEQ_LENGTH))
    logits = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {logits.shape}")
