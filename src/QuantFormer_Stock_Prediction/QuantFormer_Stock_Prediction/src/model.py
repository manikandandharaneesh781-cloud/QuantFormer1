from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, model_dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dimension)
        )

        encoding = torch.zeros(max_length, model_dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        if model_dimension > 1:
            encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.encoding[:, :inputs.size(1), :]
        return self.dropout(inputs)


class QuantFormer(nn.Module):
    """
    QuantFormer-inspired time-series model:
    feature projection + positional encoding + Transformer encoder + regression head.
    """
    def __init__(
        self,
        input_features: int = 5,
        model_dimension: int = 64,
        attention_heads: int = 4,
        encoder_layers: int = 2,
        feedforward_dimension: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if model_dimension % attention_heads != 0:
            raise ValueError("model_dimension must be divisible by attention_heads.")

        self.input_projection = nn.Linear(input_features, model_dimension)
        self.position = PositionalEncoding(model_dimension, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.normalization = nn.LayerNorm(model_dimension)
        self.regression_head = nn.Sequential(
            nn.Linear(model_dimension, model_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        hidden = self.normalization(hidden[:, -1, :])
        return self.regression_head(hidden)
