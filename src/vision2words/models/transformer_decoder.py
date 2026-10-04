"""Transformer decoder attending over the 49 spatial feature tokens.

This swaps the recurrent decoder for a standard encoder-decoder Transformer
(Vaswani et al., 2017): masked self-attention over the tokens generated so
far, cross-attention into the spatial feature grid in place of the usual
Transformer encoder output, sinusoidal positional encoding, and a final
linear layer back to vocabulary logits. Training is still teacher-forced,
but every position in the sequence is scored in parallel instead of one
LSTM step at a time, which is the main practical reason to use it here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vision2words.models.base import CaptionDecoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerDecoderModel(CaptionDecoder):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        feature_size: int = 1280,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_size: int = 2048,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.feature_proj = nn.Linear(feature_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.feature_proj(features)  # (B, 49, d_model)
        embedded = self.embedding(input_tokens) * math.sqrt(self.d_model)
        embedded = self.dropout(self.pos_encoding(embedded))

        seq_len = input_tokens.size(1)
        # Bool masks throughout (True = "may not attend here"), matching pad_mask's
        # dtype -- PyTorch warns and will eventually refuse a float causal mask
        # mixed with a bool padding mask.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_tokens.device), diagonal=1
        )
        pad_mask = input_tokens == self.pad_id  # (B, L) True where padded

        output = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )
        return self.classifier(output)
