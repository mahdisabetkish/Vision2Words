"""Plain LSTM caption decoder (no attention).

This is the Phase 1 baseline: a single pooled image-feature vector seeds the
LSTM's initial hidden state (Vinyals et al., "Show and Tell"), and the model
is trained with standard teacher forcing -- the input at each step is the
*previous* ground-truth token and the target is the *next* one. Random
scheduled sampling, which the original version of this project used, made
the loss impossible to compare across runs for no real benefit, so it's
gone; teacher forcing alone is both simpler and standard practice for this
kind of model. See models/lstm_attention_decoder.py for a Bahdanau-attention
version that reads from the spatial feature grid instead of a single pooled
vector. Greedy/beam decoding live in evaluation/decoding.py, shared across
every decoder in this package.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vision2words.models.base import CaptionDecoder


class LSTMDecoder(CaptionDecoder):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        feature_size: int = 1280,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.init_h = nn.Linear(feature_size, hidden_size)
        self.init_c = nn.Linear(feature_size, hidden_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def init_state(self, pooled_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.tanh(self.init_h(pooled_features)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(pooled_features)).unsqueeze(0)
        if self.num_layers > 1:
            h0 = h0.repeat(self.num_layers, 1, 1)
            c0 = c0.repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(self, features: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        h0, c0 = self.init_state(features)
        embedded = self.dropout(self.embedding(input_tokens))
        output, _ = self.lstm(embedded, (h0, c0))
        return self.classifier(self.dropout(output))

    @torch.no_grad()
    def generate_greedy(
        self, features: torch.Tensor, start_id: int, end_id: int, max_len: int = 20
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        h, c = self.init_state(features)
        current = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        generated = [current]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            embedded = self.embedding(current)
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.classifier(output[:, -1])
            current = logits.argmax(dim=-1, keepdim=True)
            generated.append(current)
            finished |= current.squeeze(1) == end_id
            if finished.all():
                break

        return torch.cat(generated, dim=1)
