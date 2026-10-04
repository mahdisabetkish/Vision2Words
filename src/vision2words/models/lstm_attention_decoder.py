"""LSTM decoder with Bahdanau (additive) attention over the spatial feature
grid -- "Show, Attend and Tell" (Xu et al., 2015), minus the doubly
stochastic attention regularizer, which is a nice-to-have rather than a
requirement for a dataset this size.

The Phase 1 baseline has to compress a whole image into one pooled vector
before it writes a single word. This decoder instead keeps all 49 spatial
feature tokens around and, at every decoding step, learns a soft weighting
over them from the current hidden state -- so it can look at a different
region of the image for each word, and the weights themselves double as a
visualization of what the model is "looking at".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vision2words.models.base import CaptionDecoder


class BahdanauAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, attn_size: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(feature_size, attn_size)
        self.hidden_proj = nn.Linear(hidden_size, attn_size)
        self.energy = nn.Linear(attn_size, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (B, 49, feature_size); hidden: (B, hidden_size).
        Returns (context (B, feature_size), weights (B, 49)).
        """
        feat_proj = self.feature_proj(features)
        hid_proj = self.hidden_proj(hidden).unsqueeze(1)
        scores = self.energy(torch.tanh(feat_proj + hid_proj)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * features).sum(dim=1)
        return context, weights


class LSTMAttentionDecoder(CaptionDecoder):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        feature_size: int = 1280,
        embed_size: int = 256,
        hidden_size: int = 512,
        attn_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.attention = BahdanauAttention(feature_size, hidden_size, attn_size)
        self.init_h = nn.Linear(feature_size, hidden_size)
        self.init_c = nn.Linear(feature_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + feature_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        # Populated by forward(); read by the attention-map figure in Phase 2's report.
        self.last_attention_weights: torch.Tensor | None = None

    def init_state(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_feature = features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_feature))
        c = torch.tanh(self.init_c(mean_feature))
        return h, c

    def forward(self, features: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_tokens.shape
        h, c = self.init_state(features)
        embedded = self.dropout(self.embedding(input_tokens))

        logits_steps = []
        attn_steps = []
        for t in range(seq_len):
            context, attn_weights = self.attention(features, h)
            lstm_input = torch.cat([embedded[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits_steps.append(self.classifier(self.dropout(h)))
            attn_steps.append(attn_weights)

        self.last_attention_weights = torch.stack(attn_steps, dim=1)  # (B, L, 49)
        return torch.stack(logits_steps, dim=1)

    @torch.no_grad()
    def generate_with_attention(
        self, features: torch.Tensor, start_id: int, end_id: int, max_len: int = 20
    ) -> tuple[list[int], torch.Tensor]:
        """Greedy-decode a single image (features must have batch size 1),
        keeping the per-step attention map -- used only for the attention
        visualization figure, where we want the maps as we generate rather
        than recomputed afterwards.
        """
        if features.size(0) != 1:
            raise ValueError("generate_with_attention decodes one image at a time")
        h, c = self.init_state(features)
        current = torch.tensor([[start_id]], device=features.device)
        tokens = [start_id]
        attn_maps = []

        for _ in range(max_len):
            embedded = self.embedding(current).squeeze(1)
            context, attn_weights = self.attention(features, h)
            lstm_input = torch.cat([embedded, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.classifier(h)
            next_token = int(logits.argmax(dim=-1).item())
            tokens.append(next_token)
            attn_maps.append(attn_weights.squeeze(0))
            if next_token == end_id:
                break
            current = torch.tensor([[next_token]], device=features.device)

        return tokens, torch.stack(attn_maps, dim=0)
