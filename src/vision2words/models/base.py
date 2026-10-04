"""Shared interface for caption decoders.

Both the LSTM decoder (Phase 2: with Bahdanau attention) and the Transformer
decoder plug into the same training loop and evaluation code through this
interface, so swapping one for the other is a config change, not a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CaptionDecoder(nn.Module, ABC):
    """A decoder maps encoder features + a token prefix to next-token logits."""

    @abstractmethod
    def forward(self, features: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """features: encoder output (shape depends on the decoder).
        input_tokens: (B, L) token ids, teacher-forced input (targets shifted left by one).
        returns: (B, L, vocab_size) logits.
        """

    @abstractmethod
    def generate_greedy(
        self, features: torch.Tensor, start_id: int, end_id: int, max_len: int
    ) -> torch.Tensor:
        """Greedy-decode one token at a time. Returns (B, <=max_len+1) token ids,
        starting with start_id.
        """
