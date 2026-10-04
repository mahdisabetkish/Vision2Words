"""Shared interface for caption decoders.

The LSTM baseline, the Bahdanau-attention LSTM, and the Transformer decoder
all plug into the same training loop and evaluation code through this one
method, so swapping one for the other is a config change, not a rewrite.
Greedy and beam-search decoding (evaluation/decoding.py) are written once,
against this interface, instead of once per architecture -- they just call
``forward`` again on the sequence generated so far and read off the last
position's logits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CaptionDecoder(nn.Module, ABC):
    """A decoder maps encoder features + a token prefix to next-token logits."""

    @abstractmethod
    def forward(self, features: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """features: encoder output (pooled (B, D) or spatial (B, 49, D), depending
        on the decoder). input_tokens: (B, L) token ids, teacher-forced input
        (targets shifted left by one). Returns (B, L, vocab_size) logits.
        """
