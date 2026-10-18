"""Decoder input that carries retrieved captions alongside the image features.

Every decoder takes ``forward(features, input_tokens)``, and the training
loop, greedy decoding and beam search only ever do four things with
``features``: move it to a device, read its batch size, read its device, and
(for beam search) repeat a single example once per beam. A plain tensor
already supports all four. This class supports the same four for an image
tensor plus a tensor of retrieved caption tokens, so the retrieval-augmented
decoder plugs into all of that shared code without a special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetrievalAugmentedInput:
    image: torch.Tensor  # (B, 49, feature_size) spatial features
    retrieved: torch.Tensor  # (B, R) token ids of retrieved captions, pad-filled

    def to(self, device: torch.device | str, non_blocking: bool = False) -> RetrievalAugmentedInput:
        return RetrievalAugmentedInput(
            self.image.to(device, non_blocking=non_blocking),
            self.retrieved.to(device, non_blocking=non_blocking),
        )

    def size(self, dim: int = 0) -> int:
        if dim != 0:
            raise ValueError("only the batch dimension (0) is shared by both tensors")
        return self.image.size(0)

    @property
    def device(self) -> torch.device:
        return self.image.device

    def expand_batch(self, n: int) -> RetrievalAugmentedInput:
        """Repeat a batch-size-1 input n times, for beam search."""
        if self.image.size(0) != 1:
            raise ValueError("expand_batch expects a single example")
        return RetrievalAugmentedInput(
            self.image.expand(n, *self.image.shape[1:]),
            self.retrieved.expand(n, *self.retrieved.shape[1:]),
        )


def expand_for_beams(features: torch.Tensor | RetrievalAugmentedInput, n: int):
    """Repeat a single example n times, whatever form the decoder input takes."""
    if isinstance(features, RetrievalAugmentedInput):
        return features.expand_batch(n)
    return features.expand(n, *features.shape[1:])
