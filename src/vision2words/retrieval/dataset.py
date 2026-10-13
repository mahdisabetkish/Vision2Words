"""Training/evaluation data for the retrieval-augmented decoder.

Wraps the ordinary cached-feature dataset and attaches each example's
retrieved captions. Neighbours are looked up once per split, up front, in a
single batched search -- they depend only on the image, not on which of its
five captions the example uses, so there's nothing to recompute per batch.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from vision2words.data.dataset import CaptionFeatureDataset
from vision2words.models.inputs import RetrievalAugmentedInput
from vision2words.retrieval.context import RetrievedContext


class RetrievalCaptionDataset(Dataset):
    def __init__(
        self,
        base: CaptionFeatureDataset,
        neighbors: dict[str, list[str]],
        context: RetrievedContext,
        sample_captions: bool = False,
    ) -> None:
        if base.feature_mode != "spatial":
            raise ValueError("the retrieval-augmented decoder reads spatial features")
        self.base = base
        self.neighbors = neighbors
        self.context = context
        self.sample_captions = sample_captions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_feature, tokens = self.base[idx]
        image_id = self.base.pairs[idx][0]
        # DataLoader workers seed Python's `random` per worker from the main
        # process's torch seed, so this stays reproducible under set_seed().
        rng = random.Random(random.random()) if self.sample_captions else None
        captions = self.context.captions_for(self.neighbors[image_id], rng=rng)
        return image_feature, self.context.encode(captions), tokens


def _pad(seqs: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : seq.size(0)] = seq
    return padded


def collate_retrieval(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[RetrievalAugmentedInput, torch.Tensor]:
    images, retrieved, tokens = zip(*batch, strict=True)
    features = RetrievalAugmentedInput(torch.stack(images, dim=0), _pad(list(retrieved), pad_id))
    return features, _pad(list(tokens), pad_id)
