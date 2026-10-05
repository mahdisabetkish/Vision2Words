"""Torch Dataset classes.

``ImageOnlyDataset`` is used once, during feature extraction, to read raw
images off disk through the encoder's preprocessing transform.
``CaptionFeatureDataset`` is used for training and evaluation: it reads
*cached* encoder features (no CNN forward pass at training time) and pairs
each one with a single tokenized caption. An image with five captions
becomes five training examples rather than one example with a 5-caption
dimension bolted on -- that bolted-on dimension was most of the complexity
in the original dataloader for no benefit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vision2words.data.vocabulary import Vocabulary

FeatureMode = Literal["pooled", "spatial"]


class ImageOnlyDataset(Dataset):
    def __init__(self, images_dir: str | Path, image_ids: list[str], transform) -> None:
        self.images_dir = Path(images_dir)
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        image_id = self.image_ids[idx]
        image = Image.open(self.images_dir / image_id).convert("RGB")
        return image_id, self.transform(image)


class CaptionFeatureDataset(Dataset):
    def __init__(
        self,
        feature_dir: str | Path,
        split: str,
        captions: dict[str, list[str]],
        vocab: Vocabulary,
        feature_mode: FeatureMode = "pooled",
    ) -> None:
        feature_dir = Path(feature_dir)
        self.vocab = vocab
        self.feature_mode = feature_mode

        ids = _load_ids(feature_dir, split)
        self.id_to_row = {image_id: row for row, image_id in enumerate(ids)}
        array_name = "pooled" if feature_mode == "pooled" else "spatial"
        self.features = np.load(feature_dir / f"{split}_{array_name}.npy", mmap_mode="r")

        self.pairs: list[tuple[str, str]] = [
            (image_id, caption) for image_id in ids for caption in captions.get(image_id, [])
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, caption = self.pairs[idx]
        row = self.id_to_row[image_id]
        feature = torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))
        tokens = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return feature, tokens


def _load_ids(feature_dir: Path, split: str) -> list[str]:
    import json

    return json.loads((feature_dir / f"{split}_ids.json").read_text(encoding="utf-8"))


def collate_captions(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features, token_seqs = zip(*batch, strict=True)
    features = torch.stack(features, dim=0)
    max_len = max(seq.size(0) for seq in token_seqs)
    padded = torch.full((len(token_seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(token_seqs):
        padded[i, : seq.size(0)] = seq
    return features, padded
