"""Run the frozen encoder once over the whole dataset and cache its output.

Training reads these cached arrays instead of re-running EfficientNet-B0 on
every batch of every epoch. For each split this writes:
  {split}_ids.json      -- image ids, in the same row order as the arrays
  {split}_pooled.npy     -- (N, 1280) float32, global-average-pooled features
  {split}_spatial.npy    -- (N, 49, 1280) float32, the 7x7 spatial grid
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision2words.data.dataset import ImageOnlyDataset
from vision2words.data.flickr8k import ensure_data, load_captions, load_split
from vision2words.models.encoder import FEATURE_DIM, SPATIAL_TOKENS, EncoderCNN

logger = logging.getLogger(__name__)


def extract_features(
    raw_dir: str | Path,
    feature_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
    max_images: int | None = None,
) -> None:
    """max_images caps how many images per split get processed -- for the
    Nextflow `test` profile, which needs this to finish in minutes rather
    than running the real ~8000-image dataset. Leave it unset for a real run.
    """
    raw_dir = ensure_data(raw_dir)
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    captions = load_captions(raw_dir)
    splits = load_split(raw_dir)

    encoder = EncoderCNN(fine_tune=False).to(device).eval()
    transform = EncoderCNN.preprocess()

    for split_name, image_ids in splits.items():
        image_ids = [i for i in image_ids if i in captions]
        if max_images is not None:
            image_ids = image_ids[:max_images]
        dataset = ImageOnlyDataset(raw_dir / "images", image_ids, transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        pooled_all = np.zeros((len(image_ids), FEATURE_DIM), dtype=np.float32)
        spatial_all = np.zeros((len(image_ids), SPATIAL_TOKENS, FEATURE_DIM), dtype=np.float32)

        cursor = 0
        with torch.no_grad():
            for batch_ids, images in loader:
                images = images.to(device)
                pooled, spatial = encoder(images)
                n = images.size(0)
                assert list(batch_ids) == image_ids[cursor : cursor + n], (
                    "DataLoader reordered a shuffle=False loader; feature rows would "
                    "no longer line up with image_ids."
                )
                pooled_all[cursor : cursor + n] = pooled.cpu().numpy()
                spatial_all[cursor : cursor + n] = spatial.cpu().numpy()
                cursor += n

        np.save(feature_dir / f"{split_name}_pooled.npy", pooled_all)
        np.save(feature_dir / f"{split_name}_spatial.npy", spatial_all)
        (feature_dir / f"{split_name}_ids.json").write_text(
            json.dumps(image_ids, indent=2), encoding="utf-8"
        )
        logger.info("cached %d %s features -> %s", len(image_ids), split_name, feature_dir)
