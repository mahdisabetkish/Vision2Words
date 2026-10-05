"""CaptionFeatureDataset reads a cached-feature directory + captions and
produces (feature, token) pairs; collate_captions pads a batch of those to
one tensor. These are the two places an off-by-one in array indexing or
padding would silently corrupt training data without erroring.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from vision2words.data.dataset import CaptionFeatureDataset, collate_captions
from vision2words.data.vocabulary import Vocabulary


@pytest.fixture
def tiny_feature_dir(tmp_path):
    image_ids = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    pooled = np.random.randn(3, 1280).astype(np.float32)
    spatial = np.random.randn(3, 49, 1280).astype(np.float32)

    np.save(tmp_path / "train_pooled.npy", pooled)
    np.save(tmp_path / "train_spatial.npy", spatial)
    (tmp_path / "train_ids.json").write_text(json.dumps(image_ids))

    captions = {
        "img_a.jpg": ["a dog runs", "a happy dog"],
        "img_b.jpg": ["a cat sits"],
        # img_c.jpg deliberately has no captions, to check images without
        # any caption don't silently produce a training example for one.
    }
    vocab = Vocabulary.build([c for caps in captions.values() for c in caps], min_freq=1)
    return tmp_path, pooled, spatial, image_ids, captions, vocab


def test_dataset_length_is_total_caption_count_not_image_count(tiny_feature_dir):
    feature_dir, _pooled, _spatial, _ids, captions, vocab = tiny_feature_dir
    dataset = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode="pooled")
    # 2 captions for img_a + 1 for img_b + 0 for img_c = 3 examples, not 3 images.
    assert len(dataset) == 3


def test_dataset_returns_correct_pooled_feature_row(tiny_feature_dir):
    feature_dir, pooled, _spatial, _ids, captions, vocab = tiny_feature_dir
    dataset = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode="pooled")

    feature, tokens = dataset[0]
    assert feature.shape == (1280,)
    # The feature for img_a's first example must be img_a's actual cached
    # row (index 0), not whatever row happens to be first in the array.
    assert torch.allclose(feature, torch.from_numpy(pooled[0]))
    assert tokens[0].item() == vocab.start_id
    assert tokens[-1].item() == vocab.end_id


def test_dataset_spatial_mode_shape(tiny_feature_dir):
    feature_dir, _pooled, _spatial, _ids, captions, vocab = tiny_feature_dir
    dataset = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode="spatial")
    feature, _tokens = dataset[0]
    assert feature.shape == (49, 1280)


def test_collate_pads_to_longest_sequence_in_batch(tiny_feature_dir):
    feature_dir, _pooled, _spatial, _ids, captions, vocab = tiny_feature_dir
    dataset = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode="pooled")

    batch = [dataset[i] for i in range(len(dataset))]
    features, tokens = collate_captions(batch, pad_id=vocab.pad_id)

    lengths = [len(t) for _f, t in batch]
    assert features.shape == (len(batch), 1280)
    assert tokens.shape == (len(batch), max(lengths))
    # Shorter sequences are pad_id beyond their own real length.
    for row, length in zip(tokens, lengths, strict=True):
        assert (row[length:] == vocab.pad_id).all()
        assert (row[:length] != vocab.pad_id).all() or length == tokens.shape[1]
