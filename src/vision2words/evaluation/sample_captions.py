"""Caption a handful of validation images with a trained checkpoint.

This was the Phase 1 checkpoint deliverable (evidence the model learned
something beyond a loss number) and now doubles as a quick sanity check for
any of the three decoders.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from vision2words.data.dataset import CaptionFeatureDataset
from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import greedy_decode
from vision2words.models import build_decoder, feature_mode_for


def load_checkpoint(ckpt_path: str | Path, vocab: Vocabulary, device: str = "cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_decoder(checkpoint["model_config"], vocab_size=len(vocab), pad_id=vocab.pad_id)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint["model_config"]["type"]


def sample_captions(
    ckpt_path: str | Path,
    raw_dir: str | Path,
    feature_dir: str | Path,
    split: str = "val",
    num_samples: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> list[dict]:
    feature_dir = Path(feature_dir)
    vocab = Vocabulary.load(feature_dir / "vocab.json")
    model, model_type = load_checkpoint(ckpt_path, vocab, device)
    feature_mode = feature_mode_for(model_type)

    raw_dir = ensure_data(raw_dir)
    captions = load_captions(raw_dir)
    dataset = CaptionFeatureDataset(feature_dir, split, captions, vocab, feature_mode=feature_mode)

    rng = random.Random(seed)
    image_ids = sorted({image_id for image_id, _ in dataset.pairs})
    chosen = rng.sample(image_ids, k=min(num_samples, len(image_ids)))

    results = []
    for image_id in chosen:
        row = dataset.id_to_row[image_id]
        feature = torch.from_numpy(dataset.features[row].copy()).unsqueeze(0).to(device)
        generated = greedy_decode(model, feature, vocab.start_id, vocab.end_id, max_len=20)
        predicted = vocab.decode(generated[0].tolist())
        results.append(
            {
                "image_id": image_id,
                "model_type": model_type,
                "predicted_caption": predicted,
                "reference_captions": captions[image_id],
            }
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = sample_captions(
        args.checkpoint, args.raw_dir, args.feature_dir, args.split, args.num_samples, args.device
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
