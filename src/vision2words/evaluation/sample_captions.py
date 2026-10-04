"""Caption a handful of validation images with a trained checkpoint.

This is the Phase 1 checkpoint deliverable: evidence that the model has
actually learned something from the data, beyond a validation loss number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from vision2words.data.dataset import CaptionFeatureDataset
from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.data.vocabulary import Vocabulary
from vision2words.models.lstm_decoder import LSTMDecoder


def sample_captions(
    ckpt_path: str | Path,
    raw_dir: str | Path,
    feature_dir: str | Path,
    split: str = "val",
    num_samples: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> list[dict]:
    ckpt_path = Path(ckpt_path)
    feature_dir = Path(feature_dir)
    vocab = Vocabulary.load(feature_dir / "vocab.json")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = LSTMDecoder(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        feature_size=model_config["feature_size"],
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config.get("num_layers", 1),
        dropout=model_config.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    raw_dir = ensure_data(raw_dir)
    captions = load_captions(raw_dir)
    dataset = CaptionFeatureDataset(feature_dir, split, captions, vocab, feature_mode="pooled")

    rng = random.Random(seed)
    image_ids = sorted(set(image_id for image_id, _ in dataset.pairs))
    chosen = rng.sample(image_ids, k=min(num_samples, len(image_ids)))

    results = []
    with torch.no_grad():
        for image_id in chosen:
            row = dataset.id_to_row[image_id]
            feature = torch.from_numpy(dataset.features[row].copy()).unsqueeze(0).to(device)
            generated = model.generate_greedy(feature, vocab.start_id, vocab.end_id, max_len=20)
            predicted = vocab.decode(generated[0].tolist())
            results.append(
                {
                    "image_id": image_id,
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
