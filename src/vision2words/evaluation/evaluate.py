"""Full test-split evaluation for one checkpoint: BLEU-1..4, CIDEr, param
count, mean training time per epoch, and decoder inference latency, for
both greedy and beam-search decoding.

Latency here is decoder-only (features are read from the cache, same as
training) since the encoder's cost is identical and fixed across every
decoder in this comparison; Phase 3's benchmark covers full end-to-end
(encoder + decoder) latency for Python vs the C++ port.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from vision2words.data.dataset import CaptionFeatureDataset
from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.evaluation.metrics import compute_bleu_cider
from vision2words.evaluation.sample_captions import load_checkpoint
from vision2words.models import feature_mode_for


def _mean_epoch_seconds(ckpt_dir: Path) -> float | None:
    log_path = ckpt_dir / "training_log.csv"
    if not log_path.exists():
        return None
    with open(log_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return sum(float(row["epoch_seconds"]) for row in rows) / len(rows)


def evaluate_checkpoint(
    ckpt_path: str | Path,
    raw_dir: str | Path,
    feature_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
    beam_size: int = 3,
    max_len: int = 20,
) -> dict:
    ckpt_path = Path(ckpt_path)
    feature_dir = Path(feature_dir)
    vocab = Vocabulary.load(feature_dir / "vocab.json")
    model, model_type = load_checkpoint(ckpt_path, vocab, device)
    feature_mode = feature_mode_for(model_type)

    raw_dir = ensure_data(raw_dir)
    captions = load_captions(raw_dir)
    dataset = CaptionFeatureDataset(feature_dir, split, captions, vocab, feature_mode=feature_mode)
    image_ids = sorted({image_id for image_id, _ in dataset.pairs})
    references = {image_id: captions[image_id] for image_id in image_ids}

    def _feature(image_id: str) -> torch.Tensor:
        row = dataset.id_to_row[image_id]
        return torch.from_numpy(dataset.features[row].copy()).unsqueeze(0).to(device)

    greedy_preds, start = {}, time.perf_counter()
    for image_id in image_ids:
        generated = greedy_decode(model, _feature(image_id), vocab.start_id, vocab.end_id, max_len)
        greedy_preds[image_id] = vocab.decode(generated[0].tolist())
    greedy_latency_ms = 1000 * (time.perf_counter() - start) / len(image_ids)

    beam_preds, start = {}, time.perf_counter()
    for image_id in image_ids:
        generated = beam_search(model, _feature(image_id), vocab.start_id, vocab.end_id, beam_size, max_len)
        beam_preds[image_id] = vocab.decode(generated[0].tolist())
    beam_latency_ms = 1000 * (time.perf_counter() - start) / len(image_ids)

    return {
        "model_type": model_type,
        "num_params": sum(p.numel() for p in model.parameters()),
        "mean_train_epoch_seconds": _mean_epoch_seconds(ckpt_path.parent),
        "split": split,
        "num_images": len(image_ids),
        "greedy": {**compute_bleu_cider(greedy_preds, references), "latency_ms_per_image": greedy_latency_ms},
        "beam": {
            **compute_bleu_cider(beam_preds, references),
            "latency_ms_per_image": beam_latency_ms,
            "beam_size": beam_size,
        },
        "sample_predictions": {image_id: greedy_preds[image_id] for image_id in image_ids[:10]},
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--out", help="write the result as JSON to this path")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint, args.raw_dir, args.feature_dir, args.split, args.device, args.beam_size
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
