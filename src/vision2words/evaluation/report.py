"""Build the Phase 2 comparison artifacts from three already-evaluated
checkpoints: results/metrics.json, results/comparison.md, and two figures
(sample captions from all three decoders side by side, and an attention-map
visualization for the attention-LSTM decoder). Run `v2w evaluate` for each
checkpoint first -- this only reads what that already wrote to results/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vision2words.data.dataset import CaptionFeatureDataset
from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import beam_search
from vision2words.evaluation.sample_captions import load_checkpoint
from vision2words.models import feature_mode_for

MODEL_LABELS = {
    "lstm": "LSTM (no attention)",
    "lstm_attention": "LSTM + attention",
    "transformer": "Transformer",
}
RESULT_NAMES = ("lstm_baseline", "lstm_attention", "transformer")


def build_metrics_json(results_dir: Path) -> dict:
    combined = {
        name: json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in RESULT_NAMES
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return combined


def build_comparison_table(combined: dict, results_dir: Path) -> None:
    columns = [
        "Model",
        "Params",
        "Train s/epoch",
        "Decode",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
        "CIDEr",
        "ms/img",
    ]
    lines = [
        "# Decoder comparison",
        "",
        "Flickr8k test split, 1000 images. Beam search uses beam size 3. Latency is",
        "decoder-only (cached features), on the GTX 1080 Ti.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for name in RESULT_NAMES:
        data = combined[name]
        label = MODEL_LABELS[data["model_type"]]
        for decode_key, decode_label in (("greedy", "greedy"), ("beam", "beam k=3")):
            m = data[decode_key]
            lines.append(
                f"| {label} | {data['num_params']:,} | {data['mean_train_epoch_seconds']:.1f} | "
                f"{decode_label} | {m['bleu1']:.3f} | {m['bleu2']:.3f} | {m['bleu3']:.3f} | "
                f"{m['bleu4']:.3f} | {m['cider']:.3f} | {m['latency_ms_per_image']:.1f} |"
            )
    (results_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pick_test_ids(feature_dir: Path, count: int, seed: int) -> list[str]:
    ids = json.loads((feature_dir / "test_ids.json").read_text(encoding="utf-8"))
    return random.Random(seed).sample(ids, k=count)


def build_sample_figure(
    raw_dir: str | Path,
    feature_dir: str | Path,
    ckpt_paths: dict[str, Path],
    image_ids: list[str],
    device: str,
    out_path: Path,
) -> None:
    raw_dir = ensure_data(raw_dir)
    feature_dir = Path(feature_dir)
    captions = load_captions(raw_dir)
    vocab = Vocabulary.load(feature_dir / "vocab.json")

    models = {name: load_checkpoint(path, vocab, device) for name, path in ckpt_paths.items()}
    datasets = {}
    for _name, (_model, model_type) in models.items():
        fm = feature_mode_for(model_type)
        datasets.setdefault(
            fm, CaptionFeatureDataset(feature_dir, "test", captions, vocab, feature_mode=fm)
        )

    fig, axes = plt.subplots(len(image_ids), 1, figsize=(5.5, 4.4 * len(image_ids)))
    axes = np.atleast_1d(axes)

    for ax, image_id in zip(axes, image_ids, strict=True):
        image = Image.open(raw_dir / "images" / image_id).convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        lines = []
        for _name, (model, model_type) in models.items():
            ds = datasets[feature_mode_for(model_type)]
            row = ds.id_to_row[image_id]
            feature = torch.from_numpy(ds.features[row].copy()).unsqueeze(0).to(device)
            generated = beam_search(
                model, feature, vocab.start_id, vocab.end_id, beam_size=3, max_len=20
            )
            lines.append(f"{MODEL_LABELS[model_type]}: {vocab.decode(generated[0].tolist())}")
        lines.append(f"reference: {captions[image_id][0]}")
        ax.set_title("\n".join(lines), fontsize=9, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_attention_figure(
    raw_dir: str | Path,
    feature_dir: str | Path,
    ckpt_path: str | Path,
    image_ids: list[str],
    device: str,
    out_path: Path,
    max_words: int = 6,
) -> None:
    raw_dir = ensure_data(raw_dir)
    feature_dir = Path(feature_dir)
    vocab = Vocabulary.load(feature_dir / "vocab.json")
    model, model_type = load_checkpoint(ckpt_path, vocab, device)
    if model_type != "lstm_attention":
        raise ValueError("build_attention_figure needs the lstm_attention checkpoint")

    captions = load_captions(raw_dir)
    dataset = CaptionFeatureDataset(feature_dir, "test", captions, vocab, feature_mode="spatial")

    fig, axes = plt.subplots(
        len(image_ids), max_words, figsize=(2.1 * max_words, 2.3 * len(image_ids))
    )
    axes = np.atleast_2d(axes)

    for row_idx, image_id in enumerate(image_ids):
        row = dataset.id_to_row[image_id]
        feature = torch.from_numpy(dataset.features[row].copy()).unsqueeze(0).to(device)
        tokens, attn_maps = model.generate_with_attention(
            feature, vocab.start_id, vocab.end_id, max_len=20
        )
        words = [vocab.idx2word.get(t, "<unk>") for t in tokens[1:]]  # drop the leading <START>

        image = Image.open(raw_dir / "images" / image_id).convert("RGB").resize((224, 224))
        image_arr = np.asarray(image) / 255.0

        for col in range(max_words):
            ax = axes[row_idx, col]
            ax.axis("off")
            if col >= len(words):
                continue
            attn = attn_maps[col].reshape(7, 7).cpu().numpy()
            attn = attn / (attn.max() + 1e-8)
            attn_img = Image.fromarray((attn * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
            ax.imshow(image_arr)
            ax.imshow(np.asarray(attn_img) / 255.0, cmap="jet", alpha=0.45)
            ax.set_title(words[col], fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(
    raw_dir: str | Path,
    feature_dir: str | Path,
    results_dir: str | Path,
    checkpoints: dict[str, Path],
    device: str = "cpu",
    num_sample_images: int = 4,
    num_attention_images: int = 3,
    seed: int = 1337,
) -> None:
    results_dir = Path(results_dir)
    combined = build_metrics_json(results_dir)
    build_comparison_table(combined, results_dir)

    ids = _pick_test_ids(Path(feature_dir), num_sample_images + num_attention_images, seed)
    sample_ids, attention_ids = ids[:num_sample_images], ids[num_sample_images:]

    build_sample_figure(
        raw_dir, feature_dir, checkpoints, sample_ids, device, results_dir / "sample_captions.png"
    )
    build_attention_figure(
        raw_dir,
        feature_dir,
        checkpoints["lstm_attention"],
        attention_ids,
        device,
        results_dir / "attention_maps.png",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lstm-checkpoint", required=True)
    parser.add_argument("--lstm-attention-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoints = {
        "lstm_baseline": Path(args.lstm_checkpoint),
        "lstm_attention": Path(args.lstm_attention_checkpoint),
        "transformer": Path(args.transformer_checkpoint),
    }
    build_report(args.raw_dir, args.feature_dir, args.results_dir, checkpoints, args.device)


if __name__ == "__main__":
    main()
