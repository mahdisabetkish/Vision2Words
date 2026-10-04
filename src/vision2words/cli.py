"""Command-line entry point: ``v2w <command> ...``.

Every path and hyperparameter is either a config value or a flag -- nothing
is hardcoded and nothing prompts interactively, so every command here also
works unattended inside the Docker image and the Nextflow pipeline.
"""

from __future__ import annotations

import argparse
import logging


def _add_extract_features(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("extract-features", help="cache encoder features for all splits")
    p.add_argument("--raw-dir", required=True, help="Flickr8k root (downloaded here if missing)")
    p.add_argument("--feature-dir", required=True, help="where to write cached feature arrays")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    def run(args: argparse.Namespace) -> None:
        from vision2words.data.features import extract_features

        extract_features(args.raw_dir, args.feature_dir, args.device, args.batch_size, args.num_workers)

    p.set_defaults(run=run)


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="train a caption decoder")
    p.add_argument("--config", required=True, help="path to a YAML config under configs/")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="override a config value, e.g. --set train.lr=0.0005 (repeatable)",
    )

    def run(args: argparse.Namespace) -> None:
        from vision2words.training.train import train

        best_ckpt = train(args.config, args.overrides)
        print(f"best checkpoint: {best_ckpt}")

    p.set_defaults(run=run)


def _add_sample_captions(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("sample-captions", help="caption a few images from a cached split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--device", default="cpu")

    def run(args: argparse.Namespace) -> None:
        import json

        from vision2words.evaluation.sample_captions import sample_captions

        results = sample_captions(
            args.checkpoint,
            args.raw_dir,
            args.feature_dir,
            args.split,
            args.num_samples,
            args.device,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))

    p.set_defaults(run=run)


def _add_caption(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("caption", help="caption a single image file")
    p.add_argument("image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--beam-size", type=int, default=1, help="1 = greedy")

    def run(args: argparse.Namespace) -> None:
        from vision2words.inference.caption import caption_image

        caption = caption_image(
            args.image, args.checkpoint, args.vocab, args.device, beam_size=args.beam_size
        )
        print(caption)

    p.set_defaults(run=run)


def _add_evaluate(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate", help="BLEU/CIDEr + latency on a split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--out", help="write the result as JSON to this path")

    def run(args: argparse.Namespace) -> None:
        import json

        from vision2words.evaluation.evaluate import evaluate_checkpoint

        result = evaluate_checkpoint(
            args.checkpoint,
            args.raw_dir,
            args.feature_dir,
            args.split,
            args.device,
            args.beam_size,
        )
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(text, encoding="utf-8")

    p.set_defaults(run=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v2w", description="Vision2Words CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_extract_features(subparsers)
    _add_train(subparsers)
    _add_sample_captions(subparsers)
    _add_caption(subparsers)
    _add_evaluate(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.run(args)


if __name__ == "__main__":
    main()
