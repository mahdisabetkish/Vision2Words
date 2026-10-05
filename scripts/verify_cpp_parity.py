"""Compare Python and C++ captions on the same images, for both decoders.

Run after building cpp/build/v2w_caption (or v2w_caption.exe on Windows) and
exporting ONNX graphs with `v2w export-onnx`. Writes results/cpp_parity.json
and prints a summary table; does not fabricate a result if the binary is
missing or a run fails -- it reports the failure instead.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from vision2words.inference.caption import caption_image


def find_binary(build_dir: Path) -> Path:
    candidates = [
        build_dir / "Release" / "v2w_caption.exe",
        build_dir / "v2w_caption.exe",
        build_dir / "v2w_caption",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no v2w_caption binary found under {build_dir} (build it first)")


def run_cpp(
    binary: Path, image_path: Path, model_dir: Path, decoder: str, beam_size: int
) -> tuple[str, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [
            str(binary),
            str(image_path),
            "--model-dir",
            str(model_dir),
            "--decoder",
            decoder,
            "--beam-size",
            str(beam_size),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - start
    return result.stdout.strip(), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--feature-dir", default="data/features", help="where test_ids.json and vocab.json live"
    )
    parser.add_argument("--model-dir", default="export")
    parser.add_argument("--cpp-build-dir", default="cpp/build")
    parser.add_argument("--lstm-attention-checkpoint", default="checkpoints/lstm_attention/best.pt")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer/best.pt")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--out", default="results/cpp_parity.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    model_dir = Path(args.model_dir)
    binary = find_binary(Path(args.cpp_build_dir))

    # Resolve test image ids from the feature cache next to the checkpoints,
    # not from --model-dir (the ONNX export dir doesn't carry the split list).
    feature_dir = Path(args.feature_dir)
    test_ids = json.loads((feature_dir / "test_ids.json").read_text(encoding="utf-8"))[
        : args.num_images
    ]

    checkpoints = {
        "lstm_attention": args.lstm_attention_checkpoint,
        "transformer": args.transformer_checkpoint,
    }
    vocab_path = feature_dir / "vocab.json"

    rows = []
    for decoder, ckpt in checkpoints.items():
        for image_id in test_ids:
            image_path = raw_dir / "images" / image_id
            py_start = time.perf_counter()
            py_caption = caption_image(
                image_path, ckpt, vocab_path, device="cpu", beam_size=args.beam_size
            )
            py_seconds = time.perf_counter() - py_start

            try:
                cpp_caption, cpp_seconds = run_cpp(
                    binary, image_path, model_dir, decoder, args.beam_size
                )
                error = None
            except subprocess.CalledProcessError as exc:
                cpp_caption, cpp_seconds, error = None, None, exc.stderr

            rows.append(
                {
                    "decoder": decoder,
                    "image_id": image_id,
                    "python_caption": py_caption,
                    "cpp_caption": cpp_caption,
                    "match": py_caption == cpp_caption,
                    "python_seconds": py_seconds,
                    "cpp_seconds": cpp_seconds,
                    "error": error,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"platform": platform.platform(), "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    num_match = sum(1 for r in rows if r["match"])
    print(f"{num_match}/{len(rows)} captions matched exactly between Python and C++")
    for row in rows:
        status = "OK " if row["match"] else "DIFF"
        print(f"[{status}] {row['decoder']:14s} {row['image_id']}")
        print(f"       python: {row['python_caption']}")
        print(f"       cpp:    {row['cpp_caption'] or row['error']}")


if __name__ == "__main__":
    main()
