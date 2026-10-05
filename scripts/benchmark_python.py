"""End-to-end Python inference latency (preprocess + encoder + decode),
matching what the C++ binary's --benchmark flag measures: the model is
loaded once, then only the per-image work is timed. (caption_image() alone
reloads the checkpoint from disk on every call, which is fine for a single
CLI invocation but not comparable to a repeated-inference benchmark.)
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.inference.caption import load_model
from vision2words.models import feature_mode_for
from vision2words.models.encoder import EncoderCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    encoder, decoder, vocab, model_type = load_model(args.checkpoint, args.vocab, args.device)
    feature_mode = feature_mode_for(model_type)
    transform = EncoderCNN.preprocess()

    def caption_once() -> str:
        image = Image.open(args.image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            pooled, spatial = encoder(tensor)
            features = pooled if feature_mode == "pooled" else spatial
            if args.beam_size > 1:
                generated = beam_search(decoder, features, vocab.start_id, vocab.end_id, args.beam_size, 20)
            else:
                generated = greedy_decode(decoder, features, vocab.start_id, vocab.end_id, 20)
        return vocab.decode(generated[0].tolist())

    caption_once()  # warm-up: first call pays for lazy CUDA context / cuDNN algo search on GPU
    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.runs):
        caption_once()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_ms = 1000 * elapsed / args.runs
    print(f"mean end-to-end latency over {args.runs} runs ({args.device}), model loaded once: {mean_ms:.2f} ms")


if __name__ == "__main__":
    main()
