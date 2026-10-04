"""Caption an arbitrary image file with a trained checkpoint.

Unlike training and evaluation, this runs the real encoder (there's no
cached feature for an image the model has never seen), so this is also the
code path the C++ port, the benchmark, and the Gradio app all need to match.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.models import build_decoder, feature_mode_for
from vision2words.models.encoder import EncoderCNN


def load_model(
    ckpt_path: str | Path, vocab_path: str | Path, device: str = "cpu"
) -> tuple[EncoderCNN, torch.nn.Module, Vocabulary, str]:
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = checkpoint["model_config"]["type"]

    decoder = build_decoder(checkpoint["model_config"], vocab_size=len(vocab), pad_id=vocab.pad_id)
    decoder.load_state_dict(checkpoint["model_state"])
    decoder = decoder.to(device).eval()

    encoder = EncoderCNN(fine_tune=False).to(device).eval()
    return encoder, decoder, vocab, model_type


@torch.no_grad()
def caption_image(
    image_path: str | Path,
    ckpt_path: str | Path,
    vocab_path: str | Path,
    device: str = "cpu",
    max_len: int = 20,
    beam_size: int = 1,
) -> str:
    encoder, decoder, vocab, model_type = load_model(ckpt_path, vocab_path, device)
    feature_mode = feature_mode_for(model_type)
    transform = EncoderCNN.preprocess()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    pooled, spatial = encoder(tensor)
    features = pooled if feature_mode == "pooled" else spatial

    if beam_size > 1:
        generated = beam_search(decoder, features, vocab.start_id, vocab.end_id, beam_size, max_len)
    else:
        generated = greedy_decode(decoder, features, vocab.start_id, vocab.end_id, max_len)
    return vocab.decode(generated[0].tolist())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=1)
    args = parser.parse_args()

    caption = caption_image(args.image, args.checkpoint, args.vocab, args.device, beam_size=args.beam_size)
    print(caption)


if __name__ == "__main__":
    main()
