"""Caption an arbitrary image file with a trained checkpoint.

Unlike training and evaluation, this runs the real encoder (there's no
cached feature for an image the model has never seen), so this is also the
code path the C++ port and the Hugging Face Space both need to match.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from vision2words.data.vocabulary import Vocabulary
from vision2words.models.encoder import EncoderCNN
from vision2words.models.lstm_decoder import LSTMDecoder


def load_lstm_checkpoint(
    ckpt_path: str | Path, vocab_path: str | Path, device: str = "cpu"
) -> tuple[EncoderCNN, LSTMDecoder, Vocabulary]:
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]

    decoder = LSTMDecoder(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        feature_size=model_config["feature_size"],
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config.get("num_layers", 1),
        dropout=model_config.get("dropout", 0.3),
    ).to(device)
    decoder.load_state_dict(checkpoint["model_state"])
    decoder.eval()

    encoder = EncoderCNN(fine_tune=False).to(device).eval()
    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(
    image_path: str | Path,
    ckpt_path: str | Path,
    vocab_path: str | Path,
    device: str = "cpu",
    max_len: int = 20,
) -> str:
    encoder, decoder, vocab = load_lstm_checkpoint(ckpt_path, vocab_path, device)
    transform = EncoderCNN.preprocess()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    pooled, _spatial = encoder(tensor)
    generated = decoder.generate_greedy(pooled, vocab.start_id, vocab.end_id, max_len=max_len)
    return vocab.decode(generated[0].tolist())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    caption = caption_image(args.image, args.checkpoint, args.vocab, args.device)
    print(caption)


if __name__ == "__main__":
    main()
