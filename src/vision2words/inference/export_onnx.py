"""Export the encoder and the two Phase 2 decoders to ONNX for the C++ port.

The encoder and the Transformer decoder export directly: neither has a
Python-level loop over timesteps, so a single traced graph handles any
image batch size / sequence length ONNX Runtime throws at it. The
attention-LSTM decoder is different -- its forward pass recomputes
attention from the *current* hidden state at every step, which means a
real Python loop, and tracing bakes a Python loop into a fixed number of
iterations. So it's exported as two small graphs instead: ``init_state``
(image features -> initial h, c) and ``step`` (one token + carried state ->
logits + new state), and the C++ port supplies the loop around them --
exactly mirroring the step() method vision2words.evaluation.decoding would
use if it decoded this way in Python (it doesn't need to: recomputing the
whole sequence each step, as decoding.py does, is simpler there because a
real Python loop is just a loop, not something that needs to survive
export).
"""

from __future__ import annotations

from pathlib import Path

import torch

from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.sample_captions import load_checkpoint
from vision2words.models.encoder import EncoderCNN
from vision2words.models.lstm_attention_decoder import LSTMAttentionDecoder

OPSET = 17


def export_encoder(out_dir: Path, device: str = "cpu") -> None:
    encoder = EncoderCNN(fine_tune=False).to(device).eval()
    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        encoder,
        (dummy_image,),
        str(out_dir / "encoder.onnx"),
        input_names=["image"],
        output_names=["pooled", "spatial"],
        dynamic_axes={"image": {0: "batch"}, "pooled": {0: "batch"}, "spatial": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )


class _InitStateWrapper(torch.nn.Module):
    """ONNX export needs an nn.Module whose forward IS the function to export;
    LSTMAttentionDecoder.init_state is a plain method, so wrap it.
    """

    def __init__(self, decoder: LSTMAttentionDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.init_state(features)


class _StepWrapper(torch.nn.Module):
    def __init__(self, decoder: LSTMAttentionDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self, token: torch.Tensor, h: torch.Tensor, c: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder.step(token, h, c, features)


def export_lstm_attention(
    ckpt_path: Path, vocab: Vocabulary, out_dir: Path, device: str = "cpu"
) -> None:
    model, model_type = load_checkpoint(ckpt_path, vocab, device)
    assert model_type == "lstm_attention", f"expected lstm_attention checkpoint, got {model_type}"

    hidden_size = model.lstm_cell.hidden_size
    features = torch.randn(1, 49, 1280, device=device)
    h = torch.randn(1, hidden_size, device=device)
    c = torch.randn(1, hidden_size, device=device)
    token = torch.zeros(1, dtype=torch.long, device=device)

    torch.onnx.export(
        _InitStateWrapper(model),
        (features,),
        str(out_dir / "lstm_attention_init.onnx"),
        input_names=["features"],
        output_names=["h0", "c0"],
        dynamic_axes={"features": {0: "batch"}, "h0": {0: "batch"}, "c0": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    torch.onnx.export(
        _StepWrapper(model),
        (token, h, c, features),
        str(out_dir / "lstm_attention_step.onnx"),
        input_names=["token", "h", "c", "features"],
        output_names=["logits", "h_next", "c_next", "attn_weights"],
        dynamic_axes={
            "token": {0: "batch"},
            "h": {0: "batch"},
            "c": {0: "batch"},
            "features": {0: "batch"},
            "logits": {0: "batch"},
            "h_next": {0: "batch"},
            "c_next": {0: "batch"},
            "attn_weights": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


FIXED_DECODE_LEN = 20  # matches the max_len used everywhere else in decoding


def export_transformer(
    ckpt_path: Path, vocab: Vocabulary, out_dir: Path, device: str = "cpu"
) -> None:
    """Export with a fixed sequence length rather than a dynamic one.

    The trace-based exporter bakes the dummy input's sequence length into the
    multi-head attention's internal reshape as a constant, so a dynamic_axes
    declaration doesn't actually make it dynamic (caught by the onnx_inference
    parity check). The Dynamo-based exporter handles dynamic shapes correctly
    in general, but nn.TransformerDecoder's internal fast-path branches on
    "is seq_len == 1" in a way torch.export can't guard on symbolically.

    Rather than fight that, this exports for a single fixed shape: a
    FIXED_DECODE_LEN token buffer, front-filled with generated tokens and
    pad-filled at the tail. The model's existing causal mask (position i
    can't see position >i) and padding mask (pad positions aren't attended
    to) together guarantee position i's output depends only on positions
    <=i, so re-running the whole fixed-size buffer at every decoding step
    and reading off position `step` gives exactly the same result as a
    variable-length forward pass would -- the C++ port just always passes a
    FIXED_DECODE_LEN buffer instead of a growing one.
    """
    model, model_type = load_checkpoint(ckpt_path, vocab, device)
    assert model_type == "transformer", f"expected transformer checkpoint, got {model_type}"

    features = torch.randn(1, 49, 1280, device=device)
    tokens = torch.zeros(1, FIXED_DECODE_LEN, dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        (features, tokens),
        str(out_dir / "transformer.onnx"),
        input_names=["features", "tokens"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,
    )


def export_all(
    feature_dir: str | Path,
    lstm_attention_ckpt: str | Path,
    transformer_ckpt: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.load(Path(feature_dir) / "vocab.json")

    export_encoder(out_dir, device)
    export_lstm_attention(Path(lstm_attention_ckpt), vocab, out_dir, device)
    export_transformer(Path(transformer_ckpt), vocab, out_dir, device)
    vocab.save(out_dir / "vocab.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--lstm-attention-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--out-dir", default="export")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    export_all(
        args.feature_dir,
        args.lstm_attention_checkpoint,
        args.transformer_checkpoint,
        args.out_dir,
        args.device,
    )
    print(f"exported ONNX graphs + vocab to {args.out_dir}")


if __name__ == "__main__":
    main()
