"""ONNX export parity: a freshly-initialized (untrained, random-weight) copy
of each exportable decoder must produce the same logits through ONNX
Runtime as through PyTorch directly. Skipped if onnx/onnxruntime aren't
installed (the `metrics`/`export` extras) rather than failing CI outright.

These tests build decoders directly via build_decoder() and never import
vision2words.data.flickr8k or vision2words.evaluation.sample_captions --
on Windows, importing those (which pull in pandas) in the same process as
onnxruntime has been observed to crash the interpreter (native heap
corruption from conflicting bundled math libraries), a known fragile
combination unrelated to anything these tests check.
"""

from __future__ import annotations

import pytest
import torch

from vision2words.models import build_decoder

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

pytestmark = pytest.mark.skipif(
    not _HAS_ORT, reason="onnxruntime not installed (pip install -e .[export])"
)

VOCAB_SIZE = 15
PAD_ID = 0
FEATURE_SIZE = 1280
SPATIAL_TOKENS = 49


def test_lstm_attention_init_and_step_match_pytorch(tmp_path):
    # dropout=0: this checks export correctness, not dropout behavior, and
    # an untrained (random-weight) model's unsaturated activations turn out
    # to be considerably more sensitive to any numerical wobble around
    # dropout tracing than a trained model's are -- harmless at 1e-4 for the
    # real trained checkpoints (see results/cpp_report.md), confounding here.
    model = build_decoder(
        {
            "type": "lstm_attention",
            "feature_size": FEATURE_SIZE,
            "embed_size": 8,
            "hidden_size": 12,
            "attn_size": 6,
            "context_size": 6,
            "dropout": 0.0,
        },
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
    ).eval()

    # decoder is registered as a submodule here (self.decoder = ...), not just
    # closed over -- otherwise the tracer sees its parameters as stray
    # requires_grad tensors instead of legitimate traced parameters and
    # refuses to export.
    class InitWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, features):
            return self.decoder.init_state(features)

    class StepWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, token, h, c, features):
            return self.decoder.step(token, h, c, features)

    features = torch.randn(1, SPATIAL_TOKENS, FEATURE_SIZE)
    h0 = torch.randn(1, 12)
    c0 = torch.randn(1, 12)
    token = torch.zeros(1, dtype=torch.long)

    init_path = tmp_path / "init.onnx"
    step_path = tmp_path / "step.onnx"
    torch.onnx.export(
        InitWrapper(model),
        (features,),
        str(init_path),
        input_names=["features"],
        output_names=["h0", "c0"],
        dynamo=False,
    )
    torch.onnx.export(
        StepWrapper(model),
        (token, h0, c0, features),
        str(step_path),
        input_names=["token", "h", "c", "features"],
        output_names=["logits", "h_next", "c_next", "attn_weights"],
        dynamo=False,
    )

    with torch.no_grad():
        torch_h, torch_c = model.init_state(features)
        torch_logits, _h, _c, _attn = model.step(token, torch_h, torch_c, features)

    init_sess = ort.InferenceSession(str(init_path), providers=["CPUExecutionProvider"])
    step_sess = ort.InferenceSession(str(step_path), providers=["CPUExecutionProvider"])

    onnx_h, onnx_c = init_sess.run(None, {"features": features.numpy()})
    assert torch.allclose(torch_h, torch.from_numpy(onnx_h), atol=1e-4)
    assert torch.allclose(torch_c, torch.from_numpy(onnx_c), atol=1e-4)

    onnx_logits, _h, _c, _attn = step_sess.run(
        None, {"token": token.numpy(), "h": onnx_h, "c": onnx_c, "features": features.numpy()}
    )
    assert torch.allclose(torch_logits, torch.from_numpy(onnx_logits), atol=1e-4)


def test_transformer_fixed_buffer_matches_pytorch(tmp_path):
    fixed_len = 10
    model = build_decoder(
        {
            "type": "transformer",
            "feature_size": FEATURE_SIZE,
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ff_size": 32,
            "max_len": fixed_len + 4,
            "dropout": 0.0,
        },
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
    ).eval()

    features = torch.randn(1, SPATIAL_TOKENS, FEATURE_SIZE)
    tokens = torch.zeros(1, fixed_len, dtype=torch.long)

    onnx_path = tmp_path / "transformer.onnx"
    torch.onnx.export(
        model,
        (features, tokens),
        str(onnx_path),
        input_names=["features", "tokens"],
        output_names=["logits"],
        dynamo=False,
    )

    real_tokens = torch.randint(1, VOCAB_SIZE, (1, fixed_len))
    with torch.no_grad():
        torch_logits = model(features, real_tokens)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"features": features.numpy(), "tokens": real_tokens.numpy()})[0]

    assert torch.allclose(torch_logits, torch.from_numpy(onnx_logits), atol=1e-4)
