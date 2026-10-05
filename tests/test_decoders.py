"""Shape tests for all three decoders, through the shared build_decoder()
factory -- these are the thing a typo in a Linear layer's dimensions would
break immediately, long before a real training run would surface it.
"""

from __future__ import annotations

import pytest
import torch

from vision2words.models import build_decoder, feature_mode_for

VOCAB_SIZE = 37
PAD_ID = 0
BATCH = 4
SEQ_LEN = 6
FEATURE_SIZE = 1280
SPATIAL_TOKENS = 49


def _tokens():
    return torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.mark.parametrize(
    "model_type,config",
    [
        (
            "lstm",
            {"type": "lstm", "feature_size": FEATURE_SIZE, "embed_size": 16, "hidden_size": 24},
        ),
        (
            "lstm_attention",
            {
                "type": "lstm_attention",
                "feature_size": FEATURE_SIZE,
                "embed_size": 16,
                "hidden_size": 24,
                "attn_size": 12,
                "context_size": 10,
            },
        ),
        (
            "transformer",
            {
                "type": "transformer",
                "feature_size": FEATURE_SIZE,
                "d_model": 32,
                "num_layers": 2,
                "num_heads": 4,
                "ff_size": 64,
                "max_len": 32,
            },
        ),
    ],
)
def test_forward_output_shape(model_type, config):
    model = build_decoder(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
    feature_mode = feature_mode_for(model_type)
    features = (
        torch.randn(BATCH, FEATURE_SIZE)
        if feature_mode == "pooled"
        else torch.randn(BATCH, SPATIAL_TOKENS, FEATURE_SIZE)
    )

    logits = model(features, _tokens())

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_feature_mode_matches_expected_decoder_inputs():
    # The plain LSTM baseline conditions on one pooled vector; both
    # attention-based decoders read the spatial grid. Getting this backwards
    # silently breaks training (wrong tensor rank) in a way a shape test
    # catches immediately and a stack trace later wouldn't make obvious.
    assert feature_mode_for("lstm") == "pooled"
    assert feature_mode_for("lstm_attention") == "spatial"
    assert feature_mode_for("transformer") == "spatial"


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="unknown model.type"):
        build_decoder({"type": "not_a_real_decoder"}, vocab_size=VOCAB_SIZE, pad_id=PAD_ID)


def test_lstm_attention_step_matches_forward():
    # forward() is a loop over step() (see lstm_attention_decoder.py's
    # docstring on why); this is the regression test that keeps them in
    # sync if either one is edited without the other.
    config = {
        "type": "lstm_attention",
        "feature_size": FEATURE_SIZE,
        "embed_size": 16,
        "hidden_size": 24,
        "attn_size": 12,
        "context_size": 10,
    }
    model = build_decoder(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).eval()
    features = torch.randn(1, SPATIAL_TOKENS, FEATURE_SIZE)
    tokens = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN))

    with torch.no_grad():
        forward_logits = model(features, tokens)

        h, c = model.init_state(features)
        step_logits = []
        for t in range(SEQ_LEN):
            logits, h, c, _attn = model.step(tokens[:, t], h, c, features)
            step_logits.append(logits)
        step_logits = torch.stack(step_logits, dim=1)

    assert torch.allclose(forward_logits, step_logits, atol=1e-5)
