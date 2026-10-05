"""Greedy and beam-search decoding, against a tiny decoder of each kind."""

from __future__ import annotations

import pytest
import torch

from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.models import build_decoder, feature_mode_for

VOCAB_SIZE = 20
PAD_ID, START_ID, END_ID = 0, 1, 2
FEATURE_SIZE = 1280
SPATIAL_TOKENS = 49
MAX_LEN = 10


@pytest.fixture(params=["lstm", "lstm_attention", "transformer"])
def model_and_features(request):
    configs = {
        "lstm": {"type": "lstm", "feature_size": FEATURE_SIZE, "embed_size": 8, "hidden_size": 12},
        "lstm_attention": {
            "type": "lstm_attention",
            "feature_size": FEATURE_SIZE,
            "embed_size": 8,
            "hidden_size": 12,
            "attn_size": 6,
            "context_size": 6,
        },
        "transformer": {
            "type": "transformer",
            "feature_size": FEATURE_SIZE,
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ff_size": 32,
            "max_len": MAX_LEN + 2,
        },
    }
    model_type = request.param
    model = build_decoder(configs[model_type], vocab_size=VOCAB_SIZE, pad_id=PAD_ID).eval()
    feature_mode = feature_mode_for(model_type)
    features = (
        torch.randn(1, FEATURE_SIZE)
        if feature_mode == "pooled"
        else torch.randn(1, SPATIAL_TOKENS, FEATURE_SIZE)
    )
    return model, features


def test_greedy_decode_starts_with_start_token_and_respects_max_len(model_and_features):
    model, features = model_and_features
    sequence = greedy_decode(model, features, START_ID, END_ID, max_len=MAX_LEN)

    assert sequence[0, 0].item() == START_ID
    assert sequence.shape[1] <= MAX_LEN + 1  # +1 for the leading <START>


def test_greedy_decode_stops_early_if_end_token_is_generated(model_and_features):
    model, features = model_and_features
    # Force the classifier to always prefer <END> so we can check the loop
    # actually stops instead of padding out to max_len regardless.
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        pytest.skip("this decoder type exposes no single classifier layer to patch")
    with torch.no_grad():
        # Zero the weight too, not just the bias: otherwise the logits are
        # bias + W @ x, and a large-enough random x could still outvote the
        # bias on some seeds, making this test flaky instead of a real check.
        classifier.weight.zero_()
        classifier.bias.fill_(-10.0)
        classifier.bias[END_ID] = 10.0

    sequence = greedy_decode(model, features, START_ID, END_ID, max_len=MAX_LEN)
    assert sequence.shape[1] == 2  # <START>, then immediately <END>
    assert sequence[0, 1].item() == END_ID


def test_beam_search_returns_one_sequence_starting_with_start_token(model_and_features):
    model, features = model_and_features
    sequence = beam_search(model, features, START_ID, END_ID, beam_size=3, max_len=MAX_LEN)

    assert sequence.shape[0] == 1
    assert sequence[0, 0].item() == START_ID
    assert sequence.shape[1] <= MAX_LEN + 1


def test_beam_search_rejects_batched_features(model_and_features):
    model, features = model_and_features
    batched = features.repeat(2, *([1] * (features.dim() - 1)))
    with pytest.raises(ValueError):
        beam_search(model, batched, START_ID, END_ID, beam_size=2, max_len=MAX_LEN)


def test_beam_size_one_is_deterministic_like_greedy_for_lstm():
    # Not a general equivalence (length normalization can tip a true tie
    # either way), but beam_size=1 has exactly one candidate at every step,
    # so it has no tie to break and must match greedy exactly.
    config = {"type": "lstm", "feature_size": FEATURE_SIZE, "embed_size": 8, "hidden_size": 12}
    model = build_decoder(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).eval()
    features = torch.randn(1, FEATURE_SIZE)

    greedy = greedy_decode(model, features, START_ID, END_ID, max_len=MAX_LEN)
    beam = beam_search(model, features, START_ID, END_ID, beam_size=1, max_len=MAX_LEN)

    assert torch.equal(greedy, beam)
