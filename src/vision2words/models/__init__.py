"""Decoder registry: build any of the three decoders from a config dict.

Keeping this in one place is what makes "shared training/eval code" actually
true -- train.py and evaluate.py ask for a decoder by name and get back
both the model and the feature representation (pooled vs spatial) it needs,
instead of each having its own if/else over model types.
"""

from __future__ import annotations

from typing import Any, Literal

from vision2words.models.base import CaptionDecoder
from vision2words.models.lstm_attention_decoder import LSTMAttentionDecoder
from vision2words.models.lstm_decoder import LSTMDecoder
from vision2words.models.transformer_decoder import TransformerDecoderModel

FeatureMode = Literal["pooled", "spatial"]

_FEATURE_MODE_BY_TYPE: dict[str, FeatureMode] = {
    "lstm": "pooled",
    "lstm_attention": "spatial",
    "transformer": "spatial",
}


def feature_mode_for(model_type: str) -> FeatureMode:
    return _FEATURE_MODE_BY_TYPE[model_type]


def build_decoder(model_config: dict[str, Any], vocab_size: int, pad_id: int) -> CaptionDecoder:
    model_type = model_config["type"]
    if model_type == "lstm":
        return LSTMDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            feature_size=model_config["feature_size"],
            embed_size=model_config["embed_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.3),
        )
    if model_type == "lstm_attention":
        return LSTMAttentionDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            feature_size=model_config["feature_size"],
            embed_size=model_config["embed_size"],
            hidden_size=model_config["hidden_size"],
            attn_size=model_config.get("attn_size", 256),
            context_size=model_config.get("context_size", 256),
            dropout=model_config.get("dropout", 0.3),
        )
    if model_type == "transformer":
        return TransformerDecoderModel(
            vocab_size=vocab_size,
            pad_id=pad_id,
            feature_size=model_config["feature_size"],
            d_model=model_config.get("d_model", 512),
            num_layers=model_config.get("num_layers", 4),
            num_heads=model_config.get("num_heads", 8),
            ff_size=model_config.get("ff_size", 2048),
            dropout=model_config.get("dropout", 0.1),
            max_len=model_config.get("max_len", 64),
        )
    raise ValueError(f"unknown model.type: {model_type!r} (expected lstm, lstm_attention, transformer)")
