"""Retrieval + language-model captioning, without downloading a model: the
text handling is tested directly, and the generation plumbing (chat
template, left padding, cutting the prompt off the output, cleaning the
reply) runs against small stand-ins for a tokenizer and a causal LM.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("faiss")

from vision2words.evaluation.evaluate_rag_llm import evaluate_rag_llm  # noqa: E402
from vision2words.retrieval.llm_captioner import (  # noqa: E402
    LLMCaptioner,
    build_user_prompt,
    clean_caption,
    consensus_caption,
)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("A dog runs on the beach.", "a dog runs on the beach"),
        ('"A dog runs on the beach."', "a dog runs on the beach"),
        ("Caption: Two kids play soccer.\nThey look happy.", "two kids play soccer"),
        ("\n\n- a man rides a bike\n", "a man rides a bike"),
        ("", ""),
    ],
)
def test_clean_caption(reply, expected):
    assert clean_caption(reply) == expected


def test_consensus_picks_the_caption_the_others_agree_with():
    captions = [
        "a black dog runs through the grass",
        "a dog running in the grass",
        "a man sits at a desk",
        "a black dog in the grass",
    ]
    assert consensus_caption(captions) in captions[:2] + captions[3:]
    assert consensus_caption(captions) != "a man sits at a desk"


def test_prompt_lists_every_retrieved_caption():
    captions = ["a dog runs", "a dog jumps", "a dog swims"]
    prompt = build_user_prompt(captions)
    for c in captions:
        assert f"- {c}" in prompt


class _Encoded(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None
    pad_token_id = 0

    def __init__(self, replies):
        self.replies = replies
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
        return messages[1]["content"]

    def __call__(self, prompts, return_tensors, padding):
        assert self.padding_side == "left"
        ids = torch.ones(len(prompts), 5, dtype=torch.long)
        return _Encoded(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, tokens, skip_special_tokens):
        # Only the generated part should arrive here, not the prompt.
        assert tokens.shape[1] == 3
        return self.replies[: len(tokens)]


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["do_sample"] is False
        return torch.cat([input_ids, torch.full((input_ids.size(0), 3), 7)], dim=1)


def _captioner(replies):
    return LLMCaptioner(model_name="fake", tokenizer=FakeTokenizer(replies), model=FakeModel())


def test_caption_batch_returns_one_cleaned_caption_per_image():
    captioner = _captioner(['"A dog runs on the grass."', "Two kids swim.\nextra"])
    out = captioner.caption_batch([["a dog runs"], ["kids swim"]])
    assert out == ["a dog runs on the grass", "two kids swim"]


def _tiny_dataset(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    rows = ["image,caption"]
    captions = {
        "a.jpg": ["a dog runs on grass", "a dog in a field"],
        "b.jpg": ["a dog plays outside", "a brown dog on grass"],
        "c.jpg": ["a man rides a bike", "a cyclist on a road"],
        "t1.jpg": ["a dog on the grass", "a dog running outside"],
    }
    for image_id, caps in captions.items():
        rows += [f"{image_id},{c}" for c in caps]
    (raw / "captions.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    split = {"train": ["a.jpg", "b.jpg", "c.jpg"], "val": [], "test": ["t1.jpg"]}
    (raw / "karpathy_split.json").write_text(json.dumps(split), encoding="utf-8")

    features = tmp_path / "features"
    features.mkdir()
    train = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    np.save(features / "train_pooled.npy", train)
    (features / "train_ids.json").write_text(json.dumps(split["train"]))
    np.save(features / "test_pooled.npy", np.array([[1.0, 0.05]], dtype=np.float32))
    (features / "test_ids.json").write_text(json.dumps(split["test"]))
    return raw, features


def test_evaluate_rag_llm_end_to_end_with_a_stand_in_model(tmp_path):
    pytest.importorskip("pycocoevalcap")
    raw, features = _tiny_dataset(tmp_path)
    result = evaluate_rag_llm(
        raw,
        features,
        tmp_path / "index",
        split="test",
        k=2,
        per_neighbor=1,
        batch_size=4,
        captioner=_captioner(["A dog runs on the grass."]),
    )
    sample = result["sample_predictions"]["t1.jpg"]
    # The test image sits next to the two dog photos, not the cyclist.
    assert all("dog" in c for c in sample["retrieved"])
    assert sample["rag_llm"] == "a dog runs on the grass"
    assert result["num_images"] == 1
    assert 0.0 <= result["rag_llm"]["bleu1"] <= 1.0
    assert "cider" in result["retrieval_only"]
