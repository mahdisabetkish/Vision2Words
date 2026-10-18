"""The retrieval-augmented Transformer decoder: shapes, that padding in the
retrieved captions is really ignored, that the retrieved captions really
reach the output, and that it works through the shared decoding and
training code via RetrievalAugmentedInput.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.models import build_decoder, feature_mode_for, uses_retrieval
from vision2words.models.inputs import RetrievalAugmentedInput
from vision2words.retrieval.dataset import collate_retrieval
from vision2words.training.engine import run_epoch
from vision2words.utils.seed import set_seed

VOCAB_SIZE, PAD_ID, START_ID, END_ID = 30, 0, 1, 2
FEATURE_SIZE, TOKENS = 1280, 49
CONFIG = {
    "type": "transformer_rag",
    "feature_size": FEATURE_SIZE,
    "d_model": 32,
    "num_layers": 2,
    "num_heads": 4,
    "ff_size": 64,
    "max_len": 24,
    "max_retrieved_len": 40,
    "text_encoder_layers": 1,
    "dropout": 0.0,
}


def _model():
    return build_decoder(CONFIG, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).eval()


def _input(batch: int = 2, retrieved_len: int = 12) -> RetrievalAugmentedInput:
    return RetrievalAugmentedInput(
        torch.randn(batch, TOKENS, FEATURE_SIZE),
        torch.randint(3, VOCAB_SIZE, (batch, retrieved_len)),
    )


def test_registry_knows_the_rag_decoder():
    assert uses_retrieval("transformer_rag")
    assert not uses_retrieval("transformer")
    assert feature_mode_for("transformer_rag") == "spatial"


def test_forward_shape():
    logits = _model()(_input(), torch.randint(3, VOCAB_SIZE, (2, 7)))
    assert logits.shape == (2, 7, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_padding_appended_to_retrieved_captions_changes_nothing():
    model = _model()
    features = _input()
    padded = RetrievalAugmentedInput(
        features.image,
        torch.cat([features.retrieved, torch.full((2, 9), PAD_ID)], dim=1),
    )
    tokens = torch.randint(3, VOCAB_SIZE, (2, 6))
    with torch.no_grad():
        assert torch.allclose(model(features, tokens), model(padded, tokens), atol=1e-5)


def test_retrieved_captions_actually_reach_the_output():
    model = _model()
    features = _input()
    other = RetrievalAugmentedInput(features.image, torch.randint(3, VOCAB_SIZE, (2, 12)))
    tokens = torch.randint(3, VOCAB_SIZE, (2, 6))
    with torch.no_grad():
        assert not torch.allclose(model(features, tokens), model(other, tokens))


def test_greedy_and_beam_decoding_accept_retrieval_input():
    model = _model()
    single = _input(batch=1)
    greedy = greedy_decode(model, single, START_ID, END_ID, max_len=8)
    beam = beam_search(model, single, START_ID, END_ID, beam_size=3, max_len=8)
    assert greedy[0, 0].item() == START_ID and greedy.shape[1] <= 9
    assert beam[0, 0].item() == START_ID and beam.shape[1] <= 9


def test_collate_pads_retrieved_and_target_tokens_separately():
    batch = [
        (torch.randn(TOKENS, FEATURE_SIZE), torch.tensor([1, 5, 2]), torch.tensor([1, 7, 8, 2])),
        (torch.randn(TOKENS, FEATURE_SIZE), torch.tensor([1, 5, 6, 9, 2]), torch.tensor([1, 2])),
    ]
    features, tokens = collate_retrieval(batch, pad_id=PAD_ID)
    assert features.image.shape == (2, TOKENS, FEATURE_SIZE)
    assert features.retrieved.tolist() == [[1, 5, 2, 0, 0], [1, 5, 6, 9, 2]]
    assert tokens.tolist() == [[1, 7, 8, 2], [1, 2, 0, 0]]


def test_one_batch_overfit_through_the_shared_training_loop():
    set_seed(0)
    images = torch.randn(4, TOKENS, FEATURE_SIZE)
    retrieved = torch.randint(3, VOCAB_SIZE, (4, 10))
    targets = torch.randint(3, VOCAB_SIZE, (4, 6))
    batch = [(images[i], retrieved[i], targets[i]) for i in range(4)]
    loader = DataLoader(
        batch, batch_size=4, collate_fn=lambda b: collate_retrieval(b, pad_id=PAD_ID)
    )

    model = build_decoder(CONFIG, vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    device = torch.device("cpu")

    first = run_epoch(model, loader, criterion, device, optimizer)
    for _ in range(80):
        last = run_epoch(model, loader, criterion, device, optimizer)
    assert last < first * 0.1
