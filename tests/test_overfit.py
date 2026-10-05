"""One-batch overfit test: the standard sanity check that the training loop
is actually wired correctly end to end (correct target shifting, a loss
that's actually connected to the model's parameters, an optimizer step that
actually moves them). If this doesn't drive the loss to near zero, training
is broken somewhere -- this is the fastest test in the suite that would
catch it.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from vision2words.models.lstm_decoder import LSTMDecoder
from vision2words.training.engine import run_epoch
from vision2words.utils.seed import set_seed

VOCAB_SIZE = 12
PAD_ID = 0


def _single_batch_loader():
    set_seed(0)
    batch_size, seq_len, feature_size = 4, 6, 1280
    features = torch.randn(batch_size, feature_size)
    # Non-pad tokens only, so every position actually contributes to the loss.
    tokens = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len))
    dataset = TensorDataset(features, tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_one_batch_overfit_drives_loss_near_zero():
    set_seed(0)
    loader = _single_batch_loader()
    model = LSTMDecoder(
        vocab_size=VOCAB_SIZE, pad_id=PAD_ID, feature_size=1280, embed_size=32, hidden_size=64
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    first_loss = run_epoch(model, loader, criterion, device, optimizer)
    for _ in range(100):
        last_loss = run_epoch(model, loader, criterion, device, optimizer)

    assert last_loss < first_loss * 0.05
    assert last_loss < 0.1
