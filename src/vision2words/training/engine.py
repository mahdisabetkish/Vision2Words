"""One epoch of training and one pass of validation.

The core fix this module exists to enforce: the decoder is trained to
predict the *next* token, not the current one. Given a padded batch of
token ids ``[<START>, w1, w2, ..., wn, <END>, <PAD>, ...]``, the input is
everything but the last token and the target is everything but the first --
so at position i the model sees tokens[:i+1] (via the recurrent state) and
is scored on tokens[i+1]. The previous version of this code fed the whole
sequence in as both input and target, which trains the model to echo back
what it was just shown instead of predicting what comes next.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vision2words.models.base import CaptionDecoder


def shift_for_teacher_forcing(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tokens[:, :-1], tokens[:, 1:]


def run_epoch(
    model: CaptionDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Runs one full pass over ``loader``. Trains if ``optimizer`` is given,
    otherwise evaluates in no-grad mode. Returns the mean per-token loss.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for features, tokens in loader:
            features = features.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            input_tokens, targets = shift_for_teacher_forcing(tokens)

            logits = model(features, input_tokens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_tokens = (targets != criterion.ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)
