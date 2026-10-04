"""Greedy and beam-search decoding, shared by every decoder.

Every decoder in this package exposes the same ``forward(features,
input_tokens) -> logits`` method, so decoding doesn't need to know which
architecture it's driving: each step just re-runs ``forward`` on the
sequence generated so far and reads off the last position's logits. That
costs an extra O(L) recomputation compared to hand-written recurrent state
for the LSTM decoders, but at caption-length sequences (well under 64
tokens) the difference is milliseconds, and it means greedy decoding and
beam search are written once instead of once per architecture -- which
matters more for a comparison between architectures than raw decode speed
does.
"""

from __future__ import annotations

import torch

from vision2words.models.base import CaptionDecoder


@torch.no_grad()
def greedy_decode(
    model: CaptionDecoder,
    features: torch.Tensor,
    start_id: int,
    end_id: int,
    max_len: int = 20,
) -> torch.Tensor:
    """Batched greedy decoding. Returns (B, <=max_len+1) token ids, including
    the leading start_id.
    """
    model.eval()
    batch_size = features.size(0)
    device = features.device
    sequences = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(features, sequences)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=1)
        finished |= next_token.squeeze(1) == end_id
        if finished.all():
            break

    return sequences


@torch.no_grad()
def beam_search(
    model: CaptionDecoder,
    features: torch.Tensor,
    start_id: int,
    end_id: int,
    beam_size: int = 3,
    max_len: int = 20,
) -> torch.Tensor:
    """Beam search for a single image (features must have batch size 1).

    Candidates are scored by length-normalized log-probability so a beam
    that ends early doesn't automatically win just for being short. Returns
    the single best sequence as a (1, L) tensor.
    """
    if features.size(0) != 1:
        raise ValueError("beam_search decodes one image at a time; call it in a loop over the batch")
    model.eval()
    device = features.device
    vocab_size_hint = None

    sequences = torch.full((1, 1), start_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    finished: list[tuple[float, torch.Tensor]] = []

    for _ in range(max_len):
        num_beams = sequences.size(0)
        beam_features = features.expand(num_beams, *features.shape[1:])
        logits = model(beam_features, sequences)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        vocab_size_hint = log_probs.size(-1)

        candidate_scores = (scores.unsqueeze(1) + log_probs).flatten()
        k = min(beam_size, candidate_scores.numel())
        top_scores, top_indices = candidate_scores.topk(k)
        beam_indices = top_indices // vocab_size_hint
        token_indices = top_indices % vocab_size_hint

        new_sequences = torch.cat([sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)
        is_end = token_indices == end_id

        for seq, score, ended in zip(new_sequences, top_scores, is_end):
            if ended:
                finished.append((score.item() / seq.size(0), seq))

        active = ~is_end
        if active.sum() == 0:
            break
        sequences = new_sequences[active]
        scores = top_scores[active]

    if not finished:
        best_idx = int(scores.argmax().item())
        finished.append((scores[best_idx].item() / sequences.size(1), sequences[best_idx]))

    finished.sort(key=lambda pair: pair[0], reverse=True)
    return finished[0][1].unsqueeze(0)
