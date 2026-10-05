"""BLEU-1..4 and CIDEr, the two metrics requested for the Phase 2 comparison.

pycocoevalcap's scorers are used directly on our own tokenization (lowercase,
punctuation stripped -- the same rule used everywhere else in this project)
instead of pycocoevalcap's usual PTBTokenizer step, which shells out to a
Java binary. That keeps this working on a machine with no JRE installed,
at the cost of not matching the official COCO evaluation server's
tokenization exactly; for comparing two models trained on the same data
with the same tokenizer, that mismatch doesn't matter. METEOR and SPICE are
skipped for the same reason (they also depend on Java/Stanford CoreNLP).
"""

from __future__ import annotations

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from vision2words.data.vocabulary import tokenize


def _retokenize(texts_by_id: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        image_id: [" ".join(tokenize(t)) for t in texts] for image_id, texts in texts_by_id.items()
    }


def compute_bleu_cider(
    predictions: dict[str, str], references: dict[str, list[str]]
) -> dict[str, float]:
    """predictions: {image_id: caption}. references: {image_id: [caption, ...]}.
    Returns bleu1..bleu4 and cider, corpus-level scores.
    """
    ids = sorted(predictions)
    missing = [i for i in ids if i not in references]
    if missing:
        raise KeyError(f"no reference captions for {len(missing)} ids, e.g. {missing[:3]}")

    res = _retokenize({image_id: [predictions[image_id]] for image_id in ids})
    gts = _retokenize({image_id: references[image_id] for image_id in ids})

    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)

    return {
        "bleu1": bleu_scores[0],
        "bleu2": bleu_scores[1],
        "bleu3": bleu_scores[2],
        "bleu4": bleu_scores[3],
        "cider": cider_score,
    }
