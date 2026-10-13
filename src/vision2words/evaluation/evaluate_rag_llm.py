"""Evaluate retrieval + language-model captioning on a cached split.

Same test images, same references and same BLEU/CIDEr code as the trained
decoders (evaluation/evaluate.py), so its numbers sit in the same table.
Two systems are scored from one retrieval pass:

- ``retrieval_only``: the consensus caption among the retrieved ones, no
  generation (see llm_captioner.consensus_caption).
- ``rag_llm``: the language model writing a caption from those same captions.

The gap between them is what generation adds on top of retrieval; the gap to
the trained decoders is what looking at the image adds.
"""

from __future__ import annotations

import time
from pathlib import Path

from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.evaluation.metrics import compute_bleu_cider
from vision2words.retrieval.context import neighbors_for_split
from vision2words.retrieval.index import CaptionRetriever
from vision2words.retrieval.llm_captioner import (
    DEFAULT_MODEL,
    LLMCaptioner,
    consensus_caption,
    neighbour_captions,
)


def evaluate_rag_llm(
    raw_dir: str | Path,
    feature_dir: str | Path,
    index_dir: str | Path,
    split: str = "test",
    k: int = 5,
    per_neighbor: int = 2,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    batch_size: int = 8,
    limit: int | None = None,
    captioner: LLMCaptioner | None = None,
) -> dict:
    raw_dir = ensure_data(raw_dir)
    feature_dir = Path(feature_dir)
    captions = load_captions(raw_dir)
    retriever = CaptionRetriever.load_or_build(index_dir, feature_dir, captions)

    neighbors = neighbors_for_split(retriever, feature_dir, split, k)
    image_ids = sorted(neighbors)[:limit] if limit else sorted(neighbors)
    references = {image_id: captions[image_id] for image_id in image_ids}
    context = {
        image_id: neighbour_captions(retriever, neighbors[image_id], per_neighbor)
        for image_id in image_ids
    }

    retrieval_only = {image_id: consensus_caption(context[image_id]) for image_id in image_ids}

    captioner = captioner or LLMCaptioner(model_name, device)
    llm_preds: dict[str, str] = {}
    start = time.perf_counter()
    for i in range(0, len(image_ids), batch_size):
        batch = image_ids[i : i + batch_size]
        for image_id, caption in zip(
            batch, captioner.caption_batch([context[j] for j in batch]), strict=True
        ):
            llm_preds[image_id] = caption
    llm_ms = 1000 * (time.perf_counter() - start) / len(image_ids)

    return {
        "model_type": "rag_llm",
        "llm": captioner.model_name,
        "split": split,
        "num_images": len(image_ids),
        "retrieval": {"k": k, "captions_per_neighbor": per_neighbor, "index_size": len(retriever)},
        "retrieval_only": compute_bleu_cider(retrieval_only, references),
        "rag_llm": {
            **compute_bleu_cider(llm_preds, references),
            "latency_ms_per_image": llm_ms,
            "batch_size": batch_size,
            "device": device,
        },
        "sample_predictions": {
            image_id: {
                "retrieved": context[image_id],
                "retrieval_only": retrieval_only[image_id],
                "rag_llm": llm_preds[image_id],
            }
            for image_id in image_ids[:10]
        },
    }
