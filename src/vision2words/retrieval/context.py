"""Turn retrieved neighbours into context a captioner can read.

Both captioners see the neighbours' captions, never the neighbour images:
the retrieval-augmented decoder reads them as token ids next to the image
features, and the language-model captioner reads them as plain text in its
prompt.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from vision2words.data.vocabulary import Vocabulary
from vision2words.models.inputs import RetrievalAugmentedInput
from vision2words.retrieval.index import CaptionRetriever, Neighbor


def neighbors_for_split(
    retriever: CaptionRetriever, feature_dir: str | Path, split: str, k: int
) -> dict[str, list[str]]:
    """Retrieve k neighbours for every image in a cached split, in one batched
    search. Each image's own id is excluded, which matters for the training
    split (it is in the index) and is a no-op for val/test (they aren't).
    """
    feature_dir = Path(feature_dir)
    vectors = np.load(feature_dir / f"{split}_pooled.npy")
    ids = json.loads((feature_dir / f"{split}_ids.json").read_text(encoding="utf-8"))
    hits = retriever.search(vectors, k, exclude_ids=ids)
    return {image_id: [n.image_id for n in row] for image_id, row in zip(ids, hits, strict=True)}


class RetrievedContext:
    """Pick captions from retrieved neighbours and encode them for the decoder."""

    def __init__(
        self,
        retriever: CaptionRetriever,
        vocab: Vocabulary,
        k: int = 4,
        max_caption_tokens: int = 20,
        per_neighbor: int = 1,
    ) -> None:
        self.retriever = retriever
        self.vocab = vocab
        self.k = k
        self.max_caption_tokens = max_caption_tokens
        self.per_neighbor = per_neighbor

    @classmethod
    def from_config(
        cls,
        retrieval_config: dict,
        vocab: Vocabulary,
        feature_dir: str | Path | None = None,
        captions: dict[str, list[str]] | None = None,
    ) -> RetrievedContext:
        """Rebuild the context a checkpoint was trained with. The index is
        loaded from ``index_dir``, or built from the cached training features
        if it doesn't exist yet and ``feature_dir``/``captions`` are given.
        """
        index_dir = retrieval_config["index_dir"]
        if feature_dir is not None and captions is not None:
            retriever = CaptionRetriever.load_or_build(index_dir, feature_dir, captions)
        else:
            retriever = CaptionRetriever.load(index_dir)
        return cls(
            retriever,
            vocab,
            retrieval_config["k"],
            retrieval_config["max_caption_tokens"],
            retrieval_config.get("per_neighbor", 1),
        )

    def captions_for(
        self,
        neighbor_ids: list[str],
        per_neighbor: int | None = None,
        rng: random.Random | None = None,
    ) -> list[str]:
        """One or more captions from each neighbour.

        With ``rng`` (training) the captions are sampled, so the decoder sees a
        different phrasing of the same neighbours from epoch to epoch and can't
        just memorise one string per neighbour. Without it (evaluation,
        inference) the first ``per_neighbor`` captions are used, so results are
        deterministic.
        """
        per_neighbor = per_neighbor or self.per_neighbor
        chosen = []
        for image_id in neighbor_ids[: self.k]:
            captions = self.retriever.captions[image_id]
            n = min(per_neighbor, len(captions))
            chosen.extend(rng.sample(captions, n) if rng is not None else captions[:n])
        return chosen

    def encode(self, captions: list[str]) -> torch.Tensor:
        """Concatenate the captions' token ids, each one wrapped in its own
        <START>/<END> so the decoder can tell where one caption stops and the
        next begins. A caption longer than ``max_caption_tokens`` is cut short
        but keeps its <END>.
        """
        ids: list[int] = []
        for caption in captions:
            tokens = self.vocab.encode(caption)
            if len(tokens) > self.max_caption_tokens:
                tokens = tokens[: self.max_caption_tokens - 1] + [self.vocab.end_id]
            ids.extend(tokens)
        if not ids:
            ids = [self.vocab.start_id, self.vocab.end_id]
        return torch.tensor(ids, dtype=torch.long)

    def neighbors(self, pooled: np.ndarray, exclude_id: str | None = None) -> list[Neighbor]:
        """Neighbours for a single query (e.g. a new image at inference time)."""
        return self.retriever.search(pooled, self.k, exclude_ids=[exclude_id])[0]

    def decoder_input(
        self, image: torch.Tensor, neighbor_ids: list[str]
    ) -> RetrievalAugmentedInput:
        """Batch-of-one decoder input for evaluation or inference: the image's
        spatial features (1, 49, F) plus its neighbours' captions, chosen
        deterministically.
        """
        retrieved = self.encode(self.captions_for(neighbor_ids)).unsqueeze(0)
        return RetrievalAugmentedInput(image, retrieved.to(image.device))
