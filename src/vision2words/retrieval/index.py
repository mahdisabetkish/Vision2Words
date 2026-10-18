"""Nearest-neighbour retrieval over the training images, shared by both
retrieval-augmented captioners.

The datastore is the training split and nothing else: one L2-normalised
1280-d pooled EfficientNet-B0 vector per training image, in a FAISS
inner-product index, so a search returns the training images with the
highest cosine similarity to the query. Each hit carries that image's five
reference captions, which is what the decoders actually consume.

Two choices here matter more than they look:

- Only training images go in the index. Validation and test images are
  queries, never neighbours, so no evaluation caption can leak into the
  context a model is scored on.
- A training image must not retrieve itself. Its own captions are the exact
  targets the decoder is trained to produce, so leaving it in turns
  "retrieval-augmented" into "copy the answer", which inflates training
  scores and teaches nothing that transfers to test images. ``search`` takes
  the query's own id and drops it.

An exact flat index is the right size here: 6,000 vectors fit in 30 MB and
a brute-force search takes well under a millisecond. Approximate indexes
(IVF, HNSW) only start to pay off orders of magnitude beyond that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_FILE = "train_pooled.faiss"
IDS_FILE = "train_ids.json"
CAPTIONS_FILE = "train_captions.json"


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "retrieval needs FAISS: pip install -e '.[rag]' (installs faiss-cpu)"
        ) from exc
    return faiss


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class Neighbor:
    image_id: str
    score: float  # cosine similarity to the query


class CaptionRetriever:
    def __init__(self, index, image_ids: list[str], captions: dict[str, list[str]]) -> None:
        if index.ntotal != len(image_ids):
            raise ValueError(f"index holds {index.ntotal} vectors but {len(image_ids)} ids")
        self.index = index
        self.image_ids = image_ids
        self.captions = captions

    def __len__(self) -> int:
        return len(self.image_ids)

    @classmethod
    def from_vectors(
        cls, vectors: np.ndarray, image_ids: list[str], captions: dict[str, list[str]]
    ) -> CaptionRetriever:
        faiss = _faiss()
        vectors = _normalise(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        kept = {image_id: captions[image_id] for image_id in image_ids}
        return cls(index, list(image_ids), kept)

    @classmethod
    def build(cls, feature_dir: str | Path, captions: dict[str, list[str]]) -> CaptionRetriever:
        """Index the cached training-split pooled features (see data/features.py)."""
        feature_dir = Path(feature_dir)
        vectors = np.load(feature_dir / "train_pooled.npy")
        image_ids = json.loads((feature_dir / "train_ids.json").read_text(encoding="utf-8"))
        return cls.from_vectors(vectors, image_ids, captions)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(out_dir / INDEX_FILE))
        (out_dir / IDS_FILE).write_text(json.dumps(self.image_ids), encoding="utf-8")
        (out_dir / CAPTIONS_FILE).write_text(
            json.dumps(self.captions, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> CaptionRetriever:
        index_dir = Path(index_dir)
        index = _faiss().read_index(str(index_dir / INDEX_FILE))
        image_ids = json.loads((index_dir / IDS_FILE).read_text(encoding="utf-8"))
        captions = json.loads((index_dir / CAPTIONS_FILE).read_text(encoding="utf-8"))
        return cls(index, image_ids, captions)

    @classmethod
    def load_or_build(
        cls, index_dir: str | Path, feature_dir: str | Path, captions: dict[str, list[str]]
    ) -> CaptionRetriever:
        index_dir = Path(index_dir)
        if (index_dir / INDEX_FILE).exists():
            return cls.load(index_dir)
        retriever = cls.build(feature_dir, captions)
        retriever.save(index_dir)
        return retriever

    def search(
        self,
        queries: np.ndarray,
        k: int,
        exclude_ids: list[str | None] | None = None,
    ) -> list[list[Neighbor]]:
        """Return the k most similar training images for each query row.

        ``exclude_ids[i]``, when given, is dropped from row i's results -- pass
        each query's own image id so a training image can't retrieve itself.
        """
        queries = _normalise(np.atleast_2d(queries))
        if exclude_ids is not None and len(exclude_ids) != len(queries):
            raise ValueError("exclude_ids needs one entry per query")

        # One extra hit per query covers the case where the excluded image is
        # among the top k; without it that row would come back with k - 1.
        fetch = min(k + 1, len(self.image_ids))
        scores, rows = self.index.search(queries, fetch)

        results = []
        for i in range(len(queries)):
            excluded = exclude_ids[i] if exclude_ids is not None else None
            hits = [
                Neighbor(self.image_ids[row], float(score))
                for score, row in zip(scores[i], rows[i], strict=True)
                if row >= 0 and self.image_ids[row] != excluded
            ]
            results.append(hits[:k])
        return results
