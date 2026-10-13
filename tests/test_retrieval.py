"""The shared retriever: search, the self-exclusion leakage guard, persistence,
and turning neighbours into decoder context.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

pytest.importorskip("faiss")

from vision2words.data.vocabulary import Vocabulary  # noqa: E402
from vision2words.retrieval.context import RetrievedContext, neighbors_for_split  # noqa: E402
from vision2words.retrieval.index import CaptionRetriever  # noqa: E402

IDS = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
CAPTIONS = {
    "a.jpg": ["a dog runs on grass", "a brown dog running"],
    "b.jpg": ["a dog plays in a field", "dog on the grass"],
    "c.jpg": ["a man rides a bike", "a cyclist on a road"],
    "d.jpg": ["a child swims in a pool", "a kid in the water"],
}


def _vectors() -> np.ndarray:
    # a and b point the same way (two dog photos), c and d elsewhere.
    return np.array(
        [[1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]], dtype=np.float32
    )


@pytest.fixture
def retriever() -> CaptionRetriever:
    return CaptionRetriever.from_vectors(_vectors(), IDS, CAPTIONS)


def test_nearest_neighbour_of_an_indexed_image_is_itself(retriever):
    hits = retriever.search(_vectors()[:1], k=2)[0]
    assert [h.image_id for h in hits] == ["a.jpg", "b.jpg"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_excluding_the_query_id_removes_self_and_still_returns_k(retriever):
    hits = retriever.search(_vectors()[:1], k=2, exclude_ids=["a.jpg"])[0]
    ids = [h.image_id for h in hits]
    assert "a.jpg" not in ids
    assert ids[0] == "b.jpg"
    assert len(ids) == 2


def test_scores_are_cosine_similarities_regardless_of_vector_length(retriever):
    scaled = _vectors()[:1] * 50.0
    hits = retriever.search(scaled, k=1)[0]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_save_and_load_round_trip(retriever, tmp_path):
    retriever.save(tmp_path / "index")
    loaded = CaptionRetriever.load(tmp_path / "index")
    assert loaded.image_ids == IDS
    assert loaded.captions == CAPTIONS
    original = retriever.search(_vectors(), k=3)
    reloaded = loaded.search(_vectors(), k=3)
    assert [[h.image_id for h in row] for row in original] == [
        [h.image_id for h in row] for row in reloaded
    ]


def test_neighbors_for_split_never_returns_the_image_itself(tmp_path):
    np.save(tmp_path / "train_pooled.npy", _vectors())
    (tmp_path / "train_ids.json").write_text(json.dumps(IDS))
    retriever = CaptionRetriever.build(tmp_path, CAPTIONS)

    neighbors = neighbors_for_split(retriever, tmp_path, "train", k=2)
    for image_id, found in neighbors.items():
        assert image_id not in found
        assert len(found) == 2


def _context(retriever, **kwargs) -> RetrievedContext:
    vocab = Vocabulary.build([c for caps in CAPTIONS.values() for c in caps], min_freq=1)
    return RetrievedContext(retriever, vocab, **kwargs)


def test_deterministic_caption_choice_takes_the_first_captions(retriever):
    context = _context(retriever, k=2, per_neighbor=1)
    assert context.captions_for(["a.jpg", "b.jpg"]) == [
        "a dog runs on grass",
        "a dog plays in a field",
    ]


def test_sampled_captions_come_from_the_neighbours(retriever):
    context = _context(retriever, k=2, per_neighbor=1)
    chosen = context.captions_for(["c.jpg", "d.jpg"], rng=random.Random(0))
    assert chosen[0] in CAPTIONS["c.jpg"]
    assert chosen[1] in CAPTIONS["d.jpg"]


def test_encode_wraps_each_caption_and_truncates_keeping_end(retriever):
    context = _context(retriever, k=2, max_caption_tokens=4)
    vocab = context.vocab
    ids = context.encode(["a dog runs on grass", "a cyclist on a road"]).tolist()

    # Each caption: <START> + 2 words + <END> after truncating to 4 tokens.
    assert len(ids) == 8
    assert ids[0] == vocab.start_id and ids[3] == vocab.end_id
    assert ids[4] == vocab.start_id and ids[7] == vocab.end_id
