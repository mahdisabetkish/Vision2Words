"""Vocabulary building, encode/decode round-tripping, and persistence."""

from __future__ import annotations

from vision2words.data.vocabulary import PAD_TOKEN, Vocabulary, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("A Dog, running FAST!") == ["a", "dog", "running", "fast"]


def test_build_keeps_only_frequent_words():
    captions = [
        "a dog runs",  # "dog" and "runs" appear once each
        "a cat runs",  # "runs" appears twice total, still below min_freq=3
        "a cat sits",
    ]
    vocab = Vocabulary.build(captions, min_freq=2)
    # "a" appears 3 times, "cat" 2 times: both kept. "dog", "runs", "sits" appear
    # once or twice but below the threshold where stated -- check directly.
    assert "a" in vocab.word2idx
    assert "dog" not in vocab.word2idx  # appears once, below min_freq=2


def test_special_tokens_always_present_regardless_of_frequency():
    vocab = Vocabulary.build(["only one caption here"], min_freq=100)
    for special in ("<PAD>", "<START>", "<END>", "<UNK>"):
        assert special in vocab.word2idx
    assert len(vocab) == 4  # nothing else met min_freq=100


def test_encode_wraps_with_start_and_end():
    vocab = Vocabulary.build(["a dog runs"] * 5, min_freq=1)
    ids = vocab.encode("a dog runs")
    assert ids[0] == vocab.start_id
    assert ids[-1] == vocab.end_id
    assert len(ids) == 5  # start + 3 words + end


def test_encode_unknown_word_maps_to_unk():
    vocab = Vocabulary.build(["a dog runs"] * 5, min_freq=1)
    ids = vocab.encode("a dinosaur runs")
    assert vocab.unk_id in ids


def test_decode_stops_at_end_and_skips_special_tokens():
    vocab = Vocabulary.build(["a dog runs"] * 5, min_freq=1)
    ids = [
        vocab.start_id,
        vocab.word2idx["a"],
        vocab.word2idx["dog"],
        vocab.end_id,
        vocab.word2idx["runs"],
    ]
    assert vocab.decode(ids) == "a dog"  # stops at <END>, never reaches "runs"


def test_save_and_load_round_trip(tmp_path):
    vocab = Vocabulary.build(["a dog runs fast"] * 5, min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.word2idx == vocab.word2idx
    assert loaded.min_freq == vocab.min_freq
    assert loaded.pad_id == vocab.pad_id
    assert loaded.idx2word[loaded.word2idx["dog"]] == "dog"


def test_pad_is_index_zero_by_construction():
    # LSTMDecoder relies on this: nn.Embedding(padding_idx=pad_id) needs a
    # stable, known pad index, and padding tensors are filled with pad_id.
    vocab = Vocabulary.build(["anything"], min_freq=1)
    assert vocab.word2idx[PAD_TOKEN] == 0
