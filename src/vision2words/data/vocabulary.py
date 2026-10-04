"""Caption tokenization and the word <-> index vocabulary.

The vocabulary is built once from the training captions only (using the val
or test captions here would leak information about words the model will be
asked to produce at evaluation time) and then saved as JSON next to every
checkpoint, so a checkpoint is always self-describing: you don't need the
original dataset around just to decode what it predicts.
"""

from __future__ import annotations

import json
import string
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(caption: str) -> list[str]:
    """Lowercase, strip punctuation, and split on whitespace.

    Kept as a single shared function so vocabulary building and caption
    encoding can never drift apart and tokenize the same sentence two
    different ways.
    """
    return caption.lower().translate(_PUNCTUATION_TABLE).split()


class Vocabulary:
    def __init__(self, word2idx: dict[str, int], min_freq: int) -> None:
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.min_freq = min_freq

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def start_id(self) -> int:
        return self.word2idx[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.word2idx[END_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 5) -> Vocabulary:
        counts = Counter()
        for caption in captions:
            counts.update(tokenize(caption))
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for word in words:
            word2idx[word] = len(word2idx)
        return cls(word2idx, min_freq)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        ids = [self.word2idx.get(tok, self.unk_id) for tok in tokenize(caption)]
        if add_special:
            ids = [self.start_id, *ids, self.end_id]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if strip_special and word in (PAD_TOKEN, START_TOKEN, END_TOKEN):
                if word == END_TOKEN:
                    break
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"min_freq": self.min_freq, "word2idx": self.word2idx}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["word2idx"], payload["min_freq"])
