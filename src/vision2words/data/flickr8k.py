"""Flickr8k access: download the dataset if it isn't already on disk, and
give the rest of the pipeline a single, source-agnostic way to read captions
and the train/val/test split.

The dataset is never committed to the repository. If ``raw_dir`` already
looks complete (captions.txt + images/ + karpathy_split.json), it's used as
is; this is what happens when someone points the pipeline at an existing
local copy of Flickr8k. Otherwise the data is pulled from the ``jxie/flickr8k``
mirror on the Hugging Face Hub, which packages Flickr8k with the standard
Karpathy-style 6000/1000/1000 train/val/test partition already applied, and
is written out locally in the plain (images/ + captions.txt) layout so the
rest of the code never has to know where the data came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

HF_REPO_ID = "jxie/flickr8k"
HF_FILES = {
    "train": [
        "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
        "data/train-00001-of-00002-2173151d8cd6c7fb.parquet",
    ],
    "val": ["data/validation-00000-of-00001-7025a2b596f14b7b.parquet"],
    "test": ["data/test-00000-of-00001-42a2661d12c73e48.parquet"],
}
CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]
SPLITS = ("train", "val", "test")


def _is_ready(raw_dir: Path) -> bool:
    return (
        (raw_dir / "captions.txt").exists()
        and (raw_dir / "images").is_dir()
        and (raw_dir / "karpathy_split.json").exists()
    )


def download_flickr8k(raw_dir: Path) -> None:
    """Fetch Flickr8k from the Hugging Face mirror and write it to raw_dir
    in the plain images/ + captions.txt layout, plus a karpathy_split.json
    recording which image belongs to which split.
    """
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    split_ids: dict[str, list[str]] = {}
    caption_rows = ["image,caption"]

    for split, files in HF_FILES.items():
        ids_this_split: list[str] = []
        for filename in files:
            local_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename)
            df = pd.read_parquet(local_path)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}:{Path(filename).name}"):
                image = row["image"]
                image_id = image["path"]
                out_path = images_dir / image_id
                if not out_path.exists():
                    out_path.write_bytes(image["bytes"])
                for col in CAPTION_COLUMNS:
                    caption = str(row[col]).strip()
                    if caption:
                        caption_rows.append(f"{image_id},{caption}")
                ids_this_split.append(image_id)
        split_ids[split] = ids_this_split

    (raw_dir / "captions.txt").write_text("\n".join(caption_rows) + "\n", encoding="utf-8")
    (raw_dir / "karpathy_split.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")


def ensure_data(raw_dir: str | Path) -> Path:
    """Return raw_dir, downloading the dataset into it first if needed."""
    raw_dir = Path(raw_dir)
    if not _is_ready(raw_dir):
        download_flickr8k(raw_dir)
    return raw_dir


def load_captions(raw_dir: str | Path) -> dict[str, list[str]]:
    """Read captions.txt into {image_id: [caption, ...]}."""
    captions: dict[str, list[str]] = {}
    with open(Path(raw_dir) / "captions.txt", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            image_id, caption = line.split(",", 1)
            captions.setdefault(image_id, []).append(caption)
    return captions


def load_split(raw_dir: str | Path) -> dict[str, list[str]]:
    """Read karpathy_split.json into {"train"|"val"|"test": [image_id, ...]}."""
    with open(Path(raw_dir) / "karpathy_split.json", encoding="utf-8") as f:
        return json.load(f)
