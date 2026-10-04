"""Train the LSTM caption decoder on cached EfficientNet-B0 features.

Everything a run needs -- data paths, model size, optimizer, how long to
train -- comes from a YAML config (see configs/) plus optional ``--set``
overrides, so a run is reproducible from the checkpoint directory alone: the
resolved config and the vocabulary are written there together with the
weights.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision2words.data.dataset import CaptionFeatureDataset, collate_captions
from vision2words.data.flickr8k import ensure_data, load_captions
from vision2words.data.vocabulary import Vocabulary
from vision2words.models.lstm_decoder import LSTMDecoder
from vision2words.training.engine import run_epoch
from vision2words.utils.config import load_config, save_yaml
from vision2words.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "config asks for device=cuda but torch.cuda.is_available() is False "
            "-- check the NVIDIA driver and that this torch build has CUDA support"
        )
    return torch.device(requested)


def _build_vocab(raw_dir: Path, feature_dir: Path, min_freq: int) -> Vocabulary:
    vocab_path = feature_dir / "vocab.json"
    if vocab_path.exists():
        return Vocabulary.load(vocab_path)

    captions = load_captions(raw_dir)
    from vision2words.data.flickr8k import load_split

    train_ids = set(load_split(raw_dir)["train"])
    train_captions = [cap for image_id, caps in captions.items() if image_id in train_ids for cap in caps]
    vocab = Vocabulary.build(train_captions, min_freq=min_freq)
    vocab.save(vocab_path)
    logger.info("built vocabulary: %d words (min_freq=%d) -> %s", len(vocab), min_freq, vocab_path)
    return vocab


def train(config_path: str, overrides: list[str] | None = None) -> Path:
    config = load_config(config_path, overrides)
    set_seed(config["train"]["seed"])

    raw_dir = ensure_data(config["data"]["raw_dir"])
    feature_dir = Path(config["data"]["feature_dir"])
    vocab = _build_vocab(raw_dir, feature_dir, config["data"]["min_word_freq"])
    captions = load_captions(raw_dir)

    device = _resolve_device(config["train"]["device"])

    collate = functools.partial(collate_captions, pad_id=vocab.pad_id)
    train_ds = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode="pooled")
    val_ds = CaptionFeatureDataset(feature_dir, "val", captions, vocab, feature_mode="pooled")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=config["train"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=config["train"].get("num_workers", 0),
    )

    model = LSTMDecoder(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        feature_size=config["model"]["feature_size"],
        embed_size=config["model"]["embed_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"].get("num_layers", 1),
        dropout=config["model"].get("dropout", 0.3),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    ckpt_dir = Path(config["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, ckpt_dir / "config.yaml")

    log_path = ckpt_dir / "training_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,epoch_seconds\n", encoding="utf-8")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = config["train"].get("early_stopping_patience", 5)

    for epoch in range(1, config["train"]["epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - start

        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f (%.1fs)",
            epoch,
            config["train"]["epochs"],
            train_loss,
            val_loss,
            elapsed,
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{elapsed:.1f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("early stopping: no val improvement for %d epochs", patience)
                break

    (ckpt_dir / "summary.json").write_text(
        json.dumps({"best_val_loss": best_val_loss, "epochs_ran": epoch}, indent=2),
        encoding="utf-8",
    )
    return ckpt_dir / "best.pt"
