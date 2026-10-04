"""Train any of the three caption decoders on cached EfficientNet-B0 features.

Everything a run needs -- data paths, which decoder, its size, the
optimizer, how long to train -- comes from a YAML config (see configs/)
plus optional ``--set`` overrides, so a run is reproducible from the
checkpoint directory alone: the resolved config and the vocabulary are
written there together with the weights.
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
from vision2words.data.flickr8k import ensure_data, load_captions, load_split
from vision2words.data.vocabulary import Vocabulary
from vision2words.models import build_decoder, feature_mode_for
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
    train_ids = set(load_split(raw_dir)["train"])
    train_captions = [
        cap for image_id, caps in captions.items() if image_id in train_ids for cap in caps
    ]
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
    feature_mode = feature_mode_for(config["model"]["type"])

    collate = functools.partial(collate_captions, pad_id=vocab.pad_id)
    train_ds = CaptionFeatureDataset(feature_dir, "train", captions, vocab, feature_mode=feature_mode)
    val_ds = CaptionFeatureDataset(feature_dir, "val", captions, vocab, feature_mode=feature_mode)
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

    model = build_decoder(config["model"], vocab_size=len(vocab), pad_id=vocab.pad_id).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model=%s params=%d feature_mode=%s", config["model"]["type"], num_params, feature_mode)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["train"].get("lr_patience", 2)
    )
    grad_clip_norm = config["train"].get("grad_clip_norm")

    ckpt_dir = Path(config["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, ckpt_dir / "config.yaml")

    log_path = ckpt_dir / "training_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,lr,epoch_seconds\n", encoding="utf-8")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = config["train"].get("early_stopping_patience", 5)
    epoch = 0

    for epoch in range(1, config["train"]["epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, grad_clip_norm)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f lr=%.2e (%.1fs)",
            epoch,
            config["train"]["epochs"],
            train_loss,
            val_loss,
            current_lr,
            elapsed,
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e},{elapsed:.1f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "num_params": num_params,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("early stopping: no val improvement for %d epochs", patience)
                break

    (ckpt_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_type": config["model"]["type"],
                "best_val_loss": best_val_loss,
                "epochs_ran": epoch,
                "num_params": num_params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ckpt_dir / "best.pt"
