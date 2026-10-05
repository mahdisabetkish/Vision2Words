"""Push the two decoder checkpoints + vocab to a Hugging Face model repo.

Does NOT run automatically as part of any other command -- call it by
hand, deliberately, when you've decided to publish. Reads the token from
the environment (HF_TOKEN) or falls back to the credential cached by
`huggingface-cli login`; never hardcode a token in this file or anywhere
else.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def push_model(
    repo_id: str,
    lstm_attention_ckpt: Path,
    transformer_ckpt: Path,
    vocab_path: Path,
    model_card_path: Path,
    private: bool = False,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(lstm_attention_ckpt, tmp / "lstm_attention.pt")
        shutil.copy(transformer_ckpt, tmp / "transformer.pt")
        shutil.copy(vocab_path, tmp / "vocab.json")
        shutil.copy(model_card_path, tmp / "README.md")

        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="model")

    print(f"pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/vision2words-decoders")
    parser.add_argument("--lstm-attention-checkpoint", default="checkpoints/lstm_attention/best.pt")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer/best.pt")
    parser.add_argument("--vocab", default="data/features/vocab.json")
    parser.add_argument("--model-card", default="app/model_card.md")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    push_model(
        args.repo_id,
        Path(args.lstm_attention_checkpoint),
        Path(args.transformer_checkpoint),
        Path(args.vocab),
        Path(args.model_card),
        args.private,
    )


if __name__ == "__main__":
    main()
