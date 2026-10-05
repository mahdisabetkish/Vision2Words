"""Push the Gradio app to a Hugging Face Space.

Does NOT run automatically as part of any other command -- call it by
hand, deliberately, when you've decided to publish. Reads the token from
the environment (HF_TOKEN) or falls back to the credential cached by
`huggingface-cli login`; never hardcode a token in this file or anywhere
else.

Vendors a copy of src/vision2words into the pushed folder so the Space is
self-contained (a Space is its own git repo, separate from this one, and
doesn't have access to this repo's src/ layout at runtime).
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent


def push_space(repo_id: str, model_repo_id: str, private: bool = False) -> None:
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=private
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        app_dir = REPO_ROOT / "app"
        for name in ("app.py", "requirements.txt", "style.css", "README.md"):
            shutil.copy(app_dir / name, tmp / name)
        shutil.copytree(app_dir / "examples", tmp / "examples")
        shutil.copytree(REPO_ROOT / "src" / "vision2words", tmp / "vision2words")

        app_py = (tmp / "app.py").read_text(encoding="utf-8")
        app_py, n_subs = re.subn(
            r'MODEL_REPO = os\.environ\.get\("V2W_MODEL_REPO", "[^"]+"\)',
            f'MODEL_REPO = os.environ.get("V2W_MODEL_REPO", "{model_repo_id}")',
            app_py,
        )
        # Fail loudly if app.py's MODEL_REPO line changes shape, rather than
        # publishing a Space that still points at the default repo.
        if n_subs != 1:
            raise RuntimeError("could not find the MODEL_REPO line in app/app.py")
        (tmp / "app.py").write_text(app_py, encoding="utf-8")

        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="space")

    print(f"pushed to https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/vision2words")
    parser.add_argument(
        "--model-repo-id", required=True, help="the model repo pushed by push_model_to_hub.py"
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    push_space(args.repo_id, args.model_repo_id, args.private)


if __name__ == "__main__":
    main()
