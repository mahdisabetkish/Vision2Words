"""Gradio Space: upload a photograph, see captions from both Phase 2
decoders (attention-LSTM and Transformer) side by side, plus a look at
what the attention decoder was "looking at" for each word it wrote.

Weights are never stored in this repo -- they're pulled from a Hugging
Face model repo via hf_hub_download and cached by huggingface_hub itself,
so a Space restart doesn't mean a fresh download every time. CPU only,
since that's what the free Spaces tier gives you; both decoders are small
enough that a single caption takes well under a second.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.models import build_decoder, feature_mode_for
from vision2words.models.encoder import EncoderCNN

# Set once the trained weights are actually pushed to the Hub (see the root
# README's Hugging Face section) -- never hardcode a token here, and this
# repo id is the only thing that needs to change to point at a different
# model repo.
MODEL_REPO = os.environ.get("V2W_MODEL_REPO", "vision2words/vision2words-decoders")
DEVICE = "cpu"
MAX_LEN = 20

_state: dict = {}


def _download(filename: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


def _load_decoder(filename: str, vocab: Vocabulary):
    checkpoint = torch.load(_download(filename), map_location=DEVICE, weights_only=False)
    model = build_decoder(checkpoint["model_config"], vocab_size=len(vocab), pad_id=vocab.pad_id)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(DEVICE).eval(), checkpoint["model_config"]["type"]


def get_state() -> dict:
    if _state:
        return _state
    vocab = Vocabulary.load(_download("vocab.json"))
    encoder = EncoderCNN(fine_tune=False).to(DEVICE).eval()
    lstm_attention, _ = _load_decoder("lstm_attention.pt", vocab)
    transformer, _ = _load_decoder("transformer.pt", vocab)
    _state.update(
        vocab=vocab,
        encoder=encoder,
        lstm_attention=lstm_attention,
        transformer=transformer,
        transform=EncoderCNN.preprocess(),
    )
    return _state


def _specimen_card(label: str, color_class: str, caption: str, meta: str) -> str:
    return f"""
    <div class="v2w-specimen {color_class}">
      <div class="v2w-specimen-label"><span class="v2w-dot"></span>{label}</div>
      <div class="v2w-caption-text">{caption}</div>
      <div class="v2w-specimen-meta">{meta}</div>
    </div>
    """


def _attention_filmstrip(
    model, features: torch.Tensor, image: Image.Image, vocab: Vocabulary
) -> Image.Image:
    tokens, attn_maps = model.generate_with_attention(
        features, vocab.start_id, vocab.end_id, max_len=MAX_LEN
    )
    words = [vocab.idx2word.get(t, "<unk>") for t in tokens[1:]]
    words = [w for w in words if w != "<END>"][:8] or ["<end>"]
    n = len(words)

    base = image.convert("RGB").resize((200, 200))
    base_arr = np.asarray(base) / 255.0

    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.0), facecolor="#1a1816")
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        attn = attn_maps[i].reshape(7, 7).cpu().numpy()
        attn = attn / (attn.max() + 1e-8)
        attn_img = Image.fromarray((attn * 255).astype(np.uint8)).resize((200, 200), Image.BILINEAR)
        ax.imshow(base_arr)
        ax.imshow(np.asarray(attn_img) / 255.0, cmap="inferno", alpha=0.5)
        ax.set_title(words[i], color="#ede9e0", fontsize=11, fontfamily="monospace")
        ax.axis("off")
    fig.patch.set_facecolor("#1a1816")
    fig.tight_layout(pad=0.6)

    fig.canvas.draw()
    out = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert(
        "RGB"
    )
    plt.close(fig)
    return out


def caption_image(image: Image.Image | None, beam_size: int):
    if image is None:
        placeholder = _specimen_card(
            "attention-lstm", "v2w-amber", "upload a photograph to begin", ""
        )
        return (
            placeholder,
            _specimen_card("transformer", "v2w-teal", "upload a photograph to begin", ""),
            None,
        )

    state = get_state()
    vocab = state["vocab"]
    tensor = state["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pooled, spatial = state["encoder"](tensor)

        start = time.perf_counter()
        attn_features = spatial if feature_mode_for("lstm_attention") == "spatial" else pooled
        if beam_size > 1:
            attn_tokens = beam_search(
                state["lstm_attention"],
                attn_features,
                vocab.start_id,
                vocab.end_id,
                beam_size,
                MAX_LEN,
            )
        else:
            attn_tokens = greedy_decode(
                state["lstm_attention"], attn_features, vocab.start_id, vocab.end_id, MAX_LEN
            )
        attn_ms = (time.perf_counter() - start) * 1000
        attn_caption = vocab.decode(attn_tokens[0].tolist())

        start = time.perf_counter()
        tf_features = spatial if feature_mode_for("transformer") == "spatial" else pooled
        if beam_size > 1:
            tf_tokens = beam_search(
                state["transformer"], tf_features, vocab.start_id, vocab.end_id, beam_size, MAX_LEN
            )
        else:
            tf_tokens = greedy_decode(
                state["transformer"], tf_features, vocab.start_id, vocab.end_id, MAX_LEN
            )
        tf_ms = (time.perf_counter() - start) * 1000
        tf_caption = vocab.decode(tf_tokens[0].tolist())

        filmstrip = _attention_filmstrip(state["lstm_attention"], spatial, image, vocab)

    decode_label = "greedy" if beam_size <= 1 else f"beam k={beam_size}"
    attn_card = _specimen_card(
        "attention-lstm",
        "v2w-amber",
        attn_caption,
        f"{decode_label} &middot; {attn_ms:.0f} ms &middot; CPU",
    )
    tf_card = _specimen_card(
        "transformer",
        "v2w-teal",
        tf_caption,
        f"{decode_label} &middot; {tf_ms:.0f} ms &middot; CPU",
    )
    return attn_card, tf_card, filmstrip


MASTHEAD = """
<div id="v2w-masthead">
  <div class="v2w-kicker">Flickr8k &middot; EfficientNet-B0 &middot; two decoders</div>
  <h1>Vision2Words</h1>
  <div class="v2w-sub">
    Upload a photograph and watch an attention-LSTM and a Transformer decoder
    each write their own caption for it, independently, from the same frozen
    EfficientNet-B0 features. The attention tab shows which part of the image
    the LSTM decoder was looking at for each word.
  </div>
</div>
"""

with open(Path(__file__).parent / "style.css", encoding="utf-8") as f:
    CSS = f.read()

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=gr.themes.GoogleFont("IBM Plex Sans"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="#efeee7",
    body_background_fill_dark="#1a1816",
    block_background_fill="#ffffff",
    block_background_fill_dark="#242220",
    block_border_color="#dedbd1",
    block_border_color_dark="#38352f",
    button_primary_background_fill="#b8752a",
    button_primary_background_fill_hover="#9c6322",
    button_primary_text_color="#ffffff",
)

EXAMPLES_DIR = Path(__file__).parent / "examples"

with gr.Blocks(theme=THEME, css=CSS, title="Vision2Words") as demo:
    gr.HTML(MASTHEAD)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            image_input = gr.Image(type="pil", label="Photograph", elem_id="v2w-upload", height=320)
            beam_slider = gr.Slider(1, 5, value=3, step=1, label="Beam size (1 = greedy)")
            run_button = gr.Button("Write captions", variant="primary")
            if EXAMPLES_DIR.is_dir() and any(EXAMPLES_DIR.iterdir()):
                gr.Examples(
                    examples=[[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.jpg"))],
                    inputs=image_input,
                    label="Example photographs",
                )

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("Captions"):
                    with gr.Row():
                        attn_card = gr.HTML(
                            _specimen_card("attention-lstm", "v2w-amber", "&hellip;", "")
                        )
                        tf_card = gr.HTML(_specimen_card("transformer", "v2w-teal", "&hellip;", ""))
                with gr.Tab("Attention"):
                    gr.HTML('<div class="v2w-section-label">where the attention-LSTM looked</div>')
                    filmstrip = gr.Image(label=None, show_label=False, elem_id="v2w-filmstrip")

    run_button.click(
        fn=caption_image, inputs=[image_input, beam_slider], outputs=[attn_card, tf_card, filmstrip]
    )
    image_input.change(
        fn=caption_image, inputs=[image_input, beam_slider], outputs=[attn_card, tf_card, filmstrip]
    )


if __name__ == "__main__":
    demo.launch()
