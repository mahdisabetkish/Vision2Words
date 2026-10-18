"""Gradio Space: upload a photograph, see captions from the attention-LSTM,
the Transformer and the retrieval-augmented Transformer side by side, a
look at what the attention decoder was "looking at" for each word it
wrote, and the training captions retrieval found for the photograph, which
a small language model can turn into a caption of its own.

Weights are never stored in this repo -- they're pulled from a Hugging
Face model repo via hf_hub_download and cached by huggingface_hub itself,
so a Space restart doesn't mean a fresh download every time. CPU only,
since that's what the free Spaces tier gives you; the decoders are small
enough that a single caption takes well under a second. The language model
is only loaded the first time someone asks for its caption.

The retrieval-augmented decoder and its index are optional: if the model
repo doesn't have them, the app still runs with the other two decoders.
"""

from __future__ import annotations

import html
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
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image

from vision2words.data.vocabulary import Vocabulary
from vision2words.evaluation.decoding import beam_search, greedy_decode
from vision2words.models import build_decoder, feature_mode_for
from vision2words.models.encoder import EncoderCNN
from vision2words.retrieval.context import RetrievedContext
from vision2words.retrieval.index import CAPTIONS_FILE, IDS_FILE, INDEX_FILE
from vision2words.retrieval.llm_captioner import DEFAULT_MODEL, LLMCaptioner, neighbour_captions

# Set once the trained weights are actually pushed to the Hub (see the root
# README's Hugging Face section) -- never hardcode a token here, and this
# repo id is the only thing that needs to change to point at a different
# model repo.
MODEL_REPO = os.environ.get("V2W_MODEL_REPO", "mahdisabetkish/vision2words-decoders")
LLM_MODEL = os.environ.get("V2W_LLM", DEFAULT_MODEL)
DEVICE = "cpu"
MAX_LEN = 20
# Same retrieval settings as configs/rag_llm.yaml: five neighbours, two
# captions from each go into the language model's prompt.
LLM_K, LLM_PER_NEIGHBOR = 5, 2

_state: dict = {}


def _download(filename: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


def _load_decoder(filename: str, vocab: Vocabulary):
    checkpoint = torch.load(_download(filename), map_location=DEVICE, weights_only=False)
    model = build_decoder(checkpoint["model_config"], vocab_size=len(vocab), pad_id=vocab.pad_id)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(DEVICE).eval(), checkpoint


def _load_retrieval(vocab: Vocabulary):
    """The retrieval-augmented decoder and the index it reads from, or None
    when the model repo doesn't have them.
    """
    try:
        index_dir = Path(_download(f"retrieval/{INDEX_FILE}")).parent
        for name in (IDS_FILE, CAPTIONS_FILE):
            _download(f"retrieval/{name}")
        decoder, checkpoint = _load_decoder("transformer_rag.pt", vocab)
    except EntryNotFoundError:
        return None
    config = {**checkpoint["retrieval_config"], "index_dir": index_dir}
    return decoder, RetrievedContext.from_config(config, vocab)


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
        rag=_load_retrieval(vocab),
        transform=EncoderCNN.preprocess(),
    )
    return _state


def get_llm() -> LLMCaptioner:
    if "llm" not in _state:
        _state["llm"] = LLMCaptioner(LLM_MODEL, DEVICE)
    return _state["llm"]


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


def _decode(model, features, beam_size: int, vocab: Vocabulary) -> tuple[str, float]:
    start = time.perf_counter()
    if beam_size > 1:
        tokens = beam_search(model, features, vocab.start_id, vocab.end_id, beam_size, MAX_LEN)
    else:
        tokens = greedy_decode(model, features, vocab.start_id, vocab.end_id, MAX_LEN)
    return vocab.decode(tokens[0].tolist()), (time.perf_counter() - start) * 1000


def _neighbour_list(hits, context: RetrievedContext) -> str:
    rows = []
    for rank, hit in enumerate(hits, start=1):
        caption = html.escape(context.retriever.captions[hit.image_id][0])
        used = "decoder + prompt" if rank <= context.k else "prompt"
        width = max(hit.score, 0.0) * 100
        rows.append(
            f"""
            <div class="v2w-neighbour">
              <div class="v2w-neighbour-rank">{rank:02d}</div>
              <div class="v2w-neighbour-body">
                <div class="v2w-neighbour-caption">{caption}</div>
                <div class="v2w-neighbour-meta">cosine {hit.score:.3f} &middot; {used}</div>
                <div class="v2w-neighbour-bar"><span style="width:{width:.0f}%"></span></div>
              </div>
            </div>
            """
        )
    return '<div class="v2w-neighbours">' + "".join(rows) + "</div>"


RAG_LABEL = "transformer + retrieval"
LLM_LABEL = "retrieval + language model"
NO_RETRIEVAL = "the retrieval files aren't in the model repo yet"


def caption_image(image: Image.Image | None, beam_size: int):
    if image is None:
        prompt = "upload a photograph to begin"
        return (
            _specimen_card("attention-lstm", "v2w-amber", prompt, ""),
            _specimen_card("transformer", "v2w-teal", prompt, ""),
            _specimen_card(RAG_LABEL, "v2w-sage", prompt, ""),
            None,
            "",
            [],
            _specimen_card(LLM_LABEL, "v2w-sage", prompt, ""),
        )

    state = get_state()
    vocab = state["vocab"]
    tensor = state["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pooled, spatial = state["encoder"](tensor)

        attn_features = spatial if feature_mode_for("lstm_attention") == "spatial" else pooled
        attn_caption, attn_ms = _decode(state["lstm_attention"], attn_features, beam_size, vocab)
        tf_features = spatial if feature_mode_for("transformer") == "spatial" else pooled
        tf_caption, tf_ms = _decode(state["transformer"], tf_features, beam_size, vocab)

        neighbours, llm_context = "", []
        if state["rag"] is not None:
            rag_decoder, context = state["rag"]
            # A new photograph isn't in the index, so there's nothing to
            # exclude. One search covers both: the prompt's five neighbours,
            # of which the decoder reads the first context.k.
            hits = context.retriever.search(pooled.cpu().numpy(), max(LLM_K, context.k))[0]
            ids = [h.image_id for h in hits]
            rag_features = context.decoder_input(spatial, ids)
            rag_caption, rag_ms = _decode(rag_decoder, rag_features, beam_size, vocab)
            neighbours = _neighbour_list(hits, context)
            llm_context = neighbour_captions(context.retriever, ids[:LLM_K], LLM_PER_NEIGHBOR)

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
    if state["rag"] is not None:
        rag_card = _specimen_card(
            RAG_LABEL,
            "v2w-sage",
            rag_caption,
            f"{decode_label} &middot; {rag_ms:.0f} ms &middot; CPU",
        )
        llm_card = _specimen_card(
            LLM_LABEL, "v2w-sage", "press the button to ask the language model", ""
        )
    else:
        rag_card = _specimen_card(RAG_LABEL, "v2w-sage", "unavailable", NO_RETRIEVAL)
        llm_card = _specimen_card(LLM_LABEL, "v2w-sage", "unavailable", NO_RETRIEVAL)
    return attn_card, tf_card, rag_card, filmstrip, neighbours, llm_context, llm_card


def caption_with_llm(llm_context: list[str]) -> str:
    if not llm_context:
        if _state and _state["rag"] is None:
            return _specimen_card(LLM_LABEL, "v2w-sage", "unavailable", NO_RETRIEVAL)
        return _specimen_card(LLM_LABEL, "v2w-sage", "caption a photograph first", "")
    start = time.perf_counter()
    caption = get_llm().caption(llm_context)
    ms = (time.perf_counter() - start) * 1000
    meta = f"{LLM_MODEL.split('/')[-1]} &middot; {len(llm_context)} captions &middot; {ms:.0f} ms"
    return _specimen_card(LLM_LABEL, "v2w-sage", html.escape(caption), meta + " &middot; CPU")


MASTHEAD = """
<div id="v2w-masthead">
  <div class="v2w-kicker">Flickr8k &middot; EfficientNet-B0 &middot; retrieval</div>
  <h1>Vision2Words</h1>
  <div class="v2w-sub">
    Upload a photograph and watch three decoders each write their own caption
    for it from the same frozen EfficientNet-B0 features: an attention-LSTM, a
    Transformer, and a Transformer that also reads the captions of the most
    similar training photographs. The attention tab shows where the LSTM
    looked for each word. The retrieval tab shows what was retrieved, and lets
    a small language model write a caption from those captions alone.
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
                    rag_card = gr.HTML(_specimen_card(RAG_LABEL, "v2w-sage", "&hellip;", ""))
                with gr.Tab("Attention"):
                    gr.HTML('<div class="v2w-section-label">where the attention-LSTM looked</div>')
                    filmstrip = gr.Image(label=None, show_label=False, elem_id="v2w-filmstrip")
                with gr.Tab("Retrieval"):
                    gr.HTML(
                        '<div class="v2w-section-label">'
                        "closest training photographs, by their first caption</div>"
                    )
                    neighbours_html = gr.HTML()
                    llm_card = gr.HTML(_specimen_card(LLM_LABEL, "v2w-sage", "&hellip;", ""))
                    llm_button = gr.Button("Ask the language model", variant="secondary")
                    gr.HTML(
                        '<div class="v2w-note">The language model never sees the photograph, '
                        "only the retrieved captions. The first request downloads it "
                        "(about 1 GB), and on CPU each caption takes several seconds.</div>"
                    )

    llm_context = gr.State([])
    caption_outputs = [
        attn_card,
        tf_card,
        rag_card,
        filmstrip,
        neighbours_html,
        llm_context,
        llm_card,
    ]
    run_button.click(fn=caption_image, inputs=[image_input, beam_slider], outputs=caption_outputs)
    image_input.change(fn=caption_image, inputs=[image_input, beam_slider], outputs=caption_outputs)
    llm_button.click(fn=caption_with_llm, inputs=[llm_context], outputs=[llm_card])


if __name__ == "__main__":
    demo.launch()
