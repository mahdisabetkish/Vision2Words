"""Retrieval-augmented captioning with a small local language model.

The second retrieval-augmented captioner, closer to textbook RAG than the
retrieval-augmented decoder: retrieve the training images most similar to
the query, put their human-written captions into a prompt, and have an
instruction-tuned language model write one caption from them.

The language model never sees the image. That is deliberate: a model that
could look at the photo could just describe it, and retrieval would add
nothing measurable. Here the retrieved captions are the only visual
evidence the model gets, so its score is a direct measure of how much a
caption can be reconstructed from what similar images were called.

Two baselines make that measurable, both reported next to the language
model by evaluate_rag_llm:

- ``consensus_caption`` returns the retrieved caption that agrees most with
  the other retrieved captions, with no generation at all (the nearest-
  neighbour captioning baseline of Devlin et al., 2015). If the language
  model can't beat this, it is only paraphrasing its context.
- The trained decoders in results/metrics.json, which do see the image.

The default model is Qwen2.5-0.5B-Instruct (Apache-2.0, ~0.5B parameters),
small enough to run on a CPU-only Hugging Face Space. It runs in float32 on
GPU as well: the GTX 1080 Ti this project trains on is a Pascal card, whose
half-precision throughput is a small fraction of its float32 throughput.
"""

from __future__ import annotations

import re

import torch

from vision2words.data.vocabulary import tokenize
from vision2words.retrieval.index import CaptionRetriever

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You write short image captions. You will not see the image. Instead you get "
    "captions that people wrote for other photos that look very similar to it."
)

USER_TEMPLATE = (
    "Captions of similar photos:\n{captions}\n\n"
    "Write one caption for the photo, in the same plain style. Keep to what most of "
    "these captions agree on and leave out details only one of them mentions. "
    "Reply with the caption only."
)


def neighbour_captions(
    retriever: CaptionRetriever, neighbor_ids: list[str], per_neighbor: int
) -> list[str]:
    """The first ``per_neighbor`` captions of each neighbour, in rank order."""
    return [c for image_id in neighbor_ids for c in retriever.captions[image_id][:per_neighbor]]


def build_user_prompt(captions: list[str]) -> str:
    listed = "\n".join(f"- {c.strip()}" for c in captions)
    return USER_TEMPLATE.format(captions=listed)


def clean_caption(text: str) -> str:
    """Reduce a model reply to one plain caption: first non-empty line, no
    leading label/bullet/number, no surrounding quotes, lowercase, no final
    punctuation -- the same normalised form the reference captions are
    compared in.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            break
    else:
        return ""
    line = re.sub(r"^(caption\s*:|[-*•]|\d+[.)])\s*", "", line, flags=re.IGNORECASE)
    line = line.strip().strip("\"'“”").strip()
    return line.rstrip(".!").strip().lower()


def _token_set(caption: str) -> set[str]:
    return set(tokenize(caption))


def consensus_caption(captions: list[str]) -> str:
    """Pick the retrieved caption with the highest mean word-overlap (Jaccard)
    with the other retrieved captions -- the one most of the neighbours'
    descriptions agree with.
    """
    if len(captions) == 1:
        return captions[0]
    sets = [_token_set(c) for c in captions]
    best, best_score = captions[0], -1.0
    for i, s in enumerate(sets):
        overlaps = [len(s & t) / max(len(s | t), 1) for j, t in enumerate(sets) if j != i]
        score = sum(overlaps) / len(overlaps)
        if score > best_score:
            best, best_score = captions[i], score
    return best


class LLMCaptioner:
    """Generate a caption from retrieved captions with a local causal LM.

    ``tokenizer``/``model`` can be passed in directly (the tests use tiny
    stand-ins); otherwise they're loaded from the Hugging Face Hub by name.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_new_tokens: int = 32,
        tokenizer=None,
        model=None,
    ) -> None:
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            model = model or AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        # Left padding keeps every prompt's last token at the same position,
        # which batched generation with a decoder-only model needs.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def _chat(self, captions: list[str]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(captions)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def caption_batch(self, captions_per_image: list[list[str]]) -> list[str]:
        prompts = [self._chat(captions) for captions in captions_per_image]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        replies = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [clean_caption(reply) for reply in replies]

    def caption(self, captions: list[str]) -> str:
        return self.caption_batch([captions])[0]
