---
license: mit
datasets:
  - flickr8k
language:
  - en
tags:
  - image-captioning
  - pytorch
  - lstm
  - transformer
  - attention
  - retrieval-augmented-generation
  - faiss
pipeline_tag: image-to-text
---

# Vision2Words decoders

Three image-caption decoders trained on Flickr8k, all reading features
from a frozen, ImageNet-pretrained EfficientNet-B0 encoder (never
fine-tuned): an LSTM with Bahdanau attention over the encoder's 7x7 spatial
feature grid, a Transformer decoder attending over the same grid, and a
retrieval-augmented Transformer that also attends over the captions of the
most similar training images. Files in this repo: `lstm_attention.pt`,
`transformer.pt`, `transformer_rag.pt`, `vocab.json`, and the retrieval
index under `retrieval/`.

## Architecture

- **Encoder**: EfficientNet-B0 (`torchvision`, ImageNet weights), frozen.
  224x224 input, ImageNet normalization. Produces a 7x7x1280 spatial grid.
- **Decoder A (`lstm_attention.pt`)**: Bahdanau additive attention over the
  49 spatial tokens, projected down to a 256-d context vector before the
  LSTM cell to control parameter count; embed 256, hidden 512.
- **Decoder B (`transformer.pt`)**: 4-layer Transformer decoder, d_model
  512, 8 heads, sinusoidal positional encoding, causal self-attention plus
  cross-attention into the same 49 spatial tokens.
- **Decoder C (`transformer_rag.pt`)**: Decoder B's architecture, with the
  captions of the 4 nearest training images (one each) appended to its
  cross-attention memory after the 49 image tokens. The retrieved tokens
  share the word embedding, get their own positional and segment
  embeddings, and pass through one Transformer encoder layer.
- **Retrieval index (`retrieval/`)**: an exact cosine-similarity FAISS index
  (`IndexFlatIP` on L2-normalised vectors) over the pooled 1280-d features
  of the 6,000 training images, with their captions. Only the training
  split is indexed, and an image never retrieves itself during training,
  so no evaluation image's reference captions can reach the decoder.

## Training data

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k): 8,091
photographs, 5 human-written captions each. Standard 6,000 / 1,000 / 1,000
train/val/test split. Vocabulary: 2,541 words appearing at least 5 times
in the training captions; everything else maps to `<UNK>`.

## Metrics (test split, 1,000 images)

| Decoder | Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|---|---|---|---|---|---|---|
| LSTM + attention | greedy | 0.601 | 0.416 | 0.277 | 0.182 | 0.464 |
| LSTM + attention | beam (k=3) | 0.610 | 0.430 | 0.298 | **0.202** | 0.525 |
| Transformer | greedy | 0.588 | 0.403 | 0.270 | 0.181 | 0.460 |
| Transformer | beam (k=3) | 0.599 | 0.420 | 0.290 | 0.198 | 0.502 |
| Transformer + retrieval | greedy | 0.586 | 0.406 | 0.273 | 0.181 | 0.477 |
| Transformer + retrieval | beam (k=3) | 0.600 | 0.421 | 0.292 | 0.200 | **0.531** |

Decoder C evaluated with every test image given another image's neighbours
scores CIDEr 0.446 (greedy) and 0.483 (beam), below Decoder B, so its gain
comes from retrieved captions that match the image. Each number is from a
single training run.

Tokenization for these metrics is our own (lowercase, punctuation
stripped), not the official COCO-eval PTBTokenizer, which needs a Java
runtime. For comparing these two models against each other that doesn't
matter; it does mean these numbers aren't directly comparable to numbers
reported against the official tokenizer.

The attention-LSTM edges out the Transformer here, which is a reasonable
outcome rather than a surprising one: at Flickr8k's scale (6,000 training
images), the Transformer's extra parameters (20M vs 6M) have less chance
to pay off than they would on a larger dataset like COCO or Flickr30k.

## Intended use

A portfolio / teaching demonstration of two captioning architectures on a
small, well-known dataset -- not a production captioning system. The
Hugging Face Space built on these weights (see the repo this model card
links to) is the intended way to try them interactively.

## Limitations

- Trained on Flickr8k only: everyday scenes, people, and animals. It will
  describe other kinds of images (documents, diagrams, abstract art)
  poorly or not at all.
- Greedy decoding on the Transformer decoder can fall into a repetition
  loop (e.g. repeating the same word); beam search (the Space's default)
  avoids this reliably. This is a known property of greedy decoding with
  Transformer language models, not specific to this checkpoint.
- Vocabulary is capped at 2,541 words; anything rarer becomes `<UNK>`.
- Decoder C can only borrow from what is in its index: 6,000 Flickr8k
  photographs. For an image unlike any of them, the retrieved captions are
  misleading rather than helpful.
- Not evaluated for bias in how it describes people (age, gender
  presentation, skin tone, etc.) -- Flickr8k's caption distribution
  reflects its original annotators and is not demographically audited
  here.
