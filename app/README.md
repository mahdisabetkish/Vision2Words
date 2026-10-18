---
title: Vision2Words
emoji: 🎞️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Vision2Words

Upload a photograph and compare captions from three decoders trained on
Flickr8k: an LSTM with Bahdanau attention, a Transformer decoder, and a
Transformer that also reads the captions of the most similar training
photographs, all on the same frozen EfficientNet-B0 features. The Attention
tab shows which part of the image the LSTM decoder was looking at for each
word it wrote. The Retrieval tab lists the captions retrieval found, and a
small language model (Qwen2.5-0.5B-Instruct) can write a caption from them
without seeing the photograph.

Runs on CPU. Weights are pulled from a separate Hugging Face model repo at
startup, not stored in this Space. Source, training code, and the full
writeup are in the GitHub repo linked from the project page.
