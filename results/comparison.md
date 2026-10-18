# Decoder comparison

Flickr8k test split, 1000 images. Beam search uses beam size 3. Latency is
decoder-only (cached features), on the GTX 1080 Ti.

| Model | Params | Train s/epoch | Decode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| LSTM (no attention) | 4,842,733 | 19.4 | greedy | 0.573 | 0.391 | 0.258 | 0.172 | 0.443 | 15.4 |
| LSTM (no attention) | 4,842,733 | 19.4 | beam k=3 | 0.590 | 0.411 | 0.280 | 0.188 | 0.485 | 39.4 |
| LSTM + attention | 6,154,478 | 63.5 | greedy | 0.601 | 0.416 | 0.277 | 0.182 | 0.464 | 75.9 |
| LSTM + attention | 6,154,478 | 63.5 | beam k=3 | 0.610 | 0.430 | 0.298 | 0.202 | 0.525 | 176.0 |
| Transformer | 20,076,525 | 51.3 | greedy | 0.588 | 0.403 | 0.270 | 0.181 | 0.460 | 93.9 |
| Transformer | 20,076,525 | 51.3 | beam k=3 | 0.599 | 0.420 | 0.290 | 0.198 | 0.502 | 160.5 |
| Transformer + retrieval | 23,229,933 | 77.7 | greedy | 0.586 | 0.406 | 0.273 | 0.181 | 0.477 | 125.8 |
| Transformer + retrieval | 23,229,933 | 77.7 | beam k=3 | 0.600 | 0.421 | 0.292 | 0.200 | 0.531 | 209.2 |
| Transformer + retrieval (shuffled neighbours, control) | 23,229,933 | 77.7 | greedy | 0.570 | 0.388 | 0.260 | 0.174 | 0.446 | 123.6 |
| Transformer + retrieval (shuffled neighbours, control) | 23,229,933 | 77.7 | beam k=3 | 0.588 | 0.405 | 0.277 | 0.188 | 0.483 | 210.6 |

## Retrieval + language model

Same test images and metric code. Each image retrieves its 5 nearest
training images and takes 2 captions from each.
The consensus row picks the retrieved caption the others agree with most; the
language-model rows have a small instruction-tuned model write a caption from
them. None of these rows sees the image itself. Context order is how the
captions are listed in the prompt: as retrieved, or most agreed-on first.
Latency is the language model only.

| System | Context order | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | ms/img |
|---|---|---|---|---|---|---|---|
| Retrieval only (consensus caption) | - | 0.529 | 0.336 | 0.209 | 0.127 | 0.348 | - |
| Retrieval + Qwen2.5-0.5B-Instruct | retrieval | 0.479 | 0.290 | 0.174 | 0.102 | 0.254 | 94.7 (cuda, batch 16) |
| Retrieval + Qwen2.5-1.5B-Instruct | agreement | 0.473 | 0.285 | 0.172 | 0.100 | 0.274 | 180.2 (cuda, batch 16) |
