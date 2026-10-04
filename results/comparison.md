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
