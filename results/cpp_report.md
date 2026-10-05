# C++ inference: parity and benchmark

Measured on the development machine: Windows 10, GeForce GTX 1080 Ti, MSVC
19.44 (Visual Studio 2022 Build Tools), ONNX Runtime 1.20.1 (CPU execution
provider). C++ binary built from `cpp/`; see `cpp/README.md` for the build
steps and `scripts/verify_cpp_parity.py` / `scripts/benchmark_python.py` for
how these numbers were produced.

## Caption parity (Python vs C++, beam size 3)

8 test-split images x 2 decoders = 16 comparisons: **12/16 (75%) exact
string match**. All 4 mismatches are close paraphrases of each other, not
garbage output, e.g.:

| decoder | python | cpp |
|---|---|---|
| transformer | "a man and a woman sit on a bench near an `<UNK>`" | "a man and a woman sit on a bench near a red sign" |
| transformer | "a brown dog is running through the grass" | "two brown dogs running in the grass" |

The cause is a known, documented gap rather than a bug in the exported
graphs: `stb_image_resize2`'s resize filter isn't bit-identical to PIL's
bilinear resize (see `cpp/src/preprocess.hpp`), so the encoder sees
slightly different pixel values in C++ than in Python. That's enough to
occasionally flip a close beam-search decision over a long sequence,
especially for the Transformer decoder (longer effective context per
step than the LSTM). The `encoder`, `lstm_attention_init/step`, and
`transformer` ONNX graphs themselves were separately verified to match
PyTorch to float32 precision (~1e-6 max absolute difference) on identical
input tensors -- see the commit adding `export_onnx.py` -- so the
discrepancy traces specifically to image preprocessing, not model export.

Greedy decoding on the Transformer decoder reliably hits a repetition
loop (e.g. "...a wooden wooden wooden wooden..."), reproduced identically
in both Python and C++. That's a textbook greedy-decoding failure mode for
Transformer language models, not a C++-specific bug -- beam search (what
the table above uses) resolves it in both.

## Latency (single image, model loaded once, mean of 50 runs)

| decoder | Python (CPU) | Python (CUDA, GTX 1080 Ti) | C++ (CPU, ONNX Runtime) |
|---|---|---|---|
| LSTM + attention | 138.2 ms | 147.1 ms | **27.8 ms** |
| Transformer | 221.7 ms | 202.3 ms | **182.8 ms** |

C++ on CPU beats Python on both CPU and GPU for this workload. That's not
surprising once you look at what's being measured: this is single-image
(batch size 1), short-sequence (<=20 tokens) inference, where GPU kernel
-launch and host<->device transfer overhead outweighs the actual compute,
and Python's interpreter/dispatch overhead on every decode step adds up
across ~10-20 autoregressive steps. ONNX Runtime's CPU execution provider
avoids both: no Python between steps, and no GPU round-trip for work this
small. This is a standard reason production systems serve small-batch
real-time inference from a compiled runtime on CPU rather than from a
Python/GPU training stack, even when a GPU is available.
