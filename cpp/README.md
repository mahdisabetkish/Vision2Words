# C++ inference

A small CMake project that loads an image, runs the EfficientNet-B0 encoder
and one of the two Phase 2 decoders (attention-LSTM or Transformer) through
ONNX Runtime, and prints the caption -- no Python or PyTorch involved at
inference time.

## Why ONNX Runtime

LibTorch (PyTorch's C++ API) is the other common choice, but it ties the
binary to the exact PyTorch/CUDA build it was compiled against. ONNX Runtime
decouples training (PyTorch) from inference (C++): the model is exported
once, and the C++ side only needs a Python-independent runtime.

## Export strategy

See `vision2words.inference.export_onnx` for the full rationale (in short:
the attention-LSTM decoder exports as two graphs with explicit carried
state, since its per-step Python loop doesn't survive `torch.onnx.export`;
the Transformer decoder exports with a fixed-length, pad-filled token
buffer, since dynamic sequence length hit an unresolvable issue in both the
trace-based and Dynamo-based exporters for this architecture).

## Build

```bash
# 1. Fetch header-only deps (stb_image, nlohmann/json) and the ONNX Runtime
#    prebuilt package for your platform -- not committed to git.
#    Windows:
powershell -File scripts/fetch_third_party.ps1
#    Linux/macOS:
./scripts/fetch_third_party.sh

# 2. Export the ONNX graphs from a trained checkpoint (from the repo root):
v2w export-onnx --feature-dir data/features \
    --lstm-attention-checkpoint checkpoints/lstm_attention/best.pt \
    --transformer-checkpoint checkpoints/transformer/best.pt \
    --out-dir export

# 3. Configure and build.
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release
```

## Run

```bash
# Windows: cpp/build/Release/v2w_caption.exe
# Linux:   cpp/build/v2w_caption
v2w_caption path/to/image.jpg --model-dir export --decoder lstm_attention --beam-size 3
v2w_caption path/to/image.jpg --model-dir export --decoder transformer

# Latency benchmark (end-to-end: preprocess + encoder + decode), averaged
# over N runs on the same image:
v2w_caption path/to/image.jpg --model-dir export --decoder lstm_attention --benchmark 50
```

## Parity and benchmark results

See the root README's C++ section for the actual numbers (Python vs C++
captions on the same images, and the latency comparison) -- produced by
`scripts/verify_cpp_parity.py` and the `--benchmark` flag above, not
hand-written.
