# Docker images

Two images, built from the repo root (the build context needs `src/`,
`cpp/`, and `pyproject.toml`, which is why the Dockerfiles live in
`docker/` but are invoked with `-f`):

```bash
docker build -f docker/Dockerfile.train -t vision2words-train .
docker build -f docker/Dockerfile.inference -t vision2words-infer .
```

Neither image bakes in the dataset, checkpoints, or exported models --
those are bind-mounted at `docker run` time, the same way the Nextflow
pipeline (`pipeline/`) uses them.

## Training / evaluation (`vision2words-train`)

CUDA 12.6 runtime, the project installed as a wheel, entrypoint is the
`v2w` CLI:

```bash
docker run --rm --gpus all \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    vision2words-train train --config configs/lstm_attention.yaml
```

Drop `--gpus all` to run on CPU (set `train.device: cpu` in the config, or
`--set train.device=cpu`).

## Inference (`vision2words-infer`)

No Python, no CUDA -- just the compiled C++ binary from `cpp/` and ONNX
Runtime's CPU execution provider:

```bash
docker run --rm \
    -v "$(pwd)/export/onnx:/app/export" \
    -v "$(pwd)/data/raw/images:/app/images:ro" \
    vision2words-infer /app/images/some_image.jpg --model-dir /app/export --decoder lstm_attention --beam-size 3
```

`export/onnx/` is produced by `v2w export-onnx --out-dir export/onnx` (see
the root README's C++ section) and must contain `vocab.json`, `encoder.onnx`, and either
`lstm_attention_init.onnx` + `lstm_attention_step.onnx` or
`transformer.onnx`.

## Demo (`vision2words-app`)

The Gradio demo, built from `docker/Dockerfile.app`. The trained decoders
download from Hugging Face on first start:

```bash
docker run --rm -p 7860:7860 mahdisabetkish/vision2words-app
```

Then open http://localhost:7860.

## Published images and status

Both inference images are on Docker Hub:

```bash
docker pull mahdisabetkish/vision2words-infer
docker pull mahdisabetkish/vision2words-app
```

Status on Ubuntu 24.04, CPU only:

- `vision2words-infer` builds, and the binary captions test images with both
  decoders. Its output has not yet been compared with the Python pipeline
  from inside the container. That comparison was made with the native build
  (see the root README).
- `vision2words-app` builds, serves the Gradio page, and returns captions
  from the published model.
- `vision2words-train` builds in CI (GitHub Actions, `build-docker` job). It
  has not been run on a GPU host, so the training path inside the container
  is untested on real hardware.
