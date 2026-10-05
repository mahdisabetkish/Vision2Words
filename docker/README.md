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
    -v "$(pwd)/export:/app/export" \
    -v "$(pwd)/data/raw/images:/app/images:ro" \
    vision2words-infer /app/images/some_image.jpg --model-dir /app/export --decoder lstm_attention --beam-size 3
```

`export/` is produced by `v2w export-onnx` (see the root README's C++
section) and must contain `vocab.json`, `encoder.onnx`, and either
`lstm_attention_init.onnx` + `lstm_attention_step.onnx` or
`transformer.onnx`.

## A note on where these were built

Both Dockerfiles were written and reviewed carefully but not build-tested
on the development machine for this project -- it's a remote Windows box
that can't be rebooted, which rules out WSL2/Docker Desktop there. Build
and run them on a real Docker host (the Nextflow `docker` and `gpu`
profiles assume this) before relying on them.
