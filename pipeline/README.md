# Nextflow pipeline

`main.nf` runs the whole project end to end:

```
download_or_link_data -> extract_features -> [train_lstm_baseline, train_lstm_attention, train_transformer in parallel]
    -> evaluate (each) -> build_report
    -> export_onnx (decoders A/B) -> build_cpp -> cpp_benchmark
    -> report
```

Every process body is the same `v2w` CLI command you'd run by hand -- the
pipeline's job is orchestrating what runs in parallel and wiring file
outputs to the next stage's inputs, not reimplementing any ML code.

## Running it

From the repo root:

```bash
# Local: needs `pip install -e .` done already, and for the C++ stage, a
# C++ toolchain (cmake + compiler) on PATH.
nextflow run pipeline/main.nf -profile local

# A fast, tiny-subset run that exercises every stage in a few minutes --
# use this to sanity-check a change to the pipeline itself.
nextflow run pipeline/main.nf -profile local,test

# Full run inside the project's own Docker images (see docker/README.md) --
# needs a working Docker host.
nextflow run pipeline/main.nf -profile docker

# Same, but trains on GPU (needs the NVIDIA Container Toolkit on the host).
nextflow run pipeline/main.nf -profile gpu
```

Outputs land under `data/`, `checkpoints/`, `export/`, and `results/` at
the repo root (wherever you launched `nextflow run` from), matching where
the manual CLI workflow puts them -- so inspecting the pipeline's output is
the same as inspecting a manual run's.

## What `-profile test` actually changes

`params.max_images = 24` caps each split to 24 images (`extract-features
--max-images`), `params.epochs = 1` trains for exactly one epoch per
model (and sets `data.min_word_freq=1`, since 24 images' captions don't
repeat any word 5 times), and `params.beam_size = 2`. None of that is a
separate code path -- it's the same config-override mechanism (`--set`)
used everywhere else in this project, just driven by Nextflow params
instead of typed by hand.

## Status

Written and reviewed carefully, but **not execution-tested** on the
project's own dev machine, for a platform reason rather than a pipeline
one: Nextflow requires a POSIX environment and explicitly only supports
Windows through WSL (its own `Session.registerSignalHandlers` tries to
register a `SIGHUP` handler that plain Windows' JVM doesn't have, and
fails immediately with `Unknown signal: HUP` before running a single
process -- confirmed on this machine, not a guess). This dev machine is a
remote box that can't be rebooted, which rules out installing WSL here.

So: real execution testing of `main.nf`, all four profiles included, is
still owed on a real Linux host (the `local` profile needs `pip install
-e .` and a C++ toolchain there; `docker`/`gpu` need a working Docker
host). Review the DSL2 carefully before trusting it blind.
