"""Seeding helpers for reproducible runs."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Convolution algorithms are left non-deterministic on purpose: Pascal-era
    cuDNN kernels for EfficientNet's depthwise convolutions slow down a lot
    under full determinism, and this project doesn't need bit-exact repeats
    across GPUs, just stable, comparable runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
