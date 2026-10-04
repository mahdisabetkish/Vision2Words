"""EfficientNet-B0 image encoder.

Frozen by default: the convolutional backbone is treated as a fixed feature
extractor, which is what makes feature caching possible in the first place
(see data/features.py) -- there's no point paying for the same convolution
forward pass on every epoch when the weights producing it never change.
Pass ``fine_tune=True`` to unfreeze it for end-to-end training instead.

Two feature representations come out of the same forward pass:
  * ``pooled``, a 1280-d vector (global average pool) for the plain LSTM
    decoder;
  * ``spatial``, the 7x7 grid of 1280-d vectors (49 tokens) that the
    attention and Transformer decoders attend over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEATURE_DIM = 1280
SPATIAL_TOKENS = 49  # 7 * 7


class EncoderCNN(nn.Module):
    def __init__(self, fine_tune: bool = False) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # (B, 3, 224, 224) -> (B, 1280, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.set_trainable(fine_tune)

    def set_trainable(self, trainable: bool) -> None:
        self.fine_tune = trainable
        for param in self.features.parameters():
            param.requires_grad = trainable
        self.features.train(trainable)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_map = self.features(images)  # (B, 1280, 7, 7)
        pooled = self.pool(spatial_map).flatten(1)  # (B, 1280)
        spatial = spatial_map.flatten(2).transpose(1, 2)  # (B, 49, 1280)
        return pooled, spatial

    @staticmethod
    def preprocess() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
