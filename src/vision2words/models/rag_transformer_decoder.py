"""Retrieval-augmented Transformer decoder.

Same decoder as transformer_decoder.py, with one change: what it
cross-attends to. The plain decoder's memory is the 49 spatial image tokens.
Here the memory is those 49 tokens followed by the tokens of captions
retrieved for similar training images (see vision2words.retrieval), so at
every step the decoder can attend to the image, to how people described
images that look like it, or to both. The idea follows SmallCap (Ramos et
al., CVPR 2023), adapted to an encoder-decoder captioner trained from
scratch rather than a frozen language model.

Retrieved tokens share the decoder's own word embedding -- they're the same
vocabulary, and sharing it makes "this retrieved word is a good next word"
a short path for the model to learn. A learned segment embedding marks each
memory position as image or text, the retrieved text gets its own
positional encoding, and a small Transformer encoder lets the retrieved
captions attend to each other (e.g. to notice that three of four neighbours
mention a dog) before the decoder reads them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vision2words.models.inputs import RetrievalAugmentedInput
from vision2words.models.transformer_decoder import PositionalEncoding, TransformerDecoderModel

IMAGE_SEGMENT, TEXT_SEGMENT = 0, 1


class RetrievalAugmentedTransformerDecoder(TransformerDecoderModel):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        feature_size: int = 1280,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_size: int = 2048,
        dropout: float = 0.1,
        max_len: int = 64,
        max_retrieved_len: int = 128,
        text_encoder_layers: int = 1,
    ) -> None:
        super().__init__(
            vocab_size,
            pad_id,
            feature_size,
            d_model,
            num_layers,
            num_heads,
            ff_size,
            dropout,
            max_len,
        )
        self.max_retrieved_len = max_retrieved_len
        self.segment = nn.Embedding(2, d_model)
        self.retrieved_pos = PositionalEncoding(d_model, max_retrieved_len)
        if text_encoder_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                batch_first=True,
            )
            # The nested-tensor fast path only applies in eval mode and warns
            # whenever a padding mask is passed; it buys nothing at this size.
            self.text_encoder = nn.TransformerEncoder(
                layer, num_layers=text_encoder_layers, enable_nested_tensor=False
            )
        else:
            self.text_encoder = None

    def encode_memory(self, features: RetrievalAugmentedInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the cross-attention memory and its padding mask:
        [49 image tokens | retrieved caption tokens].
        """
        retrieved = features.retrieved[:, : self.max_retrieved_len]
        image_memory = self.feature_proj(features.image) + self.segment.weight[IMAGE_SEGMENT]

        text = self.embedding(retrieved) * math.sqrt(self.d_model)
        text = self.dropout(self.retrieved_pos(text) + self.segment.weight[TEXT_SEGMENT])
        text_pad_mask = retrieved == self.pad_id
        if self.text_encoder is not None:
            text = self.text_encoder(text, src_key_padding_mask=text_pad_mask)

        memory = torch.cat([image_memory, text], dim=1)
        image_pad_mask = torch.zeros(
            image_memory.shape[:2], dtype=torch.bool, device=image_memory.device
        )
        # Image positions are never masked, so a row is never fully masked and
        # attention can't produce NaNs even if retrieval returned nothing.
        return memory, torch.cat([image_pad_mask, text_pad_mask], dim=1)

    def forward(
        self, features: RetrievalAugmentedInput, input_tokens: torch.Tensor
    ) -> torch.Tensor:
        memory, memory_pad_mask = self.encode_memory(features)
        embedded = self.embedding(input_tokens) * math.sqrt(self.d_model)
        embedded = self.dropout(self.pos_encoding(embedded))

        seq_len = input_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_tokens.device), diagonal=1
        )
        output = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=input_tokens == self.pad_id,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.classifier(output)
