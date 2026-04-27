from typing import Optional

import torch
import torch.nn as nn

from configs import m3_config as cfg
from models.encoder_spatial import EncoderSpatial
from models.decoder_transformer import TransformerCaptionDecoder


class ModelM3(nn.Module):
    """
    M3: CNN spatial encoder + Transformer decoder.

    Supports embedding strategies:
        1. random
        2. pretrained_frozen
        3. pretrained_finetune

    Input:
        images:   (B, 3, 224, 224)
        captions: (B, T)

    Output:
        logits:   (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        encoder_dim: int = cfg.encoder_dim,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        encoder_dropout: float = 0.2,

        # embedding options
        embedding_type: str = cfg.embedding_type,
        embedding_dim: int = cfg.embedding_dim,
        pretrained_embedding_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

        self.encoder = EncoderSpatial(
            encoder_dim=encoder_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            dropout=encoder_dropout,
        )

        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            encoder_dim=encoder_dim,
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            pretrained_embedding_matrix=pretrained_embedding_matrix,
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Usually:
            input captions  = captions[:, :-1]
            target captions = captions[:, 1:]

        Args:
            images:
                (B, 3, 224, 224)

            captions:
                (B, T)

        Returns:
            logits:
                (B, T, vocab_size)
        """

        encoder_output, _ = self.encoder(images)

        logits = self.decoder(
            tgt_ids=captions,
            memory=encoder_output,
            memory_key_padding_mask=None,
        )

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference using greedy decoding.

        Args:
            images:
                (B, 3, 224, 224)

        Returns:
            generated token ids:
                (B, L)
        """

        if self.start_idx is None or self.end_idx is None:
            raise ValueError(
                "start_idx and end_idx must be provided for decoding."
            )

        self.eval()

        encoder_output, _ = self.encoder(images)

        generated = self.decoder.greedy_decode(
            memory=encoder_output,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            max_len=max_len,
            memory_key_padding_mask=None,
        )

        return generated