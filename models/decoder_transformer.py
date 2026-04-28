import math
from typing import Optional

import torch
import torch.nn as nn

from configs import m_config as cfg
from models.transformer_decoder_layer import TransformerDecoderLayer
from models.positional_encoding import PositionalEncoding


class TransformerCaptionDecoder(nn.Module):
    """
    Transformer decoder for image captioning.

    Supports three embedding strategies:

        1. random
           Random initialized trainable embedding.

        2. pretrained_frozen
           Pretrained GloVe embedding, frozen.

        3. pretrained_finetune
           Pretrained GloVe embedding, trainable.

    Inputs:
        tgt_ids:  (B, T)
        memory:   (B, S, encoder_dim)

    Output:
        logits:   (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = cfg.d_model,
        nhead: int = cfg.nhead,
        num_layers: int = cfg.num_decoder_layers,
        dim_feedforward: int = cfg.dim_feedforward,
        dropout: float = cfg.dropout,
        max_len: int = cfg.max_len,
        encoder_dim: int = cfg.encoder_dim,
        use_positional_encoding: bool = cfg.use_positional_encoding,

        # embedding options
        embedding_type: str = cfg.embedding_type,
        embedding_dim: int = cfg.embedding_dim,
        pretrained_embedding_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_len = max_len

        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

        # --------------------------------------------------
        # Word embedding
        # --------------------------------------------------
        if embedding_type == "random":
            # Random trainable embedding directly in d_model dimension
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                padding_idx=pad_idx,
            )

            self.embedding_proj = nn.Identity()
            self.embedding_output_dim = d_model

        elif embedding_type in {
            "pretrained_frozen",
            "pretrained_finetune",
        }:
            if pretrained_embedding_matrix is None:
                raise ValueError(
                    "pretrained_embedding_matrix must be provided "
                    f"when embedding_type={embedding_type}"
                )

            if pretrained_embedding_matrix.shape != (
                vocab_size,
                embedding_dim,
            ):
                raise ValueError(
                    "pretrained_embedding_matrix shape mismatch. "
                    f"Expected {(vocab_size, embedding_dim)}, "
                    f"got {tuple(pretrained_embedding_matrix.shape)}"
                )

            freeze_embedding = (
                embedding_type == "pretrained_frozen"
            )

            self.embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding_matrix,
                freeze=freeze_embedding,
                padding_idx=pad_idx,
            )

            # GloVe 300d -> Transformer d_model 512
            if embedding_dim != d_model:
                self.embedding_proj = nn.Linear(
                    embedding_dim,
                    d_model,
                )
            else:
                self.embedding_proj = nn.Identity()

            self.embedding_output_dim = d_model

        else:
            raise ValueError(
                f"Unsupported embedding_type: {embedding_type}"
            )

        # --------------------------------------------------
        # CNN spatial features: encoder_dim -> d_model
        # --------------------------------------------------
        if encoder_dim != d_model:
            self.memory_proj = nn.Linear(
                encoder_dim,
                d_model,
            )
        else:
            self.memory_proj = nn.Identity()

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
            )
        else:
            self.positional_encoding = nn.Dropout(
                dropout
            )

        # --------------------------------------------------
        # Decoder layers
        # --------------------------------------------------
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            d_model
        )

        self.output_proj = nn.Linear(
            d_model,
            vocab_size,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters.

        Important:
            If using pretrained embeddings, do NOT reinitialize
            self.embedding.weight.
        """

        if self.embedding_type == "random":
            nn.init.normal_(
                self.embedding.weight,
                mean=0.0,
                std=0.02,
            )

            if self.pad_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[
                        self.pad_idx
                    ].fill_(0.0)

        # Initialize embedding projection if it is Linear
        if isinstance(
            self.embedding_proj,
            nn.Linear,
        ):
            nn.init.xavier_uniform_(
                self.embedding_proj.weight
            )
            nn.init.zeros_(
                self.embedding_proj.bias
            )

        # Output projection
        nn.init.xavier_uniform_(
            self.output_proj.weight
        )
        nn.init.zeros_(
            self.output_proj.bias
        )

    def generate_square_subsequent_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal mask.

        Shape:
            (T, T)

        True means masked.
        """

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return mask

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt_ids:
                Token ids, shape (B, T)

            memory:
                CNN spatial features, shape (B, S, encoder_dim)

        Returns:
            logits:
                Shape (B, T, vocab_size)
        """

        _, T = tgt_ids.shape
        device = tgt_ids.device

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------
        x = self.embedding(tgt_ids)

        # If pretrained embedding is 300d, project to d_model
        x = self.embedding_proj(x)

        # Standard Transformer scaling
        x = x * math.sqrt(
            self.d_model
        )

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------
        x = self.positional_encoding(x)

        # --------------------------------------------------
        # Project encoder memory if needed
        # --------------------------------------------------
        memory = self.memory_proj(memory)

        # --------------------------------------------------
        # Masks
        # --------------------------------------------------
        tgt_key_padding_mask = tgt_ids.eq(
            self.pad_idx
        )

        tgt_mask = self.generate_square_subsequent_mask(
            T,
            device,
        )

        # --------------------------------------------------
        # Decoder layers
        # --------------------------------------------------
        for layer in self.layers:
            x = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        x = self.final_norm(x)

        logits = self.output_proj(x)

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: Optional[int] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        Args:
            memory:
                CNN spatial features, shape (B, S, encoder_dim)

        Returns:
            generated:
                Generated token ids, shape (B, L)
        """

        self.eval()

        if max_len is None:
            max_len = self.max_len

        B = memory.size(0)
        device = memory.device

        generated = torch.full(
            size=(B, 1),
            fill_value=start_idx,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            B,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_len - 1):

            logits = self.forward(
                tgt_ids=generated,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
            )

            generated = torch.cat(
                [
                    generated,
                    next_token.unsqueeze(1),
                ],
                dim=1,
            )

            finished |= next_token.eq(
                end_idx
            )

            if finished.all():
                break

        return generated
