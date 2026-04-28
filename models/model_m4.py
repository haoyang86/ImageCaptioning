"""
M4: global CNN encoder + Transformer decoder.

Same encoder as M1, same decoder family as M3. The point of M4 in our
proposal is the decoder-only comparison: hold the visual representation
fixed (single global feature) and switch from LSTM to Transformer.

Memory shape into the decoder is (B, 1, d_model). Cross-attention over a
single key is mathematically a learned identity at each output position,
so M4's expressive power comes from the decoder's self-attention over
previously generated tokens, conditioned on that constant global feature.

I referenced the M3 code from my teammates and my own assignment 2/3 work
while putting this together. Using nn.TransformerDecoder here the same
way I used FullTransformerTranslator in assignment 3.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from configs import m4_config as cfg
from models.encoder_global import EncoderGlobal
from models.positional_encoding import PositionalEncoding


class ModelM4(nn.Module):
    """
    Global CNN encoder + Transformer decoder.

    embedding_type:
        random              : trainable, init from N(0, 0.02)
        pretrained_frozen   : GloVe, frozen
        pretrained_finetune : GloVe, trainable
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        encoder_dim: int = cfg.encoder_dim,
        d_model: int = cfg.d_model,
        nhead: int = cfg.nhead,
        num_layers: int = cfg.num_decoder_layers,
        dim_feedforward: int = cfg.dim_feedforward,
        dropout: float = cfg.dropout,
        max_len: int = cfg.max_len,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        encoder_dropout: float = 0.2,
        embedding_type: str = cfg.embedding_type,
        embedding_dim: int = cfg.embedding_dim,
        pretrained_embedding_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.d_model = d_model
        self.max_len = max_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = EncoderGlobal(
            encoder_dim=encoder_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            dropout=encoder_dropout,
        )
        # encoder_output is (B, 1, encoder_dim); project to d_model if needed
        if encoder_dim != d_model:
            self.memory_proj = nn.Linear(encoder_dim, d_model)
            nn.init.xavier_uniform_(self.memory_proj.weight)
            nn.init.zeros_(self.memory_proj.bias)
        else:
            self.memory_proj = nn.Identity()

        # ------------------------------------------------------------------
        # Word embedding (+ projection + positional encoding)
        # ------------------------------------------------------------------
        if embedding_type == "random":
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embedding.weight[pad_idx].fill_(0.0)

        elif embedding_type in {"pretrained_frozen", "pretrained_finetune"}:
            if pretrained_embedding_matrix is None:
                raise ValueError(
                    f"pretrained_embedding_matrix must be provided when "
                    f"embedding_type={embedding_type}"
                )
            if pretrained_embedding_matrix.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embedding_matrix shape mismatch. "
                    f"Expected {(vocab_size, embedding_dim)}, "
                    f"got {tuple(pretrained_embedding_matrix.shape)}"
                )
            freeze = embedding_type == "pretrained_frozen"
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding_matrix,
                freeze=freeze,
                padding_idx=pad_idx,
            )

        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        # GloVe is 300d; the Transformer body works in d_model (512 by default)
        if embedding_dim != d_model:
            self.embedding_proj = nn.Linear(embedding_dim, d_model)
            nn.init.xavier_uniform_(self.embedding_proj.weight)
            nn.init.zeros_(self.embedding_proj.bias)
        else:
            self.embedding_proj = nn.Identity()

        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Transformer decoder (pytorch built-in; post-norm to match m3)
        # ------------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.fc = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @staticmethod
    def _causal_mask(T: int, device):
        # True == masked (do not attend); upper triangle blocks future tokens
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced training pass.

        Args:
            images   : (B, 3, 224, 224)
            captions : (B, T)

        Returns:
            logits : (B, T, vocab_size)
        """
        B, T = captions.shape
        device = captions.device

        encoder_output, _ = self.encoder(images)       # (B, 1, encoder_dim)
        memory = self.memory_proj(encoder_output)      # (B, 1, d_model)
        # print("memory:", memory.shape)

        x = self.embedding(captions)
        x = self.embedding_proj(x)
        x = x * math.sqrt(self.d_model)                # standard transformer scaling
        x = self.pos_encoding(x)
        # print("x:", x.shape)

        tgt_mask = self._causal_mask(T, device)
        tgt_key_padding_mask = captions.eq(self.pad_idx)

        # memory has length 1 and no padding, so no memory_key_padding_mask
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc(out)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding. Transformers don't carry a running
        state like an LSTM, so we recompute the whole prefix each step.

        Args:
            images : (B, 3, 224, 224)

        Returns:
            generated : (B, L), L <= max_len
        """
        if self.start_idx is None or self.end_idx is None:
            raise ValueError("start_idx and end_idx must be set before decoding.")

        self.eval()
        if max_len is None:
            max_len = self.max_len

        device = images.device
        B = images.size(0)

        encoder_output, _ = self.encoder(images)
        memory = self.memory_proj(encoder_output)

        # start every sequence with <start>
        generated = torch.full(
            (B, 1), self.start_idx, dtype=torch.long, device=device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T = generated.size(1)

            x = self.embedding(generated)
            x = self.embedding_proj(x)
            x = x * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            tgt_mask = self._causal_mask(T, device)
            tgt_key_padding_mask = generated.eq(self.pad_idx)

            out = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            next_logits = self.fc(out[:, -1, :])       # (B, vocab_size)
            next_token = next_logits.argmax(dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | next_token.eq(self.end_idx)
            if finished.all():
                break

        return generated
