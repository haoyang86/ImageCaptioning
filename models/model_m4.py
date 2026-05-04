"""
M4: global CNN encoder + Transformer decoder.

Same encoder as M1, same decoder family as M3. The point of M4 in our
proposal is the decoder-only comparison: hold the visual representation
fixed (single global feature) and switch from LSTM to Transformer.

Memory shape into the decoder is (B, 1, d_model). Cross-attention over a
single key is mathematically a learned identity at each output position,
so M4's expressive power comes from the decoder's self-attention over
previously generated tokens, conditioned on that constant global feature.

The decoder is exposed as a submodule (model.decoder) with the
(tgt_ids, memory, memory_key_padding_mask) signature used by hao's beam
search in evaluation/decode_utils.py, so the same eval pipeline works
across M1 / M3 / M4.

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


class M4Decoder(nn.Module):
    """
    Transformer decoder for M4.

    Owns the embedding, positional encoding, memory projection, the inner
    nn.TransformerDecoder, and the final vocab projection. The shared
    beam_search calls:

        logits = decoder(tgt_ids=..., memory=..., memory_key_padding_mask=None)

    so all of those steps have to live inside this module.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        encoder_dim: int = cfg.encoder_dim,
        d_model: int = cfg.d_model,
        nhead: int = cfg.nhead,
        num_layers: int = cfg.num_decoder_layers,
        dim_feedforward: int = cfg.dim_feedforward,
        dropout: float = cfg.dropout,
        max_len: int = cfg.max_len,
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

        # ------------------------------------------------------------------
        # Memory projection (encoder_dim -> d_model)
        # ------------------------------------------------------------------
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
        # Inner transformer decoder (pytorch built-in; post-norm to match m3)
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
        self.inner_decoder = nn.TransformerDecoder(
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
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt_ids : (B, T)                  caption ids
            memory  : (B, S, encoder_dim)     raw encoder output

        Returns:
            logits : (B, T, vocab_size)
        """
        T = tgt_ids.shape[1]
        device = tgt_ids.device

        memory = self.memory_proj(memory)              # (B, S, d_model)
        # print("memory:", memory.shape)

        x = self.embedding(tgt_ids)
        x = self.embedding_proj(x)
        x = x * math.sqrt(self.d_model)                # standard transformer scaling
        x = self.pos_encoding(x)
        # print("x:", x.shape)

        tgt_mask = self._causal_mask(T, device)
        tgt_key_padding_mask = tgt_ids.eq(self.pad_idx)

        out = self.inner_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc(out)

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
        Greedy autoregressive decoding. Transformers don't carry a running
        state like an LSTM, so we recompute the whole prefix each step.
        """
        if max_len is None:
            max_len = self.max_len

        device = memory.device
        B = memory.size(0)

        # We project memory once outside the loop; the inner forward will
        # call memory_proj again on each step, which is fine since
        # nn.Identity / Linear are cheap.
        generated = torch.full(
            (B, 1), start_idx, dtype=torch.long, device=device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(
                tgt_ids=generated,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_logits = logits[:, -1, :]                # (B, vocab_size)
            next_token = next_logits.argmax(dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | next_token.eq(end_idx)
            if finished.all():
                break

        return generated


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
        self.max_len = max_len

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = EncoderGlobal(
            encoder_dim=encoder_dim,
            freeze_backbone=freeze_backbone,
            pretrained=pretrained,
            dropout=encoder_dropout,
        )

        # ------------------------------------------------------------------
        # Decoder
        # Exposed as model.decoder so hao's beam_search can call:
        #   logits = model.decoder(tgt_ids=..., memory=..., memory_key_padding_mask=None)
        # uniformly across M1 / M3 / M4.
        # ------------------------------------------------------------------
        self.decoder = M4Decoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            encoder_dim=encoder_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
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
        Teacher-forced training pass.

        Args:
            images   : (B, 3, 224, 224)
            captions : (B, T)

        Returns:
            logits : (B, T, vocab_size)
        """
        encoder_output, _ = self.encoder(images)        # (B, 1, encoder_dim)
        # print("encoder_output:", encoder_output.shape)

        return self.decoder(
            tgt_ids=captions,
            memory=encoder_output,
            memory_key_padding_mask=None,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding.

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

        encoder_output, _ = self.encoder(images)
        return self.decoder.greedy_decode(
            memory=encoder_output,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            max_len=max_len,
            memory_key_padding_mask=None,
        )
