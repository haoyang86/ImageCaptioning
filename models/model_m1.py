"""
M1 (Show and Tell): global CNN encoder + LSTM decoder, no attention.
Vinyals et al., 2015.

Mirrors the forward / greedy_decode interface used by M3 so the shared
trainer and evaluator can call it without special casing. The decoder is
exposed as a submodule (model.decoder) with the (tgt_ids, memory,
memory_key_padding_mask) signature used by hao's beam search in
evaluation/decode_utils.py.

I referenced the M3 code from my teammates and my own assignment 2/3 work
while putting this together.
"""

from typing import Optional

import torch
import torch.nn as nn

from configs import m1_config as cfg
from models.encoder_global import EncoderGlobal


class M1Decoder(nn.Module):
    """
    LSTM decoder for M1, no attention.

    memory shape:
        (B, S, encoder_dim)
        For the global encoder S=1; we mean-pool over S so the same code
        path also works if a future caller hands in spatial-shaped memory.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        encoder_dim: int = cfg.encoder_dim,
        hidden_dim: int = cfg.hidden_dim,
        num_lstm_layers: int = cfg.num_lstm_layers,
        dropout: float = cfg.dropout,
        max_len: int = cfg.max_len,
        embedding_type: str = cfg.embedding_type,
        embedding_dim: int = cfg.embedding_dim,
        pretrained_embedding_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.max_len = max_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

        # ------------------------------------------------------------------
        # Word embedding (+ projection + dropout)
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

        # GloVe is 300d; the LSTM body works in hidden_dim (default 1024 per optuna best)
        if embedding_dim != hidden_dim:
            self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
            nn.init.xavier_uniform_(self.embedding_proj.weight)
            nn.init.zeros_(self.embedding_proj.bias)
        else:
            self.embedding_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # LSTM body
        # ------------------------------------------------------------------
        # init (h_0, c_0) from the global feature, matches what m3 / decoder_lstm does
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _init_state(self, memory):
        # memory: (B, S, encoder_dim); mean over S so this works for both
        # global (S=1) and spatial memory (S>1)
        feats = memory.mean(dim=1)        # (B, encoder_dim)
        h0 = self.init_h(feats)
        c0 = self.init_c(feats)
        # nn.LSTM wants (num_layers, B, hidden_dim); repeat the visual init across layers
        h0 = h0.unsqueeze(0).expand(self.num_lstm_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.num_lstm_layers, -1, -1).contiguous()
        return h0, c0

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced training pass.

        Args:
            tgt_ids : (B, T)            caption ids
            memory  : (B, S, encoder_dim)  encoder output

        Returns:
            logits : (B, T, vocab_size)
        """
        # memory_key_padding_mask is unused here (S=1 for global encoder, no padding);
        # it's kept in the signature so the shared beam_search can call us uniformly.
        del memory_key_padding_mask

        x = self.embedding(tgt_ids)
        x = self.embedding_proj(x)
        x = self.dropout(x)
        # print("x embedded:", x.shape)

        h0, c0 = self._init_state(memory)

        lstm_out, _ = self.lstm(x, (h0, c0))   # (B, T, hidden_dim)
        return self.fc(lstm_out)

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
        Greedy autoregressive decoding. Steps the LSTM one token at a time
        using its running hidden state, instead of recomputing the whole
        sequence each step. This is faster than the forward-based path that
        beam_search uses.
        """
        del memory_key_padding_mask

        if max_len is None:
            max_len = self.max_len

        device = memory.device
        B = memory.size(0)
        h, c = self._init_state(memory)

        prev_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
        generated = [prev_token.unsqueeze(1)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            x = self.embedding(prev_token).unsqueeze(1)   # (B, 1, embed_dim)
            x = self.embedding_proj(x)                    # (B, 1, hidden_dim)
            # no dropout at inference

            lstm_out, (h, c) = self.lstm(x, (h, c))
            logits = self.fc(lstm_out.squeeze(1))         # (B, vocab_size)

            next_token = logits.argmax(dim=-1)
            generated.append(next_token.unsqueeze(1))

            finished = finished | next_token.eq(end_idx)
            if finished.all():
                break

            prev_token = next_token

        return torch.cat(generated, dim=1)


class ModelM1(nn.Module):
    """
    Global CNN encoder + LSTM decoder.

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
        hidden_dim: int = cfg.hidden_dim,
        num_lstm_layers: int = cfg.num_lstm_layers,
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
        self.decoder = M1Decoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
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
        encoder_output, _ = self.encoder(images)   # (B, 1, encoder_dim)
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
