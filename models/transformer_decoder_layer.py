import torch
import torch.nn as nn
from typing import Optional


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer for image captioning.

    Structure:
        1. Masked self-attention over target sequence
        2. Cross-attention over encoder memory (e.g. CNN spatial features)
        3. Position-wise feed-forward network

    Each sub-layer uses:
        x = LayerNorm(x + Dropout(SubLayer(x)))

    Args:
        d_model: embedding dimension
        num_heads: number of attention heads
        dim_feedforward: hidden dimension in FFN
        dropout: dropout probability
        activation: activation function in FFN ("relu" or "gelu")
        batch_first: if True, input shape is (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.batch_first = batch_first

        # 1) masked self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        # 2) cross-attention: query from target, key/value from memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        # 3) feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Dropouts
        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.dropout_activation = nn.Dropout(dropout)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:
                target embeddings, shape (B, T, D) if batch_first=True
            memory:
                encoder output / CNN spatial features, shape (B, S, D)
            tgt_mask:
                causal mask for target self-attention.
                shape can be (T, T)
                Usually upper-triangular with True or -inf above diagonal.
            tgt_key_padding_mask:
                target padding mask, shape (B, T)
                True means "ignore this position"
            memory_key_padding_mask:
                memory padding mask, shape (B, S)
                True means "ignore this memory position"

        Returns:
            output tensor of shape (B, T, D)
        """

        # --------------------------------------------------
        # 1) Masked self-attention
        # Query, Key, Value all come from tgt
        # --------------------------------------------------
        self_attn_output, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout_self_attn(self_attn_output)
        tgt = self.norm1(tgt)

        # --------------------------------------------------
        # 2) Cross-attention
        # Query from tgt, Key/Value from memory
        # --------------------------------------------------
        cross_attn_output, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout_cross_attn(cross_attn_output)
        tgt = self.norm2(tgt)

        # --------------------------------------------------
        # 3) Feed-forward network
        # --------------------------------------------------
        ffn_output = self.linear2(
            self.dropout_activation(
                self.activation(self.linear1(tgt))
            )
        )
        tgt = tgt + self.dropout_ffn(ffn_output)
        tgt = self.norm3(tgt)

        return tgt