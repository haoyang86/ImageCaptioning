import torch
import torch.nn as nn
from models.attention import Attention
from configs import m_config as cfg
from typing import Optional
import math

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with an optional attention mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        hidden_dim: int = cfg.hidden_dim,
        dropout: float = cfg.dropout,
        max_len: int = cfg.max_len,
        encoder_dim: int = cfg.encoder_dim,

        num_layers: int = cfg.num_lstm_layers,

        # attention options
        use_attention: bool = cfg.LSTM_Attention,
        attention_dim: int = cfg.attention_dim,

        # embedding options
        embedding_type: str = cfg.embedding_type,
        embedding_dim: int = cfg.embedding_dim,
        pretrained_embedding_matrix: Optional[torch.Tensor] = None,

    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.use_attention = use_attention
        self.attention_dim = attention_dim

        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

        # --------------------------------------------------
        # Word embedding
        # --------------------------------------------------
        if embedding_type == "random":
            # Random trainable embedding directly in d_model dimension
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=hidden_dim,
                padding_idx=pad_idx,
            )

            self.embedding_proj = nn.Identity()
            self.embedding_output_dim = hidden_dim

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
            if embedding_dim != hidden_dim:
                self.embedding_proj = nn.Linear(
                    embedding_dim,
                    hidden_dim,
                )
            else:
                self.embedding_proj = nn.Identity()

            self.embedding_output_dim = hidden_dim

        else:
            raise ValueError(
                f"Unsupported embedding_type: {embedding_type}"
            )

        # --------------------------------------------------

        self.dropout = nn.Dropout(dropout)


        # ------------------------------------------------------------------
        # Architecture Routing
        # ------------------------------------------------------------------

        if self.use_attention:
            self.attention = Attention(encoder_dim, hidden_dim, attention_dim)

            # For attention, we need a list of independent LSTMCells
            self.lstm_cells = nn.ModuleList()
            for i in range(num_layers):
                # Layer 0 takes the word embedding + attention context
                if i == 0:
                    input_size = hidden_dim + encoder_dim
                # Subsequent layers just take the hidden state of the layer below them
                else:
                    input_size = hidden_dim
                self.lstm_cells.append(nn.LSTMCell(input_size, hidden_dim))
        else:
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )

        # --------------------------------------------------


        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._reset_parameters()

        # --------------------------------------------------

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
            self.fc.weight
        )
        nn.init.zeros_(
            self.fc.bias
        )

        # --------------------------------------------------

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
                CNN global features, shape (B, 1, encoder_dim)

        Returns:
            logits:
                Shape (B, T, vocab_size)
        """

        batch_size, seq_len = tgt_ids.shape

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------
        embeddings = self.embedding(tgt_ids)

        # If pretrained embedding is 300d, project to hidden_dim
        embeddings = self.embedding_proj(embeddings)

        # apply dropout
        embeddings = self.dropout(embeddings)

        # initialize hidden states h0 and c0 from the image features
        features = memory.mean(dim=1) # (b, encoder_dim)
        h_init = self.init_h(features) # (B, hidden_dim)
        c_init = self.init_c(features) # (B, hidden_dim)

        if self.use_attention:
            return self._forward_with_attention(memory, embeddings, h_init, c_init, batch_size, seq_len)
        else:
            return self._forward_without_attention(embeddings, h_init, c_init)

    def _forward_with_attention(self, memory, embeddings, h_init, c_init, batch_size, seq_len):
        device = memory.device

        predictions = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        # alphas = torch.zeros(batch_size, seq_len, memory.size(1), device=device)

        # Create independent memory states for EACH layer
        h_states = [h_init.clone() for _ in range(self.num_layers)]
        c_states = [c_init.clone() for _ in range(self.num_layers)]

        for t in range(seq_len):
            # Calculate attention using the TOP layer's hidden state (h_states[-1])
            context, alpha = self.attention(memory, h_states[-1])

            # Input to the first layer (Layer 0)
            current_input = torch.cat([embeddings[:, t, :], context], dim=1)

            # Pass data up through the stacked LSTM layers
            for layer_idx in range(self.num_layers):
                h, c = self.lstm_cells[layer_idx](
                    current_input,
                    (h_states[layer_idx], c_states[layer_idx])
                )

                # Update memory for this specific layer
                h_states[layer_idx] = h
                c_states[layer_idx] = c

                # The output (h) of this layer becomes the input to the next layer
                current_input = h

            # Predict the word using the hidden state of the very top layer
            # output = self.fc(self.dropout(h_states[-1]))
            output = self.fc(h_states[-1])

            predictions[:, t, :] = output
            # alphas[:, t, :] = alpha

        return predictions

    def _forward_without_attention(self, embeddings, h_init, c_init):

        # Repeat the inital visual context for every layer in the stack
        # Transforms shape from (B, hidden_dim) to (num_layers, B, hidden_dim)
        h = h_init.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, B, hidden_dim)
        c = c_init.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, B, hidden_dim)

        lstm_out, _ = self.lstm(embeddings, (h, c))
        predictions = self.fc(lstm_out)

        return predictions

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
