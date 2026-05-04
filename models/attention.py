#additive attension (Bahdanau - style) will be used
#encoder feature dimension is 512, and decoder hidden state dimension is 512 as well
#the output of attention is A. attention weights alpha (B, 49), B. attention context vector context (B,512)
#use softmax to normalized attention scores
#scoring structure: project encoder feature and decoder hidden state to the same attention space, then combine them, then apply nonliearity then map to a scalar score per region
#attention hidden dimension is 512
#no dropout will be use

import numpy as np
import torch
from torch import nn

class Attention (nn.Module):
    """
    the attention machnism in the model M2
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        #layers needed in additive attention
        self.encoder_projection_layer = nn.Linear(encoder_dim, attention_dim)

        self.decoder_projection_layer = nn.Linear(decoder_dim, attention_dim)

        self.score_layer = nn.Linear(attention_dim, 1)

    def forward(self, encoder_output, decoder_hidden):
        encoder_proj = self.encoder_projection_layer(encoder_output)          # (B, 49, attention_dim)
        decoder_proj = self.decoder_projection_layer(decoder_hidden)          # (B, attention_dim)

        att = torch.tanh(encoder_proj + decoder_proj.unsqueeze(1))           # (B, 49, attention_dim)
        scores = self.score_layer(att).squeeze(2)                            # (B, 49)

        alpha = torch.softmax(scores, dim=1)                                 # (B, 49)
        context = (encoder_output * alpha.unsqueeze(2)).sum(dim=1)           # (B, encoder_dim)

        return context, alpha
