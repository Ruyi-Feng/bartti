import torch.nn as nn
import torch
import numpy as np

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        vocab_sz, self.d_model = config.vocab_sz, config.d_model
        self._pos_embed = nn.Parameters(torch.rand(1, config.d_model, config.seq_len))  # seq_len 也可以不固定
        self._embeding = nn.Embedding(vocab_sz, self.d_model, padding_idx)

    def forward(self, x):
        return self._embeding(x) + self._pos_embed[:, :x.size(1)]


class Bart(nn.Module):
    def __init__(self, config):
        self.embeding = Embedding(config)
        self.bart = nn.Transformer(d_model=config.d_model,
                                   nhead=config.n_heads,
                                   num_encoder_layers=config.e_layers,
                                   num_decoder_layers=config.d_layers)

    def forward(self, x1, x2):
        enc_token = self.embeding(x1)  # enc_x1 with noise
        dec_token = self.embeding(x2)  # enc_x2 with noise
        self.bart(enc_token, dec_token)
