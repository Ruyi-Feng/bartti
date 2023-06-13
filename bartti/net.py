import torch.nn as nn
import torch
import numpy as np

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        vocab_sz, self.d_model = config.vocab_sz, config.d_model
        self._pos_embed = nn.Parameters(torch.rand(config.batch, config.d_model, config.seq_len))  # seq_len 也可以不固定
        self._embeding = nn.Embedding(vocab_sz, self.d_model, padding_idx)

    def forward(self, x):
        return self._embeding(x) + self._pos_embed(np.arrange(self.d_model))


class Bart(nn.Module):
    def __init__(self, config):
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.bart = nn.Transformer(d_model=config.d_model,
                                   nhead=config.n_heads,
                                   num_encoder_layers=config.e_layers,
                                   num_decoder_layers=config.d_layers)

    def forward(self, x):
        enc_token = self.enc_embeding(x)
        dec_token = self.dec_embeding(x)
        self.bart(enc_token, dec_token)