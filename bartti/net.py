import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.d_model = config.d_model
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._pos_embed = nn.Parameter(torch.rand(1, config.max_len, 1))  # 对于每个token的pos_embed是一样的
        # cov input x: [batch, seq_len, c_in]
        self._embeding = nn.Conv1d(in_channels=config.c_in, out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._frm_embed = nn.Parameter(torch.rand(5, 1))

    def forward(self, x, mark):
        for i in mark:
            if frm_mark is None:
                frm_mark = torch.ones((1, i, 1)).float()
                continue
            else:
                frm_mark = torch.cat([frm_mark, torch.ones(1, i, 1).float()], dim=1)
        # ! 这里需要检查维度是否正确
        return self._embeding(x.permute(0, 2, 1)).transpose(1, 2) + \
            self._pos_embed.data[:, :x.shape[1], :] + frm_mark

class Bart(nn.Module):
    def __init__(self, config):
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.bart = nn.Transformer(d_model=config.d_model,
                                   nhead=config.n_heads,
                                   num_encoder_layers=config.e_layers,
                                   num_decoder_layers=config.d_layers)

    def forward(self, enc_x, enc_mark, dec_x, dec_mark):
        enc_token = self.enc_embeding(enc_x, enc_mark)
        dec_token = self.dec_embeding(dec_x, dec_mark)
        self.bart(enc_token, dec_token)
