import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.d_model = config.d_model
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._pos_embed = nn.Parameter(torch.rand(1, config.max_seq_len, 1))  # 对于每个token的pos_embed是一样的
        # cov input x: [batch, seq_len, c_in]
        self._embeding = nn.Conv1d(in_channels=config.c_in, out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._frm_embed = nn.Parameter(torch.rand(5, 1))

    def forward(self, x_pack, mark):
        x, mk = x_pack
        mks = mark[0]
        for i in range(len(mks)):
            if i == 0:
                mk[0, :mks[i], 0] = self._frm_embed[i].item()
                continue
            mk[0, sum(mks[:i-1]):sum(mks[:i]), 0] = self._frm_embed[i].item()
        # ! 这里需要检查维度是否正确
        return self._embeding(x.permute(0, 2, 1)).transpose(1, 2) + self._pos_embed.data[:, :x.shape[1], :] + mk

class Bart(nn.Module):
    def __init__(self, config):
        super(Bart, self).__init__()
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.bart = nn.Transformer(d_model=config.d_model,
                                   nhead=config.n_heads,
                                   num_encoder_layers=config.e_layers,
                                   num_decoder_layers=config.d_layers,
                                   activation='gelu')
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=config.c_in,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, enc_mark, dec_x, dec_mark, gt_x, infer=False):
        enc_token = self.enc_embeding(enc_x, enc_mark)
        dec_token = self.dec_embeding(dec_x, dec_mark)
        tgt_mask = None
        if not infer:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_token.size(1)).to(enc_token.device)
        output = self.bart(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2), tgt_mask=tgt_mask).permute(1, 2, 0)
        outputs = self.d_reduction(output).permute(0, 2, 1)  # -> batch, seq_len, d_model
        loss = self.criterion(outputs, gt_x)
        return outputs, loss
