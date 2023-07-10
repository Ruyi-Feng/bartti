import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.d_model = config.d_model
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self._pos_embed = nn.Parameter(torch.rand(1, config.max_seq_len, 1))  # 对于每个token的pos_embed是一样的
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=config.c_in, out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._frm_embed = nn.Embedding(config.frm_embed, config.d_model, padding_idx=0)
        self._car_embed = nn.Embedding(config.id_embed, config.d_model, padding_idx=0)

    def forward(self, x_group):
        """
        x_group: batch, seq_len, c_in
        """
        x = x_group[:, :, 2:]
        f = x_group[:, :, 0]
        c = x_group[:, :, 1]
        # print("c_max", c.max())
        # print("f_max", f.max())
        # print("car", c.shape)
        c_n = self._car_embed(c.int())
        # print("car_emb", c_n.shape)
        # print("frm", f.shape)
        f_n = self._frm_embed(f.int())
        # print("frm_emb", f_n.shape)
        # print("token", x.shape)
        t_n = self._token_embed(x.permute(0, 2, 1)).transpose(1, 2)
        # print("token_emb", t_n.shape)
        return t_n + f_n + c_n

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
        self.d_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=config.c_out,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, dec_x, gt_x, infer=False):
        enc_token = self.enc_embeding(enc_x)
        dec_token = self.dec_embeding(dec_x)
        tgt_mask = None
        if not infer:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_token.size(1)).to(enc_token.device)
        output = self.bart(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2), tgt_mask=tgt_mask).permute(1, 2, 0)
        outputs = self.d_reduction(output).permute(0, 2, 1)  # -> batch, seq_len, d_model
        loss = self.criterion(outputs, gt_x)
        return outputs, loss
