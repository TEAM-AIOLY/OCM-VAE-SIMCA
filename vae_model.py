import torch
from torch import nn
import numpy as np

class ConvVAE1D(nn.Module):
    def __init__(
        self,
        input_length,
        latent_dim,
        mean,
        std,
        conv_blocks=3,
        n_filters=32,
        kernel_size=9,
        stride=2,
        hidden_fc=256,
        activation="elu",
        dropout=0.0,
        use_batchnorm=True,
        beta=1.0,
    ):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.register_buffer("threshold", torch.tensor(0.0))

        act = nn.ELU if activation == "elu" else nn.GELU
        padding = kernel_size // 2

        # Encoder conv
        enc_blocks = []
        in_ch = 1
        out_len = input_length
        filters = n_filters
        for b in range(conv_blocks):
            stride_b = 1 if b == 0 else stride
            enc_blocks.append(nn.Conv1d(in_ch, filters, kernel_size, stride=stride_b, padding=padding))
            if use_batchnorm:
                enc_blocks.append(nn.BatchNorm1d(filters))
            enc_blocks.append(act())
            if dropout > 0:
                enc_blocks.append(nn.Dropout(dropout))
            in_ch = filters
            filters = min(filters * 2, 1024)
            out_len = (out_len + 2*padding - (kernel_size-1) -1)//stride_b + 1
        self.encoder_conv = nn.Sequential(*enc_blocks)
        self._enc_out_channels = in_ch
        self._enc_out_length = out_len

        # Fully connected
        fc_in = self._enc_out_channels * self._enc_out_length
        self.fc = nn.Sequential(nn.Linear(fc_in, hidden_fc), act(), nn.Dropout(dropout) if dropout>0 else nn.Identity())
        self.fc_mu = nn.Linear(hidden_fc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_fc, latent_dim)

        # Decoder
        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, hidden_fc), act(), nn.Dropout(dropout) if dropout>0 else nn.Identity(),
                                    nn.Linear(hidden_fc, fc_in), act())

        dec_blocks = []
        in_ch = self._enc_out_channels
        filters = in_ch
        for b in range(conv_blocks):
            next_filters = max(filters//2, n_filters)
            stride_b = stride if b < conv_blocks-1 else 1
            dec_blocks.append(nn.ConvTranspose1d(filters, next_filters, kernel_size, stride=stride_b, padding=padding, output_padding=(stride_b-1)))
            if use_batchnorm:
                dec_blocks.append(nn.BatchNorm1d(next_filters))
            dec_blocks.append(act())
            if dropout > 0:
                dec_blocks.append(nn.Dropout(dropout))
            filters = next_filters
        dec_blocks.append(nn.Conv1d(filters, 1, kernel_size=1))
        self.decoder_conv = nn.Sequential(*dec_blocks)

        self.register_buffer("spec_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("spec_std", torch.tensor(std, dtype=torch.float32))

        # Latent stats
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_cov_inv", torch.eye(latent_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)

    def decode(self, z):
        h = self.fc_dec(z)
        B = h.size(0)
        h = h.view(B, self._enc_out_channels, self._enc_out_length)
        x_rec = self.decoder_conv(h).squeeze(1)
        if x_rec.shape[-1] > self.input_length:
            x_rec = x_rec[..., :self.input_length]
        elif x_rec.shape[-1] < self.input_length:
            pad = x_rec.new_zeros(B, self.input_length - x_rec.shape[-1])
            x_rec = torch.cat([x_rec, pad], dim=1)
        return x_rec

    def forward(self, x):
        x_std = (x - self.spec_mean) / self.spec_std
        mu, logvar = self.encode(x_std)
        z = self.reparameterize(mu, logvar)
        x_rec_std = self.decode(z)
        x_rec = x_rec_std * self.spec_std + self.spec_mean
        return x_rec, mu, logvar
