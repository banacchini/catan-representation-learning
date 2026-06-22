"""Wspolny enkoder Transformer + glowice dla obiektywow SSL.

Pozycyjne kodowanie SINUSOIDALNE indeksowane prawdziwym action_index — dzieki
temu model wytrenowany na oknach dlugosci 256 dziala na pelnych sekwencjach
(do ~673 krokow) przy ekstrakcji embeddingow do probe.
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, positions):
        # positions: [B, T] (long) -> [B, T, d_model]
        positions = positions.clamp(max=self.pe.size(0) - 1)
        return self.pe[positions]


class SeqTransformerEncoder(nn.Module):
    """Wspolny backbone. Zwraca kontekstowe embeddingi per krok h_t [B,T,d]."""

    def __init__(self, n_features, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, positions, pad_mask, causal=False):
        # feats [B,T,F], positions [B,T], pad_mask [B,T] (True=pad)
        x = self.input_proj(feats) + self.pos_enc(positions)
        attn_mask = None
        if causal:
            T = feats.size(1)
            # maska boolowska (True = krok zabroniony) — spojny typ z pad_mask
            attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=feats.device), diagonal=1)
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.norm(h)

    @staticmethod
    def masked_mean(h, pad_mask):
        """Srednia po krokach z pominieciem paddingu -> embedding okna [B,d]."""
        keep = (~pad_mask).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)


class ProjectionHead(nn.Module):
    """MLP projektor (Barlow Twins / kontrastywne)."""

    def __init__(self, d_in, d_hidden, d_out, bn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class MAEDecoder(nn.Module):
    """Lekki dekoder rekonstruujacy cechy zamaskowanych krokow."""

    def __init__(self, d_model, n_features, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, h):
        return self.net(h)
