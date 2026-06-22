"""Konfiguracja eksperymentu — wartosci domyslne dobrane pod CPU.

Wszystko sterowalne z CLI (`src/train_all.py`). Maly model + krotka sekwencja
treningowa + podprobkowanie gier, zeby pelny przebieg byl wykonalny bez GPU.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- dane ---
    data_dir: str = "data/splits"
    subsample_games: int = 1500      # ile gier (perspektyw <= 2.4x) do pretreningu; 0 = wszystkie
    train_seq_len: int = 256         # dlugosc losowego okna podczas pretreningu
    eval_seq_len: int = 512          # dlugosc przy ekstrakcji embeddingow do probe (pokrywa 99.4% kart)
    batch_size: int = 32
    num_workers: int = 0             # Windows + multiprocessing bywa kapryśny; 0 = bezpiecznie

    # --- enkoder (wspolny backbone) ---
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    # --- pretrening ---
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # --- InfoNCE / CPC ---
    cpc_steps: int = 4               # ile krokow w przod przewidujemy (k=1..K)
    cpc_temperature: float = 0.1

    # --- Barlow Twins ---
    barlow_proj_dim: int = 512
    barlow_lambda: float = 5e-3

    # --- MAE ---
    mae_mask_ratio: float = 0.30

    # --- ewaluacja / probe ---
    probe_max_iter: int = 2000

    # --- inne ---
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "results"

    def to_dict(self):
        return asdict(self)
