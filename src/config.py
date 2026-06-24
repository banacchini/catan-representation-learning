"""Konfiguracja eksperymentu — wartosci domyslne dobrane pod CPU.

Wszystko sterowalne z CLI (`src/train_all.py`). Maly model + krotka sekwencja
treningowa + podprobkowanie gier, zeby pelny przebieg byl wykonalny bez GPU.
"""
from dataclasses import dataclass, field, asdict
import torch



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

    # --- VAE (enkoder LSTM, per-step latent + beta-annealing) ---
    vae_variant: str = "gauss"       # "gauss" (KL vs N(0,I)) | "cat" (KL vs jednostajny)
    vae_h_dim: int = 256             # rozmiar stanu ukrytego LSTM
    vae_latent_dim: int = 128        # rozmiar latentu z_t Gaussa (= embedding probe)
    vae_n_cat: int = 16              # liczba zmiennych kategorycznych — wariant "cat"
    vae_n_class: int = 16            # liczba klas na zmienna — wariant "cat"
    vae_num_layers: int = 1          # liczba warstw LSTM
    vae_dropout: float = 0.0         # dropout miedzy warstwami LSTM (gdy num_layers > 1)
    vae_beta_max: float = 4.0        # maksymalna waga KL po annealingu
    vae_warmup_epochs: int = 5       # liniowy annealing beta: 0 -> beta_max

    # --- RSSM (Dreamer-style) ---
    rssm_variant: str = "gauss"      # "gauss" | "cat" (kategoryczny, DreamerV2)
    rssm_h_dim: int = 128            # deterministyczny stan h_t (GRUCell)
    rssm_z_dim: int = 128            # stochastyczny latent (Gaussian) — wariant "gauss"
    rssm_embed_dim: int = 128        # embedding obserwacji
    rssm_n_cat: int = 16             # liczba zmiennych kategorycznych — wariant "cat"
    rssm_n_class: int = 16           # liczba klas na zmienna — wariant "cat"
    rssm_kl_balance: float = 0.8     # KL balancing (DreamerV2): waga strony posteriora
    rssm_free_nats: float = 1.0      # free-bits: dolne obciecie KL (0 = wylaczone)

    # --- supervised VAE/RSSM (warianty A/B/C; trening+ewaluacja na startach tur) ---
    # A = zamrozony enkoder SSL + glowica MLP; B = end-to-end CE + KL (VIB);
    # C = end-to-end CE + KL + rekonstrukcja (multi-task).
    sup_mode: str = ""               # "", "A", "B", "C"
    sup_recon_weight: float = 1.0    # waga rekonstrukcji (tylko C)
    sup_kl_weight: float = 1.0       # waga KL (bottleneck wariacyjny; B i C)
    sup_head_hidden: int = 128       # rozmiar warstwy ukrytej glowicy MLP (wariant A)

    # --- ewaluacja / probe ---
    probe_max_iter: int = 2000

    # --- inne ---
    model: str = "vae"               # "vae" | "rssm_gauss" | "rssm_cat" (dla search/final)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "results"

    def to_dict(self):
        return asdict(self)
