"""Per-step Sequence VAE with β-annealing.

Architecture:
  Encoder: causal GRU(F → h_dim=256) → per-step (mu, logvar) via Linear → latent_dim=128
  Decoder: MLP(latent_dim → F) — reconstructs all input features at each step
  Objective: recon(MSE continuous + BCE binary) + β·KL(N(mu,σ²) ‖ N(0,I))

β is linearly annealed from 0 to beta_max over warmup_epochs to avoid posterior collapse.
At eval time, probe uses deterministic mu_t (no sampling) as the per-step embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class _GRUEncoder(nn.Module):
    def __init__(self, n_features, h_dim, latent_dim):
        super().__init__()
        self.gru = nn.GRU(n_features, h_dim, batch_first=True)
        self.mu_head = nn.Linear(h_dim, latent_dim)
        self.lv_head = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return h, self.mu_head(h), self.lv_head(h)


class _MLPDecoder(nn.Module):
    def __init__(self, latent_dim, n_features, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, z):
        return self.net(z)


class SeqVAE(nn.Module):
    def __init__(self, n_features, h_dim=256, latent_dim=128, decoder_hidden=256):
        super().__init__()
        self.enc = _GRUEncoder(n_features, h_dim, latent_dim)
        self.dec = _MLPDecoder(latent_dim, n_features, decoder_hidden)
        self.latent_dim = latent_dim

    def forward(self, x, pad_mask, is_binary, beta=1.0):
        """Returns (total_loss, l_recon, l_kl, mu, lv)."""
        _, mu, lv = self.enc(x)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        else:
            z = mu
        x_hat = self.dec(z)

        keep = (~pad_mask).float().unsqueeze(-1)
        n = keep.sum() + 1e-6

        cont = ~is_binary
        l_cont = ((x_hat[..., cont] - x[..., cont]) ** 2 * keep).sum() / n if cont.any() else x.new_zeros(1).squeeze()
        l_bin = (F.binary_cross_entropy_with_logits(
            x_hat[..., is_binary], x[..., is_binary], reduction="none") * keep
                 ).sum() / n if is_binary.any() else x.new_zeros(1).squeeze()
        l_recon = l_cont + l_bin

        l_kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()) * keep).sum() / n
        return l_recon + beta * l_kl, l_recon, l_kl, mu, lv

    def get_embeddings(self, x, pad_mask):
        """Deterministic mu_t [B, T, latent_dim] — used during probe extraction."""
        self.eval()
        with torch.no_grad():
            _, mu, _ = self.enc(x)
        return mu


class VAEEncoderAdapter(nn.Module):
    """Matches probe.py interface: forward(feats, positions, pad_mask, causal) → [B,T,D]."""
    def __init__(self, vae: SeqVAE):
        super().__init__()
        self.vae = vae

    def forward(self, feats, positions, pad_mask, causal=False):
        return self.vae.get_embeddings(feats, pad_mask)


def train_vae(spec, cfg, device, log=print, beta_max=4.0, warmup_epochs=5):
    """Train SeqVAE with β-annealing. Returns (VAEEncoderAdapter, loss_history)."""
    from .data import build_sequences, load_split, SeqDataset, collate

    torch.manual_seed(cfg.seed + 1)
    is_binary = torch.tensor(spec.is_binary, dtype=torch.bool, device=device)

    ts = load_split(cfg.data_dir, "train", "timesteps")
    seqs = build_sequences(ts, spec, subsample_games=cfg.subsample_games, seed=cfg.seed)
    log(f"  [VAE] {len(seqs)} sequences")

    model = SeqVAE(spec.n_features, h_dim=256,
                   latent_dim=cfg.d_model, decoder_hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = SeqDataset(seqs, cfg.train_seq_len, mode="crop")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)

    history = []
    for ep in range(1, cfg.epochs + 1):
        beta = beta_max * min(1.0, (ep - 1) / max(warmup_epochs - 1, 1))
        model.train()
        tot, rec, kl_sum = 0.0, 0.0, 0.0
        for batch in loader:
            feats = batch["feats"].to(device)
            pad = batch["pad_mask"].to(device)
            opt.zero_grad()
            loss, l_r, l_kl, *_ = model(feats, pad, is_binary, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item(); rec += l_r.item(); kl_sum += l_kl.item()
        n = len(loader)
        row = {"epoch": ep, "beta": float(beta),
               "total": tot / n, "recon": rec / n, "kl": kl_sum / n}
        history.append(row)
        log(f"  [VAE] ep {ep}/{cfg.epochs} beta={beta:.2f} "
            f"L={tot/n:.4f} recon={rec/n:.4f} KL={kl_sum/n:.4f}")

    return VAEEncoderAdapter(model), history
