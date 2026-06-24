"""Per-step Sequence VAE with β-annealing (LSTM encoder).

Two latent variants (cfg.vae_variant):
  "gauss" — per-step (mu, logvar) → z ~ N(mu,σ²); KL(q ‖ N(0,I)).
  "cat"   — per-step categorical z = n_cat × n_class one-hot (straight-through);
            KL(q ‖ Uniform). Stały, nieuczony prior — kluczowa różnica vs RSSM-cat,
            którego prior p(z|h) jest uczony.

  Encoder: causal LSTM(F → h_dim) → głowica latentu
  Decoder: MLP(z → F) — reconstructs all input features at each step
  Objective: recon(MSE continuous + BCE binary) + β·KL

β is linearly annealed from 0 to beta_max over warmup_epochs to avoid posterior collapse.
At eval time, probe uses a deterministic embedding (mu_t dla gauss; prawdopodobieństwa
klas dla cat).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ssl_rssm import _kl_categorical


class _LSTMEncoder(nn.Module):
    def __init__(self, n_features, h_dim, latent_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, h_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.mu_head = nn.Linear(h_dim, latent_dim)
        self.lv_head = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
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
    def __init__(self, n_features, variant="gauss", h_dim=256, latent_dim=128,
                 n_cat=16, n_class=16, decoder_hidden=256, num_layers=1, dropout=0.0):
        super().__init__()
        assert variant in ("gauss", "cat")
        self.variant = variant
        self.n_cat = n_cat
        self.n_class = n_class
        if variant == "gauss":
            self.enc = _LSTMEncoder(n_features, h_dim, latent_dim, num_layers, dropout)
            self.z_dim = latent_dim
        else:
            self.lstm = nn.LSTM(n_features, h_dim, num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.logit_head = nn.Linear(h_dim, n_cat * n_class)
            self.z_dim = n_cat * n_class
        self.latent_dim = self.z_dim
        self.dec = _MLPDecoder(self.z_dim, n_features, decoder_hidden)

    def _cat_latent(self, x, deterministic):
        """Zwraca (z [B,T,z_dim], logits [B,T,n_cat,n_class], emb_det [B,T,z_dim])."""
        B, T, _ = x.shape
        h, _ = self.lstm(x)
        logits = self.logit_head(h).view(B, T, self.n_cat, self.n_class)
        probs = F.softmax(logits, -1)
        emb_det = probs.reshape(B, T, -1)
        if deterministic:
            z = emb_det
        else:
            idx = torch.distributions.Categorical(probs=probs).sample()
            onehot = F.one_hot(idx, self.n_class).float()
            z = (onehot + probs - probs.detach()).reshape(B, T, -1)
        return z, logits, emb_det

    def forward(self, x, pad_mask, is_binary, beta=1.0):
        """Returns (total_loss, l_recon, l_kl, emb_det). emb_det: deterministyczna
        reprezentacja per-krok (mu dla gauss, prawdopodobieństwa klas dla cat)."""
        keep = (~pad_mask).float().unsqueeze(-1)
        n = keep.sum() + 1e-6
        if self.variant == "gauss":
            _, mu, lv = self.enc(x)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if self.training else mu
            emb_det = mu
            l_kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()) * keep).sum() / n
        else:
            z, logits, emb_det = self._cat_latent(x, deterministic=not self.training)
            # KL(q ‖ Uniform) = _kl_categorical(logits_q, logity_jednostajne=0)
            kl = _kl_categorical(logits, torch.zeros_like(logits)).sum(-1)   # [B,T]
            keep2 = (~pad_mask).float()
            l_kl = (kl * keep2).sum() / (keep2.sum() + 1e-6)

        x_hat = self.dec(z)
        cont = ~is_binary
        l_cont = ((x_hat[..., cont] - x[..., cont]) ** 2 * keep).sum() / n if cont.any() else x.new_zeros(1).squeeze()
        l_bin = (F.binary_cross_entropy_with_logits(
            x_hat[..., is_binary], x[..., is_binary], reduction="none") * keep
                 ).sum() / n if is_binary.any() else x.new_zeros(1).squeeze()
        l_recon = l_cont + l_bin
        return l_recon + beta * l_kl, l_recon, l_kl, emb_det

    def get_embeddings(self, x, pad_mask):
        """Deterministyczny embedding per-krok [B, T, z_dim] — do ekstrakcji probe."""
        self.eval()
        with torch.no_grad():
            if self.variant == "gauss":
                _, mu, _ = self.enc(x)
                return mu
            _, _, emb_det = self._cat_latent(x, deterministic=True)
            return emb_det


class VAEEncoderAdapter(nn.Module):
    """Matches probe.py interface: forward(feats, positions, pad_mask, causal) → [B,T,D]."""
    def __init__(self, vae: SeqVAE):
        super().__init__()
        self.vae = vae

    def forward(self, feats, positions, pad_mask, causal=False):
        return self.vae.get_embeddings(feats, pad_mask)


def train_vae(spec, cfg, device, log=print, beta_max=None, warmup_epochs=None):
    """Train SeqVAE (LSTM encoder) with β-annealing.
    Hyperparametry brane z cfg (vae_*); beta_max/warmup_epochs mozna nadpisac argumentem.
    Returns (VAEEncoderAdapter, loss_history)."""
    from .data import build_sequences, load_split, SeqDataset, collate

    beta_max = cfg.vae_beta_max if beta_max is None else beta_max
    warmup_epochs = cfg.vae_warmup_epochs if warmup_epochs is None else warmup_epochs

    torch.manual_seed(cfg.seed + 1)
    is_binary = torch.tensor(spec.is_binary, dtype=torch.bool, device=device)

    ts = load_split(cfg.data_dir, "train", "timesteps")
    seqs = build_sequences(ts, spec, subsample_games=cfg.subsample_games, seed=cfg.seed)
    zdesc = (f"z={cfg.vae_latent_dim}" if cfg.vae_variant == "gauss"
             else f"cat={cfg.vae_n_cat}x{cfg.vae_n_class}")
    log(f"  [VAE:{cfg.vae_variant}] {len(seqs)} sequences  "
        f"(LSTM h={cfg.vae_h_dim} {zdesc} layers={cfg.vae_num_layers})")

    model = SeqVAE(spec.n_features, variant=cfg.vae_variant, h_dim=cfg.vae_h_dim,
                   latent_dim=cfg.vae_latent_dim, n_cat=cfg.vae_n_cat,
                   n_class=cfg.vae_n_class, decoder_hidden=256,
                   num_layers=cfg.vae_num_layers, dropout=cfg.vae_dropout).to(device)
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
