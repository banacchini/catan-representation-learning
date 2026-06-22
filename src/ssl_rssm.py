"""Recurrent State Space Model (RSSM) — simplified Dreamer-style.

Latent state = (h_t, z_t):
  embed_t = Linear(x_t)                                 observation embedding
  h_t = GRUCell(h_{t-1}, [z_{t-1}, embed_t])            deterministic hidden
  z_prior_t ~ N(f_prior(h_t))                           prior (past context only)
  z_post_t  ~ N(f_post(h_t, embed_t))                   posterior (includes x_t)

Decoder: MLP([h_t, z_post_t] → x_t) — self-supervised reconstruction
Loss: L_recon + KL(posterior ‖ prior)

Key difference from VAE: KL is measured against the learned prior p(z|h)
rather than a fixed N(0,I), so the prior learns to predict next-state distributions.

Probe embedding = concat(h_t, z_post_t) ∈ R^{h_dim + z_dim = 256}.
At eval time, x_t is observed so we use the posterior z_post_t.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _kl_gaussian(mu_q, lv_q, mu_p, lv_p):
    """KL(N(mu_q, σ_q²) ‖ N(mu_p, σ_p²)) element-wise."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp() - 1)


class RSSM(nn.Module):
    def __init__(self, n_features, h_dim=128, z_dim=128, embed_dim=128):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.obs_embed = nn.Sequential(nn.Linear(n_features, embed_dim), nn.ELU())
        self.gru = nn.GRUCell(z_dim + embed_dim, h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim))

        self.post_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, n_features))

    def _rollout(self, x):
        """Sequential rollout.
        Returns h_seq, z_seq, mu_q, lv_q, mu_p, lv_p — all [B, T, dim]."""
        B, T, _ = x.shape
        emb = self.obs_embed(x)
        h = x.new_zeros(B, self.h_dim)
        z = x.new_zeros(B, self.z_dim)
        hs, zs, mq_l, lq_l, mp_l, lp_l = [], [], [], [], [], []
        for t in range(T):
            e = emb[:, t]
            h = self.gru(torch.cat([z, e], -1), h)

            p_out = self.prior_net(h).chunk(2, -1)
            mu_p, lv_p = p_out
            q_out = self.post_net(torch.cat([h, e], -1)).chunk(2, -1)
            mu_q, lv_q = q_out

            if self.training:
                z = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp()
            else:
                z = mu_q

            hs.append(h); zs.append(z)
            mq_l.append(mu_q); lq_l.append(lv_q)
            mp_l.append(mu_p); lp_l.append(lv_p)

        return (torch.stack(hs, 1), torch.stack(zs, 1),
                torch.stack(mq_l, 1), torch.stack(lq_l, 1),
                torch.stack(mp_l, 1), torch.stack(lp_l, 1))

    def forward(self, x, pad_mask, is_binary):
        h_seq, z_seq, mu_q, lv_q, mu_p, lv_p = self._rollout(x)
        x_hat = self.decoder(torch.cat([h_seq, z_seq], -1))

        keep = (~pad_mask).float().unsqueeze(-1)
        n = keep.sum() + 1e-6

        cont = ~is_binary
        l_cont = ((x_hat[..., cont] - x[..., cont]) ** 2 * keep).sum() / n if cont.any() else x.new_zeros(1).squeeze()
        l_bin = (F.binary_cross_entropy_with_logits(
            x_hat[..., is_binary], x[..., is_binary], reduction="none") * keep
                 ).sum() / n if is_binary.any() else x.new_zeros(1).squeeze()
        l_recon = l_cont + l_bin

        l_kl = (_kl_gaussian(mu_q, lv_q, mu_p, lv_p) * keep).sum() / n
        return l_recon + l_kl, l_recon, l_kl, h_seq, z_seq

    def get_embeddings(self, x, pad_mask):
        """Returns concat(h_t, z_post_t) [B, T, h_dim + z_dim]."""
        self.eval()
        with torch.no_grad():
            h_seq, z_seq, *_ = self._rollout(x)
        return torch.cat([h_seq, z_seq], -1)


class RSSMEncoderAdapter(nn.Module):
    """Matches probe.py interface: forward(feats, positions, pad_mask, causal) → [B,T,D]."""
    def __init__(self, rssm: RSSM):
        super().__init__()
        self.rssm = rssm

    def forward(self, feats, positions, pad_mask, causal=False):
        return self.rssm.get_embeddings(feats, pad_mask)


def train_rssm(spec, cfg, device, log=print, seq_len_override=None):
    """Train RSSM. Returns (RSSMEncoderAdapter, loss_history)."""
    from .data import build_sequences, load_split, SeqDataset, collate

    torch.manual_seed(cfg.seed + 2)
    is_binary = torch.tensor(spec.is_binary, dtype=torch.bool, device=device)

    ts = load_split(cfg.data_dir, "train", "timesteps")
    seqs = build_sequences(ts, spec, subsample_games=cfg.subsample_games, seed=cfg.seed)
    log(f"  [RSSM] {len(seqs)} sequences")

    model = RSSM(spec.n_features, h_dim=128, z_dim=128, embed_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    seq_len = seq_len_override if seq_len_override else cfg.train_seq_len
    ds = SeqDataset(seqs, seq_len, mode="crop")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)

    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, rec, kl_sum = 0.0, 0.0, 0.0
        for batch in loader:
            feats = batch["feats"].to(device)
            pad = batch["pad_mask"].to(device)
            opt.zero_grad()
            loss, l_r, l_kl, *_ = model(feats, pad, is_binary)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item(); rec += l_r.item(); kl_sum += l_kl.item()
        n = len(loader)
        row = {"epoch": ep, "total": tot / n, "recon": rec / n, "kl": kl_sum / n}
        history.append(row)
        log(f"  [RSSM] ep {ep}/{cfg.epochs} "
            f"L={tot/n:.4f} recon={rec/n:.4f} KL={kl_sum/n:.4f}")

    return RSSMEncoderAdapter(model), history
