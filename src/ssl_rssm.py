"""Recurrent State Space Model (RSSM) — simplified Dreamer-style.

Latent state = (h_t, z_t):
  embed_t = Linear(x_t)                                 observation embedding
  h_t = GRUCell(h_{t-1}, [z_{t-1}, embed_t])            deterministic hidden
  z_prior_t ~ p(f_prior(h_t))                           prior (past context only)
  z_post_t  ~ q(f_post(h_t, embed_t))                   posterior (includes x_t)

Decoder: MLP([h_t, z_post_t] → x_t) — self-supervised reconstruction
Loss: L_recon + KL(posterior ‖ prior)

Two latent variants (cfg.rssm_variant):
  "gauss" — Gaussian z_t, KL vs learned prior (original).
  "cat"   — categorical z_t = n_cat × n_class one-hot (DreamerV2): straight-through
            sampling, KL balancing between prior/posterior, optional free-nats.

Probe embedding = concat(h_t, z_post_t).  z_post_t is deterministic at eval
(mu_q for Gaussian, posterior class-probabilities for categorical).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _kl_gaussian(mu_q, lv_q, mu_p, lv_p):
    """KL(N(mu_q, σ_q²) ‖ N(mu_p, σ_p²)) element-wise."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp() - 1)


def _kl_categorical(logits_q, logits_p):
    """KL(Cat(q) ‖ Cat(p)) na [.., n_cat, n_class] -> suma po klasach, [.., n_cat]."""
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(-1)


class RSSM(nn.Module):
    def __init__(self, n_features, variant="gauss", h_dim=128, z_dim=128, embed_dim=128,
                 n_cat=16, n_class=16, kl_balance=0.8, free_nats=1.0):
        super().__init__()
        assert variant in ("gauss", "cat")
        self.variant = variant
        self.h_dim = h_dim
        self.n_cat = n_cat
        self.n_class = n_class
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        # rozmiar stochastycznego stanu (rozmiar wektora z_t podawany do GRU/dekodera)
        self.z_dim = z_dim if variant == "gauss" else n_cat * n_class
        out_dim = 2 * z_dim if variant == "gauss" else n_cat * n_class

        self.obs_embed = nn.Sequential(nn.Linear(n_features, embed_dim), nn.ELU())
        self.gru = nn.GRUCell(self.z_dim + embed_dim, h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, out_dim))

        self.post_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, out_dim))

        self.decoder = nn.Sequential(
            nn.Linear(h_dim + self.z_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, n_features))

    # --- pojedynczy krok dla obu wariantow ---
    def _sample_z(self, params, deterministic):
        """Zwraca (z [B, z_dim], dist_params) dla biezacego wariantu."""
        if self.variant == "gauss":
            mu, lv = params.chunk(2, -1)
            if deterministic:
                z = mu
            else:
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            return z, (mu, lv)
        # categorical
        B = params.shape[0]
        logits = params.view(B, self.n_cat, self.n_class)
        probs = F.softmax(logits, -1)
        if deterministic:
            z = probs                      # miekkie prawdopodobienstwa (embedding)
        else:
            # próbka kategoryczna + straight-through gradient przez probs
            idx = torch.distributions.Categorical(probs=probs).sample()
            onehot = F.one_hot(idx, self.n_class).float()
            z = onehot + probs - probs.detach()
        return z.reshape(B, -1), logits

    def _rollout(self, x):
        """Sequential rollout. Returns h_seq, z_seq, post_params, prior_params
        (params: Gaussian -> (mu,lv) stacked w jednym tensorze 2*z; cat -> logits [.,n_cat,n_class])."""
        B, T, _ = x.shape
        deterministic = not self.training
        emb = self.obs_embed(x)
        h = x.new_zeros(B, self.h_dim)
        z = x.new_zeros(B, self.z_dim)
        hs, zs, post_l, prior_l = [], [], [], []
        for t in range(T):
            e = emb[:, t]
            h = self.gru(torch.cat([z, e], -1), h)

            prior_params = self.prior_net(h)
            post_params = self.post_net(torch.cat([h, e], -1))
            z, post_p = self._sample_z(post_params, deterministic)
            _, prior_p = self._sample_z(prior_params, deterministic)

            hs.append(h); zs.append(z)
            post_l.append(post_p); prior_l.append(prior_p)

        h_seq = torch.stack(hs, 1)
        z_seq = torch.stack(zs, 1)
        return h_seq, z_seq, post_l, prior_l

    def _kl(self, post_l, prior_l):
        """KL [B, T] z balancingiem (cat) / prostym KL (gauss)."""
        if self.variant == "gauss":
            mu_q = torch.stack([p[0] for p in post_l], 1)
            lv_q = torch.stack([p[1] for p in post_l], 1)
            mu_p = torch.stack([p[0] for p in prior_l], 1)
            lv_p = torch.stack([p[1] for p in prior_l], 1)
            return _kl_gaussian(mu_q, lv_q, mu_p, lv_p).sum(-1)
        # categorical: KL balancing wg DreamerV2
        lq = torch.stack(post_l, 1)        # [B, T, n_cat, n_class]
        lp = torch.stack(prior_l, 1)
        kl_lhs = _kl_categorical(lq.detach(), lp).sum(-1)   # ucz prior
        kl_rhs = _kl_categorical(lq, lp.detach()).sum(-1)   # ucz posterior
        a = self.kl_balance
        kl = a * kl_lhs + (1 - a) * kl_rhs
        if self.free_nats > 0:
            kl = torch.clamp(kl, min=self.free_nats)
        return kl

    def forward(self, x, pad_mask, is_binary):
        h_seq, z_seq, post_l, prior_l = self._rollout(x)
        x_hat = self.decoder(torch.cat([h_seq, z_seq], -1))

        keep = (~pad_mask).float()
        keepf = keep.unsqueeze(-1)
        n = keepf.sum() + 1e-6

        cont = ~is_binary
        l_cont = ((x_hat[..., cont] - x[..., cont]) ** 2 * keepf).sum() / n if cont.any() else x.new_zeros(1).squeeze()
        l_bin = (F.binary_cross_entropy_with_logits(
            x_hat[..., is_binary], x[..., is_binary], reduction="none") * keepf
                 ).sum() / n if is_binary.any() else x.new_zeros(1).squeeze()
        l_recon = l_cont + l_bin

        kl = self._kl(post_l, prior_l)                      # [B, T]
        l_kl = (kl * keep).sum() / (keep.sum() + 1e-6)
        return l_recon + l_kl, l_recon, l_kl, h_seq, z_seq

    def get_embeddings(self, x, pad_mask):
        """Returns concat(h_t, z_post_t) [B, T, h_dim + z_dim]. Deterministyczne."""
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
    """Train RSSM (wariant z cfg.rssm_variant). Returns (RSSMEncoderAdapter, loss_history)."""
    from .data import build_sequences, load_split, SeqDataset, collate

    torch.manual_seed(cfg.seed + 2)
    is_binary = torch.tensor(spec.is_binary, dtype=torch.bool, device=device)

    ts = load_split(cfg.data_dir, "train", "timesteps")
    seqs = build_sequences(ts, spec, subsample_games=cfg.subsample_games, seed=cfg.seed)
    log(f"  [RSSM:{cfg.rssm_variant}] {len(seqs)} sequences")

    model = RSSM(spec.n_features, variant=cfg.rssm_variant,
                 h_dim=cfg.rssm_h_dim, z_dim=cfg.rssm_z_dim, embed_dim=cfg.rssm_embed_dim,
                 n_cat=cfg.rssm_n_cat, n_class=cfg.rssm_n_class,
                 kl_balance=cfg.rssm_kl_balance, free_nats=cfg.rssm_free_nats).to(device)
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
        log(f"  [RSSM:{cfg.rssm_variant}] ep {ep}/{cfg.epochs} "
            f"L={tot/n:.4f} recon={rec/n:.4f} KL={kl_sum/n:.4f}")

    return RSSMEncoderAdapter(model), history
