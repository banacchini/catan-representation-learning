"""Pretrening InfoNCE / CPC (Contrastive Predictive Coding) — obiektyw czasowy.

Enkoder CAUSAL liczy kontekst c_t (tylko z przeszlosci). Predyktory liniowe W_k
przewiduja reprezentacje przyszlych krokow z_{t+k} (k=1..K). Strata InfoNCE z
negatywami w batchu: model ma odroznic prawdziwy przyszly krok od innych krokow.
Uczy DYNAMIKI gry (jak stan ewoluuje w czasie).
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fun

from .models import SeqTransformerEncoder


class _TargetEncoder(nn.Module):
    """Bezkontekstowa projekcja cech kroku -> z_t (cel predykcji)."""
    def __init__(self, n_features, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

    def forward(self, feats):
        return self.net(feats)


def train_infonce(encoder: SeqTransformerEncoder, loader, spec, cfg, device,
                  log=print, max_anchors=256):
    K = cfg.cpc_steps
    tau = cfg.cpc_temperature
    target_enc = _TargetEncoder(spec.n_features, encoder.d_model).to(device)
    predictors = nn.ModuleList(
        [nn.Linear(encoder.d_model, encoder.d_model) for _ in range(K)]).to(device)
    params = list(encoder.parameters()) + list(target_enc.parameters()) \
        + list(predictors.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        encoder.train(); target_enc.train(); predictors.train()
        ep_loss, n_batches = 0.0, 0
        for batch in loader:
            feats = batch["feats"].to(device)
            positions = batch["positions"].to(device)
            pad = batch["pad_mask"].to(device)
            B, T, _ = feats.shape

            c = encoder(feats, positions, pad, causal=True)   # [B,T,d] kontekst
            z = target_enc(feats)                              # [B,T,d] cele
            valid = ~pad                                        # [B,T]

            loss = 0.0
            n_terms = 0
            for k in range(1, K + 1):
                if T - k <= 0:
                    continue
                anchor_ok = valid[:, :T - k] & valid[:, k:]     # [B, T-k]
                bi, ti = anchor_ok.nonzero(as_tuple=True)
                if bi.numel() < 2:
                    continue
                if bi.numel() > max_anchors:
                    perm = torch.randperm(bi.numel(), device=device)[:max_anchors]
                    bi, ti = bi[perm], ti[perm]
                pred = predictors[k - 1](c[bi, ti])             # [N,d]
                pos = z[bi, ti + k]                              # [N,d]
                pred = Fun.normalize(pred, dim=-1)
                pos = Fun.normalize(pos, dim=-1)
                logits = pred @ pos.t() / tau                   # [N,N]
                labels = torch.arange(bi.numel(), device=device)
                loss = loss + Fun.cross_entropy(logits, labels)
                n_terms += 1
            if n_terms == 0:
                continue
            loss = loss / n_terms

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ep_loss += loss.item(); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        log(f"  [InfoNCE] epoka {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")
    return history
