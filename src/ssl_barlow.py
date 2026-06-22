"""Pretrening Barlow Twins — nie-kontrastywny SSL (bez negatywow).

Dwa zaugmentowane widoki tej samej sekwencji -> enkoder bidirectional -> pooling
do embeddingu okna -> projektor. Strata: macierz krzyzowej korelacji cech miedzy
widokami ma byc macierza jednostkowa (przekatna ~1 = niezmienniczosc,
poza-przekatna ~0 = redukcja redundancji). Uczy NIEZMIENNICZOSCI wzgledem
augmentacji.
"""
import torch

from .augment import make_two_views
from .models import ProjectionHead, SeqTransformerEncoder


def _barlow_loss(z1, z2, lam):
    # standaryzacja po batchu (kazda cecha: srednia 0, std 1)
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    B, D = z1.shape
    c = (z1.t() @ z2) / B                      # [D,D] krzyzowa korelacja
    on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
    off = c.clone()
    off.diagonal().zero_()
    off_diag = off.pow(2).sum()
    return on_diag + lam * off_diag


def train_barlow(encoder: SeqTransformerEncoder, loader, spec, cfg, device,
                 log=print):
    is_binary = torch.from_numpy(spec.is_binary).to(device)
    projector = ProjectionHead(encoder.d_model, cfg.barlow_proj_dim,
                               cfg.barlow_proj_dim).to(device)
    params = list(encoder.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)

    history = []
    for epoch in range(cfg.epochs):
        encoder.train(); projector.train()
        ep_loss, n_batches = 0.0, 0
        for batch in loader:
            if batch["feats"].shape[0] < 2:
                continue  # Barlow potrzebuje batcha > 1 do standaryzacji
            (f1, m1), (f2, m2) = make_two_views(batch, is_binary.cpu(),
                                                generator=gen)
            positions = batch["positions"].to(device)
            f1, m1 = f1.to(device), m1.to(device)
            f2, m2 = f2.to(device), m2.to(device)

            h1 = encoder(f1, positions, m1, causal=False)
            h2 = encoder(f2, positions, m2, causal=False)
            e1 = SeqTransformerEncoder.masked_mean(h1, m1)
            e2 = SeqTransformerEncoder.masked_mean(h2, m2)
            z1, z2 = projector(e1), projector(e2)
            loss = _barlow_loss(z1, z2, cfg.barlow_lambda)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ep_loss += loss.item(); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        log(f"  [Barlow] epoka {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")
    return history
