"""Pretrening Transformer MAE — masked modeling (BERT/MAE-style).

Enkoder bidirectional. Maskujemy ~30% krokow (zastepujemy uczonym tokenem maski)
i rekonstruujemy ich oryginalne cechy: MSE dla cech ciaglych + BCE dla binarnych
(one-hoty akcji, flagi). Pretekst = odszumianie/uzupelnianie stanu gry. Uczy
STRUKTURY pojedynczego kroku i lokalnego kontekstu.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fun

from .models import MAEDecoder, SeqTransformerEncoder


def train_mae(encoder: SeqTransformerEncoder, loader, spec, cfg, device, log=print):
    F = spec.n_features
    is_binary = torch.from_numpy(spec.is_binary).to(device)
    cont = ~is_binary
    decoder = MAEDecoder(encoder.d_model, F).to(device)
    mask_token = nn.Parameter(torch.zeros(F, device=device))
    params = list(encoder.parameters()) + list(decoder.parameters()) + [mask_token]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        encoder.train(); decoder.train()
        ep_loss, n_batches = 0.0, 0
        for batch in loader:
            feats = batch["feats"].to(device)
            positions = batch["positions"].to(device)
            pad = batch["pad_mask"].to(device)
            B, T, _ = feats.shape

            # maska: losowe kroki, tylko nie-padding
            rand = torch.rand(B, T, device=device)
            mask = (rand < cfg.mae_mask_ratio) & (~pad)        # [B,T] True=zamaskowany
            if mask.sum() == 0:
                continue

            corrupted = feats.clone()
            corrupted[mask] = mask_token
            h = encoder(corrupted, positions, pad, causal=False)
            recon = decoder(h)                                  # [B,T,F]

            tgt = feats[mask]                                   # [M,F]
            out = recon[mask]
            loss_cont = Fun.mse_loss(out[:, cont], tgt[:, cont]) if cont.any() else 0.0
            loss_bin = (Fun.binary_cross_entropy_with_logits(out[:, is_binary],
                        tgt[:, is_binary]) if is_binary.any() else 0.0)
            loss = loss_cont + loss_bin

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ep_loss += float(loss); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        log(f"  [MAE] epoka {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")
    return history
