"""Augmentacje okien sekwencji dla Barlow Twins (dwa widoki tej samej sekwencji).

Trzy niezalezne transformacje (zachowuja semantyke stanu gry):
  - feat_dropout : zerowanie czesci wymiarow cech (0 = srednia po standaryzacji),
  - gaussian noise : szum tylko na cechach ciaglych (nie na one-hotach/flagach),
  - time_dropout : losowe pominiecie czesci krokow (dodanie ich do maski paddingu)
                   — odpowiednik temporalnego przycinania/jitteru.
"""
import torch


def _augment_view(feats, pad_mask, is_binary, p_feat, sigma, p_time, rng):
    B, T, F = feats.shape
    out = feats.clone()
    cont = ~is_binary  # [F] bool, cechy ciagle

    # 1) szum gaussa na cechach ciaglych
    if sigma > 0:
        noise = torch.randn(B, T, F, generator=rng) * sigma
        out = out + noise * cont.view(1, 1, F)

    # 2) feature dropout (zerowanie wymiarow cech, per sample)
    if p_feat > 0:
        keep = (torch.rand(B, 1, F, generator=rng) > p_feat).float()
        out = out * keep

    # 3) time dropout — rozszerz maske paddingu o losowe kroki
    new_mask = pad_mask.clone()
    if p_time > 0:
        drop = (torch.rand(B, T, generator=rng) < p_time) & (~pad_mask)
        # nie pozwalamy wyzerowac calej sekwencji
        keep_any = (~(pad_mask | drop)).any(dim=1)
        drop[~keep_any] = False
        new_mask = pad_mask | drop
    return out, new_mask


def make_two_views(batch, is_binary, p_feat=0.2, sigma=0.1, p_time=0.15, generator=None):
    feats = batch["feats"]
    pad_mask = batch["pad_mask"]
    g = generator
    v1 = _augment_view(feats, pad_mask, is_binary, p_feat, sigma, p_time, g)
    v2 = _augment_view(feats, pad_mask, is_binary, p_feat, sigma, p_time, g)
    return v1, v2
