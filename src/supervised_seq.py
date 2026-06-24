"""Supervised VAE / RSSM — predykcja typu karty z dostepem do etykiet.

Trzy warianty (cfg.sup_mode):
  A — zamrozony enkoder SSL (recon+KL) + glowica MLP uczona CE.
  B — end-to-end: CE + sup_kl_weight*KL (gleboki wariacyjny information bottleneck,
      bez rekonstrukcji).
  C — end-to-end: CE + sup_kl_weight*KL + sup_recon_weight*recon (multi-task).

Trening i ewaluacja odbywaja sie PER KARTA, wylacznie na starcie tury dowolnego
gracza (spojnie z protokolem SSL). Predykcja idzie wprost z wyuczonej glowicy
(argmax CE), a nie z osobnej regresji logistycznej.

Re-uzywa datasetu per-karta z src.supervised oraz adapterow enkoderow z
src.ssl_vae / src.ssl_rssm.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch.utils.data import DataLoader

from .data import (CARD_FEATS, LABEL_TO_IDX, LABELS, build_sequences,
                   filter_to_turn_starts, load_split)
from .probe import _assemble_full, _f1_report, _grouped_report, extract_card_embeddings
from .ssl_rssm import RSSM, RSSMEncoderAdapter, train_rssm
from .ssl_vae import SeqVAE, VAEEncoderAdapter, train_vae
from .supervised import _SupSeqDataset, _sup_collate

SUP_MODES = ("A", "B", "C")


def parse_family(family):
    """'vae_supB' -> ('vae', 'B'); 'rssm_cat' -> ('rssm_cat', None)."""
    if "_sup" in family:
        base, mode = family.split("_sup")
        return base, mode
    return family, None


def _is_vae(base):
    return base in ("vae", "vae_cat")


def emb_dim_for(base, cfg):
    if base == "vae":
        return cfg.vae_latent_dim
    if base == "vae_cat":
        return cfg.vae_n_cat * cfg.vae_n_class
    if base == "rssm_gauss":
        return cfg.rssm_h_dim + cfg.rssm_z_dim
    if base == "rssm_cat":
        return cfg.rssm_h_dim + cfg.rssm_n_cat * cfg.rssm_n_class
    raise ValueError(f"nieznany backbone: {base}")


def _build_backbone(base, spec, cfg, device):
    if _is_vae(base):
        return SeqVAE(spec.n_features, variant=cfg.vae_variant, h_dim=cfg.vae_h_dim,
                      latent_dim=cfg.vae_latent_dim, n_cat=cfg.vae_n_cat,
                      n_class=cfg.vae_n_class, decoder_hidden=256,
                      num_layers=cfg.vae_num_layers, dropout=cfg.vae_dropout).to(device)
    variant = "cat" if base == "rssm_cat" else "gauss"
    return RSSM(spec.n_features, variant=variant,
                h_dim=cfg.rssm_h_dim, z_dim=cfg.rssm_z_dim, embed_dim=cfg.rssm_embed_dim,
                n_cat=cfg.rssm_n_cat, n_class=cfg.rssm_n_class,
                kl_balance=cfg.rssm_kl_balance, free_nats=cfg.rssm_free_nats).to(device)


def _forward_emb(base, model, feats, pad, is_binary):
    """Forward Z GRADIENTEM. Zwraca (emb [B,T,D], l_recon, l_kl)."""
    if _is_vae(base):
        _, l_recon, l_kl, emb = model(feats, pad, is_binary, beta=1.0)
        return emb, l_recon, l_kl
    _, l_recon, l_kl, h_seq, z_seq = model(feats, pad, is_binary)
    return torch.cat([h_seq, z_seq], -1), l_recon, l_kl


def _make_adapter(base, model):
    return VAEEncoderAdapter(model) if _is_vae(base) else RSSMEncoderAdapter(model)


class _MLPHead(nn.Module):
    def __init__(self, d_in, hidden, n_out=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class SupModel:
    """Wyuczony enkoder (adapter) + glowica klasyfikujaca typ karty."""
    def __init__(self, adapter, head, base, mode, emb_dim):
        self.adapter = adapter
        self.head = head
        self.base = base
        self.mode = mode
        self.emb_dim = emb_dim


def _class_weights(card_df, device):
    counts = np.bincount(card_df["label"].map(LABEL_TO_IDX), minlength=5) + 1
    return torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32, device=device)


def _train_loader(cfg, spec, n_games, log):
    """Loader per-karta TYLKO na startach tur (TRAIN)."""
    ts = load_split(cfg.data_dir, "train", "timesteps")
    card = load_split(cfg.data_dir, "train", "card_samples")
    if n_games and ts["game_id"].nunique() > n_games:
        rng = np.random.default_rng(cfg.seed)
        keep = set(rng.choice(ts["game_id"].unique(), n_games, replace=False))
        ts = ts[ts["game_id"].isin(keep)]; card = card[card["game_id"].isin(keep)]
    card = filter_to_turn_starts(card, ts)          # trening tylko na startach tur
    seqs = build_sequences(ts, spec, subsample_games=0)
    ds = _SupSeqDataset(seqs, card, cfg.eval_seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=_sup_collate, num_workers=0)
    return loader, card


def train_supervised_seq(family, cfg, spec, device, log=print, n_games=1500):
    """Trenuje supervised VAE/RSSM (wariant z family, np. 'vae_supC'). -> SupModel."""
    base, mode = parse_family(family)
    assert mode in SUP_MODES, f"bledny wariant supervised: {mode}"
    is_binary = torch.tensor(spec.is_binary, dtype=torch.bool, device=device)
    emb_dim = emb_dim_for(base, cfg)
    loader, card_tr = _train_loader(cfg, spec, n_games, log)
    w = _class_weights(card_tr, device)
    log(f"  [sup:{family}] {len(loader.dataset)} sekwencji z etykietami (start tury)")

    if mode == "A":
        # 1) pretrening SSL (recon+KL) -> 2) zamrozenie -> 3) glowica MLP (CE)
        if _is_vae(base):
            adapter, _ = train_vae(spec, cfg, device, log=log)
        else:
            adapter, _ = train_rssm(spec, cfg, device, log=log)
        enc_model = adapter.vae if _is_vae(base) else adapter.rssm
        for p in enc_model.parameters():
            p.requires_grad_(False)
        head = _MLPHead(emb_dim + len(CARD_FEATS), cfg.sup_head_hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for ep in range(1, cfg.epochs + 1):
            head.train(); tot, n = 0.0, 0
            for batch in loader:
                feats = batch["feats"].to(device); pad = batch["pad_mask"].to(device)
                with torch.no_grad():
                    emb = adapter(feats, batch["positions"].to(device), pad, causal=False)
                sel = emb[batch["bi"].to(device), batch["ti"].to(device)]
                x = torch.cat([sel, batch["cardf"].to(device)], 1)
                loss = Fun.cross_entropy(head(x), batch["y"].to(device), weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); n += 1
            log(f"  [sup:{family}] ep {ep}/{cfg.epochs} CE={tot/max(n,1):.4f}")
        return SupModel(adapter, head, base, mode, emb_dim)

    # warianty B / C — end-to-end
    model = _build_backbone(base, spec, cfg, device)
    head = nn.Linear(emb_dim + len(CARD_FEATS), 5).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    recon_w = cfg.sup_recon_weight if mode == "C" else 0.0
    kl_w = cfg.sup_kl_weight
    for ep in range(1, cfg.epochs + 1):
        model.train(); head.train()
        tot, ce_s, kl_s, rc_s, n = 0.0, 0.0, 0.0, 0.0, 0
        for batch in loader:
            feats = batch["feats"].to(device); pad = batch["pad_mask"].to(device)
            emb, l_recon, l_kl = _forward_emb(base, model, feats, pad, is_binary)
            sel = emb[batch["bi"].to(device), batch["ti"].to(device)]
            x = torch.cat([sel, batch["cardf"].to(device)], 1)
            ce = Fun.cross_entropy(head(x), batch["y"].to(device), weight=w)
            loss = ce + kl_w * l_kl + recon_w * l_recon
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
            tot += loss.item(); ce_s += ce.item(); kl_s += l_kl.item()
            rc_s += l_recon.item(); n += 1
        n = max(n, 1)
        log(f"  [sup:{family}] ep {ep}/{cfg.epochs} L={tot/n:.4f} "
            f"CE={ce_s/n:.4f} KL={kl_s/n:.4f} recon={rc_s/n:.4f}")
    return SupModel(_make_adapter(base, model), head, base, mode, emb_dim)


@torch.no_grad()
def run_supervised_eval(supmodel, spec, cfg, device, eval_split="test",
                        n_games=0, log=print):
    """Ewaluacja supervised: predykcja z glowicy na startach tur, per karta.
    Raportuje macro-F1 + per-klasa, osobno seen / unseen_mcts (jesli dostepne)."""
    def prep(split, n, seed):
        ts = load_split(cfg.data_dir, split, "timesteps")
        card = load_split(cfg.data_dir, split, "card_samples")
        if n and ts["game_id"].nunique() > n:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(ts["game_id"].unique(), n, replace=False))
            ts = ts[ts["game_id"].isin(keep)]; card = card[card["game_id"].isin(keep)]
        card = filter_to_turn_starts(card, ts)
        seqs = build_sequences(ts, spec, subsample_games=0)
        needed = {}
        for r in card[["game_id", "observed_color", "action_index"]].itertuples(index=False):
            needed.setdefault((int(r.game_id), str(r.observed_color)), set()).add(int(r.action_index))
        emb = extract_card_embeddings(supmodel.adapter, seqs, needed, cfg, device, causal=False)
        return _assemble_full(card, emb)   # X = [emb | CARD_FEATS] = wejscie glowicy

    Xte, yte, kinds, otypes = prep(eval_split, n_games, cfg.seed + 1)
    supmodel.head.eval()
    logits = supmodel.head(torch.from_numpy(Xte).to(device))
    pred = logits.argmax(-1).cpu().numpy()

    res = {"n_test": int(len(yte)), "all": _f1_report(yte, pred)}
    for kind in ("seen", "unseen_mcts"):
        m = kinds == kind
        if m.sum() > 0:
            res[kind] = _f1_report(yte[m], pred[m]); res[kind]["n"] = int(m.sum())
    res["by_observed_type"] = _grouped_report(yte, pred, otypes)
    ot = res["by_observed_type"]
    log(f"  [sup-eval:{supmodel.base}/{supmodel.mode}] "
        f"macro-F1 all={res['all']['macro_f1']:.3f} "
        f"seen={res.get('seen', {}).get('macro_f1', float('nan')):.3f} "
        f"unseen={res.get('unseen_mcts', {}).get('macro_f1', float('nan')):.3f}")
    if ot:
        log("    per observed_type: " + "  ".join(
            f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in ot.items()))
    return res
