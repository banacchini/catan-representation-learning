"""Nadzorowana gorna granica (upper bound) do porownania.

Enkoder uczony END-TO-END razem z liniowa glowica na zadaniu per-karta
(cross-entropy z wagami klas). Pokazuje ile da sie wyciagnac z tej architektury
PRZY dostepie do etykiet — punkt odniesienia dla metod samonadzorowanych, ktore
etykiet NIE widza podczas pretreningu.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch.utils.data import DataLoader, Dataset

from .data import CARD_FEATS, LABEL_TO_IDX, LABELS, build_sequences, load_split
from .models import SeqTransformerEncoder


class _SupSeqDataset(Dataset):
    """Sekwencja + lista (pozycja_w_sekwencji, cechy_karty, etykieta)."""

    def __init__(self, seqs, card_df, max_len):
        self.items = []
        by_key = {}
        for r in card_df.itertuples(index=False):
            by_key.setdefault((int(r.game_id), str(r.observed_color)), []).append(
                (int(r.action_index), [float(getattr(r, c)) for c in CARD_FEATS],
                 LABEL_TO_IDX[r.label]))
        for s in seqs:
            cards = by_key.get(s["key"])
            if not cards:
                continue
            pos_to_idx = {int(p): i for i, p in enumerate(s["positions"][:max_len])}
            labels = [(pos_to_idx[ai], cf, lab) for ai, cf, lab in cards
                      if ai in pos_to_idx]
            if labels:
                self.items.append((s["feats"][:max_len], s["positions"][:max_len],
                                   labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        feats, pos, labels = self.items[i]
        return {"feats": torch.from_numpy(feats),
                "positions": torch.from_numpy(pos), "labels": labels}


def _sup_collate(batch):
    B = len(batch)
    T = max(b["feats"].shape[0] for b in batch)
    F = batch[0]["feats"].shape[1]
    feats = torch.zeros(B, T, F)
    positions = torch.zeros(B, T, dtype=torch.long)
    pad = torch.ones(B, T, dtype=torch.bool)
    bi, ti, cardf, ys = [], [], [], []
    for i, b in enumerate(batch):
        L = b["feats"].shape[0]
        feats[i, :L] = b["feats"]; positions[i, :L] = b["positions"]; pad[i, :L] = False
        for t, cf, lab in b["labels"]:
            bi.append(i); ti.append(t); cardf.append(cf); ys.append(lab)
    return {"feats": feats, "positions": positions, "pad_mask": pad,
            "bi": torch.tensor(bi), "ti": torch.tensor(ti),
            "cardf": torch.tensor(cardf, dtype=torch.float32),
            "y": torch.tensor(ys, dtype=torch.long)}


def train_supervised(encoder: SeqTransformerEncoder, spec, cfg, device,
                     n_games=1500, log=print):
    ts = load_split(cfg.data_dir, "train", "timesteps")
    card = load_split(cfg.data_dir, "train", "card_samples")
    if n_games and ts["game_id"].nunique() > n_games:
        rng = np.random.default_rng(cfg.seed)
        keep = set(rng.choice(ts["game_id"].unique(), n_games, replace=False))
        ts = ts[ts["game_id"].isin(keep)]; card = card[card["game_id"].isin(keep)]
    seqs = build_sequences(ts, spec, subsample_games=0)
    ds = _SupSeqDataset(seqs, card, cfg.eval_seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=_sup_collate, num_workers=0)

    # wagi klas (balansowanie)
    counts = np.bincount(card["label"].map(LABEL_TO_IDX), minlength=5) + 1
    w = torch.tensor((counts.sum() / (5 * counts)), dtype=torch.float32, device=device)

    head = nn.Linear(encoder.d_model + len(CARD_FEATS), 5).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        encoder.train(); head.train()
        ep_loss, n = 0.0, 0
        for batch in loader:
            feats = batch["feats"].to(device)
            positions = batch["positions"].to(device)
            pad = batch["pad_mask"].to(device)
            h = encoder(feats, positions, pad, causal=False)
            emb = h[batch["bi"].to(device), batch["ti"].to(device)]
            x = torch.cat([emb, batch["cardf"].to(device)], dim=1)
            logits = head(x)
            loss = Fun.cross_entropy(logits, batch["y"].to(device), weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
            ep_loss += loss.item(); n += 1
        avg = ep_loss / max(n, 1); history.append(avg)
        log(f"  [Supervised] epoka {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")
    return history
