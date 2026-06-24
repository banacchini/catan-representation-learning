"""Ewaluacja reprezentacji metoda LINEAR PROBE.

Enkoder zamrozony -> dla kazdej probki per-karta bierzemy embedding kroku w jej
action_index, doklejamy jawne cechy per-karta -> regresja logistyczna
(class_weight='balanced'). Metryka: macro-F1 + F1 per klasa, raportowane osobno
dla stylu 'seen' i 'unseen_mcts'. To standardowy protokol oceny jakosci
reprezentacji samonadzorowanej.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .data import (CARD_FEATS, LABEL_TO_IDX, LABELS, SeqDataset, build_sequences,
                   collate, filter_to_turn_starts, load_split)


def run_raw_baseline(spec, cfg, log=print, n_train_games=1500, n_test_games=1500,
                     max_train_samples=80000):
    """Baseline BEZ enkodera: regresja logistyczna na surowych (znormalizowanych)
    cechach kroku w pozycji karty + cechy per-karta. Dolna granica odniesienia."""
    feat_cols = spec.feat_cols
    # cechy per-karta nieobecne juz w timesteps (n_hidden_cards / is_observed_turn_start
    # wystepuja w obu tabelach — bierzemy je z wektora cech kroku)
    card_extra = [c for c in CARD_FEATS if c not in feat_cols]

    def prep(split, n_games, seed):
        ts = load_split(cfg.data_dir, split, "timesteps")
        card = load_split(cfg.data_dir, split, "card_samples")
        if n_games and ts["game_id"].nunique() > n_games:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(ts["game_id"].unique(), n_games, replace=False))
            ts = ts[ts["game_id"].isin(keep)]; card = card[card["game_id"].isin(keep)]
        # ewaluacja TYLKO na starcie tury dowolnego gracza (+ dokleja current_rel_pos)
        card = filter_to_turn_starts(card, ts)
        key = ["game_id", "observed_color", "action_index"]
        m = card.merge(ts[key + feat_cols], on=key, how="inner", suffixes=("_card", ""))
        Xr = spec.transform(m[feat_cols].to_numpy(dtype=np.float32))
        Xc = m[card_extra].to_numpy(dtype=np.float32)
        X = np.concatenate([Xr, Xc], axis=1)
        y = m["label"].map(LABEL_TO_IDX).to_numpy()
        kinds = m["test_kind"].to_numpy() if "test_kind" in m.columns else np.array(["-"] * len(m))
        return X, y, kinds

    Xtr, ytr, _ = prep("train", n_train_games, cfg.seed)
    if len(ytr) > max_train_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(ytr), max_train_samples, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    Xte, yte, kinds = prep("test", n_test_games, cfg.seed + 1)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=cfg.probe_max_iter,
                             class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "all": _f1_report(yte, pred)}
    for kind in ("seen", "unseen_mcts"):
        msk = kinds == kind
        if msk.sum() > 0:
            res[kind] = _f1_report(yte[msk], pred[msk]); res[kind]["n"] = int(msk.sum())
    log(f"  [raw baseline] macro-F1 all={res['all']['macro_f1']:.3f}")
    return res


@torch.no_grad()
def extract_card_embeddings(encoder, seqs, needed, cfg, device, causal):
    """Zwraca dict: (game_id, color, action_index) -> embedding (np.float32).
    Liczone tylko dla pozycji obecnych w `needed` (oszczednosc pamieci)."""
    encoder.eval()
    ds = SeqDataset(seqs, cfg.eval_seq_len, mode="full")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    out = {}
    for batch in loader:
        feats = batch["feats"].to(device)
        positions = batch["positions"].to(device)
        pad = batch["pad_mask"].to(device)
        h = encoder(feats, positions, pad, causal=causal).cpu().numpy()
        lengths = batch["lengths"].numpy()
        pos_np = batch["positions"].numpy()
        for i, (gid, color) in enumerate(batch["keys"]):
            want = needed.get((gid, color))
            if not want:
                continue
            for t in range(lengths[i]):
                ai = int(pos_np[i, t])
                if ai in want:
                    out[(gid, color, ai)] = h[i, t]
    return out


def _assemble_full(card_df, emb):
    """Jak _assemble, ale zwraca tez wektor observed_type (do rozbicia per algorytm).
    Returns (X, y, kinds, otypes)."""
    rows_emb, rows_card, ys, kinds, otypes = [], [], [], [], []
    has_kind = "test_kind" in card_df.columns
    has_ot = "observed_type" in card_df.columns
    for r in card_df.itertuples(index=False):
        key = (int(r.game_id), str(r.observed_color), int(r.action_index))
        e = emb.get(key)
        if e is None:
            continue
        rows_emb.append(e)
        rows_card.append([getattr(r, c) for c in CARD_FEATS])
        ys.append(LABEL_TO_IDX[r.label])
        kinds.append(getattr(r, "test_kind", "-") if has_kind else "-")
        otypes.append(getattr(r, "observed_type", "-") if has_ot else "-")
    E = np.asarray(rows_emb, dtype=np.float32)
    C = np.asarray(rows_card, dtype=np.float32)
    X = np.concatenate([E, C], axis=1)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, np.asarray(kinds), np.asarray(otypes)


def _assemble(card_df, emb, card_scaler=None, emb_dim=None):
    """Buduje macierz X = [embedding | cechy per-karta] i wektor y."""
    X, y, kinds, _ = _assemble_full(card_df, emb)
    return X, y, kinds


def _f1_report(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(5)),
                     zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=list(range(5)),
                   zero_division=0)
    return {"macro_f1": float(macro),
            "per_class_f1": {LABELS[i]: float(per[i]) for i in range(5)}}


def _grouped_report(y_true, y_pred, groups):
    """Macro-F1 per wartosc w `groups` (np. observed_type / test_kind)."""
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() > 0:
            rep = _f1_report(y_true[m], y_pred[m])
            rep["n"] = int(m.sum())
            out[str(g)] = rep
    return out


def run_probe(encoder, spec, cfg, device, causal, name="",
              n_train_games=1500, n_test_games=1500, log=print,
              max_train_samples=80000, eval_split="test"):
    """Pelna ewaluacja: trenuje glowice na TRAIN, raportuje na `eval_split`
    (domyslnie TEST z podzialem seen/unseen; do selekcji modeli uzyj 'val').
    Ewaluacja TYLKO na starcie tury dowolnego gracza, per karta."""
    def prep(split, n_games, seed):
        ts = load_split(cfg.data_dir, split, "timesteps")
        card = load_split(cfg.data_dir, split, "card_samples")
        if n_games and ts["game_id"].nunique() > n_games:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(ts["game_id"].unique(), n_games, replace=False))
            ts = ts[ts["game_id"].isin(keep)]
            card = card[card["game_id"].isin(keep)]
        # ewaluacja TYLKO na starcie tury dowolnego gracza (+ dokleja current_rel_pos)
        card = filter_to_turn_starts(card, ts)
        seqs = build_sequences(ts, spec, subsample_games=0)
        needed = {}
        for r in card[["game_id", "observed_color", "action_index"]].itertuples(index=False):
            needed.setdefault((int(r.game_id), str(r.observed_color)), set()).add(int(r.action_index))
        emb = extract_card_embeddings(encoder, seqs, needed, cfg, device, causal)
        return _assemble_full(card, emb)

    log(f"  [probe:{name}] przygotowanie TRAIN...")
    Xtr, ytr, _, _ = prep("train", n_train_games, cfg.seed)
    if len(ytr) > max_train_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(ytr), max_train_samples, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    log(f"  [probe:{name}] przygotowanie {eval_split.upper()}...")
    Xte, yte, kinds, otypes = prep(eval_split, n_test_games, cfg.seed + 1)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=cfg.probe_max_iter, class_weight="balanced",
                             C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
           "all": _f1_report(yte, pred)}
    for kind in ("seen", "unseen_mcts"):
        m = kinds == kind
        if m.sum() > 0:
            res[kind] = _f1_report(yte[m], pred[m])
            res[kind]["n"] = int(m.sum())
    res["by_observed_type"] = _grouped_report(yte, pred, otypes)
    ot = res["by_observed_type"]
    log(f"  [probe:{name}] macro-F1 all={res['all']['macro_f1']:.3f} "
        f"seen={res.get('seen', {}).get('macro_f1', float('nan')):.3f} "
        f"unseen={res.get('unseen_mcts', {}).get('macro_f1', float('nan')):.3f}")
    if ot:
        log("    per observed_type: " + "  ".join(
            f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in ot.items()))
    return res
