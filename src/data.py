"""Wczytywanie danych i budowa sekwencji dla enkodera.

Jednostka sekwencji = (game_id, observed_color), sortowana po action_index.
Wejscie modelu = 82 kolumny numeryczne (wszystko poza meta i etykietami y_*).
Etykiety y_* sa WYKLUCZONE z wejscia (to wyciek targetu).

FeatureSpec liczy statystyki standaryzacji na splicie TRAIN i wykrywa kolumny
binarne (one-hoty akcji, flagi) — potrzebne do augmentacji i straty MAE.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

META_COLS = ["game_id", "action_index", "observed_color",
             "observed_type", "table", "winner_type", "test_kind", "split"]
TARGET_COLS = ["y_knight", "y_victory_point", "y_road_building",
               "y_monopoly", "y_year_of_plenty"]
LABELS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "MONOPOLY", "YEAR_OF_PLENTY"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
# jawne cechy per-karta doklejane do embeddingu w probe (NIE sa wyciekiem typu)
CARD_FEATS = ["rounds_held", "card_slot", "n_hidden_cards", "bought_at_action",
              "is_observed_turn_start"]


def feature_columns(df):
    drop = set(META_COLS) | set(TARGET_COLS)
    cols = [c for c in df.columns if c not in drop]
    # bezpiecznik: tylko kolumny numeryczne (gdyby split dodal inne tekstowe)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def load_split(data_dir, split, kind):
    """kind = 'timesteps' | 'card_samples'."""
    path = os.path.join(data_dir, f"{split}_{kind}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak {path} — uruchom split_dataset.py najpierw.")
    return pd.read_parquet(path)


@dataclass
class FeatureSpec:
    feat_cols: list
    mean: np.ndarray          # [F] srednia (0 dla kolumn binarnych)
    std: np.ndarray           # [F] odchylenie (1 dla kolumn binarnych)
    is_binary: np.ndarray     # [F] bool

    @property
    def n_features(self):
        return len(self.feat_cols)

    def transform(self, arr):
        return (arr - self.mean) / self.std


def fit_feature_spec(train_ts):
    """Statystyki standaryzacji + detekcja kolumn binarnych z TRAIN."""
    feat_cols = feature_columns(train_ts)
    X = train_ts[feat_cols].to_numpy(dtype=np.float32)
    # binarna = przyjmuje tylko wartosci {0,1}
    is_binary = np.array([
        np.isin(np.unique(X[:, j]), [0.0, 1.0]).all() for j in range(X.shape[1])
    ])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    # kolumny binarne zostawiamy bez standaryzacji (0/1)
    mean[is_binary] = 0.0
    std[is_binary] = 1.0
    return FeatureSpec(feat_cols, mean.astype(np.float32),
                       std.astype(np.float32), is_binary)


def build_sequences(ts_df, spec, subsample_games=0, seed=0):
    """Zwraca liste sekwencji: dict(key, feats[T,F] float32 znormalizowane,
    positions[T] = action_index)."""
    if subsample_games and ts_df["game_id"].nunique() > subsample_games:
        rng = np.random.default_rng(seed)
        games = ts_df["game_id"].unique()
        keep = set(rng.choice(games, size=subsample_games, replace=False))
        ts_df = ts_df[ts_df["game_id"].isin(keep)]

    feat_cols = spec.feat_cols
    seqs = []
    ts_df = ts_df.sort_values(["game_id", "observed_color", "action_index"])
    for (gid, color), g in ts_df.groupby(["game_id", "observed_color"], sort=False):
        feats = spec.transform(g[feat_cols].to_numpy(dtype=np.float32))
        positions = g["action_index"].to_numpy(dtype=np.int64)
        seqs.append({
            "key": (int(gid), str(color)),
            "feats": feats,
            "positions": positions,
        })
    return seqs


class SeqDataset(Dataset):
    """Dataset sekwencji. W trybie 'crop' losuje ciagle okno dlugosci max_len
    (pretrening); w trybie 'full' zwraca cala sekwencje przycieta do max_len
    (ekstrakcja embeddingow do probe)."""

    def __init__(self, seqs, max_len, mode="crop", seed=0):
        assert mode in ("crop", "full")
        self.seqs = seqs
        self.max_len = max_len
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        feats, pos = s["feats"], s["positions"]
        T = feats.shape[0]
        if self.mode == "crop" and T > self.max_len:
            start = int(self.rng.integers(0, T - self.max_len + 1))
            sl = slice(start, start + self.max_len)
            feats, pos = feats[sl], pos[sl]
        elif self.mode == "full" and T > self.max_len:
            feats, pos = feats[: self.max_len], pos[: self.max_len]
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),
            "positions": torch.from_numpy(np.ascontiguousarray(pos)),
            "key": s["key"],
        }


def collate(batch):
    """Padding do dlugosci najdluzszej sekwencji w batchu + maska paddingu."""
    B = len(batch)
    lengths = [b["feats"].shape[0] for b in batch]
    T = max(lengths)
    F = batch[0]["feats"].shape[1]
    feats = torch.zeros(B, T, F, dtype=torch.float32)
    positions = torch.zeros(B, T, dtype=torch.long)
    pad_mask = torch.ones(B, T, dtype=torch.bool)  # True = padding (ignoruj)
    for i, b in enumerate(batch):
        L = lengths[i]
        feats[i, :L] = b["feats"]
        positions[i, :L] = b["positions"]
        pad_mask[i, :L] = False
    return {
        "feats": feats,
        "positions": positions,
        "pad_mask": pad_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "keys": [b["key"] for b in batch],
    }
