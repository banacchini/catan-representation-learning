"""
Podzial zbioru Catan (struktura v2, chunki) na train / val / test bez wycieku.

Wczytuje wszystkie chunki timesteps_*.parquet i card_samples_*.parquet,
dzieli PER GRA i zapisuje 6 plikow:
    train_timesteps.parquet,  train_card_samples.parquet
    val_timesteps.parquet,    val_card_samples.parquet
    test_timesteps.parquet,   test_card_samples.parquet

ZASADA (asymetryczna):
  - TRAIN/VAL = WYLACZNIE gry bez ZADNEGO MCTS w skladzie stolu (kolumna table)
  - TEST = wszystkie gry z MCTS (niewidziany styl) + czesc czystych gier
           (widziany styl, do porownania)
Kryterium = obecnosc MCTS w `table`, NIE observed_type.
Split PER GRA: obie tabele dzielone tym samym podzialem game_id, wiec
perspektywy i karty tej samej gry trafiaja do tego samego zbioru.

Uruchom:
    uv run split_dataset.py --data-dir data --out-dir data/splits
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_chunks(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not files:
        single = os.path.join(data_dir, f"{prefix}.parquet")
        if os.path.exists(single):
            files = [single]
    if not files:
        raise FileNotFoundError(f"Brak plikow {prefix}_*.parquet w {data_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def make_split(data_dir, out_dir, val_frac=0.15, seen_in_test_frac=0.2, seed=0):
    ts = load_chunks(data_dir, "timesteps")
    card = load_chunks(data_dir, "card_samples")
    rng = np.random.default_rng(seed)

    # klasyfikacja gier po obecnosci MCTS w SKLADZIE STOLU
    game_table = ts.drop_duplicates("game_id").set_index("game_id")["table"]
    mcts_games = set(game_table[game_table.str.contains("MCTS")].index)
    clean_games = np.array(sorted(set(game_table.index) - mcts_games))
    rng.shuffle(clean_games)

    n_seen_test = int(len(clean_games) * seen_in_test_frac)
    seen_test_games = set(clean_games[:n_seen_test])
    remaining = clean_games[n_seen_test:]
    n_val = int(len(remaining) * val_frac)
    val_games = set(remaining[:n_val])
    train_games = set(remaining[n_val:])
    test_games = mcts_games | seen_test_games

    # weryfikacja braku wycieku
    leak = (train_games | val_games) & mcts_games
    assert not leak, f"WYCIEK: {len(leak)} gier z MCTS w train/val!"

    def assign(g):
        if g in train_games:
            return "train"
        if g in val_games:
            return "val"
        return "test"

    os.makedirs(out_dir, exist_ok=True)
    for tbl_name, df in [("timesteps", ts), ("card_samples", card)]:
        df = df.copy()
        df["split"] = df["game_id"].map(assign)
        if tbl_name == "timesteps" or "is_observed_turn_start" in df.columns:
            df["test_kind"] = np.where(df["game_id"].isin(mcts_games),
                                       "unseen_mcts", "seen")
            df.loc[df["split"] != "test", "test_kind"] = "-"
        for name in ["train", "val", "test"]:
            sub = df[df["split"] == name].drop(columns=["split"])
            sub.to_parquet(os.path.join(out_dir, f"{name}_{tbl_name}.parquet"),
                           index=False)

    # raport
    print("=" * 64)
    print("PODZIAL ZBIORU (bez wycieku MCTS)")
    print("=" * 64)
    print(f"Gry lacznie:         {len(game_table):,}")
    print(f"  z MCTS przy stole: {len(mcts_games):,}  -> wszystkie do TEST")
    print(f"  bez MCTS:          {len(clean_games):,}")
    print()
    ts2 = ts.copy()
    ts2["split"] = ts2["game_id"].map(assign)
    card2 = card.copy()
    card2["split"] = card2["game_id"].map(assign)
    for name, games in [("train", train_games), ("val", val_games),
                        ("test", test_games)]:
        print(f"{name:6s}: {len(games):5d} gier | "
              f"timesteps {(ts2['split']==name).sum():8,} | "
              f"card {(card2['split']==name).sum():7,}")
    print()
    test_ts = ts2[ts2["split"] == "test"]
    tk = np.where(test_ts["game_id"].isin(mcts_games), "unseen_mcts", "seen")
    import collections
    print("Sklad TEST (timesteps):", dict(collections.Counter(tk)))
    print()
    print(f"KONTROLA WYCIEKU: gry MCTS w train/val = {len(leak)} (musi byc 0)")
    print(f"Zapisano do: {out_dir}/(train|val|test)_(timesteps|card_samples).parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="data/splits")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seen-in-test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_split(args.data_dir, args.out_dir, args.val_frac,
               args.seen_in_test_frac, args.seed)
