"""
Weryfikacja wygenerowanego zbioru Catan (struktura v2, chunki).

Wczytuje WSZYSTKIE chunki przez glob:
    timesteps_*.parquet     - sekwencja per akcja
    card_samples_*.parquet  - próbki per-karta (5-klasowy label)

Sprawdza integralność, brak wycieku targetu, sensowność, spójność obu tabel
i poprawność splitu transferowego.

Uruchom:
    uv run verify_dataset.py --data-dir data
"""

import argparse
import glob
import os
import sys

import pandas as pd

DEV_TYPES = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "MONOPOLY", "YEAR_OF_PLENTY"]
Y_COLS = [f"y_{d.lower()}" for d in DEV_TYPES]
TS_META = ["game_id", "action_index", "observed_color", "observed_type",
           "table", "winner_type"]
OBSERVABLE_TYPES = {"ValueFunction", "AlphaBeta", "MCTS"}


def load_chunks(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not files:
        single = os.path.join(data_dir, f"{prefix}.parquet")
        if os.path.exists(single):
            files = [single]
    if not files:
        raise FileNotFoundError(f"Brak plików {prefix}_*.parquet w {data_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), files


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    mark = "OK" if condition else "XX"
    print(f"  [{status}] {mark} {name}" + (f"  -- {detail}" if detail else ""))
    return bool(condition)


def main(data_dir):
    ts, ts_files = load_chunks(data_dir, "timesteps")
    card, card_files = load_chunks(data_dir, "card_samples")
    feature_cols = [c for c in ts.columns if c not in Y_COLS and c not in TS_META]
    results = []

    print("=" * 64)
    print(f"WERYFIKACJA: {data_dir}")
    print(f"  timesteps: {len(ts):,} wierszy z {len(ts_files)} chunkow")
    print(f"  card_samples: {len(card):,} wierszy z {len(card_files)} chunkow")
    print(f"  gier: {ts['game_id'].nunique():,}, "
          f"perspektyw: {ts.groupby(['game_id','observed_color']).ngroups:,}")
    print("=" * 64)

    # --- 1. INTEGRALNOSC ---
    print("\n[1] Integralnosc")
    results.append(check("timesteps: brak NaN", ts.isna().sum().sum() == 0,
                         f"{ts.isna().sum().sum()} NaN"))
    results.append(check("card_samples: brak NaN", card.isna().sum().sum() == 0,
                         f"{card.isna().sum().sum()} NaN"))
    dup = ts.duplicated(["game_id", "observed_color", "action_index"]).sum()
    results.append(check("timesteps: unikalny (game,observed,action)", dup == 0,
                         f"{dup} duplikatow"))
    contig = ts.groupby(["game_id", "observed_color"])["action_index"].apply(
        lambda s: sorted(s) == list(range(len(s)))).all()
    results.append(check("action_index ciagly per perspektywa", contig))

    # --- 2. BRAK WYCIEKU ---
    print("\n[2] Brak wycieku targetu")
    ysum = ts[Y_COLS].sum(axis=1)
    results.append(check("p0_n_dev_in_hand == suma y_* (maskowanie)",
                         (ts["p0_n_dev_in_hand"] == ysum).all(),
                         f"{(ts['p0_n_dev_in_hand'] == ysum).mean()*100:.1f}%"))
    forbidden = [c for c in feature_cols
                 if any(d in c.lower() for d in
                        ["knight_in_hand", "victory_point_in_hand",
                         "road_building_in_hand", "monopoly_in_hand",
                         "year_of_plenty_in_hand"])]
    results.append(check("brak kolumn z rozbiciem kart na rece", not forbidden,
                         str(forbidden) if forbidden else "ok"))

    # --- 3. SENSOWNOSC ---
    print("\n[3] Sensownosc wartosci")
    results.append(check("obserwowani to tylko silne boty",
                         set(ts["observed_type"].unique()) <= OBSERVABLE_TYPES,
                         str(sorted(ts["observed_type"].unique()))))
    results.append(check("label per-karta w 5 typach",
                         set(card["label"].unique()) <= set(DEV_TYPES),
                         str(sorted(card["label"].unique()))))
    rob = ts["robber_on_observed"]
    results.append(check("robber_on_observed binarny", set(rob.unique()) <= {0, 1}))
    results.append(check("robber_on_observed nietrywialny",
                         0 < rob.mean() < 1, f"{rob.mean()*100:.1f}% krokow"))
    results.append(check("rounds_held nieujemne",
                         (card["rounds_held"] >= 0).all()))

    # --- 4. SPOJNOSC TS <-> CARD ---
    print("\n[4] Spojnosc timesteps <-> card_samples")
    ts_keys = set(map(tuple, ts[["game_id", "observed_color", "action_index"]].values))
    card_keys = set(map(tuple, card[["game_id", "observed_color", "action_index"]].values))
    orphans = card_keys - ts_keys
    results.append(check("kazda probka-karta ma timestep", not orphans,
                         f"{len(orphans)} sierot"))
    card_counts = card.groupby(["game_id", "observed_color", "action_index"]).size()
    merged = ts.set_index(["game_id", "observed_color", "action_index"])
    sample = merged["n_hidden_cards"]
    common = card_counts.index.intersection(sample.index)
    match = (card_counts.loc[common] == sample.loc[common]).mean() if len(common) else 1.0
    results.append(check("liczba kart-probek == n_hidden_cards",
                         match == 1.0, f"{match*100:.1f}%"))
    results.append(check("n_hidden_cards == suma y_* w timestep",
                         (ts["n_hidden_cards"] == ysum).all(),
                         f"{(ts['n_hidden_cards'] == ysum).mean()*100:.1f}%"))

    # --- 5. SPLIT TRANSFEROWY ---
    print("\n[5] Split transferowy")
    gt = ts.drop_duplicates("game_id")
    mcts_at_table = gt["table"].str.contains("MCTS").sum()
    mcts_observed = (gt["observed_type"] == "MCTS").sum()
    results.append(check("MCTS przy stole >= MCTS obserwowany",
                         mcts_at_table >= mcts_observed,
                         f"{mcts_at_table} gier z MCTS przy stole, "
                         f"{mcts_observed} z MCTS obserwowanym"))

    # --- PODSUMOWANIE ---
    print("\n" + "=" * 64)
    passed = sum(results)
    print(f"WYNIK: {passed}/{len(results)} checkow przeszlo")
    print("=" * 64)

    print("\nRozklad 5-klasowego labelu (probki per-karta):")
    print(card["label"].value_counts().to_string())
    print("\nProbki per-karta na poczatkach tur:",
          (card["is_observed_turn_start"] == 1).sum())
    print("\nMediana rounds_held per typ (sygnal: VP trzymane najdluzej?):")
    print(card.groupby("label")["rounds_held"].median()
          .sort_values(ascending=False).to_string())
    print("\nPerspektywy per typ obserwowanego:")
    print(ts.drop_duplicates(["game_id", "observed_color"])["observed_type"]
          .value_counts().to_string())

    if passed < len(results):
        print("\n[!] Czesc checkow NIE przeszla -- sprawdz wyzej.")
        sys.exit(1)
    print("\n[OK] Zbior wyglada poprawnie.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()
    main(args.data_dir)