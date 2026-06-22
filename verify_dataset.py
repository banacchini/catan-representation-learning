"""
Weryfikacja wygenerowanego zbioru Catan (struktura v2, chunki, JEDNA
perspektywa na grę).

Wczytuje wszystkie chunki przez glob:
    timesteps_*.parquet     - sekwencja per akcja
    card_samples_*.parquet  - próbki per-karta (5-klasowy label)

Sprawdza integralność, brak wycieku, sensowność (w tym NOWE kolumny: bank,
produkcja, played_*_total, dev_deck_remaining), spójność obu tabel i split.
Diagnostyka rounds_held liczona PER POJEDYNCZA KARTA (nie per wiersz!), bo
per-wiersz zawyża długo trzymane karty.

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
RESOURCES = ["wood", "brick", "sheep", "wheat", "ore"]
OBSERVABLE_TYPES = {"ValueFunction", "AlphaBeta", "MCTS"}
# limity liczby kart development w grze (do sanity-checku played_*_total)
DEV_LIMITS = {"knight": 14, "victory_point": 5, "road_building": 2,
              "monopoly": 2, "year_of_plenty": 2}


def load_chunks(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not files:
        single = os.path.join(data_dir, f"{prefix}.parquet")
        if os.path.exists(single):
            files = [single]
    if not files:
        raise FileNotFoundError(f"Brak plikow {prefix}_*.parquet w {data_dir}")
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
    # jedna perspektywa na gre
    max_persp = ts.groupby("game_id")["observed_color"].nunique().max()
    results.append(check("dokladnie jedna perspektywa na gre", max_persp == 1,
                         f"max perspektyw/gre = {max_persp}"))

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

    # --- 3. SENSOWNOSC (w tym NOWE kolumny) ---
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

    # NOWE: bank w zakresie 0-19
    bank_cols = [f"bank_{r}" for r in RESOURCES]
    if all(c in ts.columns for c in bank_cols):
        bmin, bmax = ts[bank_cols].min().min(), ts[bank_cols].max().max()
        results.append(check("bank w zakresie 0-19", 0 <= bmin and bmax <= 19,
                             f"min={bmin} max={bmax}"))
    else:
        results.append(check("kolumny bank_* obecne", False, "BRAK"))

    # NOWE: produkcja nieujemna
    prod_cols = [c for c in ts.columns if "_prod_" in c]
    if prod_cols:
        results.append(check("produkcja nieujemna (>=0)",
                             (ts[prod_cols].min().min() >= 0),
                             f"{len(prod_cols)} kolumn, min={ts[prod_cols].min().min()}"))
    else:
        results.append(check("kolumny *_prod_* obecne", False, "BRAK"))

    # NOWE: played_*_total w limitach Catana
    pt_ok = True
    pt_detail = []
    for t, lim in DEV_LIMITS.items():
        col = f"played_{t}_total"
        if col in ts.columns:
            mx = ts[col].max()
            if mx > lim:
                pt_ok = False
                pt_detail.append(f"{t}={mx}>{lim}")
        else:
            pt_ok = False
            pt_detail.append(f"brak {col}")
    results.append(check("played_*_total w limitach kart", pt_ok,
                         ", ".join(pt_detail) if pt_detail else "ok"))

    # NOWE: dev_deck_remaining w zakresie 0-25
    if "dev_deck_remaining" in ts.columns:
        dmin, dmax = ts["dev_deck_remaining"].min(), ts["dev_deck_remaining"].max()
        results.append(check("dev_deck_remaining w zakresie 0-25",
                             0 <= dmin and dmax <= 25, f"min={dmin} max={dmax}"))
    else:
        results.append(check("kolumna dev_deck_remaining obecna", False, "BRAK"))

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

    print("\nRozklad 5-klasowego labelu (PER WIERSZ):")
    print(card["label"].value_counts().to_string())

    # --- DIAGNOSTYKA rounds_held: per wiersz vs PER POJEDYNCZA KARTA ---
    # pojedyncza karta = (game_id, observed_color, bought_at_action)
    life = (card.groupby(["game_id", "observed_color", "bought_at_action", "label"])
            ["rounds_held"].max().reset_index())
    print("\n--- DIAGNOSTYKA rounds_held ---")
    print("UWAGA: mediana PER WIERSZ zawyza dlugo trzymane karty (1 karta")
    print("trzymana N akcji = N wierszy). Honest metryka = czas zycia PER KARTA.\n")
    perrow = card.groupby("label")["rounds_held"].median()
    percard = life.groupby("label")["rounds_held"].median()
    ncards = life["label"].value_counts()
    cmp = pd.DataFrame({
        "mediana_per_wiersz": perrow,
        "mediana_per_karta": percard,
        "liczba_roznych_kart": ncards,
    }).fillna(0)
    print(cmp.to_string())
    print("\nMala 'liczba_roznych_kart' => mediana jest szumem (np. road_building,")
    print("monopoly, year_of_plenty sa kupowane rzadko). VP nigdy nie zagrywane")
    print("=> zawsze trzymane do konca (wysoki czas zycia jest poprawny).")

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