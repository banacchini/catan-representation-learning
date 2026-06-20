"""
Generator zbioru danych pod projekt:
"Uczenie reprezentacji stanu przekonań o ukrytych kartach development w Catan"

Produkuje JEDEN plik parquet w formacie DŁUGIM:
  jeden wiersz = jeden krok czasowy (gra x akcja).
Ten format obsługuje wszystkie trzy modele naraz:
  - baseline heurystyczny  -> używa kolumn cech bezpośrednio (per wiersz)
  - VAE-LSTM / RSSM        -> grupują po `game_id`, sortują po `action_index`,
                              biorą sekwencję wektorów cech jako wejście
Split train/test po niewidzianym stylu -> filtr `observed_type == "MCTS"`.

================  PROJEKT WEKTORA OBSERWACJI  ================
WEJŚCIE (jawne, widoczne dla każdego przeciwnika):
  Dla każdego z 4 graczy (p0..p3, kolejność stała = kolejność przy stole):
    p{i}_total_resources     - suma kart zasobów (w Catanie liczba jest jawna,
                               rozbicie na typy NIE)
    p{i}_n_dev_in_hand       - suma kart development na ręce (jawna liczba,
                               rozbicie na typy = TARGET, nie wchodzi do wejścia)
    p{i}_public_vp           - publiczne punkty (VICTORY_POINTS, BEZ ukrytych VP!)
    p{i}_played_knight..vp   - zagrane karty (jawne)
    p{i}_has_army            - ma najwiekszą armię
    p{i}_has_longest_road    - ma najdłuższą drogę
    p{i}_longest_road_len    - długość najdłuższej drogi
    p{i}_cities/settle/roads - postawione budynki
    p{i}_is_current          - czy to jego tura teraz
  Globalne:
    action_type_*            - one-hot typu akativnej akcji
    robber_on_observed       - czy złodziej blokuje obserwowanego (kluczowy sygnał!)
    num_turns                - postęp gry
  Historia OBSERWOWANEGO gracza (stanowy sygnał - serce zadania):
    obs_rounds_since_buy     - ile akcji od ostatniego kupna karty dev
    obs_rounds_since_play    - ile akcji od ostatniego zagrania karty dev
    obs_total_dev_bought     - ile kart dev kupił łącznie
    obs_total_dev_played     - ile kart dev zagrał łącznie

TARGET (ukryte - to przewidujemy):
    y_knight, y_vp, y_road_building, y_monopoly, y_year_of_plenty
    = liczba kart każdego typu na ręce OBSERWOWANEGO gracza

METADANE (do splitu i analizy):
    game_id, action_index, observed_type, observed_index,
    table (skład stołu jako str), winner_type, game_phase
=============================================================

Uruchom:
    uv run generate_dataset.py --num 100 --out data/catan_dev.parquet
    uv run generate_dataset.py --num 10000 --out data/catan_dev.parquet --workers 16
"""

import argparse
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

import pandas as pd

from catanatron import Game, Color, GameAccumulator
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.models.player import RandomPlayer, SimplePlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.mcts import MCTSPlayer

# ---- pula botów (skład stołu) ----
BOT_POOL = {
    "Random":         {"weight": 1, "factory": lambda c: RandomPlayer(c)},
    "WeightedRandom": {"weight": 3, "factory": lambda c: WeightedRandomPlayer(c)},
    "Simple":         {"weight": 2, "factory": lambda c: SimplePlayer(c)},
    "VictoryPoint":   {"weight": 3, "factory": lambda c: VictoryPointPlayer(c)},
    "ValueFunction":  {"weight": 6, "factory": lambda c: ValueFunctionPlayer(c)},
    "AlphaBeta":      {"weight": 5, "factory": lambda c: AlphaBetaPlayer(c)},
    "MCTS":           {"weight": 1, "factory": lambda c: MCTSPlayer(c)},  # tylko test
}

# obserwujemy (przewidujemy karty) TYLKO dla silnych botów - ich gra kartami
# jest strategiczna, więc niesie sygnał. Słabe boty grają losowo = szum.
OBSERVABLE_TYPES = {"ValueFunction", "AlphaBeta", "MCTS"}

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_TARGETS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "MONOPOLY", "YEAR_OF_PLENTY"]
# typy akcji do one-hot (najczęstsze; reszta -> OTHER)
ACTION_TYPES = [
    "ROLL", "END_TURN", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD",
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING", "PLAY_MONOPOLY",
    "PLAY_YEAR_OF_PLENTY", "MARITIME_TRADE",
]


def sample_table(rng):
    names = list(BOT_POOL.keys())
    weights = [BOT_POOL[n]["weight"] for n in names]
    return rng.choices(names, weights=weights, k=4)


def robber_blocks_observed(state, observed_color):
    """Czy złodziej blokuje obserwowanego gracza?
    = czy obserwowany ma budynek na którymkolwiek z 6 węzłów kafelka złodzieja.
    To kluczowy sygnał: gracz zablokowany przez złodzieja, który NIE zagrywa
    rycerza, prawdopodobnie go nie ma."""
    b = state.board
    rc = b.robber_coordinate
    tile = b.map.tiles.get(rc)
    if tile is None or not hasattr(tile, "nodes"):
        return 0
    for node_id in tile.nodes.values():
        bld = b.buildings.get(node_id)
        if bld is not None and bld[0] == observed_color:
            return 1
    return 0


def player_public_features(ps, i, is_current):
    """Cechy JAWNE gracza i (bez wycieku ukrytych kart)."""
    pre = f"P{i}_"
    total_res = sum(ps.get(f"{pre}{r}_IN_HAND", 0) for r in RESOURCES)
    n_dev = sum(ps.get(f"{pre}{d}_IN_HAND", 0) for d in DEV_TARGETS)  # SUMA, jawna
    return {
        f"p{i}_total_resources": total_res,
        f"p{i}_n_dev_in_hand": n_dev,
        f"p{i}_public_vp": ps.get(f"{pre}VICTORY_POINTS", 0),  # NIE ACTUAL!
        f"p{i}_played_knight": ps.get(f"{pre}PLAYED_KNIGHT", 0),
        f"p{i}_played_monopoly": ps.get(f"{pre}PLAYED_MONOPOLY", 0),
        f"p{i}_played_road_building": ps.get(f"{pre}PLAYED_ROAD_BUILDING", 0),
        f"p{i}_played_yop": ps.get(f"{pre}PLAYED_YEAR_OF_PLENTY", 0),
        f"p{i}_played_vp": ps.get(f"{pre}PLAYED_VICTORY_POINT", 0),
        f"p{i}_has_army": int(bool(ps.get(f"{pre}HAS_ARMY", False))),
        f"p{i}_has_longest_road": int(bool(ps.get(f"{pre}HAS_ROAD", False))),
        f"p{i}_longest_road_len": ps.get(f"{pre}LONGEST_ROAD_LENGTH", 0),
        f"p{i}_cities_built": 4 - ps.get(f"{pre}CITIES_AVAILABLE", 4),
        f"p{i}_settlements_built": 5 - ps.get(f"{pre}SETTLEMENTS_AVAILABLE", 5),
        f"p{i}_roads_built": 15 - ps.get(f"{pre}ROADS_AVAILABLE", 15),
        f"p{i}_is_current": int(is_current),
    }


class RowCollector(GameAccumulator):
    """Buduje wiersze (per akcja) dla JEDNEJ gry."""

    def __init__(self, observed_index, game_id):
        self.obs_i = observed_index
        self.game_id = game_id
        self.rows = []
        self.action_index = 0
        # stanowe liczniki historii obserwowanego gracza
        self.last_buy_step = None
        self.last_play_step = None
        self.total_bought = 0
        self.total_played = 0

    def step(self, game_before_action, action):
        st = game_before_action.state
        ps = st.player_state
        i = self.obs_i

        # --- aktualizacja historii NA PODSTAWIE akcji obserwowanego ---
        act_name = str(action.action_type).split(".")[-1]
        is_observed_acting = (action.color == COLORS[i])
        # (liczniki aktualizujemy PO zapisaniu wiersza, by cecha opisywała
        #  stan SPRZED bieżącej akcji - bez wycieku przyszłości)

        # --- cechy per gracz (jawne) ---
        feats = {}
        current_color = st.current_color()
        for p in range(4):
            feats.update(player_public_features(ps, p, COLORS[p] == current_color))

        # --- globalne ---
        for at in ACTION_TYPES:
            feats[f"action_{at}"] = int(act_name == at)
        feats["action_OTHER"] = int(act_name not in ACTION_TYPES)

        # złodziej blokujący obserwowanego? (kluczowy sygnał wg pomysłu)
        feats["robber_on_observed"] = robber_blocks_observed(st, COLORS[i])
        feats["num_turns"] = st.num_turns

        # --- historia obserwowanego (stan SPRZED tej akcji) ---
        feats["obs_rounds_since_buy"] = (
            self.action_index - self.last_buy_step if self.last_buy_step is not None else -1)
        feats["obs_rounds_since_play"] = (
            self.action_index - self.last_play_step if self.last_play_step is not None else -1)
        feats["obs_total_dev_bought"] = self.total_bought
        feats["obs_total_dev_played"] = self.total_played

        # --- TARGET: ukryte karty obserwowanego ---
        pre = f"P{i}_"
        for d in DEV_TARGETS:
            feats[f"y_{d.lower()}"] = ps.get(f"{pre}{d}_IN_HAND", 0)

        # --- metadane ---
        feats["game_id"] = self.game_id
        feats["action_index"] = self.action_index

        self.rows.append(feats)
        self.action_index += 1

        # --- TERAZ aktualizujemy liczniki (po zapisaniu wiersza) ---
        if is_observed_acting:
            if act_name == "BUY_DEVELOPMENT_CARD":
                self.last_buy_step = self.action_index
                self.total_bought += 1
            elif act_name in ("PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
                              "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"):
                self.last_play_step = self.action_index
                self.total_played += 1


def play_one_game(game_id):
    rng = random.Random(game_id)
    # losuj stół aż obserwowany będzie silnym botem
    for _ in range(50):
        table = sample_table(rng)
        strong_positions = [k for k, name in enumerate(table)
                            if name in OBSERVABLE_TYPES]
        if strong_positions:
            observed_index = rng.choice(strong_positions)
            break
    else:
        # awaryjnie wymuś AlphaBeta na pozycji 0
        table[0] = "AlphaBeta"
        observed_index = 0

    players = [BOT_POOL[name]["factory"](COLORS[k]) for k, name in enumerate(table)]
    collector = RowCollector(observed_index, game_id)
    game = Game(players)
    game.play(accumulators=[collector])

    winning_color = game.winning_color()
    winner_type = table[COLORS.index(winning_color)] if winning_color else None
    phase_cut = game.state.num_turns

    # dopnij metadane stałe dla gry do każdego wiersza
    table_str = "|".join(table)
    observed_type = table[observed_index]
    for r in collector.rows:
        r["observed_type"] = observed_type
        r["observed_index"] = observed_index
        r["table"] = table_str
        r["winner_type"] = winner_type
        # faza gry: ostatnia 1/3 akcji = "late"
        r["game_phase"] = "late" if r["action_index"] > 0.66 * len(collector.rows) else (
            "early" if r["action_index"] < 0.33 * len(collector.rows) else "mid")
    return collector.rows


def generate(num_games, out_path, workers):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        all_rows_nested = pool.map(play_one_game, range(num_games))
    dt = time.perf_counter() - t0

    rows = [r for game_rows in all_rows_nested for r in game_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False, compression="snappy")

    # raport
    print("=" * 64)
    print(f"WYGENEROWANO: {num_games} gier -> {out_path}")
    print("=" * 64)
    print(f"Czas:            {dt:.1f}s  ({dt/num_games*1000:.0f} ms/gra)")
    print(f"Wierszy (kroków):{len(df):,}")
    print(f"Kolumn:          {df.shape[1]}")
    print(f"Rozmiar pliku:   {os.path.getsize(out_path)/1e6:.1f} MB")
    print()
    print("Obserwowany typ (powinny być tylko silne boty):")
    print(df["observed_type"].value_counts().to_string())
    print()
    print("Rozkład targetu y_knight (przykład klasy):")
    print(df["y_knight"].value_counts().sort_index().to_string())
    print()
    print("Split niewidzianego stylu:")
    test_mask = df["observed_type"] == "MCTS"
    print(f"  TEST (MCTS):  {test_mask.sum():,} wierszy")
    print(f"  TRAIN/VAL:    {(~test_mask).sum():,} wierszy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/catan_dev.parquet")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    generate(args.num, args.out, args.workers)