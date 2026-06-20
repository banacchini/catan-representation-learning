"""
Równoległy benchmark generowania danych z catanatron (multiprocessing).

Cel: zmierzyć ile REALNIE trwa generowanie na TWOIM laptopie przy pełnym
wykorzystaniu rdzeni, i wyekstrapolować do 10 000 gier.

Gry są od siebie niezależne (embarrassingly parallel), więc rozdzielamy je
na wszystkie rdzenie przez multiprocessing.Pool. Generowanie danych z Catana
jest CPU-bound - GPU (RTX 5060) tu nie pomaga w ogóle.

Uruchom (mierzy 50 gier domyślnie):
    python benchmark_parallel.py --num 50

Możesz też ograniczyć liczbę procesów:
    python benchmark_parallel.py --num 50 --workers 8

Wymaga catanatron z REPO (silne boty nie są w pip):
    git clone https://github.com/bcollazo/catanatron.git
    cd catanatron && pip install -e .
"""

import argparse
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

from catanatron import Game, Color, GameAccumulator
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.models.player import RandomPlayer, SimplePlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.mcts import MCTSPlayer

# Pula botów z wagami. MCTS i AlphaBeta są WOLNE - niska waga.
# To te same wagi co w generate_catan_data.py - dostrój pod budżet.
BOT_POOL = {
    "Random":         {"weight": 1, "factory": lambda c: RandomPlayer(c)},
    "WeightedRandom": {"weight": 7, "factory": lambda c: WeightedRandomPlayer(c)},
    "Simple":         {"weight": 2, "factory": lambda c: SimplePlayer(c)},
    "VictoryPoint":   {"weight": 7, "factory": lambda c: VictoryPointPlayer(c)},
    "ValueFunction":  {"weight": 10, "factory": lambda c: ValueFunctionPlayer(c)},
    "AlphaBeta":      {"weight": 10, "factory": lambda c: AlphaBetaPlayer(c)},
    "MCTS":           {"weight": 3, "factory": lambda c: MCTSPlayer(c)},
}

DEV_CARD_KEYS = [
    "KNIGHT_IN_HAND", "VICTORY_POINT_IN_HAND", "ROAD_BUILDING_IN_HAND",
    "MONOPOLY_IN_HAND", "YEAR_OF_PLENTY_IN_HAND",
]
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


class SampleCollector(GameAccumulator):
    def __init__(self, observed_index):
        self.obs_i = observed_index
        self.samples = 0
        self.card_types = Counter()

    def step(self, game_before_action, action):
        ps = game_before_action.state.player_state
        for key in DEV_CARD_KEYS:
            n = ps.get(f"P{self.obs_i}_{key}", 0)
            if n > 0:
                self.samples += n
                self.card_types[key] += n


def play_one_game(game_seed):
    """Rozgrywa jedną grę. Wykonywane w osobnym procesie."""
    random.seed(game_seed)
    names = list(BOT_POOL.keys())
    weights = [BOT_POOL[n]["weight"] for n in names]
    table = random.choices(names, weights=weights, k=4)
    players = [BOT_POOL[name]["factory"](COLORS[i]) for i, name in enumerate(table)]
    observed_index = random.randrange(4)
    observed_type = table[observed_index]

    collector = SampleCollector(observed_index)
    game = Game(players)
    game.play(accumulators=[collector])

    # diagnostyka win-rate (poza projektem) - kto wygrał?
    winning_color = game.winning_color()  # None jeśli limit tur
    winner_type = None
    if winning_color is not None:
        winner_index = COLORS.index(winning_color)
        winner_type = table[winner_index]

    return {
        "observed_type": observed_type,
        "samples": collector.samples,
        "card_types": dict(collector.card_types),
        "turns": game.state.num_turns,
        "table": table,            # kto siedział przy stole
        "winner_type": winner_type,  # kto wygrał (None = brak rozstrzygnięcia)
    }


def run(num_games, workers):
    seeds = list(range(num_games))

    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(play_one_game, seeds)
    dt = time.perf_counter() - t0

    total_samples = sum(r["samples"] for r in results)
    total_turns = sum(r["turns"] for r in results)
    obs_counter = Counter(r["observed_type"] for r in results)
    card_types = Counter()
    for r in results:
        card_types.update(r["card_types"])

    # --- diagnostyka win-rate (poza projektem) ---
    seats = Counter()       # ile razy bot zasiadł przy stole
    wins = Counter()        # ile razy wygrał
    undecided = 0           # gry bez rozstrzygnięcia (limit tur)
    for r in results:
        for name in r["table"]:
            seats[name] += 1
        if r["winner_type"] is None:
            undecided += 1
        else:
            wins[r["winner_type"]] += 1

    print("=" * 64)
    print(f"RÓWNOLEGŁY BENCHMARK: {num_games} gier na {workers} procesach")
    print("=" * 64)
    print(f"Czas (wall-clock):     {dt:.2f} s")
    print(f"Efektywny czas/gra:    {dt/num_games*1000:.0f} ms (wall-clock)")
    print(f"Śr. tur na grę:        {total_turns/num_games:.0f}")
    print(f"Próbek (karta-stan):   {total_samples:,}  ({total_samples/num_games:.0f}/gra)")
    print()
    print("Typ obserwowanego gracza:")
    for name in BOT_POOL:
        print(f"  {name:16s} {obs_counter.get(name,0):4d} gier")
    print()
    print("Rozkład klas kart (ground truth):")
    tot = sum(card_types.values()) or 1
    for k, v in card_types.most_common():
        print(f"  {k:24s} {v:8,}  ({100*v/tot:4.1f}%)")
    print()

    # --- diagnostyka win-rate (poza projektem, dla ciekawości) ---
    print("WIN-RATE per typ bota (zwycięstwa / zasiadania przy stole):")
    print("  (win-rate na poziomie losowym dla 4 graczy = 25%)")
    # sortuj po win-rate malejąco
    rows = []
    for name in BOT_POOL:
        s = seats.get(name, 0)
        w = wins.get(name, 0)
        wr = (100 * w / s) if s else 0.0
        rows.append((wr, name, w, s))
    for wr, name, w, s in sorted(rows, reverse=True):
        bar = "#" * int(wr / 2)  # prosty wykres tekstowy
        print(f"  {name:16s} {wr:5.1f}%  ({w:3d}/{s:3d})  {bar}")
    print(f"  gry bez rozstrzygnięcia (limit tur): {undecided}/{num_games}")
    print()

    target = 10_000
    print("-" * 64)
    print(f"EKSTRAPOLACJA do {target} gier na {workers} procesach:")
    print(f"  szac. czas:    {dt/num_games*target/60:.1f} min (wall-clock)")
    print(f"  szac. próbek:  ~{total_samples/num_games*target:,.0f}")
    print("-" * 64)
    print()
    print("Jeśli to nadal za wolno: obniż wagi MCTS/AlphaBeta w BOT_POOL")
    print("(to one zżerają większość czasu).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"liczba procesów (domyślnie wszystkie rdzenie: {os.cpu_count()})")
    args = parser.parse_args()
    print(f"Wykryto rdzeni: {os.cpu_count()}, używam procesów: {args.workers}\n")
    run(args.num, args.workers)