"""
Generator v2 zbioru Catan — per-karta, wiele perspektyw.

Produkuje DWIE powiązane tabele parquet:
  timesteps.parquet    - pełna sekwencja per akcja, per perspektywa.
                         Wejście dla VAE-LSTM / RSSM. Zawiera WSZYSTKIE kroki
                         (też te bez kart na ręce) - sekwencja musi być pełna.
  card_samples.parquet - jedna próbka na POJEDYNCZĄ kartę trzymaną w danym
                         kroku. Cel: 5-klasowa klasyfikacja typu karty.
                         Łączy się z timesteps po (game_id, observed_index,
                         action_index).

Decyzje (wg wyboru użytkownika):
  - target: per-karta, 5-klasowy softmax (label = prawdziwy typ karty)
  - granulacja: per akcja + flaga is_observed_turn_start (do filtrowania)
  - perspektywy: KAŻDY silny bot przy stole = osobna perspektywa (więcej danych)

KLUCZ: typ pojedynczej karty znamy z action_records PO grze
(BUY_DEVELOPMENT_CARD.value = prawdziwy typ). Śledzimy karty rekonstruując
zakupy/zagrania. Cecha per-karta `rounds_held` (ile akcji karta jest na ręce)
to najsilniejszy sygnał: karta trzymana długo bez zagrania -> prawdopodobnie VP.

GWARANCJA POPRAWNOŚCI: w każdym kroku śledzona ręka MUSI zgadzać się z
IN_HAND z player_state (assert w kodzie).

Uruchom:
    uv run generate_dataset_v2.py --num 100 --out-dir data
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

BOT_POOL = {
    "Random":         {"weight": 1, "factory": lambda c: RandomPlayer(c)},
    "WeightedRandom": {"weight": 7, "factory": lambda c: WeightedRandomPlayer(c)},
    "Simple":         {"weight": 2, "factory": lambda c: SimplePlayer(c)},
    "VictoryPoint":   {"weight": 7, "factory": lambda c: VictoryPointPlayer(c)},
    "ValueFunction":  {"weight": 10, "factory": lambda c: ValueFunctionPlayer(c)},
    "AlphaBeta":      {"weight": 10, "factory": lambda c: AlphaBetaPlayer(c)},
    "MCTS":           {"weight": 3, "factory": lambda c: MCTSPlayer(c)},
}
OBSERVABLE_TYPES = {"ValueFunction", "AlphaBeta", "MCTS"}
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DEV_TYPES = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "MONOPOLY", "YEAR_OF_PLENTY"]
PLAY_TO_TYPE = {  # akcja zagrania -> typ usuwanej karty
    "PLAY_KNIGHT_CARD": "KNIGHT",
    "PLAY_MONOPOLY": "MONOPOLY",
    "PLAY_ROAD_BUILDING": "ROAD_BUILDING",
    "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY",
}
ACTION_TYPES = [
    "ROLL", "END_TURN", "MOVE_ROBBER", "DISCARD_RESOURCE", "BUILD_ROAD",
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING", "PLAY_MONOPOLY",
    "PLAY_YEAR_OF_PLENTY", "MARITIME_TRADE",
]


def robber_blocks(state, observed_color):
    b = state.board
    tile = b.map.tiles.get(b.robber_coordinate)
    if tile is None or not hasattr(tile, "nodes"):
        return 0
    for node_id in tile.nodes.values():
        bld = b.buildings.get(node_id)
        if bld is not None and bld[0] == observed_color:
            return 1
    return 0


class Snapshotter(GameAccumulator):
    """Zapisuje lekki snapshot stanu PRZED każdą akcją (raz na akcję,
    współdzielony przez wszystkie perspektywy)."""

    def __init__(self):
        self.snaps = []

    def step(self, game_before_action, action):
        st = game_before_action.state
        ps = st.player_state
        cur = st.current_color()
        snap = {
            "action_type": str(action.action_type).split(".")[-1],
            "action_color": action.color,
            "current_color": cur,
            "num_turns": st.num_turns,
            "ps": {k: ps[k] for k in ps},  # płytka kopia scalarów
            "robber_block": {COLORS[i]: robber_blocks(st, COLORS[i]) for i in range(4)},
            "color_to_index": dict(st.color_to_index),  # MAPA: kolor -> P-index!
        }
        self.snaps.append(snap)


def player_public_features(ps, pi, is_current, rel_pos):
    """Czyta stan gracza o P-indeksie `pi` z player_state, ale NAZYWA kolumny
    wg pozycji względnej `rel_pos` (p0=obserwowany, p1..p3=przeciwnicy).
    Rozdzielenie jest kluczowe: catanatron tasuje graczy, więc P-index do
    odczytu != pozycja w wektorze cech."""
    pre = f"P{pi}_"
    rp = rel_pos
    return {
        f"p{rp}_total_resources": sum(ps.get(f"{pre}{r}_IN_HAND", 0) for r in RESOURCES),
        f"p{rp}_n_dev_in_hand": sum(ps.get(f"{pre}{d}_IN_HAND", 0) for d in DEV_TYPES),
        f"p{rp}_public_vp": ps.get(f"{pre}VICTORY_POINTS", 0),
        f"p{rp}_played_knight": ps.get(f"{pre}PLAYED_KNIGHT", 0),
        f"p{rp}_played_monopoly": ps.get(f"{pre}PLAYED_MONOPOLY", 0),
        f"p{rp}_played_road_building": ps.get(f"{pre}PLAYED_ROAD_BUILDING", 0),
        f"p{rp}_played_yop": ps.get(f"{pre}PLAYED_YEAR_OF_PLENTY", 0),
        f"p{rp}_played_vp": ps.get(f"{pre}PLAYED_VICTORY_POINT", 0),
        f"p{rp}_has_army": int(bool(ps.get(f"{pre}HAS_ARMY", False))),
        f"p{rp}_has_longest_road": int(bool(ps.get(f"{pre}HAS_ROAD", False))),
        f"p{rp}_longest_road_len": ps.get(f"{pre}LONGEST_ROAD_LENGTH", 0),
        f"p{rp}_cities_built": 4 - ps.get(f"{pre}CITIES_AVAILABLE", 4),
        f"p{rp}_settlements_built": 5 - ps.get(f"{pre}SETTLEMENTS_AVAILABLE", 5),
        f"p{rp}_roads_built": 15 - ps.get(f"{pre}ROADS_AVAILABLE", 15),
        f"p{rp}_is_current": int(is_current),
    }


def build_perspective(snaps, observed_color, game_id, table_str,
                      observed_type, winner_type):
    """Buduje wiersze timesteps + card_samples dla JEDNEJ perspektywy.
    observed_color = KOLOR obserwowanego gracza (jednoznaczny).
    P-index w player_state wyznaczamy z color_to_index (catanatron tasuje
    kolejność graczy, więc COLORS[i] != P{i}!)."""
    n = len(snaps)
    ts_rows, card_rows = [], []

    hand = []
    last_buy = None
    last_play = None
    total_bought = 0
    total_played = 0
    prev_current = None

    for idx, snap in enumerate(snaps):
        ps = snap["ps"]
        cur = snap["current_color"]
        c2i = snap["color_to_index"]
        obs_pi = c2i[observed_color]  # WŁAŚCIWY P-index obserwowanego

        # --- reconciliation: śledzona ręka == IN_HAND ---
        tracked = Counter(t for t, _ in hand)
        actual = {d: ps.get(f"P{obs_pi}_{d}_IN_HAND", 0) for d in DEV_TYPES}
        for d in DEV_TYPES:
            assert tracked.get(d, 0) == actual[d], (
                f"NIEZGODNOŚĆ ręki game {game_id} obs {observed_color} "
                f"action {idx}: tracked={dict(tracked)} actual={actual}")

        is_turn_start = (cur == observed_color) and (prev_current != observed_color)

        # --- wiersz timestep ---
        feats = {}
        # cechy per gracz: iterujemy po WSZYSTKICH 4 graczach wg ich P-index,
        # ale w stałej kolejności względem obserwowanego (p0=obserwowany,
        # p1..p3 = przeciwnicy wg P-index) - dzięki temu model ma spójny układ
        order = [observed_color] + [c for c in c2i if c != observed_color]
        for rel_pos, color in enumerate(order):
            pi = c2i[color]
            feats.update(player_public_features(ps, pi, color == cur,
                                                rel_pos))
        for at in ACTION_TYPES:
            feats[f"action_{at}"] = int(snap["action_type"] == at)
        feats["action_OTHER"] = int(snap["action_type"] not in ACTION_TYPES)
        feats["robber_on_observed"] = snap["robber_block"][observed_color]
        feats["num_turns"] = snap["num_turns"]
        feats["obs_rounds_since_buy"] = (idx - last_buy) if last_buy is not None else -1
        feats["obs_rounds_since_play"] = (idx - last_play) if last_play is not None else -1
        feats["obs_total_dev_bought"] = total_bought
        feats["obs_total_dev_played"] = total_played
        feats["n_hidden_cards"] = len(hand)
        feats["is_observed_turn_start"] = int(is_turn_start)
        for d in DEV_TYPES:
            feats[f"y_{d.lower()}"] = actual[d]
        feats.update({
            "game_id": game_id, "action_index": idx,
            "observed_color": str(observed_color), "observed_type": observed_type,
            "table": table_str, "winner_type": winner_type,
        })
        ts_rows.append(feats)

        # --- próbki per-karta ---
        for card_no, (ctype, bought_at) in enumerate(hand):
            card_rows.append({
                "game_id": game_id, "action_index": idx,
                "observed_color": str(observed_color), "observed_type": observed_type,
                "card_slot": card_no,
                "rounds_held": idx - bought_at,
                "bought_at_action": bought_at,
                "is_observed_turn_start": int(is_turn_start),
                "n_hidden_cards": len(hand),
                "label": ctype,
            })

        prev_current = cur

        # --- zastosuj akcję idx do śledzonej ręki ---
        if snap["action_color"] == observed_color:
            at = snap["action_type"]
            if at == "BUY_DEVELOPMENT_CARD":
                bt = snap.get("buy_type")
                if bt is not None:
                    hand.append((bt, idx))
                    last_buy = idx
                    total_bought += 1
            elif at in PLAY_TO_TYPE:
                rt = PLAY_TO_TYPE[at]
                for k, (ct, ba) in enumerate(hand):
                    if ct == rt:
                        hand.pop(k)
                        break
                last_play = idx
                total_played += 1

    return ts_rows, card_rows


def play_one_game(game_id):
    rng = random.Random(game_id)
    random.seed(game_id)  # boty używają globalnego random - seedujemy dla powtarzalności
    for _ in range(50):
        table = rng.choices(list(BOT_POOL.keys()),
                            weights=[BOT_POOL[n]["weight"] for n in BOT_POOL], k=4)
        strong = [k for k, name in enumerate(table) if name in OBSERVABLE_TYPES]
        if strong:
            break
    else:
        table[0] = "AlphaBeta"; strong = [0]

    players = [BOT_POOL[name]["factory"](COLORS[k]) for k, name in enumerate(table)]
    snapper = Snapshotter()
    game = Game(players)
    game.play(accumulators=[snapper])

    # wzbogać snapshoty o typ kupionej karty z action_records (rozstrzygnięty PO grze)
    recs = game.state.action_records
    for idx, snap in enumerate(snapper.snaps):
        if idx < len(recs) and snap["action_type"] == "BUY_DEVELOPMENT_CARD":
            snap["buy_type"] = recs[idx].action.value  # np. 'KNIGHT'

    winning_color = game.winning_color()
    winner_type = table[COLORS.index(winning_color)] if winning_color else None
    table_str = "|".join(table)

    all_ts, all_card = [], []
    for oi in strong:  # KAŻDY silny bot = osobna perspektywa
        observed_color = COLORS[oi]
        ts, card = build_perspective(snapper.snaps, observed_color, game_id,
                                     table_str, table[oi], winner_type)
        all_ts.extend(ts)
        all_card.extend(card)
    return all_ts, all_card


def generate(num_games, out_dir, workers, chunk_size=1000, start_id=0):
    import math
    import gc
    
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    
    num_chunks = math.ceil(num_games / chunk_size)
    total_ts_rows = 0
    total_card_rows = 0
    
    print("=" * 64)
    print(f"URUCHOMIENIE GENERATORA: {num_games} gier (ID od {start_id} do {start_id + num_games - 1})")
    print("=" * 64)

    for chunk_idx in range(num_chunks):
        # Przesunięcie identyfikatorów o start_id
        chunk_start_id = start_id + (chunk_idx * chunk_size)
        chunk_end_id = min(chunk_start_id + chunk_size, start_id + num_games)
        games_in_chunk = range(chunk_start_id, chunk_end_id)
        
        print(f"[{chunk_idx + 1}/{num_chunks}] Generowanie gier od ID {chunk_start_id} do {chunk_end_id - 1}...")
        
        with Pool(processes=workers) as pool:
            results = pool.map(play_one_game, games_in_chunk)
            
        ts_rows = [r for ts, _ in results for r in ts]
        card_rows = [r for _, card in results for r in card]
        
        total_ts_rows += len(ts_rows)
        total_card_rows += len(card_rows)
        
        ts_df = pd.DataFrame(ts_rows)
        card_df = pd.DataFrame(card_rows)
        
        # Bezpieczne nazywanie plików po ID gier
        ts_path = os.path.join(out_dir, f"timesteps_{chunk_start_id:05d}_to_{chunk_end_id-1:05d}.parquet")
        card_path = os.path.join(out_dir, f"card_samples_{chunk_start_id:05d}_to_{chunk_end_id-1:05d}.parquet")
        
        ts_df.to_parquet(ts_path, index=False, compression="snappy")
        card_df.to_parquet(card_path, index=False, compression="snappy")
        
        print(f" -> Zapisano {len(ts_df):,} kroków czasowych do pliku {os.path.basename(ts_path)}")
        
        del results, ts_rows, card_rows, ts_df, card_df
        gc.collect()

    dt = time.perf_counter() - t0
    print("=" * 64)
    print(f"ZAKOŃCZONO: Wygenerowano gry od {start_id} do {start_id + num_games - 1}.")
    print(f"Całkowity czas pracy: {dt:.1f}s ({dt/num_games*1000:.0f} ms/gra)")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--start-id", type=int, default=0, help="Początkowe ID gry (np. do wznawiania generowania)")
    args = parser.parse_args()
    
    generate(args.num, args.out_dir, args.workers, args.chunk_size, args.start_id)