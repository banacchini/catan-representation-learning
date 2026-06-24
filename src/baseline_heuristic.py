"""Heurystyczny baseline (nie-uczony) dla klasyfikacji kart rozwoju.

Dla kazdej probki-karty (na starcie tury dowolnego gracza) buduje rozklad p nad 5
klasami startujac od skladu talii kart rozwoju (prior) i nakladajac multiplikatywne
korekty wynikajace z regul gry, a nastepnie zwraca argmax. To czysta heurystyka:
wszystkie mnozniki i progi to recznie ustawione stale (HeuristicParams), bez zadnego
dopasowania do danych. Sluzy jako interpretowalna dolna granica odniesienia ponizej
modeli SSL (VAE/RSSM) trenowanych przez src.train_final.

Kluczowa zasada: dowod "nie zagrano karty" akumuluje sie RAZ NA TURE GRACZA
OBSERWOWANEGO. W Catanie mozna zagrac tylko jedna karte developmentu na ture, wiec:
  * liczy sie tylko tura gracza obserwowanego (tury przeciwnikow nic nie wnosza),
  * tura, w ktorej obserwowany juz zagral jakas karte developmentu, NIE jest
    niewykorzystana okazja (wykluczamy ja),
  * sila kary rosnie z kazda kolejna taka tura (kompounding) oraz z nadwyzka
    zasobow / brakiem zasobu w banku w danej turze.

Wynik zapisywany do results/baseline_metrics.json w TYM SAMYM ksztalcie co
final_metrics.json["final"] (pseudo-rodzina "heuristic", n_seeds=1, deterministyczne),
co umozliwia bezposrednie zestawienie w notebooku porownawczym.

Uruchom:
    python -m src.baseline_heuristic                       # pelny TEST
    python -m src.baseline_heuristic --eval-split test --games 200   # smoke
"""
import argparse
import dataclasses
import json
import os
from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import (LABEL_TO_IDX, LABELS, compute_turn_starts,
                   filter_to_turn_starts, load_split)
from .probe import _f1_report, _grouped_report

# indeksy klas w LABELS = ["KNIGHT","VICTORY_POINT","ROAD_BUILDING","MONOPOLY","YEAR_OF_PLENTY"]
K, VP, RB, MONO, YOP = 0, 1, 2, 3, 4

# standardowy sklad talii kart rozwoju w Catanie (K, VP, RB, MONO, YOP) — suma 25
INIT_DECK = np.array([14.0, 5.0, 2.0, 2.0, 2.0])

PLAYED_TOTAL_COLS = ["played_knight_total", "played_victory_point_total",
                     "played_road_building_total", "played_monopoly_total",
                     "played_year_of_plenty_total"]
BANK_COLS = ["bank_wood", "bank_brick", "bank_sheep", "bank_wheat", "bank_ore"]
# kolumny ze startow tury gracza obserwowanego, potrzebne do akumulacji okazji
OBS_TURN_COLS = ["robber_on_observed", "p0_total_resources",
                 "obs_total_dev_played"] + BANK_COLS


@dataclass
class HeuristicParams:
    """Recznie ustawione stale regul (zero dopasowania do danych). Mnozniki
    a_*/d_* dzialaja PER TURA GRACZA OBSERWOWANEGO (kompounding), wiec sa bliskie 1."""
    eps: float = 0.1            # dolne obciecie skladu talii (po odjeciu zagranych)
    # Regula A — rycerz: za kazda ture obserwowanego pod blokada robbera, w ktorej
    # nie zagral karty (nie odblokowal sie), P(KNIGHT) maleje.
    a_knight: float = 0.45      # P(KNIGHT) *= a_knight ** (liczba takich tur)
    # Reguly B/C — Year of Plenty / Road Building: za kazda ture obserwowanego z nadwyzka
    # zasobow ponad prog (a niezagrana karta) prior maleje; wyklad = SUMA nadwyzek po turach.
    resource_thr: float = 4.0
    d_yop: float = 0.80         # P(YOP) *= d_yop ** sum(excess_zasobow po turach okazji)
    d_rb: float = 0.80          # P(RB)  *= d_rb  ** sum(excess_zasobow po turach okazji)
    # Regula D — Monopol (proxy): niski bank zasobu => duzo zasobu na rekach (cel monopolu);
    # za kazda ture obserwowanego z niskim bankiem (a niezagranym monopolem) prior maleje.
    bank_low_thr: float = 4.0   # excess = max(0, bank_low_thr - min(bank_*)) na ture
    d_mono: float = 0.80        # P(MONO) *= d_mono ** sum(excess_banku po turach okazji)
    # Regula E — Victory Point: rosnie z liczba tur obserwowanego, w ktorych karta
    # byla trzymana i nie zostala zagrana (VP nigdy nie schodzi z reki).
    b_vp: float = 4.0           # P(VP) *= 1 + b_vp * min(n_tur_trzymania / vp_T0, vp_cap)
    vp_T0: float = 4.0          # skala (liczba tur obserwowanego) do nasycenia
    vp_cap: float = 3.0         # gorne obciecie czynnika wzrostu
    # Nasycenie dowodu: kara A/B/C/D nie moze wyzerowac klasy (wyklad obciety).
    cap_acc: float = 6.0        # max wartosc wykladnika akumulowanej kary
    # UWAGA: brak pozytywnych "boostow" dla RB/MONO/YOP. Karta developmentu jest losowana
    # ze stosu — gracz NIE wybiera typu — wiec stan/strategia gracza nie czyni konkretnego
    # typu bardziej prawdopodobnym NA RECE. Jedyny wazny prior to sklad talii (acquisition
    # losowy), a jedyne wazne korekty to zachowanie GRANIA (A/B/C/D) i fakt, ze VP nie da
    # sie zagrac (E). Z tego powodu RB/MONO/YOP nie wychodza jako argmax — to uczciwa,
    # strukturalna wlasnosc baseline'u (motywacja dla uczonych modeli).


def _obs_turn_tables(ts, params):
    """Per sekwencja (game_id, observed_color): posortowane action_index STARTOW TURY
    GRACZA OBSERWOWANEGO oraz skumulowane sumy sygnalow "niewykorzystanej okazji"
    (z prefiksem 0, do range-sum przez searchsorted).

    Tura jest okazja tylko jesli obserwowany NIE zagral w niej karty developmentu —
    wykrywane po przyroscie obs_total_dev_played do nastepnego startu tury obserwowanego.
    """
    starts = compute_turn_starts(ts)                       # + current_rel_pos
    obs = starts[starts.current_rel_pos == 0][["game_id", "observed_color", "action_index"]]
    key = ["game_id", "observed_color", "action_index"]
    obs = obs.merge(ts[key + OBS_TURN_COLS], on=key, how="left")
    obs = obs.sort_values(key)

    tables = {}
    for (gid, color), g in obs.groupby(["game_id", "observed_color"], sort=False):
        ai = g["action_index"].to_numpy()
        dev_played = g["obs_total_dev_played"].to_numpy(dtype=float)
        # przyrost licznika zagranych kart do nastepnej tury obserwowanego => zagral w tej turze
        nxt = np.append(dev_played[1:], dev_played[-1])    # ostatnia tura: brak nastepnej -> 0
        no_play = (nxt - dev_played) <= 0                  # True = realna, niewykorzystana okazja
        res = g["p0_total_resources"].to_numpy(dtype=float)
        bank_min = g[BANK_COLS].to_numpy(dtype=float).min(axis=1)

        knight_opp = (g["robber_on_observed"].to_numpy() == 1) & no_play
        res_opp = np.clip(res - params.resource_thr, 0.0, None) * no_play
        mono_opp = np.clip(params.bank_low_thr - bank_min, 0.0, None) * no_play
        turn_opp = no_play.astype(float)                   # do reguly VP (liczba tur trzymania)

        def cumsum0(a):  # skumulowana suma z prefiksem 0 (do range-sum przez searchsorted)
            return np.concatenate([[0.0], np.cumsum(a)])
        tables[(int(gid), str(color))] = {
            "ai": ai, "ck": cumsum0(knight_opp.astype(float)), "cr": cumsum0(res_opp),
            "cm": cumsum0(mono_opp), "ct": cumsum0(turn_opp),
        }
    return tables


def _attach_evidence(m, tables):
    """Dla kazdej probki-karty sumuje sygnaly okazji po turach obserwowanego, w ktorych
    karta byla trzymana: bought_at_action <= tura <= action_index (range-sum, searchsorted).
    Dodaje kolumny: acc_knight, acc_res, acc_mono, n_turns_held."""
    g = m.reset_index(drop=True)
    n = len(g)
    out = {k: np.zeros(n) for k in ("acc_knight", "acc_res", "acc_mono", "n_turns_held")}
    a_all = g["action_index"].to_numpy()
    b_all = g["bought_at_action"].to_numpy()
    for (gid, color), pos in g.groupby(["game_id", "observed_color"], sort=False).indices.items():
        T = tables.get((int(gid), str(color)))
        if T is None:
            continue
        ai = T["ai"]
        lo = np.searchsorted(ai, b_all[pos], side="left")   # pierwsza tura >= bought_at_action
        hi = np.searchsorted(ai, a_all[pos], side="right")  # pierwsza tura > action_index
        out["acc_knight"][pos] = T["ck"][hi] - T["ck"][lo]
        out["acc_res"][pos] = T["cr"][hi] - T["cr"][lo]
        out["acc_mono"][pos] = T["cm"][hi] - T["cm"][lo]
        out["n_turns_held"][pos] = T["ct"][hi] - T["ct"][lo]
    for k, v in out.items():
        g[k] = v
    return g


def predict_proba(df, params=None):
    """Wektor prawdopodobienstw [N,5] nad LABELS. Wymaga kolumn PLAYED_TOTAL_COLS oraz
    akumulatorow z _attach_evidence (acc_knight, acc_res, acc_mono, n_turns_held).

    Tylko WAZNE sygnaly: prior ze skladu talii (acquisition losowy) + korekty SUPRESYJNE
    z zachowania grania (A/B/C/D) + retencja VP (E). Brak pozytywnych boostow opartych na
    strategii gracza — gracz nie wybiera typu losowanej karty."""
    params = params or HeuristicParams()

    # --- prior: sklad talii pomniejszony o publicznie zagrane karty, znormalizowany ---
    played = df[PLAYED_TOTAL_COLS].to_numpy(dtype=float)           # [N,5]
    remaining = np.clip(INIT_DECK[None, :] - played, params.eps, None)
    p = remaining / remaining.sum(axis=1, keepdims=True)           # [N,5]

    # --- korekty SUPRESYJNE akumulowane po turach gracza obserwowanego ---
    # wyklad obciety do cap_acc — dowod nasyca sie, klasa nie spada do zera
    acc_res = np.minimum(df["acc_res"].to_numpy(), params.cap_acc)
    p[:, K] *= params.a_knight ** np.minimum(df["acc_knight"].to_numpy(), params.cap_acc)
    p[:, YOP] *= params.d_yop ** acc_res
    p[:, RB] *= params.d_rb ** acc_res
    p[:, MONO] *= params.d_mono ** np.minimum(df["acc_mono"].to_numpy(), params.cap_acc)

    # --- regula E: Victory Point rosnie z liczba tur trzymania (VP nie da sie zagrac) ---
    nth = df["n_turns_held"].to_numpy()
    p[:, VP] *= 1.0 + params.b_vp * np.minimum(nth / params.vp_T0, params.vp_cap)

    return p / p.sum(axis=1, keepdims=True)


def _agg(vals):
    """Mean dla pojedynczego ziarna (deterministyczne -> ci95=0). Kszalt jak
    train_final._agg, by JSON byl zgodny z final_metrics.json."""
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": 0.0, "ci95": 0.0,
            "n_seeds": len(arr), "values": [float(v) for v in vals]}


def run_heuristic_baseline(cfg, eval_split="test", params=None, n_games=0, log=print):
    """Ewaluacja heurystyki na `eval_split`, TYLKO na starcie tury, per karta.
    Zwraca res w tym samym ksztalcie co probe.run_probe:
    {n_test, all, seen, unseen_mcts, by_observed_type}."""
    params = params or HeuristicParams()
    ts = load_split(cfg.data_dir, eval_split, "timesteps")
    card = load_split(cfg.data_dir, eval_split, "card_samples")
    if n_games and ts["game_id"].nunique() > n_games:
        rng = np.random.default_rng(cfg.seed)
        keep = set(rng.choice(ts["game_id"].unique(), n_games, replace=False))
        ts = ts[ts["game_id"].isin(keep)]
        card = card[card["game_id"].isin(keep)]

    card = filter_to_turn_starts(card, ts)                 # probki na starcie tury DOWOLNEGO gracza
    key = ["game_id", "observed_color", "action_index"]
    m = card.merge(ts[key + PLAYED_TOTAL_COLS], on=key, how="inner")   # prior
    m = _attach_evidence(m, _obs_turn_tables(ts, params))             # akumulacja po turach obserwowanego

    pred = predict_proba(m, params).argmax(axis=1)
    y = m["label"].map(LABEL_TO_IDX).to_numpy()
    kinds = m["test_kind"].to_numpy() if "test_kind" in m.columns else np.array(["-"] * len(m))
    otypes = (m["observed_type"].to_numpy() if "observed_type" in m.columns
              else np.array(["-"] * len(m)))

    res = {"n_test": int(len(y)), "all": _f1_report(y, pred)}
    for kind in ("seen", "unseen_mcts"):
        msk = kinds == kind
        if msk.sum() > 0:
            res[kind] = _f1_report(y[msk], pred[msk])
            res[kind]["n"] = int(msk.sum())
    res["by_observed_type"] = _grouped_report(y, pred, otypes)

    log(f"  [heuristic] n_test={res['n_test']}  macro-F1 all={res['all']['macro_f1']:.3f} "
        f"seen={res.get('seen', {}).get('macro_f1', float('nan')):.3f} "
        f"unseen={res.get('unseen_mcts', {}).get('macro_f1', float('nan')):.3f}")
    if res["by_observed_type"]:
        log("    per observed_type: " + "  ".join(
            f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in res["by_observed_type"].items()))
    dist = {LABELS[i]: int((pred == i).sum()) for i in range(5)}
    log(f"    rozklad predykcji (argmax): {dist}")
    log("    per_class_f1 (all): " + "  ".join(
        f"{k}={v:.3f}" for k, v in res["all"]["per_class_f1"].items()))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--eval-split", default="test", choices=["test", "val", "train"])
    p.add_argument("--games", type=int, default=0,
                   help="0 = pelny split; >0 = podprobkowanie gier (smoke test)")
    args = p.parse_args()

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)
    params = HeuristicParams()

    print(f"Heurystyczny baseline | split={args.eval_split} | "
          f"gry={'wszystkie' if not args.games else args.games}")
    res = run_heuristic_baseline(cfg, eval_split=args.eval_split, params=params,
                                 n_games=args.games)

    summary = {}
    for k in ("all", "seen", "unseen_mcts"):
        s = res.get(k)
        summary[k] = _agg([s["macro_f1"]]) if s and s["macro_f1"] == s["macro_f1"] else None
    summary_ot = {t: _agg([v["macro_f1"]]) for t, v in res.get("by_observed_type", {}).items()}

    out = {
        "config": cfg.to_dict(),
        "params": dataclasses.asdict(params),
        "eval_split": args.eval_split,
        "final": {"heuristic": {
            "per_seed": [res],
            "summary": summary,
            "summary_observed_type": summary_ot,
        }},
    }
    out_path = os.path.join(cfg.out_dir, "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nZapisano {out_path}")


if __name__ == "__main__":
    main()
