"""Finalny (nocny) trening najlepszych konfiguracji + ewaluacja na TEST.

Bierze najlepsza konfiguracje per rodzina z results/search_results.json (selekcja
na VAL), trenuje ja na PELNYCH danych przez wiecej epok i dla kilku ziaren, a
nastepnie raportuje protokolem linear-probe NA STARCIE TURY na splicie TEST
(seen / unseen_mcts) — z usrednieniem i przedzialem ufnosci po ziarnach.

Uruchom (GPU, ~8h):
    python -m src.train_final --epochs 25 --seeds 0,1,2

Smoke test (CPU):
    python -m src.train_final --epochs 1 --seeds 0 --subsample-games 80 --probe-games 60

Wynik: results/final_metrics.json + checkpointy results/final_<rodzina>_seed<n>.pt
"""
import argparse
import dataclasses
import json
import math
import os
import random
import time

import numpy as np
import torch

from .config import Config
from .data import fit_feature_spec, load_split
from .search import (effective_games, estimate_final_seconds, evaluate_family,
                     fmt_dur, is_supervised, make_cfg, train_family)
from .supervised_seq import parse_family


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# domyslne konfiguracje (gdy brak search_results.json) — sensowne punkty startowe
DEFAULTS = {
    "vae": dict(vae_latent_dim=128, vae_h_dim=256, vae_num_layers=1),
    "vae_cat": dict(vae_n_cat=16, vae_n_class=16, vae_h_dim=256),
    "rssm_gauss": dict(rssm_h_dim=128, rssm_z_dim=128),
    "rssm_cat": dict(rssm_n_cat=16, rssm_n_class=16, rssm_h_dim=128),
}


def default_overrides(family):
    """Domyslne nadpisania dla rodziny SSL lub supervised (po backbonie)."""
    backbone, _ = parse_family(family)
    return dict(DEFAULTS[backbone])


def load_timings(search_json, families):
    """{family: sec_per_game_epoch} z najlepszych konfiguracji searcha (do estymacji)."""
    spge = {}
    if search_json and os.path.exists(search_json):
        with open(search_json) as f:
            data = json.load(f)
        by_name = {r["name"]: r for r in data["results"]}
        for fam, name in data.get("best_per_family", {}).items():
            if fam in families and name in by_name:
                spge[fam] = by_name[name].get("sec_per_game_epoch", float("nan"))
    return spge


def load_best_overrides(search_json, families):
    """Zwraca {family: overrides_dict} z search_results.json (best_per_family)."""
    best = {}
    if search_json and os.path.exists(search_json):
        with open(search_json) as f:
            data = json.load(f)
        name_to_res = {r["name"]: r for r in data["results"]}
        for fam, name in data.get("best_per_family", {}).items():
            if fam in families:
                best[fam] = name_to_res[name]["overrides"]
        print(f"Wczytano najlepsze konfiguracje z {search_json}: "
              f"{ {f: data['best_per_family'][f] for f in best} }")
    for fam in families:
        best.setdefault(fam, default_overrides(fam))
    return best


def _agg(vals):
    """Mean + 95% CI (normalny) z listy wartosci."""
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci, "n_seeds": len(arr), "values": list(vals)}


def summarize(seed_results, kind):
    """Mean + 95% CI (po ziarnach) dla macro-F1 danego `kind`."""
    vals = [r[kind]["macro_f1"] for r in seed_results
            if kind in r and r[kind]["macro_f1"] == r[kind]["macro_f1"]]
    return _agg(vals) if vals else None


def summarize_otype(seed_results):
    """Mean ± CI95 po ziarnach dla kazdego observed_type (ValueFunction/AlphaBeta/MCTS)."""
    types = set()
    for r in seed_results:
        types |= set(r.get("by_observed_type", {}).keys())
    out = {}
    for t in sorted(types):
        vals = [r["by_observed_type"][t]["macro_f1"] for r in seed_results
                if t in r.get("by_observed_type", {})]
        if vals:
            out[t] = _agg(vals)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--search-json", default="results/search_results.json")
    p.add_argument("--families", default="vae,vae_cat,rssm_gauss,rssm_cat",
                   help="rodziny do finalu (po przecinku); 'all' = wszystkie "
                        "(w tym supervised *_supA/B/C)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--subsample-games", type=int, default=0,
                   help="0 = pelne dane (domyslnie dla finalu)")
    p.add_argument("--probe-games", type=int, default=0,
                   help="0 = wszystkie gry do glowicy probe / TEST")
    p.add_argument("--dry-run", action="store_true",
                   help="tylko oszacuj czas finalu (z search_results.json), bez treningu")
    args = p.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    if "all" in families:
        from .search import search_space
        families = sorted({f for _, f, _ in search_space()})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    best = load_best_overrides(args.search_json, families)

    base = Config(data_dir=args.data_dir, out_dir=args.out_dir,
                  subsample_games=args.subsample_games, epochs=args.epochs,
                  batch_size=args.batch_size)
    os.makedirs(base.out_dir, exist_ok=True)
    device = base.device
    print(f"Urzadzenie: {device}  | rodziny: {families}  | ziarna: {seeds}")

    print("Wczytywanie TRAIN timesteps + FeatureSpec...")
    train_ts = load_split(base.data_dir, "train", "timesteps")
    spec = fit_feature_spec(train_ts)
    n_total = int(train_ts["game_id"].nunique())
    del train_ts
    print(f"  cechy: {spec.n_features}  | gier TRAIN: {n_total}")

    # --- estymacja czasu finalu z throughputu zmierzonego w searchu ---
    final_games = effective_games(args.subsample_games, n_total)
    spge = load_timings(args.search_json, families)
    print("\n" + "-" * 60)
    print(f"SZACOWANY CZAS FINALU (gry={final_games} epoki={args.epochs} "
          f"ziarna={len(seeds)}):")
    total_est = 0.0
    for fam in families:
        est = estimate_final_seconds(spge.get(fam, float("nan")),
                                     final_games, args.epochs, len(seeds))
        if est == est:
            total_est += est
        src = "z searcha" if fam in spge else "brak danych searcha"
        print(f"  {fam:16s} est. {fmt_dur(est):>10s}  ({src})")
    print(f"  {'RAZEM':16s} est. {fmt_dur(total_est):>10s}")
    print("-" * 60)
    if args.dry_run:
        print("DRY-RUN: koniec (bez treningu). Dostosuj --epochs/--seeds/"
              "--subsample-games i uruchom ponownie.")
        return

    final = {}
    for family in families:
        print(f"\n########## RODZINA: {family} ##########")
        per_seed = []
        train_secs = []
        for seed in seeds:
            print(f"\n--- {family} | seed {seed} ---")
            set_seed(seed)
            cfg = make_cfg(dataclasses.replace(base, seed=seed), family, best[family])
            t = time.time()
            model = train_family(family, cfg, spec, device)
            tr = time.time() - t
            train_secs.append(tr)
            print(f"  trening: {fmt_dur(tr)}")
            res = evaluate_family(family, model, spec, cfg, device,
                                  eval_split="test", n_train_games=args.probe_games,
                                  name=f"{family}_s{seed}")
            per_seed.append(res)
            ckpt = os.path.join(base.out_dir, f"final_{family}_seed{seed}.pt")
            if is_supervised(family):
                torch.save({"enc": model.adapter.state_dict(),
                            "head": model.head.state_dict(),
                            "base": model.base, "mode": model.mode,
                            "emb_dim": model.emb_dim}, ckpt)
            else:
                torch.save(model.state_dict(), ckpt)
            print(f"  zapisano {ckpt}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        final[family] = {
            "overrides": best[family],
            "per_seed": per_seed,
            "train_seconds": train_secs,
            "train_seconds_total": float(sum(train_secs)),
            "summary": {k: summarize(per_seed, k) for k in ("all", "seen", "unseen_mcts")},
            "summary_observed_type": summarize_otype(per_seed),
        }
        print(f"  [{family}] laczny czas treningu: {fmt_dur(sum(train_secs))}")

    with open(os.path.join(base.out_dir, "final_metrics.json"), "w") as f:
        json.dump({"config": base.to_dict(), "seeds": seeds, "final": final}, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL (TEST macro-F1, mean ± CI95 po ziarnach)")
    print("=" * 70)
    print(f"{'rodzina':14s} {'all':>16s} {'seen':>16s} {'unseen':>16s} {'trening':>10s}")
    total_train = 0.0
    for fam, d in final.items():
        def fmt(k):
            s = d["summary"][k]
            return f"{s['mean']:.3f}±{s['ci95']:.3f}" if s else "   n/a"
        total_train += d["train_seconds_total"]
        print(f"{fam:14s} {fmt('all'):>16s} {fmt('seen'):>16s} {fmt('unseen_mcts'):>16s} "
              f"{fmt_dur(d['train_seconds_total']):>10s}")
    # rozbicie per obserwowany algorytm (MCTS = nieznany w treningu)
    otypes_all = sorted({t for d in final.values() for t in d["summary_observed_type"]})
    if otypes_all:
        print("\n" + "=" * 70)
        print("PER observed_type (TEST macro-F1, mean ± CI95)  [MCTS = nieznany w treningu]")
        print("=" * 70)
        print(f"{'rodzina':16s}" + "".join(f"{t:>16s}" for t in otypes_all))
        for fam, d in final.items():
            cells = ""
            for t in otypes_all:
                s = d["summary_observed_type"].get(t)
                val = f"{s['mean']:.3f}±{s['ci95']:.3f}" if s else "-"
                cells += f"{val:>16s}"
            print(f"{fam:16s}{cells}")

    print(f"\nLaczny czas treningu (rzeczywisty): {fmt_dur(total_train)}")
    print(f"Zapisano do {base.out_dir}/final_metrics.json + final_*_seed*.pt")


if __name__ == "__main__":
    main()
