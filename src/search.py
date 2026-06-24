"""Przeszukiwanie konfiguracji VAE (LSTM) + RSSM (Gaussian / Categorical).

Kazda konfiguracja jest trenowana przy ZREDUKOWANYM budzecie (podprobkowane gry,
malo epok) i oceniana protokolem linear-probe na starcie tury, na splicie VAL.
Selekcja modeli odbywa sie WYLACZNIE na VAL — TEST pozostaje nietkniety do
finalnej ewaluacji (src/train_final.py).

Uruchom (smoke test, CPU):
    python -m src.search --subsample-games 80 --epochs 1 --probe-games 60 --quick

Pelny search (GPU, kilka godzin):
    python -m src.search --subsample-games 1500 --epochs 8 --probe-games 1200

Wynik: results/search_results.json — wszystkie konfiguracje + val macro-F1,
posortowane; oraz najlepsza konfiguracja per rodzina.
"""
import argparse
import dataclasses
import json
import os
import random
import time

import numpy as np
import torch

from .config import Config
from .data import fit_feature_spec, load_split
from .probe import run_probe
from .ssl_rssm import train_rssm
from .ssl_vae import train_vae
from .supervised_seq import parse_family, run_supervised_eval, train_supervised_seq


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def fmt_dur(sec):
    """Sekundy -> czytelny 'Hh MMm SSs'."""
    if sec != sec:                       # NaN
        return "n/a"
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def effective_games(subsample, n_total):
    """Ile gier faktycznie uzytych: 0 = wszystkie."""
    return n_total if not subsample else min(subsample, n_total)


def estimate_final_seconds(sec_per_game_epoch, final_games, final_epochs, final_seeds):
    """Liniowa ekstrapolacja czasu treningu: ~ gry x epoki x ziarna.
    (Pomija staly narzut; throughput per architektura z searcha.)"""
    if sec_per_game_epoch != sec_per_game_epoch:
        return float("nan")
    return sec_per_game_epoch * final_games * final_epochs * final_seeds


# --- przestrzen przeszukiwania (kuratorowana, nie pelny grid) ---------------
# Kazdy wpis: (nazwa, rodzina, slownik nadpisan Config). rodzina ∈
# {"vae", "rssm_gauss", "rssm_cat"} steruje wyborem trenera/wariantu.
def search_space():
    space = []

    # VAE z enkoderem LSTM — sweep pojemnosci latentu / beta / lr
    for latent in (64, 128, 256):
        space.append((f"vae_z{latent}", "vae",
                      dict(vae_latent_dim=latent, vae_h_dim=256, vae_num_layers=1)))
    space.append(("vae_z128_2L", "vae",
                  dict(vae_latent_dim=128, vae_h_dim=256, vae_num_layers=2, vae_dropout=0.1)))
    space.append(("vae_z128_beta1", "vae",
                  dict(vae_latent_dim=128, vae_h_dim=256, vae_beta_max=1.0)))
    space.append(("vae_z128_lr1e3", "vae",
                  dict(vae_latent_dim=128, vae_h_dim=256, lr=1e-3)))

    # VAE z latentem kategorycznym (stały, jednostajny prior) — domyka tabelę 2x2
    space.append(("vae_cat_16x16", "vae_cat",
                  dict(vae_n_cat=16, vae_n_class=16, vae_h_dim=256)))
    space.append(("vae_cat_32x32", "vae_cat",
                  dict(vae_n_cat=32, vae_n_class=32, vae_h_dim=256)))
    space.append(("vae_cat_16x16_beta1", "vae_cat",
                  dict(vae_n_cat=16, vae_n_class=16, vae_h_dim=256, vae_beta_max=1.0)))

    # RSSM Gaussian — sweep h_dim/z_dim/lr
    for hz in (64, 128, 256):
        space.append((f"rssm_gauss_h{hz}", "rssm_gauss",
                      dict(rssm_h_dim=hz, rssm_z_dim=hz)))
    space.append(("rssm_gauss_h128_lr1e3", "rssm_gauss",
                  dict(rssm_h_dim=128, rssm_z_dim=128, lr=1e-3)))

    # RSSM Categorical (DreamerV2) — sweep liczby zmiennych/klas, h_dim, kl_balance
    space.append(("rssm_cat_16x16", "rssm_cat",
                  dict(rssm_n_cat=16, rssm_n_class=16, rssm_h_dim=128)))
    space.append(("rssm_cat_32x32", "rssm_cat",
                  dict(rssm_n_cat=32, rssm_n_class=32, rssm_h_dim=200)))
    space.append(("rssm_cat_16x16_klb0.5", "rssm_cat",
                  dict(rssm_n_cat=16, rssm_n_class=16, rssm_h_dim=128, rssm_kl_balance=0.5)))
    space.append(("rssm_cat_24x24", "rssm_cat",
                  dict(rssm_n_cat=24, rssm_n_class=24, rssm_h_dim=200)))

    # --- warianty SUPERVISED (A/B/C) dla kazdego backbone'u ---
    base_ov = {
        "vae":        dict(vae_latent_dim=128, vae_h_dim=256),
        "vae_cat":    dict(vae_n_cat=16, vae_n_class=16, vae_h_dim=256),
        "rssm_gauss": dict(rssm_h_dim=128, rssm_z_dim=128),
        "rssm_cat":   dict(rssm_n_cat=16, rssm_n_class=16, rssm_h_dim=128),
    }
    for base, ov in base_ov.items():
        for mode in ("A", "B", "C"):
            space.append((f"{base}_sup{mode}", f"{base}_sup{mode}", dict(ov)))
    return space


def make_cfg(base_cfg, family, overrides):
    o = dict(overrides)
    backbone, mode = parse_family(family)   # 'vae_supC' -> ('vae','C'); 'rssm_cat' -> ('rssm_cat',None)
    if mode is not None:
        o["sup_mode"] = mode
    if backbone == "rssm_gauss":
        o.setdefault("rssm_variant", "gauss"); o["model"] = "rssm_gauss"
    elif backbone == "rssm_cat":
        o.setdefault("rssm_variant", "cat"); o["model"] = "rssm_cat"
    elif backbone == "vae_cat":
        o.setdefault("vae_variant", "cat"); o["model"] = "vae"
    else:  # vae (gauss)
        o.setdefault("vae_variant", "gauss"); o["model"] = "vae"
    return dataclasses.replace(base_cfg, **o)


def is_supervised(family):
    return parse_family(family)[1] is not None


def train_family(family, cfg, spec, device, log=print):
    """Zwraca adapter enkodera (SSL) lub SupModel (supervised)."""
    backbone, mode = parse_family(family)
    if mode is not None:
        return train_supervised_seq(family, cfg, spec, device, log=log,
                                    n_games=cfg.subsample_games)
    if backbone in ("vae", "vae_cat"):
        enc, _ = train_vae(spec, cfg, device, log=log)
    else:
        enc, _ = train_rssm(spec, cfg, device, log=log)
    return enc.to(device)


def evaluate_family(family, model, spec, cfg, device, eval_split, n_train_games, name):
    """Dispatch ewaluacji: supervised -> glowica; SSL -> linear probe."""
    if is_supervised(family):
        return run_supervised_eval(model, spec, cfg, device,
                                   eval_split=eval_split, n_games=0)
    return run_probe(model, spec, cfg, device, causal=False, name=name,
                     n_train_games=n_train_games, n_test_games=0, eval_split=eval_split)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--subsample-games", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--probe-games", type=int, default=1200,
                   help="ile gier do trenowania glowicy probe (TRAIN)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="vae,vae_cat,rssm_gauss,rssm_cat",
                   help="rodziny do przeszukania (po przecinku); 'all' = wszystkie "
                        "(w tym supervised *_supA/B/C)")
    p.add_argument("--quick", action="store_true",
                   help="tylko 1 konfiguracja per rodzina (dry-run)")
    # parametry FINALNEGO treningu — uzyte tylko do estymacji czasu (nie do treningu)
    p.add_argument("--final-epochs", type=int, default=25,
                   help="zakladane epoki finalu (do szacowania czasu)")
    p.add_argument("--final-seeds", type=int, default=3,
                   help="zakladana liczba ziaren finalu (do szacowania czasu)")
    p.add_argument("--final-subsample-games", type=int, default=0,
                   help="zakladane gry finalu (0 = wszystkie; do szacowania czasu)")
    args = p.parse_args()

    families = {f.strip() for f in args.families.split(",") if f.strip()}
    if "all" in families:
        families = {f for _, f, _ in search_space()}
    base = Config(data_dir=args.data_dir, out_dir=args.out_dir,
                  subsample_games=args.subsample_games, epochs=args.epochs,
                  batch_size=args.batch_size, seed=args.seed)
    set_seed(base.seed)
    os.makedirs(base.out_dir, exist_ok=True)
    device = base.device

    print(f"Urzadzenie: {device}")
    print("Wczytywanie TRAIN timesteps + FeatureSpec...")
    train_ts = load_split(base.data_dir, "train", "timesteps")
    spec = fit_feature_spec(train_ts)
    n_total = int(train_ts["game_id"].nunique())
    del train_ts
    print(f"  cechy: {spec.n_features}  | gier TRAIN: {n_total}")
    final_games = effective_games(args.final_subsample_games, n_total)
    print(f"  estymacja finalu wg: gry={final_games} epoki={args.final_epochs} "
          f"ziarna={args.final_seeds}")

    space = [e for e in search_space() if e[1] in families]
    if args.quick:
        seen = set(); pruned = []
        for e in space:
            if e[1] not in seen:
                pruned.append(e); seen.add(e[1])
        space = pruned
    print(f"Konfiguracji do przeszukania: {len(space)}")

    results = []
    for i, (name, family, overrides) in enumerate(space, 1):
        print(f"\n=== [{i}/{len(space)}] {name} ({family}) ===")
        set_seed(base.seed)
        cfg = make_cfg(base, family, overrides)
        train_sec = eval_sec = float("nan")
        try:
            t = time.time()
            model = train_family(family, cfg, spec, device)
            train_sec = time.time() - t
            t = time.time()
            res = evaluate_family(family, model, spec, cfg, device,
                                  eval_split="val", n_train_games=args.probe_games,
                                  name=name)
            eval_sec = time.time() - t
            val_f1 = res["all"]["macro_f1"]
        except Exception as ex:           # noqa: BLE001 — log i kontynuuj search
            print(f"  BLAD: {ex}")
            val_f1, res = float("nan"), {"error": str(ex)}
        # throughput treningu + ekstrapolacja na final
        eff = effective_games(cfg.subsample_games, n_total)
        spge = train_sec / max(eff * cfg.epochs, 1)      # sekundy / (gra * epoka)
        est_final = estimate_final_seconds(spge, final_games, args.final_epochs,
                                           args.final_seeds)
        print(f"  -> VAL macro-F1 = {val_f1:.4f}  | trening {fmt_dur(train_sec)} "
              f"eval {fmt_dur(eval_sec)}  | est. final {fmt_dur(est_final)}")
        results.append({"name": name, "family": family, "overrides": overrides,
                        "val_macro_f1": val_f1,
                        "train_seconds": train_sec, "eval_seconds": eval_sec,
                        "seconds": (train_sec + eval_sec),
                        "train_games": eff, "train_epochs": cfg.epochs,
                        "sec_per_game_epoch": spge, "est_final_seconds": est_final,
                        "probe": res})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ranking + najlepsza per rodzina
    ranked = sorted(results, key=lambda r: (-(r["val_macro_f1"]
                    if r["val_macro_f1"] == r["val_macro_f1"] else -1)))
    best_per_family = {}
    for r in ranked:
        f = r["family"]
        if f not in best_per_family and r["val_macro_f1"] == r["val_macro_f1"]:
            best_per_family[f] = r["name"]

    out = {"config": base.to_dict(), "results": results,
           "ranking": [(r["name"], r["family"], r["val_macro_f1"]) for r in ranked],
           "best_per_family": best_per_family,
           "n_train_games_total": n_total,
           "final_estimate_params": {"epochs": args.final_epochs,
                                     "seeds": args.final_seeds,
                                     "games": final_games}}
    with open(os.path.join(base.out_dir, "search_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    by_name = {r["name"]: r for r in results}
    print("\n" + "=" * 72)
    print("RANKING (VAL macro-F1)  |  czas treningu (search)  |  est. final / arch.")
    print("=" * 72)
    for r in ranked:
        print(f"  {r['val_macro_f1']:.4f}  {r['name']:24s} "
              f"trening {fmt_dur(r['train_seconds']):>9s}  "
              f"est.final {fmt_dur(r['est_final_seconds']):>9s}")

    # projekcja calkowitego czasu finalu dla wybranego zestawu (best per rodzina)
    total_est = sum(by_name[nm]["est_final_seconds"] for nm in best_per_family.values()
                    if by_name[nm]["est_final_seconds"] == by_name[nm]["est_final_seconds"])
    print("\n" + "-" * 72)
    print(f"PROJEKCJA FINALU (best/rodzina; gry={final_games} epoki={args.final_epochs} "
          f"ziarna={args.final_seeds}):")
    for fam, nm in best_per_family.items():
        print(f"  {fam:14s} -> {nm:24s}  est. {fmt_dur(by_name[nm]['est_final_seconds'])}")
    print(f"  {'RAZEM':14s}    {'':24s}  est. {fmt_dur(total_est)}")
    print("  (skaluje sie ~liniowo z gry x epoki x ziarna — dostosuj "
          "--final-epochs/--final-seeds/--final-subsample-games)")
    print(f"\nZapisano do {base.out_dir}/search_results.json")


if __name__ == "__main__":
    main()
