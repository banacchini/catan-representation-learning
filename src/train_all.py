"""Orkiestracja: pretrening 3 metod SSL + baseline'y + linear probe.

Uruchom (smoke test):
    .venv-ml/Scripts/python -m src.train_all --subsample-games 300 --epochs 2

Pelny przebieg (CPU, kilkadziesiat minut):
    .venv-ml/Scripts/python -m src.train_all

Wynik: results/metrics.json (F1 seen/unseen dla kazdej metody + baseline'ow),
results/losses.json (krzywe strat), results/encoder_<metoda>.pt (checkpointy).
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from .config import Config
from .data import build_sequences, collate, fit_feature_spec, load_split, SeqDataset
from .models import SeqTransformerEncoder
from .probe import run_probe, run_raw_baseline
from .ssl_barlow import train_barlow
from .ssl_infonce import train_infonce
from .ssl_mae import train_mae
from .supervised import train_supervised
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_encoder(spec, cfg):
    return SeqTransformerEncoder(
        spec.n_features, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout).to(cfg.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/splits")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--subsample-games", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--probe-games", type=int, default=1200)
    p.add_argument("--methods", default="raw,random,infonce,barlow,mae,supervised",
                   help="lista metod oddzielona przecinkami")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir,
                 subsample_games=args.subsample_games, epochs=args.epochs,
                 batch_size=args.batch_size, seed=args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("Wczytywanie TRAIN timesteps + dopasowanie FeatureSpec...")
    train_ts = load_split(cfg.data_dir, "train", "timesteps")
    spec = fit_feature_spec(train_ts)
    print(f"  cechy: {spec.n_features} (binarnych: {int(spec.is_binary.sum())}, "
          f"ciaglych: {int((~spec.is_binary).sum())})")

    # sekwencje do pretreningu (podprobkowane)
    pre_seqs = build_sequences(train_ts, spec,
                               subsample_games=cfg.subsample_games, seed=cfg.seed)
    print(f"  sekwencji do pretreningu: {len(pre_seqs)}")
    del train_ts

    def pre_loader():
        ds = SeqDataset(pre_seqs, cfg.train_seq_len, mode="crop", seed=cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0, drop_last=True)

    metrics, losses = {}, {}

    if "raw" in methods:
        print("\n=== BASELINE: surowe cechy (bez enkodera) ===")
        metrics["raw"] = run_raw_baseline(spec, cfg, n_train_games=args.probe_games,
                                          n_test_games=args.probe_games)

    def pretrain_and_probe(name, trainer, causal):
        print(f"\n=== {name.upper()} ===")
        set_seed(cfg.seed)
        enc = build_encoder(spec, cfg)
        if trainer is not None:
            losses[name] = trainer(enc, pre_loader(), spec, cfg, cfg.device)
            torch.save(enc.state_dict(), os.path.join(cfg.out_dir, f"encoder_{name}.pt"))
        metrics[name] = run_probe(enc, spec, cfg, cfg.device, causal=causal, name=name,
                                  n_train_games=args.probe_games,
                                  n_test_games=args.probe_games)

    if "random" in methods:
        pretrain_and_probe("random", None, causal=False)
    if "infonce" in methods:
        pretrain_and_probe("infonce", train_infonce, causal=True)
    if "barlow" in methods:
        pretrain_and_probe("barlow", train_barlow, causal=False)
    if "mae" in methods:
        pretrain_and_probe("mae", train_mae, causal=False)

    if "supervised" in methods:
        print("\n=== SUPERVISED (upper bound) ===")
        set_seed(cfg.seed)
        enc = build_encoder(spec, cfg)
        losses["supervised"] = train_supervised(enc, spec, cfg, cfg.device,
                                                 n_games=args.probe_games)
        torch.save(enc.state_dict(), os.path.join(cfg.out_dir, "encoder_supervised.pt"))
        metrics["supervised"] = run_probe(enc, spec, cfg, cfg.device, causal=False,
                                          name="supervised",
                                          n_train_games=args.probe_games,
                                          n_test_games=args.probe_games)

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)
    with open(os.path.join(cfg.out_dir, "losses.json"), "w") as f:
        json.dump(losses, f, indent=2)

    print("\n" + "=" * 64)
    print("PODSUMOWANIE (macro-F1)")
    print("=" * 64)
    print(f"{'metoda':14s} {'all':>7s} {'seen':>7s} {'unseen':>7s}")
    for name, r in metrics.items():
        a = r["all"]["macro_f1"]
        s = r.get("seen", {}).get("macro_f1", float("nan"))
        u = r.get("unseen_mcts", {}).get("macro_f1", float("nan"))
        print(f"{name:14s} {a:7.3f} {s:7.3f} {u:7.3f}")
    print(f"\nZapisano do {cfg.out_dir}/metrics.json, losses.json, encoder_*.pt")


if __name__ == "__main__":
    main()
