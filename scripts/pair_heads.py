# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Heads on two embeddings at once: AlphaEarth + TESSERA (192 inputs), AlphaEarth + OlmoEarth (192), TESSERA +
OlmoEarth (256), taught exactly as biomass_head.py teaches "all three" (same rows, 8 square folds, epochs, loss),
valid where both inputs are valid, scored on the same pixels (valid in all three) as biomass_head.py scores.

Why: the map reads embeddings on the fly, and each costs bandwidth (scripts/embedding_bandwidth.py); a pair may
give most of what all three give at less cost.

Writes data/heads/<a>+<b>_square.npz (same format as biomass_head.py: held-out predictions and every fold's
weights) and the "state" part of results/pair_heads.json. Peak about 5 GB.

    uv run scripts/pair_heads.py [--pairs aef+tes aef+olmo tes+olmo] [--epochs 8]
"""
import argparse
import json
import sys
import threading
import time
import os
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

RES = HERE / "results" / "pair_heads.json"
PAIRS = ["aef+tes", "aef+olmo", "tes+olmo"]


def build_pair(R, pair):
    keys = pair.split("+")
    G, Y, P, _ = R["aef"].shape
    X = np.empty((G, Y, P, sum(bh.DIMS[k] for k in keys)), np.float16)
    ok, lo = np.ones((G, Y, P), bool), 0
    for k in keys:
        _, okk = bh.features(R, k, X, lo)
        ok &= okk
        lo += bh.DIMS[k]
    return X, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    p, peak = psutil.Process(), [0.0]

    def guard():
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > args.max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
    R = dict(np.load(HERE / "data" / "rows.npz"))
    H = bh.targets(R)
    common = np.ones(R["cls"].shape, bool)
    for k in ("aef", "tes", "olmo"):
        common &= (np.isfinite(R["tes_s"]) & (R["tes_s"] > 0)) if k == "tes" else (R[k][..., 0] != -128)
    res = json.load(open(RES)) if RES.exists() else {}
    for pair in args.pairs:
        t0 = time.time()
        X, ok = build_pair(R, pair)
        G, Y, P, D = X.shape
        pa, ph, pc = np.full((G, Y), np.nan), np.zeros((G, Y, P), np.float32), np.zeros((G, Y, P), np.int8)
        weights = {}
        for name, test in bh.folds(R, "square"):
            train = np.setdiff1d(np.arange(G), test)
            net = bh.fit(X, ok, R, H, train, args.epochs, seed=0)
            a, h, c = bh.predict(net, X, ok, test)
            pa[test], ph[test], pc[test] = a, h, c
            for k, v in net.state_dict().items():
                weights[f"{name}.{k}"] = v.numpy()
            print(f"  {pair} fold {name}: {test.size} cells, {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)
        s = bh.score(R, H, common, pa, ph, pc)
        res.setdefault("state", {})[pair] = s
        a = s["all"]
        print(f"{pair:9s} agb R2 log {a.get('agb_r2_log')} mae {a.get('agb_mae')} | height R2 {a.get('height_r2')} | "
              f"class acc {a.get('cls_acc')}  [{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
        np.savez(HERE / "data" / "heads" / f"{pair}_square.npz", pa=pa.astype(np.float32), ph=ph, pc=pc, **weights)
        res["peak_gb_train"] = round(peak[0], 1)
        json.dump(res, open(RES, "w"), indent=1)
        del X, ok


if __name__ == "__main__":
    main()
