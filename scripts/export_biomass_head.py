# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""The map's biomass head: AlphaEarth alone, taught on all 92 squares (biomass_head.py's fit, same settings), written
in fire-knn's fuel_volume.json layout so the browser runs it as fire-knn runs its volume head.

Outputs kept: log1p(biomass, Mg/ha per 10 m pixel) and log1p(canopy height, m); the class logits are dropped.
The held-out scores of this head's recipe are FINDINGS section 7 (AEF, by square: R2 of log .70, median error 10 Mg/ha).

Writes web/public/models/biomass_aef.json. RAM ~2 GB (biomass_head.log), stops itself over --max-gb.

    uv run scripts/export_biomass_head.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

OUT = HERE / "web" / "public" / "models" / "biomass_aef.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    torch.set_num_threads(8)
    R = dict(np.load(HERE / "data" / "rows.npz"))
    H = bh.targets(R)
    X, ok = bh.build(R, "aef")
    G = X.shape[0]
    net = bh.fit(X, ok, R, H, np.arange(G), args.epochs, seed=0)
    gb = psutil.Process().memory_info().rss / 1e9
    print(f"fit on {G} cells, {gb:.1f} GB", flush=True)
    if gb > args.max_gb:
        raise SystemExit(f"STOP: RSS {gb:.1f} GB")

    sd = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}
    w2, b2 = sd["l2.weight"][:2], sd["l2.bias"][:2]

    # probe: real embeddings from the rows (valid ones), pixel-major int8 [n, 64], and the head's raw outputs
    rng = np.random.default_rng(0)
    q = R["aef"].reshape(-1, 64)
    good = np.flatnonzero(q[:, 0] != -128)
    pq = q[rng.choice(good, 64, replace=False)]
    with torch.no_grad():
        po = net(torch.as_tensor(bh.deq(pq)))[:, :2].numpy()

    head = {
        "name": "biomass_aef",
        "n_in": 64,
        "hidden": bh.HID,
        "n_out": 2,
        "out": ["log1p_agb_mg_ha", "log1p_height_m"],
        "years": [int(y) for y in R["years"]],
        "mean": sd["mu"].tolist(),
        "scale": sd["sd"].tolist(),
        "w1": sd["l1.weight"].reshape(-1).tolist(),
        "b1": sd["l1.bias"].tolist(),
        "w2": w2.reshape(-1).tolist(),
        "b2": b2.tolist(),
        "probe_q": pq.reshape(-1).astype(int).tolist(),
        "probe_out": po.reshape(-1).astype(float).tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(head, open(OUT, "w"))
    a = np.expm1(np.clip(po[:, 0], None, 8)).clip(0)
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size / 1e3:.0f} KB); probe biomass Mg/ha "
          f"p10 {np.percentile(a, 10):.1f} median {np.median(a):.1f} p90 {np.percentile(a, 90):.1f}")


if __name__ == "__main__":
    main()
