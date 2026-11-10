# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""How few of AlphaEarth's 64 bands the biomass head needs.

AEF's GeoTIFFs store each band in its own blocks, so a reader can fetch only some bands. That only saves anything if a
head on k bands reads about as well as the head on all 64. This fits biomass_head.py's head (same loss, same settings)
on k bands and scores it on squares it never saw (the 8 square folds), beside the 64-band head.

Bands are picked inside each fold from its training squares only, three ways:
  greedy  forward selection by a linear proxy: how much of log biomass (cell mean), log canopy height and the class
          a ridge fit on the chosen bands explains, weighted 1, 1, 0.5 as in the head's loss
  prune   the bands with the largest first-layer weights in a 64-band head fit on the same training squares
  random  k bands at random (seeds 0 to 2), the reference for whether picking matters
Then one band set per k from all 92 squares (greedy), the one a map would read.

Scores are on the same ground as FINDINGS 7 (cells and pixels valid in all three embeddings).
Writes results/band_subset.json. RAM: AEF rows 0.2 GB int8, features 0.4 GB float16, peak ~3 GB; stops over --max-gb.

    uv run scripts/band_subset.py [--ks 4 8 16 32] [--methods greedy prune random] [--epochs 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

RES = HERE / "results" / "band_subset.json"
KEYS = ["names", "kinds", "continents", "years", "classes", "aef", "tes_s", "agb", "height", "height_year", "cls", "sq"]


def load():
    z = np.load(HERE / "data" / "rows.npz")
    R = {k: z[k] for k in KEYS}
    # the "valid in all three" mask of biomass_head.py, without keeping OlmoEarth's rows
    common = (R["aef"][..., 0] != -128) & np.isfinite(R["tes_s"]) & (R["tes_s"] > 0)
    common &= z["olmo"][..., 0] != -128
    return R, common


def proxy_blocks(X, ok, R, H, train, seed=0):
    """Standardized features and targets for the ridge proxy, from up to 3000 training cells:
    (Z, T, weight) for cell-mean biomass, pixel height, pixel class."""
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(train, size=min(3000, train.size), replace=False))
    x = X[sub].astype(np.float32)  # (g, Y, P, 64)
    v = ok[sub]
    mu, sd = x[v].mean(0), x[v].std(0) + 1e-4
    x = (x - mu) / sd
    blocks = []
    # biomass: the cell's mean feature against log1p of CTrees' cell
    n = v.sum(-1)
    cmean = (x * v[..., None]).sum(2) / np.maximum(n, 1)[..., None]
    t = R["agb"][sub]
    m = np.isfinite(t) & (n >= x.shape[2] // 2)
    blocks.append((cmean[m], np.log1p(np.clip(t[m], 0, None))[:, None], 1.0))
    # height: per pixel, in the year of Meta's imagery
    hm = H[sub] & v
    ht = np.broadcast_to(R["height"][sub][:, None, :], hm.shape)[hm].astype(np.float32)
    blocks.append((x[hm], np.log1p(ht)[:, None], 1.0))
    # class: per pixel, one-hot
    c = R["cls"][sub]
    cm = (c >= 0) & v
    blocks.append((x[cm], np.eye(bh.NCLS, dtype=np.float32)[c[cm]], 0.5))
    out = []
    for Z, T, w in blocks:
        Z = Z.astype(np.float64)
        T = T.astype(np.float64)
        Z -= Z.mean(0)
        T -= T.mean(0)
        A = Z.T @ Z + 1e-3 * len(Z) * np.eye(Z.shape[1])
        out.append((A, Z.T @ T, float((T * T).sum()), w))
    return out


def greedy(blocks, k):
    chosen = []
    for _ in range(k):
        best, bs = None, -1.0
        for b in range(64):
            if b in chosen:
                continue
            S = chosen + [b]
            s = 0.0
            for A, C, tot, w in blocks:
                CS = C[S]
                s += w * float(np.sum(CS * np.linalg.solve(A[np.ix_(S, S)], CS))) / tot
            if s > bs:
                best, bs = b, s
        chosen.append(best)
    return chosen


def prune_order(net):
    """Bands by the size of their first-layer weights (inputs are standardized, so the weights compare)."""
    w = net.l1.weight.detach().numpy()
    return list(np.argsort(-np.linalg.norm(w, axis=0)))


def run(R, H, common, X, ok, bands_of, epochs, max_gb, label):
    """Held out by square: each fold's head on its own band set. bands_of(fold_name, train) -> band list."""
    G, Y, P, _ = X.shape
    pa, ph, pc = np.full((G, Y), np.nan), np.zeros((G, Y, P), np.float32), np.zeros((G, Y, P), np.int8)
    sets = {}
    t0 = time.time()
    for name, test in bh.folds(R, "square"):
        train = np.setdiff1d(np.arange(G), test)
        b = sorted(int(i) for i in bands_of(name, train))
        sets[name] = b
        Xb = np.ascontiguousarray(X[..., b])
        net = bh.fit(Xb, ok, R, H, train, epochs, seed=0)
        pa[test], ph[test], pc[test] = bh.predict(net, Xb, ok, test)
        del Xb
        gb = psutil.Process().memory_info().rss / 1e9
        if gb > max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")
    s = bh.score(R, H, common, pa, ph, pc)
    a = s["all"]
    print(f"{label:14s} agb R2 log {a.get('agb_r2_log')} mae {a.get('agb_mae')} | height R2 {a.get('height_r2')} "
          f"mae {a.get('height_mae')} | class acc {a.get('cls_acc')} f1 {a.get('cls_macro_f1')}  "
          f"[{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    return {"scores": s, "bands": sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--methods", nargs="*", default=["greedy", "prune", "random"])
    ap.add_argument("--random-ks", nargs="*", type=int, default=[4, 8, 16])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    R, common = load()
    H = bh.targets(R)
    X, ok = bh.features(R, "aef")
    G = X.shape[0]
    res = json.load(open(RES)) if RES.exists() else {}
    save = lambda: json.dump(res, open(RES, "w"), indent=1)  # noqa: E731

    # the 64-band head, and each fold's prune order from it
    order = {}

    def all64(name, train):
        return range(64)

    res["all64"] = run(R, H, common, X, ok, all64, args.epochs, args.max_gb, "all 64")
    save()
    if "prune" in args.methods:
        for name, test in bh.folds(R, "square"):
            train = np.setdiff1d(np.arange(G), test)
            order[name] = prune_order(bh.fit(X, ok, R, H, train, args.epochs, seed=0))

    gorder = {}
    if "greedy" in args.methods:
        kmax = max(args.ks)
        for name, test in bh.folds(R, "square"):
            train = np.setdiff1d(np.arange(G), test)
            gorder[name] = greedy(proxy_blocks(X, ok, R, H, train), kmax)
        print("greedy orders picked", flush=True)

    for k in args.ks:
        if "greedy" in args.methods:
            res[f"greedy_{k}"] = run(R, H, common, X, ok, lambda n, t: gorder[n][:k], args.epochs, args.max_gb, f"greedy {k}")
            save()
        if "prune" in args.methods:
            res[f"prune_{k}"] = run(R, H, common, X, ok, lambda n, t: order[n][:k], args.epochs, args.max_gb, f"prune {k}")
            save()
        if "random" in args.methods and k in args.random_ks:
            for seed in range(3):
                rb = list(np.random.default_rng(100 + seed).choice(64, k, replace=False))
                res[f"random_{k}_s{seed}"] = run(R, H, common, X, ok, lambda n, t: rb, args.epochs, args.max_gb,
                                                 f"random {k} s{seed}")
                save()

    # one band set from all 92 squares, the one a map would read, and how often the folds agreed with it
    full = greedy(proxy_blocks(X, ok, R, H, np.arange(G)), max(args.ks))
    res["map_bands_greedy"] = {str(k): sorted(int(b) for b in full[:k]) for k in args.ks}
    if gorder:
        res["fold_agreement"] = {str(k): round(float(np.mean([len(set(gorder[n][:k]) & set(full[:k])) / k for n in gorder])), 2)
                                 for k in args.ks}
    save()
    print("map bands", json.dumps(res["map_bands_greedy"]), "| fold agreement", json.dumps(res.get("fold_agreement")), flush=True)


if __name__ == "__main__":
    main()
