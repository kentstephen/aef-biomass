# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "psutil"]
# ///
"""Training rows for the biomass head: from every square, K CTrees cells x P pixels x 8 years, all three embeddings.

Squares: data/squares.json (37 picked for tree loss, kind "forest-loss") and data/veg_squares.json (55 across every
kind of vegetation). Per square, K cells of CTrees' 100 m grid (with at least 40 of their AlphaEarth pixels inside the
square) and P pixels drawn in each; for each of the 8 years 2018 to 2025:
  aef int8 (G, Y, P, 64), tes int8 (G, Y, P, 128) with tes_s float16 scale, olmo int8 (G, Y, P, 128)
  agb float32 (G, Y)      CTrees cell biomass, Mg/ha (NaN none)
  height uint8 (G, P), height_year int16 (G, P): Meta 10 m mean height and its imagery year (255 / 0 none)
  cls int8 (G, Y, P)      class where ESA WorldCover (2020, 2021) and Impact Observatory (same year) agree, else -1:
                          0 trees, 1 shrub, 2 grass, 3 crop, 4 built, 5 bare, 6 water, 7 wetland or mangrove, 8 moss
  loss uint8 (G, P)       Hansen loss year (0 none, 19 = 2019)
  sq int16 (G,)           square index into names; kinds, continents per square

Loads one square at a time (about 1.2 GB); output about 1.2 GB. Peak about 4 GB.

    uv run scripts/biomass_rows.py [--cells 500] [--px 8]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
YEARS = list(range(2018, 2026))
CLASSES = ["trees", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "moss"]
WC2C = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7, 95: 7, 100: 8}
IO_OK = {1: {6}, 2: {0}, 4: {7}, 5: {3}, 7: {4}, 8: {5}, 11: {1, 2, 8}}


def continent(lon, lat):
    if lon < -30:
        return "north-america" if lat > 12 else "south-america"
    if lon > 110 and lat < -10:
        return "oceania"
    if -20 < lon < 52 and -36 < lat < 36 and not (lon > 34 and lat > 12):
        return "africa"
    if lon < 45 and lat >= 36:
        return "europe"
    return "asia"


def squares():
    out = []
    for n, s in json.load(open(HERE / "data" / "squares.json")).items():
        out.append((n, s, "forest-loss"))
    for n, s in json.load(open(HERE / "data" / "veg_squares.json")).items():
        out.append((n, s, s["kind"]))
    return out


def classes(wc, io):
    """(n,) WorldCover and (n,) IO of the same year -> class where they agree, else -1."""
    c = np.full(wc.shape, -1, np.int8)
    for code, k in WC2C.items():
        m = wc == code
        ok = np.zeros(m.sum(), bool)
        iov = io[m]
        for io_code, allowed in IO_OK.items():
            if k in allowed:
                ok |= iov == io_code
        c[np.flatnonzero(m)[ok]] = k
    return c


def one(name, s, rng, K, P):
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    L = np.load(HERE / "data" / "labels" / f"{name}.npz")
    cell = L["cell"]
    ids, cnt = np.unique(cell, return_counts=True)
    ok = ids[cnt >= 40]
    pick = rng.choice(ok, size=min(K, ok.size), replace=False)
    order = np.argsort(cell, kind="stable")
    starts = np.searchsorted(cell[order], pick)
    counts = cnt[np.searchsorted(ids, pick)]
    pix = np.stack([order[st + rng.choice(c, P, replace=False)] for st, c in zip(starts, counts)])  # (G, P)
    G = len(pick)
    flat = pix.ravel()
    d = {}
    for k in ("aef", "tes", "olmo"):
        a = z[k]  # (Y, n, D)
        d[k] = a[:, flat].reshape(len(YEARS), G, P, -1).transpose(1, 0, 2, 3).copy()
        del a
    d["tes_s"] = z["tes_scale"][:, flat].reshape(len(YEARS), G, P).transpose(1, 0, 2).astype(np.float16)
    agb = L["agb"][:, flat].reshape(len(YEARS), G, P)[:, :, 0].T.astype(np.float32)  # same cell -> same value
    d["agb"] = np.where(agb >= 0, agb / 10, np.nan).astype(np.float32)
    d["height"] = L["height"][flat].reshape(G, P)
    hy = (L["height_date"][flat] // 10000).astype(np.int16)
    d["height_year"] = hy.reshape(G, P)
    cls = np.full((G, len(YEARS), P), -1, np.int8)
    io_years = list(L["io_years"])
    for wy in (2020, 2021):
        c = classes(z[f"wc{wy}"][flat], L["io"][io_years.index(wy)][flat])
        cls[:, YEARS.index(wy), :] = c.reshape(G, P)
    d["cls"] = cls
    d["loss"] = z["loss"][flat].reshape(G, P)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=500)
    ap.add_argument("--px", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sq = squares()
    parts, names, kinds, conts = [], [], [], []
    t0 = time.time()
    for i, (n, s, kind) in enumerate(sq):
        if not (HERE / "data" / "labels" / f"{n}.npz").exists() or not (HERE / "data" / "squares" / f"{n}.npz").exists():
            print(f"skip {n}: not read", flush=True)
            continue
        d = one(n, s, rng, args.cells, args.px)
        d["sq"] = np.full(len(d["agb"]), len(names), np.int16)
        parts.append(d)
        names.append(n)
        kinds.append(kind)
        lon, lat = s["center"]
        conts.append(continent(lon, lat))
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"{len(names):3d} {n:22s} {kind:13s} {conts[-1]:13s} cells {len(d['agb'])}  agb med "
              f"{np.nanmedian(d['agb']):6.1f}  [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(HERE / "data" / "rows.npz", names=np.array(names), kinds=np.array(kinds), continents=np.array(conts),
             years=np.array(YEARS), classes=np.array(CLASSES), **out)
    print(f"wrote data/rows.npz: {len(out['agb'])} cells x {len(YEARS)} years x {args.px} px, {len(names)} squares, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
