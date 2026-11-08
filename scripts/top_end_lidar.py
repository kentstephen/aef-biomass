# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Which biomass map follows airborne lidar where it matters: the top end of forest, and the level of chaparral.

Lidar canopy height (scripts/read_lidar.py, data/lidar) is neither GEDI, CTrees nor Meta. Per CTrees 100 m cell
(at least 40 of its AlphaEarth pixels with lidar cover of 80 % or more), against the cell's lidar mean height and its
mean 90th percentile height:
  ctrees   CTrees aboveground biomass, the lidar's year
  cci      ESA CCI Biomass v7.0, the lidar's year (2024 and before)
  heads    the square-fold heads of biomass_head.py (8 folds averaged; no lidar square was in training), on the
           embedding of the lidar's year: aef, olmo, all, and the heads taught with weight on the top (olmo_top,
           all_top)
Epoch: the first flight of two-flight squares; werk10, each pixel's campaign year (a cell takes its commonest year).

(a) Top end, the lidar squares with forest: rank correlation of each map with lidar height over all cells, over the
    cells 20 m and taller, and 30 m and taller; each map's median per lidar height band (where it saturates);
    per square medians.
(b) Shrub level, the chaparral lidar squares (Eaton and Palisades 2023 flights before the 2025 fires, Los Padres
    statewide): cells where most pixels are WorldCover shrub and the lidar mean height is 0.3 to 6 m. CTrees, the
    SoCal chaparral map (2021, its last year before the 2022 fall; data/chaparral) and the heads against lidar
    height: rank correlation, median biomass per meter of lidar height, and how steady that ratio is across
    height bands and squares (a map whose level is right for one stand height but not another changes ratio).
(c) Tundra: what labels the tundra squares carry that could check CTrees there.

Writes results/top_end_lidar.json. One square's one-year features at a time: peak about 3 GB (stops past 4 GB).

    uv run scripts/top_end_lidar.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import class_hard_calibrate as chc  # noqa: E402
import lidar_check as lc  # noqa: E402

YEARS = list(range(2018, 2026))
DIMS = {"aef": 64, "olmo": 128, "all": 320}
HEADS = {"aef": "aef_square", "olmo": "olmo_square", "all": "all_square", "olmo_top": "olmo_square_top",
         "all_top": "all_square_top"}
FOREST = ["lidar-us-redwood", "lidar-us-sierra-teak", "lidar-nz-glenbervie", "lidar-ee-lahemaa", "lidar-ee-tartu",
          "lidar-nz-esk-valley", "lidar-nz-banks", "lidar-us-eaton", "lidar-us-palisades", "lidar-us-sjer",
          "lidar-us-los-padres"]
CHAP = ["lidar-us-eaton", "lidar-us-palisades", "lidar-us-los-padres"]
TUNDRA = ["yamal-ru", "noatak-us", "nunavik-ca"]
BANDS = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 200]
SHRUB_BANDS = [0.3, 1, 2, 3, 4, 6]
MAX_GB = 4.0
peak = [0.0]


def guard():
    p = psutil.Process()
    while True:
        gb = p.memory_info().rss / 1e9
        peak[0] = max(peak[0], gb)
        if gb > MAX_GB:
            print(f"STOP: RSS {gb:.1f} GB", flush=True)
            import os
            os._exit(3)
        time.sleep(0.5)


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(bh.spearman(a[m], b[m]), 3) if m.sum() >= 30 else None


def cell_means(cell, valid, **vals):
    """Per CTrees cell: count of valid pixels and the mean of each value over them."""
    ids, inv = np.unique(cell, return_inverse=True)
    cnt = np.bincount(inv, valid, ids.size)
    out = {"_n": cnt}
    for k, v in vals.items():
        ok = valid & np.isfinite(v)
        s = np.bincount(inv, np.where(ok, v, 0), ids.size)
        c = np.bincount(inv, ok, ids.size)
        out[k] = np.where(c >= 0.8 * np.maximum(cnt, 1), s / np.maximum(c, 1), np.nan)
    return out, inv


def square_cells(name, nets, shrub=False):
    """Cell table for one lidar square: lidar heights, CTrees, CCI, chaparral map, heads."""
    L = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    lab = np.load(HERE / "data" / "labels" / f"{name}.npz")
    cci_f = HERE / "data" / "cci" / f"{name}.npz"
    chap_f = HERE / "data" / "chaparral" / f"{name}.npz"
    chm, p90, cov = L["chm"][0].astype(np.float32), L["chm_p90"][0].astype(np.float32), L["cover"][0]
    n = chm.size
    if str(L["source"]) == "werk10":
        ypx = L["year_px"].astype(int)
        ok_l = (cov >= 80) & (ypx >= 2018) & (ypx <= 2025)
    else:
        ypx = np.full(n, int(min(max(int(L["year"][0]), 2018), 2025)))
        ok_l = cov >= 80
    vals = {"h": np.where(ok_l, chm, np.nan), "h90": np.where(ok_l, p90, np.nan)}
    agb = lab["agb"].astype(np.float32)
    yi = np.clip(ypx - 2018, 0, 7)
    ct = agb[yi, np.arange(n)]
    vals["ctrees"] = np.where(ct >= 0, ct / 10, np.nan)
    if cci_f.exists():
        c = np.load(cci_f)["agb"].astype(np.float32)
        c[c == 65535] = np.nan
        ci = np.clip(ypx - 2018, 0, c.shape[0] - 1)
        vals["cci"] = np.where(ypx <= 2024, c[ci, np.arange(n)], np.nan)
    if chap_f.exists():
        m = np.load(chap_f)
        vals["chaparral_map"] = m["agb"][list(m["years"]).index(2021)]
    preds = {k: np.full(n, np.nan, np.float32) for k in nets}
    for y in sorted(set(ypx[ok_l].tolist())):
        sel = ok_l & (ypx == y)
        for inp in ("aef", "olmo", "all"):
            x, okE = chc.feats(z, inp, y - 2018)
            m = sel & okE
            for k, ns in nets.items():
                if k.split("_")[0] == inp and m.any():
                    a, _ = lc.state(ns, x[m])
                    preds[k][m] = a
            del x
    vals |= preds
    wc = z["wc2021"]
    valid = ok_l.copy()
    if shrub:
        valid &= (wc == 20) & (chm >= 0.3) & (chm <= 6)
    cm, _ = cell_means(lab["cell"], valid, **vals)
    keep = cm["_n"] >= 40 if not shrub else cm["_n"] >= 40
    out = {k: v[keep] for k, v in cm.items() if k != "_n"}
    out["sq"] = np.full(int(keep.sum()), name)
    del z, L, lab
    return out


def bands(h, v, edges):
    o = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi) & np.isfinite(v)
        if m.sum() >= 20:
            o[f"{lo}-{hi}"] = [int(m.sum()), round(float(np.median(v[m])), 1)]
    return o


def main():
    threading.Thread(target=guard, daemon=True).start()
    nets = {k: lc.nets_of(HERE / "data" / "heads" / f"{f}.npz", DIMS[k.split("_")[0]]) for k, f in HEADS.items()}
    maps = ["ctrees", "cci"] + list(HEADS)
    res = {"note": "per CTrees 100 m cell; lidar mean height (h) and mean 90th percentile height (h90)"}
    t0 = time.time()

    # (a) top end
    parts = []
    for name in FOREST:
        if not (HERE / "data" / "lidar" / f"{name}.npz").exists():
            continue
        d = square_cells(name, nets)
        parts.append(d)
        print(f"  {name}: {d['h'].size} cells, lidar h median {np.nanmedian(d['h']):.1f} m  "
              f"[{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
    A = {k: np.concatenate([p[k] for p in parts if k in p]) if all(k in p for p in parts) else
         np.concatenate([p.get(k, np.full(p["h"].size, np.nan)) for p in parts]) for k in parts[0]}
    top = {"cells": int(A["h"].size), "rank_all": {}, "rank_h20": {}, "rank_h30": {}, "rank_h90_all": {},
           "rank_map200": {}, "by_height_band": {}, "per_square": {}}
    for k in maps:
        v = A[k]
        top["rank_all"][k] = spear(v, A["h"])
        top["rank_h90_all"][k] = spear(v, A["h90"])
        top["rank_h20"][k] = spear(v[A["h"] >= 20], A["h"][A["h"] >= 20])
        top["rank_h30"][k] = spear(v[A["h"] >= 30], A["h"][A["h"] >= 30])
        top["by_height_band"][k] = bands(A["h"], v, BANDS)
    # where a map reads 200 Mg/ha or more: does lidar height still order its values?
    for k in ("ctrees", "cci", "all", "all_top"):
        m = A[k] >= 200
        top["rank_map200"][k] = {"cells": int(m.sum()), "rank": spear(A[k][m], A["h"][m]),
                                 "lidar_h_median": round(float(np.nanmedian(A["h"][m])), 1) if m.any() else None}
    for name in sorted(set(A["sq"])):
        m = A["sq"] == name
        top["per_square"][name] = {"cells": int(m.sum()), "lidar_h": round(float(np.nanmedian(A["h"][m])), 1),
                                   "lidar_h90": round(float(np.nanmedian(A["h90"][m])), 1)} | {
            k: round(float(np.nanmedian(A[k][m])), 1) if np.isfinite(A[k][m]).any() else None for k in maps} | {
            f"rank_{k}": spear(A[k][m], A["h"][m]) for k in ("ctrees", "cci", "all")}
    res["top_end"] = top
    print("rank with lidar h, all cells " + json.dumps(top["rank_all"]), flush=True)
    print("rank, cells >= 20 m      " + json.dumps(top["rank_h20"]), flush=True)
    print("rank, cells >= 30 m      " + json.dumps(top["rank_h30"]), flush=True)
    print("map >= 200: " + json.dumps(top["rank_map200"]), flush=True)
    for k in maps:
        print(f"  band {k:9s} " + "  ".join(f"{b}:{v[1]}" for b, v in top["by_height_band"][k].items()), flush=True)
    for nm, v in top["per_square"].items():
        print(f"  {nm:22s} {json.dumps(v)}", flush=True)

    # (b) shrub level
    parts = [square_cells(n, nets, shrub=True) for n in CHAP]
    S = {k: np.concatenate([p.get(k, np.full(p["h"].size, np.nan)) for p in parts]) for k in parts[0]}
    smaps = ["ctrees", "chaparral_map", "all", "olmo", "cci"]
    sh = {"cells": int(S["h"].size), "cells_per_square": {n: int((S["sq"] == n).sum()) for n in CHAP}}
    for k in smaps:
        v = S[k]
        ratio = v / np.maximum(S["h"], 0.1)
        sh[k] = {"median": round(float(np.nanmedian(v)), 1), "rank_h": spear(v, S["h"]), "rank_h90": spear(v, S["h90"]),
                 "per_m_median": round(float(np.nanmedian(ratio)), 2),
                 "per_m_by_band": {b: x[1] for b, x in bands(S["h"], ratio, SHRUB_BANDS).items()},
                 "median_by_band": {b: x[1] for b, x in bands(S["h"], v, SHRUB_BANDS).items()},
                 "per_square": {n: [round(float(np.nanmedian(v[S["sq"] == n])), 1),
                                    round(float(np.nanmedian(ratio[S["sq"] == n])), 2)] for n in CHAP if (S["sq"] == n).any()}}
    sh["lidar_h_median"] = round(float(np.nanmedian(S["h"])), 2)
    sh["lidar_h_by_square"] = {n: round(float(np.nanmedian(S["h"][S["sq"] == n])), 2) for n in CHAP if (S["sq"] == n).any()}
    sh["ctrees_over_map"] = round(float(np.nanmedian(S["ctrees"] / np.maximum(S["chaparral_map"], 0.1))), 2)
    res["shrub_level"] = sh
    print("shrub cells", sh["cells"], sh["cells_per_square"], "lidar h", sh["lidar_h_median"], sh["lidar_h_by_square"],
          "ctrees/map", sh["ctrees_over_map"], flush=True)
    for k in smaps:
        print(f"  {k:14s} {json.dumps(sh[k])}", flush=True)

    # (c) tundra
    tu = {}
    for name in TUNDRA:
        lab = np.load(HERE / "data" / "labels" / f"{name}.npz")
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        h = lab["height"]
        hy = lab["height_date"] // 10000
        okh = h != 255
        agb = lab["agb"].astype(np.float32)
        agb[agb < 0] = np.nan
        cci_f = HERE / "data" / "cci" / f"{name}.npz"
        cc = None
        if cci_f.exists():
            c = np.load(cci_f)["agb"].astype(np.float32)
            c[c == 65535] = np.nan
            cc = round(float(np.nanmedian(c)), 1)
        lat = float(np.mean([z["rc"][0], z["rc"][1]]))
        tu[name] = {"meta_height_share": round(float(okh.mean()), 3),
                    "meta_height_m_median_p90": [float(np.median(h[okh])), float(np.percentile(h[okh], 90))] if okh.any() else None,
                    "meta_share_2m_plus": round(float((h[okh] >= 2).mean()), 3) if okh.any() else None,
                    "meta_years": [int(np.percentile(hy[okh & (hy > 0)], 5)), int(np.percentile(hy[okh & (hy > 0)], 95))] if (okh & (hy > 0)).any() else None,
                    "ctrees_median": round(float(np.nanmedian(agb)) / 10, 1), "cci_median": cc,
                    "lidar_on_disk": (HERE / "data" / "lidar" / f"{name}.npz").exists(),
                    "wc2021": {int(k): round(float(v), 3) for k, v in zip(*np.unique(z["wc2021"], return_counts=True)) if v / z["wc2021"].size >= 0.02}}
        tu[name]["wc2021"] = {k: round(v / z["wc2021"].size, 3) for k, v in zip(*[a.tolist() for a in np.unique(z["wc2021"], return_counts=True)]) if v / z["wc2021"].size >= 0.02}
        print(f"  tundra {name}: {json.dumps(tu[name])}", flush=True)
    res["tundra"] = tu
    res["peak_gb"] = round(peak[0], 2)
    json.dump(res, open(HERE / "results" / "top_end_lidar.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
