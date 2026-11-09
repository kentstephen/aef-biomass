"""Grass years outside the US against Wu et al. 2026 global grassland biomass (Zenodo 18044163).

Wu: yearly peak aboveground grassland biomass 2000 to 2022, 0.005 degree cells (~500 m), g/m2, inside a grassland
mask (steppe, savanna, tundra, alpine meadow). A random forest on MODIS NDVI, climate, CO2 and terrain, taught on
11,265 field samples worldwide: a model, not a measurement, but field-taught and not GEDI or CTrees.

Every square of veg_squares.json outside the US, and the US rangeland squares RAP covers (where Wu is checked
against RAP, a field-taught 30 m map, to see how much Wu itself follows grass years). Per Wu cell, 2018 to 2022:
  map      the map's head (web/public/models/biomass_aef.json, CTrees-taught total biomass), mean of its 10 m pixels
  rap head a head taught on RAP's herbaceous biomass from one year's AlphaEarth (map_check.py's 64-unit head),
           taught on every RAP square (in the US checks, each square held out in turn)
  rap      RAP itself (US squares only), mean of its pixels in the cell
A cell counts when 80 % of its 10 m pixels are valid in AlphaEarth in every year and Wu has a value every year.

Scores per square, then the median over squares: rank of cells within a square, each cell's departure from its own
mean over the years (correlation), the square's yearly median (correlation over 5 years), and the level (square
medians, Mg/ha). Writes results/grass_abroad.json. RAM about 3 GB (one square at a time plus a 1.2 M-row table).

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with torch python scripts/grass_abroad.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import burn_check as bc  # noqa: E402  (the map's head, as the browser runs it)
import map_check as mc  # noqa: E402

OUT = HERE / "results" / "grass_abroad.json"
WU = "https://zenodo.org/api/records/18044163/files/AGB_{}.tif/content"
SQ_YEARS = list(range(2018, 2026))
YEARS = list(range(2018, 2023))
YI = [SQ_YEARS.index(y) for y in YEARS]
RES, X0, Y0 = 8.983111749910169e-05, -180.0, 83.68570533713473
WRES = 0.005


def pix_lonlat(rc):
    y0, y1, x0, x1 = rc
    lat = Y0 - (np.arange(y0, y1) + 0.5) * RES
    lon = X0 + (np.arange(x0, x1) + 0.5) * RES
    return lon, lat


def wu_cells(rc):
    """Wu cell index of every pixel (n,), the cells' (row, col) window, and Wu (Y, cells) Mg/ha."""
    lon, lat = pix_lonlat(rc)
    cc = np.floor((lon + 180) / WRES).astype(int)
    rr = np.floor((90 - lat) / WRES).astype(int)
    r0, c0 = rr.min(), cc.min()
    h, w = rr.max() - r0 + 1, cc.max() - c0 + 1
    cell = ((rr - r0)[:, None] * w + (cc - c0)[None, :]).ravel()
    vals = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
        for y in YEARS:
            with rasterio.open("/vsicurl/" + WU.format(y)) as s:
                a = s.read(1, window=Window(c0, r0, w, h)).astype(np.float32)
            vals.append(a.ravel() / 100.0)  # g/m2 -> Mg/ha
    return cell, int(h * w), np.stack(vals)


def cell_mean(v, cell, nc):
    """(Y, n) pixel values with NaN -> (Y, nc) cell means and valid share."""
    out, share = np.full((v.shape[0], nc), np.nan, np.float32), np.zeros((v.shape[0], nc), np.float32)
    tot = np.bincount(cell, minlength=nc).astype(np.float32)
    for k in range(v.shape[0]):
        ok = np.isfinite(v[k])
        n = np.bincount(cell[ok], minlength=nc)
        s = np.bincount(cell[ok], weights=v[k][ok], minlength=nc)
        out[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        share[k] = n / np.maximum(tot, 1)
    return out, share


def map_head(q):
    agb = np.full(q.shape[:2], np.nan, np.float32)
    for y in range(q.shape[0]):
        ok = q[y, :, 0] != -128
        agb[y, ok] = bc.map_agb(q[y, ok])
    return agb


def rap_table(names, px, rng):
    X, t, s = [], [], []
    for si, name in enumerate(names):
        lab, ly = mc.label_of("rap", name)
        q = np.load(HERE / "data" / "squares" / f"{name}.npz")["aef"]
        for k, y in enumerate(ly):
            if y not in SQ_YEARS:
                continue
            e = q[SQ_YEARS.index(y)]
            cand = np.flatnonzero((e[:, 0] != -128) & np.isfinite(lab[k]))
            pick = np.sort(rng.choice(cand, min(px, cand.size), replace=False))
            X.append(bh.deq(e[pick]).astype(np.float16)), t.append(lab[k][pick]), s.append(np.full(pick.size, si, np.int16))
        del q
    return np.concatenate(X), np.concatenate(t), np.concatenate(s)


def rap_pred(net, q):
    out = np.full(q.shape[:2], np.nan, np.float32)
    for y in range(q.shape[0]):
        ok = q[y, :, 0] != -128
        out[y, ok] = mc.run_one(net, bh.deq(q[y, ok]))
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() == 0 or b[m].std() == 0:
        return None
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    return round(bh.spearman(a[m], b[m]), 3)


def square_scores(ref, P):
    """ref, P: (Y, cells) on the kept cells."""
    dep_r, dep_p = ref - ref.mean(0), P - P.mean(0)
    mr, mp = np.median(ref, 1), np.median(P, 1)
    return {"rank in square": spear(ref.mean(0), P.mean(0)),
            "cell departures": corr(dep_r.ravel(), dep_p.ravel()),
            "yearly median": corr(mr, mp),
            "steps": corr(np.diff(ref, axis=0).ravel(), np.diff(P, axis=0).ravel()),
            "median Mg/ha": round(float(np.median(P)), 3)}


def med(xs):
    xs = [x for x in xs if x is not None]
    return (round(float(np.median(xs)), 3), sum(x > 0 for x in xs), len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=8)
    ap.add_argument("--px", type=int, default=12000)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    veg = json.load(open(HERE / "data" / "veg_squares.json"))
    us = json.load(open(HERE / "data" / "us_squares.json"))
    abroad = [n for n, v in veg.items() if not n.endswith("-us") and v["kind"] in
              ("grass", "shrub", "savanna", "tundra", "desert-edge", "wetland", "crop", "city-edge")]
    abroad = [n for n in abroad if (HERE / "data" / "squares" / f"{n}.npz").exists()]
    rapsq = mc.squares_of("rap")
    kind = {n: veg[n]["kind"] for n in abroad} | {n: (us.get(n) or veg.get(n))["kind"] for n in rapsq}
    print(f"{len(abroad)} squares abroad, {len(rapsq)} RAP squares", flush=True)

    X, T, S = rap_table(rapsq, args.px, rng)
    print(f"RAP table {X.shape} [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    net_all = mc.fit_one(X, T)
    nets_out = {}
    for si, n in enumerate(rapsq):
        m = S != si
        nets_out[n] = mc.fit_one(X[m], T[m])
    del X, T, S
    print(f"RAP heads fit [{time.time() - t0:.0f} s]", flush=True)

    res = {"note": __doc__.split("\n\n")[0], "years": YEARS, "squares": {}}
    for name in abroad + rapsq:
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        rc = tuple(int(v) for v in z["rc"])
        q = z["aef"][YI]
        try:
            cell, nc, wu = wu_cells(rc)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: Wu read failed {e!r}", flush=True)
            continue
        agb, share = cell_mean(map_head(q), cell, nc)
        net = nets_out.get(name, net_all)
        rp, _ = cell_mean(rap_pred(net, q), cell, nc)
        keep = np.all(share >= 0.8, 0) & np.all(np.isfinite(wu), 0)
        o = {"kind": kind[name], "cells": int(keep.sum()), "cells in square": nc}
        if keep.sum() >= 10:
            W = wu[:, keep]
            o["wu median Mg/ha"] = round(float(np.median(W)), 3)
            o["wu yearly medians"] = [round(float(v), 3) for v in np.median(W, 1)]
            o["map"] = square_scores(W, agb[:, keep])
            o["rap head"] = square_scores(W, rp[:, keep])
            if name in rapsq:
                lab, ly = mc.label_of("rap", name)
                lab = lab[[ly.index(y) for y in YEARS]]
                r, rs = cell_mean(lab, cell, nc)
                k2 = keep & np.all(rs >= 0.8, 0)
                if k2.sum() >= 10:
                    o["wu against rap"] = square_scores(r[:, k2], wu[:, k2])
                    o["rap head against rap"] = square_scores(r[:, k2], rp[:, k2])
                    o["map against rap"] = square_scores(r[:, k2], agb[:, k2])
        res["squares"][name] = o
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"{name} ({kind[name]}): {json.dumps(o)} [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        del z, q
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")

    summ = {}
    for grp, names in (("abroad", abroad), ("us", rapsq)):
        got = [res["squares"][n] for n in names if n in res["squares"] and "map" in res["squares"][n]]
        g = {"squares": len(got)}
        for who in ("map", "rap head", "wu against rap", "rap head against rap", "map against rap"):
            if not any(who in o for o in got):
                continue
            g[who] = {k: med([o[who][k] for o in got if who in o]) for k in ("rank in square", "cell departures", "yearly median", "steps")}
        if grp == "abroad":
            for who in ("map", "rap head"):
                g[who]["rank of square medians against wu"] = spear(np.array([o["wu median Mg/ha"] for o in got]),
                                                                     np.array([o[who]["median Mg/ha"] for o in got]))
        summ[grp] = g
    res["summary (median over squares, squares positive, squares)"] = summ
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    print(f"wrote {OUT} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
