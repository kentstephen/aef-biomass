"""Burned forest against before and after fire lidar that can tell standing dead trees from live canopy: Jasper.

Section 32 found lidar's canopy height is no referee on burns: dead trees keep standing, so the top of the canopy
barely falls. Canopy cover can: a crown that lost its needles lets more of the 1 m grid through to the ground.
The Jasper fire (July 2024, Jasper National Park, Alberta) has NRCan HRDEM 1 m surface and ground models from
September 2019 (NRCAN-Jasper_Park_2019_2) and October 2024, three months after the fire (PC-Jasper_2024), on the
same 1 m grid (EPSG:3979), overlapping on about 227 km2 around the town and the burn.

Per AlphaEarth pixel (10 m), from the 1 m canopy height (surface minus ground) of each flight: mean height, and
cover = the share of 1 m cells above 2 m (and above 5 m). Forest = 2019 mean height 5 m or more and cover 50 % or
more. Burned = Hansen loss in 2024 (stand-replacing loss in the fire year); steady = no Hansen loss 2001 to 2024.
Beside them the map's head (AlphaEarth 2019 and 2025, also 2024) and CTrees (2019, 2024).

Tests: which lidar measure tells burned from steady forest (AUC of height fall and of cover fall); in the burned
forest, how far each falls; the map's fall against each per pixel and per 100 m cell (rank); and the map's fall where
the trees stand dead (height kept, cover gone). Lytton (2021) has no pre-fire flight over the burn (the 2017 and
2019 projects listed for it overlap the 2023 flight only 100 km south), and Fort McMurray (May 2016) has no pre-fire
AlphaEarth year, so Jasper is the one pair here.

Writes results/burn_lidar_ca.json. Peak RAM about 2 GB.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with icechunk --with xarray \
        --with requests --with pyproj --with pyogrio --with scipy python scripts/burn_lidar_ca.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import aef_read as ar  # noqa: E402
import top_end_plots as tp  # noqa: E402

OUT = HERE / "results" / "burn_lidar_ca.json"
EXT = HERE / "data" / "labels_new" / "hrdem"
URL = "https://canelevation-dem.s3.ca-central-1.amazonaws.com/hrdem-lidar/{}-{}.tif"
P19, P24 = "NRCAN-Jasper_Park_2019_2-1m", "PC-Jasper_2024-1m"
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
           GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
TILE = 1000  # m
YEARS = (2019, 2023, 2024, 2025)


def overlap():
    import pyogrio
    g = [shapely.union_all(pyogrio.read_dataframe(EXT / f"{p}-extent.geojson").to_crs("EPSG:3979").geometry.values)
         for p in (P19, P24)]
    return g[0].intersection(g[1])


def chm(ds_s, ds_t, b):
    w = from_bounds(*b, ds_s.transform).round_offsets().round_lengths()
    s = ds_s.read(1, window=w, boundless=True, fill_value=-32767).astype(np.float32)
    t = ds_t.read(1, window=w, boundless=True, fill_value=-32767).astype(np.float32)
    h = s - t
    h[(s <= -32000) | (t <= -32000) | ~np.isfinite(h)] = np.nan
    h[h < 0] = 0
    h[h > 90] = np.nan
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    ar.guard(args.max_gb)
    t0 = time.time()
    proc = psutil.Process()
    O = overlap()
    to_ll = Transformer.from_crs("EPSG:3979", "EPSG:4326", always_xy=True)
    bx = O.bounds
    lons, lats = to_ll.transform([bx[0], bx[2], bx[0], bx[2]], [bx[1], bx[1], bx[3], bx[3]])
    rc = ar.rc_of(min(lons), min(lats), max(lons), max(lats))
    y0, y1, x0, x1 = rc
    Hh, Ww = y1 - y0, x1 - x0
    npx = Hh * Ww
    print(f"overlap {O.area / 1e6:.0f} km2, AlphaEarth box {Hh} x {Ww}", flush=True)
    acc = {k: np.zeros(npx, np.float64) for k in ("n", "h19", "h24", "c19", "c24", "d19", "d24")}
    tiles = [(x, y) for x in np.arange(np.floor(bx[0]), bx[2], TILE) for y in np.arange(np.floor(bx[1]), bx[3], TILE)
             if O.intersects(shapely.box(x, y, x + TILE, y + TILE))]
    print(f"{len(tiles)} lidar tiles of {TILE} m", flush=True)
    with rasterio.Env(**ENV):
        dss = {(p, k): rasterio.open("/vsicurl/" + URL.format(p, k)) for p in (P19, P24) for k in ("dsm", "dtm")}
        for ti, (x, y) in enumerate(tiles):
            b = (x, y, x + TILE, y + TILE)
            a = chm(dss[(P19, "dsm")], dss[(P19, "dtm")], b)
            c = chm(dss[(P24, "dsm")], dss[(P24, "dtm")], b)
            ok = np.isfinite(a) & np.isfinite(c)
            if ok.sum() < 1000:
                continue
            rr, cc = np.nonzero(ok)
            X = x + cc + 0.5
            Y = y + TILE - rr - 0.5
            lo, la = to_ll.transform(X, Y)
            pc = np.floor((np.asarray(lo) - ar.X0) / ar.RES).astype(np.int64) - x0
            pr = np.floor((ar.Y0 - np.asarray(la)) / ar.RES).astype(np.int64) - y0
            ins = (pc >= 0) & (pc < Ww) & (pr >= 0) & (pr < Hh)
            g = (pr * Ww + pc)[ins]
            va, vc = a[ok][ins], c[ok][ins]
            acc["n"] += np.bincount(g, minlength=npx)
            acc["h19"] += np.bincount(g, weights=va, minlength=npx)
            acc["h24"] += np.bincount(g, weights=vc, minlength=npx)
            acc["c19"] += np.bincount(g, weights=(va > 2).astype(float), minlength=npx)
            acc["c24"] += np.bincount(g, weights=(vc > 2).astype(float), minlength=npx)
            acc["d19"] += np.bincount(g, weights=(va > 5).astype(float), minlength=npx)
            acc["d24"] += np.bincount(g, weights=(vc > 5).astype(float), minlength=npx)
            if ti % 20 == 0:
                gb = proc.memory_info().rss / 1e9
                print(f"  tile {ti + 1}/{len(tiles)} [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
                if gb > args.max_gb:
                    raise SystemExit(f"STOP: RSS {gb:.1f} GB")
        for d in dss.values():
            d.close()
    n = acc.pop("n")
    # 1 m cells per AlphaEarth pixel at this latitude (about 10 m north-south by 6 m east-west at 53 N); 80 % of them
    mid_lat = float(np.mean(lats))
    per_px = (ar.RES * 111320) ** 2 * np.cos(np.radians(mid_lat))
    keep = n >= 0.8 * per_px
    T = {k: np.where(keep, v / np.maximum(n, 1), np.nan).astype(np.float32) for k, v in acc.items()}
    idx = np.flatnonzero(keep)
    print(f"pixels with both flights: {idx.size} [{time.time() - t0:.0f} s]", flush=True)
    lon, lat = ar.centers(rc)
    LON, LAT = np.meshgrid(lon, lat)
    LONk, LATk = LON.ravel()[idx], LAT.ravel()[idx]
    del LON, LAT
    tab = {k: v[idx] for k, v in T.items()}
    tab["loss"] = tp.hansen_loss(LONk[None, :], LATk[None, :]).ravel()
    ct = tp.ctrees(LONk[None, :], LATk[None, :], [2019, 2024])
    tab["ct19"], tab["ct24"] = ct[0].ravel(), ct[1].ravel()
    # AlphaEarth in strips of rows, the head in batches: the whole box at once is several GB
    STRIP = 256
    rows_of = idx // Ww
    for Y in YEARS:
        agb = np.full(idx.size, np.nan, np.float32)
        for r0 in range(0, Hh, STRIP):
            sel = np.flatnonzero((rows_of >= r0) & (rows_of < r0 + STRIP))
            if not sel.size:
                continue
            q = ar.raw((y0 + r0, y0 + min(r0 + STRIP, Hh), x0, x1), Y)
            local = idx[sel] - r0 * Ww
            agb[sel] = ar.head(q[local])[0]
            del q
        tab[f"map{Y}"] = agb
        print(f"  AlphaEarth {Y} [{time.time() - t0:.0f} s, {proc.memory_info().rss / 1e9:.1f} GB]", flush=True)
    tab["block"] = ((idx // Ww) // 10 * (Ww // 10 + 1) + (idx % Ww) // 10).astype(np.int64)
    tab["lon"], tab["lat"] = LONk.astype(np.float32), LATk.astype(np.float32)
    res = analyze(tab) | {"overlap km2": round(O.area / 1e6, 1), "pixels": int(idx.size), "seconds": round(time.time() - t0),
                          "peak_gb": round(proc.memory_info().rss / 1e9, 2)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def fall(a, b):
    return np.clip(1 - b / np.maximum(a, 1e-6), -1, 1)


def auc(pos, neg):
    """Mann-Whitney AUC, NaN dropped: how often a burned pixel scores above a steady one."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 20 or neg.size < 20:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return round(float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)), 4)


def analyze(t):
    forest = (t["h19"] >= 5) & (t["c19"] >= 0.5)
    burned = forest & (t["loss"] == 24)
    steady = forest & (t["loss"] == 0)
    fH, fC, fD = fall(t["h19"], t["h24"]), fall(t["c19"], t["c24"]), fall(t["d19"], t["d24"])
    fM = fall(t["map2019"], t["map2025"])
    fM24 = fall(t["map2019"], t["map2024"])
    fCT = fall(t["ct19"], t["ct24"])
    o = {"forest px": int(forest.sum()), "burned px (Hansen 2024)": int(burned.sum()), "steady px": int(steady.sum())}
    o["tells burned from steady forest (AUC)"] = {k: auc(v[burned], v[steady]) for k, v in
                                                   (("lidar height fall", fH), ("lidar cover fall (>2 m)", fC),
                                                    ("lidar cover fall (>5 m)", fD), ("map fall 2019-2025", fM),
                                                    ("map fall 2019-2024", fM24), ("CTrees fall 2019-2024", fCT))}
    med = lambda v, m: round(float(np.nanmedian(v[m])), 3)  # noqa: E731
    o["median fall"] = {grp: {k: med(v, m) for k, v in (("lidar height", fH), ("lidar cover >2 m", fC),
                                                         ("lidar cover >5 m", fD), ("map 2019-2025", fM),
                                                         ("map 2019-2024", fM24), ("CTrees 2019-2024", fCT))}
                        for grp, m in (("burned", burned), ("steady", steady))}
    o["levels, burned"] = {k: med(t[k], burned) for k in ("h19", "h24", "c19", "c24", "map2019", "map2024", "map2025", "ct19", "ct24")}
    # standing dead: height kept (fell under 25 %) but cover gone (fell 50 % or more)
    dead = burned & (fH < 0.25) & (fC >= 0.5)
    gone = burned & (fH >= 0.5)
    o["burned, height kept and cover gone (standing dead)"] = {"share of burned": round(float(dead.sum() / max(burned.sum(), 1)), 3),
                                                               "map fall": med(fM, dead), "lidar height fall": med(fH, dead),
                                                               "lidar cover fall": med(fC, dead)}
    o["burned, height fell by half (trees down)"] = {"share of burned": round(float(gone.sum() / max(burned.sum(), 1)), 3),
                                                     "map fall": med(fM, gone), "lidar cover fall": med(fC, gone)}
    sp = lambda a, b, m: tp.spear(a[m], b[m])  # noqa: E731
    o["rank of the map's fall, per pixel"] = {grp: {"with height fall": sp(fM, fH, m), "with cover fall": sp(fM, fC, m),
                                                    "with cover fall >5 m": sp(fM, fD, m)}
                                              for grp, m in (("forest", forest), ("burned", burned))}
    # 100 m cells: means over forest pixels, cells with 50 or more
    B = t["block"]
    m = forest
    ub, inv = np.unique(B[m], return_inverse=True)
    cnt = np.bincount(inv)

    def bm(v):
        return np.bincount(inv, weights=np.nan_to_num(v[m]), minlength=ub.size) / np.maximum(cnt, 1)
    k = cnt >= 50
    cH, cC = fall(bm(t["h19"]), bm(t["h24"])), fall(bm(t["c19"]), bm(t["c24"]))
    cM = fall(bm(t["map2019"]), bm(t["map2025"]))
    burned_share = np.bincount(inv, weights=burned[m].astype(float), minlength=ub.size) / np.maximum(cnt, 1)
    o["per 100 m cell (forest pixels)"] = {"cells": int(k.sum()), "map fall with height fall": tp.spear(cM[k], cH[k]),
                                           "map fall with cover fall": tp.spear(cM[k], cC[k]),
                                           "map fall with Hansen's burned share": tp.spear(cM[k], burned_share[k]),
                                           "cover fall with Hansen's burned share": tp.spear(cC[k], burned_share[k]),
                                           "height fall with Hansen's burned share": tp.spear(cH[k], burned_share[k])}
    # the map's fall by lidar cover fall, in burned and steady forest
    rows = {}
    for lo, hi in ((-1, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        mm = forest & (fC >= lo) & (fC < hi)
        if mm.sum() >= 50:
            rows[f"{lo}-{hi}"] = {"px": int(mm.sum()), "map fall": med(fM, mm), "lidar height fall": med(fH, mm),
                                  "Hansen 2024 share": round(float((t["loss"][mm] == 24).mean()), 3)}
    o["by lidar cover fall (forest)"] = rows
    return o


if __name__ == "__main__":
    main()
