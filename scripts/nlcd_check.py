"""The map's biomass by land cover, every year 2018 to 2024, against
  --source nlcd  Annual NLCD (USGS, Collection 1 v1, 30 m, CONUS): the test squares inside CONUS;
  --source io    Esri / Impact Observatory annual land cover (10 m, worldwide, s3://io-10m-annual-lulc, one COG per
                 UTM zone and latitude band): every test square.

1. Per NLCD class, pooled over pixel-years: the median of the map's head (web/public/models/biomass_aef.json), of
   CTrees, of RAP's herbaceous biomass (rangeland squares) and of the SoCal chaparral map (chaparral squares).
   Grass and shrub reading high shows here, beside what each label says.
2. Class changes from one year to the next: forest to shrub, grass or barren (fire, harvest), and back, against the
   pixels whose class stayed. The head's and CTrees' median change on each.
3. Per square: its main class and the medians of each source.

NLCD read from Source Cooperative (kylebarron/usgs-landcover, annual-nlcd/c1/v1/cu/mosaic), nearest 30 m pixel at
each 10 m pixel center; kept in data/nlcd/<square>.npz (lndcov uint8 (7, n)). Writes results/nlcd_check.json.
One square at a time; RAM ~2 GB; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with icechunk --with xarray --with requests \
        python scripts/nlcd_check.py
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import burn_check as bc  # noqa: E402  (the map's head)
import read_labels as rl  # noqa: E402  (the raster sampler)

NLCD = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/"
        "c1/v1/cu/mosaic/Annual_NLCD_LndCov_{}_CU_C1V1.tif")
NYEARS = list(range(2018, 2025))
YEARS = list(range(2018, 2026))
CLASSES = {11: "water", 21: "developed open", 22: "developed low", 23: "developed med", 24: "developed high",
           31: "barren", 41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest", 52: "shrub",
           71: "grassland", 81: "pasture", 82: "crops", 90: "woody wetland", 95: "herbaceous wetland"}
FOREST, OPEN = {41, 42, 43}, {52, 71, 31}
IO = "https://s3.us-west-2.amazonaws.com/io-10m-annual-lulc/{}_{}.tif"
IO_CLASSES = {1: "water", 2: "trees", 4: "flooded vegetation", 5: "crops", 7: "built", 8: "bare", 9: "snow/ice",
              10: "clouds", 11: "rangeland"}
IO_FOREST, IO_OPEN = {2}, {11, 8}
BANDS = "CDEFGHJKLMNPQRSTUVWX"
LB_ACRE = 0.00112085
SRC = "nlcd"
RES, X0, Y0 = 8.983111749910169e-05, -180.0, 83.68570533713473


def squares(world=False):
    out = {}
    for f in ("squares.json", "veg_squares.json", "us_squares.json", "lidar_squares.json"):
        p = HERE / "data" / f
        if not p.exists():
            continue
        d = json.load(open(p))
        for k, v in (d.items() if isinstance(d, dict) else [(x.get("name"), x) for x in d]):
            if not (isinstance(v, dict) and "rc" in v) or not (HERE / "data" / "squares" / f"{k}.npz").exists() \
                    or not (HERE / "data" / "labels" / f"{k}.npz").exists():
                continue
            y0, y1, x0, x1 = v["rc"]
            lon, lat = X0 + (x0 + x1) / 2 * RES, Y0 - (y0 + y1) / 2 * RES
            if world or (-125 < lon < -66 and 24 < lat < 49.5):
                out[k] = v
    return out


def io_year(LON, LAT, y):
    """IO's classes at each pixel center for a year: the zone and band file of each pixel, nearest pixel; 250 none."""
    zone = np.clip(((LON + 180) // 6).astype(int) + 1, 1, 60)
    band = np.clip(((LAT + 80) // 8).astype(int), 0, len(BANDS) - 1)
    out = np.full(LON.shape, 250, np.uint8)
    for z, b in sorted(set(zip(zone.ravel().tolist(), band.ravel().tolist()))):
        m = (zone == z) & (band == b)
        try:
            v = rl.sample(IO.format(f"{z:02d}{BANDS[b]}", y), LON, LAT, 0)
        except Exception as ex:  # noqa: BLE001
            print(f"  IO {z:02d}{BANDS[b]} {y}: {ex}", flush=True)
            continue
        out[m] = np.where(v[m] == 0, 250, v[m])
    return out


def nlcd_for(name, rc):
    f = HERE / "data" / SRC / f"{name}.npz"
    if f.exists():
        return np.load(f)["lndcov"]
    LON, LAT = rl.grid(rc)
    if SRC == "io":
        a = np.stack([io_year(LON, LAT, y).ravel() for y in NYEARS]).astype(np.uint8)
    else:
        a = np.stack([rl.sample(NLCD.format(y), LON, LAT, 250).ravel() for y in NYEARS]).astype(np.uint8)
    f.parent.mkdir(exist_ok=True)
    np.savez_compressed(f, lndcov=a, years=np.array(NYEARS), rc=np.array(rc))
    return a


def labels_of(name):
    """(Y, n) Mg/ha with NaN for CTrees, RAP herbaceous and the chaparral map (None where the square has none)."""
    z = np.load(HERE / "data" / "labels" / f"{name}.npz")
    a = z["agb"].astype(np.float32)
    out = {"ctrees": np.where(a < 0, np.nan, a / 10)}
    p = HERE / "data" / "rap" / f"{name}.npz"
    if p.exists():
        d = np.load(p)
        out["rap"] = np.where((d["afg"] != 65535) & (d["pfg"] != 65535),
                              (d["afg"].astype(np.float32) + d["pfg"]) * LB_ACRE, np.nan)
    p = HERE / "data" / "chaparral" / f"{name}.npz"
    if p.exists():
        d = np.load(p)
        c = np.full((len(YEARS), d["agb"].shape[1]), np.nan, np.float32)
        for i, y in enumerate(d["years"]):
            if int(y) in YEARS:
                c[YEARS.index(int(y))] = d["agb"][i]
        out["chaparral"] = c
    return out


def head_of(name):
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    q = z["aef"]
    out = np.full(q.shape[:2], np.nan, np.float32)
    for i in range(q.shape[0]):
        ok = q[i, :, 0] != -128
        out[i, ok] = bc.map_agb(q[i, ok])
    return out


def guard(max_gb):
    def run():
        p = psutil.Process()
        while True:
            if p.memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def med(v):
    v = v[np.isfinite(v)]
    return round(float(np.median(v)), 1) if v.size >= 200 else None


def main():
    global SRC, CLASSES, FOREST, OPEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nlcd", "io"], default="nlcd")
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    SRC = args.source
    if SRC == "io":
        CLASSES, FOREST, OPEN = IO_CLASSES, IO_FOREST, IO_OPEN
    out_file = HERE / "results" / f"{SRC}_check.json"
    guard(args.max_gb)
    t0 = time.time()
    sq = squares(world=SRC == "io")
    print(f"{len(sq)} squares ({SRC})", flush=True)
    # pooled per class: a subsample of pixel-years per source; transitions: the change per source
    pool = {}
    trans = {}
    per_sq = {}
    rng = np.random.default_rng(0)
    for k, (name, v) in enumerate(sorted(sq.items())):
        cls = nlcd_for(name, tuple(v["rc"]))
        hd = head_of(name)
        lab = labels_of(name)
        src = {"head": hd} | lab
        ny = len(NYEARS)
        c = cls[:ny]
        # per class, every 7th pixel of every year
        sub = np.arange(rng.integers(7), c.shape[1], 7)
        for code in CLASSES:
            m = c[:, sub] == code
            if not m.any():
                continue
            d = pool.setdefault(code, {})
            for s, a in src.items():
                d.setdefault(s, []).append(a[:ny, sub][m])
        # class changes between consecutive years
        # every 7th pixel here too: "trees stayed" alone is millions of pixels a square
        for i in range(ny - 1):
            a0, a1 = c[i, sub], c[i + 1, sub]
            if SRC == "io":
                kinds = {"trees to rangeland/bare": np.isin(a0, list(FOREST)) & np.isin(a1, list(OPEN)),
                         "rangeland/bare to trees": np.isin(a0, list(OPEN)) & np.isin(a1, list(FOREST)),
                         "trees stayed": (a0 == 2) & (a1 == 2), "rangeland stayed": (a0 == 11) & (a1 == 11),
                         "bare stayed": (a0 == 8) & (a1 == 8)}
            else:
                kinds = {"forest to shrub/grass/barren": np.isin(a0, list(FOREST)) & np.isin(a1, list(OPEN)),
                         "shrub/grass/barren to forest": np.isin(a0, list(OPEN)) & np.isin(a1, list(FOREST)),
                         "forest stayed": np.isin(a0, list(FOREST)) & np.isin(a1, list(FOREST)),
                         "shrub stayed": (a0 == 52) & (a1 == 52), "grass stayed": (a0 == 71) & (a1 == 71)}
            for kind, m in kinds.items():
                if not m.any():
                    continue
                d = trans.setdefault(kind, {})
                for s, a in src.items():
                    d.setdefault(s, {"before": [], "after": []})
                    d[s]["before"].append(a[i, sub][m])
                    d[s]["after"].append(a[i + 1, sub][m])
        vals, cnt = np.unique(c[c != 250], return_counts=True)
        main_c = int(vals[cnt.argmax()]) if vals.size else None
        per_sq[name] = {"main_class": CLASSES.get(main_c, main_c), "share": round(float(cnt.max() / cnt.sum()), 2) if vals.size else None,
                        **{s: [med(a[i]) for i in range(len(YEARS))] for s, a in src.items()}}
        print(f"square {k + 1} of {len(sq)}: {name} main {per_sq[name]['main_class']} "
              f"head {med(hd)} ctrees {med(lab['ctrees'])}" + (f" rap {med(lab['rap'])}" if "rap" in lab else "")
              + (f" chaparral {med(lab['chaparral'])}" if "chaparral" in lab else "")
              + f" [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        del hd, lab, src, cls

    res = {"classes": {}, "transitions": {}, "squares": per_sq}
    print(f"\nPER {SRC.upper()} CLASS, median Mg/ha over pixel-years 2018 to 2024 (n px-years)", flush=True)
    print(f"{'class':20s} {'n':>9s} {'head':>7s} {'ctrees':>7s} {'rap':>7s} {'chap':>7s}", flush=True)
    for code, d in sorted(pool.items()):
        row = {s: med(np.concatenate(a)) for s, a in d.items()}
        n = int(sum(x.size for x in d["head"]))
        res["classes"][CLASSES[code]] = row | {"n": n}
        f = lambda s: f"{row.get(s):7.1f}" if row.get(s) is not None else f"{'':>7s}"  # noqa: E731
        print(f"{CLASSES[code]:20s} {n:9d} {f('head')} {f('ctrees')} {f('rap')} {f('chaparral')}", flush=True)
    print("\nCLASS CHANGES, one year to the next: median before -> after (median per-pixel change)", flush=True)
    for kind, d in trans.items():
        o = {}
        for s, ba in d.items():
            b, a = np.concatenate(ba["before"]), np.concatenate(ba["after"])
            ok = np.isfinite(b) & np.isfinite(a)
            if ok.sum() < 50:
                continue
            o[s] = {"n": int(ok.sum()), "before": round(float(np.median(b[ok])), 1), "after": round(float(np.median(a[ok])), 1),
                    "change": round(float(np.median(a[ok] - b[ok])), 1),
                    "change_pct": round(float(np.median(100 * (a[ok] - b[ok]) / np.maximum(b[ok], 1))), 1)}
        res["transitions"][kind] = o
        print(f"  {kind}:", flush=True)
        for s, e in o.items():
            print(f"     {s:10s} n {e['n']:8d}  {e['before']:6.1f} -> {e['after']:6.1f}  ({e['change']:+.1f}, {e['change_pct']:+.0f} %)", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(out_file, "w"), indent=1)
    print(f"\nwrote {out_file} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
