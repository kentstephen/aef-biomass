"""Tundra biomass against Orndahl et al. 30 m Arctic AGB (2020) and the Berner et al. 2024 harvest plots.

Section 28 found Esri files tundra as rangeland, and nothing on disk checked the heads' or CTrees' tundra values.
  Orndahl  30 m plant and woody-plant AGB for 2020 (Arctic Data Center doi:10.18739/A2NS0M06B), a Landsat + climate +
           terrain model taught on harvest plots, most of them from the Berner synthesis; its plant AGB is vascular
           plants (herbs, shrubs, trees), not mosses or lichens. Median (p500) and its 95 % range (p025, p975).
  Berner   the synthesis plots from 2017 on (doi:10.18739/A2QJ78081), vascular (herb + shrub + tree) and woody
           (shrub + tree) per plot, the median per site and year. Field harvest, but not independent of Orndahl:
           its datasets taught Orndahl's model (flagged per site).
Against both: the map's head (2020 for Orndahl; the plot's year, 2018 at the earliest, for Berner), CTrees (100 m)
and ESA CCI v7 (100 m) the same year.

Orndahl part: 11 tundra squares (600 x 600 AlphaEarth pixels; the three of veg_squares.json and eight new ones across
the Arctic), compared on 100 m blocks (10 x 10 pixels, Orndahl nearest at each pixel center). Berner part: each
site's AlphaEarth is the 3 x 3 pixels around its mean coordinate. Writes results/tundra_check.json. RAM under 2 GB.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with icechunk --with xarray \
        --with requests --with pyproj --with tifffile --with imagecodecs python scripts/tundra_check.py
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
from pyproj import Transformer

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import aef_read as ar  # noqa: E402
import orndahl_read as orn  # noqa: E402
import top_end_plots as tp  # noqa: E402  (CTrees, CCI samplers and scores)

OUT = HERE / "results" / "tundra_check.json"
BERNER = HERE / "data" / "labels_new" / "berner" / "berner_v1_2.csv"
ORN_TRAIN = HERE / "data" / "labels_new" / "orndahl" / "plant_agb_train_data.csv"
TO = Transformer.from_crs("EPSG:4326", "EPSG:3571", always_xy=True)
SQUARES = {
    "yamal-ru": (70.3118, 67.4062, "veg_squares.json"),
    "noatak-us": (-160.7018, 67.6128, "veg_squares.json"),
    "nunavik-ca": (-71.8872, 59.7222, "veg_squares.json"),
    "taymyr-ru": (98.0, 73.0, "Taymyr lowland tundra"),
    "kytalyk-ru": (147.5, 70.8, "Indigirka lowland shrub tundra (Kytalyk)"),
    "chukotka-ru": (174.0, 65.3, "Anadyr lowland, Chukotka"),
    "kangerlussuaq-gl": (-50.5, 67.1, "west Greenland dry shrub tundra"),
    "banks-island-ca": (-119.6, 73.2, "Banks Island, high Arctic"),
    "daring-lake-ca": (-111.6, 64.87, "Low Arctic shrub tundra, Northwest Territories"),
    "toolik-us": (-149.6, 68.63, "Alaska North Slope foothills (Toolik)"),
    "finnmark-no": (25.5, 69.6, "Finnmark plateau heath and birch"),
}
HALF = 300


def blocks(v, B=10):
    w = v.reshape(600 // B, B, 600 // B, B)
    ok = np.isfinite(w)
    cnt = ok.sum((1, 3))
    return np.where(cnt >= 0.8 * B * B, np.nansum(w, (1, 3)) / np.maximum(cnt, 1), np.nan).ravel()


def square(name, lon, lat):
    cx, cy = int((lon - ar.X0) / ar.RES), int((ar.Y0 - lat) / ar.RES)
    rc = (cy - HALF, cy + HALF, cx - HALF, cx + HALF)
    lo, la = ar.centers(rc)
    LON, LAT = np.meshgrid(lo, la)
    X, Y = TO.transform(LON, LAT)
    X, Y = np.asarray(X), np.asarray(Y)
    agb, hgt, cls = ar.head(ar.raw(rc, 2020))
    d = {"map": agb, "ctrees": tp.ctrees(LON, LAT, [2020])[0], "cci": tp.cci(LON, LAT, 2020).ravel()}
    for lay, key in (("plant_agb_2020_p500", "orndahl"), ("plant_agb_2020_p025", "orndahl lo"),
                     ("plant_agb_2020_p975", "orndahl hi"), ("woodyplant_agb_2020_p500", "orndahl woody")):
        d[key] = orn.sample(lay, X, Y).ravel() / 100.0  # g/m2 -> Mg/ha
    cl = np.bincount(cls[cls >= 0], minlength=len(ar.CLASSES))
    return {k: blocks(v.reshape(600, 600)) for k, v in d.items()}, {ar.CLASSES[i]: round(float(c / max(cl.sum(), 1)), 3)
                                                                       for i, c in enumerate(cl) if c}


def berner_sites():
    rows = list(csv.DictReader(open(BERNER, encoding="latin-1")))
    train = {r["dataset_id"] for r in csv.DictReader(open(ORN_TRAIN, encoding="utf-8-sig"))}
    plots = defaultdict(dict)
    meta = {}
    for r in rows:
        if int(r["year"]) < 2017 or r["vascular_measured"] != "TRUE":
            continue
        k = (r["site_code"], int(r["year"]), r["plot_code"])
        try:
            v = float(r["biomass_density_gm2"])
        except ValueError:
            v = np.nan
        plots[k][r["pft"]] = v
        meta[k] = (float(r["longitude"]), float(r["latitude"]), r["dataset_id"], r["country"], r["coord_type"])
    sites = defaultdict(lambda: {"vasc": [], "woody": [], "lon": [], "lat": []})
    for k, p in plots.items():
        vals = [p.get(a) for a in ("herb", "shrub", "tree")]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        s = sites[(k[0], k[1])]
        s["vasc"].append(sum(vals)), s["woody"].append(vals[1] + vals[2])
        s["lon"].append(meta[k][0]), s["lat"].append(meta[k][1])
        s["ds"], s["country"] = meta[k][2], meta[k][3]
    out = []
    for (code, yr), s in sites.items():
        out.append({"site": code, "year": yr, "plots": len(s["vasc"]), "vasc": float(np.median(s["vasc"])) / 100,
                    "woody": float(np.median(s["woody"])) / 100, "lon": float(np.mean(s["lon"])), "lat": float(np.mean(s["lat"])),
                    "dataset": s["ds"], "country": s["country"], "in orndahl training": s["ds"] in train})
    return out


def berner_one(s):
    Y = min(max(s["year"], 2018), 2025)
    cx, cy = int((s["lon"] - ar.X0) / ar.RES), int((ar.Y0 - s["lat"]) / ar.RES)
    agb, _, _ = ar.head(ar.raw((cy - 1, cy + 2, cx - 1, cx + 2), Y))
    s["map"] = float(np.nanmean(agb)) if np.isfinite(agb).any() else None
    LON, LAT = np.array([[s["lon"]]]), np.array([[s["lat"]]])
    s["ctrees"] = float(tp.ctrees(LON, LAT, [Y])[0][0])
    s["cci"] = float(tp.cci(LON, LAT, Y).ravel()[0])
    X, Yp = TO.transform(LON, LAT)
    X, Yp = np.asarray(X, float).reshape(1, 1), np.asarray(Yp, float).reshape(1, 1)
    s["orndahl"] = float(orn.sample("plant_agb_2020_p500", X, Yp).ravel()[0]) / 100
    s["orndahl woody"] = float(orn.sample("woodyplant_agb_2020_p500", X, Yp).ravel()[0]) / 100
    return s


def berner(rows, workers=8):
    """Every site, several at a time (each is a few small remote reads)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    for name in ("plant_agb_2020_p500", "woodyplant_agb_2020_p500"):  # the TIFF tags once, before the threads
        for t in sorted({orn.tile_of(*TO.transform(s["lon"], s["lat"])) for s in rows}):
            orn.tags(f"{name}_{t}.tif")
    ar.mosaic()
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(berner_one, rows)):
            out.append(s)
            if i % 50 == 0:
                print(f"  berner site {i + 1}/{len(rows)} [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    return out


def sc(ref, P, bins):
    o = {"n": int(np.isfinite(ref).sum()), "ref median": round(float(np.nanmedian(ref)), 3)}
    for k, p in P.items():
        m = np.isfinite(ref) & np.isfinite(p)
        o[k] = {"rank": tp.spear(ref, p), "r2 log": tp.r2log(ref, p),
                "median": round(float(np.median(p[m])), 3) if m.any() else None,
                "median ratio": round(float(np.median((p[m] + 0.1) / (ref[m] + 0.1))), 3) if m.any() else None}
    bo = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ref >= lo) & (ref < hi)
        if m.sum() < 5:
            continue
        b = {"n": int(m.sum()), "ref median": round(float(np.median(ref[m])), 3)}
        for k, p in P.items():
            v = p[m]
            if np.isfinite(v).sum() >= 5:
                b[k] = round(float(np.nanmedian(v)), 3)
        bo[f"{lo}-{hi}"] = b
    o["by reference"] = bo
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    ar.guard(args.max_gb)
    t0 = time.time()
    res = {"note": __doc__.split("\n\n")[0]}
    bins = [0, 1, 2, 5, 10, 20, 1000]
    if "squares" not in args.skip:
        per, pool = {}, defaultdict(list)
        for name, (lon, lat, note) in SQUARES.items():
            B, cls = square(name, lon, lat)
            ref = B["orndahl"]
            P = {k: B[k] for k in ("map", "ctrees", "cci", "orndahl woody")}
            o = sc(ref, P, bins) | {"center": [lon, lat], "note": note, "head classes 2020": cls}
            m = np.isfinite(ref) & np.isfinite(B["map"])
            o["map inside Orndahl's 95 % range"] = round(float(((B["map"][m] >= B["orndahl lo"][m]) & (B["map"][m] <= B["orndahl hi"][m])).mean()), 3) if m.any() else None
            o["ctrees inside Orndahl's 95 % range"] = round(float(((B["ctrees"][m] >= B["orndahl lo"][m]) & (B["ctrees"][m] <= B["orndahl hi"][m])).mean()), 3) if m.any() else None
            per[name] = o
            for k, v in B.items():
                pool[k].append(v)
            pool["sq"].append(np.full(v.size, len(per) - 1))
            print(f"{name}: {json.dumps({k: v for k, v in o.items() if k != 'by reference'})} [{time.time() - t0:.0f} s, "
                  f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        P = {k: np.concatenate(v) for k, v in pool.items()}
        meds = {k: np.array([np.nanmedian(v) for v in pool[k]]) for k in ("orndahl", "map", "ctrees", "cci")}
        res["squares"] = {"pooled blocks": sc(P["orndahl"], {k: P[k] for k in ("map", "ctrees", "cci", "orndahl woody")}, bins),
                          "rank of square medians against Orndahl": {k: tp.spear(meds["orndahl"], meds[k]) for k in ("map", "ctrees", "cci")},
                          "rank within a square, median over squares": {k: float(np.median([per[n][k]["rank"] for n in per if per[n][k]["rank"] is not None]))
                                                                         for k in ("map", "ctrees", "cci")},
                          "per square": per}
        print("SQUARES", json.dumps({k: v for k, v in res["squares"].items() if k != "per square"}), flush=True)
    if "berner" not in args.skip:
        rows = berner(berner_sites())
        json.dump(rows, open(HERE / "data" / "labels_new" / "berner" / "sites.json", "w"))
        g = {}
        for grp, sel in (("all sites", lambda s: True), ("not in Orndahl's training", lambda s: not s["in orndahl training"])):
            R = [s for s in rows if sel(s)]
            ref = np.array([s["vasc"] for s in R])
            P = {k: np.array([np.nan if s.get(k) is None else s[k] for s in R], float) for k in ("map", "ctrees", "cci", "orndahl")}
            g[grp] = {"vascular": sc(ref, P, bins)}
            refw = np.array([s["woody"] for s in R])
            Pw = {k: np.array([np.nan if s.get(k) is None else s[k] for s in R], float) for k in ("map", "ctrees", "cci", "orndahl woody")}
            g[grp]["woody"] = sc(refw, Pw, bins)
        g["sites"] = len(rows)
        g["by country"] = {c: sum(s["country"] == c for s in rows) for c in sorted({s["country"] for s in rows})}
        res["berner"] = g
        print("BERNER", json.dumps(g), flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    prev = json.load(open(OUT)) if OUT.exists() and args.skip else {}
    json.dump(prev | res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
