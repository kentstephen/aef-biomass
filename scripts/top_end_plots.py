"""The top end against plot biomass: GEO-TREES census plots and Rodda et al. 2024 lidar + plot maps.

Section 23 found the heads flatten against lidar height above about 250 Mg/ha, and that which map's level is right
there needs plot biomass. Two references, both independent of GEDI (CTrees' and AlphaEarth's link):
  GEO-TREES   tree-census subplots (mostly 50 x 50 m polygons, some plot points), RAINFOR, ForestGEO, AfriTRON, TmFO,
              IIASA, Fluxnet, TForces; the 105 "NASA GEDI cal/val" subplots dropped. Censuses 1999 to 2021, mostly
              before AlphaEarth: each is read in max(census, 2018) and dropped if Hansen shows any loss on it from
              the census year on (through that year). A point plot is read as the 1 ha around it.
  Rodda 2024  13 tropical sites (India, Thailand, Cameroon, Gabon), field plots scaled up with airborne or UAV lidar,
              100 m cells. Lidar years 2012 to 2023: each cell is the median of 2018 to 2022, and dropped if Hansen
              shows loss on any of its pixels 2012 to 2022.
Beside the map's head (mean of its 10 m pixels on the plot or cell): CTrees (100 m, the same year) and ESA CCI v7
(100 m, the same year, 2024 at most), each nearest at the pixel centers and averaged the same way.

Writes results/top_end_plots.json: per reference, by the reference's own biomass (bins of 100 Mg/ha), the median of
each map and its median ratio to the reference; rank and R2 of log overall; per Rodda site. RAM under 2 GB.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with icechunk --with xarray \
        --with requests --with pyproj python scripts/top_end_plots.py
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import rasterio
import shapely
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import aef_read as ar  # noqa: E402
import read_labels as rl  # noqa: E402  (CTrees reader)

OUT = HERE / "results" / "top_end_plots.json"
GT = HERE / "data" / "labels_new" / "geotrees" / "plots.geojson"
RODDA = HERE / "data" / "labels_new" / "rodda"
HANSEN = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_lossyear_{}.tif"
CCI = "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/{y}/{t}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-{y}-fv7.0.tif"
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
BINS = [0, 100, 200, 300, 400, 500, 2000]


def hansen_tile(lat, lon, fmt):
    top, left = int(math.ceil(lat / 10) * 10), int(math.floor(lon / 10) * 10)
    if fmt == "hansen":
        return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"
    return f"{'N' if top >= 0 else 'S'}{abs(top):02d}{'E' if left >= 0 else 'W'}{abs(left):03d}"


def sample_tiled(url_of, LON, LAT, fill, fmt):
    """Nearest value at each (LON, LAT) from 10 degree EPSG:4326 tiles; fill outside."""
    out = np.full(LON.shape, fill, np.float32)
    tiles = defaultdict(list)
    top = np.ceil(LAT.ravel() / 10).astype(int) * 10
    left = np.floor(LON.ravel() / 10).astype(int) * 10
    key = top * 1000 + left
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        tiles[hansen_tile(LAT.ravel()[idx[0]], LON.ravel()[idx[0]], fmt)].extend(idx.tolist())
    flat = out.ravel()
    with rasterio.Env(**ENV):
        for t, idx in tiles.items():
            idx = np.array(idx)
            x, y = LON.ravel()[idx], LAT.ravel()[idx]
            try:
                with rasterio.open("/vsicurl/" + url_of(t)) as s:
                    pad = 2 * abs(s.transform.a)
                    w = from_bounds(x.min() - pad, y.min() - pad, x.max() + pad, y.max() + pad, s.transform)
                    w = w.round_offsets().round_lengths()
                    a = s.read(1, window=w, boundless=True, fill_value=fill)
                    tr = s.window_transform(w)
            except Exception as e:  # noqa: BLE001
                print(f"    read failed {t}: {e!r}", flush=True)
                continue
            r = np.floor((y - tr.f) / tr.e).astype(int)
            c = np.floor((x - tr.c) / tr.a).astype(int)
            ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
            v = np.full(idx.size, fill, np.float32)
            v[ok] = a[r[ok], c[ok]]
            flat[idx] = v
    return flat.reshape(LON.shape)


def hansen_loss(LON, LAT):
    return sample_tiled(lambda t: HANSEN.format(t), LON, LAT, 0, "hansen").astype(np.int16)


def cci(LON, LAT, year):
    v = sample_tiled(lambda t: CCI.format(y=min(year, 2024), t=t), LON, LAT, 65535, "cci")
    return np.where(v == 65535, np.nan, v)


def ctrees(LON, LAT, years):
    """CTrees Mg/ha (len(years), n), NaN none."""
    try:
        d = rl.read_ctrees(LON, LAT)
    except Exception as e:  # noqa: BLE001
        print(f"    ctrees failed {e!r}", flush=True)
        return np.full((len(years), LON.size), np.nan, np.float32)
    a = d["agb"].astype(np.float32)
    a = np.where(a < 0, np.nan, a / 10)
    return a[[rl.YEARS.index(y) for y in years]]


def nanmean_by(v, g, ng):
    ok = np.isfinite(v)
    n = np.bincount(g[ok], minlength=ng)
    s = np.bincount(g[ok], weights=v[ok], minlength=ng)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


# ---------------------------------------------------------------- GEO-TREES
def geotrees():
    feats = [f for f in json.load(open(GT))["features"] if "GEDI" not in f["properties"]["Network"]
             and f["properties"].get("AGB") is not None]
    plots = defaultdict(list)
    for f in feats:
        plots[f["properties"]["Plot_ID"]].append(f)
    t0 = time.time()

    def one(item):
        pid, fs = item
        rows = []
        geoms = []
        for f in fs:
            g = shapely.geometry.shape(f["geometry"])
            if g.geom_type == "Point":  # the 1 ha around a plot point
                d = 50 / 111320
                g = shapely.box(g.x - d / math.cos(math.radians(g.y)), g.y - d, g.x + d / math.cos(math.radians(g.y)), g.y + d)
            geoms.append(shapely.make_valid(g))
        bs = np.array([g.bounds for g in geoms])
        b = (bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max())
        rc = ar.rc_of(b[0] - 2e-4, b[1] - 2e-4, b[2] + 2e-4, b[3] + 2e-4)
        lon, lat = ar.centers(rc)
        LON, LAT = np.meshgrid(lon, lat)
        loss = hansen_loss(LON, LAT).ravel()
        by_year = defaultdict(list)
        for f, g in zip(fs, geoms):
            p = f["properties"]
            Y = min(max(int(p["CensusYear"]), 2018), 2025)
            inside = shapely.contains_xy(g, LON.ravel(), LAT.ravel())
            if inside.sum() == 0:
                continue
            lost = (loss[inside] >= int(p["CensusYear"]) - 2000 + 1) & (loss[inside] <= Y - 2000)
            by_year[Y].append((f, inside, bool(lost.any())))
        for Y, items in by_year.items():
            agb, _, _ = ar.head(ar.raw(rc, Y))
            ct = ctrees(LON, LAT, [Y])[0]
            cc = cci(LON, LAT, Y).ravel()
            for f, inside, lost in items:
                p = f["properties"]
                rows.append({"plot": pid, "sub": p["SubPlot_Id"], "network": p["Network"].strip(), "census": int(p["CensusYear"]),
                             "year": Y, "lost": lost, "px": int(inside.sum()), "ref": float(p["AGB"]),
                             "map": float(np.nanmean(agb[inside])) if np.isfinite(agb[inside]).any() else None,
                             "ctrees": float(np.nanmean(ct[inside])) if np.isfinite(ct[inside]).any() else None,
                             "cci": float(np.nanmean(cc[inside])) if np.isfinite(cc[inside]).any() else None,
                             "lon": float(LON.ravel()[inside].mean()), "lat": float(LAT.ravel()[inside].mean())})
        return rows

    from concurrent.futures import ThreadPoolExecutor
    ar.mosaic()
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for pi, got in enumerate(ex.map(one, sorted(plots.items()))):
            rows += got
            if pi % 20 == 0:
                print(f"  geotrees plot {pi + 1}/{len(plots)}: {len(rows)} rows [{time.time() - t0:.0f} s, "
                      f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    return rows


# ---------------------------------------------------------------- Rodda
def rodda_site(path):
    with rasterio.open(path) as s:
        A = s.read().astype(np.float32)
        nd = s.nodata
        crs, tr = s.crs, s.transform
        bnd = transform_bounds(crs, "EPSG:4326", *s.bounds)
    mean, sd = A[0], A[1]
    valid = np.isfinite(mean) & (mean >= 0) & ((mean != nd) if nd is not None else True)
    H, W = mean.shape
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rc = ar.rc_of(*bnd)
    years = list(range(2018, 2023))
    ncell = H * W
    acc = {k: np.zeros((len(years), ncell), np.float64) for k in ("map", "ctrees", "cci")}
    cnt = {k: np.zeros((len(years), ncell), np.float64) for k in acc}
    lost = np.zeros(ncell, bool)
    npx = np.zeros(ncell, np.int64)
    B = 768
    y0, y1, x0, x1 = rc
    for by in range(y0, y1, B):
        for bx in range(x0, x1, B):
            sub = (by, min(by + B, y1), bx, min(bx + B, x1))
            lon, lat = ar.centers(sub)
            LON, LAT = np.meshgrid(lon, lat)
            X, Y = to_utm.transform(LON.ravel(), LAT.ravel())
            cc = np.floor((np.asarray(X) - tr.c) / tr.a).astype(int)
            rr = np.floor((np.asarray(Y) - tr.f) / tr.e).astype(int)
            ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            cell = np.where(ins, rr * W + cc, 0)
            ins &= valid.ravel()[cell]
            if not ins.any():
                continue
            g = cell[ins]
            npx += np.bincount(g, minlength=ncell)
            ls = hansen_loss(LON, LAT).ravel()[ins]
            lost |= np.bincount(g, weights=((ls >= 12) & (ls <= 22)).astype(float), minlength=ncell) > 0
            ct = ctrees(LON, LAT, years)
            for k, Yr in enumerate(years):
                agb, _, _ = ar.head(ar.raw(sub, Yr))
                cv = cci(LON, LAT, Yr).ravel()
                for key, v in (("map", agb[ins]), ("ctrees", ct[k][ins]), ("cci", cv[ins])):
                    ok = np.isfinite(v)
                    acc[key][k] += np.bincount(g[ok], weights=v[ok], minlength=ncell)
                    cnt[key][k] += np.bincount(g[ok], minlength=ncell)
    keep = valid.ravel() & (npx >= 50) & ~lost
    out = {"ref": mean.ravel()[keep], "ref_sd": sd.ravel()[keep], "lost cells": int((valid.ravel() & lost).sum())}
    for key in acc:
        m = np.where(cnt[key] > 0, acc[key] / np.maximum(cnt[key], 1), np.nan)
        out[key] = np.nanmedian(m[:, keep], 0) if keep.any() else np.array([])
    return out


# ---------------------------------------------------------------- scores
def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def r2log(ref, p):
    m = np.isfinite(ref) & np.isfinite(p) & (ref >= 0) & (p >= 0)
    if m.sum() < 10:
        return None
    t, q = np.log1p(ref[m]), np.log1p(p[m])
    return round(float(1 - ((t - q) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 3)


def score(ref, P):
    o = {"n": int(np.isfinite(ref).sum()), "median ref": round(float(np.nanmedian(ref)), 1)}
    for k, p in P.items():
        o[k] = {"rank": spear(ref, p), "r2 log": r2log(ref, p), "median": round(float(np.nanmedian(p)), 1)
                if np.isfinite(p).any() else None}
    bins = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ref >= lo) & (ref < hi)
        if m.sum() < 5:
            continue
        b = {"n": int(m.sum()), "ref median": round(float(np.median(ref[m])), 1)}
        for k, p in P.items():
            v = p[m]
            ok = np.isfinite(v)
            if ok.sum() >= 5:
                b[k] = round(float(np.median(v[ok])), 1)
                b[k + " / ref"] = round(float(np.median(v[ok] / np.maximum(ref[m][ok], 1))), 3)
        bins[f"{lo}-{hi}"] = b
    o["by reference biomass"] = bins
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    ar.guard(args.max_gb)
    t0 = time.time()
    res = {"note": __doc__.split("\n\n")[0]}
    if "rodda" not in args.skip:
        sites, pool = {}, defaultdict(list)
        for f in sorted(RODDA.glob("*_AGB100.tif")):
            name = f.stem.replace("_AGB100", "")
            S = rodda_site(f)
            P = {k: S[k] for k in ("map", "ctrees", "cci")}
            sites[name] = score(S["ref"], P) | {"lost cells dropped": S["lost cells"]}
            for k in ("ref", "map", "ctrees", "cci"):
                pool[k].append(S[k])
            print(f"rodda {name}: {json.dumps({k: v for k, v in sites[name].items() if k != 'by reference biomass'})} "
                  f"[{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        P = {k: np.concatenate(pool[k]) for k in pool}
        res["rodda"] = {"pooled": score(P["ref"], {k: P[k] for k in ("map", "ctrees", "cci")}), "sites": sites}
        print("RODDA POOLED", json.dumps(res["rodda"]["pooled"]), flush=True)
    if "geotrees" not in args.skip:
        rows = geotrees()
        kept = [r for r in rows if not r["lost"]]
        ref = np.array([r["ref"] for r in kept])
        P = {k: np.array([np.nan if r[k] is None else r[k] for r in kept]) for k in ("map", "ctrees", "cci")}
        g = {"subplots read": len(rows), "dropped for Hansen loss since census": sum(r["lost"] for r in rows),
             "all": score(ref, P)}
        recent = np.array([r["census"] >= 2015 for r in kept])
        g["census 2015 on"] = score(ref[recent], {k: v[recent] for k, v in P.items()})
        byn = {}
        for net in sorted({r["network"] for r in kept}):
            m = np.array([r["network"] == net for r in kept])
            if m.sum() >= 10:
                byn[net] = score(ref[m], {k: v[m] for k, v in P.items()})
                byn[net].pop("by reference biomass")
        g["by network"] = byn
        res["geotrees"] = g
        json.dump(rows, open(HERE / "data" / "labels_new" / "geotrees" / "rows.json", "w"))
        print("GEOTREES", json.dumps({k: v for k, v in g.items() if k != "by network"}), flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    prev = json.load(open(OUT)) if OUT.exists() and args.skip else {}
    json.dump(prev | res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
