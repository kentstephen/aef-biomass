# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio", "icechunk", "xarray", "zarr>=3", "requests", "psutil"]
# ///
"""World labels under every pixel of every square (data/squares.json and data/veg_squares.json), 2018 to 2025.

Per square, data/labels/<name>.npz (n = 600 x 600 AlphaEarth pixels, row-major as data/squares/<name>.npz):
  agb, agb_err  int16 (8, n)   CTrees aboveground biomass and its residual standard error, Mg/ha x 10, -9999 none
                               (100 m: each pixel carries its cell's value); cell int32 (n,) the cell's id
  height        uint8 (n,)     Meta CHM v2 10 m mean canopy height, m, 255 none (water is a valid 0; above 70 N the
                               source reads a false ~10 m), height_date int32 (n,) imagery date yyyymmdd, 0 unknown
  io            uint8 (6, n)   Impact Observatory annual land use v02, 2018 to 2023 (1 water, 2 trees, 4 flooded veg,
                               5 crops, 7 built, 8 bare, 9 snow, 10 clouds, 11 rangeland), 0 none
  bii           uint16 (8, n)  Biodiversity intactness v1.1 x 10000, 0 none (the source writes 0 on water)

All public, no sign-in. One square takes about 40 s; four at a time; peak well under 2 GB. Resumable.

    uv run scripts/read_labels.py [--workers 4] [--max-gb 6] [--picks squares.json veg_squares.json]
"""
import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import rasterio
import requests
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_xy
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "labels"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
YEARS = list(range(2018, 2026))
IO_YEARS = list(range(2018, 2024))
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3")
META = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global"
BII = "https://data.source.coop/vizzuality/biodiversity-intactness-100m-v1-1/bii_{}.tif"
PC = "https://planetarycomputer.microsoft.com/api"
_ct = threading.local()


def grid(rc):
    y0, y1, x0, x1 = rc
    lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
    lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT


def sample(url, LON, LAT, fill):
    """Nearest value of a raster at each pixel center; `fill` outside it."""
    with rasterio.Env(**ENV), rasterio.open(url) as s:
        if s.crs.to_epsg() == 4326:
            X, Y = LON, LAT
        else:
            X, Y = warp_xy("EPSG:4326", s.crs, LON.ravel(), LAT.ravel())
            X, Y = np.asarray(X).reshape(LON.shape), np.asarray(Y).reshape(LON.shape)
        pad = 3 * abs(s.transform.a)
        w = from_bounds(X.min() - pad, Y.min() - pad, X.max() + pad, Y.max() + pad, s.transform)
        a = s.read(1, window=w, boundless=True, fill_value=fill)
        tr = s.window_transform(w)
    r = np.floor((Y - tr.f) / tr.e).astype(int)
    c = np.floor((X - tr.c) / tr.a).astype(int)
    inside = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.full(LON.shape, fill, a.dtype)
    out[inside] = a[r[inside], c[inside]]
    return out


def ctrees():
    if not hasattr(_ct, "ds"):
        import icechunk
        import xarray as xr
        st = icechunk.s3_storage(bucket="ctrees-agb-100m-global", prefix="agb_100m_global", region="us-west-2", anonymous=True)
        repo = icechunk.Repository.open(st)
        _ct.ds = xr.open_zarr(repo.readonly_session(branch="main").store, zarr_format=3, group="aboveground_biomass",
                              chunks=None, mask_and_scale=False)
    return _ct.ds


def read_ctrees(LON, LAT):
    ds = ctrees()
    res = float(ds.x[1] - ds.x[0])
    x0, y0 = float(ds.x[0]) - res / 2, float(ds.y[0]) + res / 2
    ci = np.floor((LON - x0) / res).astype(int)
    ri = np.floor((y0 - LAT) / res).astype(int)
    r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
    yrs = [int(y) for y in ds.time.dt.year.values]
    ti = [yrs.index(y) for y in YEARS]
    out = {}
    for v, k in (("agb", "agb"), ("uncertainty", "agb_err")):
        for attempt in range(4):
            try:
                A = np.asarray(ds[v].isel(time=ti, y=slice(r0, r1), x=slice(c0, c1)).values)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + 5 * attempt)
        out[k] = A[:, ri - r0, ci - c0].reshape(len(YEARS), -1).astype(np.int16)
    out["cell"] = ((ri - r0) * (c1 - c0) + (ci - c0)).ravel().astype(np.int32)
    return out


def quadkey(lon, lat, z):
    x = int((lon + 180) / 360 * 2**z)
    s = math.sin(math.radians(max(min(lat, 85.05), -85.05)))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2**z)
    return "".join(str((1 if x & (1 << (i - 1)) else 0) + (2 if y & (1 << (i - 1)) else 0)) for i in range(z, 0, -1))


def corners(LON, LAT):
    return [(float(LON[i, j]), float(LAT[i, j])) for i in (0, -1) for j in (0, -1)]


def read_meta(LON, LAT):
    h = np.full(LON.shape, 255, np.uint8)
    for q in sorted({quadkey(lo, la, 7) for lo, la in corners(LON, LAT)}):
        try:
            v = sample(f"/vsicurl/{META}/dinov3_global_chm_v2_ml3_10m_mean/chm/{q}.tif", LON, LAT, 255)
        except rasterio.errors.RasterioIOError:
            continue  # no tile: no source imagery there
        h = np.where(v != 255, v, h)
    date = np.zeros(LON.shape, np.int32)
    tr = from_origin(LON[0, 0] - AEF_RES / 2, LAT[0, 0] + AEF_RES / 2, AEF_RES, AEF_RES)
    for q in sorted({quadkey(lo, la, 10) for lo, la in corners(LON, LAT)}):
        r = requests.get(f"{META}/dinov3_global_chm_v2_ml3/metadata/{q}.geojson", timeout=120)
        if not r.ok:
            continue
        shapes = [(f["geometry"], int(f["properties"]["acq_date"].replace("-", "")))
                  for f in r.json()["features"] if f.get("properties", {}).get("acq_date")]
        if shapes:
            d = features.rasterize(shapes, out_shape=LON.shape, transform=tr, fill=0, dtype="int32")
            date = np.where(d > 0, d, date)
    return {"height": h.ravel(), "height_date": date.ravel()}


def read_io(LON, LAT):
    W, S, E, N = float(LON.min()), float(LAT.min()), float(LON.max()), float(LAT.max())
    tok = requests.get(f"{PC}/sas/v1/token/io-lulc-annual-v02", timeout=60).json()["token"]
    items = requests.post(f"{PC}/stac/v1/search", timeout=120, json={
        "collections": ["io-lulc-annual-v02"], "bbox": [W, S, E, N], "limit": 100}).json()["features"]
    out = np.zeros((len(IO_YEARS), LON.size), np.uint8)
    for it in items:
        y = int(it["properties"]["start_datetime"][:4])
        if y not in IO_YEARS:
            continue
        v = sample("/vsicurl/" + it["assets"]["data"]["href"] + "?" + tok, LON, LAT, 0).ravel()
        k = IO_YEARS.index(y)
        out[k] = np.where(out[k] == 0, v, out[k])
    return {"io": out}


def read_bii(LON, LAT):
    out = np.zeros((len(YEARS), LON.size), np.uint16)
    for k, y in enumerate(YEARS):
        v = sample("/vsicurl/" + BII.format(y), LON, LAT, 0).ravel()
        v = np.nan_to_num(v, nan=0.0)
        out[k] = np.clip(np.round(v * 10000), 0, 10000).astype(np.uint16)
    return {"bii": out}


def one(name, sq):
    t = time.time()
    LON, LAT = grid(sq["rc"])
    d, secs = {}, {}
    for k, f in (("ctrees", read_ctrees), ("meta", read_meta), ("io", read_io), ("bii", read_bii)):
        t1 = time.time()
        d.update(f(LON, LAT))
        secs[k] = round(time.time() - t1, 1)
    tmp = OUT / f"{name}.tmp.npz"
    np.savez_compressed(tmp, rc=np.array(sq["rc"]), years=np.array(YEARS), io_years=np.array(IO_YEARS), **d)
    os.replace(tmp, OUT / f"{name}.npz")
    agb = d["agb"][d["agb"] >= 0]
    return (f"{name:22s} {time.time() - t:4.0f} s {secs}  agb none {(d['agb'] < 0).mean():.2f} med "
            f"{np.median(agb) / 10 if agb.size else float('nan'):6.1f}  height none {(d['height'] == 255).mean():.2f}  "
            f"dated {(d['height_date'] > 0).mean():.2f}  io none {(d['io'] == 0).mean():.2f}  bii none {(d['bii'] == 0).mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--picks", nargs="*", default=["squares.json", "veg_squares.json"], help="picks files in data/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
    sq = {}
    for f in args.picks:
        sq.update(json.load(open(HERE / "data" / f)))
    todo = [n for n in sq if not (OUT / f"{n}.npz").exists() and (not args.only or n in args.only)]
    print(f"{len(todo)} squares to read", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, n, sq[n]): n for n in todo}
        for k, f in enumerate(futs):
            try:
                print(f"{k + 1}/{len(todo)} {f.result()}  [peak {peak[0]:.1f} GB]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{k + 1}/{len(todo)} {futs[f]} FAILED: {e!r}", flush=True)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
