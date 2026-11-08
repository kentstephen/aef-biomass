# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio", "psutil"]
# ///
"""Rangeland Analysis Platform v3 under every square inside the lower 48, 2018 to 2025, on AlphaEarth's grid.

RAP (Univ. of Montana, Allred et al.): yearly 30 m maps from Landsat, taught on BLM AIM and NRCS NRI field plots,
not on GEDI or CTrees. Two products, both public COGs (EPSG:4326, 30 m):
  vegetation-biomass-v3-<year>.tif   band 1 annual forb and grass, band 2 perennial forb and grass:
                                     aboveground herbaceous biomass, lbs/acre (x 0.00112085 = Mg/ha), 65535 none
  vegetation-cover-v3-<year>.tif     % cover: annual forb and grass, bare ground, litter, perennial forb and grass,
                                     shrub, tree; 255 none
Herbaceous biomass only: shrub and tree mass are not in it (their cover is).

Per square, data/rap/<name>.npz (n = 600 x 600, row-major as data/squares/<name>.npz):
  afg, pfg  uint16 (8, n)     lbs/acre, 65535 none
  cover     uint8 (8, 6, n)   afg, bgr, ltr, pfg, shr, tre %, 255 none
  rcell     int32 (n,)        the RAP 30 m pixel each 10 m pixel falls in (for averaging)

Squares: data/squares.json, veg_squares.json, us_squares.json, lidar_squares.json, those inside RAP's bounds.
About 10 s a square; peak under 1 GB.

    uv run scripts/read_rap.py [--workers 4]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "rap"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
YEARS = list(range(2018, 2026))
BASE = "https://rangeland.ntsg.umt.edu/data/rap"
BIO = BASE + "/rap-vegetation-biomass/v3/vegetation-biomass-v3-{}.tif"
COV = BASE + "/rap-vegetation-cover/v3/vegetation-cover-v3-{}.tif"
BOUNDS = (-124.77, 24.52, -66.87, 49.39)
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3")


def grid(rc):
    y0, y1, x0, x1 = rc
    lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
    lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
    return lon, lat


def read(url, lon, lat, fill):
    """All bands at each pixel center (nearest), and the source pixel id."""
    with rasterio.Env(**ENV), rasterio.open("/vsicurl/" + url) as s:
        pad = 3 * abs(s.transform.a)
        w = from_bounds(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad, s.transform)
        a = s.read(window=w, boundless=True, fill_value=fill)
        tr = s.window_transform(w)
    r = np.floor((lat - tr.f) / tr.e).astype(int)
    c = np.floor((lon - tr.c) / tr.a).astype(int)
    R, C = np.meshgrid(r, c, indexing="ij")
    return a[:, R, C].reshape(a.shape[0], -1), (R * a.shape[2] + C).ravel().astype(np.int32)


def one(name, sq):
    t = time.time()
    lon, lat = grid(sq["rc"])
    n = lon.size * lat.size
    afg, pfg = np.full((8, n), 65535, np.uint16), np.full((8, n), 65535, np.uint16)
    cov = np.full((8, 6, n), 255, np.uint8)
    rcell = None
    for k, y in enumerate(YEARS):
        for attempt in range(4):
            try:
                b, rcell = read(BIO.format(y), lon, lat, 65535)
                c, _ = read(COV.format(y), lon, lat, 255)
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    raise
                time.sleep(5 + 10 * attempt)
        afg[k], pfg[k], cov[k] = b[0], b[1], c
    tmp = OUT / f"{name}.tmp.npz"
    np.savez_compressed(tmp, rc=np.array(sq["rc"]), years=np.array(YEARS), afg=afg, pfg=pfg, cover=cov, rcell=rcell)
    os.replace(tmp, OUT / f"{name}.npz")
    ok = (afg != 65535) & (pfg != 65535)
    herb = np.where(ok, afg.astype(np.float32) + pfg, np.nan) * 0.00112085
    by_year = " ".join(f"{np.nanmedian(h):.2f}" if np.isfinite(h).any() else "-" for h in herb)
    return f"{name:22s} {time.time() - t:4.0f} s  valid {ok.mean():.2f}  herb Mg/ha median by year {by_year}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sq = {}
    for f in ("squares.json", "veg_squares.json", "us_squares.json", "lidar_squares.json"):
        if (HERE / "data" / f).exists():
            sq.update(json.load(open(HERE / "data" / f)))
    W, S, E, N = BOUNDS
    todo = [k for k, v in sq.items() if W < v["center"][0] < E and S < v["center"][1] < N and not (OUT / f"{k}.npz").exists()]
    print(f"{len(todo)} squares inside RAP: {todo}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, k, sq[k]): k for k in todo}
        for i, f in enumerate(futs):
            try:
                print(f"{i + 1}/{len(todo)} {f.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{i + 1}/{len(todo)} {futs[f]} FAILED: {e!r}", flush=True)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
