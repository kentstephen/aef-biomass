# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio", "psutil"]
# ///
"""ESA CCI Biomass v7.0 under every square: a second biomass map, made from radar (ALOS-2 PALSAR-2, Sentinel-1)
and calibrated with ICESat-2 rather than GEDI, so independent of CTrees' GEDI link and of AlphaEarth's GEDI target.

Source: https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/{year}/ (10 x 10 degree GeoTIFFs,
100 m, uint16 Mg/ha, 0 = no biomass; AGB and its standard deviation; years 2015 to 2024 here 2018 to 2024). Free
and open, citation required (DOI 10.5285/6429d1aafe1e43b9b414e4a5a7f8b903). Its biomass is the woody parts of
living trees only, so grass and crops read 0 by definition. The files are stripped, not tiled: a 6 km window is
one range request per row, a few seconds a layer.

Per square, data/cci/<name>.npz: agb, sd uint16 (7, n) at each AlphaEarth pixel's center, 2018 to 2024
(65535 = outside the files). n = 600 x 600, row-major as data/squares/<name>.npz.

    uv run scripts/read_cci.py [--picks squares.json veg_squares.json lidar_squares.json] [--workers 4]
"""
import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "cci"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
YEARS = list(range(2018, 2025))
URL = "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/{y}/{t}_ESACCI-BIOMASS-L4-{L}-MERGED-100m-{y}-fv7.0.tif"
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")


def tile(lat, lon):
    la, lo = math.ceil(lat / 10) * 10, math.floor(lon / 10) * 10
    if la == lat and lat != 90:  # a pixel exactly on a tile's top edge belongs to the tile above
        la += 10
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def one(name, s):
    t0 = time.time()
    y0, y1, x0, x1 = s["rc"]
    lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
    lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
    LON, LAT = np.meshgrid(lon, lat)
    tiles = np.array([[tile(a, b) for b in (lon[0], lon[-1])] for a in (lat[0], lat[-1])]).ravel()
    out = {L: np.full((len(YEARS), LAT.size), 65535, np.uint16) for L in ("AGB", "AGB_SD")}
    with rasterio.Env(**ENV):
        for t in sorted(set(tiles)):
            for yi, y in enumerate(YEARS):
                for L in ("AGB", "AGB_SD"):
                    with rasterio.open("/vsicurl/" + URL.format(y=y, t=t, L=L)) as ds:
                        b = ds.bounds
                        m = (LON >= b.left) & (LON < b.right) & (LAT > b.bottom) & (LAT <= b.top)
                        if not m.any():
                            continue
                        w = from_bounds(max(lon[0], b.left) - 0.002, max(lat[-1], b.bottom) - 0.002,
                                        min(lon[-1], b.right) + 0.002, min(lat[0], b.top) + 0.002, ds.transform)
                        w = w.round_offsets().round_lengths()
                        a = ds.read(1, window=w, boundless=True, fill_value=65535)
                        wt = ds.window_transform(w)
                        cc = ((LON[m] - wt.c) / wt.a).astype(int)
                        rr = ((LAT[m] - wt.f) / wt.e).astype(int)
                        ok = (rr >= 0) & (rr < a.shape[0]) & (cc >= 0) & (cc < a.shape[1])
                        v = np.full(m.sum(), 65535, np.uint16)
                        v[ok] = a[rr[ok], cc[ok]]
                        out[L][yi, m.ravel()] = v
    np.savez_compressed(OUT / f"{name}.npz", agb=out["AGB"], sd=out["AGB_SD"], years=np.array(YEARS))
    a = out["AGB"][-1]
    a = a[a != 65535]
    return f"{name}: tiles {sorted(set(tiles))} median agb 2024 {np.median(a) if a.size else None}  {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", nargs="*", default=["squares.json", "veg_squares.json", "lidar_squares.json"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sq = {}
    for f in args.picks:
        sq.update(json.load(open(HERE / "data" / f)))
    todo = [n for n in sq if not (OUT / f"{n}.npz").exists()]
    proc = psutil.Process()

    def guard():
        while True:
            if proc.memory_info().rss / 1e9 > args.max_gb:
                print("STOP: memory", flush=True)
                import os
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
    print(f"{len(todo)} squares", flush=True)
    t0 = time.time()

    def run(n):
        try:
            return one(n, sq[n])
        except Exception as e:  # noqa: BLE001
            return f"{n} FAILED: {e!r}"
    with ThreadPoolExecutor(args.workers) as ex:
        for k, msg in enumerate(ex.map(run, todo)):
            print(f"{k + 1}/{len(todo)} {msg}", flush=True)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
