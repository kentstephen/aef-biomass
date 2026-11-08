# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio"]
# ///
"""The SoCal annual biomass map under the chaparral squares (and any other square inside it), 2018 to 2023.

Schrader-Patton, Underwood et al., "Annual biomass data (2001-2023) for southern California", Dryad
doi:10.5061/dryad.qz612jmjt, CC0. Aboveground live biomass from a random forest on 766 field plots (FIA, LANDFIRE
reference plots, research plots) with Landsat July-August NDVI and PRISM precipitation of each year: Landsat-driven,
so it shares inputs with the embeddings, but not GEDI and not CTrees. EPSG:4326, ~30 m, g/m2 (x 0.01 = Mg/ha).

Needs the yearly zips in data/chaparral/zips (WWETAC_UCD_SoCal_Biomass_<year>.zip, 2018 to 2023; Dryad's
download needs a browser or an account token). The aboveground layer is read straight from each zip.

Per square, data/chaparral/<name>.npz: agb float32 (6, n) Mg/ha, NaN none; years.

    uv run scripts/read_chaparral.py
"""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
ZIPS = HERE / "data" / "chaparral" / "zips"
OUT = HERE / "data" / "chaparral"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
YEARS = list(range(2018, 2024))
BOUNDS = (-121.96, 32.47, -115.23, 36.79)


def grid(rc):
    y0, y1, x0, x1 = rc
    return (AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)


def layer(year):
    z = ZIPS / f"WWETAC_UCD_SoCal_Biomass_{year}.zip"
    names = [n for n in zipfile.ZipFile(z).namelist() if n.lower().endswith(".tif") and "above" in n.lower()]
    if len(names) != 1:
        raise SystemExit(f"{z.name}: expected one aboveground tif, found {names}")
    return f"/vsizip/{z}/{names[0]}"


def read(path, lon, lat):
    with rasterio.open(path) as s:
        pad = 3 * abs(s.transform.a)
        w = from_bounds(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad, s.transform)
        a = s.read(1, window=w, boundless=True, fill_value=s.nodata if s.nodata is not None else -1).astype(np.float32)
        tr, nod = s.window_transform(w), s.nodata
    if nod is not None:
        a[a == nod] = np.nan
    a[a < 0] = np.nan
    r = np.floor((lat - tr.f) / tr.e).astype(int)
    c = np.floor((lon - tr.c) / tr.a).astype(int)
    R, C = np.meshgrid(r, c, indexing="ij")
    return a[R, C].ravel() * 0.01


def main():
    sq = {}
    for f in ("us_squares.json", "lidar_squares.json", "veg_squares.json", "squares.json"):
        sq.update(json.load(open(HERE / "data" / f)))
    W, S, E, N = BOUNDS
    todo = [k for k, v in sq.items() if W < v["center"][0] < E and S < v["center"][1] < N]
    paths = {y: layer(y) for y in YEARS}
    for name in todo:
        lon, lat = grid(sq[name]["rc"])
        agb = np.stack([read(paths[y], lon, lat) for y in YEARS])
        valid = np.isfinite(agb).mean()
        if valid < 0.3:
            print(f"{name:24s} outside the map ({valid:.2f} valid), skipped", flush=True)
            continue
        tmp = OUT / f"{name}.tmp.npz"
        np.savez_compressed(tmp, agb=agb, years=np.array(YEARS), rc=np.array(sq[name]["rc"]))
        os.replace(tmp, OUT / f"{name}.npz")
        med = " ".join(f"{np.nanmedian(a):.1f}" for a in agb)
        print(f"{name:24s} valid {valid:.2f}  Mg/ha median by year {med}", flush=True)


if __name__ == "__main__":
    main()
