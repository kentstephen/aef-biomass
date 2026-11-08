# /// script
# dependencies = ["numpy", "rasterio", "icechunk", "xarray", "zarr>=3", "requests", "psutil"]
# ///
"""PLAN.md step 1 probe: read each label under one square's pixels, report access, grid, nodata, read time.

    uv run scripts/probe_labels.py rondonia-br
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
import requests
from rasterio.warp import transform as warp_xy
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "rondonia-br"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
YEARS = list(range(2018, 2026))
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES")

sq = json.load(open(HERE / "data/squares.json"))[NAME]
y0, y1, x0, x1 = sq["rc"]
lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
LON, LAT = np.meshgrid(lon, lat)
W, S, E, N = lon.min(), lat.min(), lon.max(), lat.max()
sqz = np.load(HERE / f"data/squares/{NAME}.npz", mmap_mode="r")
wc = sqz["wc2021"].reshape(LAT.shape)
print(f"{NAME}: {LAT.shape} px, box {W:.4f} {S:.4f} {E:.4f} {N:.4f}")


def rss():
    return psutil.Process().memory_info().rss / 1e9


def sample_raster(url, crs_xy=None, band=1, label=""):
    """Value of a raster at each pixel center (nearest)."""
    t = time.time()
    with rasterio.Env(**ENV), rasterio.open(url) as s:
        if s.crs.to_epsg() == 4326:
            X, Y = LON, LAT
        else:
            X, Y = warp_xy("EPSG:4326", s.crs, LON.ravel(), LAT.ravel())
            X, Y = np.asarray(X).reshape(LON.shape), np.asarray(Y).reshape(LON.shape)
        pad = 3 * abs(s.transform.a)
        w = from_bounds(X.min() - pad, Y.min() - pad, X.max() + pad, Y.max() + pad, s.transform)
        a = s.read(band, window=w, boundless=True, fill_value=s.nodata if s.nodata is not None else 0)
        tr = s.window_transform(w)
        info = dict(crs=str(s.crs), res=(s.transform.a, s.transform.e), dtype=s.dtypes[0], nodata=s.nodata,
                    size=(s.width, s.height), tags=dict(list(s.tags().items())[:6]))
    r = np.clip(((Y - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
    c = np.clip(((X - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
    info["secs"] = round(time.time() - t, 1)
    return a[r, c], info


def quadkey(lo, la, z):
    x = int((lo + 180) / 360 * 2**z)
    s = math.sin(math.radians(la))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2**z)
    q = ""
    for i in range(z, 0, -1):
        d, m = 0, 1 << (i - 1)
        d += 1 if x & m else 0
        d += 2 if y & m else 0
        q += str(d)
    return q


trees = wc == 10
print(f"WorldCover 2021 trees {trees.mean():.2f}, crop {(wc == 40).mean():.2f}, grass {(wc == 30).mean():.2f}")

# ---- CTrees AGB --------------------------------------------------------------------------------------------
print("\n== CTrees AGB (Icechunk, anonymous)")
try:
    import icechunk
    import xarray as xr
    t = time.time()
    st = icechunk.s3_storage(bucket="ctrees-agb-100m-global", prefix="agb_100m_global", region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(st)
    ds = xr.open_zarr(repo.readonly_session(branch="main").store, zarr_format=3, group="aboveground_biomass",
                      chunks=None, mask_and_scale=False)
    print(f"  open {time.time() - t:.1f}s; vars {list(ds.data_vars)}; agb {ds.agb.shape} {ds.agb.dtype}; "
          f"years {int(ds.time.dt.year.min())}-{int(ds.time.dt.year.max())}")
    print(f"  x {float(ds.x[0]):.6f} step {float(ds.x[1] - ds.x[0]):.8f}; y {float(ds.y[0]):.6f} step {float(ds.y[1] - ds.y[0]):.8f}")
    RES = float(ds.x[1] - ds.x[0])
    XE0, YE0 = float(ds.x[0]) - RES / 2, float(ds.y[0]) + RES / 2
    ci = ((LON - XE0) / RES).astype(int)
    ri = ((YE0 - LAT) / RES).astype(int)
    r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
    ti = [i for i, y in enumerate(ds.time.dt.year.values) if int(y) in YEARS]
    t = time.time()
    A = np.asarray(ds.agb.isel(time=ti, y=slice(r0, r1), x=slice(c0, c1)).values)
    ta = time.time() - t
    t = time.time()
    U = np.asarray(ds.uncertainty.isel(time=ti, y=slice(r0, r1), x=slice(c0, c1)).values)
    print(f"  read agb {ta:.1f}s, uncertainty {time.time() - t:.1f}s for {A.shape} cells x {len(ti)} years")
    a = A[:, ri - r0, ci - c0]
    u = U[:, ri - r0, ci - c0]
    nod = (a == -9999).mean()
    print(f"  nodata share {nod:.3f}; cells under square {(r1 - r0) * (c1 - c0)} (~{(r1-r0)*(c1-c0) and 360000/((r1-r0)*(c1-c0)):.0f} px each)")
    for k, y in enumerate(YEARS[: len(ti)]):
        v = a[k][a[k] >= 0] / 10
        vt = a[k][trees & (a[k] >= 0)] / 10
        uu = u[k][u[k] >= 0] / 10
        print(f"  {y}: AGB median {np.median(v):6.1f} p90 {np.percentile(v, 90):6.1f}; on WC trees {np.median(vt) if vt.size else float('nan'):6.1f}; "
              f"err median {np.median(uu):5.1f} Mg/ha")
    loss = sqz["loss"].reshape(LAT.shape)
    for ly in (19, 20):
        m = loss == ly
        if m.sum():
            print(f"  Hansen loss 20{ly} px ({m.sum()}): AGB 2018 {np.median(a[0][m]) / 10:.1f} -> 2022 {np.median(a[4][m]) / 10:.1f} -> 2025 {np.median(a[-1][m]) / 10:.1f}")
except Exception as e:
    print("  FAILED", type(e).__name__, e)
print(f"  rss {rss():.2f} GB")

# ---- Meta canopy height v2, 10 m mean -----------------------------------------------------------------------
print("\n== Meta CHM v2 10 m mean (+ dates from source metadata)")
try:
    q7 = quadkey(float(lon.mean()), float(lat.mean()), 7)
    url = f"https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3_10m_mean/chm/{q7}.tif"
    h, info = sample_raster("/vsicurl/" + url)
    print(f"  tile {q7}: {info}")
    hv = h[h != info["nodata"]].astype(float) if info["nodata"] is not None else h.astype(float)
    print(f"  nodata share {(h == info['nodata']).mean():.3f}; height median {np.median(hv):.1f}, p90 {np.percentile(hv, 90):.1f}")
    print(f"  on WC trees median {np.median(h[trees & (h != info['nodata'])]):.1f}; on WC non-trees {np.median(h[~trees & (h != info['nodata'])]):.1f}")
    q10 = quadkey(float(lon.mean()), float(lat.mean()), 10)
    t = time.time()
    r = requests.get(f"https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/metadata/{q10}.geojson", timeout=60)
    print(f"  metadata {q10}: HTTP {r.status_code} in {time.time() - t:.1f}s")
    if r.ok:
        g = r.json()
        props = [f["properties"] for f in g["features"]]
        print(f"  {len(props)} date polygons; keys {list(props[0].keys()) if props else None}; first {props[:3]}")
except Exception as e:
    print("  FAILED", type(e).__name__, e)

print("\n== Meta CHM v1 acquisition date (global raster)")
try:
    d, info = sample_raster("/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/CHM_acquisition_date.tif")
    print(f"  {info}")
    vals, cnt = np.unique(d, return_counts=True)
    print(f"  values under square: {list(zip(vals[:8].tolist(), cnt[:8].tolist()))}")
except Exception as e:
    print("  FAILED", type(e).__name__, e)

# ---- Impact Observatory annual LULC v02 ----------------------------------------------------------------------
print("\n== Impact Observatory annual LULC v02 (Planetary Computer)")
try:
    t = time.time()
    tok = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/io-lulc-annual-v02", timeout=60).json()["token"]
    items = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", timeout=60, json={
        "collections": ["io-lulc-annual-v02"], "bbox": [W, S, E, N], "limit": 50}).json()["features"]
    print(f"  token + search {time.time() - t:.1f}s; {len(items)} items")
    io = {}
    for it in sorted(items, key=lambda i: i["properties"]["start_datetime"]):
        y = int(it["properties"]["start_datetime"][:4])
        href = it["assets"]["data"]["href"] + "?" + tok
        v, info = sample_raster("/vsicurl/" + href)
        io[y] = v
        print(f"  {y} {it['id']}: {info['secs']}s, {info['crs']}, res {info['res'][0]:.1f}, {info['dtype']}, nodata {info['nodata']}")
    print("  classes:", it["assets"]["data"].get("classification:classes", [])[:12] and
          {c["value"]: c["description"] for c in it["assets"]["data"]["classification:classes"]})
    if 2021 in io:
        v = io[2021]
        vals, cnt = np.unique(v, return_counts=True)
        print(f"  2021 class shares: {dict(zip(vals.tolist(), np.round(cnt / v.size, 3).tolist()))}")
        print(f"  IO 2021 trees (2) vs WorldCover 2021 trees (10): agree {((v == 2) == trees).mean():.3f}")
except Exception as e:
    print("  FAILED", type(e).__name__, e)

# ---- BII v1.1 -----------------------------------------------------------------------------------------------
print("\n== Vizzuality BII v1.1 (Source Coop COGs)")
try:
    for y in (2018, 2022, 2025):
        b, info = sample_raster(f"/vsicurl/https://data.source.coop/vizzuality/biodiversity-intactness-100m-v1-1/bii_{y}.tif")
        nd = info["nodata"]
        bv = b[~np.isnan(b)] if nd is None or (isinstance(nd, float) and math.isnan(nd)) else b[b != nd]
        print(f"  {y}: {info['secs']}s, {info['crs']}, res {info['res'][0]:.6f}, {info['dtype']}, nodata {nd}; "
              f"median {np.median(bv):.3f} range {bv.min():.3f}-{bv.max():.3f}; WC trees {np.median(b[trees]):.3f} non-trees {np.median(b[~trees]):.3f}")
except Exception as e:
    print("  FAILED", type(e).__name__, e)

print(f"\npeak-ish rss {rss():.2f} GB")
