"""Every pixel of the new test squares (data/squares.json), every year 2018 to 2025, in all three embeddings, with
the reference layers under them, on AlphaEarth's grid (as Earthwork aligns TESSERA).

Per square, data/squares/<name>.npz:
  aef      int8 (8, n, 64)    AlphaEarth mosaic (tge-labs/aef-mosaic), raw int8
  tes      int8 (8, n, 128)   TESSERA v1.1, with tes_scale float32 (8, n) and s1 int16 (8, n) radar looks
  olmo     int8 (8, n, 128)   OlmoEarth v1.3 (-128 = nodata)
  loss     uint8 (n,)         Hansen GFC v1.12 loss year (0 none, 1 to 24 = 2001 to 2024); cover uint8 tree cover 2000
  wc2020, wc2021  uint8 (n,)  ESA WorldCover classes (10 trees, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare,
                              80 water, 90 wetland, 95 mangrove, 100 moss); 0 outside the tile read
n = 600 x 600 pixels. Resumable. Four squares at a time; peak RAM about 6 GB; stops past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil python scripts/read_squares.py [--workers 4]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW))
sys.path.insert(0, str(HERE / "scripts"))
os.chdir(EW)
import zarr  # noqa: E402
from obstore.store import HTTPStore  # noqa: E402
from zarr.storage import ObjectStore  # noqa: E402

import ground  # noqa: E402
import read_olmo  # noqa: E402
import tessera_sites as ts  # noqa: E402

YEARS = list(range(2018, 2026))
OUT = HERE / "data" / "squares"
WC = {2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{}_Map.tif",
      2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"}
_init = threading.Lock()


def centers(rc):
    y0, y1, x0, x1 = rc
    lat = ts.AEF_Y0 - (np.arange(y0, y1) + 0.5) * ts.AEF_RES
    lon = ts.AEF_X0 + (np.arange(x0, x1) + 0.5) * ts.AEF_RES
    return lon, lat


def at(a, tr, lon, lat):
    rr = np.clip(((lat - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
    cc = np.clip(((lon - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
    return a[rr][:, cc].ravel()


def worldcover(year, lon, lat):
    clon, clat = float(lon.mean()), float(lat.mean())
    la, lo = int(np.floor(clat / 3) * 3), int(np.floor(clon / 3) * 3)
    tile = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open("/vsicurl/" + WC[year].format(tile)) as s:
        w = from_bounds(lon.min() - 1e-4, lat.min() - 1e-4, lon.max() + 1e-4, lat.max() + 1e-4, s.transform)
        a = s.read(1, window=w, boundless=True, fill_value=0)
        tr = s.window_transform(w)
    v = at(a, tr, lon, lat)
    inside = (lon[None, :] >= lo) & (lon[None, :] < lo + 3) & (lat[:, None] >= la) & (lat[:, None] < la + 3)
    return np.where(inside.ravel(), v, 0).astype(np.uint8)


def aef_raw(rc):
    with _init:
        if "arr" not in ts._aef:
            ts.aef_unit(YEARS[0], (rc[0], rc[0] + 1, rc[2], rc[2] + 1))
    arr, times = ts._aef["arr"], ts._aef["times"]
    y0, y1, x0, x1 = rc
    out = np.empty((len(YEARS), (y1 - y0) * (x1 - x0), 64), np.int8)
    for i, y in enumerate(YEARS):
        e = np.asarray(arr[int(np.where(times == y)[0][0]), :, y0:y1, x0:x1])
        out[i] = e.reshape(64, -1).T
    return out


def one(g, name, sq):
    t = time.time()
    rc = tuple(sq["rc"])
    lon, lat = centers(rc)
    box = (lon.min() - 0.001, lat.min() - 0.001, lon.max() + 0.001, lat.max() + 0.001)
    d = {"loss": at(*ground.hansen("lossyear", *box), lon, lat).astype(np.uint8),
         "cover": at(*ground.hansen("treecover2000", *box), lon, lat).astype(np.uint8),
         "wc2020": worldcover(2020, lon, lat), "wc2021": worldcover(2021, lon, lat)}
    d["aef"] = aef_raw(rc)
    got = ts.read_rc(g, name, rc, YEARS, tags=[f"y{y}" for y in YEARS])
    d["tes"] = np.stack([got[f"y{y}_q"] for y in YEARS])
    d["tes_scale"] = np.stack([got[f"y{y}_scales"] for y in YEARS]).astype(np.float32)
    d["s1"] = np.stack([got[f"y{y}_s1_asc_obs_count"].astype(np.int32) + got[f"y{y}_s1_desc_obs_count"] for y in YEARS]).astype(np.int16)
    del got
    d["olmo"] = read_olmo.read(name, rc, None, YEARS)
    tmp = OUT / f"{name}.tmp.npz"
    np.savez(tmp, rc=np.array(rc), years=np.array(YEARS), **d)
    os.replace(tmp, OUT / f"{name}.npz")
    return name, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 128})
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
    sq = json.load(open(HERE / "data" / "squares.json"))
    todo = [n for n in sq if not (OUT / f"{n}.npz").exists()]
    print(f"{len(todo)} squares to read", flush=True)
    g = ObjectStore(HTTPStore.from_url(ts.TESSERA), read_only=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, g, n, sq[n]) for n in todo]
        for k, f in enumerate(futs):
            try:
                n, dt = f.result()
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} {n} {dt:.0f} s  [eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min, peak {peak[0]:.1f} GB]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{k + 1}/{len(todo)} {todo[k]} FAILED: {e!r}", flush=True)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
