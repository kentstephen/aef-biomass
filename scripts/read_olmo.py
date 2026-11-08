# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pyproj", "psutil"]
# ///
"""OlmoEarth v1.3 under the pixels Earthwork already holds AlphaEarth and TESSERA for, so all three compare on
the same ground. Each pixel center on AlphaEarth's grid is projected into the square's UTM zone and the
OlmoEarth pixel under it is taken (as Earthwork's tessera_sites.read_rc does for TESSERA).

Sets (from ~/dev/projects/aef-tessera-earthwork/data):
  forest  the 19 forest squares of signal_trees.py (separate/<name>.npz idx), years 2018 to 2025
  quiet   the 20 quiet natural squares (years/<name>.npz, every pixel), years 2018 to 2025
  ground  the burn, reservoir and lookalike squares (places/ground-*.npz, every pixel), their two years
The crop, farm and marsh sets are already in Earthwork's data/olmoearth (same rows as its TESSERA files).

Writes data/olmo/<name>.npz: q int8 (years, n, 128), years. Resumable. Four squares at a time; peak RAM about
1.5 GB; stops past --max-gb.

    uv run scripts/read_olmo.py [forest quiet ground] [--workers 4] [--max-gb 6]
"""
import argparse
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import zarr
from obstore.store import S3Store
from pyproj import Transformer
from zarr.storage import ObjectStore

EW = Path.home() / "dev/projects/aef-tessera-earthwork/data"
OUT = Path(__file__).resolve().parent.parent / "data" / "olmo"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
T0 = 2017
YEARS = list(range(2018, 2026))
HALF = 300
FOREST = ["coast-range-or", "willapa-wa", "ware-ga", "pine-ms", "north-woods-me", "norrland-se", "kainuu-fi",
          "kaingaroa-nz", "tres-lagoas-br", "near-kolwezi-cd", "near-fort-hills-ab", "near-detour-lake-on",
          "near-adaro-id", "near-sangatta-id", "near-madre-de-dios-pe", "near-boke-gn", "near-mobimbi-sl",
          "near-weda-bay-id", "near-morowali-id"]
STORE = ObjectStore(S3Store.from_url("s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
                                     endpoint="https://data.source.coop", skip_signature=True,
                                     client_options={"timeout": "120s", "connect_timeout": "10s"}), read_only=True)
_zones, _lock = {}, threading.Lock()


def zone(lon):
    n = int((lon + 180) // 6) + 1
    with _lock:
        if n not in _zones:
            _zones[n] = zarr.open_group(STORE, mode="r", path=f"utm{n:02d}")
        return _zones[n]


def read(name, rc, idx, years):
    y0, y1, x0, x1 = rc
    LON, LAT = np.meshgrid(AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES, AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES)
    LON, LAT = LON.ravel(), LAT.ravel()
    if idx is not None:
        LON, LAT = LON[idx], LAT[idx]
    z = zone(float(LON.mean()))
    X, Y = Transformer.from_crs(4326, z.attrs["proj:code"], always_xy=True).transform(LON, LAT)
    a, _, tx0, _, e, ty0 = z.attrs["spatial:transform"]
    col = np.floor((X - tx0) / a).astype(np.int64)
    row = np.floor((Y - ty0) / e).astype(np.int64)
    r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
    at = (row - r0) * (c1 - c0) + (col - c0)
    out = np.empty((len(years), len(at), 128), np.int8)
    for i, yr in enumerate(years):
        for attempt in range(4):
            try:
                q = z["embeddings"][yr - T0, :, r0:r1, c0:c1]
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + 5 * attempt)
        out[i] = q.reshape(128, -1)[:, at].T
        del q
    return out


def jobs(sets):
    js = []
    if "forest" in sets:
        for n in FOREST:
            z = np.load(EW / "separate" / f"{n}.npz", allow_pickle=True)
            if n.startswith("near-"):
                lon, lat = z["point"]
                r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
                rc = (r - HALF, r + HALF, c - HALF, c + HALF)
            else:
                rc = tuple(json.loads(str(np.load(EW / "places" / f"ground-{n}.npz", allow_pickle=True)["meta"]))["rc"])
            js.append((n, rc, z["idx"], YEARS))
    if "quiet" in sets:
        for f in sorted(glob.glob(str(EW / "years" / "*.npz"))):
            js.append((Path(f).stem, tuple(np.load(f)["rc"]), None, YEARS))
    if "ground" in sets:
        for f in sorted(glob.glob(str(EW / "places" / "ground-*.npz"))):
            meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
            if meta["kind"] != "cut":
                js.append((Path(f).stem, tuple(meta["rc"]), None, list(meta["aef"])))
    return [j for j in js if not (OUT / f"{j[0]}.npz").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=["forest", "quiet", "ground"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 64})
    p, peak = psutil.Process(), [0.0]

    def guard():
        while True:
            g = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], g)
            if g > args.max_gb:
                print(f"STOP: RSS {g:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
    js = jobs(args.sets)
    print(f"{len(js)} squares to read", flush=True)
    t0 = time.time()

    def one(j):
        n, rc, idx, years = j
        t = time.time()
        q = read(n, rc, idx, years)
        tmp = OUT / f"{n}.tmp.npz"
        np.savez(tmp, q=q, years=np.array(years))
        os.replace(tmp, OUT / f"{n}.npz")
        return n, q.shape, time.time() - t, float((q[..., 0] == -128).mean())

    with ThreadPoolExecutor(args.workers) as ex:
        for k, (n, shape, dt, nod) in enumerate(ex.map(one, js)):
            el = time.time() - t0
            print(f"{k + 1}/{len(js)} {n} {shape} nodata {nod:.3f} {dt:.0f} s  [eta {el / (k + 1) * (len(js) - k - 1) / 60:.0f} min, "
                  f"peak {peak[0]:.1f} GB]", flush=True)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
