"""Raw AlphaEarth (int8, the mosaic tge-labs/aef-mosaic on Source Coop) over any lon/lat box, with the map's head.

Shared by the label checks that read places outside the squares (top_end_plots.py, tundra_check.py, burn_lidar_ca.py).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RES, X0, Y0 = 8.983111749910169e-05, -180.0, 83.68570533713473
_m = {}

H = json.load(open(HERE / "web" / "public" / "models" / "biomass_aef.json"))
MU, SD = np.array(H["mean"], np.float32), np.array(H["scale"], np.float32)
W1 = np.array(H["w1"], np.float32).reshape(H["hidden"], H["n_in"])
B1 = np.array(H["b1"], np.float32)
W2 = np.array(H["w2"], np.float32).reshape(H["n_out"], H["hidden"])
B2 = np.array(H["b2"], np.float32)
CLASSES = H["classes"]


def mosaic():
    if "arr" not in _m:
        import zarr
        from obstore.store import S3Store
        from zarr.storage import ObjectStore
        zarr.config.set({"async.concurrency": 64})
        store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                        virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
        g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
        _m["times"], _m["arr"] = [int(t) for t in g["time"][:]], g["embeddings"]
    return _m["arr"], _m["times"]


def rc_of(lon0, lat0, lon1, lat1):
    """Mosaic rows and columns (y0, y1, x0, x1) whose pixel centers fall in the box."""
    x0, x1 = int(np.ceil((lon0 - X0) / RES - 0.5)), int(np.floor((lon1 - X0) / RES - 0.5)) + 1
    y0, y1 = int(np.ceil((Y0 - lat1) / RES - 0.5)), int(np.floor((Y0 - lat0) / RES - 0.5)) + 1
    return y0, y1, x0, x1


def centers(rc):
    y0, y1, x0, x1 = rc
    return X0 + (np.arange(x0, x1) + 0.5) * RES, Y0 - (np.arange(y0, y1) + 0.5) * RES


def raw(rc, year):
    """int8 (h * w, 64), -128 = no data."""
    arr, times = mosaic()
    y0, y1, x0, x1 = rc
    for attempt in range(4):
        try:
            e = np.asarray(arr[times.index(year), :, y0:y1, x0:x1])
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            import time
            time.sleep(3 + 5 * attempt)
    return e.reshape(64, -1).T


def head_raw(q):
    f = q.astype(np.float32) / 127.5
    x = (np.sign(f) * f * f - MU) / SD
    return np.maximum(x @ W1.T + B1, 0) @ W2.T + B2


def head(q):
    """The map's head on raw int8 (n, 64): biomass Mg/ha (NaN where no data), height m, class index (-1 none)."""
    ok = q[:, 0] != -128
    agb = np.full(len(q), np.nan, np.float32)
    hgt = np.full(len(q), np.nan, np.float32)
    cls = np.full(len(q), -1, np.int8)
    if ok.any():
        r = head_raw(q[ok])
        agb[ok] = np.expm1(np.clip(r[:, 0], 0, 8))
        hgt[ok] = np.expm1(np.clip(r[:, 1], 0, 6))
        cls[ok] = r[:, 2:].argmax(1)
    return agb, hgt, cls


def guard(max_gb):
    """Stop the process on its own when its memory passes max_gb."""
    import os
    import threading
    import time

    import psutil

    def run():
        p = psutil.Process()
        while True:
            gb = p.memory_info().rss / 1e9
            if gb > max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
