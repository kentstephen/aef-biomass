# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pyproj"]
# ///
"""Which years does OlmoEarth v1.3 have at a few forest squares worldwide? Reads one 4 MB chunk per year."""
import time
import numpy as np
import zarr
from obstore.store import S3Store
from pyproj import Transformer
from zarr.storage import ObjectStore

S = ObjectStore(S3Store.from_url("s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr", endpoint="https://data.source.coop",
                                 skip_signature=True, client_options={"timeout": "120s"}), read_only=True)
PTS = {"kainuu-fi": (27.0, 63.0), "tres-lagoas-br": (-51.9, -20.6), "kolwezi-cd": (25.47, -10.71),
       "kaingaroa-nz": (176.6, -38.6), "ware-ga": (-82.4, 31.2), "jau-br": (-62.5, -2.0), "sangatta-id": (117.5, 0.5)}
root = zarr.open_group(S, mode="r")
print("root attrs:", dict(root.attrs) if root.attrs else "-")
for n, (lon, lat) in PTS.items():
    z = int((lon + 180) // 6) + 1
    t = time.time()
    try:
        g = zarr.open_group(S, mode="r", path=f"utm{z:02d}")
    except Exception as e:
        print(n, f"utm{z:02d}", "no group:", e); continue
    arr = g["embeddings"]
    a_, _, x0, _, e_, y0 = g.attrs["spatial:transform"]
    x, y = Transformer.from_crs(4326, g.attrs["proj:code"], always_xy=True).transform(lon, lat)
    col, row = int((x - x0) // a_), int((y - y0) // e_)
    inside = 0 <= row < arr.shape[2] and 0 <= col < arr.shape[3]
    out = []
    if inside:
        for yi in range(arr.shape[0]):
            q = arr[yi, 0, max(row - 32, 0):row + 32, max(col - 32, 0):col + 32]
            out.append(round(float((q != -128).mean()), 2))
    print(f"{n:16s} utm{z:02d} {g.attrs['proj:code']} shape {arr.shape} chunks {arr.chunks} inside {inside} valid per year {out} [{time.time()-t:.0f} s]", flush=True)
