"""What each embedding costs to read on the fly: bytes and requests to read one 6 km square (600 x 600 AlphaEarth
pixels, 36 km2) for one year, from each public store, counted at the store as zarr fetches it (whole chunks, so a
browser reading the same chunks pays the same).

  AlphaEarth  tge-labs/aef-mosaic on Source Cooperative (int8, chunks 64 x 256 x 256, shards 4096)
  TESSERA     tessera v1.1 (dclimate Zarr) through aef-tessera-earthwork's reader; this also reads the per-year
              observation counts, which a map would not need, so the map's cost is somewhat lower
  OlmoEarth   olmoearth-v1-3-embeddings, one group per UTM zone, through read_olmo.py

Writes results/embedding_bandwidth.json. Light (well under 1 GB).

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil python scripts/embedding_bandwidth.py
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW)); sys.path.insert(0, str(HERE / "scripts")); os.chdir(EW)
import numpy as np, zarr
from obstore.store import HTTPStore, S3Store
from zarr.storage import ObjectStore
import tessera_sites as ts
import read_olmo

class Counting(ObjectStore):
    n = 0; req = 0
    async def get(self, key, prototype, byte_range=None):
        b = await super().get(key, prototype, byte_range)
        if b is not None:
            Counting.n += len(b); Counting.req += 1
        return b
    async def get_partial_values(self, prototype, key_ranges):
        out = await super().get_partial_values(prototype, key_ranges)
        for b in out:
            if b is not None:
                Counting.n += len(b); Counting.req += 1
        return out

def reset():
    Counting.n = 0; Counting.req = 0

sq = json.load(open(HERE / "data/us_squares.json")) | json.load(open(HERE / "data/veg_squares.json"))
zarr.config.set({"async.concurrency": 64})
res = {}
for name in ("flint-hills-ks", "olympic-us", "congo-intact-cg"):
    rc = tuple(sq[name]["rc"]); year = 2021
    out = {}
    # AlphaEarth mosaic
    st = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2", virtual_hosted_style_request=False,
                 skip_signature=True, prefix="aef-mosaic")
    g = zarr.open_group(Counting(st, read_only=True), mode="r")
    arr, times = g["embeddings"], g["time"][:]
    reset(); t = time.time()
    np.asarray(arr[int(np.where(times == year)[0][0]), :, rc[0]:rc[1], rc[2]:rc[3]])
    out["aef"] = {"MB": round(Counting.n / 1e6, 2), "requests": Counting.req, "s": round(time.time() - t, 1),
                  "chunks": list(arr.chunks), "shards": list(arr.shards) if getattr(arr, "shards", None) else None}
    # TESSERA
    gt = Counting(HTTPStore.from_url(ts.TESSERA), read_only=True)
    reset(); t = time.time()
    ts.read_rc(gt, name, rc, [year], tags=["y"])
    out["tes"] = {"MB": round(Counting.n / 1e6, 2), "requests": Counting.req, "s": round(time.time() - t, 1)}
    # OlmoEarth
    read_olmo.STORE = Counting(S3Store.from_url("s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
                               endpoint="https://data.source.coop", skip_signature=True, region="us-west-2"), read_only=True)
    read_olmo._zones.clear()
    reset(); t = time.time()
    read_olmo.read(name, rc, None, [year])
    out["olmo"] = {"MB": round(Counting.n / 1e6, 2), "requests": Counting.req, "s": round(time.time() - t, 1)}
    res[name] = out
    print(name, json.dumps(out), flush=True)
json.dump(res, open(HERE / "results" / "embedding_bandwidth.json", "w"), indent=1)
