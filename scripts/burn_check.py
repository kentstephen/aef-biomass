"""Does the map's biomass fall inside burn scars? The map's head (web/public/models/biomass_aef.json) on AlphaEarth,
every year 2018 to 2025, inside a few large California fires against unburned ground around each, beside CTrees and
fire-knn's vegetation volume head (models/fuel_volume.json, LANDFIRE-taught) on the same pixels.

Fires: perimeters from MTBS (Source Cooperative, through 2023 fires) and NIFC WFIGS (the Park Fire, 2024).
Per fire: up to --chunks 256 px mosaic chunks with their center inside the perimeter, and as many in a ring 2 to 8 km
outside it that no fire of 2015 to 2024 touched; every 4th pixel of each kept, by where it falls.

Writes results/burn_check.json and prints, per fire and year, the median inside, outside, and inside / outside for
each of the three. RAM ~1 GB; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with duckdb --with shapely --with icechunk \
        --with xarray python scripts/burn_check.py
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
import shapely
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import volume_head as vh  # noqa: E402

YEARS = list(range(2018, 2026))
RES, X0, Y0, CHUNK = 8.983111749910169e-05, -180.0, 83.68570533713473, 256
MTBS = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/cboettig/fire/mtbs-perimeters-1984-2024.parquet"
WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
# (name, ignition year, source): Paradise's two, the fire-knn test fires, and a chaparral one
FIRES = [("CAMP", 2018, "mtbs"), ("PARK", 2024, "wfigs"), ("CREEK", 2020, "mtbs"), ("DIXIE", 2021, "mtbs"),
         ("CALDOR", 2021, "mtbs"), ("BOBCAT", 2020, "mtbs")]
OUT = HERE / "results" / "burn_check.json"
KM = 1 / 111.0


# ---------------------------------------------------------------- the map's head, as the browser runs it
H = json.load(open(HERE / "web" / "public" / "models" / "biomass_aef.json"))
MU, SD = np.array(H["mean"], np.float32), np.array(H["scale"], np.float32)
W1 = np.array(H["w1"], np.float32).reshape(H["hidden"], H["n_in"])
B1 = np.array(H["b1"], np.float32)
W2 = np.array(H["w2"], np.float32).reshape(H["n_out"], H["hidden"])
B2 = np.array(H["b2"], np.float32)


def map_raw(q):
    f = q.astype(np.float32) / 127.5
    x = (np.sign(f) * f * f - MU) / SD
    return np.maximum(x @ W1.T + B1, 0) @ W2.T + B2


def map_agb(q):
    return np.expm1(np.clip(map_raw(q)[:, 0], 0, 8))


# ---------------------------------------------------------------- perimeters
def perimeters():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(f"""SELECT upper(Incid_Name), year(Ig_Date), ST_AsWKB(geom) FROM '{MTBS}'
        WHERE TRY_CAST(BurnBndLat AS DOUBLE) BETWEEN 32 AND 42.5 AND TRY_CAST(BurnBndLon AS DOUBLE) BETWEEN -125 AND -114
        AND year(Ig_Date) BETWEEN 2015 AND 2024""").fetchall()
    allg = [(n, int(y), shapely.make_valid(shapely.from_wkb(bytes(g)))) for n, y, g in rows]
    import requests
    for name, yr, src in FIRES:
        if src != "wfigs":
            continue
        r = requests.get(WFIGS, params={"where": f"poly_IncidentName='{name.title()}' AND attr_POOState='US-CA'",
                                        "outFields": "poly_GISAcres,attr_FireDiscoveryDateTime", "outSR": 4326,
                                        "f": "geojson"}, timeout=120).json()
        best = max((f for f in r["features"]
                    if time.gmtime(f["properties"]["attr_FireDiscoveryDateTime"] / 1000).tm_year == yr),
                   key=lambda f: f["properties"]["poly_GISAcres"])
        allg.append((name, yr, shapely.make_valid(shape(best["geometry"]))))
    return allg


# ---------------------------------------------------------------- the mosaic
_aef, _lock = {}, threading.Lock()


def mosaic():
    with _lock:
        if "arr" not in _aef:
            import zarr
            from obstore.store import S3Store
            from zarr.storage import ObjectStore
            store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                            virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
            g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
            _aef["times"], _aef["arr"] = list(np.asarray(g["time"][:]).astype(int)), g["embeddings"]
    return _aef["arr"], _aef["times"]


def chunk_of(lon, lat):
    return int((Y0 - lat) / RES) // CHUNK, int((lon - X0) / RES) // CHUNK


def chunk_center(r, c):
    return X0 + (c + 0.5) * CHUNK * RES, Y0 - (r + 0.5) * CHUNK * RES


def pick_chunks(geom, rng, n, inside):
    x0, y0, x1, y1 = geom.bounds
    r0, c0 = chunk_of(x0, y1)
    r1, c1 = chunk_of(x1, y0)
    cand = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    pts = np.array([chunk_center(r, c) for r, c in cand])
    ok = shapely.contains_xy(geom, pts[:, 0], pts[:, 1]) if len(cand) else np.zeros(0, bool)
    cand = [k for k, o in zip(cand, ok) if o]
    rng.shuffle(cand)
    return cand[:n]


def read_chunk(r, c, step=4):
    """Every `step`th pixel of a chunk, all years: lon, lat (m,), int8 embeddings (Y, m, 64)."""
    arr, times = mosaic()
    y0, x0 = r * CHUNK, c * CHUNK
    iy, ix = np.meshgrid(np.arange(0, CHUNK, step), np.arange(0, CHUNK, step), indexing="ij")
    lon = X0 + (x0 + ix.ravel() + 0.5) * RES
    lat = Y0 - (y0 + iy.ravel() + 0.5) * RES
    E = np.empty((len(YEARS), lon.size, 64), np.int8)
    for i, y in enumerate(YEARS):
        for attempt in range(4):
            try:
                e = np.asarray(arr[times.index(y), :, y0:y0 + CHUNK, x0:x0 + CHUNK])
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + 5 * attempt)
        E[i] = e[:, ::step, ::step].reshape(64, -1).T
    return lon, lat, E


def ctrees_at(lon, lat):
    import read_labels as rl
    LON, LAT = lon[None, :], lat[None, :]
    try:
        d = rl.read_ctrees(LON, LAT)
    except Exception as ex:  # noqa: BLE001
        print(f"  CTrees read failed: {ex}", flush=True)
        return None
    a = d["agb"].astype(np.float32)
    return np.where(a < 0, np.nan, a / 10)


def guard(max_gb):
    def run():
        p = psutil.Process()
        while True:
            if p.memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12, help="chunks inside and outside, per fire")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    guard(args.max_gb)
    t0 = time.time()
    q = np.array(H["probe_q"], np.int8).reshape(-1, 64)
    gap = float(np.abs(map_raw(q) - np.array(H["probe_out"], np.float32).reshape(len(q), -1)).max())
    print(f"map head probe gap {gap:.2e}", flush=True)
    allg = perimeters()
    print(f"{len(allg)} perimeters, 2015 to 2024", flush=True)
    burned_any = shapely.union_all([g for _, _, g in allg])
    rng = np.random.default_rng(0)
    res = {"years": YEARS, "fires": {}}
    jobs = []
    for name, yr, _ in FIRES:
        m = [g for n, y, g in allg if n == name and y == yr]
        if not m:
            print(f"{name} {yr}: no perimeter", flush=True)
            continue
        fire = shapely.union_all(m)
        ring = fire.buffer(8 * KM).difference(fire.buffer(2 * KM)).difference(burned_any)
        ins, outs = pick_chunks(fire, rng, args.chunks, True), pick_chunks(ring, rng, args.chunks, False)
        res["fires"][name] = {"year": yr, "acres_deg2": fire.area, "chunks_in": len(ins), "chunks_out": len(outs)}
        for zone, ks, geom in (("inside", ins, fire), ("outside", outs, ring)):
            for k in ks:
                jobs.append((name, zone, k, geom))
    print(f"{len(jobs)} chunks to read, 8 years each", flush=True)
    acc = {}
    done = [0]

    def one(job):
        name, zone, (r, c), geom = job
        lon, lat, E = read_chunk(r, c)
        keep = shapely.contains_xy(geom, lon, lat) & (E[0, :, 0] != -128)
        lon, lat, E = lon[keep], lat[keep], E[:, keep]
        agb = np.stack([map_agb(E[i]) for i in range(len(YEARS))])
        vol = np.stack([vh.heads(E[i])["vol"] for i in range(len(YEARS))])
        ok = (E[:, :, 0] != -128)
        agb[~ok], vol[~ok] = np.nan, np.nan
        ct = ctrees_at(lon, lat)
        with _lock:
            d = acc.setdefault((name, zone), {"agb": [], "vol": [], "ct": []})
            d["agb"].append(agb), d["vol"].append(vol)
            if ct is not None:
                d["ct"].append(ct)
            done[0] += 1
            print(f"chunk {done[0]} of {len(jobs)}: {name} {zone} ({r}, {c}) {keep.sum()} px "
                  f"[{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            f.result()

    print("\nmedian per year; ratio = inside / outside (1 = the scar reads like unburned ground)", flush=True)
    for name, yr, _ in FIRES:
        if name not in res["fires"]:
            continue
        o = res["fires"][name]
        print(f"\n{name} (ignited {yr})   " + " ".join(f"{y:>7d}" for y in YEARS), flush=True)
        for key, label in (("agb", "map head Mg/ha"), ("ct", "CTrees Mg/ha"), ("vol", "fire-knn volume m")):
            row = {}
            for zone in ("inside", "outside"):
                v = acc.get((name, zone), {}).get(key)
                if not v:
                    continue
                a = np.concatenate(v, 1)
                row[zone] = [float(np.nanmedian(a[i])) for i in range(len(YEARS))]
                row[f"{zone}_px"] = int(np.isfinite(a[0]).sum())
            if "inside" in row and "outside" in row:
                row["ratio"] = [i / max(u, 1e-6) for i, u in zip(row["inside"], row["outside"])]
                for zone in ("inside", "outside", "ratio"):
                    fmt = "{:7.2f}" if zone == "ratio" or key == "vol" else "{:7.1f}"
                    print(f"  {label:18s} {zone:7s} " + " ".join(fmt.format(v) for v in row[zone]), flush=True)
            o[key] = row
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
