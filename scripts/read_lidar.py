"""Airborne lidar canopy height under new 6 km squares: an independent check of the heads' height and change.

CTrees is partly calibrated on GEDI, which AlphaEarth was taught to reconstruct; Meta's heights come from imagery.
Repeat airborne lidar is neither. Canopy height = surface model (highest returns) minus ground model, from the
national programs Earthwork already reads for bare ground (~/dev/projects/aef-tessera-earthwork, world_*.py),
with the surface product of the same flights swapped in, plus California's NASA WERK canopy height:
  nz     LINZ nz-elevation dsm_1m minus dem_1m (COGs, NZTM), two surveys per place
  ca     NRCan HRDEM <project>-dsm.tif minus -dtm.tif (COGs), two projects
  nl     AHN4 (PDOK) and AHN5 0.5 m DSM minus DTM
  ee     Estonia's LAZ point clouds already on disk in Earthwork (3 x 3 km per site): highest non-noise return
         minus mean ground return per 2 m cell
  werk   NASA WERK 1 m canopy height (band 3 of its COGs): Los Angeles 2023 before, the 2025 post-fire flights
  werk10 NASA WERK statewide 10 m canopy height, one flight per place, with its campaign year
Each source is read onto a 2 m grid in its own CRS (Earthwork's world_sites.read_cogs: full resolution averaged
down), then sampled 5 x 5 per AlphaEarth pixel of the square: mean and 90th percentile of the 25 heights and the
share of them with data.

Per square, data/lidar/<name>.npz (n = 600 x 600, row-major as data/squares/<name>.npz):
  chm, chm_p90 float16 (E, n)  canopy height, m (NaN none), per epoch E (2, or 1 for werk10)
  cover uint8 (E, n)            share of the 25 samples with data, %
  year int16 (E,)               the embedding year each epoch is paired with (flight's middle date);
                                werk10: year_px int16 (n,) per pixel
  dates (E, 2) str              first and last flight day (or survey span)
data/lidar_squares.json lists the squares (rc, center, kind "lidar", source, note) for read_squares.py and
read_labels.py.

Network-bound; peak RAM under 3 GB per square. Run from this repo, in Earthwork's environment:

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with "laspy[lazrs]" \\
        python scripts/read_lidar.py [--sites-only] [--only name ...]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
OUT = HERE / "data" / "lidar"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
HALF = 300
D = datetime.date

NORTH = ("northland/northland_2018-2020", "northland/northland_2024")
HAWKES = ("hawkes-bay/hawkes-bay_2020-2021", "hawkes-bay/hawkes-bay_2023")
OTAGO = ("otago/central-otago_2021", "otago/central-otago_2022-2023")
BANKS = ("canterbury/banks-peninsula_2018-2019", "canterbury/banks-peninsula_2023")
FMM = ("NRCAN-Fort_McMurray_2018-1m", "NRCAN-Athabasca_Clearwater_UTM12_2022-1m")
TIMMINS = ("ON-SPL_ON_Cochrane_UTM17_2018-1m", "ON-Timmins_UTM17_2024-1m")


def S(name, src, lat, lon, note, pair=None, ew=None):
    return dict(name=f"lidar-{name}", src=src, lat=lat, lon=lon, pair=pair, ew=ew, note=note)


SITES = [
    S("nz-glenbervie", "nz", -35.64, 174.36, "Glenbervie plantation pine and native forest, Northland", NORTH),
    S("nz-esk-valley", "nz", -39.405, 176.825, "Esk Valley, Cyclone Gabrielle slips and silt (Feb 2023)", HAWKES),
    S("nz-roxburgh", "nz", -45.45, 169.38, "Central Otago, semi-arid tussock, shrub and orchards", OTAGO),
    S("nz-banks", "nz", -43.75, 172.85, "Banks Peninsula pasture, shrub and forest remnants", BANKS),
    S("ca-fort-mcmurray", "ca", 56.784, -111.142, "boreal forest and muskeg east of Fort McMurray (partial cover)", FMM),
    S("ca-timmins", "ca", 48.50, -81.33, "boreal forest near Timmins", TIMMINS),
    S("nl-heuvelrug", "nl", 52.060, 5.330, "Utrechtse Heuvelrug forest and heath"),
    S("ee-lahemaa", "ee", 59.550, 25.850, "Lahemaa boreal forest and bog (3 x 3 km of lidar)", ew="world-estonia-lahemaa"),
    S("ee-kivioli", "ee", 59.380, 27.005, "Kivioli forest, clearcuts and fields (3 x 3 km)", ew="world-estonia-kivioli-north"),
    S("ee-tartu", "ee", 58.380, 26.720, "Tartu farmland and forest edge (3 x 3 km)", ew="world-estonia-tartu"),
    S("us-eaton", "werk", 34.200, -118.100, "Eaton fire (Jan 2025): Altadena and San Gabriel foothill chaparral",
      ("ca-losangeles-1-b23", "ca-lapostwildfire-eaton-c25")),
    S("us-palisades", "werk", 34.070, -118.560, "Palisades fire (Jan 2025): Santa Monica Mountains chaparral",
      ("ca-losangeles-1-b23", "ca-lapostwildfire-palisades-c25")),
    S("us-sierra-teak", "werk10", 37.006, -119.006, "Sierra Nevada mixed conifer (NEON TEAK)"),
    S("us-sjer", "werk10", 37.109, -119.732, "oak savanna, Sierra foothills (NEON SJER)"),
    S("us-redwood", "werk10", 40.330, -123.980, "Humboldt Redwoods, very tall forest"),
    S("us-los-padres", "werk10", 34.600, -119.750, "Los Padres chaparral, Santa Ynez Mountains"),
    S("us-mojave", "werk10", 35.100, -116.500, "Mojave desert shrub"),
    S("us-central-valley", "werk10", 36.600, -120.300, "Central Valley crops and orchards"),
]


def rc_of(lat, lon):
    r, c = int((AEF_Y0 - lat) / AEF_RES), int((lon - AEF_X0) / AEF_RES)
    return [r - HALF, r + HALF, c - HALF, c + HALF]


def write_sites():
    picks = {s["name"]: {"rc": rc_of(s["lat"], s["lon"]), "center": [s["lon"], s["lat"]], "kind": "lidar",
                         "source": s["src"], "note": s["note"]} for s in SITES}
    json.dump(picks, open(HERE / "data" / "lidar_squares.json", "w"), indent=1)
    return picks


def mid_year(a, b):
    a, b = (d if isinstance(d, D) else D.fromisoformat(str(d)[:10]) for d in (a, b))
    return (a + (b - a) / 2).year


# ---- sampling a 2 m grid onto the square's AlphaEarth pixels ----

def sample(arr, tr, crs, rc):
    """arr (h, w) on a grid with transform tr in crs -> mean, p90 and cover (%) per pixel of rc, 5 x 5 samples."""
    from pyproj import Transformer
    from scipy import ndimage
    y0, y1, x0, x1 = rc
    sub = (np.arange(5) + 0.5) / 5
    cols = x0 + (np.arange(x1 - x0)[:, None] + sub[None]).ravel()
    rows = y0 + (np.arange(y1 - y0)[:, None] + sub[None]).ravel()
    out_m = np.full((y1 - y0, x1 - x0), np.nan, np.float32)
    out_p = out_m.copy()
    out_c = np.zeros(out_m.shape, np.uint8)
    T = Transformer.from_crs(4326, crs, always_xy=True)
    for b in range(0, rows.size, 500):  # 100 pixel rows at a time
        LON, LAT = np.meshgrid(AEF_X0 + cols * AEF_RES, AEF_Y0 - rows[b:b + 500] * AEF_RES)
        X, Y = T.transform(LON, LAT)
        cc, rr = (~tr) * (X, Y)
        v = ndimage.map_coordinates(arr, [rr - 0.5, cc - 0.5], order=0, mode="constant", cval=np.nan)
        v = v.reshape(-1, 5, x1 - x0, 5).transpose(0, 2, 1, 3).reshape(-1, x1 - x0, 25)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            out_m[b // 5:b // 5 + v.shape[0]] = np.nanmean(v, -1)
            out_p[b // 5:b // 5 + v.shape[0]] = np.nanpercentile(v, 90, axis=-1)
        out_c[b // 5:b // 5 + v.shape[0]] = (np.isfinite(v).mean(-1) * 100).astype(np.uint8)
    return out_m.ravel(), out_p.ravel(), out_c.ravel()


def chm(dsm, dtm):
    h = dsm - dtm
    h[(h < -2) | (h > 120)] = np.nan  # misregistered water, birds, wires
    return np.clip(h, 0, None)


# ---- sources ----

def read_nz(site, ws):
    import world_nz as nz
    crs = nz.CRS_NZ
    grid = ws.window(site, crs, half_km=3.3)
    out = []
    for survey in site["pair"]:
        it = nz.items(survey)
        pad = 0.08
        urls = [u for u, (w, s, e, n) in it["tiles"] if w - pad < site["lon"] < e + pad and s - pad < site["lat"] < n + pad]
        with ThreadPoolExecutor(2) as ex:
            dtm, dsm = ex.map(lambda us: ws.read_cogs(us, crs, grid)[0], [urls, [u.replace("/dem_1m/", "/dsm_1m/") for u in urls]])
        out.append((chm(dsm, dtm), (it["start"], it["end"])))
    from rasterio.transform import from_origin
    W, S_, E, N, w, h = grid
    return out, from_origin(W, N, ws.STEP, ws.STEP), crs


def read_ca(site, ws):
    import world_canada as wc
    crs = wc.utm(site)
    grid = ws.window(site, crs, half_km=3.3)
    out = []
    for p in site["pair"]:
        dtm, tr = ws.read_cogs([f"{wc.BUCKET}/{p}-dtm.tif"], crs, grid)
        dsm, _ = ws.read_cogs([f"{wc.BUCKET}/{p}-dsm.tif"], crs, grid)
        a, b = wc.PROJECTS[p]
        out.append((chm(dsm, dtm), (a.isoformat(), b.isoformat())))
    return out, tr, crs


def read_nl(site, ws):
    import pyogrio
    import shapely
    import world_ahn as ahn
    grid = ws.window(site, ahn.CRS_RD, half_km=3.3)
    W, S_, E, N = grid[:4]
    df = pyogrio.read_dataframe(ahn.aux("AUX/bladwijzer.gpkg"), columns=["AHN", "AHN5_05M_M", "AHN5_05M_R"], bbox=(W, S_, E, N))
    df = df[df.intersects(shapely.box(W, S_, E, N))]
    have4 = ahn.pdok()
    dsm4 = ahn.PDOK.replace("dtm_05m", "dsm_05m")
    u4t = [ahn.full(f"/vsicurl/{ahn.PDOK}/M_{r.AHN}.tif") for r in df.itertuples() if r.AHN in have4]
    u4s = [ahn.full(f"/vsicurl/{dsm4}/R_{r.AHN}.tif") for r in df.itertuples() if r.AHN in have4]
    fix = lambda u: ahn.full("/vsicurl/" + u.replace("https://basisdata.nl/hwh-ahn", ahn.HOST))
    both = [r for r in df.itertuples() if isinstance(r.AHN5_05M_M, str) and isinstance(r.AHN5_05M_R, str)]
    u5t = [fix(r.AHN5_05M_M) for r in both]
    u5s = [fix(r.AHN5_05M_R) for r in both]
    (a, b, c, d), tr = ahn.read([(u4t, ahn.CRS_RD), (u4s, ahn.CRS_RD), (u5t, ahn.CRS_RD_NAP), (u5s, ahn.CRS_RD_NAP)], grid)
    d4, d5 = ahn.flown(4, grid), ahn.flown(5, grid)
    return [(chm(b, a), tuple(x.isoformat() for x in d4)), (chm(d, c), tuple(x.isoformat() for x in d5))], tr, ahn.CRS_RD


def grid_surface(paths, grid, step):
    """Highest non-noise return and mean ground return per cell -> canopy height, and the flight days."""
    import laspy
    W, S_, E, N, w, h = grid
    top = np.full(h * w, -np.inf)
    gsum = np.zeros(h * w)
    gcnt = np.zeros(h * w, np.int64)
    days = {}
    for p in paths:
        with laspy.open(p) as f:
            for pts in f.chunk_iterator(2_000_000):
                cl = np.asarray(pts.classification)
                x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
                c = np.floor((x - W) / step).astype(np.int64)
                r = np.floor((N - y) / step).astype(np.int64)
                ok = (c >= 0) & (c < w) & (r >= 0) & (r < h) & (cl != 7) & (cl != 18)
                i = r[ok] * w + c[ok]
                np.maximum.at(top, i, z[ok])
                g = cl[ok] == 2
                gsum += np.bincount(i[g], z[ok][g], h * w)
                gcnt += np.bincount(i[g], None, h * w)
                d = np.floor((np.asarray(pts.gps_time)[ok] + 1e9) / 86400).astype(np.int64)
                u, k = np.unique(d, return_counts=True)
                for a_, b_ in zip(u.tolist(), k.tolist()):
                    days[a_] = days.get(a_, 0) + b_
    from scipy import ndimage
    with np.errstate(invalid="ignore", divide="ignore"):
        gz = (gsum / gcnt).reshape(h, w)
    gz[gcnt.reshape(h, w) == 0] = np.nan
    hole = np.isnan(gz)
    if hole.any() and (~hole).any():
        dist, (ri, ci) = ndimage.distance_transform_edt(hole, return_indices=True)
        fill = hole & (dist <= 10)  # ground under dense canopy: nearest ground within 20 m
        gz[fill] = gz[ri[fill], ci[fill]]
    tz = top.reshape(h, w)
    tz[~np.isfinite(tz)] = np.nan
    tot = sum(days.values())
    keep = sorted(d for d, k in days.items() if k >= 0.005 * tot)
    gps0 = datetime.datetime(1980, 1, 6)
    span = tuple((gps0 + datetime.timedelta(days=d)).date().isoformat() for d in (keep[0], keep[-1]))
    return chm(tz.astype(np.float32), gz.astype(np.float32)), span


def read_ee(site, ws):
    import world_estonia as ee
    from rasterio.transform import from_origin
    s0 = next(s for s in ee.SITES if s["name"] == site["ew"])
    grid = ws.window(site, ee.CRS_EE, half_km=3.3)
    out = []
    for year, product in (s0["first"], s0["second"]):
        paths = [os.path.join(ee.DIR, f"{sh}_{year}_{product}.laz") for sh, _, _ in ee.sheet_ids(s0)]
        paths = [p for p in paths if os.path.exists(p)]
        out.append(grid_surface(paths, grid, ws.STEP))
    W, S_, E, N, w, h = grid
    return out, from_origin(W, N, ws.STEP, ws.STEP), ee.CRS_EE


WERK = "https://scil-data.sdsc.edu/data/nasa-werk/stac"


def werk_tiles(col, box_utm):
    import requests
    import shapely
    s = requests.Session()
    c = s.get(f"{WERK}/nasa-werk-dem-{col}/collection.json", timeout=60).json()
    hrefs = [l["href"] for l in c["links"] if l["rel"] == "item"]
    if "losangeles" in col:  # 1 km tile names carry their corner (..._11slu039000383700_ -> 390000, 3837000)
        def cell(h):
            m = re.search(r"_11s[a-z]{2}(\d{6})(\d{6})_", h)
            x, y = int(m.group(1)) * 10, int(m.group(2)) * 10
            return shapely.box(x, y, x + 1000, y + 1000)
        hrefs = [h for h in hrefs if cell(h).intersects(box_utm)]
    with ThreadPoolExecutor(16) as ex:
        items = list(ex.map(lambda h: s.get(f"{WERK}/nasa-werk-dem-{col}/{h[2:]}", timeout=60).json(), hrefs))
    keep = [i for i in items if shapely.box(*i["properties"]["proj:bbox"]).intersects(box_utm)]
    dates = sorted(i["properties"].get("datetime") or i["properties"].get("start_datetime") for i in keep)
    return [f"vrt:///vsicurl/{i['assets']['data']['href']}?bands=3&ovr=NONE" for i in keep], (dates[0][:10], dates[-1][:10])


def read_werk(site, ws):
    import shapely
    from rasterio.crs import CRS
    crs = CRS.from_epsg(6340)
    grid = ws.window(site, crs, half_km=3.3)
    box = shapely.box(*grid[:4])
    out = []
    for col in site["pair"]:
        urls, span = werk_tiles(col, box)
        a, tr = ws.read_cogs(urls, crs, grid)
        a[(a < -2) | (a > 120)] = np.nan
        out.append((np.clip(a, 0, None), span))
    return out, tr, crs


W10 = "https://scil-data.sdsc.edu/data/nasa-werk"


def read_werk10(site, ws):
    from rasterio.crs import CRS
    crs = CRS.from_epsg(3310)
    grid = ws.window(site, crs, half_km=3.3)
    a, tr = ws.read_cogs([f"/vsicurl/{W10}/statewide-10m/california_statewide_10m_chm_epsg3310.tif"], crs, grid,
                         nodata_below=-1000)
    a[(a < -2) | (a > 120)] = np.nan
    yr, _ = ws.read_cogs([f"/vsicurl/{W10}/california_statewide_10m_epsg3310_campaign_year.tif"], crs, grid)
    yr[yr <= 0] = np.nan
    return [(np.clip(a, 0, None), None)], tr, crs, yr


def one(site, ws, rc):
    t0 = time.time()
    src = site["src"]
    yr_px = None
    if src == "werk10":
        epochs, tr, crs, yr = read_werk10(site, ws)
    else:
        epochs, tr, crs = {"nz": read_nz, "ca": read_ca, "nl": read_nl, "ee": read_ee, "werk": read_werk}[src](site, ws)
    ms, ps, cs, years, dates = [], [], [], [], []
    for a, span in epochs:
        m, p, c = sample(a, tr, crs, rc)
        ms.append(m)
        ps.append(p)
        cs.append(c)
        if span:
            years.append(mid_year(*span))
            dates.append(list(span))
    out = dict(chm=np.stack(ms).astype(np.float16), chm_p90=np.stack(ps).astype(np.float16), cover=np.stack(cs),
               source=src, note=site["note"])
    if src == "werk10":
        ym, _, _ = sample(yr, tr, crs, rc)
        out["year_px"] = np.where(np.isfinite(ym), np.round(ym), 0).astype(np.int16)
        out["year"] = np.array([0], np.int16)
        out["dates"] = np.array([["", ""]])
    else:
        out["year"] = np.array(years, np.int16)
        out["dates"] = np.array(dates)
    np.savez(OUT / f"{site['name']}.npz", **out)
    cov = [float((c >= 80).mean()) for c in cs]
    med = [float(np.nanmedian(m)) if np.isfinite(m).any() else None for m in ms]
    return (f"{site['name']}: years {out['year'].tolist()} cover>=80% {[round(c, 2) for c in cov]} "
            f"median height {[None if v is None else round(v, 1) for v in med]}  {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-only", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    picks = write_sites()
    print(f"wrote data/lidar_squares.json ({len(picks)} squares)", flush=True)
    if args.sites_only:
        return
    sys.path.insert(0, str(EW))
    os.chdir(EW)
    import psutil
    import rasterio
    import world_sites as ws
    OUT.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    todo = [s for s in SITES if (not args.only or s["name"] in args.only) and not (OUT / f"{s['name']}.npz").exists()]

    def run(s):
        try:
            with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_CACHEMAX=512):
                msg = one(s, ws, picks[s["name"]]["rc"])
        except Exception as e:  # noqa: BLE001
            msg = f"{s['name']} FAILED: {e!r}\n{traceback.format_exc(limit=3)}"
        gb = proc.memory_info().rss / 1e9
        if gb > args.max_gb:
            print(f"STOP: RSS {gb:.1f} GB", flush=True)
            os._exit(3)
        return msg + f"  [{gb:.1f} GB]"
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(run, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
