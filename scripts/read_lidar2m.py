"""More airborne lidar canopy height, read at 2 m from the start (section 16 read full resolution and took over an
hour a square). Canopy height = surface model minus ground model of the same program, or the program's own canopy
height model, on a 2 m grid in the source's CRS, then sampled 5 x 5 per AlphaEarth pixel by read_lidar.sample:
  fr     IGN LiDAR HD canopy height (MNH), one GetMap at 2 m from the Geoplateforme WMS-R (the server averages);
         flight days from the WFS tile metadata. One epoch (France is flown once, 2021 to 2025).
  gb     Scotland's SRSP Phase 4 (2018) and Phase 5 (2020) 0.5 m DSM and DTM COGs, each opened at its 2 m
         overview; flight days from the first point of each 1 km LAZ tile (Earthwork's world_scotland.flown).
  ca     NRCan HRDEM 1 m DSM and DTM COGs of two projects, opened at their 2 m overview and warped onto UTM.
  nl     AHN4 (PDOK) and AHN5 0.5 m DSM and DTM, opened at their 2 m overview; dates from the flight swaths.
  no     Kartverket's per-project DOM and DTM (hoydedata.no ImageServers, CC BY 4.0), exported at 2 m in 1 km blocks
         with the mosaic held to one lidar project (LAS_PROJECT_ID): Oslo 2019 and Oslo 2024, the pair Earthwork
         reads; each flight is the 90 days up to its last flight day.
  ch     swisstopo swissSURFACE3D raster (0.5 m DSM; releases 2018 and 2024) at its 2 m overview, minus swissALTI3D
         2 m. Written, not in SITES: the Sihlwald read passed 20 minutes (forty-nine 1 km tiles per release) and was
         stopped.
  nl     note: AHN4's DSM read through its 2 m overview halves the canopy (median 8.4 m where AHN5 reads 17.3 and
         Meta over 15 m), so only the AHN5 epoch of the Heuvelrug square is kept (done after the read).
Overviews instead of full resolution: a few centimeters of overview error matters for bare-ground change (why
Earthwork reads full resolution) but not for canopy height in meters.

Output as read_lidar.py: data/lidar/<name>.npz (chm, chm_p90, cover, year, dates, source, note), and the picks in
data/lidar2_squares.json. Each site is read in its own process (--only), so a driver can stop one that runs long.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil python scripts/read_lidar2m.py --sites-only
    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil python scripts/read_lidar2m.py --only lidar-fr-landes
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(HERE / "scripts"))
import read_lidar as rl  # noqa: E402

OUT = HERE / "data" / "lidar"
PICKS = HERE / "data" / "lidar2_squares.json"
D = datetime.date


def S(name, src, lat, lon, note, **kw):
    return dict(name=f"lidar-{name}", src=src, lat=lat, lon=lon, note=note, **kw)


SITES = [
    S("fr-landes", "fr", 44.20, -0.90, "Landes pine plantations, clearcuts and regrowth (one flight)"),
    S("gb-devilla", "gb", 56.072, -3.690, "Devilla Forest conifer plantation and farmland, Fife", phases=(4, 5)),
    S("ca-timmins-dome", "ca", 48.440, -81.212, "Timmins: boreal forest, the Dome mine and town (the pair Earthwork "
      "checked for overlap)", pair=rl.TIMMINS),
    S("nl-heuvelrug", "nl", 52.060, 5.330, "Utrechtse Heuvelrug forest and heath"),
    S("no-holmenkollen", "no", 59.965, 10.67, "Holmenkollen and Nordmarka's edge, Oslo: forest, suburbs, ski "
      "slopes", pair=(1174, 6346)),
]
NOT_READ = [S("ch-sihlwald", "ch", 47.255, 8.555, "Sihlwald and Albis mixed forest, Zurich")]


def write_sites():
    picks = {s["name"]: {"rc": rl.rc_of(s["lat"], s["lon"]), "center": [s["lon"], s["lat"]], "kind": "lidar",
                         "source": s["src"], "note": s["note"]} for s in SITES}
    json.dump(picks, open(PICKS, "w"), indent=1)
    return picks


def read_ovr(urls, crs, grid, step=2.0, nodata_below=-1000):
    """The rasters at urls mosaicked onto the 2 m grid, each opened at the overview level nearest 2 m (not
    finer than the source), warped when its CRS differs: float32, NaN none."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds
    W, S_, E, N, w, h = grid
    out = np.full((h, w), np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="5",
                      GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.vrt"):
        for u in urls:
            with rasterio.open(u) as s0:
                res0 = abs(s0.transform.a)
                ovr = s0.overviews(1)
                lvl = None
                for i, f in enumerate(ovr):
                    if res0 * f <= step * 1.01:
                        lvl = i
            s = rasterio.open(u, overview_level=lvl) if lvl is not None else rasterio.open(u)
            with s:
                src = s if s.crs == crs else WarpedVRT(s, crs=crs, resampling=Resampling.bilinear)
                b = src.bounds
                x0, x1 = max(W, b.left), min(E, b.right)
                y0, y1 = max(S_, b.bottom), min(N, b.top)
                if x1 - x0 < step or y1 - y0 < step:
                    continue
                c0, c1 = int(np.ceil((x0 - W) / step)), int(np.floor((x1 - W) / step))
                r0, r1 = int(np.ceil((N - y1) / step)), int(np.floor((N - y0) / step))
                if c1 <= c0 or r1 <= r0:
                    continue
                win = from_bounds(W + c0 * step, N - r1 * step, W + c1 * step, N - r0 * step, src.transform)
                a = src.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.average,
                             masked=True).astype(np.float32).filled(np.nan)
                a[a < nodata_below] = np.nan
                a[a > 1e30] = np.nan
                cur = out[r0:r1, c0:c1]
                fill = np.isnan(cur) & np.isfinite(a)
                cur[fill] = a[fill]
    return out, from_origin(W, N, step, step)


# ---- sources ----

def read_fr(site, ws):
    import rasterio
    import requests
    from pyproj import Transformer
    from rasterio.crs import CRS
    crs = CRS.from_epsg(2154)
    grid = ws.window(site, crs, half_km=3.3)
    W, S_, E, N, w, h = grid
    raw = HERE / "data" / "lidar_raw" / "fr"
    raw.mkdir(parents=True, exist_ok=True)
    f = raw / f"{site['name']}_mnh_2m.tif"
    if not f.exists():
        p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", STYLES="", CRS="EPSG:2154", BBOX=f"{W},{S_},{E},{N}",
                 LAYERS="IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", WIDTH=w, HEIGHT=h,
                 FORMAT="image/geotiff")
        for k in range(5):
            r = requests.get("https://data.geopf.fr/wms-r", params=p, timeout=600)
            if r.status_code == 200 and "tiff" in r.headers.get("content-type", ""):
                break
            time.sleep(2 + 3 * k)
        else:
            raise RuntimeError(f"MNH GetMap failed: {r.status_code} {r.text[:200]}")
        f.write_bytes(r.content)
    with rasterio.open(f) as ds:
        a = ds.read(1).astype(np.float32)
    a[(a < -2) | (a > 120)] = np.nan
    a = np.clip(a, 0, None)
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = inv.transform([W, E, W, E], [S_, S_, N, N])
    q = dict(SERVICE="WFS", VERSION="2.0.0", REQUEST="GetFeature", TYPENAMES="IGNF_LIDAR-HD_METADONNEE:metadata",
             OUTPUTFORMAT="application/json", COUNT=1000,
             BBOX=f"{min(lat)},{min(lon)},{max(lat)},{max(lon)},urn:ogc:def:crs:EPSG::4326",
             PROPERTYNAME="date_debut_acquisition,date_fin_acquisition")
    P = [ft["properties"] for ft in requests.get("https://data.geopf.fr/wfs/ows", params=q, timeout=120).json()["features"]]
    d0 = min(p["date_debut_acquisition"][:10] for p in P)
    d1 = max(p["date_fin_acquisition"][:10] for p in P)
    from rasterio.transform import from_origin
    return [(a, (d0, d1))], from_origin(W, N, 2.0, 2.0), crs


def gb_keys(n, sq, kind):
    import requests
    out, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": f"lidar/phase-{n}/{kind}/27700/gridded/{sq}"}
        if tok:
            q["continuation-token"] = tok
        t = requests.get("https://srsp-open-data.s3.eu-west-2.amazonaws.com/", params=q, timeout=60).text
        out += re.findall(r"<Key>([^<]*)</Key>", t)
        m = re.search(r"<NextContinuationToken>([^<]*)<", t)
        if not m:
            return set(out)
        tok = m.group(1)


def read_gb(site, ws):
    import world_scotland as sc
    crs = sc.CRS_GB
    grid = ws.window(site, crs, half_km=3.3)
    quarters = sc.tiles(grid, 5000, sc.quarter)
    out = []
    for n in site["phases"]:
        keys = set()
        for sq in {q[:2] for q in quarters}:
            keys |= gb_keys(n, sq, "dsm") | gb_keys(n, sq, "dtm")
        dsm = [f"/vsicurl/{sc.BUCKET}/{k}" for k in sorted(keys) if "/dsm/" in k and k.split("/")[-1][:6] in quarters]
        dtm = [f"/vsicurl/{sc.BUCKET}/{k}" for k in sorted(keys) if "/dtm/" in k and k.split("/")[-1][:6] in quarters]
        a, tr = read_ovr(dsm, crs, grid)
        b, _ = read_ovr(dtm, crs, grid)
        (d0, d1), how = sc.flown(dict(site, name=f"lidar2-{site['name']}"), n, grid)
        out.append((rl.chm(a, b), (d0.isoformat(), d1.isoformat())))
    return out, tr, crs


def read_ca(site, ws):
    import world_canada as wc
    crs = wc.utm(site)
    grid = ws.window(site, crs, half_km=3.3)
    out = []
    for p in site["pair"]:
        a, tr = read_ovr([f"{wc.BUCKET}/{p}-dsm.tif"], crs, grid)
        b, _ = read_ovr([f"{wc.BUCKET}/{p}-dtm.tif"], crs, grid)
        d0, d1 = wc.PROJECTS[p]
        out.append((rl.chm(a, b), (d0.isoformat(), d1.isoformat())))
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
    u4t = [f"/vsicurl/{ahn.PDOK}/M_{r.AHN}.tif" for r in df.itertuples() if r.AHN in have4]
    u4s = [f"/vsicurl/{dsm4}/R_{r.AHN}.tif" for r in df.itertuples() if r.AHN in have4]
    fix = lambda u: "/vsicurl/" + u.replace("https://basisdata.nl/hwh-ahn", ahn.HOST)
    both = [r for r in df.itertuples() if isinstance(r.AHN5_05M_M, str) and isinstance(r.AHN5_05M_R, str)]
    u5t, u5s = [fix(r.AHN5_05M_M) for r in both], [fix(r.AHN5_05M_R) for r in both]
    out = []
    for (ut, us, crs_h), ver in (((u4t, u4s, ahn.CRS_RD), 4), ((u5t, u5s, ahn.CRS_RD_NAP), 5)):
        a, tr = read_ovr(us, ahn.CRS_RD, grid)
        b, _ = read_ovr(ut, ahn.CRS_RD, grid)
        d = ahn.flown(ver, grid)
        out.append((rl.chm(a, b), tuple(x.isoformat() for x in d)))
    return out, tr, ahn.CRS_RD


def stac(collection, bbox):
    """{tile: {release year: [asset hrefs]}}, as world_swiss.stac, skipping items whose id is not
    collection_year_tile."""
    import collections

    import requests
    out = collections.defaultdict(dict)
    body = {"collections": [collection], "bbox": list(bbox), "limit": 100}
    url, method = "https://data.geo.admin.ch/api/stac/v1/search", "POST"
    while url:
        r = requests.post(url, json=body, timeout=120) if method == "POST" else requests.get(url, timeout=120)
        d = json.loads(r.text, strict=False)
        for f in d.get("features", []):
            parts = f["id"].split("_")
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            out[parts[2]][int(parts[1])] = [a["href"] for a in f["assets"].values()]
        nxt = [lk for lk in d.get("links", []) if lk["rel"] == "next"]
        url = nxt[0]["href"] if nxt else None
        method = nxt[0].get("method", "GET") if nxt else None
        body = nxt[0].get("body", body) if nxt else None
    return out


def read_ch(site, ws):
    from pyproj import Transformer
    import world_swiss as sw
    crs = sw.CRS_CH
    grid = ws.window(site, crs, half_km=3.3)
    W, S_, E, N, _, _ = grid
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    xs, ys = inv.transform([W, E, W, E], [S_, S_, N, N])
    bbox = (min(xs) - 1e-3, min(ys) - 1e-3, max(xs) + 1e-3, max(ys) + 1e-3)
    alti = stac("ch.swisstopo.swissalti3d", bbox)
    surfr = stac("ch.swisstopo.swisssurface3d-raster", bbox)
    cloud = stac("ch.swisstopo.swisssurface3d", bbox)
    out = []
    for dsm_year, dtm_years, days in ((2018, sw.FIRST, range(2016, 2020)), (2024, sw.SECOND, range(2023, 2026))):
        A, B, dd = [], [], []
        for tile, rel in sorted(surfr.items()):
            e, n = (int(v) * 1000 for v in tile.split("-"))
            if e + 1000 <= W or e >= E or n + 1000 <= S_ or n >= N or dsm_year not in rel:
                continue
            A.append("/vsicurl/" + next(h for h in rel[dsm_year] if h.endswith(".tif")))
            ya = [y for y in dtm_years if y in alti.get(tile, {})]
            if ya:
                B.append("/vsicurl/" + next(h for h in alti[tile][ya[-1]] if h.endswith("_2_2056_5728.tif")))
            d = sw.tile_days(tile, cloud, days)
            if d:
                dd.append(d)
        a, tr = read_ovr(A, crs, grid)
        b, _ = read_ovr(B, crs, grid)
        span = sw.span(dd) if dd else sw.year_days(dsm_year)
        out.append((rl.chm(a, b), tuple(x.isoformat() for x in span)))
    return out, tr, crs


def no_block(svc, pid, box, n, crs):
    import io

    import rasterio
    import requests
    W, S_, E, N = box
    p = {"bbox": f"{W},{S_},{E},{N}", "bboxSR": crs.to_epsg(), "imageSR": crs.to_epsg(), "size": f"{n},{n}",
         "format": "tiff", "pixelType": "F32", "noData": -9999, "interpolation": "RSP_BilinearInterpolation",
         "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicNone", "where": f"LAS_PROJECT_ID={pid}"}), "f": "image"}
    for k in range(6):
        try:
            r = requests.get(f"https://hoydedata.no/arcgis/rest/services/{svc}/ImageServer/exportImage", params=p, timeout=300)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    else:
        raise RuntimeError(f"exportImage failed: {svc} {pid}")
    with rasterio.MemoryFile(io.BytesIO(r.content)) as m, m.open() as s:
        a = s.read(1).astype(np.float32)
    a[a <= -9998] = np.nan
    return a


def read_no(site, ws):
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    crs = CRS.from_epsg(25833)
    grid = ws.window(site, crs, half_km=3.3)
    W, S_, E, N, w, h = grid
    nb = 500  # 1 km blocks of 2 m cells
    out = []
    for pid in site["pair"]:
        q = requests.get("https://hoydedata.no/arcgis/rest/services/Prosjekt_DTM/ImageServer/query",
                         params={"where": f"LAS_PROJECT_ID={pid}", "outFields": "SISTEFLYDATO", "returnGeometry": "false",
                                 "resultRecordCount": 1, "f": "json"}, timeout=120).json()
        last = datetime.datetime.fromtimestamp(q["features"][0]["attributes"]["SISTEFLYDATO"] / 1000, datetime.UTC).date()
        arrs = {}
        for svc in ("Prosjekt_DOM", "Prosjekt_DTM"):
            a = np.full((h, w), np.nan, np.float32)

            def blk(rc, a=a, svc=svc):
                r0, c0 = rc
                x0, y1 = W + c0 * 2.0, N - r0 * 2.0
                b = no_block(svc, pid, (x0, y1 - nb * 2.0, x0 + nb * 2.0, y1), nb, crs)
                a[r0:r0 + nb, c0:c0 + nb] = b[:min(nb, h - r0), :min(nb, w - c0)]
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(blk, [(r0, c0) for r0 in range(0, h, nb) for c0 in range(0, w, nb)]))
            arrs[svc] = a
        out.append((rl.chm(arrs["Prosjekt_DOM"], arrs["Prosjekt_DTM"]),
                    ((last - datetime.timedelta(days=90)).isoformat(), last.isoformat())))
    return out, from_origin(W, N, 2.0, 2.0), crs


def one(site, ws, rc):
    t0 = time.time()
    epochs, tr, crs = {"fr": read_fr, "gb": read_gb, "ca": read_ca, "nl": read_nl, "ch": read_ch, "no": read_no}[site["src"]](site, ws)
    ms, ps, cs, years, dates = [], [], [], [], []
    for a, span in epochs:
        m, p, c = rl.sample(a, tr, crs, rc)
        ms.append(m)
        ps.append(p)
        cs.append(c)
        years.append(rl.mid_year(*span))
        dates.append(list(span))
    np.savez(OUT / f"{site['name']}.npz", chm=np.stack(ms).astype(np.float16), chm_p90=np.stack(ps).astype(np.float16),
             cover=np.stack(cs), source=site["src"], note=site["note"], year=np.array(years, np.int16),
             dates=np.array(dates))
    cov = [round(float((c >= 80).mean()), 2) for c in cs]
    med = [round(float(np.nanmedian(m)), 1) if np.isfinite(m).any() else None for m in ms]
    return f"{site['name']}: years {years} dates {dates} cover>=80% {cov} median height {med}  {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-only", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-gb", type=float, default=12)
    args = ap.parse_args()
    picks = write_sites()
    print(f"wrote {PICKS.name} ({len(picks)} squares)", flush=True)
    if args.sites_only:
        return
    sys.path.insert(0, str(EW))
    os.chdir(EW)
    import threading

    import psutil
    import world_sites as ws
    OUT.mkdir(parents=True, exist_ok=True)
    proc, peak = psutil.Process(), [0.0]

    def guard():
        while True:
            gb = proc.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > args.max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
    for s in SITES:
        if args.only and s["name"] not in args.only:
            continue
        if (OUT / f"{s['name']}.npz").exists():
            print(f"{s['name']}: already read", flush=True)
            continue
        print(one(s, ws, picks[s["name"]]["rc"]) + f"  [peak {peak[0]:.1f} GB]", flush=True)


if __name__ == "__main__":
    main()
