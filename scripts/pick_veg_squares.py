"""Pick 6 km squares across every kind of vegetation, worldwide, from ESA WorldCover 2021.

The 37 squares of pick_squares.py were picked for tree loss, so a biomass head taught on them would only know
forest. These cover the rest of the range: grass, shrub, savanna and open woodland, crop, wetland and mangrove,
the edges of growing cities, desert edge, tundra, and intact closed forest (the high end of biomass).

For each named region: WorldCover 2021 over a box 0.25 degrees either side of the point, read at about 55 m. Every
6 km window (600 x 600 AlphaEarth pixels) is scored on how close its class shares come to the kind's target (below);
water and ground outside the tile are kept out. The best window is kept.

Writes data/veg_squares.json: name -> rc (AlphaEarth mosaic rows / cols), center, kind, WorldCover shares.

    uv run --project ~/dev/projects/aef-tessera-earthwork python scripts/pick_veg_squares.py
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter

OUT = Path(__file__).resolve().parent.parent / "data" / "veg_squares.json"
AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
BOX, STEP, HALF = 0.25, 0.0005, 300
CODES = {"trees": [10], "shrub": [20], "grass": [30], "crop": [40], "built": [50], "bare": [60], "water": [80],
         "wet": [90, 95], "moss": [100]}

# kind: {class group: target share}; score = -sum |share - target|
KINDS = {
    "grass": {"grass": 0.75},
    "shrub": {"shrub": 0.6},
    "savanna": {"trees": 0.35, ("grass", "shrub"): 0.55},
    "crop": {"crop": 0.75},
    "wetland": {"wet": 0.55},
    "city-edge": {"built": 0.35, ("crop", "grass", "shrub", "trees"): 0.6},
    "desert-edge": {"bare": 0.6, ("shrub", "grass"): 0.35},
    "tundra": {("moss", "shrub", "grass"): 0.85},
    "closed-forest": {"trees": 0.97},
}

# name: (lon, lat, kind)
REGIONS = {
    "mongolia-steppe": (106.5, 46.5, "grass"), "kazakh-steppe": (69.5, 50.5, "grass"),
    "serengeti-tz": (34.8, -2.6, "grass"), "pampas-ar": (-61.5, -35.5, "grass"),
    "sandhills-us": (-101.0, 42.0, "grass"), "mitchell-grass-au": (142.5, -21.0, "grass"),
    "tibet-cn": (91.5, 31.5, "grass"),
    "karoo-za": (21.5, -32.5, "shrub"), "great-basin-us": (-116.5, 40.0, "shrub"),
    "patagonia-ar": (-69.5, -44.5, "shrub"), "chihuahua-mx": (-105.0, 28.5, "shrub"),
    "sahel-ne": (2.5, 14.0, "shrub"), "mulga-au": (135.5, -26.0, "shrub"), "caatinga-br": (-40.0, -8.5, "shrub"),
    "cerrado-br": (-47.5, -13.5, "savanna"), "miombo-zm": (28.5, -13.5, "savanna"),
    "sudan-savanna-bf": (-1.5, 11.5, "savanna"), "top-end-au": (132.5, -14.5, "savanna"),
    "llanos-ve": (-67.5, 7.5, "savanna"), "mopane-bw": (25.5, -19.5, "savanna"),
    "dehesa-es": (-6.0, 38.8, "savanna"), "central-india": (79.5, 22.5, "savanna"),
    "iowa-us": (-93.5, 42.2, "crop"), "punjab-in": (75.5, 30.5, "crop"), "north-china-plain": (115.5, 36.5, "crop"),
    "ukraine": (33.0, 49.0, "crop"), "beauce-fr": (1.8, 48.2, "crop"), "matopiba-br": (-45.5, -11.8, "crop"),
    "ethiopia-highlands": (39.5, 9.5, "crop"), "java-rice-id": (107.5, -6.5, "crop"),
    "sundarbans-bd": (89.4, 22.0, "wetland"), "pantanal-br": (-57.0, -18.0, "wetland"),
    "okavango-bw": (22.8, -19.3, "wetland"), "sudd-ss": (30.8, 7.5, "wetland"),
    "hudson-lowlands-ca": (-84.0, 52.5, "wetland"), "niger-delta-ng": (6.2, 4.6, "wetland"),
    "everglades-us": (-80.8, 25.8, "wetland"),
    "lagos-ng": (3.3, 6.6, "city-edge"), "dhaka-bd": (90.4, 23.8, "city-edge"), "addis-et": (38.8, 9.0, "city-edge"),
    "phoenix-us": (-112.3, 33.5, "city-edge"), "hanoi-vn": (105.8, 21.0, "city-edge"),
    "cairo-east-eg": (31.6, 30.0, "city-edge"), "delhi-ncr-in": (77.2, 28.5, "city-edge"),
    "sahara-dz": (5.0, 25.5, "desert-edge"), "gobi-mn": (105.0, 43.5, "desert-edge"),
    "yamal-ru": (70.5, 67.5, "tundra"), "noatak-us": (-160.5, 67.5, "tundra"), "nunavik-ca": (-72.0, 59.5, "tundra"),
    "amazon-intact-br": (-64.5, -3.0, "closed-forest"), "congo-intact-cg": (16.5, 1.5, "closed-forest"),
    "olympic-us": (-123.8, 47.8, "closed-forest"), "borneo-intact-id": (114.5, 1.0, "closed-forest"),
    "boreal-intact-ca": (-79.0, 50.5, "closed-forest"), "tasmania-au": (146.0, -42.5, "closed-forest"),
}


def tile_of(lon, lat):
    la, lo = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}", lo, la


def read_box(lon, lat):
    """WorldCover classes over the box at ~55 m, 0 outside the center's 3-degree tile."""
    tile, lo, la = tile_of(lon, lat)
    W, S, E, N = lon - BOX, lat - BOX, lon + BOX, lat + BOX
    n = int(round(2 * BOX / STEP))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open("/vsicurl/" + WC.format(tile)) as s:
        w = from_bounds(W, S, E, N, s.transform)
        a = s.read(1, window=w, out_shape=(n, n), boundless=True, fill_value=0, resampling=Resampling.nearest)
    return a


def pick(lon, lat, kind):
    a = read_box(lon, lat)
    win = int(round(2 * HALF * AEF_RES / STEP))
    share = {g: uniform_filter(np.isin(a, c).astype(np.float32), win) for g, c in CODES.items()}
    share["none"] = uniform_filter((a == 0).astype(np.float32), win)
    score = np.zeros_like(share["none"])
    for groups, target in KINDS[kind].items():
        groups = groups if isinstance(groups, tuple) else (groups,)
        score -= np.abs(sum(share[g] for g in groups) - target)
    bad = share["none"] > 0.01
    if kind != "wetland":
        bad |= share["water"] > 0.15
    h = win // 2 + 1
    bad[:h], bad[-h:], bad[:, :h], bad[:, -h:] = True, True, True, True
    score[bad] = -np.inf
    i, j = np.unravel_index(np.argmax(score), score.shape)
    clat, clon = lat + BOX - (i + 0.5) * STEP, lon - BOX + (j + 0.5) * STEP
    r, c = int((AEF_Y0 - clat) / AEF_RES), int((clon - AEF_X0) / AEF_RES)
    return {"rc": [r - HALF, r + HALF, c - HALF, c + HALF], "center": [round(clon, 4), round(clat, 4)], "kind": kind,
            "score": round(float(score[i, j]), 3),
            "wc2021": {g: round(float(share[g][i, j]), 3) for g in CODES if share[g][i, j] >= 0.005}}


def main():
    out = json.load(open(OUT)) if OUT.exists() else {}
    for name, (lon, lat, kind) in REGIONS.items():
        if name in out:
            continue
        try:
            p = pick(lon, lat, kind)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed {e!r}", flush=True)
            continue
        out[name] = p
        print(f"{name:22s} {kind:13s} {p['center']}  score {p['score']:6.2f}  {p['wc2021']}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
