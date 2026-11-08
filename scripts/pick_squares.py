"""Pick new 6 km squares for the wider tree-loss test from Hansen's Global Forest Change (v1.12).

For each named region: Hansen loss year and tree cover 2000 in a box 0.25 degrees either side of the point; every
6 km square (600 x 600 AlphaEarth pixels, as Earthwork's ground.py) is scored by the share of its ground that lost
cover 2019 to 2024 and the share that was forest (cover >= 30 %) in 2000. The square kept is the one whose loss share
is closest to 20 % among squares at least 30 % forest, so each holds cut ground, standing forest and other land.
None of these places was used to set the frozen detectors (models/detectors.npz).

Writes data/squares.json: name -> rc (AlphaEarth mosaic rows / cols), center, driver, loss share per year.

    uv run --project ~/dev/projects/aef-tessera-earthwork python scripts/pick_squares.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW))
os.chdir(EW)
import ground  # noqa: E402
from tessera_sites import AEF_RES, AEF_X0, AEF_Y0  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "squares.json"
BOX = 0.25
HALF = 300
TARGET = 0.20

# name: (lon, lat, driver). Points kept off 10-degree Hansen tile edges.
REGIONS = {
    # clearing for farms and pasture
    "rondonia-br": (-63.0, -9.5, "commodity"), "xingu-br": (-52.0, -6.5, "commodity"),
    "mato-grosso-br": (-55.5, -11.5, "commodity"), "santa-cruz-bo": (-62.5, -16.5, "commodity"),
    "chaco-py": (-60.5, -22.0, "commodity"), "riau-id": (101.5, 0.5, "commodity"),
    "west-kalimantan-id": (110.5, 0.4, "commodity"), "sarawak-my": (113.5, 3.0, "commodity"),
    "cote-divoire": (-7.0, 6.0, "commodity"), "guaviare-co": (-72.5, 2.3, "commodity"),
    "ucayali-pe": (-74.5, -8.5, "commodity"), "campeche-mx": (-89.5, 18.3, "commodity"),
    "nicaragua": (-84.5, 13.5, "commodity"), "cambodia": (104.5, 13.0, "commodity"),
    # shifting cultivation
    "laos": (102.5, 20.4, "shifting"), "equateur-cd": (23.0, 0.5, "shifting"), "madagascar": (48.5, -18.5, "shifting"),
    "liberia": (-9.5, 6.5, "shifting"), "sagaing-mm": (95.5, 24.0, "shifting"),
    # forestry
    "arkhangelsk-ru": (42.0, 62.0, "forestry"), "irkutsk-ru": (102.0, 57.0, "forestry"),
    "british-columbia-ca": (-122.5, 54.0, "forestry"), "quebec-ca": (-71.5, 49.5, "forestry"),
    "uruguay": (-56.0, -32.5, "forestry"), "mpumalanga-za": (30.5, -25.5, "forestry"),
    "latvia": (25.5, 57.0, "forestry"), "maule-cl": (-72.0, -35.5, "forestry"), "portugal": (-8.3, 39.7, "forestry"),
    "galicia-es": (-8.0, 43.0, "forestry"), "guangxi-cn": (108.5, 23.0, "forestry"), "vietnam": (107.8, 14.5, "forestry"),
    "kyushu-jp": (131.0, 32.5, "forestry"),
    # insects and salvage, fire
    "harz-de": (10.6, 51.8, "insects"), "vysocina-cz": (15.5, 49.5, "insects"),
    "yakutia-fire-ru": (129.5, 62.5, "fire"), "manavgat-tr": (31.5, 36.9, "fire"), "nsw-fire-au": (150.0, -35.5, "fire"),
}


def pick(name, lon, lat):
    box = (lon - BOX, lat - BOX, lon + BOX, lat + BOX)
    loss, tr = ground.hansen("lossyear", *box)
    cover, _ = ground.hansen("treecover2000", *box)
    win = max(3, int(round(HALF * 2 * AEF_RES / abs(tr.e))))  # 6 km in Hansen pixels (both axes in degrees)
    recent = ((loss >= 19) & (loss <= 24)).astype(np.float32)
    forest = (cover >= 30).astype(np.float32)
    ls, fs = uniform_filter(recent, win), uniform_filter(forest, win)
    h = win // 2
    ls, fs = ls[h:-h, h:-h], fs[h:-h, h:-h]
    ok = fs >= 0.3
    if not ok.any():
        ok = fs >= fs.max() * 0.8
    score = np.where(ok, -np.abs(ls - TARGET), -np.inf)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    i, j = i + h, j + h
    clat, clon = tr.f + (i + 0.5) * tr.e, tr.c + (j + 0.5) * tr.a
    r, c = int((AEF_Y0 - clat) / AEF_RES), int((clon - AEF_X0) / AEF_RES)
    sub = loss[i - h:i + h, j - h:j + h]
    by_year = {str(2000 + y): round(float((sub == y).mean()), 3) for y in range(18, 25)}
    return {"rc": [r - HALF, r + HALF, c - HALF, c + HALF], "center": [round(clon, 4), round(clat, 4)],
            "loss_2019_2024": round(float(ls[i - h, j - h]), 3), "forest_2000": round(float(fs[i - h, j - h]), 3),
            "loss_by_year": by_year}


def main():
    out = json.load(open(OUT)) if OUT.exists() else {}
    for name, (lon, lat, driver) in REGIONS.items():
        if name in out:
            continue
        try:
            p = pick(name, lon, lat)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed {e}", flush=True)
            continue
        p["driver"] = driver
        out[name] = p
        print(f"{name:22s} {driver:10s} center {p['center']}  loss 19-24 {p['loss_2019_2024']}  forest {p['forest_2000']}  "
              f"by year {p['loss_by_year']}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
