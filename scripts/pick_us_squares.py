"""Pick 6 km squares for checks against two biomass maps that do not come from GEDI or CTrees.

  rap        US rangeland under the Rangeland Analysis Platform (RAP v3: yearly herbaceous biomass and cover, 30 m):
             tallgrass, mixed and shortgrass prairie, desert grassland, California annual grassland, sagebrush,
             desert shrub and oak-juniper savanna. Searched over a box 0.25 degrees either side of the point, as
             pick_veg_squares.py does. sandhills-us and great-basin-us (veg_squares.json) are under RAP too.
  chaparral  southern California shrubland under the SoCal annual biomass map (Schrader-Patton et al., Dryad
             doi:10.5061/dryad.qz612jmjt, 2001 to 2023). Five fire scars from 2017 to 2022, each square placed
             where it lies most inside the fire's MTBS perimeter (Valley 91 %, the rest 100 %; MTBS 1984 to 2024
             on Source Cooperative), and one place last burned in 2007 (Witch; Cedar 2003), searched over a box
             0.05 degrees either side.

Writes data/us_squares.json: name -> rc, center, kind, group (rap / chaparral), note, WorldCover shares.

    uv run --project ~/dev/projects/aef-tessera-earthwork python scripts/pick_us_squares.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import pick_veg_squares as pv  # noqa: E402

OUT = HERE / "data" / "us_squares.json"
pv.KINDS["chaparral"] = {("shrub", "trees", "grass"): 0.97}

# name: (lon, lat, kind, group, box, note)
REGIONS = {
    "flint-hills-ks": (-96.5, 38.4, "grass", "rap", 0.25, "tallgrass prairie, burned and grazed"),
    "mixed-grass-nd": (-101.5, 46.6, "grass", "rap", 0.25, "northern mixed-grass prairie; 2021 drought"),
    "shortgrass-co": (-104.2, 40.7, "grass", "rap", 0.25, "Pawnee shortgrass steppe"),
    "rolling-plains-tx": (-100.3, 33.3, "grass", "rap", 0.25, "southern mixed-grass; 2022 drought"),
    "san-rafael-az": (-110.6, 31.45, "grass", "rap", 0.25, "desert grassland, monsoon-driven"),
    "foothill-annual-ca": (-120.5, 37.4, "grass", "rap", 0.25, "California annual grassland, rain-year swings"),
    "sagebrush-wy": (-108.2, 42.3, "shrub", "rap", 0.25, "Wyoming big sagebrush"),
    "snake-river-id": (-115.8, 43.1, "shrub", "rap", 0.25, "sagebrush and cheatgrass, frequent fire"),
    "mojave-ca": (-116.0, 35.3, "shrub", "rap", 0.25, "creosote desert shrub"),
    "edwards-plateau-tx": (-99.9, 30.4, "savanna", "rap", 0.25, "oak-juniper savanna"),
    "thomas-fire-ca": (-119.0095, 34.4279, "chaparral", "chaparral", None, "Thomas Fire, Dec 2017: regrowth 2018 on"),
    "woolsey-fire-ca": (-118.6988, 34.1953, "chaparral", "chaparral", None, "Woolsey Fire, Nov 2018"),
    "holy-fire-ca": (-117.4747, 33.7003, "chaparral", "chaparral", None, "Holy Fire, Aug 2018"),
    "bobcat-fire-ca": (-117.8529, 34.3936, "chaparral", "chaparral", None, "Bobcat Fire, Sep 2020"),
    "valley-fire-ca": (-116.6723, 32.7221, "chaparral", "chaparral", None, "Valley Fire, Sep 2020"),
    "fairview-fire-ca": (-116.8295, 33.6558, "chaparral", "chaparral", None, "Fairview Fire, Sep 2022"),
    "cleveland-steady-ca": (-116.75, 33.05, "chaparral", "chaparral", 0.05, "chaparral last burned 2007 (Witch; Cedar 2003)"),
}


def main():
    out = json.load(open(OUT)) if OUT.exists() else {}
    for name, (lon, lat, kind, group, box, note) in REGIONS.items():
        if name in out:
            continue
        try:
            if box is None:  # fixed center (inside a fire perimeter)
                r, c = int((pv.AEF_Y0 - lat) / pv.AEF_RES), int((lon - pv.AEF_X0) / pv.AEF_RES)
                p = {"rc": [r - pv.HALF, r + pv.HALF, c - pv.HALF, c + pv.HALF], "center": [lon, lat], "kind": kind}
            else:
                pv.BOX = box
                p = pv.pick(lon, lat, kind)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed {e!r}", flush=True)
            continue
        p |= {"group": group, "note": note}
        out[name] = p
        print(f"{name:22s} {kind:10s} {p['center']}  {p.get('wc2021', 'fixed center')}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
