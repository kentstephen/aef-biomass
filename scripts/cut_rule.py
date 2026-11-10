"""A cut check for the change view: where the head's own height and class say the forest was cut, the map draws a
cut instead of trusting biomass (FINDINGS 42). Which rule, at what cost?

The head gives three answers a year: biomass, canopy height and class. On a fresh cut in New Hampshire (Stoddard,
43.1536, -72.1087) height fell 12 to 2 m and the class went trees to grass between 2022 and 2023 while biomass rose
65 to 101 Mg/ha, so the map drew gain. Rules tried, from year a to year b of the window (head height at a 5 m or more):
  h3, h5, h8        the head's height fell 3, 5 or 8 m
  half              the head's height fell to under half
  h5 moved          h5 and the embedding moved past the map's cut (1 - cos >= AEF_CUT, as loss needs)
  half moved        half and moved
  class             the class went trees to grass, bare, crop or built
  class moved       class and moved
  h5 or class, moved

Scored on the 12 two-flight lidar squares over each flight window (cut_as_gain.py's sets: lost forest fell 3 m or
more from 5 m, clear cut under half, steady forest within 0.5 m, gained 2 m or more), each square weighed the same:
how much lost forest and clear cut the rule finds, how much of what the map draws as gain on lost forest it takes back,
how much of the loss the map misses it adds, and what it costs: steady forest called cut, and gained ground (and the
gain the map draws there) called cut. Then the Stoddard chunk read live: the cut patch (trees to grass with biomass up)
and the steady forest around it.

Run (RAM under 3 GB, one square at a time):
    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \\
        --with icechunk --with xarray --with requests --with torch --with zarr --with obstore python scripts/cut_rule.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import burn_check as bc  # noqa: E402
import change_confidence_test as cct  # noqa: E402
import cut_as_gain as cg  # noqa: E402
import window_test as wt  # noqa: E402

OUT = HERE / "results" / "cut_rule.json"
YEARS = cct.YEARS
CLASSES = ["trees", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "moss"]
OPEN = [CLASSES.index(c) for c in ("grass", "bare", "crop", "built")]
NH = (-72.1087, 43.1536)


def rules(hgt, cls, mv, ya, yb, idx):
    """Every rule's call on pixels idx, from year index ya to yb."""
    ha, hb = hgt[ya, idx], hgt[yb, idx]
    tall = ha >= 5
    fall = ha - hb
    moved = mv >= 1
    cl = (cls[ya, idx] == 0) & np.isin(cls[yb, idx], OPEN)
    r = {"h3": tall & (fall >= 3), "h5": tall & (fall >= 5), "h8": tall & (fall >= 8), "half": tall & (hb < 0.5 * ha)}
    r["h5 moved"] = r["h5"] & moved
    r["half moved"] = r["half"] & moved
    r["class"] = cl
    r["class moved"] = cl & moved
    r["h5 or class, moved"] = (r["h5"] | cl) & moved
    return r


def share(m, of):
    n = int(of.sum())
    return round(float((m & of).sum()) / n, 4) if n else None


def one(name, conf):
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    q = z["aef"]
    Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    fy = [int(y) for y in Ld["year"]]
    i1, i2 = [YEARS.index(min(max(y, YEARS[0]), YEARS[-1])) for y in fy]
    ok = q[:, :, 0] != -128
    agb, hgt, cls = cg.head_years(q, ok)
    chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
    m = np.all(cov >= 80, 0) & np.all(ok[i1:i2 + 1], 0)
    h1, dh = chm[0], chm[1] - chm[0]
    sets = {"lost": m & (h1 >= 5) & (dh <= -3)}
    sets["clear cut"] = sets["lost"] & (chm[1] < 0.5 * h1)
    sets["steady forest"] = m & (h1 >= 5) & (np.abs(dh) < 0.5)
    sets["gained"] = m & (dh >= 2)
    idx = np.flatnonzero(m)
    drawn = np.zeros(m.size, np.int8)
    drawn[idx] = cg.drawn(q, agb, cls, conf, i1, i2, idx)
    mv = cg.move(q, i1, i2, idx)
    R = {}
    for k, v in rules(hgt, cls, mv, i1, i2, idx).items():
        full = np.zeros(m.size, bool)
        full[idx] = v
        R[k] = full
    res = {"flights": fy, "years": [YEARS[i1], YEARS[i2]], "px": {k: int(s.sum()) for k, s in sets.items()},
           "map today": {k: {"gain": share(drawn == 1, s), "loss": share(drawn == -1, s)} for k, s in sets.items()},
           "rules": {}}
    for k, c in R.items():
        res["rules"][k] = {
            "lost found": share(c, sets["lost"]),
            "clear cut found": share(c, sets["clear cut"]),
            "gain on lost taken back": share(c, sets["lost"] & (drawn == 1)),
            "missed loss added": share(c, sets["lost"] & (drawn == 0)),
            "steady forest called cut": share(c, sets["steady forest"]),
            "gained called cut": share(c, sets["gained"]),
            "gain on gained taken away": share(c, sets["gained"] & (drawn == 1)),
            "n gain on lost": int((sets["lost"] & (drawn == 1)).sum()),
        }
    del z, q, agb, hgt, cls
    return res


def nh_chunk():
    """The Stoddard chunk, read live: the cut patch and the steady forest around it, window 2021 to 2025."""
    r, c = bc.chunk_of(*NH)
    _, _, E = bc.read_chunk(r, c, step=1)
    ok = E[:, :, 0] != -128
    agb, hgt, cls = cg.head_years(E, ok)
    ya, yb = YEARS.index(2021), YEARS.index(2025)
    idx = np.flatnonzero(np.all(ok, 0))
    patch = (cls[ya, idx] == 0) & (cls[yb, idx] == CLASSES.index("grass")) & (agb[yb, idx] - agb[ya, idx] > 20)
    steady = np.all(cls[:, idx] == 0, 0) & (np.abs(agb[yb, idx] - agb[ya, idx]) < 10)
    R = rules(hgt, cls, cg.move(E, ya, yb, idx), ya, yb, idx)
    return {"px": {"cut patch": int(patch.sum()), "steady forest": int(steady.sum())},
            "rules": {k: {"cut patch called cut": share(v, patch), "steady forest called cut": share(v, steady)}
                      for k, v in R.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=4.0)
    args = ap.parse_args()
    conf = wt.Conf()
    out, t0 = {"squares": {}}, time.time()
    for si, name in enumerate(cg.SQUARES):
        out["squares"][name] = one(name, conf)
        rss = psutil.Process().memory_info().rss / 1e9
        print(f"{si + 1}/{len(cg.SQUARES)} {name} {out['squares'][name]['years']} [{time.time() - t0:.0f} s, {rss:.1f} GB]",
              flush=True)
        if rss > args.max_gb:
            sys.exit(f"stopped: {rss:.1f} GB past --max-gb {args.max_gb}")
    names = list(out["squares"])
    keys = list(out["squares"][names[0]]["rules"]["h3"])
    pooled = {}
    for k in out["squares"][names[0]]["rules"]:
        row = {}
        for m in keys:
            if m.startswith("n "):
                row[m] = int(sum(out["squares"][n]["rules"][k][m] for n in names))
                continue
            need = {"lost found": "lost", "clear cut found": "clear cut", "gain on lost taken back": "lost",
                    "missed loss added": "lost", "steady forest called cut": "steady forest", "gained called cut": "gained",
                    "gain on gained taken away": "gained"}[m]
            v = [out["squares"][n]["rules"][k][m] for n in names
                 if out["squares"][n]["px"][need] >= 200 and out["squares"][n]["rules"][k][m] is not None]
            row[m] = round(float(np.mean(v)), 4) if v else None
        pooled[k] = row
    out["pooled_mean_of_squares"] = pooled
    out["stoddard"] = nh_chunk()
    out["seconds"] = round(time.time() - t0)
    out["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 1)
    OUT.write_text(json.dumps(out, indent=1))
    cols = ["lost found", "clear cut found", "gain on lost taken back", "missed loss added", "steady forest called cut",
            "gained called cut", "gain on gained taken away"]
    print("\nmean of squares (200+ px)   " + " | ".join(cols) + " | Stoddard patch / steady")
    for k, row in pooled.items():
        s = out["stoddard"]["rules"][k]
        print(f"  {k:20s} " + "  ".join(f"{row[c]:.3f}" if row[c] is not None else "  -  " for c in cols)
              + f"   {s['cut patch called cut']:.3f} / {s['steady forest called cut']:.3f}")
    print(f"gain drawn on lost forest, all squares: {pooled['h3']['n gain on lost']} px")
    print(f"done {out['seconds']} s, {out['peak_gb']} GB")


if __name__ == "__main__":
    main()
