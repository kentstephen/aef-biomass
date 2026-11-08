"""How much ground that was NOT cut does a tree-cut detector light up? Earthwork's trouble spots.

The signal test (signal_trees.py) only set cut forest against standing forest beside it. Earthwork's own false
alarms were crop fields, wetlands and changing woodland, so here the detector meets them. Every set comes from
aef-tessera-earthwork (both embeddings, the two years Earthwork compared):
  crop    crop switches in 16 US farm regions (USDA CDL), 2019 -> 2021 and 2023 -> 2025, nothing dug or cut
  farm    farm lidar sites (fields leveled, pivots); marsh: Texas / Florida marsh and storm coast lidar
  look    ground-look-*: floods, landslides, river banks, solar arrays, pits and the like (the event's ground)
  lake    reservoirs drawn down or refilled (the bathtub ring)
  burn    forest burned (Hansen loss in the fire year): tree loss, but not cutting
  years   20 quiet natural squares, every year 2018 to 2025 (6 are forest: Amazon, Borneo, Congo, taiga, Ontario,
          Adirondacks), each two-year pair Y-1 -> Y+1

Detectors, each with its cutoff set to find 80 % of cut pixel-years in the 20 forest squares, held out by square:
  plain_aef, plain_tes   1 - cosine between the two years
  model                  logistic regression on both embeddings (years either side and their difference), as
                         signal_trees.py, final fit on all forest squares
Gate: "was this forest the year before": logistic regression on the before year's embeddings (AEF + TESSERA),
  forest = the forest squares' pixels before their cut; not forest = the non-forest quiet squares (desert, steppe,
  savanna, grassland). Crops, marsh, floods and the rest are never taught to it. Gate cutoff: keeps 95 % of the
  forest squares' cut pixels.

Run with Earthwork's environment (it imports compare.py's loader):
    uv run --project ~/dev/projects/aef-tessera-earthwork python scripts/false_alarms_trees.py
RAM: one site in flight; the Texas marsh (1.1 M pixels) peaks about 4 GB. Stops past MAX_GB (default 22); peak was ~14 GB.
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW))
os.chdir(EW)  # Earthwork's modules use paths relative to their own folder
import compare  # noqa: E402
import tessera_sites as ts  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results"
SEP = EW / "data/separate"
YEARS = list(range(2018, 2026))
CAND = list(range(2019, 2025))
FOREST_SQ = ["coast-range-or", "willapa-wa", "ware-ga", "pine-ms", "north-woods-me", "norrland-se", "kainuu-fi",
             "kaingaroa-nz", "tres-lagoas-br", "near-kolwezi-cd", "near-fort-hills-ab", "near-detour-lake-on",
             "near-adaro-id", "near-sangatta-id", "near-madre-de-dios-pe", "near-boke-gn", "near-mobimbi-sl",
             "near-weda-bay-id", "near-morowali-id"]  # biobio-cl (no AEF 2025) and sembehun (= mobimbi) left out
QUIET_FOREST = ["jau-br", "kerihun-id", "salonga-cd", "yakutia-taiga-ru", "ontario-shield-ca", "adirondack-ny-us"]
FARM = ["antelope-farm-ne", "box-butte-farm-ne", "fresno-farm-ca", "hamilton-farm-in", "arkansas-delta-ar",
        "othello-pivots-wa"]
MARSH = ["anahuac-tx", "myakka-fl", "brazoria-tx", "mexico-beach-fl"]
MAX_GB = float(os.environ.get("MAX_GB", 22))
PX = 3000
RNG = np.random.default_rng(0)


def watch():
    p, peak = psutil.Process(), [0.0]

    def run():
        while True:
            g = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], g)
            if g > MAX_GB:
                print(f"STOP: RSS {g:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return peak


def unit(x):
    x = np.nan_to_num(x.astype(np.float32))
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def pair_feats(ab, aa, tb, ta):
    return np.concatenate([ab, aa, aa - ab, tb, ta, ta - tb], 1).astype(np.float32)


def forest_rows():
    """Per forest square: pixel-year rows for the cut model, before-year rows for the gate."""
    rows = []
    for n in FOREST_SQ:
        z = np.load(SEP / f"{n}.npz", allow_pickle=True)
        A, T = unit(z["aef"][1:]), unit(z["tes"][1:])
        ok = z["aok"][1:].all(0) & z["tok"][1:].all(0)
        ev = int(z["b"][0]) + 1
        cut = z["grp"] == 4
        keep = RNG.choice(np.flatnonzero(ok), min(PX, ok.sum()), replace=False)
        X, y, pa, pt = [], [], [], []
        for Y in CAND:
            a, b = YEARS.index(Y - 1), YEARS.index(Y + 1)
            X.append(pair_feats(A[a, keep], A[b, keep], T[a, keep], T[b, keep]))
            y.append((cut[keep] & (ev == Y)).astype(np.int8))
            pa.append(1 - (A[a, keep] * A[b, keep]).sum(1))
            pt.append(1 - (T[a, keep] * T[b, keep]).sum(1))
        i0 = YEARS.index(max(ev - 1, YEARS[0]))
        gate_x = np.concatenate([A[i0, keep], T[i0, keep]], 1)
        rows.append({"name": n, "X": np.concatenate(X), "y": np.concatenate(y), "pa": np.concatenate(pa),
                     "pt": np.concatenate(pt), "gate_x": gate_x, "cut_keep": cut[keep]})
        del z, A, T
    return rows


def quiet_squares():
    out = {}
    for f in sorted(glob.glob(str(EW / "data/years/*.npz"))):
        n = Path(f).stem
        z = np.load(f, allow_pickle=True)
        A = np.stack([unit(z[f"aef_{y}"]) for y in YEARS])
        T = np.stack([unit(z[f"{y}_q"].astype(np.float32) * np.nan_to_num(z[f"{y}_scales"])[:, None]) for y in YEARS])
        ok = np.stack([z[f"aok_{y}"] & np.isfinite(z[f"{y}_scales"]) & (z[f"{y}_scales"] > 0) for y in YEARS]).all(0)
        keep = RNG.choice(np.flatnonzero(ok), min(PX, ok.sum()), replace=False) if ok.sum() else np.array([], int)
        out[n] = (A[:, keep], T[:, keep])
        del z
    return out


def fit(X, y, C=0.1):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=300, C=C, class_weight="balanced").fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))


def main():
    peak = watch()
    t0 = time.time()
    rows = forest_rows()
    print(f"forest squares read [{time.time() - t0:.0f} s, {peak[0]:.1f} GB]", flush=True)

    # held-out scores on the forest squares -> cutoffs at 80 % of cut pixel-years found
    held = []
    for i in range(len(rows)):
        tr = [r for j, r in enumerate(rows) if j != i]
        f = fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
        held.append(f(rows[i]["X"]))
    pos = np.concatenate([r["y"] for r in rows]) == 1
    s_model = np.concatenate(held)
    s_pa, s_pt = np.concatenate([r["pa"] for r in rows]), np.concatenate([r["pt"] for r in rows])
    cut_at = {k: float(np.quantile(s[pos], 0.2)) for k, s in (("model", s_model), ("plain_aef", s_pa), ("plain_tes", s_pt))}
    neg_forest = ~pos  # pixel-years with no cut that year (standing forest, or cut ground in other years)
    print("cutoffs (80 % of cuts found):", {k: round(v, 3) for k, v in cut_at.items()}, flush=True)
    model = fit(np.concatenate([r["X"] for r in rows]), np.concatenate([r["y"] for r in rows]))

    # the gate: forest before vs non-forest quiet ground
    quiet = quiet_squares()
    nonforest = [n for n in quiet if n not in QUIET_FOREST]
    gx_pos = np.concatenate([r["gate_x"] for r in rows])
    gx_neg = np.concatenate([np.concatenate([quiet[n][0][0], quiet[n][1][0]], 1) for n in nonforest if quiet[n][0].shape[1]])
    gate = fit(np.concatenate([gx_pos, gx_neg]), np.r_[np.ones(len(gx_pos)), np.zeros(len(gx_neg))].astype(int), C=1.0)
    g_cut = gate(np.concatenate([r["gate_x"][r["cut_keep"]] for r in rows]))
    gate_at = float(np.quantile(g_cut, 0.05))

    def lit(ab, aa, tb, ta):
        X = pair_feats(ab, aa, tb, ta)
        g = gate(np.concatenate([ab, tb], 1)) >= gate_at
        sm = model(X) >= cut_at["model"]
        spa = (1 - (ab * aa).sum(1)) >= cut_at["plain_aef"]
        spt = (1 - (tb * ta).sum(1)) >= cut_at["plain_tes"]
        r = lambda v: round(float(v.mean()), 3) if len(v) else None  # noqa: E731
        return {"n": int(len(X)), "plain_aef": r(spa), "plain_tes": r(spt), "model": r(sm), "model_gated": r(sm & g),
                "plain_aef_gated": r(spa & g), "plain_tes_gated": r(spt & g),
                "gate_forest": r(g)}

    res = {"cutoffs": cut_at, "gate_at": gate_at, "sets": {}}
    # forest squares themselves, held out: what is found and what standing forest lights
    hm = s_model >= cut_at["model"]
    res["forest_squares"] = {"cut_found_model": round(float(hm[pos].mean()), 3),
                             "no_cut_that_year_lit_model": round(float(hm[neg_forest].mean()), 3)}
    gcut = np.concatenate([np.tile(gate(r["gate_x"]) >= gate_at, len(CAND)) for r in rows])
    for k, s in (("plain_aef", s_pa), ("plain_tes", s_pt), ("model", s_model)):
        res["forest_squares"][f"{k}_gated_cut_found"] = round(float(((s >= cut_at[k]) & gcut)[pos].mean()), 3)

    def add(group, name, ab, aa, tb, ta, years):
        r = lit(ab, aa, tb, ta)
        r["years"] = years
        res["sets"].setdefault(group, {})[name] = r
        print(f"{group:6s} {name:34s} {str(years):12s} n {r['n']:6d}  plain AEF {r['plain_aef']}  TES {r['plain_tes']}  "
              f"model {r['model']}  gated {r['model_gated']}  (gate says forest {r['gate_forest']})  "
              f"[{time.time() - t0:.0f} s, {peak[0]:.1f} GB]", flush=True)

    # quiet squares, every two-year pair
    for n, (A, T) in quiet.items():
        if not A.shape[1]:
            continue
        ab = np.concatenate([A[YEARS.index(Y - 1)] for Y in CAND]); aa = np.concatenate([A[YEARS.index(Y + 1)] for Y in CAND])
        tb = np.concatenate([T[YEARS.index(Y - 1)] for Y in CAND]); ta = np.concatenate([T[YEARS.index(Y + 1)] for Y in CAND])
        add("years-forest" if n in QUIET_FOREST else "years-open", n, ab, aa, tb, ta, "every Y")
    del quiet

    def site(group, name, which=None):
        try:
            d = compare.load(name)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}", flush=True)
            return
        m = d["ok"].copy()
        if which is not None:
            m &= which(name, d)
        idx = np.flatnonzero(m)
        if len(idx) > 50000:
            idx = np.sort(RNG.choice(idx, 50000, replace=False))
        f = lambda k: unit(d[k][idx])  # noqa: E731
        ab, aa, tb, ta = f("ab"), f("aa"), f("tb"), f("ta")
        del d
        add(group, name, ab, aa, tb, ta, None)

    for f_name, region, b, a in ts.cdl_sets():
        site("crop", f_name)
    for n in FARM:
        site("farm", n, lambda nm, d: d["y"] == 0)
    for n in MARSH:
        site("marsh", n, lambda nm, d: d["y"] == 0)

    def ground_grp(want):
        def w(nm, d):
            g = np.load(ts.site_path(nm), allow_pickle=True)["grp"]
            return g == want
        return w
    for p in sorted(glob.glob(str(EW / "data/places/ground-*.npz"))):
        n = Path(p).stem
        kind = json.loads(str(np.load(p, allow_pickle=True)["meta"]))["kind"]
        if kind == "cut":
            continue
        grp = {"burn": "burn", "lake": "lake"}.get(kind, "look")
        site(grp, n, ground_grp(1))
        if kind == "burn":
            site("burn-forest-beside", n, ground_grp(2))

    res["peak_gb"] = round(peak[0], 2)
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "false_alarms_trees.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
