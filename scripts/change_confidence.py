"""How big a change has to be to be believed, per land cover class and window length: the noise of the map's head on
steady ground, measured with the land cover maps, so the change map can draw each pixel with its confidence.

Steady ground, per window a -> b (2018 to 2024, every pair): Esri annual land cover (data/io, nlcd_check.py --source
io) gives the same class every year from a to b, and CTrees' 100 m cell moved under 10 Mg/ha and under 10 %. On it,
the head's change (web/public/models/biomass_aef.json) is noise. Recorded per Esri class and window length L:
  d      the change per 10 m pixel, Mg/ha, and as a percent of the pixel's year-a biomass
  dbar   the change averaged over the 100 m around each pixel (a sliding 10 x 10 window), Mg/ha and percent of the
         window's year-a biomass: gain drawn without the fixed squares
  mv     the embedding's own move, (1 - cosine) / the frozen clearcut cutoff (the loss gate is mv >= 1)
For contrast, the same on pixels whose Esri class went from trees to rangeland or bare between a and b (loss) and
back (gain). In CONUS the same steady test with Annual NLCD's classes too (data/nlcd), as a cross-check.

Writes results/change_confidence.json: per class and L, quantiles of each, the lookup the map uses. One square at a
time, every 31st steady pixel kept; RAM ~3 GB; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        python scripts/change_confidence.py
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import burn_check as bc  # noqa: E402  (the map's head)
import nlcd_check as nc  # noqa: E402  (the squares, the label readers, the classes)

CUT = float(np.load(HERE / "models" / "detectors.npz")["aef_cut"])
NY = 7  # 2018 to 2024: the years with Esri and NLCD classes
IO_NAMES = nc.IO_CLASSES
NLCD_GROUP = {41: "forest", 42: "forest", 43: "forest", 52: "shrub", 71: "grassland", 81: "pasture/crops",
              82: "pasture/crops", 90: "wetland", 95: "wetland", 31: "barren", 21: "developed", 22: "developed",
              23: "developed", 24: "developed", 11: "water"}
Q = [50, 75, 80, 85, 90, 95, 97.5, 99, 99.5, 99.9]
STEP = 31
OUT = HERE / "results" / "change_confidence.json"


def guard(max_gb):
    def run():
        p = psutil.Process()
        while True:
            if p.memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def box(v, ok, side=600):
    """Mean of v over the 10 x 10 pixels around each pixel (the valid ones), on the square's grid."""
    v2 = np.where(ok, v, 0).reshape(side, side).astype(np.float32)
    w2 = ok.reshape(side, side).astype(np.float32)
    s = uniform_filter(v2, 10, mode="nearest")
    c = uniform_filter(w2, 10, mode="nearest")
    return np.where(c > 0.3, s / np.maximum(c, 1e-6), np.nan).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    guard(args.max_gb)
    t0 = time.time()
    sq = nc.squares(world=True)
    print(f"{len(sq)} squares", flush=True)
    rng = np.random.default_rng(0)
    pools = {}  # (table, class, L) -> {key: [arrays]}

    def add(table, cls, L, **vals):
        d = pools.setdefault((table, cls, L), {})
        for k, v in vals.items():
            d.setdefault(k, []).append(v.astype(np.float32))

    for si, (name, v) in enumerate(sorted(sq.items())):
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        io = np.load(fi)["lndcov"][:NY]
        fn = HERE / "data" / "nlcd" / f"{name}.npz"
        nl = np.load(fn)["lndcov"][:NY] if fn.exists() else None
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"][:NY]
        n = q.shape[1]
        if n != 360000:
            print(f"  {name}: {n} px, not 600 x 600, skipped", flush=True)
            continue
        ok = q[:, :, 0] != -128
        agb = np.full((NY, n), np.nan, np.float32)
        U = np.zeros((NY, n, 64), np.float16)
        for i in range(NY):
            agb[i, ok[i]] = bc.map_agb(q[i, ok[i]])
            f = q[i].astype(np.float32) / 127.5
            e = np.sign(f) * f * f
            U[i] = (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float16)
            del f, e
        del q
        ct = nc.labels_of(name)["ctrees"][:NY]
        keep = np.zeros(n, bool)
        keep[rng.integers(STEP)::STEP] = True
        for a in range(NY - 1):
            for b in range(a + 1, NY):
                L = b - a
                valid = ok[a] & ok[b] & np.isfinite(agb[a]) & np.isfinite(agb[b])
                d = agb[b] - agb[a]
                dbar = box(d, valid)
                abar = box(agb[a], valid)
                mv = np.where(valid, (1 - (U[a].astype(np.float32) * U[b].astype(np.float32)).sum(1)) / CUT, np.nan)
                cta, ctb = ct[a], ct[b]
                ct_steady = np.isfinite(cta) & np.isfinite(ctb) & (np.abs(ctb - cta) < 10) & \
                    (np.abs(ctb - cta) < 0.1 * np.maximum(cta, 1))
                stable = np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250)
                vals = lambda m: dict(d=d[m], dpct=100 * d[m] / np.maximum(agb[a][m], 1), dbar=dbar[m],  # noqa: E731
                                      dbarpct=100 * dbar[m] / np.maximum(abar[m], 1), mv=mv[m], a=agb[a][m])
                m = valid & stable & ct_steady & keep & np.isfinite(dbar)
                for c in np.unique(io[a][m]):
                    mc = m & (io[a] == c)
                    add("io_steady", IO_NAMES.get(int(c), str(c)), L, **vals(mc))
                # the contrast: Esri's trees to rangeland or bare (loss), and back (gain), every such pixel
                lost = valid & (io[a] == 2) & np.isin(io[b], [8, 11]) & np.isfinite(dbar)
                grew = valid & np.isin(io[a], [8, 11]) & (io[b] == 2) & np.isfinite(dbar)
                if lost.any():
                    add("io_trees_lost", "trees", L, **vals(lost))
                if grew.any():
                    add("io_trees_grew", "trees", L, **vals(grew))
                if nl is not None:
                    sn = np.all(nl[a:b + 1] == nl[a], 0) & (nl[a] != 250)
                    mn = valid & sn & ct_steady & keep & np.isfinite(dbar)
                    for c in np.unique(nl[a][mn]):
                        g = NLCD_GROUP.get(int(c))
                        if g:
                            add("nlcd_steady", g, L, **vals(mn & (nl[a] == c)))
                del d, dbar, abar, mv
        del U, agb, ct
        print(f"square {si + 1} of {len(sq)}: {name} [{time.time() - t0:.0f} s, "
              f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)

    res = {"q": Q, "step": STEP, "cut": CUT, "tables": {}}
    for (table, cls, L), d in sorted(pools.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        e = {k: np.concatenate(v) for k, v in d.items()}
        if e["d"].size < 300:
            continue
        o = {"n": int(e["d"].size)}
        for k in ("d", "dpct", "dbar", "dbarpct"):
            x = e[k][np.isfinite(e[k])]
            o[f"{k}_abs"] = [round(float(v), 2) for v in np.percentile(np.abs(x), Q)]
            o[f"{k}_neg"] = [round(float(v), 2) for v in np.percentile(-np.minimum(x, 0), Q)]
            o[f"{k}_pos"] = [round(float(v), 2) for v in np.percentile(np.maximum(x, 0), Q)]
        x = e["mv"][np.isfinite(e["mv"])]
        o["mv"] = [round(float(v), 3) for v in np.percentile(x, Q)]
        o["mv_gate_share"] = round(float((x >= 1).mean()), 4)
        o["a_median"] = round(float(np.nanmedian(e["a"])), 1)
        res["tables"].setdefault(table, {}).setdefault(cls, {})[str(L)] = o

    print("\nSTEADY GROUND (Esri class held, CTrees steady): the change the head shows anyway", flush=True)
    print("90th / 99th percentile of |change|; mv share past the loss gate", flush=True)
    for table in ("io_steady", "nlcd_steady", "io_trees_lost", "io_trees_grew"):
        print(f"\n{table}", flush=True)
        print(f"  {'class':20s} L {'n':>8s}  {'pixel Mg/ha':>13s}  {'pixel %':>11s}  {'100 m Mg/ha':>13s}  {'100 m %':>11s}  {'gate':>6s}  {'biomass':>7s}", flush=True)
        for cls, byL in res["tables"].get(table, {}).items():
            for L in ("1", "3", "6"):
                o = byL.get(L)
                if not o:
                    continue
                i90, i99 = Q.index(90), Q.index(99)
                print(f"  {cls:20s} {L} {o['n']:8d}  {o['d_abs'][i90]:6.1f} {o['d_abs'][i99]:6.1f}  {o['dpct_abs'][i90]:5.0f} {o['dpct_abs'][i99]:5.0f}  "
                      f"{o['dbar_abs'][i90]:6.1f} {o['dbar_abs'][i99]:6.1f}  {o['dbarpct_abs'][i90]:5.0f} {o['dbarpct_abs'][i99]:5.0f}  "
                      f"{o['mv_gate_share']:6.3f}  {o['a_median']:7.1f}", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} in {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
