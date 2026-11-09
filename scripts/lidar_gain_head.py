"""Gain taught on lidar's canopy rise directly: the only teacher that is not itself a model.

The nine two-flight lidar squares (FINDINGS 16, 22, 24). A pixel's input is AlphaEarth at both flight years and
their difference (64 + 64 + 64), the window's length, and (variant "+head") the map's head's biomass in the first
year and its change. Target: lidar's canopy rose 2 m or more (gained), against everything else. Gradient-boosted
trees, held out by square (leave one square out), so every score is on a square the model never saw.

Scored on the held-out pixels against lidar, as FINDINGS 31: gained (rose 2 m) against steady (moved under 0.5 m),
AUC and found at 2 % of steady lit, per square and pooled; beside it the map's own gain (its rise in Mg/ha, and its
two-part floor) on the same pixels. Then, taught on all nine squares, on Esri's "became trees" against steady
ground over every square and window 2018 to 2024 (a check that it carries past nine places), beside the map's rise.

Only nine squares, six of them with much regrowth, and Glenbervie (New Zealand pine) holds over half the gained
pixels: expect thin, noisy scores. RAM about 2 to 3 GB (one square's embeddings at a time).

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests --with torch python scripts/lidar_gain_head.py
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import change_confidence_test as cct  # noqa: E402
import growth_test as gt  # noqa: E402
import nlcd_check as nc  # noqa: E402

OUT = HERE / "results" / "lidar_gain_head.json"
YEARS, NY = cct.YEARS, cct.NY
PER_SQUARE = 60000  # training pixels drawn per square (gained kept up to half of them)


def feats(qa, qb, L, a, d, head):
    ea, eb = bh.deq(qa), bh.deq(qb)
    cols = [ea, eb, eb - ea, np.full((ea.shape[0], 1), L, np.float32)]
    if head:
        cols += [np.log1p(np.maximum(a, 0))[:, None], d[:, None], (d / np.maximum(a, 1))[:, None]]
    return np.concatenate(cols, 1).astype(np.float32)


def lidar_square(name):
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
    i1, i2 = YEARS.index(y1), YEARS.index(y2)
    q = z["aef"]
    s = z["tes_scale"]
    ok = (q[:, :, 0] != -128) & np.isfinite(s) & (s > 0)
    agb, _ = gt.predict("map", None, z, "0")
    chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
    m = np.all(cov >= 80, 0) & ok[i1] & ok[i2] & np.isfinite(agb[i1]) & np.isfinite(agb[i2])
    idx = np.flatnonzero(m)
    a, d = agb[i1, idx], agb[i2, idx] - agb[i1, idx]
    dH = (chm[1] - chm[0])[idx]
    out = {"qa": q[i1, idx], "qb": q[i2, idx], "a": a, "d": d, "dH": dH, "L": i2 - i1, "years": (y1, y2)}
    del z, q, agb
    return out


def found_at(real, steady, lit=0.02):
    thr = float(np.quantile(steady, 1 - lit))
    return round(float((real > thr).mean()), 4)


def scores(p, d, a, L, gained, steady):
    tau = 20 + 20 * min(1, max(0, (L - 1) / 6)) if np.isscalar(L) else 20 + 20 * np.clip((L - 1) / 6, 0, 1)
    floor = (d >= tau) & (100 * d / np.maximum(a, 1) >= 25)
    o = {"gained": int(gained.sum()), "steady": int(steady.sum())}
    if gained.sum() < 50 or steady.sum() < 50:
        return o
    o["lidar head"] = {"auc": cct.auc(p[gained], p[steady]), "found_at_2pct_lit": found_at(p[gained], p[steady])}
    o["map rise"] = {"auc": cct.auc(d[gained], d[steady]), "found_at_2pct_lit": found_at(d[gained], d[steady]),
                     "floor_found": round(float(floor[gained].mean()), 4), "floor_lit": round(float(floor[steady].mean()), 4)}
    # the lidar head at the map floor's own lit, so the two can be read at the same false alarm rate
    fl = max(float(floor[steady].mean()), 1e-4)
    o["lidar head at map floor's lit"] = found_at(p[gained], p[steady], fl)
    return o


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def sample(S, rng):
    g = S["dH"] >= 2
    gi, oi = np.flatnonzero(g), np.flatnonzero(~g)
    ng = min(gi.size, PER_SQUARE // 2)
    no = min(oi.size, PER_SQUARE - ng)
    return np.concatenate([rng.choice(gi, ng, replace=False), rng.choice(oi, no, replace=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=6)
    ap.add_argument("--no-esri", action="store_true")
    args = ap.parse_args()
    cct.guard(args.max_gb)
    t0 = time.time()
    rng = np.random.default_rng(0)
    sq = {}
    for name in cct.LIDAR:
        sq[name] = lidar_square(name)
        print(f"read {name} {sq[name]['years']}: {sq[name]['a'].size} px, gained {(sq[name]['dH'] >= 2).sum()} "
              f"[{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    picks = {n: sample(S, rng) for n, S in sq.items()}
    res = {"note": "held out by square; gained = canopy rose 2 m, steady = moved under 0.5 m", "variants": {}}
    finals = {}
    for head in (False, True):
        vname = "embeddings + head" if head else "embeddings"
        X = {n: feats(S["qa"][picks[n]], S["qb"][picks[n]], S["L"], S["a"][picks[n]], S["d"][picks[n]], head) for n, S in sq.items()}
        y = {n: (S["dH"][picks[n]] >= 2).astype(np.int8) for n, S in sq.items()}
        per, pool = {}, {"p": [], "d": [], "a": [], "L": [], "g": [], "s": []}
        for held in sq:
            tr = [n for n in sq if n != held]
            clf = model().fit(np.concatenate([X[n] for n in tr]), np.concatenate([y[n] for n in tr]))
            S = sq[held]
            Xh = feats(S["qa"], S["qb"], S["L"], S["a"], S["d"], head)
            p = clf.predict_proba(Xh)[:, 1].astype(np.float32)
            del Xh
            gained, steady = S["dH"] >= 2, np.abs(S["dH"]) < 0.5
            per[held] = scores(p, S["d"], S["a"], S["L"], gained, steady) | {"years": S["years"]}
            keep = gained | steady
            for k, v in (("p", p), ("d", S["d"]), ("a", S["a"]), ("L", np.full(p.size, S["L"])), ("g", gained), ("s", steady)):
                pool[k].append(v[keep])
            print(f"{vname} held {held}: {json.dumps(per[held])}", flush=True)
        P = {k: np.concatenate(v) for k, v in pool.items()}
        pooled = scores(P["p"], P["d"], P["a"], P["L"], P["g"], P["s"])
        # pooled without Glenbervie, which holds most of the gained pixels
        res["variants"][vname] = {"pooled": pooled, "per square": per}
        print(f"{vname} POOLED {json.dumps(pooled)}", flush=True)
        finals[vname] = model().fit(np.concatenate(list(X.values())), np.concatenate(list(y.values())))
    del sq
    if not args.no_esri:
        res["esri"] = esri_check(finals, rng)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "variants"}, indent=1), flush=True)
    print(f"wrote {OUT} in {res['seconds']} s, peak {res['peak_gb']} GB", flush=True)


def esri_check(finals, rng):
    """Taught on all nine lidar squares; Esri's 'became trees' against steady ground, every other square."""
    sq = nc.squares(world=True)
    acc = {v: {"p": [], "d": [], "a": [], "L": [], "set": []} for v in finals}
    t0 = time.time()
    for si, name in enumerate(sorted(sq)):
        if name in cct.LIDAR:
            continue
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"]
        if q.shape[1] != 360000:
            continue
        s = z["tes_scale"]
        ok = (q[:, :, 0] != -128) & np.isfinite(s) & (s > 0)
        agb, _ = gt.predict("map", None, z, "0")
        ok &= np.isfinite(agb)
        io = np.load(fi)["lndcov"][:NY]
        ct = nc.labels_of(name)["ctrees"]
        ks = np.zeros(360000, bool)
        ks[rng.integers(cct.STEP_STEADY * 4)::cct.STEP_STEADY * 4] = True
        kr = np.zeros(360000, bool)
        kr[rng.integers(cct.STEP_REAL)::cct.STEP_REAL] = True
        for a in range(NY - 1):
            for b in range(a + 1, NY):
                valid = ok[a] & ok[b]
                ct_st = np.isfinite(ct[a]) & np.isfinite(ct[b]) & (np.abs(ct[b] - ct[a]) < 10) & \
                    (np.abs(ct[b] - ct[a]) < 0.1 * np.maximum(ct[a], 1))
                stable = np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250)
                st = valid & stable & ct_st & ks
                gr = valid & kr & np.isin(io[a], [8, 11]) & (io[b] == 2)
                idx = np.flatnonzero(st | gr)
                if idx.size == 0:
                    continue
                av, dv = agb[a, idx], agb[b, idx] - agb[a, idx]
                for v, clf in finals.items():
                    X = feats(q[a, idx], q[b, idx], b - a, av, dv, v != "embeddings")
                    acc[v]["p"].append(clf.predict_proba(X)[:, 1].astype(np.float32))
                    acc[v]["d"].append(dv), acc[v]["a"].append(av)
                    acc[v]["L"].append(np.full(idx.size, b - a, np.int8)), acc[v]["set"].append(np.where(gr[idx], 2, 0).astype(np.int8))
        del z, q, agb
        if si % 10 == 0:
            print(f"esri square {si + 1} of {len(sq)} [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    out = {}
    for v, A in acc.items():
        A = {k: np.concatenate(x) for k, x in A.items()}
        o = {"all": scores(A["p"], A["d"], A["a"], A["L"], A["set"] == 2, A["set"] == 0)}
        for L in range(1, 7):
            m = A["L"] == L
            o[f"L{L}"] = scores(A["p"][m], A["d"][m], A["a"][m], A["L"][m], (A["set"] == 2)[m], (A["set"] == 0)[m])
        out[v] = o
        print(f"ESRI {v}: {json.dumps(o['all'])}", flush=True)
    return out


if __name__ == "__main__":
    main()
