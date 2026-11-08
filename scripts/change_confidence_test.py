"""Which way of drawing biomass change with its confidence works best: every variant scored on the same pixels,
held out by square, against airborne lidar and against Esri's land cover changes.

Confidence of a change = how unusual it is on steady ground: the share of steady pixels (Esri class held from a to
b, CTrees' 100 m cell moved under 10 Mg/ha and 10 %) whose change is smaller, from the steady pixels of the OTHER
folds' squares (5 folds by square), per window length L (1 to 6), and per Esri class where the variant says so.

Variants, loss (a fall) and gain (a rise) apart:
  rule           the map's rule now: loss where the embedding moved past the clearcut cutoff and the head fell;
                 gain where the fixed 100 m block rose at least tau (20 to 40 Mg/ha) and a quarter (binary)
  pct            one table for all ground, the change in percent of the pixel's year-a biomass
  cls_mg         a table per Esri class, the change in Mg/ha
  cls_pct        a table per Esri class, in percent
  ..._mv         the lower of that and the embedding move's own confidence (its share of steady pixels below it)
  gain at three sizes: px (the pixel), slide (the 100 m around the pixel, a sliding mean), block (fixed 100 m blocks)

References:
  esri   all squares, every window 2018 to 2024: lost = trees to rangeland or bare, grew = rangeland or bare to
         trees, steady as above (a sample)
  lidar  the nine two-flight squares (FINDINGS 16, 22), per 10 m pixel: lost = canopy fell 3 m from 5 m or more,
         gained = rose 2 m, steady = moved under 0.5 m; and per CTrees 100 m cell (40+ lidar pixels): the mean of
         (gain confidence - loss confidence) against the cell's mean canopy change
Scores: AUC (real against steady); found at 2 % of steady lit (each variant's own cutoff, so all are compared at
the same false alarm rate); found and lit at confidence 0.95; the rule at its own cut.

Writes results/change_confidence_test.json. RAM ~4 GB; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests python scripts/change_confidence_test.py
"""
import argparse
import json
import os
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import psutil
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import burn_check as bc  # noqa: E402  (the map's head)
import nlcd_check as nc  # noqa: E402  (squares, labels)

CUT = float(np.load(HERE / "models" / "detectors.npz")["aef_cut"])
YEARS = list(range(2018, 2026))
NY = 7  # Esri classes 2018 to 2024
FOLDS = 5
STEP_STEADY, STEP_REAL = 61, 7
TAU = {1: 20, 7: 40}
OUT = HERE / "results" / "change_confidence_test.json"
LIDAR = ["lidar-nz-glenbervie", "lidar-nz-esk-valley", "lidar-nz-roxburgh", "lidar-nz-banks", "lidar-ee-lahemaa",
         "lidar-ee-kivioli", "lidar-ee-tartu", "lidar-us-eaton", "lidar-us-palisades"]
F = ["d", "dpct", "dbar", "dbarpct", "dblk", "dblkpct", "ablk", "mv", "a"]


def guard(max_gb):
    def run():
        p = psutil.Process()
        while True:
            if p.memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def slide(v, ok, side=600):
    v2 = np.where(ok, v, 0).reshape(side, side).astype(np.float32)
    w2 = ok.reshape(side, side).astype(np.float32)
    s, c = uniform_filter(v2, 10, mode="nearest"), uniform_filter(w2, 10, mode="nearest")
    return np.where(c > 0.3, s / np.maximum(c, 1e-6), np.nan).ravel()


def block(v, ok, side=600):
    v2 = np.where(ok, v, 0).reshape(side // 10, 10, side // 10, 10)
    w2 = ok.reshape(side // 10, 10, side // 10, 10)
    s, c = v2.sum((1, 3)), w2.sum((1, 3))
    m = np.where(c >= 30, s / np.maximum(c, 1), np.nan)
    return np.repeat(np.repeat(m, 10, 0), 10, 1).ravel()


def feats(agbA, agbB, uA, uB, ok):
    d = agbB - agbA
    dbar, abar = slide(d, ok), slide(agbA, ok)
    dblk, ablk = block(d, ok), block(agbA, ok)
    mv = np.where(ok, (1 - (uA.astype(np.float32) * uB.astype(np.float32)).sum(1)) / CUT, np.nan)
    return {"d": d, "dpct": 100 * d / np.maximum(agbA, 1), "dbar": dbar, "dbarpct": 100 * dbar / np.maximum(abar, 1),
            "dblk": dblk, "dblkpct": 100 * dblk / np.maximum(ablk, 1), "ablk": ablk, "mv": mv, "a": agbA}


def unit(q):
    f = q.astype(np.float32) / 127.5
    e = np.sign(f) * f * f
    return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float16)


def collect(sq, rows, lid):
    rng = np.random.default_rng(0)
    t0 = time.time()
    for si, name in enumerate(sorted(sq)):
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        io = np.load(fi)["lndcov"][:NY]
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"]
        if q.shape[1] != 360000:
            continue
        ok = q[:, :, 0] != -128
        agb = np.full(q.shape[:2], np.nan, np.float32)
        for i in range(q.shape[0]):
            agb[i, ok[i]] = bc.map_agb(q[i, ok[i]])
        U = np.stack([unit(q[i]) for i in range(q.shape[0])])
        del q
        ct = nc.labels_of(name)["ctrees"]
        fold = zlib.crc32(name.encode()) % FOLDS
        ks = np.zeros(360000, bool)
        ks[rng.integers(STEP_STEADY)::STEP_STEADY] = True
        kr = np.zeros(360000, bool)
        kr[rng.integers(STEP_REAL)::STEP_REAL] = True
        for a in range(NY - 1):
            for b in range(a + 1, NY):
                valid = ok[a] & ok[b]
                f = feats(agb[a], agb[b], U[a], U[b], valid)
                fin = valid & np.isfinite(f["dbar"]) & np.isfinite(f["dblk"])
                ct_st = np.isfinite(ct[a]) & np.isfinite(ct[b]) & (np.abs(ct[b] - ct[a]) < 10) & \
                    (np.abs(ct[b] - ct[a]) < 0.1 * np.maximum(ct[a], 1))
                stable = np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250)
                sets = {0: fin & stable & ct_st & ks,
                        1: fin & kr & (io[a] == 2) & np.isin(io[b], [8, 11]),
                        2: fin & kr & np.isin(io[a], [8, 11]) & (io[b] == 2)}
                for s_, m in sets.items():
                    if not m.any():
                        continue
                    r = {k: v[m].astype(np.float32) for k, v in f.items()}
                    r |= {"set": np.full(m.sum(), s_, np.int8), "L": np.full(m.sum(), b - a, np.int8),
                          "cls": io[a][m].astype(np.int16), "fold": np.full(m.sum(), fold, np.int8)}
                    rows.append(r)
                del f
        if name in LIDAR:
            Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
            y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
            i1, i2 = YEARS.index(y1), YEARS.index(y2)
            chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
            m = np.all(cov >= 80, 0) & ok[i1] & ok[i2]
            f = feats(agb[i1], agb[i2], U[i1], U[i2], ok[i1] & ok[i2])
            m &= np.isfinite(f["dbar"]) & np.isfinite(f["dblk"])
            cls = np.load(fi)["lndcov"][min(i1, NY - 1)]
            cell = np.load(HERE / "data" / "labels" / f"{name}.npz")["cell"]
            lid.append({**{k: v[m].astype(np.float32) for k, v in f.items()}, "dH": (chm[1] - chm[0])[m], "h1": chm[0][m],
                        "L": np.full(m.sum(), i2 - i1, np.int8), "cls": cls[m].astype(np.int16),
                        "fold": np.full(m.sum(), fold, np.int8), "cell": cell[m] + 10_000_000 * len(lid), "name": name})
            print(f"  lidar {name} {y1}-{y2}: {m.sum()} px", flush=True)
        del U, agb
        print(f"square {si + 1} of {len(sq)}: {name} [{time.time() - t0:.0f} s, "
              f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)


class Tables:
    """Per (fold left out, key): the sorted steady values of the other folds, subsampled; conf = share below."""

    def __init__(self, S):
        self.S = S
        self.cache = {}

    def get(self, hold, var, L, cls=None):
        k = (hold, var, L, cls)
        if k not in self.cache:
            S = self.S
            m = (S["fold"] != hold) & (S["L"] == L)
            if cls is not None:
                m &= S["cls"] == cls
            x = S[var][m]
            x = np.sort(x[np.isfinite(x)])
            if x.size > 40000:
                x = x[np.linspace(0, x.size - 1, 40000).astype(int)]
            self.cache[k] = x if x.size >= 200 else None
        return self.cache[k]

    def conf(self, R, var, x, per_cls):
        """x: the value to judge (larger = more unusual); per-row table by its fold, L (capped at 6), class."""
        out = np.full(x.shape, np.nan, np.float32)
        Lc = np.minimum(R["L"], 6)
        keys = np.unique(np.stack([R["fold"], Lc, R["cls"] if per_cls else np.zeros_like(R["cls"])], 1), axis=0)
        for fo, L, c in keys:
            m = (R["fold"] == fo) & (Lc == L) & ((R["cls"] == c) if per_cls else True)
            t = self.get(int(fo), var, int(L), int(c) if per_cls else None)
            if t is None and per_cls:
                t = self.get(int(fo), var, int(L), None)  # a class too rare: the pooled table
            if t is None:
                continue
            out[m] = np.searchsorted(t, x[m], side="left") / t.size
        return out


def variants(T, R):
    """Each variant's loss and gain confidence for rows R (dict of arrays)."""
    neg = lambda v: np.maximum(-v, 0)  # noqa: E731
    pos = lambda v: np.maximum(v, 0)  # noqa: E731
    # the steady tables are of one-sided values too
    V = {}
    mv = T.conf(R, "mv", R["mv"], False)
    mvc = T.conf(R, "mv", R["mv"], True)
    tau = np.where(R["L"] <= 1, TAU[1], TAU[1] + (TAU[7] - TAU[1]) * (np.minimum(R["L"], 7) - 1) / 6)
    V["loss rule"] = ((R["mv"] >= 1) & (R["d"] < 0)).astype(np.float32)
    V["loss pct"] = T.conf(R, "neg_dpct", neg(R["dpct"]), False)
    V["loss cls_mg"] = T.conf(R, "neg_d", neg(R["d"]), True)
    V["loss cls_pct"] = T.conf(R, "neg_dpct", neg(R["dpct"]), True)
    V["loss pct_mv"] = np.minimum(V["loss pct"], mv)
    V["loss cls_pct_mv"] = np.minimum(V["loss cls_pct"], mvc)
    V["loss cls_mg_mv"] = np.minimum(V["loss cls_mg"], mvc)
    V["gain rule (block)"] = ((R["dblk"] >= tau) & (R["dblk"] >= 0.25 * np.maximum(R["ablk"], 1))).astype(np.float32)
    # the rule's floor (tau Mg/ha and a quarter) at the finer sizes: the pixel, and the sliding 100 m
    V["gain rule px"] = ((R["d"] >= tau) & (R["dpct"] >= 25)).astype(np.float32)
    V["gain rule slide"] = ((R["dbar"] >= tau) & (R["dbarpct"] >= 25)).astype(np.float32)
    for size, dv, pv in (("px", "d", "dpct"), ("slide", "dbar", "dbarpct"), ("block", "dblk", "dblkpct")):
        V[f"gain pct {size}"] = T.conf(R, f"pos_{pv}", pos(R[pv]), False)
        V[f"gain cls_mg {size}"] = T.conf(R, f"pos_{dv}", pos(R[dv]), True)
        V[f"gain cls_pct {size}"] = T.conf(R, f"pos_{pv}", pos(R[pv]), True)
    V["gain pct px_mv"] = np.minimum(V["gain pct px"], mv)
    V["gain cls_pct slide_mv"] = np.minimum(V["gain cls_pct slide"], mvc)
    return V


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 50 or neg.size < 50:
        return None
    allv = np.concatenate([pos, neg])
    # ranks, ties averaged
    sv = np.sort(allv)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    first = np.searchsorted(sv, uniq, side="left") + 1
    r = (first + (cnt - 1) / 2)[inv]
    return round(float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)), 4)


def score(real, steady, binary=False):
    real, steady = real[np.isfinite(real)], steady[np.isfinite(steady)]
    if real.size < 50 or steady.size < 50:
        return None
    o = {"n_real": int(real.size), "n_steady": int(steady.size), "auc": auc(real, steady)}
    if binary:
        o["found"], o["lit"] = round(float((real >= 0.5).mean()), 4), round(float((steady >= 0.5).mean()), 4)
    else:
        thr = float(np.quantile(steady, 0.98))
        o["found_at_2pct_lit"] = round(float((real > thr).mean()), 4)
        o["found_95"], o["lit_95"] = round(float((real >= 0.95).mean()), 4), round(float((steady >= 0.95).mean()), 4)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=10)
    args = ap.parse_args()
    guard(args.max_gb)
    t0 = time.time()
    sq = nc.squares(world=True)
    rows, lid = [], []
    collect(sq, rows, lid)
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    del rows
    for k in ("d", "dpct", "dbar", "dbarpct", "dblk", "dblkpct"):
        R[f"neg_{k}"] = np.maximum(-R[k], 0)
        R[f"pos_{k}"] = np.maximum(R[k], 0)
    S = {k: v[R["set"] == 0] for k, v in R.items()}
    T = Tables(S)
    print(f"\nrows: steady {int((R['set'] == 0).sum())}, lost {int((R['set'] == 1).sum())}, grew {int((R['set'] == 2).sum())} "
          f"[{time.time() - t0:.0f} s]", flush=True)
    res = {"esri": {}, "lidar_px": {}, "lidar_cell": {}}
    V = variants(T, R)
    for k, v in V.items():
        kind = 1 if k.startswith("loss") else 2
        res["esri"][k] = score(v[R["set"] == kind], v[R["set"] == 0], binary="rule" in k)
    # the same by the ground the window starts on (Esri class at year a), for the main variants
    res["esri_by_class"] = {}
    for c, cname in ((2, "trees"), (11, "rangeland"), (5, "crops")):
        for k in ("loss rule", "loss pct", "loss cls_pct", "loss cls_pct_mv", "gain rule (block)", "gain pct px", "gain cls_pct px", "gain cls_pct slide"):
            v = V[k]
            kind = 1 if k.startswith("loss") else 2
            real = v[(R["set"] == kind)]  # real change starts on trees (loss) or rangeland/bare (gain): judged against this class's steady
            st = v[(R["set"] == 0) & (R["cls"] == c)]
            res["esri_by_class"][f"{cname}: {k} (steady lit)"] = round(float((st >= (0.5 if 'rule' in k else 0.95)).mean()), 4) if st.size else None
    # lidar
    L = {k: np.concatenate([x[k] for x in lid]) for k in lid[0] if k != "name"}
    for k in ("d", "dpct", "dbar", "dbarpct", "dblk", "dblkpct"):
        L[f"neg_{k}"], L[f"pos_{k}"] = np.maximum(-L[k], 0), np.maximum(L[k], 0)
    VL = variants(T, L)
    lost = (L["dH"] <= -3) & (L["h1"] >= 5)
    gained = L["dH"] >= 2
    steady = np.abs(L["dH"]) < 0.5
    for k, v in VL.items():
        res["lidar_px"][k] = score(v[lost if k.startswith("loss") else gained], v[steady], binary="rule" in k)
    # per 100 m cell: mean signed confidence (gain - loss) against mean canopy change, for loss + gain pairings
    ids, inv, cnt = np.unique(L["cell"], return_inverse=True, return_counts=True)
    keep = cnt >= 40
    dHc = np.bincount(inv, L["dH"]) / cnt
    pairs = [("loss rule", "gain rule (block)"), ("loss rule", "gain rule px"), ("loss rule", "gain rule slide"),
             ("loss cls_pct_mv", "gain rule (block)"), ("loss cls_pct_mv", "gain rule slide"), ("loss cls_pct_mv", "gain rule px"), ("loss pct", "gain pct px"), ("loss pct", "gain pct slide"),
             ("loss pct", "gain pct block"), ("loss cls_pct", "gain cls_pct px"), ("loss cls_pct", "gain cls_pct slide"),
             ("loss cls_pct_mv", "gain cls_pct slide"), ("loss cls_pct_mv", "gain cls_pct px"), ("loss cls_mg_mv", "gain cls_mg slide"),
             ("loss pct_mv", "gain pct slide"), ("loss pct_mv", "gain pct px_mv"), ("loss cls_pct_mv", "gain cls_pct slide_mv")]
    for lk, gk in pairs:
        s = np.nan_to_num(VL[gk]) - np.nan_to_num(VL[lk])
        sc = np.bincount(inv, s) / cnt
        x, y = sc[keep], dHc[keep]
        lc, gc, stc = (y <= -3), (y >= 2), (np.abs(y) < 0.5)
        o = {"cells": int(keep.sum()), "corr": round(float(np.corrcoef(x, y)[0, 1]), 4),
             "auc_lost": auc(-x[lc], -x[stc]), "auc_gained": auc(x[gc], x[stc])}
        res["lidar_cell"][f"{lk} + {gk}"] = o
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)

    def show(title, d):
        print(f"\n{title}", flush=True)
        for k, o in d.items():
            if o is None:
                continue
            if "found" in o:
                print(f"  {k:28s} auc {o['auc']}  found {o['found']}  lit {o['lit']}  (n {o['n_real']} / {o['n_steady']})", flush=True)
            else:
                print(f"  {k:28s} auc {o['auc']}  found@2%lit {o['found_at_2pct_lit']}  found@.95 {o['found_95']}  lit@.95 {o['lit_95']}", flush=True)
    show("ESRI: lost = trees to rangeland/bare, grew = back; against steady (held out by square)", res["esri"])
    show("LIDAR per pixel: lost = canopy -3 m from 5 m+, gained = +2 m; against |change| < 0.5 m", res["lidar_px"])
    print("\nLIDAR per 100 m cell: mean (gain conf - loss conf) against mean canopy change", flush=True)
    for k, o in res["lidar_cell"].items():
        print(f"  {k:52s} corr {o['corr']}  auc lost {o['auc_lost']}  auc gained {o['auc_gained']}  ({o['cells']} cells)", flush=True)
    print("\nsteady lit by the class the window starts on (rule at its cut, others at .95)", flush=True)
    for k, v in res["esri_by_class"].items():
        print(f"  {k:52s} {v}", flush=True)
    print(f"\nwrote {OUT} in {res['seconds']} s, peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
