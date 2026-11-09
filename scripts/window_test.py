"""Two checks on the map's change view, with the map's own head (one year of AlphaEarth, taught on all 92 squares,
none of them lidar squares):

1. Loss size. The browser draws a loss's size as the head's fall f (a share of the pixel's biomass in year a)
   blended 2/3 of the way toward tanh(f / loss factor), capped at the pixel's biomass (web/src/change.ts). Beside it:
   the raw fall, the loss factor with the cap (the rule before), and the full soft curve. All against the canopy
   fall of the two-flight lidar squares, on the pixels the browser draws as loss (confidence .9 and up), per 10 m
   pixel and per 100 m cell, for every square and split into burns (Eaton, Palisades) and the rest (New Zealand
   harvest and storm, Estonian clearcuts).
2. Gain by window length. The two-part floor (tau Mg/ha and a quarter of the pixel's biomass) and the rise itself,
   split by the window's length in years, on Esri's "became trees" (every square, every window 2018 to 2024) and on
   lidar (each square's own flight gap). Beside the two-year difference: a trend over every year of the window (the
   least-squares and the Theil-Sen slope of the head's yearly biomass, times the window's length), which a longer
   window can average where two years cannot.

Confidence as the browser computes it: the steady tables of web/public/models/change_conf.json, keyed on the head's
class at year a. Those tables were built from every square's steady pixels, the lidar squares' too, so the loss
confidence is not held out; the gain comparisons do not use the tables (each variant at its own floor, and at the
cutoff that lights 2 % of steady ground).

All variants are scored on the same pixels: valid in AlphaEarth and TESSERA (as FINDINGS 31) in every year of the
window. Writes results/window_test.json. RAM about 6 to 7 GB; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests --with torch python scripts/window_test.py
"""
import argparse
import json
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import change_confidence_test as cct  # noqa: E402
import growth_test as gt  # noqa: E402
import nlcd_check as nc  # noqa: E402

OUT = HERE / "results" / "window_test.json"
TABLES = HERE / "web" / "public" / "models" / "change_conf.json"
YEARS, NY = cct.YEARS, cct.NY
# web/src/change.ts
FIT = {1: (20.0, 0.34219714999198914), 7: (40.0, 0.8100864887237549)}
SOFT, LOSS_FROM = 2 / 3, 0.9
BURNS = {"lidar-us-eaton", "lidar-us-palisades"}
GAIN_VARIANTS = ("end", "ols", "theil")


def rule_for(L):
    f = np.clip((np.asarray(L, np.float32) - 1) / 6, 0, 1)
    return FIT[1][0] + (FIT[7][0] - FIT[1][0]) * f, FIT[1][1] + (FIT[7][1] - FIT[1][1]) * f


def loss_sizes(f, L):
    """The drawn share of the pixel's biomass, per rule; f = the head's raw fall share."""
    lf = rule_for(L)[1]
    return {"raw": np.minimum(1, f), "factor + cap": np.minimum(1, f / lf),
            "soft 2/3 (map)": np.minimum(1, (1 - SOFT) * f + SOFT * np.tanh(f / lf)), "soft": np.tanh(f / lf)}


class Conf:
    """The browser's steady tables: share of steady pixels below x, per head class and window length."""

    def __init__(self):
        T = json.load(open(TABLES))
        self.p, self.cls, self.q = np.array(T["p"]), T["classes"], T["q"]
        self.L = T["L"]

    def share(self, var, cls, L, x):
        out = np.zeros(x.shape, np.float32)
        Lc = np.clip(L, self.L[0], self.L[-1])
        for c in np.unique(cls):
            for l_ in np.unique(Lc):
                m = (cls == c) & (Lc == l_)
                key = f"{self.cls[c] if 0 <= c < len(self.cls) else 'all'}:{l_}"
                q = np.array(self.q[var].get(key, self.q[var][f"all:{l_}"]))
                v = np.interp(x[m], q, self.p)
                v[x[m] <= q[0]] = 0
                v[x[m] > q[-1]] = 1
                out[m] = v
        return out


def trends(A, a, b):
    """The window's change from its yearly values A (Y, m), years a..b: two-year difference, least-squares slope and
    Theil-Sen slope (each times the window's length), and each one's start value (for the percent floor)."""
    Ys = A[a:b + 1].astype(np.float32)
    L = b - a
    t = np.arange(L + 1, dtype=np.float32)
    tc = t - t.mean()
    s_ols = (tc[:, None] * (Ys - Ys.mean(0))).sum(0) / (tc ** 2).sum()
    if L == 1:
        s_ts = Ys[1] - Ys[0]
    else:
        s_ts = np.median(np.stack([(Ys[j] - Ys[i]) / (j - i) for i in range(L + 1) for j in range(i + 1, L + 1)]), 0)
    med = np.median(Ys, 0)
    return {"end": (Ys[-1] - Ys[0], Ys[0]),
            "ols": (s_ols * L, Ys.mean(0) - s_ols * t.mean()),
            "theil": (s_ts * L, med - s_ts * float(np.median(t)))}


def collect(sq, rows, lid, conf):
    rng = np.random.default_rng(0)
    t0 = time.time()
    for si, name in enumerate(sorted(sq)):
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"]
        if q.shape[1] != 360000:
            continue
        s = z["tes_scale"]
        ok = (q[:, :, 0] != -128) & np.isfinite(s) & (s > 0)
        agb, hcls = gt.predict("map", None, z, "0")
        ok &= np.isfinite(agb)
        io = np.load(fi)["lndcov"][:NY]
        ct = nc.labels_of(name)["ctrees"]
        ks = np.zeros(360000, bool)
        ks[rng.integers(cct.STEP_STEADY)::cct.STEP_STEADY] = True
        kr = np.zeros(360000, bool)
        kr[rng.integers(cct.STEP_REAL)::cct.STEP_REAL] = True
        for a in range(NY - 1):
            for b in range(a + 1, NY):
                valid = np.all(ok[a:b + 1], 0)
                ct_st = np.isfinite(ct[a]) & np.isfinite(ct[b]) & (np.abs(ct[b] - ct[a]) < 10) & \
                    (np.abs(ct[b] - ct[a]) < 0.1 * np.maximum(ct[a], 1))
                stable = np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250)
                st = valid & stable & ct_st & ks
                gr = valid & kr & np.isin(io[a], [8, 11]) & (io[b] == 2)
                idx = np.flatnonzero(st | gr)
                if idx.size == 0:
                    continue
                tr = trends(agb[:, idx], a, b)
                r = {"set": np.where(gr[idx], 2, 0).astype(np.int8), "L": np.full(idx.size, b - a, np.int8)}
                for k, (d, a0) in tr.items():
                    r[f"d_{k}"] = d.astype(np.float32)
                    r[f"pct_{k}"] = (100 * d / np.maximum(a0, 1)).astype(np.float32)
                rows.append(r)
        if name in cct.LIDAR:
            lid.append(lidar_rows(name, z, ok, agb, hcls, conf))
        del z, q, agb
        print(f"square {si + 1} of {len(sq)}: {name} [{time.time() - t0:.0f} s, "
              f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)


def lidar_rows(name, z, ok, agb, hcls, conf):
    Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
    i1, i2 = YEARS.index(y1), YEARS.index(y2)
    chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
    m = np.all(cov >= 80, 0) & np.all(ok[i1:i2 + 1], 0)
    idx = np.flatnonzero(m)
    L = i2 - i1
    tr = trends(agb[:, idx], i1, i2)
    a = agb[i1, idx]
    d = agb[i2, idx] - a
    ua, ub = cct.unit(z["aef"][i1, idx]), cct.unit(z["aef"][i2, idx])
    mv = (1 - (ua.astype(np.float32) * ub.astype(np.float32)).sum(1)) / cct.CUT
    c = hcls[i1, idx].astype(np.int64)
    Lv = np.full(idx.size, L)
    c_pct = conf.share("neg_dpct", c, Lv, np.maximum(-100 * d / np.maximum(a, 1), 0))
    c_mv = conf.share("mv", c, Lv, mv)
    r, col = idx // 600, idx % 600
    out = {"a": a, "d": d, "conf_loss": np.where(d < 0, np.minimum(c_pct, c_mv), 0), "h1": chm[0, idx], "h2": chm[1, idx],
           "L": Lv.astype(np.int8), "cell": (r // 10) * 60 + col // 10, "name": name}
    for k, (dv, a0) in tr.items():
        out[f"d_{k}"] = dv
        out[f"pct_{k}"] = 100 * dv / np.maximum(a0, 1)
    print(f"  lidar {name} {y1}-{y2}: {idx.size} px", flush=True)
    return out


def auc(pos, neg):
    return cct.auc(pos, neg)


def gain_scores(d, pct, L, real, steady):
    """Floor found / lit, AUC and found at 2 % lit, for one variant on rows split into real and steady."""
    tau = rule_for(L)[0]
    floor = (d >= tau) & (pct >= 25)
    if real.sum() < 50 or steady.sum() < 50:
        return None
    thr = float(np.quantile(d[steady], 0.98))
    return {"n_real": int(real.sum()), "n_steady": int(steady.sum()),
            "floor_found": round(float(floor[real].mean()), 4), "floor_lit": round(float(floor[steady].mean()), 4),
            "auc": auc(d[real], d[steady]), "found_at_2pct_lit": round(float((d[real] > thr).mean()), 4),
            "real_median_Mg": round(float(np.median(d[real])), 1),
            "steady_p90_abs_Mg": round(float(np.quantile(np.abs(d[steady]), 0.9)), 1)}


def gain_part(R, Ld):
    res = {"esri": {}, "lidar": {}}
    for v in GAIN_VARIANTS:
        d, pct = R[f"d_{v}"], R[f"pct_{v}"]
        res["esri"][v] = {"all": gain_scores(d, pct, R["L"], R["set"] == 2, R["set"] == 0)}
        for L in range(1, 7):
            mL = R["L"] == L
            res["esri"][v][f"L{L}"] = gain_scores(d, pct, R["L"], (R["set"] == 2) & mL, (R["set"] == 0) & mL)
    dH = Ld["h2"] - Ld["h1"]
    gained, steady = dH >= 2, np.abs(dH) < 0.5
    for v in GAIN_VARIANTS:
        d, pct = Ld[f"d_{v}"], Ld[f"pct_{v}"]
        res["lidar"][v] = {"all": gain_scores(d, pct, Ld["L"], gained, steady)}
        for L in np.unique(Ld["L"]):
            mL = Ld["L"] == L
            names = sorted(set(Ld["name"][mL]))
            o = gain_scores(d, pct, Ld["L"], gained & mL, steady & mL)
            if o is not None:
                o["squares"] = names
            res["lidar"][v][f"L{int(L)}"] = o
    return res


def loss_part(Ld):
    """The drawn size of loss against lidar's canopy fall, per rule."""
    res = {}
    a, d, L = Ld["a"], Ld["d"], Ld["L"]
    h1, h2 = Ld["h1"], Ld["h2"]
    drawn = (d < 0) & (Ld["conf_loss"] >= LOSS_FROM)
    f = np.where(d < 0, -d / np.maximum(a, 1), 0)
    S = {k: np.where(drawn, v, 0) for k, v in loss_sizes(f, L).items()}
    g = np.clip(1 - h2 / np.maximum(h1, 0.1), 0, 1)  # lidar's canopy fall, a share of the first flight's height
    forest = h1 >= 5
    lost = (h2 - h1 <= -3) & forest
    bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    burn = np.isin(Ld["name"], list(BURNS))
    for grp, gm in (("all", np.ones_like(burn)), ("burns (Eaton, Palisades)", burn), ("harvest and storm (NZ, EE)", ~burn)):
        o = {"lidar lost px": int((lost & gm).sum()), "drawn share of lidar lost": round(float(drawn[lost & gm].mean()), 4),
             "drawn share of forest with lidar fall < 10 %": round(float(drawn[forest & gm & (g < 0.1)].mean()), 4),
             "rules": {}}
        m = drawn & forest & gm
        o["drawn forest px"] = int(m.sum())
        o["lidar fall of drawn px, median"] = round(float(np.median(g[m])), 3)
        o["head raw fall of drawn px, median"] = round(float(np.median(f[m])), 3)
        for k, s in S.items():
            ro = {"spearman with lidar fall": round(float(spearmanr(s[m], g[m])[0]), 4),
                  "median drawn": round(float(np.median(s[m])), 3),
                  "share drawn at 99 % or more": round(float((s[m] >= 0.99).mean()), 4),
                  "median drawn minus lidar fall": round(float(np.median(s[m] - g[m])), 3), "by lidar fall": {}}
            for lo, hi in bins:
                mb = m & (g >= lo) & (g < hi)
                if mb.sum() >= 50:
                    ro["by lidar fall"][f"{lo:.2f}-{min(hi, 1):.2f}"] = {
                        "px": int(mb.sum()), "median drawn": round(float(np.median(s[mb])), 3),
                        "p25": round(float(np.quantile(s[mb], 0.25)), 3), "p75": round(float(np.quantile(s[mb], 0.75)), 3)}
            o["rules"][k] = ro
        # per 100 m cell (40 or more lidar pixels): drawn loss share of the cell's biomass vs lidar's cell canopy fall
        key = np.array([f"{n}:{c}" for n, c in zip(Ld["name"][gm], Ld["cell"][gm])])
        ids, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        keep = cnt >= 40
        H1, H2 = np.bincount(inv, h1[gm]) / cnt, np.bincount(inv, h2[gm]) / cnt
        gc = np.clip(1 - H2 / np.maximum(H1, 0.1), 0, 1)
        A = np.bincount(inv, a[gm])
        cf = keep & (H1 >= 5)
        o["cells"] = {"cells": int(cf.sum()), "cells lidar fall >= 25 %": int((cf & (gc >= 0.25)).sum()), "rules": {}}
        for k, s in S.items():
            sc = np.bincount(inv, (s * a)[gm]) / np.maximum(A, 1e-6)
            lo_ = cf & (gc >= 0.25)
            o["cells"]["rules"][k] = {"pearson with lidar fall": round(float(np.corrcoef(sc[cf], gc[cf])[0, 1]), 4),
                                      "spearman with lidar fall": round(float(spearmanr(sc[cf], gc[cf])[0]), 4),
                                      "median drawn, cells lidar fall >= 25 %": round(float(np.median(sc[lo_])), 3) if lo_.any() else None,
                                      "median lidar fall, same cells": round(float(np.median(gc[lo_])), 3) if lo_.any() else None,
                                      "median drawn minus lidar fall, same cells": round(float(np.median(sc[lo_] - gc[lo_])), 3) if lo_.any() else None}
        res[grp] = o
    # per square, the map's rule only
    res["per square (map rule)"] = {}
    for n in sorted(set(Ld["name"])):
        m = drawn & forest & (Ld["name"] == n)
        if m.sum() < 50:
            continue
        s = S["soft 2/3 (map)"]
        res["per square (map rule)"][n] = {"drawn px": int(m.sum()), "median drawn": round(float(np.median(s[m])), 3),
                                           "median lidar fall": round(float(np.median(g[m])), 3),
                                           "spearman": round(float(spearmanr(s[m], g[m])[0]), 4)}
    return res


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=12)
    ap.add_argument("--only", nargs="*", help="a few squares, for a quick check (results to the temp folder)")
    args = ap.parse_args()
    cct.guard(args.max_gb)
    t0 = time.time()
    sq = nc.squares(world=True)
    if args.only:
        sq = {k: v for k, v in sq.items() if k in args.only}
        OUT = Path(tempfile.gettempdir()) / "window_test_check.json"
    rows, lid = [], []
    collect(sq, rows, lid, Conf())
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    del rows
    Ld = {k: np.concatenate([x[k] if k != "name" else np.full(x["a"].size, x["name"]) for x in lid]) for k in lid[0]}
    res = {"rows": {"steady": int((R["set"] == 0).sum()), "grew": int((R["set"] == 2).sum()), "lidar px": int(Ld["a"].size)},
           "gain": gain_part(R, Ld), "loss": loss_part(Ld)}
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"\nwrote {OUT} in {res['seconds']} s, peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
