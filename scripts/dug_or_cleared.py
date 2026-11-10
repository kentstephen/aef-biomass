"""Where forest was cleared: was the ground also dug? An AlphaEarth-only head for the map's change view (FINDINGS 42).

The question of Earthwork's section 19 (aef-tessera-earthwork separate_test.py), asked with what the map reads: AlphaEarth
at the change window's two years, no TESSERA, no later years. Two feature sets:
  pair   [b, a, a*b, (a-b)^2] of the unit vectors (Earthwork's own features)
  ctx    pair, plus the mean around the pixel (about 110 m and 510 m) at both years: [ca - cb, ca] at each
Taught as dug: cleared and dug (lidar), moved (satellite lasers). As not dug: cleared and not dug (lidar), the ten
clearcut regions' cut ground, the clearing picked near each mine, the satellite sites' unchanged ground. One logistic
regression (scaled, C=0.1), each place weighed the same and its two groups balanced, at most PER_CLASS pixels per
place and group.

Every place is scored held out: each lidar site alone, the others in FOLDS folds. Held-out scores at the lidar sites go
to data/four_box/dug_or_cleared.npz (rows of the separate files, for four_box.py); the head taught on every place to
models/dug_or_cleared_<set>.npz.

Run: uv run --project ../aef-tessera-earthwork --with psutil python scripts/dug_or_cleared.py [--max-gb 12]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW))
os.chdir(EW)
from separate import CUTS, DIR, LIDAR, NEAR, YEARS, near_name, space_list  # noqa: E402

OUT = HERE / "results" / "dug_or_cleared.json"
SCORES = HERE / "data" / "four_box" / "dug_or_cleared.npz"
SETS = ("pair", "ctx")
PER_CLASS = 3000
FOLDS = 4
RNG = np.random.default_rng(0)


def build(n, groups=(1, 2, 4, 6, 7)):
    """Both feature sets (float16) for a place's pixels in `groups`, their labels (dug: groups 1 and 6), groups and
    rows in the separate file."""
    z = np.load(f"{DIR}/{n}.npz")
    g = z["grp"]
    m0 = np.isin(g, groups)
    rows = np.flatnonzero(m0)
    E = z["aef"][:, rows].astype(np.float32)
    C = z["aef_ctx"][:, :, rows].astype(np.float32)
    ok = z["aok"][:, rows]
    r = np.arange(len(rows))
    bi, ai = z["b"][rows].astype(int) - YEARS[0], z["a"][rows].astype(int) - YEARS[0]
    valid = ok[bi, r] & ok[ai, r]
    b, a = E[bi, r], E[ai, r]
    pair = np.concatenate([b, a, a * b, (a - b) ** 2], 1)
    ctx = [pair]
    for k in range(C.shape[1]):
        cb, ca = C[bi, k, r], C[ai, k, r]
        ctx += [ca - cb, ca]
    F = {"pair": pair.astype(np.float16), "ctx": np.concatenate(ctx, 1).astype(np.float16)}
    gg = g[rows]
    out = {k: v[valid] for k, v in F.items()}
    return out, np.isin(gg, (1, 6)).astype(np.int8)[valid], gg[valid], rows[valid], str(z["source"])


def weights(y):
    w = np.zeros(len(y))
    for v in (0, 1):
        if (y == v).any():
            w[y == v] = 0.5 / (y == v).sum()
    return w


def fit(P, names, fs):
    Xs, ys, ws = [], [], []
    for n in names:
        y = P[n]["y"]
        i = np.concatenate([RNG.choice(np.flatnonzero(y == v), min(PER_CLASS, int((y == v).sum())), replace=False)
                            for v in (0, 1) if (y == v).any()])
        Xs.append(P[n]["F"][fs][i].astype(np.float32))
        ys.append(y[i])
        ws.append(weights(y[i]))
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
    return sc, m


def score(model, X):
    sc, m = model
    return m.predict_proba(sc.transform(X.astype(np.float32)))[:, 1]


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 3) if 0 < y.sum() < len(y) else None


def guard(max_gb):
    def run():
        while True:
            if psutil.Process().memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(1)
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=12)
    args = ap.parse_args()
    guard(args.max_gb)
    t0 = time.time()
    names = [n for n in LIDAR + CUTS + space_list() + [near_name(m) for m in NEAR] if os.path.exists(f"{DIR}/{n}.npz")]
    P = {}
    for n in names:
        F, y, g, rows, src = build(n)
        if len(y) == 0:
            continue
        P[n] = {"F": F, "y": y, "g": g, "rows": rows, "src": src}
        if src == "lidar":  # every labeled pixel (other ground too), scored held out for four_box.py
            P[n]["all"] = build(n, (1, 2, 3))
        print(f"{n:26s} {src:5s} dug {int(y.sum()):6d}  not {int((1 - y).sum()):6d} "
              f"[{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    lid = [n for n in P if P[n]["src"] == "lidar"]
    rest = sorted(n for n in P if n not in lid)
    folds = [[n] for n in lid] + [[n for i, n in enumerate(rest) if i % FOLDS == k] for k in range(FOLDS)]
    held = {fs: {} for fs in SETS}
    for fs in SETS:
        for k, out in enumerate(folds):
            model = fit(P, [n for n in P if n not in out], fs)
            for n in out:
                held[fs][n] = score(model, P[n]["F"][fs])
                if "all" in P[n]:
                    held[fs][n + "|all"] = score(model, P[n]["all"][0][fs])
            print(f"{fs}: fold {k + 1} of {len(folds)} held out {out[:3]}{'...' if len(out) > 3 else ''} "
                  f"[{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        sc, m = fit(P, list(P), fs)
        np.savez(HERE / "models" / f"dug_or_cleared_{fs}.npz", mean=sc.mean_.astype(np.float32),
                 scale=sc.scale_.astype(np.float32), w=m.coef_.ravel().astype(np.float32), b=np.float32(m.intercept_[0]))
    res = {"per_place": {}, "summary": {}}
    for n in P:
        y, src = P[n]["y"], P[n]["src"]
        res["per_place"][n] = {"src": src, "dug": int(y.sum()), "not": int((1 - y).sum()),
                               **{f"{fs}_auc": auc(y, held[fs][n]) for fs in SETS},
                               **{f"{fs}_dug_called": round(float((held[fs][n][y == 1] > 0.5).mean()), 3) if y.any() else None for fs in SETS},
                               **{f"{fs}_not_called": round(float((held[fs][n][y == 0] > 0.5).mean()), 3) if (1 - y).any() else None for fs in SETS}}
    for fs in SETS:
        s = {}
        for src in ("lidar", "cut", "near", "space"):
            ns = [n for n in P if P[n]["src"] == src]
            pp = [res["per_place"][n] for n in ns]
            a_ = [p[f"{fs}_auc"] for p in pp if p[f"{fs}_auc"] is not None and min(p["dug"], p["not"]) >= 50]
            dc = [p[f"{fs}_dug_called"] for p in pp if p[f"{fs}_dug_called"] is not None and p["dug"] >= 50]
            nc = [p[f"{fs}_not_called"] for p in pp if p[f"{fs}_not_called"] is not None and p["not"] >= 50]
            s[src] = {"places": len(ns), "mean_auc": round(float(np.mean(a_)), 3) if a_ else None,
                      "dug_called_dug": round(float(np.mean(dc)), 3) if dc else None,
                      "not_called_dug": round(float(np.mean(nc)), 3) if nc else None}
        res["summary"][fs] = s
        print(f"\n{fs} (mean of places; called dug above .5)")
        for src, v in s.items():
            print(f"  {src:6s} {v['places']:3d} places  AUC {v['mean_auc']}  dug called dug {v['dug_called_dug']}  "
                  f"not dug called dug {v['not_called_dug']}")
    print("\nlidar sites, held out: AUC pair / ctx")
    for n in lid:
        p = res["per_place"][n]
        print(f"  {n:20s} dug {p['dug']:6d} not {p['not']:6d}  {p['pair_auc']} / {p['ctx_auc']}")
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES, **{f"{n}|rows": P[n]["all"][3] for n in lid},
             **{f"{n}|{fs}": held[fs][n + "|all"].astype(np.float32) for n in lid for fs in SETS})
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 1)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"done {res['seconds']} s, {res['peak_gb']} GB")


if __name__ == "__main__":
    main()
