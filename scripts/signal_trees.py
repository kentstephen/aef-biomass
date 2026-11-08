# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "psutil"]
# ///
"""Is there signal for trees cut, and for new building, in yearly embeddings? AlphaEarth (AEF) and TESSERA.

Reads Earthwork's saved pixels (aef-tessera-earthwork/data/separate, every year 2017 to 2025 of both embeddings):
  cut    ten commercial clearcut regions (US, Sweden, Finland, New Zealand, Brazil, Chile)
  near   eleven 6 km squares of forest cleared near mines (Congo, Canada, Indonesia, Peru, Guinea, Sierra Leone)
  Each: 15,000 pixels Hansen GFC v1.12 says lost tree cover in one event year, 5,000 forest that never lost cover.
  lidar  US building sites (forest cleared for building, per-pixel Hansen loss year, repeat lidar moved / unchanged)
  space  new cities with satellite-laser moved / unchanged labels (Egypt's new capital, Eko Atlantic, Xiongan,
         Amaravati)

Tests:
  1 detect   AUC of plain change (1 - cosine) between the year before and the year after the event, cut vs forest
  2 when     for cut pixels, the year of the largest one-year change against Hansen's loss year
  3 new place a small model (logistic regression on the years either side and their difference) taught on every
             other forest square, scored on the one left out: is pixel p cut in year Y, for every Y 2018 to 2024
  4 regrowth cosine of cut pixels to their own pre-cut embedding, year by year after the cut, against forest's
             own year-to-year drift
  5 building plain change on moved vs unchanged ground at the building and new-city sites

RAM: one region in flight (~150 MB) plus the test-3 table (~0.9 GB); peak about 3 GB. Stops past --max-gb.

    uv run scripts/signal_trees.py [--px 3000] [--max-gb 10]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SRC = Path.home() / "dev/projects/aef-tessera-earthwork/data/separate"
OUT = Path(__file__).resolve().parent.parent / "results"
YEARS = list(range(2018, 2026))  # 2017 dropped: TESSERA lacks much of it, AEF rebuilt it differently
CUT = ["coast-range-or", "willapa-wa", "ware-ga", "pine-ms", "north-woods-me", "norrland-se", "kainuu-fi",
       "kaingaroa-nz", "tres-lagoas-br", "biobio-cl"]
NEAR = ["near-kolwezi-cd", "near-fort-hills-ab", "near-detour-lake-on", "near-adaro-id", "near-sangatta-id",
        "near-madre-de-dios-pe", "near-boke-gn", "near-mobimbi-sl", "near-sembehun-sl", "near-weda-bay-id",
        "near-morowali-id"]
BUILD = ["lakewood-ranch-fl", "four-corners-fl", "huntsville-al", "new-albany-oh"]
CITIES = ["space-new-capital-eg", "space-eko-atlantic-ng", "space-xiongan-cn", "space-amaravati-in"]
CAND = list(range(2019, 2025))  # candidate cut years Y for test 3 (needs Y-1 and Y+1)


def watch(max_gb):
    p = psutil.Process()
    peak = [0.0]

    def run():
        while True:
            g = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], g)
            if g > max_gb:
                print(f"STOP: RSS {g:.1f} GB over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return peak


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def load(name):
    z = np.load(SRC / f"{name}.npz", allow_pickle=True)
    d = {k: z[k] for k in ("grp", "b", "a") + (("loss", "y") if "loss" in z.files else ("y",))}
    d["aok"], d["tok"] = z["aok"][1:], z["tok"][1:]
    d["aef"], d["tes"] = unit(z["aef"][1:]), unit(z["tes"][1:])
    return d


def cos(e, i, j):
    return (e[i] * e[j]).sum(-1)


def auc(y, s):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def forest(name, d, px, rng):
    ev = int(d["b"][0]) + 1
    cut = d["grp"] == 4
    ok_all = d["aok"].all(0) & d["tok"].all(0)
    r = {"event": ev, "cut_px": int((cut & ok_all).sum()), "forest_px": int((~cut & ok_all).sum())}
    i0, i1 = YEARS.index(max(ev - 1, YEARS[0])), YEARS.index(ev + 1)
    for k in ("aef", "tes"):
        e = d[k]
        ch = 1 - cos(e, i0, i1)
        r[f"{k}_detect_auc"] = auc(cut[ok_all], ch[ok_all])
        step = 1 - (e[1:] * e[:-1]).sum(-1)  # [8, n]: change into year YEARS[t+1]
        top = np.array(YEARS[1:])[step.argmax(0)]
        off = top[cut & ok_all] - ev
        r[f"{k}_when"] = {str(o): round(float((off == o).mean()), 3) for o in (-2, -1, 0, 1, 2)}
        r[f"{k}_when_0_or_1"] = round(float(((off == 0) | (off == 1)).mean()), 3)
        # regrowth: cosine to the pre-cut year, cut minus forest, each year from ev-1 to 2025
        base = e[i0]
        c = (e * base[None]).sum(-1)  # [9, n]
        r[f"{k}_back"] = {str(y): round(float(np.nanmean(c[t, cut & ok_all])), 3) for t, y in enumerate(YEARS) if y >= ev - 1}
        r[f"{k}_forest_drift"] = {str(y): round(float(np.nanmean(c[t, ~cut & ok_all])), 3) for t, y in enumerate(YEARS) if y >= ev - 1}
        r[f"{k}_forest_step"] = round(float(np.nanmedian(step[:, ~cut & ok_all])), 4)
    # rows for test 3: px pixels (cut and forest in the file's 3:1 ratio), every candidate year
    keep = np.flatnonzero(ok_all)
    keep = rng.choice(keep, min(px, len(keep)), replace=False)
    X, y, yr, pix = [], [], [], []
    for Y in CAND:
        a, b = YEARS.index(Y - 1), YEARS.index(Y + 1)
        feats = []
        for k in ("aef", "tes"):
            e0, e1 = d[k][a, keep], d[k][b, keep]
            feats += [e0, e1, e1 - e0]
        X.append(np.concatenate(feats, 1).astype(np.float32))
        y.append((cut[keep] & (ev == Y)).astype(np.int8))
        yr.append(np.full(len(keep), Y, np.int16))
        pix.append(np.arange(len(keep)))
    return r, (np.concatenate(X), np.concatenate(y), np.concatenate(yr), np.concatenate(pix), cut[keep], ev)


def held_out(rows, names):
    """Leave one forest square out; logistic regression on AEF, TESSERA, both. AUC and AP over pixel-years, and
    for cut pixels whether the top-scored year is Hansen's (or the year after)."""
    nA = 64 * 3
    cols = {"aef": slice(0, nA), "tes": slice(nA, None), "both": slice(0, None)}
    out = {}
    for which, sl in cols.items():
        per = {}
        for i, n in enumerate(names):
            t = time.time()
            if rows[i][1].sum() == 0:  # nothing to score here (no valid pixels, or the cut year is outside CAND)
                continue
            Xtr = np.concatenate([rows[j][0][:, sl] for j in range(len(names)) if j != i])
            ytr = np.concatenate([rows[j][1] for j in range(len(names)) if j != i])
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=300, C=0.1, class_weight="balanced").fit(sc.transform(Xtr), ytr)
            del Xtr
            X, y, yr, pix, cut, ev = rows[i]
            s = m.decision_function(sc.transform(X[:, sl]))
            k = len(np.unique(pix))
            S = s.reshape(len(CAND), k)
            best = np.array(CAND)[S.argmax(0)]
            per[n] = {"auc": round(auc(y, s), 3), "ap": round(float(average_precision_score(y, s)), 3),
                      "base_rate": round(float(y.mean()), 3),
                      "when_0_or_1": round(float(((best[cut] == ev) | (best[cut] == ev + 1)).mean()), 3),
                      "cut_vs_forest_auc": round(auc(cut.astype(int), S.max(0)), 3)}
            print(f"  {which:4s} {n:24s} AUC {per[n]['auc']:.3f} AP {per[n]['ap']:.3f} (base {per[n]['base_rate']:.3f}) "
                  f"year right {per[n]['when_0_or_1']:.2f}  cut vs forest {per[n]['cut_vs_forest_auc']:.3f}  "
                  f"{time.time() - t:.0f} s", flush=True)
        out[which] = per
    return out


def building(name, d):
    ok = d["aok"].all(0) & d["tok"].all(0)
    r = {}
    lab = d["y"]
    m = ok & (lab >= 0)
    if "loss" not in d:  # space sites: per-pixel pair b, a; moved grp 6, unchanged 7
        lab = (d["grp"] == 6).astype(int)
        m = ok & np.isin(d["grp"], (6, 7))
    m = m & (d["b"] >= YEARS[0])
    for k in ("aef", "tes"):
        e = d[k]
        ia = np.array([YEARS.index(max(int(v), YEARS[0])) for v in d["b"]])
        ib = np.array([YEARS.index(int(v)) for v in d["a"]])
        ch = 1 - (e[ia, np.arange(e.shape[1])] * e[ib, np.arange(e.shape[1])]).sum(-1)
        r[f"{k}_moved_vs_unchanged_auc"] = auc(lab[m], ch[m])
        if "loss" in d:  # forest cleared for building: largest one-year change vs Hansen's year
            step = 1 - (e[1:] * e[:-1]).sum(-1)
            top = np.array(YEARS[1:])[step.argmax(0)]
            g = ok & (d["grp"] == 1) & (d["loss"] >= 18)
            off = top[g] - (2000 + d["loss"][g].astype(int))
            r[f"{k}_cleared_for_building_year_0_or_1"] = round(float(((off == 0) | (off == 1)).mean()), 3)
    r["moved_px"], r["unchanged_px"] = int(lab[m].sum()), int((lab[m] == 0).sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=3000)
    ap.add_argument("--max-gb", type=float, default=10)
    args = ap.parse_args()
    peak = watch(args.max_gb)
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    res = {"forest": {}, "held_out": {}, "building": {}}
    rows, names = [], []
    t0 = time.time()
    for n in CUT + NEAR:
        d = load(n)
        r, row = forest(n, d, args.px, rng)
        del d
        res["forest"][n] = r
        rows.append(row)
        names.append(n)
        print(f"{n:24s} event {r['event']}  px {r['cut_px']}/{r['forest_px']}  detect AEF {r['aef_detect_auc']} TES {r['tes_detect_auc']}  "
              f"year right AEF {r['aef_when_0_or_1']} TES {r['tes_when_0_or_1']}  "
              f"[{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
    for n in BUILD + CITIES:
        d = load(n)
        res["building"][n] = building(n, d)
        del d
        print(n, res["building"][n], flush=True)
    json.dump(res, open(OUT / "signal_trees.json", "w"), indent=1)
    print("held out (test 3):", flush=True)
    res["held_out"] = held_out(rows, names)
    res["peak_gb"] = round(peak[0], 2)
    json.dump(res, open(OUT / "signal_trees.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
