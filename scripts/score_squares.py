# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "joblib", "psutil"]
# ///
"""Score the new test squares (data/squares/, read_squares.py) with the frozen detectors (models/detectors.npz)
and fire-knn's volume head (scripts/volume_head.py). Every pixel, every year; nothing here was tuned on these places.

Tree loss in year Y (2019 to 2024), per embedding: plain change 1 - cos(e[Y-1], e[Y+1]) at or over the frozen cutoff,
and the frozen forest gate on e[Y-1]; vote = at least 2 of the 3. A pixel's flagged year = its first flagged Y.
Against Hansen GFC v1.12 (30 m, its own errors):
  found         Hansen loss 2019 to 2024 pixels flagged within a year of Hansen's year (L-1 to L+1)
  year right    of those found, flagged in L or L+1
  forest lit    pixels never lost (2001 to 2024) with tree cover 2000 >= 30 %: share flagged in any year
  open lit      never lost, cover < 10 %, WorldCover 2020 not trees: share flagged in any year
  agrees        flagged pixels with any Hansen loss 2017 to 2024 (precision against Hansen)
Volume head (AlphaEarth only; trained on LANDFIRE in California):
  wc check      canopy cover from the 2020 embedding against WorldCover 2020 trees (10): AUC, and lifeform "tree"
                share on WorldCover trees vs not
  loss size     Hansen loss pixels: volume the year before vs the year after (median ratio); never-lost forest same
  after         lifeform shares on Hansen loss pixels in 2025
Writes results/squares.json. Squares scored in parallel: 4 processes, ~6 GB each at peak (~25 GB together; 6 reached 37 GB);
stops everything if this process and its workers pass --max-gb (40).

    uv run scripts/score_squares.py [--jobs 6] [--max-gb 40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
SQ = HERE / "data" / "squares"
YEARS = list(range(2018, 2026))
CAND = list(range(2019, 2025))
EMB = ("aef", "tes", "olmo")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def deq_signed(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def auc(y, s):
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return None
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r3(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 3)


def flags(z, k, D):
    """Per embedding: gated flags and ungated flags (len(CAND), n), the gate on e[Y-1] (len(CAND), n), the year of the
    largest single-year change (2019 to 2025), and the valid mask over every year."""
    if k == "aef":
        E = unit(deq_signed(z["aef"]))
        ok = (z["aef"][..., 0] != -128).all(0)
    elif k == "tes":
        s = z["tes_scale"]
        E = unit(z["tes"].astype(np.float32) * np.nan_to_num(s)[..., None])
        ok = (np.isfinite(s) & (s > 0)).all(0)
    else:
        E = unit(deq_signed(z["olmo"]))
        ok = (z["olmo"][..., 0] != -128).all(0)
    n = E.shape[1]
    out, raw, gate = (np.zeros((len(CAND), n), bool) for _ in range(3))
    for i, Y in enumerate(CAND):
        a, b = E[YEARS.index(Y - 1)], E[YEARS.index(Y + 1)]
        plain = 1 - (a * b).sum(1)
        g = ((a - D[f"{k}_gate_mean"]) / D[f"{k}_gate_scale"]) @ D[f"{k}_gate_coef"] + D[f"{k}_gate_b"]
        raw[i], gate[i] = plain >= D[f"{k}_cut"], g >= D[f"{k}_gate_at"]
        out[i] = raw[i] & gate[i]
    step = 1 - (E[1:] * E[:-1]).sum(-1)  # change into 2019 .. 2025
    date = np.array(YEARS[1:])[step.argmax(0)]
    return out, raw, gate, date, ok


def dilate(mask, side, r=3):
    """mask (side*side,) grown by r pixels (a 30 m Hansen pixel is about 3 of ours)."""
    m = mask.reshape(side, side)
    p = np.pad(m, r)
    o = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= p[r + dy:r + dy + side, r + dx:r + dx + side]
    return o.ravel()


def score(name):
    import volume_head as vh
    D = dict(np.load(HERE / "models" / "detectors.npz"))
    z = np.load(SQ / f"{name}.npz")
    loss, cover, wc20 = z["loss"].astype(int), z["cover"].astype(int), z["wc2020"]
    side = int(round(np.sqrt(len(loss))))
    L = np.where((loss >= 19) & (loss <= 24), 2000 + loss, 0)
    near_loss = dilate(loss > 0, side)
    forest = (loss == 0) & ~near_loss & (cover >= 30)
    open_ = (loss == 0) & ~near_loss & (cover < 10) & (wc20 != 10) & (wc20 > 0)
    anyloss = (loss >= 17) & (loss <= 24)
    F, R, G, DT, OKS = {}, {}, {}, {}, {}
    for k in EMB:
        F[k], R[k], G[k], DT[k], OKS[k] = flags(z, k, D)
        F[k] &= OKS[k]
    nvalid = sum(OKS[k].astype(int) for k in EMB)
    F["vote"] = (F["aef"].astype(int) + F["tes"] + F["olmo"]) >= 2
    OKS["vote"] = nvalid >= 2
    res = {"n": {k: int(v.sum()) for k, v in OKS.items()}, "hansen_loss_px": int((L > 0).sum()),
           "forest_px": int(forest.sum()), "open_px": int(open_.sum()), "det": {}}
    cand = np.array(CAND)
    col = np.arange(len(L))
    for k, f in F.items():
        OK = OKS[k]
        anyf = f.any(0)
        hl = (L > 0) & OK
        near = np.zeros(len(L), bool)
        near_raw = np.zeros(len(L), bool)
        for d in (-1, 0, 1):
            Ly = L + d
            idx = np.clip(np.searchsorted(cand, Ly), 0, len(CAND) - 1)
            valid = (Ly >= CAND[0]) & (Ly <= CAND[-1])
            near |= valid & f[idx, col]
            if k != "vote":
                near_raw |= valid & R[k][idx, col]
        found = hl & near
        if k == "vote":  # date by the embeddings that flagged, majority of their dates
            dates = np.stack([np.where(F[e].any(0), DT[e], 0) for e in EMB])
            date = np.array([np.bincount(c[c > 0]).argmax() if (c > 0).any() else 0 for c in dates.T]) if found.any() else np.zeros(len(L), int)
        else:
            date = DT[k]
        right = found & ((date == L) | (date == L + 1))
        r = {"found": r3(found.sum() / max(hl.sum(), 1)), "year_right": r3(right.sum() / max(found.sum(), 1)),
             "forest_lit": r3((anyf & forest & OK).sum() / max((forest & OK).sum(), 1)),
             "open_lit": r3((anyf & open_ & OK).sum() / max((open_ & OK).sum(), 1)),
             "agrees": r3((anyf & OK & anyloss).sum() / max((anyf & OK).sum(), 1)),
             "flagged_px": int((anyf & OK).sum()),
             "offset": {str(o): r3(((date - L)[found] == o).mean()) if found.any() else None for o in (-1, 0, 1, 2)}}
        if k != "vote":
            idx = np.clip(np.searchsorted(cand, L), 0, len(CAND) - 1)
            r["found_ungated"] = r3((hl & near_raw).sum() / max(hl.sum(), 1))
            r["gate_pass_on_loss"] = r3(G[k][idx, col][hl & (L >= CAND[0])].mean()) if (hl & (L >= CAND[0])).any() else None
        res["det"][k] = r
    OK = OKS["vote"]
    # flagged with no Hansen loss at all: a few positions for a look against imagery later
    nohl = np.flatnonzero(F["vote"].any(0) & OK & (loss == 0))
    rc = z["rc"]
    w = rc[3] - rc[2]
    pick = nohl[np.linspace(0, len(nohl) - 1, min(5, len(nohl))).astype(int)] if len(nohl) else []
    res["vote_no_hansen_examples"] = [[round(float(-180 + (rc[2] + i % w + 0.5) * 8.983111749910169e-05), 5),
                                       round(float(83.68570533713473 - (rc[0] + i // w + 0.5) * 8.983111749910169e-05), 5)]
                                      for i in pick]
    del F, R, G
    # volume head
    V = [vh.heads(z["aef"][i]) for i in range(len(YEARS))]
    i20 = YEARS.index(2020)
    OK = OKS["aef"]
    m = OK & (wc20 > 0)
    tree = wc20 == 10
    res["wc_check"] = {"cc_auc_vs_wc_trees": r3(auc(tree[m], V[i20]["cc"][m])),
                       "life_tree_on_wc_trees": r3((V[i20]["life"][m & tree] == 1).mean()) if (m & tree).any() else None,
                       "life_tree_on_wc_other": r3((V[i20]["life"][m & ~tree] == 1).mean()) if (m & ~tree).any() else None,
                       "wc_tree_share": r3(tree[m].mean())}
    ratios, steady = [], []
    for Y in CAND:
        a, b = V[YEARS.index(Y - 1)]["vol"], V[YEARS.index(Y + 1)]["vol"]
        sel = OK & (L == Y)
        if sel.sum():
            ratios.append(np.log((b[sel] + 0.05) / (a[sel] + 0.05)))
        steady.append(np.log((b[forest & OK] + 0.05) / (a[forest & OK] + 0.05)))
    r = np.concatenate(ratios) if ratios else np.array([np.nan])
    s = np.concatenate(steady) if steady else np.array([np.nan])
    sel = OK & (L > 0)
    bef = np.concatenate([V[YEARS.index(Y - 1)]["vol"][OK & (L == Y)] for Y in CAND]) if sel.any() else np.array([np.nan])
    res["loss_size"] = {"vol_before_median_m": r3(np.nanmedian(bef)), "loss_ratio_median": r3(np.exp(np.nanmedian(r))),
                        "forest_ratio_median": r3(np.exp(np.nanmedian(s))),
                        "cc_before": r3(np.nanmedian(np.concatenate([V[YEARS.index(Y - 1)]["cc"][OK & (L == Y)] for Y in CAND]))) if sel.any() else None}
    life25 = V[-1]["life"][sel]
    res["after_2025"] = {vh.LIFE[i]: r3((life25 == i).mean()) for i in range(4)} if sel.any() else {}
    life_before = np.concatenate([V[YEARS.index(Y - 1)]["life"][OK & (L == Y)] for Y in CAND]) if sel.any() else np.array([])
    res["before"] = {vh.LIFE[i]: r3((life_before == i).mean()) for i in range(4)} if len(life_before) else {}
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=40)
    a = ap.parse_args()
    import os
    import threading
    import time
    import psutil
    me, peak = psutil.Process(), [0.0]

    def guard():
        while True:
            tot = me.memory_info().rss
            for c in me.children(recursive=True):
                try:
                    tot += c.memory_info().rss
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], tot / 1e9)
            if tot / 1e9 > a.max_gb:
                print(f"STOP: {tot / 1e9:.1f} GB across workers", flush=True)
                for c in me.children(recursive=True):
                    c.kill()
                os._exit(3)
            time.sleep(1)
    threading.Thread(target=guard, daemon=True).start()
    meta = json.load(open(HERE / "data" / "squares.json"))
    names = [n for n in meta if (SQ / f"{n}.npz").exists()]
    print(f"scoring {len(names)} squares", flush=True)
    out = dict(Parallel(n_jobs=a.jobs, verbose=0)(delayed(score)(n) for n in names))
    for n in out:
        out[n]["driver"] = meta[n]["driver"]
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "squares.json", "w"), indent=1)
    print(f"peak {peak[0]:.1f} GB across workers", flush=True)
    print(f"{'square':22s} {'driver':10s} " + " ".join(f"{k:>34s}" for k in ("aef", "tes", "olmo", "vote")))
    print(" " * 34 + "   found/year/forest/open/agrees" * 4)
    for n, r in out.items():
        cells = []
        for k in ("aef", "tes", "olmo", "vote"):
            d = r["det"][k]
            cells.append(f"{d['found']:.2f}/{d['year_right']:.2f}/{d['forest_lit']:.3f}/{d['open_lit']:.3f}/{d['agrees']:.2f}")
        print(f"{n:22s} {r['driver']:10s} " + " ".join(f"{c:>34s}" for c in cells))
    print()
    for k in ("aef", "tes", "olmo", "vote"):
        ms = ("found", "year_right", "forest_lit", "open_lit", "agrees") + (("found_ungated", "gate_pass_on_loss") if k != "vote" else ())
        med = {m: r3(np.median([r["det"][k][m] for r in out.values() if r["det"][k].get(m) is not None])) for m in ms}
        print(f"median {k:5s} {med}")
    print()
    for n, r in out.items():
        print(f"{n:22s} wc {r['wc_check']}  loss {r['loss_size']}  before {r['before']}  after {r['after_2025']}")


if __name__ == "__main__":
    main()
