# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""The squeezed top end of biomass: how much is CTrees' own error, and two fixes.

1. CTrees' error. The training cells (data/rows.npz) were drawn with a seeded generator; replaying the draw gives
   each cell's id, and so CTrees' residual standard error per cell (data/labels/<name>.npz, agb_err). The replay is
   checked against the stored biomass. Per bin of CTrees biomass: CTrees' error beside the heads' miss.
2. Quantile mapping. The held-out predictions (data/heads/<input>_square.npz) mapped through
   F_true^-1(F_pred(p)), both quantile functions fit on the other folds' cells: undoes the pull toward the middle
   without retraining (also spreads the noise).
3. A head taught with more weight on high biomass (biomass_head.py --weight-top, written to
   data/heads/<input>_square_top.npz), scored here beside the plain head when present.

Writes results/top_end.json and data/rows_err.npz (agb_err per row cell). RAM about 2 GB.

    uv run scripts/top_end.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import biomass_rows as br  # noqa: E402

BINS = [0, 10, 50, 150, 250, 300, 400, 1e9]


def replay(K=500, P=8):
    """agb_err (G, Y) and cell ids for the rows, by replaying biomass_rows.py's draws (seed 0, same order)."""
    rng = np.random.default_rng(0)
    errs, agbs = [], []
    for n, s, kind in br.squares():
        L = np.load(HERE / "data" / "labels" / f"{n}.npz")
        cell = L["cell"]
        ids, cnt = np.unique(cell, return_counts=True)
        ok = ids[cnt >= 40]
        pick = rng.choice(ok, size=min(K, ok.size), replace=False)
        order = np.argsort(cell, kind="stable")
        starts = np.searchsorted(cell[order], pick)
        counts = cnt[np.searchsorted(ids, pick)]
        pix = np.stack([order[st + rng.choice(c, P, replace=False)] for st, c in zip(starts, counts)])
        first = pix[:, 0]
        e = L["agb_err"][:, first].T.astype(np.float32)
        a = L["agb"][:, first].T.astype(np.float32)
        errs.append(np.where(e >= 0, e / 10, np.nan))
        agbs.append(np.where(a >= 0, a / 10, np.nan))
    return np.concatenate(errs), np.concatenate(agbs)


def qmap(p_tr, t_tr, p_te, n=201):
    qs = np.linspace(0, 1, n)
    fp = np.quantile(np.log1p(p_tr), qs)
    ft = np.quantile(np.log1p(t_tr), qs)
    u = np.interp(np.log1p(p_te), fp, qs)
    return np.expm1(np.interp(u, qs, ft))


def table(t, p, err=None):
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    out = {"r2_log": round(bh.r2(np.log1p(t), np.log1p(p)), 3), "median_err": round(float(np.median(np.abs(t - p))), 1),
           "bins": {}}
    e = err[m] if err is not None else None
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = (t >= lo) & (t < hi)
        if b.sum() < 30:
            continue
        row = {"n": int(b.sum()), "true_median": round(float(np.median(t[b])), 1),
               "pred_median": round(float(np.median(p[b])), 1),
               "miss_median": round(float(np.median(np.abs(t[b] - p[b]))), 1)}
        if e is not None:
            ee = e[b]
            row["ctrees_err_median"] = round(float(np.nanmedian(ee)), 1)
            row["miss_within_2err"] = round(float((np.abs(t[b] - p[b]) <= 2 * ee).mean()), 3)
        out["bins"][f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"] = row
    return out


def main():
    R = dict(np.load(HERE / "data" / "rows.npz"))
    err, agb = replay()
    same = np.isfinite(agb) == np.isfinite(R["agb"])
    agree = np.nanmax(np.abs(agb - R["agb"])) if same.all() else np.inf
    print(f"replay check: finite pattern equal {bool(same.all())}, max |agb diff| {agree:.3f}", flush=True)
    if not (same.all() and agree < 0.06):
        raise SystemExit("replay does not match rows.npz")
    np.savez(HERE / "data" / "rows_err.npz", agb_err=err)
    res = {"replay_max_diff": round(float(agree), 3)}
    t = R["agb"]
    fold = np.zeros(len(t), int)
    for fname, test in bh.folds(R, "square"):
        fold[test] = int(fname)
    for inp in ("aef", "tes", "olmo", "all"):
        res[inp] = {}
        for tag in ("", "_top"):
            f = HERE / "data" / "heads" / f"{inp}_square{tag}.npz"
            if not f.exists():
                continue
            p = np.load(f)["pa"].astype(np.float64)
            res[inp]["plain" + tag] = table(t.ravel(), p.ravel(), err.ravel())
            q = np.full_like(p, np.nan)
            for k in np.unique(fold):
                tr, te = fold != k, fold == k
                m = np.isfinite(t[tr]) & np.isfinite(p[tr])
                pt = p[te]
                ok = np.isfinite(pt)
                qq = np.full_like(pt, np.nan)
                qq[ok] = qmap(p[tr][m], t[tr][m], pt[ok])
                q[te] = qq
            res[inp]["quantile_mapped" + tag] = table(t.ravel(), q.ravel(), err.ravel())
        for meth, v in res[inp].items():
            b = v["bins"]
            print(f"{inp:5s} {meth:20s} R2 log {v['r2_log']} med err {v['median_err']} | " + "  ".join(
                f"{k}: {x['pred_median']} vs {x['true_median']}" + (f" (CTrees err {x['ctrees_err_median']}, within 2 err {x['miss_within_2err']})" if k in ("300-400", "400+") else "")
                for k, x in b.items() if k in ("10-50", "150-250", "250-300", "300-400", "400+")), flush=True)
    json.dump(res, open(HERE / "results" / "top_end.json", "w"), indent=1)


if __name__ == "__main__":
    main()
