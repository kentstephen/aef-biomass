# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Biomass change calibrated per window length, from the cell tables of biomass_change.py (no new predictions).

data/heads/change_cells_<input>.npz holds, per CTrees cell (341,571 of them), CTrees' biomass every year 2018 to
2025, and the heads' change for each one-year window and for 2018 to 2025 (plain difference, gated). One-year plain
changes add up, so every window of every length 1 to 7 years comes from them.

The first calibration (biomass_change.py) divided by one muting factor pooled over all windows; one-year windows
outnumber the rest seven to one, so the factor fit the yearly muting and overshot the long span. Here the factor is
fit per window length, across folds (the squares of fold f are scaled by a factor fit on the other folds' cells):
  plain / s_L    plain difference over a window of L years, divided by its own slope s_L (predicted on true)
  gated / s_L    the same on the gated change (counted only where the embedding moved), L = 1 and 7
  plain / l_L, gated / l_L   divided instead by the muting on clear loss cells only (median predicted over median
                 true): the slope s_L is dragged down by CTrees' own noise on small changes, so dividing by it
                 overshoots
Scored per window length on all cells, the tree-loss squares and the rest, with biomass_change.stats.

Writes results/change_calibrate.json. RAM under 2 GB.

    uv run scripts/change_calibrate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402

YEARS = bc.YEARS


def windows(L):
    return [(a, a + L) for a in range(0, len(YEARS) - L)]


def slope(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    c = np.cov(p[m], t[m])
    return c[0, 1] / c[1, 1]


def loss_ratio(p, t, base):
    """Median predicted over median true change on clear loss cells (CTrees fell by half and 20 Mg/ha or more), where
    the true change stands well above CTrees' own year-to-year noise."""
    m = np.isfinite(p) & np.isfinite(t) & np.isfinite(base)
    p, t, base = p[m], t[m], base[m]
    loss = (t / np.maximum(base, 1) <= -0.5) & (t <= -20)
    return np.median(p[loss]) / np.median(t[loss])


def main():
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    names = list(R["names"])
    fold_of_sq = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of_sq[int(s)] = fname
    res = {"note": "factor per window length, fit on the other folds' cells; stats as in biomass_change.py"}
    for inp in ("aef", "tes", "olmo", "all"):
        Z = np.load(HERE / "data" / "heads" / f"change_cells_{inp}.npz")
        T, kind, sq = Z["truth"], Z["kind"], Z["sq"]
        fold = np.array([fold_of_sq[int(s)] for s in sq])
        yearly = Z["plain"][:, :7]  # (C, 7): 2018-19 .. 2024-25
        cum = np.concatenate([np.zeros((len(T), 1)), np.cumsum(yearly, 1)], 1)  # (C, 8) change since 2018
        out = {}
        for L in range(1, 8):
            ws = windows(L)
            t = np.stack([T[:, b] - T[:, a] for a, b in ws], 1)
            base = np.stack([T[:, a] for a, b in ws], 1)
            meth = {"plain": np.stack([cum[:, b] - cum[:, a] for a, b in ws], 1)}
            if L == 1:
                meth["gated"] = Z["gated"][:, :7]
            if L == 7:
                meth["gated"] = Z["gated"][:, 7:8]
            for nm in list(meth):
                p = meth[nm]
                cal, cal_l = np.full_like(p, np.nan), np.full_like(p, np.nan)
                factors, factors_l = [], []
                for f in np.unique(fold):
                    tr = fold != f
                    s = slope(p[tr].ravel(), t[tr].ravel())
                    sl = loss_ratio(p[tr].ravel(), t[tr].ravel(), base[tr].ravel())
                    factors.append(s)
                    factors_l.append(sl)
                    cal[fold == f] = p[fold == f] / s
                    cal_l[fold == f] = p[fold == f] / sl
                meth[nm + "/s"] = cal
                meth[nm + "/l"] = cal_l
                out.setdefault(f"L{L}", {})[f"{nm}_factor"] = [round(float(min(factors)), 3), round(float(max(factors)), 3)]
                out[f"L{L}"][f"{nm}_loss_factor"] = [round(float(min(factors_l)), 3), round(float(max(factors_l)), 3)]
            for nm, p in meth.items():
                for gname in ("all", "forest-loss", "other"):
                    gm = np.ones(len(kind), bool) if gname == "all" else (kind == "forest-loss") if gname == "forest-loss" else (kind != "forest-loss")
                    tt, pp, aa = t[gm].ravel(), p[gm].ravel(), base[gm].ravel()
                    m = np.isfinite(tt) & np.isfinite(pp) & np.isfinite(aa)
                    out[f"L{L}"][f"{nm}/{gname}"] = bc.stats({"d": tt[m], "a": aa[m]}, pp[m])
        res[inp] = out
        print(f"--- {inp}")
        for L in range(1, 8):
            o = out[f"L{L}"]
            line = f"  L{L} factor {o['plain_factor']} loss factor {o['plain_loss_factor']}"
            for nm in ("plain", "plain/l", "gated", "gated/l"):
                if f"{nm}/forest-loss" in o:
                    a, b = o[f"{nm}/forest-loss"], o[f"{nm}/other"]
                    line += (f" | {nm}: slope {a.get('slope')} loss {a.get('loss_ratio')}/{a.get('loss_found')} "
                             f"steady {a.get('steady_lit')} other-steady {b.get('steady_lit')}")
            print(line, flush=True)
    json.dump(res, open(HERE / "results" / "change_calibrate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
