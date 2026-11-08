# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""CTrees against ESA CCI Biomass v7.0 against the heads, on the training cells of every square.

The cells of data/rows.npz (500 a square, 8 pixels each) are recovered by replaying their seeded draw (as top_end.py
does); CCI (data/cci/<name>.npz, scripts/read_cci.py) is averaged over the same 8 pixels, 2018 to 2024. The heads'
numbers are the square-held-out predictions (data/heads/<input>_square.npz). Per kind of square: medians of the three,
rank correlation of each pair, and for 2018 to 2024 the change. CCI counts the woody parts of living trees only, so
grass and crop read near 0 there by definition (CTrees' own definition is not checked here).

Writes results/cci_compare.json. RAM about 2 GB.

    uv run scripts/cci_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import biomass_rows as br  # noqa: E402


def replay_pix(K=500, P=8):
    rng = np.random.default_rng(0)
    out = []
    for n, s, kind in br.squares():
        L = np.load(HERE / "data" / "labels" / f"{n}.npz")
        cell = L["cell"]
        ids, cnt = np.unique(cell, return_counts=True)
        ok = ids[cnt >= 40]
        pick = rng.choice(ok, size=min(K, ok.size), replace=False)
        order = np.argsort(cell, kind="stable")
        starts = np.searchsorted(cell[order], pick)
        counts = cnt[np.searchsorted(ids, pick)]
        out.append((n, np.stack([order[st + rng.choice(c, P, replace=False)] for st, c in zip(starts, counts)])))
    return out


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(bh.spearman(a[m], b[m]), 3) if m.sum() > 50 else None


def main():
    R = np.load(HERE / "data" / "rows.npz")
    kinds = R["kinds"][R["sq"]]
    t = R["agb"][:, :7]  # 2018..2024
    cci = []
    for n, pix in replay_pix():
        C = np.load(HERE / "data" / "cci" / f"{n}.npz")
        a = C["agb"][:, pix].astype(np.float32)  # (7, G, P)
        a[a == 65535] = np.nan
        cci.append(np.nanmean(a, -1).T)  # (G, 7)
    cci = np.concatenate(cci)
    heads = {inp: np.load(HERE / "data" / "heads" / f"{inp}_square.npz")["pa"][:, :7] for inp in ("aef", "olmo", "all")}
    res = {}
    for g in ["all"] + sorted(set(R["kinds"])):
        m = np.ones(len(kinds), bool) if g == "all" else kinds == g
        o = {"cells": int(m.sum()),
             "median_ctrees": round(float(np.nanmedian(t[m])), 1), "median_cci": round(float(np.nanmedian(cci[m])), 1),
             "share_cci_zero": round(float(np.nanmean(cci[m] < 1)), 3),
             "spearman_ctrees_cci": sp(t[m].ravel(), cci[m].ravel())}
        for inp, p in heads.items():
            o[f"median_{inp}"] = round(float(np.nanmedian(p[m])), 1)
            o[f"spearman_{inp}_ctrees"] = sp(p[m].ravel(), t[m].ravel())
            o[f"spearman_{inp}_cci"] = sp(p[m].ravel(), cci[m].ravel())
        dt, dc = t[m, 6] - t[m, 0], cci[m, 6] - cci[m, 0]
        o["change_2018_2024"] = {"median_ctrees": round(float(np.nanmedian(dt)), 1), "median_cci": round(float(np.nanmedian(dc)), 1),
                                 "spearman_ctrees_cci": sp(dt, dc)}
        for inp, p in heads.items():
            dp = p[m, 6] - p[m, 0]
            o["change_2018_2024"][f"spearman_{inp}_ctrees"] = sp(dp, dt)
            o["change_2018_2024"][f"spearman_{inp}_cci"] = sp(dp, dc)
        res[g] = o
        ch = o["change_2018_2024"]
        print(f"{g:13s} CTrees {o['median_ctrees']:6.1f}  CCI {o['median_cci']:6.1f} (zero {o['share_cci_zero']})  "
              f"olmo {o['median_olmo']:6.1f}  | rank: CTrees-CCI {o['spearman_ctrees_cci']}  olmo-CTrees {o['spearman_olmo_ctrees']}  "
              f"olmo-CCI {o['spearman_olmo_cci']}  | change rank: CTrees-CCI {ch['spearman_ctrees_cci']}  "
              f"olmo-CTrees {ch['spearman_olmo_ctrees']}  olmo-CCI {ch['spearman_olmo_cci']}", flush=True)
    json.dump(res, open(HERE / "results" / "cci_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
