"""Freeze the tree-cut detectors exactly as trees3.py fit them, before any new place is looked at.

Per embedding (AEF, TESSERA, OlmoEarth): the plain-change cutoff (80 % of cut pixel-years found over the 19 forest
squares) and the "forest the year before" gate (logistic regression on the before-year embedding, forest squares
vs the non-forest quiet squares; cutoff keeps 95 % of cut pixels). Same rows, seed and code as trees3.py.

Writes models/detectors.npz: <emb>_cut, <emb>_gate_mean, _gate_scale, _gate_coef, _gate_b, _gate_at.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with threadpoolctl python scripts/freeze_detectors.py
RAM: about 2 GB.
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trees3 as t  # noqa: E402  (chdirs into Earthwork, as trees3 itself does)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models"


def main():
    rng = np.random.default_rng(0)
    before, plain, ys, cutmask = {k: [] for k in t.EMB}, {k: [] for k in t.EMB}, [], {k: [] for k in t.EMB}
    for n in t.FOREST:
        E, ok, ev, cut = t.forest_square(n)
        i0 = t.YEARS.index(max(ev - 1, t.YEARS[0]))
        keep = rng.choice(np.flatnonzero(ok), min(t.PX, ok.sum()), replace=False)
        y = np.concatenate([(cut[keep] & (ev == Y)).astype(np.int8) for Y in t.CAND])
        ys.append(y)
        for k in t.EMB:
            before[k].append(E[k][i0, keep])
            cutmask[k].append(cut[keep])
            plain[k].append(np.concatenate([1 - (E[k][t.YEARS.index(Y - 1), keep] * E[k][t.YEARS.index(Y + 1), keep]).sum(1)
                                            for Y in t.CAND]))
    pos = np.concatenate(ys) == 1
    quiet = {}
    for f in sorted(glob.glob(str(t.EW / "data/years/*.npz"))):
        n = Path(f).stem
        if n in t.QUIET_FOREST:
            continue
        z = np.load(f, allow_pickle=True)
        o, ook = t.olmo_deq(np.load(t.OLMO / f"{n}.npz")["q"][0])
        y0 = t.YEARS[0]
        A = t.unit(z[f"aef_{y0}"])
        T = t.unit(z[f"{y0}_q"].astype(np.float32) * np.nan_to_num(z[f"{y0}_scales"])[:, None])
        ok = z[f"aok_{y0}"] & np.isfinite(z[f"{y0}_scales"]) & (z[f"{y0}_scales"] > 0) & ook
        keep = rng.choice(np.flatnonzero(ok), min(3000, ok.sum()), replace=False) if ok.sum() else np.array([], int)
        quiet[n] = {"aef": A[keep], "tes": T[keep], "olmo": o[keep]}
    out = {}
    for k in t.EMB:
        gp, gn = np.concatenate(before[k]), np.concatenate([q[k] for q in quiet.values()])
        X = np.concatenate([gp, gn])
        y = np.r_[np.ones(len(gp)), np.zeros(len(gn))].astype(int)
        sc = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced").fit(sc.transform(X), y)
        g = lambda Z: m.decision_function(sc.transform(Z))  # noqa: E731
        out[f"{k}_gate_mean"], out[f"{k}_gate_scale"] = sc.mean_, sc.scale_
        out[f"{k}_gate_coef"], out[f"{k}_gate_b"] = m.coef_[0], m.intercept_
        out[f"{k}_gate_at"] = np.quantile(g(np.concatenate([b[c] for b, c in zip(before[k], cutmask[k])])), 0.05)
        out[f"{k}_cut"] = np.quantile(np.concatenate(plain[k])[pos], 0.2)
        print(f"{k}: cut {float(out[f'{k}_cut']):.4f}  gate_at {float(out[f'{k}_gate_at']):.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "detectors.npz", **out)
    print("wrote", OUT / "detectors.npz")


if __name__ == "__main__":
    main()
