# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Per-pixel margin of error for the land cover classes: temperature scaling and conformal class sets.

Same pixels as class_report.py: 2020 and 2021, where ESA WorldCover and Impact Observatory agree, each square by the
fold head that never saw it (data/heads/<input>_square.npz). Both fixes are fit across folds: for the squares of fold
f, the temperature and the set cutoffs come from the held-out predictions on the other folds only, so no square is
calibrated on itself.
  temperature  one number T dividing the logits, fit by log loss; the top probability then reads as the chance of
               being right (calibration error before and after, by confidence bin)
  sets         split conformal: every class whose probability passes a cutoff, the cutoff chosen so the true class
               is inside 90 % (and 95 %) of the time on the other folds; a pixel's set is its margin of
               error (one class = sure, two or three = which classes it may be). Coverage and set size per class and
               per kind of square, since coverage holds on average, not for every kind.
These are the easy pixels (two maps agree): on ground where maps disagree the error is larger than this says.

Writes results/class_calibrate.json and models/class_temperature.json. RAM about 5 GB.

    uv run scripts/class_calibrate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

YI = [2, 3]  # 2020, 2021
BINS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
LEVELS = (0.90, 0.95)


def logits(R, X, ok, W, d):
    """Held-out class logits on the agreed pixels: (N, 9) float32, true class, square, fold name."""
    L, T, S, F = [], [], [], []
    for fname, test in bh.folds(R, "square"):
        net = bh.Head(d, np.zeros(d, np.float32), np.ones(d, np.float32))
        net.load_state_dict({k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(fname + ".")})
        net.eval()
        for b in range(0, test.size, 2000):
            g = test[b:b + 2000]
            with torch.no_grad():
                o = net(torch.as_tensor(X[g][:, YI].astype(np.float32)))[..., 2:].numpy()
            lab = R["cls"][g][:, YI]
            m = (lab >= 0) & ok[g][:, YI]
            L.append(o[m])
            T.append(lab[m])
            S.append(np.broadcast_to(R["sq"][g][:, None, None], lab.shape)[m])
            F.append(np.full(int(m.sum()), int(fname)))
    return (np.concatenate(v) for v in (L, T, S, F))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_t(z, t):
    lt = torch.zeros(1, requires_grad=True)  # log T
    zt, tt = torch.as_tensor(z), torch.as_tensor(t.astype(np.int64))
    opt = torch.optim.LBFGS([lt], lr=0.5, max_iter=100)

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zt / lt.exp(), tt)
        loss.backward()
        return loss
    opt.step(step)
    return float(lt.detach().exp())


def reliability(p, t):
    conf, pred = p.max(-1), p.argmax(-1)
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 100:
            out.append({"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "share_of_pixels": round(float(m.mean()), 3),
                        "mean_confidence": round(float(conf[m].mean()), 3), "right": round(float((pred[m] == t[m]).mean()), 3)})
    ece = sum(abs(b["mean_confidence"] - b["right"]) * b["share_of_pixels"] for b in out)
    return out, round(float(ece), 4)


def set_scores(p, t):
    """Conformal score: one minus the probability given to the true class."""
    return 1 - p[np.arange(len(t)), t]


def sets(p, q):
    """Every class whose probability is at least 1 - q (q one number, or one per class; the top class always, so no
    set is empty)."""
    member = p >= 1 - np.asarray(q)
    member[np.arange(len(p)), p.argmax(-1)] = True
    return member


def main():
    R = dict(np.load(HERE / "data" / "rows.npz"))
    classes = list(R["classes"])
    kinds = np.array(R["kinds"])
    res = {"classes": classes, "note": "2020-2021 pixels where WorldCover and Impact Observatory agree; fits across folds"}
    temps = {}
    qc_all = {}
    for inp in ("aef", "tes", "olmo", "all"):
        X, ok = bh.build(R, inp)
        W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
        z, t, s, f = logits(R, X, ok, W, X.shape[-1])
        del X, ok
        pt = np.empty_like(z)
        member = {lv: np.zeros_like(z, bool) for lv in LEVELS}
        member |= {("class", lv): np.zeros_like(z, bool) for lv in LEVELS}
        tf = {}
        qf = {lv: [] for lv in LEVELS}
        for fold in np.unique(f):
            tr, te = f != fold, f == fold
            T = fit_t(z[tr], t[tr])
            tf[int(fold)] = round(T, 3)
            pt[te] = softmax(z[te] / T)
            sc = set_scores(softmax(z[tr] / T), t[tr])
            for lv in LEVELS:
                n = sc.size
                q = np.quantile(sc, min(1, np.ceil((n + 1) * lv) / n), method="higher")
                member[lv][te] = sets(pt[te], q)
                qf[lv].append(float(q))
                # one cutoff per class, from that class's pixels on the other folds (each class keeps its level)
                qc = np.ones(z.shape[1])
                for c in range(z.shape[1]):
                    sc_c = sc[t[tr] == c]
                    if sc_c.size >= 50:
                        qc[c] = np.quantile(sc_c, min(1, np.ceil((sc_c.size + 1) * lv) / sc_c.size), method="higher")
                member[("class", lv)][te] = sets(pt[te], qc)
        T_all = fit_t(z, t)
        temps[inp] = round(T_all, 4)
        # cutoffs fit on every fold's pixels, for squares outside these 92 (a pixel's set: classes with p >= 1 - q)
        sc_all = set_scores(softmax(z / T_all), t)
        qc_all[inp] = {}
        for lv in LEVELS:
            q = [1.0] * z.shape[1]
            for c in range(z.shape[1]):
                s_c = sc_all[t == c]
                if s_c.size >= 50:
                    q[c] = float(np.quantile(s_c, min(1, np.ceil((s_c.size + 1) * lv) / s_c.size), method="higher"))
            qc_all[inp][f"{int(lv * 100)}"] = [round(v, 5) for v in q]
        raw, ece_raw = reliability(softmax(z), t)
        cal, ece_cal = reliability(pt, t)
        out = {"pixels": int(t.size), "temperature_by_fold": tf, "temperature_all": temps[inp],
               "ece_raw": ece_raw, "ece_calibrated": ece_cal, "reliability_raw": raw, "reliability_calibrated": cal,
               "sets": {}}
        for key in list(member):
            lv = key[1] if isinstance(key, tuple) else key
            mem = member[key]
            cov = mem[np.arange(t.size), t]
            size = mem.sum(-1)
            sv = {"coverage": round(float(cov.mean()), 3), "mean_size": round(float(size.mean()), 2),
                  "share_single": round(float((size == 1).mean()), 3),
                  "single_right": round(float(cov[size == 1].mean()), 3) if (size == 1).any() else None,
                  "per_class": {}, "per_kind": {}}
            for i, nm in enumerate(classes):
                m = t == i
                if m.sum() >= 50:
                    sv["per_class"][nm] = {"n": int(m.sum()), "coverage": round(float(cov[m].mean()), 3),
                                           "mean_size": round(float(size[m].mean()), 2)}
            for kd in sorted(set(kinds)):
                m = kinds[s] == kd
                if m.sum() >= 50:
                    sv["per_kind"][kd] = {"n": int(m.sum()), "coverage": round(float(cov[m].mean()), 3),
                                          "mean_size": round(float(size[m].mean()), 2)}
            # coverage per square: how often a single square falls well short
            per_sq = np.array([cov[s == q].mean() for q in np.unique(s)])
            sv["squares_below"] = {"under_0.8": int((per_sq < 0.8).sum()), "squares": int(per_sq.size),
                                   "min": round(float(per_sq.min()), 3)}
            if not isinstance(key, tuple):
                sv["min_prob_in_set"] = [round(1 - max(qf[lv]), 4), round(1 - min(qf[lv]), 4)]
            out["sets"][("by_class_" if isinstance(key, tuple) else "") + f"{int(lv * 100)}"] = sv
        res[inp] = out
        s90 = out["sets"]["90"]
        print(f"{inp:5s} T {temps[inp]} (folds {min(tf.values())}-{max(tf.values())})  ECE {ece_raw} -> {ece_cal}  "
              f"90% sets: coverage {s90['coverage']} size {s90['mean_size']} single {s90['share_single']} "
              f"(right {s90['single_right']})  [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        print("      calibrated " + "  ".join(f"{b['confidence']}: {b['mean_confidence']}->{b['right']} ({b['share_of_pixels']})" for b in cal))
        print("      90% coverage by class " + "  ".join(f"{k} {v['coverage']}/{v['mean_size']}" for k, v in s90["per_class"].items()))
        print("      by kind " + "  ".join(f"{k} {v['coverage']}/{v['mean_size']}" for k, v in s90["per_kind"].items()), flush=True)
        c90 = out["sets"]["by_class_90"]
        print(f"      class-wise 90%: coverage {c90['coverage']} size {c90['mean_size']} single {c90['share_single']} (right {c90['single_right']})")
        print("        by class " + "  ".join(f"{k} {v['coverage']}/{v['mean_size']}" for k, v in c90["per_class"].items()))
        print("        by kind " + "  ".join(f"{k} {v['coverage']}/{v['mean_size']}" for k, v in c90["per_kind"].items()), flush=True)
    json.dump(res, open(HERE / "results" / "class_calibrate.json", "w"), indent=1)
    json.dump({"temperature": temps, "class_cutoffs": qc_all, "classes": classes,
               "note": "divide the class logits of the <input> heads by the temperature before softmax; a pixel's "
                       "90 % (95 %) set holds every class c with probability >= 1 - class_cutoffs[level][c]"},
              open(HERE / "models" / "class_temperature.json", "w"), indent=1)


if __name__ == "__main__":
    main()
