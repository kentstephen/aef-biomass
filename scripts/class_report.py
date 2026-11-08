# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Land cover from the biomass head, with its margins of error: per class, and per pixel from its own confidence.

The class output of biomass_head.py, square-held-out (each square by the fold head that never saw it), on the pixels
of 2020 and 2021 where ESA WorldCover and Impact Observatory agree (data/rows.npz). Per input:
  confusion matrix; per class precision and recall with a 95 % interval from resampling whole squares (1,000 draws);
  overall accuracy with its interval; and calibration: pixels binned by the head's top class probability, the share
  right in each bin (a head whose 0.8 means right 80 % of the time gives a usable per-pixel error).
The labels are the easy pixels (two maps agree), so these scores run high against ground where maps disagree.

Writes results/class_report.json. RAM about 4 GB (rows and one input's features).

    uv run scripts/class_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

YI = [2, 3]  # 2020, 2021 in 2018..2025
BINS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def main():
    R = dict(np.load(HERE / "data" / "rows.npz"))
    classes = list(R["classes"])
    nsq = len(R["names"])
    res = {"classes": classes, "note": "labels: WorldCover 2020 / 2021 where Impact Observatory agrees"}
    rng = np.random.default_rng(0)
    for inp in ("aef", "tes", "olmo", "all"):
        X, ok = bh.build(R, inp)
        W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
        d = X.shape[-1]
        T, P, C, S = [], [], [], []
        for fname, test in bh.folds(R, "square"):
            net = bh.Head(d, np.zeros(d, np.float32), np.ones(d, np.float32))
            net.load_state_dict({k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(fname + ".")})
            net.eval()
            for b in range(0, test.size, 2000):
                g = test[b:b + 2000]
                x = torch.as_tensor(X[g][:, YI].astype(np.float32))
                with torch.no_grad():
                    pr = torch.softmax(net(x)[..., 2:], -1).numpy()  # (B, 2, P, 9)
                lab = R["cls"][g][:, YI]
                m = (lab >= 0) & ok[g][:, YI]
                T.append(lab[m])
                P.append(pr.argmax(-1)[m])
                C.append(pr.max(-1)[m])
                S.append(np.broadcast_to(R["sq"][g][:, None, None], lab.shape)[m])
        t, p, c, s = (np.concatenate(v) for v in (T, P, C, S))
        k = len(classes)
        conf = np.zeros((k, k), np.int64)
        np.add.at(conf, (t, p), 1)
        # per-square counts, for resampling whole squares
        per = np.zeros((nsq, k, k), np.int64)
        np.add.at(per, (s, t, p), 1)
        draws = [per[rng.integers(0, nsq, nsq)].sum(0) for _ in range(1000)]

        def ci(fn):
            v = np.array([fn(m) for m in draws], float)
            v = v[np.isfinite(v)]
            return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v.size else None

        out = {"pixels": int(t.size), "accuracy": round(float(np.trace(conf) / conf.sum()), 3),
               "accuracy_ci": ci(lambda m: np.trace(m) / m.sum()), "per_class": {}, "confusion": conf.tolist()}
        for i, nm in enumerate(classes):
            if conf[i].sum() < 50:
                continue
            out["per_class"][nm] = {
                "n": int(conf[i].sum()),
                "recall": round(float(conf[i, i] / conf[i].sum()), 3),
                "recall_ci": ci(lambda m, i=i: m[i, i] / m[i].sum() if m[i].sum() else np.nan),
                "precision": round(float(conf[i, i] / max(conf[:, i].sum(), 1)), 3),
                "precision_ci": ci(lambda m, i=i: m[i, i] / m[:, i].sum() if m[:, i].sum() else np.nan),
                "mostly_confused_with": classes[int(np.argsort(conf[i])[-2] if np.argmax(conf[i]) == i else np.argmax(conf[i]))],
            }
        cal = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (c >= lo) & (c < hi)
            if m.sum() >= 100:
                cal.append({"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "share_of_pixels": round(float(m.mean()), 3),
                            "mean_confidence": round(float(c[m].mean()), 3), "right": round(float((t[m] == p[m]).mean()), 3)})
        out["calibration"] = cal
        out["ece"] = round(float(sum(abs(b["mean_confidence"] - b["right"]) * b["share_of_pixels"] for b in cal)), 3)
        res[inp] = out
        print(f"{inp:5s} acc {out['accuracy']} {out['accuracy_ci']}  ece {out['ece']}  " + "  ".join(
            f"{nm} R{v['recall']} P{v['precision']}" for nm, v in out["per_class"].items()), flush=True)
        print("      calibration " + "  ".join(f"{b['confidence']}: {b['right']} ({b['share_of_pixels']})" for b in cal), flush=True)
        del X, ok
    json.dump(res, open(HERE / "results" / "class_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
