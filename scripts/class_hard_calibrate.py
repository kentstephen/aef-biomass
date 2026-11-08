# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""A margin of error for the classes that holds on hard ground: calibration fit on referee-labeled hard pixels.

Section 15 (class_hard_ground.py): where ESA WorldCover and Impact Observatory disagree, the head's confidence runs
high (.77 stated, about .63 right on trees against not trees). The temperature and set cutoffs of
class_calibrate.py were fit on pixels where the two maps agree. Here the same is fit on hard pixels instead.

Hard pixels: 2020 and 2021, both maps have a class, they disagree, and one says trees while the other does not.
Referee: Meta's 10 m mean canopy height (imagery 2017 on), 5 m or more = trees, under 2 m = not trees (between:
left out). Only trees against not trees can be refereed this way, so the calibration here is for that call:
P(trees) = the head's calibrated probability of the trees class.

Every square is scored by its held-out fold head; every fix is fit on the other folds' hard pixels only:
  before    the per-fold temperature of class_calibrate.py (fit on agreed pixels) and the class-wise 90 % cutoffs
            (trees in the set if p(trees) passes the trees cutoff; not trees if any other class is in the set)
  hard T    a second temperature, fit by log loss on P(trees) against the referee; used where the maps disagree
            (known at map time: both maps are public for 2020 and 2021)
  hard sets 90 % sets over {trees, not trees}, one cutoff per side, fit on the other folds' hard pixels
Reported: reliability of the trees / not call (confidence = max(P, 1 - P)) and the error between stated and actual,
set coverage and share of single-answer sets, overall, per kind of square and per referee side.

Transfer check, independent of Meta: the 18 lidar squares (none in training), canopy from airborne lidar
(data/lidar, the epoch nearest 2021, pixels whose canopy moved under 2 m between epochs), the same 5 m / 2 m rule,
on hard and on agreed pixels; heads = the 8 fold heads' calibrated probabilities averaged, the fits from all 92
squares' hard pixels.

Writes results/class_hard_calibrate.json and adds "hard_ground" to models/class_temperature.json. One square's two
years of features at a time: peak about 3 GB.

    uv run scripts/class_hard_calibrate.py [--inputs olmo all] [--per-square 40000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402
import biomass_rows as br  # noqa: E402

YEARS = bc.YEARS
BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
LEVEL = 0.90
DIMS = {"aef": 64, "tes": 128, "olmo": 128, "all": 320}


def feats(z, inp, yi):
    """float32 (n, D) and valid (n,) for one year."""
    def one(k):
        q = z[k][yi]
        if k == "tes":
            s = z["tes_scale"][yi]
            return q.astype(np.float32) * np.nan_to_num(s)[:, None], np.isfinite(s) & (s > 0)
        return bh.deq(q), q[:, 0] != -128
    if inp != "all":
        return one(inp)
    xs, ok = [], None
    for k in bc.EMB:
        x, o = one(k)
        xs.append(x)
        ok = o if ok is None else ok & o
    return np.concatenate(xs, -1), ok


def logits_of(net, x):
    with torch.no_grad():
        return np.concatenate([net(torch.as_tensor(x[b:b + 100000]))[:, 2:].numpy() for b in range(0, len(x), 100000)]) \
            if len(x) else np.zeros((0, 9), np.float32)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ptree(z, T):
    return softmax(z / T)[:, 0]


def fit_t_binary(z, truth):
    """Temperature by log loss on P(trees) against the referee."""
    lt = torch.zeros(1, requires_grad=True)
    zt, tt = torch.as_tensor(z), torch.as_tensor(truth.astype(np.float32))
    opt = torch.optim.LBFGS([lt], lr=0.5, max_iter=100)

    def step():
        opt.zero_grad()
        p = torch.softmax(zt / lt.exp(), -1)[:, 0].clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, tt)
        loss.backward()
        return loss
    opt.step(step)
    return float(lt.detach().exp())


def reliability(p, truth):
    """The trees / not call: confidence max(P, 1 - P), right when the call matches the referee."""
    conf = np.maximum(p, 1 - p)
    right = (p >= 0.5) == truth
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 100:
            out.append({"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "share": round(float(m.mean()), 3),
                        "stated": round(float(conf[m].mean()), 3), "right": round(float(right[m].mean()), 3)})
    ece = sum(abs(b["stated"] - b["right"]) * b["share"] for b in out)
    return {"right": round(float(right.mean()), 3), "stated": round(float(conf.mean()), 3), "ece": round(float(ece), 4),
            "bins": out}


def cutoffs(p, truth, level=LEVEL):
    """One cutoff per side: side s is in the set when its probability >= 1 - q_s."""
    q = []
    for side, ps in ((True, p), (False, 1 - p)):
        sc = 1 - ps[truth == side]
        n = sc.size
        q.append(float(np.quantile(sc, min(1, np.ceil((n + 1) * level) / n), method="higher")) if n >= 50 else 1.0)
    return q


def binary_sets(p, q):
    t_in, n_in = p >= 1 - q[0], (1 - p) >= 1 - q[1]
    top = p >= 0.5
    return t_in | top, n_in | ~top


def old_sets(probs, qc):
    """Class-wise 90 % sets of class_calibrate.py, read as trees / not trees."""
    mem = probs >= 1 - np.asarray(qc)[None]
    mem[np.arange(len(probs)), probs.argmax(-1)] = True
    return mem[:, 0], mem[:, 1:].any(-1)


def set_stats(t_in, n_in, truth, groups=None):
    cov = np.where(truth, t_in, n_in)
    single = t_in != n_in
    out = {"coverage": round(float(cov.mean()), 3), "single": round(float(single.mean()), 3),
           "single_right": round(float(cov[single].mean()), 3) if single.any() else None,
           "coverage_trees": round(float(cov[truth].mean()), 3) if truth.any() else None,
           "coverage_not": round(float(cov[~truth].mean()), 3) if (~truth).any() else None}
    if groups is not None:
        out["per_kind"] = {g: {"n": int((groups == g).sum()), "coverage": round(float(cov[groups == g].mean()), 3),
                               "single": round(float(single[groups == g].mean()), 3)}
                           for g in sorted(set(groups)) if (groups == g).sum() >= 200}
    return out


def hard_mask(wc, io):
    """Pixels with a class in both maps that disagree, one saying trees and the other not; and the agreed pixels."""
    wcc = np.full(wc.shape, -1, np.int8)
    for code, c in br.WC2C.items():
        wcc[wc == code] = c
    valid = (wcc >= 0) & np.isin(io, list(br.IO_OK))
    agree = br.classes(wc, io) >= 0
    tree_dis = (wcc == 0) != (io == 2)
    return valid & ~agree & tree_dis, valid & agree


def collect(inp, names, kinds, fold_of, rng, per_sq):
    """Held-out logits on refereed hard pixels of the 92 squares."""
    W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
    Z, TR, F, K, S = [], [], [], [], []
    for si, name in enumerate(names):
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        L = np.load(HERE / "data" / "labels" / f"{name}.npz")
        net = bc.head_from(W, fold_of[name], DIMS[inp])
        io_years = list(L["io_years"])
        hgt, hy = L["height"], L["height_date"] // 10000
        for wy in (2020, 2021):
            yi = YEARS.index(wy)
            hard, _ = hard_mask(z[f"wc{wy}"], L["io"][io_years.index(wy)])
            hok = (hgt != 255) & (hy >= 2017)
            tall, low = hok & (hgt >= 5), hok & (hgt < 2)
            x, ok = feats(z, inp, yi)
            sel = np.flatnonzero(hard & (tall | low) & ok)
            if sel.size > per_sq // 2:
                sel = np.sort(rng.choice(sel, per_sq // 2, replace=False))
            Z.append(logits_of(net, x[sel]))
            TR.append(tall[sel])
            F.append(np.full(sel.size, int(fold_of[name])))
            K.append(np.full(sel.size, kinds[si]))
            S.append(np.full(sel.size, si))
            del x
        del z, L
        if si % 20 == 19:
            print(f"  {inp} {si + 1}/{len(names)} [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    return (np.concatenate(v) for v in (Z, TR, F, K, S))


def lidar_check(inp, W, T_agree, T_hard, qc_old, q_hard):
    """The 18 lidar squares: lidar canopy as referee on hard and agreed pixels."""
    picks = json.load(open(HERE / "data" / "lidar_squares.json"))
    folds = sorted({k.split(".", 1)[0] for k in W.files if "." in k})
    nets = [bc.head_from(W, f, DIMS[inp]) for f in folds]
    pool = {g: {"z": [], "truth": [], "sq": []} for g in ("hard", "agree")}
    per = {}
    for name in picks:
        fl, fs, flab = (HERE / "data" / d / f"{name}.npz" for d in ("lidar", "squares", "labels"))
        if not (fl.exists() and fs.exists() and flab.exists()):
            continue
        Ld, z, L = np.load(fl), np.load(fs), np.load(flab)
        chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
        if str(Ld["source"]) == "werk10":
            ypx = Ld["year_px"]
            okl = (cov[0] >= 80) & (ypx >= 2018) & (ypx <= 2023)
            h = chm[0]
        else:
            e = int(np.argmin(np.abs(Ld["year"].astype(int) - 2021)))
            okl = np.all(cov >= 80, 0)
            if chm.shape[0] > 1:
                okl &= np.abs(chm[1] - chm[0]) < 2
            h = chm[e]
        io_years = list(L["io_years"])
        hard, agree = hard_mask(z["wc2021"], L["io"][io_years.index(2021)])
        x, ok = feats(z, inp, YEARS.index(2021))
        ref = okl & ok & ((h >= 5) | (h < 2))
        per[name] = {}
        for g, m in (("hard", hard), ("agree", agree)):
            sel = np.flatnonzero(m & ref)
            if sel.size < 200:
                continue
            zz = np.stack([logits_of(n, x[sel]) for n in nets])  # (8, n, 9)
            pool[g]["z"].append(zz)
            pool[g]["truth"].append(h[sel] >= 5)
            pool[g]["sq"].append(np.full(sel.size, name))
            per[name][g] = int(sel.size)
        del x, z, L, Ld
    out = {"pixels_per_square": per}
    for g, d in pool.items():
        if not d["z"]:
            continue
        zz = np.concatenate(d["z"], 1)
        truth = np.concatenate(d["truth"])
        o = {"px": int(truth.size), "trees_share": round(float(truth.mean()), 3)}
        for label, T in (("before", T_agree), ("hard_T", T_hard)):
            probs = np.mean([softmax(zf / T) for zf in zz], 0)
            p = probs[:, 0]
            o[label] = reliability(p, truth)
            if label == "before":
                o[label]["sets_90"] = set_stats(*old_sets(probs, qc_old), truth)
            else:
                o[label]["sets_90"] = set_stats(*binary_sets(p, q_hard), truth)
        out[g] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["olmo", "all"])
    ap.add_argument("--per-square", type=int, default=40000, help="refereed hard pixels kept per square (both years)")
    args = ap.parse_args()
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    names, kinds = list(R["names"]), list(R["kinds"])
    fold_of = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of[names[s]] = fname
    C = json.load(open(HERE / "models" / "class_temperature.json"))
    cal = json.load(open(HERE / "results" / "class_calibrate.json"))
    res = {"note": "trees against not trees on hard pixels (WorldCover and IO disagree, one says trees), referee Meta "
                   "height >= 5 m / < 2 m; every fit on the other folds; lidar squares as an outside check"}
    hard_models = {}
    for inp in args.inputs:
        rng = np.random.default_rng(0)
        z, truth, f, kd, sq = collect(inp, names, kinds, fold_of, rng, args.per_square)
        print(f"{inp}: {truth.size} refereed hard pixels, trees {truth.mean():.2f} "
              f"[{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        T_fold = {int(k): v for k, v in cal[inp]["temperature_by_fold"].items()}
        qc_old = C["class_cutoffs"][inp]["90"]
        p_before = np.empty(truth.size)
        p_hard = np.empty(truth.size)
        probs_before = np.empty((truth.size, z.shape[1]))
        t_in_h, n_in_h = np.empty(truth.size, bool), np.empty(truth.size, bool)
        th = {}
        for fold in np.unique(f):
            tr, te = f != fold, f == fold
            pr = softmax(z[te] / T_fold[int(fold)])
            probs_before[te], p_before[te] = pr, pr[:, 0]
            T = fit_t_binary(z[tr], truth[tr])
            th[int(fold)] = round(T, 3)
            p_hard[te] = ptree(z[te], T)
            q = cutoffs(ptree(z[tr], T), truth[tr])
            t_in_h[te], n_in_h[te] = binary_sets(p_hard[te], q)
        o = {"px": int(truth.size), "trees_share": round(float(truth.mean()), 3), "hard_T_by_fold": th}
        o["before"] = reliability(p_before, truth)
        o["before"]["sets_90"] = set_stats(*old_sets(probs_before, qc_old), truth, kd)
        o["hard_T"] = reliability(p_hard, truth)
        o["hard_T"]["sets_90"] = set_stats(t_in_h, n_in_h, truth, kd)
        o["per_kind_right"] = {k: {"n": int((kd == k).sum()), "right": round(float(((p_hard >= 0.5) == truth)[kd == k].mean()), 3),
                                   "stated_before": round(float(np.maximum(p_before, 1 - p_before)[kd == k].mean()), 3),
                                   "stated_hard": round(float(np.maximum(p_hard, 1 - p_hard)[kd == k].mean()), 3)}
                               for k in sorted(set(kd)) if (kd == k).sum() >= 200}
        # one fit on every square's hard pixels, for the lidar check and for maps
        T_all = fit_t_binary(z, truth)
        q_all = cutoffs(ptree(z, T_all), truth)
        hard_models[inp] = {"temperature": round(T_all, 4), "cutoffs_90": [round(v, 5) for v in q_all]}
        o["hard_T_all"] = hard_models[inp]
        del z
        o["lidar"] = lidar_check(inp, np.load(HERE / "data" / "heads" / f"{inp}_square.npz"),
                                 C["temperature"][inp], T_all, qc_old, q_all)
        res[inp] = o
        b, h = o["before"], o["hard_T"]
        print(f"{inp}: hard T {min(th.values())}-{max(th.values())}  call right {b['right']}  stated {b['stated']} -> "
              f"{h['stated']}  ECE {b['ece']} -> {h['ece']}", flush=True)
        print(f"   90% sets before {b['sets_90']['coverage']} single {b['sets_90']['single']}  |  hard "
              f"{h['sets_90']['coverage']} single {h['sets_90']['single']} (trees {h['sets_90']['coverage_trees']}, "
              f"not {h['sets_90']['coverage_not']})", flush=True)
        print("   hard bins " + "  ".join(f"{x['confidence']}: {x['stated']}->{x['right']} ({x['share']})" for x in h["bins"]), flush=True)
        print("   per kind " + "  ".join(f"{k} {v['right']} ({v['stated_before']}->{v['stated_hard']})" for k, v in o["per_kind_right"].items()), flush=True)
        for g in ("hard", "agree"):
            if g in o["lidar"]:
                lv = o["lidar"][g]
                print(f"   lidar {g}: px {lv['px']} trees {lv['trees_share']}  before right {lv['before']['right']} stated "
                      f"{lv['before']['stated']} ECE {lv['before']['ece']} sets {lv['before']['sets_90']['coverage']}/"
                      f"{lv['before']['sets_90']['single']}  |  hard T stated {lv['hard_T']['stated']} ECE "
                      f"{lv['hard_T']['ece']} sets {lv['hard_T']['sets_90']['coverage']}/{lv['hard_T']['sets_90']['single']}",
                      flush=True)
        json.dump(res, open(HERE / "results" / "class_hard_calibrate.json", "w"), indent=1)
    C["hard_ground"] = hard_models | {
        "note": "where WorldCover and IO disagree (one says trees): P(trees) = softmax(logits / temperature)[trees]; "
                "the 90 % set holds trees if P >= 1 - cutoffs_90[0], not trees if 1 - P >= 1 - cutoffs_90[1]; fit on "
                "Meta-height-refereed hard pixels of all 92 squares"}
    json.dump(C, open(HERE / "models" / "class_temperature.json", "w"), indent=1)


if __name__ == "__main__":
    main()
