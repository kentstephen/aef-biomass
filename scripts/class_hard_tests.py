# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Classes on hard ground, the directions left by section 18 (class_hard_calibrate.py).

Stages (run in order; each saves what the next needs):
  dump        held-out class logits of the 92 squares' Meta-refereed hard pixels (as class_hard_calibrate.py: 2020
              and 2021, WorldCover and IO disagree, one says trees; Meta mean height >= 5 m trees, < 2 m not), with
              fold, kind and continent; the 18 lidar squares' hard and agreed pixels (all 8 fold heads' logits,
              lidar referee, same rule); and the lidar squares' shrub and grass pixels for (c).
              -> data/heads/class_hard_logits_<input>.npz
  fit         (a) an offset fit, P(trees) = sigmoid(a * L + b), L the head's log odds of trees, against the plain
              hard temperature of section 18; (b) the same per continent (known at map time), each fit on the other
              folds' pixels of that continent (the global fit where a continent has too few). All fit across folds;
              then fit on all 92 squares and tried on the lidar referee.
  shrubgrass  (c) on the lidar squares, pixels where WorldCover or IO says shrub or grass (IO's rangeland is both):
              lidar mean height 0.5 to 5 m = shrub, under 0.3 m = grass (and a p90 rule: p90 1 to 5 m with mean
              under 5 = shrub, p90 under 0.3 m = grass). How often WorldCover and the head are right, and how sure
              the head is (calibrated, mean of the 8 fold heads).
  woodland    (d) open woodland as its own class: agreed trees, shrub or grass pixels (2020, 2021) whose Meta mean
              height (imagery 2017 on) is 2 to 5 m become class 9. A 10-class head (biomass_head.py's head and
              folds otherwise unchanged) per input; per-class recall held out, and the hard trees-vs-not pixels of
              dump scored again with it.
Writes results/class_hard_tests.json, models/class_hard_tests.json (fits for maps), data/heads/<input>_square_ow.npz.
Peak RAM about 4 GB (dump, woodland with all three); guard at --max-gb.

    uv run scripts/class_hard_tests.py --stage dump fit shrubgrass woodland [--inputs olmo all]
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
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402
import biomass_rows as br  # noqa: E402
import class_hard_calibrate as chc  # noqa: E402

YEARS = bc.YEARS
DIMS = chc.DIMS
RES = HERE / "results" / "class_hard_tests.json"
MOD = HERE / "models" / "class_hard_tests.json"
CLASSES = ["trees", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "moss"]


def guard(cap):
    p = psutil.Process()
    peak = [0.0]

    def run():
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > cap:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return peak


def rows_meta():
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    names, kinds, conts = list(R["names"]), list(R["kinds"]), list(R["continents"])
    fold_of = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of[names[s]] = fname
    return names, kinds, conts, fold_of


def log_odds(z):
    """Log odds of trees under the plain softmax: z_trees - logsumexp(z_other)."""
    o = z[:, 1:]
    m = o.max(-1, keepdims=True)
    return z[:, 0] - (m[:, 0] + np.log(np.exp(o - m).sum(-1)))


def lidar_ref(name):
    """Lidar mean height, p90 and the usable mask (epoch nearest 2021, stable under 2 m), or None."""
    Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    chm, p90, cov = Ld["chm"].astype(np.float32), Ld["chm_p90"].astype(np.float32), Ld["cover"]
    src = str(Ld["source"])
    if src == "werk10":
        ypx = Ld["year_px"]
        ok = (cov[0] >= 80) & (ypx >= 2018) & (ypx <= 2023)
        return chm[0], p90[0], ok, src
    e = int(np.argmin(np.abs(Ld["year"].astype(int) - 2021)))
    ok = np.all(cov >= 80, 0)
    if chm.shape[0] > 1:
        ok &= np.abs(chm[1] - chm[0]) < 2
    return chm[e], p90[e], ok, src


def dump(inp, args):
    names, kinds, conts, fold_of = rows_meta()
    rng = np.random.default_rng(0)
    z, truth, f, kd, sq = chc.collect(inp, names, kinds, fold_of, rng, args.per_square)
    out = {"z": z.astype(np.float32), "truth": truth, "fold": f, "kind": kd, "sq": sq,
           "cont": np.array([conts[s] for s in sq])}
    W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
    folds = sorted({k.split(".", 1)[0] for k in W.files if "." in k})
    nets = [bc.head_from(W, fd, DIMS[inp]) for fd in folds]
    picks = json.load(open(HERE / "data" / "lidar_squares.json"))
    acc = {k: [] for k in ("hz", "ht", "hs", "az", "at", "as_", "gz", "gwc", "gio", "gh", "gp", "gs")}
    srng = np.random.default_rng(1)
    for name in picks:
        fl, fs, flab = (HERE / "data" / d / f"{name}.npz" for d in ("lidar", "squares", "labels"))
        if not (fl.exists() and fs.exists() and flab.exists()):
            continue
        h, p90, okl, src = lidar_ref(name)
        zq, L = np.load(fs), np.load(flab)
        io = L["io"][list(L["io_years"]).index(2021)]
        wc = zq["wc2021"]
        hard, agree = chc.hard_mask(wc, io)
        x, ok = chc.feats(zq, inp, YEARS.index(2021))
        ref = okl & ok & ((h >= 5) | (h < 2))
        for key, m in (("h", hard), ("a", agree)):
            sel = np.flatnonzero(m & ref)
            if sel.size > args.per_square:
                sel = np.sort(srng.choice(sel, args.per_square, replace=False))
            if sel.size < 200:
                continue
            acc[f"{key}z"].append(np.stack([chc.logits_of(n, x[sel]) for n in nets]).astype(np.float16))
            acc[f"{key}t"].append(h[sel] >= 5)
            acc["hs" if key == "h" else "as_"].append(np.full(sel.size, name))
        # shrub / grass pixels: WorldCover shrub or grass, or IO rangeland
        sg = okl & ok & (np.isin(wc, [20, 30]) | (io == 11))
        sel = np.flatnonzero(sg)
        if sel.size > args.per_square:
            sel = np.sort(srng.choice(sel, args.per_square, replace=False))
        if sel.size >= 200:
            acc["gz"].append(np.stack([chc.logits_of(n, x[sel]) for n in nets]).astype(np.float16))
            acc["gwc"].append(wc[sel])
            acc["gio"].append(io[sel])
            acc["gh"].append(h[sel])
            acc["gp"].append(p90[sel])
            acc["gs"].append(np.full(sel.size, name))
        del x, zq, L
        print(f"  lidar {name} [{src}]: shrub/grass {sel.size}  [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
    for k in ("hz", "az", "gz"):
        out[f"lidar_{k}"] = np.concatenate(acc[k], 1)
    for k in ("ht", "hs", "at", "as_", "gwc", "gio", "gh", "gp", "gs"):
        out[f"lidar_{k}"] = np.concatenate(acc[k])
    cont_of = {n: br.continent(*picks[n]["center"]) for n in picks}
    out["lidar_hcont"] = np.array([cont_of[n] for n in out["lidar_hs"]])
    np.savez(HERE / "data" / "heads" / f"class_hard_logits_{inp}.npz", **out)
    print(f"{inp}: dumped {truth.size} hard pixels; lidar hard {out['lidar_ht'].size}, agreed {out['lidar_at'].size}, "
          f"shrub/grass {out['lidar_gh'].size}", flush=True)


def fit_platt(L, t, iters=200):
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Lt, tt = torch.as_tensor(L.astype(np.float32)), torch.as_tensor(t.astype(np.float32))
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * Lt + b, tt)
        loss.backward()
        return loss
    opt.step(step)
    a_, b_ = float(a.detach()), float(b.detach())
    if not (np.isfinite(a_) and np.isfinite(b_)):  # diverged: fall back to plain gradient steps from the start
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a * Lt + b, tt)
            loss.backward()
            opt.step()
        a_, b_ = float(a.detach()), float(b.detach())
    return a_, b_


def sig(x):
    return 1 / (1 + np.exp(-x))


def summary(p, truth, kd=None, cont=None, q=None):
    o = chc.reliability(p, truth)
    o["bins"] = [{k: v for k, v in b.items()} for b in o["bins"]]
    if q is not None:
        o["sets_90"] = chc.set_stats(*chc.binary_sets(p, q), truth, kd)
        if cont is not None:
            t_in, n_in = chc.binary_sets(p, q)
            cov = np.where(truth, t_in, n_in)
            o["sets_90"]["per_continent"] = {c: {"n": int((cont == c).sum()), "coverage": round(float(cov[cont == c].mean()), 3),
                                                 "single": round(float((t_in != n_in)[cont == c].mean()), 3)}
                                             for c in sorted(set(cont)) if (cont == c).sum() >= 200}
    return o


def fit_stage(inp, res, models):
    D = np.load(HERE / "data" / "heads" / f"class_hard_logits_{inp}.npz")
    z, truth, f, kd, cont = D["z"], D["truth"], D["fold"], D["kind"], D["cont"]
    L = log_odds(z)
    n = truth.size
    pT, pP, pC = np.empty(n), np.empty(n), np.empty(n)
    sets = {k: [np.empty(n, bool), np.empty(n, bool)] for k in ("T", "P", "C")}
    fits = {"T": {}, "P": {}, "C": {}}
    for fold in np.unique(f):
        tr, te = f != fold, f == fold
        T = chc.fit_t_binary(z[tr], truth[tr])
        pT[te] = chc.ptree(z[te], T)
        qT = chc.cutoffs(chc.ptree(z[tr], T), truth[tr])
        a, b = fit_platt(L[tr], truth[tr])
        pP[te] = sig(a * L[te] + b)
        qP = chc.cutoffs(sig(a * L[tr] + b), truth[tr])
        fits["T"][int(fold)], fits["P"][int(fold)] = round(T, 3), [round(a, 3), round(b, 3)]
        # per continent: fit on the other folds' pixels of the continent, the global fit when too few
        for c in np.unique(cont[te]):
            trc, tec = tr & (cont == c), te & (cont == c)
            if trc.sum() >= 2000 and truth[trc].sum() >= 200 and (~truth[trc]).sum() >= 200:
                ac, bc_ = fit_platt(L[trc], truth[trc])
                qc = chc.cutoffs(sig(ac * L[trc] + bc_), truth[trc])
            else:
                ac, bc_, qc = a, b, qP
            pC[tec] = sig(ac * L[tec] + bc_)
            ti, ni = chc.binary_sets(pC[tec], qc)
            sets["C"][0][tec], sets["C"][1][tec] = ti, ni
        for k, p, q in (("T", pT, qT), ("P", pP, qP)):
            ti, ni = chc.binary_sets(p[te], q)
            sets[k][0][te], sets[k][1][te] = ti, ni
    o = {"px": int(n), "fits_by_fold": fits}
    for k, p, label in (("T", pT, "hard_temperature"), ("P", pP, "offset"), ("C", pC, "offset_per_continent")):
        s = chc.reliability(p, truth)
        s["sets_90"] = chc.set_stats(sets[k][0], sets[k][1], truth, kd)
        cov = np.where(truth, sets[k][0], sets[k][1])
        s["sets_90"]["per_continent"] = {c: {"n": int((cont == c).sum()), "coverage": round(float(cov[cont == c].mean()), 3),
                                             "single": round(float((sets[k][0] != sets[k][1])[cont == c].mean()), 3)}
                                         for c in sorted(set(cont))}
        o[label] = s
    # fits on all 92 squares, tried on the lidar referee
    T_all = chc.fit_t_binary(z, truth)
    a_all, b_all = fit_platt(L, truth)
    q_T = chc.cutoffs(chc.ptree(z, T_all), truth)
    q_P = chc.cutoffs(sig(a_all * L + b_all), truth)
    per_c = {}
    for c in sorted(set(cont)):
        m = cont == c
        if m.sum() >= 2000 and truth[m].sum() >= 200 and (~truth[m]).sum() >= 200:
            ac, bc_ = fit_platt(L[m], truth[m])
            per_c[c] = {"a": round(ac, 4), "b": round(bc_, 4), "cutoffs_90": [round(v, 5) for v in chc.cutoffs(sig(ac * L[m] + bc_), truth[m])]}
    models[inp] = {"offset": {"a": round(a_all, 4), "b": round(b_all, 4), "cutoffs_90": [round(v, 5) for v in q_P]},
                   "offset_per_continent": per_c, "hard_temperature": round(T_all, 4)}
    lid = {}
    for g, zk, tk in (("hard", "lidar_hz", "lidar_ht"), ("agree", "lidar_az", "lidar_at")):
        zz, tt = D[zk].astype(np.float32), D[tk]
        Lf = np.stack([log_odds(zf) for zf in zz])
        pT_l = np.mean([chc.ptree(zf, T_all) for zf in zz], 0)
        pP_l = np.mean(sig(a_all * Lf + b_all), 0)
        lid[g] = {"px": int(tt.size), "hard_temperature": summary(pT_l, tt, q=q_T), "offset": summary(pP_l, tt, q=q_P)}
        if g == "hard":
            hc = D["lidar_hcont"]
            pC_l = np.empty(tt.size)
            covC = np.empty(tt.size, bool)
            singC = np.empty(tt.size, bool)
            for c in np.unique(hc):
                m = hc == c
                if c in per_c:
                    ac, bc_, qc = per_c[c]["a"], per_c[c]["b"], per_c[c]["cutoffs_90"]
                else:
                    ac, bc_, qc = a_all, b_all, q_P
                pC_l[m] = np.mean(sig(ac * Lf[:, m] + bc_), 0)
                ti, ni = chc.binary_sets(pC_l[m], qc)
                covC[m] = np.where(tt[m], ti, ni)
                singC[m] = ti != ni
            s = chc.reliability(pC_l, tt)
            s["sets_90"] = {"coverage": round(float(covC.mean()), 3), "single": round(float(singC.mean()), 3),
                            "per_continent": {c: {"n": int((hc == c).sum()), "coverage": round(float(covC[hc == c].mean()), 3)}
                                              for c in np.unique(hc)}}
            lid[g]["offset_per_continent"] = s
    o["lidar"] = lid
    res.setdefault(inp, {})["fit"] = o
    print(f"{inp} fit (held out by fold, Meta referee):", flush=True)
    for label in ("hard_temperature", "offset", "offset_per_continent"):
        s = o[label]
        print(f"   {label:22s} stated {s['stated']} right {s['right']} ECE {s['ece']}  sets {s['sets_90']['coverage']}/"
              f"{s['sets_90']['single']}  bins " + " ".join(f"{b['stated']}->{b['right']}" for b in s["bins"]), flush=True)
        print("      per kind " + " ".join(f"{k} {v['coverage']}" for k, v in s["sets_90"]["per_kind"].items()), flush=True)
    for g, v in lid.items():
        for label, s in v.items():
            if label == "px":
                continue
            print(f"   lidar {g:5s} {label:22s} stated {s['stated']} right {s['right']} ECE {s['ece']}  sets "
                  f"{s['sets_90']['coverage']}/{s['sets_90']['single']}", flush=True)


def shrubgrass(inp, res):
    D = np.load(HERE / "data" / "heads" / f"class_hard_logits_{inp}.npz")
    C = json.load(open(HERE / "models" / "class_temperature.json"))
    T = C["temperature"][inp]
    zz = D["lidar_gz"].astype(np.float32)
    P = np.mean([chc.softmax(zf / T) for zf in zz], 0)
    wc, io, h, p90, sq = D["lidar_gwc"], D["lidar_gio"], D["lidar_gh"], D["lidar_gp"], D["lidar_gs"]
    out = {}
    for rule, shrub, grass in (("mean", (h >= 0.5) & (h <= 5), h < 0.3),
                               ("p90", (p90 >= 1) & (p90 <= 5) & (h < 5), p90 < 0.3)):
        m = shrub | grass
        truth = shrub[m]
        ps, pg = P[m, 1], P[m, 2]
        head_shrub = ps > pg
        conf = np.maximum(ps, pg) / np.maximum(ps + pg, 1e-9)
        wcm = wc[m]
        top = P[m].argmax(-1)
        o = {"px": int(m.sum()), "shrub_share": round(float(truth.mean()), 3),
             "worldcover": {"says_shrub_or_grass": round(float(np.isin(wcm, [20, 30]).mean()), 3),
                            "right_where_it_says_shrub_or_grass": round(float(((wcm == 20) == truth)[np.isin(wcm, [20, 30])].mean()), 3),
                            "says_other": {CLASSES[br.WC2C[c]]: round(float((wcm == c).mean()), 3) for c in br.WC2C if (wcm == c).mean() >= 0.01 and c not in (20, 30)}},
             "io_rangeland_share": round(float((io[m] == 11).mean()), 3),
             "head_two_way": {"right": round(float((head_shrub == truth).mean()), 3),
                              "right_on_shrub": round(float(head_shrub[truth].mean()), 3) if truth.any() else None,
                              "right_on_grass": round(float((~head_shrub)[~truth].mean()), 3) if (~truth).any() else None,
                              "stated": round(float(conf.mean()), 3),
                              "stated_when_right": round(float(conf[head_shrub == truth].mean()), 3),
                              "stated_when_wrong": round(float(conf[head_shrub != truth].mean()), 3)},
             "head_top_class": {CLASSES[c]: round(float((top == c).mean()), 3) for c in range(9) if (top == c).mean() >= 0.01}}
        both = np.isin(wcm, [20, 30])
        o["on_worldcover_shrub_or_grass"] = {"px": int(both.sum()), "worldcover_right": round(float(((wcm == 20) == truth)[both].mean()), 3),
                                             "head_right": round(float((head_shrub == truth)[both].mean()), 3)}
        per = {}
        for s in np.unique(sq[m]):
            mm = sq[m] == s
            if mm.sum() >= 300:
                b2 = mm & both
                per[str(s)] = {"px": int(mm.sum()), "shrub_share": round(float(truth[mm].mean()), 3),
                               "head_right": round(float((head_shrub == truth)[mm].mean()), 3),
                               "worldcover_right": round(float(((wcm == 20) == truth)[b2].mean()), 3) if b2.sum() >= 100 else None}
        o["per_square"] = per
        out[rule] = o
        print(f"{inp} shrub/grass [{rule}] px {o['px']} shrub {o['shrub_share']}  WC right {o['on_worldcover_shrub_or_grass']['worldcover_right']} "
              f"head right {o['on_worldcover_shrub_or_grass']['head_right']} (all px {o['head_two_way']['right']}; shrub "
              f"{o['head_two_way']['right_on_shrub']} grass {o['head_two_way']['right_on_grass']}) stated {o['head_two_way']['stated']} "
              f"(right {o['head_two_way']['stated_when_right']} / wrong {o['head_two_way']['stated_when_wrong']})  top {o['head_top_class']}", flush=True)
        for s, v in per.items():
            print(f"      {s:24s} {v}", flush=True)
    res.setdefault(inp, {})["shrub_grass_lidar"] = out


def woodland(inp, res, args):
    R = dict(np.load(HERE / "data" / "rows.npz"))
    h, hy = R["height"], R["height_year"]
    ow = (h != 255) & (hy >= 2017) & (h >= 2) & (h <= 5)  # (G, P)
    cls = R["cls"].copy()
    owY = np.broadcast_to(ow[:, None, :], cls.shape)
    cls9 = np.where(np.isin(cls, [0, 1, 2]) & owY, 9, cls).astype(np.int8)
    agreed = cls >= 0
    moved = {CLASSES[c]: round(float(((cls == c) & owY)[agreed].sum() / max((cls == c).sum(), 1)), 3) for c in (0, 1, 2)}
    share = round(float((cls9 == 9).sum() / agreed.sum()), 3)
    print(f"{inp} woodland: open woodland {share} of agreed pixels; share of each class moved {moved}", flush=True)
    R10 = dict(R)
    R10["cls"] = cls9
    old_ncls = bh.NCLS
    bh.NCLS = 10
    try:
        X, ok = bh.build(R10, inp)
        H = bh.targets(R10)
        common = np.ones(cls.shape, bool)
        for k in ("aef", "tes", "olmo"):
            common &= (np.isfinite(R["tes_s"]) & (R["tes_s"] > 0)) if k == "tes" else (R[k][..., 0] != -128)
        G = X.shape[0]
        pc = np.zeros(cls.shape, np.int8)
        weights = {}
        t0 = time.time()
        for name, test in bh.folds(R10, "square"):
            train = np.setdiff1d(np.arange(G), test)
            net = bh.fit(X, ok, R10, H, train, args.epochs, seed=0)
            _, _, c = bh.predict(net, X, ok, test)
            pc[test] = c
            for k, v in net.state_dict().items():
                weights[f"{name}.{k}"] = v.numpy()
            print(f"  {inp} woodland fold {name} [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        del X, ok
        np.savez(HERE / "data" / "heads" / f"{inp}_square_ow.npz", pc=pc, **weights)
        old = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")["pc"]
        cm = (cls9 >= 0) & common
        names10 = CLASSES + ["open woodland"]
        rec10 = {names10[c]: round(float((pc[cm] == c)[cls9[cm] == c].mean()), 3) for c in range(10) if (cls9[cm] == c).sum() >= 200}
        cm9 = agreed & common
        rec9 = {CLASSES[c]: round(float((old[cm9] == c)[cls[cm9] == c].mean()), 3) for c in range(9) if (cls[cm9] == c).sum() >= 200}
        # where the 9-class head puts the pixels that became open woodland
        owm = cm & (cls9 == 9)
        old_on_ow = {CLASSES[c]: round(float((old[owm] == c).mean()), 3) for c in range(9) if (old[owm] == c).mean() >= 0.02}
        o = {"rule": "agreed trees, shrub or grass with Meta mean height 2 to 5 m (imagery 2017 on) -> open woodland",
             "share_of_agreed": share, "moved_from": moved,
             "recall_10_class": rec10, "recall_9_class": rec9,
             "acc_10": round(float((pc[cm] == cls9[cm]).mean()), 3), "acc_9": round(float((old[cm9] == cls[cm9]).mean()), 3),
             "nine_class_head_on_open_woodland": old_on_ow}
        print(f"   recall 10-class {rec10}\n   recall 9-class  {rec9}\n   9-class head on open woodland px {old_on_ow}", flush=True)
        # hard pixels again with the 10-class head
        names, kinds, conts, fold_of = rows_meta()
        W10 = np.load(HERE / "data" / "heads" / f"{inp}_square_ow.npz")
        D = np.load(HERE / "data" / "heads" / f"class_hard_logits_{inp}.npz")
        rng = np.random.default_rng(0)
        orig_head_path = HERE / "data" / "heads" / f"{inp}_square.npz"
        z10 = collect_with(W10, inp, names, kinds, fold_of, rng, args.per_square)
        assert z10.shape[0] == D["truth"].size, (z10.shape, D["truth"].size)
        truth = D["truth"]
        for label, zz in (("nine_class", D["z"]), ("ten_class", z10)):
            p = chc.softmax(zz)
            top = p.argmax(-1)
            conf = p.max(-1)
            called_ow = top == 9 if zz.shape[1] == 10 else np.zeros(top.size, bool)
            call_trees = top == 0
            right = call_trees == truth
            right_ow_not = np.where(called_ow, ~truth, right)
            o[f"hard_{label}"] = {"called_open_woodland": round(float(called_ow.mean()), 3),
                                  "right_ow_as_not_trees": round(float(right_ow_not.mean()), 3),
                                  "right_excluding_ow_calls": round(float(right[~called_ow].mean()), 3) if (~called_ow).any() else None,
                                  "confidence": round(float(conf.mean()), 3),
                                  "ow_calls_truth_trees_share": round(float(truth[called_ow].mean()), 3) if called_ow.any() else None,
                                  "binary_trees_vs_rest": chc.reliability(p[:, 0] / np.maximum(p.sum(-1), 1e-9), truth)["right"]}
            print(f"   hard {label}: {o[f'hard_{label}']}", flush=True)
        res.setdefault(inp, {})["open_woodland"] = o
    finally:
        bh.NCLS = old_ncls


def collect_with(W, inp, names, kinds, fold_of, rng, per_sq):
    """chc.collect's pixels with another set of fold heads (same rng draws, so the same pixels)."""
    Z = []
    for si, name in enumerate(names):
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        L = np.load(HERE / "data" / "labels" / f"{name}.npz")
        net = bc.head_from(W, fold_of[name], DIMS[inp])
        io_years = list(L["io_years"])
        hgt, hy = L["height"], L["height_date"] // 10000
        for wy in (2020, 2021):
            yi = YEARS.index(wy)
            hard, _ = chc.hard_mask(z[f"wc{wy}"], L["io"][io_years.index(wy)])
            hok = (hgt != 255) & (hy >= 2017)
            tall, low = hok & (hgt >= 5), hok & (hgt < 2)
            x, ok = chc.feats(z, inp, yi)
            sel = np.flatnonzero(hard & (tall | low) & ok)
            if sel.size > per_sq // 2:
                sel = np.sort(rng.choice(sel, per_sq // 2, replace=False))
            Z.append(chc.logits_of(net, x[sel]))
            del x
        del z, L
    return np.concatenate([a for a in Z if len(a)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["dump", "fit", "shrubgrass", "woodland"])
    ap.add_argument("--inputs", nargs="*", default=["olmo", "all"])
    ap.add_argument("--per-square", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=6)
    args = ap.parse_args()
    torch.set_num_threads(8)
    peak = guard(args.max_gb)
    res = json.load(open(RES)) if RES.exists() else {}
    models = json.load(open(MOD)) if MOD.exists() else {}
    for stage in args.stage:
        for inp in args.inputs:
            t0 = time.time()
            if stage == "dump":
                dump(inp, args)
            elif stage == "fit":
                fit_stage(inp, res, models)
            elif stage == "shrubgrass":
                shrubgrass(inp, res)
            elif stage == "woodland":
                woodland(inp, res, args)
            print(f"[{stage} {inp}: {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
            res["peak_gb"] = max(res.get("peak_gb", 0), round(peak[0], 1))
            json.dump(res, open(RES, "w"), indent=1)
            if models:
                models["note"] = ("hard pixels (WorldCover and IO disagree, one says trees): L = trees logit minus logsumexp of "
                                  "the other class logits; P(trees) = sigmoid(a * L + b); 90 % set holds trees if P >= 1 - "
                                  "cutoffs_90[0], not trees if 1 - P >= 1 - cutoffs_90[1]; per continent where fit")
                json.dump(models, open(MOD, "w"), indent=1)


if __name__ == "__main__":
    main()
