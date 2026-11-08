# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Change for the map: loss from one head, gain from another, every pairing of embeddings, and a gate for slow change.

Section 12: the one-year head, gated by the frozen clearcut cutoffs and divided by the muting on clear loss cells,
gives loss at full size with steady ground quiet. Section 14: the trajectory head shows regrowth four times better,
but the clearcut gate removes it. Here loss and gain are taken from separate sources and every pairing is scored:

  loss source X   the gated one-year head of X, X in aef, tes, olmo, all (the gate of X: its own frozen cutoff,
                  or 2 of 3 for all); its negative part, divided by its loss factor (the muting on clear loss cells)
  gain source Y   the trajectory head of Y where one exists (olmo, all; none was trained for aef or tes), or the
                  one-year head of Y (aef, tes, olmo, all); its positive part, divided by its gain factor (the muting
                  on clear gain cells, CTrees rose by half and 20 Mg/ha), ungated or behind a gradual gate
  combined        per pixel: negative part of X's gated change + positive part of Y's change
  gradual gate    the embedding's move as a fraction of the clearcut cutoff (1 - cosine over <emb>_cut; for all, the
                  median of the three, so 1 is the frozen 2-of-3 gate), from the first year to the last
                  ("distance"), or summed over the consecutive years between ("chain", 2018 to 2025); the cutoff q,
                  a fraction of the clearcut cutoff from a grid, is picked on the other folds' steady cells as the most
                  open one that keeps steady ground no noisier than X's gated loss alone (+ .01); the gate of the
                  gain part uses Y's embeddings, so a pairing reads only X's and Y's
  bandwidth       each pairing's embeddings to read: the union of X's and Y's (all = aef, tes, olmo)
Scored on CTrees' 100 m cells (biomass_change.stats: slope, correlation, loss / gain size and found, steady lit),
one-year windows and 2018 to 2025, every factor and cutoff fit on the other folds; then against airborne lidar
canopy change on the two-flight lidar squares (fits from all 92 squares; the mean of the 8 fold heads); then the
spread of the combined change per pixel by WorldCover class, for the change map's ramp limits.

Stage "cells" (one pass over the 92 squares, each by its held-out fold heads) writes
data/heads/change_combined_cells.npz and a pixel sample data/heads/change_combined_px.npz; stage "score" writes
results/change_combined.json. Peak RAM about 8 GB in "cells" (one square's features), about 6 GB in "score".

    uv run scripts/change_combined.py --stage cells
    uv run scripts/change_combined.py --stage score
    uv run scripts/change_combined.py --stage floor   (gain behind a magnitude floor on the 100 m cell; after score)
    uv run scripts/change_combined.py --stage ramp2 --ramp-pair aef+oney_aef   (ramp limits for the floor rule)
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
import lidar_check as lc  # noqa: E402
import trajectory_head as th  # noqa: E402

YEARS = bc.YEARS
EMB = bc.EMB
INPUTS = ("aef", "tes", "olmo", "all")
TRAJ = ("olmo", "all")
DIMS = {"aef": 64, "tes": 128, "olmo": 128, "all": 320}
WIN = bc.WINDOWS  # 7 one-year windows, then 2018 to 2025
IA = [YEARS.index(a) for a, _ in WIN]
IB = [YEARS.index(b) for _, b in WIN]
Q = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]  # gradual gate cutoffs, fractions of the clearcut cutoff
WC = {10: "trees", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland",
      95: "mangrove", 100: "moss"}
CELLS = HERE / "data" / "heads" / "change_combined_cells.npz"
PX = HERE / "data" / "heads" / "change_combined_px.npz"
RES = HERE / "results" / "change_combined.json"
# gain sources: (name, head kind, input)
GAINS = [(f"traj_{y}", "traj", y) for y in TRAJ] + [(f"oney_{y}", "oney", y) for y in INPUTS]


def reads(*inp):
    out = set()
    for i in inp:
        out |= set(EMB) if i == "all" else {i}
    return [e for e in EMB if e in out]


def guard(max_gb, peak):
    p = psutil.Process()

    def run():
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def traj_net(W, fold, d):
    net = th.Traj(d, np.zeros(d, np.float32), np.ones(d, np.float32))
    net.load_state_dict({k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f"{fold}.")})
    return net.eval()


def x_of(F, inp, y, sl=slice(None)):
    parts = EMB if inp == "all" else (inp,)
    return np.concatenate([F[k][0][y][sl] for k in parts], -1).astype(np.float32)


def ok_of(F, inp):
    return F["aef"][1] & F["tes"][1] & F["olmo"][1] if inp == "all" else F[inp][1]


def moves(F, pairs):
    """Move per pixel per embedding for each (a, b) year pair: (1 - cosine) / cutoff, 0 where either year is
    missing; plus all = the median of the three."""
    D = np.load(HERE / "models" / "detectors.npz")
    out = {}
    for k in EMB:
        E, okk = F[k]
        cut = float(D[f"{k}_cut"])
        s = np.zeros((len(pairs), E.shape[1]), np.float32)
        for i, (a, b) in enumerate(pairs):
            ua, ub = bc.unit(E[a].astype(np.float32)), bc.unit(E[b].astype(np.float32))
            s[i] = np.where(okk[a] & okk[b], (1 - (ua * ub).sum(-1)) / cut, 0)
        out[k] = s
    out["all"] = np.median(np.stack([out[k] for k in EMB]), 0)
    return out


@torch.no_grad()
def predict(F, snets, tnets):
    """snets {inp: [nets]}, tnets {inp: [nets]} -> biomass (Y, n) per head (mean over the nets given)."""
    n = F["aef"][0].shape[1]
    pa = {}
    for inp, nets in snets.items():
        p = np.zeros((len(YEARS), n), np.float32)
        for y in range(len(YEARS)):
            x = torch.as_tensor(x_of(F, inp, y))
            for net in nets:
                p[y] += np.clip(np.expm1(np.minimum(net(x).numpy()[:, 0], 8)), 0, None) / len(nets)
        pa[inp] = p
    pt = {}
    for inp, nets in tnets.items():
        ok = ok_of(F, inp)
        p = np.zeros((len(YEARS), n), np.float32)
        for b in range(0, n, 40000):
            sl = slice(b, b + 40000)
            x = np.stack([x_of(F, inp, y, sl) for y in range(len(YEARS))], 1)
            xt, vt = torch.as_tensor(x), torch.as_tensor(ok[:, sl].T.copy())
            for net in nets:
                p[:, sl] += np.clip(np.expm1(np.minimum(net(xt, vt).numpy().T, 8)), 0, None) / len(nets)
        pt[inp] = p
    return pa, pt


def stage_cells(args):
    peak = [0.0]
    guard(args.max_gb, peak)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    names, kinds = list(R["names"]), list(R["kinds"])
    fold_of = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of[names[s]] = fname
    WS = {i: np.load(HERE / "data" / "heads" / f"{i}_square.npz") for i in INPUTS}
    WT = {i: np.load(HERE / "data" / "heads" / f"traj_{i}.npz") for i in TRAJ}
    rng = np.random.default_rng(0)
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(v)
    truth, cell_sq = [], []
    smp = {}
    ncell = 0
    for si, name in enumerate(names):
        if args.limit and si >= args.limit:
            break
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        L = np.load(HERE / "data" / "labels" / f"{name}.npz")
        F = bc.square_feats(z)
        cell = L["cell"]
        ids, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
        keep = cnt >= 40
        agb = L["agb"].astype(np.float32)
        first = np.zeros(ids.size, int)
        first[inv[::-1]] = np.arange(cell.size)[::-1]
        truth.append(np.where(agb[:, first] >= 0, agb[:, first] / 10, np.nan).T[keep])
        cell_sq.append(np.full(int(keep.sum()), si))
        gid = np.full(ids.size, -1)
        gid[keep] = ncell + np.arange(int(keep.sum()))
        ncell += int(keep.sum())

        def cellmean(v, valid):  # (K, n) -> (C, K), as biomass_change.py
            outm = np.full((ids.size, v.shape[0]), np.nan, np.float32)
            for k in range(v.shape[0]):
                s = np.bincount(inv, np.where(valid[k], v[k], 0), ids.size)
                c = np.bincount(inv, valid[k], ids.size)
                outm[:, k] = np.where(c >= 20, s / np.maximum(c, 1), np.nan)
            return outm[keep]

        fold = fold_of[name]
        pa, pt = predict(F, {i: [bc.head_from(WS[i], fold, DIMS[i])] for i in INPUTS},
                         {i: [traj_net(WT[i], fold, DIMS[i])] for i in TRAJ})
        mv = moves(F, list(zip(IA, IB)))  # {emb: (W, n)}
        chain7 = {k: v[:7].sum(0) for k, v in mv.items()}
        okv = {i: ok_of(F, i) for i in INPUTS}
        del F
        for i in INPUTS:
            okw = np.stack([okv[i][a] & okv[i][b] for a, b in zip(IA, IB)])
            ds = np.stack([pa[i][b] - pa[i][a] for a, b in zip(IA, IB)])
            g1 = mv[i] >= 1
            add(f"s_plain_{i}", cellmean(ds, okw))
            add(f"s_gate_{i}", cellmean(np.where(g1, ds, 0), okw))
            add(f"s_neg_{i}", cellmean(np.where(g1, np.minimum(ds, 0), 0), okw))
            pos = np.maximum(ds, 0)
            add(f"oney_pos_{i}", cellmean(pos, okw))
            add(f"oney_posq_{i}", np.stack([cellmean(np.where(mv[i] >= q, pos, 0), okw) for q in Q], -1))
            add(f"oney_posc_{i}", np.stack([cellmean(np.where((chain7[i] >= q)[None], pos[7:], 0), okw[7:])[:, 0] for q in Q], -1))
        for i in TRAJ:
            okw = np.stack([okv[i][a] & okv[i][b] for a, b in zip(IA, IB)])
            dt = np.stack([pt[i][b] - pt[i][a] for a, b in zip(IA, IB)])
            pos = np.maximum(dt, 0)
            add(f"t_plain_{i}", cellmean(dt, okw))
            add(f"traj_pos_{i}", cellmean(pos, okw))
            add(f"traj_posq_{i}", np.stack([cellmean(np.where(mv[i] >= q, pos, 0), okw) for q in Q], -1))
            add(f"traj_posc_{i}", np.stack([cellmean(np.where((chain7[i] >= q)[None], pos[7:], 0), okw[7:])[:, 0] for q in Q], -1))
        # pixel sample for the ramp limits: per window 2500 pixels valid in all three, in kept cells
        wc = z["wc2021"]
        incell = gid[inv] >= 0
        oka = okv["all"]
        for w in range(len(WIN)):
            a, b = IA[w], IB[w]
            cand = np.flatnonzero(oka[a] & oka[b] & incell)
            pick = rng.choice(cand, min(2500, cand.size), replace=False)
            row = {"wc": wc[pick], "sq": np.full(pick.size, si, np.int16), "cell": gid[inv[pick]],
                   "win": np.full(pick.size, w, np.int8)}
            for i in INPUTS:
                row[f"ds_{i}"] = (pa[i][b] - pa[i][a])[pick]
                row[f"a_{i}"] = pa[i][a][pick]
                row[f"mv_{i}"] = mv[i][w][pick]
                row[f"ch_{i}"] = chain7[i][pick] if w == 7 else mv[i][w][pick]
            for i in TRAJ:
                row[f"dt_{i}"] = (pt[i][b] - pt[i][a])[pick]
            for k, v in row.items():
                smp.setdefault(k, []).append(v)
        print(f"{si + 1}/{len(names)} {name} [{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
        del z, L, pa, pt, mv
    out = {k: np.concatenate(v) for k, v in acc.items()}
    T = np.concatenate(truth)
    n = len(T)
    for i in INPUTS:
        Z = np.load(HERE / "data" / "heads" / f"change_cells_{i}.npz")
        m = np.isfinite(Z["plain"][:n]) & np.isfinite(out[f"s_plain_{i}"])
        print(f"check {i}: truth equal {np.allclose(np.nan_to_num(Z['truth'][:n], nan=-1), np.nan_to_num(T, nan=-1))}, "
              f"plain corr {np.corrcoef(Z['plain'][:n][m], out[f's_plain_{i}'][m])[0, 1]:.4f}, gated corr "
              f"{np.corrcoef(np.nan_to_num(Z['gated'][:n][m]), np.nan_to_num(out[f's_gate_{i}'][m]))[0, 1]:.4f}", flush=True)
    for i in TRAJ:
        Zt = np.load(HERE / "data" / "heads" / f"change_cells_traj_{i}.npz")["state"][:n]
        d = np.stack([Zt[:, b] - Zt[:, a] for a, b in zip(IA, IB)], 1)
        m = np.isfinite(d) & np.isfinite(out[f"t_plain_{i}"])
        print(f"check traj {i}: corr {np.corrcoef(d[m], out[f't_plain_{i}'][m])[0, 1]:.4f}", flush=True)
    np.savez(CELLS, truth=T, sq=np.concatenate(cell_sq), kind=np.array(kinds)[np.concatenate(cell_sq)], q=np.array(Q),
             names=np.array(names), **out)
    np.savez(PX, **{k: np.concatenate(v) for k, v in smp.items()})
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


# ---------------------------------------------------------------- scoring

def ratio(p, t, base, kind="loss"):
    """Median predicted over median true change on clear loss (gain) cells: CTrees fell (rose) by half and 20 Mg/ha."""
    m = np.isfinite(p) & np.isfinite(t) & np.isfinite(base)
    p, t, base = p[m], t[m], base[m]
    rel = t / np.maximum(base, 1)
    sel = (rel <= -0.5) & (t <= -20) if kind == "loss" else (rel >= 0.5) & (t >= 20)
    return float(np.median(p[sel]) / np.median(t[sel])) if sel.sum() >= 20 else np.nan


def lit(p, t, base):
    """Share of steady cells lit (biomass_change.stats' rule)."""
    m = np.isfinite(p) & np.isfinite(t) & np.isfinite(base)
    p, t, base = p[m], t[m], base[m]
    rel, prel = t / np.maximum(base, 1), p / np.maximum(base, 1)
    st = (np.abs(rel) < 0.1) & (np.abs(t) < 10)
    return float(((np.abs(prel[st]) >= 0.25) & (np.abs(p[st]) >= 10)).mean())


def gain_src(Z, g, w):
    """Positive part of gain source g for window set w: ungated (C, K), distance-gated (C, K, Q), chain (C, 1, Q)."""
    _, kind, i = g
    pre = "traj" if kind == "traj" else "oney"
    sl = slice(0, 7) if w == "L1" else slice(7, 8)
    return Z[f"{pre}_pos_{i}"][:, sl], Z[f"{pre}_posq_{i}"][:, sl], Z[f"{pre}_posc_{i}"][:, None, :]


def stage_score(args):
    peak = [0.0]
    guard(args.max_gb, peak)
    Z = dict(np.load(CELLS))
    T, kind, sq = Z["truth"], Z["kind"], Z["sq"]
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    fold_sq = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_sq[int(s)] = int(fname)
    fold = np.array([fold_sq[int(s)] for s in sq])
    folds = np.unique(fold)
    ta, tb = T[:, IA], T[:, IB]
    TD = tb - ta
    groups = {"all": np.ones(len(kind), bool), "forest-loss": kind == "forest-loss", "other": kind != "forest-loss"}
    res = {"cells": int(len(T)), "q_grid": Q, "gain_sources": [g[0] for g in GAINS],
           "note": "trajectory heads exist for olmo and all only; oney = the one-year head's rise", "factors_all_folds": {}}

    def stats_of(P, sl):
        o = {}
        for gname, gm in groups.items():
            t, p, b = TD[gm][:, sl].ravel(), P[gm].ravel(), ta[gm][:, sl].ravel()
            m = np.isfinite(t) & np.isfinite(p) & np.isfinite(b)
            o[gname] = bc.stats({"d": t[m], "a": b[m]}, p[m])
        return o

    table = []
    for w, sl in (("L1", slice(0, 7)), ("L7", slice(7, 8))):
        t_w, b_w = TD[:, sl], ta[:, sl]
        fac = {}
        # factors per fold (fit on the other folds) and on all folds
        trs = {f: fold != f for f in folds} | {"all": np.ones(len(T), bool)}

        def fit(arr, kind_):
            return {f: ratio(arr[tr].ravel(), t_w[tr].ravel(), b_w[tr].ravel(), kind_) for f, tr in trs.items()}
        neg = {x: Z[f"s_neg_{x}"][:, sl] for x in INPUTS}
        gate = {x: Z[f"s_gate_{x}"][:, sl] for x in INPUTS}
        fneg = {x: fit(neg[x], "loss") for x in INPUTS}
        fgate = {x: fit(gate[x], "loss") for x in INPUTS}

        def assemble(fn):
            out = np.full(t_w.shape, np.nan, np.float32)
            for f in folds:
                te = fold == f
                out[te] = fn(f, te)
            return out
        ref_lit = {x: {f: lit((gate[x][fold != f] / fgate[x][f]).ravel(), t_w[fold != f].ravel(), b_w[fold != f].ravel())
                       for f in folds} for x in INPUTS}
        gsrc = {g[0]: gain_src(Z, g, w) for g in GAINS}
        fpos = {g: fit(v[0], "gain") for g, v in gsrc.items()}
        fac_out = {"loss": {x: fneg[x]["all"] for x in INPUTS}, "gated_loss": {x: fgate[x]["all"] for x in INPUTS},
                   "gain": {g: fpos[g]["all"] for g in gsrc}, "gradual": {}}
        res[w] = {"loss_alone": {}, "pairs": {}}
        for x in INPUTS:
            P = assemble(lambda f, te: gate[x][te] / fgate[x][f])
            res[w]["loss_alone"][x] = stats_of(P, sl)
        for g, (pos, posq, posc) in gsrc.items():
            gi = g.split("_", 1)[1]
            # gain alone (ungated), as a reference
            res[w].setdefault("gain_alone", {})[g] = stats_of(assemble(lambda f, te: pos[te] / fpos[g][f]), sl)
            for x in INPUTS:
                key = f"loss {x} + gain {g}"
                o = {"reads": reads(x, gi)}
                Pu = assemble(lambda f, te: neg[x][te] / fneg[x][f] + pos[te] / fpos[g][f])
                o["ungated"] = stats_of(Pu, sl)
                for gname, src in (("distance", posq), ("chain", posc if w == "L7" else None)):
                    if src is None:
                        continue
                    picks = {}

                    def fn(f, te, src=src, picks=picks):
                        tr = fold != f
                        best, quiet = None, (np.inf, None)
                        for qi in range(len(Q)):
                            gq = ratio(src[tr][..., qi].ravel(), t_w[tr].ravel(), b_w[tr].ravel(), "gain")
                            if not np.isfinite(gq) or gq <= 0.02:  # the gate blocks nearly all gain
                                continue
                            pr = neg[x][tr] / fneg[x][f] + src[tr][..., qi] / gq
                            lt = lit(pr.ravel(), t_w[tr].ravel(), b_w[tr].ravel())
                            quiet = min(quiet, (lt, qi))
                            if lt <= ref_lit[x][f] + 0.01:
                                best = qi
                                break
                        best = quiet[1] if best is None else best
                        if best is None:
                            picks[int(f)] = None
                            return np.full(src[te][..., 0].shape, np.nan, np.float32)
                        gq = ratio(src[tr][..., best].ravel(), t_w[tr].ravel(), b_w[tr].ravel(), "gain")
                        picks[int(f)] = Q[best]
                        return neg[x][te] / fneg[x][f] + src[te][..., best] / gq
                    Pg = assemble(fn)
                    o[f"gradual_{gname}"] = stats_of(Pg, sl)
                    o[f"gradual_{gname}_q_by_fold"] = picks
                    qs = [q for q in picks.values() if q is not None]
                    if not qs:
                        continue
                    qm = max(set(qs), key=qs.count)
                    qi = Q.index(qm)
                    fac_out["gradual"][f"{key}/{gname}"] = {
                        "q": qm, "gain": ratio(src[..., qi].ravel(), t_w.ravel(), b_w.ravel(), "gain")}
                res[w]["pairs"][key] = o
                for var in ("ungated", "gradual_distance", "gradual_chain"):
                    if var in o:
                        a, fl, ot = o[var]["all"], o[var]["forest-loss"], o[var]["other"]
                        table.append({"window": w, "loss": x, "gain": g, "variant": var, "reads": "+".join(o["reads"]),
                                      "corr": a.get("corr"), "slope": a.get("slope"),
                                      "loss_size": fl.get("loss_ratio"), "loss_found": fl.get("loss_found"),
                                      "gain_size": fl.get("gain_ratio"), "gain_found": fl.get("gain_found"),
                                      "steady_lit_forest": fl.get("steady_lit"), "steady_lit_other": ot.get("steady_lit")})
        res["factors_all_folds"][w] = fac_out
        print(f"--- {w} loss alone (gated / loss factor): " + "  ".join(
            f"{x} corr {v['all'].get('corr')} loss {v['forest-loss'].get('loss_ratio')}/{v['forest-loss'].get('loss_found')} "
            f"steady {v['forest-loss'].get('steady_lit')}/{v['other'].get('steady_lit')}" for x, v in res[w]["loss_alone"].items()), flush=True)
        for r in [r for r in table if r["window"] == w]:
            print(f"{w} loss {r['loss']:4s} gain {r['gain']:9s} {r['variant']:16s} [{r['reads']:13s}] corr {r['corr']} "
                  f"slope {r['slope']} loss {r['loss_size']}/{r['loss_found']} gain {r['gain_size']}/{r['gain_found']} "
                  f"steady {r['steady_lit_forest']}/{r['steady_lit_other']}", flush=True)
    res["table"] = table
    json.dump(res, open(RES, "w"), indent=1, default=float)
    res["lidar"] = lidar(res["factors_all_folds"], args)
    json.dump(res, open(RES, "w"), indent=1, default=float)
    res["ramp"] = ramp(res["factors_all_folds"], T, args.ramp_pair)
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(f"done, peak {peak[0]:.1f} GB", flush=True)


def interp(fac, key_path, L):
    """A factor for a window of L years: linear between the one-year and seven-year fits."""
    a, b = fac["L1"], fac["L7"]
    for k in key_path:
        a, b = a[k], b[k]
    return a + (b - a) * (L - 1) / 6


def lidar(fac, args):
    """Two-flight lidar squares: lidar canopy change against every pairing (mean of the 8 fold heads)."""
    picks = json.load(open(HERE / "data" / "lidar_squares.json"))
    snets = {i: lc.nets_of(HERE / "data" / "heads" / f"{i}_square.npz", DIMS[i]) for i in INPUTS}
    tnets = {i: lc.nets_of(HERE / "data" / "heads" / f"traj_{i}.npz", DIMS[i], traj=True) for i in TRAJ}
    pool, per = {}, {}
    for name in picks:
        fl, fs = HERE / "data" / "lidar" / f"{name}.npz", HERE / "data" / "squares" / f"{name}.npz"
        if not (fl.exists() and fs.exists()):
            continue
        Ld = np.load(fl)
        if str(Ld["source"]) == "werk10" or Ld["chm"].shape[0] < 2:
            continue
        y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
        if y2 <= y1:
            continue
        i1, i2 = YEARS.index(y1), YEARS.index(y2)
        L = i2 - i1
        z = np.load(fs)
        F = bc.square_feats(z)
        ok = ok_of(F, "all")
        chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
        m = np.all(cov >= 80, 0) & ok[i1] & ok[i2]
        if m.sum() < 2000:
            continue
        pa, pt = predict(F, snets, tnets)
        mv = {k: v[0][m] for k, v in moves(F, [(i1, i2)]).items()}
        ch = {k: v.sum(0)[m] for k, v in moves(F, [(y, y + 1) for y in range(i1, i2)]).items()}
        del F
        c1, c2 = chm[0][m], chm[1][m]
        dH = c2 - c1
        lost, gain = (dH <= -5) & (c1 >= 5), dH >= 3
        steady = (np.abs(dH) < 1) & (np.abs(dH) < 0.3 * np.maximum(c1, 1))
        per[name] = {"years": [y1, y2], "n_lost": int(lost.sum()), "n_gain": int(gain.sum()), "n_steady": int(steady.sum())}
        ds = {i: (pa[i][i2] - pa[i][i1])[m] for i in INPUTS}
        a1 = {i: pa[i][i1][m] for i in INPUTS}
        gpos = {f"oney_{i}": np.maximum(ds[i], 0) for i in INPUTS} | {f"traj_{i}": np.maximum((pt[i][i2] - pt[i][i1])[m], 0) for i in TRAJ}
        meth = {}
        for x in INPUTS:
            negx = np.where(mv[x] >= 1, np.minimum(ds[x], 0), 0) / interp(fac, ("loss", x), L)
            meth[f"loss {x} alone"] = (np.where(mv[x] >= 1, ds[x], 0) / interp(fac, ("gated_loss", x), L), x)
            for g in gpos:
                gi = g.split("_", 1)[1]
                key = f"loss {x} + gain {g}"
                meth[f"{key}/ungated"] = (negx + gpos[g] / interp(fac, ("gain", g), L), x)
                # the gradual gate: distance for one year, chain past one year (its cutoff and factor from 2018 to 2025)
                gk = f"{key}/{'distance' if L == 1 else 'chain'}"
                gw = fac["L1" if L == 1 else "L7"]["gradual"].get(gk)
                if gw:
                    sc = mv[gi] if L == 1 else ch[gi]
                    meth[f"{key}/gradual"] = (negx + np.where(sc >= gw["q"], gpos[g], 0) / gw["gain"], x)
        for k, (da, x) in meth.items():
            base = np.maximum(a1[x], 1)
            sl_ = (da <= -10) & (da <= -0.25 * base)
            sg_ = (da >= 5) & (da >= 0.25 * base)
            d = pool.setdefault(k, {"dH": [], "da": [], "lost": [], "gain": [], "steady": [], "sl": [], "sg": []})
            for kk, v in (("dH", dH), ("da", da), ("lost", lost), ("gain", gain), ("steady", steady), ("sl", sl_), ("sg", sg_)):
                d[kk].append(v)
        print(f"lidar {name} {y1}-{y2}: lost {per[name]['n_lost']} gain {per[name]['n_gain']} steady {per[name]['n_steady']}", flush=True)
        del z, pa, pt
    out = {"per_square": per, "pooled": {}}
    for k, d in pool.items():
        d = {kk: np.concatenate(v) for kk, v in d.items()}
        c = np.cov(d["dH"], d["da"])
        out["pooled"][k] = {"lost_found": round(float(d["sl"][d["lost"]].mean()), 3),
                            "gain_found": round(float(d["sg"][d["gain"]].mean()), 3),
                            "steady_lit": round(float((d["sl"] | d["sg"])[d["steady"]].mean()), 3),
                            "corr": lc.corr(d["da"], d["dH"]), "slope_Mg_per_m": round(float(c[0, 1] / c[0, 0]), 3)}
        print(f"LIDAR {k:44s} {json.dumps(out['pooled'][k])}", flush=True)
    return out


def ramp(fac, T, pair):
    """Spread of one pairing's combined change per pixel, yearly and 2018 to 2025, by WorldCover class."""
    x, g = pair.split("+")
    gi = g.split("_", 1)[1]
    S = np.load(PX)
    win, wc, cell = S["win"], S["wc"], S["cell"]
    ta, tb = T[cell, np.array(IA)[win]], T[cell, np.array(IB)[win]]
    td = tb - ta
    steady_cell = (np.abs(td / np.maximum(ta, 1)) < 0.1) & (np.abs(td) < 10)
    key = f"loss {x} + gain {g}"
    out = {"pair": key}
    for wname, sel_w, wk, gname in (("yearly", win < 7, "L1", "distance"), ("2018-2025", win == 7, "L7", "chain")):
        f = fac[wk]
        neg = np.where(S[f"mv_{x}"] >= 1, np.minimum(S[f"ds_{x}"], 0), 0) / f["loss"][x]
        pos = np.maximum(S[f"dt_{gi}"] if g.startswith("traj") else S[f"ds_{gi}"], 0)
        gw = f["gradual"][f"{key}/{gname}"]
        sc = S[f"mv_{gi}"] if gname == "distance" else S[f"ch_{gi}"]
        variants = {"ungated gain": neg + pos / f["gain"][g],
                    "gradual gate on gain": neg + np.where(sc >= gw["q"], pos, 0) / gw["gain"]}
        base = np.maximum(S[f"a_{x}"], 1)
        o = {}
        for vn, v in variants.items():
            ov = {}
            for cname, cm in [("all", np.ones(len(v), bool))] + [(nm, wc == code) for code, nm in WC.items()]:
                m = sel_w & cm & np.isfinite(v)
                if m.sum() < 500:
                    continue
                vv, pp = v[m], 100 * v[m] / base[m]
                moving = np.abs(vv) > 1e-6
                st = steady_cell[m]
                e = {"px": int(m.sum()), "share_nonzero": round(float(moving.mean()), 3),
                     "all_p2_p98_Mg": [round(float(np.percentile(vv, 2)), 1), round(float(np.percentile(vv, 98)), 1)]}
                if moving.sum() >= 100:
                    e["moving_p2_p98_Mg"] = [round(float(np.percentile(vv[moving], 2)), 1), round(float(np.percentile(vv[moving], 98)), 1)]
                    e["moving_p2_p98_pct"] = [round(float(np.percentile(pp[moving], 2)), 1), round(float(np.percentile(pp[moving], 98)), 1)]
                if st.sum() >= 100:
                    e["steady_band90_Mg"] = round(float(np.percentile(np.abs(vv[st]), 90)), 2)
                    e["steady_band90_pct"] = round(float(np.percentile(np.abs(pp[st]), 90)), 1)
                ov[cname] = e
            o[vn] = ov
        out[wname] = o
        for vn in variants:
            print(f"RAMP {wname} {vn}:", flush=True)
            for cname, e in o[vn].items():
                print(f"   {cname:9s} {e}", flush=True)
    return out


# ---------------------------------------------------------------- a floor on gain

TAU = [0, 2, 5, 10, 15, 20, 30, 40, 60]  # Mg/ha


def gain_scale(gp, t, base, how):
    """raw: 1 (the head's own size); mean: mean predicted over mean true change on clear gain cells."""
    if how == "raw":
        return 1.0
    m = np.isfinite(gp) & np.isfinite(t) & np.isfinite(base)
    gp, t, base = gp[m], t[m], base[m]
    sel = (t / np.maximum(base, 1) >= 0.5) & (t >= 20)
    return float(gp[sel].mean() / t[sel].mean()) if sel.sum() >= 20 else np.nan


def stage_floor(args):
    """Gain counted only where the gain source's 100 m cell change is at least tau Mg/ha (and a quarter of the cell's
    before value), divided by the gain factor of what passes; tau picked per fold on the other folds' steady cells as
    the lowest that keeps steady ground no noisier than the loss source alone (+ .01). The gain then reads as a 100 m
    layer (in a map: the trajectory change averaged over 10 x 10 pixels before the floor). Loss stays per pixel
    (cell mean of the gated negative part)."""
    peak = [0.0]
    guard(args.max_gb, peak)
    Z = dict(np.load(CELLS))
    T, kind, sq = Z["truth"], Z["kind"], Z["sq"]
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    fold_sq = {}
    for fname, test in bh.folds(rows, "square"):
        for s_ in np.unique(rows["sq"][test]):
            fold_sq[int(s_)] = int(fname)
    fold = np.array([fold_sq[int(s_)] for s_ in sq])
    folds = np.unique(fold)
    ta, tb = T[:, IA], T[:, IB]
    TD = tb - ta
    groups = {"all": np.ones(len(kind), bool), "forest-loss": kind == "forest-loss", "other": kind != "forest-loss"}
    res = json.load(open(RES))
    out, table, fac = {"tau_grid": TAU, "scales": list(args.scales)}, [], {}
    for scale, w, sl in [(sc, w, sl) for sc in args.scales for w, sl in (("L1", slice(0, 7)), ("L7", slice(7, 8)))]:
        t_w, b_w = TD[:, sl], ta[:, sl]
        neg = {x: Z[f"s_neg_{x}"][:, sl] for x in INPUTS}
        gate = {x: Z[f"s_gate_{x}"][:, sl] for x in INPUTS}
        gsig = {f"traj_{i}": Z[f"t_plain_{i}"][:, sl] for i in TRAJ} | {f"oney_{i}": Z[f"s_plain_{i}"][:, sl] for i in INPUTS}
        out.setdefault(scale, {})[w] = {}
        fac.setdefault(scale, {})[w] = {}
        for x in INPUTS:
            fn_ = {f: ratio(neg[x][fold != f].ravel(), t_w[fold != f].ravel(), b_w[fold != f].ravel()) for f in folds}
            fg_ = {f: ratio(gate[x][fold != f].ravel(), t_w[fold != f].ravel(), b_w[fold != f].ravel()) for f in folds}
            ref = {f: lit((gate[x][fold != f] / fg_[f]).ravel(), t_w[fold != f].ravel(), b_w[fold != f].ravel()) for f in folds}
            for g, G in gsig.items():
                gi = g.split("_", 1)[1]
                P = np.full(t_w.shape, np.nan, np.float32)
                picks = {}
                for f in folds:
                    tr, te = fold != f, fold == f
                    choice = None
                    for tau in TAU:
                        pas = (G >= tau) & (G >= 0.25 * np.maximum(b_w, 1)) & (G > 0)
                        gp = np.where(pas, G, 0)
                        gq = gain_scale(gp[tr].ravel(), t_w[tr].ravel(), b_w[tr].ravel(), scale)
                        if not np.isfinite(gq) or gq <= 0.02:
                            continue
                        pr = neg[x][tr] / fn_[f] + gp[tr] / gq
                        if lit(pr.ravel(), t_w[tr].ravel(), b_w[tr].ravel()) <= ref[f] + 0.01:
                            choice = (tau, gq)
                            break
                    if choice is None:
                        picks[int(f)] = None
                        P[te] = neg[x][te] / fn_[f]
                        continue
                    tau, gq = choice
                    picks[int(f)] = tau
                    pas = (G[te] >= tau) & (G[te] >= 0.25 * np.maximum(b_w[te], 1)) & (G[te] > 0)
                    P[te] = neg[x][te] / fn_[f] + np.where(pas, G[te], 0) / gq
                o = {"reads": reads(x, gi), "tau_by_fold": picks}
                for gname, gm in groups.items():
                    t, pp, b = t_w[gm].ravel(), P[gm].ravel(), b_w[gm].ravel()
                    m = np.isfinite(t) & np.isfinite(pp) & np.isfinite(b)
                    o[gname] = bc.stats({"d": t[m], "a": b[m]}, pp[m])
                taus = [v for v in picks.values() if v is not None]
                if taus:
                    tm = max(set(taus), key=taus.count)
                    pas = (G >= tm) & (G >= 0.25 * np.maximum(b_w, 1)) & (G > 0)
                    fac[scale][w][f"{x}+{g}"] = {"tau": tm, "loss": ratio(neg[x].ravel(), t_w.ravel(), b_w.ravel()),
                                                 "gain": gain_scale(np.where(pas, G, 0).ravel(), t_w.ravel(), b_w.ravel(), scale)}
                key = f"loss {x} + gain {g}"
                out[scale][w][key] = o
                a, fl, ot = o["all"], o["forest-loss"], o["other"]
                table.append({"window": w, "loss": x, "gain": g, "variant": f"floor, gain {scale}", "reads": "+".join(o["reads"]),
                              "tau": picks, "corr": a.get("corr"), "slope": a.get("slope"),
                              "loss_size": fl.get("loss_ratio"), "loss_found": fl.get("loss_found"),
                              "gain_size": fl.get("gain_ratio"), "gain_found": fl.get("gain_found"),
                              "steady_lit_forest": fl.get("steady_lit"), "steady_lit_other": ot.get("steady_lit")})
                print(f"{w} floor[{scale}] loss {x:4s} gain {g:9s} [{'+'.join(o['reads']):13s}] tau {sorted(set(map(str, picks.values())))} "
                      f"corr {a.get('corr')} slope {a.get('slope')} loss {fl.get('loss_ratio')}/{fl.get('loss_found')} "
                      f"gain {fl.get('gain_ratio')}/{fl.get('gain_found')} steady {fl.get('steady_lit')}/{ot.get('steady_lit')}", flush=True)
    res["floor"] = out | {"table": table, "factors_all_folds": fac}
    json.dump(res, open(RES, "w"), indent=1, default=float)
    res["floor"]["lidar_cells"] = {sc: lidar_cells(fac[sc], args) for sc in args.scales}
    json.dump(res, open(RES, "w"), indent=1, default=float)
    print(f"done, peak {peak[0]:.1f} GB", flush=True)


def lidar_cells(fac, args):
    """Lidar squares, per CTrees 100 m cell: mean lidar height change against the floor rule and against loss alone
    (cells with 40 or more lidar pixels); found / lit by the same cell rules as biomass_change.stats, with lidar
    change read as biomass by the pooled slope of the loss-alone head (Mg/ha per m)."""
    picks = json.load(open(HERE / "data" / "lidar_squares.json"))
    pairs = [p for p in args.lidar_pairs]
    need_s = sorted({p.split("+")[0] for p in pairs} | {p.split("+")[1].split("_", 1)[1] for p in pairs if "oney_" in p})
    need_t = sorted({p.split("+")[1].split("_", 1)[1] for p in pairs if "traj_" in p})
    snets = {i: lc.nets_of(HERE / "data" / "heads" / f"{i}_square.npz", DIMS[i]) for i in need_s}
    tnets = {i: lc.nets_of(HERE / "data" / "heads" / f"traj_{i}.npz", DIMS[i], traj=True) for i in need_t}
    pool = {}
    for name in picks:
        fl, fs, flab = (HERE / "data" / d / f"{name}.npz" for d in ("lidar", "squares", "labels"))
        if not (fl.exists() and fs.exists() and flab.exists()):
            continue
        Ld = np.load(fl)
        if str(Ld["source"]) == "werk10" or Ld["chm"].shape[0] < 2:
            continue
        y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
        if y2 <= y1:
            continue
        i1, i2 = YEARS.index(y1), YEARS.index(y2)
        L = i2 - i1
        wk = "L1" if L == 1 else "L7"
        z = np.load(fs)
        F = bc.square_feats(z)
        ok = ok_of(F, "all")
        chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
        m = np.all(cov >= 80, 0) & ok[i1] & ok[i2]
        if m.sum() < 2000:
            continue
        pa, pt = predict(F, snets, tnets)
        mv = {k: v[0] for k, v in moves(F, [(i1, i2)]).items()}
        del F
        cell = np.load(flab)["cell"]
        ids, inv = np.unique(cell, return_inverse=True)

        def cm(v, valid):
            s_ = np.bincount(inv, np.where(valid, v, 0), ids.size)
            c = np.bincount(inv, valid, ids.size)
            return np.where(c >= 40, s_ / np.maximum(c, 1), np.nan)
        dH = cm(chm[1] - chm[0], m)
        h1 = cm(chm[0], m)
        for p in pairs:
            x, g = p.split("+")
            gi = g.split("_", 1)[1]
            f = fac[wk].get(p)
            if f is None:
                continue
            ds = pa[x][i2] - pa[x][i1]
            neg = cm(np.where(mv[x] >= 1, np.minimum(ds, 0), 0), m) / interp_floor(fac, p, "loss", L)
            alone = cm(np.where(mv[x] >= 1, ds, 0), m) / interp_floor(fac, p, "loss", L)
            G = cm((pt[gi][i2] - pt[gi][i1]) if g.startswith("traj") else (pa[gi][i2] - pa[gi][i1]), m)
            a1 = cm(pa[x][i1], m)
            pas = (G >= f["tau"]) & (G >= 0.25 * np.maximum(a1, 1)) & (G > 0)
            comb = neg + np.where(pas, G, 0) / interp_floor(fac, p, "gain", L)
            pairs_alone = [(f"{x} loss alone", alone)] if not any(k.startswith(f"{x} loss alone") for k in pool) or \
                p == next(q for q in pairs if q.split("+")[0] == x) else []
            for k, v in [(f"{p} floor", comb)] + pairs_alone:
                d = pool.setdefault(k, {"dH": [], "v": [], "h1": [], "a1": []})
                d["dH"].append(dH), d["v"].append(v), d["h1"].append(h1), d["a1"].append(a1)
        print(f"lidar cells {name} {y1}-{y2}: {int(np.isfinite(dH).sum())} cells", flush=True)
        del z, pa, pt
    out = {}
    for k, d in pool.items():
        d = {kk: np.concatenate(v) for kk, v in d.items()}
        ok_ = np.isfinite(d["dH"]) & np.isfinite(d["v"])
        dH, v, h1, a1 = d["dH"][ok_], d["v"][ok_], d["h1"][ok_], np.maximum(d["a1"][ok_], 1)
        lost = (dH <= -3) & (h1 >= 5)
        gain = dH >= 2
        steady = np.abs(dH) < 0.5
        seen_l = (v <= -10) & (v <= -0.25 * a1)
        seen_g = (v >= 5) & (v >= 0.25 * a1)
        out[k] = {"cells": int(ok_.sum()), "corr": lc.corr(v, dH), "lost_cells": int(lost.sum()),
                  "lost_found": round(float(seen_l[lost].mean()), 3) if lost.any() else None,
                  "gain_cells": int(gain.sum()), "gain_found": round(float(seen_g[gain].mean()), 3) if gain.any() else None,
                  "steady_cells": int(steady.sum()), "steady_lit": round(float((seen_l | seen_g)[steady].mean()), 3) if steady.any() else None}
        print(f"LIDAR CELLS {k:34s} {json.dumps(out[k])}", flush=True)
    return out


def interp_floor(fac, p, key, L):
    a, b = fac["L1"].get(p, {}).get(key), fac["L7"].get(p, {}).get(key)
    if a is None or b is None:
        return a if b is None else b
    return a + (b - a) * (L - 1) / 6


def stage_ramp2(args):
    """Ramp limits for the floor rule of one pairing (--ramp-pair, gain at its own size): the loss part per pixel
    (gated negative change over its loss factor), the gain part per 100 m cell (the gain source's cell change at or
    above tau), and steady ground; plus the plain one-year difference of the loss head per pixel, ungated."""
    x, g = args.ramp_pair.split("+")
    gi = g.split("_", 1)[1]
    res = json.load(open(RES))
    fac = res["floor"]["factors_all_folds"]["raw"]
    S = np.load(PX)
    Z = np.load(CELLS)
    T = Z["truth"]
    win, wc, cell = S["win"], S["wc"], S["cell"]
    ta_p, tb_p = T[cell, np.array(IA)[win]], T[cell, np.array(IB)[win]]
    steady_px = (np.abs((tb_p - ta_p) / np.maximum(ta_p, 1)) < 0.1) & (np.abs(tb_p - ta_p) < 10)
    TD = T[:, IB] - T[:, IA]
    TA = T[:, IA]
    steady_c = (np.abs(TD / np.maximum(TA, 1)) < 0.1) & (np.abs(TD) < 10)
    out = {"pair": args.ramp_pair, "rule": "loss per pixel (gated negative / loss factor) + gain per 100 m cell (cell change >= tau, own size)"}

    def pc(v, q):
        return [round(float(np.percentile(v, k)), 1) for k in q] if v.size >= 100 else None
    for wname, wk, sel_w, sl in (("yearly", "L1", win < 7, slice(0, 7)), ("2018-2025", "L7", win == 7, slice(7, 8))):
        f = fac[wk][f"{x}+{g}"]
        loss_px = np.where(S[f"mv_{x}"] >= 1, np.minimum(S[f"ds_{x}"], 0), 0) / f["loss"]
        base = np.maximum(S[f"a_{x}"], 1)
        plain = S[f"ds_{x}"]
        G = (Z[f"t_plain_{gi}"] if g.startswith("traj") else Z[f"s_plain_{gi}"])[:, sl]
        Bc = np.maximum(TA[:, sl], 1)
        pas = (G >= f["tau"]) & (G >= 0.25 * Bc) & (G > 0)
        o = {"tau_Mg": f["tau"], "loss_factor": round(f["loss"], 3), "classes": {}}
        o["gain_cells"] = {"share_of_cells_lit": round(float(pas.mean()), 4),
                           "lit_p50_p90_p98_Mg": pc(G[pas], (50, 90, 98)),
                           "lit_p50_p90_p98_pct": pc(100 * G[pas] / Bc[pas], (50, 90, 98)),
                           "steady_cells_lit": round(float(pas[steady_c[:, sl]].mean()), 4)}
        for cname, cm in [("all", np.ones(len(win), bool))] + [(nm, wc == code) for code, nm in WC.items()]:
            m = sel_w & cm
            if m.sum() < 500:
                continue
            lp = loss_px[m]
            nz = lp < 0
            e = {"px": int(m.sum()), "loss_share_px": round(float(nz.mean()), 4),
                 "loss_p2_p10_p50_Mg": pc(lp[nz], (2, 10, 50)), "loss_p2_p10_p50_pct": pc(100 * lp[nz] / base[m][nz], (2, 10, 50)),
                 "steady_px_loss_lit": round(float(nz[steady_px[m]].mean()), 4) if steady_px[m].sum() >= 100 else None,
                 "plain_p2_p98_Mg": pc(plain[m], (2, 98)), "plain_p2_p98_pct": pc(100 * plain[m] / base[m], (2, 98)),
                 "plain_steady_band90_Mg": round(float(np.percentile(np.abs(plain[m][steady_px[m]]), 90)), 2) if steady_px[m].sum() >= 100 else None,
                 "plain_steady_band90_pct": round(float(np.percentile(np.abs(100 * plain[m] / base[m])[steady_px[m]], 90)), 1) if steady_px[m].sum() >= 100 else None}
            o["classes"][cname] = e
        out[wname] = o
        print(f"RAMP2 {wname} {args.ramp_pair}: tau {f['tau']} gain cells {o['gain_cells']}", flush=True)
        for cname, e in o["classes"].items():
            print(f"   {cname:9s} {e}", flush=True)
    res["floor"]["ramp"] = out
    json.dump(res, open(RES, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cells", "score", "floor", "ramp2"], required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=12)
    ap.add_argument("--limit", type=int, default=0, help="first N squares only (smoke test; writes the same files)")
    ap.add_argument("--ramp-pair", default="all+traj_all", help="loss input + gain source for the ramp spread")
    ap.add_argument("--scales", nargs="*", default=["raw", "mean"], help="gain scale above the floor (stage floor)")
    ap.add_argument("--lidar-pairs", nargs="*", default=["aef+traj_all", "aef+traj_olmo", "aef+oney_aef", "all+traj_all",
                    "olmo+traj_olmo"], help="pairings scored against lidar per 100 m cell in stage floor")
    args = ap.parse_args()
    {"cells": stage_cells, "score": stage_score, "floor": stage_floor, "ramp2": stage_ramp2}[args.stage](args)


if __name__ == "__main__":
    main()
