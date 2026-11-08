# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""lidar_check.py on the lidar read at 2 m (scripts/read_lidar2m.py, data/lidar2_squares.json): France, Scotland,
Canada, the Netherlands, Switzerland. Same tests; writes results/lidar_check2.json.

The heads against airborne lidar canopy height: independent of GEDI, CTrees and Meta.

None of the lidar squares was in the training rows, so each is predicted by all 8 square-fold heads of
biomass_head.py, averaged (and by the trajectory head's folds, data/heads/traj_<input>.npz, where present).
Pixels: lidar cover of 80 % or more in the epochs used, and valid in the embedding.

State, each lidar epoch with the embedding of its year (werk10: each pixel's campaign year, 2018 to 2023):
  height     the head's canopy height against the lidar's mean height per 10 m pixel: R2, mean error, by
             WorldCover class; Meta's height (the label the head was taught on) against the lidar beside it
  biomass    rank correlation of the head's biomass with lidar height, beside CTrees' and CCI's (100 m), as a
             check that amounts order the way canopy does
Change, two epochs (lidar dH = later minus earlier mean height):
  canopy lost   pixels with dH <= -5 m that stood 5 m or more: share where the head saw it (height down 2.5 m,
                or biomass down by a quarter and 10 Mg/ha), plain and gated (the frozen embedding cutoffs)
  canopy gained dH >= +3 m: share where the head's height rose 1.5 m or biomass rose by a quarter and 5 Mg/ha
  steady        |dH| < 1 m and height < 30 % change: share lit by the same rules
  correlation of the head's height change and biomass change with dH, per pixel and per CTrees 100 m cell,
  beside CTrees' own change at its cells

Writes results/lidar_check2.json. Peak RAM about 6 GB (one square's float features).

    uv run scripts/lidar_check2.py [--inputs aef tes olmo all]
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

YEARS = bc.YEARS
WC = {10: "trees", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def nets_of(path, d, traj=False):
    W = np.load(path)
    folds = sorted({k.split(".", 1)[0] for k in W.files if "." in k})
    out = []
    for f in folds:
        if traj:
            import trajectory_head as th
            net = th.Traj(d, np.zeros(d, np.float32), np.ones(d, np.float32))
        else:
            net = bh.Head(d, np.zeros(d, np.float32), np.ones(d, np.float32))
        net.load_state_dict({k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f + ".")})
        out.append(net.eval())
    return out


@torch.no_grad()
def state(nets, x):
    """x (n, D) -> biomass Mg/ha, height m: mean of the fold heads in linear units."""
    a, h = 0, 0
    for net in nets:
        o = torch.cat([net(torch.as_tensor(x[b:b + 100000])) for b in range(0, len(x), 100000)]).numpy()
        a = a + np.clip(np.expm1(np.minimum(o[:, 0], 8)), 0, None)
        h = h + np.clip(np.expm1(o[:, 1]), 0, None)
    return a / len(nets), h / len(nets)


@torch.no_grad()
def traj_state(nets, xs, ok):
    """xs (Y, n, D), ok (Y, n) -> biomass (Y, n), mean of the trajectory fold heads."""
    Y, n, D = xs.shape
    out = np.zeros((Y, n), np.float32)
    for net in nets:
        for b in range(0, n, 40000):
            xt = torch.as_tensor(xs[:, b:b + 40000].astype(np.float32)).permute(1, 0, 2)
            vt = torch.as_tensor(ok[:, b:b + 40000].T.copy())
            out[:, b:b + 40000] += np.clip(np.expm1(np.minimum(net(xt, vt).numpy().T, 8)), 0, None)
    return out / len(nets)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-9))


def spearman(a, b):
    if a.size < 30:
        return None
    return round(bh.spearman(a, b), 3)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3) if m.sum() > 30 and a[m].std() > 0 and b[m].std() > 0 else None


def cellmean(v, cell, valid):
    ids, inv = np.unique(cell, return_inverse=True)
    s = np.bincount(inv, np.where(valid, v, 0), ids.size)
    c = np.bincount(inv, valid, ids.size)
    out = np.where(c >= 40, s / np.maximum(c, 1), np.nan)
    return out, inv, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["aef", "tes", "olmo", "all"])
    args = ap.parse_args()
    picks = json.load(open(HERE / "data" / "lidar2_squares.json"))
    Dt = np.load(HERE / "models" / "detectors.npz")
    dims = {"aef": 64, "tes": 128, "olmo": 128, "all": 320}
    nets = {inp: nets_of(HERE / "data" / "heads" / f"{inp}_square.npz", dims[inp]) for inp in args.inputs}
    tnets = {inp: nets_of(HERE / "data" / "heads" / f"traj_{inp}.npz", dims[inp], traj=True)
             for inp in args.inputs if (HERE / "data" / "heads" / f"traj_{inp}.npz").exists()}
    res = {"squares": {}}
    pool = {}  # pooled pixels per (input, test) for overall numbers

    def add(key, **kv):
        d = pool.setdefault(key, {})
        for k, v in kv.items():
            d.setdefault(k, []).append(np.asarray(v))

    for name, p in picks.items():
        fl, fs, flab = (HERE / "data" / d / f"{name}.npz" for d in ("lidar", "squares", "labels"))
        if not (fl.exists() and fs.exists()):
            print(f"skip {name}: not read", flush=True)
            continue
        L = np.load(fl)
        z = np.load(fs)
        lab = np.load(flab) if flab.exists() else None
        cci = np.load(HERE / "data" / "cci" / f"{name}.npz") if (HERE / "data" / "cci" / f"{name}.npz").exists() else None
        F = bc.square_feats(z)
        wc = z["wc2021"]
        chm, cov = L["chm"].astype(np.float32), L["cover"]
        src = str(L["source"])
        out = {"source": src, "note": str(L["note"]), "years": L["year"].tolist(), "dates": L["dates"].tolist()}
        E = chm.shape[0]
        one_epoch = src == "werk10" or E == 1  # one flight: state only (France's LiDAR HD)
        if one_epoch:
            ypx = L["year_px"] if src == "werk10" else np.full(chm.shape[1], int(L["year"][0]), np.int16)
            ok_l = (cov[0] >= 80) & (ypx >= 2018) & (ypx <= 2025)
            epochs = None
        else:
            years = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in L["year"]]
            ok_l = np.all(cov >= 80, 0)
            epochs = years
        out["lidar_px"] = int(ok_l.sum())
        if ok_l.sum() < 2000:
            print(f"{name}: only {ok_l.sum()} pixels with lidar", flush=True)
            res["squares"][name] = out
            continue
        for inp in args.inputs:
            if inp == "all":
                okE = F["aef"][1] & F["tes"][1] & F["olmo"][1]
                X = lambda y: np.concatenate([F[k][0][y] for k in bc.EMB], -1).astype(np.float32)
            else:
                okE = F[inp][1]
                X = lambda y, _k=inp: F[_k][0][y].astype(np.float32)
            o = {}
            if one_epoch:
                hp = np.full(chm.shape[1], np.nan, np.float32)
                ap_ = np.full(chm.shape[1], np.nan, np.float32)
                for y in range(2018, 2026):
                    m = ok_l & (ypx == y) & okE[YEARS.index(y)]
                    if m.sum():
                        a, h = state(nets[inp], X(YEARS.index(y))[m])
                        hp[m], ap_[m] = h, a
                m = ok_l & np.isfinite(hp)
                t = chm[0][m]
                o["height_r2"], o["height_mae"] = round(r2(t, hp[m]), 3), round(float(np.mean(np.abs(t - hp[m]))), 2)
                o["height_bias"] = round(float(np.mean(hp[m] - t)), 2)
                o["biomass_vs_height_spearman"] = spearman(ap_[m], t)
                add((inp, "state"), t=t, p=hp[m], wc=wc[m], a=ap_[m])
                if inp == args.inputs[0] and lab is not None:
                    mh = m & (lab["height"] != 255)
                    o_meta = {"meta_r2": round(r2(chm[0][mh], lab["height"][mh].astype(np.float32)), 3),
                              "meta_mae": round(float(np.mean(np.abs(chm[0][mh] - lab["height"][mh]))), 2)}
                    out["meta"] = o_meta
                    add(("meta", "state"), t=chm[0][mh], p=lab["height"][mh].astype(np.float32), wc=wc[mh])
                out[inp] = o
                continue
            # two epochs
            y1, y2 = epochs
            i1, i2 = YEARS.index(y1), YEARS.index(y2)
            m = ok_l & okE[i1] & okE[i2]
            a1, h1 = state(nets[inp], X(i1)[m])
            a2, h2 = state(nets[inp], X(i2)[m])
            c1, c2 = chm[0][m], chm[1][m]
            for e, (t, hp) in enumerate(((c1, h1), (c2, h2))):
                o[f"height_r2_e{e}"] = round(r2(t, hp), 3)
                o[f"height_mae_e{e}"] = round(float(np.mean(np.abs(t - hp))), 2)
            o["biomass_vs_height_spearman"] = spearman(np.concatenate([a1, a2]), np.concatenate([c1, c2]))
            add((inp, "state"), t=np.concatenate([c1, c2]), p=np.concatenate([h1, h2]), wc=np.concatenate([wc[m], wc[m]]),
                a=np.concatenate([a1, a2]))
            dH = c2 - c1
            dh, da = h2 - h1, a2 - a1
            if y2 > y1:
                if inp == "all":
                    mv = np.zeros(m.sum(), int)
                    for k in bc.EMB:
                        u1, u2 = bc.unit(F[k][0][i1][m].astype(np.float32)), bc.unit(F[k][0][i2][m].astype(np.float32))
                        mv += (1 - (u1 * u2).sum(-1)) >= float(Dt[f"{k}_cut"])
                    moved = mv >= 2
                else:
                    u1, u2 = bc.unit(F[inp][0][i1][m].astype(np.float32)), bc.unit(F[inp][0][i2][m].astype(np.float32))
                    moved = (1 - (u1 * u2).sum(-1)) >= float(Dt[f"{inp}_cut"])
                lost = (dH <= -5) & (c1 >= 5)
                gain = dH >= 3
                steady = (np.abs(dH) < 1) & (np.abs(dH) < 0.3 * np.maximum(c1, 1))
                seen_l = (dh <= -2.5) | ((da <= -10) & (da <= -0.25 * np.maximum(a1, 1)))
                seen_g = (dh >= 1.5) | ((da >= 5) & (da >= 0.25 * np.maximum(a1, 1)))
                o["n_lost"], o["n_gain"], o["n_steady"] = int(lost.sum()), int(gain.sum()), int(steady.sum())
                if lost.sum() >= 30:
                    o["lost_found"] = round(float(seen_l[lost].mean()), 3)
                    o["lost_found_gated"] = round(float((seen_l & moved)[lost].mean()), 3)
                    o["lost_dh_median"] = round(float(np.median(dh[lost])), 2)
                    o["lost_dH_median"] = round(float(np.median(dH[lost])), 2)
                if gain.sum() >= 30:
                    o["gain_found"] = round(float(seen_g[gain].mean()), 3)
                    o["gain_dh_median"] = round(float(np.median(dh[gain])), 2)
                    o["gain_dH_median"] = round(float(np.median(dH[gain])), 2)
                if steady.sum() >= 30:
                    o["steady_lit"] = round(float((seen_l | seen_g)[steady].mean()), 3)
                    o["steady_lit_gated"] = round(float(((seen_l | seen_g) & moved)[steady].mean()), 3)
                o["corr_dh_dH"] = corr(dh, dH)
                o["corr_da_dH"] = corr(da, dH)
                add((inp, "change"), dH=dH, dh=dh, da=da, lost=lost, gain=gain, steady=steady, seen_l=seen_l,
                    seen_g=seen_g, moved=moved, c1=c1)
                if inp in tnets:
                    ok_all = okE
                    xs = np.stack([X(y) for y in range(len(YEARS))]) if inp != "all" else None
                    if xs is None:
                        xs = np.stack([X(y) for y in range(len(YEARS))])
                    ta = traj_state(tnets[inp], xs[:, m], ok_all[:, m])
                    tda = ta[i2] - ta[i1]
                    seen_tl = (tda <= -10) & (tda <= -0.25 * np.maximum(ta[i1], 1))
                    seen_tg = (tda >= 5) & (tda >= 0.25 * np.maximum(ta[i1], 1))
                    o["traj_corr_da_dH"] = corr(tda, dH)
                    if lost.sum() >= 30:
                        o["traj_lost_found"] = round(float(seen_tl[lost].mean()), 3)
                    if gain.sum() >= 30:
                        o["traj_gain_found"] = round(float(seen_tg[gain].mean()), 3)
                    if steady.sum() >= 30:
                        o["traj_steady_lit"] = round(float((seen_tl | seen_tg)[steady].mean()), 3)
                    add((inp, "traj"), dH=dH, tda=tda, lost=lost, gain=gain, steady=steady, seen_l=seen_tl, seen_g=seen_tg)
                    del xs
                # 100 m: lidar change, head change, CTrees change per CTrees cell
                if lab is not None and inp == args.inputs[0] or (lab is not None and inp == "all"):
                    cell = lab["cell"][m]
                    agb = lab["agb"].astype(np.float32)
                    ct = np.where((agb[i1] >= 0) & (agb[i2] >= 0), (agb[i2] - agb[i1]) / 10, np.nan)[m]
                    vH, inv, cnt = cellmean(dH, cell, np.ones(m.sum(), bool))
                    va, _, _ = cellmean(da, cell, np.ones(m.sum(), bool))
                    vc, _, _ = cellmean(ct, cell, np.isfinite(ct))
                    o["cell_corr_head_dH"] = corr(va, vH)
                    o["cell_corr_ctrees_dH"] = corr(vc, vH)
                    o["cells"] = int(np.isfinite(vH).sum())
                    add((inp, "cells"), vH=vH, va=va, vc=vc)
                    if cci is not None and y1 <= 2024 and y2 <= 2024:
                        ca = cci["agb"].astype(np.float32)
                        ca[ca == 65535] = np.nan
                        dcc = (ca[YEARS.index(y2)] - ca[YEARS.index(y1)])[m]
                        vcc, _, _ = cellmean(dcc, cell, np.isfinite(dcc))
                        o["cell_corr_cci_dH"] = corr(vcc, vH)
                        add((inp, "cells_cci"), vH=vH, vcc=vcc, va=va)
            if inp == args.inputs[0] and lab is not None:
                hd = lab["height"][m]
                mh = hd != 255
                hy = (lab["height_date"][m] // 10000)
                # Meta against the lidar epoch nearest its imagery date
                near = np.where(np.abs(hy - y1) <= np.abs(hy - y2), 0, 1)
                t = np.where(near == 0, c1, c2)
                out["meta"] = {"meta_r2": round(r2(t[mh], hd[mh].astype(np.float32)), 3),
                               "meta_mae": round(float(np.mean(np.abs(t[mh] - hd[mh]))), 2),
                               "meta_years": [int(np.percentile(hy[mh], 5)), int(np.percentile(hy[mh], 95))] if mh.any() else None}
                add(("meta", "state"), t=t[mh], p=hd[mh].astype(np.float32), wc=wc[m][mh])
            out[inp] = o
        res["squares"][name] = out
        print(f"{name} [{src}] years {out['years']} px {out['lidar_px']} meta {out.get('meta')}", flush=True)
        for inp in args.inputs:
            if inp in out:
                print(f"   {inp:5s} {json.dumps(out[inp])}", flush=True)
        print(f"   [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        del F, z

    # pooled
    tot = {}
    for (inp, test), d in pool.items():
        d = {k: np.concatenate(v) for k, v in d.items()}
        o = {}
        if test == "state":
            t, p = d["t"], d["p"]
            o["height_r2"], o["height_mae"], o["height_bias"] = round(r2(t, p), 3), round(float(np.mean(np.abs(t - p))), 2), round(float(np.mean(p - t)), 2)
            o["px"] = int(t.size)
            o["by_class"] = {}
            for code, nm in WC.items():
                mm = d["wc"] == code
                if mm.sum() >= 500:
                    o["by_class"][nm] = {"px": int(mm.sum()), "lidar_mean": round(float(t[mm].mean()), 2),
                                         "pred_mean": round(float(p[mm].mean()), 2),
                                         "mae": round(float(np.mean(np.abs(t[mm] - p[mm]))), 2)}
            bins = [0, 2, 5, 10, 20, 30, 45, 200]
            o["by_height"] = {f"{lo}-{hi}": round(float(np.median(p[(t >= lo) & (t < hi)])), 1)
                              for lo, hi in zip(bins[:-1], bins[1:]) if ((t >= lo) & (t < hi)).sum() >= 200}
            if "a" in d:
                o["biomass_vs_height_spearman"] = spearman(d["a"], t)
        elif test == "change":
            for k, sel in (("lost", d["lost"]), ("gain", d["gain"]), ("steady", d["steady"])):
                o[f"n_{k}"] = int(sel.sum())
            o["lost_found"] = round(float(d["seen_l"][d["lost"]].mean()), 3)
            o["lost_found_gated"] = round(float((d["seen_l"] & d["moved"])[d["lost"]].mean()), 3)
            o["lost_dh_vs_dH"] = [round(float(np.median(d["dh"][d["lost"]])), 2), round(float(np.median(d["dH"][d["lost"]])), 2)]
            o["gain_found"] = round(float(d["seen_g"][d["gain"]].mean()), 3)
            o["gain_dh_vs_dH"] = [round(float(np.median(d["dh"][d["gain"]])), 2), round(float(np.median(d["dH"][d["gain"]])), 2)]
            st = d["steady"]
            o["steady_lit"] = round(float((d["seen_l"] | d["seen_g"])[st].mean()), 3)
            o["steady_lit_gated"] = round(float(((d["seen_l"] | d["seen_g"]) & d["moved"])[st].mean()), 3)
            o["corr_dh_dH"], o["corr_da_dH"] = corr(d["dh"], d["dH"]), corr(d["da"], d["dH"])
            c = np.cov(d["dH"], d["dh"])
            o["slope_dh_on_dH"] = round(float(c[0, 1] / c[0, 0]), 3)
        elif test == "traj":
            o["lost_found"] = round(float(d["seen_l"][d["lost"]].mean()), 3)
            o["gain_found"] = round(float(d["seen_g"][d["gain"]].mean()), 3)
            o["steady_lit"] = round(float((d["seen_l"] | d["seen_g"])[d["steady"]].mean()), 3)
            o["corr_da_dH"] = corr(d["tda"], d["dH"])
        elif test == "cells":
            o["cells"] = int(np.isfinite(d["vH"]).sum())
            o["corr_head_dH"], o["corr_ctrees_dH"] = corr(d["va"], d["vH"]), corr(d["vc"], d["vH"])
        elif test == "cells_cci":
            o["corr_cci_dH"], o["corr_head_dH_same_cells"] = corr(d["vcc"], d["vH"]), corr(d["va"], d["vH"])
        tot[f"{inp}/{test}"] = o
        print(f"POOLED {inp}/{test}: {json.dumps({k: v for k, v in o.items() if k not in ('by_class',)})}", flush=True)
        if "by_class" in o:
            print("        by class " + "  ".join(f"{k} {v['lidar_mean']}->{v['pred_mean']} ({v['px']})" for k, v in o["by_class"].items()), flush=True)
    res["pooled"] = tot
    json.dump(res, open(HERE / "results" / "lidar_check2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
