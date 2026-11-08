# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Non-tree biomass against two maps that do not come from GEDI or CTrees: RAP (grass) and the SoCal chaparral map.

  --label rap        RAP v3 aboveground herbaceous biomass (annual + perennial forb and grass), Mg/ha, 2018 to 2025
                     (scripts/read_rap.py). Squares: us_squares.json group "rap" plus sandhills-us and great-basin-us.
                     Herbaceous only: shrub and tree mass are not in it, so amounts are compared by rank, not size.
  --label chaparral  SoCal annual aboveground live biomass (Schrader-Patton et al.), Mg/ha, 2018 to 2023
                     (scripts/read_chaparral.py). Squares: us_squares.json group "chaparral" and every other square
                     the map covers. Total live mass, so amounts compare directly. Its 2022 and 2023 fall on every
                     square, burned or not (the later-added years); --years 2018 2021 keeps the years before.

Everything is scored on 100 m blocks (10 x 10 AlphaEarth pixels; a block counts when 80 % of its pixels are valid
in the label and in all three embeddings), per year. Three things against the label:
  ctrees_head  the heads of biomass_head.py, taught on CTrees (a square in the training rows by its held-out fold
               head, any other by the mean of the 8 fold heads)
  ctrees       CTrees' own 100 m biomass under the block
  own_head     a head taught on the label itself from one year's embedding, each square held out in turn (the same
               64-unit head; log biomass per 10 m pixel, the label's value where the pixel falls)
Tests:
  amounts      R2 of log and median error (own_head, and ctrees_head on chaparral), rank correlation over blocks
               (pooled; and the median within a square), rank correlation of square medians
  years        each block's departure from its own mean over the years: correlation with the label's departure;
               each square's median by year: correlation with the label's (the median over squares); year-to-year
               steps correlated
  swings       consecutive-year steps where the label moves by at least --swing Mg/ha and 25 %: share the model
               moved the same way (by at least a third of the swing threshold and 10 %); steady steps (label under
               5 %): share the model lit by the same rule. Section 9's 10 Mg/ha floor would hide every grass swing
               here, so the share of label swings above 10 Mg/ha is reported too.
On RAP, also on herbaceous ground only (blocks under 10 % tree + shrub cover in RAP every year), since RAP leaves
woody mass out. Writes results/map_check_<label>.json and the block arrays to data/heads/map_check_<label>.npz. One square's features at a time plus a training table of about 1 GB:
peak about 6 GB.

    uv run scripts/map_check.py --label rap [--inputs aef tes olmo all] [--px 12000]
"""
import argparse
import json
import sys
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

YEARS = bc.YEARS
DIMS = {"aef": 64, "tes": 128, "olmo": 128, "all": 320}
LB_ACRE = 0.00112085  # lbs/acre -> Mg/ha
B = 10  # block side, pixels


def woody_of(kind, name):
    """(Y_label, n) RAP tree + shrub cover %, NaN none (rap only; None otherwise)."""
    if kind != "rap":
        return None
    c = np.load(HERE / "data" / "rap" / f"{name}.npz")["cover"]
    ok = (c[:, 4] != 255) & (c[:, 5] != 255)
    return np.where(ok, c[:, 4].astype(np.float32) + c[:, 5], np.nan)


def label_of(kind, name):
    """(Y_label, n) Mg/ha with NaN none, and the label's years."""
    if kind == "rap":
        d = np.load(HERE / "data" / "rap" / f"{name}.npz")
        a, p = d["afg"].astype(np.float32), d["pfg"].astype(np.float32)
        v = np.where((d["afg"] != 65535) & (d["pfg"] != 65535), (a + p) * LB_ACRE, np.nan)
        return v, list(d["years"])
    d = np.load(HERE / "data" / "chaparral" / f"{name}.npz")
    return d["agb"].astype(np.float32), list(d["years"])


def squares_of(kind):
    us = json.load(open(HERE / "data" / "us_squares.json"))
    names = [k for k, v in us.items() if v["group"] == kind]
    if kind == "rap":
        names += ["sandhills-us", "great-basin-us"]
    else:  # every square the chaparral map covers (the lidar squares in Los Angeles and Los Padres too)
        names += sorted(p.stem for p in (HERE / "data" / "chaparral").glob("*.npz") if p.stem not in names)
    return [n for n in names if (HERE / "data" / ("rap" if kind == "rap" else "chaparral") / f"{n}.npz").exists()
            and (HERE / "data" / "squares" / f"{n}.npz").exists()]


def blocks(v):
    """(..., n) row-major 600 x 600 -> (..., 3600) block means (NaN-aware) and valid share."""
    s = v.shape[:-1]
    w = v.reshape(*s, 600 // B, B, 600 // B, B)
    ok = np.isfinite(w)
    cnt = ok.sum((-3, -1))
    m = np.where(cnt > 0, np.nansum(w, (-3, -1)) / np.maximum(cnt, 1), np.nan)
    return m.reshape(*s, -1), (cnt / (B * B)).reshape(*s, -1)


def x_of(F, inp, y):
    if inp == "all":
        return np.concatenate([F[k][0][y] for k in bc.EMB], -1).astype(np.float32)
    return F[inp][0][y].astype(np.float32)


class One(torch.nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(d, bh.HID)
        self.l2 = torch.nn.Linear(bh.HID, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1((x - self.mu) / self.sd)))[..., 0]


def fit_one(X, t, epochs=6, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), min(200000, len(X)), replace=False)
    xs = X[sub].astype(np.float32)
    net = One(X.shape[1], xs.mean(0), xs.std(0) + 1e-4)
    del xs
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    bs = 2048
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    tt = torch.as_tensor(np.log1p(t).astype(np.float32))
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for b in range(0, len(X), bs):
            g = perm[b:b + bs]
            loss = torch.nn.functional.huber_loss(net(torch.as_tensor(X[g].astype(np.float32))), tt[g], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def run_one(net, x):
    return np.concatenate([np.clip(np.expm1(net(torch.as_tensor(x[b:b + 100000])).numpy()), 0, None)
                           for b in range(0, len(x), 100000)])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return None
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    return round(bh.spearman(a[m], b[m]), 3)


def score(T, P, sq, swing):
    """T, P (S, Y, nb) label and model on blocks (NaN none); sq names. Returns the tests above."""
    S, Y, nb = T.shape
    m = np.isfinite(T) & np.isfinite(P)
    o = {"blocks": int(m.any(1).sum()), "block_years": int(m.sum())}
    t, p = T[m], P[m]
    o["r2_log"] = round(lc.r2(np.log1p(t), np.log1p(p)), 3)
    o["median_error"] = round(float(np.median(np.abs(t - p))), 3)
    o["label_median"], o["model_median"] = round(float(np.median(t)), 3), round(float(np.median(p)), 3)
    o["rank_blocks"] = spear(t, p)
    within = [spear(T[s][m[s]], P[s][m[s]]) for s in range(S)]
    o["rank_within_square_median"] = round(float(np.median([w for w in within if w is not None])), 3) if any(w is not None for w in within) else None
    med_t = np.array([np.median(T[s][m[s]]) if m[s].any() else np.nan for s in range(S)])
    med_p = np.array([np.median(P[s][m[s]]) if m[s].any() else np.nan for s in range(S)])
    o["rank_square_medians"] = spear(med_t, med_p) if S >= 10 else (round(bh.spearman(med_t, med_p), 3) if S >= 4 else None)
    # years
    Tm = np.where(m, T, np.nan)
    Pm = np.where(m, P, np.nan)
    full = m.sum(1) >= Y - 1  # blocks with nearly every year
    with np.errstate(invalid="ignore"):
        dT = Tm - np.nanmean(Tm, 1, keepdims=True)
        dP = Pm - np.nanmean(Pm, 1, keepdims=True)
    sel = full[:, None, :] & m
    o["departure_corr"] = corr(dT[sel], dP[sel])
    series = {}
    for s in range(S):
        ts_ = np.array([np.nanmedian(Tm[s, y][full[s]]) if (full[s] & m[s, y]).any() else np.nan for y in range(Y)])
        ps_ = np.array([np.nanmedian(Pm[s, y][full[s]]) if (full[s] & m[s, y]).any() else np.nan for y in range(Y)])
        ok = np.isfinite(ts_) & np.isfinite(ps_)
        c = float(np.corrcoef(ts_[ok], ps_[ok])[0, 1]) if ok.sum() >= 4 and ts_[ok].std() > 0 and ps_[ok].std() > 0 else None
        series[sq[s]] = {"corr": round(c, 3) if c is not None else None, "label": [round(float(v), 2) for v in ts_],
                         "model": [round(float(v), 2) for v in ps_]}
    cs = [v["corr"] for v in series.values() if v["corr"] is not None]
    o["square_series_corr_median"] = round(float(np.median(cs)), 3) if cs else None
    o["square_series_corr_positive"] = f"{sum(c > 0 for c in cs)}/{len(cs)}"
    o["series"] = series
    sT, sP = Tm[:, 1:] - Tm[:, :-1], Pm[:, 1:] - Pm[:, :-1]
    ok = np.isfinite(sT) & np.isfinite(sP)
    o["step_corr"] = corr(sT[ok], sP[ok])
    base = np.maximum(Tm[:, :-1], 1e-3)[ok]
    st, sp = sT[ok], sP[ok]
    pbase = np.maximum(Pm[:, :-1], 1e-3)[ok]
    big = (np.abs(st) >= swing) & (np.abs(st) >= 0.25 * base)
    moved = (np.abs(sp) >= swing / 3) & (np.abs(sp) >= 0.10 * pbase)
    steady = np.abs(st) < 0.05 * base
    o["swings"] = {"n": int(big.sum()), "share_of_steps": round(float(big.mean()), 3),
                   "up": int((big & (st > 0)).sum()), "down": int((big & (st < 0)).sum()),
                   "found_same_way": round(float((moved & (np.sign(sp) == np.sign(st)))[big].mean()), 3) if big.any() else None,
                   "found_down": round(float((moved & (sp < 0))[big & (st < 0)].mean()), 3) if (big & (st < 0)).any() else None,
                   "found_up": round(float((moved & (sp > 0))[big & (st > 0)].mean()), 3) if (big & (st > 0)).any() else None,
                   "steady_n": int(steady.sum()),
                   "steady_lit": round(float(moved[steady].mean()), 3) if steady.any() else None,
                   "label_steps_over_10": round(float((np.abs(st) >= 10).mean()), 4)}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["rap", "chaparral"], required=True)
    ap.add_argument("--inputs", nargs="*", default=["aef", "tes", "olmo", "all"])
    ap.add_argument("--px", type=int, default=12000, help="training pixels per square-year for own_head")
    ap.add_argument("--max-gb", type=float, default=16)
    ap.add_argument("--years", type=int, nargs=2, help="first and last label year to use (chaparral: 2018 2021)")
    args = ap.parse_args()
    swing = 0.25 if args.label == "rap" else 5.0
    names = squares_of(args.label)
    print(f"{len(names)} squares: {names}", flush=True)
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    train_names = list(R["names"])
    fold_of = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of[train_names[s]] = fname
    heads = {inp: np.load(HERE / "data" / "heads" / f"{inp}_square.npz") for inp in args.inputs}
    all_nets = {inp: lc.nets_of(HERE / "data" / "heads" / f"{inp}_square.npz", DIMS[inp]) for inp in args.inputs}
    rng = np.random.default_rng(0)
    t0 = time.time()
    lab_years = None
    Tb, Cb, Wb = [], [], []  # label, CTrees, RAP tree + shrub cover on blocks (S, Y, nb)
    Hb = {inp: [] for inp in args.inputs}  # CTrees heads on blocks
    tab = {inp: [] for inp in args.inputs}
    tab_t, tab_s = [], []
    for si, name in enumerate(names):
        lab, lab_years = label_of(args.label, name)
        if args.years:
            keep_y = [i for i, y in enumerate(lab_years) if args.years[0] <= y <= args.years[1]]
            lab, lab_years = lab[keep_y], [lab_years[i] for i in keep_y]
        yi = [YEARS.index(y) for y in lab_years]
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        L = np.load(HERE / "data" / "labels" / f"{name}.npz")
        F = bc.square_feats(z)
        okE = F["aef"][1] & F["tes"][1] & F["olmo"][1]  # (8, n)
        okE = okE[yi]
        tb, tv = blocks(np.where(okE, lab, np.nan))
        tb[tv < 0.8] = np.nan
        Tb.append(tb)
        wd = woody_of(args.label, name)
        if wd is not None:
            Wb.append(blocks(np.where(okE, wd, np.nan))[0])
        agb = L["agb"].astype(np.float32)[yi]
        cb, cv = blocks(np.where((agb >= 0) & okE, agb / 10, np.nan))
        cb[cv < 0.8] = np.nan
        Cb.append(cb)
        for inp in args.inputs:
            nets = [bc.head_from(heads[inp], fold_of[name], DIMS[inp])] if name in fold_of else all_nets[inp]
            pred = np.full((len(yi), lab.shape[1]), np.nan, np.float32)
            for k, y in enumerate(yi):
                m = okE[k]
                a, _ = lc.state(nets, x_of(F, inp, y)[m])
                pred[k, m] = a
            pb, pv = blocks(pred)
            pb[pv < 0.8] = np.nan
            Hb[inp].append(pb)
        # training rows for own_head: pixels valid in label and embeddings
        for k, y in enumerate(yi):
            cand = np.flatnonzero(okE[k] & np.isfinite(lab[k]))
            pick = np.sort(rng.choice(cand, min(args.px, cand.size), replace=False))
            for inp in args.inputs:
                tab[inp].append(x_of(F, inp, y)[pick].astype(np.float16))
            tab_t.append(lab[k][pick])
            tab_s.append(np.full(pick.size, si, np.int16))
        del F, z, L
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"  {si + 1}/{len(names)} {name}: label median {np.nanmedian(lab):.2f}  [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")
    Tb, Cb = np.stack(Tb), np.stack(Cb)
    ts_, ss_ = np.concatenate(tab_t), np.concatenate(tab_s)
    # herbaceous ground (rap): blocks under 10 % tree + shrub cover in every year, where RAP's herbaceous mass is
    # most of the living mass and a total-biomass model should rank the same way
    herb = None
    if Wb:
        Wb = np.stack(Wb)
        herb = np.all(Wb < 10, 1, keepdims=True) & np.isfinite(Wb).all(1, keepdims=True)
        herb = np.broadcast_to(herb, Tb.shape)

    def both(P):
        o = score(Tb, P, names, swing)
        if herb is not None:
            o["herb_ground"] = {k: v for k, v in score(np.where(herb, Tb, np.nan), P, names, swing).items() if k != "series"}
        return o
    res = {"label": args.label, "years": [int(y) for y in lab_years], "squares": names, "swing_threshold": swing,
           "ctrees": both(Cb)}
    if herb is not None:
        res["herb_ground_share"] = round(float(herb[:, 0].mean()), 3)
    keep = {"label": Tb, "ctrees": Cb} | ({"woody": Wb} if herb is not None else {})
    print(f"ctrees      {json.dumps({k: v for k, v in res['ctrees'].items() if k != 'series'})}", flush=True)
    for inp in args.inputs:
        res.setdefault("ctrees_head", {})[inp] = both(np.stack(Hb[inp]))
        keep[f"ctrees_head_{inp}"] = np.stack(Hb[inp])
        X = np.concatenate(tab[inp])
        tab[inp] = None
        own = []
        for si, name in enumerate(names):
            net = fit_one(X[ss_ != si], ts_[ss_ != si])
            z = np.load(HERE / "data" / "squares" / f"{name}.npz")
            F = bc.square_feats(z)
            yi = [YEARS.index(y) for y in lab_years]
            okE = (F["aef"][1] & F["tes"][1] & F["olmo"][1])[yi]
            pred = np.full(okE.shape, np.nan, np.float32)
            for k, y in enumerate(yi):
                pred[k, okE[k]] = run_one(net, x_of(F, inp, y)[okE[k]])
            pb, pv = blocks(pred)
            pb[pv < 0.8] = np.nan
            own.append(pb)
            del F, z
        del X
        res.setdefault("own_head", {})[inp] = both(np.stack(own))
        keep[f"own_head_{inp}"] = np.stack(own)
        np.savez(HERE / "data" / "heads" / f"map_check_{args.label}.npz", names=np.array(names),
                 years=np.array(lab_years), **keep)
        for model in ("ctrees_head", "own_head"):
            print(f"{model:11s} {inp:5s} {json.dumps({k: v for k, v in res[model][inp].items() if k != 'series'})}", flush=True)
        print(f"  [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        json.dump(res, open(HERE / "results" / f"map_check_{args.label}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
