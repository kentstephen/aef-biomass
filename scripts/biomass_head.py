# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""PLAN.md step 2 and 3a: one head per year's embedding -> biomass, canopy height, class; held out by place.

Rows: data/rows.npz (scripts/biomass_rows.py). Input: one year's embedding, AlphaEarth (64), TESSERA (128),
OlmoEarth (128) or all three (320), standardized on the training squares. Head: 64 hidden units (fire-knn's size),
outputs log biomass per 10 m pixel, log canopy height, 9 class logits.
  biomass  the mean of the head's linear Mg/ha over a cell's P pixels is matched to CTrees' cell (log space loss)
  height   Meta 10 m mean height, only in the year of Meta's imagery, 2017 on (2017 with 2018; taught per pixel)
  class    WorldCover where Impact Observatory agrees, 2020 and 2021 (taught per pixel)

Held out: "square" = 8 folds of squares (kinds spread over the folds), every square scored by a head that never saw
it; "continent" = each continent left out in turn. Scores are on cells and pixels valid in all three embeddings, so
the inputs compare on the same ground. fire-knn's California volume head is scored beside them as the baseline.

Writes data/heads/<input>_<scheme>.npz (held-out predictions and every fold's weights), results/biomass_head.json.
RAM: rows ~1.2 GB int8, one input's float16 features up to ~1.9 GB, peak ~6 GB.

    uv run scripts/biomass_head.py [--inputs aef tes olmo all] [--schemes square continent] [--epochs 8]
        [--weight-top 1 --tag _top]   more weight on high biomass, written beside the plain heads
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
OUT = HERE / "data" / "heads"
RES = HERE / "results" / "biomass_head.json"
DIMS = {"aef": 64, "tes": 128, "olmo": 128}
HID = 64
NCLS = 9
BINS = [0, 10, 50, 150, 300, 1e9]


def deq(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def features(R, key, out=None, lo=0):
    """float16 (G, Y, P, D) and valid mask (G, Y, P) for one embedding, converted in chunks (no full float32 copy);
    written into out[..., lo:lo + D] when given."""
    q = R[key]
    D = q.shape[-1]
    if out is None:
        out, lo = np.empty(q.shape, np.float16), 0
    if key == "tes":
        s = R["tes_s"]
        ok = np.isfinite(s) & (s > 0)
    else:
        ok = q[..., 0] != -128
    for b in range(0, len(q), 2000):
        if key == "tes":
            v = q[b:b + 2000].astype(np.float32) * np.nan_to_num(s[b:b + 2000].astype(np.float32))[..., None]
        else:
            v = deq(q[b:b + 2000])
        out[b:b + 2000, ..., lo:lo + D] = v
    return out, ok


def build(R, inp):
    if inp == "all":
        G, Y, P, _ = R["aef"].shape
        X = np.empty((G, Y, P, sum(DIMS.values())), np.float16)
        ok, lo = np.ones((G, Y, P), bool), 0
        for k in ("aef", "tes", "olmo"):
            _, okk = features(R, k, X, lo)
            ok &= okk
            lo += DIMS[k]
        return X, ok
    return features(R, inp)


class Head(torch.nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(d, HID)
        self.l2 = torch.nn.Linear(HID, 2 + NCLS)

    def forward(self, x):
        return self.l2(torch.relu(self.l1((x - self.mu) / self.sd)))


def targets(R):
    Y = R["agb"].shape[1]
    years = list(R["years"])
    hy = np.clip(R["height_year"], years[0], years[-1])
    # imagery from 2017 on only (2017 paired with 2018); older imagery may predate change in the embedding years
    hmask = (R["height"] != 255) & (R["height_year"] >= years[0] - 1)
    # height taught in the year of the imagery only: (G, Y, P)
    hy_idx = hy - years[0]
    H = np.zeros(R["cls"].shape, bool)
    for y in range(Y):
        H[:, y, :] = hmask & (hy_idx == y)
    return H


def fit(X, ok, R, H, train, epochs, seed, weight_top=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G, Y, P, D = X.shape
    sub = rng.choice(train, size=min(3000, train.size), replace=False)
    xs = X[sub].reshape(-1, D).astype(np.float32)
    m = ok[sub].reshape(-1)
    mu, sd = xs[m].mean(0), xs[m].std(0) + 1e-4
    del xs, m
    net = Head(D, mu, sd)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(train.size / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    agb = torch.as_tensor(np.nan_to_num(R["agb"], nan=-1.0))
    hgt = torch.as_tensor(R["height"].astype(np.float32))
    cls = torch.as_tensor(R["cls"].astype(np.int64))
    Ht = torch.as_tensor(H)
    okt = torch.as_tensor(ok)
    for ep in range(epochs):
        perm = rng.permutation(train)
        for b in range(0, perm.size, 256):
            g = np.sort(perm[b:b + 256])
            x = torch.as_tensor(X[g].astype(np.float32))
            o = net(x)  # (B, Y, P, 11)
            v = okt[g]
            pa = torch.expm1(o[..., 0].clamp(max=8)).clamp(min=0) * v
            n = v.sum(-1)
            cm = pa.sum(-1) / n.clamp(min=1)
            t = agb[g]
            mb = (t >= 0) & (n >= P // 2)
            # --weight-top w: cells above 100 Mg/ha count up to (1 + w * 4) times (weight 1 + w * clip(agb/100 - 1, 0, 4))
            wb = 1 + weight_top * ((t / 100 - 1).clamp(min=0, max=4))
            lb = (wb * (torch.log1p(cm) - torch.log1p(t.clamp(min=0))) ** 2)[mb].sum() / wb[mb].sum() if mb.any() else 0
            mh = Ht[g] & v
            lh = torch.nn.functional.huber_loss(o[..., 1][mh], torch.log1p(hgt[g][:, None, :].expand_as(mh)[mh]), delta=0.5) if mh.any() else 0
            c = cls[g]
            mc = (c >= 0) & v
            lc = torch.nn.functional.cross_entropy(o[..., 2:][mc], c[mc]) if mc.any() else 0
            loss = lb + lh + 0.5 * lc
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def predict(net, X, ok, idx):
    G, Y, P, D = X.shape
    pa, ph, pc = np.zeros((idx.size, Y)), np.zeros((idx.size, Y, P), np.float32), np.zeros((idx.size, Y, P), np.int8)
    for b in range(0, idx.size, 1024):
        g = idx[b:b + 1024]
        o = net(torch.as_tensor(X[g].astype(np.float32))).numpy()
        v = ok[g]
        a = np.clip(np.expm1(np.minimum(o[..., 0], 8)), 0, None) * v
        n = v.sum(-1)
        pa[b:b + g.size] = np.where(n >= P // 2, a.sum(-1) / np.maximum(n, 1), np.nan)
        ph[b:b + g.size] = np.expm1(o[..., 1])
        pc[b:b + g.size] = o[..., 2:].argmax(-1)
    return pa, ph, pc


def folds(R, scheme, k=8):
    sq, kinds, conts = R["sq"], R["kinds"], R["continents"]
    if scheme == "continent":
        names = sorted(set(conts))
        return [(c, np.flatnonzero(conts[sq] == c)) for c in names]
    rng = np.random.default_rng(1)
    order = []
    for kd in sorted(set(kinds)):
        s = np.flatnonzero(kinds == kd)
        order += list(rng.permutation(s))
    fold_of = {s: i % k for i, s in enumerate(order)}
    f = np.array([fold_of[s] for s in range(len(kinds))])
    return [(str(i), np.flatnonzero(f[sq] == i)) for i in range(k)]


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def score(R, H, common, pa, ph, pc):
    """Biomass on cell-years, height on its pixel-year, class on agreed pixel-years; overall and per kind."""
    kinds = R["kinds"][R["sq"]]
    t = R["agb"]
    out = {}
    groups = {"all": np.ones(len(kinds), bool)} | {k: kinds == k for k in sorted(set(R["kinds"]))}
    n_common = common.sum(-1)
    for gname, gm in groups.items():
        s = {}
        m = gm[:, None] & np.isfinite(t) & np.isfinite(pa) & (n_common >= R["height"].shape[1] // 2)
        if m.sum() > 50:
            tt, pp = t[m], pa[m]
            s["agb_r2_log"] = round(r2(np.log1p(tt), np.log1p(pp)), 3)
            s["agb_r2"] = round(r2(tt, pp), 3)
            s["agb_mae"] = round(float(np.median(np.abs(tt - pp))), 1)
            s["agb_true_med"] = round(float(np.median(tt)), 1)
            s["agb_pred_med"] = round(float(np.median(pp)), 1)
            s["agb_bins"] = {f"{BINS[i]}-{BINS[i + 1] if BINS[i + 1] < 1e8 else ''}": [int(((tt >= BINS[i]) & (tt < BINS[i + 1])).sum()),
                             round(float(np.median(pp[(tt >= BINS[i]) & (tt < BINS[i + 1])])), 1) if ((tt >= BINS[i]) & (tt < BINS[i + 1])).any() else None]
                             for i in range(len(BINS) - 1)}
        hm = gm[:, None, None] & H & common
        if hm.sum() > 50:
            ht = np.broadcast_to(R["height"][:, None, :], H.shape)[hm].astype(np.float32)
            hp = ph[hm]
            s["height_r2"] = round(r2(ht, hp), 3)
            s["height_mae"] = round(float(np.mean(np.abs(ht - hp))), 2)
        cm = gm[:, None, None] & (R["cls"] >= 0) & common
        if cm.sum() > 50:
            ct, cp = R["cls"][cm], pc[cm]
            s["cls_acc"] = round(float((ct == cp).mean()), 3)
            f1 = []
            for c in range(NCLS):
                tp, fp, fn = ((cp == c) & (ct == c)).sum(), ((cp == c) & (ct != c)).sum(), ((cp != c) & (ct == c)).sum()
                if tp + fn >= 50:
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            s["cls_macro_f1"] = round(float(np.mean(f1)), 3) if f1 else None
        out[gname] = s
    return out


def baseline(R, common):
    """fire-knn's California volume head (AlphaEarth): its cell-mean volume ranked against CTrees, per kind."""
    import volume_head as vh
    G, Y, P, _ = R["aef"].shape
    vol = np.zeros((G, Y))
    for b in range(0, G, 2000):
        q = R["aef"][b:b + 2000].reshape(-1, 64)
        v = vh.heads(q)["vol"].reshape(-1, Y, P)
        vv = np.where(common[b:b + 2000], v, np.nan)
        cnt = np.isfinite(vv).sum(-1)
        vol[b:b + 2000] = np.where(cnt >= P // 2, np.nansum(vv, -1) / np.maximum(cnt, 1), np.nan)
    kinds = R["kinds"][R["sq"]]
    out = {}
    for gname in ["all"] + sorted(set(R["kinds"])):
        gm = np.ones(G, bool) if gname == "all" else kinds == gname
        m = gm[:, None] & np.isfinite(R["agb"]) & np.isfinite(vol)
        if m.sum() > 50:
            out[gname] = {"vol_spearman_agb": round(spearman(vol[m], R["agb"][m]), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["aef", "tes", "olmo", "all"])
    ap.add_argument("--schemes", nargs="*", default=["square", "continent"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=14)
    ap.add_argument("--weight-top", type=float, default=0.0, help="extra weight on cells above 100 Mg/ha (0 = none)")
    ap.add_argument("--tag", default="", help="suffix for the output names, e.g. _top")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(np.load(HERE / "data" / "rows.npz"))
    H = targets(R)
    common = np.ones(R["cls"].shape, bool)
    for k in ("aef", "tes", "olmo"):
        common &= (np.isfinite(R["tes_s"]) & (R["tes_s"] > 0)) if k == "tes" else (R[k][..., 0] != -128)
    res = json.load(open(RES)) if RES.exists() else {}
    res["rows"] = {"cells": int(len(R["agb"])), "squares": int(len(R["names"])), "common_px_years": round(float(common.mean()), 3)}
    if "baseline_firekn_volume" not in res:
        res["baseline_firekn_volume"] = baseline(R, common)
        print("baseline", json.dumps(res["baseline_firekn_volume"]["all"]), flush=True)
    for inp in args.inputs:
        X, ok = build(R, inp)
        for scheme in args.schemes:
            t0 = time.time()
            G, Y, P, D = X.shape
            pa, ph, pc = np.full((G, Y), np.nan), np.zeros((G, Y, P), np.float32), np.zeros((G, Y, P), np.int8)
            weights = {}
            for name, test in folds(R, scheme):
                train = np.setdiff1d(np.arange(G), test)
                net = fit(X, ok, R, H, train, args.epochs, seed=0, weight_top=args.weight_top)
                a, h, c = predict(net, X, ok, test)
                pa[test], ph[test], pc[test] = a, h, c
                for k, v in net.state_dict().items():
                    weights[f"{name}.{k}"] = v.numpy()
                gb = psutil.Process().memory_info().rss / 1e9
                print(f"  {inp} {scheme} fold {name}: {test.size} cells, {time.time() - t0:.0f} s, {gb:.1f} GB", flush=True)
                if gb > args.max_gb:
                    raise SystemExit(f"STOP: RSS {gb:.1f} GB")
            s = score(R, H, common, pa, ph, pc)
            res.setdefault(inp, {})[scheme + args.tag] = s
            a = s["all"]
            print(f"{inp:5s} {scheme:9s} agb R2 log {a.get('agb_r2_log')} lin {a.get('agb_r2')} mae {a.get('agb_mae')} | "
                  f"height R2 {a.get('height_r2')} mae {a.get('height_mae')} | class acc {a.get('cls_acc')} f1 {a.get('cls_macro_f1')}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
            np.savez(OUT / f"{inp}_{scheme}{args.tag}.npz", pa=pa.astype(np.float32), ph=ph, pc=pc, **weights)
            json.dump(res, open(RES, "w"), indent=1)
        del X, ok


if __name__ == "__main__":
    main()
