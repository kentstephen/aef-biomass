# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""PLAN.md step 4 (the "step 3" of this phase): biomass change between years, against CTrees' own change.

Every square is predicted, every pixel and year, by the square-held-out heads of biomass_head.py (each square by the
fold that never saw it), then folded to CTrees' 100 m cells (mean of linear Mg/ha over the cell's pixels in the
square; cells with at least 40 pixels). Windows: each pair of consecutive years 2018 to 2025, and 2018 to 2025.

Change, five ways, per embedding input:
  plain       later minus earlier state
  calibrated  plain, divided by its muting (slope of predicted on true change) measured on the other folds' cells
  gated       state change kept only on pixels whose embedding moved past the frozen cutoff (models/detectors.npz,
              1 - cosine >= <emb>_cut; "all" = 2 of 3); pixels that did not move count as no change
  pair        a head taught on two years at once (both embeddings and their difference) to the cell's change in log
              biomass, from data/rows.npz, same square folds (direct change instead of differencing two states)
Scored overall, outside the tree-loss squares ("other") and per kind of square, against CTrees' change: slope of predicted on true (1 = full size, below 1 = muted), correlation; loss cells
(CTrees fell by half and 20 Mg/ha or more): median predicted / true, share found (predicted fall of a quarter and
10 Mg/ha); gain cells likewise; steady cells (CTrees moved under 10 % and under 10 Mg/ha): share lit.
And at 10 m, in the tree-loss squares: the head's biomass on Hansen-loss pixels the year before and after the loss,
beside never-lost forest, and beside CTrees' 100 m cell under the same pixels.

Peak RAM: one square (~1.2 GB) and its float features (~0.5 GB per year) plus the pair rows (~3 GB): ~8 GB.

    uv run scripts/biomass_change.py [--inputs aef tes olmo all] [--pair-epochs 8]
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
import biomass_head as bh  # noqa: E402

RES = HERE / "results" / "biomass_change.json"
YEARS = list(range(2018, 2026))
WINDOWS = [(y, y + 1) for y in range(2018, 2025)] + [(2018, 2025)]
EMB = ("aef", "tes", "olmo")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def square_feats(z):
    """float16 (Y, n, D) and valid (Y, n) per embedding, for one square, converted a year at a time."""
    out = {}
    for k in EMB:
        q = z[k]
        if k == "tes":
            s = z["tes_scale"]
            ok = np.isfinite(s) & (s > 0)
        else:
            ok = q[..., 0] != -128
        e = np.empty(q.shape, np.float16)
        for y in range(q.shape[0]):
            e[y] = (q[y].astype(np.float32) * np.nan_to_num(s[y])[:, None]) if k == "tes" else bh.deq(q[y])
        out[k] = (e, ok)
        del q
    return out


def head_from(W, fold, d):
    net = bh.Head(d, np.zeros(d, np.float32), np.ones(d, np.float32))
    sd = {k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f"{fold}.")}
    net.load_state_dict(sd)
    return net.eval()


class Pair(torch.nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(3 * d, bh.HID)
        self.l2 = torch.nn.Linear(bh.HID, 1)

    def forward(self, a, b):
        a, b = (a - self.mu) / self.sd, (b - self.mu) / self.sd
        return self.l2(torch.relu(self.l1(torch.cat([a, b, b - a], -1))))[..., 0]


def fit_pair(X, ok, agb, train, epochs, seed=0):
    """X float16 (G, Y, P, D). Target: cell change in log1p biomass for every window; the mean of the pixels'
    predicted change is matched to it."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G, Y, P, D = X.shape
    sub = rng.choice(train, size=min(3000, train.size), replace=False)
    xs = X[sub].reshape(-1, D).astype(np.float32)
    m = ok[sub].reshape(-1)
    net = Pair(D, xs[m].mean(0), xs[m].std(0) + 1e-4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(train.size / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    la = np.log1p(np.nan_to_num(agb, nan=-0.5).clip(min=-0.5))
    ia = np.array([YEARS.index(a) for a, _ in WINDOWS])
    ib = np.array([YEARS.index(b) for _, b in WINDOWS])
    for _ in range(epochs):
        perm = rng.permutation(train)
        for s in range(0, perm.size, 256):
            g = np.sort(perm[s:s + 256])
            x = torch.as_tensor(X[g].astype(np.float32))
            v = torch.as_tensor(ok[g][:, ia] & ok[g][:, ib])  # (B, W, P)
            o = net(x[:, ia], x[:, ib]) * v
            n = v.sum(-1)
            pm = o.sum(-1) / n.clamp(min=1)
            t = torch.as_tensor(la[g][:, ib] - la[g][:, ia])
            mt = torch.as_tensor(np.isfinite(agb[g][:, ia]) & np.isfinite(agb[g][:, ib])) & (n >= P // 2)
            loss = ((pm - t) ** 2)[mt].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def stats(t, p, kind=None):
    """t, p: true and predicted change in Mg/ha; plus relative versions from the before values."""
    out = {}
    if t["d"].size < 50:
        return out
    tt, pp, base = t["d"], p, t["a"]
    c = np.cov(tt, pp)
    out["slope"] = round(float(c[0, 1] / c[0, 0]), 3)
    out["corr"] = round(float(np.corrcoef(tt, pp)[0, 1]), 3)
    rel = tt / np.maximum(base, 1)
    prel = pp / np.maximum(base, 1)
    loss = (rel <= -0.5) & (tt <= -20)
    gain = (rel >= 0.5) & (tt >= 20)
    steady = (np.abs(rel) < 0.1) & (np.abs(tt) < 10)
    if loss.sum() >= 20:
        out["loss_n"] = int(loss.sum())
        out["loss_ratio"] = round(float(np.median(pp[loss]) / np.median(tt[loss])), 3)
        out["loss_found"] = round(float(((prel[loss] <= -0.25) & (pp[loss] <= -10)).mean()), 3)
    if gain.sum() >= 20:
        out["gain_n"] = int(gain.sum())
        out["gain_ratio"] = round(float(np.median(pp[gain]) / np.median(tt[gain])), 3)
        out["gain_found"] = round(float(((prel[gain] >= 0.25) & (pp[gain] >= 10)).mean()), 3)
    if steady.sum() >= 20:
        out["steady_n"] = int(steady.sum())
        out["steady_lit"] = round(float(((np.abs(prel[steady]) >= 0.25) & (np.abs(pp[steady]) >= 10)).mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["aef", "tes", "olmo", "all"])
    ap.add_argument("--pair-epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N squares only (smoke test)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    R = dict(np.load(HERE / "data" / "rows.npz"))
    names, kinds = list(R["names"]), list(R["kinds"])
    fold_of = {}
    for fname, test in bh.folds(R, "square"):
        for s in np.unique(R["sq"][test]):
            fold_of[names[s]] = fname
    D = np.load(HERE / "models" / "detectors.npz")
    W = {inp: np.load(HERE / "data" / "heads" / f"{inp}_square.npz") for inp in args.inputs}

    # pair heads on the rows, one per fold and input
    pairs = {}
    for inp in args.inputs:
        X, ok = bh.build(R, inp)
        for fname, test in bh.folds(R, "square"):
            train = np.setdiff1d(np.arange(len(R["agb"])), test)
            pairs[(inp, fname)] = fit_pair(X, ok, R["agb"], train, args.pair_epochs)
        print(f"pair heads {inp}: {time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB", flush=True)
        del X, ok
    del R

    cells = {inp: {m: [] for m in ("plain", "gated", "pair")} for inp in args.inputs}
    truth, cell_kind, cell_sq = [], [], []
    px = {inp: {"loss_before": [], "loss_after": [], "forest_before": [], "forest_after": [],
                "ct_loss_before": [], "ct_loss_after": []} for inp in args.inputs}
    ia = [YEARS.index(a) for a, _ in WINDOWS]
    ib = [YEARS.index(b) for _, b in WINDOWS]
    for si, name in enumerate(names):
        if args.limit and si >= args.limit:
            break
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        L = np.load(HERE / "data" / "labels" / f"{name}.npz")
        F = square_feats(z)
        cell = L["cell"]
        ids, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
        keep = cnt >= 40
        agb = L["agb"].astype(np.float32)  # (Y, n) x10
        first = np.zeros(ids.size, int)
        first[inv[::-1]] = np.arange(cell.size)[::-1]
        ct = np.where(agb[:, first] >= 0, agb[:, first] / 10, np.nan).T[keep]  # (C, Y)
        truth.append(ct)
        cell_kind += [kinds[si]] * int(keep.sum())
        cell_sq += [si] * int(keep.sum())
        moved = {}
        for k in EMB:
            E, okk = F[k]
            U = [unit(E[y].astype(np.float32)) for y in range(len(YEARS))]
            mv = np.stack([1 - (U[a] * U[b]).sum(-1) >= float(D[f"{k}_cut"]) for a, b in zip(ia, ib)])
            moved[k] = mv & np.stack([okk[a] & okk[b] for a, b in zip(ia, ib)])
            del U
        moved["all"] = (moved["aef"].astype(int) + moved["tes"] + moved["olmo"]) >= 2
        fold = fold_of[name]

        def cellmean(v, valid):  # v (K, n) -> (C, K)
            outm = np.full((ids.size, v.shape[0]), np.nan)
            for k in range(v.shape[0]):
                s = np.bincount(inv, np.where(valid[k], v[k], 0), ids.size)
                c = np.bincount(inv, valid[k], ids.size)
                outm[:, k] = np.where(c >= 20, s / np.maximum(c, 1), np.nan)
            return outm[keep]

        loss = z["loss"]
        cover = z["cover"]
        for inp in args.inputs:
            if inp == "all":
                ok = F["aef"][1] & F["tes"][1] & F["olmo"][1]

                def xs(y):
                    return torch.as_tensor(np.concatenate([F[k][0][y] for k in EMB], -1).astype(np.float32))
                d = sum(F[k][0].shape[-1] for k in EMB)
            else:
                ok = F[inp][1]

                def xs(y, _e=F[inp][0]):
                    return torch.as_tensor(_e[y].astype(np.float32))
                d = F[inp][0].shape[-1]
            net = head_from(W[inp], fold, d)
            with torch.no_grad():
                pa = np.stack([np.clip(np.expm1(np.minimum(net(xs(y)).numpy()[:, 0], 8)), 0, None)
                               for y in range(len(YEARS))])  # (Y, n)
                pnet = pairs[(inp, fold)]
                pdl = np.stack([pnet(xs(a), xs(b)).numpy() for a, b in zip(ia, ib)])
            state = cellmean(pa, ok)  # (C, Y)
            cells[inp]["plain"].append(state)
            dpix = np.stack([pa[b] - pa[a] for a, b in zip(ia, ib)])
            okw = np.stack([ok[a] & ok[b] for a, b in zip(ia, ib)])
            cells[inp]["gated"].append(cellmean(np.where(moved[inp], dpix, 0.0), okw))
            cells[inp]["pair"].append(cellmean(pdl, okw))  # change in log1p, converted below
            if kinds[si] == "forest-loss":
                never = (loss == 0) & (cover >= 30)
                for ly in range(19, 25):
                    m = (loss == ly) & ok[YEARS.index(2000 + ly - 1)] & ok[min(YEARS.index(2000 + ly + 1), len(YEARS) - 1)]
                    if m.sum() < 50:
                        continue
                    a, b = YEARS.index(2000 + ly - 1), min(YEARS.index(2000 + ly + 1), len(YEARS) - 1)
                    px[inp]["loss_before"].append(float(np.median(pa[a][m])))
                    px[inp]["loss_after"].append(float(np.median(pa[b][m])))
                    fm = never & ok[a] & ok[b]
                    px[inp]["forest_before"].append(float(np.median(pa[a][fm])) if fm.sum() > 50 else np.nan)
                    px[inp]["forest_after"].append(float(np.median(pa[b][fm])) if fm.sum() > 50 else np.nan)
                    px[inp]["ct_loss_before"].append(float(np.median(agb[a][m & (agb[a] >= 0)]) / 10))
                    px[inp]["ct_loss_after"].append(float(np.median(agb[b][m & (agb[b] >= 0)]) / 10))
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"{si + 1}/{len(names)} {name} [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")
        del z, L, F

    T = np.concatenate(truth)  # (C, Y)
    ck = np.array(cell_kind)
    cs = np.array(cell_sq)
    cf = np.array([fold_of[names[s]] for s in cs])
    ta, tb = T[:, ia], T[:, ib]
    TD = tb - ta  # (C, W)
    res = {"cells": int(len(T)), "windows": [f"{a}-{b}" for a, b in WINDOWS]}
    # CTrees' own year-to-year noise on cells it calls steady over the whole span
    res["ctrees"] = {"median_abs_yearly_change": round(float(np.nanmedian(np.abs(TD[:, :7]))), 1)}
    for inp in args.inputs:
        S = np.concatenate(cells[inp]["plain"])
        G = np.concatenate(cells[inp]["gated"])
        Pl = np.concatenate(cells[inp]["pair"])
        PD = {"plain": S[:, ib] - S[:, ia], "gated": G}
        # pair: predicted change in log1p from the true before value, back to Mg/ha
        PD["pair"] = np.expm1(np.log1p(np.clip(ta, 0, None)) + Pl) - np.clip(ta, 0, None)
        # calibrated: per fold, undo the muting measured on the other folds' cells: if predicted change runs at a
        # slope s of true change there, divide by s (variance matching; restores size, also scales up noise)
        cal = np.full_like(PD["plain"], np.nan)
        for f in np.unique(cf):
            tr = (cf != f)[:, None] & np.isfinite(TD) & np.isfinite(PD["plain"])
            x, y = PD["plain"][tr], TD[tr]
            b = np.var(y) / np.cov(x, y)[0, 1]
            a = y.mean() - b * x.mean()
            cal[cf == f] = a + b * PD["plain"][cf == f]
        PD["calibrated"] = cal
        res[inp] = {}
        for meth, P in PD.items():
            res[inp][meth] = {}
            for wname, wsel in (("yearly", slice(0, 7)), ("2018-2025", slice(7, 8))):
                for gname in ["all", "other"] + sorted(set(ck)):
                    gm = np.ones(len(ck), bool) if gname == "all" else (ck != "forest-loss") if gname == "other" else (ck == gname)
                    t = TD[gm][:, wsel].ravel()
                    p = P[gm][:, wsel].ravel()
                    a = ta[gm][:, wsel].ravel()
                    m = np.isfinite(t) & np.isfinite(p) & np.isfinite(a)
                    res[inp][meth][f"{wname}/{gname}"] = stats({"d": t[m], "a": a[m]}, p[m])
        np.savez(HERE / "data" / "heads" / f"change_cells_{inp}.npz", truth=T, kind=ck, sq=cs,
                 **{m: P.astype(np.float32) for m, P in PD.items()})
        res[inp]["pixels_10m"] = {k: round(float(np.nanmedian(v)), 1) if v else None for k, v in px[inp].items()}
        res[inp]["pixels_10m"]["events"] = len(px[inp]["loss_before"])
        for meth in PD:
            r = res[inp][meth]
            print(f"{inp:5s} {meth:10s} yearly slope {r['yearly/all'].get('slope')} corr {r['yearly/all'].get('corr')} "
                  f"loss {r['yearly/all'].get('loss_ratio')}/{r['yearly/all'].get('loss_found')} "
                  f"steady lit {r['yearly/all'].get('steady_lit')} | 2018-25 slope {r['2018-2025/all'].get('slope')} "
                  f"corr {r['2018-2025/all'].get('corr')} loss {r['2018-2025/all'].get('loss_ratio')}/{r['2018-2025/all'].get('loss_found')} "
                  f"gain {r['2018-2025/all'].get('gain_ratio')}/{r['2018-2025/all'].get('gain_found')}", flush=True)
        print(f"{inp:5s} 10 m at Hansen loss: {res[inp]['pixels_10m']}", flush=True)
    RES.parent.mkdir(exist_ok=True)
    json.dump(res, open(RES, "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
