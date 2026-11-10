# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""A zoomed-out map from k of AlphaEarth's 64 bands: four ways, scored against the map's own head on all 64.

AEF's GeoTIFFs keep each band in its own blocks, so reading k bands costs k/64 of the download. Zoomed out the map
needs a rough picture, not the 10 m answer, so each way is scored per pixel and over blocks of 6, 12 and 30 pixels
(60, 120, 300 m: about one screen pixel at zoom 10, 9 and 8).

  plain    the k-band head of band_subset.py (taught on CTrees, Meta and the classes, like the map's head)
  copy     a small network taught to give the map head's answers from k bands, on every pixel of the training squares
  fill     a network that rebuilds the other 64 - k bands from the k read, then the map's head on the rebuilt vector
  ctx      copy, also given the same k bands of the 5 x 5 pixels around (neighbors come in the same block, no cost)
  yearfill (change view) year A read whole and year B on k bands; how B's other bands differ from A's, rebuilt from
           A and B's k bands (no difference is copyA)
  copyA    (change view) the same reads, B's other bands taken from A as they are

Teacher: in each of the 8 square folds, a 64-band head fit on the training squares' rows (biomass_head.py's fit);
every network here is taught on training squares only and scored on the fold's held-out squares. Bands: each fold's
greedy picks from results/band_subset.json. Pixels: 4 patches of 64 x 64 per square (one per quarter), 8 years, from
data/squares/*.npz, scored on the inner 60 x 60.
The CTrees / Meta / class scores of plain, copy and fill on the rows are kept beside them (rows have no neighbors).

Writes results/zoomed_out.json. RAM: patches 0.8 GB int8, rows 0.6 GB, training sets up to 1.6 GB float16; peak
~5 GB, stops over --max-gb.

    uv run scripts/zoomed_out.py [--ks 8 16 32] [--folds 0 1 ...] [--n 1000000] [--epochs 4]
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
import band_subset as bsub  # noqa: E402
import biomass_head as bh  # noqa: E402

RES = HERE / "results" / "zoomed_out.json"
PATCH, NP, M = 64, 4, 2  # patch side, patches per square, context margin (5 x 5)
IN = PATCH - 2 * M  # 60, the scored inner side
LEVELS = [1, 6, 12, 30]
PAIRS = [(y, y + 1) for y in range(7)] + [(0, 7)]
MB = lambda: psutil.Process().memory_info().rss / 1e9  # noqa: E731


def square_folds(R, k=8):
    """The fold of each square, as biomass_head.folds makes them."""
    kinds = R["kinds"]
    rng = np.random.default_rng(1)
    order = []
    for kd in sorted(set(kinds)):
        order += list(rng.permutation(np.flatnonzero(kinds == kd)))
    fold_of = {s: i % k for i, s in enumerate(order)}
    return np.array([fold_of[s] for s in range(len(kinds))])


def read_patches(names):
    out = np.empty((len(names), NP, 8, PATCH, PATCH, 64), np.int8)
    for s, n in enumerate(names):
        a = np.load(HERE / "data" / "squares" / f"{n}.npz")["aef"].reshape(8, 600, 600, 64)
        rng = np.random.default_rng(1000 + s)
        for p, (qy, qx) in enumerate([(0, 0), (0, 300), (300, 0), (300, 300)]):
            y, x = qy + rng.integers(0, 300 - PATCH + 1), qx + rng.integers(0, 300 - PATCH + 1)
            out[s, p] = a[:, y:y + PATCH, x:x + PATCH]
        del a
    return out


def dq(q):
    """Dequantized, with nodata pixels set to 0."""
    f = bh.deq(q)
    f[q[..., 0] == -128] = 0
    return f


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, mu, sd, hid=256):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.f = torch.nn.Sequential(torch.nn.Linear(d_in, hid), torch.nn.ReLU(), torch.nn.Linear(hid, hid),
                                     torch.nn.ReLU(), torch.nn.Linear(hid, d_out))

    def forward(self, x):
        return self.f((x - self.mu) / self.sd)


@torch.no_grad()
def apply(net, x, bs=1 << 18):
    sh = x.shape
    x = x.reshape(-1, sh[-1])
    out = [net(torch.as_tensor(np.ascontiguousarray(x[i:i + bs], np.float32))).numpy() for i in range(0, len(x), bs)]
    o = np.concatenate(out)
    return o.reshape(*sh[:-1], o.shape[-1])


def distill_loss(o, t):
    kl = torch.nn.functional.kl_div(torch.log_softmax(o[:, 2:], -1), torch.softmax(t[:, 2:], -1), reduction="batchmean")
    return torch.nn.functional.mse_loss(o[:, 0], t[:, 0]) + torch.nn.functional.mse_loss(o[:, 1], t[:, 1]) + 0.5 * kl


def train(X, Y, loss, epochs, seed, standardize_y=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s = X[:200000].astype(np.float32)
    net = MLP(X.shape[1], Y.shape[1], s.mean(0), s.std(0) + 1e-4)
    ym, ys = (Y.mean(0), Y.std(0) + 1e-6) if standardize_y else (np.zeros(Y.shape[1]), np.ones(Y.shape[1]))
    Yt = ((Y - ym) / ys).astype(np.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 4096
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for b in range(0, len(X), bs):
            i = np.sort(perm[b:b + bs])
            o = net(torch.as_tensor(X[i].astype(np.float32)))
            l = loss(o, torch.as_tensor(Yt[i]))
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
    net.eval()
    return net, ym.astype(np.float32), ys.astype(np.float32)


def outputs(o):
    """agb Mg/ha, height m, class from the 11 raw outputs."""
    return np.clip(np.expm1(np.minimum(o[..., 0], 8)), 0, None), np.expm1(o[..., 1]), o[..., 2:].argmax(-1)


def blocks(x, v, b, how="mean"):
    """(8, IN, IN) values over b x b blocks, blocks under half valid dropped -> flat values; b == 1 keeps every 6th
    pixel each way. how: mean, or mode for classes."""
    if b == 1:
        sl = (slice(None), slice(0, IN, 6), slice(0, IN, 6))
        return x[sl][v[sl]]
    Y, n = x.shape[0], IN // b
    vv = v.reshape(Y, n, b, n, b)
    cnt = vv.sum((2, 4))
    keep = cnt >= b * b / 2
    if how == "mode":
        oh = (x[..., None] == np.arange(bh.NCLS)).reshape(Y, n, b, n, b, bh.NCLS) & vv[..., None]
        return oh.sum((2, 4)).argmax(-1)[keep]
    s = np.where(vv, x.reshape(Y, n, b, n, b), 0).sum((2, 4))
    return (s / np.maximum(cnt, 1))[keep]


class Pool:
    """Pooled values over folds: key -> list of arrays."""

    def __init__(self):
        self.d = {}

    def add(self, key, a):
        self.d.setdefault(key, []).append(np.asarray(a, np.float32))

    def get(self, key):
        return np.concatenate(self.d[key]) if key in self.d else np.zeros(0, np.float32)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


def rank(x):
    return np.argsort(np.argsort(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(8)))
    ap.add_argument("--n", type=int, default=1_000_000, help="training pixel-years per network")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=12)
    ap.add_argument("--out", default=str(RES))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    R, common = bsub.load()
    H = bh.targets(R)
    X, ok = bh.features(R, "aef")
    picks = json.load(open(HERE / "results" / "band_subset.json"))
    names = [str(n) for n in R["names"]]
    sqf = square_folds(R)
    PA = read_patches(names)
    print(f"patches {PA.shape} read, {MB():.1f} GB, {time.time() - t0:.0f} s", flush=True)
    G, Y, P, _ = X.shape
    pool = Pool()
    # rows-based label scores: (method, k) -> pa, ph, pc over all cells
    lab = {}

    def lab_arrays(key):
        if key not in lab:
            lab[key] = [np.full((G, Y), np.nan), np.zeros((G, Y, P), np.float32), np.zeros((G, Y, P), np.int8)]
        return lab[key]

    def rows_predict(key, fn, test):
        pa, ph, pc = lab_arrays(key)
        for b in range(0, test.size, 1024):
            g = test[b:b + 1024]
            o = fn(X[g].astype(np.float32))
            v = ok[g]
            a = np.clip(np.expm1(np.minimum(o[..., 0], 8)), 0, None) * v
            n = v.sum(-1)
            pa[g] = np.where(n >= P // 2, a.sum(-1) / np.maximum(n, 1), np.nan)
            ph[g] = np.expm1(o[..., 1])
            pc[g] = o[..., 2:].argmax(-1)

    for fold in args.folds:
        tf = time.time()
        test_cells = np.flatnonzero(sqf[R["sq"]] == fold)
        train_cells = np.setdiff1d(np.arange(G), test_cells)
        tr_sq, te_sq = np.flatnonzero(sqf != fold), np.flatnonzero(sqf == fold)
        teacher = bh.fit(X, ok, R, H, train_cells, 8, seed=0)
        rows_predict(("teacher", 64), lambda x: apply(teacher, x), test_cells)

        # one training sample of pixel-years (and year pairs) from the training squares, shared by every k
        rng = np.random.default_rng(fold)
        n = args.n
        s_, p_ = rng.choice(tr_sq, n), rng.integers(0, NP, n)
        y_, i_, j_ = rng.integers(0, 8, n), rng.integers(M, PATCH - M, n), rng.integers(M, PATCH - M, n)
        cq = PA[s_, p_, y_, i_, j_]
        keep = cq[:, 0] != -128
        s_, p_, y_, i_, j_, cq = s_[keep], p_[keep], y_[keep], i_[keep], j_[keep], cq[keep]
        cf = dq(cq)
        tout = apply(teacher, cf)
        yb = (y_ + rng.integers(1, 8, len(y_))) % 8  # a second year for the pairs
        bq = PA[s_, p_, yb, i_, j_]
        pk = bq[:, 0] != -128
        bf = dq(bq)
        del cq, bq

        for k in args.ks:
            tk = time.time()
            bands = picks[f"greedy_{k}"]["bands"][str(fold)]
            miss = [b for b in range(64) if b not in bands]
            nets = {}
            # plain: the k-band head on the labels
            Xb = np.ascontiguousarray(X[..., bands])
            plain = bh.fit(Xb, ok, R, H, train_cells, 8, seed=0)
            del Xb
            rows_predict(("plain", k), lambda x: apply(plain, x[..., bands]), test_cells)
            # copy
            cp, _, _ = train(cf[:, bands].astype(np.float16), tout, distill_loss, args.epochs, fold)
            rows_predict(("copy", k), lambda x: apply(cp, x[..., bands]), test_cells)
            # fill
            fl, fm, fs = train(cf[:, bands].astype(np.float16), cf[:, miss], torch.nn.functional.mse_loss, args.epochs,
                               fold, standardize_y=True)

            def rebuild(x, fl=fl, fm=fm, fs=fs):
                r = x.copy()
                r[..., miss] = apply(fl, x[..., bands]) * fs + fm
                return unit(r)

            rows_predict(("fill", k), lambda x: apply(teacher, rebuild(x)), test_cells)
            # ctx: the k bands of the 5 x 5 around, in (di, dj, band) order
            ctx = np.empty((len(s_), 25 * k), np.float16)
            for o, (di, dj) in enumerate([(a, b) for a in range(-M, M + 1) for b in range(-M, M + 1)]):
                ctx[:, o * k:(o + 1) * k] = dq(PA[s_, p_, y_, i_ + di, j_ + dj])[:, bands]
            cx, _, _ = train(ctx, tout, distill_loss, args.epochs, fold)
            del ctx
            # yearfill: A whole and B's k bands -> how B's other bands differ from A's (0 is copyA)
            yin = np.concatenate([cf[pk], bf[pk][:, bands]], 1).astype(np.float16)
            yf, ym, ys = train(yin, bf[pk][:, miss] - cf[pk][:, miss], torch.nn.functional.mse_loss, args.epochs, fold,
                               standardize_y=True)
            del yin
            if MB() > args.max_gb:
                raise SystemExit(f"STOP: RSS {MB():.1f} GB")

            # score on the held-out squares' patches
            for s in te_sq:
                for p in range(NP):
                    q = PA[s, p]
                    f = dq(q)
                    v = (q[..., 0] != -128)[:, M:-M, M:-M]
                    fi = f[:, M:-M, M:-M]
                    fb = fi[..., bands]
                    res = {"map": apply(teacher, fi), "plain": apply(plain, fb), "copy": apply(cp, fb),
                           "fill": apply(teacher, rebuild(fi))}
                    win = np.lib.stride_tricks.sliding_window_view(f[..., bands], (5, 5), axis=(1, 2))  # (8,60,60,k,5,5)
                    res["ctx"] = apply(cx, win.transpose(0, 1, 2, 4, 5, 3).reshape(8, IN, IN, 25 * k))
                    out = {m: outputs(o) for m, o in res.items()}
                    for m in ("plain", "copy", "fill", "ctx"):
                        for lv in LEVELS:
                            pool.add(("agb_t", m, k, lv), blocks(out["map"][0], v, lv))
                            pool.add(("agb_p", m, k, lv), blocks(out[m][0], v, lv))
                            pool.add(("h_t", m, k, lv), blocks(out["map"][1], v, lv))
                            pool.add(("h_p", m, k, lv), blocks(out[m][1], v, lv))
                            pool.add(("c_t", m, k, lv), blocks(out["map"][2], v, lv, "mode"))
                            pool.add(("c_p", m, k, lv), blocks(out[m][2], v, lv, "mode"))
                    # the change view, year pairs
                    for a, b in PAIRS:
                        vb = v[a] & v[b]
                        if vb.sum() < 100:
                            continue
                        A, B = fi[a], fi[b]
                        yfB = B.copy()
                        yfB[..., miss] = A[..., miss] + apply(yf, np.concatenate([A, B[..., bands]], -1)) * ys + ym
                        yfB = unit(yfB)
                        caB = B.copy()
                        caB[..., miss] = A[..., miss]
                        caB = unit(caB)
                        agbA = out["map"][0][a]
                        dmap = out["map"][0][b] - agbA
                        ds = {m: out[m][0][b] - out[m][0][a] for m in ("plain", "copy", "fill", "ctx")}
                        ds["yearfill"] = outputs(apply(teacher, yfB))[0] - agbA
                        ds["copyA"] = outputs(apply(teacher, caB))[0] - agbA
                        for m, d in ds.items():
                            for lv in LEVELS:
                                pool.add(("d_t", m, k, lv), blocks(dmap[None], vb[None], lv))
                                pool.add(("d_p", m, k, lv), blocks(d[None], vb[None], lv))
                        cos = lambda x, y: 1 - (x * y).sum(-1) / np.maximum(  # noqa: E731
                            np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-9)
                        sl = (slice(0, IN, 2), slice(0, IN, 2))
                        vs = vb[sl]
                        pool.add(("mv_t", k), cos(A, B)[sl][vs])
                        rA, rB = rebuild(A), rebuild(B)
                        for m, mv in (("kband", cos(A[..., bands], B[..., bands])), ("yearfill", cos(A, yfB)),
                                      ("copyA", cos(A, caB)), ("fill", cos(rA, rB))):
                            pool.add(("mv", m, k), mv[sl][vs])
            print(f"  fold {fold} k {k}: {time.time() - tk:.0f} s, {MB():.1f} GB", flush=True)
        del cf, tout, bf
        print(f"fold {fold} done {time.time() - tf:.0f} s, {MB():.1f} GB", flush=True)

    # summary
    out = {"folds": args.folds, "n": args.n, "epochs": args.epochs, "levels_m": [10 * lv for lv in LEVELS], "maps": {},
           "change": {}, "move": {}, "labels": {}}
    for k in args.ks:
        for m in ("plain", "copy", "fill", "ctx"):
            row = {}
            for lv in LEVELS:
                at, ap_ = pool.get(("agb_t", m, k, lv)), pool.get(("agb_p", m, k, lv))
                ht, hp = pool.get(("h_t", m, k, lv)), pool.get(("h_p", m, k, lv))
                ct, cp_ = pool.get(("c_t", m, k, lv)), pool.get(("c_p", m, k, lv))
                row[f"{10 * lv}m"] = {"agb_r2_log": round(r2(np.log1p(at), np.log1p(ap_)), 3),
                                      "agb_mae": round(float(np.median(np.abs(at - ap_))), 1),
                                      "agb_med": round(float(np.median(at)), 1),
                                      "h_r2": round(r2(ht, hp), 3), "h_r2_log": round(r2(np.log1p(np.clip(ht, 0, None)), np.log1p(np.clip(hp, 0, None))), 3),
                                      "h_mae": round(float(np.median(np.abs(ht - hp))), 2), "cls_agree": round(float((ct == cp_).mean()), 3),
                                      "n": int(at.size)}
            out["maps"][f"{m}_{k}"] = row
        for m in ("plain", "copy", "fill", "ctx", "yearfill", "copyA"):
            row = {}
            for lv in LEVELS:
                dt, dp = pool.get(("d_t", m, k, lv)), pool.get(("d_p", m, k, lv))
                big_l, big_g, still = dt < -20, dt > 20, np.abs(dt) < 5
                sub = np.random.default_rng(0).choice(dt.size, min(dt.size, 300000), replace=False)
                row[f"{10 * lv}m"] = {"d_r2": round(r2(dt, dp), 3),
                                      "d_rank": round(float(np.corrcoef(rank(dt[sub]), rank(dp[sub]))[0, 1]), 3),
                                      "loss_seen": round(float((dp[big_l] < -10).mean()), 3) if big_l.any() else None,
                                      "gain_seen": round(float((dp[big_g] > 10).mean()), 3) if big_g.any() else None,
                                      "false_loss": round(float((dp[still] < -10).mean()), 4) if still.any() else None,
                                      "false_gain": round(float((dp[still] > 10).mean()), 4) if still.any() else None,
                                      "n_loss": int(big_l.sum()), "n_gain": int(big_g.sum())}
            out["change"][f"{m}_{k}"] = row
        mt = pool.get(("mv_t", k))
        top = mt >= np.quantile(mt, 0.99)
        sub = np.random.default_rng(0).choice(mt.size, min(mt.size, 400000), replace=False)
        for m in ("kband", "yearfill", "copyA", "fill"):
            mp = pool.get(("mv", m, k))
            out["move"][f"{m}_{k}"] = {"rank_agree": round(float(np.corrcoef(rank(mt[sub]), rank(mp[sub]))[0, 1]), 3),
                                       "top1_kept": round(float((mp[top] >= np.quantile(mp, 0.99)).mean()), 3)}
    for key, (pa, ph, pc) in lab.items():
        s = bh.score(R, H, common, pa, ph, pc)["all"] if len(args.folds) == 8 else None
        if s:
            out["labels"][f"{key[0]}_{key[1]}"] = {x: s.get(x) for x in ("agb_r2_log", "agb_mae", "height_r2", "cls_acc",
                                                                         "cls_macro_f1")}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["labels"]), flush=True)
    print(f"done {time.time() - t0:.0f} s, {MB():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
