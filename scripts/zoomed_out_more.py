# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Room to improve in zoomed_out.py's networks: more pixels, a bigger network, and fill-in taught through the map.

zoomed_out.py taught copy, fill and yearfill on 1 M pixel-years from 4 patches per training square. They need no
labels, so they can learn from any AlphaEarth pixels. Here:
  pool     60,000 pixel-years from anywhere in each square (and a second year at the same pixel), from the 92 squares
           and the 40 others on disk (lidar, fire and range squares, all 148 km or more from any of the 92); in each
           fold the held-out squares are left out, so about 7 M pixel-years
  base     zoomed_out.py's networks (256 x 2, 4 epochs, 1 M from the patches), run again here for the comparison
  more     the pool, same network
  big      the pool, 512 x 3, 6 epochs
  fill+    big fill whose loss also asks that the map's head give the same answers on the rebuilt vector
  yearfill+ the same for yearfill

Scored as zoomed_out.py scores (the same patches of the held-out squares, against each fold's 64-band head).
Networks train on the GPU (MPS) when there is one. Writes results/zoomed_out_more.json.
RAM: pool 1 GB int8, patches 0.8 GB, rows 0.6 GB, teacher outputs 0.35 GB; peak ~5 GB, stops over --max-gb.

    uv run scripts/zoomed_out_more.py [--ks 4 8 16 32] [--folds 0 1 ...] [--per-sq 60000]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import band_subset as bsub  # noqa: E402
import biomass_head as bh  # noqa: E402
import zoomed_out as zo  # noqa: E402

RES = HERE / "results" / "zoomed_out_more.json"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LEVELS = zo.LEVELS
NP, M, IN, PATCH = zo.NP, zo.M, zo.IN, zo.PATCH


def deq_t(q):
    """int8 tensor -> dequantized float, nodata 0."""
    f = q.float() / 127.5
    f = torch.sign(f) * f * f
    return torch.where((q[..., :1] == -128), torch.zeros_like(f), f)


def read_pool(names, per_sq, seed=0):
    """A and B int8 (n, 64) at the same pixels in two different years, and the square of each."""
    A, B, S = [], [], []
    for s, n in enumerate(names):
        a = np.load(HERE / "data" / "squares" / f"{n}.npz")["aef"]  # (8, 360000, 64)
        rng = np.random.default_rng(5000 + s)
        px = rng.integers(0, a.shape[1], per_sq)
        y = rng.integers(0, 8, per_sq)
        yb = (y + rng.integers(1, 8, per_sq)) % 8
        qa, qb = a[y, px], a[yb, px]
        k = qa[:, 0] != -128
        A.append(qa[k])
        B.append(qb[k])
        S.append(np.full(k.sum(), s, np.int16))
        del a
    return np.concatenate(A), np.concatenate(B), np.concatenate(S)


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, mu, sd, hid, layers):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        mods, d = [], d_in
        for _ in range(layers):
            mods += [torch.nn.Linear(d, hid), torch.nn.ReLU()]
            d = hid
        self.f = torch.nn.Sequential(*mods, torch.nn.Linear(d, d_out))

    def forward(self, x):
        return self.f((x - self.mu) / self.sd)


def distill(o, t):
    kl = torch.nn.functional.kl_div(torch.log_softmax(o[:, 2:], -1), torch.softmax(t[:, 2:], -1), reduction="batchmean")
    return torch.nn.functional.mse_loss(o[:, 0], t[:, 0]) + torch.nn.functional.mse_loss(o[:, 1], t[:, 1]) + 0.5 * kl


def train(kind, A, B, TA, TB, bands, miss, teacher, hid, layers, epochs, seed, bs=8192):
    """kind: copy (k bands of A -> teacher(A)), fill / fill+ (k bands of A -> A's other bands), yearfill / yearfill+
    (A whole and B's k bands -> B's other bands minus A's). A, B int8 numpy; TA, TB teacher outputs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(A)
    bt, mt = torch.as_tensor(bands, device=DEV), torch.as_tensor(miss, device=DEV)

    def batch(i):
        a = deq_t(torch.as_tensor(A[i], device=DEV))
        b = deq_t(torch.as_tensor(B[i], device=DEV)) if kind.startswith("year") else None
        if kind == "copy":
            return a[:, bt], torch.as_tensor(TA[i], device=DEV).float(), a, b
        if kind.startswith("fill"):
            return a[:, bt], a[:, mt], a, b
        return torch.cat([a, b[:, bt]], 1), b[:, mt] - a[:, mt], a, b

    i0 = np.sort(rng.choice(n, min(n, 200000), replace=False))
    x0, y0, _, _ = batch(i0)
    mu, sd = x0.mean(0), x0.std(0) + 1e-4
    ym, ys = (torch.zeros(y0.shape[1], device=DEV), torch.ones(y0.shape[1], device=DEV)) if kind == "copy" else (
        y0.mean(0), y0.std(0) + 1e-6)
    net = MLP(x0.shape[1], y0.shape[1], mu.cpu(), sd.cpu(), hid, layers).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    through = kind.endswith("+")
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, bs):
            i = np.sort(perm[b0:b0 + bs])
            x, y, a, b = batch(i)
            o = net(x)
            if kind == "copy":
                loss = distill(o, y)
            else:
                loss = torch.nn.functional.mse_loss(o, (y - ym) / ys)
                if through:
                    base = a if kind.startswith("fill") else b
                    r = base.clone()
                    r[:, mt] = (a[:, mt] if kind.startswith("year") else 0) + o * ys + ym
                    r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    t = torch.as_tensor((TA if kind.startswith("fill") else TB)[i], device=DEV).float()
                    loss = loss + distill(teacher(r), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net, ym, ys


def teach(tdev, Q, step=1 << 20):
    """The teacher's outputs on int8 rows, a million at a time (no full float copy)."""
    return np.concatenate([run(tdev, zo.dq(Q[i:i + step])).astype(np.float16) for i in range(0, len(Q), step)])


@torch.no_grad()
def run(net, x, bs=1 << 17):
    sh = x.shape
    x = torch.as_tensor(np.ascontiguousarray(x.reshape(-1, sh[-1]), np.float32))
    out = torch.cat([net(x[i:i + bs].to(DEV)).cpu() for i in range(0, len(x), bs)]).numpy()
    return out.reshape(*sh[:-1], out.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(8)))
    ap.add_argument("--per-sq", type=int, default=60000)
    ap.add_argument("--base-ks", nargs="*", type=int, default=[16], help="ks to also run base and more at")
    ap.add_argument("--max-gb", type=float, default=12)
    ap.add_argument("--out", default=str(RES))
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    R, common = bsub.load()
    H = bh.targets(R)
    X, ok = bh.features(R, "aef")
    G = X.shape[0]
    picks = json.load(open(HERE / "results" / "band_subset.json"))
    names = [str(n) for n in R["names"]]
    extra = sorted(f[:-4] for f in os.listdir(HERE / "data" / "squares") if f.endswith(".npz") and f[:-4] not in names)
    sqf = zo.square_folds(R)
    PA = zo.read_patches(names)
    PoA, PoB, PoS = read_pool(names + extra, args.per_sq)
    print(f"patches {PA.shape}, pool {len(PoA):,} from {len(names) + len(extra)} squares, {zo.MB():.1f} GB, "
          f"{time.time() - t0:.0f} s, device {DEV}", flush=True)
    pool = zo.Pool()
    configs = []  # (name, kind, hid, layers, epochs, source)
    for k in args.ks:
        if k in args.base_ks:
            configs += [(f"{m}_base_{k}", m, 256, 2, 4, "patches", k) for m in ("copy", "fill", "yearfill")]
            configs += [(f"{m}_more_{k}", m, 256, 2, 4, "pool", k) for m in ("copy", "fill", "yearfill")]
        configs += [(f"{m}_big_{k}", m.rstrip("+"), 512, 3, 6, "pool", k) if not m.endswith("+") else
                    (f"{m}_big_{k}", m, 512, 3, 6, "pool", k) for m in ("copy", "fill", "fill+", "yearfill", "yearfill+")]

    for fold in args.folds:
        tf = time.time()
        test_cells = np.flatnonzero(sqf[R["sq"]] == fold)
        train_cells = np.setdiff1d(np.arange(G), test_cells)
        tr_sq, te_sq = np.flatnonzero(sqf != fold), np.flatnonzero(sqf == fold)
        teacher = bh.fit(X, ok, R, H, train_cells, 8, seed=0)
        tdev = bh.Head(64, teacher.mu.numpy(), teacher.sd.numpy()).to(DEV)
        tdev.load_state_dict(teacher.state_dict())
        tdev.eval()
        for p_ in tdev.parameters():
            p_.requires_grad_(False)
        # the pool without the held-out squares, with the teacher's answers
        keep = ~np.isin(PoS, te_sq)
        A, B = PoA[keep], PoB[keep]
        TA = teach(tdev, A)
        TB = teach(tdev, B)
        # zoomed_out.py's training set: 1 M pixel-years from the training squares' patches (same draw)
        rng = np.random.default_rng(fold)
        n = 1_000_000
        s_, p_, y_ = rng.choice(tr_sq, n), rng.integers(0, NP, n), rng.integers(0, 8, n)
        i_, j_ = rng.integers(M, PATCH - M, n), rng.integers(M, PATCH - M, n)
        cq = PA[s_, p_, y_, i_, j_]
        kk = cq[:, 0] != -128
        s_, p_, y_, i_, j_, cq = s_[kk], p_[kk], y_[kk], i_[kk], j_[kk], cq[kk]
        yb = (y_ + rng.integers(1, 8, len(y_))) % 8
        bq = PA[s_, p_, yb, i_, j_]
        pA, pB = cq, bq
        pTA = teach(tdev, pA)
        pTB = teach(tdev, pB)

        for name, kind, hid, layers, epochs, src, k in configs:
            tk = time.time()
            bands = picks[f"greedy_{k}"]["bands"][str(fold)]
            miss = [b for b in range(64) if b not in bands]
            if src == "patches":
                a_, b_, ta_, tb_ = pA, pB, pTA, pTB
            else:
                a_, b_, ta_, tb_ = A, B, TA, TB
            if kind.startswith("year"):
                ok2 = b_[:, 0] != -128
                a_, b_, ta_, tb_ = a_[ok2], b_[ok2], ta_[ok2], tb_[ok2]
            net, ym, ys = train(kind, a_, b_, ta_, tb_, bands, miss, tdev, hid, layers, epochs, fold)
            ym, ys = ym.cpu().numpy(), ys.cpu().numpy()

            # score on the held-out squares' patches
            for s in te_sq:
                for p in range(NP):
                    q = PA[s, p]
                    f = zo.dq(q)
                    v = (q[..., 0] != -128)[:, M:-M, M:-M]
                    fi = f[:, M:-M, M:-M]
                    mp = run(tdev, fi)
                    out_map = zo.outputs(mp)
                    if kind == "copy" or kind.startswith("fill"):
                        if kind == "copy":
                            o = run(net, fi[..., bands])
                        else:
                            r = fi.copy()
                            r[..., miss] = run(net, fi[..., bands]) * ys + ym
                            o = run(tdev, zo.unit(r))
                        out = zo.outputs(o)
                        for lv in LEVELS:
                            pool.add(("agb_t", name, lv), zo.blocks(out_map[0], v, lv))
                            pool.add(("agb_p", name, lv), zo.blocks(out[0], v, lv))
                            pool.add(("h_t", name, lv), zo.blocks(out_map[1], v, lv))
                            pool.add(("h_p", name, lv), zo.blocks(out[1], v, lv))
                            pool.add(("c_t", name, lv), zo.blocks(out_map[2], v, lv, "mode"))
                            pool.add(("c_p", name, lv), zo.blocks(out[2], v, lv, "mode"))
                    for a, b in zo.PAIRS:
                        vb = v[a] & v[b]
                        if vb.sum() < 100:
                            continue
                        dmap = out_map[0][b] - out_map[0][a]
                        if kind.startswith("year"):
                            Bh = fi[b].copy()
                            Bh[..., miss] = fi[a][..., miss] + run(net, np.concatenate([fi[a], fi[b][..., bands]], -1)) * ys + ym
                            d = zo.outputs(run(tdev, zo.unit(Bh)))[0] - out_map[0][a]
                        else:
                            d = out[0][b] - out[0][a]
                        for lv in LEVELS:
                            pool.add(("d_t", name, lv), zo.blocks(dmap[None], vb[None], lv))
                            pool.add(("d_p", name, lv), zo.blocks(d[None], vb[None], lv))
            if zo.MB() > args.max_gb:
                raise SystemExit(f"STOP: RSS {zo.MB():.1f} GB")
            print(f"  fold {fold} {name}: {time.time() - tk:.0f} s, {zo.MB():.1f} GB", flush=True)
        del A, B, TA, TB, pA, pB, pTA, pTB
        print(f"fold {fold} done {time.time() - tf:.0f} s", flush=True)

    out = {"folds": args.folds, "per_sq": args.per_sq, "maps": {}, "change": {}}
    for name, kind, *_ in configs:
        if not kind.startswith("year"):
            row = {}
            for lv in LEVELS:
                at, ap_ = pool.get(("agb_t", name, lv)), pool.get(("agb_p", name, lv))
                ht, hp = pool.get(("h_t", name, lv)), pool.get(("h_p", name, lv))
                ct, cp_ = pool.get(("c_t", name, lv)), pool.get(("c_p", name, lv))
                row[f"{10 * lv}m"] = {"agb_r2_log": round(zo.r2(np.log1p(at), np.log1p(ap_)), 3),
                                      "agb_mae": round(float(np.median(np.abs(at - ap_))), 1),
                                      "h_mae": round(float(np.median(np.abs(ht - hp))), 2),
                                      "cls_agree": round(float((ct == cp_).mean()), 3)}
            out["maps"][name] = row
        row = {}
        for lv in LEVELS:
            dt, dp = pool.get(("d_t", name, lv)), pool.get(("d_p", name, lv))
            big_l, big_g, still = dt < -20, dt > 20, np.abs(dt) < 5
            sub = np.random.default_rng(0).choice(dt.size, min(dt.size, 300000), replace=False)
            row[f"{10 * lv}m"] = {"d_rank": round(float(np.corrcoef(zo.rank(dt[sub]), zo.rank(dp[sub]))[0, 1]), 3),
                                  "loss_seen": round(float((dp[big_l] < -10).mean()), 3),
                                  "gain_seen": round(float((dp[big_g] > 10).mean()), 3),
                                  "false_loss": round(float((dp[still] < -10).mean()), 4),
                                  "false_gain": round(float((dp[still] > 10).mean()), 4)}
        out["change"][name] = row
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, {zo.MB():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
