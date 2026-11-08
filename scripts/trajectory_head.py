# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""A head on the whole 2018 to 2025 series of a pixel: biomass every year, from all eight years at once.

The state head (biomass_head.py) sees one year; differencing two of its years mutes change, and slow regrowth barely
shows (FINDINGS 9). This head sees the pixel's eight embeddings together: each year through the same layer of 64,
then mixed across neighboring years (a width-3 convolution over time, twice), then biomass per year. A missing year
enters as zeros with a flag. Taught on data/rows.npz like the state head (mean over a cell's pixels matched to CTrees
in log), plus a change term: for every pair of years, the cell's change in log biomass matched to CTrees' change.
Same 8 square folds; each square is predicted by the fold that never saw it.

Then every pixel of every square, folded to CTrees' cells exactly as biomass_change.py does (the cell tables are
checked equal), and scored with biomass_change.stats: plain difference, gated by the frozen cutoffs, and divided by
the muting on clear loss cells of the other folds (per window length, as change_calibrate.py). The state head's
numbers on the same cells come from data/heads/change_cells_<input>.npz.

Writes data/heads/traj_<input>.npz (weights), data/heads/change_cells_traj_<input>.npz, results/trajectory_head.json.
Peak RAM about 9 GB (rows, one input's features, one square's float features).

    uv run scripts/trajectory_head.py [--inputs olmo all aef] [--epochs 8] [--limit N]
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
import change_calibrate as cc  # noqa: E402

YEARS = bc.YEARS
EMB = bc.EMB
RES = HERE / "results" / "trajectory_head.json"
PAIRS = [(a, b) for a in range(len(YEARS)) for b in range(a + 1, len(YEARS))]


class Traj(torch.nn.Module):
    def __init__(self, d, mu, sd, hid=bh.HID):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(d + 1, hid)
        self.t1 = torch.nn.Conv1d(hid, hid, 3, padding=1)
        self.t2 = torch.nn.Conv1d(hid, hid, 3, padding=1)
        self.out = torch.nn.Linear(hid, 1)

    def forward(self, x, ok):
        """x (N, Y, D), ok (N, Y) -> log1p biomass (N, Y)."""
        z = torch.where(ok[..., None], (x - self.mu) / self.sd, torch.zeros_like(x))
        h = torch.relu(self.l1(torch.cat([z, ok[..., None].float()], -1)))  # (N, Y, H)
        g = h.transpose(1, 2)
        g = g + torch.relu(self.t1(g))
        g = g + torch.relu(self.t2(g))
        return self.out(g.transpose(1, 2))[..., 0]


def fit(X, ok, agb, train, epochs, seed=0, w_change=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G, Y, P, D = X.shape
    sub = rng.choice(train, size=min(3000, train.size), replace=False)
    xs = X[sub].reshape(-1, D).astype(np.float32)
    m = ok[sub].reshape(-1)
    net = Traj(D, xs[m].mean(0), xs[m].std(0) + 1e-4)
    del xs
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(train.size / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    A = torch.as_tensor(np.nan_to_num(agb, nan=-1.0))
    ia = torch.as_tensor([a for a, _ in PAIRS])
    ib = torch.as_tensor([b for _, b in PAIRS])
    for _ in range(epochs):
        perm = rng.permutation(train)
        for s in range(0, perm.size, 256):
            g = np.sort(perm[s:s + 256])
            x = torch.as_tensor(X[g].astype(np.float32)).permute(0, 2, 1, 3).reshape(-1, Y, D)  # (B*P, Y, D)
            v = torch.as_tensor(ok[g]).permute(0, 2, 1).reshape(-1, Y)
            o = net(x, v).reshape(len(g), P, Y).permute(0, 2, 1)  # (B, Y, P)
            vv = torch.as_tensor(ok[g])
            pa = torch.expm1(o.clamp(max=8)).clamp(min=0) * vv
            n = vv.sum(-1)
            cm = torch.log1p(pa.sum(-1) / n.clamp(min=1))  # (B, Y)
            t = A[g]
            mb = (t >= 0) & (n >= P // 2)
            lt = torch.log1p(t.clamp(min=0))
            lb = ((cm - lt) ** 2)[mb].mean()
            mp = mb[:, ia] & mb[:, ib]
            lch = (((cm[:, ib] - cm[:, ia]) - (lt[:, ib] - lt[:, ia])) ** 2)[mp].mean() if mp.any() else 0
            loss = lb + w_change * lch
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["olmo", "all", "aef"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    R = dict(np.load(HERE / "data" / "rows.npz"))
    names, kinds = list(R["names"]), list(R["kinds"])
    fold_of = {}
    for fname, test in bh.folds(R, "square"):
        for s in np.unique(R["sq"][test]):
            fold_of[names[s]] = fname
    nets = {}
    for inp in args.inputs:
        X, ok = bh.build(R, inp)
        weights = {}
        for fname, test in bh.folds(R, "square"):
            train = np.setdiff1d(np.arange(len(R["agb"])), test)
            nets[(inp, fname)] = fit(X, ok, R["agb"], train, args.epochs)
            for k, v in nets[(inp, fname)].state_dict().items():
                weights[f"{fname}.{k}"] = v.numpy()
            gb = psutil.Process().memory_info().rss / 1e9
            print(f"  traj {inp} fold {fname}: {time.time() - t0:.0f} s, {gb:.1f} GB", flush=True)
            if gb > args.max_gb:
                raise SystemExit(f"STOP: RSS {gb:.1f} GB")
        np.savez(HERE / "data" / "heads" / f"traj_{inp}.npz", **weights)
        del X, ok
    del R

    D = np.load(HERE / "models" / "detectors.npz")
    ia = [YEARS.index(a) for a, _ in bc.WINDOWS]
    ib = [YEARS.index(b) for _, b in bc.WINDOWS]
    state = {inp: [] for inp in args.inputs}
    gated = {inp: [] for inp in args.inputs}
    truth = []
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
        moved = {}
        for k in EMB:
            E, okk = F[k]
            U = [bc.unit(E[y].astype(np.float32)) for y in range(len(YEARS))]
            mv = np.stack([1 - (U[a] * U[b]).sum(-1) >= float(D[f"{k}_cut"]) for a, b in zip(ia, ib)])
            moved[k] = mv & np.stack([okk[a] & okk[b] for a, b in zip(ia, ib)])
            del U
        moved["all"] = (moved["aef"].astype(int) + moved["tes"] + moved["olmo"]) >= 2

        def cellmean(v, valid):
            outm = np.full((ids.size, v.shape[0]), np.nan)
            for k in range(v.shape[0]):
                s = np.bincount(inv, np.where(valid[k], v[k], 0), ids.size)
                c = np.bincount(inv, valid[k], ids.size)
                outm[:, k] = np.where(c >= 20, s / np.maximum(c, 1), np.nan)
            return outm[keep]

        n = cell.size
        for inp in args.inputs:
            if inp == "all":
                ok = F["aef"][1] & F["tes"][1] & F["olmo"][1]
                parts = [F[k][0] for k in EMB]
            else:
                ok = F[inp][1]
                parts = [F[inp][0]]
            net = nets[(inp, fold_of[name])]
            pa = np.zeros((len(YEARS), n), np.float32)
            with torch.no_grad():
                for b in range(0, n, 40000):
                    x = np.concatenate([p[:, b:b + 40000] for p in parts], -1).astype(np.float32)  # (Y, m, D)
                    xt = torch.as_tensor(x).permute(1, 0, 2)
                    vt = torch.as_tensor(ok[:, b:b + 40000].T.copy())
                    o = net(xt, vt).numpy().T  # (Y, m)
                    pa[:, b:b + 40000] = np.clip(np.expm1(np.minimum(o, 8)), 0, None)
            state[inp].append(cellmean(pa, ok))
            dpix = np.stack([pa[b] - pa[a] for a, b in zip(ia, ib)])
            okw = np.stack([ok[a] & ok[b] for a, b in zip(ia, ib)])
            gated[inp].append(cellmean(np.where(moved[inp], dpix, 0.0), okw))
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"{si + 1}/{len(names)} {name} [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")
        del z, L, F

    T = np.concatenate(truth)
    res = json.load(open(RES)) if RES.exists() else {}
    for inp in args.inputs:
        Z = np.load(HERE / "data" / "heads" / f"change_cells_{inp}.npz")
        if not args.limit:
            same = np.allclose(np.nan_to_num(Z["truth"], nan=-1), np.nan_to_num(T, nan=-1))
            if not same:
                raise SystemExit("cell table differs from biomass_change.py's")
        kind = Z["kind"][:len(T)]
        sq = Z["sq"][:len(T)]
        fold = np.array([fold_of[names[s]] for s in sq])
        S = np.concatenate(state[inp])
        G = np.concatenate(gated[inp])
        out = {}
        for L in range(1, 8):
            ws = cc.windows(L)
            t = np.stack([T[:, b] - T[:, a] for a, b in ws], 1)
            base = np.stack([T[:, a] for a, b in ws], 1)
            meth = {"traj": np.stack([S[:, b] - S[:, a] for a, b in ws], 1)}
            cum = np.concatenate([np.zeros((len(T), 1)), np.cumsum(Z["plain"][:len(T), :7], 1)], 1)
            meth["state"] = np.stack([cum[:, b] - cum[:, a] for a, b in ws], 1)
            if L in (1, 7):
                sel = slice(0, 7) if L == 1 else slice(7, 8)
                meth["traj_gated"] = G[:, sel]
                meth["state_gated"] = Z["gated"][:len(T), sel]
            for nm in list(meth):
                p = meth[nm]
                cal = np.full_like(p, np.nan)
                for f in np.unique(fold):
                    tr = fold != f
                    cal[fold == f] = p[fold == f] / cc.loss_ratio(p[tr].ravel(), t[tr].ravel(), base[tr].ravel())
                meth[nm + "/l"] = cal
            for nm, p in meth.items():
                for gname in ("all", "forest-loss", "other"):
                    gm = np.ones(len(kind), bool) if gname == "all" else (kind == "forest-loss") if gname == "forest-loss" else (kind != "forest-loss")
                    tt, pp, aa = t[gm].ravel(), p[gm].ravel(), base[gm].ravel()
                    m = np.isfinite(tt) & np.isfinite(pp) & np.isfinite(aa)
                    out.setdefault(f"L{L}", {})[f"{nm}/{gname}"] = bc.stats({"d": tt[m], "a": aa[m]}, pp[m])
        # state accuracy, per year, against CTrees on the cells
        m = np.isfinite(T) & np.isfinite(S)
        out["state_r2_log"] = round(bh.r2(np.log1p(T[m]), np.log1p(S[m])), 3)
        res[inp] = out
        np.savez(HERE / "data" / "heads" / f"change_cells_traj_{inp}.npz", state=S.astype(np.float32), gated=G.astype(np.float32))
        print(f"--- {inp}: cell state R2 log {out['state_r2_log']}")
        for L in (1, 3, 7):
            o = out[f"L{L}"]
            line = f"  L{L}"
            for nm in ("state", "traj", "state/l", "traj/l", "state_gated/l", "traj_gated/l"):
                k = f"{nm}/forest-loss"
                if k in o:
                    a, b = o[k], o[f"{nm}/other"]
                    line += (f" | {nm}: corr {a.get('corr')} loss {a.get('loss_ratio')}/{a.get('loss_found')} "
                             f"gain {a.get('gain_ratio')}/{a.get('gain_found')} steady {a.get('steady_lit')} other {b.get('steady_lit')}")
            print(line, flush=True)
    json.dump(res, open(RES, "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
