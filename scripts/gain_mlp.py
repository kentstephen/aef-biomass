"""The lidar-taught gain as a small network the browser can run (FINDINGS 32, 33), and its export.

Same teacher and harness as lidar_gain_head.py: the nine two-flight lidar squares, target lidar's canopy rose 2 m,
held out by square, scored against steady (moved under 0.5 m) and then, taught on all nine, on Esri's "became
trees" over every other square. In place of the boosted trees, one hidden layer of 32 (ReLU) and a logistic output,
so the browser can run it per pixel beside the biomass head. Two inputs:
  difference   e_b - e_a and the window's length (section 33: the part that carries to other places)
  embeddings   e_a, e_b, e_b - e_a and the window's length (section 32's input)
Then the steady ground table for the browser: the network's output on steady pixels of every square (Esri class held
over the window, CTrees steady, as change_confidence_test.py), quantiles per window length, so a pixel's confidence
is the share of steady ground scored below it, as for the map's other change rules.

Writes results/gain_mlp.json and, with --export <input>, web/public/models/gain_lidar.json. RAM about 2 GB.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests --with torch python scripts/gain_mlp.py [--export difference]
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
import change_confidence_test as cct  # noqa: E402
import growth_test as gt  # noqa: E402
import lidar_gain_head as lg  # noqa: E402
import nlcd_check as nc  # noqa: E402

OUT = HERE / "results" / "gain_mlp.json"
EXPORT = HERE / "web" / "public" / "models" / "gain_lidar.json"
HID = 32
P = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999]


def feats(qa, qb, L, v):
    ea, eb = bh.deq(qa), bh.deq(qb)
    Lc = np.full((ea.shape[0], 1), L, np.float32) if np.isscalar(L) else np.asarray(L, np.float32)[:, None]
    cols = [eb - ea] if v == "difference" else [ea, eb, eb - ea]
    return np.concatenate(cols + [Lc], 1).astype(np.float32)


class Net(torch.nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(d, HID)
        self.l2 = torch.nn.Linear(HID, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1((x - self.mu) / self.sd)))[..., 0]


class Clf:
    """predict_proba like sklearn's, so lidar_gain_head's scoring and Esri check run unchanged."""

    def __init__(self, net, v):
        self.net, self.v = net, v

    @torch.no_grad()
    def predict_proba(self, X):
        z = np.concatenate([self.net(torch.as_tensor(X[b:b + 200000])).numpy() for b in range(0, len(X), 200000)])
        p = 1 / (1 + np.exp(-z))
        return np.stack([1 - p, p], 1)


def fit(X, y, seed=0, epochs=10):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(X.shape[1], X.mean(0), X.std(0) + 1e-4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    bs = 1024
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y.astype(np.float32))
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(len(X)))
        for b in range(0, len(X), bs):
            g = perm[b:b + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[g]), yt[g])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def steady_table(clf, v, rng):
    """Quantiles P of the network's output on steady pixels (Esri class held, CTrees steady), per window length."""
    sq = nc.squares(world=True)
    got = {L: [] for L in range(1, 8)}
    for name in sorted(sq):
        if name in cct.LIDAR:
            continue
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"]
        if q.shape[1] != 360000:
            continue
        s = z["tes_scale"]
        ok = (q[:, :, 0] != -128) & np.isfinite(s) & (s > 0)
        io = np.load(fi)["lndcov"][:cct.NY]
        ct = nc.labels_of(name)["ctrees"]
        ks = np.zeros(360000, bool)
        ks[rng.integers(cct.STEP_STEADY * 4)::cct.STEP_STEADY * 4] = True
        for a in range(cct.NY - 1):
            for b in range(a + 1, cct.NY):
                ct_st = np.isfinite(ct[a]) & np.isfinite(ct[b]) & (np.abs(ct[b] - ct[a]) < 10) & \
                    (np.abs(ct[b] - ct[a]) < 0.1 * np.maximum(ct[a], 1))
                st = ok[a] & ok[b] & np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250) & ct_st & ks
                idx = np.flatnonzero(st)
                if idx.size:
                    got[b - a].append(clf.predict_proba(feats(q[a, idx], q[b, idx], b - a, v))[:, 1].astype(np.float32))
        del z, q
    out = {}
    for L, xs in got.items():
        if xs:
            x = np.concatenate(xs)
            out[str(L)] = {"n": int(x.size), "q": [float(np.quantile(x, p)) for p in P]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=6)
    ap.add_argument("--export", choices=["difference", "embeddings"])
    args = ap.parse_args()
    cct.guard(args.max_gb)
    t0 = time.time()
    rng = np.random.default_rng(0)
    sq = {n: lg.lidar_square(n) for n in cct.LIDAR}
    picks = {n: lg.sample(S, rng) for n, S in sq.items()}
    res = {"note": __doc__.split("\n\n")[0], "variants": {}}
    finals = {}
    for v in ("difference", "embeddings"):
        X = {n: feats(S["qa"][picks[n]], S["qb"][picks[n]], S["L"], v) for n, S in sq.items()}
        y = {n: (S["dH"][picks[n]] >= 2).astype(np.int8) for n, S in sq.items()}
        per, pool = {}, {k: [] for k in ("p", "d", "a", "L", "g", "s")}
        for held in sq:
            tr = [n for n in sq if n != held]
            clf = Clf(fit(np.concatenate([X[n] for n in tr]), np.concatenate([y[n] for n in tr])), v)
            S = sq[held]
            p = clf.predict_proba(feats(S["qa"], S["qb"], S["L"], v))[:, 1].astype(np.float32)
            gained, steady = S["dH"] >= 2, np.abs(S["dH"]) < 0.5
            per[held] = lg.scores(p, S["d"], S["a"], S["L"], gained, steady) | {"years": S["years"]}
            keep = gained | steady
            for k, val in (("p", p), ("d", S["d"]), ("a", S["a"]), ("L", np.full(p.size, S["L"])), ("g", gained), ("s", steady)):
                pool[k].append(val[keep])
            print(f"{v} held {held}: {json.dumps(per[held])}", flush=True)
        Pp = {k: np.concatenate(x) for k, x in pool.items()}
        pooled = lg.scores(Pp["p"], Pp["d"], Pp["a"], Pp["L"], Pp["g"], Pp["s"])
        res["variants"][v] = {"pooled": pooled, "per square": per}
        print(f"{v} POOLED {json.dumps(pooled)} [{time.time() - t0:.0f} s]", flush=True)
        finals[v] = Clf(fit(np.concatenate(list(X.values())), np.concatenate(list(y.values()))), v)
    del sq
    # lidar_gain_head's Esri check builds its own features from the variant name: give it this script's
    lg.feats = lambda qa, qb, L, a, d, v: feats(qa, qb, L, v)
    res["esri"] = lg.esri_check(finals, rng)
    if args.export:
        v = args.export
        tab = steady_table(finals[v], v, np.random.default_rng(1))
        net = finals[v].net
        sd = {k: t.detach().numpy().astype(np.float64) for k, t in net.state_dict().items()}
        n_in = sd["l1.weight"].shape[1]
        out = {"name": "gain_lidar", "input": v, "n_in": n_in, "hidden": HID,
               "note": "lidar-taught gain (FINDINGS 34): p(canopy rose 2 m) from AlphaEarth; inputs " +
                       ("e_b - e_a (64), years" if v == "difference" else "e_a, e_b, e_b - e_a (192), years"),
               "mean": sd["mu"].round(6).tolist(), "scale": sd["sd"].round(6).tolist(),
               "w1": sd["l1.weight"].ravel().round(6).tolist(), "b1": sd["l1.bias"].round(6).tolist(),
               "w2": sd["l2.weight"].ravel().round(6).tolist(), "b2": float(sd["l2.bias"][0]),
               "steady": {"p": P, "L": tab}}
        # a probe so the browser can check its own arithmetic: 32 steady-table pixels and their outputs
        rp = np.random.default_rng(2)
        qa = rp.integers(-60, 60, (32, 64)).astype(np.int8)
        qb = (qa + rp.integers(-20, 20, (32, 64))).clip(-127, 127).astype(np.int8)
        out["probe_a"], out["probe_b"] = qa.ravel().tolist(), qb.ravel().tolist()
        out["probe_L"] = 3
        out["probe_p"] = finals[v].predict_proba(feats(qa, qb, 3, v))[:, 1].round(6).tolist()
        json.dump(out, open(EXPORT, "w"))
        res["export"] = {"input": v, "steady": tab}
        print(f"wrote {EXPORT}", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "variants"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
