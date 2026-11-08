# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Grass tests after section 17 (RAP): a grass change threshold, California annual grass, a two-part head, and a
RAP-taught head abroad.

  noise       a change threshold for grass set against RAP's year noise, from the saved blocks
              (data/heads/map_check_rap.npz): for each model, the smallest relative step r such that steady ground
              (RAP's step under 5 %) is lit at most 10 % (lit = the model's step at least r of its own previous year
              and at least 0.05 Mg/ha); RAP's swings (0.25 Mg/ha and 25 %) caught the same way at that r
  california  foothill-annual-ca, where the RAP head runs opposite to RAP (-.85): RAP's annual and perennial
              herbaceous biomass and its cover classes by year, beside the head
  twopart     one head, two outputs: total biomass taught on CTrees (rows.npz, all 92 squares, the cell-mean loss of
              biomass_head.py) and herbaceous biomass taught on RAP (the 12 RAP squares, per pixel as map_check.py).
              The 8 square folds of biomass_head.py; the 10 RAP squares outside the rows are dealt to the folds in
              turn (sandhills-us and great-basin-us keep theirs). Three fits per fold with the same code: CTrees
              only, RAP only, both. Scored on held-out squares only: CTrees R2 of log on cells (all, grass, shrub),
              RAP on 100 m blocks with map_check.score.
  abroad      a RAP head taught on all 12 US squares, run on the world grass, shrub and desert-edge squares of
              veg_squares.json: level and yearly median per square beside CTrees, and a few droughts checked

Writes results/grass_tests.json (one key per part). RAM: twopart about 6 GB with one input per run (rows features plus
the RAP table), the rest under 4 GB; each part stops past --max-gb.

    uv run scripts/grass_tests.py noise california
    uv run scripts/grass_tests.py twopart --inputs all olmo
    uv run scripts/grass_tests.py abroad --inputs all tes
"""
import argparse
import json
import sys
import threading
import time
import os
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402
import map_check as mc  # noqa: E402

RES = HERE / "results" / "grass_tests.json"
YEARS = bc.YEARS
DIMS = {"aef": 64, "tes": 128, "olmo": 128, "all": 320}


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


def save(key, val):
    res = json.load(open(RES)) if RES.exists() else {}
    res[key] = val
    json.dump(res, open(RES, "w"), indent=1)


# ---------------------------------------------------------------- noise
def noise():
    z = np.load(HERE / "data" / "heads" / "map_check_rap.npz")
    T = z["label"]
    sT = T[:, 1:] - T[:, :-1]
    baseT = np.maximum(T[:, :-1], 1e-3)
    out = {}
    models = ["ctrees"] + [k for k in z.files if k.startswith(("ctrees_head_", "own_head_"))]
    for k in models:
        P = z[k]
        sP = P[:, 1:] - P[:, :-1]
        baseP = np.maximum(P[:, :-1], 1e-3)
        ok = np.isfinite(sT) & np.isfinite(sP)
        st, sp, bt, bp = sT[ok], sP[ok], baseT[ok], baseP[ok]
        steady = np.abs(st) < 0.05 * bt
        big = (np.abs(st) >= 0.25) & (np.abs(st) >= 0.25 * bt)
        rel = np.abs(sp) / bp
        lit_ok = np.abs(sp) >= 0.05
        # smallest r with steady lit <= 10 %
        r_grid = np.round(np.arange(0.02, 1.01, 0.01), 2)
        pick = None
        rows = []
        for r in r_grid:
            moved = lit_ok & (rel >= r)
            sl = float(moved[steady].mean())
            found = float((moved & (np.sign(sp) == np.sign(st)))[big].mean())
            rows.append((r, sl, found))
            if pick is None and sl <= 0.10:
                pick = (r, sl, found)
        o = {"steady_n": int(steady.sum()), "swings_n": int(big.sum())}
        for r_show in (0.10, 0.25):
            rr = [x for x in rows if abs(x[0] - r_show) < 1e-9][0]
            o[f"at_{int(r_show * 100)}pct"] = {"steady_lit": round(rr[1], 3), "swings_found": round(rr[2], 3)}
        if pick:
            o["threshold"] = {"r": pick[0], "steady_lit": round(pick[1], 3), "swings_found": round(pick[2], 3)}
            # in Mg/ha at the median base of the model, for scale
            o["threshold"]["mg_ha_at_median"] = round(float(pick[0] * np.median(bp)), 3)
        # swings caught, also restricted to large swings (50 %)
        big50 = (np.abs(st) >= 0.25) & (np.abs(st) >= 0.5 * bt)
        if pick:
            moved = lit_ok & (rel >= pick[0])
            o["threshold"]["swings50_found"] = round(float((moved & (np.sign(sp) == np.sign(st)))[big50].mean()), 3)
        out[k] = o
        print(f"{k:18s} {json.dumps(o)}", flush=True)
    # RAP's own year noise, for scale: median |step| / base on all blocks
    ok = np.isfinite(sT)
    out["rap_year_noise"] = {"median_rel_step": round(float(np.median(np.abs(sT[ok]) / baseT[ok])), 3),
                             "p90_rel_step": round(float(np.percentile(np.abs(sT[ok]) / baseT[ok], 90)), 3),
                             "steady_share": round(float((np.abs(sT[ok]) < 0.05 * baseT[ok]).mean()), 3)}
    print("rap year noise", out["rap_year_noise"], flush=True)
    save("noise", out)


# ---------------------------------------------------------------- california
def california():
    name = "foothill-annual-ca"
    d = np.load(HERE / "data" / "rap" / f"{name}.npz")
    z = np.load(HERE / "data" / "heads" / "map_check_rap.npz")
    si = list(z["names"]).index(name)
    afg = np.where(d["afg"] != 65535, d["afg"] * mc.LB_ACRE, np.nan)
    pfg = np.where(d["pfg"] != 65535, d["pfg"] * mc.LB_ACRE, np.nan)
    cov = d["cover"].astype(np.float32)
    cov[cov == 255] = np.nan
    names = ["annual_forb_grass", "bare", "litter", "perennial_forb_grass", "shrub", "tree"]
    out = {"years": YEARS,
           "rap_annual_herb_mg_ha": [round(float(np.nanmedian(a)), 2) for a in afg],
           "rap_perennial_herb_mg_ha": [round(float(np.nanmedian(a)), 2) for a in pfg],
           "annual_share_of_herb": [round(float(np.nanmedian(a / np.maximum(a + p, 1e-6))), 3) for a, p in zip(afg, pfg)],
           "cover_pct_mean": {nm: [round(float(np.nanmean(cov[y, i])), 1) for y in range(8)] for i, nm in enumerate(names)}}
    for k in ("own_head_all", "own_head_tes", "ctrees_head_all", "ctrees"):
        out[k] = [round(float(np.nanmedian(v)), 2) for v in z[k][si]]
    out["label_blocks"] = [round(float(np.nanmedian(v)), 2) for v in z["label"][si]]
    # per block: does the head follow annual or perennial, or neither, year to year
    T = z["label"][si]
    P = z["own_head_all"][si]
    A, _ = mc.blocks(afg)
    Pf, _ = mc.blocks(pfg)
    C, _ = mc.blocks(cov[:, 1])  # bare
    L, _ = mc.blocks(cov[:, 2])  # litter
    def dep(x):
        return x - np.nanmean(x, 0, keepdims=True)
    out["departure_corr_head_vs"] = {"rap_total": mc.corr(dep(T).ravel(), dep(P).ravel()),
                                     "rap_annual": mc.corr(dep(A).ravel(), dep(P).ravel()),
                                     "rap_perennial": mc.corr(dep(Pf).ravel(), dep(P).ravel()),
                                     "bare_cover": mc.corr(dep(C).ravel(), dep(P).ravel()),
                                     "litter_cover": mc.corr(dep(L).ravel(), dep(P).ravel())}
    for k, v in out.items():
        print(f"{k:28s} {v}", flush=True)
    save("california", out)


# ---------------------------------------------------------------- shared: one year of one input, light on RAM
def year_x(z, inp, yi):
    """float32 (n, D) for one input and year, and the pixels valid in all three embeddings that year."""
    q = {k: z[k][yi] for k in bc.EMB}
    sc = z["tes_scale"][yi]
    ok = (q["aef"][:, 0] != -128) & (q["olmo"][:, 0] != -128) & np.isfinite(sc) & (sc > 0)

    def one(k):
        return q[k].astype(np.float32) * np.nan_to_num(sc)[:, None] if k == "tes" else bh.deq(q[k])
    x = np.concatenate([one(k) for k in bc.EMB], -1) if inp == "all" else one(inp)
    return x, ok


# ---------------------------------------------------------------- shared: the RAP table
def rap_table(names, inputs, px, rng, years=None):
    """Training pixels per RAP square: features per input (float16), target Mg/ha, square index."""
    tab = {inp: [] for inp in inputs}
    tt, ts = [], []
    for si, name in enumerate(names):
        lab, ly = mc.label_of("rap", name)
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        for k, y in enumerate(ly):
            yi = YEARS.index(y)
            pick = None
            for inp in inputs:
                x, okE = year_x(z, inp, yi)
                if pick is None:
                    cand = np.flatnonzero(okE & np.isfinite(lab[k]))
                    pick = np.sort(rng.choice(cand, min(px, cand.size), replace=False))
                tab[inp].append(x[pick].astype(np.float16))
                del x
            tt.append(lab[k][pick])
            ts.append(np.full(pick.size, si, np.int16))
        del z
    return {inp: np.concatenate(v) for inp, v in tab.items()}, np.concatenate(tt), np.concatenate(ts)


# ---------------------------------------------------------------- twopart
class Two(torch.nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.l1 = torch.nn.Linear(d, bh.HID)
        self.l2 = torch.nn.Linear(bh.HID, 2)  # 0 total biomass (CTrees), 1 herbaceous (RAP); log1p Mg/ha per pixel

    def forward(self, x):
        return self.l2(torch.relu(self.l1((x - self.mu) / self.sd)))


def fit_two(X, ok, agb, train_c, Xr, tr, train_r, w_c, w_r, epochs=8, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G, Y, P, D = X.shape
    parts = []
    if w_c > 0:
        sub = rng.choice(train_c, size=min(3000, train_c.size), replace=False)
        xs = X[sub].reshape(-1, D).astype(np.float32)
        parts.append(xs[ok[sub].reshape(-1)])
    if w_r > 0:
        parts.append(Xr[rng.choice(train_r, min(100000, train_r.size), replace=False)].astype(np.float32))
    xs = np.concatenate(parts)
    net = Two(D, xs.mean(0), xs.std(0) + 1e-4)
    del xs, parts
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    nb = int(np.ceil(train_c.size / 256)) if w_c > 0 else int(np.ceil(train_r.size / 2048))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * nb)
    agb_t = torch.as_tensor(np.nan_to_num(agb, nan=-1.0))
    okt = torch.as_tensor(ok)
    trt = torch.as_tensor(np.log1p(tr).astype(np.float32))
    for _ in range(epochs):
        perm_c = rng.permutation(train_c) if w_c > 0 else None
        for b in range(nb):
            loss = 0
            if w_c > 0:
                g = np.sort(perm_c[b * 256:(b + 1) * 256])
                o = net(torch.as_tensor(X[g].astype(np.float32)))[..., 0]
                v = okt[g]
                pa = torch.expm1(o.clamp(max=8)).clamp(min=0) * v
                n = v.sum(-1)
                cm = pa.sum(-1) / n.clamp(min=1)
                t = agb_t[g]
                mb = (t >= 0) & (n >= P // 2)
                if mb.any():
                    loss = loss + w_c * ((torch.log1p(cm) - torch.log1p(t.clamp(min=0))) ** 2)[mb].mean()
            if w_r > 0:
                gr = rng.choice(train_r, 2048, replace=False)
                o = net(torch.as_tensor(Xr[gr].astype(np.float32)))[:, 1]
                loss = loss + w_r * torch.nn.functional.huber_loss(o, trt[gr], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def pred_cells(net, X, ok, idx):
    G, Y, P, D = X.shape
    out = np.full((idx.size, Y), np.nan)
    for b in range(0, idx.size, 1024):
        g = idx[b:b + 1024]
        o = net(torch.as_tensor(X[g].astype(np.float32)))[..., 0].numpy()
        v = ok[g]
        a = np.clip(np.expm1(np.minimum(o, 8)), 0, None) * v
        n = v.sum(-1)
        out[b:b + g.size] = np.where(n >= P // 2, a.sum(-1) / np.maximum(n, 1), np.nan)
    return out


@torch.no_grad()
def pred_square_blocks(net, name, inp, years, out_idx):
    """The net's output out_idx on every pixel of a square, 100 m block means (Y, 3600)."""
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    pred = np.full((len(years), 360000), np.nan, np.float32)
    for k, y in enumerate(years):
        x, m = year_x(z, inp, YEARS.index(y))
        x = x[m]
        o = np.concatenate([net(torch.as_tensor(x[b:b + 100000]))[:, out_idx].numpy() for b in range(0, len(x), 100000)])
        pred[k, m] = np.clip(np.expm1(np.minimum(o, 8)), 0, None)
        del x
    del z
    pb, pv = mc.blocks(pred)
    pb[pv < 0.8] = np.nan
    return pb


def r2log(t, p):
    m = np.isfinite(t) & np.isfinite(p)
    return round(bh.r2(np.log1p(t[m]), np.log1p(p[m])), 3) if m.sum() > 50 else None


def twopart(inputs, px, max_gb):
    peak = [0.0]
    guard(max_gb, peak)
    R = dict(np.load(HERE / "data" / "rows.npz"))
    rows_names = list(R["names"])
    zb = np.load(HERE / "data" / "heads" / "map_check_rap.npz")
    rnames = list(zb["names"])
    T_blocks = zb["label"]
    fold_list = bh.folds(R, "square")
    fold_of = {}
    for fname, test in fold_list:
        for s in np.unique(R["sq"][test]):
            fold_of[rows_names[s]] = int(fname)
    k = 0
    for n in rnames:
        if n not in fold_of:
            fold_of[n] = k % 8
            k += 1
    rng = np.random.default_rng(0)
    Xr_all, tr, ts = rap_table(rnames, inputs, px, rng)
    r_fold = np.array([fold_of[rnames[s]] for s in ts])
    kinds = R["kinds"][R["sq"]]
    common = np.ones(R["cls"].shape, bool)
    for kk in ("aef", "tes", "olmo"):
        common &= (np.isfinite(R["tes_s"]) & (R["tes_s"] > 0)) if kk == "tes" else (R[kk][..., 0] != -128)
    prev = json.load(open(RES)).get("twopart", {}) if RES.exists() else {}
    out = prev | {"folds": {n: fold_of[n] for n in rnames}, "rap_px_per_square_year": px}
    for inp in inputs:
        X, ok = bh.build(R, inp)
        ok = ok & common
        if len(inputs) == 1:
            for kk in ("aef", "tes", "olmo", "tes_s", "cls", "height", "height_year", "loss"):
                R.pop(kk, None)
        Xr = Xr_all[inp]
        G = X.shape[0]
        res = {}
        for variant, (wc, wr) in {"ctrees_only": (1.0, 0.0), "rap_only": (0.0, 1.0), "both": (1.0, 1.0)}.items():
            t0 = time.time()
            pa = np.full((G, len(YEARS)), np.nan)
            rap_pred = np.full(T_blocks.shape, np.nan, np.float32)
            for fname, test in fold_list:
                f = int(fname)
                train_c = np.setdiff1d(np.arange(G), test)
                train_r = np.flatnonzero(r_fold != f)
                net = fit_two(X, ok, R["agb"], train_c, Xr, tr, train_r, wc, wr)
                if wc > 0:
                    pa[test] = pred_cells(net, X, ok, test)
                if wr > 0:
                    for si, n in enumerate(rnames):
                        if fold_of[n] == f:
                            rap_pred[si] = pred_square_blocks(net, n, inp, list(zb["years"]), 1)
                print(f"  {inp} {variant} fold {f} [{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
            o = {}
            if wc > 0:
                t = R["agb"]
                o["ctrees_r2_log"] = r2log(t, pa)
                for kd in ("grass", "shrub", "savanna", "closed-forest"):
                    o[f"ctrees_r2_log_{kd}"] = r2log(t[kinds == kd], pa[kinds == kd])
                o["ctrees_median_error"] = round(float(np.nanmedian(np.abs(t - pa))), 2)
            if wr > 0:
                s = mc.score(T_blocks, rap_pred, rnames, 0.25)
                o["rap"] = {kk: v for kk, v in s.items() if kk != "series"}
                o["rap_series_corr"] = {n: v["corr"] for n, v in s["series"].items()}
            res[variant] = o
            print(f"{inp} {variant}: {json.dumps({kk: v for kk, v in o.items() if kk not in ('rap_series_corr',)})}", flush=True)
        out[inp] = res
        del X, ok
        save("twopart", out)
    out[f"peak_gb_{'_'.join(inputs)}"] = round(peak[0], 1)
    save("twopart", out)


# ---------------------------------------------------------------- abroad
DROUGHTS = {  # place: (year expected lower, years expected higher, source of the expectation)
    "pampas-ar": ([2023], [2019, 2024], "2022/23 Pampas drought, among the worst in decades"),
    "karoo-za": ([2019], [2021, 2022], "2018 to 2020 Karoo and Northern Cape drought, broken by rains from 2021"),
    "mitchell-grass-au": ([2019], [2021, 2022], "2019 eastern Australia drought; La Nina wet years 2020 to 2022"),
    "mulga-au": ([2019], [2021, 2022], "2019 drought in central Australia; La Nina wet years 2020 to 2022"),
}


def abroad(inputs, px, max_gb):
    peak = [0.0]
    guard(max_gb, peak)
    zb = np.load(HERE / "data" / "heads" / "map_check_rap.npz")
    rnames = list(zb["names"])
    rng = np.random.default_rng(0)
    Xr_all, tr, ts = rap_table(rnames, inputs, px, rng)
    veg = json.load(open(HERE / "data" / "veg_squares.json"))
    targets = [n for n, v in veg.items() if v["kind"] in ("grass", "shrub", "desert-edge") and not n.endswith("-us")]
    out = {"squares": targets, "droughts_checked": {k: v[2] for k, v in DROUGHTS.items()}}
    for inp in inputs:
        net = mc.fit_one(Xr_all[inp], tr)
        # the same head on the US squares it was taught on, for the level it gives at home
        per = {}
        med_h, med_c = [], []
        for name in targets:
            z = np.load(HERE / "data" / "squares" / f"{name}.npz")
            L = np.load(HERE / "data" / "labels" / f"{name}.npz")
            pred = np.full((len(YEARS), 360000), np.nan, np.float32)
            okE = np.zeros((len(YEARS), 360000), bool)
            for y in range(len(YEARS)):
                x, okE[y] = year_x(z, inp, y)
                if okE[y].sum():
                    pred[y, okE[y]] = mc.run_one(net, x[okE[y]])
                del x
            hb, hv = mc.blocks(pred)
            hb[hv < 0.8] = np.nan
            agb = L["agb"].astype(np.float32)
            cb, cv = mc.blocks(np.where((agb >= 0) & okE, agb / 10, np.nan))
            cb[cv < 0.8] = np.nan
            hs = [round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None for v in hb]
            cs = [round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None for v in cb]
            ok = [i for i in range(8) if hs[i] is not None and cs[i] is not None]
            within = mc.spear(hb[ok].ravel(), cb[ok].ravel()) if ok else None
            o = {"kind": veg[name]["kind"], "head": hs, "ctrees": cs, "rank_within_vs_ctrees": within,
                 "series_corr_vs_ctrees": round(float(np.corrcoef([hs[i] for i in ok], [cs[i] for i in ok])[0, 1]), 3)
                 if len(ok) >= 4 and np.std([hs[i] for i in ok]) > 0 and np.std([cs[i] for i in ok]) > 0 else None}
            if name in DROUGHTS:
                low, high, _ = DROUGHTS[name]
                hl = np.mean([hs[YEARS.index(y)] for y in low if hs[YEARS.index(y)] is not None])
                hh = np.mean([hs[YEARS.index(y)] for y in high if hs[YEARS.index(y)] is not None])
                cl = np.mean([cs[YEARS.index(y)] for y in low if cs[YEARS.index(y)] is not None])
                ch = np.mean([cs[YEARS.index(y)] for y in high if cs[YEARS.index(y)] is not None])
                o["drought"] = {"low_years": low, "high_years": high, "head_low_over_high": round(float(hl / hh), 3),
                                "ctrees_low_over_high": round(float(cl / ch), 3)}
            per[name] = o
            med_h.append(np.nanmedian([v for v in hs if v is not None]))
            med_c.append(np.nanmedian([v for v in cs if v is not None]))
            print(f"  {inp} {name:20s} head {hs}  ctrees {cs}  within {within}  {o.get('drought', '')}  "
                  f"[peak {peak[0]:.1f} GB]", flush=True)
            del z, L
        out[inp] = {"per_square": per, "rank_square_medians_vs_ctrees": round(bh.spearman(np.array(med_h), np.array(med_c)), 3),
                    "head_median_of_medians": round(float(np.median(med_h)), 2),
                    "ctrees_median_of_medians": round(float(np.median(med_c)), 2)}
        print(f"{inp}: rank of square medians vs CTrees {out[inp]['rank_square_medians_vs_ctrees']}", flush=True)
        save("abroad", out)
    out["peak_gb"] = round(peak[0], 1)
    save("abroad", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", choices=["noise", "california", "twopart", "abroad"])
    ap.add_argument("--inputs", nargs="*", default=["all"])
    ap.add_argument("--px", type=int, default=12000)
    ap.add_argument("--max-gb", type=float, default=8)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for p in args.parts:
        if p == "noise":
            noise()
        elif p == "california":
            california()
        elif p == "twopart":
            twopart(args.inputs, args.px, args.max_gb)
        else:
            abroad(args.inputs, args.px, args.max_gb)


if __name__ == "__main__":
    main()
