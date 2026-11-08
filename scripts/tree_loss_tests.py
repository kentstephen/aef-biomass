# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil", "scikit-learn", "rasterio", "pystac-client"]
# ///
"""The open tree-loss items of section 4, on the 37 tree-loss squares (data/squares.json).

Same definitions as score_squares.py: tree loss in year Y (2019 to 2024) = plain change 1 - cos(e[Y-1], e[Y+1]) at
or over the frozen cutoff, passed by a gate on e[Y-1]; vote = 2 of 3 embeddings; against Hansen GFC v1.12:
  found       Hansen loss 2019 to 2024 flagged within a year of Hansen's year
  forest lit  never lost, 30 m from any loss, tree cover 2000 >= 30 %: share flagged in any year
  open lit    never lost, cover < 10 %, WorldCover 2020 not trees
Variants (models/detectors.npz is not changed):
  frozen      the frozen cutoff and gate
  ungated     the frozen cutoff, no gate
  height      the frozen cutoff, gate = the biomass head's canopy height the year before >= 3 m (the square's
              held-out fold head of biomass_head.py, same embedding)
  refit       the frozen cutoff, the gate re-fit on these squares (held out: 5 folds of squares, drivers spread):
              before-year embeddings of Hansen loss and of standing forest against open ground (these squares'
              and the 55 vegetation squares' non-tree pixels), logistic, cutoff keeping 95 % of loss pixels
  scaled      refit gate, cutoff scaled by Hansen tree cover 2000: cut x clip(cover / 60, 0.35, 1)
  lower       refit gate, cutoff x 0.7 everywhere
By kind: the squares' drivers (commodity, shifting, forestry, insects, fire) and the dry or open set named in
section 4 (chaco-py, manavgat-tr, mpumalanga-za, yakutia-fire-ru, cote-divoire). By cover band (Hansen 2000):
found on loss pixels and lit on never-lost pixels, 10-30, 30-60, 60-100 %.

  uv run scripts/tree_loss_tests.py gates     writes results/tree_loss_tests.json ("gates"), and the vote flags of
                                              riau-id, nicaragua and latvia to data/tree_loss_tests/
  uv run scripts/tree_loss_tests.py fire      GLAD forest loss due to fire 2001 to 2025 (Tyukavina et al., public
                                              tiles, glad.umd.edu) under the squares: found on fire and on other loss
  uv run scripts/tree_loss_tests.py s2        Sentinel-2 L2A (Earth Search, no sign-in): NBR before and after at
                                              flags Hansen does not have, against Hansen loss and steady forest
One square at a time; peak about 5 GB (gates).
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
import biomass_head as bh  # noqa: E402

SQ = HERE / "data" / "squares"
OUT = HERE / "data" / "tree_loss_tests"
RES = HERE / "results" / "tree_loss_tests.json"
YEARS = list(range(2018, 2026))
CAND = list(range(2019, 2025))
EMB = ("aef", "tes", "olmo")
DIMS = {"aef": 64, "tes": 128, "olmo": 128}
DRY = {"chaco-py", "manavgat-tr", "mpumalanga-za", "yakutia-fire-ru", "cote-divoire"}
BANDS = [(10, 30), (30, 60), (60, 101)]
VARIANTS = ("frozen", "ungated", "height", "refit", "scaled", "lower")
S2_SQUARES = ("riau-id", "nicaragua", "latvia")


def guard(cap):
    p = psutil.Process()
    peak = [0.0]

    def run():
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > cap:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return peak


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def year_feats(z, k, y):
    """Head input (float32 (n, D)) and valid mask for embedding k, year index y."""
    if k == "tes":
        s = z["tes_scale"][y]
        return z["tes"][y].astype(np.float32) * np.nan_to_num(s)[:, None], np.isfinite(s) & (s > 0)
    q = z[k][y]
    return bh.deq(q), q[:, 0] != -128


def dilate(mask, side, r=3):
    m = mask.reshape(side, side)
    p = np.pad(m, r)
    o = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= p[r + dy:r + dy + side, r + dx:r + dx + side]
    return o.ravel()


def ground(z):
    loss, cover, wc20 = z["loss"].astype(int), z["cover"].astype(int), z["wc2020"]
    side = int(round(np.sqrt(len(loss))))
    L = np.where((loss >= 19) & (loss <= 24), 2000 + loss, 0)
    near = dilate(loss > 0, side)
    forest = (loss == 0) & ~near & (cover >= 30)
    open_ = (loss == 0) & ~near & (cover < 10) & (wc20 != 10) & (wc20 > 0)
    return L, cover, forest, open_, (loss == 0) & ~near


def frozen_gate(D, k, u):
    return ((u - D[f"{k}_gate_mean"]) / D[f"{k}_gate_scale"]) @ D[f"{k}_gate_coef"] + D[f"{k}_gate_b"]


def square_folds(names, drivers, k=5):
    rng = np.random.default_rng(3)
    order = []
    for d in sorted(set(drivers.values())):
        order += list(rng.permutation([n for n in names if drivers[n] == d]))
    return {n: i % k for i, n in enumerate(order)}


def collect_gate_rows(names, veg, rng):
    """Before-year unit embeddings: Hansen loss (y = 1), standing forest (y = 1), open ground (y = 0)."""
    rows = {k: {"x": [], "y": [], "sq": []} for k in EMB}
    for n in names:
        z = np.load(SQ / f"{n}.npz")
        L, cover, forest, open_, _ = ground(z)
        for k in EMB:
            U, OK = [], []
            for yi in range(len(YEARS) - 1):
                x, ok = year_feats(z, k, yi)
                U.append(unit(x).astype(np.float16))
                OK.append(ok)
            lp = np.flatnonzero(L > 0)
            sel_l = rng.choice(lp, min(4000, lp.size), replace=False) if lp.size else np.array([], int)
            yb = L[sel_l] - 1 - YEARS[0]  # year index of the year before
            xs = [U[i][j] for i, j in zip(yb, sel_l) if OK[i][j]]
            fp, op = np.flatnonzero(forest), np.flatnonzero(open_)
            sel_f = rng.choice(fp, min(2000, fp.size), replace=False) if fp.size else np.array([], int)
            sel_o = rng.choice(op, min(2000, op.size), replace=False) if op.size else np.array([], int)
            yr = rng.integers(0, len(U), sel_f.size + sel_o.size)
            xf = [U[i][j] for i, j in zip(yr[:sel_f.size], sel_f) if OK[i][j]]
            xo = [U[i][j] for i, j in zip(yr[sel_f.size:], sel_o) if OK[i][j]]
            for xx, lab in ((xs, 1), (xf, 1), (xo, 0)):
                if xx:
                    rows[k]["x"].append(np.stack(xx))
                    rows[k]["y"].append(np.full(len(xx), lab, np.int8))
                    rows[k]["sq"].append(np.full(len(xx), n))
            rows[k].setdefault("loss_x", []).append((n, np.stack(xs) if xs else np.zeros((0, DIMS[k]), np.float16)))
            del U
        del z
    # open ground from the vegetation squares (never test squares here)
    for n, kind in veg.items():
        if kind in ("closed-forest", "savanna"):
            continue
        z = np.load(SQ / f"{n}.npz")
        wc = z["wc2021"]
        cand = np.flatnonzero((wc > 0) & (wc != 10) & (z["cover"] < 10))
        if not cand.size:
            continue
        sel = rng.choice(cand, min(1500, cand.size), replace=False)
        for k in EMB:
            yi = int(rng.integers(0, len(YEARS) - 1))
            x, ok = year_feats(z, k, yi)
            u = unit(x[sel])[ok[sel]].astype(np.float16)
            rows[k]["x"].append(u)
            rows[k]["y"].append(np.zeros(len(u), np.int8))
            rows[k]["sq"].append(np.full(len(u), "veg:" + n))
        del z
    for k in EMB:
        rows[k]["x"] = np.concatenate(rows[k]["x"])
        rows[k]["y"] = np.concatenate(rows[k]["y"])
        rows[k]["sq"] = np.concatenate(rows[k]["sq"])
    return rows


def fit_gates(rows, fold_of, nfold=5):
    """Per embedding, per fold: (scaler mean, scale, coef, b, cutoff) fit without that fold's squares."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {}
    for k in EMB:
        X, y, s = rows[k]["x"].astype(np.float32), rows[k]["y"], rows[k]["sq"]
        fold = np.array([fold_of.get(v, -1) for v in s])  # -1: vegetation squares, always in training
        loss_x = dict(rows[k]["loss_x"])
        for f in range(nfold):
            tr = fold != f
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            lx = np.concatenate([v for n, v in loss_x.items() if fold_of[n] != f and len(v)]).astype(np.float32)
            g = m.decision_function(sc.transform(lx))
            out[(k, f)] = (sc.mean_, sc.scale_, m.coef_[0], float(m.intercept_[0]), float(np.quantile(g, 0.05)))
    return out


def fold_heads(names):
    """The held-out fold head of biomass_head.py for each square, per embedding."""
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    rn = list(R["names"])
    fold = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold[rn[s]] = fname
    heads = {}
    for k in EMB:
        W = np.load(HERE / "data" / "heads" / f"{k}_square.npz")
        for n in names:
            net = bh.Head(DIMS[k], np.zeros(DIMS[k], np.float32), np.ones(DIMS[k], np.float32))
            net.load_state_dict({kk.split(".", 1)[1]: torch.as_tensor(W[kk]) for kk in W.files
                                 if kk.startswith(fold[n] + ".")})
            heads[(k, n)] = net.eval()
    return heads


@torch.no_grad()
def height_of(net, x):
    return np.concatenate([np.expm1(net(torch.as_tensor(x[b:b + 100000]))[:, 1].numpy())
                           for b in range(0, len(x), 100000)])


def score_flags(F, OK, L, forest, open_, cover, steady_any):
    """found / forest lit / open lit, and by cover band."""
    cand = np.array(CAND)
    col = np.arange(len(L))
    anyf = F.any(0)
    hl = (L > 0) & OK
    near = np.zeros(len(L), bool)
    for d in (-1, 0, 1):
        Ly = L + d
        idx = np.clip(np.searchsorted(cand, Ly), 0, len(CAND) - 1)
        near |= ((Ly >= CAND[0]) & (Ly <= CAND[-1])) & F[idx, col]
    found = hl & near
    o = {"loss_px": int(hl.sum()), "found": float(found.sum() / max(hl.sum(), 1)),
         "forest_lit": float((anyf & forest & OK).sum() / max((forest & OK).sum(), 1)),
         "open_lit": float((anyf & open_ & OK).sum() / max((open_ & OK).sum(), 1)), "bands": {}}
    for lo, hi in BANDS:
        b = (cover >= lo) & (cover < hi)
        st = steady_any & b & OK
        o["bands"][f"{lo}-{min(hi, 100)}"] = {"loss_px": int((hl & b).sum()), "found": float((found & b).sum() / max((hl & b).sum(), 1)),
                                               "steady_px": int(st.sum()), "lit": float((anyf & st).sum() / max(st.sum(), 1))}
    o["_found_mask"] = found
    return o


def cmd_gates(args):
    peak = guard(args.max_gb)
    meta = json.load(open(HERE / "data" / "squares.json"))
    names = [n for n in meta if (SQ / f"{n}.npz").exists()]
    drivers = {n: meta[n]["driver"] for n in names}
    veg = {n: v["kind"] for n, v in json.load(open(HERE / "data" / "veg_squares.json")).items() if (SQ / f"{n}.npz").exists()}
    fold_of = square_folds(names, drivers)
    rng = np.random.default_rng(0)
    t0 = time.time()
    rows = collect_gate_rows(names, veg, rng)
    print(f"gate rows: " + ", ".join(f"{k} {len(rows[k]['y'])} ({rows[k]['y'].mean():.2f} forest)" for k in EMB)
          + f"  [{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
    gates = fit_gates(rows, fold_of)
    del rows
    D = dict(np.load(HERE / "models" / "detectors.npz"))
    heads = fold_heads(names)
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    for n in names:
        z = np.load(SQ / f"{n}.npz")
        L, cover, forest, open_, steady_any = ground(z)
        flags = {v: {} for v in VARIANTS}
        oks = {}
        for k in EMB:
            Us, Hs, OKs = [], [], []
            for yi in range(len(YEARS)):
                x, ok = year_feats(z, k, yi)
                Us.append(unit(x).astype(np.float16))
                OKs.append(ok)
                Hs.append(height_of(heads[(k, n)], x).astype(np.float16) if yi < len(YEARS) - 2 else None)
            ok_all = np.all(OKs, 0)
            oks[k] = ok_all
            mu, scl, coef, b, at = gates[(k, fold_of[n])]
            cut = float(D[f"{k}_cut"])
            scale = np.clip(cover / 60, 0.35, 1.0)
            out = {v: np.zeros((len(CAND), len(L)), bool) for v in VARIANTS}
            for i, Y in enumerate(CAND):
                a = Us[YEARS.index(Y - 1)].astype(np.float32)
                bb = Us[YEARS.index(Y + 1)].astype(np.float32)
                plain = 1 - (a * bb).sum(1)
                g_fz = frozen_gate(D, k, a) >= D[f"{k}_gate_at"]
                g_rf = ((a - mu) / scl) @ coef + b >= at
                g_h = Hs[YEARS.index(Y - 1)].astype(np.float32) >= 3.0
                raw = plain >= cut
                out["frozen"][i] = raw & g_fz
                out["ungated"][i] = raw
                out["height"][i] = raw & g_h
                out["refit"][i] = raw & g_rf
                out["scaled"][i] = (plain >= cut * scale) & g_rf
                out["lower"][i] = (plain >= cut * 0.7) & g_rf
            for v in VARIANTS:
                flags[v][k] = out[v] & ok_all
            del Us, Hs
        nvalid = sum(oks[k].astype(int) for k in EMB)
        res = {"driver": drivers[n], "dry_open": n in DRY}
        for v in VARIANTS:
            flags[v]["vote"] = (flags[v]["aef"].astype(int) + flags[v]["tes"] + flags[v]["olmo"]) >= 2
            res[v] = {}
            for k in EMB + ("vote",):
                OK = oks[k] if k != "vote" else nvalid >= 2
                s = score_flags(flags[v][k], OK, L, forest, open_, cover, steady_any)
                s.pop("_found_mask")
                res[v][k] = s
        if n in S2_SQUARES:
            f = flags["frozen"]["vote"]
            first = np.where(f.any(0), np.array(CAND)[f.argmax(0)], 0)
            np.savez_compressed(OUT / f"{n}_flags.npz", flag_year=first.astype(np.int16), loss=z["loss"], cover=z["cover"],
                                rc=z["rc"], forest=forest, valid=nvalid >= 2)
        per[n] = res
        fv, rv = res["frozen"]["vote"], res["refit"]["vote"]
        print(f"{n:22s} {drivers[n]:10s} vote found frozen {fv['found']:.2f} height {res['height']['vote']['found']:.2f} "
              f"refit {rv['found']:.2f} scaled {res['scaled']['vote']['found']:.2f} | forest lit {fv['forest_lit']:.3f} "
              f"{res['height']['vote']['forest_lit']:.3f} {rv['forest_lit']:.3f} {res['scaled']['vote']['forest_lit']:.3f}  "
              f"[{time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)
        del z
    summary = summarize(per)
    out = json.load(open(RES)) if RES.exists() else {}
    out["gates"] = {"per_square": per, "summary": summary, "peak_gb": round(peak[0], 1)}
    json.dump(out, open(RES, "w"), indent=1)
    print(json.dumps(summary["median"], indent=None)[:3000], flush=True)


def summarize(per):
    groups = {"all": list(per), "dry_open": [n for n in per if per[n]["dry_open"]],
              "other": [n for n in per if not per[n]["dry_open"]]}
    for d in sorted({v["driver"] for v in per.values()}):
        groups[d] = [n for n in per if per[n]["driver"] == d]
    med = {}
    for g, ns in groups.items():
        med[g] = {}
        for v in VARIANTS:
            med[g][v] = {}
            for k in EMB + ("vote",):
                med[g][v][k] = {m: round(float(np.median([per[n][v][k][m] for n in ns])), 3) for m in ("found", "forest_lit", "open_lit")}
    # bands pooled over squares (pixel-weighted)
    bands = {}
    for v in VARIANTS:
        bands[v] = {}
        for k in EMB + ("vote",):
            bands[v][k] = {}
            for b in per[next(iter(per))][v][k]["bands"]:
                lp = sum(per[n][v][k]["bands"][b]["loss_px"] for n in per)
                fd = sum(per[n][v][k]["bands"][b]["found"] * per[n][v][k]["bands"][b]["loss_px"] for n in per)
                sp = sum(per[n][v][k]["bands"][b]["steady_px"] for n in per)
                lt = sum(per[n][v][k]["bands"][b]["lit"] * per[n][v][k]["bands"][b]["steady_px"] for n in per)
                bands[v][k][b] = {"loss_px": lp, "found": round(fd / max(lp, 1), 3), "steady_px": sp, "lit": round(lt / max(sp, 1), 3)}
    return {"median": med, "bands_pooled": bands, "groups": {g: len(ns) for g, ns in groups.items()}}


# ---------- fire ----------
GLAD = "https://glad.umd.edu/users/Alexandra/Fire_GFL_data/2001-25/{}_fire_forest_loss_2001-25.tif"
REGIONS = ("AFR", "EUR", "LAM", "NAM", "SEA-AUS")


def cmd_fire(args):
    import rasterio
    from rasterio.windows import from_bounds
    peak = guard(args.max_gb)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3")
    bounds = {}
    with rasterio.Env(**env):
        for r in REGIONS:
            with rasterio.open("/vsicurl/" + GLAD.format(r)) as s:
                bounds[r] = tuple(s.bounds)
    print("regions", bounds, flush=True)
    meta = json.load(open(HERE / "data" / "squares.json"))
    res = json.load(open(RES)) if RES.exists() else {}
    per_gates = res.get("gates", {}).get("per_square")
    D = dict(np.load(HERE / "models" / "detectors.npz"))
    out = {}
    for n, v in meta.items():
        z = np.load(SQ / f"{n}.npz")
        rc = z["rc"]
        lon = -180 + (np.arange(rc[2], rc[3]) + 0.5) * 8.983111749910169e-05
        lat = 83.68570533713473 - (np.arange(rc[0], rc[1]) + 0.5) * 8.983111749910169e-05
        cx, cy = float(lon.mean()), float(lat.mean())
        reg = [r for r, b in bounds.items() if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if not reg:
            print(f"{n}: no GLAD fire region", flush=True)
            continue
        fire = None
        for r in reg:
            with rasterio.Env(**env), rasterio.open("/vsicurl/" + GLAD.format(r)) as s:
                w = from_bounds(lon.min() - 0.001, lat.min() - 0.001, lon.max() + 0.001, lat.max() + 0.001, s.transform)
                a = s.read(1, window=w, boundless=True, fill_value=0)
                tr = s.window_transform(w)
            rr = np.clip(((lat - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
            cc = np.clip(((lon - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
            f = a[rr][:, cc].ravel()
            fire = f if fire is None else np.maximum(fire, f)
        L, cover, forest, open_, _ = ground(z)
        vals, cnt = np.unique(fire, return_counts=True)
        # flags as frozen, per embedding and vote (recomputed here: plain change and frozen gate)
        F, OKS = {}, {}
        for k in EMB:
            U, OK = [], []
            for yi in range(len(YEARS)):
                x, ok = year_feats(z, k, yi)
                U.append(unit(x).astype(np.float16))
                OK.append(ok)
            OKS[k] = np.all(OK, 0)
            fl = np.zeros((len(CAND), len(L)), bool)
            for i, Y in enumerate(CAND):
                a_, b_ = U[YEARS.index(Y - 1)].astype(np.float32), U[YEARS.index(Y + 1)].astype(np.float32)
                fl[i] = ((1 - (a_ * b_).sum(1)) >= float(D[f"{k}_cut"])) & (frozen_gate(D, k, a_) >= D[f"{k}_gate_at"])
            F[k] = fl & OKS[k]
            del U
        F["vote"] = (F["aef"].astype(int) + F["tes"] + F["olmo"]) >= 2
        OKS["vote"] = sum(OKS[k].astype(int) for k in EMB) >= 2
        # GLAD codes: 1 other (non-fire) loss, 2 to 4 fire at low, medium, high certainty, 5 all fire loss in Africa,
        # 0 no data. Fire = 3 to 5 (the map that matches GLAD's sample estimate); other = 1; 2 left out of both.
        is_fire = np.isin(fire, [3, 4, 5]) & (L > 0)
        is_other = (fire == 1) & (L > 0)
        o = {"driver": v["driver"], "glad_values": {int(a): int(c) for a, c in zip(vals, cnt)},
             "loss_px": int((L > 0).sum()), "fire_loss_px": int(is_fire.sum()),
             "other_loss_px": int(is_other.sum()),
             "fire_share_of_loss": round(float(is_fire.sum() / max((is_fire | is_other).sum(), 1)), 3)}
        for k in EMB + ("vote",):
            s_all = score_flags(F[k], OKS[k], L, forest, open_, cover, forest)
            fm = s_all.pop("_found_mask")
            hl = (L > 0) & OKS[k]
            o[k] = {"found_fire": round(float(fm[hl & is_fire].mean()), 3) if (hl & is_fire).sum() >= 100 else None,
                    "found_other": round(float(fm[hl & is_other].mean()), 3) if (hl & is_other).sum() >= 100 else None}
        out[n] = o
        print(f"{n:22s} {v['driver']:10s} fire share of loss {o['fire_share_of_loss']:.2f}  vote found fire "
              f"{o['vote']['found_fire']} other {o['vote']['found_other']}  [peak {peak[0]:.1f} GB]", flush=True)
        del z, F
    # pooled by kind of loss: driver (squares' notes) x fire (GLAD)
    pooled = {}
    for k in EMB + ("vote",):
        pooled[k] = {}
        for lab, key in (("fire", "found_fire"), ("not fire", "found_other")):
            vals_ = [(o[k][key], o["fire_loss_px"] if lab == "fire" else o["other_loss_px"]) for o in out.values() if o[k][key] is not None]
            pooled[k][lab] = round(sum(a * w for a, w in vals_) / max(sum(w for _, w in vals_), 1), 3) if vals_ else None
        for d in sorted({o["driver"] for o in out.values()}):
            vals_ = [(o[k]["found_other"], o["other_loss_px"]) for o in out.values() if o["driver"] == d and o[k]["found_other"] is not None]
            pooled[k][f"fire, {d}"] = None
            vf = [(o[k]["found_fire"], o["fire_loss_px"]) for o in out.values() if o["driver"] == d and o[k]["found_fire"] is not None]
            if vf:
                pooled[k][f"fire, {d}"] = round(sum(a * w for a, w in vf) / max(sum(w for _, w in vf), 1), 3)
            pooled[k][f"not fire, {d}"] = round(sum(a * w for a, w in vals_) / max(sum(w for _, w in vals_), 1), 3) if vals_ else None
    res = json.load(open(RES)) if RES.exists() else {}
    res["fire"] = {"per_square": out, "pooled_found": pooled, "peak_gb": round(peak[0], 1),
                   "note": "GLAD forest loss due to fire 2001-2025 on Hansen 2019-2024 loss pixels: fire = codes 3-5, other = code 1, code 2 (low certainty) left out"}
    json.dump(res, open(RES, "w"), indent=1)
    print(json.dumps(pooled, indent=1), flush=True)


# ---------- Sentinel-2 ----------
def cmd_s2(args):
    import rasterio
    from pystac_client import Client
    from rasterio.warp import transform as warp_xy
    from rasterio.windows import from_bounds
    peak = guard(args.max_gb)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="3", AWS_NO_SIGN_REQUEST="YES")
    seasons = {"riau-id": (6, 9), "nicaragua": (1, 4), "latvia": (5, 9)}
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    res = json.load(open(RES)) if RES.exists() else {}
    out = {}
    for n in S2_SQUARES:
        f = np.load(OUT / f"{n}_flags.npz")
        rc = f["rc"]
        lon = -180 + (np.arange(rc[2], rc[3]) + 0.5) * 8.983111749910169e-05
        lat = 83.68570533713473 - (np.arange(rc[0], rc[1]) + 0.5) * 8.983111749910169e-05
        LON, LAT = np.meshgrid(lon, lat)
        bbox = [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
        m0, m1 = seasons[n]
        nbr = np.full((len(YEARS), LON.size), np.nan, np.float32)
        used = {}
        for yi, y in enumerate(YEARS):
            items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{y}-{m0:02d}-01/{y}-{m1:02d}-28",
                                    query={"eo:cloud_cover": {"lt": 40}}, max_items=60).items())
            items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:args.scenes]
            stack = []
            for it in items:
                try:
                    vals = {}
                    for band in ("nir", "swir22", "scl"):
                        href = it.assets[band].href
                        with rasterio.Env(**env), rasterio.open(href) as s:
                            X, Y = warp_xy("EPSG:4326", s.crs, LON.ravel(), LAT.ravel())
                            X, Y = np.asarray(X), np.asarray(Y)
                            w = from_bounds(X.min() - 40, Y.min() - 40, X.max() + 40, Y.max() + 40, s.transform)
                            a = s.read(1, window=w, boundless=True, fill_value=0)
                            tr = s.window_transform(w)
                        rr = np.clip(((Y - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
                        cc = np.clip(((X - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
                        vals[band] = a[rr, cc]
                    clear = np.isin(vals["scl"], [4, 5, 6, 7, 11]) & (vals["nir"] > 0)  # vegetation, bare, water, unclassified, snow
                    clear &= ~np.isin(vals["scl"], [3, 8, 9, 10])
                    nir, sw = vals["nir"].astype(np.float32), vals["swir22"].astype(np.float32)
                    # baseline 04.00 on (from 2022-01-25) adds 1000 to reflectance; remove it unless already done
                    pb = str(it.properties.get("s2:processing_baseline", "00.00"))
                    if pb >= "04.00" and not it.properties.get("earthsearch:boa_offset_applied", False):
                        nir, sw = nir - 1000, sw - 1000
                    v = np.where(clear, (nir - sw) / np.maximum(nir + sw, 1), np.nan)
                    stack.append(v)
                except Exception as e:  # noqa: BLE001
                    print(f"  {n} {y} {it.id}: {e!r}", flush=True)
            if stack:
                with np.errstate(all="ignore"):
                    nbr[yi] = np.nanmedian(np.stack(stack), 0)
            used[y] = len(stack)
            print(f"  {n} {y}: {len(stack)} scenes, clear {np.isfinite(nbr[yi]).mean():.2f}  [peak {peak[0]:.1f} GB]", flush=True)
        fy, loss, valid, forest = f["flag_year"], f["loss"].astype(int), f["valid"], f["forest"]
        groups = {"flagged, no Hansen loss": (fy > 0) & (loss == 0) & valid,
                  "Hansen loss 2019-2024, flagged": (fy > 0) & (loss >= 19) & (loss <= 24) & valid,
                  "steady forest, not flagged": forest & (fy == 0) & valid}
        o = {"scenes_per_year": used}
        for g, m in groups.items():
            idx = np.flatnonzero(m)
            if not idx.size:
                o[g] = None
                continue
            ev = np.where(fy[idx] > 0, fy[idx], 2000 + loss[idx])
            ev = np.where(ev > 0, ev, 2021)  # steady forest: a middle year
            if g.startswith("steady"):
                ev = np.full(idx.size, 2021)
            b = np.array([nbr[YEARS.index(e - 1), j] for e, j in zip(ev, idx)])
            a = np.array([nbr[YEARS.index(min(e + 1, YEARS[-1])), j] for e, j in zip(ev, idx)])
            d = a - b
            ok = np.isfinite(d)
            o[g] = {"px": int(idx.size), "with_both_years": int(ok.sum()), "nbr_before": round(float(np.nanmedian(b)), 3),
                    "nbr_after": round(float(np.nanmedian(a)), 3), "drop_median": round(float(np.median(-d[ok])), 3) if ok.any() else None,
                    "share_drop_over_0.1": round(float((-d[ok] > 0.1).mean()), 3) if ok.any() else None,
                    "share_drop_over_0.2": round(float((-d[ok] > 0.2).mean()), 3) if ok.any() else None}
        out[n] = o
        print(n, json.dumps({k: v for k, v in o.items() if k != "scenes_per_year"}), flush=True)
    res = json.load(open(RES)) if RES.exists() else {}
    res["s2"] = {"per_square": out, "peak_gb": round(peak[0], 1),
                 "note": "NBR = (B8 - B12) / (B8 + B12), Sentinel-2 L2A from Earth Search, SCL clear pixels, per-year "
                         "median of the clearest scenes in the dry or growing season; before = year before the flag "
                         "(or Hansen's year), after = year after"}
    json.dump(res, open(RES, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gates", "fire", "s2"])
    ap.add_argument("--max-gb", type=float, default=10)
    ap.add_argument("--scenes", type=int, default=4)
    args = ap.parse_args()
    {"gates": cmd_gates, "fire": cmd_fire, "s2": cmd_s2}[args.cmd](args)


if __name__ == "__main__":
    main()
