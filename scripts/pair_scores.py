# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Which embeddings for the map: every input (aef, tes, olmo, the three pairs of pair_heads.py, all three) on the
tests that matter for the map's two settings, beside what each costs to read (results/embedding_bandwidth.json).

Every input is scored here the same way, so singles and pairs line up (the singles and "all" reproduce
lidar_check.py and map_check.py, which is checked in the output):
  lidar      the 18 lidar squares of section 16 (none in training), as lidar_check.py: the mean of the 8 square-fold
             heads; height R2 on every lidar epoch; on two-epoch squares canopy lost (dH <= -5 m, stood 5 m) found,
             gained (dH >= 3 m) found, steady (|dH| < 1 m) lit, plain and gated (the frozen embedding cutoffs: a
             majority of the input's embeddings moved); biomass change against lidar height change per CTrees cell
  rap        a head taught on RAP per input, each of the 12 RAP squares held out in turn (map_check.py own_head, the
             same training pixels): R2 of log, rank within a square, block departures, year-to-year steps
  chaparral  the CTrees-taught heads on the 10 chaparral-map squares, 2018 to 2021 (map_check.py --label chaparral):
             rank within a square, block departures, steps
  state      from biomass_head.json (singles, all) and pair_heads.json (pairs): biomass R2 of log against CTrees
             held out by square, height against Meta, class share right
Then one table, the read cost per square-year and per change pair, and per map setting the frontier: inputs that no
other input beats on cost while matching or beating it on every test of that setting.

Writes results/pair_heads.json (adds lidar, rap, chaparral, table, frontier). One square's features at a time and a
training table of about 0.8 GB: peak about 5 GB.

    uv run scripts/pair_scores.py [--parts lidar rap chaparral table]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402
import lidar_check as lc  # noqa: E402
import map_check as mc  # noqa: E402

RES = HERE / "results" / "pair_heads.json"
YEARS = bc.YEARS
INPUTS = ["aef", "tes", "olmo", "aef+tes", "aef+olmo", "tes+olmo", "all"]
EMB = ("aef", "tes", "olmo")


def members(inp):
    return list(EMB) if inp == "all" else inp.split("+")


def dim(inp):
    return sum(bh.DIMS[k] for k in members(inp))


def emb_year(z, k, yi):
    """float32 (n, D) and valid (n,) for one embedding, one year."""
    q = z[k][yi]
    if k == "tes":
        s = z["tes_scale"][yi]
        return q.astype(np.float32) * np.nan_to_num(s)[:, None], np.isfinite(s) & (s > 0)
    return bh.deq(q), q[:, 0] != -128


class Square:
    """One square's embeddings, converted per year on demand and kept as float16 per embedding."""

    def __init__(self, name):
        self.z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        self.cache = {}

    def emb(self, k, yi):
        if (k, yi) not in self.cache:
            x, ok = emb_year(self.z, k, yi)
            self.cache[(k, yi)] = (x.astype(np.float16), ok)
        return self.cache[(k, yi)]

    def x(self, inp, yi, m=None):
        parts, ok = [], None
        for k in members(inp):
            e, o = self.emb(k, yi)
            parts.append(e if m is None else e[m])
            ok = o if ok is None else ok & o
        return np.concatenate(parts, -1).astype(np.float32), ok

    def ok(self, inp, yi):
        ok = None
        for k in members(inp):
            o = self.emb(k, yi)[1]
            ok = o if ok is None else ok & o
        return ok


def heads_path(inp):
    return HERE / "data" / "heads" / f"{inp}_square.npz"


def part_lidar(res, peak):
    picks = json.load(open(HERE / "data" / "lidar_squares.json"))
    Dt = np.load(HERE / "models" / "detectors.npz")
    nets = {inp: lc.nets_of(heads_path(inp), dim(inp)) for inp in INPUTS}
    pool = {inp: {} for inp in INPUTS}

    def add(inp, **kv):
        for k, v in kv.items():
            pool[inp].setdefault(k, []).append(np.asarray(v))

    for name in picks:
        fl, fs, flab = (HERE / "data" / d / f"{name}.npz" for d in ("lidar", "squares", "labels"))
        if not (fl.exists() and fs.exists()):
            continue
        L = np.load(fl)
        S = Square(name)
        lab = np.load(flab) if flab.exists() else None
        chm, cov = L["chm"].astype(np.float32), L["cover"]
        src = str(L["source"])
        if src == "werk10":
            ypx = L["year_px"]
            ok_l = (cov[0] >= 80) & (ypx >= 2018) & (ypx <= 2023)
        else:
            years = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in L["year"]]
            ok_l = np.all(cov >= 80, 0)
        if ok_l.sum() < 2000:
            continue
        for inp in INPUTS:
            if src == "werk10":
                hp = np.full(chm.shape[1], np.nan, np.float32)
                for y in range(2018, 2024):
                    yi = YEARS.index(y)
                    m = ok_l & (ypx == y) & S.ok(inp, yi)
                    if m.sum():
                        _, h = lc.state(nets[inp], S.x(inp, yi, m)[0])
                        hp[m] = h
                m = ok_l & np.isfinite(hp)
                add(inp, t=chm[0][m], p=hp[m])
                continue
            y1, y2 = years
            i1, i2 = YEARS.index(y1), YEARS.index(y2)
            m = ok_l & S.ok(inp, i1) & S.ok(inp, i2)
            a1, h1 = lc.state(nets[inp], S.x(inp, i1, m)[0])
            a2, h2 = lc.state(nets[inp], S.x(inp, i2, m)[0])
            c1, c2 = chm[0][m], chm[1][m]
            add(inp, t=np.concatenate([c1, c2]), p=np.concatenate([h1, h2]))
            if y2 <= y1:
                continue
            dH, dh, da = c2 - c1, h2 - h1, a2 - a1
            mv = np.zeros(m.sum(), int)
            for k in members(inp):
                u1 = bc.unit(S.emb(k, i1)[0][m].astype(np.float32))
                u2 = bc.unit(S.emb(k, i2)[0][m].astype(np.float32))
                mv += (1 - (u1 * u2).sum(-1)) >= float(Dt[f"{k}_cut"])
            moved = mv >= (len(members(inp)) // 2 + 1)
            lost = (dH <= -5) & (c1 >= 5)
            gain = dH >= 3
            steady = (np.abs(dH) < 1) & (np.abs(dH) < 0.3 * np.maximum(c1, 1))
            seen_l = (dh <= -2.5) | ((da <= -10) & (da <= -0.25 * np.maximum(a1, 1)))
            seen_g = (dh >= 1.5) | ((da >= 5) & (da >= 0.25 * np.maximum(a1, 1)))
            add(inp, lost=lost, gain=gain, steady=steady, seen_l=seen_l, seen_g=seen_g, moved=moved)
            if lab is not None:
                cell = lab["cell"][m]
                vH, _, _ = lc.cellmean(dH, cell, np.ones(m.sum(), bool))
                va, _, _ = lc.cellmean(da, cell, np.ones(m.sum(), bool))
                add(inp, vH=vH, va=va)
        print(f"  lidar {name} [{psutil.Process().memory_info().rss / 1e9:.1f} GB, peak {peak[0]:.1f}]", flush=True)
        del S, L
    out = {}
    for inp, d in pool.items():
        d = {k: np.concatenate(v) for k, v in d.items()}
        lo, ga, st = d["lost"], d["gain"], d["steady"]
        any_ = d["seen_l"] | d["seen_g"]
        out[inp] = {"height_r2": round(lc.r2(d["t"], d["p"]), 3),
                    "lost_found": round(float(d["seen_l"][lo].mean()), 3),
                    "lost_found_gated": round(float((d["seen_l"] & d["moved"])[lo].mean()), 3),
                    "gain_found": round(float(d["seen_g"][ga].mean()), 3),
                    "steady_lit": round(float(any_[st].mean()), 3),
                    "steady_lit_gated": round(float((any_ & d["moved"])[st].mean()), 3),
                    "cell_corr_dH": lc.corr(d["va"], d["vH"]), "cells": int(np.isfinite(d["vH"]).sum())}
        print(f"lidar {inp:9s} {json.dumps(out[inp])}", flush=True)
    res["lidar"] = out
    old = json.load(open(HERE / "results" / "lidar_check.json"))["pooled"]
    res["lidar_reproduces"] = {inp: {"height_r2": old.get(f"{inp}/state", {}).get("height_r2"),
                                     "lost_found": old.get(f"{inp}/change", {}).get("lost_found"),
                                     "cell_corr_dH": old.get(f"{inp}/cells", {}).get("corr_head_dH")}
                               for inp in ("aef", "tes", "olmo", "all")}


def part_rap(res, peak, px=12000):
    names = mc.squares_of("rap")
    rng = np.random.default_rng(0)
    Tb, tab, tab_t, tab_s, lab_years = [], {k: [] for k in EMB}, [], [], None
    for si, name in enumerate(names):
        lab, lab_years = mc.label_of("rap", name)
        yi = [YEARS.index(y) for y in lab_years]
        S = Square(name)
        okE = np.stack([S.ok("all", y) for y in yi])
        tb, tv = mc.blocks(np.where(okE, lab, np.nan))
        tb[tv < 0.8] = np.nan
        Tb.append(tb)
        for k, y in enumerate(yi):
            cand = np.flatnonzero(okE[k] & np.isfinite(lab[k]))
            pick = np.sort(rng.choice(cand, min(px, cand.size), replace=False))
            for e in EMB:
                tab[e].append(S.emb(e, y)[0][pick])
            tab_t.append(lab[k][pick])
            tab_s.append(np.full(pick.size, si, np.int16))
        del S
    Tb = np.stack(Tb)
    tab = {e: np.concatenate(v) for e, v in tab.items()}
    ts_, ss_ = np.concatenate(tab_t), np.concatenate(tab_s)
    nets = {inp: [] for inp in INPUTS}
    for inp in INPUTS:
        X = np.concatenate([tab[e] for e in members(inp)], -1)
        for si in range(len(names)):
            nets[inp].append(mc.fit_one(X[ss_ != si], ts_[ss_ != si]))
        del X
        print(f"  rap fits {inp} [{psutil.Process().memory_info().rss / 1e9:.1f} GB, peak {peak[0]:.1f}]", flush=True)
    del tab
    own = {inp: [] for inp in INPUTS}
    for si, name in enumerate(names):
        S = Square(name)
        yi = [YEARS.index(y) for y in lab_years]
        okE = np.stack([S.ok("all", y) for y in yi])
        for inp in INPUTS:
            pred = np.full(okE.shape, np.nan, np.float32)
            for k, y in enumerate(yi):
                pred[k, okE[k]] = mc.run_one(nets[inp][si], S.x(inp, y, okE[k])[0])
            pb, pv = mc.blocks(pred)
            pb[pv < 0.8] = np.nan
            own[inp].append(pb)
        del S
    res["rap"] = {}
    old = json.load(open(HERE / "results" / "map_check_rap.json"))["own_head"]
    for inp in INPUTS:
        s = mc.score(Tb, np.stack(own[inp]), names, 0.25)
        res["rap"][inp] = {k: s[k] for k in ("r2_log", "median_error", "rank_within_square_median", "departure_corr",
                                             "step_corr", "square_series_corr_median")} | \
            {"swings_found": s["swings"]["found_same_way"], "steady_lit": s["swings"]["steady_lit"]}
        if inp in old:
            res["rap"][inp]["reproduces_departure_corr"] = old[inp]["departure_corr"]
        print(f"rap {inp:9s} {json.dumps(res['rap'][inp])}", flush=True)


def part_chaparral(res, peak):
    names = mc.squares_of("chaparral")
    R = np.load(HERE / "data" / "rows.npz")
    train_names = set(R["names"])
    nets = {inp: lc.nets_of(heads_path(inp), dim(inp)) for inp in INPUTS}
    Tb, Hb = [], {inp: [] for inp in INPUTS}
    for name in names:
        if name in train_names:
            raise SystemExit(f"{name} is in the training rows; use its fold head")
        lab, ly = mc.label_of("chaparral", name)
        keep = [i for i, y in enumerate(ly) if 2018 <= y <= 2021]
        lab, ly = lab[keep], [ly[i] for i in keep]
        yi = [YEARS.index(y) for y in ly]
        S = Square(name)
        okE = np.stack([S.ok("all", y) for y in yi])
        tb, tv = mc.blocks(np.where(okE, lab, np.nan))
        tb[tv < 0.8] = np.nan
        Tb.append(tb)
        for inp in INPUTS:
            pred = np.full(okE.shape, np.nan, np.float32)
            for k, y in enumerate(yi):
                a, _ = lc.state(nets[inp], S.x(inp, y, okE[k])[0])
                pred[k, okE[k]] = a
            pb, pv = mc.blocks(pred)
            pb[pv < 0.8] = np.nan
            Hb[inp].append(pb)
        del S
        print(f"  chaparral {name} [peak {peak[0]:.1f} GB]", flush=True)
    Tb = np.stack(Tb)
    res["chaparral"] = {}
    old = json.load(open(HERE / "results" / "map_check_chaparral.json"))["ctrees_head"]
    for inp in INPUTS:
        s = mc.score(Tb, np.stack(Hb[inp]), names, 5.0)
        res["chaparral"][inp] = {k: s[k] for k in ("rank_within_square_median", "rank_square_medians",
                                                   "departure_corr", "step_corr", "median_error")}
        if inp in old:
            res["chaparral"][inp]["reproduces_departure_corr"] = old[inp]["departure_corr"]
        print(f"chaparral {inp:9s} {json.dumps(res['chaparral'][inp])}", flush=True)


# map settings: (test name, source, key, higher is better)
SETTINGS = {
    "change": [("lidar canopy lost found", "lidar", "lost_found", True),
               ("lidar canopy gained found", "lidar", "gain_found", True),
               ("lidar steady lit, gated", "lidar", "steady_lit_gated", False),
               ("biomass change vs lidar, per cell", "lidar", "cell_corr_dH", True),
               ("grass block departures (RAP head)", "rap", "departure_corr", True),
               ("grass year steps (RAP head)", "rap", "step_corr", True),
               ("chaparral departures (CTrees head)", "chaparral", "departure_corr", True)],
    "each_year": [("biomass R2 of log vs CTrees", "state", "agb_r2_log", True),
                  ("height R2 vs Meta", "state", "height_r2", True),
                  ("class share right", "state", "cls_acc", True),
                  ("height R2 vs lidar", "lidar", "height_r2", True),
                  ("grass R2 of log (RAP head)", "rap", "r2_log", True),
                  ("grass rank within square (RAP head)", "rap", "rank_within_square_median", True),
                  ("chaparral rank within square (CTrees head)", "chaparral", "rank_within_square_median", True)],
}


def part_table(res):
    bw = json.load(open(HERE / "results" / "embedding_bandwidth.json"))
    places = list(bw)
    cost = {inp: round(float(np.median([sum(bw[p][k]["MB"] for k in members(inp)) for p in places])), 1)
            for inp in INPUTS}
    old = json.load(open(HERE / "results" / "biomass_head.json"))
    state = {inp: old[inp]["square"]["all"] for inp in ("aef", "tes", "olmo", "all")}
    state |= {inp: res["state"][inp]["all"] for inp in res.get("state", {})}
    src = {"state": state, "lidar": res.get("lidar", {}), "rap": res.get("rap", {}), "chaparral": res.get("chaparral", {})}
    table = {inp: {"MB_per_square_year": cost[inp], "MB_per_change_pair": round(2 * cost[inp], 1)} for inp in INPUTS}
    for setting, tests in SETTINGS.items():
        for label, s, key, _ in tests:
            for inp in INPUTS:
                table[inp][label] = src[s].get(inp, {}).get(key)
    res["table"] = table
    res["frontier"] = {}
    for setting, tests in SETTINGS.items():
        def better_eq(a, b):  # a at least as good as b on every test
            for label, _, _, hi in tests:
                va, vb = table[a][label], table[b][label]
                if va is None or vb is None:
                    continue
                if (va < vb) if hi else (va > vb):
                    return False
            return True
        front = []
        for a in INPUTS:
            dominated = any(b != a and cost[b] <= cost[a] and better_eq(b, a) and
                            (cost[b] < cost[a] or any(table[b][l] != table[a][l] for l, *_ in tests)) for b in INPUTS)
            if not dominated:
                front.append(a)
        res["frontier"][setting] = sorted(front, key=lambda k: cost[k])
    res["cost_MB_per_square_year"] = cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["lidar", "rap", "chaparral", "table"])
    ap.add_argument("--max-gb", type=float, default=8)
    args = ap.parse_args()
    p, peak = psutil.Process(), [0.0]

    def guard():
        while True:
            gb = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], gb)
            if gb > args.max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
    res = json.load(open(RES)) if RES.exists() else {}
    t0 = time.time()
    for part in args.parts:
        if part == "lidar":
            part_lidar(res, peak)
        elif part == "rap":
            part_rap(res, peak)
        elif part == "chaparral":
            part_chaparral(res, peak)
        elif part == "table":
            part_table(res)
            for inp, row in res["table"].items():
                print(f"{inp:9s} " + "  ".join(f"{k}: {v}" for k, v in row.items()), flush=True)
            print("frontier", json.dumps(res["frontier"]), flush=True)
        res[f"peak_gb_{part}"] = round(peak[0], 1)
        json.dump(res, open(RES, "w"), indent=1)
        print(f"[{part} done, {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB]", flush=True)


if __name__ == "__main__":
    main()
