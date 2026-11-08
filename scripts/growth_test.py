"""The two ways to a stronger growth signal, under FINDINGS 30's harness: the trajectory head (all eight years of a
pixel at once, FINDINGS 14) and TESSERA (FINDINGS 25), against the one-year AlphaEarth head the map runs.

Every head is held out: a square in data/rows.npz is scored by the fold that never saw it, any other square (the
lidar squares, the Esri-only squares) by fold 0, which never saw it either. The map's own head ("map", taught on
all 92 squares) is scored too, as the reference FINDINGS 30 used. Every head is scored on the same pixels: valid in
AlphaEarth and TESSERA both, in both years of the window.

Heads (--head):
  map            the browser's head (web/public/models/biomass_aef.json), one year of AlphaEarth
  aef tes aef+tes  one-year heads, held out (data/heads/<inp>_square.npz, biomass_head.py / pair_heads.py)
  traj_aef traj_tes traj_aef+tes  trajectory heads, held out (data/heads/traj_<inp>.npz, trajectory_head.py)

For each head: the steady-ground tables from that head's own change (section 29's steady rule), the variants of
change_confidence_test.py, and the same scores (Esri's changes on every square, airborne lidar per pixel and per
100 m cell). The embedding move (loss) is AlphaEarth's for every head, as on the map.

For the map's head, two more things:
  - the confidence keyed on the head's own class (its argmax of nine), not Esri's, as the browser has it;
  - the tables the browser reads, built from every square's steady pixels keyed on the head's class:
    web/public/models/change_conf.json (with --write-tables).

Writes results/growth_test.json (one entry per head, merged). RAM about 7 GB per head; stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests --with torch python scripts/growth_test.py --head map --write-tables
"""
import argparse
import json
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402
import burn_check as bc  # noqa: E402  (the map's head, as the browser runs it)
import change_confidence_test as cct  # noqa: E402  (the harness of FINDINGS 30)
import nlcd_check as nc  # noqa: E402
import trajectory_head as th  # noqa: E402

OUT = HERE / "results" / "growth_test.json"
TABLES = HERE / "web" / "public" / "models" / "change_conf.json"
YEARS = cct.YEARS
NY = cct.NY
CLASSES = ["trees", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "moss"]


def fold_map():
    R = np.load(HERE / "data" / "rows.npz")
    names = list(R["names"])
    Rd = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    out = {}
    for fname, test in bh.folds(Rd, "square"):
        for s in np.unique(Rd["sq"][test]):
            out[names[s]] = fname
    return out


def emb(z, key):
    """float32 (Y, n, D) and valid (Y, n), one embedding of one square."""
    q = z[key]
    if key == "tes":
        s = z["tes_scale"]
        ok = np.isfinite(s) & (s > 0)
        e = q.astype(np.float32) * np.nan_to_num(s)[..., None]
    else:
        ok = q[..., 0] != -128
        e = bh.deq(q)
    return e.astype(np.float16), ok


class OneYear:
    def __init__(self, inp):
        self.inp = inp
        W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
        self.nets = {}
        for f in {k.split(".")[0] for k in W.files if k[0].isdigit()}:
            sd = {k[len(f) + 1:]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f + ".")}
            net = bh.Head(sd["mu"].numel(), sd["mu"].numpy(), sd["sd"].numpy())
            net.load_state_dict(sd)
            self.nets[f] = net.eval()

    @torch.no_grad()
    def __call__(self, X, fold):
        net = self.nets[fold]
        Y, n, _ = X.shape
        agb = np.empty((Y, n), np.float32)
        cls = np.empty((Y, n), np.int8)
        for y in range(Y):
            for b in range(0, n, 60000):
                o = net(torch.as_tensor(X[y, b:b + 60000].astype(np.float32))).numpy()
                agb[y, b:b + 60000] = np.clip(np.expm1(np.minimum(o[:, 0], 8)), 0, None)
                cls[y, b:b + 60000] = o[:, 2:].argmax(1)
        return agb, cls


class Trajectory:
    def __init__(self, inp):
        W = np.load(HERE / "data" / "heads" / f"traj_{inp}.npz")
        self.nets = {}
        for f in {k.split(".")[0] for k in W.files}:
            sd = {k[len(f) + 1:]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f + ".")}
            net = th.Traj(sd["mu"].numel(), sd["mu"].numpy(), sd["sd"].numpy())
            net.load_state_dict(sd)
            self.nets[f] = net.eval()

    @torch.no_grad()
    def __call__(self, X, ok, fold):
        net = self.nets[fold]
        Y, n, _ = X.shape
        agb = np.empty((Y, n), np.float32)
        for b in range(0, n, 30000):
            xt = torch.as_tensor(X[:, b:b + 30000].astype(np.float32)).permute(1, 0, 2)
            vt = torch.as_tensor(ok[:, b:b + 30000].T.copy())
            o = net(xt, vt).numpy().T
            agb[:, b:b + 30000] = np.clip(np.expm1(np.minimum(o, 8)), 0, None)
        return agb, None


def head_of(name):
    if name == "map":
        return None
    if name.startswith("traj_"):
        return Trajectory(name[5:])
    return OneYear(name)


def predict(head_name, head, z, fold):
    """(agb (Y, n) with NaN where the head has no input, head class (Y, n) or None)."""
    inp = "aef" if head_name == "map" else head_name.replace("traj_", "")
    keys = inp.split("+")
    if head_name == "map":
        q = z["aef"]
        ok = q[..., 0] != -128
        agb = np.full(ok.shape, np.nan, np.float32)
        cls = np.zeros(ok.shape, np.int8)
        for y in range(q.shape[0]):
            raw = bc.map_raw(q[y, ok[y]])
            agb[y, ok[y]] = np.expm1(np.clip(raw[:, 0], 0, 8))
            cls[y, ok[y]] = raw[:, 2:].argmax(1)
        return agb, cls
    parts = [emb(z, k) for k in keys]
    X = parts[0][0] if len(parts) == 1 else np.concatenate([p[0] for p in parts], -1)
    ok = parts[0][1].copy()
    for p in parts[1:]:
        ok &= p[1]
    del parts
    agb, cls = head(X, ok, fold) if head_name.startswith("traj_") else head(X, fold)
    agb[~ok] = np.nan
    return agb, cls


def collect(head_name, head, folds_of, sq, rows, lid, tab_rows):
    rng = np.random.default_rng(0)
    t0 = time.time()
    for si, name in enumerate(sorted(sq)):
        fi = HERE / "data" / "io" / f"{name}.npz"
        if not fi.exists():
            continue
        io = np.load(fi)["lndcov"][:NY]
        z = np.load(HERE / "data" / "squares" / f"{name}.npz")
        q = z["aef"]
        if q.shape[1] != 360000:
            continue
        s = z["tes_scale"]
        ok = (q[:, :, 0] != -128) & np.isfinite(s) & (s > 0)  # the same pixels for every head
        agb, hcls = predict(head_name, head, z, folds_of.get(name, "0"))
        ok &= np.isfinite(agb)
        U = np.stack([cct.unit(q[i]) for i in range(q.shape[0])])
        del q, z
        ct = nc.labels_of(name)["ctrees"]
        fold = zlib.crc32(name.encode()) % cct.FOLDS
        ks = np.zeros(360000, bool)
        ks[rng.integers(cct.STEP_STEADY)::cct.STEP_STEADY] = True
        kr = np.zeros(360000, bool)
        kr[rng.integers(cct.STEP_REAL)::cct.STEP_REAL] = True
        for a in range(NY - 1):
            for b in range(a + 1, NY):
                valid = ok[a] & ok[b]
                f = cct.feats(agb[a], agb[b], U[a], U[b], valid)
                fin = valid & np.isfinite(f["dbar"]) & np.isfinite(f["dblk"])
                ct_st = np.isfinite(ct[a]) & np.isfinite(ct[b]) & (np.abs(ct[b] - ct[a]) < 10) & \
                    (np.abs(ct[b] - ct[a]) < 0.1 * np.maximum(ct[a], 1))
                stable = np.all(io[a:b + 1] == io[a], 0) & (io[a] != 250)
                sets = {0: fin & stable & ct_st & ks,
                        1: fin & kr & (io[a] == 2) & np.isin(io[b], [8, 11]),
                        2: fin & kr & np.isin(io[a], [8, 11]) & (io[b] == 2)}
                for s_, m in sets.items():
                    if not m.any():
                        continue
                    r = {k: v[m].astype(np.float32) for k, v in f.items()}
                    r |= {"set": np.full(m.sum(), s_, np.int8), "L": np.full(m.sum(), b - a, np.int8),
                          "cls": io[a][m].astype(np.int16), "fold": np.full(m.sum(), fold, np.int8),
                          "hcls": (hcls[a][m] if hcls is not None else np.zeros(m.sum())).astype(np.int16)}
                    rows.append(r)
                if tab_rows is not None:
                    # the browser's tables: steady pixels at three times the sampling, keyed on the head's class
                    m = fin & stable & ct_st & (ks | np.roll(ks, 1) | np.roll(ks, 2))
                    tab_rows.append({"neg_dpct": np.maximum(-f["dpct"][m], 0).astype(np.float32),
                                     "pos_d": np.maximum(f["d"][m], 0).astype(np.float32),
                                     "mv": f["mv"][m].astype(np.float32),
                                     "L": np.full(m.sum(), b - a, np.int8), "hcls": hcls[a][m].astype(np.int8)})
                del f
        if name in cct.LIDAR:
            Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
            y1, y2 = [min(max(int(y), YEARS[0]), YEARS[-1]) for y in Ld["year"]]
            i1, i2 = YEARS.index(y1), YEARS.index(y2)
            chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
            m = np.all(cov >= 80, 0) & ok[i1] & ok[i2]
            f = cct.feats(agb[i1], agb[i2], U[i1], U[i2], ok[i1] & ok[i2])
            m &= np.isfinite(f["dbar"]) & np.isfinite(f["dblk"])
            cls = np.load(fi)["lndcov"][min(i1, NY - 1)]
            cell = np.load(HERE / "data" / "labels" / f"{name}.npz")["cell"]
            lid.append({**{k: v[m].astype(np.float32) for k, v in f.items()}, "dH": (chm[1] - chm[0])[m], "h1": chm[0][m],
                        "L": np.full(m.sum(), i2 - i1, np.int8), "cls": cls[m].astype(np.int16),
                        "hcls": (hcls[i1][m] if hcls is not None else np.zeros(m.sum())).astype(np.int16),
                        "fold": np.full(m.sum(), fold, np.int8), "cell": cell[m] + 10_000_000 * len(lid), "name": name})
            print(f"  lidar {name} {y1}-{y2}: {m.sum()} px", flush=True)
        del U, agb
        print(f"square {si + 1} of {len(sq)}: {name} [{time.time() - t0:.0f} s, "
              f"{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)


def with_split(R):
    for k in ("d", "dpct", "dbar", "dbarpct", "dblk", "dblkpct"):
        R[f"neg_{k}"] = np.maximum(-R[k], 0)
        R[f"pos_{k}"] = np.maximum(R[k], 0)
    return R


def head_class_variants(R, S):
    """The browser's two confidences, keyed on the head's class (not Esri's)."""
    Rh, Sh = dict(R), dict(S)
    Rh["cls"], Sh["cls"] = R["hcls"], S["hcls"]
    T = cct.Tables(Sh)
    mvc = T.conf(Rh, "mv", Rh["mv"], True)
    loss = np.minimum(T.conf(Rh, "neg_dpct", Rh["neg_dpct"], True), mvc)
    gain = T.conf(Rh, "pos_d", Rh["pos_d"], True)
    return {"loss cls_pct_mv (head class)": loss, "gain cls_mg px (head class)": gain}


def write_tables(tab_rows):
    """Per head class and window length (1 to 6), the steady pixels' quantiles of the three values the browser
    judges: the fall in percent of the pixel's biomass, the rise in Mg/ha, and the embedding's move."""
    T = {k: np.concatenate([r[k] for r in tab_rows]) for k in tab_rows[0]}
    p = np.unique(np.round(np.concatenate([np.linspace(0, 0.8, 41), np.linspace(0.8, 0.99, 96), np.linspace(0.99, 1, 21)]), 5))
    out = {"about": "Steady ground's change per 10 m pixel, by the head's class at the window's first year and the "
                    "window's length in years: quantiles (q) at the shares p. A pixel's confidence is the share of "
                    "steady pixels below its value. scripts/growth_test.py, FINDINGS 29 to 31.",
           "classes": CLASSES, "p": p.tolist(), "vars": ["neg_dpct", "pos_d", "mv"], "L": list(range(1, 7)),
           "n": {}, "q": {}}
    for v in out["vars"]:
        out["q"][v] = {}
        for L in range(1, 7):
            mL = T["L"] == L
            pooled = np.quantile(T[v][mL], p)
            for ci, c in enumerate(CLASSES + ["all"]):
                m = mL & (T["hcls"] == ci) if c != "all" else mL
                n = int(m.sum())
                out["n"][f"{c}:{L}"] = n
                qv = np.quantile(T[v][m], p) if n >= 2000 else pooled
                out["q"][v][f"{c}:{L}"] = [round(float(x), 4) for x in qv]
    json.dump(out, open(TABLES, "w"), separators=(",", ":"))
    print(f"wrote {TABLES} ({TABLES.stat().st_size / 1e3:.0f} KB, {T['L'].size} steady pixels)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--write-tables", action="store_true")
    ap.add_argument("--max-gb", type=float, default=12)
    ap.add_argument("--only", nargs="*", help="a few squares, for a quick check (results not written)")
    args = ap.parse_args()
    cct.guard(args.max_gb)
    torch.set_num_threads(6)
    t0 = time.time()
    head = head_of(args.head)
    sq = nc.squares(world=True)
    if args.only:
        sq = {k: v for k, v in sq.items() if k in args.only}
        global OUT, TABLES
        OUT = Path(tempfile.gettempdir()) / "growth_test_check.json"
        TABLES = Path(tempfile.gettempdir()) / "change_conf_check.json"
    rows, lid = [], []
    tab_rows = [] if args.write_tables and args.head == "map" else None
    collect(args.head, head, fold_map(), sq, rows, lid, tab_rows)
    if tab_rows is not None:
        write_tables(tab_rows)
        del tab_rows
    R = with_split({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    del rows
    S = {k: v[R["set"] == 0] for k, v in R.items()}
    T = cct.Tables(S)
    res = {"rows": {"steady": int((R["set"] == 0).sum()), "lost": int((R["set"] == 1).sum()), "grew": int((R["set"] == 2).sum())},
           "esri": {}, "lidar_px": {}, "lidar_cell": {}}
    V = cct.variants(T, R)
    if args.head == "map":
        V |= head_class_variants(R, S)
    for k, v in V.items():
        kind = 1 if k.startswith("loss") else 2
        res["esri"][k] = cct.score(v[R["set"] == kind], v[R["set"] == 0], binary="rule" in k)
    # the size of the head's change on Esri's "became trees" and on steady trees, 90th percentile, one year
    m1 = R["L"] == 1
    res["size_p90_1yr"] = {
        "grew (Mg/ha)": round(float(np.quantile(R["d"][(R["set"] == 2) & m1], 0.9)), 1),
        "steady trees (Mg/ha)": round(float(np.quantile(np.abs(R["d"][(R["set"] == 0) & m1 & (R["cls"] == 2)]), 0.9)), 1),
        "grew (%)": round(float(np.quantile(R["dpct"][(R["set"] == 2) & m1], 0.9)), 1),
        "steady trees (%)": round(float(np.quantile(np.abs(R["dpct"][(R["set"] == 0) & m1 & (R["cls"] == 2)]), 0.9)), 1),
        "lost (%)": round(float(np.quantile(-R["dpct"][(R["set"] == 1) & m1], 0.9)), 1)}
    # the browser's display cuts on the map's head: loss drawn at confidence .9 and up, gain at the floor
    if args.head == "map":
        lc, gc = V["loss cls_pct_mv (head class)"], V["gain rule px"]
        st = R["set"] == 0
        res["display"] = {}
        for cut in (0.9, 0.95, 0.99):
            res["display"][f"loss head class >= {cut}"] = {"found": round(float((lc[R["set"] == 1] >= cut).mean()), 4),
                                                         "lit": round(float((lc[st] >= cut).mean()), 4)}
        for c, cn in ((2, "trees"), (11, "rangeland"), (5, "crops")):
            res["display"][f"steady {cn} lit, loss head class >= .9"] = round(float((lc[st & (R["cls"] == c)] >= 0.9).mean()), 4)
            res["display"][f"steady {cn} lit, gain floor"] = round(float((gc[st & (R["cls"] == c)] >= 0.5).mean()), 4)
    del V
    L = with_split({k: np.concatenate([x[k] for x in lid]) for k in lid[0] if k != "name"})
    VL = cct.variants(T, L)
    if args.head == "map":
        VL |= head_class_variants(L, S)
    lost = (L["dH"] <= -3) & (L["h1"] >= 5)
    gained = L["dH"] >= 2
    steady = np.abs(L["dH"]) < 0.5
    for k, v in VL.items():
        res["lidar_px"][k] = cct.score(v[lost if k.startswith("loss") else gained], v[steady], binary="rule" in k)
    if args.head == "map":
        lc = VL["loss cls_pct_mv (head class)"]
        for cut in (0.9, 0.95, 0.99):
            res["display"][f"lidar loss head class >= {cut}"] = {"found": round(float((lc[lost] >= cut).mean()), 4),
                                                               "lit": round(float((lc[steady] >= cut).mean()), 4)}
    ids, inv, cnt = np.unique(L["cell"], return_inverse=True, return_counts=True)
    keep = cnt >= 40
    dHc = np.bincount(inv, L["dH"]) / cnt
    pairs = [("loss rule", "gain rule (block)"), ("loss rule", "gain rule px"), ("loss cls_pct_mv", "gain rule px"),
             ("loss cls_pct_mv", "gain cls_mg px"), ("loss cls_pct", "gain cls_mg px")]
    if args.head == "map":
        pairs.append(("loss cls_pct_mv (head class)", "gain rule px"))
    for lk, gk in pairs:
        s = np.nan_to_num(VL[gk]) - np.nan_to_num(VL[lk])
        sc = np.bincount(inv, s) / cnt
        x, y = sc[keep], dHc[keep]
        lcm, gcm, stc = (y <= -3), (y >= 2), (np.abs(y) < 0.5)
        res["lidar_cell"][f"{lk} + {gk}"] = {"cells": int(keep.sum()), "corr": round(float(np.corrcoef(x, y)[0, 1]), 4),
                                             "auc_lost": cct.auc(-x[lcm], -x[stc]), "auc_gained": cct.auc(x[gcm], x[stc])}
    # the head's own rise against lidar's, per pixel, where lidar's canopy rose 2 m (the size of regrowth it sees)
    res["lidar_gained_rise"] = {"median Mg/ha": round(float(np.median(L["d"][gained])), 1),
                                "steady |change| p90 Mg/ha": round(float(np.quantile(np.abs(L["d"][steady]), 0.9)), 1),
                                "gained px": int(gained.sum()), "steady px": int(steady.sum())}
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    allr = json.load(open(OUT)) if OUT.exists() else {}
    allr[args.head] = res
    json.dump(allr, open(OUT, "w"), indent=1)

    print(f"\n=== {args.head}  {json.dumps(res['rows'])}  size p90 1 yr {json.dumps(res['size_p90_1yr'])}", flush=True)
    for title, d in (("ESRI", res["esri"]), ("LIDAR per pixel", res["lidar_px"])):
        print(f"\n{title}", flush=True)
        for k, o in d.items():
            if o is None:
                continue
            if "found" in o:
                print(f"  {k:34s} auc {o['auc']}  found {o['found']}  lit {o['lit']}", flush=True)
            else:
                print(f"  {k:34s} auc {o['auc']}  found@2%lit {o['found_at_2pct_lit']}  found@.95 {o['found_95']}  lit@.95 {o['lit_95']}", flush=True)
    print("\nLIDAR per 100 m cell", flush=True)
    for k, o in res["lidar_cell"].items():
        print(f"  {k:52s} corr {o['corr']}  auc lost {o['auc_lost']}  auc gained {o['auc_gained']}", flush=True)
    print(f"lidar gained rise {json.dumps(res['lidar_gained_rise'])}", flush=True)
    if "display" in res:
        print(f"display {json.dumps(res['display'])}", flush=True)
    print(f"\nwrote {args.head} to {OUT} in {res['seconds']} s, peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
