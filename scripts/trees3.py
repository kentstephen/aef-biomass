"""Trees cut, three embeddings side by side: AlphaEarth (AEF), TESSERA and OlmoEarth (OLMO), on the same pixels.

Repeats signal_trees.py and false_alarms_trees.py with OlmoEarth added (scripts/read_olmo.py read it under
Earthwork's pixels). Years 2018 to 2025.

Forest squares (19; Hansen GFC v1.12 loss in one event year vs forest never lost):
  detect     AUC of plain change (1 - cosine), year before vs year after the event
  when       share of cut pixels whose largest one-year change falls in the Hansen year or the year after
  back       cosine of cut pixels to their pre-cut self, each year after (regrowth), vs never-cut forest
  model      logistic regression held out by square ("cut in year Y?", pixel-years 2019 to 2024), per embedding
             and on all three
Detectors with cutoffs at 80 % of cut pixel-years found (pooled over the forest squares):
  plain_<emb>        plain change, gated by that embedding's own "forest the year before" (logistic on the
                     before-year embedding: forest squares vs the non-forest quiet squares only)
  vote               at least 2 of the 3 gated plain detectors
False alarms: share lit on Earthwork's trouble spots (quiet squares every pair, crop switches, farm and marsh
lidar, reservoirs, other lookalikes, burns), none taught.

Run with Earthwork's environment (compare.py's loader):
    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil python scripts/trees3.py
RAM: forest rows ~1 GB; the held-out fits run 6 at a time in worker processes (~2 GB each, ~15 GB together);
the Texas marsh file ~4 GB on top of the main process later. The main process stops past MAX_GB (30).
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EW = Path.home() / "dev/projects/aef-tessera-earthwork"
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EW))
os.chdir(EW)
import compare  # noqa: E402
import tessera_sites as ts  # noqa: E402

OLMO = HERE / "data" / "olmo"
OUT = HERE / "results"
YEARS = list(range(2018, 2026))
CAND = list(range(2019, 2025))
EMB = ("aef", "tes", "olmo")
FOREST = ["coast-range-or", "willapa-wa", "ware-ga", "pine-ms", "north-woods-me", "norrland-se", "kainuu-fi",
          "kaingaroa-nz", "tres-lagoas-br", "near-kolwezi-cd", "near-fort-hills-ab", "near-detour-lake-on",
          "near-adaro-id", "near-sangatta-id", "near-madre-de-dios-pe", "near-boke-gn", "near-mobimbi-sl",
          "near-weda-bay-id", "near-morowali-id"]
QUIET_FOREST = ["jau-br", "kerihun-id", "salonga-cd", "yakutia-taiga-ru", "ontario-shield-ca", "adirondack-ny-us"]
FARM = ["antelope-farm-ne", "box-butte-farm-ne", "fresno-farm-ca", "hamilton-farm-in", "arkansas-delta-ar", "othello-pivots-wa"]
MARSH = ["anahuac-tx", "myakka-fl", "brazoria-tx", "mexico-beach-fl"]
MAX_GB = float(os.environ.get("MAX_GB", 30))
WORKERS, BLAS_THREADS = 6, 2  # 16 cores (12 performance): 6 fits at a time, 2 math threads each
PX = 2000
RNG = np.random.default_rng(0)


def watch():
    p, peak = psutil.Process(), [0.0]

    def run():
        while True:
            g = p.memory_info().rss / 1e9
            peak[0] = max(peak[0], g)
            if g > MAX_GB:
                print(f"STOP: RSS {g:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()
    return peak


def unit(x):
    x = np.nan_to_num(x.astype(np.float32))
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def olmo_deq(q):
    """int8 -> unit float32, and the valid mask (-128 = nodata)."""
    f = q.astype(np.float32) / 127.5
    return unit(np.sign(f) * f * f), q[..., 0] != -128


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def r3(v):
    return None if v is None else round(float(v), 3)


def fit(X, y, C=0.1):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=200, C=C, class_weight="balanced").fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z))


def held_fold(Xall, yall, fold, i, sl):
    """One held-out fit in a worker process: train on every other square, score square i."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(BLAS_THREADS):
        tr = fold != i
        f = fit(np.ascontiguousarray(Xall[tr][:, sl]), yall[tr])
        return f(np.ascontiguousarray(Xall[fold == i][:, sl]))


def forest_square(n):
    z = np.load(EW / "data/separate" / f"{n}.npz", allow_pickle=True)
    o, ook = olmo_deq(np.load(OLMO / f"{n}.npz")["q"])
    E = {"aef": unit(z["aef"][1:]), "tes": unit(z["tes"][1:]), "olmo": o}
    ok = z["aok"][1:].all(0) & z["tok"][1:].all(0) & ook.all(0)
    return E, ok, int(z["b"][0]) + 1, z["grp"] == 4


def main():
    peak = watch()
    t0 = time.time()
    res = {"forest": {}, "held_out": {}, "detectors": {}, "sets": {}}
    rows = []
    for n in FOREST:
        E, ok, ev, cut = forest_square(n)
        i0, i1 = YEARS.index(max(ev - 1, YEARS[0])), YEARS.index(ev + 1)
        r = {"event": ev, "cut_px": int((cut & ok).sum())}
        for k in EMB:
            e = E[k]
            ch = 1 - (e[i0] * e[i1]).sum(-1)
            r[f"{k}_detect"] = r3(auc(cut[ok], ch[ok]))
            step = 1 - (e[1:] * e[:-1]).sum(-1)
            top = np.array(YEARS[1:])[step.argmax(0)]
            off = top[cut & ok] - ev
            r[f"{k}_when"] = r3(((off == 0) | (off == 1)).mean())
            c = (e * e[i0][None]).sum(-1)
            r[f"{k}_back"] = {str(y): r3(np.nanmean(c[t, cut & ok])) for t, y in enumerate(YEARS) if y >= ev - 1}
            r[f"{k}_forest"] = {str(y): r3(np.nanmean(c[t, ~cut & ok])) for t, y in enumerate(YEARS) if y >= ev - 1}
        res["forest"][n] = r
        keep = RNG.choice(np.flatnonzero(ok), min(PX, ok.sum()), replace=False)
        X, y, pl = [], [], {k: [] for k in EMB}
        for Y in CAND:
            a, b = YEARS.index(Y - 1), YEARS.index(Y + 1)
            X.append(np.concatenate([np.concatenate([E[k][a, keep], E[k][b, keep], E[k][b, keep] - E[k][a, keep]], 1) for k in EMB], 1))
            y.append((cut[keep] & (ev == Y)).astype(np.int8))
            for k in EMB:
                pl[k].append(1 - (E[k][a, keep] * E[k][b, keep]).sum(1))
        before = {k: E[k][i0, keep] for k in EMB}
        rows.append({"name": n, "X": np.concatenate(X).astype(np.float32), "y": np.concatenate(y),
                     "plain": {k: np.concatenate(v) for k, v in pl.items()}, "before": before, "cut": cut[keep], "ev": ev})
        print(f"{n:24s} {ev} detect " + " ".join(f"{k} {r[f'{k}_detect']}" for k in EMB) + "  when " +
              " ".join(f"{k} {r[f'{k}_when']}" for k in EMB) + f"  [{time.time() - t0:.0f} s, {peak[0]:.1f} GB]", flush=True)
        del E

    # held-out model per embedding and all three
    cols, o = {}, 0
    for k, d in (("aef", 64), ("tes", 128), ("olmo", 128)):
        cols[k] = slice(o, o + 3 * d)
        o += 3 * d
    cols["all3"] = slice(0, o)
    Xall = np.concatenate([r["X"] for r in rows])
    yall = np.concatenate([r["y"] for r in rows])
    fold = np.concatenate([np.full(len(r["y"]), i, np.int16) for i, r in enumerate(rows)])
    for r in rows:
        del r["X"]
    tasks = [(which, i) for which in cols for i, r in enumerate(rows) if r["y"].sum() > 0]
    scores = dict(zip(tasks, Parallel(n_jobs=WORKERS, max_nbytes="1M")(
        delayed(held_fold)(Xall, yall, fold, i, cols[which]) for which, i in tasks)))
    del Xall
    for which in cols:
        per = {}
        for i, r in enumerate(rows):
            if (which, i) not in scores:
                continue
            s = scores[(which, i)]
            S = s.reshape(len(CAND), -1)
            best = np.array(CAND)[S.argmax(0)]
            per[r["name"]] = {"auc": r3(auc(r["y"], s)), "when": r3(((best[r["cut"]] == r["ev"]) | (best[r["cut"]] == r["ev"] + 1)).mean())}
        res["held_out"][which] = per
        print(f"held out {which:4s} median AUC {np.median([v['auc'] for v in per.values()]):.3f} "
              f"year right {np.median([v['when'] for v in per.values()]):.3f}  [{time.time() - t0:.0f} s]", flush=True)

    # quiet squares, every embedding, every year
    quiet = {}
    for f in sorted(glob.glob(str(EW / "data/years/*.npz"))):
        n = Path(f).stem
        z = np.load(f, allow_pickle=True)
        o, ook = olmo_deq(np.load(OLMO / f"{n}.npz")["q"])
        A = np.stack([unit(z[f"aef_{y}"]) for y in YEARS])
        T = np.stack([unit(z[f"{y}_q"].astype(np.float32) * np.nan_to_num(z[f"{y}_scales"])[:, None]) for y in YEARS])
        ok = np.stack([z[f"aok_{y}"] & np.isfinite(z[f"{y}_scales"]) & (z[f"{y}_scales"] > 0) for y in YEARS]).all(0) & ook.all(0)
        keep = RNG.choice(np.flatnonzero(ok), min(3000, ok.sum()), replace=False) if ok.sum() else np.array([], int)
        quiet[n] = {"aef": A[:, keep], "tes": T[:, keep], "olmo": o[:, keep]}

    # gates and cutoffs
    pos = np.concatenate([r["y"] for r in rows]) == 1
    gates, gate_at, cut_at = {}, {}, {}
    nonforest = [n for n in quiet if n not in QUIET_FOREST and quiet[n]["aef"].shape[1]]
    for k in EMB:
        gp = np.concatenate([r["before"][k] for r in rows])
        gn = np.concatenate([quiet[n][k][0] for n in nonforest])
        g = fit(np.concatenate([gp, gn]), np.r_[np.ones(len(gp)), np.zeros(len(gn))].astype(int), C=1.0)
        gates[k] = g
        gate_at[k] = float(np.quantile(g(np.concatenate([r["before"][k][r["cut"]] for r in rows])), 0.05))
        s = np.concatenate([r["plain"][k] for r in rows])
        cut_at[k] = float(np.quantile(s[pos], 0.2))
    gcut = {k: np.concatenate([np.tile(gates[k](r["before"][k]) >= gate_at[k], len(CAND)) for r in rows]) for k in EMB}
    hit = {k: (np.concatenate([r["plain"][k] for r in rows]) >= cut_at[k]) & gcut[k] for k in EMB}
    vote = (hit["aef"].astype(int) + hit["tes"] + hit["olmo"]) >= 2
    res["detectors"] = {"cut_at": cut_at, "gate_at": gate_at,
                        "cuts_found": {**{k: r3(hit[k][pos].mean()) for k in EMB}, "vote": r3(vote[pos].mean())}}
    print("cuts found (gated, 80 % cutoffs):", res["detectors"]["cuts_found"], flush=True)

    def lit(E):
        h = {}
        for k in EMB:
            b, a = E[k]
            h[k] = ((1 - (b * a).sum(1)) >= cut_at[k]) & (gates[k](b) >= gate_at[k])
            h[k + "_ungated"] = (1 - (b * a).sum(1)) >= cut_at[k]
        h["vote"] = (h["aef"].astype(int) + h["tes"] + h["olmo"]) >= 2
        return {k: r3(v.mean()) for k, v in h.items()} | {"n": int(len(E["aef"][0]))}

    def add(group, name, E):
        r = lit(E)
        res["sets"].setdefault(group, {})[name] = r
        print(f"{group:18s} {name:34s} n {r['n']:6d}  " + " ".join(f"{k} {r[k]}" for k in EMB) + f"  vote {r['vote']}  "
              f"ungated " + " ".join(f"{k} {r[k + '_ungated']}" for k in EMB) + f"  [{time.time() - t0:.0f} s, {peak[0]:.1f} GB]", flush=True)

    for n, Q in quiet.items():
        if not Q["aef"].shape[1]:
            continue
        E = {k: (np.concatenate([Q[k][YEARS.index(Y - 1)] for Y in CAND]), np.concatenate([Q[k][YEARS.index(Y + 1)] for Y in CAND])) for k in EMB}
        add("years-forest" if n in QUIET_FOREST else "years-open", n, E)
    del quiet

    def site(group, name, olmo_q, which=None):
        d = compare.load(name)
        ob, oa = olmo_q
        (ob, okb), (oa, oka) = olmo_deq(ob), olmo_deq(oa)
        m = d["ok"] & okb & oka
        if which is not None:
            m &= which(name, d)
        idx = np.flatnonzero(m)
        if len(idx) > 50000:
            idx = np.sort(RNG.choice(idx, 50000, replace=False))
        E = {"aef": (unit(d["ab"][idx]), unit(d["aa"][idx])), "tes": (unit(d["tb"][idx]), unit(d["ta"][idx])),
             "olmo": (ob[idx], oa[idx])}
        del d
        add(group, name, E)

    def ew_olmo(name):
        z = np.load(EW / "data/olmoearth" / f"{name}.npz")
        return z["b_q"], z["a_q"]

    for f_name, region, b, a in ts.cdl_sets():
        site("crop", f_name, ew_olmo(f_name))
    for n in FARM:
        site("farm", n, ew_olmo(n), lambda nm, d: d["y"] == 0)
    for n in MARSH:
        site("marsh", n, ew_olmo(n), lambda nm, d: d["y"] == 0)

    def ground_grp(want):
        return lambda nm, d: np.load(ts.site_path(nm), allow_pickle=True)["grp"] == want
    for p in sorted(glob.glob(str(EW / "data/places/ground-*.npz"))):
        n = Path(p).stem
        kind = json.loads(str(np.load(p, allow_pickle=True)["meta"]))["kind"]
        if kind == "cut":
            continue
        q = np.load(OLMO / f"{n}.npz")["q"]
        grp = {"burn": "burn", "lake": "lake"}.get(kind, "look")
        site(grp, n, (q[0], q[1]), ground_grp(1))
        if kind == "burn":
            site("burn-forest-beside", n, (q[0], q[1]), ground_grp(2))

    res["peak_gb"] = round(peak[0], 2)
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "trees3.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, peak {peak[0]:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
