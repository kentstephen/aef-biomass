"""Biomass lost x ground moved: the map's two answers against truth for both, at the lidar sites of Earthwork's
section 19 (FINDINGS 42).

Truth, per 10 m pixel (aef-tessera-earthwork/data/separate, built by separate.py): Hansen's forest loss in a year of
the lidar window, split by the repeat lidar's ground change into cleared and dug and cleared and not dug; other
labeled ground (no Hansen loss there) split by the same lidar into dug and not dug. So four boxes:

  cleared + dug      Hansen loss, ground moved (mean |dz| > .5 m)
  cleared only       Hansen loss, ground unchanged (< .1 m)
  dug only           no Hansen loss, ground moved
  neither            no Hansen loss, ground unchanged

The map's answers, on the same pair of years (b: the year before the first flight, a: the year of the second):

  biomass lost   the map's head (web/public/models/biomass_aef.json) on each year, the browser's loss rule: the
                 head fell and its confidence (web/public/models/change_conf.json, the lower of the fall's and the
                 embedding move's share of steady ground below them) is .9 or more (web/src/change.ts)
  ground moved   Earthwork's AlphaEarth layer (aef-tessera-earthwork web/public/models/world.json): logistic on
                 [b, a, a*b, (a - b)^2] of the unit vectors, times the gate clip((1 - cos - .05) / .10, 0, 1), from
                 its cutoff (.6035, 60 % of held-out US digging found). At the sites that taught it (the five US
                 sites here) the world recipe's held-out scores are used (data/scores-world/world); Kentucky never
                 taught it, so its final model is used there. Both are reported at the US sites, for the gap.

Run: uv run --project ../aef-tessera-earthwork --with psutil python scripts/four_box.py [--max-gb 8]
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
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(EW))
os.chdir(EW)
import compare  # noqa: E402

OUT = HERE / "results" / "four_box.json"
SEPD = EW / "data" / "separate"
HELD = EW / "data" / "scores-world" / "world"
SITES = ["ky-clay", "ky-perry", "ky-pike", "hibbing-mn", "lakewood-ranch-fl", "four-corners-fl", "huntsville-al",
         "new-albany-oh"]
Y0 = 2017  # the separate files hold 2017 to 2025
BOXES = ["cleared + dug", "cleared only", "dug only", "neither"]
LOSS_FROM = 0.9

# ---------------------------------------------------------------- the biomass map's head and loss rule
H = json.load(open(HERE / "web" / "public" / "models" / "biomass_aef.json"))
MU, SD = np.array(H["mean"], np.float32), np.array(H["scale"], np.float32)
W1 = np.array(H["w1"], np.float32).reshape(H["hidden"], H["n_in"])
B1 = np.array(H["b1"], np.float32)
W2 = np.array(H["w2"], np.float32).reshape(H["n_out"], H["hidden"])
B2 = np.array(H["b2"], np.float32)
CUT = float(np.load(HERE / "models" / "detectors.npz")["aef_cut"])


def head(e):
    """agb (Mg/ha) and class index from dequantized bands (here unit length: AlphaEarth's own norm is within .01)."""
    z = np.maximum(((e - MU) / SD) @ W1.T + B1, 0) @ W2.T + B2
    return np.expm1(np.clip(z[:, 0], 0, 8)), np.argmax(z[:, 2:], 1)


class Conf:
    """The browser's steady tables: the share of steady pixels below x, per head class and window length."""

    def __init__(self):
        T = json.load(open(HERE / "web" / "public" / "models" / "change_conf.json"))
        self.p, self.cls, self.q, self.L = np.array(T["p"]), T["classes"], T["q"], T["L"]

    def share(self, var, cls, L, x):
        out = np.zeros(x.shape, np.float32)
        Lc = np.clip(L, self.L[0], self.L[-1])
        for c in np.unique(cls):
            for l_ in np.unique(Lc):
                m = (cls == c) & (Lc == l_)
                q = np.array(self.q[var].get(f"{self.cls[c]}:{l_}", self.q[var][f"all:{l_}"]))
                v = np.interp(x[m], q, self.p)
                v[x[m] <= q[0]] = 0
                v[x[m] > q[-1]] = 1
                out[m] = v
        return out


def biomass_lost(eb, ea, L, conf):
    ab, cb = head(eb)
    aa, _ = head(ea)
    d = aa - ab
    mv = (1 - (eb * ea).sum(1)) / CUT
    c = np.minimum(conf.share("neg_dpct", cb, L, np.maximum(-100 * d / np.maximum(ab, 1), 0)),
                   conf.share("mv", cb, L, mv))
    return (d < 0) & (c >= LOSS_FROM), ab, d


# ---------------------------------------------------------------- Earthwork's AlphaEarth layer
EWM = json.load(open(EW / "web" / "public" / "models" / "world.json"))
EW_W, EW_B, EW_CUT = np.array(EWM["aef"]["w"], np.float32), float(EWM["aef"]["b"]), float(EWM["cut"]["aef"])


def earthwork(eb, ea):
    f = np.concatenate([eb, ea, ea * eb, (ea - eb) ** 2], 1)
    p = 1 / (1 + np.exp(-(f @ EW_W + EW_B)))
    return compare.gate(p, 1 - (eb * ea).sum(1))


_LAB = {}


def labeled_rows(name):
    """The rows of a site's labeled pixels, in the order its saved score files keep them (compare.labeled)."""
    if name not in _LAB:
        d = compare.load(name)
        _LAB[name] = compare.labeled(d)
        del d
    return _LAB[name]


def held_out(name, idx, scores):
    """A recipe's held-out AlphaEarth score (gated) at the separate file's pixels; NaN where not labeled; None if
    the recipe saved no scores for the site."""
    fp = scores / f"{name}.npz"
    if not fp.exists():
        return None
    lab = labeled_rows(name)
    S = np.load(fp, allow_pickle=True)
    assert len(S["aef"]) == len(lab), (name, len(S["aef"]), len(lab))
    pos = np.clip(np.searchsorted(lab, idx), 0, len(lab) - 1)
    hit = lab[pos] == idx
    g = compare.gate(S["aef"][pos], S["aplain"][pos]).astype(np.float32)
    g[~hit] = np.nan
    return g


# ---------------------------------------------------------------- scoring
def table(truth, pred):
    """Rows: truth box; columns: the map's box; each row the share of that truth's pixels."""
    out = {}
    for i, t in enumerate(BOXES):
        m = truth == i
        out[t] = {"n": int(m.sum()), **{BOXES[j]: round(float((pred[m] == j).mean()), 3) if m.any() else None
                                        for j in range(4)}}
    return out


def boxes(lost, dug):
    return np.where(lost & dug, 0, np.where(lost, 1, np.where(dug, 2, 3))).astype(np.int8)


def guard(max_gb):
    def run():
        while True:
            if psutil.Process().memory_info().rss / 1e9 > max_gb:
                print(f"STOP: RSS over {max_gb} GB", flush=True)
                os._exit(3)
            time.sleep(1)
    threading.Thread(target=run, daemon=True).start()


# Earthwork recipes with saved held-out scores at the five US sites, and each one's cutoff (world_eval.cutoffs: 60 %
# of held-out US digging found, gated AlphaEarth)
RECIPES = {
    "world (the map today)": ("scores-world/world", 0.6035),
    "burns quiet": ("scores-fire-abroad", 0.618),
    "burns + every cut quiet": ("scores-fire-abroad-cut", 0.6118),
    "burns + cuts least like digging": ("scores-fire-abroad-cut-far", 0.6076),
    "burns + 5,000 cut px per region": ("scores-fire-abroad-cut-px5000", 0.6051),
    "burns + cuts at weight .25": ("scores-fire-abroad-cut-w0.25", 0.6055),
}
SEP_SCORES = HERE / "data" / "four_box" / "dug_or_cleared.npz"
SEP_CUT = 0.5


def summary(table):
    """The share of each truth box the map puts in the right box, and their mean."""
    right = {t: table[t][t] for t in BOXES if table[t]["n"] >= 50}
    return {"right": right, "mean_right": round(float(np.mean(list(right.values()))), 3) if right else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=10)
    args = ap.parse_args()
    guard(args.max_gb)
    conf = Conf()
    SEP = np.load(SEP_SCORES)
    res, t0 = {"sites": {}, "variants": {}}, time.time()
    tabs = {}  # variant -> site -> table
    for name in SITES:
        S = np.load(SEPD / f"{name}.npz", allow_pickle=True)
        grp, y, b, a, idx = S["grp"], S["y"], S["b"].astype(int), S["a"].astype(int), S["idx"]
        aef, aok = S["aef"], S["aok"]
        n = len(grp)
        r = np.arange(n)
        ib, ia = b - Y0, a - Y0
        truth = np.select([grp == 1, grp == 2, (grp == 3) & (y == 1), (grp == 3) & (y == 0)], [0, 1, 2, 3], -1)
        ok = aok[ib, r] & aok[ia, r] & (truth >= 0)
        eb, ea = aef[ib, r].astype(np.float32), aef[ia, r].astype(np.float32)
        del aef
        lost, agb_b, d = biomass_lost(eb, ea, a - b, conf)
        # the dug-or-cleared head's held-out score at every labeled pixel (the fold that left this site out)
        sep = {fs: np.full(n, np.nan, np.float32) for fs in ("pair", "ctx")}
        rows = SEP[f"{name}|rows"]
        for fs in sep:
            sep[fs][rows] = SEP[f"{name}|{fs}"]
        ew = {"world (the map today)": held_out(name, idx, EW / "data" / RECIPES["world (the map today)"][0])}
        final = earthwork(eb, ea)
        if ew["world (the map today)"] is None:
            ew["world (the map today)"] = final  # Kentucky never taught the world recipe
        for k, (dir_, cut) in RECIPES.items():
            if k not in ew:
                ew[k] = held_out(name, idx, EW / "data" / dir_)
        res["sites"][name] = {"px": int(ok.sum()), "pairs": sorted({f"{bb}-{aa}" for bb, aa in zip(b[ok], a[ok])}),
                              "before_2018": round(float((b[ok] < 2018).mean()), 3),
                              "world_used": "final" if name.startswith("ky-") else "held_out",
                              "biomass_lost": {t: round(float(lost[ok & (truth == i)].mean()), 3) for i, t in enumerate(BOXES) if (ok & (truth == i)).any()}}
        for k, (dir_, cut) in RECIPES.items():
            g = ew[k]
            if g is None:
                continue
            m = ok & np.isfinite(g)
            tabs.setdefault(f"A/B: biomass lost x Earthwork {k}", {})[name] = table(truth[m], boxes(lost[m], g[m] >= cut))
        # option 2: where biomass was lost the head decides; elsewhere the map's Earthwork (world) decides
        gw = ew["world (the map today)"]
        cw = RECIPES["world (the map today)"][1]
        for fs in ("pair", "ctx"):
            s_ = sep[fs]
            m = ok & np.isfinite(gw) & np.isfinite(s_)
            for tag, dug_lost in ((f"C: lost -> {fs} head; else Earthwork", s_ >= SEP_CUT),
                                  (f"C: lost -> {fs} head AND Earthwork; else Earthwork", (s_ >= SEP_CUT) & (gw >= cw))):
                dug = np.where(lost, dug_lost, gw >= cw)
                tabs.setdefault(tag, {})[name] = table(truth[m], boxes(lost[m], dug[m]))
        print(f"{name}: {int(ok.sum())} px [{time.time() - t0:.0f} s, {psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        _LAB.pop(name, None)
        del S, eb, ea
    US = [n for n in SITES if not n.startswith("ky-")]
    for v, per in tabs.items():
        out = {"sites": per}
        for scope, names in (("five US sites", US), ("all eight", SITES)):
            names = [n for n in names if n in per]
            if scope == "all eight" and len(names) < len(SITES):
                continue
            pooled = {}
            for t in BOXES:
                rows_ = [per[n][t] for n in names if per[n][t]["n"] >= 50]
                pooled[t] = {"n": sum(rw["n"] for rw in rows_), **{c: round(float(np.mean([rw[c] for rw in rows_])), 3) for c in BOXES}}
            out[scope] = {"pooled": pooled, **summary(pooled)}
        res["variants"][v] = out
    print("\nright box, mean of sites (cleared + dug / cleared only / dug only / neither; mean)")
    for scope in ("five US sites", "all eight"):
        print(f"  {scope}")
        for v, out in res["variants"].items():
            if scope in out:
                rr = out[scope]["right"]
                print(f"    {v:62s} " + " ".join(f"{rr.get(t, float('nan')):5.2f}" for t in BOXES) + f"   {out[scope]['mean_right']:.3f}")
    res["peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 1)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"done {res['seconds']} s, {res['peak_gb']} GB")


if __name__ == "__main__":
    main()
