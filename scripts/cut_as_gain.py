"""Cut forest drawn as gain: on the two-flight lidar squares, of the forest lidar says lost canopy, how much does the
map draw as gain, as loss, or not at all, and does the head's own canopy height fall there?

Two New Hampshire harvests (2023 and 2024) drew blue in the change view: the head's canopy height fell from 12 to 3 m and
14 to 5 m, but its biomass rose 74 to 141 and 63 to 92 Mg/ha, and the gain rule reads only biomass.

Per square, on pixels with lidar cover in both flights and AlphaEarth in every year between them:
- Lost forest: first flight 5 m or taller, canopy fell 3 m or more (as FINDINGS 32). Clear cut: also fell to under
  half its height. Steady forest: 5 m or taller, moved under 0.5 m. Gained: rose 2 m or more.
- The flight window (the years of the two flights): the map's loss (confidence .9 and up, web/src/change.ts) and
  gain (the two-part floor), as the browser draws them.
- The year after the cut: on lost pixels, the one-year step inside the window where the embedding moved most (the
  cut's year as AlphaEarth sees it), drawn by the one-year rule. The New Hampshire cases are this step.
- The head's height: of the pixels not drawn as loss, the share where the head's height fell 3 m or more (what
  height would add to loss, and its false rate on steady forest). Of the pixels drawn as gain, the share where the head's canopy height fell 3 m or more (a
  height check on gain would remove them), on lost, gained and steady forest.

Writes results/cut_as_gain.json. RAM under 2 GB (one square at a time); stops itself past --max-gb.

    uv run --project ~/dev/projects/aef-tessera-earthwork --with psutil --with shapely --with duckdb --with scipy \
        --with icechunk --with xarray --with requests --with torch python scripts/cut_as_gain.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import burn_check as bc  # noqa: E402  (the map's head, as the browser runs it)
import change_confidence_test as cct  # noqa: E402
import window_test as wt  # noqa: E402

OUT = HERE / "results" / "cut_as_gain.json"
YEARS = cct.YEARS
SQUARES = ["lidar-ca-timmins-dome", "lidar-ee-kivioli", "lidar-ee-lahemaa", "lidar-ee-tartu", "lidar-gb-devilla",
           "lidar-no-holmenkollen", "lidar-nz-banks", "lidar-nz-esk-valley", "lidar-nz-glenbervie", "lidar-nz-roxburgh",
           "lidar-us-eaton", "lidar-us-palisades"]
HFALL = 3.0  # the head's height fall, m, that counts as a cut


def head_years(q, ok):
    """The head's biomass (Mg/ha), canopy height (m) and class per year, NaN / -1 without an embedding."""
    Y, n = ok.shape
    agb = np.full((Y, n), np.nan, np.float32)
    hgt = np.full((Y, n), np.nan, np.float32)
    cls = np.full((Y, n), -1, np.int16)
    for y in range(Y):
        i = np.flatnonzero(ok[y])
        for s in range(0, i.size, 60000):
            j = i[s:s + 60000]
            raw = bc.map_raw(q[y, j])
            agb[y, j] = np.expm1(np.clip(raw[:, 0], 0, 8))
            hgt[y, j] = np.expm1(np.clip(raw[:, 1], 0, 8))
            cls[y, j] = raw[:, 2:].argmax(1)
    return agb, hgt, cls


def move(q, ya, yb, idx):
    ua, ub = cct.unit(q[ya, idx]).astype(np.float32), cct.unit(q[yb, idx]).astype(np.float32)
    return (1 - (ua * ub).sum(1)) / cct.CUT


def drawn(q, agb, cls, conf, ya, yb, idx):
    """The browser's change from year ya to yb on pixels idx: -1 loss, +1 gain, 0 nothing."""
    L = yb - ya
    a, d = agb[ya, idx], agb[yb, idx] - agb[ya, idx]
    tau = float(wt.rule_for(L)[0])
    c = cls[ya, idx].astype(np.int64)
    Lv = np.full(idx.size, L)
    out = np.zeros(idx.size, np.int8)
    neg = d < 0
    if neg.any():
        cp = conf.share("neg_dpct", c[neg], Lv[neg], -100 * d[neg] / np.maximum(a[neg], 1))
        cm = conf.share("mv", c[neg], Lv[neg], move(q, ya, yb, idx[neg]))
        out[np.flatnonzero(neg)[np.minimum(cp, cm) >= wt.LOSS_FROM]] = -1
    out[(d >= tau) & (d >= 0.25 * np.maximum(a, 1))] = 1
    return out


def share(m, of):
    n = int(of.sum())
    return {"n": n, "share": round(float((m & of).sum()) / n, 4) if n else None}


def one(name, conf):
    z = np.load(HERE / "data" / "squares" / f"{name}.npz")
    q = z["aef"]
    Ld = np.load(HERE / "data" / "lidar" / f"{name}.npz")
    fy = [int(y) for y in Ld["year"]]
    i1, i2 = [YEARS.index(min(max(y, YEARS[0]), YEARS[-1])) for y in fy]
    ok = q[:, :, 0] != -128
    agb, hgt, cls = head_years(q, ok)
    chm, cov = Ld["chm"].astype(np.float32), Ld["cover"]
    m = np.all(cov >= 80, 0) & np.all(ok[i1:i2 + 1], 0)
    h1, h2 = chm[0], chm[1]
    dh = h2 - h1
    sets = {"lost": m & (h1 >= 5) & (dh <= -3)}
    sets["clear cut"] = sets["lost"] & (h2 < 0.5 * h1)
    sets["steady forest"] = m & (h1 >= 5) & (np.abs(dh) < 0.5)
    sets["gained"] = m & (dh >= 2)
    idx = np.flatnonzero(m)
    full = np.zeros(m.size, np.int8)
    full[idx] = drawn(q, agb, cls, conf, i1, i2, idx)
    hf = np.zeros(m.size, bool)
    hf[idx] = (hgt[i1, idx] - hgt[i2, idx]) >= HFALL
    res = {"flights": fy, "years": [YEARS[i1], YEARS[i2]], "window": {}}
    for k, s in sets.items():
        g = (full == 1) & s
        res["window"][k] = {"px": int(s.sum()), "drawn gain": share(full == 1, s)["share"],
                            "drawn loss": share(full == -1, s)["share"], "nothing": share(full == 0, s)["share"],
                            "gain with head height fall": share(hf, g),
                            "head height fell (all)": share(hf, s)["share"],
                            "not drawn as loss, head height fell": share(hf, s & (full != -1))}
    # the year after the cut, as AlphaEarth sees it
    if i2 - i1 >= 1:
        li = np.flatnonzero(sets["lost"])
        mv = np.stack([move(q, y, y + 1, li) for y in range(i1, i2)])
        step = i1 + mv.argmax(0)
        one_ = np.zeros(m.size, np.int8)
        hf1 = np.zeros(m.size, bool)
        rise = np.full(m.size, np.nan, np.float32)
        for y in np.unique(step):
            j = li[step == y]
            one_[j] = drawn(q, agb, cls, conf, y, y + 1, j)
            hf1[j] = (hgt[y, j] - hgt[y + 1, j]) >= HFALL
            rise[j] = agb[y + 1, j] - agb[y, j]
        res["cut step"] = {}
        for k in ("lost", "clear cut"):
            s = sets[k]
            g = (one_ == 1) & s
            res["cut step"][k] = {"px": int(s.sum()), "drawn gain": share(one_ == 1, s)["share"],
                                  "drawn loss": share(one_ == -1, s)["share"],
                                  "biomass rose": share(rise > 0, s)["share"],
                                  "gain with head height fall": share(hf1, g),
                                  "head height fell (all)": share(hf1, s)["share"]}
    del z, q, agb, hgt, cls
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=4.0)
    args = ap.parse_args()
    conf = wt.Conf()
    out, t0 = {}, time.time()
    for si, name in enumerate(SQUARES):
        out[name] = one(name, conf)
        rss = psutil.Process().memory_info().rss / 1e9
        w = out[name]["window"]["lost"]
        c = out[name].get("cut step", {}).get("lost", {})
        print(f"{si + 1}/{len(SQUARES)} {name} {out[name]['years']}: lost {w['px']} px, window gain {w['drawn gain']}"
              f" loss {w['drawn loss']}; cut step gain {c.get('drawn gain')} loss {c.get('drawn loss')}"
              f" [{time.time() - t0:.0f} s, {rss:.1f} GB]", flush=True)
        if rss > args.max_gb:
            sys.exit(f"stopped: {rss:.1f} GB past --max-gb {args.max_gb}")
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
