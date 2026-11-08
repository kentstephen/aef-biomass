"""Earthwork's trouble spots through the biomass head: how much ground does it say gained or lost biomass?

The tree-cut detector lit crop switches (98 % in AlphaEarth), marsh and storm coast (40 %), reservoirs (16 %) and
farms until the forest gate cut them out (false_alarms_trees.py, trees3.py). A biomass map has no forest gate, so
the same places are run through the square-held-out biomass heads of biomass_head.py (the 8 fold heads averaged;
none of these sites is in their rows). Sets, from aef-tessera-earthwork, each over the two years Earthwork compared:
  crop     crop switches in US farm regions (USDA CDL), nothing dug or cut
  farm     farm lidar sites, unchanged ground; marsh: Texas / Florida marsh and storm coast lidar, unchanged ground
  look     floods, landslides, river banks, solar arrays, pits (the event's ground)
  lake     reservoirs drawn down or refilled
  burn     forest burned; burn-beside: the standing forest beside it
  (Earthwork's clearcut places have no OlmoEarth read and are skipped; biomass_change.py's Hansen-loss pixels are the
  real-loss reference)
Per site, share of pixels "lit": predicted biomass changed by a quarter or more and by 10 Mg/ha or more (fell or
rose), plain, and gated (only where the embedding moved past its frozen cutoff, models/detectors.npz). Per set, the
median over sites. No biomass truth here: for crop, marsh and farms, a large lit share is change the map would show
where the ground did not change in kind; for cut, lake, look and burn it is change that happened.

Run with Earthwork's environment:
    uv run --project ~/dev/projects/aef-tessera-earthwork --with torch --with psutil python scripts/biomass_lookalikes.py
RAM: one site in flight (up to 20,000 pixels kept), peak about 4 GB; stops past --max-gb.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
EW = Path.home() / "dev/projects/aef-tessera-earthwork"
sys.path.insert(0, str(HERE / "scripts"))
import biomass_head as bh  # noqa: E402

sys.path.insert(0, str(EW))
os.chdir(EW)
import compare  # noqa: E402
import tessera_sites as ts  # noqa: E402

OLMO = HERE / "data" / "olmo"
FARM = ["antelope-farm-ne", "box-butte-farm-ne", "fresno-farm-ca", "hamilton-farm-in", "arkansas-delta-ar",
        "othello-pivots-wa"]
MARSH = ["anahuac-tx", "myakka-fl", "brazoria-tx", "mexico-beach-fl"]
INPUTS = ("aef", "tes", "olmo", "all")
PX = 20000
RNG = np.random.default_rng(0)


def heads(inp):
    W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
    d = bh.DIMS[inp] if inp != "all" else sum(bh.DIMS.values())
    out = []
    for f in sorted({k.split(".", 1)[0] for k in W.files if "." in k}):
        net = bh.Head(d, np.zeros(d, np.float32), np.ones(d, np.float32))
        net.load_state_dict({k.split(".", 1)[1]: torch.as_tensor(W[k]) for k in W.files if k.startswith(f + ".")})
        out.append(net.eval())
    return out


@torch.no_grad()
def biomass(nets, x):
    x = torch.as_tensor(x.astype(np.float32))
    return np.mean([np.clip(np.expm1(np.minimum(n(x).numpy()[:, 0], 8)), 0, None) for n in nets], 0)


def deq(q):
    return bh.deq(q), q[..., 0] != -128


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def site(name, olmo_q, which=None):
    """Before / after per embedding (head inputs: AlphaEarth as stored, TESSERA q x scale, OlmoEarth dequantized)."""
    d = compare.load(name)
    t = np.load(f"{compare.OUT}/{name}.npz", allow_pickle=True)
    (ob, okb), (oa, oka) = deq(olmo_q[0]), deq(olmo_q[1])
    m = d["ok"] & okb & oka
    if which is not None:
        m &= which(name, d)
    idx = np.flatnonzero(m)
    if len(idx) > PX:
        idx = np.sort(RNG.choice(idx, PX, replace=False))
    E = {"aef": (d["ab"][idx].astype(np.float32), d["aa"][idx].astype(np.float32)),
         "tes": ((t["b_q"][idx].astype(np.float32) * t["b_scales"][idx, None]),
                 (t["a_q"][idx].astype(np.float32) * t["a_scales"][idx, None])),
         "olmo": (ob[idx], oa[idx])}
    E["all"] = tuple(np.concatenate([E[k][i] for k in ("aef", "tes", "olmo")], 1) for i in (0, 1))
    return E, d["years"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=10)
    args = ap.parse_args()
    t0 = time.time()
    D = np.load(HERE / "models" / "detectors.npz")
    NETS = {inp: heads(inp) for inp in INPUTS}
    res = {"sites": {}}

    def add(group, name, olmo_q, which=None):
        E, years = site(name, olmo_q, which)
        n = len(E["aef"][0])
        if n < 200:
            return
        moved = {k: 1 - (unit(E[k][0]) * unit(E[k][1])).sum(1) >= float(D[f"{k}_cut"]) for k in ("aef", "tes", "olmo")}
        moved["all"] = (moved["aef"].astype(int) + moved["tes"] + moved["olmo"]) >= 2
        r = {"group": group, "n": n, "years": years}
        for inp in INPUTS:
            b, a = biomass(NETS[inp], E[inp][0]), biomass(NETS[inp], E[inp][1])
            dlt = a - b
            lit = (np.abs(dlt) >= 0.25 * np.maximum(b, 1)) & (np.abs(dlt) >= 10)
            r[inp] = {"before": round(float(np.median(b)), 1), "after": round(float(np.median(a)), 1),
                      "lit": round(float(lit.mean()), 3), "fell": round(float((lit & (dlt < 0)).mean()), 3),
                      "lit_gated": round(float((lit & moved[inp]).mean()), 3)}
        res["sites"][f"{group}/{name}"] = r
        gb = psutil.Process().memory_info().rss / 1e9
        print(f"{group:11s} {name:34s} n {n:6d} " + "  ".join(
            f"{k} {r[k]['before']}->{r[k]['after']} lit {r[k]['lit']} gated {r[k]['lit_gated']}" for k in INPUTS)
            + f"  [{time.time() - t0:.0f} s, {gb:.1f} GB]", flush=True)
        if gb > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb:.1f} GB")

    def ew_olmo(name):
        z = np.load(EW / "data/olmoearth" / f"{name}.npz")
        return z["b_q"], z["a_q"]

    for f_name, region, b, a in ts.cdl_sets():
        add("crop", f_name, ew_olmo(f_name))
    for n in FARM:
        add("farm", n, ew_olmo(n), lambda nm, d: d["y"] == 0)
    for n in MARSH:
        add("marsh", n, ew_olmo(n), lambda nm, d: d["y"] == 0)

    def grp(want):
        return lambda nm, d: np.load(ts.site_path(nm), allow_pickle=True)["grp"] == want
    for p in sorted(glob.glob(str(EW / "data/places/ground-*.npz"))):
        n = Path(p).stem
        if not (OLMO / f"{n}.npz").exists():
            continue
        kind = json.loads(str(np.load(p, allow_pickle=True)["meta"]))["kind"]
        q = np.load(OLMO / f"{n}.npz")["q"]
        g = {"burn": "burn", "lake": "lake", "cut": "cut"}.get(kind, "look")
        add(g, n, (q[0], q[1]), grp(1))
        if kind in ("burn", "cut"):
            add(f"{g}-beside", n, (q[0], q[1]), grp(2))

    groups = sorted({r["group"] for r in res["sites"].values()})
    res["sets"] = {}
    for g in groups:
        rs = [r for r in res["sites"].values() if r["group"] == g]
        res["sets"][g] = {"sites": len(rs)} | {
            inp: {k: round(float(np.median([r[inp][k] for r in rs])), 3) for k in ("lit", "fell", "lit_gated")}
            for inp in INPUTS}
        print(f"SET {g:11s} sites {len(rs):3d}  " + "  ".join(
            f"{inp} lit {res['sets'][g][inp]['lit']} gated {res['sets'][g][inp]['lit_gated']}" for inp in INPUTS), flush=True)
    json.dump(res, open(HERE / "results" / "biomass_lookalikes.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
