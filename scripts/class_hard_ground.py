# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Land cover on hard ground: the pixels where ESA WorldCover and Impact Observatory disagree.

The class scores so far (class_report.py, class_calibrate.py) are on pixels where the two maps agree. Here every
pixel of every square in 2020 and 2021 where both maps have a class, each square by its held-out fold head, with
the calibrated probabilities and class-wise 90 % sets of models/class_temperature.json (the set cutoffs were fit on
all folds' agreed pixels, the square's own included: a small leak, descriptive only):
  margin of error   confidence, share of single-class sets and set size, on agreed against disagreed pixels: the
                    per-pixel error should widen where the maps disagree
  whom it follows   on disagreed pixels, the share where the head picks WorldCover's class, one of IO's, or other
  trees settled by height   where one map says trees and the other does not, Meta's 10 m canopy height (imagery
                    2017 on) as the referee: 5 m or more = trees, under 2 m = not trees (between: left out);
                    share right for WorldCover, IO and the head
Writes results/class_hard_ground.json. RAM about 4 GB.

    uv run scripts/class_hard_ground.py [--inputs olmo all]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import biomass_change as bc  # noqa: E402
import biomass_head as bh  # noqa: E402
import biomass_rows as br  # noqa: E402

YEARS = bc.YEARS


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["olmo", "all"])
    args = ap.parse_args()
    R = np.load(HERE / "data" / "rows.npz")
    rows = {"sq": R["sq"], "kinds": R["kinds"], "continents": R["continents"]}
    names, kinds = list(R["names"]), list(R["kinds"])
    fold_of = {}
    for fname, test in bh.folds(rows, "square"):
        for s in np.unique(rows["sq"][test]):
            fold_of[names[s]] = fname
    C = json.load(open(HERE / "models" / "class_temperature.json"))
    classes = C["classes"]
    io_classes = {k: v for k, v in br.IO_OK.items()}
    res = {"classes": classes}
    for inp in args.inputs:
        W = np.load(HERE / "data" / "heads" / f"{inp}_square.npz")
        T = C["temperature"][inp]
        q = 1 - np.array(C["class_cutoffs"][inp]["90"])
        acc = {g: {"conf": [], "single": [], "size": []} for g in ("agree", "disagree")}
        follow = {"worldcover": 0, "io": 0, "other": 0, "n": 0}
        per_kind = {}
        pairs = {}
        referee = {"n": 0, "wc_right": 0, "io_right": 0, "head_right": 0, "head_conf_right": [], "head_conf_wrong": []}
        for si, name in enumerate(names):
            z = np.load(HERE / "data" / "squares" / f"{name}.npz")
            L = np.load(HERE / "data" / "labels" / f"{name}.npz")
            F = bc.square_feats(z) if inp == "all" else None
            d = 320 if inp == "all" else z[inp].shape[-1]
            net = bc.head_from(W, fold_of[name], d)
            io_years = list(L["io_years"])
            hgt = L["height"].astype(np.float32)
            hy = L["height_date"] // 10000
            for wy in (2020, 2021):
                yi = YEARS.index(wy)
                wc = z[f"wc{wy}"]
                io = L["io"][io_years.index(wy)]
                wcc = np.full(wc.shape, -1, np.int8)
                for code, c in br.WC2C.items():
                    wcc[wc == code] = c
                valid = (wcc >= 0) & np.isin(io, list(io_classes))
                agree = br.classes(wc, io) >= 0
                if inp == "all":
                    ok = F["aef"][1][yi] & F["tes"][1][yi] & F["olmo"][1][yi]
                    x = np.concatenate([F[e][0][yi] for e in bc.EMB], -1)
                else:
                    qv = z[inp][yi]
                    if inp == "tes":
                        s = z["tes_scale"][yi]
                        ok = np.isfinite(s) & (s > 0)
                        x = qv.astype(np.float32) * np.nan_to_num(s)[:, None]
                    else:
                        ok = qv[:, 0] != -128
                        x = bh.deq(qv)
                m = valid & ok
                with torch.no_grad():
                    lg = np.concatenate([net(torch.as_tensor(x[m][b:b + 100000].astype(np.float32)))[:, 2:].numpy()
                                         for b in range(0, int(m.sum()), 100000)])
                p = softmax(lg / T)
                pred = p.argmax(-1)
                conf = p.max(-1)
                members = p >= q[None]
                members[np.arange(len(p)), pred] = True
                size = members.sum(-1)
                ag = agree[m]
                for g, sel in (("agree", ag), ("disagree", ~ag)):
                    acc[g]["conf"].append(conf[sel])
                    acc[g]["single"].append(size[sel] == 1)
                    acc[g]["size"].append(size[sel])
                dis = ~ag
                wcm = wcc[m][dis]
                iom = io[m][dis]
                pr = pred[dis]
                in_io = np.zeros(dis.sum(), bool)
                for code, allowed in io_classes.items():
                    sel = iom == code
                    in_io[sel] = np.isin(pr[sel], list(allowed))
                follow["worldcover"] += int((pr == wcm).sum())
                follow["io"] += int((in_io & (pr != wcm)).sum())
                follow["other"] += int((~in_io & (pr != wcm)).sum())
                follow["n"] += int(dis.sum())
                pk = per_kind.setdefault(kinds[si], {"agree": 0, "disagree": 0, "single_dis": 0, "single_ag": 0})
                pk["agree"] += int(ag.sum())
                pk["disagree"] += int(dis.sum())
                pk["single_dis"] += int((size[dis] == 1).sum())
                pk["single_ag"] += int((size[ag] == 1).sum())
                # the most common disagreements, by name
                ion = {1: "water", 2: "trees", 4: "flooded veg", 5: "crops", 7: "built", 8: "bare", 11: "rangeland"}
                keys, cnts = np.unique(wcm.astype(np.int32) * 100 + iom, return_counts=True)
                for kk, cc in zip(keys, cnts):
                    nm = f"WC {classes[kk // 100]} / IO {ion.get(kk % 100, kk % 100)}"
                    pp = pairs.setdefault(nm, {"n": 0, "head_wc": 0, "head_io": 0, "single": 0})
                    sel = (wcm.astype(np.int32) * 100 + iom) == kk
                    pp["n"] += int(cc)
                    pp["head_wc"] += int((pr[sel] == wcm[sel]).sum())
                    pp["head_io"] += int((in_io[sel] & (pr[sel] != wcm[sel])).sum())
                    pp["single"] += int((size[dis][sel] == 1).sum())
                # trees settled by Meta's height
                wc_tree = wcm == 0
                io_tree = iom == 2
                tree_dis = wc_tree != io_tree
                hm = hgt[m][dis]
                hok = (hm != 255) & (hy[m][dis] >= 2017)
                tall, low = hok & (hm >= 5), hok & (hm < 2)
                sel = tree_dis & (tall | low)
                truth = tall[sel]
                referee["n"] += int(sel.sum())
                referee["wc_right"] += int((wc_tree[sel] == truth).sum())
                referee["io_right"] += int((io_tree[sel] == truth).sum())
                hr = (pr[sel] == 0) == truth
                referee["head_right"] += int(hr.sum())
                referee["head_conf_right"].append(conf[dis][sel][hr])
                referee["head_conf_wrong"].append(conf[dis][sel][~hr])
            del z, L, F
            if si % 20 == 19:
                print(f"  {inp} {si + 1}/{len(names)} [{psutil.Process().memory_info().rss / 1e9:.1f} GB]", flush=True)
        out = {}
        for g, v in acc.items():
            c = np.concatenate(v["conf"])
            out[g] = {"px": int(c.size), "mean_confidence": round(float(c.mean()), 3),
                      "share_single": round(float(np.concatenate(v["single"]).mean()), 3),
                      "mean_set_size": round(float(np.concatenate(v["size"]).mean()), 2)}
        out["disagree"]["share_of_pixels"] = round(out["disagree"]["px"] / (out["disagree"]["px"] + out["agree"]["px"]), 3)
        n = max(follow["n"], 1)
        out["follows"] = {k: round(v / n, 3) for k, v in follow.items() if k != "n"}
        out["per_kind"] = {k: {"disagree_share": round(v["disagree"] / max(v["agree"] + v["disagree"], 1), 3),
                               "single_agree": round(v["single_ag"] / max(v["agree"], 1), 3),
                               "single_disagree": round(v["single_dis"] / max(v["disagree"], 1), 3)} for k, v in per_kind.items()}
        top = sorted(pairs.items(), key=lambda kv: -kv[1]["n"])[:12]
        out["top_disagreements"] = {k: {"share": round(v["n"] / n, 3), "head_wc": round(v["head_wc"] / v["n"], 3),
                                        "head_io": round(v["head_io"] / v["n"], 3), "single": round(v["single"] / v["n"], 3)}
                                    for k, v in top}
        rn = max(referee["n"], 1)
        cr, cw = np.concatenate(referee["head_conf_right"]), np.concatenate(referee["head_conf_wrong"])
        out["trees_by_height"] = {"px": referee["n"], "worldcover_right": round(referee["wc_right"] / rn, 3),
                                  "io_right": round(referee["io_right"] / rn, 3), "head_right": round(referee["head_right"] / rn, 3),
                                  "head_conf_when_right": round(float(cr.mean()), 3) if cr.size else None,
                                  "head_conf_when_wrong": round(float(cw.mean()), 3) if cw.size else None}
        res[inp] = out
        print(f"{inp}: agree {out['agree']}  disagree {out['disagree']}", flush=True)
        print(f"   follows {out['follows']}  trees by height {out['trees_by_height']}", flush=True)
        for k, v in out["top_disagreements"].items():
            print(f"   {k:32s} {v}", flush=True)
        print("   per kind " + "  ".join(f"{k} {v['disagree_share']}/{v['single_agree']}/{v['single_disagree']}" for k, v in out["per_kind"].items()), flush=True)
    json.dump(res, open(HERE / "results" / "class_hard_ground.json", "w"), indent=1)


if __name__ == "__main__":
    main()
