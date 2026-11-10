# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "psutil"]
# ///
"""Fill-in networks for the map's comparison of 16, 32 and 64 bands: from k of AlphaEarth's 64 bands, rebuild the
other 64 - k, and the map's own head (web/public/models/biomass_aef.json) runs on the rebuilt vector.

Taught as section 41's fill+ (zoomed_out_more.py's train): the loss asks for the missing bands and for the map's head
to give the same answers on the rebuilt vector. 256 x 2 (section 41's "more" size, about 0.08 M weights): zoomed out a
view holds four times the pixels, and the browser runs the network on every one. Bands: band_subset.json's
map_bands_greedy (the 16 are inside the 32). Taught on the pool of zoomed_out_more.py (60,000 pixel-years per square
from 132 squares), the 92 squares' fold 0 held out; scored on those held-out squares' pool pixels at 10 m, against the
map's head, through the browser's own int8 round trip (rebuilt, scaled to unit length, quantized).

Writes web/public/models/fill_aef_<k>.json. RAM ~4 GB, stops over --max-gb.

    uv run scripts/export_fill.py [--ks 16 32]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import band_subset as bsub  # noqa: E402
import biomass_head as bh  # noqa: E402
import zoomed_out as zo  # noqa: E402
import zoomed_out_more as zm  # noqa: E402

MODELS = HERE / "web" / "public" / "models"
DEV = zm.DEV


def gb():
    return psutil.Process().memory_info().rss / 1e9


def map_head():
    """The map's head as a torch module on dequantized vectors."""
    h = json.load(open(MODELS / "biomass_aef.json"))
    m = bh.Head(64, np.array(h["mean"], np.float32), np.array(h["scale"], np.float32))
    with torch.no_grad():
        m.l1.weight.copy_(torch.tensor(h["w1"]).reshape(h["hidden"], 64))
        m.l1.bias.copy_(torch.tensor(h["b1"]))
        m.l2.weight.copy_(torch.tensor(h["w2"]).reshape(h["n_out"], h["hidden"]))
        m.l2.bias.copy_(torch.tensor(h["b2"]))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEV), h


def quant(f):
    """Float (dequantized) to int8 the way the stores hold it: q = sign(f) sqrt(|f|) 127.5."""
    return np.clip(np.round(np.sign(f) * np.sqrt(np.abs(f)) * 127.5), -127, 127).astype(np.int8)


def rebuild(net, ym, ys, q, bands, miss):
    """int8 rows (n, 64) -> the browser's rebuilt int8 rows from the k bands only."""
    f = zo.dq(q)
    r = f.copy()
    r[:, miss] = zm.run(net, f[:, bands]) * ys + ym
    out = quant(zo.unit(r))
    out[q[:, 0] == -128] = -128
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[16, 32])
    ap.add_argument("--per-sq", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=10)
    ap.add_argument("--prefix", action="store_true",
                    help="bands 0 to k-1 (the front of each Zarr chunk) instead of the greedy picks; writes fill_aef_<k>p.json")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    R, _ = bsub.load()
    names = [str(n) for n in R["names"]]
    extra = sorted(f[:-4] for f in os.listdir(HERE / "data" / "squares") if f.endswith(".npz") and f[:-4] not in names)
    held = np.flatnonzero(zo.square_folds(R) == 0)
    A, B, S = zm.read_pool(names + extra, args.per_sq)
    del B
    del R
    teacher, hj = map_head()
    tr = ~np.isin(S, held)
    At, Ah = A[tr], A[~tr]
    del A
    TA = zm.teach(teacher, At)
    print(f"pool {len(At):,} taught, {len(Ah):,} held out ({len(held)} squares), {gb():.1f} GB, "
          f"{time.time() - t0:.0f} s, device {DEV}", flush=True)
    # the map's answers on the held-out pixels, through the same int8 path
    ok = Ah[:, 0] != -128
    Ah = Ah[ok]
    T = zm.run(teacher, zo.dq(Ah))
    picks = json.load(open(HERE / "results" / "band_subset.json"))["map_bands_greedy"]
    summary = {}
    for k in args.ks:
        tk = time.time()
        bands = list(range(k)) if args.prefix else [int(b) for b in picks[str(k)]]
        miss = [b for b in range(64) if b not in bands]
        net, ym, ys = zm.train("fill+", At, At, TA, TA, bands, miss, teacher, 256, 2, args.epochs, 0)
        ym, ys = ym.cpu().numpy(), ys.cpu().numpy()
        q = rebuild(net, ym, ys, Ah, bands, miss)
        P = zm.run(teacher, zo.dq(q))
        agb_t, agb_p = np.expm1(np.minimum(T[:, 0], 8)), np.expm1(np.minimum(P[:, 0], 8))
        h_t, h_p = np.expm1(np.minimum(T[:, 1], 8)), np.expm1(np.minimum(P[:, 1], 8))
        sc = {"agb_off_median": round(float(np.median(np.abs(agb_t - agb_p))), 2),
              "agb_median": round(float(np.median(agb_t)), 1),
              "agb_r2_log": round(zo.r2(np.log1p(agb_t), np.log1p(agb_p)), 3),
              "h_off_median": round(float(np.median(np.abs(h_t - h_p))), 2),
              "cls_same": round(float((T[:, 2:].argmax(1) == P[:, 2:].argmax(1)).mean()), 3),
              "n": int(len(Ah))}
        summary[k] = sc
        # the network for the browser: x = (deq(q[bands]) - mu) / sd, Linear ReLU Linear ReLU Linear, then
        # missing = out * ys + ym; the vector scaled to unit length, quantized as quant()
        lin = [m for m in net.f if isinstance(m, torch.nn.Linear)]
        rng = np.random.default_rng(7)
        pi = rng.choice(len(Ah), 64, replace=False)
        out = {
            "name": f"fill_aef_{k}{'p' if args.prefix else ''}", "k": k, "bands": bands, "miss": miss, "hidden": 256, "layers": len(lin),
            "mu": net.mu.cpu().numpy().round(6).tolist(), "sd": net.sd.cpu().numpy().round(6).tolist(),
            "ym": ym.round(6).tolist(), "ys": ys.round(6).tolist(),
            "w": [l.weight.detach().cpu().numpy().astype(np.float32).ravel().round(6).tolist() for l in lin],
            "b": [l.bias.detach().cpu().numpy().astype(np.float32).ravel().round(6).tolist() for l in lin],
            "dims": [[l.out_features, l.in_features] for l in lin],
            # 64 held-out pixels: their 64 bands (int8, pixel-major) and the rebuilt vector the browser should give
            "probe_q": Ah[pi].astype(int).ravel().tolist(),
            "probe_out": q[pi].astype(int).ravel().tolist(),
            "held_out": sc,
        }
        json.dump(out, open(MODELS / f"fill_aef_{k}{'p' if args.prefix else ''}.json", "w"), separators=(",", ":"))
        print(f"  k {k}: {sc}, {time.time() - tk:.0f} s, {gb():.1f} GB", flush=True)
        if gb() > args.max_gb:
            raise SystemExit(f"STOP: RSS {gb():.1f} GB")
    # the held-out score of the 64 bands through the same round trip (quantization only)
    q64 = quant(zo.unit(zo.dq(Ah)))
    P = zm.run(teacher, zo.dq(q64))
    print("64 round trip, agb off median",
          round(float(np.median(np.abs(np.expm1(np.minimum(T[:, 0], 8)) - np.expm1(np.minimum(P[:, 0], 8))))), 3))
    res = HERE / "results" / "export_fill.json"
    allres = json.load(open(res)) if res.exists() else {}
    allres["prefix" if args.prefix else "greedy"] = summary
    json.dump(allres, open(res, "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s, {gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
