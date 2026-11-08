"""fire-knn's vegetation volume head (models/fuel_volume.json, copied from aef-hls-s1-fire-knn), in numpy.

Input: the year's AlphaEarth embedding, dequantized sign(q) (q / 127.5)^2, not renormalized (as fire-knn fit it).
Outputs: vol (cover x height of the leading lifeform, m), vc (its cover, 0..1), vh (its height, m), cc (canopy
cover, 0..1), ch (canopy height, m), 11 fuel-kind logits, 4 lifeform logits (none, tree, shrub, herb).
Trained on LANDFIRE in California only.
"""
import json
from pathlib import Path

import numpy as np

H = json.load(open(Path(__file__).resolve().parent.parent / "models" / "fuel_volume.json"))
MU, SD = np.array(H["mean"], np.float32), np.array(H["scale"], np.float32)
W1 = np.array(H["w1"], np.float32).reshape(H["hidden"], H["n_in"])
B1 = np.array(H["b1"], np.float32)
W2 = np.array(H["w2"], np.float32).reshape(H["n_out"], H["hidden"])
B2 = np.array(H["b2"], np.float32)
LIFE = H["life"]


def raw(q):
    """int8 AlphaEarth (n, 64) -> raw outputs (n, 20)."""
    f = q.astype(np.float32) / 127.5
    x = (np.sign(f) * f * f - MU) / SD
    return np.maximum(x @ W1.T + B1, 0) @ W2.T + B2


def heads(q):
    o = raw(q)
    vol = np.exp(o[:, 0] * (H["vol_hi"] - H["vol_lo"]) + H["vol_lo"]) - 0.02
    return {"vol": np.maximum(vol, 0), "cc": np.clip(o[:, 3], 0, 1), "ch": np.maximum(o[:, 4] * 50, 0),
            "life": o[:, 16:20].argmax(1)}


if __name__ == "__main__":
    q = np.array(H["probe_q"], np.int8).reshape(-1, 64)
    want = np.array(H["probe_out"], np.float32).reshape(len(q), -1)
    print("probe max abs diff", float(np.abs(raw(q) - want).max()), "life order", LIFE, "reg", H["reg"])
