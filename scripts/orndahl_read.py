"""Window reads of the Orndahl et al. 30 m Arctic AGB tiles (Arctic Data Center doi:10.18739/A2NS0M06B) over HTTP.

The tiles are striped GeoTIFFs (one row per strip) served without a file size (Content-Range "bytes 0-1023/*"), which
GDAL's /vsicurl cannot open. This reads the TIFF tags with tifffile through ranged GETs, then fetches only the strips
of the rows wanted and decodes them. Tiles: 89,600 x 89,600, 30 m, EPSG:3571, uint16, nodata 65535.
"""
import io
import json
from pathlib import Path

import imagecodecs
import numpy as np
import requests
import tifffile

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data" / "labels_new" / "orndahl" / "list.json"
OBJ = "https://arcticdata.io/metacat/d1/mn/v2/object/{}"
ORIGIN, SIDE, RES, N = -4889340.0, 2688000.0, 30.0, 89600
_files = {}
_tags = {}


def files():
    if not _files:
        for d in json.load(open(LIST))["response"]["docs"]:
            if d.get("fileName", "").endswith(".tif"):
                _files[d["fileName"]] = (d["identifier"], int(d["size"]))
    return _files


class Ranged(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.s = url, size, 0, requests.Session()
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def get(self, a, b):
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
                r.raise_for_status()
                return r.content
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                import time
                time.sleep(2 + 3 * attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        B = 1 << 20
        out = bytearray()
        p, end = self.pos, min(self.pos + n, self.size)
        while p < end:
            k = p // B
            if k not in self.cache:
                self.cache[k] = self.get(k * B, min((k + 1) * B, self.size))
            blk = self.cache[k]
            take = blk[p - k * B: min(end - k * B, len(blk))]
            out += take
            p += len(take)
        self.pos = p
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def tile_of(x, y):
    c, r = int((x - ORIGIN) // SIDE), int((-ORIGIN - y) // SIDE)
    return "ABCD"[r] + str(c + 1)


def tags(fname):
    if fname not in _tags:
        ident, size = files()[fname]
        fh = Ranged(OBJ.format(ident), size)
        with tifffile.TiffFile(fh) as t:
            p = t.pages[0]
            _tags[fname] = {"offsets": np.asarray(p.dataoffsets, np.int64), "counts": np.asarray(p.databytecounts, np.int64),
                            "rps": p.rowsperstrip, "predictor": p.predictor, "compression": p.compression,
                            "shape": p.shape, "dtype": p.dtype, "origin": p.geotiff_tags.get("ModelTiepoint") if hasattr(p, "geotiff_tags") and p.geotiff_tags else None,
                            "url": OBJ.format(ident), "session": fh.s}
    return _tags[fname]


def window(layer, tile, r0, r1, c0, c1):
    """uint16 (r1 - r0, c1 - c0) of e.g. layer 'plant_agb_2020_p500' in tile 'A2' (rows and columns in the tile)."""
    T = tags(f"{layer}_{tile}.tif")
    rps = T["rps"]
    s0, s1 = r0 // rps, (r1 - 1) // rps + 1
    a, b = int(T["offsets"][s0:s1].min()), int((T["offsets"][s0:s1] + T["counts"][s0:s1]).max())
    for attempt in range(5):
        try:
            r = requests.get(T["url"], headers={"Range": f"bytes={a}-{b - 1}"}, timeout=300)
            r.raise_for_status()
            buf = r.content
            break
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            import time
            time.sleep(2 + 3 * attempt)
    W = T["shape"][1]
    rows = []
    for s in range(s0, s1):
        o, n = int(T["offsets"][s]) - a, int(T["counts"][s])
        raw = imagecodecs.lzw_decode(buf[o:o + n]) if int(T["compression"]) == 5 else buf[o:o + n]
        v = np.frombuffer(raw, T["dtype"]).reshape(-1, W)
        if int(T["predictor"]) == 2:
            v = np.cumsum(v, axis=1, dtype=T["dtype"])
        rows.append(v)
    A = np.concatenate(rows)[r0 - s0 * rps: r1 - s0 * rps, c0:c1]
    return A


def sample(layer, X, Y):
    """Nearest Orndahl value (float32 g/m2, NaN nodata) at EPSG:3571 points X, Y (one tile; points must share it)."""
    out = np.full(X.shape, np.nan, np.float32)
    tiles = np.array([tile_of(x, y) for x, y in zip(X.ravel()[[0, -1]], Y.ravel()[[0, -1]])])
    for t in sorted(set(tiles)):
        r0t, c0t = "ABCD".index(t[0]), int(t[1]) - 1
        tx0, ty0 = ORIGIN + c0t * SIDE, -ORIGIN - r0t * SIDE
        cc = np.floor((X - tx0) / RES).astype(int)
        rr = np.floor((ty0 - Y) / RES).astype(int)
        ins = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
        if not ins.any():
            continue
        A = window(layer, t, rr[ins].min(), rr[ins].max() + 1, cc[ins].min(), cc[ins].max() + 1)
        v = A[rr[ins] - rr[ins].min(), cc[ins] - cc[ins].min()].astype(np.float32)
        v[v == 65535] = np.nan
        out[ins] = v
    return out
