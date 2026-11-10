import { Tiff, TiffTag } from "@cogeotiff/core";
import { asyncBufferFromUrl, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { Zstd } from "numcodecs";
import proj4 from "proj4";

import { RES, X0, Y0 } from "./aef";
import { cacheGet, cachePut, ZARR_TTL } from "./disk-cache";

/**
 * AlphaEarth's own GeoTIFFs (Source Cooperative tge-labs/aef/v1/annual), read at their full-resolution 10 m level
 * only. Each file is one year of one 8192 px UTM tile, planar: every band in its own 1024 x 1024 zstd tiles, so only
 * the bands asked for are read. The rows run south to north (a positive y pixel size).
 *
 * A chunk of the map's grid (the Zarr mosaic's: EPSG:4326, 256 px) takes, at each pixel center, the GeoTIFF pixel
 * under it (nearest), from the first file of the year that covers it with data. The band tiles are kept on disk as
 * read (compressed) and a few decoded in memory, so the chunks that share a tile, a later switch to more bands or a
 * step back to a year read nothing twice.
 */

const S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/";
const INDEX = `${S3}tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet`;
const NODATA = -128;
const TILE = 1024;

type Entry = { url: string; epsg: number; w: number; s: number; e: number; n: number };
type Opened = {
  url: string;
  offs: number[];
  cnts: number[];
  /** x = c + a col, y = f + e row (UTM meters) */
  a: number;
  c: number;
  e: number;
  f: number;
  W: number;
  H: number;
  nx: number;
  per: number;
  fwd: (p: [number, number]) => [number, number];
};

let index: Promise<Map<number, Entry[]>> | null = null;
/** Every file by year, from the index (5 MB, read once per worker). */
function files(): Promise<Map<number, Entry[]>> {
  if (index) return index;
  index = (async () => {
    const file = await asyncBufferFromUrl({ url: INDEX });
    const rows = (await parquetReadObjects({ file, compressors, columns: ["assets", "bbox", "proj:epsg"] })) as Record<string, unknown>[];
    const by = new Map<number, Entry[]>();
    for (const r of rows) {
      const href = (r["assets"] as { data: { href: string } }).data.href;
      const y = Number(/\/annual\/(\d{4})\//.exec(href)?.[1]);
      const bb = r["bbox"] as { xmin: number; ymin: number; xmax: number; ymax: number };
      const ent = { url: href.replace("s3://us-west-2.opendata.source.coop/", S3), epsg: Number(String(r["proj:epsg"]).split(":")[1]),
        w: bb.xmin, s: bb.ymin, e: bb.xmax, n: bb.ymax };
      let l = by.get(y);
      if (!l) by.set(y, (l = []));
      l.push(ent);
    }
    return by;
  })();
  index.catch(() => (index = null));
  return index;
}

const utm = (epsg: number) => {
  const z = epsg % 100, south = Math.floor(epsg / 100) === 327;
  const t = proj4(proj4.WGS84, `+proj=utm +zone=${z}${south ? " +south" : ""} +datum=WGS84 +units=m +no_defs`);
  return (p: [number, number]) => t.forward(p) as [number, number];
};

const range = async (url: string, off: number, len: number, signal?: AbortSignal) => {
  const r = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` }, cache: "no-store", signal });
  if (!r.ok) throw new Error(`${r.status} for ${url}`);
  const b = await r.arrayBuffer();
  if (b.byteLength !== len) throw new Error(`short read ${b.byteLength} of ${len} from ${url}`);
  return b;
};

const opened = new Map<string, Promise<Opened>>();
/** A file's header and its whole tile table (three requests), once. */
function open(ent: Entry): Promise<Opened> {
  let p = opened.get(ent.url);
  if (p) return p;
  p = (async () => {
    const tif = await Tiff.create({ url: new URL(ent.url), fetch: (o: number, l?: number) => range(ent.url, o, l ?? 16384) });
    const im = tif.images[0]!;
    const [offs, cnts, mt] = await Promise.all([
      im.fetch(TiffTag.TileOffsets), im.fetch(TiffTag.TileByteCounts), im.fetch(TiffTag.ModelTransformation),
    ]);
    if (im.tileSize.width !== TILE || im.tileSize.height !== TILE) throw new Error(`tile ${im.tileSize.width} in ${ent.url}`);
    let a: number, c: number, e: number, f: number;
    if (mt) [a, c, e, f] = [mt[0]!, mt[3]!, mt[5]!, mt[7]!];
    else {
      const [ox, oy] = im.origin, [rx, ry] = im.resolution;
      [a, c, e, f] = [rx, ox, -Math.abs(ry), oy];
    }
    return {
      url: ent.url, offs: Array.from(offs as ArrayLike<number>), cnts: Array.from(cnts as ArrayLike<number>), a, c, e, f,
      W: im.size.width, H: im.size.height, nx: im.tileCount.x, per: im.tileCount.x * im.tileCount.y, fwd: utm(ent.epsg),
    };
  })();
  opened.set(ent.url, p);
  p.catch(() => opened.delete(ent.url));
  return p;
}

const zstd = Zstd.fromConfig({ id: "zstd", level: 1 });
/** Decoded band tiles in memory, the last used last (1 MB each). */
const MEM_TILES = 96;
const mem = new Map<string, Promise<Int8Array | null>>();
const stats = { fetched: 0, fetchedBytes: 0, disk: 0, mem: 0 };
export const tiffStats = () => ({ ...stats });

/** One band's 1024 x 1024 tile, decoded; null where the file holds none (sparse). */
function bandTile(o: Opened, band: number, t: number): Promise<Int8Array | null> {
  const idx = band * o.per + t;
  const k = `${o.url}#${idx}`;
  const have = mem.get(k);
  if (have) {
    mem.delete(k);
    mem.set(k, have);
    stats.mem++;
    return have;
  }
  const p = (async () => {
    const off = o.offs[idx]!, len = o.cnts[idx]!;
    if (!len) return null;
    const dk = `tif|${k}`;
    let raw = await cacheGet(dk);
    if (raw) stats.disk++;
    else {
      raw = new Uint8Array(await range(o.url, off, len));
      stats.fetched++;
      stats.fetchedBytes += len;
      cachePut(dk, raw, ZARR_TTL);
    }
    const d = await zstd.decode(raw);
    if (d.byteLength !== TILE * TILE) throw new Error(`tile of ${d.byteLength} bytes`);
    return new Int8Array(d.buffer, d.byteOffset, d.byteLength);
  })();
  mem.set(k, p);
  p.catch(() => mem.delete(k));
  while (mem.size > MEM_TILES) mem.delete(mem.keys().next().value!);
  return p;
}

const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

/**
 * One chunk of the map's grid for a year, only the bands asked for (bands[0] tells nodata; AlphaEarth's nodata is all
 * bands at once): band-major int8 [bands.length, w * h], -128 where no file has data.
 */
export async function readChunk(year: number, py0: number, px0: number, h: number, w: number, bands: number[], signal?: AbortSignal): Promise<Int8Array> {
  const n = w * h, nb = bands.length;
  const out = new Int8Array(nb * n).fill(NODATA);
  const lw = X0 + px0 * RES, le = X0 + (px0 + w) * RES, ln = Y0 - py0 * RES, ls = Y0 - (py0 + h) * RES;
  const all = (await files()).get(year) ?? [];
  // the files over the chunk; those of the chunk's own UTM zone first
  const zone = Math.floor(((lw + le) / 2 + 180) / 6) + 1;
  const cands = all.filter((f) => f.w < le && f.e > lw && f.s < ln && f.n > ls)
    .sort((a, b) => Number(b.epsg % 100 === zone) - Number(a.epsg % 100 === zone));
  const filled = new Uint8Array(n);
  let left = n;
  // pixel centers into each file's UTM on a 17 x 17 lattice, bilinear between (a 2.6 km chunk is flat to well under a cm)
  const G = 16, sx = Math.max(1, w - 1), sy = Math.max(1, h - 1);
  for (const ent of cands) {
    if (!left) break;
    if (signal?.aborted) throw aborted();
    const o = await open(ent);
    const LE = new Float64Array((G + 1) * (G + 1)), LN = new Float64Array((G + 1) * (G + 1));
    for (let gi = 0; gi <= G; gi++)
      for (let gj = 0; gj <= G; gj++) {
        const [E, N] = o.fwd([X0 + (px0 + (sx * gj) / G + 0.5) * RES, Y0 - (py0 + (sy * gi) / G + 0.5) * RES]);
        LE[gi * (G + 1) + gj] = E;
        LN[gi * (G + 1) + gj] = N;
      }
    // the pixels this file covers, by its tile: [pixel, place in the tile] pairs
    const byTile = new Map<number, number[]>();
    for (let i = 0; i < h; i++) {
      const fy = (i * G) / sy, g0 = Math.min(G - 1, Math.floor(fy)), ty = fy - g0;
      for (let j = 0; j < w; j++) {
        const p = i * w + j;
        if (filled[p]) continue;
        const fx = (j * G) / sx, h0 = Math.min(G - 1, Math.floor(fx)), tx = fx - h0;
        const k00 = g0 * (G + 1) + h0, k10 = k00 + G + 1;
        const E = (1 - ty) * ((1 - tx) * LE[k00]! + tx * LE[k00 + 1]!) + ty * ((1 - tx) * LE[k10]! + tx * LE[k10 + 1]!);
        const N = (1 - ty) * ((1 - tx) * LN[k00]! + tx * LN[k00 + 1]!) + ty * ((1 - tx) * LN[k10]! + tx * LN[k10 + 1]!);
        const col = Math.floor((E - o.c) / o.a), row = Math.floor((N - o.f) / o.e);
        if (col < 0 || row < 0 || col >= o.W || row >= o.H) continue;
        const t = (row >> 10) * o.nx + (col >> 10);
        let l = byTile.get(t);
        if (!l) byTile.set(t, (l = []));
        l.push(p, (row & (TILE - 1)) * TILE + (col & (TILE - 1)));
      }
    }
    for (const [t, l] of byTile) {
      if (signal?.aborted) throw aborted();
      const tiles = await Promise.all(bands.map((b) => bandTile(o, b, t)));
      const t0 = tiles[0];
      if (!t0) continue;
      for (let q = 0; q < l.length; q += 2) {
        const p = l[q]!, at = l[q + 1]!;
        if (filled[p] || t0[at] === NODATA) continue;
        filled[p] = 1;
        left--;
        for (let bi = 0; bi < nb; bi++) out[bi * n + p] = tiles[bi]?.[at] ?? NODATA;
      }
    }
  }
  return out;
}
