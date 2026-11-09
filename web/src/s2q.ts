/**
 * Sentinel-2 imagery for years EOX has no clean mosaic for (2018), from the Copernicus Sentinel-2 Global Mosaics
 * (ESA, quarterly cloudless composites, 10 m) that TGE Labs mirrors on Source Cooperative: one COG per band (B04
 * red, B03 green, B02 blue) per 100 km tile, int16 reflectance x 10000, in the tile's UTM zone.
 *
 * Drawn as ordinary web mercator map tiles, made in the browser: the quarter's tile index (items.parquet) is read
 * once, then each map tile reads the band blocks under it at the overview nearest its own pixel size, reprojects
 * from UTM (a 17 x 17 grid of exact points, linear between them) and colors the reflectance. Nothing is stored
 * but a small in-memory cache of decoded blocks.
 *
 * One quarter stands in for the year: July to September (Q3) from the Tropic of Capricorn north, January to March
 * (Q1, summer) south of it, so neither has the winter's snow and the southern tropics avoid their wet-season cloud.
 */
import { defaultDecoderPool, GeoTIFF, type Overview } from "@developmentseed/geotiff";
import { asyncBufferFromUrl, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import proj4 from "proj4";

const ROOT = "https://data.source.coop/tge-labs/sentinel-2-quarterly-cloudless-mosaics/";
export const S2Q_CREDIT =
  "Copernicus Sentinel-2 Global Mosaics 2018 (ESA, contains modified Copernicus Sentinel data 2018) via TGE Labs on Source Cooperative";
/** The latitude where the quarter switches (the Tropic of Capricorn). */
const SOUTH = -23.44;
export const quarterAt = (lat: number) => (lat >= SOUTH ? "Q3" : "Q1");
/** Reflectance (x 10000) to 0..255: 0 to REFL_TOP linear, then a gamma that lifts the darks. */
const REFL_TOP = 2500;
const GAMMA = 1 / 1.6;
const LUT = new Uint8ClampedArray(REFL_TOP + 1).map((_, v) => Math.round(255 * (v / REFL_TOP) ** GAMMA));
const NODATA = -32768;
const SIZE = 256;
const GRID = 16;

type Item = { w: number; s: number; e: number; n: number; epsg: number; r: string; g: string; b: string };
const index = new Map<string, Promise<Item[]>>();

/** The quarter's tiles (bbox, UTM code, band hrefs), read once from its items.parquet. */
function itemsOf(year: number, q: string): Promise<Item[]> {
  const k = `${year}.${q}`;
  let p = index.get(k);
  if (!p) {
    p = (async () => {
      const file = await asyncBufferFromUrl({ url: `${ROOT}mosaics/quarter=${k}/items.parquet` });
      const rows = (await parquetReadObjects({ file, compressors, columns: ["bbox", "proj:code", "B04_href", "B03_href", "B02_href"] })) as Record<string, unknown>[];
      return rows.map((r) => {
        const bb = r["bbox"] as number[];
        return { w: bb[0]!, s: bb[1]!, e: bb[2]!, n: bb[3]!, epsg: Number(String(r["proj:code"]).split(":")[1]),
                 r: String(r["B04_href"]), g: String(r["B03_href"]), b: String(r["B02_href"]) };
      });
    })();
    p.catch(() => index.delete(k));
    index.set(k, p);
  }
  return p;
}

const tiffs = new Map<string, Promise<GeoTIFF>>();
function tiff(url: string): Promise<GeoTIFF> {
  let p = tiffs.get(url);
  if (!p) {
    p = GeoTIFF.fromUrl(url);
    p.catch(() => tiffs.delete(url));
    tiffs.set(url, p);
    // a few hundred open headers at most (each holds a small header cache)
    if (tiffs.size > 300) tiffs.delete(tiffs.keys().next().value!);
  }
  return p;
}

let pool: ReturnType<typeof defaultDecoderPool> | null = null;
/** Decoded blocks by url, level and block, the last used last; at most CAP_BYTES. */
const blocks = new Map<string, Promise<{ data: Int16Array; w: number; h: number }>>();
const blockBytes = new Map<string, number>();
let held = 0;
const CAP_BYTES = 192 * 1024 ** 2;
function block(img: GeoTIFF | Overview, key: string, tx: number, ty: number) {
  const have = blocks.get(key);
  if (have) {
    blocks.delete(key);
    blocks.set(key, have);
    return have;
  }
  pool ??= defaultDecoderPool();
  const p = img.fetchTile(tx, ty, { pool, boundless: true }).then((t) => {
    const a = t.array;
    const data = (a.layout === "band-separate" ? a.bands[0] : a.data) as Int16Array;
    blockBytes.set(key, data.byteLength);
    held += data.byteLength;
    for (const [k] of blocks) {
      if (held <= CAP_BYTES) break;
      if (k === key) continue;
      blocks.delete(k);
      held -= blockBytes.get(k) ?? 0;
      blockBytes.delete(k);
    }
    return { data, w: a.width, h: a.height };
  });
  p.catch(() => blocks.delete(key));
  blocks.set(key, p);
  return p;
}

const projs = new Map<number, { forward: (p: [number, number]) => number[] }>();
function toUtm(epsg: number) {
  let c = projs.get(epsg);
  if (!c) {
    const zone = epsg % 100, south = Math.floor(epsg / 100) === 327;
    c = proj4("EPSG:4326", `+proj=utm +zone=${zone}${south ? " +south" : ""} +datum=WGS84 +units=m +no_defs`) as { forward: (p: [number, number]) => number[] };
    projs.set(epsg, c);
  }
  return c;
}

/** The finest image whose pixel is at least the map tile's own (in meters); the full image if none is. */
function levelFor(g: GeoTIFF, meters: number): { img: GeoTIFF | Overview; lvl: number } {
  const imgs: (GeoTIFF | Overview)[] = [g, ...g.overviews];
  let best = 0;
  for (let i = 0; i < imgs.length; i++) if (Math.abs(imgs[i]!.transform[0]) <= meters * 1.01) best = i;
  return { img: imgs[best]!, lvl: best };
}

const lonOf = (x: number, z: number) => (x / 2 ** z) * 360 - 180;
const latOf = (y: number, z: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** z))) * 180) / Math.PI;

/**
 * One web mercator map tile of the year's imagery, as RGBA; null where no tile of the mosaic covers it. Pixels with
 * no data stay clear.
 */
export async function s2qTile(year: number, z: number, x: number, y: number, signal?: AbortSignal): Promise<ImageData | null> {
  const w = lonOf(x, z), e = lonOf(x + 1, z), n = latOf(y, z), s = latOf(y + 1, z);
  const qs = [...new Set([quarterAt(n), quarterAt(s)])];
  const lists = await Promise.all(qs.map((q) => itemsOf(year, q).then((l) => [q, l] as const)));
  const cands: { q: string; it: Item }[] = [];
  for (const [q, l] of lists) for (const it of l) if (it.e > w && it.w < e && it.n > s && it.s < n) cands.push({ q, it });
  if (!cands.length || signal?.aborted) return null;
  // meters per map pixel at the tile's middle
  const meters = ((40075016.686 / 2 ** z / SIZE) * Math.cos((((n + s) / 2) * Math.PI) / 180));
  const out = new ImageData(SIZE, SIZE);
  const done = new Uint8Array(SIZE * SIZE);
  // each pixel's latitude (rows) and longitude (columns), at pixel centers
  const lat = Float64Array.from({ length: SIZE }, (_, r) => latOf(y + (r + 0.5) / SIZE, z));
  for (const { q, it } of cands) {
    const [gr, gg, gb] = await Promise.all([tiff(it.r), tiff(it.g), tiff(it.b)]);
    if (signal?.aborted) return null;
    const { img: ir, lvl } = levelFor(gr, meters);
    const ig = lvl === 0 ? gg : gg.overviews[lvl - 1]!, ib = lvl === 0 ? gb : gb.overviews[lvl - 1]!;
    const [a, , c, , ee, f] = ir.transform;
    const tw = ir.tileWidth, th = ir.tileHeight, W = ir.width, H = ir.height;
    // the UTM position of a grid of exact points, linear between them
    const P = toUtm(it.epsg);
    const gx = new Float64Array((GRID + 1) ** 2), gy = new Float64Array((GRID + 1) ** 2);
    for (let j = 0; j <= GRID; j++)
      for (let i = 0; i <= GRID; i++) {
        const [X, Y] = P.forward([lonOf(x + i / GRID, z), latOf(y + j / GRID, z)]);
        gx[j * (GRID + 1) + i] = X!;
        gy[j * (GRID + 1) + i] = Y!;
      }
    // source pixel (col, row) of every map pixel this item may fill; -1 outside it
    const col = new Int32Array(SIZE * SIZE).fill(-1), row = new Int32Array(SIZE * SIZE);
    const need = new Set<number>();
    const step = SIZE / GRID;
    for (let r = 0; r < SIZE; r++) {
      const fj = (r + 0.5) / step, j = Math.min(GRID - 1, Math.floor(fj)), v = fj - j;
      const qq = quarterAt(lat[r]!);
      if (qq !== q) continue;
      for (let cc = 0; cc < SIZE; cc++) {
        const p = r * SIZE + cc;
        if (done[p]) continue;
        const fi = (cc + 0.5) / step, i = Math.min(GRID - 1, Math.floor(fi)), u = fi - i;
        const k0 = j * (GRID + 1) + i, k1 = k0 + GRID + 1;
        const X = (1 - v) * ((1 - u) * gx[k0]! + u * gx[k0 + 1]!) + v * ((1 - u) * gx[k1]! + u * gx[k1 + 1]!);
        const Y = (1 - v) * ((1 - u) * gy[k0]! + u * gy[k0 + 1]!) + v * ((1 - u) * gy[k1]! + u * gy[k1 + 1]!);
        const sc = Math.floor((X - c) / a), sr = Math.floor((Y - f) / ee);
        if (sc < 0 || sr < 0 || sc >= W || sr >= H) continue;
        col[p] = sc;
        row[p] = sr;
        need.add(Math.floor(sr / th) * 65536 + Math.floor(sc / tw));
      }
    }
    if (!need.size) continue;
    const got = new Map<number, [Int16Array, Int16Array, Int16Array, number]>();
    await Promise.all([...need].map(async (k) => {
      const ty = Math.floor(k / 65536), tx = k % 65536;
      const [br, bg, bb] = await Promise.all([[ir, it.r], [ig, it.g], [ib, it.b]].map(([im, u]) =>
        block(im as GeoTIFF | Overview, `${u}|${lvl}|${tx}|${ty}`, tx, ty)));
      got.set(k, [br!.data, bg!.data, bb!.data, br!.w]);
    }));
    if (signal?.aborted) return null;
    const px = out.data;
    for (let p = 0; p < SIZE * SIZE; p++) {
      const sc = col[p]!;
      if (sc < 0) continue;
      const sr = row[p]!;
      const k = Math.floor(sr / th) * 65536 + Math.floor(sc / tw);
      const B = got.get(k);
      if (!B) continue;
      const o = (sr % th) * B[3] + (sc % tw);
      const R = B[0][o]!, G = B[1][o]!, Bl = B[2][o]!;
      if (R === NODATA || G === NODATA || Bl === NODATA) continue;
      px[4 * p] = LUT[Math.max(0, Math.min(REFL_TOP, R))]!;
      px[4 * p + 1] = LUT[Math.max(0, Math.min(REFL_TOP, G))]!;
      px[4 * p + 2] = LUT[Math.max(0, Math.min(REFL_TOP, Bl))]!;
      px[4 * p + 3] = 255;
      done[p] = 1;
    }
  }
  return out;
}
