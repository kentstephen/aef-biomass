/**
 * Earth Genome's Sentinel-2 yearly mosaics in true color, read and drawn in the page: Earthwork's imagery
 * (aef-earthwork-web/web/src/ui.js: s2Discover, s2Tiff, s2TileData, S2Look, s2CogLayers), the TCI path only.
 * The STAC search names the footprints under the view (s2-items.ts); each footprint's TCI COG on data.source.coop
 * is opened and its tiles decoded in workers by deck.gl-geotiff (MosaicLayer + COGLayer). The year shown loads
 * first, the other years once it is in, so a step between them is instant. Yearly footprints paint over the fill
 * ones (cloud gaps).
 */
import { SourceHttp } from "@chunkd/source-http";
import { COGLayer, MosaicLayer } from "@developmentseed/deck.gl-geotiff";
import { CreateTexture } from "@developmentseed/deck.gl-raster/gpu-modules";
import { DecoderPool, GeoTIFF, PerOriginSemaphore } from "@developmentseed/geotiff";
import GeoTiffWorker from "@developmentseed/geotiff/pool/worker?worker";
import type * as maplibregl from "maplibre-gl";

import { s2Items } from "./s2-items";

/** The years with yearly mosaics (2025 from the bucket where the STAC lags it, s2-items.ts). */
export const S2Y = [2022, 2023, 2024, 2025];
const MIN_Z = 7;

// every COG read tries again when the bucket is busy: data.source.coop answers some of a burst with 503, often
// without CORS headers (fetch only says "Failed to fetch"), and a tile that fails is not asked for again. A busy
// answer waits 0.4, 0.8, 1.6 s (and a little more at random) inside its slot (Earthwork's ui.js)
const S2_BUSY = new Set([429, 500, 502, 503, 504]);
const s2Wait = (ms: number, signal?: AbortSignal | null) =>
  new Promise<void>((ok, no) => {
    if (signal?.aborted) return no(signal.reason);
    const t = setTimeout(ok, ms);
    signal?.addEventListener("abort", () => (clearTimeout(t), no(signal.reason)), { once: true });
  });
SourceHttp.fetch = async (url: string | URL | Request, init?: RequestInit) => {
  const signal = init?.signal;
  for (let i = 0; ; i++) {
    const last = i === 3;
    try {
      const res = await fetch(url, init);
      if (last || !S2_BUSY.has(res.status)) return res;
    } catch (e) {
      if (last || signal?.aborted || (e as Error)?.name === "AbortError") throw e;
    }
    await s2Wait(400 * 2 ** i + Math.random() * 300, signal);
  }
};

type Src = { id: string; url: string; bbox: [number, number, number, number] };
type Batch = { key: string; yearly: Src[]; fill: Src[]; bbox: [number, number, number, number] };

/**
 * nodata (alpha 0) is dropped; at a footprint edge the filtered color is the edge pixel's color times the filtered
 * alpha (nodata is black), so dividing by alpha gives the pixel back with no dark fringe (Earthwork's S2Look).
 */
const S2Look = {
  name: "s2Look",
  fs: `uniform s2LookUniforms {
  float gamma;
} s2Look;
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  color = vec4(pow(color.rgb / color.a, vec3(1.0 / s2Look.gamma)), 1.0);
`,
  },
  uniformTypes: { gamma: "f32" },
  getUniforms: (q: { gamma: number }) => ({ gamma: q.gamma }),
};

/**
 * The imagery for the map(s). `views` gives the maps showing it (the main map while Space is held, the pair's left);
 * `onChange` asks for a redraw when footprints arrive or the year shown is in.
 */
export function createS2(views: () => maplibregl.Map[], onChange: () => void) {
  let pool: DecoderPool | null = null;
  const poolGet = () =>
    pool || (pool = new DecoderPool({ size: Math.min(6, navigator.hardwareConcurrency || 4), createWorker: () => new GeoTiffWorker() }));
  const tiffs = new Map<string, Promise<GeoTIFF>>();
  const src = new Map<number, { ids: Set<string>; batches: Batch[] }>();
  // the requests to data.source.coop, 16 at a time (HTTP/2), the year shown first, yearly before fill
  const limiter = new PerOriginSemaphore({ maxRequests: 16 });
  const asked = new Set<string>();
  const stat = { tiles: 0, pending: 0, failed: 0 };
  let shown = S2Y[S2Y.length - 1]!;
  let warm = false;

  const tiff = (url: string, opts: { getPriority?: () => number } | undefined, year: number, fill: boolean) => {
    let p = tiffs.get(url);
    if (!p) {
      const near = opts?.getPriority;
      const getPriority = () => [year === shown ? 0 : 1, fill ? 1 : 0, near ? near() : 0];
      p = GeoTIFF.fromUrl(url, { concurrencyLimiter: limiter, getPriority } as never);
      tiffs.set(url, p);
      p.catch(() => tiffs.get(url) === p && tiffs.delete(url));
    }
    return p;
  };

  // the view and half of it again on every side: where footprints are looked for and mounted
  function view(): [number, number, number, number] {
    const bs = views().map((m) => m.getBounds());
    const W = Math.min(...bs.map((b) => b.getWest())), E = Math.max(...bs.map((b) => b.getEast()));
    const S = Math.min(...bs.map((b) => b.getSouth())), N = Math.max(...bs.map((b) => b.getNorth()));
    const dx = (E - W) / 2, dy = (N - S) / 2;
    return [W - dx, Math.max(-85, S - dy), E + dx, Math.min(85, N + dy)];
  }

  function discover() {
    const ms = views();
    if (!ms.length) return;
    const z = Math.max(MIN_Z, Math.min(9, Math.floor(ms[0]!.getZoom()))), n = 2 ** z;
    const tx = (lon: number) => Math.min(n - 1, Math.max(0, Math.floor(((lon + 180) / 360) * n)));
    const ty = (lat: number) => {
      const v = Math.sin((lat * Math.PI) / 180);
      return Math.min(n - 1, Math.max(0, Math.floor((0.5 - Math.log((1 + v) / (1 - v)) / (4 * Math.PI)) * n)));
    };
    const [W, S, E, N] = view(), x0 = tx(W), x1 = tx(E), y0 = ty(N), y1 = ty(S);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;
    for (const year of [shown, ...S2Y.filter((y) => y !== shown)])
      for (let x = x0; x <= x1; x++)
        for (let y = y0; y <= y1; y++) {
          const k = `${year}/${z}/${x}/${y}`;
          if (asked.has(k)) continue;
          asked.add(k);
          s2Items(year, { z, x, y }).then(
            (items) => {
              let e = src.get(year);
              if (!e) src.set(year, (e = { ids: new Set(), batches: [] }));
              const bt: Batch = { key: k.replace(/\//g, "-"), yearly: [], fill: [], bbox: [180, 90, -180, -90] };
              for (const it of items) {
                if (!it.bbox || e.ids.has(it.id)) continue;
                e.ids.add(it.id);
                bt[it.fill ? "fill" : "yearly"].push({ id: it.id, url: it.url, bbox: it.bbox });
                bt.bbox = [Math.min(bt.bbox[0], it.bbox[0]), Math.min(bt.bbox[1], it.bbox[1]), Math.max(bt.bbox[2], it.bbox[2]), Math.max(bt.bbox[3], it.bbox[3])];
              }
              if (bt.yearly.length || bt.fill.length) {
                e.batches.push(bt);
                onChange();
              }
            },
            () => asked.delete(k),
          );
        }
  }

  async function tileData(image: any, { device, x, y, signal, pool: p }: any, year: number) {
    stat.pending++;
    try {
      const { array } = await image.fetchTile(x, y, { boundless: false, pool: p, signal });
      const { width, height, data } = array, px = width * height;
      // TCI's black (0, 0, 0) is nodata: alpha 0 there, so the linear filter's blend of an edge pixel with its
      // nodata neighbor carries a falling alpha the shader can undo
      let rgba = data;
      if (data.length === 3 * px) {
        rgba = new Uint8Array(4 * px);
        for (let i = 0; i < px; i++) {
          const r = data[3 * i], g = data[3 * i + 1], b = data[3 * i + 2];
          rgba[4 * i] = r;
          rgba[4 * i + 1] = g;
          rgba[4 * i + 2] = b;
          rgba[4 * i + 3] = r || g || b ? 255 : 0;
        }
      }
      stat.tiles++;
      return { texture: device.createTexture({ data: rgba, format: "rgba8unorm", width, height, sampler: { magFilter: "linear", minFilter: "linear" } }), width, height };
    } catch (e) {
      if (!signal?.aborted) stat.failed++;
      throw e;
    } finally {
      stat.pending--;
      // the year shown is in: mount the others
      if (!warm && stat.pending === 0 && year === shown) {
        warm = true;
        setTimeout(onChange, 250);
      }
    }
  }

  /**
   * One year's layers for one map (`side` keeps the ids apart). Shown or hidden by visible, not opacity: the tile
   * layers inside MosaicLayer and COGLayer keep each tile's layer as it came in, while deck.gl checks every parent's
   * visible on each draw. Hidden years still load their tiles.
   */
  function yearLayers(year: number, side: string, visible: boolean, beforeId?: string) {
    poolGet();
    const e = src.get(year);
    if (!e) return [];
    const [W, S, E, N] = view();
    const live = e.batches.filter((bt) => bt.bbox[0] < E && bt.bbox[2] > W && bt.bbox[1] < N && bt.bbox[3] > S);
    const parts = [
      ...live.filter((bt) => bt.fill.length).map((bt) => [bt, "fill"] as const),
      ...live.filter((bt) => bt.yearly.length).map((bt) => [bt, "yearly"] as const),
    ];
    return parts.map(
      ([bt, part]) =>
        new MosaicLayer({
          id: `s2${side}-${bt.key}-${part}`,
          sources: bt[part],
          maxCacheSize: 0,
          minZoom: MIN_Z,
          visible,
          // read by the MapLibre overlay: the imagery draws under the basemap's labels
          beforeId,
          // a footprint that cannot be read is nothing there (a rejection would leave MosaicLayer a tile with no content)
          getSource: (s: Src, o: any) => tiff(s.url, o, year, part === "fill").catch(() => null),
          renderSource: (s: Src, { data, signal }: any) =>
            !data
              ? null
              : new COGLayer({
                  id: `s2${side}-${year}-${s.id}`,
                  geotiff: data,
                  getTileData: (img: any, o: any) => tileData(img, o, year),
                  pool: pool!,
                  signal,
                  refinementStrategy: "best-available",
                  maxRequests: 16,
                  renderTile: (d: any) => ({
                    renderPipeline: [{ module: CreateTexture, props: { textureName: d.texture } }, { module: S2Look, props: { gamma: 1 } }],
                  }),
                } as never),
        } as never),
    );
  }

  /** The layers for a map: the year shown, and once it is in, the other years hidden (their tiles load ahead). */
  function layers(year: number, side: string, beforeId?: string) {
    if (year !== shown) {
      shown = year;
    }
    discover();
    const ys = warm ? [year, ...S2Y.filter((y) => y !== year)] : [year];
    return ys.flatMap((y) => yearLayers(y, side, y === year, beforeId));
  }

  return { layers, stat, MIN_Z };
}

/** The imagery year for a map year: the same year where there is a mosaic, else the nearest. */
export const s2YearFor = (y: number) => S2Y.reduce((a, b) => (Math.abs(b - y) < Math.abs(a - y) ? b : a));
