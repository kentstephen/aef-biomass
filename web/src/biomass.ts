import type { MinimalTileData, RasterModule } from "@developmentseed/deck.gl-raster";
import { ZarrLayer, ZarrSource } from "@developmentseed/deck.gl-zarr";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import * as zarr from "zarrita";

import { CHUNK, MIN_ZOOM, openMosaic, RES, X0, Y0, YEAR_ORIGIN } from "./aef";
import { CH_NODATA, CONF_SCALE, type ConfTables, CUT, LOSS_FROM, MG_SCALE, PCT_SCALE } from "./change";
import type { FillNet } from "./fill";
import type { GainHead } from "./gain";
import { type BiomassHead, CODE, codeValue } from "./head";
import { Pool } from "./pool";
import { CHANGE_50, glslRamp, LAND_COVER, LIGHT_HI, LIGHT_LO, toOklab, VIRIDIS_50 } from "./ramp";

/** One chunk on the GPU: a year's biomass codes and classes side by side (rg16uint) or a window's change, percent x 100 and confidence x 10000 (rg16sint). */
type TileData = MinimalTileData & { key: string; tex: Texture };

/**
 * A ZarrLayer that says when deck has taken it off the map. deck applies a new layer list at its next frame, not at
 * setProps: a key taken off and put back before that frame keeps its old layer and tiles (deck hands the state to the
 * new layer of the same id) and is never finalized. So a layer's textures are freed here, never on a timer.
 */
class BioLayer extends ZarrLayer<TileData, "int8"> {
  static override layerName = "BioLayer";
  override finalizeState(context: Parameters<ZarrLayer<TileData, "int8">["finalizeState"]>[0]) {
    super.finalizeState(context);
    (this.props as unknown as { onFinalize?: () => void }).onFinalize?.();
  }
}
/**
 * One chunk's planes on the page: a year's [agb codes n, height codes n, class n] (Uint16), or a window's
 * [percent x 100 n, Mg/ha x 4 n, confidence x 10000 n, classes n] (Int16, change.ts).
 */
type Planes = { planes: Uint16Array | Int16Array; w: number; h: number; rough?: boolean };

/**
 * A layer's key: a year ("2025") or a window ("2018-2025", the change from the first year to the second); a window
 * ending in L draws gain by the lidar-taught network (gain.ts) instead of the two-part floor.
 */
export const yearKey = (y: number) => String(y);
export const windowKey = (a: number, b: number, lidarGain = false) => `${a}-${b}${lidarGain ? "L" : ""}`;
const windowOf = (lk: string) => (lk.includes("-") ? (lk.replace("L", "").split("-").map(Number) as [number, number]) : null);

/** The change ramp's ends, percent of the pixel's own biomass: loss to all of it, gain to one and a half times it. */
export const LOSS_END = 100;
export const GAIN_END = 150;
/**
 * Opacity from confidence, not from size: a pixel is drawn at ALPHA_MIN at the low end of its confidence and fully
 * at the top. Loss from LOSS_FROM (below it nothing is drawn), gain from GAIN_FROM (it is drawn wherever it passes
 * the floor, faint where steady ground of its class often rises as much). Exactly 0 is not drawn at all.
 */
export const GAIN_FROM = 0.5;

/** The card's live histogram of change in view: columns along the change ramp, one per ramp stop. */
export const HIST_N = 50;
type SideStats = { ha: number; t: number; pairs: { from: string; to: string; ha: number; t: number }[] };
export type ChangeStats = {
  sure: Float64Array;
  faint: Float64Array;
  loss: SideStats;
  gain: SideStats;
  /** where the head's height and class say the forest was cut (change.ts): hectares only, no amount */
  cut: SideStats;
  /** hectares in view with data in both years; chunks of the view counted and wanted; every k-th pixel counted */
  dataHa: number;
  inHand: number;
  wanted: number;
  every: number;
};
export const ALPHA_MIN = 0.2;

/** The biomass ramp's ends in log1p(Mg/ha): 0 to 350 Mg/ha, the same in every view and year. */
export const AGB_LO = 0;
export const AGB_HI = Math.log1p(350);
/** The height view's top, m: lightness runs over log1p(0..HEIGHT_HI). */
export const HEIGHT_HI = 40;
/** The relief view's slope: this many units of rise per unit of log biomass between neighbors. */
export const RELIEF_Z = 3;

/**
 * A year per pixel, by the view's VIEW_CODE (ramp.ts). 0: biomass on viridis over AGB_LO..AGB_HI in log1p(Mg/ha). The class
 * views take the head's class color's hue and chroma (OKLab) and keep the detail: 1, its lightness from biomass over
 * the same range; 2, its lightness from canopy height over 0..HEIGHT_HI; 3, its flat color lit as a relief of log
 * biomass (light from the northwest, RELIEF_Z times the log step between neighbors as the slope).
 * The texel: biomass code, then the height code with the class in its low four bits.
 */
const BiomassColor = {
  name: "biomass",
  fs: `uniform biomassUniforms {
  float lo;
  float hi;
  float ramp;
} biomass;`,
  inject: {
    "fs:#decl": `precision highp usampler2D;
uniform highp usampler2D agbTex;
${glslRamp("viridis", VIRIDIS_50)}
const vec3 CLASS_LAB[${LAND_COVER.length}] = vec3[${LAND_COVER.length}](${LAND_COVER.map(([, h]) => `vec3(${toOklab(h).map((v) => v.toFixed(5)).join(", ")})`).join(", ")});
vec3 oklabToSrgb(vec3 c) {
  float l = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
  float m = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
  float s = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
  l = l * l * l; m = m * m * m; s = s * s * s;
  vec3 lin = clamp(vec3(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s), 0.0, 1.0);
  return mix(12.92 * lin, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
}
// the class's hue at lightness t (0 least, 1 most), as ramp.ts classShade
vec3 classShade(int k, float t) {
  float L = ${LIGHT_HI.toFixed(3)} + (${LIGHT_LO.toFixed(3)} - ${LIGHT_HI.toFixed(3)}) * clamp(t, 0.0, 1.0);
  float d = min(1.0, abs(L - 0.62) / 0.36);
  float f = 1.0 - 0.6 * d * d;
  return oklabToSrgb(vec3(L, CLASS_LAB[k].yz * f));
}
float agbAt(ivec2 p, ivec2 size, float here) {
  uint c = texelFetch(agbTex, clamp(p, ivec2(0), size - 1), 0).r;
  return c == 0u ? here : float(c - 1u) / ${CODE.toFixed(1)};
}`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(agbTex, 0);
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      uvec2 cc = texelFetch(agbTex, px, 0).rg;
      uint c = cc.r;
      if (c == 0u) discard;
      float v = float(c - 1u) / ${CODE.toFixed(1)};
      float t = (v - biomass.lo) / max(biomass.hi - biomass.lo, 1e-6);
      int r = int(biomass.ramp + 0.5);
      int k = min(int(cc.g & 15u), ${LAND_COVER.length - 1});
      if (r == 1) color = vec4(classShade(k, t), 1.0);
      else if (r == 2) {
        float h = float((cc.g & 0xFFF0u) - min(cc.g & 0xFFF0u, 1u)) / ${CODE.toFixed(1)};
        color = vec4(classShade(k, h / ${Math.log1p(HEIGHT_HI).toFixed(5)}), 1.0);
      } else if (r == 3) {
        float dx = agbAt(px + ivec2(1, 0), size, v) - agbAt(px - ivec2(1, 0), size, v);
        float dy = agbAt(px + ivec2(0, 1), size, v) - agbAt(px - ivec2(0, 1), size, v);
        // texture rows run north to south; the light from the northwest, 45 degrees up
        vec3 n = normalize(vec3(-${RELIEF_Z.toFixed(2)} * dx * 0.5, ${RELIEF_Z.toFixed(2)} * dy * 0.5, 1.0));
        vec3 sun = normalize(vec3(-1.0, 1.0, 1.4142));
        float shade = dot(n, sun) / sun.z;
        vec3 base = CLASS_LAB[k];
        color = vec4(oklabToSrgb(vec3(clamp(base.x * shade, 0.05, 0.99), base.yz)), 1.0);
      } else color = vec4(viridis(t), 1.0);
    `,
  },
  uniformTypes: { lo: "f32", hi: "f32", ramp: "f32" },
  getUniforms: (p: Record<string, unknown>) => p,
} as unknown as ShaderModule<{ agbTex: Texture; lo: number; hi: number; ramp: number }>;

/**
 * A window's change per pixel on the 50-stop orange (loss) to white to blue (gain) ramp: loss to -LOSS_END %, gain
 * to +GAIN_END %; its opacity from its confidence. No change is not drawn.
 */
const ChangeColor = {
  name: "change",
  fs: `uniform changeUniforms {
  float lossEnd;
  float gainEnd;
  float lossFrom;
  float gainFrom;
  float alphaMin;
} change;`,
  inject: {
    "fs:#decl": `precision highp isampler2D;
uniform highp isampler2D chTex;
${glslRamp("changeRamp", CHANGE_50)}`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      ivec2 size = textureSize(chTex, 0);
      ivec2 px = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
      ivec2 cc = texelFetch(chTex, px, 0).rg;
      int c = cc.r;
      if (c == 0 || c == ${CH_NODATA}) discard;
      if (cc.g == ${CUT}) {
        // a cut (change.ts): the loss ramp's end, fully drawn
        color = vec4(changeRamp(0.0), 1.0);
      } else {
        float pct = float(c) / ${PCT_SCALE.toFixed(1)};
        float conf = float(cc.g) / ${CONF_SCALE.toFixed(1)};
        float t = pct < 0.0 ? 0.5 - 0.5 * min(-pct / change.lossEnd, 1.0) : 0.5 + 0.5 * min(pct / change.gainEnd, 1.0);
        float from = pct < 0.0 ? change.lossFrom : change.gainFrom;
        float a = mix(change.alphaMin, 1.0, smoothstep(from, 0.995, conf));
        color = vec4(changeRamp(t), a);
      }
    `,
  },
  uniformTypes: { lossEnd: "f32", gainEnd: "f32", lossFrom: "f32", gainFrom: "f32", alphaMin: "f32" },
  getUniforms: (p: Record<string, unknown>) => p,
} as unknown as ShaderModule<{ chTex: Texture; lossEnd: number; gainEnd: number; lossFrom: number; gainFrom: number; alphaMin: number }>;

/**
 * fill: the network that rebuilds the bands after the first 32 (scripts/export_fill.py --prefix), so each chunk draws
 * first from the front of its bytes and sharpens to the exact picture once the rest is in (zread.ts); onSharpen is
 * called when chunks on the map have sharpened (the page repaints).
 */
export async function createBiomass(head: BiomassHead, conf: ConfTables, gain: GainHead | null, beforeId: () => string | undefined, fill: FillNet | null = null, onSharpen: () => void = () => {}) {
  const { arr, attrs } = await openMosaic();
  const [, , H, W] = arr.shape as number[];
  // the tile grid and CRS, read once from the group's GeoZarr attributes and shared by every layer
  const opened = await ZarrSource.open(arr, { geoZarrMetadata: attrs });
  if (!opened.is("int8")) throw new Error("Expected int8 embeddings");
  const source: ZarrSource<"int8"> = opened;
  const pool = new Pool(head, conf, gain, fill);

  // the planes by "year:row-col", the last used last; 384 KB each, at most CAP_BYTES (a zoom 11 view in a large
  // window is about 1,000 chunks, 375 MB a year): the reading ahead takes only the years that fit (main.ts)
  const planes = new Map<string, Planes>();
  const CAP_BYTES = 768 * 1024 ** 2;
  const capChunks = Math.floor(CAP_BYTES / (384 * 1024));
  let planeBytes = 0;
  const keyOf = (lk: string, r: number, c: number) => `${lk}:${r}-${c}`;
  const keep = (k: string, p: Planes) => {
    const old = planes.get(k);
    if (old) {
      planes.delete(k);
      planeBytes -= old.planes.byteLength;
    }
    planes.set(k, p);
    planeBytes += p.planes.byteLength;
    for (const [ok, v] of planes) {
      if (planeBytes <= CAP_BYTES) break;
      planes.delete(ok);
      planeBytes -= v.planes.byteLength;
    }
  };
  // reads in flight, shared by everyone who wants the same chunk-year; stopped once nobody does (fire-knn's pattern)
  const inflight = new Map<string, { p: Promise<Planes>; ac: AbortController; wants: number }>();
  // the chunks on the GPU now, by key: whether the view's chunks are in
  const loaded = new Set<string>();
  // each layer's textures: deck.gl does not unload the tiles of a layer taken off the map, so its finalize frees them
  const texOf = new Map<string, Set<Texture>>();
  // the textures of each chunk (by its planes' key), to write the exact planes into when they come
  const texAt = new Map<string, Set<Texture>>();
  // chunks drawn rough whose exact planes are still to come: stopped once no tile shows them
  const sharpening = new Map<string, AbortController>();
  const untex = (k: string, t: Texture) => {
    const s = texAt.get(k);
    if (!s) return;
    s.delete(t);
    if (!s.size) {
      texAt.delete(k);
      sharpening.get(k)?.abort();
    }
  };
  /** A chunk's planes as its texture holds them: a year's biomass codes, or a window's percent and confidence side by side. */
  const texData = (p: Planes, win: boolean) => {
    const n = p.w * p.h;
    if (!win) {
      // biomass code; the height code with the class in its low four bits (codes reach 64,001, classes 9)
      const data = new Uint16Array(2 * n);
      for (let i = 0; i < n; i++) {
        data[2 * i] = p.planes[i]!;
        data[2 * i + 1] = (p.planes[n + i]! & 0xfff0) | (p.planes[2 * n + i]! & 15);
      }
      return data;
    }
    const data = new Int16Array(2 * n);
    for (let i = 0; i < n; i++) {
      data[2 * i] = p.planes[i]!;
      data[2 * i + 1] = p.planes[2 * n + i]!;
    }
    return data;
  };
  let sharpTimer = 0;
  const sharpened = () => {
    if (sharpTimer) return;
    sharpTimer = window.setTimeout(() => {
      sharpTimer = 0;
      onSharpen();
    }, 100);
  };
  /**
   * deck took a layer off the map (its state is gone, nothing draws its tiles again): free its textures and forget its
   * tiles for the loading count. Not on a timer from the step: put back before deck's next frame, the same layer and
   * tiles carry on drawing.
   */
  function freeTextures(lk: string) {
    const set = texOf.get(lk);
    texOf.delete(lk);
    for (const [k, s] of [...texAt]) if (k.startsWith(`${lk}:`)) for (const t of [...s]) if (set?.has(t)) untex(k, t);
    set?.forEach((t) => t.destroy());
    for (const k of [...loaded]) if (k.startsWith(`${lk}:`)) loaded.delete(k);
  }

  /** A chunk's planes for a layer key: the year read and scored, or the window's change (both years read). */
  async function planesFor(lk: string, py0: number, px0: number, h: number, w: number, signal?: AbortSignal, ahead = false): Promise<Planes> {
    const k = keyOf(lk, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
    const have = planes.get(k);
    if (have) {
      keep(k, have);
      return have;
    }
    let f = inflight.get(k);
    if (!f || f.ac.signal.aborted) {
      const ac = new AbortController();
      const win = windowOf(lk);
      // the rough planes answer the tile; the exact ones replace them in the cache and in its texture
      let answer!: (p: Planes) => void;
      let fail!: (e: unknown) => void;
      const p = new Promise<Planes>((ok, no) => {
        answer = ok;
        fail = no;
      });
      let answered = false;
      const rough = (pl: Uint16Array | Int16Array) => {
        if (answered) return;
        answered = true;
        const out = { planes: pl, w, h, rough: true };
        keep(k, out);
        sharpening.set(k, ac);
        answer(out);
      };
      const ask = win
        ? pool.change(win[0] - YEAR_ORIGIN, win[1] - YEAR_ORIGIN, py0, px0, h, w, lk.endsWith("L"), ac.signal, rough, ahead)
        : pool.read(Number(lk) - YEAR_ORIGIN, py0, px0, h, w, ac.signal, rough, ahead);
      ask.then(
        (pl) => {
          const out = { planes: pl, w, h };
          keep(k, out);
          if (sharpening.get(k) === ac) sharpening.delete(k);
          if (!answered) {
            answered = true;
            answer(out);
            return;
          }
          const data = texData(out, !!win);
          for (const t of texAt.get(k) ?? []) t.writeData(data);
          sharpened();
        },
        (e) => {
          if (sharpening.get(k) === ac) sharpening.delete(k);
          // a rough chunk stopped before it sharpened: read afresh when next asked
          const had = planes.get(k);
          if (answered && had?.rough) {
            planes.delete(k);
            planeBytes -= had.planes.byteLength;
          } else if (!answered) fail(e);
        },
      );
      const g = { p, ac, wants: 0 };
      inflight.set(k, g);
      const done = () => inflight.get(k) === g && inflight.delete(k);
      p.then(done, done);
      f = g;
    }
    const g = f;
    g.wants++;
    const drop = () => {
      if (--g.wants === 0) g.ac.abort();
    };
    signal?.addEventListener("abort", drop, { once: true });
    try {
      return await g.p;
    } finally {
      signal?.removeEventListener("abort", drop);
      if (!signal?.aborted) g.wants--;
    }
  }

  /**
   * Read a layer's chunks under a box ahead of a step to it, a few at a time, the nearest the middle first; the
   * planes land in the cache, so the step draws without a read. Stopped by `signal` (a move, a step).
   */
  async function prefetch(lk: string, bbox: [number, number, number, number], signal: AbortSignal, onChunk?: () => void) {
    const { c0, c1, r0, r1 } = chunkRange(bbox);
    const rm = (r0 + r1) / 2, cm = (c0 + c1) / 2;
    const todo: [number, number][] = [];
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) if (!planes.has(keyOf(lk, r, c))) todo.push([r, c]);
    todo.sort((a, b) => Math.hypot(a[0] - rm, a[1] - cm) - Math.hypot(b[0] - rm, b[1] - cm));
    let i = 0;
    const lane = async () => {
      while (i < todo.length && !signal.aborted) {
        const [r, c] = todo[i++]!;
        const py0 = r * CHUNK, px0 = c * CHUNK;
        try {
          await planesFor(lk, py0, px0, Math.min(CHUNK, H - py0), Math.min(CHUNK, W - px0), signal, true);
          onChunk?.();
        } catch {
          // aborted, or a chunk with no data: the step reads it again if it needs it
        }
      }
    };
    await Promise.all([lane(), lane(), lane(), lane()]);
  }

  /** How many of a layer's chunks under a box are in hand (read and scored), for the prefetch's count. */
  function inHand(bbox: [number, number, number, number], lk: string) {
    const { c0, c1, r0, r1 } = chunkRange(bbox);
    let n = 0;
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) if (planes.has(keyOf(lk, r, c))) n++;
    return { inHand: n, wanted: Math.max(0, (r1 - r0 + 1) * (c1 - c0 + 1)) };
  }

  const chunkRange = (bbox: [number, number, number, number]) => {
    const [w, s, e, n] = bbox;
    return {
      c0: Math.max(0, Math.floor((w - X0) / RES / CHUNK)),
      c1: Math.min(Math.ceil(W / CHUNK) - 1, Math.floor((e - X0) / RES / CHUNK)),
      r0: Math.max(0, Math.floor((Y0 - n) / RES / CHUNK)),
      r1: Math.min(Math.ceil(H / CHUNK) - 1, Math.floor((Y0 - s) / RES / CHUNK)),
    };
  };

  /**
   * The change in view, counted live for the card: hectares per step of the ramp (HIST_N columns, the ramp's own
   * positions), each split into pixels the map draws more than half opaque ("sure") and the rest; each side's
   * hectares and tonnes; and hectares and tonnes per class pair (the head's class in year a, then in year b). Only
   * pixels the map draws count. A pixel's area is the ground it covers (the grid is in degrees, so it narrows with
   * latitude). Big views are counted on every k-th pixel each way and scaled by k x k, at most ~1.5 M pixels.
   */
  function changeStats(bbox: [number, number, number, number], lk: string): ChangeStats {
    const { c0, c1, r0, r1 } = chunkRange(bbox);
    const [bw, bs, be, bn] = bbox;
    // the view's pixel box on the grid
    const gx0 = Math.floor((bw - X0) / RES), gx1 = Math.ceil((be - X0) / RES);
    const gy0 = Math.floor((Y0 - bn) / RES), gy1 = Math.ceil((Y0 - bs) / RES);
    const k = Math.max(1, Math.ceil(Math.sqrt(((gx1 - gx0) * (gy1 - gy0)) / 1.5e6)));
    const sure = new Float64Array(HIST_N), faint = new Float64Array(HIST_N);
    const side = () => ({ ha: 0, t: 0, pairs: new Map<number, { ha: number; t: number }>() });
    const loss = side(), gain = side(), cut = side();
    let dataHa = 0, inHand = 0, wanted = 0;
    const sureLoss = (LOSS_FROM + 0.995) / 2, sureGain = (GAIN_FROM + 0.995) / 2;
    const M_PER_DEG_LAT = 110574, M_PER_DEG_LON = 111320, D2R = Math.PI / 180;
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        wanted++;
        const p = planes.get(keyOf(lk, r, c));
        if (!p) continue;
        inHand++;
        const n = p.w * p.h, P = p.planes;
        // the chunk's part of the view, on the k-pixel lattice of the whole grid (no seams between chunks)
        const x0 = Math.max(gx0, c * CHUNK), x1 = Math.min(gx1, c * CHUNK + p.w);
        const y0 = Math.max(gy0, r * CHUNK), y1 = Math.min(gy1, r * CHUNK + p.h);
        for (let gy = Math.ceil(y0 / k) * k; gy < y1; gy += k) {
          const lat = Y0 - (gy + 0.5) * RES;
          const ha = (k * k * RES * M_PER_DEG_LON * Math.cos(lat * D2R) * RES * M_PER_DEG_LAT) / 1e4;
          const row = (gy - r * CHUNK) * p.w - c * CHUNK;
          for (let gx = Math.ceil(x0 / k) * k; gx < x1; gx += k) {
            const i = row + gx;
            const v = P[i]!;
            if (v === CH_NODATA) continue;
            dataHa += ha;
            if (v === 0) continue;
            if (P[2 * n + i] === CUT) {
              cut.ha += ha;
              const pk = P[3 * n + i] ?? -1;
              const q = cut.pairs.get(pk);
              if (q) q.ha += ha;
              else cut.pairs.set(pk, { ha, t: 0 });
              continue;
            }
            const pct = v / PCT_SCALE, mg = P[n + i]! / MG_SCALE, conf = P[2 * n + i]! / CONF_SCALE;
            const t = pct < 0 ? 0.5 - 0.5 * Math.min(-pct / LOSS_END, 1) : 0.5 + 0.5 * Math.min(pct / GAIN_END, 1);
            const b = Math.min(HIST_N - 1, Math.floor(t * HIST_N));
            const s = pct < 0 ? loss : gain;
            if (conf >= (pct < 0 ? sureLoss : sureGain)) sure[b] += ha;
            else faint[b] += ha;
            s.ha += ha;
            s.t += Math.abs(mg) * ha;
            const pk = P[3 * n + i] ?? -1;
            const q = s.pairs.get(pk);
            if (q) {
              q.ha += ha;
              q.t += Math.abs(mg) * ha;
            } else s.pairs.set(pk, { ha, t: Math.abs(mg) * ha });
          }
        }
      }
    const named = (sd: ReturnType<typeof side>) => ({
      ha: sd.ha,
      t: sd.t,
      pairs: [...sd.pairs]
        .map(([pk, v]) => ({
          from: pk < 0 ? "" : (head.classes[pk % 256] ?? ""),
          to: pk < 0 ? "" : (head.classes[pk >> 8] ?? ""),
          ...v,
        }))
        .sort((a, b) => b.ha - a.ha),
    });
    return { sure, faint, loss: named(loss), gain: named(gain), cut: named(cut), dataHa, inHand, wanted, every: k };
  }

  /** A layer: a year's biomass (key "2025") or a window's change (key "2018-2025"). */
  function layer(opts: { key: string; visible: boolean; ramp?: number; onTile?: () => void; onViewportLoad?: () => void }) {
    const lk = opts.key;
    const win = windowOf(lk);
    return new BioLayer({
      id: win ? `change-${lk}` : `biomass-${lk}`,
      source,
      selection: { time: (win ? win[1] : Number(lk)) - YEAR_ORIGIN, band: null },
      getTileData: async (_a, o) => {
        const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
        const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
        const { width, height } = o;
        const p = await planesFor(lk, py0, px0, height, width, o.signal);
        const key = keyOf(lk, Math.floor(py0 / CHUNK), Math.floor(px0 / CHUNK));
        loaded.add(key);
        // the change's percent and confidence side by side, one texel each
        const data = texData(p, !!win);
        const tex = o.device.createTexture({
          format: win ? "rg16sint" : "rg16uint",
          width,
          height,
          mipLevels: 1,
          data,
          sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
        });
        let set = texOf.get(lk);
        if (!set) texOf.set(lk, (set = new Set()));
        set.add(tex);
        let at = texAt.get(key);
        if (!at) texAt.set(key, (at = new Set()));
        at.add(tex);
        // exact planes may have come between the rough answer and this texture
        const now = planes.get(key);
        if (p.rough && now && !now.rough) tex.writeData(texData(now, !!win));
        return { key, tex, width, height, byteLength: data.byteLength };
      },
      renderTile: (data: TileData) => ({
        renderPipeline: [
          win
            ? {
                module: ChangeColor,
                props: { chTex: data.tex, lossEnd: LOSS_END, gainEnd: GAIN_END, lossFrom: LOSS_FROM, gainFrom: GAIN_FROM, alphaMin: ALPHA_MIN },
              }
            : {
                module: BiomassColor,
                props: { agbTex: data.tex, lo: AGB_LO, hi: AGB_HI, ramp: opts.ramp ?? 0 },
              },
        ] as RasterModule[],
      }),
      // a new ramp redraws the tiles in hand, no read
      updateTriggers: { renderTile: opts.ramp ?? 0 },
      visible: opts.visible,
      minZoom: MIN_ZOOM,
      // the view's chunks all asked at once: the workers read them a few at a time, every front before any rest
      // (zread.ts), so the whole view draws rough before it sharpens
      maxRequests: 64,
      // a tile on the GPU is 256 KB (two 16-bit values a pixel); a zoom 11 view in a large window is about 1,000
      maxCacheSize: 1200,
      onTileLoad: () => opts.onTile && opts.onTile(),
      onTileUnload: (tile: { content?: TileData | null }) => {
        const t = tile.content;
        if (!t) return;
        loaded.delete(t.key);
        texOf.get(lk)?.delete(t.tex);
        untex(t.key, t.tex);
        t.tex?.destroy();
      },
      onViewportLoad: () => opts.onViewportLoad && opts.onViewportLoad(),
      // read by BioLayer.finalizeState (not a ZarrLayer prop, so spread past the type check)
      ...({ onFinalize: () => freeTextures(lk) } as object),
      // @ts-expect-error beforeId is read by the MapLibre overlay (the layer draws under the labels), not typed on the layer
      beforeId: beforeId(),
    });
  }

  /** A layer's plane values at a point (three for a year, four for a change), from the planes in hand; null where none. */
  function planeAt(lon: number, lat: number, lk: string): [number, number, number, number] | null {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const r = Math.floor(py / CHUNK), c = Math.floor(px / CHUNK);
    const p = planes.get(keyOf(lk, r, c));
    if (!p) return null;
    const ix = px - c * CHUNK, iy = py - r * CHUNK;
    if (ix < 0 || iy < 0 || ix >= p.w || iy >= p.h) return null;
    const i = iy * p.w + ix, n = p.w * p.h;
    return [p.planes[i]!, p.planes[n + i]!, p.planes[2 * n + i]!, p.planes[3 * n + i] ?? -1];
  }

  /** Biomass (Mg/ha), canopy height (m) and the head's class at a point for a year, from the planes in hand; null where none. */
  function valueAtPoint(lon: number, lat: number, year: number): { agb: number; height: number; cls: string } | null {
    const v = planeAt(lon, lat, yearKey(year));
    if (!v || v[0] === 0) return null;
    return { agb: codeValue(v[0]), height: codeValue(v[1]), cls: head.classes[v[2]] ?? "" };
  }

  /**
   * A window's change at a point: percent of the pixel's biomass, Mg/ha, its confidence (0 to 1), and the head's
   * class in each of the two years.
   */
  function changeAtPoint(
    lon: number, lat: number, lk: string,
  ): { pct: number; mg: number; conf: number; clsA: string; clsB: string; cut: boolean } | null {
    const v = planeAt(lon, lat, lk);
    if (!v || v[0] === CH_NODATA) return null;
    const k = v[3] < 0 ? -1 : v[3];
    const clsA = k < 0 ? "" : (head.classes[k % 256] ?? ""), clsB = k < 0 ? "" : (head.classes[k >> 8] ?? "");
    const cut = v[2] === CUT;
    return { pct: v[0] / PCT_SCALE, mg: v[1] / MG_SCALE, conf: cut ? 0 : v[2] / CONF_SCALE, clsA, clsB, cut };
  }

  /**
   * When a window's change happened at a point: the head's biomass there every year from a to b (the year planes of
   * its chunk, read where not in hand, and kept for the each-year view), and the one-year step that moved most in
   * the change's direction (sign -1 loss, +1 gain). Null where a year has no data.
   */
  async function whenAtPoint(lon: number, lat: number, a: number, b: number, sign: number, signal?: AbortSignal): Promise<[number, number] | null> {
    if (b - a <= 1) return [a, b];
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const r = Math.floor(py / CHUNK), c = Math.floor(px / CHUNK);
    const py0 = r * CHUNK, px0 = c * CHUNK, h = Math.min(CHUNK, H - py0), w = Math.min(CHUNK, W - px0);
    const ix = px - px0, iy = py - py0;
    if (ix < 0 || iy < 0 || ix >= w || iy >= h) return null;
    const years = Array.from({ length: b - a + 1 }, (_, k) => a + k);
    const got = await Promise.all(years.map((y) => planesFor(yearKey(y), py0, px0, h, w, signal)));
    const v = got.map((p) => codeValue(p.planes[iy * p.w + ix]!));
    if (v.some((x) => !(x >= 0))) return null;
    let best = 0;
    for (let k = 1; k < v.length - 1; k++) if (sign * (v[k + 1]! - v[k]!) > sign * (v[best + 1]! - v[best]!)) best = k;
    return [years[best]!, years[best + 1]!];
  }

  /** How many of the chunks under a box are on the GPU for a layer, and how many of those are still rough. */
  function loadedIn(bbox: [number, number, number, number], lk: string) {
    const { c0, c1, r0, r1 } = chunkRange(bbox);
    let n = 0, rough = 0;
    for (let r = r0; r <= r1; r++)
      for (let c = c0; c <= c1; c++) {
        const k = keyOf(lk, r, c);
        if (!loaded.has(k)) continue;
        n++;
        if (planes.get(k)?.rough) rough++;
      }
    return { loaded: n, rough, wanted: Math.max(0, (r1 - r0 + 1) * (c1 - c0 + 1)) };
  }

  return {
    layer, valueAtPoint, changeAtPoint, whenAtPoint, changeStats, loadedIn, inHand, prefetch, pool, capChunks,
  };
}
