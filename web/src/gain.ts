import { deq, NODATA } from "./head";

/**
 * The lidar-taught gain (scripts/gain_mlp.py, FINDINGS 38): a 32-unit network that gives the chance that lidar's
 * canopy rose 2 m or more between two years, from AlphaEarth alone, taught on nine two-flight lidar squares. Input
 * "difference": the two years' dequantized difference and the window's length; "embeddings": year a, year b, their
 * difference and the length. Its confidence is the share of steady ground (127 squares, Esri class held, CTrees
 * steady) scored below it at the same window length, as the map's other change rules.
 */
export type GainHead = {
  name: string;
  input: "difference" | "embeddings";
  n_in: number;
  hidden: number;
  mean: number[];
  scale: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number;
  /** The longest window the network was taught on; longer windows are fed as this. */
  L_max?: number;
  steady: { p: number[]; L: Record<string, { n: number; q: number[] }> };
  probe_a: number[];
  probe_b: number[];
  probe_L: number;
  probe_p: number[];
};

/** Gain is drawn where the confidence is at least this: 1.5 % of steady ground lit, as the map's two-part floor. */
export const GAIN_LIDAR_FROM = 0.985;

export type GainNet = { h: GainHead; inv: Float32Array; w1: Float32Array; x: Float32Array; hid: Float32Array };

export function gainNet(h: GainHead): GainNet {
  return {
    h,
    inv: Float32Array.from(h.scale, (s) => 1 / s),
    w1: Float32Array.from(h.w1),
    x: new Float32Array(h.n_in),
    hid: new Float32Array(h.hidden),
  };
}

/** The chance canopy rose, for pixel p of two band-major [64, n] chunks over a window of L years; NaN no data. */
export function gainP(g: GainNet, ea: Int8Array, eb: Int8Array, n: number, p: number, L: number): number {
  if (ea[p] === NODATA || eb[p] === NODATA) return NaN;
  const { h, x, hid, w1, inv } = g;
  let o = 0;
  if (h.input === "embeddings") {
    for (let b = 0; b < 64; b++) x[o++] = deq(ea[b * n + p]!);
    for (let b = 0; b < 64; b++) x[o++] = deq(eb[b * n + p]!);
  }
  for (let b = 0; b < 64; b++) x[o++] = deq(eb[b * n + p]!) - deq(ea[b * n + p]!);
  x[o++] = Math.min(L, h.L_max ?? Infinity);
  for (let i = 0; i < h.n_in; i++) x[i] = (x[i]! - h.mean[i]!) * inv[i]!;
  let z = h.b2;
  for (let j = 0; j < h.hidden; j++) {
    let s = h.b1[j]!;
    const r = j * h.n_in;
    for (let i = 0; i < h.n_in; i++) s += w1[r + i]! * x[i]!;
    if (s > 0) z += h.w2[j]! * s;
  }
  return 1 / (1 + Math.exp(-z));
}

/** The largest gap between this code's outputs and the export's own on its probe (should be ~1e-6). */
export function gainProbeGap(g: GainNet): number {
  const n = g.h.probe_p.length;
  const ea = new Int8Array(64 * n), eb = new Int8Array(64 * n);
  for (let p = 0; p < n; p++)
    for (let b = 0; b < 64; b++) {
      ea[b * n + p] = g.h.probe_a[p * 64 + b]!;
      eb[b * n + p] = g.h.probe_b[p * 64 + b]!;
    }
  let gap = 0;
  for (let p = 0; p < n; p++) gap = Math.max(gap, Math.abs(gainP(g, ea, eb, n, p, g.h.probe_L) - g.h.probe_p[p]!));
  return gap;
}
