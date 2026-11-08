/**
 * The biomass head (scripts/export_biomass_head.py): one 64 -> 64 -> 2 network on a single year's AlphaEarth
 * embedding, dequantized sign(q) * (q / 127.5)^2 and standardized, in fire-knn's fuel_volume.json layout.
 * Outputs: log1p(biomass, Mg/ha) and log1p(canopy height, m).
 *
 * Per pixel the scorer writes two u16 codes: 0 = no embedding, else 1 + round(log1p(value) * CODE), so a code
 * is the log value at 1 / 8000 steps (biomass to about 3,600 Mg/ha).
 */
export type BiomassHead = {
  name: string;
  n_in: number;
  hidden: number;
  n_out: number;
  out: string[];
  years: number[];
  mean: number[];
  scale: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
  probe_q: number[];
  probe_out: number[];
};

export const NODATA = -128;
export const CODE = 8000;
/** The head's own clamp on log biomass (biomass_head.py: expm1(min(o, 8))). */
export const LOG_MAX = 8;

/** A code back to log1p(value); NaN for 0. */
export const codeLog = (c: number) => (c > 0 ? (c - 1) / CODE : NaN);
/** A code back to the value (Mg/ha or m); NaN for 0. */
export const codeValue = (c: number) => (c > 0 ? Math.expm1((c - 1) / CODE) : NaN);
export const toCode = (logv: number) => 1 + Math.round(Math.min(LOG_MAX, Math.max(0, logv)) * CODE);

/** The weights in one array, as the GPU reads them: mean, 1 / scale, w1, b1, w2, b2. */
export function packHead(h: BiomassHead): Float32Array {
  const out = new Float32Array(2 * h.n_in + h.w1.length + h.b1.length + h.w2.length + h.b2.length);
  let o = 0;
  for (const a of [h.mean, h.scale.map((s) => 1 / s), h.w1, h.b1, h.w2, h.b2]) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}

const deq = (q: number) => {
  const f = q / 127.5;
  return f * Math.abs(f);
};

/** The raw outputs for one pixel of a band-major chunk; false where it has no embedding (band 0 at -128). */
function forward(h: BiomassHead, w: Float32Array, e: Int8Array, n: number, p: number, x: Float32Array, hid: Float32Array, z: Float32Array): boolean {
  const NI = h.n_in;
  if (e[p] === NODATA) return false;
  for (let b = 0; b < NI; b++) x[b] = (deq(e[b * n + p]!) - w[b]!) * w[NI + b]!;
  const o1 = 2 * NI, ob1 = o1 + h.hidden * NI, o2 = ob1 + h.hidden, ob2 = o2 + h.n_out * h.hidden;
  for (let j = 0; j < h.hidden; j++) {
    let s = w[ob1 + j]!;
    const r = o1 + j * NI;
    for (let i = 0; i < NI; i++) s += w[r + i]! * x[i]!;
    hid[j] = s > 0 ? s : 0;
  }
  for (let c = 0; c < h.n_out; c++) {
    let s = w[ob2 + c]!;
    const r = o2 + c * h.hidden;
    for (let j = 0; j < h.hidden; j++) s += w[r + j]! * hid[j]!;
    z[c] = s;
  }
  return true;
}

/** The head on the CPU, for when WebGPU is missing: e is one year, band-major int8 [64, n]; out [agb codes n, height codes n]. */
export function scoreCpu(h: BiomassHead, w: Float32Array, e: Int8Array, n: number): Uint16Array {
  const out = new Uint16Array(2 * n);
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  for (let p = 0; p < n; p++) {
    if (!forward(h, w, e, n, p, x, hid, z)) continue;
    out[p] = toCode(z[0]!);
    out[n + p] = toCode(z[1]!);
  }
  return out;
}

/** The largest gap between this code's outputs and the export's own on its probe embeddings. */
export function probeGap(h: BiomassHead, w: Float32Array): number {
  const n = h.probe_q.length / h.n_in;
  const e = new Int8Array(h.probe_q.length);
  for (let p = 0; p < n; p++) for (let b = 0; b < h.n_in; b++) e[b * n + p] = h.probe_q[p * h.n_in + b]!;
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  let worst = 0;
  for (let p = 0; p < n; p++) {
    forward(h, w, e, n, p, x, hid, z);
    for (let c = 0; c < h.n_out; c++) worst = Math.max(worst, Math.abs(z[c]! - h.probe_out[p * h.n_out + c]!));
  }
  return worst;
}
