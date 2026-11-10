import { deq, NODATA } from "./head";

/**
 * A fill-in network (scripts/export_fill.py): from k of AlphaEarth's 64 bands, the other 64 - k, so the map's own head
 * runs on the rebuilt vector. x = (deq(q[bands]) - mu) / sd; Linear, ReLU, Linear, ReLU, Linear; a missing band is
 * out * ys + ym. The 64 are then scaled to unit length (AlphaEarth's vectors are) and quantized as the stores hold
 * them, q = sign(f) sqrt(|f|) 127.5, so everything after the read (the head, change, gain) is unchanged.
 */
export type FillNet = {
  name: string;
  k: number;
  bands: number[];
  miss: number[];
  hidden: number;
  layers: number;
  mu: number[];
  sd: number[];
  ym: number[];
  ys: number[];
  /** each layer's weights, [out][in] row-major, and biases */
  w: number[][];
  b: number[][];
  dims: [number, number][];
  /** 64 held-out pixels: their 64 bands and the rebuilt 64 the export gives, int8 pixel-major */
  probe_q: number[];
  probe_out: number[];
  held_out: Record<string, number>;
};

/** The weights in one array, as the GPU reads them: mu, 1 / sd, ym, ys, then each layer's w and b. */
export function packFill(f: FillNet): Float32Array {
  const parts = [f.mu, f.sd.map((s) => 1 / s), f.ym, f.ys];
  for (let l = 0; l < f.w.length; l++) parts.push(f.w[l]!, f.b[l]!);
  const out = new Float32Array(parts.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of parts) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}

/** Where each part starts in packFill's array. */
export function fillOffsets(f: FillNet) {
  const k = f.k, m = f.miss.length;
  let o = 2 * k + 2 * m;
  const layers = f.dims.map(([out, inp]) => {
    const w = o;
    o += out * inp;
    const b = o;
    o += out;
    return { w, b, out, inp };
  });
  return { mu: 0, isd: k, ym: 2 * k, ys: 2 * k + m, layers };
}

const quant = (v: number) => {
  const q = Math.round(Math.sign(v) * Math.sqrt(Math.abs(v)) * 127.5);
  return q > 127 ? 127 : q < -127 ? -127 : q;
};

/** The fill on the CPU: e band-major int8 [k, n] (the network's bands, in its order); out band-major [64, n]. */
export function fillCpu(f: FillNet, wts: Float32Array, e: Int8Array, n: number): Int8Array {
  const out = new Int8Array(64 * n).fill(NODATA);
  const O = fillOffsets(f);
  const k = f.k;
  const acts = [new Float32Array(k), ...f.dims.map(([o]) => new Float32Array(o))];
  const full = new Float32Array(64);
  for (let p = 0; p < n; p++) {
    if (e[p] === NODATA) continue;
    const x = acts[0]!;
    for (let i = 0; i < k; i++) {
      const d = deq(e[i * n + p]!);
      full[f.bands[i]!] = d;
      x[i] = (d - wts[O.mu + i]!) * wts[O.isd + i]!;
    }
    for (let l = 0; l < O.layers.length; l++) {
      const L = O.layers[l]!, a = acts[l]!, z = acts[l + 1]!;
      const last = l === O.layers.length - 1;
      for (let j = 0; j < L.out; j++) {
        let s = wts[L.b + j]!;
        const r = L.w + j * L.inp;
        for (let i = 0; i < L.inp; i++) s += wts[r + i]! * a[i]!;
        z[j] = last || s > 0 ? s : 0;
      }
    }
    const o = acts[acts.length - 1]!;
    for (let j = 0; j < f.miss.length; j++) full[f.miss[j]!] = o[j]! * wts[O.ys + j]! + wts[O.ym + j]!;
    let nn = 0;
    for (let b = 0; b < 64; b++) nn += full[b]! * full[b]!;
    const inv = 1 / Math.max(Math.sqrt(nn), 1e-6);
    for (let b = 0; b < 64; b++) out[b * n + p] = quant(full[b]! * inv);
  }
  return out;
}

/** The probe's k bands, band-major [k, n], as the reader gives them. */
export function probeInput(f: FillNet): { e: Int8Array; n: number } {
  const n = f.probe_q.length / 64;
  const e = new Int8Array(f.k * n);
  for (let p = 0; p < n; p++) for (let i = 0; i < f.k; i++) e[i * n + p] = f.probe_q[p * 64 + f.bands[i]!]!;
  return { e, n };
}

/** How far a fill's output is from the export's on the probe: the largest step in int8, and how many values differ. */
export function fillProbeGap(f: FillNet, out: Int8Array, n: number): { worst: number; off: number } {
  let worst = 0, off = 0;
  for (let p = 0; p < n; p++)
    for (let b = 0; b < 64; b++) {
      const d = Math.abs(out[b * n + p]! - f.probe_out[p * 64 + b]!);
      if (d) off++;
      worst = Math.max(worst, d);
    }
  return { worst, off };
}
