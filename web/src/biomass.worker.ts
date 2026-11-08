import * as zarr from "zarrita";

import { type AefArray, NUM_BANDS, openMosaic } from "./aef";
import { type BiomassHead, packHead, probeGap, scoreCpu, toCode } from "./head";
import { makeScorer, type Scorer } from "./head-gpu";

/**
 * One chunk-year per ask: AlphaEarth's 64 bands read from the mosaic, then the biomass head on them (WebGPU,
 * else the CPU). Only the two u16 planes go back to the page (256 KB per 256 x 256 chunk), never the embedding.
 */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; timeIdx: number };
type Msg = { type: "init"; head: BiomassHead } | Read | { type: "abort"; id: number };

let head: BiomassHead | null = null;
let weights: Float32Array | null = null;
let ready: Promise<Scorer> | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

/** The GPU scorer if it gives the export's own outputs on the probe, else the CPU. */
async function scorer(h: BiomassHead, w: Float32Array): Promise<Scorer> {
  const cpu: Scorer = async (e, n) => scoreCpu(h, w, e, n);
  const gap = probeGap(h, w);
  let mode = "cpu", gpuGap = NaN;
  let use = cpu;
  try {
    const g = await makeScorer(h, w, 256 * 256);
    if (g) {
      const n = h.probe_q.length / h.n_in;
      const e = new Int8Array(h.probe_q.length);
      for (let p = 0; p < n; p++) for (let b = 0; b < h.n_in; b++) e[b * n + p] = h.probe_q[p * h.n_in + b]!;
      const out = await g(e, n);
      gpuGap = 0;
      for (let p = 0; p < n; p++)
        for (let c = 0; c < 2; c++) gpuGap = Math.max(gpuGap, Math.abs(out[c * n + p]! - toCode(h.probe_out[p * h.n_out + c]!)));
      if (gpuGap <= 4) {
        use = g;
        mode = "gpu";
      }
    }
  } catch {
    // no WebGPU in this worker: the CPU path
  }
  postMessage({ type: "mode", mode, probeGap: gap, gpuCodeGap: gpuGap });
  return use;
}

async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  try {
    if (!mosaic) mosaic = openMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const [arr, score] = await Promise.all([mosaic, ready!]);
    const c = await zarr.get(arr, [m.timeIdx, null, zarr.slice(m.py0, m.py0 + m.h), zarr.slice(m.px0, m.px0 + m.w)], { signal: ac.signal });
    if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w)
      throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
    if (ac.signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
    const planes = await score(c.data as Int8Array, m.w * m.h);
    postMessage({ id: m.id, planes }, { transfer: [planes.buffer as ArrayBuffer] });
  } catch (e) {
    const err = e as { name?: string; message?: string };
    postMessage({ id: m.id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
  } finally {
    running.delete(m.id);
  }
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const m = ev.data;
  if (m.type === "init") {
    head = m.head;
    weights = packHead(head);
    ready = scorer(head, weights);
  } else if (m.type === "abort") running.get(m.id)?.abort();
  else void read(m);
};
