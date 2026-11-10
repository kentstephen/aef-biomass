import * as zarr from "zarrita";

import { type AefArray, NUM_BANDS, openMosaic, YEAR_ORIGIN } from "./aef";
import { changeOf, type ConfTables } from "./change";
import { fillCpu, type FillNet, fillProbeGap, packFill, probeInput } from "./fill";
import { type Filler, makeFiller } from "./fill-gpu";
import { type GainHead, type GainNet, gainNet, gainProbeGap } from "./gain";
import { argClass, type BiomassHead, packHead, probeGap, scoreCpu, toCode } from "./head";
import { makeScorer, type Scorer } from "./head-gpu";
import { readChunk, tiffStats } from "./tiff";

/**
 * One chunk-year per ask: AlphaEarth's 64 bands read from the mosaic, then the biomass head on them (WebGPU,
 * else the CPU). Only the three u16 planes go back to the page (biomass, height, class: 384 KB per 256 x 256
 * chunk), never the embedding. A change ask reads the chunk in two years (the second from the disk cache, mostly)
 * and sends back the change planes of change.ts (Int16, 384 KB).
 */
/** bands: read that many from AlphaEarth's GeoTIFFs (tiff.ts), the rest rebuilt by a fill network (fill.ts); else the mosaic. */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; timeIdx: number; timeIdxB?: number; lidarGain?: boolean; bands?: number };
type Msg = { type: "init"; head: BiomassHead; conf: ConfTables; gain: GainHead | null; fills?: FillNet[] } | Read | { type: "abort"; id: number };

let head: BiomassHead | null = null;
let conf: ConfTables | null = null;
let gain: GainNet | null = null;
let weights: Float32Array | null = null;
let ready: Promise<Scorer> | null = null;
let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();
let fillNets: FillNet[] = [];
const fillers = new Map<number, Promise<{ net: FillNet; run: Filler }>>();
const ALL = Array.from({ length: NUM_BANDS }, (_, b) => b);

/** The fill for k bands: on the GPU if it gives the export's own output on the probe (to a step of 1 in int8), else the CPU. */
function filler(k: number) {
  let f = fillers.get(k);
  if (f) return f;
  f = (async () => {
    const net = fillNets.find((x) => x.k === k);
    if (!net) throw new Error(`no fill network for ${k} bands`);
    const w = packFill(net);
    const cpu: Filler = async (e, n) => fillCpu(net, w, e, n);
    const pr = probeInput(net);
    const cpuGap = fillProbeGap(net, fillCpu(net, w, pr.e, pr.n), pr.n);
    let run = cpu, mode = "cpu", gpuGap: { worst: number; off: number } | null = null;
    try {
      const g = await makeFiller(net, w, 256 * 256);
      if (g) {
        gpuGap = fillProbeGap(net, await g(pr.e, pr.n), pr.n);
        if (gpuGap.worst <= 1) {
          run = g;
          mode = "gpu";
        }
      }
    } catch (e) {
      postMessage({ type: "note", note: `fill ${k} on the GPU failed: ${(e as Error).message}` });
    }
    postMessage({ type: "fill", k, mode, cpuGap, gpuGap });
    return { net, run };
  })();
  fillers.set(k, f);
  return f;
}

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
      let clsOff = 0;
      for (let p = 0; p < n; p++) {
        for (let c = 0; c < 2; c++) gpuGap = Math.max(gpuGap, Math.abs(out[c * n + p]! - toCode(h.probe_out[p * h.n_out + c]!)));
        if (out[2 * n + p] !== argClass(h.probe_out.slice(p * h.n_out, (p + 1) * h.n_out), h.n_out)) clsOff++;
      }
      // a class can flip on a near tie between float paths; more than one of the probe's 64 is a wrong shader
      if (gpuGap <= 4 && clsOff <= 1) {
        use = g;
        mode = "gpu";
      }
    }
  } catch {
    // no WebGPU in this worker: the CPU path
  }
  postMessage({ type: "mode", mode, probeGap: gap, gpuCodeGap: gpuGap, gainGap: gain ? gainProbeGap(gain) : undefined });
  return use;
}

async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  try {
    if (!mosaic) mosaic = openMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const [arr, score] = await Promise.all([mosaic, ready!]);
    const k = m.bands;
    const get = async (t: number): Promise<Int8Array> => {
      if (k) {
        if (k >= NUM_BANDS) return readChunk(YEAR_ORIGIN + t, m.py0, m.px0, m.h, m.w, ALL, ac.signal);
        const f = await filler(k);
        const e = await readChunk(YEAR_ORIGIN + t, m.py0, m.px0, m.h, m.w, f.net.bands, ac.signal);
        if (ac.signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
        return f.run(e, m.w * m.h);
      }
      const c = await zarr.get(arr, [t, null, zarr.slice(m.py0, m.py0 + m.h), zarr.slice(m.px0, m.px0 + m.w)], { signal: ac.signal });
      if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w)
        throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
      if (ac.signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
      return c.data as Int8Array;
    };
    const n = m.w * m.h;
    let planes: Uint16Array | Int16Array;
    if (m.timeIdxB == null) planes = await score(await get(m.timeIdx), n);
    else {
      const [ea, eb] = await Promise.all([get(m.timeIdx), get(m.timeIdxB)]);
      const ca = await score(ea, n);
      const cb = await score(eb, n);
      planes = changeOf(ea, eb, ca, cb, n, m.timeIdxB - m.timeIdx, conf!, m.lidarGain ? gain : null);
    }
    postMessage({ id: m.id, planes, tiff: k ? tiffStats() : undefined }, { transfer: [planes.buffer as ArrayBuffer] });
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
    conf = m.conf;
    fillNets = m.fills ?? [];
    gain = m.gain ? gainNet(m.gain) : null;
    weights = packHead(head);
    ready = scorer(head, weights);
  } else if (m.type === "abort") running.get(m.id)?.abort();
  else void read(m);
};
