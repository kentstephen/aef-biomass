import { changeOf, type ConfTables } from "./change";
import { fillCpu, type FillNet, fillProbeGap, packFill, probeInput } from "./fill";
import { type Filler, makeFiller } from "./fill-gpu";
import { type GainHead, type GainNet, gainNet, gainProbeGap } from "./gain";
import { argClass, type BiomassHead, packHead, probeGap, scoreCpu, toCode } from "./head";
import { makeScorer, type Scorer } from "./head-gpu";
import { front, FRONT_BANDS, type Part } from "./zread";

/**
 * One chunk-year per ask: AlphaEarth's 64 bands read from the mosaic in two parts (zread.ts), the head on them
 * (WebGPU, else the CPU) after each: a rough picture from the first 32 bands, then the exact one. Only the three u16 planes go back to the page (biomass, height, class: 384 KB per 256 x 256
 * chunk), never the embedding. A change ask reads the chunk in two years (the second from the disk cache, mostly)
 * and sends back the change planes of change.ts (Int16, 384 KB).
 */
/** ahead: reading ahead, after the view's own reads (zread.ts) */
type Read = { type: "read"; id: number; px0: number; py0: number; w: number; h: number; timeIdx: number; timeIdxB?: number; lidarGain?: boolean; ahead?: boolean };
type Msg = { type: "init"; head: BiomassHead; conf: ConfTables; gain: GainHead | null; fill: FillNet | null } | Read | { type: "abort"; id: number };

let head: BiomassHead | null = null;
let conf: ConfTables | null = null;
let gain: GainNet | null = null;
let weights: Float32Array | null = null;
let ready: Promise<Scorer> | null = null;
let fillNet: FillNet | null = null;
let filling: Promise<Filler> | null = null;
const running = new Map<number, AbortController>();

/** The fill for the front's bands: on the GPU if it gives the export's own output on the probe (to 1 in int8), else the CPU. */
async function filler(f: FillNet): Promise<Filler> {
  const w = packFill(f);
  const cpu: Filler = async (e, n) => fillCpu(f, w, e, n);
  const pr = probeInput(f);
  let run = cpu, mode = "cpu";
  let gpuGap: { worst: number; off: number } | null = null;
  try {
    const g = await makeFiller(f, w, 256 * 256);
    if (g) {
      gpuGap = fillProbeGap(f, await g(pr.e, pr.n), pr.n);
      if (gpuGap.worst <= 1) {
        run = g;
        mode = "gpu";
      }
    }
  } catch {
    // no WebGPU here: the CPU
  }
  postMessage({ type: "fill", mode, cpuGap: fillProbeGap(f, fillCpu(f, w, pr.e, pr.n), pr.n), gpuGap });
  return run;
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

/**
 * A chunk-year (or a window's two): from the front of each chunk, the first FRONT_BANDS bands and the rest rebuilt,
 * scored and sent at once (rough: true); then the rest of each chunk read and the exact planes sent. A chunk already
 * on disk is sent exact only.
 */
async function read(m: Read) {
  const ac = new AbortController();
  running.set(m.id, ac);
  const stop = () => {
    if (ac.signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  try {
    if (m.w !== 256 || m.h !== 256) throw new Error(`a ${m.w} x ${m.h} chunk`);
    const score = await ready!;
    const n = m.w * m.h;
    const ts = m.timeIdxB == null ? [m.timeIdx] : [m.timeIdx, m.timeIdxB];
    const parts: Part[] = await Promise.all(ts.map((t) => front(t, m.py0, m.px0, ac.signal, !!m.ahead)));
    stop();
    const planesOf = async (es: Int8Array[]): Promise<Uint16Array | Int16Array> => {
      if (es.length === 1) return score(es[0]!, n);
      const ca = await score(es[0]!, n);
      const cb = await score(es[1]!, n);
      return changeOf(es[0]!, es[1]!, ca, cb, n, m.timeIdxB! - m.timeIdx, conf!, m.lidarGain ? gain : null);
    };
    if (parts.some((p) => !p.full) && fillNet) {
      if (!filling) filling = filler(fillNet);
      const fill = await filling;
      const rough = await Promise.all(parts.map((p) => p.full ?? fill(p.front!, n)));
      stop();
      const planes = await planesOf(rough);
      postMessage({ id: m.id, planes, rough: true }, { transfer: [planes.buffer as ArrayBuffer] });
    }
    const full = await Promise.all(parts.map((p) => p.full ?? p.rest(ac.signal)));
    stop();
    const planes = await planesOf(full);
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
    conf = m.conf;
    fillNet = m.fill && m.fill.k === FRONT_BANDS ? m.fill : null;
    gain = m.gain ? gainNet(m.gain) : null;
    weights = packHead(head);
    ready = scorer(head, weights);
  } else if (m.type === "abort") running.get(m.id)?.abort();
  else void read(m);
};
