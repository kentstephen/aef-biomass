import type { ConfTables } from "./change";
import type { GainHead } from "./gain";
import type { BiomassHead } from "./head";
import BiomassWorker from "./biomass.worker.ts?worker";

/** A few read-and-score workers (after fire-knn's aef/pool.ts); a chunk always goes to the same one. */
type W = { w: Worker; busy: number; waiting: Map<number, { ok: (p: Uint16Array | Int16Array) => void; no: (e: Error) => void }> };
export type Mode = { mode: string; probeGap: number; gpuCodeGap: number; gainGap?: number };

const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

export class Pool {
  private ws: W[];
  private nextId = 0;
  modes: Mode[] = [];
  onMode: (m: Mode) => void = () => {};

  constructor(head: BiomassHead, conf: ConfTables, gain: GainHead | null, size = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2))) {
    this.ws = Array.from({ length: size }, () => {
      const w = new BiomassWorker();
      const ent: W = { w, busy: 0, waiting: new Map() };
      w.onmessage = (ev: MessageEvent<{ type?: string; id: number; planes?: Uint16Array | Int16Array; error?: { name: string; message: string } } & Partial<Mode>>) => {
        const d = ev.data;
        if (d.type === "mode") {
          const m = { mode: d.mode!, probeGap: d.probeGap!, gpuCodeGap: d.gpuCodeGap!, gainGap: d.gainGap };
          this.modes.push(m);
          this.onMode(m);
          return;
        }
        const p = ent.waiting.get(d.id);
        if (!p) return;
        ent.waiting.delete(d.id);
        ent.busy--;
        if (d.error) p.no(Object.assign(new Error(d.error.message), { name: d.error.name }));
        else p.ok(d.planes!);
      };
      w.postMessage({ type: "init", head, conf, gain });
      return ent;
    });
  }

  /** One chunk-year of the mosaic, read and scored: [agb codes n, height codes n, class n]. */
  read(timeIdx: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal): Promise<Uint16Array> {
    return this.ask({ timeIdx }, py0, px0, h, w, signal) as Promise<Uint16Array>;
  }

  /**
   * One chunk's change from one year to a later one (change.ts): [percent x 100 n, Mg/ha x 4 n, confidence x 10000 n];
   * lidarGain: gain by the lidar-taught network (gain.ts).
   */
  change(timeIdx: number, timeIdxB: number, py0: number, px0: number, h: number, w: number, lidarGain: boolean, signal?: AbortSignal): Promise<Int16Array> {
    return this.ask({ timeIdx, timeIdxB, lidarGain }, py0, px0, h, w, signal) as Promise<Int16Array>;
  }

  private ask(t: { timeIdx: number; timeIdxB?: number; lidarGain?: boolean }, py0: number, px0: number, h: number, w: number, signal?: AbortSignal): Promise<Uint16Array | Int16Array> {
    if (signal?.aborted) return Promise.reject(aborted());
    // a small hash of the chunk's place, so neighbors spread over the workers
    const slot = Math.abs(Math.imul(py0 >> 8, 73856093) ^ Math.imul(px0 >> 8, 19349663));
    const ent = this.ws[slot % this.ws.length]!;
    const id = ++this.nextId;
    ent.busy++;
    return new Promise<Uint16Array | Int16Array>((ok, no) => {
      ent.waiting.set(id, { ok, no });
      ent.w.postMessage({ type: "read", id, px0, py0, w, h, ...t });
      signal?.addEventListener(
        "abort",
        () => {
          if (!ent.waiting.has(id)) return;
          ent.waiting.delete(id);
          ent.busy--;
          ent.w.postMessage({ type: "abort", id });
          no(aborted());
        },
        { once: true },
      );
    });
  }
}
