import type { ConfTables } from "./change";
import type { FillNet } from "./fill";
import type { GainHead } from "./gain";
import type { BiomassHead } from "./head";
import BiomassWorker from "./biomass.worker.ts?worker";

/** A few read-and-score workers (after fire-knn's aef/pool.ts); a chunk always goes to the same one. */
type W = { w: Worker; busy: number; waiting: Map<number, { ok: (p: Uint16Array | Int16Array) => void; no: (e: Error) => void }> };
export type Mode = { mode: string; probeGap: number; gpuCodeGap: number; gainGap?: number };
export type FillMode = { k: number; mode: string; cpuGap: { worst: number; off: number }; gpuGap: { worst: number; off: number } | null };
export type TiffStats = { fetched: number; fetchedBytes: number; disk: number; mem: number };

const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

export class Pool {
  private ws: W[];
  private nextId = 0;
  modes: Mode[] = [];
  onMode: (m: Mode) => void = () => {};
  fillModes: FillMode[] = [];
  onFill: (m: FillMode) => void = () => {};
  /** the GeoTIFF reads so far, per worker (each counts its own) */
  tiff: TiffStats[] = [];

  constructor(head: BiomassHead, conf: ConfTables, gain: GainHead | null, fills: FillNet[] = [], size = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2))) {
    this.ws = Array.from({ length: size }, () => {
      const w = new BiomassWorker();
      const ent: W = { w, busy: 0, waiting: new Map() };
      const wi = this.tiff.length;
      this.tiff.push({ fetched: 0, fetchedBytes: 0, disk: 0, mem: 0 });
      w.onmessage = (ev: MessageEvent<{ type?: string; id: number; planes?: Uint16Array | Int16Array; error?: { name: string; message: string }; tiff?: TiffStats; note?: string } & Partial<Mode> & Partial<FillMode>>) => {
        const d = ev.data;
        if (d.type === "note") {
          console.warn(d.note);
          return;
        }
        if (d.type === "fill") {
          const m = { k: d.k!, mode: d.mode!, cpuGap: d.cpuGap!, gpuGap: d.gpuGap ?? null };
          this.fillModes.push(m);
          this.onFill(m);
          return;
        }
        if (d.tiff) this.tiff[wi] = d.tiff;
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
      w.postMessage({ type: "init", head, conf, gain, fills });
      return ent;
    });
  }

  /**
   * One chunk-year, read and scored: [agb codes n, height codes n, class n]. bands: that many read from the GeoTIFFs,
   * the rest rebuilt (64: all read); none: the mosaic.
   */
  read(timeIdx: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal, bands?: number): Promise<Uint16Array> {
    return this.ask({ timeIdx, bands }, py0, px0, h, w, signal) as Promise<Uint16Array>;
  }

  /**
   * One chunk's change from one year to a later one (change.ts): [percent x 100 n, Mg/ha x 4 n, confidence x 10000 n];
   * lidarGain: gain by the lidar-taught network (gain.ts).
   */
  change(timeIdx: number, timeIdxB: number, py0: number, px0: number, h: number, w: number, lidarGain: boolean, signal?: AbortSignal, bands?: number): Promise<Int16Array> {
    return this.ask({ timeIdx, timeIdxB, lidarGain, bands }, py0, px0, h, w, signal) as Promise<Int16Array>;
  }

  private ask(t: { timeIdx: number; timeIdxB?: number; lidarGain?: boolean; bands?: number }, py0: number, px0: number, h: number, w: number, signal?: AbortSignal): Promise<Uint16Array | Int16Array> {
    if (signal?.aborted) return Promise.reject(aborted());
    // a small hash of the chunk's place, so neighbors spread over the workers; from the GeoTIFFs, by 4 x 4 chunk
    // blocks (about a GeoTIFF tile), so the chunks that share a band tile go to the worker that holds it
    const sh = t.bands ? 10 : 8;
    const slot = Math.abs(Math.imul(py0 >> sh, 73856093) ^ Math.imul(px0 >> sh, 19349663));
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
