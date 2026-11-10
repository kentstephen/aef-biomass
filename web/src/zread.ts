import { Decompress } from "fzstd";
import { Zstd } from "numcodecs";

import { fromS3, NUM_BANDS, ZARR_URL } from "./aef";

/**
 * The mosaic's chunks read in two parts. Each 256 px inner chunk is the 64 bands one after another (band-major,
 * "bytes" then zstd, no shuffle), so the front of its compressed bytes decodes to the first bands: the first 32 are
 * about half of a chunk's bytes (at most 53 % over 30 chunks in five places). front() reads 56 % of the chunk and
 * gives those 32 bands, rest() reads the other 44 % and gives all 64. Nothing is read twice for one draw, and
 * nothing is kept on the reader's disk: the scored planes stay in memory (biomass.ts) and a chunk read again later
 * is fetched again.
 */

const ROOT = fromS3(ZARR_URL);
const ARRAY = "/embeddings";
const SHARD = 4096, CHUNK = 256, PER = SHARD / CHUNK;
/** 256 [offset, length] pairs of uint64, then a crc32c */
const INDEX_BYTES = PER * PER * 16 + 4;
/** The share of a chunk read first; FRONT_BANDS decode from it (refetched whole in the rare chunk where they do not). */
export const FRONT_BANDS = 32;
const FRONT_SHARE = 0.56;
const EMPTY = 0xffffffffffffffffn;

const zstd = Zstd.fromConfig({ id: "zstd", level: 3 });

/**
 * The reads of this worker, a few at a time, by priority: the view's fronts (0), then its rests (1), then reading
 * ahead the next year, fronts (2) and rests (3); in order of asking within each. So the whole view draws rough before
 * any chunk sharpens, and the view sharpens before the next year is read.
 */
const LANES = 2;
type Job = { prio: number; go: () => void; signal?: AbortSignal };
const queue: Job[] = [];
let active = 0;
function pump() {
  while (active < LANES && queue.length) {
    let bi = 0;
    for (let i = 1; i < queue.length; i++) if (queue[i]!.prio < queue[bi]!.prio) bi = i;
    const j = queue.splice(bi, 1)[0]!;
    if (j.signal?.aborted) {
      j.go();
      continue;
    }
    active++;
    j.go();
  }
}
function scheduled<T>(prio: number, fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((ok, no) => {
    queue.push({
      prio,
      signal,
      go: () => {
        if (signal?.aborted) {
          no(Object.assign(new Error("aborted"), { name: "AbortError" }));
          return;
        }
        fn().then(ok, no).finally(() => {
          active--;
          pump();
        });
      },
    });
    pump();
  });
}

const fetchRange = async (url: string, range: string, signal?: AbortSignal) => {
  const r = await fetch(url, { headers: { Range: `bytes=${range}` }, cache: "no-store", signal });
  if (!r.ok) throw new Error(`${r.status} for ${url}`);
  return new Uint8Array(await r.arrayBuffer());
};

const shardIndex = new Map<string, Promise<DataView>>();
/** A shard's index (its last 4,100 bytes), once per worker, kept in memory. */
function index(key: string, prio: number, signal?: AbortSignal): Promise<DataView> {
  let p = shardIndex.get(key);
  if (p) return p;
  p = (async () => {
    const b = await scheduled(prio, () => fetchRange(`${ROOT}${key}`, `-${INDEX_BYTES}`, signal), signal);
    if (b.byteLength !== INDEX_BYTES) throw new Error(`shard index of ${b.byteLength} bytes`);
    return new DataView(b.buffer, b.byteOffset, b.byteLength);
  })();
  shardIndex.set(key, p);
  p.catch(() => shardIndex.delete(key));
  return p;
}

export type Part = {
  /** all 64 bands, band-major int8 [64, n], when the whole chunk is in hand (the front was all of it, or empty) */
  full?: Int8Array;
  /** the first FRONT_BANDS bands, band-major int8 [FRONT_BANDS, n] */
  front?: Int8Array;
  /** the rest of the chunk read, and all 64 bands */
  rest: (signal?: AbortSignal) => Promise<Int8Array>;
};

const asInt8 = (u: Uint8Array) => new Int8Array(u.buffer, u.byteOffset, u.byteLength);

/**
 * A chunk of year index t at (py0, px0) on the mosaic's grid: its front (all of it where the front is the whole chunk).
 * ahead: reading ahead (after the view's own reads).
 */
export async function front(t: number, py0: number, px0: number, signal?: AbortSignal, ahead = false): Promise<Part> {
  const pf = ahead ? 2 : 0, pr = pf + 1;
  const n = CHUNK * CHUNK, need = FRONT_BANDS * n;
  const key = `${ARRAY}/c/${t}/0/${Math.floor(py0 / SHARD)}/${Math.floor(px0 / SHARD)}`;
  const ix = Math.floor((px0 % SHARD) / CHUNK), iy = Math.floor((py0 % SHARD) / CHUNK);
  const view = await index(key, pf, signal);
  const off = view.getBigUint64(16 * (iy * PER + ix), true), len = view.getBigUint64(16 * (iy * PER + ix) + 8, true);
  if (off === EMPTY && len === EMPTY) {
    const full = new Int8Array(NUM_BANDS * n).fill(-128);
    return { full, rest: async () => full };
  }
  const o = Number(off), L = Number(len);
  const decodeAll = async (bytes: Uint8Array) => {
    const d = await zstd.decode(bytes);
    if (d.byteLength !== NUM_BANDS * n) throw new Error(`chunk of ${d.byteLength} bytes`);
    return asInt8(d);
  };
  const F = Math.min(L, Math.ceil(FRONT_SHARE * L));
  const head = await scheduled(pf, () => fetchRange(`${ROOT}${key}`, `${o}-${o + F - 1}`, signal), signal);
  const whole = async (tail: Uint8Array) => {
    const all = new Uint8Array(L);
    all.set(head);
    all.set(tail, F);
    return decodeAll(all);
  };
  if (F === L) {
    const full = await whole(new Uint8Array(0));
    return { full, rest: async () => full };
  }
  const rest = async (sig?: AbortSignal) => whole(await scheduled(pr, () => fetchRange(`${ROOT}${key}`, `${o + F}-${o + L - 1}`, sig), sig));
  // the front's bands, decoded as a stream until they are all out
  const out = new Uint8Array(need);
  let got = 0;
  const ds = new Decompress((c) => {
    if (got >= need) return;
    const k = Math.min(c.length, need - got);
    out.set(c.subarray(0, k), got);
    got += k;
  });
  for (let p = 0; p < head.length && got < need; p += 65536) ds.push(head.subarray(p, Math.min(head.length, p + 65536)));
  if (got < need) {
    // a chunk whose first bands take more than the share read: the rest now, and all of it at once
    const full = await rest(signal);
    return { full, rest: async () => full };
  }
  return { front: asInt8(out), rest };
}
