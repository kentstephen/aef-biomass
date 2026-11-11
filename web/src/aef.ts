import * as zarr from "zarrita";

/**
 * The AlphaEarth Zarr mosaic on Source Cooperative, read as fire-knn reads it (web/src/aef/mosaic.ts): one level at
 * 10 m, no overviews, 256 px inner chunks of all 64 bands in 4096 px shards. Read through the gateway, which speaks
 * HTTP/2 (the S3 host behind it speaks HTTP/1.1, six reads at a time: a zoom 11 view of 162 chunks sharpened in 15 to
 * 17 s from the gateway against 26 to 32 s from S3, headless). The browser's HTTP cache is off: a suffix range
 * answered with a whole shard was coming back short from it, and with it on, the ranges of one shard (one URL) were
 * read one after another (the same view 15 s against 8.5 s, and no faster seen again). Nothing is written to the
 * reader's disk.
 */
export const ZARR_URL = "https://data.source.coop/tge-labs/aef-mosaic";
export const VARIABLE = "embeddings";
export const NUM_BANDS = 64;
export const YEAR_ORIGIN = 2017;

/** Below this the mosaic would read too much (every zoom step out is four times the chunks); nothing is read. */
export const MIN_ZOOM = 11;

/** The mosaic's grid (EPSG:4326, pixel registered) and its chunk size. */
export const RES = 8.983111749910169e-5;
export const X0 = -180;
export const Y0 = 83.68570533713473;
export const CHUNK = 256;

export type AefArray = zarr.Array<"int8", zarr.Readable>;

export async function openMosaic(): Promise<{ arr: AefArray; attrs: unknown }> {
  const store = new zarr.FetchStore(ZARR_URL, { overrides: { cache: "no-store" } });
  const root = await zarr.open.v3(store, { kind: "group" });
  const arr = await zarr.open.v3(root.resolve(VARIABLE), { kind: "array" });
  if (!arr.is("int8")) throw new Error(`Expected int8 embeddings, got ${arr.dtype}`);
  return { arr, attrs: root.attrs };
}

/**
 * Earlier versions kept every chunk read in the browser's IndexedDB (up to 16 GB, "aef-biomass-cache"); the map no
 * longer writes to disk, so a visit removes what an earlier one left.
 */
export function dropOldCache() {
  try {
    indexedDB.deleteDatabase("aef-biomass-cache");
  } catch {
    // no IndexedDB here: nothing was kept
  }
}
