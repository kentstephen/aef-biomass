import * as zarr from "zarrita";

import { DiskCachedStore } from "./disk-cache";

/**
 * The AlphaEarth Zarr mosaic on Source Cooperative, read as fire-knn reads it (web/src/aef/mosaic.ts): one level at
 * 10 m, no overviews, 256 px inner chunks of all 64 bands in 4096 px shards. Read from the S3 host behind the gateway
 * (the gateway throttles many requests at once), with the browser's HTTP cache off (a suffix range answered with a
 * whole shard was coming back short from it) and the chunks kept on disk instead (disk-cache.ts).
 */
export const ZARR_URL = "https://data.source.coop/tge-labs/aef-mosaic";
export const fromS3 = (url: string) =>
  url.replace("https://data.source.coop/", "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/");
export const VARIABLE = "embeddings";
export const NUM_BANDS = 64;
export const YEAR_ORIGIN = 2017;

/** Below this the mosaic would read too much (every zoom step out is four times the chunks); nothing is read. */
export const MIN_ZOOM = 11;

/**
 * The bands read per pixel (the comparison of 16, 32 and 64 from AlphaEarth's GeoTIFFs, tiff.ts) and the zoom each
 * is read from: a quarter of the bands at a quarter of the read per pixel holds four times the ground, one zoom step
 * out, at what zoom 11 costs with all 64; half the bands half a step.
 */
export type Bands = 16 | 32 | 64;
export const BANDS: Bands[] = [16, 32, 64];
export const BAND_FLOOR: Record<Bands, number> = { 16: 10, 32: 10.5, 64: 11 };

/** The mosaic's grid (EPSG:4326, pixel registered) and its chunk size. */
export const RES = 8.983111749910169e-5;
export const X0 = -180;
export const Y0 = 83.68570533713473;
export const CHUNK = 256;

export type AefArray = zarr.Array<"int8", zarr.Readable>;

export async function openMosaic(): Promise<{ arr: AefArray; attrs: unknown }> {
  const store = new DiskCachedStore(ZARR_URL, new zarr.FetchStore(fromS3(ZARR_URL), { overrides: { cache: "no-store" } }));
  const root = await zarr.open.v3(store, { kind: "group" });
  const arr = await zarr.open.v3(root.resolve(VARIABLE), { kind: "array" });
  if (!arr.is("int8")) throw new Error(`Expected int8 embeddings, got ${arr.dtype}`);
  return { arr, attrs: root.attrs };
}
