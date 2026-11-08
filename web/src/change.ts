import { codeValue, deq, NODATA } from "./head";

/**
 * Biomass change between two years, the rule FINDINGS section 22 found best against airborne lidar (.84 per 100 m
 * cell) and the cheapest to read: AlphaEarth for both parts.
 * - Loss, per pixel: the head's fall from year a to year b, kept only where the embedding itself moved past the
 *   frozen clearcut cutoff ((1 - cosine) / AEF_CUT >= 1, models/detectors.npz), divided by the loss factor (the
 *   head's muting on clear loss cells), and capped at the pixel's own biomass in year a.
 * - Gain, per 100 m cell (10 x 10 pixels on the mosaic's grid): the head's mean rise, counted at its own size where
 *   it is at least TAU Mg/ha and a quarter of the cell's biomass in year a. Scaling it up again brings the noise back.
 * Factors and floors from results/change_combined.json (floor, raw, aef+oney_aef), linear between the one-year and
 * seven-year fits. Everything else is exactly 0, which the map leaves fully transparent (96 % of steady pixels).
 */
export const AEF_CUT = 0.1035354882478714;
const FIT = { L1: { tau: 20, loss: 0.34219714999198914 }, L7: { tau: 40, loss: 0.8100864887237549 } };
export const CELL = 10;
/** The fewest pixels with both years for a cell's gain to count (a cell cut by a chunk edge keeps its part). */
const CELL_MIN = 30;

export function ruleFor(years: number) {
  const f = Math.min(1, Math.max(0, (years - 1) / 6));
  return { tau: FIT.L1.tau + (FIT.L7.tau - FIT.L1.tau) * f, loss: FIT.L1.loss + (FIT.L7.loss - FIT.L1.loss) * f };
}

/** The change plane's codes: percent x 100 (no data at CH_NODATA), and Mg/ha x 4. */
export const CH_NODATA = -32768;
export const PCT_SCALE = 100;
export const MG_SCALE = 4;
const clamp16 = (v: number) => Math.max(-32767, Math.min(32767, Math.round(v)));

/**
 * One chunk's change from year a to year b. ea, eb: band-major int8 [64, n]; ca, cb: the head's codes for each
 * ([agb n, height n]). px0, py0: the chunk's place on the mosaic's grid, so the 100 m cells line up across chunks.
 * Out: Int16 [percent x 100 n, Mg/ha x 4 n]. Loss is a percent of the pixel's own biomass in year a, gain of its
 * cell's.
 */
export function changeOf(
  ea: Int8Array, eb: Int8Array, ca: Uint16Array, cb: Uint16Array,
  w: number, h: number, px0: number, py0: number, years: number,
): Int16Array {
  const n = w * h;
  const { tau, loss } = ruleFor(years);
  const a = new Float32Array(n), d = new Float32Array(n);
  const ok = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    if (ca[p] === 0 || cb[p] === 0 || ea[p] === NODATA || eb[p] === NODATA) continue;
    ok[p] = 1;
    a[p] = codeValue(ca[p]!);
    d[p] = codeValue(cb[p]!) - a[p]!;
  }
  // the cells: the mean rise and the mean year-a biomass of the pixels with both years
  const cx0 = Math.floor(px0 / CELL), cy0 = Math.floor(py0 / CELL);
  const ncx = Math.floor((px0 + w - 1) / CELL) - cx0 + 1, ncy = Math.floor((py0 + h - 1) / CELL) - cy0 + 1;
  const sd = new Float32Array(ncx * ncy), sa = new Float32Array(ncx * ncy), cnt = new Uint16Array(ncx * ncy);
  const cellOf = (p: number) => {
    const x = p % w, y = (p / w) | 0;
    return (Math.floor((py0 + y) / CELL) - cy0) * ncx + (Math.floor((px0 + x) / CELL) - cx0);
  };
  for (let p = 0; p < n; p++) {
    if (!ok[p]) continue;
    const c = cellOf(p);
    sd[c] += d[p]!;
    sa[c] += a[p]!;
    cnt[c]!++;
  }
  const gain = new Float32Array(ncx * ncy), gainBase = new Float32Array(ncx * ncy);
  for (let c = 0; c < gain.length; c++) {
    if (cnt[c]! < CELL_MIN) continue;
    const g = sd[c]! / cnt[c]!, base = Math.max(sa[c]! / cnt[c]!, 1);
    if (g > 0 && g >= tau && g >= 0.25 * base) {
      gain[c] = g;
      gainBase[c] = base;
    }
  }
  const out = new Int16Array(2 * n).fill(CH_NODATA, 0, n);
  for (let p = 0; p < n; p++) {
    if (!ok[p]) continue;
    let mg = 0, pct = 0;
    if (d[p]! < 0) {
      // the gate: the embedding's own move between the two years, as a cosine on the dequantized bands
      let ab = 0, aa = 0, bb = 0;
      for (let b = 0; b < 64; b++) {
        const u = deq(ea[b * n + p]!), v = deq(eb[b * n + p]!);
        ab += u * v;
        aa += u * u;
        bb += v * v;
      }
      const cos = ab / Math.max(Math.sqrt(aa * bb), 1e-12);
      if ((1 - cos) / AEF_CUT >= 1) {
        const l = Math.max(d[p]! / loss, -a[p]!);
        mg += l;
        pct += (100 * l) / Math.max(a[p]!, 1);
      }
    }
    const c = cellOf(p);
    if (gain[c]! > 0) {
      mg += gain[c]!;
      pct += (100 * gain[c]!) / gainBase[c]!;
    }
    out[p] = mg === 0 ? 0 : clamp16(pct * PCT_SCALE) || (mg < 0 ? -1 : 1);
    out[n + p] = clamp16(mg * MG_SCALE);
  }
  return out;
}
