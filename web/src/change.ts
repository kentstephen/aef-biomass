import { GAIN_LIDAR_FROM, type GainNet, gainP } from "./gain";
import { codeValue, deq, NODATA } from "./head";

/**
 * Biomass change between two years, per 10 m pixel, drawn with its confidence (FINDINGS 29 to 31). A pixel's
 * confidence is how unusual its change is on steady ground: the share of steady pixels of the same class (the
 * head's own, at the first year) and window length whose change is smaller (models/change_conf.json, built by
 * scripts/growth_test.py from the steady pixels of 127 squares).
 * - Loss: the head's fall, its confidence the lower of two: the fall in percent of the pixel's own biomass, and the
 *   embedding's own move ((1 - cosine) / AEF_CUT). Drawn from LOSS_FROM up. Its size is the head's own fall f (a
 *   share of the pixel's biomass in year a) blended toward a soft curve, tanh(f / loss factor), by SOFT. The loss
 *   factor (the head's muting on clear loss cells, a 100 m mean) alone pushed every fall over ~42 % to the cap, so
 *   a half-burned and a cleared pixel drew the same; the raw fall keeps partial loss apart, the curve lifts it.
 * - Gain: the pixel's own rise, where it is at least TAU Mg/ha and a quarter of the pixel's biomass in year a (the
 *   two-part floor); its confidence from the rise in Mg/ha. Per pixel finds more of lidar's regrowth than 100 m
 *   blocks for the same steady ground lit: averaging over 100 m does not lower the noise, whole patches drift.
 * - Gain, lidar-taught (when asked): the pixel's own rise, where the lidar-taught network (gain.ts) scores it above
 *   all but 1.5 % of steady ground at the window's length; its confidence that share.
 * - Cut: where gain would be drawn but the head's own height and class say the forest was cut (CUT_FROM_M tall in
 *   year a and fell CUT_FALL_M or more, or trees in year a and open ground in year b), the pixel is drawn as a cut
 *   instead (FINDINGS 42: on a fresh harvest the head's biomass can rise while its height falls 12 to 2 m and its
 *   class turns to grass). A cut has no amount: its biomass is the answer that failed. Its confidence code is CUT.
 * Factors and floors from results/change_combined.json, linear between the one-year and seven-year fits.
 * Everything else is exactly 0, which the map leaves fully transparent.
 */
export const AEF_CUT = 0.1035354882478714;
const FIT = { L1: { tau: 20, loss: 0.34219714999198914 }, L7: { tau: 40, loss: 0.8100864887237549 } };
/** Loss is drawn from this confidence up (FINDINGS 30: at .95 it lights 1 to 3 % of steady ground). */
export const LOSS_FROM = 0.9;
/** How far loss's size goes from the head's raw fall (0) toward the soft curve (1). */
export const SOFT = 2 / 3;
/** The cut check on gain (scripts/cut_rule.py): the head's height in year a, and its fall to year b, in m. */
export const CUT_FROM_M = 5;
export const CUT_FALL_M = 5;
/** The classes a cut forest turns into. */
const OPEN = ["grass", "crop", "built", "bare"];

export function ruleFor(years: number) {
  const f = Math.min(1, Math.max(0, (years - 1) / 6));
  return { tau: FIT.L1.tau + (FIT.L7.tau - FIT.L1.tau) * f, loss: FIT.L1.loss + (FIT.L7.loss - FIT.L1.loss) * f };
}

/** The steady-ground tables: per value, "class:L" -> quantiles at the shares p. */
export type ConfTables = {
  classes: string[];
  p: number[];
  L: number[];
  q: Record<"neg_dpct" | "pos_d" | "mv", Record<string, number[]>>;
};

/** The share of steady pixels below x, from a table's quantiles (linear between them). */
function shareBelow(p: number[], q: number[], x: number): number {
  if (!(x > q[0]!)) return 0;
  const n = q.length;
  if (x > q[n - 1]!) return 1;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (q[m]! < x) lo = m;
    else hi = m;
  }
  const span = q[hi]! - q[lo]!;
  return span > 0 ? p[lo]! + ((p[hi]! - p[lo]!) * (x - q[lo]!)) / span : p[lo]!;
}

/**
 * The change plane's codes: percent x 100 (no data at CH_NODATA), Mg/ha x 4, the confidence x 10000 (CUT for a cut,
 * drawn at -100 % with no amount), and the head's class in each year (year a's index + 256 x year b's; -1 where
 * either year has no data).
 */
export const CH_NODATA = -32768;
export const CUT = -1;
export const PCT_SCALE = 100;
export const MG_SCALE = 4;
export const CONF_SCALE = 10000;
const clamp16 = (v: number) => Math.max(-32767, Math.min(32767, Math.round(v)));

/**
 * One chunk's change from year a to year b. ea, eb: band-major int8 [64, n]; ca, cb: the head's planes for each
 * ([agb codes n, height codes n, class n]). Out: Int16 [percent x 100 n, Mg/ha x 4 n, confidence x 10000 n,
 * classes n].
 * Both loss and gain are a percent of the pixel's own biomass in year a.
 */
export function changeOf(
  ea: Int8Array, eb: Int8Array, ca: Uint16Array, cb: Uint16Array, n: number, years: number, T: ConfTables,
  g: GainNet | null = null,
): Int16Array {
  const { tau, loss } = ruleFor(years);
  const L = Math.min(Math.max(years, T.L[0]!), T.L[T.L.length - 1]!);
  const tab = (v: keyof ConfTables["q"], c: number) => T.q[v][`${T.classes[c] ?? "all"}:${L}`] ?? T.q[v][`all:${L}`]!;
  const out = new Int16Array(4 * n).fill(CH_NODATA, 0, n).fill(-1, 3 * n);
  // the lidar gain's steady table at this window length (the nearest length it has)
  const gLs = g ? Object.keys(g.h.steady.L).map(Number).sort((x, y) => x - y) : [];
  const gL = gLs.length ? gLs.reduce((best, x) => (Math.abs(x - years) < Math.abs(best - years) ? x : best), gLs[0]!) : 0;
  const gQ = g && gLs.length ? g.h.steady.L[String(gL)]!.q : null;
  const trees = T.classes.indexOf("trees");
  const open = new Set(OPEN.map((c) => T.classes.indexOf(c)).filter((i) => i >= 0));
  for (let p = 0; p < n; p++) {
    if (ca[p] === 0 || cb[p] === 0 || ea[p] === NODATA || eb[p] === NODATA) continue;
    const a = codeValue(ca[p]!), d = codeValue(cb[p]!) - a;
    const cls = ca[2 * n + p]!;
    out[3 * n + p] = cls + 256 * cb[2 * n + p]!;
    let mg = 0, pct = 0, conf = 0;
    // the head's own height and class say the forest was cut
    const ha = codeValue(ca[n + p]!), hb = codeValue(cb[n + p]!);
    const cut = ha >= CUT_FROM_M && (ha - hb >= CUT_FALL_M || (cls === trees && open.has(cb[2 * n + p]!)));
    if (d < 0) {
      const cPct = shareBelow(T.p, tab("neg_dpct", cls), (-100 * d) / Math.max(a, 1));
      if (cPct >= LOSS_FROM) {
        // the embedding's own move between the two years, as a cosine on the dequantized bands
        let ab = 0, aa = 0, bb = 0;
        for (let b = 0; b < 64; b++) {
          const u = deq(ea[b * n + p]!), v = deq(eb[b * n + p]!);
          ab += u * v;
          aa += u * u;
          bb += v * v;
        }
        const mv = (1 - ab / Math.max(Math.sqrt(aa * bb), 1e-12)) / AEF_CUT;
        conf = Math.min(cPct, shareBelow(T.p, tab("mv", cls), mv));
        if (conf >= LOSS_FROM) {
          const f = -d / Math.max(a, 1);
          mg = -a * Math.min(1, (1 - SOFT) * f + SOFT * Math.tanh(f / loss));
          pct = (100 * mg) / Math.max(a, 1);
        }
      }
    } else if (g && gQ) {
      if (d > 0) {
        const pr = gainP(g, ea, eb, n, p, years);
        const c = pr === pr ? shareBelow(g.h.steady.p, gQ, pr) : 0;
        if (c >= GAIN_LIDAR_FROM && cut) {
          out[p] = -100 * PCT_SCALE;
          out[2 * n + p] = CUT;
          continue;
        }
        if (c >= GAIN_LIDAR_FROM) {
          mg = d;
          pct = (100 * d) / Math.max(a, 1);
          conf = c;
        }
      }
    } else if (d >= tau && d >= 0.25 * Math.max(a, 1) && cut) {
      out[p] = -100 * PCT_SCALE;
      out[2 * n + p] = CUT;
      continue;
    } else if (d >= tau && d >= 0.25 * Math.max(a, 1)) {
      mg = d;
      pct = (100 * d) / Math.max(a, 1);
      conf = shareBelow(T.p, tab("pos_d", cls), d);
    }
    out[p] = mg === 0 ? 0 : clamp16(pct * PCT_SCALE) || (mg < 0 ? -1 : 1);
    out[n + p] = clamp16(mg * MG_SCALE);
    out[2 * n + p] = mg === 0 ? 0 : clamp16(conf * CONF_SCALE);
  }
  return out;
}
