/**
 * Viridis as 50 stops (fire-knn's aef/ramp.ts): matplotlib's path (dark purple, blue, teal, green, yellow) with
 * five darker stops below its first color toward near-black and three paler stops above its last toward a pale
 * lemon, the way five-min-away-redux remade CARTO Emrld as 50 stops. Seven stops washed out; fifty do not.
 */
export const VIRIDIS_50 = [
  "#07020f", "#13021d", "#1f022b", "#2c0138", "#380146", "#440154", "#460a5d", "#471365", "#481b6d", "#482374",
  "#472c7a", "#46337f", "#443a83", "#424186", "#3e4989", "#3c508b", "#39568c", "#365d8d", "#32648e", "#306a8e",
  "#2d708e", "#2b758e", "#287c8e", "#26828e", "#24878e", "#228d8d", "#20938c", "#1f998a", "#1f9f88", "#20a486",
  "#25ab82", "#2ab07f", "#32b67a", "#3bbb75", "#48c16e", "#54c568", "#60ca60", "#6ece58", "#7fd34e", "#8ed645",
  "#9dd93b", "#addc30", "#c0df25", "#d0e11c", "#dfe318", "#efe51c", "#fde725", "#feef54", "#fef784", "#ffffb3",
];

/**
 * Land cover: one flat color per class of the head, in the head's class order (biomass_aef.json). Spread over
 * lightness (OKLab L .44 to .94) and the blue to yellow axis so every pair stays apart under a protanope or deuteranope
 * simulation (Machado 2009, OKLab distance .096 or more); no red. Built is a cool gray, not the red land cover maps use.
 */
export const LAND_COVER: [string, string][] = [
  ["trees", "#24613f"],
  ["shrub", "#7b8d4b"],
  ["grass", "#c9d890"],
  ["crop", "#e5a62f"],
  ["built", "#7d7b8c"],
  ["bare", "#f2e9d6"],
  ["water", "#2a63ad"],
  ["wetland", "#6cb6cf"],
  ["moss", "#d8cdee"],
];

/**
 * The each-year views (R cycles them): biomass on viridis, or the head's class in its flat color lit as a relief of the
 * biomass. Two more the shader still draws are off: the class's hue with its lightness from biomass (shrub and grass
 * come out alike), and with its lightness from canopy height (ground without a canopy washes out to white).
 */
export type YearView = { name: string; kind: "ramp" | "classBiomass" | "classHeight" | "classRelief"; stops: string[] };
/** The shader's number for each view (biomass.ts BiomassColor), fixed whichever views are in the list. */
export const VIEW_CODE: Record<YearView["kind"], number> = { ramp: 0, classBiomass: 1, classHeight: 2, classRelief: 3 };
export const YEAR_RAMPS: YearView[] = [
  { name: "viridis", kind: "ramp", stops: VIRIDIS_50 },
  // { name: "class, lightness by biomass", kind: "classBiomass", stops: [] },
  // { name: "class, lightness by height", kind: "classHeight", stops: [] },
  { name: "class, relief from biomass", kind: "classRelief", stops: [] },
];

/** OKLab of a hex color, and back (clipped to sRGB); the class views mix in it. */
const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const gam = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.max(0, c) ** (1 / 2.4) - 0.055);
export function toOklab(h: string): [number, number, number] {
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(h.slice(i, i + 2), 16) / 255)) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
export function oklabToHex([L, a, b]: [number, number, number]): string {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return `#${rgb.map((c) => Math.round(255 * Math.min(1, Math.max(0, gam(Math.min(1, Math.max(0, c)))))).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * The lightness views' range: OKLab L from LIGHT_HI where the value is least to LIGHT_LO where it is most (dense
 * forest dark), the class's own hue and chroma throughout, chroma eased toward the light and dark ends to stay in sRGB.
 */
export const LIGHT_HI = 0.95;
export const LIGHT_LO = 0.3;
export const easeChroma = (L: number) => 1 - 0.6 * Math.min(1, Math.abs(L - 0.62) / 0.36) ** 2;
/** A class's color at lightness t (0 least, 1 most), as the shader draws it. */
export function classShade(k: number, t: number): string {
  const [, a, b] = toOklab(LAND_COVER[k]![1]);
  const L = LIGHT_HI + (LIGHT_LO - LIGHT_HI) * Math.min(1, Math.max(0, t));
  const f = easeChroma(L);
  return oklabToHex([L, a * f, b * f]);
}

/**
 * Biomass change as 50 stops: the Landsat CTrees notebooks' change ramp (ne-landsat-ctrees-pair.py DIV_RAMP: deep
 * orange, orange, pale orange, white, pale blue, blue, deep blue) run in OKLab, with one deeper stop past each end.
 * Loss (orange) first, the white middle no change, gain (blue) last: the blue to orange axis a protanope keeps.
 */
export const CHANGE_50 = [
  "#6e3600", "#7e3e00", "#8f4600", "#9f4e00", "#ac5502", "#b45b06", "#bc600a", "#c4660e", "#cc6c13", "#d47216",
  "#dc781a", "#e27f24", "#e58732", "#e8903e", "#eb9849", "#eda053", "#f0a85d", "#f3b066", "#f4b976", "#f4c28a",
  "#f5cb9e", "#f4d4b1", "#f4ddc4", "#f3e5d6", "#f3eee9", "#edeff1", "#e2e8f0", "#d8e2ef", "#cddced", "#c3d5ec",
  "#b9cfea", "#aec8e9", "#a3c1e6", "#98b9e3", "#8db2e0", "#82aadc", "#77a2d9", "#6c9bd5", "#6193d2", "#578ccd",
  "#4f85c7", "#477ec2", "#3e77bc", "#3670b6", "#2d69b1", "#2462ab", "#1d5aa1", "#1a5194", "#164887", "#133f7a",
];

export const hexToRgb = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16) / 255,
  parseInt(h.slice(3, 5), 16) / 255,
  parseInt(h.slice(5, 7), 16) / 255,
];

/** The ramp as a GLSL const array and its lookup, `name(t)` for t in 0..1. */
export function glslRamp(name: string, stops: string[]): string {
  const rgb = stops.map(hexToRgb);
  const N = rgb.length;
  return `const vec3 ${name.toUpperCase()}[${N}] = vec3[${N}](${rgb.map((c) => `vec3(${c.map((v) => v.toFixed(4)).join(", ")})`).join(", ")});
vec3 ${name}(float t) {
  float x = clamp(t, 0.0, 1.0) * ${N - 1}.0;
  int i = min(int(floor(x)), ${N - 2});
  return mix(${name.toUpperCase()}[i], ${name.toUpperCase()}[i + 1], x - float(i));
}`;
}

export const cssGradient = (stops: string[]) => `linear-gradient(90deg, ${stops.join(", ")})`;

/** Flat bands, one per color (a class key folded into the card's header). */
export const cssBands = (stops: string[]) =>
  `linear-gradient(90deg, ${stops.map((c, i) => `${c} ${(i / stops.length) * 100}% ${((i + 1) / stops.length) * 100}%`).join(", ")})`;
