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
