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
