import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { PathLayer } from "@deck.gl/layers";
import { MapLibreOverlay as MapboxOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { BAND_FLOOR, type Bands, BANDS, MIN_ZOOM, RES, X0, Y0 } from "./aef";
import { AGB_HI, type ChangeStats, createBiomass, GAIN_END, HIST_N, LOSS_END, windowKey, yearKey } from "./biomass";
import type { FillNet } from "./fill";
import type { GainHead } from "./gain";
import type { ConfTables } from "./change";
import type { BiomassHead } from "./head";
import { CHANGE_50, cssGradient, VIRIDIS_50 } from "./ramp";
import { S2_CREDIT, s2Layers, s2YearFor } from "./s2";
import { makeSearch } from "./search";

maplibregl.setWorkerUrl(maplibreWorkerUrl);

const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
// The Sierra Nevada's west slope by Yosemite: standing conifer forest, the 2013 Rim Fire's scar growing back in shrub,
// oak woodland and grass lower down
const HOME = { center: [-119.98, 37.86] as [number, number], zoom: 12.3 };
// the key's ticks in linear mode, Mg/ha
const TICKS = [0, 2, 10, 30, 100, 300];
// the map never tilts (Earthwork's FLAT); turning is Shift + left, right only
const FLAT = { maxPitch: 0, pitchWithRotate: false, touchPitch: false, dragRotate: false };

// the change key's ticks, percent of the pixel's own biomass
const CH_TICKS = [-100, -50, 0, 50, 150];
const MODE_KEY = "aef-biomass-mode";
const WIN_KEY = "aef-biomass-window";

type Mode = "year" | "change";
const stored = (k: string) => {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
};
const store = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {
    // no storage: not remembered
  }
};

const [head, confTables]: [BiomassHead, ConfTables] = await Promise.all(
  ["biomass_aef.json", "change_conf.json"].map(async (f) => (await fetch(`${import.meta.env.BASE_URL}models/${f}`)).json()),
) as [BiomassHead, ConfTables];
// the lidar-taught gain (G in change); the map runs without it if the file is missing
const gainHead: GainHead | null = await fetch(`${import.meta.env.BASE_URL}models/gain_lidar.json`)
  .then((r) => (r.ok ? r.json() : null)).catch(() => null);
// the fill-in networks for 16 and 32 bands (scripts/export_fill.py)
const fills: FillNet[] = await Promise.all(
  [16, 32].map(async (k) => (await fetch(`${import.meta.env.BASE_URL}models/fill_aef_${k}.json`)).json()),
);
const YEARS = head.years;
const savedWin = (stored(WIN_KEY) ?? "").split("-").map(Number);
const winOk = savedWin.length === 2 && YEARS.includes(savedWin[0]!) && YEARS.includes(savedWin[1]!) && savedWin[0]! < savedWin[1]!;
const st = {
  // each year (one year's biomass on viridis) or change (a window's loss and gain on orange to blue)
  mode: (stored(MODE_KEY) === "change" ? "change" : "year") as Mode,
  year: YEARS[YEARS.length - 1]!,
  // the change window, the whole run at first
  y0: winOk ? savedWin[0]! : YEARS[0]!,
  y1: winOk ? savedWin[1]! : YEARS[YEARS.length - 1]!,
  // the layer left, drawn under the new one until the new one's view is in
  prev: null as string | null,
  dir: -1,
  hover: null as [number, number] | null,
  // the year's view all in (deck.gl's onViewportLoad): the bar goes away even where a chunk has no data
  viewLoaded: false,
  ahead: null as AbortController | null,
  aheadKey: null as string | null,
  // Sentinel-2: held over the whole map (Space, one map only), or the pair's left side (P)
  holding: false,
  pair: false,
  // picking (Q, Earthwork's): the tooltip by the pointer and the picked pixel's gold boundary; remembered. Where the pointer is
  // on screen; the pixel clicked
  tips: stored("aef-biomass-tips") !== "0",
  // the basemap's place names (L), on both maps; remembered
  labels: stored("aef-biomass-labels") !== "0",
  hoverPt: null as [number, number] | null,
  picked: null as [number, number] | null,
  // the imagery's gamma (; and '), remembered
  gamma: Number(stored("aef-biomass-s2-gamma")) || 1,
  // gain by the lidar-taught network instead of the two-part floor (G, change only), remembered
  lidarGain: !!gainHead && stored("aef-biomass-gain") === "lidar",
  // the bands read per pixel from AlphaEarth's GeoTIFFs (N cycles), the rest rebuilt; remembered
  bands: (BANDS.find((b) => String(b) === stored("aef-biomass-bands")) ?? 16) as Bands,
};
/** The layer shown: the year, or the window. */
const curKey = () => (st.mode === "year" ? yearKey(st.year, st.bands) : windowKey(st.y0, st.y1, st.lidarGain, st.bands));
/** A layer key as words: "2025, 16 bands", or "2018 to 2025" (", lidar-taught gain" for a window ending in L). */
const keyName = (k: string) => k.replace(/b(\d+)$/, ", $1 bands").replace("-", " to ").replace("L", ", lidar-taught gain");
/** The zoom the bands are read from. */
const floorZ = () => BAND_FLOOR[st.bands];

// ---- the page: the map fills the window; search and the card down the top left, the loading bar along the top
const mapEl = document.getElementById("map")!;
// the pair's left side (P): Sentinel-2, one camera with the map (Earthwork's .at-map2)
const mapEl2 = document.createElement("div");
mapEl2.id = "map2";
document.body.prepend(mapEl2);
// the imagery's year, big, top left: over the pair's left side, or over the whole map while Space is held
const s2card = document.createElement("div");
s2card.className = "s2card at-glass";

const bar = document.createElement("div");
bar.className = "ew-load";
bar.innerHTML = "<i></i>";
const tag = document.createElement("div");
tag.className = "ew-load-tag at-glass";
// Earthwork's fold chevron (ui.js ICON.chev)
const CHEV = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>';
const top = document.createElement("div");
top.className = "at-top";
const card = document.createElement("div");
card.className = "card at-glass";
card.innerHTML = `
  <div class="at-hd"><span class="t">Biomass</span><i class="at-ramp"></i><span class="hy"></span><button class="at-cb">${CHEV}</button></div>
  <div class="body">
  <div class="sub"></div>
  <div class="seg-f mode"><button data-m="year">Each year</button><button data-m="change">Change</button><kbd>C</kbd></div>
  <div class="bandrow"><span class="at-lab">Bands</span><div class="seg-f bandsel">${BANDS.map((b) => `<button data-b="${b}" title="${b} of 64 bands, read from zoom ${BAND_FLOOR[b]} (N cycles)">${b}</button>`).join("")}<kbd>N</kbd></div></div>
  <div class="yearrow">
    <button class="step" data-d="-1" title="Earlier year ([)">&lsaquo;</button>
    <div class="year"></div>
    <button class="step" data-d="1" title="Later year (])">&rsaquo;</button>
    <div class="years"></div>
  </div>
  <div class="at-row winrow">
    <span class="at-lab">Years</span>
    <span class="at-win"><span class="trk"></span><span class="spn" title="drag to move the window, its length kept"></span><span class="tks"></span></span>
    <span class="at-wtxt"></span>
  </div>
  <div class="key"><div class="hist"><div class="hl"></div><div class="cov"></div><div class="chart"><svg viewBox="0 0 ${HIST_N} 64" preserveAspectRatio="none"></svg><span class="pk l"></span><span class="pk r"></span></div></div><div class="bar"></div><div class="ticks"></div><div class="unit"></div></div>
  <div class="pairs"></div>
  <div class="readout">&nbsp;</div>
  <div class="zoom"></div>
  <div class="status"></div>
  <div class="keys"></div>
  <div class="repo"><a href="https://github.com/kentstephen/aef-biomass" target="_blank" rel="noopener">The code and how it works on GitHub</a></div>
  <div class="credits">AlphaEarth &copy; Google, Google DeepMind, via Source Cooperative &middot; head taught on CTrees AGB and Meta canopy height (CC BY 4.0) &middot; ${S2_CREDIT} &middot; search by Photon and Nominatim over OpenStreetMap &middot; &copy; CARTO &copy; OpenStreetMap</div>
  </div>`;
// the pixel's tooltip, by the pointer
const tip = document.createElement("div");
tip.className = "tip at-glass";
document.body.append(bar, tag, top, s2card, tip);
const $ = <T extends HTMLElement>(s: string) => card.querySelector(s) as T;
const fold = (on: boolean) => {
  card.classList.toggle("collapsed", on);
  $(".at-cb").title = on ? "open the card" : "fold the card";
  try {
    localStorage.setItem("aef-biomass-fold", on ? "1" : "0");
  } catch {
    // no storage: the fold is not remembered
  }
};
$(".at-hd").onclick = (e) => {
  e.stopPropagation();
  fold(!card.classList.contains("collapsed"));
};

const yearsEl = $(".years");
yearsEl.innerHTML = YEARS.map((y) => `<span data-y="${y}">${String(y).slice(2)}</span>`).join("");

// ---- the change window (Earthwork's "Years" range, ui.js rWin): two handles, each end its own; the span between
// them is a grab that moves the whole window, its length kept. It rereads when let go.
const win = $(".at-win");
const wSpn = $(".spn");
const wTks = $(".tks");
for (const y of YEARS) wTks.insertAdjacentHTML("beforeend", `<span><i>’${String(y).slice(-2)}</i></span>`);
const mkR = () => {
  const r = document.createElement("input");
  r.type = "range";
  r.min = "0";
  r.max = String(YEARS.length - 1);
  r.step = "1";
  r.title = "the window's ends: drag either one, it rereads when you let go";
  win.append(r);
  return r;
};
const rFrom = mkR(), rTo = mkR();
// the window while dragged, before it is let go
const drag = { y0: st.y0, y1: st.y1 };
function styleWin() {
  const n = YEARS.length - 1;
  const i0 = YEARS.indexOf(drag.y0), i1 = YEARS.indexOf(drag.y1);
  rFrom.value = String(i0);
  rTo.value = String(i1);
  // the end at the edge stays grabbable over the other
  rFrom.style.zIndex = i0 === n ? "3" : "2";
  rTo.style.zIndex = i1 === 0 ? "3" : "2";
  const usable = (win.clientWidth || 200) - 16;
  wSpn.style.left = `${8 + (usable * i0) / n}px`;
  wSpn.style.width = `${(usable * (i1 - i0)) / n}px`;
  $(".at-wtxt").textContent = `${drag.y0} to ${drag.y1}`;
}
const onDrag = (which: "from" | "to") => {
  let a = Number(rFrom.value), b = Number(rTo.value);
  if (a >= b) {
    if (which === "from") a = b - 1;
    else b = a + 1;
  }
  a = Math.max(0, a);
  b = Math.min(YEARS.length - 1, b);
  drag.y0 = YEARS[a]!;
  drag.y1 = YEARS[b]!;
  styleWin();
};
rFrom.addEventListener("input", () => onDrag("from"));
rTo.addEventListener("input", () => onDrag("to"));
// let go, the ends give the keys back (Earthwork's winRelease blurs them), so [ and ] step at once
const winRelease = () => {
  setWindow(drag.y0, drag.y1);
  rFrom.blur();
  rTo.blur();
};
rFrom.addEventListener("change", winRelease);
rTo.addEventListener("change", winRelease);
// the middle grab: the window slides a year per year's width dragged
wSpn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  wSpn.setPointerCapture(e.pointerId);
  wSpn.classList.add("held");
  const x0 = e.clientX, i0 = YEARS.indexOf(drag.y0), len = YEARS.indexOf(drag.y1) - i0;
  const perYear = ((win.clientWidth || 200) - 16) / (YEARS.length - 1);
  const move = (ev: PointerEvent) => {
    const i = Math.max(0, Math.min(YEARS.length - 1 - len, i0 + Math.round((ev.clientX - x0) / perYear)));
    drag.y0 = YEARS[i]!;
    drag.y1 = YEARS[i + len]!;
    styleWin();
  };
  const up = () => {
    wSpn.classList.remove("held");
    wSpn.removeEventListener("pointermove", move);
    wSpn.removeEventListener("pointerup", up);
    wSpn.removeEventListener("pointercancel", up);
    setWindow(drag.y0, drag.y1);
  };
  wSpn.addEventListener("pointermove", move);
  wSpn.addEventListener("pointerup", up);
  wSpn.addEventListener("pointercancel", up);
});
new ResizeObserver(styleWin).observe(win);

const map = new maplibregl.Map({ container: mapEl, style: STYLE, center: HOME.center, zoom: HOME.zoom, attributionControl: false, hash: true, ...FLAT });
map.touchZoomRotate.disableRotation();
// caught now: the head and the mosaic open while the style loads, and its load event can fire before they are done
const mapLoaded = new Promise<void>((ok) => map.once("load", () => ok()));
const overlay = new MapboxOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
new ResizeObserver(() => map.resize()).observe(mapEl);

// ---- the pair (P), Earthwork's: a second map on the left, made the first time the pair opens; either map moves the other
let map2: maplibregl.Map | null = null;
let overlay2: MapboxOverlay | null = null;
let firstLabel2: string | undefined;
function boot2() {
  if (map2) return;
  const m2 = new maplibregl.Map({ container: mapEl2, style: STYLE, center: map.getCenter(), zoom: map.getZoom(), attributionControl: false, ...FLAT });
  map2 = m2;
  m2.keyboard.disable();
  m2.touchZoomRotate.disableRotation();
  overlay2 = new MapboxOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck, left: ${e?.message ?? e}`) });
  m2.addControl(overlay2 as unknown as maplibregl.IControl);
  m2.on("load", () => {
    firstLabel2 = m2.getStyle().layers.find((l) => l.type === "symbol")?.id;
    styleLabels(m2);
    paint();
  });
  let syncing = false;
  const follow = (a: maplibregl.Map, b: maplibregl.Map) =>
    a.on("move", () => {
      if (syncing || !st.pair) return;
      syncing = true;
      b.jumpTo({ center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing() });
      syncing = false;
    });
  follow(map, m2);
  follow(m2, map);
  m2.on("moveend", () => paint());
  m2.on("idle", () => addMissingGroups(m2, [...s2Left, ...(st.pair ? outlines("l") : [])]));
  // the left side hovers as the right does: one camera, so a point is the same place on both
  m2.on("mousemove", hoverAt);
  m2.on("mouseout", hoverOff);
  m2.on("click", pickAt);
  new ResizeObserver(() => m2.resize()).observe(mapEl2);
}
function setPair(on: boolean) {
  if (on === st.pair) return;
  if (on && st.holding) endHold();
  st.pair = on;
  document.body.classList.toggle("pair", on);
  if (on) {
    // the left shows the map's year (imgYear)
    boot2();
    map2!.jumpTo({ center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing() });
  }
  render();
}

// ---- hold Space: the imagery over the whole map (one map only; paired, it is already on the left), on the map's year
function beginHold() {
  if (st.holding || st.pair) return;
  st.holding = true;
  mapEl.classList.add("holding");
  paint();
}
function endHold() {
  if (!st.holding) return;
  st.holding = false;
  mapEl.classList.remove("holding");
  paint();
}
function setGamma(g: number) {
  st.gamma = Math.round(Math.max(0.4, Math.min(2.5, g)) * 10) / 10;
  store("aef-biomass-s2-gamma", String(st.gamma));
  paint();
}
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") endHold();
});
// a lost window (Cmd-Tab with Space held) gets no keyup
window.addEventListener("blur", () => endHold());

const search = makeSearch(map, MIN_ZOOM);
top.append(search.el, card);
// the card loads as it was left, folded or open (now that it and the search are on the page)
fold((() => {
  try {
    return localStorage.getItem("aef-biomass-fold") === "1";
  } catch {
    return false;
  }
})());

function say(s: string) {
  $(".status").textContent = s;
}

// the biomass draws under the basemap's labels
let firstLabel: string | undefined;
const bio = await createBiomass(head, confTables, gainHead, () => firstLabel, fills);
let modeNote = "";
const setModeNote = () => {
  const ms = bio.pool.modes;
  const gpu = ms.filter((m) => m.mode === "gpu").length;
  modeNote = gpu ? `head on WebGPU (${gpu} of ${ms.length} workers)` : "head on the CPU (no WebGPU)";
  const fm = bio.pool.fillModes.filter((m) => m.k === st.bands);
  if (fm.length) {
    const g = fm.filter((m) => m.mode === "gpu").length;
    modeNote += g ? `, fill on WebGPU (${g} of ${fm.length})` : ", fill on the CPU (slow)";
  }
  status();
};
bio.pool.onMode = setModeNote;
bio.pool.onFill = setModeNote;
/** The GeoTIFF bytes read from the network so far, and the band tiles taken from the disk cache. */
const tiffNote = () => {
  const t = bio.pool.tiff;
  const mb = t.reduce((s, x) => s + x.fetchedBytes, 0) / 1e6, disk = t.reduce((s, x) => s + x.disk, 0);
  return mb || disk ? ` · GeoTIFF ${mb >= 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${mb.toFixed(0)} MB`} read${disk ? `, ${disk} tiles from disk` : ""}` : "";
};

const bbox = (): [number, number, number, number] => {
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
};
const readable = () => map.getZoom() >= floorZ();

// ---- the change in view, counted live (change mode): totals, columns over the ramp, what it was and became
const HIST_TOP = 12; // room at the chart's top for the scale's label
const num = (v: number) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v.toFixed(1));
let histTimer = 0;
/** Recount at most every 300 ms while the view loads or moves, and once after the last ask. */
function queueHist() {
  if (histTimer) return;
  histTimer = window.setTimeout(() => {
    histTimer = 0;
    drawHist();
  }, 300);
}
function drawHist() {
  if (st.mode !== "change") return;
  const svg = $(".hist svg"), hl = $(".hist .hl"), cov = $(".hist .cov"), pairs = $(".pairs");
  const pkL = $(".hist .pk.l"), pkR = $(".hist .pk.r");
  if (!readable()) {
    svg.innerHTML = "";
    hl.innerHTML = "";
    pkL.textContent = pkR.textContent = "";
    cov.textContent = `Zoom in to ${floorZ()} to count the change in view`;
    pairs.innerHTML = "";
    return;
  }
  const S: ChangeStats = bio.changeStats(bbox(), curKey());
  // loss and gain on one scale, so their heights compare; a column with any hectares keeps at least 1 px
  const tot = (b: number) => S.sure[b]! + S.faint[b]!;
  let m = 0;
  for (let b = 0; b < HIST_N; b++) m = Math.max(m, tot(b));
  const H = 64 - HIST_TOP;
  let bars = "";
  for (let b = 0; b < HIST_N; b++) {
    if (!m || !tot(b)) continue;
    const lift = Math.max(1, (H * tot(b)) / m) / ((H * tot(b)) / m);
    const hs = (lift * H * S.sure[b]!) / m, hf = (lift * H * S.faint[b]!) / m;
    const col = CHANGE_50[b]!;
    // solid: pixels the map draws more than half opaque; faint on top: the rest
    if (hs > 0) bars += `<rect x="${b + 0.1}" y="${64 - hs}" width="0.8" height="${hs}" fill="${col}"/>`;
    if (hf > 0) bars += `<rect x="${b + 0.1}" y="${64 - hs - hf}" width="0.8" height="${hf}" fill="${col}" fill-opacity="0.35"/>`;
  }
  svg.innerHTML = bars;
  pkL.textContent = m ? `tallest ${num(m)} ha` : "";
  pkR.textContent = "";
  const net = S.gain.t - S.loss.t;
  hl.innerHTML =
    `<span><i class="sw" style="background:${CHANGE_50[8]}"></i>Loss <b>${num(S.loss.ha)}</b> ha &middot; ${num(S.loss.t)} t</span>` +
    `<span><i class="sw" style="background:${CHANGE_50[41]}"></i>Gain <b>${num(S.gain.ha)}</b> ha &middot; ${num(S.gain.t)} t</span>`;
  const part = S.inHand < S.wanted ? ` &middot; ${S.inHand} of ${S.wanted} chunks counted so far` : "";
  cov.innerHTML = `In view: ${num(S.dataHa)} ha with data &middot; net ${net > 0 ? "+" : net < 0 ? "-" : ""}${num(Math.abs(net))} t${part}`;
  // what it was in the first year, and what it became in the last (the head's class)
  const table = (name: string, sd: ChangeStats["loss"], top: number) => {
    if (!sd.ha) return "";
    const rows = sd.pairs.slice(0, top).map((q) => {
      const what = !q.from ? "no class" : q.from === q.to ? `${q.from}, still ${q.to}` : `${q.from} to ${q.to}`;
      return `<tr><td>${what}</td><td>${num(q.ha)} ha</td><td>${Math.round((100 * q.ha) / sd.ha)}%</td><td>${num(q.t)} t</td></tr>`;
    });
    return `<div class="h">${name}: what it was, what it became</div><table>${rows.join("")}</table>`;
  };
  pairs.innerHTML = table("Loss", S.loss, 4) + table("Gain", S.gain, 3);
}

function status() {
  queueHist();
  const z = map.getZoom();
  const zEl = $(".zoom");
  zEl.classList.toggle("out", !readable());
  zEl.innerHTML = readable()
    ? `Zoom ${z.toFixed(1)} &middot; ${st.bands} of 64 bands, read from zoom ${floorZ()}`
    : `Zoom ${z.toFixed(1)}: zoom in to ${floorZ()} to read ${st.bands} bands. AlphaEarth is 10 m with no coarser levels; fewer bands read less, so they are read further out (16 from 10, 32 from 10.5, 64 from 11).`;
  if (!readable()) {
    bar.classList.remove("on");
    tag.classList.remove("on");
    say(modeNote);
    return;
  }
  const k = curKey();
  const { loaded, wanted } = bio.loadedIn(bbox(), k);
  // done when the view's chunks are all in, even if deck.gl's view-loaded call has not come
  const loading = !st.viewLoaded && wanted > 0 && loaded < wanted;
  bar.classList.toggle("on", loading);
  (bar.firstElementChild as HTMLElement).style.width = wanted ? `${(100 * loaded) / wanted}%` : "0";
  tag.classList.toggle("on", loading);
  // a window reads two years per chunk
  tag.innerHTML = `Reading ${keyName(k)}<span>${loaded} of ${wanted} chunks${st.mode === "change" ? ", two years each" : ""}</span>`;
  let ahead = "";
  if (st.aheadKey != null) {
    const h = bio.inHand(bbox(), st.aheadKey);
    const nm = keyName(st.aheadKey);
    ahead = h.inHand < h.wanted ? ` · reading ${nm} ahead, ${h.inHand} of ${h.wanted}` : ` · ${nm} read ahead`;
  }
  say(`${loaded} of ${wanted} chunks for ${keyName(k)}${ahead}${modeNote ? ` · ${modeNote}` : ""}${tiffNote()}`);
}

function keyTicks() {
  const ticks = $(".ticks");
  if (st.mode === "change") {
    ticks.innerHTML = CH_TICKS.map((v) => {
      const t = v < 0 ? 0.5 - (0.5 * -v) / LOSS_END : 0.5 + (0.5 * v) / GAIN_END;
      return `<span style="left:${t * 100}%">${v > 0 ? "+" : ""}${v}${v ? "%" : ""}</span>`;
    }).join("");
    return;
  }
  ticks.innerHTML =
    TICKS.map((v) => {
      const t = Math.log1p(v) / AGB_HI;
      // a tick near the end would sit on the end's own label
      return t <= 0.9 ? `<span style="left:${t * 100}%">${v}</span>` : "";
    }).join("") + `<span style="left:100%">${Math.round(Math.expm1(AGB_HI))}+</span>`;
}

/** The pixel under the pointer: the card's readout line, and the tooltip by the pointer (Q hides it). */
/**
 * When the change under the pointer happened (change view): the one-year step of the head's biomass that moved most
 * in the change's direction, by pixel and window; read once per pixel, the readout redrawn when it lands.
 */
const when = { key: "", years: null as [number, number] | null, busy: false, ac: null as AbortController | null };
function whenFor(lon: number, lat: number, sign: number): string {
  const k = `${Math.floor((lon - X0) / RES)}:${Math.floor((Y0 - lat) / RES)}:${st.y0}-${st.y1}:${sign}`;
  if (when.key !== k) {
    when.ac?.abort();
    const ac = (when.ac = new AbortController());
    Object.assign(when, { key: k, years: null, busy: true });
    bio.whenAtPoint(lon, lat, st.y0, st.y1, sign, ac.signal, st.bands).then(
      (y) => {
        if (when.key !== k) return;
        Object.assign(when, { years: y, busy: false });
        readout();
      },
      () => {
        if (when.key === k) when.busy = false;
      },
    );
  }
  if (when.years) return `between ${when.years[0]} and ${when.years[1]}`;
  return when.busy ? "finding the years" : "";
}

function readout() {
  const el = $(".readout");
  const off = () => {
    el.innerHTML = "&nbsp;";
    tip.classList.remove("on");
  };
  // the pixel under the pointer, else the one picked
  const pt = st.hover ?? st.picked;
  if (!pt) return off();
  const at = `${pt[1].toFixed(4)}, ${pt[0].toFixed(4)}`;
  // the tooltip, vertical: one item a line; the value, what it means, then when and where (muted)
  let line = "", big = "", mid = "", sub = "";
  if (st.mode === "change") {
    const c = bio.changeAtPoint(pt[0], pt[1], curKey());
    if (!c) return off();
    const sg = (v: number, d: number) => `${v > 0 ? "+" : ""}${v.toFixed(d)}`;
    if (c.mg === 0) {
      big = '<span class="none">No change counted</span>';
      line = `no change counted &middot; ${at}`;
    } else {
      big = `<b>${sg(c.mg, Math.abs(c.mg) >= 10 ? 0 : 1)}</b> Mg/ha<div class="m">${sg(c.pct, 0)}%</div>`;
      line = `<b>${sg(c.mg, Math.abs(c.mg) >= 10 ? 0 : 1)}</b> Mg/ha (${sg(c.pct, 0)}%) &middot; ${at}`;
    }
    // the confidence: the share of steady ground of its class with less change
    const share = `${Math.min(99.9, 100 * c.conf).toFixed(c.conf >= 0.99 ? 1 : 0)}%`;
    // what the head calls the pixel in each year: one class if it stayed, "a to b" if it changed
    const cls = !c.clsA ? "" : c.clsA === c.clsB ? c.clsA : `${c.clsA} to ${c.clsB}`;
    if (c.mg !== 0) {
      const yrs = whenFor(pt[0], pt[1], Math.sign(c.mg));
      mid = `<div>${c.mg < 0 ? "Loss" : "Gain"}${yrs ? ` ${yrs}` : ""}</div>${cls ? `<div>${cls}</div>` : ""}<div>more than ${share} of steady ground</div>`;
      line += `${yrs ? ` &middot; ${yrs}` : ""}${cls ? ` &middot; ${cls}` : ""} &middot; more than ${share} of steady ground`;
    } else if (cls) {
      mid = `<div>${cls}</div>`;
      line += ` &middot; ${cls}`;
    }
    sub = keyName(curKey());
  } else {
    const v = bio.valueAtPoint(pt[0], pt[1], st.year, st.bands);
    if (!v) return off();
    const a = v.agb >= 10 ? v.agb.toFixed(0) : v.agb.toFixed(1);
    big = `<b>${a}</b> Mg/ha`;
    mid = `<div>canopy ${v.height.toFixed(1)} m</div><div>${v.cls}</div>`;
    line = `<b>${a}</b> Mg/ha &middot; canopy ${v.height.toFixed(1)} m &middot; ${v.cls} &middot; ${at}`;
    sub = String(st.year);
  }
  el.innerHTML = line;
  if (!st.tips || !st.hoverPt || !st.hover) return void tip.classList.remove("on");
  tip.innerHTML = `<div class="v">${big}</div>${mid ? `<div class="c">${mid}</div>` : ""}<div class="s">${sub}</div><div class="s">${pt[1].toFixed(4)}</div><div class="s">${pt[0].toFixed(4)}</div>`;
  tip.classList.add("on");
  // by the pointer, kept inside the window
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const x = st.hoverPt[0] + 16 + w > innerWidth ? st.hoverPt[0] - 16 - w : st.hoverPt[0] + 16;
  const y = st.hoverPt[1] + 16 + h > innerHeight ? st.hoverPt[1] - 16 - h : st.hoverPt[1] + 16;
  tip.style.transform = `translate(${x}px, ${y}px)`;
}

/** Stop reading ahead (a move, a step, zoomed out). */
function stopAhead() {
  st.ahead?.abort();
  st.ahead = null;
  st.aheadKey = null;
}

/** With the view in: read ahead the next year in the stepping direction, then the one the other way (each year). */
function readAhead() {
  stopAhead();
  if (!readable()) return;
  const ac = (st.ahead = new AbortController());
  const next: string[] = [];
  for (const d of [st.dir, -st.dir]) {
    if (st.mode !== "year") break;
    const y = YEARS[YEARS.indexOf(st.year) + d];
    if (y !== undefined) next.push(yearKey(y, st.bands));
  }
  void (async () => {
    for (const k of next) {
      if (ac.signal.aborted) return;
      st.aheadKey = k;
      status();
      await bio.prefetch(k, bbox(), ac.signal, status);
    }
    if (st.ahead === ac) st.aheadKey = null;
    status();
  })();
}

/** The card for the mode: only the controls it uses, its ramp, key, words and keys. */
function styleMode() {
  const ch = st.mode === "change";
  card.classList.toggle("ch", ch);
  for (const b of card.querySelectorAll<HTMLButtonElement>(".mode button")) b.classList.toggle("on", b.dataset.m === st.mode);
  // folded, the short title (the card's width goes to the ramp and the years)
  $(".at-hd .t").innerHTML = ch ? '<span class="tl">Biomass change</span><span class="ts">Change</span>' : "Biomass";
  const ramp = cssGradient(ch ? CHANGE_50 : VIRIDIS_50);
  $(".bar").style.background = ramp;
  // folded, the card is its header: the short title, the ramp on the map and the year (Earthwork's card fold)
  $(".at-ramp").style.background = ramp;
  $(".sub").textContent = ch
    ? st.lidarGain
      ? "AlphaEarth, per 10 m pixel: loss where the head fell and the embedding itself moved, gain where the lidar-taught network sees canopy rise"
      : "AlphaEarth, per 10 m pixel: loss where the head fell and the embedding itself moved, gain where it rose past a floor"
    : "AlphaEarth, one year's embedding";
  $(".unit").textContent = ch
    ? "percent of the pixel's own biomass the first year. Faint: steady ground of its class often changes as much. Clear: no change counted"
    : "Mg/ha aboveground, per 10 m pixel";
  $(".keys").innerHTML = ch
    ? "<kbd>[</kbd><kbd>]</kbd>&nbsp;imagery year &nbsp; <kbd>B</kbd>&nbsp;blink &nbsp; <kbd>C</kbd>&nbsp;each year &nbsp; <kbd>N</kbd>&nbsp;bands &nbsp; " +
      (gainHead ? `<kbd>G</kbd>&nbsp;lidar gain, ${st.lidarGain ? "on" : "off"} &nbsp; ` : "") +
      "<kbd>Space</kbd>&nbsp;imagery &nbsp; <kbd>P</kbd>&nbsp;pair &nbsp; <kbd>Q</kbd>&nbsp;picking &nbsp; <kbd>L</kbd>&nbsp;labels &nbsp; <kbd>F</kbd>&nbsp;full screen &nbsp; <kbd>/</kbd>&nbsp;search &nbsp; <kbd>?</kbd>&nbsp;about"
    : '<kbd>[</kbd><kbd>]</kbd>&nbsp;year &nbsp; <kbd>B</kbd>&nbsp;blink &nbsp; <kbd>C</kbd>&nbsp;change &nbsp; <kbd>N</kbd>&nbsp;bands &nbsp; ' +
      (gainHead ? '<kbd>G</kbd>&nbsp;lidar gain (in change) &nbsp; ' : '') +
      '<kbd>Space</kbd>&nbsp;imagery &nbsp; <kbd>P</kbd>&nbsp;pair &nbsp; <kbd>Q</kbd>&nbsp;picking &nbsp; <kbd>L</kbd>&nbsp;labels &nbsp; <kbd>F</kbd>&nbsp;full screen &nbsp; <kbd>/</kbd>&nbsp;search &nbsp; <kbd>?</kbd>&nbsp;about';
  drag.y0 = st.y0;
  drag.y1 = st.y1;
  styleWin();
}

let bioLayers: unknown[] = [];
/** The map's layers: the biomass, and Sentinel-2 over it while Space is held; the pair's left: Sentinel-2. */
let s2Main: unknown[] = [], s2Left: unknown[] = [];
/**
 * The imagery's year is the year ([ ]): the embeddings' in each year; in change the imagery's own, which each year
 * opens on. The nearest with a mosaic (EOX's start in 2019).
 */
const mapYear = () => st.year;
const imgYear = () => s2YearFor(mapYear());
function paint() {
  s2Main = st.holding ? s2Layers(imgYear(), "", st.gamma, firstLabel) : [];
  s2Left = st.pair ? s2Layers(imgYear(), "l", st.gamma, firstLabel2) : [];
  drawLayers();
  styleS2card();
}
/** The 10 m pixel's square on the mosaic's grid, as a closed path. */
function pixelRing(lon: number, lat: number): [number, number][] {
  const x0 = X0 + Math.floor((lon - X0) / RES) * RES, y0 = Y0 - Math.floor((Y0 - lat) / RES) * RES;
  return [[x0, y0], [x0 + RES, y0], [x0 + RES, y0 - RES], [x0, y0 - RES], [x0, y0]];
}
/** The picked pixel's boundary, in gold, on both maps. */
function outlines(side: string) {
  const pt = st.picked;
  if (!st.tips || !readable() || !pt) return [];
  return [new PathLayer({ id: `picked${side}`, data: [pixelRing(pt[0], pt[1])], getPath: (d: [number, number][]) => d,
    getColor: [255, 200, 40, 255], widthUnits: "pixels", getWidth: 2, pickable: false })];
}
function drawLayers() {
  overlay.setProps({ layers: [...bioLayers, ...s2Main, ...outlines("")] as never });
  if (map2 && overlay2) overlay2.setProps({ layers: [...s2Left, ...(st.pair ? outlines("l") : [])] as never });
}
/**
 * deck adds its layer groups (the slot under the labels) only while MapLibre reports the style loaded, which is false
 * while any tile loads; the left map is always loading when its layers are set, so it never got one and its imagery
 * drew over the labels. Once a map is idle, any group missing is added by setting the layers again.
 */
const groupId = (l: any) => (l.props.beforeId ? `deck-maplibre-layer-group-before:${l.props.beforeId}` : "deck-maplibre-layer-group-last");
function addMissingGroups(m: maplibregl.Map, layers: unknown[]) {
  if (layers.some((l) => !m.getLayer(groupId(l)))) drawLayers();
}
function styleS2card() {
  s2card.classList.toggle("on", st.holding || st.pair);
  const y = imgYear(), off = y !== mapYear() ? `<div class="keys">no ${mapYear()} mosaic</div>` : "";
  s2card.innerHTML = `<div class="lab">Sentinel-2</div><div class="yr">${y}</div>${off}<div class="keys"><kbd>;</kbd><kbd>'</kbd>&nbsp;gamma ${st.gamma.toFixed(1)}</div>`;
}


function render() {
  const k = curKey();
  $(".year").textContent = String(st.year);
  $(".hy").textContent = keyName(k);
  for (const s of yearsEl.querySelectorAll<HTMLElement>("span")) s.classList.toggle("on", Number(s.dataset.y) === st.year);
  const want: string[] = [];
  const layers = [];
  // zoomed out nothing is on the map and nothing is read
  if (readable()) {
    // the layer left keeps drawing under the new one until the new one's view is in, so a step never flashes empty
    if (st.prev != null && st.prev !== k) {
      want.push(st.prev);
      layers.push(bio.layer({ key: st.prev, visible: true }));
    }
    want.push(k);
    layers.push(
      bio.layer({
        key: k,
        visible: true,
        onTile: () => {
          status();
          readout();
        },
        onViewportLoad: () => {
          st.viewLoaded = true;
          if (st.prev != null) {
            st.prev = null;
            render();
          }
          status();
          readAhead();
        },
      }),
    );
  } else stopAhead();
  bioLayers = layers;
  // the imagery steps with the map's year
  paint();
  keyTicks();
  status();
  readout();
}

function setYear(y: number) {
  if (y === st.year) return;
  st.dir = y > st.year ? 1 : -1;
  if (st.prev == null) st.prev = curKey();
  st.year = y;
  st.viewLoaded = false;
  stopAhead();
  render();
}
function setWindow(a: number, b: number) {
  drag.y0 = a;
  drag.y1 = b;
  styleWin();
  if (a === st.y0 && b === st.y1) return;
  st.dir = a + b > st.y0 + st.y1 ? 1 : -1;
  if (st.prev == null) st.prev = curKey();
  st.y0 = a;
  st.y1 = b;
  store(WIN_KEY, windowKey(a, b));
  st.viewLoaded = false;
  stopAhead();
  render();
}
/** The code and the README (how the map works), on GitHub. */
const REPO_URL = "https://github.com/kentstephen/aef-biomass";
/** What G does, the first time it is pressed (then remembered); a close button or Esc puts it away. */
const gainHelp = document.createElement("div");
gainHelp.className = "help at-glass";
gainHelp.innerHTML = `<button class="x" aria-label="Close">&times;</button>
  <div class="t">Gain taught on lidar</div>
  <p>G switches how the map decides where biomass went <b>up</b> (blue). Loss (orange) does not change.</p>
  <p><b>Off</b> (the default): a pixel is gain where its biomass rose past a floor, 20 to 40 Mg/ha by window length
  and at least a quarter of what was there.</p>
  <p><b>On</b>: a second model, taught on airborne lidar flown twice over the same forest, marks gain only where the
  change looks like canopy that grew. The size is still the biomass rise. It usually draws less, more concentrated
  gain.</p>
  <p>Neither is clearly better: on the lidar places the floor finds a little more regrowth; elsewhere the lidar-taught
  one agrees with Esri's "became trees" about three times as often. Press G again to switch back.</p>
  <p><a href="${REPO_URL}" target="_blank" rel="noopener">github.com/kentstephen/aef-biomass</a></p>`;
document.body.append(gainHelp);
const closeHelp = () => gainHelp.classList.remove("on");
gainHelp.querySelector<HTMLButtonElement>(".x")!.onclick = closeHelp;

/**
 * The first visit's card (Earthwork's welcome): what this is and what to do, in a few lines; ? opens the about card
 * with the rest. Both close on their button, a click outside, or Esc.
 */
function overlay_(cls: string, html: string) {
  const el = document.createElement("div");
  el.className = `at-about ${cls}`;
  el.innerHTML = `<div class="box at-glass">${html}</div>`;
  document.body.append(el);
  el.onclick = (e) => {
    if (e.target === el) el.classList.remove("on");
  };
  el.querySelector<HTMLButtonElement>("button.at-chip")!.onclick = () => el.classList.remove("on");
  return el;
}
const welcome = overlay_("welcome", `<h2>Biomass</h2>
  <p>This map shows how much living plant matter stands on every 10 m of land, each year from 2018 to 2025, and where
  it was lost or gained between two years. A small model reads it in your browser from AlphaEarth (Google DeepMind),
  a yearly summary of each 10 m pixel from satellites, taught on biomass, canopy height and land cover maps.</p>
  <p>It starts in the Sierra Nevada foothills along the Tuolumne River, California. Zoom in to 11 or closer to read it;
  anywhere on land works.</p>
  <ul>
    <li><b>Hover</b> a pixel for its biomass (Mg/ha), canopy height and land cover; in change, how much it changed and between which years.</li>
    <li><kbd>C</kbd> switches between <b>each year</b> and <b>change</b>: orange is loss, blue is gain, faint where steady ground often changes as much.</li>
    <li><kbd>[</kbd> <kbd>]</kbd> step the year, <kbd>B</kbd> blinks between the first and the latest; in change, drag the window of years on the card.</li>
    <li><b>Hold</b> <kbd>Space</kbd> to see the Sentinel-2 imagery of that year; <kbd>P</kbd> pairs it on the left, moving with the map.</li>
    <li><kbd>G</kbd> in change: gain decided by a model taught on airborne lidar instead of a floor.</li>
  </ul>
  <p><small><kbd>?</kbd> has everything else. The code and how it works:
  <a href="${REPO_URL}" target="_blank" rel="noopener">github.com/kentstephen/aef-biomass</a></small></p>
  <div><button class="at-chip">Start</button></div>`);
const about = overlay_("about", `<h2>Biomass</h2>
  <p><b>Biomass</b> here is aboveground living plant matter, dry, in megagrams per hectare (Mg/ha; 1 Mg/ha is 100 g
  per square meter). A small model (the head) turns each 10 m pixel's AlphaEarth embedding for a year into biomass,
  canopy height and a land cover class. It is taught on CTrees biomass, Meta canopy height and land cover, so its
  biomass is mostly the woody kind: trees and shrubs. It runs on your GPU, from zoom 11 in; nothing is precomputed.</p>
  <p><b>Each year</b> shows one year's biomass on a viridis ramp, 0 to 350+ Mg/ha (log scale). <b>Change</b> shows a window
  of years (drag its ends or the span between them on the card): loss in orange where the biomass fell and the
  embedding itself changed, gain in blue where it rose past a floor (20 to 40 Mg/ha by window length and a quarter
  of what was there), as a percent of the pixel's first-year biomass. Faint is change that steady ground of the same
  class often matches; solid is change it rarely does; clear is no change counted. Hovering a changed pixel says
  between which two years it happened. The card's histogram counts the change in view by hectare, and what each
  pixel was and became.</p>
  <p><b>G</b> (change) decides gain with a second model, taught on airborne lidar flown twice over the same forest (New
  Zealand, Estonia, California): gain only where the change looks like canopy that grew. It usually draws less, more
  concentrated gain; on the lidar places the floor finds a little more regrowth, elsewhere the lidar-taught one
  agrees with Esri's "became trees" about three times as often.</p>
  <p><b>What it does not do well yet.</b> Grass and crops get a rough number, not their own mass (the head is taught on
  woody biomass). Tundra reads four to five times too high. Tall tropical forest reads short above about 150 Mg/ha.
  Burned stands with dead trees still standing are hard for it and for lidar. Slow regrowth is small against the
  year-to-year noise.</p>
  <p><small>Keys: C each year or change; [ and ] the year (in change, the imagery's year); B the first year or the latest;
  G gain by lidar, on or off (change); hold Space for the imagery; P pairs the map with the imagery; Q the tooltip
  and picking off and on; L place names; ; and ' the imagery's brightness; F full screen; / search; ? this page;
  Esc closes a card, else folds the main card.</small></p>
  <p><small>AlphaEarth Foundations by Google and Google DeepMind, via Source Cooperative (CC BY 4.0). The head taught on
  CTrees aboveground biomass and Meta and WRI canopy height (CC BY 4.0), with ESA WorldCover and Impact Observatory
  land cover. Imagery: EOxCloudless by EOX IT Services GmbH, 2019 to 2025 (contains modified Copernicus Sentinel data;
  free for non-commercial use); 2018 from the Copernicus Sentinel-2 Global Mosaics (ESA), via TGE Labs on Source
  Cooperative. Search by Photon and Nominatim over OpenStreetMap. Basemap by CARTO, &copy; OpenStreetMap. The code and how
  it works: <a href="${REPO_URL}" target="_blank" rel="noopener">github.com/kentstephen/aef-biomass</a></small></p>
  <div style="margin-top:12px"><button class="at-chip">Close</button></div>`);
if (!stored("aef-biomass-welcome")) welcome.classList.add("on");
store("aef-biomass-welcome", "1");
const overlayOpen = () => welcome.classList.contains("on") || about.classList.contains("on");

/** G: gain by the lidar-taught network or by the two-part floor (change only). */
function setLidarGain(on: boolean) {
  if (!gainHead || on === st.lidarGain) return;
  st.lidarGain = on;
  if (on && !stored("aef-biomass-gain-help")) {
    gainHelp.classList.add("on");
    store("aef-biomass-gain-help", "1");
  }
  store("aef-biomass-gain", on ? "lidar" : "floor");
  st.prev = null;
  st.viewLoaded = false;
  stopAhead();
  styleMode();
  render();
  say(on ? "gain: the lidar-taught network" : "gain: the two-part floor");
}
/**
 * The bands read per pixel (N cycles 16, 32, 64): the layer left keeps drawing under the new one until the new one's view is in.
 * Below the new count's zoom nothing is drawn.
 */
function setBands(b: Bands) {
  if (b === st.bands) return;
  if (st.prev == null && readable()) st.prev = curKey();
  st.bands = b;
  store("aef-biomass-bands", String(b));
  if (!readable()) st.prev = null;
  st.viewLoaded = false;
  stopAhead();
  for (const el of card.querySelectorAll<HTMLButtonElement>(".bandsel button")) el.classList.toggle("on", Number(el.dataset.b) === b);
  wasReadable = readable();
  setModeNote();
  render();
}
function setMode(m: Mode) {
  if (m === st.mode) return;
  st.mode = m;
  store(MODE_KEY, m);
  // the other mode's layer is not kept under: viridis showing through the clear no change would read as change
  st.prev = null;
  st.viewLoaded = false;
  stopAhead();
  styleMode();
  render();
}
/** [ and ]: the year: the embeddings' and the imagery's together (each year), the imagery's alone (change). */
function step(d: number) {
  const y = YEARS[YEARS.indexOf(st.year) + d];
  if (y === undefined) return;
  goYear(y);
}
/** B: the latest year, then the first, back and forth (from any other year, the latest first), as [ ] step. */
function blink() {
  const a = YEARS[0]!, b = YEARS[YEARS.length - 1]!;
  goYear(st.year === b ? a : b);
}
function goYear(y: number) {
  if (st.mode === "year") return setYear(y);
  // in change the window stays: the year steps the imagery, and each year opens on it (C)
  st.year = y;
  $(".year").textContent = String(y);
  for (const s of yearsEl.querySelectorAll<HTMLElement>("span")) s.classList.toggle("on", Number(s.dataset.y) === y);
  paint();
}

/** The basemap's labels (its symbol layers) shown or hidden, as st.labels says. */
function styleLabels(m: maplibregl.Map) {
  if (!m.isStyleLoaded() && !m.getStyle()?.layers) return;
  for (const l of m.getStyle().layers) if (l.type === "symbol") m.setLayoutProperty(l.id, "visibility", st.labels ? "visible" : "none");
}
function toggleLabels() {
  st.labels = !st.labels;
  store("aef-biomass-labels", st.labels ? "1" : "0");
  styleLabels(map);
  if (map2) styleLabels(map2);
}

function toggleFull() {
  if (document.fullscreenElement) void document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => say("full screen is not allowed here"));
}
document.addEventListener("fullscreenchange", () => setTimeout(() => map.resize(), 30));

// on the window, before the map or anything on the page sees the key (capture)
window.addEventListener("keydown", (e) => {
  // typing in the search keeps its keys; a slider never does
  if ((e.target instanceof HTMLInputElement && e.target.type !== "range") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === "Space") {
    if (!e.repeat) beginHold();
  }
  // ; and ': the imagery's gamma (Earthwork's brightness keys)
  else if (e.code === "Semicolon" || e.code === "Quote") setGamma(st.gamma + (e.code === "Quote" ? 0.1 : -0.1));
  // Q: picking off and on (the tooltip and the picked pixel's boundary)
  else if (e.code === "KeyQ") {
    st.tips = !st.tips;
    store("aef-biomass-tips", st.tips ? "1" : "0");
    readout();
    drawLayers();
  }
  // Esc folds the card
  else if (e.code === "Escape") {
    if (overlayOpen()) {
      welcome.classList.remove("on");
      about.classList.remove("on");
    } else if (gainHelp.classList.contains("on")) closeHelp();
    else fold(true);
  }
  else if (e.key === "?") {
    welcome.classList.remove("on");
    about.classList.toggle("on");
  }
  // [ and ]: the year, the embeddings' and the imagery's together (Earthwork's); by the key's code, so Shift or a
  // keyboard layout does not change them
  else if (e.code === "BracketLeft") step(-1);
  else if (e.code === "BracketRight") step(1);
  else if (e.code === "KeyB" && !e.shiftKey) blink();
  else if (e.code === "KeyP" && !e.shiftKey) setPair(!st.pair);
  else if (e.code === "KeyC" && !e.shiftKey) setMode(st.mode === "year" ? "change" : "year");
  else if (e.code === "KeyG" && !e.shiftKey && st.mode === "change" && gainHead) setLidarGain(!st.lidarGain);
  // N: the bands read per pixel, 16, 32, 64 and round again
  else if (e.code === "KeyN" && !e.shiftKey) setBands(BANDS[(BANDS.indexOf(st.bands) + 1) % BANDS.length]!);
  else if (e.code === "KeyF") toggleFull();
  else if (e.code === "KeyL" && !e.shiftKey) toggleLabels();
  else if (e.key === "/") search.focus();
  else return;
  e.preventDefault();
}, { capture: true });
for (const b of card.querySelectorAll<HTMLButtonElement>(".step")) b.onclick = () => step(Number(b.dataset.d));
for (const b of card.querySelectorAll<HTMLButtonElement>(".mode button")) b.onclick = () => setMode(b.dataset.m as Mode);
for (const b of card.querySelectorAll<HTMLButtonElement>(".bandsel button")) {
  b.classList.toggle("on", Number(b.dataset.b) === st.bands);
  b.onclick = () => setBands(Number(b.dataset.b) as Bands);
}
yearsEl.onclick = (e) => {
  const y = Number((e.target as HTMLElement).dataset.y);
  if (y) setYear(y);
};

// hovering reads the pixel (tooltip and card) but draws nothing on the map
const hoverAt = (e: maplibregl.MapMouseEvent) => {
  st.hover = [e.lngLat.lng, e.lngLat.lat];
  st.hoverPt = [e.originalEvent.clientX, e.originalEvent.clientY];
  readout();
};
const hoverOff = () => {
  st.hover = null;
  st.hoverPt = null;
  readout();
};
// a click outlines the pixel's boundary, on both maps; with one outlined, a click anywhere lets it go
const pickAt = (e: maplibregl.MapMouseEvent) => {
  if (!st.tips) return;
  st.picked = st.picked ? null : [e.lngLat.lng, e.lngLat.lat];
  readout();
  drawLayers();
};
map.on("click", pickAt);
map.on("mousemove", hoverAt);
map.on("mouseout", hoverOff);
map.on("movestart", () => {
  stopAhead();
  st.viewLoaded = false;
});
let wasReadable = readable();
map.on("zoom", () => {
  // crossing the zoom limit puts the layers on or takes them off at once
  if (readable() !== wasReadable) {
    wasReadable = readable();
    render();
  }
});
map.on("moveend", () => status());


styleMode();
await mapLoaded;
firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
styleLabels(map);
render();
map.on("idle", () => addMissingGroups(map, [...bioLayers, ...s2Main, ...outlines("")]));
// dev only: lets a headless probe read both maps' layer order and the head's values
if (import.meta.env.DEV) Object.assign(window, { __pair: { map, map2: () => map2, bio, st } });
