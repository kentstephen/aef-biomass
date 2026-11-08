import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { PathLayer } from "@deck.gl/layers";
import { MapLibreOverlay as MapboxOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { MIN_ZOOM, RES, X0, Y0 } from "./aef";
import { createBiomass, GAIN_END, LOSS_END, windowKey, yearKey } from "./biomass";
import type { BiomassHead } from "./head";
import { CHANGE_50, cssGradient, VIRIDIS_50 } from "./ramp";
import { S2_CREDIT, S2Y, s2Layers, s2YearFor } from "./s2";
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

// the change key's ticks, percent (loss of the pixel's own biomass, gain of its 100 m cell's)
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

const head: BiomassHead = await (await fetch(`${import.meta.env.BASE_URL}models/biomass_aef.json`)).json();
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
  viewDirty: true,
  // the year's view all in (deck.gl's onViewportLoad): the bar goes away even where a chunk has no data
  viewLoaded: false,
  on: [] as string[],
  ahead: null as AbortController | null,
  aheadKey: null as string | null,
  // Sentinel-2: held over the whole map (Space, one map only), or the pair's left side (P)
  holding: false,
  pair: false,
  imgYear: 2024,
  // picking (Q, Earthwork's): the tooltip by the pointer and the gold outlines; remembered. Where the pointer is
  // on screen; the pixel clicked
  tips: stored("aef-biomass-tips") !== "0",
  hoverPt: null as [number, number] | null,
  picked: null as [number, number] | null,
  // the imagery's gamma (; and '), remembered
  gamma: Number(stored("aef-biomass-s2-gamma")) || 1,
};
/** The layer shown: the year, or the window. */
const curKey = () => (st.mode === "year" ? yearKey(st.year) : windowKey(st.y0, st.y1));
/** A layer key as words: "2025", or "2018 to 2025". */
const keyName = (k: string) => k.replace("-", " to ");

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
  <div class="yearrow">
    <button class="step" data-d="-1" title="Earlier year (-)">&lsaquo;</button>
    <div class="year"></div>
    <button class="step" data-d="1" title="Later year (=)">&rsaquo;</button>
    <div class="years"></div>
  </div>
  <div class="at-row winrow">
    <span class="at-lab">Years</span>
    <span class="at-win"><span class="trk"></span><span class="spn" title="drag to move the window, its length kept"></span><span class="tks"></span></span>
    <span class="at-wtxt"></span>
  </div>
  <div class="key"><div class="bar"></div><div class="ticks"></div><div class="unit"></div></div>
  <div class="readout">&nbsp;</div>
  <div class="zoom"></div>
  <div class="status"></div>
  <div class="keys"></div>
  <div class="credits">AlphaEarth &copy; Google, Google DeepMind, via Source Cooperative &middot; head taught on CTrees AGB and Meta canopy height (CC BY 4.0) &middot; ${S2_CREDIT} &middot; search by Photon and Nominatim over OpenStreetMap &middot; &copy; CARTO &copy; OpenStreetMap</div>
  </div>`;
// the pixel's tooltip, by the pointer
const tip = document.createElement("div");
tip.className = "tip at-glass";
document.body.append(bar, tag, top, s2card, tip);
const $ = <T extends HTMLElement>(s: string) => card.querySelector(s) as T;
// the search, while not in use, as wide as the card under it (Earthwork's fitSearch); in use, full width
const fitSearch = () => {
  const folded = card.classList.contains("collapsed");
  if (folded && card.offsetWidth) search.el.style.setProperty("--yw", `${card.offsetWidth}px`);
  else search.el.style.removeProperty("--yw");
};
const fold = (on: boolean) => {
  card.classList.toggle("collapsed", on);
  requestAnimationFrame(fitSearch);
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
    // the left opens on the map's year (the nearest with a mosaic); [ and ] step it from there
    st.imgYear = s2YearFor(st.mode === "year" ? st.year : st.y1);
    boot2();
    map2!.jumpTo({ center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing() });
  }
  render();
}

// ---- hold Space: the imagery over the whole map (one map only; paired, it is already on the left). It opens on
// the map's year (the nearest with a mosaic); while it shows (held, or paired), scroll or [ ] steps its year
function beginHold() {
  if (st.holding || st.pair) return;
  st.holding = true;
  st.imgYear = s2YearFor(st.mode === "year" ? st.year : st.y1);
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
function stepImg(d: number) {
  const i = S2Y.indexOf(st.imgYear);
  const y = S2Y[Math.max(0, Math.min(S2Y.length - 1, i + d))]!;
  if (y === st.imgYear) return;
  st.imgYear = y;
  paint();
}
let wheelAcc = 0, lastStep = 0;
document.addEventListener(
  "wheel",
  (e) => {
    if (!st.holding) return;
    e.preventDefault();
    e.stopPropagation();
    wheelAcc += e.deltaY;
    const now = performance.now();
    if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) {
      stepImg(wheelAcc > 0 ? 1 : -1);
      wheelAcc = 0;
      lastStep = now;
    }
  },
  { capture: true, passive: false },
);
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") endHold();
});
// a lost window (Cmd-Tab with Space held) gets no keyup
window.addEventListener("blur", () => endHold());

const search = makeSearch(map, MIN_ZOOM);
top.append(search.el, card);
// the search follows the card's width as it changes (the year filled in, fonts in)
new ResizeObserver(fitSearch).observe(card);
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
const bio = await createBiomass(head, () => firstLabel);
let modeNote = "";
bio.pool.onMode = () => {
  const ms = bio.pool.modes;
  const gpu = ms.filter((m) => m.mode === "gpu").length;
  modeNote = gpu ? `head on WebGPU (${gpu} of ${ms.length} workers)` : "head on the CPU (no WebGPU)";
  status();
};

const bbox = (): [number, number, number, number] => {
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
};
const readable = () => map.getZoom() >= MIN_ZOOM;

function status() {
  const z = map.getZoom();
  const zEl = $(".zoom");
  zEl.classList.toggle("out", !readable());
  zEl.innerHTML = readable()
    ? `Zoom ${z.toFixed(1)} &middot; AlphaEarth is read from zoom ${MIN_ZOOM}`
    : `Zoom ${z.toFixed(1)}: zoom in to ${MIN_ZOOM} to read AlphaEarth. It is 10 m with no coarser levels, so a wider view would read too much.`;
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
  say(`${loaded} of ${wanted} chunks for ${keyName(k)}${ahead}${modeNote ? ` · ${modeNote}` : ""}`);
}

function keyTicks() {
  const s = bio.stretch;
  const ticks = $(".ticks");
  if (st.mode === "change") {
    ticks.innerHTML = CH_TICKS.map((v) => {
      const t = v < 0 ? 0.5 - (0.5 * -v) / LOSS_END : 0.5 + (0.5 * v) / GAIN_END;
      return `<span style="left:${t * 100}%">${v > 0 ? "+" : ""}${v}${v ? "%" : ""}</span>`;
    }).join("");
    return;
  }
  const fmt = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
  if (s.rank) {
    ticks.innerHTML = [0, 0.25, 0.5, 0.75, 1].map((t) => `<span style="left:${t * 100}%">${fmt(bio.valueAt(t))}</span>`).join("");
  } else {
    ticks.innerHTML =
      TICKS.map((v) => {
        const t = Math.log1p(v) / s.hi;
        // a tick near the end would sit on the end's own label
        return t <= 0.9 ? `<span style="left:${t * 100}%">${v}</span>` : "";
      }).join("") + `<span style="left:100%">${Math.round(Math.expm1(s.hi))}+</span>`;
  }
  $(".rk").textContent = s.rank ? "rank stretch on (view)" : "rank stretch";
}

/** The pixel under the pointer: the card's readout line, and the tooltip by the pointer (Q hides it). */
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
  let line = "", big = "", sub = "";
  if (st.mode === "change") {
    const c = bio.changeAtPoint(pt[0], pt[1], curKey());
    if (!c) return off();
    const sg = (v: number, d: number) => `${v > 0 ? "+" : ""}${v.toFixed(d)}`;
    if (c.mg === 0) {
      big = "No change counted";
      line = `no change counted &middot; ${at}`;
    } else {
      big = `<b>${sg(c.mg, Math.abs(c.mg) >= 10 ? 0 : 1)}</b> Mg/ha<div class="m">${sg(c.pct, 0)}%</div>`;
      line = `<b>${sg(c.mg, Math.abs(c.mg) >= 10 ? 0 : 1)}</b> Mg/ha (${sg(c.pct, 0)}%) &middot; ${at}`;
    }
    sub = `${c.mg < 0 ? "<div>loss, this pixel</div>" : c.mg > 0 ? "<div>gain, its 100 m cell</div>" : ""}<div>${keyName(curKey())}</div>`;
  } else {
    const v = bio.valueAtPoint(pt[0], pt[1], st.year);
    if (!v) return off();
    const a = v.agb >= 10 ? v.agb.toFixed(0) : v.agb.toFixed(1);
    big = `<b>${a}</b> Mg/ha<div class="m">canopy ${v.height.toFixed(1)} m</div>`;
    line = `<b>${a}</b> Mg/ha &middot; canopy ${v.height.toFixed(1)} m &middot; ${at}`;
    sub = `<div>${st.year}</div>`;
  }
  el.innerHTML = line;
  if (!st.tips || !st.hoverPt || !st.hover) return void tip.classList.remove("on");
  tip.innerHTML = `<div class="v">${big}</div><div class="s">${sub}<div>${pt[1].toFixed(4)},</div><div>${pt[0].toFixed(4)}</div></div>`;
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
    if (y !== undefined) next.push(yearKey(y));
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
    ? "AlphaEarth: loss where the embedding itself moved, gain per 100 m cell above a floor"
    : "AlphaEarth, one year's embedding";
  $(".unit").textContent = ch
    ? "loss: percent of the pixel's own biomass the first year; gain: of its 100 m cell's. Clear: no change counted"
    : "Mg/ha aboveground, per 10 m pixel";
  $(".keys").innerHTML = ch
    ? "<kbd>[</kbd><kbd>]</kbd> imagery year &nbsp; <kbd>C</kbd> each year &nbsp; <kbd>Space</kbd> imagery &nbsp; <kbd>P</kbd> pair &nbsp; <kbd>Q</kbd> picking &nbsp; <kbd>F</kbd> full screen &nbsp; <kbd>/</kbd> search"
    : '<kbd>-</kbd><kbd>=</kbd> year &nbsp; <kbd>[</kbd><kbd>]</kbd> imagery year &nbsp; <kbd>R</kbd> <span class="rk">rank stretch</span> &nbsp; <kbd>C</kbd> change &nbsp; <kbd>Space</kbd> imagery &nbsp; <kbd>P</kbd> pair &nbsp; <kbd>Q</kbd> picking &nbsp; <kbd>F</kbd> full screen &nbsp; <kbd>/</kbd> search';
  drag.y0 = st.y0;
  drag.y1 = st.y1;
  styleWin();
}

let bioLayers: unknown[] = [];
/** The map's layers: the biomass, and Sentinel-2 over it while Space is held; the pair's left: Sentinel-2. */
let s2Main: unknown[] = [], s2Left: unknown[] = [];
function paint() {
  s2Main = st.holding ? s2Layers(st.imgYear, "", st.gamma, firstLabel) : [];
  s2Left = st.pair ? s2Layers(st.imgYear, "l", st.gamma, firstLabel2) : [];
  drawLayers();
  styleS2card();
}
/** The 10 m pixel's square on the mosaic's grid, as a closed path. */
function pixelRing(lon: number, lat: number): [number, number][] {
  const x0 = X0 + Math.floor((lon - X0) / RES) * RES, y0 = Y0 - Math.floor((Y0 - lat) / RES) * RES;
  return [[x0, y0], [x0 + RES, y0], [x0 + RES, y0 - RES], [x0, y0 - RES], [x0, y0]];
}
/** Earthwork's outlines, in gold: the pixel under the pointer (thin) and the one picked (thick), on both maps. */
function outlines(side: string) {
  if (!st.tips || !readable()) return [];
  const out = [];
  for (const [id, pt, w] of [["hover", st.hover, 1.5], ["picked", st.picked, 3]] as const) {
    if (!pt) continue;
    out.push(new PathLayer({ id: `${id}${side}`, data: [pixelRing(pt[0], pt[1])], getPath: (d: [number, number][]) => d,
      getColor: [255, 200, 40, 255], widthUnits: "pixels", getWidth: w, pickable: false }));
  }
  return out;
}
function drawLayers() {
  overlay.setProps({ layers: [...bioLayers, ...s2Main, ...outlines("")] as never });
  if (map2 && overlay2) overlay2.setProps({ layers: [...s2Left, ...(st.pair ? outlines("l") : [])] as never });
}
function styleS2card() {
  s2card.classList.toggle("on", st.holding || st.pair);
  s2card.innerHTML = `<div class="lab">Sentinel-2</div><div class="yr">${st.imgYear}</div><div class="keys"><kbd>[</kbd><kbd>]</kbd> year &nbsp; <kbd>;</kbd><kbd>'</kbd> gamma ${st.gamma.toFixed(1)}</div>`;
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
          if (st.mode === "year" && st.viewDirty && bio.rebuildCdf(bbox(), st.year)) {
            st.viewDirty = false;
            if (bio.stretch.rank) render();
            keyTicks();
          }
          status();
          readAhead();
        },
      }),
    );
  } else stopAhead();
  bioLayers = layers;
  drawLayers();
  for (const o of st.on) if (!want.includes(o)) bio.release(o);
  st.on = want;
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
function setMode(m: Mode) {
  if (m === st.mode) return;
  st.mode = m;
  store(MODE_KEY, m);
  // the other mode's layer is not kept under: viridis showing through the clear no change would read as change
  st.prev = null;
  st.viewLoaded = false;
  st.viewDirty = true;
  stopAhead();
  styleMode();
  render();
}
/** - and =: the embeddings' year (each year). */
function step(d: number) {
  if (st.mode !== "year") return;
  const y = YEARS[YEARS.indexOf(st.year) + d];
  if (y !== undefined) setYear(y);
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
  // matched by the key's code, so Shift or a keyboard layout does not change what [ and ] do
  if (e.code === "Space") {
    if (!e.repeat) beginHold();
  } else if (e.code === "BracketLeft" || e.key === "[" || e.key === "{") stepImg(-1);
  else if (e.code === "BracketRight" || e.key === "]" || e.key === "}") stepImg(1);
  // ; and ': the imagery's gamma (Earthwork's brightness keys)
  else if (e.code === "Semicolon" || e.code === "Quote") setGamma(st.gamma + (e.code === "Quote" ? 0.1 : -0.1));
  // Q: picking off and on (the tooltip and the gold outlines, Earthwork's)
  else if (e.code === "KeyQ") {
    st.tips = !st.tips;
    store("aef-biomass-tips", st.tips ? "1" : "0");
    readout();
    drawLayers();
  }
  // Esc folds the card
  else if (e.code === "Escape") fold(true);
  // - and =: the embeddings' year
  else if (e.code === "Minus" || e.key === "-" || e.key === "_") step(-1);
  else if (e.code === "Equal" || e.key === "=" || e.key === "+") step(1);
  else if (e.code === "KeyP" && !e.shiftKey) setPair(!st.pair);
  else if (e.code === "KeyR" && !e.shiftKey && st.mode === "year") {
    bio.setRank(!bio.stretch.rank);
    render();
  } else if (e.code === "KeyC" && !e.shiftKey) setMode(st.mode === "year" ? "change" : "year");
  else if (e.code === "KeyF") toggleFull();
  else if (e.key === "/") search.focus();
  else return;
  e.preventDefault();
}, { capture: true });
for (const b of card.querySelectorAll<HTMLButtonElement>(".step")) b.onclick = () => step(Number(b.dataset.d));
for (const b of card.querySelectorAll<HTMLButtonElement>(".mode button")) b.onclick = () => setMode(b.dataset.m as Mode);
yearsEl.onclick = (e) => {
  const y = Number((e.target as HTMLElement).dataset.y);
  if (y) setYear(y);
};

const samePixel = (a: [number, number] | null, b: [number, number] | null) =>
  !!a && !!b && Math.floor((a[0] - X0) / RES) === Math.floor((b[0] - X0) / RES) && Math.floor((Y0 - a[1]) / RES) === Math.floor((Y0 - b[1]) / RES);
const hoverAt = (e: maplibregl.MapMouseEvent) => {
  const was = st.hover;
  st.hover = [e.lngLat.lng, e.lngLat.lat];
  st.hoverPt = [e.originalEvent.clientX, e.originalEvent.clientY];
  readout();
  if (st.tips && !samePixel(was, st.hover)) drawLayers();
};
const hoverOff = () => {
  st.hover = null;
  st.hoverPt = null;
  readout();
  if (st.tips) drawLayers();
};
// a click picks the pixel (gold, thick); the same pixel again lets it go
const pickAt = (e: maplibregl.MapMouseEvent) => {
  if (!st.tips) return;
  const p: [number, number] = [e.lngLat.lng, e.lngLat.lat];
  st.picked = samePixel(st.picked, p) ? null : p;
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
map.on("moveend", () => {
  if (st.mode !== "year") return void status();
  st.viewDirty = !bio.rebuildCdf(bbox(), st.year);
  if (!st.viewDirty && bio.stretch.rank) render();
  else status();
});


styleMode();
await mapLoaded;
firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
render();
