import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay as MapboxOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { MIN_ZOOM } from "./aef";
import { createBiomass } from "./biomass";
import type { BiomassHead } from "./head";
import { cssGradient, VIRIDIS_50 } from "./ramp";
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

const head: BiomassHead = await (await fetch(`${import.meta.env.BASE_URL}models/biomass_aef.json`)).json();
const YEARS = head.years;
const st = {
  year: YEARS[YEARS.length - 1]!,
  prev: null as number | null,
  dir: -1,
  hover: null as [number, number] | null,
  viewDirty: true,
  // the year's view all in (deck.gl's onViewportLoad): the bar goes away even where a chunk has no data
  viewLoaded: false,
  on: [] as number[],
  ahead: null as AbortController | null,
  aheadYear: null as number | null,
};

// ---- the page: the map fills the window; search and the card down the top left, the loading bar along the top
const mapEl = document.getElementById("map")!;
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
  <div class="sub">AlphaEarth, one year's embedding</div>
  <div class="yearrow">
    <button class="step" data-d="-1" title="Earlier year ([)">&lsaquo;</button>
    <div class="year"></div>
    <button class="step" data-d="1" title="Later year (])">&rsaquo;</button>
    <div class="years"></div>
  </div>
  <div class="key"><div class="bar"></div><div class="ticks"></div><div class="unit">Mg/ha aboveground, per 10 m pixel</div></div>
  <div class="readout">&nbsp;</div>
  <div class="zoom"></div>
  <div class="status"></div>
  <div class="keys"><kbd>[</kbd><kbd>]</kbd> year &nbsp; <kbd>R</kbd> <span class="rk">rank stretch</span> &nbsp; <kbd>F</kbd> full screen &nbsp; <kbd>/</kbd> search</div>
  <div class="credits">AlphaEarth &copy; Google, Google DeepMind, via Source Cooperative &middot; head taught on CTrees AGB and Meta canopy height (CC BY 4.0) &middot; search by Photon and Nominatim over OpenStreetMap &middot; &copy; CARTO &copy; OpenStreetMap</div>
  </div>`;
document.body.append(bar, tag, top);
const $ = <T extends HTMLElement>(s: string) => card.querySelector(s) as T;
$(".bar").style.background = cssGradient(VIRIDIS_50);
// folded, the card is its header: the short title, the ramp on the map and the year (Earthwork's card fold)
$(".at-ramp").style.background = cssGradient(VIRIDIS_50);
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

const map = new maplibregl.Map({ container: mapEl, style: STYLE, center: HOME.center, zoom: HOME.zoom, attributionControl: false, hash: true, ...FLAT });
map.touchZoomRotate.disableRotation();
// caught now: the head and the mosaic open while the style loads, and its load event can fire before they are done
const mapLoaded = new Promise<void>((ok) => map.once("load", () => ok()));
const overlay = new MapboxOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
new ResizeObserver(() => map.resize()).observe(mapEl);

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
  const { loaded, wanted } = bio.loadedIn(bbox(), st.year);
  const loading = !st.viewLoaded && wanted > 0;
  bar.classList.toggle("on", loading);
  (bar.firstElementChild as HTMLElement).style.width = wanted ? `${(100 * loaded) / wanted}%` : "0";
  tag.classList.toggle("on", loading);
  tag.innerHTML = `Reading ${st.year}<span>${loaded} of ${wanted} chunks</span>`;
  let ahead = "";
  if (st.aheadYear != null) {
    const h = bio.inHand(bbox(), st.aheadYear);
    ahead = h.inHand < h.wanted ? ` · reading ${st.aheadYear} ahead, ${h.inHand} of ${h.wanted}` : ` · ${st.aheadYear} read ahead`;
  }
  say(`${loaded} of ${wanted} chunks for ${st.year}${ahead}${modeNote ? ` · ${modeNote}` : ""}`);
}

function keyTicks() {
  const s = bio.stretch;
  const ticks = $(".ticks");
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

function readout() {
  const el = $(".readout");
  if (!st.hover) return void (el.innerHTML = "&nbsp;");
  const v = bio.valueAtPoint(st.hover[0], st.hover[1], st.year);
  el.innerHTML = v
    ? `<b>${v.agb >= 10 ? v.agb.toFixed(0) : v.agb.toFixed(1)}</b> Mg/ha &middot; canopy ${v.height.toFixed(1)} m &middot; ${st.hover[1].toFixed(4)}, ${st.hover[0].toFixed(4)}`
    : "&nbsp;";
}

/** Stop reading ahead (a move, a step, zoomed out). */
function stopAhead() {
  st.ahead?.abort();
  st.ahead = null;
  st.aheadYear = null;
}

/** With the year's view in: read the next year in the stepping direction, then the one the other way. */
function readAhead() {
  stopAhead();
  if (!readable()) return;
  const ac = (st.ahead = new AbortController());
  const i = YEARS.indexOf(st.year);
  const next = [YEARS[i + st.dir], YEARS[i - st.dir]].filter((y): y is number => y !== undefined);
  void (async () => {
    for (const y of next) {
      if (ac.signal.aborted) return;
      st.aheadYear = y;
      status();
      await bio.prefetch(y, bbox(), ac.signal, status);
    }
    if (st.ahead === ac) st.aheadYear = null;
    status();
  })();
}

function render() {
  $(".year").textContent = String(st.year);
  $(".hy").textContent = String(st.year);
  for (const s of yearsEl.querySelectorAll<HTMLElement>("span")) s.classList.toggle("on", Number(s.dataset.y) === st.year);
  const want: number[] = [];
  const layers = [];
  // zoomed out nothing is on the map and nothing is read
  if (readable()) {
    // the year left keeps drawing under the new one until the new one's view is in, so a step never flashes empty
    if (st.prev != null && st.prev !== st.year) {
      want.push(st.prev);
      layers.push(bio.layer({ year: st.prev, visible: true }));
    }
    want.push(st.year);
    layers.push(
      bio.layer({
        year: st.year,
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
          if (st.viewDirty && bio.rebuildCdf(bbox(), st.year)) {
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
  overlay.setProps({ layers });
  for (const y of st.on) if (!want.includes(y)) bio.release(y);
  st.on = want;
  keyTicks();
  status();
  readout();
}

function setYear(y: number) {
  if (y === st.year) return;
  st.dir = y > st.year ? 1 : -1;
  if (st.prev == null) st.prev = st.year;
  st.year = y;
  st.viewLoaded = false;
  stopAhead();
  render();
}
function step(d: number) {
  const y = YEARS[YEARS.indexOf(st.year) + d];
  if (y !== undefined) setYear(y);
}

function toggleFull() {
  if (document.fullscreenElement) void document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => say("full screen is not allowed here"));
}
document.addEventListener("fullscreenchange", () => setTimeout(() => map.resize(), 30));

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  // matched by the key's code, so Shift or a keyboard layout does not change what [ and ] do
  if (e.code === "BracketLeft") step(-1);
  else if (e.code === "BracketRight") step(1);
  else if (e.code === "KeyR" && !e.shiftKey) {
    bio.setRank(!bio.stretch.rank);
    render();
  } else if (e.code === "KeyF") toggleFull();
  else if (e.key === "/") search.focus();
  else return;
  e.preventDefault();
});
for (const b of card.querySelectorAll<HTMLButtonElement>(".step")) b.onclick = () => step(Number(b.dataset.d));
yearsEl.onclick = (e) => {
  const y = Number((e.target as HTMLElement).dataset.y);
  if (y) setYear(y);
};

map.on("mousemove", (e) => {
  st.hover = [e.lngLat.lng, e.lngLat.lat];
  readout();
});
map.on("mouseout", () => {
  st.hover = null;
  readout();
});
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
  st.viewDirty = !bio.rebuildCdf(bbox(), st.year);
  if (!st.viewDirty && bio.stretch.rank) render();
  else status();
});


await mapLoaded;
firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
render();
