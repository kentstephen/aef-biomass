import type * as maplibregl from "maplibre-gl";

/**
 * Place search, from aef-tessera-earthwork's ui.js: Photon as you type; when it is slow or finds nothing, Enter asks
 * Nominatim (whose usage policy allows no search-as-you-type and at most one request a second, so it runs on Enter
 * only). A place is flown to no wider than `minZoom`, where the map is read.
 */
const PHOTON = "https://photon.komoot.io/api/", NOMINATIM = "https://nominatim.openstreetmap.org/search";
const PHOTON_MS = 6000, PHOTON_ENTER_MS = 2500;
const ICON =
  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>';

type Hit = { geometry: { coordinates: [number, number] }; properties: Record<string, unknown> & { extent?: number[] } };

const el_ = (tag: string, cls = "", html = "") => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

export function makeSearch(map: maplibregl.Map, minZoom: number) {
  const box = el_("div", "at-search at-glass", ICON);
  const gc = el_("input") as HTMLInputElement;
  gc.type = "search";
  gc.placeholder = "Search";
  const hits = el_("div", "at-hits at-glass");
  const key = el_("kbd", "", "/");
  key.title = "/ searches";
  box.append(gc, key, hits);

  let gcHits: Hit[] = [], gcSel = -1, gcTimer: ReturnType<typeof setTimeout> | undefined, gcSeq = 0;
  let gcBusy: string | null = null, gcAbort: AbortController | null = null, nomLast = 0;
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  const hitName = (f: Hit) => {
    const p = f.properties;
    return [str(p.name), str(p.street) && !p.name ? str(p.street) : null, str(p.city) && p.city !== p.name ? str(p.city) : null, str(p.state), str(p.country)]
      .filter(Boolean)
      .join(", ");
  };
  const hitKind = (f: Hit) => [str(f.properties.osm_value), str(f.properties.type)].filter((x) => x && x !== "yes").join(", ");
  const gcHide = () => {
    hits.style.display = "none";
    hits.replaceChildren();
    gcSel = -1;
  };
  const gcShow = () => {
    hits.replaceChildren();
    if (!gcHits.length) return gcHide();
    gcHits.forEach((f, i) => {
      const r = el_("div", "at-hit" + (i === gcSel ? " sel" : ""));
      r.textContent = hitName(f);
      const k = el_("small");
      k.textContent = hitKind(f);
      r.appendChild(k);
      r.onmousedown = (e) => {
        e.preventDefault();
        gcFly(f);
      };
      r.onmouseenter = () => {
        gcSel = i;
        gcShow();
      };
      hits.appendChild(r);
    });
    hits.style.display = "block";
  };
  // one gray line in the list in place of hits: searching, or why there are none
  const gcStatus = (t: string) => {
    hits.replaceChildren();
    gcSel = -1;
    const r = el_("div", "at-hit wait");
    r.textContent = t;
    hits.appendChild(r);
    hits.style.display = "block";
  };
  const photon = async (q: string, ms: number): Promise<Hit[]> => {
    const params = new URLSearchParams({ q, limit: "6", lang: "en" });
    const c = map.getCenter();
    params.set("lon", c.lng.toFixed(4));
    params.set("lat", c.lat.toFixed(4));
    gcAbort?.abort();
    const ac = (gcAbort = new AbortController()), t = setTimeout(() => ac.abort(), ms);
    try {
      const d = await (await fetch(PHOTON + "?" + params.toString(), { signal: ac.signal })).json();
      return (d.features || []).filter((f: Hit) => f.geometry && f.geometry.coordinates);
    } finally {
      clearTimeout(t);
    }
  };
  // Nominatim's results in Photon's shape (extent is [west, north, east, south]), so the list and the fly-to read both alike
  const nominatim = async (q: string): Promise<Hit[]> => {
    const wait = nomLast + 1100 - Date.now();
    if (wait > 0) await new Promise((ok) => setTimeout(ok, wait));
    nomLast = Date.now();
    const params = new URLSearchParams({ q, format: "jsonv2", addressdetails: "1", limit: "6", "accept-language": "en" });
    const b = map.getBounds();
    params.set("viewbox", [b.getWest(), b.getNorth(), b.getEast(), b.getSouth()].map((v) => v.toFixed(4)).join(","));
    const d = await (await fetch(NOMINATIM + "?" + params.toString())).json();
    return (d || []).map((h: Record<string, unknown>) => {
      const a = (h.address || {}) as Record<string, string>, bb = ((h.boundingbox || []) as string[]).map(Number);
      return {
        geometry: { coordinates: [Number(h.lon), Number(h.lat)] },
        properties: {
          name: h.name || String(h.display_name || "").split(",")[0],
          city: a.city || a.town || a.village,
          state: a.state,
          country: a.country,
          osm_value: h.type,
          type: "via Nominatim",
          extent: bb.length === 4 ? [bb[2], bb[1], bb[3], bb[0]] : undefined,
        },
      };
    });
  };
  // enter: Enter was pressed, so Photon gets a shorter wait and Nominatim follows when it fails
  const gcAsk = async (enter = false) => {
    const q = gc.value.trim();
    if (q.length < 2) {
      gcHits = [];
      gcBusy = null;
      return gcHide();
    }
    const s = ++gcSeq;
    // Enter while Photon is still on this same query: it has been waited on already
    const skipPhoton = enter && gcBusy === q;
    if (skipPhoton) {
      gcAbort?.abort();
      gcBusy = null;
    }
    gcStatus("searching…");
    let found: Hit[] = [], why = "";
    if (!skipPhoton) {
      gcBusy = q;
      try {
        found = await photon(q, enter ? PHOTON_ENTER_MS : PHOTON_MS);
      } catch (e) {
        why = (e as Error).name === "AbortError" ? "slow" : "down";
      }
      if (s !== gcSeq) return;
      gcBusy = null;
    }
    if (!found.length && enter) {
      gcStatus("searching Nominatim…");
      try {
        found = await nominatim(q);
      } catch {
        why = "down";
      }
      if (s !== gcSeq) return;
    }
    gcHits = found;
    gcSel = found.length ? 0 : -1;
    if (found.length) gcShow();
    else if (enter) gcStatus(why ? "search is not answering, try again in a moment" : "no match");
    else
      gcStatus(
        why === "slow"
          ? "Photon is slow: press Enter to search Nominatim"
          : why
            ? "Photon is not answering: press Enter to search Nominatim"
            : "no match: press Enter to search Nominatim",
      );
  };
  const gcFly = (f: Hit) => {
    const [lon, lat] = f.geometry.coordinates;
    const ext = f.properties.extent;
    let zoom = 12.5;
    if (ext && ext.length === 4) {
      const span = Math.max(Math.abs(ext[2]! - ext[0]!), Math.abs(ext[1]! - ext[3]!) * 2, 0.01);
      zoom = Math.log2((360 * ((map.getContainer().clientWidth || 1200) / 512)) / span) - 0.3;
    }
    // never out past where the map is read: a whole region's extent lands near zoom 8, where nothing is read
    zoom = Math.max(minZoom + 0.5, Math.min(16, zoom));
    gc.value = hitName(f);
    gcHits = [];
    gcHide();
    gc.blur();
    map.flyTo({ center: [lon, lat], zoom, duration: 2200, essential: true });
  };
  gc.addEventListener("input", () => {
    clearTimeout(gcTimer);
    gcTimer = setTimeout(gcAsk, 250);
  });
  gc.addEventListener("focus", () => gcHits.length && gcShow());
  gc.addEventListener("blur", () => setTimeout(gcHide, 120));
  gc.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "ArrowDown" && gcHits.length) {
      gcSel = (gcSel + 1) % gcHits.length;
      gcShow();
      e.preventDefault();
    } else if (e.key === "ArrowUp" && gcHits.length) {
      gcSel = (gcSel - 1 + gcHits.length) % gcHits.length;
      gcShow();
      e.preventDefault();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]!);
      else {
        clearTimeout(gcTimer);
        void gcAsk(true).then(() => gcHits.length && gcFly(gcHits[0]!));
      }
    } else if (e.key === "Escape") {
      gcHide();
      gc.blur();
    }
  });

  return { el: box, focus: () => gc.focus() };
}
