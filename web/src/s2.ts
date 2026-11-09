/**
 * Sentinel-2 imagery, one mosaic a year. 2019 on: EOxCloudless by EOX (web mercator JPEG tiles over WMTS; free for
 * non-commercial use with the credit on the map, CC BY-NC-SA 4.0). 2018: EOX's mosaic is hazy and pale (milky over
 * the Amazon, a blue-green cast in the boreal), so it comes from the Copernicus quarterly mosaics instead, made into
 * map tiles in the browser (s2q.ts). Each
 * year is a deck.gl TileLayer of BitmapLayers; the years beside the one shown stay mounted at opacity 0, so their
 * tiles load ahead and a step to them is instant. A gamma on the imagery (; and ') is one uniform, so changing it
 * is a redraw, not a reload.
 */
import { LayerExtension } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";

import { S2Q_CREDIT, s2qTile } from "./s2q";

export const S2Y = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
/** Years drawn from the Copernicus quarterly mosaics (s2q.ts) instead of EOX's tiles. */
const FROM_S2Q = new Set([2018]);
/** Below this zoom a year from the quarterly mosaics is not drawn (too many 100 km tiles to read). */
const S2Q_MIN_ZOOM = 8;
const URL = (y: number) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${y}_3857/default/g/{z}/{y}/{x}.jpg`;
/** EOX's attribution wording (June 2026), with the years the map shows. */
export const S2_CREDIT =
  'EOxCloudless <a href="https://cloudless.eox.at" target="_blank" rel="noopener">https://cloudless.eox.at</a> by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2019 to 2025) &middot; ' +
  S2Q_CREDIT;

const gammaModule = {
  name: "s2gamma",
  fs: `uniform s2gammaUniforms {
  float gamma;
} s2gamma;
`,
  inject: { "fs:DECKGL_FILTER_COLOR": "color.rgb = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / s2gamma.gamma));" },
  uniformTypes: { gamma: "f32" },
};

/** v -> v ** (1 / gamma) on whatever the layer draws; the layer's s2Gamma prop sets it. */
class GammaExtension extends LayerExtension {
  static extensionName = "GammaExtension";
  static defaultProps = { s2Gamma: 1 };
  getShaders() {
    return { modules: [gammaModule] };
  }
  draw(this: any) {
    this.setShaderModuleProps({ s2gamma: { gamma: this.props.s2Gamma ?? 1 } });
  }
}
const GAMMA = new GammaExtension();

/** One year's tiles for one map (`side` keeps the ids apart): shown at opacity 1, or 0 to load ahead. */
function yearLayer(year: number, side: string, shown: boolean, gamma: number, beforeId?: string) {
  return new TileLayer({
    id: `s2${side}-${year}`,
    ...(FROM_S2Q.has(year)
      ? { getTileData: (t: any) => (t.index.z < S2Q_MIN_ZOOM ? null : s2qTile(year, t.index.z, t.index.x, t.index.y, t.signal)) }
      : { data: URL(year) }),
    minZoom: 0,
    maxZoom: 15,
    tileSize: 256,
    refinementStrategy: "best-available",
    opacity: shown ? 1 : 0,
    // read by the MapLibre overlay: the imagery draws under the basemap's labels
    beforeId,
    renderSubLayers: (p: any) => {
      const { west, south, east, north } = p.tile.bbox;
      return new BitmapLayer(p, {
        data: undefined,
        image: p.data,
        bounds: [west, south, east, north],
        extensions: [GAMMA],
        s2Gamma: gamma,
      } as never);
    },
    updateTriggers: { renderSubLayers: [gamma] },
  } as never);
}

/** The layers for a map: the year shown, and the years either side of it loading ahead. */
export function s2Layers(year: number, side: string, gamma: number, beforeId?: string) {
  const i = S2Y.indexOf(year);
  const ys = [S2Y[i - 1], S2Y[i + 1]].filter((y): y is number => y !== undefined);
  return [...ys.map((y) => yearLayer(y, side, false, gamma, beforeId)), yearLayer(year, side, true, gamma, beforeId)];
}

/** The imagery year for a map year: the same year where there is a mosaic, else the nearest. */
export const s2YearFor = (y: number) => S2Y.reduce((a, b) => (Math.abs(b - y) < Math.abs(a - y) ? b : a));
