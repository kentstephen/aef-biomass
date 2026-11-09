# aef-biomass

Tests of whether yearly 10 m satellite embeddings can map land cover built around biomass, every year, and how it
changes between years. Three embeddings on the same pixels: [AlphaEarth Foundations](https://source.coop/tge-labs/aef)
(AEF), [TESSERA](https://github.com/ucam-eo/geotessera) v1.1 and [OlmoEarth](https://huggingface.co/allenai/OlmoEarth-v1_3-Base)
v1.3, 2018 to 2025.

**Live map:** [kentstephen.github.io/aef-biomass](https://kentstephen.github.io/aef-biomass/), biomass every year and its change
between years, from AlphaEarth, run in the browser (zoom in to 11 to read it). Built from `web/` and deployed on each
push to `main` (`.github/workflows/pages.yml`). How to read it: [How the map works](#how-the-map-works).

## How the map works

Everything is computed in your browser from [AlphaEarth](https://source.coop/tge-labs/aef) embeddings, one 64-number
summary per 10 m pixel per year (2018 to 2025), read from zoom 11 in. A small model (the "head", taught on CTrees
biomass, Meta canopy height and land cover) turns each pixel's embedding into aboveground biomass in Mg/ha, a canopy
height and a land cover class. Nothing is precomputed; the head runs on the GPU in your browser.

**Each year (C to switch).** One year's biomass per pixel, on a viridis ramp from 0 to 350+ Mg/ha (log scale).
Hover a pixel for its biomass, canopy height and class.

**Change (C).** The window of years at the top of the card (drag either end, or the span between them). Orange is
loss, blue is gain, as a percent of the pixel's own biomass in the first year; clear means no change counted.
- Loss: drawn where the head's biomass fell and the embedding itself changed, both by more than nearly all steady
  ground of the same class does (FINDINGS 29 to 32).
- Gain, by default: where the pixel's biomass rose past a floor (20 to 40 Mg/ha by window length, and at least a
  quarter of what was there).
- Faint against solid: faint is a change that steady ground of the same class often matches; solid is one it rarely
  does. Hovering a changed pixel says between which two years it happened (the one-year step that moved most).
- The card's histogram counts the change in view, by hectare, and what each pixel was and became.

**G: gain taught on lidar (change only, off by default).** Switches how gain is decided. Instead of the floor, a
second model taught on airborne lidar flown twice over the same forest (New Zealand, Estonia, California) marks a
pixel as gain only where the change looks like canopy that grew; the size is still the biomass rise. It usually
draws less, more concentrated gain. On the lidar places it finds slightly less regrowth than the floor; everywhere
else it agrees with Esri's "became trees" about three times as often (FINDINGS 33, 38). G again switches back.

**Imagery.** Hold Space for Sentinel-2 over the whole map, or P for it on the left beside the map. 2019 to 2025:
EOxCloudless by EOX. 2018: the Copernicus Sentinel-2 quarterly mosaics (July to September north of the Tropic of
Capricorn, January to March south of it), made into map tiles in the browser.

**Keys.** C each year or change; `[` `]` the year (in change, the imagery's year); B blink between the first and the
latest year; G lidar gain on or off (change view); Space imagery; P pair; Q picking and tooltip; L place names;
`;` `'` the imagery's brightness; F full screen; / search; Esc folds the card.

**What it does not do well yet.**
- Grass and crops: the head is taught on woody biomass, so fields and grass get a rough number, not their own crop
  or grass mass (FINDINGS 34).
- Tundra reads four to five times too high, CTrees' level (FINDINGS 36).
- Tall tropical forest reads short above about 150 Mg/ha, about half the plot biomass above 400 (FINDINGS 35).
- Burns: a burned stand can keep standing dead trees, which the map and lidar height both struggle with (FINDINGS 32).
- Regrowth is the weak half of change: slow growth year to year is small against the noise.

## Where this starts

This repo extends the fuel map of [aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr)
(`main`) from California to the world, and from fuel to biomass. There, small heads on one year's AlphaEarth
embedding read, per 10 m pixel and per year:

- fuel or not fuel: P(fuel) over the 11 LANDFIRE FBFM40 fuel kinds, held-out AUC .956
  ([`scripts/fit_fuel.py`](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr/blob/main/scripts/fit_fuel.py),
  `models/fuel_head.json`);
- vegetation volume (cover x height), leading lifeform (tree, shrub, herb), canopy cover and height, fuel kind:
  log volume R2 .79, lifeform right on 80 % of pixels
  ([`scripts/fit_volume.py`](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr/blob/main/scripts/fit_volume.py),
  `models/fuel_volume.json`, copied here);
- the same read year after year, 2019 to 2025, on the web map (`?app=fuel`,
  [`web/src/fuel/`](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr/tree/main/web/src/fuel)): Caldor's
  volume 11, 13 m before the fire, 4.0 in the fire year, 2.5 m by 2025; Creek forest 91 % tree before, 51 % herb in
  2022, 56 % shrub in 2025.

The change side follows Earthwork, which found digging from two years of embeddings, taught and checked by repeat
lidar: [aef-earthwork](https://github.com/kentstephen/aef-earthwork) (model, notebooks),
[aef-earthwork-web](https://github.com/kentstephen/aef-earthwork-web) (the change map with a window of years) and
[aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork) (TESSERA and AlphaEarth, the repeat
lidar sites in 11 countries and the US, the lookalike sets). Its lidar sites are the independent check for
vegetation amount and change here; its lookalikes (crop switches, farms, marsh, reservoirs, burns) are kinds of
vegetation change to map, not false alarms. One "what changed" model joining the two is the long-term aim.

That much detail in classes and amounts, every year, is the reason to try a biomass map: the same kind of head,
taught on world labels (CTrees biomass, Meta canopy height, land cover classes), over all kinds of ground, not only
forest.

## Files

- `FINDINGS.md`: every test behind the map, in order, with what did not work. Tree loss (sections 1 to 5);
  biomass, canopy height and land cover from one year's embedding over 92 squares of every kind of ground, held out
  by square and by continent (6 to 10); margins of error, calibrated change, the top end, regrowth, hard ground and
  airborne lidar (11 to 16); grass, shrub, combined change, lidar at 2 m, tree loss and which embeddings to read (17
  to 26); the map's head on burns and land cover, steady-ground noise and drawing change with its confidence (27 to
  30); growth, loss size and gain taught on lidar (31 to 33); and checks against references the head never saw:
  grass abroad, plot biomass in tall forest, tundra, a burn flown before and after, and the lidar-taught gain in the
  browser (34 to 38).
- `PLAN.md`: the original testing plan (November 2026), kept as written; the map has since been built.
- `web/`: the map ([How the map works](#how-the-map-works)). The head runs on the GPU in the browser (WebGPU in
  workers, the CPU without it); AlphaEarth is read straight from S3 behind Source Cooperative, its chunks kept on disk
  between visits. Reading, the head on the GPU, search, the card and the window range come from fire-knn and
  Earthwork's web apps. Run: `cd web && npm install && npm run dev` (port 3010). The head:
  `uv run scripts/export_biomass_head.py` writes `web/public/models/biomass_aef.json`; the lidar-taught gain:
  `scripts/gain_mlp.py --export difference` writes `web/public/models/gain_lidar.json`.

## Scripts

| Script | What |
|---|---|
| `signal_trees.py` | Is there signal for trees cut and for construction, AEF and TESSERA |
| `false_alarms_trees.py` | The tree-cut detector on crop switches, farms, marsh, reservoirs, lookalikes and burns |
| `probe_olmo.py`, `read_olmo.py` | OlmoEarth coverage, and OlmoEarth under the pixels the other two were read at |
| `trees3.py` | AEF, TESSERA and OlmoEarth side by side on the forest squares and the trouble spots |
| `freeze_detectors.py` | Saves the detectors (`models/detectors.npz`) before any new place is scored |
| `pick_squares.py`, `read_squares.py`, `score_squares.py` | 37 new squares from Hansen loss, every pixel and year in all three embeddings, scored |
| `pick_veg_squares.py` | 55 more squares from ESA WorldCover, each dominated by one kind of ground (grass, shrub, savanna, crop, wetland, city edge, desert edge, tundra, intact forest); read with `read_squares.py --picks veg_squares.json` |
| `probe_labels.py`, `read_labels.py` | The world labels under every pixel of all 92 squares: CTrees biomass and its error, Meta canopy height and its date, Impact Observatory annual classes, biodiversity intactness |
| `biomass_rows.py`, `biomass_head.py` | One head per year's embedding (biomass, canopy height, class), held out by square and by continent, per embedding and all three |
| `class_report.py` | Land cover with its margins of error: per class with intervals, and the head's confidence against how often it is right |
| `biomass_change.py` | Biomass change between years against CTrees' change: plain difference, calibrated, gated by embedding change, a pair head; and the 10 m view at Hansen loss |
| `biomass_lookalikes.py` | Earthwork's trouble spots (crop switches, farms, marsh, reservoirs, floods, burns) through the biomass head |
| `class_calibrate.py` | Temperature scaling and conformal class sets, fit across folds (`models/class_temperature.json`) |
| `change_calibrate.py` | Biomass change calibrated per window length, from the saved cell tables |
| `top_end.py` | CTrees' own error at the top end, quantile mapping, and the head taught with weight on high biomass (`biomass_head.py --weight-top 1 --tag _top`) |
| `read_cci.py`, `cci_compare.py` | ESA CCI Biomass v7.0 under every square; CTrees against CCI against the heads |
| `trajectory_head.py` | A head on all eight years of a pixel at once, scored on change and regrowth |
| `class_hard_ground.py` | The classes where WorldCover and IO disagree, with Meta's height as referee for trees |
| `read_lidar.py`, `lidar_check.py` | Airborne lidar canopy height (LINZ, Estonia, NASA WERK; surface minus ground through Earthwork's readers) under 18 new squares, and the heads against it; read with `read_squares.py --picks lidar_squares.json` and `read_labels.py --picks lidar_squares.json` |
| `pick_us_squares.py` | 10 US rangeland squares under RAP and 7 southern California chaparral squares (six inside MTBS fire perimeters); read with `read_squares.py --picks us_squares.json` and `read_labels.py --picks us_squares.json` |
| `read_rap.py` | Rangeland Analysis Platform v3 herbaceous biomass and cover, 2018 to 2025, under every square in the lower 48 |
| `read_chaparral.py` | The SoCal annual biomass map (Dryad doi:10.5061/dryad.qz612jmjt), 2018 to 2023, from the yearly zips in `data/chaparral/zips` |
| `map_check.py` | The CTrees-taught heads, CTrees, and a head taught on the map itself (held out by square) against RAP (`--label rap`) or the chaparral map (`--label chaparral`), on amounts, years and swings |
| `class_hard_calibrate.py` | Class calibration fit on hard pixels (the maps disagree, Meta height as referee), checked against airborne lidar |
| `grass_tests.py` | Grass: a change threshold set against RAP's year noise, California annual grass, a two-part head (CTrees total and RAP grass), the RAP head abroad |
| `class_hard_tests.py` | Hard-ground classes: an offset calibration, per-continent fits, shrub against grass refereed by lidar, open woodland as a tenth class (`models/class_hard_tests.json`) |
| `top_end_lidar.py` | The top end and the shrub level against airborne lidar: CTrees, CCI, the chaparral map and the heads by lidar height |
| `read_lidar2m.py`, `lidar_check2.py` | More airborne lidar, read at 2 m (France, Scotland, Canada, the Netherlands, Norway), and the heads against it |
| `change_combined.py` | Change for the map: loss from one head and gain from another, every pairing, against CTrees and lidar; a gradual gate; a floor on gain; the ramp limits |
| `embedding_bandwidth.py` | What each embedding costs to read on the fly: bytes and requests per square-year from each public store |
| `pair_heads.py`, `pair_scores.py` | Heads on two embeddings at once, scored beside one and all three on the map's tests, with read cost |
| `tree_loss_tests.py` | Tree loss: a height gate for dry and open forest, cutoffs by cover, loss from fire (GLAD), Sentinel-2 under the flags Hansen lacks |
| `burn_check.py` | The map's head inside six California burn scars (MTBS, NIFC WFIGS) against unburned ground around each, beside CTrees and fire-knn's volume head |
| `nlcd_check.py` | Biomass by land cover and by class change from year to year, against Annual NLCD (`--source nlcd`, CONUS) or Esri's annual land cover (`--source io`, worldwide) |
| `change_confidence.py` | The head's change on steady ground (Esri class held, CTrees steady) per class and window length: how big a change must be to be believed |
| `change_confidence_test.py` | Every way of drawing change with its confidence (one percent table, per class, with the embedding's move; gain per pixel, sliding 100 m or fixed blocks) against lidar and Esri's changes, held out by square |
| `growth_test.py` | The ways to a stronger growth signal (the trajectory head, TESSERA, the two together) under `change_confidence_test.py`'s harness, every head held out; for the map's head, the confidence keyed on its own class, and the browser's tables (`--write-tables`) |
| `window_test.py` | The map's loss size (its soft rule and the others) against lidar's canopy fall; gain by window length, two years against a trend over every year of the window, on Esri's changes and lidar |
| `lidar_gain_head.py` | Gain taught on lidar's canopy rise directly (AlphaEarth at both flight years), held out by square, beside the map's gain; then on Esri's "became trees" over every other square; `--variants difference,"first year"` for the difference alone and the first year alone |
| `grass_abroad.py` | Grass years outside the US against Wu et al. 2026 global grassland biomass (500 m, 2018 to 2022), the map and a RAP-taught head; Wu against RAP at home |
| `export_biomass_head.py` | Exports the map's head (one year's AlphaEarth to biomass, canopy height and class) to `web/public/models/biomass_aef.json` |
| `aef_read.py` | AlphaEarth (raw int8) over any lon/lat box from the mosaic, and the map's head on it, for the label checks outside the squares |
| `top_end_plots.py` | The top end against plot biomass: Rodda et al. 2024 lidar and plot maps (13 tropical sites) and GEO-TREES census subplots, beside CTrees and CCI |
| `tundra_check.py`, `orndahl_read.py` | Tundra against Orndahl's 30 m Arctic biomass (2020) on 11 squares and the Berner harvest plots from 2017 on; a reader for Orndahl's striped tiles over HTTP |
| `gain_mlp.py` | The lidar-taught gain as a small network the browser runs (on the difference of two years), held out by lidar square, with its steady-ground table; exports `web/public/models/gain_lidar.json` (the map's G switch) |
| `burn_lidar_ca.py` | The Jasper fire (2024) against NRCan lidar flown in 2019 and 2024: canopy height and cover per pixel, beside the map's fall and CTrees |
| `volume_head.py` | The vegetation volume head of [aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr) (`models/fuel_volume.json`, trained on LANDFIRE in California) in numpy |

Several scripts read pixels and labels that the Earthwork repos already hold
(`../aef-tessera-earthwork/data`, built by [aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork)) and
run with its environment: `uv run --project ../aef-tessera-earthwork --with psutil python scripts/<script>.py`. The
others carry their dependencies inline (`uv run scripts/<script>.py`). Those scripts expect aef-tessera-earthwork
cloned at `~/dev/projects/aef-tessera-earthwork`. Data and logs are not in the repo; results are in `results/`.

## Data

All public, no credentials: AlphaEarth Foundations (tge-labs on Source Cooperative), TESSERA v1.1 (Source
Cooperative), OlmoEarth v1.3 embeddings (Source Cooperative; OlmoEarth Artifact License), Hansen Global Forest Change
v1.12, ESA WorldCover 2020 and 2021, CTrees aboveground biomass (100 m, AWS Open Data, CC-BY 4.0), Meta and WRI canopy
height v2 (AWS, CC BY 4.0), Impact Observatory annual land use v02 (Planetary Computer), Vizzuality and Impact
Observatory biodiversity intactness v1.1 (Source Cooperative, CC-BY 4.0), and the Earthwork labels (USGS 3DEP, USDA
Cropland Data Layer), ESA CCI Biomass v7.0 (CEDA, free and open with citation), LINZ nz-elevation DSM and DEM (CC BY
4.0), Estonia's Maa- ja Ruumiamet lidar point clouds, NASA WERK California lidar canopy height (CC0), the
Rangeland Analysis Platform v3 (Univ. of Montana), MTBS burned area perimeters (USGS and USFS, via Source
Cooperative), and the SoCal annual biomass map (Schrader-Patton et al., Dryad, CC0; its download needs a browser or a
Dryad account token), GLAD forest loss due to fire (University of Maryland), Sentinel-2 L2A (Earth Search), and airborne
lidar read at 2 m: IGN LiDAR HD canopy height (France), the Scottish Remote Sensing Portal, NRCan HRDEM (Canada),
AHN5 (the Netherlands) and Kartverket (Norway). For the label checks of sections 34 to 37: Wu et al. 2026 global
grassland biomass (Zenodo 18044163, CC BY 4.0), Rodda et al. 2024 lidar-based biomass maps (IRD Dataverse
doi:10.23708/H2MHXF, CC BY 4.0), GEO-TREES plot biomass (IIASA, GEO-TREES terms), Orndahl et al. 30 m Arctic plant
biomass (Arctic Data Center doi:10.18739/A2NS0M06B), the Berner et al. 2024 Arctic plant biomass synthesis (Arctic
Data Center doi:10.18739/A2QJ78081), and NRCan HRDEM lidar over Jasper (Open Government Licence Canada). The map's
imagery: EOxCloudless by EOX IT Services GmbH, 2019 to 2025 (free for non-commercial use with attribution), and the
Copernicus Sentinel-2 Global Mosaics (ESA) for 2018, via TGE Labs on Source Cooperative. The picks are in `results/veg_squares.json`,
`results/lidar_squares.json`, `results/lidar2_squares.json` and `results/us_squares.json`; trained heads and cell
tables stay in `data/`.
