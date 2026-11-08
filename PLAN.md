# Plan: biomass classes and their change, from yearly embeddings

Started November 2026. Findings so far are in `FINDINGS.md`. This file is the plan; append dated sections, do not
rewrite old ones.

## Goal

A classified land cover map built around biomass, every year, and where it changed between years. It extends the
fuel heads of `aef-hls-s1-fire-knn` (fuel kinds, lifeform, canopy cover and height, vegetation volume, all trained
on LANDFIRE in California) to the whole world with world labels, and adds biomass in Mg/ha.

Two map forms, built later (not in this phase):

1. **A series of years.** Each year's embeddings, read for the view, run through the model in the browser: class
   and biomass per pixel, one year at a time.
2. **A change map with a window**, as Earthwork does for digging: pick two years (or a span), see what changed
   from what to what, and how much biomass went or came back.

This phase is testing only: does the model hold up across places it never saw, in classes, in amounts, and in
change between years.

## What is already known (FINDINGS.md)

- Plain embedding change finds tree loss and dates it well; a "what was here before" read is what removes false
  alarms (crop switches, wetlands, reservoirs). That read was a two-class gate; it failed on dry and open forest
  (Chaco, Mediterranean pine). Classes with an open woodland class are the fix.
- fire-knn's volume head abroad: canopy cover ranks trees well everywhere (median AUC .964 against ESA WorldCover
  trees); lifeform and amounts do not carry outside California (tropical forest read as "not tree").
- fire-knn's change was muted pixel by pixel (volume fell by a factor of 4 where LANDFIRE fell by 18), the usual
  pull toward the mean when a per-year model is differenced.
- Three embeddings tested on the same pixels: AEF, TESSERA, OlmoEarth. All detect change; TESSERA shows regrowth
  most clearly, OlmoEarth least; AEF lacks 2025 in central Chile.

## Labels

| Label | What | Grain | Years | Access | Notes |
|---|---|---|---|---|---|
| CTrees AGB | aboveground biomass, Mg/ha, with a residual standard error | 100 m | 2000 to 2025 | Icechunk on AWS Open Data, bucket `ctrees-agb-100m-global`, prefix `agb_100m_global`, group `aboveground_biomass`; anonymous; int16 / 10, nodata -9999; reader in `../ne-landsat-ctrees-marimo/ne-landsat-ctrees-pair.py` | CC-BY 4.0. Partly calibrated on GEDI, which AEF was taught to reconstruct: AEF's agreement with it runs flattering |
| Meta canopy height | tree height | 1 m | one date per place (v1 imagery mostly 2009 to 2020; v2 2026 with observation dates) | `s3://dataforgood-fb-data/forests/` (v1, v2), no sign-in | CC BY 4.0. From very high resolution imagery, not the embeddings' inputs. Access and date layer to verify |
| Land cover classes | class per pixel | 10 m | ESA WorldCover 2020, 2021; Impact Observatory annual 2017 to 2023 | WorldCover already read (`read_squares.py`); IO annual on Planetary Computer, access to verify | Class definitions differ; use places where two agree |
| LANDFIRE | FBFM40 fuel kinds, EVC / EVH lifeform, CC, CH | 30 m | 2016, 2022, 2024, 2025 | fire-knn's `sample_change.py` (lfps.usgs.gov) | US only; keeps the fuel kinds |
| BII v1.1 | biodiversity intactness 0 to 1 | ~100 m | 2017 to 2025 | Source Coop `vizzuality/biodiversity-intactness-100m-v1-1` | A model output itself; a third opinion, not truth. Coverage, grid, nodata to verify |
| Hansen GFC v1.12 | tree loss year, tree cover 2000 | 30 m | 2001 to 2024 | already read | Reference for loss timing |
| Repeat lidar | height change | 1 to 2 m | two flights per site | Earthwork: 3DEP sites and 128 sites in 11 countries (`world_*.py`); NEON AOP | The independent check for amounts and change |

## Pixels

The 37 squares of section 4 and the 19 forest squares of section 1 already hold all three embeddings for every year
2018 to 2025 (`data/squares/`, Earthwork's `data/separate` plus `data/olmo/`). Labels are read under those pixels:
no new embedding download for the first tests. More squares (grassland, savanna, cropland, cities, wetlands, dry
forest) are added the same way (`pick_squares.py` with a different picking rule, `read_squares.py`) once the first
tests run.

## Steps

### 1. Read the labels under the existing pixels
- CTrees: every year 2018 to 2025, AGB and its error, for each square; each 10 m pixel gets its 100 m cell's value and
  a cell id.
- Meta: canopy height at each pixel (mean of the 1 m heights inside the 10 m pixel), with the observation date.
- Impact Observatory annual classes 2018 to 2023 at each pixel; WorldCover is already there.
- Check first, on one square, before the full read: access, grid, nodata, read time.

### 2. One head, several outputs
- Input: one year's embedding (AEF 64, TESSERA 128, OlmoEarth 128, or all three).
- Outputs: biomass (log Mg/ha), canopy height, canopy cover, class (a short list built around biomass: bare or built,
  water, crop, grass, shrub, open woodland, closed forest, wetland), and the fuel kinds where LANDFIRE exists.
- Biomass is taught at CTrees' 100 m: the mean of the head's 10 m predictions inside a cell is matched to the cell.
  Height and class are taught per 10 m pixel.
- Same size as fire-knn's heads (a layer of 64), so it can run in the browser later.

### 3. Tests, held out by place
- Every square scored by a head taught without it; then a whole continent left out.
- Biomass: R2 and median error against CTrees at 100 m; against repeat lidar where it exists (independent of GEDI).
- Height: against Meta (held-out squares) and repeat lidar.
- Classes: against WorldCover and IO where both agree.
- Per embedding and all three together.

### 4. Change between years
- Per pixel, the head's biomass and class each year. Change = later minus earlier, for any window of years.
- Score against CTrees' own year-to-year change (100 m) and repeat lidar: the slope of predicted on true change (1 is
  unmuted), loss found, steady ground left alone.
- Fixes for muting, compared: teaching on pairs of years (two embeddings in, the label change out), calibrating
  predicted change on held-out pixels, and counting a class change only where the embedding itself changed past the
  frozen cutoffs (`models/detectors.npz`), which kept steady ground quiet in section 4.
- The dry-forest and sparse-forest failures of section 4 are rerun with the class head in place of the forest gate.

### 5. Then the map
- One of the two forms above, decided after the tests.

## Rules for running
- Estimate peak RAM before every heavy job and give it a cap; this machine has 64 GB. Section 4's scoring reached
  37 GB with 6 workers and 25 GB with 4.
- Drop 2017 (TESSERA gaps, AEF rebuilt it). Check per embedding for missing years (AEF 2025 in central Chile).
- Detectors and heads are frozen before new places are scored.
- Source Coop and the other public stores: four squares at a time has been fine.

## 2026-11-07, later: the starting point, and the squares

The starting point is fire-knn's fuel map ([aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr),
`main`; README "Where this starts" has the links): fuel or not fuel, and per year a lifeform, canopy cover, height,
fuel kind and vegetation volume at 10 m, which tracked fires year by year. This plan widens that to biomass over all
kinds of ground, worldwide. The tree-loss work in `FINDINGS.md` came first; at the end of that session the target moved to biomass, and its
results (canopy cover ranks trees abroad, AUC .964; loss sizes right where forest reads as forest) are what say a
biomass map could work.

The 37 squares on disk were picked for tree loss (Hansen ~20 % loss, at least 30 % forest), and section 1's are
cut-forest samples. A training set from them alone would be a forest set. Squares across the whole range (bare,
built, water, crop, grass, shrub, savanna, open woodland, closed forest, wetland, mangrove), worldwide, come before
the label read, not after. The forest squares stay as some of the forest rows.

Labels checked under one square (2026-11-07, `scripts/probe_labels.py`): CTrees AGB and its error, Meta CHM v2
10 m mean height with acquisition dates, Impact Observatory annual classes 2017 to 2023, BII v1.1 2017 to 2025;
all public, no sign-in. CTrees and Meta are strongest on trees; no worldwide label for grass or shrub amount is
in hand yet.

## 2026-11-08: notes for the map phase (not built yet)

### Shade the embeddings on the GPU, in the browser
The map runs the heads on the embeddings in the browser, on the GPU, through deck.gl-raster (Development Seed), as
Earthwork and fire-knn already do. Start from that code rather than anew:
- [aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork) `web/` (deck.gl-raster, deck.gl-zarr,
  deck.gl-geotiff 0.8.1): `src/earthwork-score.ts` (a luma.gl shader module: each year's 64 AlphaEarth bands as an r8sint
  Texture2DArray, the weights as a texture, the score per pixel in the fragment shader), `src/aef.ts`, `src/tessera.ts`
  (both read straight from Source Cooperative), `src/disk-cache.ts` (kept on disk between visits), `src/chunk.worker.ts`.
- [aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr) `web/` (`main`): `src/fuel/head-gpu.ts`,
  `src/fuel/volume-gpu.ts` (the fuel and volume heads as WebGPU compute, int8 packed in u32, one thread per pixel, CPU
  fallback in `volume.ts`), `src/aef/aef.worker.ts` and `pool.ts` (IndexedDB chunk cache, per-worker memory, a chunk
  always on the same worker so the next year needs one read), `src/aef/render.ts`, `src/gpu/sample-texture-2d-array.ts`.
- [aef-landcover-fosberg-cog](https://github.com/kentstephen/aef-landcover-fosberg-cog) `web/` (newest, 2026-11-07): the
  same `gpu/` and `aef/render.ts` lineage.
The biomass head is the same shape as fire-knn's (one layer of 64), so it drops into that shader path: a year's
embedding in, biomass, height and class out per pixel; the change map is two years through it, or the pair head.

### Color
- Long ramps the way [five-min-away-redux](https://github.com/kentstephen/five-min-away-redux) and usgs-mrms-flood made
  them: a CARTO ramp remade as 50 stops along its own hue path, extended at both ends (Emrld 50 in
  `five_min_away.py`; the GLSL lookup `emrld(t)` in usgs-mrms-flood `web/src/elev.ts`, with its rank stretch, color by
  percentile of the pixels in view). Seven stops washed out; fifty did not.
- Directions, none picked: biomass amount on one lightness ramp (Emrld 50 as is); change on a cool to warm diverging
  ramp, blue for gain and orange for loss (or the other way), each side remade to many stops, with the middle (no
  change, status quo) light or dark. Blue against orange holds up without red; CARTO ArmyRose's rose end leans red.
- No red-against-green anywhere; strokes in their own color, no dark casing.

### Land cover with its error
Classes come out of the same head, with an error a map can show: per class from the held-out tests
(`results/class_report.json`, precision and recall with intervals), and per pixel from the head's own confidence once
it is calibrated (raw, it runs overconfident; see FINDINGS section 8).

## 2026-11-08: steps 1 to 4 run once (FINDINGS sections 6 to 10)

- Step 1 done with 55 new squares over every kind of ground (`pick_veg_squares.py`) and the labels under all 92
  (`read_labels.py`). BII read, not used yet.
- Step 2 done (`biomass_head.py`, `class_report.py`): biomass R2 of log .70 to .74 held out by square, .65 to .67 by
  continent; height R2 up to .78; classes 91 to 92 % right with OlmoEarth or all three, intervals per class.
- Step 3 (held out by place) is inside step 2. Repeat lidar not done: Earthwork's lidar is bare ground only.
- Step 4 run (`biomass_change.py`, `biomass_lookalikes.py`): loss over 2018 to 2025 at about four-fifths of CTrees'
  size, a single year under half; the gate and the pair head keep steady ground quiet; regrowth barely shows;
  Earthwork's trouble spots mostly stay quiet in the biomass head, wetland the exception, handled by the gate.
- Open, in FINDINGS "Next": class confidence calibration, change calibration per window length, the squeezed top end,
  a canopy lidar check, tundra, hard-ground class labels, open woodland as its own class.

## 2026-11-08, later: the open items, and where each one's data comes from

The "Next" list of FINDINGS sections 6 to 10, in this order:

1. **Class margin of error** (`class_calibrate.py`): a temperature on the class logits and split conformal class
   sets, both fit across folds. No new data.
2. **Change calibrated per window length** (`change_calibrate.py`): from the saved cell tables
   (`data/heads/change_cells_*.npz`); one-year changes add up to every window. No new data.
3. **The squeezed top end** (`top_end.py`, `biomass_head.py --weight-top`): CTrees' own error per cell (the training
   draw replayed), quantile mapping across folds, and a head taught with more weight above 100 Mg/ha.
4. **Trajectory head** (`trajectory_head.py`): the eight years of a pixel in at once, biomass per year out, for
   regrowth.
5. **Independent lidar check** (`read_lidar.py`, `lidar_check.py`): 18 new squares, canopy height = surface minus
   ground from the programs Earthwork reads for bare ground (LINZ, NRCan HRDEM, AHN, Estonia's point clouds on disk)
   plus NASA WERK in California (1 m before and after the 2025 Los Angeles fires; statewide 10 m, one flight per
   place). NEON AOP needs an account token since 2026-06-30 (not used).
6. **A second biomass map** (`read_cci.py`): ESA CCI Biomass v7.0 (radar, ICESat-2 calibrated, 100 m, 2018 to
   2024) under every square, for tundra and as a check on CTrees.
7. **Hard ground for classes** (`class_hard_ground.py`): pixels where WorldCover and IO disagree, with Meta's
   height as referee for trees.

## 2026-11-08, afternoon: non-tree checks against maps that are not CTrees

1. **Grass against RAP** (`pick_us_squares.py`, `read_rap.py`, `map_check.py --label rap`): 10 new US rangeland
   squares plus two of section 6; the CTrees-taught heads, CTrees, and a head taught on RAP (held out by square),
   on amounts, years and swings. Done: FINDINGS 17.
2. **Shrub against the SoCal chaparral map** (`read_chaparral.py`, `map_check.py --label chaparral`): six fire
   squares inside MTBS perimeters and one steady square, read; the map's yearly zips are still to download.
3. **Classes calibrated on hard ground** (`class_hard_calibrate.py`): a temperature and set cutoffs fit on
   Meta-refereed hard pixels, checked against the lidar squares. Done: FINDINGS 18.

## The map: Earthwork's pair map, for biomass

Fork the pair map of aef-earthwork-web: one year against another. Two settings, both on the long 50-stop ramps:
- **Change** (first): biomass change between the two years, blue through a low-alpha white at no change to
  orange, in the manner of the CTrees Landsat change viewer.
- **Each year**: biomass of one year in viridis, stepping year over year.
Full window by default, no dark casings on strokes. GPU shading notes above (deck.gl-raster, Earthwork and fire-knn
web code).

### How the fork goes (read from the Earthwork, fire-knn and usgs-mrms-flood web code)
- Base: `aef-earthwork-web/web` (Vite, TypeScript, deck.gl 9.4, deck.gl-raster / deck.gl-zarr 0.8.1, MapLibre,
  zarrita; GitHub Pages workflow). It reads the AlphaEarth GeoZarr mosaic (`data.source.coop/tge-labs/aef-mosaic`,
  int8 [time, band, y, x], 256-pixel chunks) in a worker, takes two years from the "AEF Years" range slider
  (`ui.js`), and has the pair (P: a second synced map), full screen (F), the loading bar and the key handling to keep.
- The head on the GPU: fire-knn's `fuel/volume-gpu.ts` (WebGPU compute, int8 packed in u32, one thread per pixel:
  dequantize, standardize, Linear 64, ReLU, Linear) and `fuel/volume.ts` (CPU fallback, `probeGap` check), with
  weights in fire-knn's `fuel_volume.json` format (`mean`, `scale`, `w1`, `b1`, `w2`, `b2`, probe inputs and
  outputs). The biomass head has the same shape: export it in that format, one output (log biomass).
- Which embeddings (open): all three are public Zarr on Source Cooperative and readable in the browser with
  zarrita. AlphaEarth: the mosaic, already on the map's grid. TESSERA: `aef-tessera-earthwork/web/src/tessera.ts`
  reads it already, onto AlphaEarth's grid. OlmoEarth: `s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr`
  (https://data.source.coop), one group per UTM zone, so a worker projects each pixel center into the zone as
  `read_olmo.py` and TESSERA's `utm.ts` do; no browser reader yet. The heads exist for each alone and all three
  together (section 7 and the side-by-side table). One head taught on all squares for the map, or the mean of
  the fold heads.
- The worker scores each chunk-year once and keeps the biomass planes (256 KB per chunk-year), so stepping a year
  or changing the pair reads nothing new.
- Shaders, after usgs-mrms-flood `elev.ts` (`Elev`: a 50-stop GLSL const array, a 256-bin CDF over the pixels in
  view for the rank stretch):
  - change: d = B - A, t = clamp(d / dmax, -1, 1), 50 stops blue to white to orange, alpha low near zero
    (rising with |t|), rank stretch on |d| with the sign kept;
  - each year: 50-stop viridis (`VIRIDIS_STOPS` in fire-knn `aef/ramp.ts`), stepping the year with [ ].
- First build: no side-by-side imagery and no Earth Genome Sentinel-2 mosaic. Only the biomass layers, and the work
  goes into getting their look right: the change map, then the individual years. Imagery can come later.
- Drop from the Earthwork base for that first build: Sentinel-2 and the pair's imagery, the paint, hexes. Which
  embeddings it reads is open (section 25). Keep the tooltip value (CPU forward pass).
