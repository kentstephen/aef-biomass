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
