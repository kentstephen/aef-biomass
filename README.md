# aef-biomass

Tests of whether yearly 10 m satellite embeddings can map land cover built around biomass, every year, and how it
changes between years. Three embeddings on the same pixels: [AlphaEarth Foundations](https://source.coop/tge-labs/aef)
(AEF), [TESSERA](https://github.com/ucam-eo/geotessera) v1.1 and [OlmoEarth](https://huggingface.co/allenai/OlmoEarth-v1_3-Base)
v1.3, 2018 to 2025.

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

- `FINDINGS.md`: what the tests found so far. Biomass, canopy height and land cover from one year's embedding over 92
  squares of every kind of ground, held out by square and by continent; biomass change against CTrees; Earthwork's
  trouble spots through the biomass head; the three embeddings side by side. Before that, tree loss (signal, false
  alarms, 37 squares wall to wall, fire-knn's volume head abroad).
- `PLAN.md`: the plan for the biomass classes and their change.

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
| `volume_head.py` | The vegetation volume head of [aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr) (`models/fuel_volume.json`, trained on LANDFIRE in California) in numpy |

Several scripts read pixels and labels that the Earthwork repos already hold
(`../aef-tessera-earthwork/data`, built by [aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork)) and
run with its environment: `uv run --project ../aef-tessera-earthwork --with psutil python scripts/<script>.py`. The
others carry their dependencies inline (`uv run scripts/<script>.py`). Data and logs are not in the repo; results are
in `results/`.

## Data

All public, no credentials: AlphaEarth Foundations (tge-labs on Source Cooperative), TESSERA v1.1 (Source
Cooperative), OlmoEarth v1.3 embeddings (Source Cooperative; OlmoEarth Artifact License), Hansen Global Forest Change
v1.12, ESA WorldCover 2020 and 2021, CTrees aboveground biomass (100 m, AWS Open Data, CC-BY 4.0), Meta and WRI canopy
height v2 (AWS, CC BY 4.0), Impact Observatory annual land use v02 (Planetary Computer), Vizzuality and Impact
Observatory biodiversity intactness v1.1 (Source Cooperative, CC-BY 4.0), and the Earthwork labels (USGS 3DEP, USDA
Cropland Data Layer). The picks are in `results/veg_squares.json`; trained heads and cell tables stay in `data/`.
