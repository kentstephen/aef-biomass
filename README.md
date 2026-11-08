# aef-biomass

Tests of whether yearly 10 m satellite embeddings can map land cover built around biomass, every year, and how it
changes between years. Three embeddings on the same pixels: [AlphaEarth Foundations](https://source.coop/tge-labs/aef)
(AEF), [TESSERA](https://github.com/ucam-eo/geotessera) v1.1 and [OlmoEarth](https://huggingface.co/allenai/OlmoEarth-v1_3-Base)
v1.3, 2018 to 2025.

- `FINDINGS.md`: what the tests found so far (tree loss: signal, false alarms, 37 squares wall to wall, fire-knn's
  volume head abroad).
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
| `volume_head.py` | The vegetation volume head of [aef-hls-s1-fire-knn](https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr) (`models/fuel_volume.json`, trained on LANDFIRE in California) in numpy |

Several scripts read pixels and labels that the Earthwork repos already hold
(`../aef-tessera-earthwork/data`, built by [aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork)) and
run with its environment: `uv run --project ../aef-tessera-earthwork --with psutil python scripts/<script>.py`. The
others carry their dependencies inline (`uv run scripts/<script>.py`). Data and logs are not in the repo; results are
in `results/`.

## Data

All public, no credentials: AlphaEarth Foundations (tge-labs on Source Cooperative), TESSERA v1.1 (Source
Cooperative), OlmoEarth v1.3 embeddings (Source Cooperative; OlmoEarth Artifact License), Hansen Global Forest Change
v1.12, ESA WorldCover 2020 and 2021, and the Earthwork labels (USGS 3DEP, USDA Cropland Data Layer).
