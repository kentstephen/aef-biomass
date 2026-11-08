# Findings: tree loss from yearly embeddings

What the tests in this repo found, November 2026. Three 10 m yearly embeddings, compared on the same pixels:
AlphaEarth Foundations (AEF, 64 numbers), TESSERA v1.1 (128) and OlmoEarth v1.3 (128). Years 2018 to 2025 (2017
is left out: TESSERA lacks much of it and AEF rebuilt it differently). Reference: Hansen Global Forest Change v1.12
(30 m, loss year to 2024), which has its own errors. The pixels in sections 1 to 3 come from Earthwork
(`~/dev/projects/aef-tessera-earthwork/data`); section 4's squares were picked and read here.

## 1. The signal is there (`scripts/signal_trees.py`, `scripts/trees3.py`)

Nineteen forest squares, 6 km each: ten commercial clearcut regions (US, Sweden, Finland, New Zealand, Brazil) and
nine clearings near mines (Congo, Canada, Indonesia, Peru, Guinea, Sierra Leone). In each, 15,000 pixels Hansen
says lost cover in one year and 5,000 forest that never did.

| | AEF | TESSERA | OlmoEarth |
|---|---|---|---|
| Cut vs standing forest, plain change (1 - cosine, year before vs after), median AUC | .993 | .975 | .987 |
| Largest one-year change in Hansen's year or the next, median | .94 | .94 | .90 |
| Small model taught on the other squares, median AUC / year right | .885 / .73 | .935 / .93 | .869 / .61 |

- Plain change is near perfect everywhere except Sierra Leone (.65 to .74), small clearings that regrow within a year.
- The small taught models carry less well to a new place than plain change does.
- Regrowth after a cut is visible, at the speed of the tree. Cosine to the pixel's pre-cut self, lowest to 2025
  (standing forest beside it in parentheses):

| | AEF | TESSERA | OlmoEarth |
|---|---|---|---|
| Tres Lagoas, Brazil (eucalyptus) | .55 -> .95 (.96) | .50 -> .91 (.91) | .68 -> .92 (.95) |
| Kaingaroa, New Zealand (pine) | .43 -> .67 (.81) | .36 -> .80 (.80) | .59 -> .59 (.88) |
| Coast Range, Oregon (Douglas-fir) | .52 -> .60 (.91) | .39 -> .55 (.87) | .62 -> .66 (.92) |

  TESSERA shows recovery most clearly; OlmoEarth barely moves after a cut.

## 2. False alarms, and the forest gate (`scripts/false_alarms_trees.py`, `scripts/trees3.py`)

Earthwork's trouble spots, never taught: crop switches in 16 US farm regions, farm and marsh lidar, reservoirs,
21 other lookalikes (floods, landslides, river banks, solar arrays), burns, and 20 quiet natural squares. Cutoffs
find 80 % of cuts. The gate asks "was this forest the year before", a logistic regression on the before year's
embedding, taught on the forest squares against non-forest quiet ground (desert, steppe, savanna, grassland).

| Median share lit | AEF plain | AEF gated | TESSERA gated | OlmoEarth gated | 2 of 3 gated |
|---|---|---|---|---|---|
| Crop switches (27) | .985 | .000 | .004 | .000 | .000 |
| Farm lidar (6) | .39 | .000 | .009 | .000 | .000 |
| Reservoirs (6) | .16 | .000 | .000 | .000 | .000 |
| Other lookalikes (21) | .87 | .04 | .07 | .000 | .002 |
| Marsh and storm coast (4) | .40 | .07 | .08 | .004 | .03 |
| Standing forest, 6 quiet squares worldwide | .000 | .000 | .009 | .000 | .000 |
| Cuts found | .80 | .77 | .77 | .77 | .80 |

- Without the gate a crop switch moves AEF as much as a clearcut. With it, crops, farms, reservoirs and open ground
  go to zero.
- Most of what the gate leaves is real tree loss of other kinds: landslides (Wayanad, Haiti), a river eating forest
  (Ucayali), Hurricane Michael (Mexico Beach), forested wetland (Myakka).
- Burns light up (AEF .77, TESSERA .72 gated): fire is tree loss too; telling fire from cutting needs its own class.

## 3. Detectors frozen (`scripts/freeze_detectors.py`, `models/detectors.npz`)

Before any new place was looked at: each embedding's plain-change cutoff and forest gate, as fit on sections 1 and 2.

## 4. Thirty-seven new squares, every pixel (`scripts/pick_squares.py`, `read_squares.py`, `score_squares.py`)

Squares picked from Hansen where about 20 % of the ground lost cover in 2019 to 2024 and at least 30 % was forest in
2000, so each holds cut ground, standing forest and other land. 22 countries; drivers: clearing for farms (14),
shifting cultivation (5), forestry (13), bark beetle and salvage (2), fire (3). Every one of the 360,000 pixels in
each square scored with the frozen detectors, every year. Read in 18 min (32 GB on disk); scored in parallel.

| Median over squares | AEF | TESSERA | OlmoEarth | 2 of 3 |
|---|---|---|---|---|
| Hansen loss found (within a year of Hansen's year) | .83 | .76 | .74 | .76 |
| Dated to Hansen's year or the next | .91 | .90 | .88 | .94 |
| Never-lost forest lit (30 m from any Hansen loss) | .062 | .106 | .026 | .033 |
| Never-lost open ground lit | .008 | .068 | .001 | .003 |
| Flagged pixels Hansen also calls loss (2017 to 2024) | .53 | .47 | .66 | .64 |

Works well: Rondonia, Arkhangelsk, Quebec, Irkutsk, British Columbia, Portugal, Harz, Vysocina, Kyushu, Maule
(vote: 76 to 94 % found, 94 to 99 % dated right, 0.2 to 4 % of standing forest lit).

Three ways it fails:
1. **The gate rejects dry and open forest.** Gran Chaco (Paraguay): plain change sees 100 % of the 2023 clearing in
   all three embeddings, the gate passes 0 %. Manavgat (Turkish pine, burned): seen 91 to 94 %, gate passes 0 to
   15 %. Mpumalanga plantations partly. The gate learned "forest" from humid, boreal and plantation forest against
   desert and steppe; open dry forest falls on the wrong side.
2. **Sparse forest changes too little.** Yakutia larch burn: 0 to 4 % seen even without the gate. Cote d'Ivoire
   (cocoa land): about 40 %. The cutoff was set on dense forest.
3. **Many flags Hansen does not have.** Never-lost forest lit: Riau 74 to 91 %, Nicaragua 40 to 73 %, Latvia AEF
   71 % (TESSERA 12 %). Either Hansen missed clearing (plantation rotations, smallholder clearing, degradation) or
   the detector is wrong there; not checked against imagery yet. Sample points are in `results/squares.json`
   (`vote_no_hansen_examples`).

Data gap: AEF has no 2025 at Maule (Chile), as at Biobio; there the vote runs on TESSERA and OlmoEarth.

## 5. The vegetation volume head abroad (fire-knn's `models/fuel_volume.json`, `scripts/volume_head.py`)

The head was trained on LANDFIRE in California (AEF only). On the 37 squares:

- **Canopy cover ranks trees well everywhere.** Against ESA WorldCover 2020 trees: median AUC .964 (Nicaragua .57,
  Cote d'Ivoire .74, Liberia .72 the low ones).
- **Lifeform and absolute size do not carry.** Tropical forests in Cote d'Ivoire, Laos, Liberia, Cambodia, Vietnam,
  Myanmar and Congo read "tree" on 0 % of WorldCover tree pixels, and their volume before the loss under 1 m.
- **Loss size is right where it reads forest as forest:** volume after over before, median over squares .41; Rondonia
  .06 (5.8 m to 0.4 m), Riau .06, Guaviare .11, Ucayali .14; never-lost forest .97. Burns that resprout (NSW
  eucalyptus .96) and fire-driven loss (Mato Grosso 2020, 1.10) show none.
- So: its canopy cover is a usable forest signal worldwide (a candidate for the gate); its amounts and lifeforms need
  global labels before they mean anything outside California.

## Next

- A gate that knows dry and open forest: the volume head's canopy cover, or the gate taught with dry forest
  (Chaco, Mediterranean, savanna woodland) added.
- A lower or per-cover cutoff for sparse forest (larch, cocoa, savanna).
- A look at Sentinel-2 under the flags Hansen does not have (Riau, Nicaragua, Latvia).
- Loss kinds: cut, burned, storm, landslide, river.
- Volume head retrained on global labels (canopy height maps, WorldCover) for amounts outside California.
