# Findings: biomass, land cover and tree loss from yearly embeddings

What the tests in this repo found, November 2026. Sections 1 to 5 are tree loss (the first session); sections 6 to
10 are biomass and land cover over every kind of ground (the second). Three 10 m yearly embeddings, compared on the same pixels:
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

## 6. Squares over every kind of ground, and the world labels (`scripts/pick_veg_squares.py`, `read_labels.py`)

The 37 squares of section 4 were picked for tree loss. 55 more, 6 km each, picked from ESA WorldCover 2021 so one
kind of ground dominates: grass (7), shrub (7), savanna and open woodland (8), crop (8), wetland and mangrove (7),
the edge of a growing city (7), desert edge (2), tundra (3), intact closed forest (6); every continent but
Antarctica. All three embeddings read for every year 2018 to 2025 (25 min, 49 GB). Under all 92 squares, every pixel
(22 min, all public, no sign-in):

| Label | Grain | Years | Notes |
|---|---|---|---|
| CTrees aboveground biomass, with its error | 100 m | 2018 to 2025 | Median by square from 0.2 Mg/ha (Mongolian steppe, Gobi) to 354 (Congo). Its floor is about 3 to 4 Mg/ha on bare sand; it gives tundra about 19 to 32, which looks high |
| Meta canopy height v2, 10 m mean, with imagery date | 10 m | one date, 2012 to 2020s | Used only where the imagery is from 2017 on |
| Impact Observatory annual land use | 10 m | 2018 to 2023 | Agrees with WorldCover trees on 89 % of a forest square |
| Biodiversity intactness v1.1 | 100 m | 2018 to 2025 | Read, not used yet |

## 7. Biomass, height and class from one year's embedding (`scripts/biomass_rows.py`, `biomass_head.py`)

One small head (a layer of 64, fire-knn's size) on one year's embedding: biomass, canopy height and class. Biomass is
taught at CTrees' 100 m: the mean of the head's 10 m values over 8 pixels of a cell is matched to the cell. 500 cells
per square, every year. Every square scored by a head that never saw it (8 folds); then each continent left out.

| Held out by square | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| Biomass, R2 of log, cell-years | .70 | .61 | .74 | .73 |
| Biomass, median error, Mg/ha | 10 | 12 | 9 | 9 |
| Biomass, continent left out, R2 of log | .66 | .53 | .65 | .67 |
| Canopy height vs Meta, R2 / mean error m | .63 / 2.3 | .56 / 2.6 | .72 / 2.0 | .78 / 1.8 |
| Class, share right (WorldCover and IO agree) | .86 | .85 | .92 | .91 |

fire-knn's California volume head, for comparison, ranks with CTrees at .64 overall and backward on crops (-.20),
city edges (-.30) and desert (-.33). World labels are what make the amounts carry.

By kind (held out by square; R2 of log / median error Mg/ha / median predicted, against CTrees' median):

| Kind | CTrees | AEF | OlmoEarth | All three |
|---|---|---|---|---|
| Intact closed forest | 215 | .57 / 54 / 133 | .69 / 51 / 176 | .69 / 46 / 165 |
| Tree-loss squares | 60 | .57 / 21 / 57 | .61 / 18 / 53 | .61 / 18 / 51 |
| Wetland | 17 | .58 / 10 / 21 | .55 / 9 / 14 | .49 / 12 / 16 |
| Savanna | 19 | .27 / 8 / 15 | .29 / 8 / 12 | .28 / 8 / 13 |
| City edge | 16 | .27 / 6 / 16 | .18 / 7 / 14 | .28 / 6 / 16 |
| Crop | 14 | .30 / 5 / 12 | .31 / 5 / 8 | .31 / 6 / 9 |
| Shrub | 5 | .16 / 4 / 7 | .23 / 4 / 6 | .46 / 3 / 5 |
| Grass | 3 | .25 / 3 / 5 | .43 / 3 / 5 | .28 / 2 / 3 |
| Desert edge | 0.6 | -.32 / 2 / 4 | .43 / 1 / 2 | .63 / 1 / 1 |
| Tundra | 19 | -.62 / 17 / 13 | -.14 / 11 / 12 | -.75 / 13 / 9 |

- The amounts carry across the whole range, from desert to rainforest: errors of 1 to 4 Mg/ha on grass, shrub and
  desert, 5 to 10 on crops, cities, savanna and wetland. The low R2 inside those kinds is their narrow range.
- The top is squeezed: cells CTrees puts above 300 Mg/ha come out 205 (AEF) to 252 (all three); intact forest at 215
  comes out 133 to 176. The usual pull toward the middle.
- Tundra fails in every embedding; CTrees' own tundra values (19 to 32 Mg/ha) are the first suspect.
- OlmoEarth is the best single embedding for amounts, height and class here; TESSERA the weakest for amounts. All
  three together help height most.

## 8. Land cover, with its margin of error (`scripts/class_report.py`, `results/class_report.json`)

The class output, held out by square, on 2020 and 2021 pixels where WorldCover and Impact Observatory agree (the easy
pixels; harder ground will score lower). Intervals: 95 %, resampling whole squares.

| | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| Share right | .86 [.82, .90] | .85 [.81, .88] | .92 [.89, .94] | .91 [.88, .93] |
| Calibration error (gap between confidence and share right) | .074 | .064 | .031 | .043 |

Per class, OlmoEarth (recall [interval] / precision): trees .98 [.97, .99] / .97, built .98 / .98, crop .96 [.93, .98]
/ .96, water .96 / .94, grass .89 [.83, .93] / .84, bare .77 [.39, .89] / .76, moss and lichen .74 / .71, shrub .65
[.49, .78] / .75, wetland .37 [.22, .66] / .66. Shrub and wetland are mostly mistaken for grass.

Per pixel: 79 % of pixels get a confidence of .95 or more and are right 98 % of the time. Below that the head is
overconfident: at .8 to .9 it is right 74 %, at .5 to .6 about half. Calibrated (a temperature fit on held-out
squares, not done yet), the confidence becomes a per-pixel margin of error a map can show.

## 9. Biomass change between years, against CTrees' change (`scripts/biomass_change.py`, `results/biomass_change.json`)

Every pixel of every square, every year, by the square-held-out heads, folded to CTrees' cells (341,571 cells).
CTrees itself barely moves from year to year (median 1.1 Mg/ha); its large changes are nearly all in the tree-loss
squares, so elsewhere this is mostly a test of steady ground staying steady.

Tree-loss squares, all three embeddings (AEF in parentheses):

| | Plain difference | Gated | Pair head |
|---|---|---|---|
| 2018 to 2025: slope of predicted on true change (1 = full size) | .63 (.72) | .58 | .55 |
| 2018 to 2025: loss cells found / size against CTrees | .82 / .81 (.83 / .86) | .77 / .75 | .75 / .55 |
| 2018 to 2025: steady cells lit | .28 | .07 | .04 |
| Year to year: loss found / size | .55 / .44 | .45 / .30 | .47 / .33 |
| Year to year: steady cells lit | .14 | .014 | .008 |
| Regrowth (gain cells), 2018 to 2025: found / size | .33 / .08 | .12 / 0 | .08 / .04 |

- Loss over several years comes through at about four-fifths of its size; a single year at under half (muted, as in
  fire-knn). Regrowth barely shows: CTrees' gains are slow and the heads are flatter still.
- The frozen tree-loss gate (count change only where the embedding moved) keeps steady ground quiet (14 % to 1.4 %
  a year) for a small loss in what is found, and it drops regrowth entirely (gradual change never passes the cut).
- The pair head (two years in, change out) is the quietest and correlates best over 2018 to 2025 (.69 against .49),
  but it is not larger: the muting is not fixed by teaching on change.
- The calibration tried here (divide by the muting measured on other squares, pooled over all windows) overshoots on
  the long span (slope 1.43) and lights 29 % of steady cells: a calibration per window length is the fix to try.
- Outside the tree-loss squares, steady cells lit by plain difference: 3.7 % a year, 8.9 % over 2018 to 2025. The
  highest is wetland: 8.5 % a year, 20 % over the span (AEF 14 %); gated, 0.1 % and 1.1 %. The Earthwork wetland
  trouble (water levels moving the embedding) carries into biomass change, and the gate handles it.

At 10 m, on Hansen's loss pixels in the tree-loss squares (204 square-years): the head falls from 81 to 44 Mg/ha
(year before to year after), while CTrees' 100 m cell over the same pixels falls only from 83 to 64; never-lost
forest holds (78 to 76). The 10 m map shows the clearing at about twice the depth of the 100 m cell it sits in.
AEF, TESSERA and OlmoEarth agree here within a few Mg/ha.

## 10. Earthwork's trouble spots through the biomass head (`scripts/biomass_lookalikes.py`, `results/biomass_lookalikes.json`)

The places that fooled the tree-cut detector (section 2) and Earthwork before it. No forest gate here. Median share
of pixels whose biomass changed by a quarter and by 10 Mg/ha or more, plain (gated):

| Set | AEF | TESSERA | OlmoEarth | All three | Tree-cut detector, AEF ungated (section 2) |
|---|---|---|---|---|---|
| Crop switches (27) | 0 (0) | .035 (.019) | 0 (0) | 0 (0) | .985 |
| Farm lidar (6) | .021 (.013) | .052 (.012) | .029 (.016) | .015 (.003) | .39 |
| Reservoirs (6) | 0 (0) | .087 (.047) | 0 (0) | .003 (.003) | .16 |
| Floods, landslides, river banks, solar, pits (21) | .021 (.021) | .17 (.10) | .12 (.10) | .026 (.023) | .87 |
| Marsh and storm coast (4) | .13 (.10) | .35 (.16) | .13 (.06) | .054 (.026) | .40 |
| Burns (15), real loss | .93 (.87) | .92 (.85) | .92 (.87) | .91 (.82) | |
| Standing forest beside the burns (15) | .55 (.27) | .53 (.09) | .36 (.13) | .43 (.11) | |

- A crop switch moves the embedding but not the biomass head: crops and farms stay put. The 10 Mg/ha floor helps
  here (crops hold 5 to 15), so a crop's own seasonal biomass is below what this counts.
- Marsh is the one that still lights, and most of it is Mexico Beach (Hurricane Michael flattened forest there: real
  loss); the Texas and Florida marshes light 0.3 to 9 % with all three, 2 to 43 % in TESSERA alone.
- TESSERA alone is the noisiest everywhere (radar looks changing; Earthwork's finding 25 holds here too).
- The forest beside the burns lights 36 to 55 %: much of it is probably real (scorch, edge mortality, salvage), but
  not checked.

## The embeddings, side by side (all sections)

| | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| Tree loss: plain change, cut vs standing (1) | .993 | .975 | .987 | |
| Tree loss: regrowth visible after a cut (1) | moderate | clearest | nearly flat | |
| Tree loss, 37 squares: found / standing forest lit (4) | .83 / .06 | .76 / .11 | .74 / .03 | .76 / .03 (vote) |
| Biomass, held out by square, R2 of log (7) | .70 | .61 | .74 | .73 |
| Canopy height, R2 (7) | .63 | .56 | .72 | .78 |
| Land cover, share right (8) | .86 | .85 | .92 | .91 |
| Biomass change 2018 to 2025, tree-loss squares: loss size / steady lit (9) | .86 / .26 | .73 / .40 | .86 / .27 | .81 / .28 |
| Trouble spots lit, worst set (10) | .13 marsh | .35 marsh | .13 marsh | .054 marsh |

AEF dates and finds loss best; OlmoEarth reads amounts, height and classes best and is quietest; TESSERA shows
regrowth best and is the noisiest; together they are the safest on trouble spots. Gaps: AEF has no 2025 in central
Chile; TESSERA lacks much of 2017 (dropped everywhere).

## Next

Biomass and land cover:
- Calibrate the class confidence (temperature on held-out squares) so each pixel carries its margin of error.
- Calibrate change per window length; try a pair head on longer spans, and a head on the whole 2018 to 2025 trajectory
  for regrowth.
- Squeeze at the top: CTrees above 300 Mg/ha comes out near 250 at best. A loss weighted toward the high end, or the
  error layer CTrees gives, to check how much is CTrees' own noise.
- An independent check: repeat-lidar canopy change. Earthwork's lidar is bare-ground only; surface models from the
  same programs (AHN in the Netherlands, swissSURFACE3D, England's NLP, NRCan HRDEM, 3DEP point clouds) or NEON would
  be needed.
- Tundra: check CTrees there against another map before trusting either.
- Labels where WorldCover and IO disagree, to score the classes on hard ground; open woodland vs closed forest as
  their own classes.

Tree loss (from section 4, still open):
- A gate that knows dry and open forest: the volume head's canopy cover, or the gate taught with dry forest
  (Chaco, Mediterranean, savanna woodland) added.
- A lower or per-cover cutoff for sparse forest (larch, cocoa, savanna).
- A look at Sentinel-2 under the flags Hansen does not have (Riau, Nicaragua, Latvia).
- Loss kinds: cut, burned, storm, landslide, river.
- Volume head retrained on global labels (canopy height maps, WorldCover) for amounts outside California.
