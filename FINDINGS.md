# Findings: biomass, land cover and tree loss from yearly embeddings

What the tests in this repo found, November 2026. Sections 1 to 5 are tree loss (the first session); sections 6 to
10 are biomass and land cover over every kind of ground (the second); sections 11 to 16 work through the open items
of the second (margins of error, calibrated change, the top end, regrowth, hard ground, airborne lidar); sections 17
to 26 test the open directions; sections 27 to 30 check the map's head on burn scars and land cover, and how to
draw change with its confidence; sections 31 to 33 test the growth signal, the size of loss against lidar and gain
taught on lidar; sections 34 to 38 check the map against references its head never saw (grass abroad, plot biomass
in tall forest, tundra, a burn flown before and after) and put the lidar-taught gain in the browser. Three 10 m
yearly embeddings, compared on the same pixels:
AlphaEarth Foundations (AEF, 64 numbers), TESSERA v1.1 (128) and OlmoEarth v1.3 (128). Years 2018 to 2025 (2017
is left out: TESSERA lacks much of it and AEF rebuilt it differently). Reference: Hansen Global Forest Change v1.12
(30 m, loss year to 2024), which has its own errors. The pixels in sections 1 to 3 come from Earthwork
([aef-tessera-earthwork](https://github.com/kentstephen/aef-tessera-earthwork)'s `data/`); section 4's squares were picked and read here.

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

## 11. A margin of error per pixel for the classes (`scripts/class_calibrate.py`, `models/class_temperature.json`)

Same pixels as section 8 (2020 and 2021, WorldCover and IO agree, each square by its held-out fold head). Both fixes
are fit across folds: a square's fold is calibrated on the other folds only.

- **Temperature.** One number dividing the logits (1.47 for OlmoEarth, 1.64 all three, 1.78 AEF, 1.60 TESSERA).
  Calibration error falls from .031 to .006 (OlmoEarth), .043 to .006 (all three), .074 to .012 (AEF). After it,
  a pixel at .75 confidence is right 74 % of the time, at .85 right 84 %, at .93 right 92 %.
- **Class sets.** Every class whose calibrated probability passes a cutoff, the cutoff set so the true class is
  inside 90 % of the time. One cutoff for all classes holds 90 % on average but not per class (OlmoEarth: wetland
  37 %, shrub 65 %), because trees fill most pixels. One cutoff per class fixes that:

| 90 % sets, one cutoff per class | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| True class inside the set | .93 | .92 | .95 | .94 |
| Classes per set, mean | 1.37 | 1.55 | 1.13 | 1.17 |
| Pixels with a single class (right) | .71 (.95) | .61 (.93) | .88 (.95) | .85 (.95) |

  OlmoEarth per class, inside the set / mean set size: trees .98 / 1.04, built .98 / 1.02, crop .96 / 1.10, water
  .96 / 1.39, grass .90 / 1.20, shrub .89 / 1.34, bare .89 / 1.20, wetland .88 / 1.71, moss .82 / 1.67. A map
  can show the set: one class where it is sure, two where it is not (wetland and grass, shrub and grass).
- These are the easy pixels; section 15 is the hard ground.

## 12. Change calibrated per window length (`scripts/change_calibrate.py`, `results/change_calibrate.json`)

From the cell tables of section 9 (one-year changes add up, so every window of 1 to 7 years is there). Section 9's
factor was pooled over all windows and overshot the long span. Fit per window length it reaches a slope of 1 at
every length, but loss cells then come out 1.2 to 1.5 times CTrees' size: CTrees' own year-to-year noise on small
changes pulls the fitted slope down, so dividing by it overshoots. The factor that works is the muting measured on
clear loss cells only (CTrees fell by half and 20 Mg/ha), applied to the gated change:

| All three embeddings, tree-loss squares | Plain | Gated | Gated / loss factor |
|---|---|---|---|
| One year: loss size / found | .44 / .55 | .31 / .45 | 1.06 / .65 |
| One year: steady cells lit (tree-loss squares; elsewhere) | .14; .037 | .014; .002 | .049; .024 |
| 2018 to 2025: loss size / found | .83 / .83 | .77 / .77 | 1.02 / .82 |
| 2018 to 2025: steady lit (tree-loss squares; elsewhere) | .28; .089 | .072; .010 | .093; .020 |

The factor by window length (fit on the other folds) runs from about .40 to .53 for one year to .69 to .91 for
seven, over the folds and embeddings. Loss now comes out at full size over any span; the price is that
steady ground inside the tree-loss squares lights at 5 % a year (elsewhere 2 %).

## 13. The squeezed top end (`scripts/top_end.py`, `results/top_end.json`, `scripts/read_cci.py`, `scripts/cci_compare.py`)

The training cells' draw replayed (checked equal to the stored biomass) gives CTrees' own error per cell.

| CTrees bin, Mg/ha | 150 to 250 | 250 to 300 | 300 to 400 | 400 and up |
|---|---|---|---|---|
| CTrees median | 189 | 272 | 331 | 440 |
| CTrees' own error, median | | | 75 | 89 |
| OlmoEarth head | 142 | 203 | 234 | 238 |
| OlmoEarth, quantile mapped | 169 | 250 | 286 | 287 |
| OlmoEarth, taught with weight on the top | 148 | 217 | 217 | 198 |
| All three, quantile mapped | 165 | 252 | 296 | 291 |

- Above about 250 Mg/ha the heads stop separating one level from the next: 300 to 400 and 400 and up come out the
  same. Weighting the loss toward the top does not help (worse above 400). Quantile mapping across folds lifts the
  top by 50 to 60 Mg/ha but costs R2 of log everywhere (.73 to .69).
- 77 to 80 % of the head's misses in the 300 to 400 bin are within twice CTrees' own error.
- **A second map disagrees with CTrees at the top.** ESA CCI Biomass v7.0 (radar, ICESat-2 calibrated, not GEDI;
  read under all 110 squares, 2018 to 2024) puts intact closed forest at 260 Mg/ha where CTrees puts 215, and the
  two rank its cells together at only .35 (the head ranks with CTrees at .80, with CCI at .41). The top end is
  squeezed against a label that another map does not share.

CTrees against CCI against the head (OlmoEarth), median over the training cells and rank correlation:

| Kind | CTrees | CCI (share 0) | Head | CTrees with CCI | Head with CTrees | Head with CCI | Change 2018 to 2024, CTrees with CCI |
|---|---|---|---|---|---|---|---|
| All | 24 | 14 (.39) | 19 | .75 | .87 | .79 | .17 |
| Closed forest | 215 | 260 (.02) | 177 | .35 | .80 | .41 | .01 |
| Tree-loss squares | 60 | 88 (.13) | 54 | .67 | .80 | .69 | .28 |
| Savanna | 19 | 9 (.30) | 12 | .52 | .75 | .63 | .05 |
| Wetland | 17 | 15 (.30) | 14 | .64 | .74 | .63 | .07 |
| Tundra | 19 | 0 (.93) | 11 | .44 | .22 | .18 | -.07 |
| Crop | 14 | 0 (.76) | 8 | .32 | .61 | .33 | .05 |
| Grass | 3 | 0 (.88) | 4 | .17 | .68 | .28 | .07 |

- CCI counts the woody parts of living trees only, so grass, crop, shrub and tundra read 0; that is a definition,
  not a disagreement. Tundra's 19 Mg/ha rests on CTrees alone, and no lidar here covers tundra (NEON's Alaska
  sites need an account token since June 2026).
- The two maps barely agree on change from 2018 to 2024 (rank .17 overall, .28 on the tree-loss squares), and the
  head agrees with CTrees' change about as well as CCI does (.24, .39 on tree-loss squares). Year-to-year change
  in any one map is a weak reference; section 16 uses lidar instead.

## 14. A head on the whole series, for regrowth (`scripts/trajectory_head.py`, `results/trajectory_head.json`)

Eight years of a pixel in at once (each year through a layer of 64, then mixed across neighboring years), biomass
for every year out, taught on CTrees' state and on its change between every pair of years. Same folds, scored on
the same 341,571 cells as section 9. State is as good as the one-year head (R2 of log .73 to .74 on cells).

| 2018 to 2025, tree-loss squares | One-year head (all three) | Trajectory head (all three) |
|---|---|---|
| Regrowth (gain cells): size / found | .09 / .34 | .37 / .61 |
| Regrowth, divided by the loss factor (section 12): size / found | .11 / .37 | .55 / .72 |
| Loss: size / found | .83 / .83 | .67 / .76 |
| Steady lit, tree-loss squares; elsewhere | .28; .089 | .25; .048 |
| Correlation with CTrees' change | .49 | .54 |

- Regrowth comes through at four times the size, and steady ground elsewhere is half as noisy, but loss is more
  muted than with the one-year head; the loss factor brings both to full size. OlmoEarth alone is similar
  (regrowth .24 / .47, divided .43 / .58).
- The forest gate (the embedding moved past the frozen cutoff) removes regrowth for both heads: gradual change
  never passes a cutoff set on clearcuts.

## 15. Classes on hard ground (`scripts/class_hard_ground.py`, `results/class_hard_ground.json`)

Every pixel of the 92 squares in 2020 and 2021 where both WorldCover and IO have a class: they disagree on 25 % of
them (wetland squares 54 %, desert edge 86 %, city edge 43 %, savanna 39 %, closed forest 4 %). The commonest
disagreements: WorldCover trees where IO says rangeland (15 % of them), bare against rangeland (13 %), grass
against crops (9 %), wetland against rangeland (8 %), shrub against trees (8 %).

| OlmoEarth, calibrated | Maps agree | Maps disagree |
|---|---|---|
| Confidence, mean | .92 | .77 |
| Single-class sets | .88 | .70 |
| Classes per set | 1.13 | 1.34 |

- The margin of error widens where the maps disagree, as it should. On disagreed pixels the head takes
  WorldCover's class 51 % of the time, IO's 38 %, another 11 %.
- **Trees, settled by Meta's height** (4.3 million pixels where one map says trees and the other does not; 5 m or
  more = trees, under 2 m = not): WorldCover is right 49 %, IO 51 %, the head 63 % (all three: 63 %). The head,
  taught only where the maps agree, beats both on the pixels where they do not.
- But it is still overconfident there: .78 confidence when right, .73 when wrong, so a pixel at .77 is right
  about 63 % of the time on this ground, not 77 %. A calibration fit on agreed pixels understates the error on
  hard ground; one fit on referee-labeled hard pixels is the fix to try.
- Meta's mean height is a rough referee for WorldCover's tree definition (tree cover, not mean height): open
  woodland with a low mean height counts as not trees here.

## 16. Against airborne lidar (`scripts/read_lidar.py`, `scripts/lidar_check.py`, `results/lidar_check.json`)

The independent check. Canopy height = surface model minus ground model of the same flight, from the national
programs Earthwork reads for bare ground, with the surface product swapped in: LINZ (New Zealand, 1 m), Estonia's
point clouds already on disk in Earthwork (highest return minus ground, 3 x 3 km a site), and NASA WERK's 1 m
canopy height over Los Angeles before (2023) and just after (January 2025) the Eaton and Palisades fires; plus WERK's
statewide 10 m height, one flight per place, at six California squares. Each lidar square is a new 6 km square, so
it is predicted by all 8 fold heads averaged; none was in training. The epoch is paired with the embedding of its
flight year (a survey's middle date). Pixels with lidar on 80 % or more of their 2 m samples.

**Height.** Pooled over 15 squares (6.7 million pixel-epochs):

| | AEF | TESSERA | OlmoEarth | All three | Meta (the label) |
|---|---|---|---|---|---|
| R2 against lidar | .73 | .61 | .66 | .65 | .72 |
| Mean error, m | 2.5 | 3.0 | 2.8 | 2.8 | 2.8 |
| Bias, m | -1.1 | -2.1 | -2.3 | -2.4 | -1.4 |
| Lidar 30 to 45 m reads as | 33 | 22 | 24 | 21 | 27 |

- Against lidar, AEF's height is as good as Meta's, the label it was taught on; against Meta (section 7) OlmoEarth
  and all three looked better. AEF was taught to reconstruct GEDI canopy height, and here, against lidar that is
  neither GEDI nor Meta, that shows. OlmoEarth and all three squeeze tall forest more.
- Inside single squares the agreement drops (R2 .47 to .76 at Glenbervie, Esk Valley, Banks and Lahemaa; near
  zero or below at most California statewide 10 m squares), but Meta drops as much there (.04 to .28 in the
  Sierra, redwoods, oak savanna and chaparral; .71 in the Central Valley): the statewide 10 m product likely measures height differently (its definition
  not checked). Those six squares count for the pooled height only.
- Biomass ranks with lidar height at .68 to .71 (rank correlation, all pixel-epochs).

**Change between flights** (9 squares with two flights; canopy lost = lidar height fell 5 m or more from 5 m or
more, median fall 23 m; gained = rose 3 m or more, median 5.8 m; steady = under 1 m):

| Pooled | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| Canopy lost: found (gated) | .85 (.79) | .89 (.71) | .89 (.84) | .87 (.81) |
| Height fall at lost pixels, head (lidar 22.9 m) | 12.9 m | 11.0 m | 12.7 m | 14.6 m |
| Canopy gained: found | .70 | .78 | .48 | .67 |
| Steady lit (gated) | .23 (.08) | .34 (.08) | .24 (.10) | .21 (.07) |
| Correlation of height change with lidar's, per pixel | .73 | .66 | .66 | .72 |
| **Biomass change against lidar height change, per 100 m cell** | **.82** | | | **.70** |
| CTrees' change on the same cells | .50 | | | .50 |
| ESA CCI's change on the same cells (2024 and before) | .58 | | | .58 |

- Per CTrees cell, the heads' biomass change follows lidar canopy change more closely than CTrees' own change does
  (.82 AEF, .70 all three, against .50), and more closely than CCI's (.58). The head was taught on CTrees' state,
  not its change; differencing it at 10 m and averaging up beats the label's own year-to-year change.
- The size is muted as before: lost canopy reads as about 13 m of a 23 m fall (slope of the head's height change
  on lidar's, .42 to .60).
- The trajectory head (section 14) on the same pixels, judged on biomass alone: lost found .74 (all three), gained
  .53, steady lit .14, correlation with lidar change .67 (one-year head .60). Its regrowth gain over CTrees
  (section 14) shows here as a better correlation and quieter steady ground, not as more gained pixels found (the
  one-year head's count also uses its height output).

By square (all three; AEF where it differs much):

| Square, flights | Lost px: found | Gained px: found | Steady lit (gated) | Cell correlation: head / CTrees / CCI |
|---|---|---|---|---|
| Glenbervie NZ plantation, 2019 / 2024 | 47,855: .96 | 127,595: .68 (AEF .86) | .23 (.02) | .78 (AEF .88) / .49 / .68 |
| Esk Valley NZ, cyclone, 2020 / 2024 | 13,052: .91 | 6,838: .65 | .43 (.15) | .72 / .52 / .49 |
| Banks Peninsula NZ, 2018 / 2023 | 4,543: .95 | 11,152: .47 | .08 (.00) | .82 / .77 / -.32 |
| Lahemaa EE, boreal, 2018 / 2022 | 5,030: .76 (gated .46) | 33,328: .78 | .49 (.01) | .54 (AEF .71) / .07 / .18 |
| Kivioli EE, 2018 / 2022 | 461: .42 | 2,199: .17 | .04 (.02) | -.14 (AEF .36) / .11 / .17 |
| Tartu EE, 2019 / 2023 | 2,138: .42 | 3,955: .35 | .16 (.01) | .34 / .13 / .09 |
| Eaton fire LA, 2023 / Jan 2025 | 4,286: .60 | 335: .02 | .34 (.20) | .27 / .39 / none |
| Palisades fire LA, 2023 / Jan 2025 | 3,764: .42 | 57: .11 | .20 (.12) | .74 / .43 / none |
| Roxburgh NZ, 2021 / 2022 | 4 | 111: .01 | .00 (.00) | -.06 / .17 / .12 |

- Harvest, cyclone slips and storm loss in New Zealand: 91 to 96 % of lost canopy found, cell correlation .72 to
  .88 against CTrees' .49 to .77.
- The Los Angeles fires are the weak case: 42 to 60 % found. The post-fire flight was days after the fires, while
  the 2025 embedding is a whole year of resprouting chaparral; burned houses also count as lost height there.
- Estonia's spring flights at the small sites (Kivioli, Tartu) change little and are read poorly; Lahemaa's loss is
  found plainly (.76) but the gate drops much of it (.46).
- Not read: Canada (HRDEM Fort McMurray, Timmins) and the Netherlands (AHN4 / AHN5 Heuvelrug) did not finish
  in time (full-resolution 0.5 to 1 m surfaces read over HTTP, over an hour each); Mojave has lidar on only 10 %.

## 17. Grass against RAP, a map that is not CTrees (`scripts/pick_us_squares.py`, `read_rap.py`, `map_check.py`, `results/map_check_rap.json`)

Every amount check before this one leaned on CTrees (which uses GEDI, one of AEF's training targets) or on canopy
(CCI, Meta, lidar). The Rangeland Analysis Platform (RAP v3, Univ. of Montana) is a yearly 30 m map of herbaceous
biomass across the lower 48, built from Landsat and weather and taught on BLM AIM and NRCS NRI field plots: not
GEDI, not CTrees. It shares Landsat with the embeddings as an input, not as a target. It counts herbaceous mass only
(annual and perennial forbs and grasses; shrubs and trees are left out), so amounts here compare by rank.

Ten new US rangeland squares, picked from WorldCover as in section 6 (Flint Hills tallgrass, North Dakota mixed-grass,
Pawnee shortgrass, Texas Rolling Plains, San Rafael desert grassland, California foothill annual grass, Wyoming
sagebrush, Snake River sagebrush and cheatgrass, Mojave creosote, Edwards Plateau oak-juniper savanna), plus
sandhills-us and great-basin-us: 12 squares, 2018 to 2025, 43,200 blocks of 100 m. RAP's median is 1.1 Mg/ha; its
year-to-year swings are real (Snake River 0.7 to 0.2 in the 2021 drought and back to 1.0; Great Basin 0.45 to 0.04).

Three things against RAP on the same blocks: the CTrees-taught heads of section 7 (never saw these squares, or
by their held-out fold), CTrees itself, and a head taught on RAP from one year's embedding, each square held out in
turn (same 64-unit head).

| Against RAP, 12 squares | CTrees | CTrees head, all three | RAP head AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|---|---|
| R2 of log, blocks | -3.4 | -2.5 | .45 | **.71** | .63 | .69 |
| Median error, Mg/ha | .90 | 1.18 | .24 | .30 | .32 | **.23** |
| Rank within a square (median of 12) | .10 | -.16 | .68 | .70 | **.71** | .69 |
| Rank of the square medians | **.87** | .60 | .74 | .89 | .85 | .89 |
| Each block's departure from its own mean, correlation | .22 | .04 | .39 | .44 | .35 | **.51** |
| Square's yearly median: correlation (median), squares positive | .40, 12/12 | .13, 7/12 | .49, 11/12 | .41, 10/12 | .28, 11/12 | **.51, 11/12** |
| Year-to-year steps, correlation | .10 | .03 | .33 | .28 | .25 | **.46** |
| RAP swings (25 % of steps) caught the same way | .36 | .40 | .56 | .53 | .49 | **.65** |

- **The CTrees-taught heads do not see grass.** Across squares they rank roughly right (.57 to .72 for single
  embeddings), but inside a square they rank backward (AEF -.20, OlmoEarth -.28), and they do not follow RAP's
  years (departure correlation .02 to .08).
- **The embeddings do.** Taught on RAP and scored on squares it never saw, the head reads herbaceous amounts
  (R2 of log .69 all three, median error 0.23 Mg/ha), ranks within a square at .69 and follows the years: .51 on
  each block's departure, .46 on year-to-year steps, 65 % of RAP's swings caught the same way.
- **The fault is the label.** CTrees itself ranks squares well (.87) but barely moves inside a square (.10) or from
  year to year (.22): a head taught on it learns what it has to give, which for grass is a level per place.
- **Herbaceous ground only** (blocks under 10 % tree plus shrub cover in RAP every year, 37 % of them, where RAP's
  herbaceous mass is most of the living mass): the same. The CTrees heads rank within a square at -.19 to .00 and
  follow the years at .00 to .13; the RAP head at .79 and .51. So the backward ranking is not only woody pixels
  that RAP leaves out.
- TESSERA, the weakest on CTrees amounts (section 7), is the best single embedding for grass amounts here (.71; on
  herbaceous ground .63 against AEF .12). All three together follow the years best.
- Per square (all three, RAP head; correlation of the yearly median with RAP's): Great Basin .94, Snake River .88
  (the 2021 low read as 0.10 against RAP's 0.19), Pawnee .79, San Rafael .66, Sandhills .55, North Dakota .54,
  Flint Hills .48, Rolling Plains .45, Edwards Plateau .40, Mojave .37, Wyoming sagebrush .10, and California
  foothill annual grass **-.85**: there RAP drops in 2025 (3.1 to 2.5) while the head rises. Annual grass peaks
  and cures in spring; a yearly embedding and RAP's yearly production may weigh the season differently.
- Swings are small: no RAP step reaches 10 Mg/ha, so section 9's 10 Mg/ha floor would hide all of them. The rule
  used here (the model moves a third of 0.25 Mg/ha and 10 %) also lights about half of RAP's steady steps, for
  every model, CTrees included: a yearly grass map moves 10 % in most years, and this rule cannot tell a real
  small change from noise. A grass change map needs its own threshold, set against RAP's year noise.
- RAP is US only. A grass head taught on it would need a check abroad before it is used there.

## 18. Classes calibrated on hard ground (`scripts/class_hard_calibrate.py`, `results/class_hard_calibrate.json`)

Section 15 found the class confidence too high where WorldCover and IO disagree. Here a second calibration is fit
on those pixels. Hard pixels: 2020 and 2021, both maps have a class, they disagree, and one says trees while the
other does not (1.8 million kept, 43 % trees). Referee: Meta's mean height, 5 m or more trees, under 2 m not
(between: left out). Only trees against not trees can be refereed this way. Each square by its held-out fold head;
each fix fit on the other folds only. Whether a pixel is hard is known at map time (both maps are public).

| Trees or not, on hard pixels | OlmoEarth before | OlmoEarth hard fit | All three before | All three hard fit |
|---|---|---|---|---|
| Call right | .64 | .64 | .65 | .65 |
| Confidence stated, mean | .84 | .73 | .85 | .73 |
| Calibration error | .20 | .074 | .19 | .068 |
| 90 % sets: truth inside | .70 | .90 | .72 | .90 |
| 90 % sets: one answer | .86 | .40 | .83 | .42 |

- The fix: a temperature of 3.5 (OlmoEarth) or 3.9 (all three) on the logits where the maps disagree, against
  1.5 to 1.6 where they agree; and 90 % sets with one cutoff each for trees and not trees. Saved under
  "hard_ground" in models/class_temperature.json.
- After it, stated confidence is closer but still high in the upper bins (all three: .85 stated, .73 right;
  .97 stated, .87 right). One temperature cannot fix it all; a fit with an offset, or one per region, is the next
  step.
- The honest set is wide: to hold 90 %, more than half of hard pixels get both answers. That is the margin of
  error on this ground.
- Coverage holds for most kinds of square (all three: closed forest .90, savanna .95, wetland .92, city edge .92,
  grass .99) but not shrub squares (.81) or crop (.86). On shrub squares the head is right only 37 % of the time
  on hard pixels: worse than a coin, so the trees call there runs the wrong way, not only too sure.
- **Against airborne lidar** (the 18 lidar squares of section 16, none in training; lidar canopy as the referee,
  fits from all 92 squares), on 423,000 hard pixels:

| Lidar referee, hard pixels | OlmoEarth before | after | All three before | after |
|---|---|---|---|---|
| Call right | .55 | .55 | .72 | .72 |
| Confidence stated | .84 | .71 | .85 | .76 |
| Calibration error | .29 | .096 | .14 | .043 |
| 90 % sets: truth inside / one answer | .66 / .85 | .82 / .42 | .80 / .86 | **.90** / .45 |

  With all three embeddings the hard fit holds on the independent referee: 90 % coverage, calibration error .043.
  OlmoEarth alone falls short (.82) and is barely better than a coin on this ground (.55). On agreed pixels the
  lidar referee agrees with the maps-based calibration (right .94, stated .98, error .04), so that one stands.

## 19. Shrub against the SoCal chaparral map (`scripts/read_chaparral.py`, `map_check.py --label chaparral`, `results/map_check_chaparral.json`)

The SoCal annual biomass map (Schrader-Patton, Underwood et al., Dryad doi:10.5061/dryad.qz612jmjt, CC0): yearly
aboveground live biomass at ~30 m over the national forests' ecoregions of southern California, from a random forest
on 766 field plots with Landsat July-August NDVI and the year's precipitation. Not GEDI, not CTrees. Ten squares
under it: the six fire squares inside MTBS perimeters and the steady one (`pick_us_squares.py`), and three lidar
squares (Eaton, Palisades, Los Padres).

The map itself, first:
- **Its 2022 and 2023 fall on every square, burned or not**: the steady square 5.9 to 3.7 to 1.4 Mg/ha, Los Padres
  7.8 to 3.6, while 2023 was a wet year. The later-added years do not join the earlier ones. Only 2018 to 2021
  are used below.
- **It barely sees fire.** On pixels where the embedding moved past the frozen cutoff in the fire's step, Holy
  drops 9.9 to 6.1 and is back to 9.1 two years later (chaparral takes a decade or more); Woolsey's and Valley's
  burned pixels hardly move (2.2 flat; 5.8 to 5.1); Bobcat's rise the year after (14.0 to 15.1). CTrees shows the
  fires more plausibly (Holy 17.4 to 10.8, Fairview 20 to 13, slow regrowth after). So this map checks shrub
  amounts and steady years, not fire change.
- Its level is about half of CTrees' (median 6.7 against 14 Mg/ha; Eaton and Palisades 8 to 11 against 39 to 40).

| Against the chaparral map, 2018 to 2021 | CTrees | CTrees head AEF | TESSERA | OlmoEarth | All three | Map head, all three |
|---|---|---|---|---|---|---|
| R2 of log, blocks | -1.5 | -1.0 | -.12 | .01 | -.22 | **.64** |
| Median error, Mg/ha (map median 6.7) | 7.7 | 9.1 | 5.2 | 5.1 | 5.8 | **1.9** |
| Rank within a square (median of 10) | .39 | .36 | .66 | .58 | .58 | **.78** |
| Rank of the square medians | .33 | .31 | **.81** | .71 | .73 | .76 |
| Each block's departure from its own mean | .29 | .22 | **.56** | .20 | .23 | .54 |
| Year-to-year steps, correlation | .29 | .11 | .55 | .30 | .15 | **.57** |
| Map swings (5 Mg/ha and 25 %) caught / steady lit | .24 / .09 | .44 / .49 | .77 / .57 | .56 / .50 | .40 / .45 | .77 / .20 |

(Map head: taught on the map, each square held out; single embeddings .57 to .65 R2 of log.)

- **Shrubs are not grass.** Here the CTrees-taught heads do rank inside a square (.58 all three, .66 TESSERA),
  better than CTrees itself (.39): the head's 10 m pixels separate chaparral stands that CTrees' 100 m cells blur.
  Their level is about twice the map's, as CTrees' is.
- Taught on the map, the head takes the map's level (median 6.4 against 6.7, error 1.9 Mg/ha) and follows its
  years (steps .57, swings caught .77 with steady ground lit .20). It also learns the map's habits, including its
  weak fire response, so this is agreement with the map, not proof of the fires.
- TESSERA again follows the years best among the CTrees heads (.56 on departures, .55 on steps), as it shows
  regrowth best in section 1.
- Which level is right, CTrees' or the map's, these tests cannot say. The map was fit on shrub plots; CTrees
  leans on GEDI, whose canopy heights over low shrub are least sure. Lidar canopy height and cover under the same
  stands (NASA WERK statewide 10 m covers them) with a shrub allometry would be one referee.

## 20. Grass, the open directions (`scripts/grass_tests.py`, `results/grass_tests.json`)

Four tests after section 17, on the same 12 RAP squares and 100 m blocks.

**A change threshold for grass, set against RAP's year noise.** RAP's own grass moves a lot from year to year:
the median step is 21 % of the year before, the 90th percentile 92 %, and only 17 % of steps are steady (under
5 %). For each model, the smallest relative step r (with at least 0.05 Mg/ha) that lights steady ground at most
10 %, and RAP's swings (0.25 Mg/ha and 25 %) caught the same way at that r:

| Model | r for 10 % steady lit | Mg/ha at the model's median | RAP swings caught | Swings over 50 % caught |
|---|---|---|---|---|
| CTrees | .83 | 1.15 | .07 | .10 |
| CTrees head, all three | .85 | 1.50 | .06 | .11 |
| RAP head AEF | .44 | .32 | .29 | .42 |
| RAP head TESSERA | .44 | .45 | .26 | .40 |
| RAP head OlmoEarth | .56 | .52 | .16 | .25 |
| RAP head, all three | .48 | .45 | .26 | .41 |

- A grass change map needs a step of about half the year before (r .44 to .48) to keep steady ground quiet; at
  that cutoff the RAP heads catch a quarter of RAP's swings and about 40 % of the large ones. The CTrees heads
  need r .85 and then catch almost nothing (.06).
- Looser cutoffs light steady ground for every model, the label included: at r .25 the RAP head (all three)
  catches .49 of swings with steady ground lit .27. Grass years are noisy in the label as well as the heads, so a
  grass change map is a map of large swings, or of several years' trend, not of single small steps.

**California annual grass** (foothill-annual-ca, where the RAP head ran opposite to RAP, -.85). RAP's herbaceous
mass there is 77 to 96 % annual grass. RAP's annual grass is highest in 2022 (3.4 Mg/ha) and falls in 2024 and
2025 (2.6, 1.9), with annual grass cover 63 % in 2022 and 37 % in 2025, while its perennial grass rises (0.1 to
0.5 Mg/ha). The head's year-to-year departures follow RAP's perennial grass (.28), not its annual grass (-.02), and
run against bare ground (-.28) and litter (-.17) cover. Taught on squares that are mostly perennial grass, the head
reads the green, lasting cover a yearly embedding sees, while RAP's annual grass is a spring crop that has cured by
summer. It also sets the level low (1.2 against RAP's 3.2 Mg/ha). Annual grassland needs its own teaching, or a
seasonal embedding.

**A two-part head.** One head, two outputs: total biomass taught on CTrees (all 92 squares, the cell-mean loss of
section 7) and herbaceous biomass taught on RAP (the 12 squares). The 8 square folds of section 7; the 10 new RAP
squares dealt to the folds in turn. Three fits per fold with the same code, scored on held-out squares only:

| | CTrees only, all three | RAP only | Both | CTrees only, OlmoEarth | RAP only | Both |
|---|---|---|---|---|---|---|
| CTrees, R2 of log, all cells | .729 | | .726 | .724 | | .720 |
| CTrees, R2 of log, grass squares | .38 | | .29 | .42 | | .49 |
| CTrees, R2 of log, shrub squares | .56 | | .38 | .36 | | .36 |
| RAP, R2 of log | | .61 | .68 | | .62 | .71 |
| RAP, rank within a square | | .74 | .70 | | .63 | .70 |
| RAP, block departures | | .52 | .45 | | .34 | .44 |
| RAP, square years positive | | 11/12 | 11/12 | | 9/12 | 12/12 |
| RAP, swings caught | | .60 | .57 | | .59 | .64 |

- Adding RAP's target keeps CTrees' overall score (.726 against .729; OlmoEarth .720 against .724), and the RAP
  side stays near the RAP-only head (within a square .70 against .74; section 17's leave-one-out head .69).
  OlmoEarth gains on the RAP side when the targets are shared (departures .34 to .44, every square positive).
- Inside the kinds the trade shows: with all three embeddings, CTrees' grass squares fall .38 to .29 and shrub
  squares .56 to .38; with OlmoEarth, grass squares rise .42 to .49. One shared hidden layer of 64 serves both;
  a wider layer or separate last layers are the next things to try.
- So one head can carry both, the woody total from CTrees and the herbaceous part from RAP, at little cost overall.

**The RAP head abroad.** A RAP head taught on all 12 US squares, run on the 7 world grass, 7 shrub and 2
desert-edge squares of section 6, beside CTrees:

| Square | RAP head, all three, yearly medians 2018 to 2025 (Mg/ha) | CTrees |
|---|---|---|
| Mongolian steppe | .45 to .83 | .1 to .3 |
| Kazakh steppe | .36 to 1.16 | 1.2 to 1.8 |
| Serengeti | .59 to 1.10 | 6.1 to 7.0 |
| Pampas | 1.35 to 1.97 | 16 to 18 |
| Mitchell grass, Queensland | .09 to .53 | 2.5 to 4.5 |
| Tibet | .50 to .60 | 4.5 to 4.9 |
| Karoo | .18 to .42 | 11.7 to 12.7 |
| Patagonia | .38 to .68 | .5 to 1.0 |
| Chihuahua | .42 to .98 | 1.7 to 2.1 |
| Sahel | .29 to .50 | .5 to 1.0 |
| Mulga, central Australia | .06 to .27 | 13.4 to 14.1 |
| Caatinga | 1.00 to 1.42 | 11.6 to 12.4 |
| Sahara edge | .02 to .09 | 3.7 to 4.0 |
| Gobi edge | .23 to .46 | .1 to .3 |

- The head's grass levels abroad are in the range RAP gives at home (0 to 2 Mg/ha) and order the dry end
  sensibly (Sahara .04, Mulga .1 to .3, Pampas and Caatinga highest). They rank with CTrees across squares at
  .24 (all three) and .60 (TESSERA), but the two measure different things: CTrees counts the shrubs and trees in
  Karoo, Mulga and Caatinga (12 to 14 Mg/ha), and its Sahara edge reading of 3.7 Mg/ha is the bare-sand floor of
  section 6.
- Inside a square it does not rank with CTrees (mostly near 0), as at home: CTrees does not see grass inside a
  square.
- Four droughts were checked, each expected with moderate confidence only: the 2022/23 Pampas drought (2023
  against 2019 and 2024), and 2019 against the wet years 2021 and 2022 in the Karoo, Mitchell grass and Mulga.
  None came out lower in the expected dry year (head dry over wet 1.18, 1.37, 3.23, 1.58; TESSERA 1.10 to 1.75).
  CTrees is flat in all four (.95 to 1.05). Either the head does not carry grass years abroad, or the expected
  years are wrong for these squares (northwest Queensland had a large monsoon flood in February 2019, for one).
  A label abroad is needed to say which: national rangeland maps or field plots; none was read here.
- Mulga falls steadily from 2022 (.27 to .06), with CTrees flat.

RAM peaks: two-part 6.8 GB (one input per run; a first run with two inputs stopped itself at 8.2 GB), abroad
4.5 GB, the rest under 2 GB.

## 21. Classes on hard ground, the next steps (`scripts/class_hard_tests.py`, `results/class_hard_tests.json`, `models/class_hard_tests.json`)

Section 18 left four directions: a fit with an offset, a fit per region, shrub against grass with lidar as the
referee, and open woodland as its own class. Same hard pixels as section 18 (1.8 million Meta-refereed trees-against-
not pixels where WorldCover and IO disagree), each square by its held-out fold head, every fit on the other folds
only. The 18 lidar squares are the outside referee, with 40,000 pixels kept per square, so the lidar numbers differ
a little from section 18's. Held-out logits are saved to data/heads/class_hard_logits_<input>.npz. Peak RAM 3.6 GB.

**An offset fit.** P(trees) = sigmoid(a * L + b), where L is the head's log odds of trees (a = .17, b = -.30 for
OlmoEarth; .16 and -.34 for all three).

| Trees or not, hard pixels | OlmoEarth temperature | OlmoEarth offset | All three temperature | All three offset |
|---|---|---|---|---|
| Stated / right (Meta referee) | .73 / .65 | .66 / .65 | .73 / .67 | .66 / .66 |
| Calibration error | .074 | **.009** | .068 | **.007** |
| Top bins, stated to right | .85 to .69, .97 to .93 | .84 to .83, .96 to .97 | .85 to .73, .97 to .87 | .84 to .83, .96 to .88 |
| 90 % sets: truth inside / one answer | .90 / .40 | .90 / .41 | .90 / .42 | .90 / .42 |
| Lidar referee: stated / right | .71 / .63 | .66 / .63 | .77 / .71 | .70 / .72 |
| Lidar referee: calibration error | .085 | .024 | .054 | .044 |
| Lidar referee: 90 % sets inside / one answer | .84 / .41 | .85 / .43 | .90 / .44 | .90 / .50 |

- The offset closes the gap section 18 left in the upper bins. Stated confidence now matches the share right on
  the Meta referee (error under .01). On the lidar referee the error falls to .024 (OlmoEarth) and .044 (all
  three). The sets keep their 90 %, and against lidar with all three they give one answer on half the pixels.
- OlmoEarth alone still falls short against lidar (.85 inside); all three hold .90.
- On agreed pixels the offset is too modest (lidar: stated .87, right .92). It is for hard pixels only, as the
  temperature was. Fits for maps are in models/class_hard_tests.json.

**One fit per continent** (continent known at map time; fit on the other folds' pixels of that continent). Worse,
not better. Held out, coverage falls to .87 (OlmoEarth) and .88 (all three), calibration error rises to .065 and
.036, and shrub squares fall further (.63 and .54 inside). It is worse on the lidar referee too (OlmoEarth error
.13). A continent's other squares do not predict a held-out square's bias. With the single offset fit, coverage per
continent is already .83 to .92 for OlmoEarth (Oceania lowest) and .89 to .93 for all three. Shrub squares stay
short under every fit (.78 to .81): their trees call runs the wrong way (section 18), and no recalibration fixes
that.

**Shrub against grass, lidar as referee.** On the lidar squares, pixels where WorldCover says shrub or grass or IO
says rangeland. IO's rangeland covers both, so IO cannot be scored. Lidar mean height 0.5 to 5 m = shrub, under
0.3 m = grass (418,000 pixels, 43 % shrub). A p90 rule (p90 1 to 5 m = shrub, under 0.3 m = grass) gives the same
picture.

| Shrub or grass, lidar referee | WorldCover | Head, OlmoEarth | Head, all three |
|---|---|---|---|
| Right, on WorldCover's shrub or grass pixels | .75 | .78 | .77 |
| Right on shrub / on grass (all pixels) | | .52 / .97 | .50 / .97 |
| Stated confidence (right / wrong) | | .91 (.92 / .86) | .95 (.96 / .90) |

- The head is about as good as WorldCover (.78 against .75) but leans to grass: right on grass 97 % of the time,
  on shrub only half. It is too sure: .91 stated, .78 right.
- By square, the head beats WorldCover in southern California chaparral (Los Padres .69 against .50, Palisades
  .81 against .76, Eaton .95 against .91). It loses in Sierra and San Joaquin oak woodland with a grass floor
  (Sierra TEAK .55 / .35 and SJER .38 / .40 for OlmoEarth / all three; WorldCover .24 and .43). Both are poor
  there, because the lidar's 0.5 to 5 m mean often catches oak crowns over grass. New Zealand and Estonia: both
  .76 to .90, close.
- Shrub and grass is where the class margin of error is least honest. A calibration fit on lidar-refereed shrub
  and grass pixels would be the same move as the offset fit. But only 14 lidar squares have shrub or grass, so it
  would be fit and checked on the same handful of places.

**Open woodland as its own class.** Rule: agreed trees, shrub or grass pixels (2020, 2021) whose Meta mean height
(imagery 2017 on) is 2 to 5 m become class 9, open woodland. That is 5.6 % of agreed pixels (9.6 % of agreed trees,
6.3 % of shrub, 2.8 % of grass). One 10-class OlmoEarth head, otherwise as in section 7. The all-three head hit the
6 GB cap and was stopped.
- The head barely learns the class: recall .04, and it still calls most of those pixels trees. A mean height of 2
  to 5 m at 10 m does not mark a kind of ground the embedding separates. It marks sparse or short trees inside
  forest as often as woodland.
- The other classes are unchanged within run-to-run noise (trees .97, crop .95, grass .88). Wetland moves between
  .37 and .61 across identical reruns of the 9-class head, so its change here is noise.
- On the hard trees-against-not pixels it helps a little: right .665 (open woodland counted as not trees), against
  .640 to .645 for 9-class heads retrained with two other seeds, at a lower raw confidence (.70 against .80). It
  calls open woodland on 7.6 % of hard pixels, and the referee says a third of those are trees.
- Still open: a class from a better rule, such as canopy cover at 100 m or lidar-refereed woodland.

## 22. Change for the map: loss from one head, gain from another, every pairing (`scripts/change_combined.py`, `results/change_combined.json`)

Section 12 gave loss at full size from the gated one-year head. Section 14 gave regrowth from the trajectory head,
which the clearcut gate removes. Here loss and gain come from separate sources and every pairing is scored, so the
map's embeddings can be picked by evidence.
- **Loss source X:** the gated one-year head of AEF, TESSERA, OlmoEarth or all three. Its negative part is divided
  by the muting on clear loss cells.
- **Gain source Y:** the trajectory head where one exists (OlmoEarth and all three; none was trained for AEF or
  TESSERA), or the one-year head of any input (its rise).
- A pairing reads X's and Y's embeddings.

One pass over the 92 squares scores each square by its held-out fold heads. The cell tables match sections 9 and 14
to a correlation of .99 to 1.00. Every factor and cutoff is fit on the other folds. Scored against CTrees' cells as
in section 9, and against airborne lidar on section 16's two-flight squares, using the mean of the 8 fold heads.
Peak RAM 10.5 GB.

**Gain at full size lights steady ground.** The gain factors are small (all-three trajectory head: .12 for one
year, .36 for seven; one-year heads: .15 to .39). Scaling the rise up to CTrees' size scales its noise with it.
2018 to 2025, tree-loss squares:

| CTrees cells, 2018 to 2025 | Reads | Corr. | Loss: size / found | Gain found | Steady lit (elsewhere) |
|---|---|---|---|---|---|
| Loss alone, AEF | aef | .53 | 1.02 / .82 | 0 | .082 (.006) |
| Loss alone, OlmoEarth | olmo | .57 | 1.00 / .86 | 0 | .100 (.010) |
| Loss alone, all three | all three | .56 | 1.02 / .82 | 0 | .075 (.009) |
| Loss alone, TESSERA | tes | .43 | 1.03 / .79 | 0 | .124 (.011) |
| Loss all three + gain trajectory all three, full size | all three | .41 | 1.02 / .81 | .83 | .44 (.18) |
| Loss AEF + gain trajectory all three, full size | all three | .41 | 1.01 / .81 | .84 | .45 (.17) |
| Loss AEF + gain AEF one-year, full size | aef | .34 | 1.02 / .81 | .73 | .48 (.33) |

- "Loss alone" here is the negative part only. It is quieter than section 12's signed gated change, with the same
  loss found. All three, one year: steady lit .035 against .049; seven years: .075 against .093.
- Every full-size gain pairing finds 61 to 84 % of CTrees' gain cells. It also lights 42 to 58 % of steady cells in
  the tree-loss squares and 16 to 43 % elsewhere. The trajectory heads are the quietest gain source; the one-year
  heads' rise is noisier.
- **The gradual gate does not help.** Cutoffs below the clearcut one, on the move from the first year and on the
  move summed over the years, each picked on steady cells, keep nearly the same noise: the embedding moves on
  steady ground about as much as on slow regrowth.

**A floor on gain, at the head's own size.** Gain counts where the gain source's 100 m cell rises by at least τ and
by a quarter of the cell's biomass. τ is the lowest value that keeps steady ground within .01 of the loss-only map,
picked on the other folds. The gain is not scaled up, so it reads as a 100 m layer; in a map, that means averaging
the gain source's change over 10 x 10 pixels before applying the floor. Scaling the floored gain up again by its
mean muting brings the noise back and drops the lidar correlation to .49 to .56.

| Floor rule | Reads | τ (Mg/ha): 1 yr; 7 yr | CTrees, 1 yr: corr / gain found / steady (else) | CTrees, 2018 to 2025: corr / gain found / steady (else) | Lidar per 100 m cell: corr / lost / gained / steady |
|---|---|---|---|---|---|
| Loss AEF alone | aef | | .31 / 0 / .043 (.012) | .53 / 0 / .082 (.006) | .80 / .85 / .29 / .070 |
| Loss AEF + gain AEF one-year | aef | 20; 30 to 40 | .30 / .15 / .076 (.020) | .51 / .14 / .111 (.021) | **.84 / .87 / .41 / .032** |
| Loss AEF + gain trajectory all three | all three | 0; 40 | .31 / .22 / .063 (.016) | .52 / .18 / .123 (.011) | .83 / .87 / .36 / .030 |
| Loss AEF + gain trajectory OlmoEarth | aef + olmo | 0; 20 to 30 | .31 / .17 / .052 (.013) | .53 / .20 / .116 (.014) | .82 / .87 / .28 / .034 |
| Loss all three + gain trajectory all three | all three | 0; 40 | .34 / .23 / .056 (.016) | .55 / .18 / .116 (.015) | .72 / .80 / .36 / .052 |
| Loss all three alone | all three | | .34 / 0 / .035 (.012) | .56 / 0 / .075 (.009) | .69 / .79 / .24 / .080 |
| Loss OlmoEarth + gain trajectory OlmoEarth | olmo | 0; 20 to 30 | .32 / .17 / .047 (.012) | .57 / .26 / .143 (.019) | .75 / .86 / .31 / .15 |
| Loss OlmoEarth alone | olmo | | .32 / 0 / .038 (.011) | .57 / 0 / .100 (.010) | .71 / .84 / .18 / .19 |

Lidar cells are CTrees 100 m cells with at least 40 lidar pixels, on the nine two-flight squares: 25,378 cells.
Lost: mean canopy fell 3 m, from 5 m or more. Gained: mean canopy rose 2 m. Steady: canopy moved under 0.5 m. Loss
sizes stay at 1.00 to 1.06 throughout; gain size reads 0 as a median, because most gain cells stay under the floor.
(In the lidar-cells stage the "loss alone" pools were appended once per pairing: the statistics are the same, only
the cell counts printed were tripled; the script is fixed, not rerun.)

- **The floor buys regrowth cheaply.** Lidar's gained cells are found .29 to .41 of the time (.18 to .29 with loss
  alone), fewer steady cells light (.030 against .070, since the negative part alone is quieter), and the
  correlation rises (.80 to .84). On CTrees' cells the floor finds 14 to 26 % of gain cells, with steady ground
  near the loss-only level.
- **By the evidence, AEF for both parts:** against lidar, AEF loss leads every other loss source per pixel
  (correlation .71 against .52 to .60; lost found .77) and per cell (.80 against .69 to .71). AEF loss with AEF's
  own rise above the floor is the best lidar pairing (.84), and it reads one embedding. On CTrees' cells, OlmoEarth
  and all three are slightly ahead on seven-year loss (.56 to .57 against .53), and the trajectory heads find more
  gain (.18 to .26 against .14); against lidar they are behind, and OlmoEarth's steady ground is the noisiest (.15
  to .19). Read cost (section 25): the AEF-only pairing reads about a fifth of what all three need.
- Regrowth is still the weak half: at most four in ten of lidar's gained cells, and one in seven of CTrees'.

**Ramp limits for the change map.** From a pixel sample of 1.6 million one-year and 227,500 seven-year pixels valid
in all three embeddings, using AEF loss plus AEF one-year gain above the floor. Percentages are of the pixel's own
biomass the year before.
- **Loss, per pixel:** the loss factor (.34 for one year) restores a 100 m cell's loss size, but per pixel it
  overshoots: the median lit one-year loss is -92 %, and about half of lit pixels fall past -100 % (one in eight
  over seven years). The map should cap loss at the pixel's own biomass. Median lit loss: one year -13 Mg/ha
  (trees -77, grass -6.5, shrub -6, crop -7.5); seven years -19 (trees -65); the 10th percentile -141 (trees
  -250) for one year. 5.5 % of pixels show loss in a year (crop 14 %, the noisiest; built 1 %), 14 % over seven.
- **Gain, per 100 m cell:** 1.7 % of cells gain in a year, 1.4 % over seven years. Median gain 30 Mg/ha for one
  year and 56 for seven; 98th percentile 91 and 144. In percent: median 42 to 54 %, 90th percentile 105 to 165 %.
- **Steady ground:** with the gate on loss and the floor on gain, steady ground is exactly zero in 96 % of one-year
  pixels (4 % lit; crop 13 %) and in 98 % of cells. So the white at the center can be the zero itself, fully
  transparent, with the low-alpha band only around it.
- **The plain difference instead** (the one-year AEF head, ungated, unscaled): 90 % of steady ground sits within
  ±21 Mg/ha in a year (trees ±31, grass ±9, shrub ±8, crop ±7.5, bare ±3.4) and ±32 over seven years (trees ±50).
  In percent the band is more even across classes: ±36 to 48 % in a year, ±43 to 66 % over seven. The 2nd and
  98th percentiles are -36 and +31 Mg/ha in a year (-51 and +79 %), and -91 and +47 over seven (-80 and +107 %).
- A percent ramp fits every kind of ground with one setting: loss to -100 %, gain to about +100 to +150 %, and a
  low-alpha band around ±40 % for a plain difference. An Mg/ha ramp needs a range per kind: trees ±150 to 250,
  grass and shrub ±20 to 40.

## 23. The top end and the shrub level against airborne lidar (`scripts/top_end_lidar.py`, `results/top_end_lidar.json`)

Lidar canopy height (section 16) is neither GEDI, CTrees nor Meta. Per CTrees 100 m cell with lidar under 40 or more
of its pixels: the cell's lidar mean height, against CTrees and ESA CCI in the lidar's year and against the heads
(square-fold heads averaged; no lidar square was in training) on the embedding of that year. 11 lidar squares with
woody cover, 36,539 cells. Lidar height alone cannot set a biomass level (that needs an allometry); it can say which
map orders the cells the way canopy does, and where each one stops rising.

**The top end.**

| Rank correlation with lidar mean height | CTrees | CCI | Head AEF | OlmoEarth | All three | All three, top-weighted |
|---|---|---|---|---|---|---|
| All cells | **.80** | .72 | .75 | .79 | **.80** | .77 |
| Cells 20 m and taller | .62 | .38 | .50 | .58 | **.65** | .64 |
| Cells 30 m and taller (mostly the redwoods) | .26 | .20 | -.17 | .08 | .25 | **.34** |

Median biomass, Mg/ha, by lidar mean height:

| Lidar height | 10-15 m | 15-20 | 20-25 | 25-30 | 30-40 | 40-50 | 50 m + |
|---|---|---|---|---|---|---|---|
| CTrees | 102 | 114 | 170 | 247 | 331 | 363 | 393 |
| CCI | 136 | 189 | 245 | 295 | 336 | 370 | 443 |
| Head, all three | 67 | 125 | 150 | 181 | 241 | 244 | 279 |
| Head, all three, top-weighted | 67 | 141 | 178 | 218 | 268 | 276 | 325 |
| Head, OlmoEarth | 61 | 130 | 150 | 181 | 225 | 227 | 230 |

- Neither map saturates against lidar height: CTrees and CCI both keep rising to the tallest cells (50 m and more:
  393 and 443 Mg/ha). The squeeze of section 13 is in the heads: OlmoEarth's stops near 230 Mg/ha above 30 m,
  all three near 245 to 280, while lidar height keeps rising.
- CTrees orders tall forest by height better than CCI: .62 against .38 on cells 20 m and taller, .26 against .20
  above 30 m. CCI reads more per meter in mid-height forest: its cells at 200 Mg/ha or more have a lidar median of
  25 m, CTrees' 33 m. The largest gap is Glenbervie's plantation pine (lidar 10 m mean, 14 m p90): CCI 214, CTrees
  63, the heads 97 to 124.
- CCI reads 0 under short woody cover that lidar measures: Tartu (5 m mean height, CCI 0, CTrees 41), Banks
  Peninsula, SJER oak savanna. Its definition counts trees only, so short stands drop out.
- The top-weighted head (section 13) helps here: it ranks the cells above 30 m best of all (.34) and lifts the top
  band from 279 to 325 Mg/ha without losing rank over all cells (.77). AEF alone reads the redwoods highest (341,
  against CTrees 306 and CCI 318) but ranks inside them backward (-.17).
- On this evidence CTrees is the better guide to the top end for rank; nothing here says CCI's higher level is
  right or wrong, which needs plot biomass, not height. Redwood cells: lidar median 32.5 m, CTrees 306, CCI 318,
  the all-three head 227, top-weighted 255.

**The shrub level.** Cells on the chaparral lidar squares (Eaton and Palisades, 2023 flights before the fires;
Los Padres, statewide 10 m) where most pixels are WorldCover shrub with a lidar mean height of 0.3 to 6 m: 1,869
cells, median lidar height 1.75 m. The chaparral map is its 2021 layer (the last before its 2022 fall).

| Shrub cells | CTrees | Chaparral map | CCI | Head, all three | Head, OlmoEarth |
|---|---|---|---|---|---|
| Median, Mg/ha | 31.7 | 6.5 | 66 | 18.3 | 19.6 |
| Rank with lidar mean height | .31 | .17 | .41 | .37 | **.53** |
| Median per meter of lidar height | 18.6 | 3.8 | 35.7 | 10.6 | 11.3 |
| Median by height, 0.3-1 / 1-2 / 2-3 / 3-4 m | 15 / 31 / 38 / 29 | 5.6 / 6.4 / 6.7 / 8.4 | 39 / 65 / 71 / 74 | 13 / 18 / 20 / 21 | 14 / 19 / 23 / 25 |

- On these cells CTrees reads about 4.9 times the chaparral map, not twice (section 19's whole squares mix in
  grass and burned ground). CCI reads twice CTrees; the heads sit between the map and CTrees.
- The chaparral map barely follows stand height: 5.6 Mg/ha in stands under a meter, 8.4 in stands of 3 to 4 m,
  and it ranks cells by lidar height worst of all (.17). A stand four times as tall holding about 1.5 times the
  biomass is hard to square with structure. The OlmoEarth head follows height best (.53) and roughly doubles from
  the lowest to the tallest band.
- Which level is right the lidar cannot settle without a chaparral height-to-biomass relation, and none is applied
  here (none known with enough confidence to cite). What it does say: the map's level is nearly flat across stand
  heights, so its low median is at least partly a missing response to structure, not a well-measured low level;
  and no map is consistent per meter (each one's biomass per meter falls as stands get taller, CTrees' most
  steeply above 3 m).
- Caveats: lidar mean height averages 2 m cells and gaps, so it runs below stand top height (p90 ranks the same
  way); the lidar years (2023, and the statewide campaign years) and the map's 2021 differ; NASA WERK notes
  unclassified high returns that can inflate height; WorldCover's shrub class is the only cover filter.

**Tundra.** Nothing on disk can check CTrees' tundra values (yamal-ru 32, noatak-us 18, nunavik-ca 6 Mg/ha):
Meta's height reads 0 m at the median and 90th percentile on all three (0 to 2 % of pixels at 2 m or more), CCI
reads 0, and there is no lidar. Both only say there are no trees, which does not test shrub, moss and lichen mass.

## 24. More airborne lidar, read at 2 m (`scripts/read_lidar2m.py`, `scripts/lidar_check2.py`, `results/lidar_check2.json`)

Section 16 read surface and ground models at full resolution and averaged down, which took over an hour a square.
Here each COG is opened at its overview nearest 2 m (or the server is asked for 2 m): a few centimeters of overview
error matter for bare-ground change, not for canopy height in meters. Five new 6 km squares, none in training,
canopy height = the program's surface model minus its ground model (France: IGN's own canopy height model):

| Square | Source | Flights | Read |
|---|---|---|---|
| Landes, France: pine plantations and clearcuts | IGN LiDAR HD canopy height (MNH), WMS at 2 m | Aug to Oct 2023 | 22 s |
| Devilla Forest, Fife, Scotland: conifer plantation and farms | SRSP Phase 4 and 5, 0.5 m DSM and DTM COGs at their 2 m overview | May to Jun 2018, Jul 2020 | 2 min |
| Timmins, Ontario: boreal forest, the Dome mine, town | NRCan HRDEM 1 m DSM and DTM (Cochrane 2018, Timmins 2024) | 2018 to 2019 (46 % of the square), Jan to May 2024 | 42 s |
| Utrechtse Heuvelrug, Netherlands: forest and heath | AHN5 0.5 m DSM and DTM at 2 m overview | Feb 2023 | 10 min |
| Holmenkollen, Oslo: forest edge, suburbs | Kartverket per-project DOM and DTM, exported at 2 m | Jun to Sep 2019, Feb to May 2024 | 74 s |

Not read: Switzerland (swissSURFACE3D 2018 and 2024 releases minus swissALTI3D; forty-nine 0.5 m tiles a release,
stopped at the 20 minute limit), England (the EA's DSM tiles come only as whole 50 to 70 MB zips per 5 km tile).
AHN4 was read but dropped: its DSM through the 2 m overview reads tall forest at half height (8.4 m where AHN5
reads 17.3 m and Meta over 15 m), so the Dutch square is AHN5 alone.

**Height (all five squares, 2.4 million pixels):**

| Against the new lidar | AEF | TESSERA | OlmoEarth | All three | Meta |
|---|---|---|---|---|---|
| Height R2 (section 16) | .55 (.73) | .63 (.61) | .70 (.66) | **.72** (.65) | .61 |
| Mean error, m | 2.50 | 2.36 | 2.16 | **2.05** | 2.61 |
| Biomass rank against lidar height | .75 | **.78** | .74 | .76 | |

- On these squares the order flips from section 16: all three embeddings read height best (.72) and AEF worst
  (.55); the heads beat Meta's map (.61), the label they were taught on. AEF's drop is the Dutch square: leaf-off
  February lidar, where AEF reads 3.7 m taller than the lidar (R2 .01; OlmoEarth .67).
- By square, all three: Devilla .79, Timmins .84 (2018) and .63 (2024), Landes .58, Heuvelrug .66, Holmenkollen
  .50 and .36 (Meta itself -.17 there).

**Change, three two-flight squares (13,300 lost-canopy pixels, 2,100 gained):**

| | AEF | TESSERA | OlmoEarth | All three |
|---|---|---|---|---|
| Canopy lost: found (gated); section 16 | .38 (.21); .85 | .51 (.25); .89 | .43 (.29); .89 | .45 (.24); .87 |
| Canopy gained: found | .49 | .47 | .34 | .40 |
| Steady lit (gated) | .23 (.01) | .42 (.05) | .29 (.04) | .27 (.01) |
| Biomass change against lidar change, per 100 m cell (CTrees .09) | **.48** | | | -.10 |

- **Devilla, the clean pair** (both flights in summer, two years apart): canopy lost found .69 (AEF) to .84 (all
  three), .63 to .70 gated; steady ground lit .14 to .24, gated .01 to .06; per cell, AEF's biomass change follows
  lidar's at .66 against CTrees' .24. That repeats section 16 on new ground.
- **Timmins and Holmenkollen mix seasons**: a summer flight against a January to May one. Much of the "lost
  canopy" there is likely leaves and snow, not trees: at Holmenkollen the heads' height falls a median 0.8 to 1.4 m
  where the lidar falls 7.2 m, and the gate passes only 9 to 20 % of it; at Timmins 46 to 68 % is found. These two
  pull the pooled numbers down and are not a fair test of change; a pair from the same season is needed there.
- Per cell, AEF's change follows lidar change at .48 pooled against CTrees' .09; all three together do not
  (-.10, Timmins and Holmenkollen).
- The trajectory head (section 14) finds little here (gain .01 to .08): the two-flight squares are short, and
  seasonal loss is not what it was taught.

Memory: the reads stayed under 2 GB; lidar_check2 (a copy of lidar_check.py, which keeps every square's pixels
for the pooled numbers and has no stop) reached 20 GB, over its 12 GB budget.

## 25. Which embeddings for the map: accuracy against what each costs to read (`scripts/embedding_bandwidth.py`, `pair_heads.py`, `pair_scores.py`, `results/pair_heads.json`, `results/embedding_bandwidth.json`)

The map reads embeddings on the fly, so each one costs bandwidth: one 6 km square, one year, as zarr fetches it
(whole chunks; median of Flint Hills, Olympic and Congo): AlphaEarth 21 MB, TESSERA 46, OlmoEarth 57, all three
123. The change map reads two years, so twice that. Heads on two embeddings at once (AlphaEarth + TESSERA,
AlphaEarth + OlmoEarth, TESSERA + OlmoEarth) were taught exactly as the all-three head (same rows, 8 square folds,
epochs). Every input is scored here the same way on the tests of the map's two settings; the singles and all three
reproduce sections 16, 17 and 19 exactly. Gate for a pair: both of its embeddings moved (a majority, as two of three
for all three).

| Input | MB per square-year (per change pair) | Lidar lost found | Lidar gained found | Lidar steady lit, gated | Change vs lidar, per cell | Grass departures (RAP head) | Grass steps (RAP head) | Chaparral departures (CTrees head) |
|---|---|---|---|---|---|---|---|---|
| AlphaEarth | 21 (42) | .85 | .70 | .075 | **.82** | .39 | .33 | .22 |
| TESSERA | 46 (93) | **.89** | **.78** | .076 | .64 | .44 | .28 | **.56** |
| OlmoEarth | 57 (114) | **.89** | .48 | .104 | .72 | .35 | .25 | .20 |
| AlphaEarth + TESSERA | 68 (136) | .87 | .72 | .051 | .80 | .44 | .25 | .32 |
| AlphaEarth + OlmoEarth | 77 (154) | .87 | .45 | **.037** | .81 | .44 | .38 | .35 |
| TESSERA + OlmoEarth | 103 (207) | .87 | .51 | .040 | .67 | .51 | .43 | .30 |
| All three | 123 (246) | .87 | .67 | .068 | .70 | **.51** | **.46** | .23 |

| Input | MB per square-year | Biomass vs CTrees, R2 of log | Height vs Meta | Classes right | Height vs lidar | Grass amounts, R2 of log (RAP head) | Grass rank within square | Chaparral rank within square |
|---|---|---|---|---|---|---|---|---|
| AlphaEarth | 21 | .70 | .63 | .86 | .73 | .45 | .68 | .36 |
| TESSERA | 46 | .61 | .56 | .85 | .61 | .71 | .70 | **.66** |
| OlmoEarth | 57 | **.74** | .72 | .92 | .66 | .63 | .71 | .58 |
| AlphaEarth + TESSERA | 68 | .72 | .69 | .88 | **.77** | **.75** | .74 | .59 |
| AlphaEarth + OlmoEarth | 77 | .73 | .76 | .91 | .66 | .53 | .71 | .64 |
| TESSERA + OlmoEarth | 103 | .71 | .76 | **.92** | .71 | .60 | **.77** | .60 |
| All three | 123 | .73 | **.78** | .91 | .65 | .69 | .69 | .58 |

Mean rank over each setting's seven tests (1 best of 7; wins in brackets):

| Input | Change map | Each-year map |
|---|---|---|
| AlphaEarth | 4.6 (1) | 5.9 (0) |
| TESSERA | 3.7 (3) | 5.1 (1) |
| OlmoEarth | 5.6 (1) | 3.4 (1) |
| AlphaEarth + TESSERA | 3.6 (0) | 3.1 (2) |
| AlphaEarth + OlmoEarth | 3.4 (1) | 3.6 (0) |
| TESSERA + OlmoEarth | 3.9 (0) | **2.9** (2) |
| All three | **3.3** (2) | 4.0 (1) |

- No input wins every test, so on the strict frontier (no other input both cheaper and at least as good on every
  test of the setting) all seven stand, for both settings. The tests pull different ways: woody change against
  lidar favors AlphaEarth, regrowth and shrub years TESSERA, grass years all three, steady ground the pairs with
  OlmoEarth.
- **All three is not clearly best.** For the change map it ranks first on average, by a small margin (3.3 against
  3.4 for AlphaEarth + OlmoEarth at 63 % of the cost), and it loses woody change per cell (.70 against
  AlphaEarth's .82). For the each-year map it ranks fifth: TESSERA + OlmoEarth, AlphaEarth + TESSERA and
  OlmoEarth alone all rank above it.
- What each step up in cost buys, change map (per change pair):
  - AlphaEarth alone, 42 MB: the best woody change against lidar (.82), weakest on shrub years (.22).
  - TESSERA alone, 93 MB: the most regrowth found (.78) and shrub years (.56), but woody change per cell drops
    to .64 and plain steady ground lights most (.34 before the gate).
  - AlphaEarth + TESSERA, 136 MB: keeps most of AlphaEarth's woody change (.80) with TESSERA's regrowth (.72) and
    quieter steady ground (.051); shrub years .32; grass steps lower (.25).
  - AlphaEarth + OlmoEarth, 154 MB: the quietest steady ground (.037), woody change .81, grass steps .38; regrowth
    found lowest of all (.45).
  - All three, 246 MB: grass years best (.51 / .46) and regrowth .67, at woody change .70 and shrub years .23.
- What each step buys, each-year map (per square-year): OlmoEarth alone (57 MB) holds CTrees amounts best (.74)
  and classes (.92); AlphaEarth + TESSERA (68 MB) gives the best height against lidar (.77) and grass amounts
  (.75); TESSERA + OlmoEarth (103 MB) the best grass rank (.77), classes (.92) and the best mean rank; all three
  (123 MB) the best height against Meta (.78) and nothing else.
- Caveats: the grass and chaparral tests are US only (RAP, the SoCal map); lidar is 18 squares; costs are for
  three places and include TESSERA's per-year observation counts, which a map would not need; the change tests
  above use the one-year heads, not the trajectory head or combined loss and gain of section 22.

## 26. Tree loss, the open items of section 4 (`scripts/tree_loss_tests.py`, `results/tree_loss_tests.json`)

The 37 tree-loss squares, the definitions of section 4 (plain change over the frozen cutoff, a gate on the year
before, vote 2 of 3; found within a year of Hansen's year; standing forest and open ground lit). The frozen
detectors are unchanged; each variant below is scored beside them. The squares' held-out fold heads of section 7
give the height gate, so no square is judged by a head that saw it.

**The gate, and dry and open forest.** Variants: the frozen gate; no gate; a height gate (the biomass head's
canopy height the year before, 3 m or more); the gate re-fit on these squares (5 folds of squares, held out:
before-year embeddings of Hansen loss and standing forest against open ground of these and the 55 vegetation
squares); the re-fit gate with the cutoff scaled by Hansen 2000 cover (cut x clip(cover / 60, .35, 1)); the
re-fit gate with the cutoff x .7.

| Vote, median of squares: found / standing forest lit / open lit | Frozen | No gate | Height gate | Re-fit gate | Re-fit, scaled cut | Re-fit, cut x .7 |
|---|---|---|---|---|---|---|
| All 37 | .76 / .033 / .003 | .87 / .053 / .152 | **.81 / .036 / .000** | .78 / .038 / .000 | .80 / .045 / .006 | .88 / .101 / .001 |
| Dry and open (5: Chaco, Manavgat, Mpumalanga, Yakutia, Cote d'Ivoire) | .08 / .000 / .000 | .87 / .033 / .109 | **.73 / .011 / .000** | .27 / .003 / .000 | .46 / .010 / .000 | .48 / .018 / .000 |
| The other 32 | .79 / .045 / .004 | .87 / .070 / .168 | .82 / .039 / .001 | .84 / .043 / .000 | .84 / .050 / .006 | .91 / .141 / .001 |
| Fire squares (3) | .08 / .000 / .000 | .91 / .005 / .190 | .73 / .004 / .000 | .64 / .003 / .000 | .67 / .010 / .000 | .69 / .021 / .000 |

- **The height gate fixes dry and open forest.** "Was there canopy 3 m tall the year before," from the biomass
  head, replaces "did it look like the forests the gate was taught on." On the five dry or open squares it finds
  .73 of the loss against .08, with standing forest lit .011 and open ground 0; over all 37 it finds .81 against
  .76 at about the same standing lit (.036 against .033). Gran Chaco goes from 0 to .96 (AEF .96, OlmoEarth 1.00),
  Manavgat .08 to .73, Mpumalanga .31 to .80, NSW .62 to .90, Maule .76 to .85.
- **Re-fitting the logistic gate does not fix Chaco** (.01): with dry forest among the positives, open dry ground
  still sits beside it. It helps Mpumalanga (.77) and Manavgat (.64). Height is the better gate.
- **The cost in Chaco**: its standing forest moves from year to year in the embeddings. With no gate, TESSERA lights
  99.5 % of it, OlmoEarth 52 %, AEF 16 %; with the height gate the vote lights 36 % (AEF 11 %). The frozen gate
  looked quiet there only because it passed nothing. Dry forest needs its own change cutoff, or the change read
  over more than one year.
- Shifting cultivation loses a little under the height gate (.65 to .52): fallow regrowth under 3 m is cut again.
  Portugal and Galicia too (.82 to .54, .67 to .46): eucalyptus cut young. A lower height (2 m) or the gate only
  where the head's cover is low are the next things to try.

**Sparse forest, by cover band** (Hansen 2000 cover; vote, pooled over the 37 squares; found on loss pixels / lit
on never-lost pixels away from loss):

| Cover | Loss px | Frozen | No gate | Height gate | Re-fit | Scaled cut | Cut x .7 |
|---|---|---|---|---|---|---|---|
| 10 to 30 % | 82,877 | .48 / .031 | .82 / .133 | .58 / .022 | .58 / .024 | .72 / .217 | .66 / .063 |
| 30 to 60 % | 405,127 | .42 / .034 | .70 / .167 | .56 / .087 | .44 / .033 | .52 / .068 | .57 / .088 |
| 60 % and up | 1,806,209 | .76 / .073 | .82 / .088 | .78 / .075 | .80 / .078 | .80 / .078 | .90 / .185 |

- Sparse and middle cover is where the frozen detector loses most (.48 and .42 against .76 in dense forest). The
  height gate lifts both (.58, .56) without more lit at low cover; scaling the cutoff by cover lifts the sparse
  band to .72 but lights 22 % of its never-lost ground. The middle band (30 to 60 %) holds most of the dry-forest
  loss.
- **Yakutia's larch burn** needs a lower cutoff, not a different gate: .01 found under every gate, .42 at cut x .7
  (TESSERA .54, OlmoEarth .58, AEF 0) with standing forest lit .003. AEF does not move there at any cutoff tried.
- Cote d'Ivoire (cocoa land) gains only from a lower or scaled cutoff (.29 to .46 / .48), at standing lit .074
  to .080.

**Loss kinds.** The driver label of WRI and Google DeepMind (1 km, 2001 to 2025) needs a Global Forest Watch API
key for its tiles, so it was skipped. GLAD's forest loss due to fire (Tyukavina et al., 2001 to 2025, public
continental tiles, 30 m; fire = its codes 3 to 5, other = code 1, low-certainty code 2 left out) splits Hansen's
2019 to 2024 loss under the 37 squares: about 395,000 pixels from fire, 2.07 million from other causes. Frozen
detectors:

| Found (pixel-weighted) | AEF | TESSERA | OlmoEarth | Vote |
|---|---|---|---|---|
| Fire-related loss, all | .62 | .55 | .48 | .55 |
| Fire-related loss, without Yakutia | .76 | .67 | .58 | .67 |
| Other loss | .78 | .73 | .65 | .71 |
| Fire in clearing squares (commodity) | .89 | .83 | .58 | .80 |
| Fire in the wildfire squares | .35 | .16 | .27 | .24 |

- Fire as such is not the hard kind: fire used in clearing (Mato Grosso 77 % of its loss from fire, Ucayali 46 %)
  is found as well as other loss or better (.80 against .72 in the commodity squares). The hard kind is wildfire in
  open or sparse forest (Yakutia larch, Manavgat pine), where the frozen gate and the dense-forest cutoff both
  fail; the height gate and the lower cutoff above are the fixes for those.
- Storm, landslide and river loss have no public label read here; the lidar squares (Esk Valley slips) are the
  closest check, in section 16.

**The flags Hansen does not have** (section 4's third failure). Sentinel-2 L2A from Earth Search (no sign-in): the
4 clearest dry- or growing-season scenes a year (Riau June to September, Nicaragua January to April, Latvia May to
September), cloud-masked by the scene classification, NBR = (B8 - B12) / (B8 + B12) with the 2022-on reflectance
offset removed, per-year median; the drop from the year before the flag to the year after.

| NBR drop, median (share over .1) | Flagged, no Hansen loss | Hansen loss, flagged | Steady forest, not flagged |
|---|---|---|---|
| Riau | .17 (.84) | .16 (.73) | -.04 (.06) |
| Nicaragua | .14 (.59) | .18 (.63) | .04 (.26) |
| Latvia | -.02 (.28) | .22 (.70) | -.07 (.02) |

- **Riau's flags are real loss Hansen missed**: flagged pixels outside Hansen's loss drop as much as Hansen's own
  loss pixels (.17 against .16), while steady forest greens. The square is plantation land; short acacia or oil
  palm rotations cleared and replanted between Hansen's passes fit.
- **Nicaragua's are mostly real**: .14 against Hansen's .18, 59 % dropping over .1 against 26 % of steady forest
  (the dry season is noisier there).
- **Latvia's are mostly not stand-replacing**: the median flagged pixel does not drop at all, though 28 % drop over
  .1 against 2 % of steady forest: a minority of real change, likely thinning or partial cuts, and a majority lit
  with no canopy loss. This fits section 4: AEF lit 71 % there, TESSERA 12 %.
- So "standing forest lit" against Hansen overstates false alarms in Riau and Nicaragua; in Latvia it is mostly
  false alarm.

RAM peaks: gates 4.4 GB, fire 3.6 GB, Sentinel-2 0.5 GB.


## 27. The map's head inside burn scars (`scripts/burn_check.py`, `results/burn_check.json`)

The map's head (the one the web map runs, `web/public/models/biomass_aef.json`) on AlphaEarth inside six California
fires, every year 2018 to 2025, against unburned ground 2 to 8 km outside each (no fire of 2015 to 2024 there),
beside CTrees and fire-knn's LANDFIRE-taught volume head on the same pixels. Perimeters from MTBS, and NIFC WFIGS
for the Park Fire (MTBS has no 2024 fires yet). Twelve 256 px chunks inside and twelve outside per fire, every 4th
pixel. 12 minutes, 1.6 GB.

| Fire | Map head, median inside (Mg/ha) | CTrees | fire-knn volume |
|---|---|---|---|
| Caldor 2021 | 70 to 12 (-83 %) | 151 to 89 (-41 %) | -96 % |
| Dixie 2021 | 35 to 10 (-72 %) | 79 to 49 (-38 %) | -96 % |
| Creek 2020 | 24 to 9 | 64 to 48 | -90 % |
| Camp 2018 (Paradise) | 48 to 20, back to 25 by 2025 | 43 to 35, flat after | -70 % |
| Park 2024 | 24 to 17 in 2024 (-29 %), 12 in 2025 | 34 to 32 | -73 % in 2024 |
| Bobcat 2020 (chaparral) | 13 to 10 | 18 to 13 | -65 % |

- The head does not read high regardless of the burn: at scar scale it falls more than CTrees in every fire.
- The Park Fire started in late July 2024, so the 2024 embedding is mostly the months before it; the head's drop
  finishes in 2025. fire-knn's volume head drops harder in 2024 (-73 %). CTrees barely sees the Park Fire.
- Camp's scar regrows in the head (20 to 25 Mg/ha over four years, +25 %); CTrees stays flat after the drop.
- Chaparral (Bobcat) is the weak case, as in section 19.

## 28. Biomass by land cover: Annual NLCD and Esri's annual land cover (`scripts/nlcd_check.py`, `results/nlcd_check.json`, `results/io_check.json`)

The map's head by land cover class and by class change from one year to the next, 2018 to 2024. Annual NLCD
(USGS Collection 1, 30 m COGs on Source Cooperative, `kylebarron/usgs-landcover`) under the 33 test squares in
CONUS; Esri / Impact Observatory annual land cover (10 m, `s3://io-10m-annual-lulc`, one COG per UTM zone, 2017 to
2024) under all 127. Classes read at each 10 m pixel center, kept in `data/nlcd/`, `data/io/`.

| Class change, one year to the next | Map head | CTrees | Chaparral map |
|---|---|---|---|
| NLCD forest to shrub, grass or barren | 14 to 15 Mg/ha (+7 %) | -16 % | -40 % |
| Esri trees to rangeland or bare (127 squares) | 41 to 32 Mg/ha (-13 %) | -3 % | -26 % |
| Esri rangeland or bare to trees | +7 % | 0 % | +18 % |
| Esri trees, rangeland, bare held | about 0 % | about 0 % | |

- Against Esri, worldwide, the head follows the class changes in the right direction and further than CTrees.
  Against NLCD it does not; NLCD's forest flips are mostly thin forest (14 Mg/ha before), so edge flips and label
  noise are a large share of them. Section 27's scars agree with Esri.
- By class (NLCD): shrub reads a little high (head 6.0 Mg/ha, CTrees 4.1, the chaparral map 5.1), grassland low
  against CTrees (5.7 against 7.4), evergreen forest high (241 against 214; Redwood 489 against 308).
- Grass and shrub abroad: the head's levels sit near CTrees on every square (Serengeti 5.9 against 6.9, Karoo 10.9
  against 12.4, Mongolia 0.2 against 0.1); CTrees is a woody label, so this is not a grass check.
- Tundra: Esri files tundra as rangeland, so it says nothing about tundra biomass. The head reads Noatak 18,
  Nunavik 4, Yamal 22 Mg/ha. The Berner / Orndahl Arctic maps are the next check.
- Peaks: NLCD 2.4 GB, 2 minutes; Esri 3.2 GB once the transitions were sampled (the first run stopped itself at
  6 GB pooling every pixel).

## 29. The noise on steady ground, per land cover class (`scripts/change_confidence.py`, `results/change_confidence.json`)

Steady ground for a window a to b: Esri's class held every year from a to b, and CTrees' 100 m cell moved under
10 Mg/ha and 10 %. The head's change there is noise. 127 squares, every window 2018 to 2024, every 31st steady
pixel. 90th percentile of the change's size:

| Ground | 1 year | 6 years | Steady past the loss gate (1 year) |
|---|---|---|---|
| Trees | 33 Mg/ha (28 %) | 47 Mg/ha (42 %) | 3 % |
| Rangeland | 6 Mg/ha (38 %) | 8 Mg/ha (48 %) | 8 % |
| Bare | 2 Mg/ha (32 %) | 3 Mg/ha (52 %) | 6 % |
| Crops | 6 Mg/ha (43 %) | 8 Mg/ha (61 %) | 28 % (37 % over 3 years) |

- In Mg/ha the noise differs by class (trees five times rangeland); in percent it is similar for most classes.
- **Averaging over 100 m hardly lowers it** (trees 33 to 29.5 Mg/ha): the noise is not pixel by pixel but whole
  patches moving together from year to year. The fixed gain blocks of section 22 are not quiet because averaging
  cancels noise; the 20 to 40 Mg/ha floor is what keeps them quiet.
- Real loss stands clear of it: where Esri says trees went to rangeland or bare, the 90th percentile is 54 Mg/ha
  (59 %) in a year and 121 (83 %) over six, and 32 to 66 % pass the gate.
- Real gain mostly sits inside it in Mg/ha: where Esri says ground became trees, 18 Mg/ha in a year and 31 over six,
  against 33 to 47 on steady trees. In percent there is some room (48 to 93 % against 28 to 42 %).
- NLCD's steady classes give the same picture in CONUS (forest 59 Mg/ha at the 90th percentile in a year, at
  175 Mg/ha median biomass; grassland 3.3).

## 30. Drawing change with its confidence: which way works (`scripts/change_confidence_test.py`, `results/change_confidence_test.json`)

A pixel's confidence = the share of steady pixels (section 29's rule) whose change is smaller, from the steady
pixels of other squares (5 folds by square), per window length, and per Esri class where the variant says so.
Every variant scored on the same pixels against Esri's changes (all 127 squares, every window 2018 to 2024: 3.9
million lost, 2.6 million grew, 8.9 million steady) and against airborne lidar (the nine two-flight squares of
sections 16 and 22: per 10 m pixel, lost = canopy fell 3 m from 5 m or more, gained = rose 2 m, steady = moved under
0.5 m; and per CTrees 100 m cell). "Found at 2 % lit": each variant's cutoff set so 2 % of steady ground lights.
7 minutes, 6.8 GB.

**Loss.**

| Loss variant | Esri AUC | Esri found / steady lit at .95 | Lidar AUC | Lidar found / steady lit at .95 |
|---|---|---|---|---|
| The rule now (embedding gate, the head fell) | .717 | .49 / .054 (at its cut) | .836 | .75 / .080 (at its cut) |
| Percent, one table | .788 | .51 / .050 | .856 | .68 / .042 |
| Per class, Mg/ha | .723 | .27 / .051 | .842 | .62 / .042 |
| Per class, percent | .793 | .53 / .051 | .858 | .69 / .045 |
| **Per class, percent, with the embedding's move** | **.808** | .43 / **.011** | **.869** | .67 / **.029** |

- Per class in percent, combined with the embedding move's own confidence (the lower of the two), is the best loss:
  the highest AUC against both references, and at .95 it lights a fifth (Esri) to a third (lidar) of the steady
  ground the rule lights, for a little less found.
- It fixes the rule's worst ground: steady crops lit 13.8 % by the rule, 0.7 % by it; rangeland 6.4 % against 0.9 %.
- Per 100 m cell against lidar, loss AUC .937 against the rule's .941: as good at the cell, finer at the pixel.

**Gain.**

| Gain variant | Esri AUC | Lidar AUC | Lidar found at 2 % lit |
|---|---|---|---|
| The rule now (fixed 100 m block, 20 to 40 Mg/ha and 25 %) | .525 (found .07, lit .019) | .675 (found .36, lit .014) | |
| Percent, pixel / sliding 100 m / fixed block | .635 / .641 / .636 | .665 / .667 / .663 | .18 / .18 / .17 |
| Per class Mg/ha, pixel / sliding / block | .689 / .693 / .688 | .706 / .711 / .707 | .19 / .19 / .18 |
| Per class percent, pixel / sliding / block | .624 / .627 / .622 | .682 / .687 / .682 | .21 / .21 / .20 |

- **The size makes no difference**: pixel, sliding 100 m and fixed blocks score within .01 of each other in every
  variant. Gain at pixel size costs nothing in accuracy.
- Per class in Mg/ha is the best confidence for gain (lidar AUC .71), but no confidence version beats the rule's
  two-part floor at a low false alarm rate against lidar: the rule finds .36 of lidar's gained pixels with 1.4 % of
  steady lit, the best confidence version .19 to .21 at 2 %. Against Esri the order flips (the rule finds .07).
- Combining gain with the embedding move hurts it (lidar AUC .61 to .67): regrowth does not move the embedding much.
- Per 100 m cell against lidar, the rule's pairing still correlates best with canopy change (.54 against .36 to .44
  for the confidence pairings), since a cell mean of confidences is not a size.

**The rule's floor at finer sizes.** The same two-part floor (at least 20 to 40 Mg/ha by window length, and at least
25 % of the year-a biomass) applied to the pixel's own change and to the sliding 100 m mean, not to fixed blocks:

| Gain, two-part floor | Lidar found / steady lit (pixel) | Lidar per 100 m cell: AUC gained / corr (with the rule's loss) | Esri found / lit |
|---|---|---|---|
| Fixed 100 m block (the map now) | .363 / .014 | .774 / .537 | .070 / .019 |
| Sliding 100 m | .369 / .014 | .790 / .539 | .071 / .019 |
| **Per pixel** | **.387 / .015** | **.858 / .544** | .080 / .025 |

- **Per pixel is the finest and the best**: it finds more of lidar's regrowth for the same steady ground lit, and
  per 100 m cell it separates gained cells far better (.86 against .77). The blocks were costing gain, not
  quieting it (section 29: averaging does not lower the noise).
- With the per class, percent, with the embedding's move loss, the per-pixel gain keeps the same picture (cell AUC
  lost .941, gained .821, corr .531).
- Esri's "grew" (rangeland or bare to trees) finds little under any variant (AUC .52 to .69): Esri's class changes
  into trees are mostly not regrowth a lidar would see in the window, or not visible to the head.

**For the map.** Loss: per pixel, opacity from the per class, percent, with the embedding's move confidence (loss
AUC .87 against lidar; a fifth to a third of the rule's steady ground lit; crops and rangeland quiet). Gain: per
pixel with the two-part floor, opacity from the per class Mg/ha confidence above it. Both at 10 m: no blocks.
Peak RAM 6.9 GB (above the 4 GB estimated: the variant arrays for 15 million rows), 7 minutes.

## 31. The two ways to a stronger growth signal, per pixel (`scripts/growth_test.py`, `results/growth_test.json`)

Section 30's harness, run on seven heads: the map's head (one year of AlphaEarth, taught on all squares), the
one-year heads on AEF, TESSERA and AEF + TESSERA, and the trajectory heads (section 14, all eight years in) on the
same three inputs. Every head is held out (a square scored by the fold that never saw it) and every head is scored on
the same pixels, valid in AlphaEarth and TESSERA both. Gain against the nine lidar squares per 10 m pixel (250,000
gained, 1.3 million steady) and per 100 m cell; loss confidence per class, percent, with AlphaEarth's embedding move,
as on the map. About 22 minutes and 7.3 to 8.0 GB per trajectory head.

| Head | Gain floor, per pixel: lidar found / steady lit | Gain confidence: lidar AUC, found at 2 % lit | Per 100 m cell: AUC gained / corr | Lidar gained: median rise / steady p90, Mg/ha | Steady trees, 1 yr p90, Mg/ha | Loss AUC, lidar / Esri |
|---|---|---|---|---|---|---|
| Map (AEF, all squares) | **.387** / .015 | **.706**, .189 | .822 / .531 | 25.0 / 13.7 | 32.7 | .869 / **.808** |
| AEF, one year | .373 / .015 | .699, .135 | **.857 / .565** | 24.1 / 14.2 | 32.4 | .846 / .784 |
| TESSERA, one year | .348 / .017 | .655, .201 | .795 / .480 | 12.9 / 23.0 | 64.0 | **.881** / .750 |
| AEF + TESSERA, one year | .354 / .009 | .663, .213 | .808 / .538 | 19.0 / 14.3 | 41.2 | .868 / .777 |
| Trajectory, AEF | .360 / .012 | .644, .197 | .692 / .524 | 20.9 / 10.3 | **16.5** | .839 / .788 |
| Trajectory, TESSERA | .300 / .014 | .654, .168 | .716 / .460 | 12.7 / 12.8 | 20.7 | .864 / .748 |
| Trajectory, AEF + TESSERA | .289 / **.005** | .670, **.238** | .751 / .501 | 13.5 / **8.8** | 18.8 | .844 / .788 |

- **No head finds more of lidar's regrowth than the map's at the map's floor.** The map's head finds .387 of the
  gained pixels with 1.5 % of steady ground lit; the best other is the held-out AEF head (.373 at the same lit).
- **The trajectory heads are quieter, as section 14 found, but the regrowth shrinks with the noise.** Steady trees
  move half as much in a year (p90 16 to 21 Mg/ha against 32 to 64), but lidar's gained pixels also rise less (AEF
  21 against 24 to 25 Mg/ha). Rise over steady noise: trajectory AEF 2.0, the map 1.8, AEF + TESSERA one year 1.3,
  TESSERA one year 0.6.
- **What the quiet buys is a stricter cutoff.** The trajectory head on AEF + TESSERA lights a third of the map's
  steady ground at the floor (.005 against .015) for three quarters of its found (.289), and its gain confidence finds
  the most at 2 % lit (.238 against the map's .189). The AEF + TESSERA one-year pair is second on both (.009; .213).
- **Per 100 m cell the trajectory heads are worse** (gained AUC .69 to .75 against .80 to .86). Section 14's
  four-times regrowth was measured against CTrees' change, which the trajectory head is taught on; against lidar,
  which neither head saw, it does not carry over. The held-out one-year AEF head is the best per cell (.857 / .565).
- **TESSERA alone is the noisiest** (steady trees p90 64 Mg/ha one year) and shows the smallest lidar rise; it reads
  loss best against lidar (.881) and worst against Esri (.750).
- Loss is a little more muted with the trajectory heads (lidar AUC .839 to .864; per cell .90 to .92 against .91 to
  .96), as in section 14.

**Cost in the browser.** A one-year change view reads two years: 2 x 21 MB per square for AEF, 2 x 68 MB for AEF +
TESSERA. A trajectory head reads all eight years for any change view (8 x 21 = 168 MB for AEF, 8 x 68 = 544 MB for
AEF + TESSERA) and needs a new GPU pass that mixes across years. TESSERA in the browser is a second embedding read
(46 MB per square-year) the map does not do yet.

## 32. Loss size against lidar, gain by window length, gain taught on lidar (`scripts/window_test.py`, `scripts/lidar_gain_head.py`, `results/window_test.json`, `results/lidar_gain_head.json`)

Three checks on the change view with the map's head (one year of AlphaEarth, taught on all 92 squares, none of them
lidar squares). The harness reproduces section 31's map row exactly on the lidar pixels (gain floor .387 found, .015
lit). About 2 minutes and 3.5 GB for `window_test.py`; 8 minutes and 1.5 GB for `lidar_gain_head.py`.

**Loss size.** The map draws a loss's size as the head's fall f blended 2/3 of the way toward tanh(f / loss factor),
capped at the pixel's biomass (`web/src/change.ts`). Against lidar's canopy fall (a share of the first flight's
height) on the forest pixels (5 m or taller) the map draws as loss, nine two-flight squares:

| Rule | Rank with lidar fall, per pixel: all / harvest and storm | Median drawn, harvest and storm (lidar .958) | Drawn at -100 % | Per 100 m cell, lidar fell 25 % or more: median drawn (lidar .933, harvest and storm) |
|---|---|---|---|---|
| Raw fall | **.61** / **.48** | .751 | 0 | .70 |
| Loss factor + cap (the rule before) | .34 / .37 | 1.0 | .65 (.76) | **.97** |
| Soft 2/3 (the map now) | .48 / .47 | .799 | 0 | .76 |
| Soft curve | .29 / .46 | .822 | 0 | .79 |

- **The map's rule keeps partial loss apart about as well as the raw fall** on harvest and storm (rank .47 against
  .48) and much better than the factor with the cap (.37), which draws three quarters of those pixels at -100 %.
- **It draws clear loss short.** Lidar says 96 % of the canopy went (median, per pixel) where the map draws 80 %; per
  100 m cell 76 % against 93 %. The factor with the cap gets the size of clear loss right (97 %) and flattens all the
  rest.
- **The head barely tells partial from full loss.** Within the pixels drawn, the drawn size is nearly flat from 10 %
  to 75 % lidar fall under every rule (the map: .66, .67, .69, .68), rising only above 75 % (.81). The limit is the
  head's fall, not the rule laid on it.
- **On burns lidar is not a referee.** On Eaton and Palisades (flights 2023 and 2025) the pixels drawn as loss have a
  median lidar canopy fall of 8 %, and 36 % of the forest where lidar's canopy fell under 10 % is drawn (2.7 % on
  harvest and storm). Lidar's canopy height counts trees that stand dead, and both the map and lidar miss what the
  other sees (rank .14). Burn severity needs a reference that tells dead from live.
- Lidar finds .71 of its lost forest drawn (.75 harvest and storm, .55 burns).
- Caveat: canopy height fall is not biomass fall. On clear loss both reach about 100 %, so the size comparison is
  fair there; partial fall in height can under- or overstate the biomass that went.

**Gain by window length.** Esri's "became trees" against steady ground, every square and window 2018 to 2024; every
variant on the same pixels (valid in every year of the window). "Trend" is the least-squares slope of the head's
yearly biomass over every year of the window, times its length (Theil-Sen within .005 of it).

| Window | AUC: two years / trend | Found at 2 % lit | Grew: median rise, Mg/ha | Steady: p90 of the change, Mg/ha | Floor: found / lit |
|---|---|---|---|---|---|
| 1 year | .587 / .587 | .016 | 1.7 | 19.4 | .061 / .019 |
| 2 | .616 / .616 | .023 | 2.7 | 22.8 | .077 / .027 |
| 3 | .631 / .632 | .031 | 3.6 | 25.1 | .084 / .028 |
| 4 | .645 / .651 | .043 | 4.2 | 26.3 | .093 / .027 |
| 5 | .660 / .667 | .037 | 4.8 | 27.2 | .083 / .026 |
| 6 | .655 / .658 | .042 | 4.7 | 29.4 | .071 / .024 |

- **Longer windows help a little**: AUC .59 over one year to .66 over five, as the rise grows faster than the steady
  noise. Esri's "became trees" stays a weak signal at every length (a median rise of 2 to 5 Mg/ha against steady
  noise of 19 to 29).
- **A trend over every year of the window adds next to nothing** (+.006 AUC at four and five years) and against lidar
  it is slightly worse (found at 2 % lit .447 against .473 for the two years).
- **Lidar cannot split by window length**: each square's gap is tied to its place. Five years is the two New Zealand
  squares (found at the floor .505), four is Estonia and Esk Valley (.156), one and two years have almost no
  regrowth. Window and place cannot be told apart there.

**Gain taught on lidar.** The only teacher that is not itself a model. Gradient-boosted trees on AlphaEarth at both
flight years and their difference (and the window's length), target: lidar's canopy rose 2 m. Held out by square
(leave one out), scored as section 31 against steady (moved under 0.5 m):

| Square (flights) | Gained px | AUC: lidar-taught / map | Found at 2 % lit: lidar-taught / map | At the map floor's lit: lidar-taught / map floor |
|---|---|---|---|---|
| NZ Glenbervie (2019, 2024) | 148,092 | .899 / .819 | .340 / .421 | .608 / .566 (lit .087) |
| NZ Esk Valley (2020, 2024) | 11,856 | .862 / .771 | .659 / .517 | .704 / .585 (lit .043) |
| NZ Banks (2018, 2023) | 18,129 | .982 / .309 | .804 / .159 | .176 / .006 (lit 0) |
| NZ Roxburgh (2021, 2022) | 376 | .968 / .481 | .548 / .021 | (the floor lights nothing) |
| EE Lahemaa (2018, 2022) | 56,194 | .857 / .670 | .164 / .096 | .183 / .105 (lit .023) |
| EE Kivioli (2018, 2022) | 4,126 | .848 / .718 | .605 / .198 | (the floor lights nothing) |
| EE Tartu (2019, 2023) | 10,033 | .781 / .458 | .172 / .036 | (the floor lights nothing) |
| US Eaton (2023, 2025) | 992 | .564 / .408 | .044 / .019 | |
| US Palisades (2023, 2025) | 193 | .691 / .498 | .176 / .042 | |
| **Pooled** | 249,991 | .964 / .785 | **.671** / .473 | **.597** / .387 (lit .015) |

- **On every held-out square the lidar-taught head ranks regrowth above steady ground better than the map.** At the
  map's own false alarm rate (1.5 % of steady lit) it finds six in ten of lidar's regrowth pixels against four in ten.
  On Banks the map's rise runs against lidar's (AUC .31); the lidar-taught head reads it at .98.
- Adding the map's biomass and change to the inputs does not help (pooled found at the floor's lit .567 against .597).
- **It does not carry over to Esri's "became trees"** on the other squares: AUC .58 against the map's .61, found at
  2 % lit .020 for both. Either Esri's label is the weak referee section 31 found it to be (AUC .52 to .69 for every
  head), or the head learned what canopy rise looks like in New Zealand and Estonian forest, where all its regrowth
  comes from, and not elsewhere. Nine squares in three countries; Glenbervie holds 59 % of the gained pixels.
- The pooled AUC (.96) is above every square's because it also ranks whole squares against each other; the per-square
  numbers (.78 to .98 on the seven with regrowth) are the fair ones.
- Not tested yet: whether it sees the rise or predicts it. Each year's embedding is an input, so it may learn where
  canopy tends to rise (young plantation) as much as the rise itself; a difference-only input would tell.

## 33. Gain taught on lidar: does it see the rise or know where canopy rises? (`scripts/lidar_gain_head.py --variants`, `results/lidar_gain_diff.json`)

Section 32's lidar-taught gain head gets both years of AlphaEarth and their difference, so it could rank regrowth
from what the place is (young plantation, a cut two years back) as much as from the change. The same trees, the same
held-out squares, two more inputs: the difference alone (e_b - e_a: it can only see the change) and the first year
alone (e_a: it can only know where canopy tends to rise). Every variant also gets the window's length. 5 minutes,
1.0 GB.

| Input | Pooled AUC | Found at 2 % lit | At the map floor's lit (1.5 %) | Esri "became trees": AUC / found at 2 % lit |
|---|---|---|---|---|
| Both years and the difference (section 32) | .964 | **.671** | **.597** | .581 / .020 |
| The difference alone | .822 | .373 | .346 | .577 / **.042** |
| The first year alone | .910 | .342 | .297 | .539 / .015 |
| The map's rise, for reference | .785 | .473 | .387 (the floor) | **.614** / .020 |

Per square, AUC (found at 2 % lit):

| Square | Both | Difference alone | First year alone |
|---|---|---|---|
| NZ Glenbervie | .899 (.340) | .788 (**.367**) | .789 (.140) |
| NZ Esk Valley | .862 (.659) | .707 (.445) | **.866** (.350) |
| NZ Roxburgh | .968 (.548) | .658 (.040) | .964 (.476) |
| NZ Banks | .982 (.804) | **.315** (.014) | .900 (.291) |
| EE Lahemaa | .857 (.164) | .757 (.062) | .834 (.087) |
| EE Kivioli | .848 (.605) | .666 (.116) | .840 (.585) |
| EE Tartu | .781 (.172) | .600 (.040) | .773 (.148) |
| US Eaton | .564 (.044) | .494 (.027) | .552 (.051) |
| US Palisades | .691 (.176) | .561 (.042) | .643 (.052) |

- **Both, and neither alone is enough.** Each input alone finds about half of what the full head finds at the map's
  false alarm rate (.35 and .30 against .60), so the full head uses the change and the place together.
- **The ranking inside a square comes mostly from the place.** The first year alone ranks regrowth above steady
  ground nearly as well as both years on seven of nine squares (AUC within .05). Where canopy will rise is
  written in the first year: young stands, recent cuts, scrub on its way up.
- **Banks is all place.** The difference alone reads it backward (AUC .31), the first year alone at .90. The head
  knows which Banks pixels are the kind that grow, not that they grew; section 32's .98 there is not a sighting of
  the rise.
- **Where the rise is large the difference sees it.** On Glenbervie (pine planted and grown, half the gained pixels)
  the difference alone finds more of the rise at a strict cutoff than both inputs (.367 against .340 at 2 % lit),
  and the first year alone finds .140.
- **What carries to other places is the difference.** On Esri's "became trees" over every other square, the
  difference alone finds twice what the full head finds at 2 % lit (.042 against .020); the first year alone does
  worst (.015). What the full head learned about the place is New Zealand and Estonia; what it learned about the
  change is closer to general. The map's own rise still ranks Esri's label best (AUC .61).
- So a lidar-taught gain for the map would be taught on the difference (or with the place inputs weakened), not on
  both years as section 32 had it, if it is to work outside the nine squares.

## 34. Grass years outside the US against Wu et al. 2026 global grassland biomass (`scripts/grass_abroad.py`, `results/grass_abroad.json`)

Wu et al. 2026 (Zenodo 18044163, CC BY 4.0): yearly peak aboveground grassland biomass, 2000 to 2022, 0.005 degree
cells (~500 m), inside a grassland mask (steppe, savanna, tundra, alpine meadow). A random forest on MODIS NDVI,
climate, CO2 and terrain, taught on 11,265 field samples worldwide: a model, not a measurement, and it leans on
greenness as the embeddings do, but it is field-taught and not GEDI or CTrees. Read under every square of
veg_squares.json outside the US (19 with 10 or more cells inside Wu's mask; the crop squares are almost all outside
it) and the 11 US rangeland squares, 2018 to 2022, the map's 10 m pixels averaged into each Wu cell. Beside the map,
a head taught on RAP (US herbaceous biomass) from one year's AlphaEarth, every RAP square (held out in turn at home).
At home Wu is also checked against RAP, to see how well Wu itself follows grass years. 17 minutes, 1.3 GB.

Median over squares (squares positive of all):

| | Rank of cells in a square | Each cell's departure from its mean | Square's yearly median | Year-to-year steps |
|---|---|---|---|---|
| **Abroad, 19 squares: the map against Wu** | .18 (13/19) | .11 (12/19) | **.26 (15/19)** | .26 (14/19) |
| Abroad: the RAP head against Wu | -.03 (9/19) | .00 (11/19) | .05 (11/19) | -.10 (7/19) |
| **US, 11 squares: Wu against RAP** | .19 (9/11) | .25 (8/11) | .27 (8/11) | -.17 (5/11) |
| US: the RAP head against RAP (held out, 500 m cells) | .75 (10/11) | .60 (10/11) | .68 (9/11) | .57 (10/11) |
| US: the map against RAP (500 m cells) | -.44 (3/11) | .06 (8/11) | .18 (7/11) | -.01 (5/11) |

- **Wu is a weak referee for grass years.** At home it follows RAP's years about as well as the map follows Wu's
  abroad (.27 against .26 for the square's yearly median, .25 against .11 per cell). RAP, field-taught at 30 m,
  and Wu, field-taught at 500 m on MODIS, disagree on most of the year-to-year swing. A score against Wu abroad
  near .25 is about as high as Wu itself would reach against RAP.
- **The map follows Wu's years on most squares** (15 of 19 with a positive yearly correlation), strongly on some:
  Mongolian steppe .76 (Wu's 2022 rise from .63 to .95 Mg/ha), Top End savanna .76 per year and .69 per cell,
  Okavango .79 and .59, Sahel and Serengeti .74. Backward on Dehesa (-.52), central India (-.41) and the Matopiba
  farmland edge (-.54), all three where trees or crops carry most of the map's mass.
- **The RAP head does not carry abroad.** Taught on US rangeland, it follows Wu at about zero everywhere and orders
  the squares' levels poorly (.27 against the map's .73). Section 20 found the same head's levels plausible abroad;
  its years are not.
- **The map orders the squares' levels with Wu** (rank of square medians .73) but reads two to ten times higher
  on wooded grassland, because it counts the woody mass Wu leaves out: Cerrado 29 against 5 Mg/ha, central India 49
  against 2.5, Top End 15 against 1.7. On open grass it is near Wu: Mongolia .36 against .61, Llanos 5.3 against
  5.4, Sahel 1.3 against .8; above it on Kazakh steppe (2.0 against .6) and Tibet (3.8 against .9).
- Caveats: five years only (2018 to 2022, the overlap), so a square's yearly correlation is over five points;
  both Wu and the embeddings read greenness, so agreement is partly shared input, not independent truth.

## 35. The top end against plot biomass: Rodda lidar maps and GEO-TREES census plots (`scripts/top_end_plots.py`, `results/top_end_plots.json`)

Section 23 found the heads flatten against lidar height above about 250 Mg/ha and left open which map's level is
right there. Two plot-based references, neither from GEDI:
- **Rodda et al. 2024** (IRD Dataverse doi:10.23708/H2MHXF, CC BY 4.0): field plots scaled up with airborne or UAV
  lidar, 100 m cells, 13 tropical sites in India, Thailand, Cameroon and Gabon (100,414 cells kept). Lidar 2012 to
  2023; each cell is compared with the median of 2018 to 2022, and dropped where Hansen shows loss on it 2012 to 2022.
- **GEO-TREES** (the Forest Observation System's successor): 1,264 census subplots (mostly 50 x 50 m) from RAINFOR,
  ForestGEO, AfriTRON, TmFO, IIASA and others, the 105 GEDI cal/val subplots dropped; censuses mostly 2009 to 2016,
  read in max(census, 2018) and dropped where Hansen shows loss since the census (9). CTrees and ESA CCI likely
  calibrated on plot networks of this kind, so GEO-TREES may not be independent of them.
Beside the map's head (mean of its 10 m pixels): CTrees and ESA CCI v7 the same year. 30 minutes, under 1.5 GB.

| | Rank / R2 of log: map | CTrees | CCI |
|---|---|---|---|
| Rodda, 100,414 cells (median 199 Mg/ha) | **.89 / .77** | .75 / .65 | .72 / .25 |
| GEO-TREES, 1,264 subplots (median 307) | .41 / .21 | **.50 / .60** | .39 / .49 |
| GEO-TREES, censuses from 2015 on (61) | **.86** / .79 | .77 / .85 | .75 / .90 |

Median of each map against the reference's own biomass (ratio to the reference in brackets):

| Reference biomass, Mg/ha | Rodda: map | CTrees | CCI | GEO-TREES: map | CTrees | CCI |
|---|---|---|---|---|---|---|
| 0 to 100 | 41 (1.12) | 34 (.97) | 27 (.59) | 44 (1.65) | 51 (1.63) | 35 (1.16) |
| 100 to 200 | 113 (.74) | 144 (.96) | 81 (.52) | 148 (.93) | 259 (1.50) | 177 (1.11) |
| 200 to 300 | 160 (.65) | 168 (.69) | 127 (.52) | 192 (.75) | 299 (1.23) | 275 (1.03) |
| 300 to 400 | 272 (.78) | 261 (.77) | 260 (.75) | 211 (.59) | 373 (1.03) | 307 (.86) |
| 400 to 500 | 238 (.56) | 268 (.62) | 234 (.54) | 211 (.49) | 380 (.88) | 308 (.71) |
| 500 and up | 216 (.39) | 256 (.47) | 223 (.40) | 232 (.42) | 387 (.71) | 314 (.59) |

- **The map ranks tropical forest biomass best of the three against lidar and plots** (Rodda .89 against CTrees
  .75), ahead of CTrees on 12 of 13 sites, and on the recent censuses (.86). Its order is right.
- **Its level is short above about 150 Mg/ha.** Against Rodda it reads three quarters of the reference at 100 to
  300 Mg/ha and about half above 400; against GEO-TREES 59 % at 300 to 400 and 42 % above 500. The map does not
  rise above about 275 Mg/ha anywhere: its median stays 210 to 275 from 300 Mg/ha up.
- **CTrees holds the level better at the top, but against different references in different ways.** Against Rodda
  it flattens almost as the map does (.62 at 400 to 500); against GEO-TREES it is near the plots at 300 to 500 and
  half again too high at 100 to 200, which is what calibration on plots of this kind would look like, so its
  GEO-TREES level is not an independent check. CCI is short everywhere against Rodda and between the two on GEO-TREES.
- **Where both references agree: everything flattens above 400 Mg/ha** (every map at .4 to .7 of the reference),
  and the map is the flattest. The map inherits CTrees' top-end flattening and adds its own (a head on a single
  10 m embedding averages toward the middle).
- Directions: the top-weighted head of section 13 (trained with weight on high biomass), a calibration of the map's
  top end on Rodda (the one reference here independent of all three), or a cell-level correction above 200 Mg/ha.
- Caveats: Rodda's 13 sites are South Asia and Central Africa, no Amazon or Southeast Asian lowland; most GEO-TREES
  censuses predate AlphaEarth by 2 to 9 years (old-growth biomass changes slowly, and loss since was dropped); a 50 m
  subplot is 25 AlphaEarth pixels, so plot edges and geolocation add noise to GEO-TREES that the 100 m Rodda cells do
  not have.

## 36. Tundra against Orndahl's 30 m Arctic biomass and the Berner harvest plots (`scripts/tundra_check.py`, `scripts/orndahl_read.py`, `results/tundra_check.json`)

Nothing on disk checked tundra values before (section 23); Esri files tundra as rangeland (section 28). Two references,
not independent of each other (Orndahl's model is taught largely on the Berner plots), both independent of GEDI and
CTrees:
- **Orndahl et al.** (Arctic Data Center doi:10.18739/A2NS0M06B): 30 m plant (vascular: herbs, shrubs, trees) and
  woody-plant aboveground biomass for 2020, a Landsat, climate and terrain model, with its 95 % range. Its tiles are
  striped GeoTIFFs served without a file size, which GDAL cannot open over HTTP; `orndahl_read.py` reads the TIFF tags
  and only the strips a window needs. Compared on 100 m blocks over 11 tundra squares, the three of veg_squares.json
  and eight new ones across the Arctic (Taymyr, the Indigirka lowland, Chukotka, west Greenland, Banks Island, Daring
  Lake, Toolik, Finnmark). Finnmark is outside Orndahl's map, so 10 squares count.
- **Berner et al. 2024** (doi:10.18739/A2QJ78081): harvest plots from 2017 on with vascular plants measured, the
  median per site and year (473 site-years: Finland 333, Canada 58, Sweden 33, USA 33, Norway 10, Russia 6), against
  the 3 x 3 AlphaEarth pixels around each site in the plot's year. 324 of them are from datasets Orndahl did not
  train on.
Beside the map: CTrees and ESA CCI v7 (CCI counts living trees only, so it reads 0 on tundra by definition).
52 minutes (the Berner sites' reads), 0.6 GB.

| | Map | CTrees | Orndahl woody |
|---|---|---|---|
| 10 squares, 35,351 blocks of 100 m: median (Orndahl plant 3.3 Mg/ha) | 15.0 (4.7 x) | 15.2 (5.2 x) | 1.6 |
| Blocks inside Orndahl's 95 % range, by square | 0 to 10 % | 1 to 56 % | |
| Rank with Orndahl within a square (median of 10) | **.55** | .04 | .97 |
| Rank of the square medians with Orndahl | .70 | **.81** | |
| Berner sites, 473: median (plots, vascular, 2.1 Mg/ha; Orndahl there 5.4) | 9.4 (4.1 x) | 15.6 (6.0 x) | |
| Rank with the Berner sites | -.02 | -.04 | (Orndahl plant .28) |

By Orndahl's plant biomass per block (pooled): the map reads 6.6 Mg/ha where Orndahl reads under 1, 15.6 at 2 to 5,
19 at 5 to 10 and 24 at 10 to 20; CTrees 7.2, 16.8, 28 and 24.

- **The map reads tundra four to five times too high**, as CTrees does: a median 15 Mg/ha where Orndahl reads 3.3,
  and outside Orndahl's 95 % range on nine tenths of the blocks or more on every square. It is above Orndahl's total
  even where Orndahl counts no woody mass at all. The level is CTrees' (the map's teacher), not the embedding's.
- **Inside a square the map follows Orndahl and CTrees does not** (rank .55 against .04): the map sees where the
  tundra is richer (shrubby valleys, wet sedge against dry heath) even at the wrong level. Across squares CTrees
  orders them slightly better.
- **At the plots nothing ranks** (map -.02, CTrees -.04, Orndahl .28, Orndahl partly taught on them): a site's few
  harvest quadrats of 0.04 to 0.25 m2 do not stand for a 10 to 30 m pixel. The plots do agree on the level: the
  map reads four times their median, CTrees six times; even Orndahl reads two and a half times.
- The head calls tundra grass (63 to 100 % of a square) or moss (Nunavik, Banks Island, Noatak); never trees or shrub.
- Directions: a tundra correction from Orndahl (the map's within-square order is usable, its level is not), a head
  taught with Orndahl's level north of the treeline, or a lower cap where the head's class is grass or moss above
  60 N. Orndahl covers 2020 only, so it sets a level, not years.

## 37. A burn against lidar flown before and after: Jasper (`scripts/burn_lidar_ca.py`, `results/burn_lidar_ca.json`)

Section 32 found lidar's canopy height no referee on the Los Angeles burns, where dead stems kept standing. Canopy
cover (the share of 1 m cells above 2 m) was the hoped-for way to tell standing dead from live. The Jasper fire (July
2024, Alberta) has NRCan HRDEM 1 m surface and ground models from September 2019 and October 2024 on the same grid,
overlapping on 227 km2 around the town and the burn (Lytton has no pre-fire flight over its burn; Fort McMurray no
pre-fire AlphaEarth year). Per AlphaEarth pixel: mean canopy height and cover from the 1 m cells of each flight.
Forest = 2019 height 5 m or more and cover 50 % or more (3.1 million pixels); burned = Hansen loss in 2024 (2.8
million), steady = no Hansen loss since 2001 (0.3 million). 14 minutes, 2.7 GB.

| | Tells burned from steady (AUC) | Median fall, burned | Median fall, steady |
|---|---|---|---|
| Lidar canopy height | .907 | 94 % | 19 % |
| Lidar cover above 2 m | .894 | 90 % | 3 % |
| Lidar cover above 5 m | .891 | 98 % | 11 % |
| **The map, 2019 to 2025** | **.957** | 85 % | 16 % |
| The map, 2019 to 2024 (the fire year) | .778 | 21 % | 1 % |
| CTrees, 2019 to 2024 | .571 | 2 % | 1 % |

- **Here the lidar sees the burn as trees gone, by height and by cover alike.** Burned forest went from a median
  13 m and full cover to 0.7 m and 9 % cover. Not one burned pixel kept its height while losing its cover (the
  standing-dead pattern section 32 hoped cover would catch). Either the stems fell or were consumed, or the 1 m
  surface model passes through bare stems; height and cover cannot tell which, so the dead-or-live question needs
  the point clouds (return counts per stem), not these grids.
- **The map matches Hansen's burn more closely than either lidar measure does** (AUC .957 against .907), which
  is not the same as finding the burn better: Hansen is the referee here, and it is a Landsat product, so the map
  (built on optical imagery too) and Hansen share what they see, while lidar measures structure. Lidar's score is
  also lowered by noise between its two flights: on steady forest its height fell a median 19 % (season, sensor and
  processing), against the map's 16 %. What holds without Hansen: the map and lidar agree on the burn, a median
  fall of 85 % against lidar's 94 % on burned forest, ranked together .60 per pixel and .70 per 100 m cell.
- **The fire year's embedding sees a fifth of it.** From 2019 to 2024 the map falls 21 %; to 2025, 85 %. A fire in
  July leaves most of the year's picture pre-fire. A burn shows in full the year after; the change tooltip's "between
  2024 and 2025" for Jasper is that, not a late fire. CTrees' 2024 layer shows no fall at all.
- **Partial burns are drawn smaller, in order.** By lidar cover fall the map's median fall is 30 %, 63 %, 76 %, 83 %
  and 87 % (cover fall under 10 %, 10 to 25 %, 25 to 50 %, 50 to 75 %, 75 to 100 %); rank with lidar per pixel .60,
  per 100 m cell .70. Where cover barely fell, 43 % of pixels are still Hansen-burned: the burn's lightly touched edges.
- Caveat: one fire, one forest type (lodgepole pine and spruce in the Rockies); 89 % of the overlap burned, so steady
  forest is a thin set (0.3 million pixels).

## 38. The lidar-taught gain in the browser (`scripts/gain_mlp.py`, `results/gain_mlp.json`, `web/src/gain.ts`, `web/public/models/gain_lidar.json`)

Sections 32 and 33 taught gain on lidar's canopy rise with boosted trees. For the map the same teacher and harness
with a network the browser runs per pixel: one hidden layer of 32, on the difference of the two years' AlphaEarth
and the window's length (section 33: the part that carries to other places), held out by lidar square. The window
input is capped at 5 years, the longest the nine squares teach; the map's longer windows are fed as 5 (uncapped, its
output saturated on steady ground over six and seven years). Its confidence is the share of steady ground (127
squares, Esri class held, CTrees steady) scored below it at the window's length, as the map's other change rules;
gain is drawn from 98.5 %, the 1.5 % of steady ground the map's two-part floor lights. 3 minutes, 1.6 GB.

| | Lidar, pooled: AUC | Found at the floor's lit (1.5 %) | Esri "became trees": AUC | Found at 2 % lit |
|---|---|---|---|---|
| The network, difference (exported) | .786 | .353 | .580 | **.064** |
| The network, both years and the difference | **.956** | **.568** | .603 | .023 |
| Boosted trees, difference (section 33) | .822 | .346 | .577 | .042 |
| The map's rise and floor | .785 | .387 | **.614** | .020 |

On Esri's label by window length the exported network finds .049, .056, .069, .074, .063 and .052 at 2 % lit for one
to six years, about three times the map's at every length.

- **The network matches the boosted trees** on the difference (.35 against .35 at the floor's lit), so the browser
  loses nothing to the smaller model.
- **It trades lidar for reach.** On the nine lidar squares it finds slightly less regrowth than the map's floor at
  the same false alarm rate (.35 against .39); on Esri's "became trees" everywhere else it finds three times as much
  at 2 % lit (.064 against .020). Both-years input would win on lidar and lose elsewhere, as section 33 found.
- **In the browser it is a switch, off by default**: G in the change view, remembered. The worker runs it on the
  pixels the head says rose; its output matches the export's own on a probe of 32 pixels to 1e-5. At Glenbervie
  (2019 to 2024) it draws 610 ha of gain where the floor draws 1,455, holding the plantation and dropping the faint
  fringe; at the Arkhangelsk clearcuts (2018 to 2025) 0.5 ha against 7. The loss side is unchanged.
- Caveat: taught on nine squares in three countries; the regrowth lidar found since (New Brunswick,
  Finland, Brazil, Poland, Alabama) would widen it.

## 39. How few of the 64 bands the head needs (`scripts/band_subset.py`, `results/band_subset.json`)

The map reads AlphaEarth from zoom 11 in, because a view costs whole chunks of all 64 bands and every zoom step out
is four times the ground. The head only gives three numbers per pixel, so the question is whether it needs all 64.
Two parts to it: what the stores let a reader skip, and what the head uses.

- **The Zarr mosaic cannot skip bands.** Each 256 px inner chunk is one zstd block of all 64 bands (`tge-labs/aef-mosaic`,
  `zarr.json`); a median chunk is 2.6 MB, 63 % of raw (one 2024 shard over the Sierra Nevada foothills).
- **AEF's GeoTIFFs can.** They are planar: at the full-resolution 10 m level, each band in its own 1024 x 1024 zstd tiles, a median 580 KB, 55 % of raw
  (one 2024 tile, zone 10N). A reader such as geotiff.js fetches only the bands asked for.
- **The map's head uses nearly all 64.** Of its first-layer weight (on the standardized input), 53 bands carry 90 % and
  the weakest band has a third of the strongest's; as directions, 27 carry 90 %.

Heads on k bands, taught exactly as section 7's (same rows, loss and epochs), held out by square (8 folds). Bands
picked inside each fold from its training squares only: greedy forward selection by a ridge proxy (log biomass on
the cell mean, log height and class per pixel, weighted as the head's loss), or the largest first-layer weights of
a 64-band head on the same squares, or at random (3 draws). 10 minutes, 2.2 GB.

| Bands | Biomass, R2 of log | Biomass, median error (Mg/ha) | Height R2 | Class right | Class F1 |
|---|---|---|---|---|---|
| 64 | **.723** | **10.0** | .656 | **.858** | **.651** |
| 32, greedy | .695 | 10.3 | .655 | .815 | .533 |
| 32, weights | .659 | 11.3 | **.678** | .823 | .630 |
| 16, greedy | .638 | 11.0 | .551 | .770 | .479 |
| 16, weights | .512 | 14.4 | .307 | .755 | .498 |
| 16, random (3) | .44 to .54 | 13.6 to 14.1 | .30 to .50 | .71 to .73 | .45 to .47 |
| 8, greedy | .449 | 13.9 | .214 | .709 | .393 |
| 8, random (3) | .21 to .45 | 15.1 to 17.4 | .03 to .18 | .61 to .67 | .37 to .41 |
| 4, greedy | .301 | 17.0 | .00 | .598 | .236 |

- **Half the bands is nearly free; a quarter is not.** 32 bands keep biomass and height; classes drop a few points,
  mostly the rare ones. 16 lose about a tenth of the biomass fit and more of height and classes; 8 or fewer fall apart.
- **Which bands matters little.** Greedy beats random at 8 and 16 by about what random draws differ from each other,
  and the folds agree on only .56, .62, .64 and .75 of their 4, 8, 16 and 32 picks. AlphaEarth spreads what it
  knows across the vector.
- The 64-band head scores .723 here against section 7's .702, from the same recipe; differences of .02 are run noise.
- **The skipped bands are partly predictable from the rest** (pixels of a quarter of the squares held out): from 8,
  16 and 32 bands, a linear fit recovers a median .10, .19 and .42 of each other band's variance, a fit with squared
  and pairwise terms .22, .45 and .74. The pattern is there but curved.
- **The change map's own test holds better than the head.** Loss is drawn only where the embedding moved between
  years (1 - cosine over all 64 bands). Over the rows' one-year steps, the move on the 16 and 32 picked bands ranks
  .87 and .95 with the 64-band move, and keeps 76 % and 87 % of its top 1 %.

## 40. A zoomed-out map from a few bands: four ways, against the map's own head (`scripts/zoomed_out.py`, `results/zoomed_out.json`)

Zoomed out, a screen pixel covers many 10 m pixels (about 60 m at zoom 10, 120 m at 9, 300 m at 8), so the map needs a
rough picture that agrees with what zooming in will show, not the 10 m answer. Four ways from k bands, each scored
against the 64-band head of the same fold, on the fold's held-out squares, per pixel and over 6, 12 and 30 pixel
blocks:
- **plain**: section 39's k-band head, taught on the labels.
- **copy**: a small network (256 x 2) taught to give the 64-band head's answers from k bands, on every training pixel.
- **fill**: a network that rebuilds the other 64 - k bands from the k read; the 64-band head runs on the result.
- **neighbors**: copy, also given the same k bands of the 5 x 5 pixels around (they come in the same block, at no cost).

For the change map, two more that read year A whole and year B on k bands (64 + k of 128 bands, against 2k for the
four above): **copyA** takes B's missing bands from A as they are; **years alike** rebuilds how B's missing bands
differ from A's. Teacher in each fold: a 64-band head on the training squares' rows; networks taught on 1 M
pixel-years from 4 patches of 64 x 64 per training square (all 8 years); scored on the same patches of the held-out
squares (inner 60 x 60, 92 squares in all), bands from each fold's greedy picks. 14 minutes, 7.5 GB.

Each year, at 60 m (the median pixel there is 21 Mg/ha):

| Bands | Way | Biomass, R2 of log | Biomass off the map, median (Mg/ha) | Height off, median (m) | Class same |
|---|---|---|---|---|---|
| 8 | plain | .59 | 10.4 | .65 | .70 |
| 8 | copy | **.62** | **9.7** | **.54** | .71 |
| 8 | fill | .55 | 11.0 | .58 | .70 |
| 8 | neighbors | .51 | 10.4 | .55 | .70 |
| 16 | plain | .82 | 6.7 | .44 | .79 |
| 16 | copy | .85 | 5.7 | **.34** | .80 |
| 16 | fill | **.87** | **5.3** | .36 | .80 |
| 16 | neighbors | .80 | 6.7 | .37 | **.82** |
| 32 | plain | .91 | 5.1 | .42 | .86 |
| 32 | copy | .92 | 4.4 | .25 | .88 |
| 32 | fill | **.94** | **3.2** | **.21** | **.90** |
| 32 | neighbors | .93 | 4.0 | .25 | .90 |

Change, at 60 m. Big: the 64-band head changed by more than 20 Mg/ha; seen: the rough way changed by more than 10 the
same way; false: the head changed by less than 5 and the rough way by more than 10.

| Bands | Way | Bands read per pair | Big loss seen | Big gain seen | False loss | False gain |
|---|---|---|---|---|---|---|
| 8 | fill | 16 | .69 | .50 | .110 | .124 |
| 8 | copyA | 72 | .60 | .35 | **.008** | **.010** |
| 8 | years alike | 72 | .65 | .63 | .016 | .061 |
| 16 | plain | 32 | .72 | .56 | .053 | .060 |
| 16 | copy | 32 | .73 | .54 | .039 | .047 |
| 16 | fill | 32 | **.76** | .59 | .053 | .064 |
| 16 | neighbors | 32 | .74 | .58 | .062 | .071 |
| 16 | copyA | 80 | .73 | .49 | **.011** | **.015** |
| 16 | years alike | 80 | .72 | **.72** | .013 | .059 |
| 32 | fill | 64 | .86 | .81 | .029 | .035 |
| 32 | copyA | 96 | **.88** | .74 | .014 | **.020** |
| 32 | years alike | 96 | .87 | **.92** | **.008** | .042 |

Against the labels on the rows (section 7's scores; the rows have no neighbors):

| Bands | Way | Biomass, R2 of log | Biomass, median error | Height R2 | Class right | Class F1 |
|---|---|---|---|---|---|---|
| 64 | the head | .723 | 10.0 | .656 | .858 | .651 |
| 16 | plain / copy / fill | .638 / .651 / .660 | 11.0 / 10.7 / 10.8 | .551 / .505 / .340 | .770 / .787 / .779 | .479 / .501 / .520 |
| 32 | plain / copy / fill | .695 / .691 / .690 | 10.3 / 10.4 / 10.4 | .655 / .663 / .615 | .815 / .835 / .830 | .533 / .561 / .581 |

- **16 bands give a usable rough map; 8 do not.** Fill at 16 is a median 5.3 Mg/ha off the map at 60 m (about a
  quarter of the median pixel) with the class the same on 80 %; at 8 the best way is 9.7 off. On the GeoTIFFs, 16
  bands are a quarter of the read: zoom 10 for what zoom 11 costs now. Zoom 9 at that cost needs 4 bands.
- **Zooming out does not hide the errors.** From 10 m to 300 m, fill at 16 goes from .86 to .89 and 5.6 to 5.1 Mg/ha:
  the misses are whole areas reading high or low, not speckle that averages away.
- **Neighbors add nothing.** The 5 x 5 around does as well as copy or worse at every k.
- **Copy and fill follow the map; they are not truer.** Against the 64-band head they beat plain (5.3 to 5.7 Mg/ha
  against 6.7 at 16 bands); against the labels all three score about the same. Following the map is the point for a
  view that should not jump when zoomed in.
- **Reading one year whole is what removes false change.** Few bands of both years light 4 to 12 % of steady blocks
  as change; with A whole, about 1 %. copyA is the quietest and misses gain; years alike sees gain best (.72 at 16,
  .92 at 32) at about 5 % false gain. It reads 80 or 96 of 128 bands, less than one zoom step.
- The move test on fold picks, held-out patches: 16 bands keep 70 % of the 64-band move's top 1 % (copyA 72 %, fill
  69 %), 32 bands 82 % (copyA 84 %).
- Caveat: copy and fill learned from about 70 squares' patches per fold; they need no labels, so they can learn
  from any AlphaEarth pixels (section 41).

## 41. A zoomed-out map from a few bands: more pixels, a bigger network, fill-in taught through the map (`scripts/zoomed_out_more.py`, `results/zoomed_out_more.json`)

Copy, fill and years alike need no labels (section 40's caveat), so here they learn from far more AlphaEarth pixels:
60,000 pixel-years from anywhere in each square (and a second year at the same pixel) from the 92 squares and the 40
others on disk (lidar, fire and range squares, all 148 km or more from any of the 92), 7.9 M pixel-years in all; in
each fold the held-out squares are left out. Ways, each against the same fold's 64-band head:
- **base**: section 40's networks (256 x 2, 4 epochs, 1 M pixel-years from the patches), run again for the comparison.
- **more**: the pool, same network.
- **big**: the pool, 512 x 3, 6 epochs.
- **fill+**: big fill whose loss also asks that the map's head give the same answers on the rebuilt vector; **years
  alike+** the same for years alike.

Scored as section 40 (the same patches of the held-out squares, 8 folds, each fold's greedy band picks). 106 minutes,
7.0 GB, on the GPU.

Each year, at 60 m (the median pixel there is 21 Mg/ha):

| Bands | Way | Biomass, R2 of log | Biomass off the map, median (Mg/ha) | Height off, median (m) | Class same |
|---|---|---|---|---|---|
| 4 | copy, big | .43 | **13.2** | .96 | **.57** |
| 4 | fill, big | .27 | 15.2 | 1.10 | **.57** |
| 4 | fill+ | **.44** | **13.2** | **.95** | **.57** |
| 8 | copy, big | .61 | 8.9 | .51 | **.70** |
| 8 | fill, big | .59 | 9.5 | .52 | **.70** |
| 8 | fill+ | **.62** | **8.7** | .51 | **.70** |
| 16 | copy, base | .83 | 6.5 | .37 | .80 |
| 16 | fill, base | .87 | 5.4 | .38 | .79 |
| 16 | copy, more | .88 | 5.2 | .32 | .82 |
| 16 | fill, more | .88 | 5.1 | .33 | .82 |
| 16 | copy, big | .90 | **4.5** | **.28** | .83 |
| 16 | fill, big | .88 | 4.8 | .32 | .83 |
| 16 | fill+ | **.90** | **4.5** | .29 | **.84** |
| 32 | copy, big | .95 | 2.9 | .18 | .92 |
| 32 | fill, big | .95 | 2.8 | .17 | **.93** |
| 32 | fill+ | **.96** | **2.6** | **.16** | **.93** |

Change, at 60 m, as section 40 (big: the 64-band head changed by more than 20 Mg/ha; seen: the rough way changed by
more than 10 the same way; false: the head changed by less than 5 and the rough way by more than 10). Rank: rank
agreement of the rough way's change with the head's, every pixel.

| Bands | Way | Bands read per pair | Rank with the map's change | Big loss seen | Big gain seen | False loss | False gain |
|---|---|---|---|---|---|---|---|
| 8 | fill+ | 16 | .30 | .69 | .56 | .095 | .108 |
| 8 | years alike, big | 72 | .38 | .65 | **.63** | **.016** | .071 |
| 8 | years alike+ | 72 | **.44** | **.75** | .60 | .019 | **.042** |
| 16 | copy, base | 32 | .34 | .73 | .53 | .053 | .059 |
| 16 | fill, base | 32 | .42 | .76 | .60 | .054 | .064 |
| 16 | copy, big | 32 | .47 | .78 | .63 | .040 | .045 |
| 16 | fill+ | 32 | .47 | .78 | .66 | .043 | .049 |
| 16 | years alike, base | 80 | .48 | .70 | .71 | **.012** | .061 |
| 16 | years alike, more | 80 | .54 | .76 | .73 | .020 | .047 |
| 16 | years alike, big | 80 | .54 | .78 | **.74** | .017 | .050 |
| 16 | years alike+ | 80 | **.57** | **.85** | .69 | .020 | **.041** |
| 32 | fill+ | 64 | .75 | **.93** | .89 | .020 | **.024** |
| 32 | years alike, big | 96 | .74 | .89 | **.91** | **.012** | .033 |
| 32 | years alike+ | 96 | **.77** | **.93** | .90 | .013 | **.024** |

- **More pixels and a bigger network cut the distance to the map by 10 to 20 %.** At 16 bands the best way goes
  from 5.3 Mg/ha off (section 40's fill) to 4.5 (copy big, fill+), class same from 80 to 84 %; at 32 from 3.2 to 2.6
  and 90 to 93 %; at 8 from 9.7 to 8.7. Copy gains most from the pool (6.5 to 5.2), then the network (to 4.5); fill
  gains less from either (5.4, 5.1, 4.8).
- **It does not buy a zoom step.** Big at 8 bands (8.7 off, 70 % same) stays well short of base at 16 (5.4, 79 %).
  At 4 bands the map would be 13 Mg/ha off with the class the same on 57 %: zoom 9 at zoom 11's read is still out.
- **Fill+ is the best or level at every k, by a little.** Taught through the head it mends plain fill where it was
  worst (4 bands: 15.2 to 13.2) and leads at 32 (2.6 against 2.8 and 2.9). At 16, copy big ties it (copy gives the
  head's answers itself; fill+ gives the rebuilt bands, and the map's own small head runs on them).
- **Zooming out still does not hide the errors.** Fill+ at 16 is 4.8 Mg/ha off at 10 m and 4.3 at 300 m.
- **Change: years alike+ finds more loss with less false gain.** At 16 bands (80 read per pair) it sees 85 % of big
  loss against 72 % in section 40, false gain 4.1 % against 5.9 %, rank with the map's change .57 against .48 for
  base; big gain seen drops a little (.69 against .72) and false loss rises (2.0 % against 1.3 %). Years alike+ at 8
  (72 read per pair) sees more loss than section 40's years alike at 16 (.75 against .72) with less false gain (4.2
  against 5.9 %), but less gain (.60 against .72).
- **Reading one year whole now mostly quiets false loss.** At 16 bands, few bands of both years (fill+, copy big)
  light 4 % of steady blocks as loss and 5 % as gain; years alike+ 2 % as loss and still 4 % as gain.
- Network size in the browser: big is about 0.6 M weights (about 2.3 MB in float32), base about 0.08 M.
- Caveat: base is section 40's set-up run again, and copy at 16 came out 6.5 Mg/ha off here against 5.7 there
  (fill 5.4 against 5.3, years alike within a point); single runs of copy can differ by about 1 Mg/ha, so gaps of a few
  tenths between ways at the same k are not settled. copyA was not run again.

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
| Class calibration error, raw / calibrated (11) | .074 / .012 | .064 / .010 | .031 / .006 | .043 / .006 |
| 90 % class sets, one cutoff per class: single-class pixels (11) | .71 | .61 | .88 | .85 |
| Height against airborne lidar, R2 (16) | .73 | .61 | .66 | .65 |
| Lidar canopy lost: found (16) | .85 | .89 | .89 | .87 |
| Biomass change against lidar canopy change, per 100 m cell (16; CTrees .50) | .82 | | | .70 |
| Grass amounts against RAP, head taught on RAP, R2 of log (17) | .45 | .71 | .63 | .69 |
| Grass years against RAP: block departures / year-to-year steps (17) | .39 / .33 | .44 / .28 | .35 / .25 | .51 / .46 |
| Hard-ground class sets against lidar: truth inside 90 % sets (18) | | | .82 | .90 |
| Chaparral, CTrees head: rank within a square / year departures (19) | .36 / .22 | .66 / .56 | .58 / .20 | .58 / .23 |
| Shrub stands, rank of biomass with lidar height (23) | | | .53 | .37 |
| Height against the 2 m lidar of five new squares, R2 (24; Meta .61) | .55 | .63 | .70 | .72 |
| Change against lidar per cell, loss and floored gain from the same input (22) | .84 | | .75 | .72 |
| Read cost, MB per 6 km square-year (25) | 21 | 46 | 57 | 123 |
| Mean rank over the change map's tests / the each-year map's tests, of seven inputs (25) | 4.6 / 5.9 | 3.7 / 5.1 | 5.6 / 3.4 | 3.3 / 4.0 |

AEF dates and finds loss best, and against section 16's lidar it reads height and change best (on section 24's
leaf-off Dutch lidar it reads height worst); OlmoEarth reads amounts and classes best against the map labels and is
quietest; TESSERA shows regrowth best, follows shrub years and grass amounts best, and is the noisiest; together
they are the safest on trouble spots, follow grass years best and hold the hard-ground class sets. The pairs of
section 25 sit between: AlphaEarth + TESSERA and TESSERA + OlmoEarth rank above all three for the each-year map,
AlphaEarth + OlmoEarth nearly level with it for change at 63 % of its read cost. Gaps: AEF has no 2025 in central
Chile; TESSERA lacks much of 2017 (dropped everywhere).

## Next

Tested in sections 17 to 31; what is still open (directions, not ranked):
- **The map's embeddings**: section 25 lays out accuracy against read cost for every single, pair and all three;
  no input wins every test, so it is a choice by setting (change, each year), still to make.
- **Grass**: one head can carry the woody total (CTrees) and the grass part (RAP) at little cost (section 20); a
  wider shared layer or separate last layers; a grass label abroad to check the RAP head's years (the four drought
  checks there did not come out); annual grassland, which a yearly embedding reads as its lasting green, not its
  spring crop; a grass change map of large swings or of trends, since single small steps are noise in the label too.
  Section 34: abroad the map follows Wu's grass years on 15 of 19 squares (median .26), as well as Wu follows RAP at
  home; the RAP-taught head does not carry abroad. A field label of grass years abroad is still missing (NutNet).
- **Shrub**: no map of the three (CTrees, the chaparral map, CCI) is consistent with lidar height across stands
  (section 23), and none is refereed; a chaparral height-to-biomass relation from field plots would set the level.
  The head leans grass on shrub against grass (section 21).
- **Classes**: the offset fit (section 21) is the hard-ground calibration to use; shrub squares stay short under
  every fit; open woodland needs a better rule than a 2 to 5 m mean height (canopy cover at 100 m, or lidar).
- **Change for the map** (section 30): loss per pixel with its confidence per class, in percent, with the
  embedding's move; gain per pixel with the two-part floor and its per class Mg/ha confidence. Not built yet: the
  browser needs a class per pixel (the head's class output, dropped at export) and the confidence tables. Gain's
  signal is the limit, not its size. Per pixel against lidar (section 31) neither the trajectory head nor TESSERA
  finds more regrowth than the map's head at its floor. The trajectory heads and the AEF + TESSERA pair lower the
  steady ground lit (the trajectory head on AEF + TESSERA to a third), so they serve a stricter cutoff, at three to
  thirteen times the map's read for a change view.
- **Tundra**: Esri files it as rangeland (section 28). Section 36: against Orndahl's Arctic map the map reads four
  to five times too high (CTrees' level) while ranking within a square with it (.55); a level correction from
  Orndahl is the open step.
- **Loss size** (section 32): the map's soft 2/3 rule keeps partial loss apart but draws clear loss 15 to 20 points
  short of lidar; the head's fall is nearly flat across partial loss, so a rule cannot fix it alone. Burn severity
  needs a reference that tells dead trees from live ones; lidar height does not. Section 37 (Jasper): the 1 m grids'
  cover falls with height (no standing dead seen), the map's fall agrees with lidar's (85 % against 94 %, rank .70
  per cell), and a fire shows in full the year after it; dead-or-live needs point clouds.
- **Gain taught on lidar** (section 32): beats the map on every held-out lidar square, not on Esri's "became trees".
  Section 33: inside a square its ranking comes mostly from the first year (where canopy tends to rise); the part
  that carries to Esri's label is the difference. Section 38: in the browser as a small network on the difference,
  a switch (G) beside the floor. Open: lidar squares with regrowth outside New Zealand and Estonia (found, not read:
  New Brunswick, Finland, Brazil, Poland, Alabama).
- **Change** (earlier): loss and gain from separate sources (section 22); AEF for both parts, gain above a floor at the head's
  own size, is the best against lidar (.84 per cell) and the cheapest to read. Regrowth stays the weak half (at
  most four in ten of lidar's gained cells); full-size gain lights steady ground, and the gradual gate did not
  quiet it. Per-pixel loss overshoots the pixel's own biomass and needs a cap.
- **The top end**: the heads, not CTrees or CCI, are what flattens against lidar height (section 23); the
  top-weighted head helps. Section 35 (plots and lidar-scaled plots): the map ranks tropical forest best but reads
  three quarters of the reference at 100 to 300 Mg/ha and about half above 400; Rodda's maps are the independent
  reference to calibrate the top end on.
- **Lidar**: Switzerland and England still unread (whole-tile reads); Canada and Norway need same-season flight
  pairs before they test change (section 24).
- **Tundra**: section 36 checks CTrees' and the map's values against Orndahl and the Berner plots; NEON's Alaska
  lidar and plots still need a free account token.
- **The Los Angeles fires**: a flight a year after, when one is flown.

Tree loss (section 26):
- The height gate (canopy 3 m the year before, from the biomass head) fixes dry and open forest; Chaco's standing
  forest still moves year to year and needs its own cutoff or a longer window; young eucalyptus and fallow need a
  lower height.
- Yakutia's larch responds only to a lower cutoff; wildfire in open forest is the hard kind of loss.
- The WRI and Google DeepMind driver label needs a Global Forest Watch API key; storm, landslide and river loss
  have no public label read.
