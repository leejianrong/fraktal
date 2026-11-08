# A tropical real test set (Q15, FRK-43)

Written 2026-11-04 from web searches only. Every licence below is a claim to check against the
dataset's own terms before any commercial use; nothing here was downloaded or verified. See
ADR-0007 (we sell synthetic datasets, so licence mistakes cost money) and ADR-0008 (V1 used
Khartoum, tropical deferred).

## What the test set must do

Measure whether a model trained on synthetic data does better on **real tropical imagery with
building labels**. It is for evaluation only: we never train on it, never ship it, never leak its
footprints into a synthetic world (ADR-0006). A small set is enough, since it steers and does not
gate (ADR-0009): about 30 to 60 chips of 500 by 500 pixels at roughly 0.5 m.

## Candidates

| Candidate | Tropical? | Imagery | Labels | Licence as reported | Fit |
|---|---|---|---|---|---|
| [Open Cities AI Challenge](https://source.coop/open-cities) | Yes: 10 African cities and regions (Dar es Salaam, Zanzibar, Kampala and others) | Drone, not satellite; far finer than 0.5 m, so it would be resampled | About 790k footprints, community-mapped and validated | **Conflict**: the [source.coop README](https://data.source.coop/open-cities/ai-challenge/README.md) says ODbL-1.0, a [GFDRR/OpenDRI post](https://opendri.org/?p=6981) shows CC BY 4.0 | Best ready-made labels. Wrong continent and sensor |
| [Maxar Open Data](https://registry.opendata.aws/maxar-open-data/) | Yes: Indonesia earthquakes (2018, 2022), Philippines Typhoon Goni (2020), Myanmar earthquake (2025) | Real VHR satellite, the right sensor class | None supplied | CC BY-NC 4.0 per [Maxar](https://blog.maxar.com/open-data-program/2020/open-data-response-to-typhoon-goni) and AWS; a Myanmar release is described as for humanitarian response; Maxar is now Vantor, so recheck terms | Right sensor and region, no labels, and the NC term needs a ruling |
| [xBD / xView2](https://arxiv.org/pdf/1911.09296) | Partly: includes the 2018 Palu tsunami | Pre and post event DigitalGlobe | Building polygons | **Unverified**: only the code licence was found; data terms need checking on the xView2 site | Labelled and disaster-skewed, so rubble and damage. Licence unknown |
| [Google Open Buildings V3](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons) | Yes: South and Southeast Asia | None | About 1.8 billion model-derived outlines with a confidence score | CC BY 4.0 (some HDX copies list ODbL) | Good as seed labels for hand correction, not as ground truth on its own |
| [GBSS](https://arxiv.org/pdf/2401.01178) | Unclear | 0.25 m tiles from Google Maps | OSM-derived | Not found | Imagery source terms look doubtful; skip |
| OSM plus a licensed or open image | Yes, anywhere | Whatever we can use | Volunteer footprints, uneven quality in informal areas | ODbL for the labels | Fallback building block |

No SpaceNet product for Southeast Asia turned up, and no labelled Bangkok, Jakarta or Manila
imagery dataset was found.

## Recommendation (provisional, for Jian to confirm)

1. **Primary, ready now: Open Cities AI**, resampled to about 0.5 m and blurred to a satellite-like
   response. It is the only candidate with real labels, real tropical cities and a permissive
   licence on either reading. Its weaknesses are the drone sensor and that it is Africa, not
   Southeast Asia. Treat results as "tropical, not yet Southeast Asian".
2. **Secondary, for the Southeast Asian claim: a small hand-corrected Maxar set** (about 30 to 40
   chips from the Indonesian or Philippine releases), seeded from Google Open Buildings or OSM so
   that labelling is correction, not drawing. Use it for private evaluation only, and settle the
   NC question before relying on any number from it in sales material.
3. **Do not use xBD or GBSS** until their data licences are checked.

## Hand-label plan, sized

If no candidate is accepted: 40 chips, each corrected from seed footprints. Assuming about 100
buildings a chip and 3 to 6 minutes a chip to fix or add the misses, that is roughly 2 to 4 hours
of labelling plus a few hours of setup. These are estimates, not measurements. Informal
settlements are slower, because roofs touch and seed footprints are poor there.

## Open items

- Which licence does Open Cities AI really carry? Check the Radiant MLHub or source.coop
  dataset card and the dataset documentation.
- Does a private, evaluation-only use of CC BY-NC imagery fit ADR-0007? This is a decision for
  Jian, possibly with advice.
- Access cost and size of the Open Cities and Maxar files were not checked.
- An ingest path is not built: the benchmark harness takes the dataset as config (ADR-0008), so a
  loader for chips plus a label raster is the work once a choice is confirmed.

## Decision, 2026-11-04 (Jian)

Open Cities AI and Maxar Open Data are used **for private evaluation only**. We do not train on
them, redistribute them or their derived chips, ship them inside a dataset product, or quote
them as training data in sales material. Only aggregate metrics (a score on the test set) may
leave the machine. This is what makes the CC BY-NC imagery acceptable under ADR-0007. The
licence checks in "Open items" still apply, in particular that a published metric computed on
ODbL or CC BY data keeps its attribution.

## Licence checks, 2026-11-04

Read from the primary pages, not summaries. Not legal advice.

- **Open Cities AI: ODbL-1.0.** The [source.coop dataset page](https://source.coop/open-cities/ai-challenge) and its [README](https://data.source.coop/open-cities/ai-challenge/README.md) both say ODbL-1.0, citing GFDRR Labs (2020), Radiant MLHub, DOI 10.34911/rdnt.f94cxb. The CC BY 4.0 seen earlier came from a STAC excerpt in an older post, so the source of record is ODbL. ODbL asks for attribution and, if a derived database is publicly used, share-alike; keeping the chips private triggers neither, and a score is a produced work. Keep the citation in `chips.json` (the loader does).
- **Maxar Open Data: CC BY-NC 4.0.** The [AWS registry](https://registry.opendata.aws/maxar-open-data/) lists "Creative Commons Attribution Non Commercial 4.0" with no custom terms. Maxar is now Vantor, and a [Vantor registry entry](https://registry.opendata.aws/vantor-open-data/) (bucket `vantor-opendata`, full-strip COGs) carries newer activations under the same non-commercial licence. Our decision to use it for private evaluation only is made, but "non-commercial" is about the use, and evaluating a commercial product is a grey area. Settle it with Vantor, or avoid relying on Maxar numbers in sales material.
- **Labels:** neither Maxar nor Vantor supplies building labels, so a Maxar set still needs the hand-correction plan above.

## What is built

`fraktal eval-set CITY/ID ...` (`src/fraktal/evalset.py`) builds an Open Cities evaluation chip
directory in the same format as the SpaceNet chips. Scenes are cloud-optimised GeoTIFFs on
`data.source.coop` with anonymous HTTP access, so a window is read at 0.5 m straight from the
overviews (area averaging does the low-pass) and only the scene's label file is downloaded.
Layout: `train_tier_1/<city>/<id>/<id>.tif` and `<id>-labels/<id>.geojson`. The chip directory
has `eval_only: true`, an empty train split and an `EVAL_ONLY.txt`. A live fetch of
`mon/207cc7` gave five aligned 500 px chips of dense informal settlement in about 30 seconds.
The other scene folders were not listed (the bucket listing was too slow to page); `dar/a017f9`
also exists. Maxar has no loader because it has no labels.

## First comparison against real tropical imagery (2026-11-04)

`fraktal compare` logic on the 5 Open Cities chips (Monrovia `mon/207cc7`, 0.5 m, evaluation only)
against 3 to 5 chips per tropical biome (`v10_tropical`, 1 m render, 600 m region). The sample is
tiny and one scene, so read the pattern, not the digits.

| biome | colour W1 | edge energy | noise | contrast | strong edge |
|---|---|---|---|---|---|
| dense_urban | 0.098 | 0.40 | 0.50 | 0.19 | 0.72 |
| informal_settlement | 0.101 | 0.43 | 0.46 | 0.32 | 0.81 |
| oil_palm_plantation | 0.056 | 0.44 | 0.47 | 0.24 | 0.81 |
| paddy_terraces | 0.071 | 0.49 | 0.48 | 0.31 | 0.90 |
| mangrove_coast | 0.103 | 0.45 | 0.47 | 0.32 | 0.82 |

- Colour is close: 0.06 to 0.10, against 0.27 to 0.29 when the same biomes were scored against
  arid Khartoum. The tropical palette is in the right place.
- Edges, noise and strong edges are far off (0.4 to 0.9). The synthetic look is too smooth, not
  the wrong colour. That points at texture and fine detail, and at scene density, not at palette.
- Next: more scenes (more cities and a built-up Monrovia scene) before trusting any number.

### After FRK-54 (denser buildings)

Building cover over a 2 km tile: dense_urban 15.4% to 22.2%, informal_settlement 5.8% to 11.3%.
That came from filling more of each parcel (smaller setbacks, wider width and depth shares) and,
for informal, closer and longer footpaths. Same 5 real chips, re-rendered:

| biome | edge energy | contrast | strong edge |
|---|---|---|---|
| dense_urban | 0.40 to 0.32 | 0.19 to 0.10 | 0.72 to 0.57 |
| informal_settlement | 0.43 to 0.41 | 0.32 to 0.32 | 0.81 to 0.78 |

Tried and dropped: a second street tier (alleys) made roads denser but buildings sparser, because
alleys take space from the parcels on the streets already there; lifting the cap of 8 settlement
centres changed nothing. The targets on the card (25% and 20%) were not reached. What limits
cover now is that parcels exist only in strips along streets, so a street-block generator is the
next step, not more parameters.

### After FRK-60 (street blocks)

`ParcelConfig.blocks` (off by default, on for dense_urban and informal_settlement) carves the land
between the streets, out to `block_depth_m` from a street edge, and splits each block across its long
side until a lot is about one parcel in area. Interior lots are included, so a lot need not touch a
street; each takes the id of its nearest street, and its building is fitted as a rectangle turned to
face that street. Informal cuts are slanted by up to half of `skew_deg`, so plots are not parallel.
Block names come from the rounded block centre, so a block's lots do not move when another block
changes. The default biome and every other biome keep their ids and bytes (the pinned test passes);
the two new settings are left out of the world id while at their defaults.

Building cover over a 2 km tile, seed 3: dense_urban 22.2% to 26.9% (depth 30 m); informal_settlement
11.3% to 22.5% (depth 32 m). The targets were 25% and 20%. Building counts rose from 2,960 to 6,415 and
from 7,518 to 18,620. Cover depends on tile size and seed (a 1 km tile gave lower numbers), so these
are one seed, one size. The effect on edge energy and contrast against the 5 real chips was **not
re-measured**; the earlier table above is from before blocks. The overview sheet
(`docs/img/biomes/biome-overview.jpg`) is regenerated.

### After FRK-70 (ten evaluation scenes)

The evaluation set grew from one scene (5 chips) to ten scenes and 90 chips, all from Open Cities AI
tier 1 (ODbL-1.0, evaluation only, `eval_only: true`, empty train split, `EVAL_ONLY.txt`):

| city | scenes (chips) |
|---|---|
| Monrovia | `mon/207cc7` (5), `mon/401175` (10) |
| Accra | `acc/665946` (10), `acc/a42435` (10) |
| Pointe-Noire | `ptn/abe1a3` (10) |
| Kampala | `kam/4e7c7f` (10) |
| Dar es Salaam | `dar/a017f9` (10), `dar/b15fce` (10) |
| Zanzibar | `znz/33cae6` (10) |
| Niamey | `nia/825a50` (5) |

Rebuild with `fraktal eval-set mon/207cc7 mon/401175 acc/665946 acc/a42435 ptn/abe1a3 kam/4e7c7f
dar/a017f9 dar/b15fce znz/33cae6 nia/825a50 --max-chips 10 --out data/eval/open-cities-10` (about
2.5 minutes). The scene list comes from the tier's STAC files, which the earlier note could not get
by listing the bucket: `train_tier_1/catalog.json` names the seven cities and each city's
`collection.json` names its scenes (`acc` 4, `mon` 4, `ptn` 2, `kam` 1, `dar` 6, `znz` 13, `nia` 1;
the `-labels` entries are the label folders). Picked here: every scene of the small cities, two of
Accra and Dar, and one of Zanzibar, so the set spans dense red-roof settlement (Monrovia, Accra,
Dar), planned estates, a dry Sahelian scene (Niamey, the closest to Khartoum) and a coast (Zanzibar).

Caveats: chips are random windows, so some are nearly empty (one Zanzibar chip is open sea; the
loader keeps at most 10% empty chips per scene by design); 10 of the 60 scenes are used; Zanzibar and
Niamey have few built-up chips; and the set is **not committed** (it lives under `data/`, which git
ignores). Not done: re-running the biome comparison on the larger set, which is FRK-67.
