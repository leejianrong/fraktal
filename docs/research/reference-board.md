# The reference board: real imagery to look at (FRK-99, ADR-0012)

Realism is judged by looking at our renders next to real chips. The chips are for looking and for
evaluation only: never trained on, redistributed, committed or used in sales material. They live under
`data/eval/` (git-ignored); only our own synthetic strips go in the repo.

| Set | View | Where | Licence | Notes |
|---|---|---|---|---|
| Open Cities AI | drone, 4 to 8 cm averaged to 0.5 m | `data/eval/open-cities-10/` (90 chips, 4 cities by file prefix) | ODbL-1.0 | built by `fraktal eval-set` |
| SpaceNet 2 Khartoum | satellite, 650 px | `data/spacenet2/chips/` | see ADR-0008 | benchmark chips carry a green cast; use `refboard.spacenet_view` to look (below) |
| Maxar Open Data | satellite, 0.31 m | `data/eval/maxar-reference/` (47 windows of 1024 px from 6 events, 42 of them in before and after pairs) | CC BY-NC 4.0 (registry page, 2026-11-07) | `scripts/refboard/pull_maxar.py` |

Maxar citation, as the registry asks: "Maxar Open Data Program was accessed on DATE from
https://registry.opendata.aws/maxar-open-data". The script writes it with the access date into
`EVAL_ONLY.txt` beside the chips.

## Rebuild

```bash
uv run python scripts/refboard/pull_maxar.py --out data/eval/maxar-reference
```

Reads only the windows it keeps from the cloud-optimised scenes (about 150 MB, three to four minutes). Per
event it looks for ARD tiles that have a scene before and a scene after the event date, since the grid is
fixed and the same pixel window then shows the same ground twice. It tries candidates, best first, until three
tiles give chips: tiles with both sides first, then one-sided ones. Single-band (panchromatic) scenes are
skipped. From each tile it takes two 1024 px windows chosen on the first scene's thumbnail (the one with the
most large-scale structure and one from the upper middle) and reads the same ground from the other scene. A
window that is cloudy, empty or featureless is dropped by a simple rule; if only one side is clear, that side
is kept and the chip is marked `paired: false`.

Each chip's metadata has `phase` (`pre` or `post`), `event_date` and `paired`. The review sheets and the board
show the phase. The event dates are in `EVENT_DATES` in the script, taken from the event descriptions and public
reporting, so a scene within a few days of the date may be labelled wrongly: Myanmar 2025-03-28, Kalmaegi
2025-11-04, Mocha 2023-05-14, Kenya 2024-04-29 (the Mai Mahiu dam burst, rain began in March), Sudan 2022-07-01
(the start of the flood season), Indonesia 2022-11-21 (Cianjur; the set ends a week later).

## What is in it

Events: Indonesia earthquake 2022, Myanmar earthquake 2025 (Mandalay area), Typhoon Kalmaegi 2025
(Philippines), Cyclone Mocha 2023 (Rakhine coast), Kenya flooding 2024 (Nairobi), Sudan flooding 2022.
Chips per event, pre and post: Mocha 5 and 5, Myanmar 4 and 3, Indonesia 3 and 4, Kenya 4 and 4, Sudan 6 and 6,
Kalmaegi 2 and 1 (the typhoon scenes are mostly sea or cloud). Cyclone Senyar 2025 gave nothing: its scenes are
single-band. Content: dense housing with red or tin
roofs, informal settlement with packed tin roofs, tank farms and warehouses, a stadium, walled compounds
and irrigated farmland, paddy, mangrove, palm coast.

Limits: these are disaster-response acquisitions, so the post scenes show damage and floods as well as
seasons; a thin cloud can pass the filter (one Indonesia post chip has haze); a few chips are mostly vegetation
or water; the "biome" a window resembles is a judgement, not a label. A pair is two acquisitions with different
sun angles and sensors, so colour and shadows differ between its sides as well as the ground.

## SpaceNet colour (FRK-100)

The files are `RGB-PanSharpen`, uint16, bands in R, G, B order, so the band order is right. Over a sample
of 26 tiles the raw medians are R 534, G 592, B 409 (2nd to 98th percentile: R 364 to 800, G 475 to 887,
B 344 to 595). `chips.stretch_range` maps all three bands with one range, so blue stays far below green and
the benchmark chips come out strongly green (channel means R 88, G 118, B 34, from the fit recorded beside `KHARTOUM_TONE`). A per-band
stretch (`refboard.band_ranges`, `refboard.spacenet_view`) gives a natural reddish-brown desert city.

Consequence: `KHARTOUM_TONE`, `palette.json` and the synthetic Khartoum chips were fitted to the cast, so the
benchmark pair is consistent and earlier results hold, but SpaceNet is not a colour target for realism. See
the ADR-0012 addendum; the rebuild is FRK-117, at the FRK-45 milestone. I have not measured how much the
cast changes the colour gauge numbers: that needs the rebuilt chips.

## Review sheets (FRK-101)

```bash
uv run fraktal review --synthetic data/dataset_v0 --view satellite --board
uv run fraktal review --synthetic data/dataset_v0 --view drone
```

`--synthetic` is a `render3d-batch` output (`<biome>/tile_*/sensor.png`, falling back to `image.png`). Each
row of `data/review/sheet_<view>.jpg` is one biome: renders on the left (orange labels are the real chips),
real chips on the right, each cell the middle 250 m of the chip (`--ground-m`), so a house is the same size
on both sides. `--board` also writes `data/review/reference-board.html`, with every chip. The output must be
inside the data directory, which git ignores; the command refuses anywhere else.

Satellite rows use Maxar chips matched by event to the biome (a judgement, in `refboard.SATELLITE_EVENTS`)
and a SpaceNet tile with the per-band stretch for Khartoum. Drone rows use Open Cities chips, rotated so each
row shows different ones. Chips that are mostly cloud, no-data or featureless are skipped, which leaves few
good ones for some biomes (informal settlement 2, Khartoum 3 Maxar chips). Oil palm and mangrove have no
really close Maxar match; the coast and the Indonesian fields are the nearest.

## Ground-level photos (FRK-119)

```bash
# MAPILLARY_TOKEN in the environment or in ./.env (git-ignored); Commons needs no token
uv run python scripts/refboard/pull_ground.py --out data/eval/ground-reference
```

Two sources, one folder (`data/eval/ground-reference`, a metadata file beside each JPEG):

- **Mapillary**, about 100 photos from 9 spots (Nairobi centre and Mathare, Kampala, Dar es Salaam, Accra,
  Jakarta, Cebu, Mandalay, Quezon City), one per sequence, panoramas skipped, 1024 px. Mostly dashcam views:
  roads, kerbs, boundary walls, shopfronts, motorbikes, power lines. Reported as CC BY-SA 4.0 (not re-read).
  The token is a read-only client token, sent in an `Authorization` header, never in a URL. Manila's centre,
  Khartoum and Khartoum Bahri returned no images; there is no Mapillary coverage there in the boxes tried.
- **Wikimedia Commons**, about 20 photos found by a search phrase per city, kept only when the title names a
  street scene (a search for "Nairobi street" also returns portraits and interiors). Licence and author are
  recorded per photo. Khartoum comes only from here.

Limits: the Mapillary photos are one hash-chosen sample, not a survey; the API returns a different 100 images
on each call, so a re-run adds new ones. Some are blurry or show the car bonnet. Commons photos are what people
chose to upload, which skews to landmarks. `fraktal review --board` shows a few per place; there is no
side-by-side sheet for eye-level renders yet (FRK-115).

## Eye-level check (FRK-144)

`fraktal streets FRAMES_DIR --city 'nairobi-*' --sheet out.png` compares a directory of synthetic street
frames with the photos in `data/eval/ground-reference` (Commons and Mapillary together, eval only). It
prints nine coarse statistics (sky, green, warm and vertical-structure shares, brightness, contrast,
saturation, edge energy, road brightness) as photo mean and spread against synthetic, with a z score and a
`*` when the synthetic mean is more than 2 photo standard deviations away. The report always states the
photo count: under 8 photos a city's numbers are a hint. The sheet puts the frames above the photos; read
it first, since a frame can pass every number and still look wrong. Some Mapillary images are dashcam
frames with overlays or odd colour (a pink false-colour tree in Manila-Quezon); they stay in the sample and
widen the photo spread, so look at the sheet before trusting a flag.
