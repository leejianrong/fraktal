# The reference board: real imagery to look at (FRK-99, ADR-0012)

Realism is judged by looking at our renders next to real chips. The chips are for looking and for
evaluation only: never trained on, redistributed, committed or used in sales material. They live under
`data/eval/` (git-ignored); only our own synthetic strips go in the repo.

| Set | View | Where | Licence | Notes |
|---|---|---|---|---|
| Open Cities AI | drone, 4 to 8 cm averaged to 0.5 m | `data/eval/open-cities-10/` (90 chips, 4 cities by file prefix) | ODbL-1.0 | built by `fraktal eval-set` |
| SpaceNet 2 Khartoum | satellite, 650 px | `data/spacenet2/chips/` | see ADR-0008 | benchmark chips carry a green cast; use `refboard.spacenet_view` to look (below) |
| Maxar Open Data | satellite, 0.31 m | `data/eval/maxar-reference/` (28 windows of 1024 px, 6 events) | CC BY-NC 4.0 (registry page, 2026-11-07) | `scripts/refboard/pull_maxar.py` |

Maxar citation, as the registry asks: "Maxar Open Data Program was accessed on DATE from
https://registry.opendata.aws/maxar-open-data". The script writes it with the access date into
`EVAL_ONLY.txt` beside the chips.

## Rebuild

```bash
uv run python scripts/refboard/pull_maxar.py --out data/eval/maxar-reference
```

Reads only the windows it keeps from the cloud-optimised scenes (about 85 MB, about a minute). It takes
the three biggest scenes per event, one per area, skips single-band (panchromatic) scenes, and from each
keeps two 1024 px windows: the one with the most large-scale structure and one from the upper middle.
Windows with cloud, open sea or scene-edge gaps are rejected by a simple rule on a thumbnail.

## What is in it

Events: Indonesia earthquake 2022, Myanmar earthquake 2025 (Mandalay area), Typhoon Kalmaegi 2025
(Philippines), Cyclone Mocha 2023 (Rakhine coast), Kenya flooding 2024 (Nairobi), Sudan flooding 2022.
Cyclone Senyar 2025 gave nothing: its scenes are single-band. Content: dense housing with red or tin
roofs, informal settlement with packed tin roofs, tank farms and warehouses, a stadium, walled compounds
and irrigated farmland, paddy, mangrove, palm coast.

Limits: these are disaster-response acquisitions, so some scenes are post-event; about 4 of the 28
windows are mostly cloud or reef; one window has a black scene-edge strip; a Maxar scene is a single
date, so there is no season or haze variation per place; the "biome" a window resembles is a judgement,
not a label.

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
