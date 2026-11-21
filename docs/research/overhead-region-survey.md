# Overhead region survey: what our satellite worlds lack (2026-11)

A quick qualitative look at real satellite tiles from many regions beside our renders, to choose
which looks to build next. Not a measurement. The real tiles are evaluation only and stay under
`data/eval/spacenet-regions/` (git-ignored, CC BY-SA 4.0, ADR-0008); no image from them is in the repo.

## What was looked at

`scripts/refboard/pull_spacenet_regions.py` pulls 108 SpaceNet tiles and writes two contact sheets:

- **Close-ups, 0.3 to 0.5 m:** 8 tiles each from Rio (SN1), Las Vegas, Paris, Shanghai (SN2), Mumbai and
  Moscow (SN5). Sheet: `data/review/regions/spacenet_regions.jpg`, the middle 200 m of each tile.
- **Global, 4 m (Planet):** one mid-year scene from each of the 60 SN7 areas, labelled with latitude and
  longitude. They cover West and North Africa, Egypt, Khartoum, Kampala, Lusaka, Durban, the Gulf, India,
  China, the Philippines, the US, Europe and South America. Sheet: `data/review/regions/sn7_grid.jpg`.

Ours: the six biome tiles on the review sheet (`fraktal review`, `docs/img/pipeline-3d/batch-biomes.jpg`)
and the shipped sample tile.

## Caveats

The 4 m scenes show land-use layout only, not building detail. Tiles are spread evenly through each
folder, so some are water or image edge. The comparison used an older batch sheet and the sample tile,
not a fresh render of every biome.

## What we lack, most important first

1. **Terrain and water.** Real tiles show relief (dry valleys, eroded slopes, mountains, quarries and
   mines) and a lot of water (bays, rivers, canals). Our terrain is flat with visible blocks and open
   water appears only in paddy and mangrove. Card FRK-187.
2. **Planned layouts.** Las Vegas, Gulf and Chinese tiles are near-identical houses on curved streets
   or strict grids. We only generate organic dense and informal layouts. Card FRK-188.
3. **Industry and infrastructure.** Large flat-roof warehouses, blue and tin roof clusters, highways with
   interchanges, bridges and car parks. FRK-107 covers part (industrial compounds); highways, bridges
   and parking are card FRK-189.
4. **Building detail.** Real roofs vary a lot in colour, show ridge lines and L or T shapes, and cast
   crisp long shadows. Ours are small flat boxes (FRK-110 added roof colour mixes).
5. **Roads.** Real roads have lane markings, kerbs, sidewalks and visible cars. Ours are low-contrast
   bands (part of FRK-149).
6. **Crop fields.** Rectilinear parcels with row structure and varied colour; ours are neon squares
   (FRK-136, FRK-137, FRK-141).
7. **Atmosphere and season.** Snow (Moscow), cloud, haze and large sun-angle changes. We render one
   clean look.
8. **Small water features:** pools and ponds.

## Target looks, in build order

Chosen by look, not by city: a generator that reproduces a look covers every region that shares it.
Numbers are the cell index on `sn7_grid.jpg` (latitude and longitude are decoded from the tile
names; the grouping is a judgement from 4 m thumbnails).

| # | Look | Real examples (SN7 cells) | Needs |
|---|---|---|---|
| 1 | **Arid low-rise sprawl.** Sand and rock relief, flat roofs, grid or radial streets, bare plots. Next to Khartoum, so it reuses the desert palette | Cairo 34 to 36, Khartoum 37, Benghazi 30, Riyadh 40, Kuwait 41, Abu Dhabi 43, Sanaa 39, Algeria 26, Calama 22 | FRK-187 (arid relief), FRK-188 (grid and radial layouts) |
| 2 | **Planned suburbs.** Uniform houses on curved streets, pools, large blocks | Las Vegas 09 and the Vegas close-ups, San Diego 06 and 07, Boise 08, Salt Lake 10, Mexico City 12 | FRK-188 |
| 3 | **Industrial and farmed plain.** Blue and tin roof clusters, warehouses, rectilinear fields, rivers and canals | Chengdu 04 and 49 to 50, Guangzhou 51 to 52, Wuhan 53, Hefei 54 to 55, Shanghai 57 | FRK-107 (industrial), FRK-187 (water), FRK-189 (highways), the paddy and field cards |
| 4 | **Dense tropical** (already covered by the dense-urban and informal biomes; check only) | Kumasi 00, Dhaka 01, Clark 03, Kampala 38, Mumbai 45 and 46, Andhra 47 and 48, Kingston 19 | None new |
| 5 | **Temperate sprawl and ports.** Tree-heavy suburbs, field mosaics, container ports and bays | Atlanta 15, Raleigh 18, Birmingham 27, London 28, Rotterdam 29, Dakar 25, Daejeon 58 | Later |

Skipped for now: snow and cloud as a look of their own (Moscow), and mountain or mining terrain beyond
the arid relief in FRK-187.

Why this order: look 1 extends the pilot we already have and needs two cards; look 2 is the most visibly
wrong gap and only needs FRK-188; look 3 is the largest set of regions but needs the most cards, so it
comes after its parts exist. Dense tropical is the benchmark's home ground and mostly built.

## Next

Build the cards above in the order of the table.
Measure with the free checks (`fraktal compare`, distribution plots, chip strips), not a training run
(ADR-0009).

## Progress: planned suburbs (FRK-188)

`planned_suburb` (rigid grid) and `planned_cul_de_sac` (dead-end spurs with turning circles) now
exist, from `src/fraktal/subdivision.py`. Rendered at 0.5 m beside the Vegas tiles, the layout and
the near-identical houses on uniform lots read right. Still missing against those tiles: local
streets draw as tan dirt, not dark asphalt with kerbs and pavements (FRK-149); roof colours are
random greys and browns where a subdivision repeats one or two; no pools, driveways or street
trees; one district per world, not the patchwork of grids and loops that real tiles show.

## Progress: arid relief (FRK-187, step 1)

`arid_badlands` (terrain style `arid`) makes ridges, scarps and gullies, with no open water and
little cover; steep bare ground turns to banded rock in the Blender ground colours. A 500 m tile at
0.8 m shows no terrain blocks. Still missing: water look (shore blend, depth colour, waves) and
rivers and lakes (steps 2 and 3), wadis drawn as pale channels, and a sheet row beside real chips.
