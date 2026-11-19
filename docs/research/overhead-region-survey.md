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

## Next

Pick the target regions to widen to from the global sheet, then build the cards above in that order.
Measure with the free checks (`fraktal compare`, distribution plots, chip strips), not a training run
(ADR-0009).
