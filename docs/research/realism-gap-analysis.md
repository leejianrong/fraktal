# Realism gap analysis (FRK-102, ADR-0012)

Written 2026-11-07. Question: why do our renders look fake next to real tropical imagery, and what
should we build first? Evidence is the review sheets of FRK-101 (our renders beside real chips at the
same ground size) plus native-pixel crops of both. Real chips are not in the repo (eval only); they are
named here and can be viewed with `fraktal review` (see [reference-board.md](reference-board.md)). Our own
renders are in [docs/img/realism/gap-synthetic-native.jpg](../img/realism/gap-synthetic-native.jpg) (dense_urban,
informal_settlement, khartoum and oil_palm_plantation, 250 m each, native pixels).

## How far to trust this

- One reviewer, about 60 real chips (28 Maxar, 90 Open Cities of which 29 pass the filter, a few
  SpaceNet tiles) and one synthetic tile set per biome. Gaps listed here are what I saw, not a survey.
- The Maxar chips are disaster-response acquisitions; some scenes are post-event.
- There is **no real ground-level or oblique reference** in the board. Every statement about eye-level and
  oblique views compares our renders with my sense of real streets, not with a chip. That is itself a gap.
- Where I say what our code does, I read `roads.py`, `settlement.py`, `building_grammar.py`, `objects.py`,
  `trees3d.py` and `blender_scene.py`. Rankings are judgement, for Jian to overrule in FRK-103.

## The ranked list

Payoff is my guess at how much a gap makes images look artificial, weighted by how many pixels and views it
touches. Layer says who owns the fix: **world** is the Python generators (benefits 2.5D and Blender),
**asset** is meshes, materials and textures, **blender** is shading and light, **sensor** is the camera chain.

| # | Gap | What real shows, what we show | Layer | Card | Points | Payoff |
|---|---|---|---|---|---|---|
| 1 | Ground | soil, dust, tracks, edges that are not square | world + blender | FRK-108 | 5 | highest |
| 2 | Street layout | organic, hierarchical, curved | world | FRK-104 | 5 | high |
| 3 | Parcels and footprints | irregular, mixed sizes, attached sheds | world | FRK-105 | 5 | high |
| 4 | Roofs | mixed, worn, ribbed tin | asset + blender | FRK-109, 110 | 3 + 3 | high |
| 5 | Large industrial buildings | halls, tanks, yards | world + asset | FRK-107 | 5 | medium, high for satellite |
| 6 | Walled compounds | walls and courtyards | world + asset | FRK-106 | 3 | medium |
| 7 | Trees | big irregular crowns, clustered | asset | FRK-111 | 3 | medium |
| 8 | Light | hard warm sun, readable shadows | blender | FRK-113 | 3 | medium |
| 9 | Vehicles, people, street objects | varied and everywhere | asset | FRK-112 | 5 | medium |
| 10 | Street-level facades | doors, shutters, signs, kerbs | asset | FRK-115 | 5 | highest for ground views only; needs a reference first |
| 11 | Drone lens | vignette, aberration, compression | sensor | FRK-114 | 2 | low |

## 1. Ground (FRK-108)

Real: in the Open Cities, Maxar Sudan, Maxar Myanmar and SpaceNet chips the ground is mostly bare red or
ochre soil, with paths worn into it, patchy dry grass, and edges between land uses that
follow fences, tracks and drainage. Vegetation is dark trees on that soil, not a green carpet.

Ours: a mottled green ground in every biome, including khartoum, whose real chips are tan. The world's
land cover is a 5 m grid (`gsd_m` 5.0 in `world.json`), so at 0.5 m each cell is ten pixels wide; edge
warping and noise hide this in some tiles and not in others. The paddy tiles show it plainly as coloured
squares. There is no soil class with a palette of its own, no worn paths and no dust. In the sheets, the
ground is the first thing that reads as fake because it covers most of every frame.

Fix: finer ground synthesis per class, a soil palette per biome taken from the board (Maxar and Open Cities,
not SpaceNet, see ADR-0012), tracks and desire lines, edges that follow roads and plots. World change for the
class map, Blender change for the ground texture.

## 2. Street layout (FRK-104)

Real: informal streets curve, taper, branch irregularly and dead-end; footpaths run between houses; main
roads are wide with several lanes and moving traffic (the Maxar Kenya road and the SpaceNet highway tiles show
this, the SpaceNet ones with a median).

Ours: arterials are least-cost paths between centres, and local streets branch off them at a fixed spacing
(`local_spacing_m`, 70 m in dense_urban) as parallel combs; the road is a flat grey band with no markings
(`_road_raster` paints a class colour). The result is a regular herringbone that is visible at once in the
dense_urban and informal_settlement renders.

Fix: a hierarchy with curved local roads, varied spacing, dead ends and footpaths, and road markings as a
ground detail. Existing worlds change, so this ships behind a config flag that defaults off, the way
`ParcelConfig.blocks` and `skew_deg` did, so existing world ids still hold (ADR-0003).

## 3. Parcels and footprints (FRK-105)

Real: footprints are irregular (L-shapes, additions, sheds against a main house) and sizes vary by an order
of magnitude within one street. Orientation follows the lane, not a global grid.

Ours: parcels are rectangles laid along each road's frontage; a building is a rectangle inside it, with a few
free-form lots when `blocks` is on. The grammar has three typologies (`block`, `shed`, `courtyard`). The
informal_settlement render shows the effect: a cloud of similar rectangles, rotated slightly, with even gaps.

Fix: footprint types (L, T, attached shed, subdivided plot), a heavier-tailed size mix, orientation that
follows the nearest street, per biome.

## 4. Roofs (FRK-109, FRK-110)

Real: corrugated tin is the common roof in the informal chips, in grey, rust, dull red, green and blue,
with ribs, stains and patchy colour; tiles and concrete appear too. Rooftops carry tanks, panels and
clutter. Large halls have long ribbed roofs in one colour with a visible seam pattern.

Ours: flat-shaded colours, one clutter prop (`CLUTTER = ("tank",)`), a small set of roof materials, and a
saturated light blue and orange in informal_settlement that no real chip has at that saturation (the Kenya
chips have blue sheds, but darker and with ribs). Tin has no relief at drone scale.

Fix: ribbing as geometry or normal map, rust and staining, a colour distribution fitted to the board, more
clutter types.

## 5. Large industrial buildings and 6. Walled compounds (FRK-107, FRK-106)

Real: the Myanmar chips show a tank farm (circular tanks tens of metres across inside square bunds) beside a
long ribbed shed; the Kenya chips show large blue-roofed and pale-roofed sheds with loading yards; SpaceNet
Khartoum has warehouses with ribbed roofs and walled lots; Maxar Sudan shows rectangular walled compounds with
outbuildings.

Ours: none of these exist. There are no large halls, no tanks on the ground and no yard walls; a wall appears
only as a roof parapet.

Fix: a hall typology and yard objects (class `yard_object`) and compound walls (class `wall`), as decided in
the ADR-0012 addendum. Halls stay `building` with typology `industrial_hall`. For satellite data the halls
and tanks matter most; they are big, high-contrast and common near roads.

## 7. Trees (FRK-111)

Real: crowns are large and irregular, overlap, line streets and fill yards; at 0.3 m the texture of a
mangrove or forest canopy is clear, with crown shadows.

Ours: three species (broadleaf, palm, shrub) placed by `objects.place` as a density per hectare per land cover
class, so they are spread evenly like seeds. Crowns are lumpy low-poly balls, small next to the real ones, and
they read as dots in the sheets.

Fix: size and shape variety, placement by plot and road edge, hedges, denser canopy where the biome has it.

## 8. Light (FRK-113)

Real: hard warm sun, long shadows that keep detail, a pale haze and a sky with a gradient and clouds.

Ours: the Blender world is one flat colour (0.62, 0.72, 0.9) at strength 0.55 and the sun's strength is fixed;
only its direction comes from the scene (`build_light`). The 2.5D path varies sun, haze and noise per scene
(`randomise.py`) but the 3D path does not. Dense_urban streets render as dark canyons in the sheets, and the
oblique and eye-level renders have a plain blue-grey sky.

Fix: a physical sky or CC0 HDRI, haze and a tone curve, all seeded.

## 9. Vehicles, people, street objects (FRK-112)

Real: Kenya road chips show buses, vans and cars in several lanes; the SpaceNet junction tile is full of
vehicles; informal streets hold stalls, motorbikes and people.

Ours: cars are two boxes in five colours, people three boxes, poles are boxes. The eye-level render shows the
result: a star-shaped palm, box people and an empty grey road.

## 10. Street-level facades (FRK-115)

The widest gap in absolute terms, and it only affects ground and low oblique views. Our facade-detail
render has windows, balconies and parapets but flat colours and no wear; the eye-level render has almost
no buildings. There is no real ground-level reference in the board, so FRK-115 should start by choosing one.
Likely sources are community photo sites and Wikimedia Commons, whose licences need reading before use;
for looking only, the same eval-only rules would apply.

## 11. Drone lens (FRK-114)

Open Cities has the colour and contrast of a drone camera (warm, saturated, slightly soft), and ADR-0010's
comparison found our sensor chain gets noise and blur about right once applied. The lens-specific effects
are small next to the gaps above.

## Things that are not gaps

- Blur and noise: with the sensor chain, 3D output was as close to the real chips as 2.5D on 9 of 10 gauge
  pairs (ADR-0010). Do not spend effort here first.
- Building heights: the storey profiles per biome (`heights.PROFILES`) are plausible, but heights are not visible
  in nadir views and I have no real oblique reference, so this is unchecked rather than fine.
- Label exactness: unaffected by anything above.

## Proposed order for Phase B

1. FRK-108 ground, then FRK-104 and FRK-105 layout. They cover most pixels in every view and benefit 2.5D.
2. FRK-109 and FRK-110 roofs, then FRK-107 and FRK-106 industrial buildings and compounds.
3. FRK-111 trees, FRK-113 light, FRK-112 objects.
4. FRK-115 after a ground-level reference is chosen; FRK-114 last.

Each step is judged by a before and after strip against the board (Jian looks), with no training run until the
chosen gaps are closed (ADR-0012). FRK-103 is where Jian reorders, drops or approves this list.

## Open questions

- Which ground-level reference to use, and is its licence acceptable for looking only?
- Should the khartoum biome keep a green-tinted look for the benchmark, or move to the tan the real chips
  show? The SpaceNet cast explains the green (ADR-0012 addendum); a colour-correct Khartoum waits for FRK-117.
- Do we want Maxar scenes from before the disaster dates only, to avoid damaged buildings as a reference?

## Answers (2026-11-07)

1. Ground-level reference: Mapillary, starting with Wikimedia Commons (FRK-119); licence is not a worry for
   looking only, but the photos are never used to train or condition anything.
2. Khartoum moves to the tan the real chips show; the benchmark path keeps its cast until FRK-117 (FRK-120).
3. Maxar scenes from before and after the event, both (FRK-118).

Details in the ADR-0012 addendum 2.

## Progress

- **FRK-108 (soil ground, 3D path, 2026-11-08):** `src/fraktal/ground_soil.py` turns a per-biome share of
  grass, shrub and crop ground into crisp soil patches (laterite for the wet biomes, ochre elsewhere) and
  mutes the remaining green. Colours are set by eye from the board, not fitted to pixels. Ground-colour
  before and after, from `fraktal scene`: `docs/img/realism/soil-ground-before-after.jpg`. Not yet looked at as a full
  Blender render beside the real chips; paddy and mangrove tiles change little because their classes are not
  soil-capable. The 2.5D benchmark path is untouched.
- **FRK-120 (Khartoum tan, 2026-11-08):** the `khartoum` biome now gets a tan soil style (80% of its soil-capable
  ground) and, in the 3D batch path only, near-neutral `ARID` sensor ranges (`randomise.BIOME_RANGES`) instead of
  the humid-tropics tone. The 2.5D benchmark path keeps `KHARTOUM_TONE`, and `palette.json` and world ids are
  untouched. Ground colour before and after: `docs/img/realism/khartoum-tan-before-after.jpg`. The standing water in the
  generated world still looks odd in a dry biome; that is the world generator, not this card.
- **FRK-109 (corrugated roofs, 2026-11-08):** the four tin roof materials now carry procedural ribs (a 0.1 m
  pitch, rib direction chosen per ~14 m cell), rust patches and a warm neutral tone in place of the pale blue
  (`add_corrugation` in `blender_scene.py`; the manifest hashes were updated). `docs/img/realism/roofs-drone-scale.jpg` is
  two informal-settlement tiles at 3 cm per pixel: ribs are faint but visible and rust spots show; at 0.1 m per
  pixel (one rib is about one pixel) they wash out, as intended. Weak spots: rib contrast is low, the clay-tile roofs
  show a dimple pattern from their texture, and the ground looks soft at 3 cm because its colour map is not that fine.
- **FRK-110 (roof colours and clutter, 2026-11-08):** tin roofs now draw from a per-biome mix of eight colours
  (silver, red, rust, dark, plus blue, green, white and pink; `building_grammar.TIN_MIXES`, shares set by eye from
  the board), and the silver and dark tints no longer read teal (the texture is blue-grey, so the tint divides it
  out). Flat roofs get solar panels, brick stacks and black or blue tanks beside the existing tank, on their own
  random stream so earlier draws are unchanged (`_more_clutter`; shares are assumed, 20, 15 and 20 percent).
  `docs/img/realism/roof-colours-clutter.jpg` is dense-urban and informal-settlement tiles at 0.12 m per pixel. Not done:
  clutter on tin and tile roofs, laundry, and checking the shares against the board by counting.
- **FRK-124 (props in the Blender scene, 2026-11-08):** the code-written props (`propgen`) are now placed by
  `props_scene.sample` (open ground clear of buildings and roads; tuk-tuks on streets; 60 per km2 by default,
  `--prop-density` on `render3d` and `render3d-batch`), packed as arrays and imported as one mesh. Class comes from
  the catalogue (`yard_object` 9, tuk-tuks `vehicle` 4), and each prop is a labelled instance numbered from 20001 so
  a tuk-tuk never shares an id with a parked car. Checked on a pod at 1500 per km2 and 0.15 m: 135 props, 130 and
  126 of them visible in the two tiles, with exact class and instance in `labels.png` (`docs/img/realism/props-in-scene.jpg`).
  Import cost: 0.46 to 0.64 s per 1000 props (the geometry is built once per variant, eight per kind, and copied).
  Not done: meshes from the image-to-3D route are not imported (their QA lives in `scripts/props/meshqa.py`), the
  props are one merged mesh, not Blender instances (per-instance labels need per-face colours), and placement is a
  plain scatter, not tied to parcels or roads beyond the tuk-tuks (FRK-128).
- **FRK-126 (normal and roughness maps, 2026-11-08):** `fraktal.pbr` derives a tangent-space normal map and a
  roughness map from each tileable colour texture (height is lightness minus its slow variation, the normal from
  the height's slope, the roughness from brightness), every step wrapping at the edge so tiles stay seamless.
  `scripts/textures/derive_pbr.py` wrote 18 JPEG maps (2.1 MB) for the nine textures, added `normal` and
  `roughness_map` to the materials that use them and a manifest row per map (licence and generator copied from the
  source row). Blender reads them in `load_material`; the procedural tin ribs ride on top of the derived relief.
  A/B on a pod, same two informal-settlement tiles at 4 cm per pixel (`docs/img/realism/pbr-ab.jpg`, top row without, bottom
  with): the maps load without errors and change the buildings by 5 to 9 grey levels of 255 on average, so the
  effect is real but subtle under this soft lighting. Not done: the card's first goal, new diffusion textures made
  with their own maps (this derives maps for the existing ones), a comparison with a learned estimator, and a
  look under a low sun, where relief shows most.
- **FRK-106 (walled compounds, 2026-11-08):** `compounds.build` puts a wall (0.22 m thick, 1.8 to 2.4 m high, set 0.4 m
  inside the parcel edge) round a per-biome share of the parcels that hold a building (khartoum 55, dense urban
  40, informal settlement 25 percent, set by eye), with a 3 m gate gap on the edge nearest the road. Derived from
  the parcel, building and road layers at export, so worlds and ids do not change. Label class `wall` is 8, one
  instance per compound; the gate is a gap. `--wall-share` overrides the share on `render3d` and `render3d-batch`.
  Pod check on two Khartoum tiles at 0.15 m (`docs/img/realism/compound-walls.jpg`): 15 and 32 compounds, each a separate
  instance in `labels.png`, gate gaps visible. Not done: courtyards and outbuildings inside compounds, wall
  colour or material variety beyond four flat colours, trees and props are not kept out of walls, and the
  informal-settlement run at 200 m tiles failed with `landcover.tif does not cover the render region` (not
  caused by the walls; fixed 2026-11-09, see the tile-size note below). Jian has not yet looked at a
  before and after strip next to a reference chip, which the card asks for.
- **Side by side with real chips (2026-11-08):** two of our Khartoum tiles at 0.3 m next to Maxar Sudan chips (the
  sheet is `data/eval/side-by-side-khartoum-v1.jpg`, git-ignored: it holds real pixels). The Sudan chips are rural,
  so this judges ground, trees and colour, not buildings. Findings: our ground was a pale cream (the brightest 60
  percent of pixels averaged about 195, 185, 153 against about 148 to 164, 123 to 136, 111 to 122 in the three
  dry-season chips), where the real earth is a red-brown; our trees are bigger, rounder, darker and sparser than the
  real scrub; our soil patches have smooth blob edges and no fine speckle. Only the colour was acted on (below). The
  numbers were used to see the gap; the new colour was chosen by eye and judged by looking, not fitted to them.
- **FRK-120 follow-up (earth colour, 2026-11-08):** `ground_soil.TAN` is now a red-brown (150, 108, 86 to 172, 128,
  102) instead of the pale cream. Under a low sun in a Khartoum tile it reads as earth. Still wrong: the muted
  green between the soil patches looks like camouflage, and the patch shapes are too smooth.
- **FRK-126 follow-up (ribs under a low sun, 2026-11-08):** `render3d-batch --sun-elevation-range LOW HIGH` draws the
  sun elevation from a range. At 2 cm per pixel and a 10 to 18 degree sun the tin ribs were not visible, and three
  separate render changes left tiles pixel-identical. Causes, found with a flat-sheet probe
  (`scripts/props/material_probe.py`, run by `scripts/runpod/probe_run.sh`): (1) the Wave texture band is
  `2 pi / 20 = 0.314 m` at scale 1, so a scale of `1 / ribs_m` made the ribs 3 cm apart, not 10 cm; (2) the
  sawtooth profile has a constant slope, which shades uniformly, so it is now a sine; (3) the bump depth was fixed,
  and is now `rib_depth_m` (1 cm in the eight ribbed tin materials); (4) the rib direction was random per 14 m
  cell, so ribs ran along the ridge on some roofs, and now follows each face's normal and runs down the slope
  (flat faces keep the random direction). The probe measures a 0.100 m stripe period. In the scene
  (`docs/img/realism/tin-ribs-low-sun.jpg`, informal-settlement tiles at 2 cm, top row the probe, bottom row before and
  after the direction fix) the ribs show on red and blue roofs, and are faint under the sensor blur and sky fill.
  The grey roof in one tile shows none, probably the sun running parallel to its ribs; not checked. The Khartoum
  roofs I first looked at were flat concrete and terracotta, so they had no ribs to show. Still not done on this
  card: diffusion textures with their own maps and a learned-estimator comparison. The derived normal map gives
  rusty steel a leathery grain at 2 cm, which is worth a second look.
- **FRK-104 (organic road networks, 2026-11-08):** `roads.build_roads` can now grow local streets that bend
  (`local_wander_deg`, `local_curl_deg`), side streets that branch off them and dead-end (`branch_rate`,
  `branch_length_m`), and footpaths (new road class `footpath`, 1.6 m wide, `footpath_rate`). Arterials are never
  touched, branches and footpaths start on their parent's final line and share its vertex, and each kind draws from
  its own sub-stream (`roads:wander`, `roads:branch`, `roads:foot`) that is only used above zero, so with the
  defaults every older world, id and geometry is unchanged (the new fields join `LATER_FIELDS`). Biomes: informal
  settlement winds, branches and gets footpaths most (about 570 streets and 520 footpaths in a 1.5 km world, from
  345 straight streets); dense urban stays a close grid with gentle bends and a few short alleys; the three rural
  biomes get loose bends, spurs and a few footpaths. **Khartoum is left straight on purpose**: its config is the
  default, which the benchmark worlds use. `docs/img/realism/organic-roads.jpg` is plan views of all six biomes, top row
  straight (the old generator, settings zeroed), bottom row new. Traffic and people route on the graph of streets:
  footpaths are left out of it (too narrow for a car), and they paint as the laterite track surface, so no tuk-tuk
  or parked car is placed on one (`test_cars_and_people_still_route_on_an_organic_informal_settlement`). Not done:
  people walking along footpaths, junction geometry (out of scope), and parcels and walls ignore footpaths. The
  bend test uses a strong setting; at the biome settings the curves are gentle and the plan view is the real check.
- **FRK-126 follow-up (ground textures and a learned-normal comparison, 2026-11-08):** SDXL base 1.0 on an A40
  (4.6 s per 1024 px image, 16 images, about 75 s of GPU time) made four roles with four seeds each, wrapped with
  circular padding (`scripts/textures/sdxl_ground.py`, `scripts/runpod/ground_textures_run.sh`;
  `docs/img/realism/ground-textures.jpg`, left: rows are red earth, tan soil, scrub, sand plaster). Results: `soil_red`
  seed 0 is plain fine-grained red earth, seeds 1 to 3 are cracked dried mud; `soil_tan` came out as river
  cobbles in all four seeds, so it is rejected; `soil_scrub` ignored the grass tufts and gave earth and cracked mud
  again; `sand plaster` is nearly flat. Seam scores 0.73 to 1.14 (about 1 is seamless); the worst (0.73, a tan
  seed) was not used. **Learned normals (Marigold normals v1.1, OpenRAIL++-M, same licence family as SDXL; its card
  lists no training data, which I could not check) were compared with the derived ones on seed 0 of each role
  (right: texture, derived normal, Marigold normal): the correlation is 0.00 on red earth, scrub and plaster, where
  Marigold returns a flat map, and 0.40 (x) and 0.59 (y) on the cobbles. It reads scene geometry, not
  microrelief, so for these textures the procedural derivation stays and Marigold is not used.** Kept and used:
  `gen-soil-red` (seed 0) with derived maps, applied in `blender_scene.add_ground_detail` as fine grain on the
  ground (the colour follows the texture's lightness by 35 percent at a 2 m tile, and its normal map tilts the
  surface), in world metres so it follows the terrain. Checked on a pod on two Khartoum tiles at 0.15 m: the ground
  is a pinker red-brown with cream patches, and the grain is barely visible at that scale (a 2 m tile is about 13
  pixels), so it mostly adds a little tone. The cracked mud is not in the library. Not done: a texture for the
  muted-green patches, SDXL tin or plaster with own maps (the existing ones are used), checking the ground grain
  at the 3 cm scale.
- **FRK-128 (prop sampler variation, 2026-11-08):** `props_scene.sample` now gives each prop a size (log-normal,
  4 percent, clamped to 0.92 to 1.08), one of four brightness steps and one of three wear levels (a darker colour
  and more roughness), drawn from their own stream (`props:vary`) so where props go is unchanged. `pack` applies the
  size and writes one material per distinct look, named `<part>~t<tint>w<wear>`, so Blender reads the same
  arrays. `repeat_stats` reports `distinct_shapes`, `distinct_looks`, `shape_repeat_rate` and `look_repeat_rate`
  per tile; `blender_export` writes it as `props_stats` in `scene.json`. On a 300 m informal-settlement tile at
  60 per km2 the stats were 5 props, 4 shapes, so a shape repeat rate of 0.2 and a look repeat rate of 0.0.
  Golden test: `test_golden_sampler_output`. A scaled prop can reach up to 8 percent past its clearance box, which
  is a few centimetres to 0.25 m for a container; the clearance test was left as it was so placements stay the same.
- **FRK-111 trees and the Khartoum ground (2026-11-09):** the side-by-side sheet showed our trees too big, round,
  dark and evenly spread, and our ground cream with smooth blobs and bright green ribbons. The 3D path now places trees
  with `trees3d.scatter` (the 2.5D renderer keeps `objects.place`): thickets (a Gaussian cloud round a parent, 60
  percent of trees) plus loners, a log-normal crown radius (median 1.3 m, clamped 0.5 to 4.5 m) with 2 percent
  shade trees of 3 to 5 m, heights from the radius, 110 per hectare in Khartoum, none on a building or within a
  crown of a wall, and hedges of small shrubs along one wall of 20 percent of compounds. Foliage is olive and
  grey-green, set by eye. The Khartoum soil style now also colours bare ground and mutes tree-cover ground,
  patch edges are ragged (a fourth, 2.5 m octave) and fine dark tufts sit in zones. Evidence, our renders only,
  before above and after below: `docs/img/realism/trees-ground-before-after.jpg`. Tests: clustering by the index of
  dispersion, size mix, no trees on buildings or walls. Not done: the other biomes keep the old ground (bare ground
  stays cream there), wall-edge hedges ignore which side is the yard, trees still show hard crisp shadows, and
  buildings and roads have no real reference in the rural chips.
- **Tile-size bug behind the landcover failure (2026-11-09):** reproduced locally. Two causes. (1) A tile size that is
  not a whole number of pixels (200 m at 0.15 m is 1333.3) was rounded down, so `region_grid` counted two tiles
  and rendered a region twice as wide, past the world's edge; 120 m and 300 m at 0.15 m divide evenly, which is why
  those runs worked. `render.tile_px_for` now rounds up. (2) A tile centred within a pixel of the world's edge
  snapped outward past the raster; `render3d_batch.plan` now keeps `EDGE_SLACK_PX` (3) pixels inside, moving only
  centres that would have landed in that strip, so other plans are unchanged. Tests: one tile for odd sizes, and
  no planned tile past the raster edge (it fails with the slack set to 0). Image sizes change by one pixel for
  sizes that used to round down.
- **FRK-106 leftovers: wall kinds, yard trees, props off walls (2026-11-09):** `compounds.build` now picks a kind per
  compound from a per-biome mix: a block wall, a block wall with piers (a 0.45 m post at each end and every 3.5 m,
  0.25 m taller) or a thin sheet-metal fence (zinc, rust or blue, at most 2.1 m, 5 cm thick). Informal
  settlement leans to sheet fences (50 percent), Khartoum to block walls (90 percent). Block walls draw from seven
  colours instead of four. `build` also returns a free point inside each compound (`yard`, 2 m clear of the
  building); `trees3d.yard_trees` puts a shade tree there in 30 percent of compounds, and `props_scene.sample` takes
  the wall segments and keeps props off them (placements are unchanged when there are no walls). Pod check, 200 m
  tiles at 0.15 m: `docs/img/realism/walls-kinds-yard.jpg` (two Khartoum tiles, our render only). The informal
  settlement tiles rendered at 200 m too, which confirms the tile-size fix above. Not done: outbuildings and
  courtyards as buildings of their own, wall texture (colours are still flat), and the informal-settlement,
  dense-urban and other biomes still have the old cream, salmon and bright-green ground, which looks worse than
  Khartoum's now; that is the next slice.

- **Ground for the urban biomes** (informal settlement, dense urban): they now use the Khartoum treatment (bare land as red-brown laterite, muted greens, dark tufts, ragged patch edges), with the earth set browner and the greens more olive by eye. See `docs/img/realism/ground-urban-biomes.jpg` (top: informal settlement, bottom: dense urban). Oil palm, paddy and mangrove keep the old ground, since they have real vegetation to show; the real reference chips are all rural Sudan, so they cannot judge these colours.

- **Footprint variety (FRK-105)**: `BuildingConfig` gains `shape_mix` (L, T, shed-attached), `size_tail` (log-normal spread of the footprint size) and `shed_height_m`. Informal settlement and dense urban use them; the shape and size draws come from their own substream (`buildings:shape:<parcel>`), so no existing draw moves and the default Khartoum world keeps its id (the three fields are in `LATER_FIELDS`). A lean-to shed is a second, low building in the same parcel, touching the house, with its own instance id. See `docs/img/realism/footprints-urban-biomes.jpg` (top: informal settlement, bottom: dense urban). Khartoum is left alone because its world id is pinned in `test_biomes.py`; giving it the same mix means repinning that golden on purpose. Parcel shape and orientation per block are unchanged (informal already has sheared, winding blocks).

- **Industrial sites (FRK-107)**: `fraktal.industrial.carve` cuts lots of 45 to 90 m by 30 to 46 m beside arterial roads (dense urban; `IndustrialConfig.sites_per_km`), replacing every house parcel they touch, and puts one or two long halls (8 to 11 m high, tin roof, concrete walls, typology `industrial_hall`) at the back, with an open apron at the front. `props_scene.yard_props` fills the apron with containers, tanks and a new code-written `silo` prop, all class `yard_object` (9). Positions are retried (25 tries a site) because a close street grid rejects most; a world gets roughly 7 to 18 sites. The whole `industrial` config section is left out of the world id while at its default, so existing worlds keep their ids. Evidence: `docs/img/realism/industrial-hall-yard.jpg` (a crop of a dense-urban tile; the hall runs off the left edge because batch tiles are placed at random, not on sites). Not done: Khartoum (its pinned world id needs a deliberate repin, as with FRK-105), a before and after strip next to a real chip, and the hall's roller doors.

- **Sky, dust and shadow tone (FRK-113)**: `fraktal.lighting.draw` gives each scene, from its seed and sun elevation, a warm sun colour (orange when low, warm white when high), a sky gradient from a clear zenith to a dusty horizon, a warm glow around the sun and a sky-fill strength. `blender_scene.build_light` builds the sun lamp and a node-based world (gradient plus sun lobe, no Nishita node, so it does not depend on the Blender version); scenes exported before this keep the old flat light. The fill is much stronger than before (1.3 to 1.9, was 0.55), so street shadows are lit by the sky: over 8 dense-urban tiles the darkest tenth of pixels went from 20 to 41 (of 255) and the median from 75 to 108, while the brightest 5 percent went from 136 to 189, so the picture is brighter and flatter overall. Both are judged by eye and by those numbers; there is no real urban chip to compare shadows against. See `docs/img/realism/sky-shadow-before-after.jpg` (top before, bottom after). Haze already varies per tile in the sensor chain. Not done: a tone curve in the renderer (the view transform stays Raw so colours compare with 2.5D) and softer tree shadows.

- **Drone lens effects (FRK-114)**: `kernels.sensor.lens_effects` adds a vignette (corner darkening by a share), chromatic aberration (red magnified and blue shrunk, a fringe of N pixels at the corner) and JPEG-style block compression at a given quality (8 x 8 DCT blocks, JPEG's quantisation table, plain numpy so there is no encoder dependency and the output is the same on every machine), in that order, after the sensor chain and noise. `render3d_batch.plan(..., sensor=True, lens=(...))` draws each tile's values from a stream of its own (every effect is always drawn, so leaving one out never moves the others); an effect left out is 0, which is off. `fraktal render3d-batch --sensor --lens` turns on all three, or name some (`--lens vignette compression`). Ranges are assumed, not fitted: vignette 0.15 to 0.45, fringe 0.4 to 1.6 px, JPEG quality 55 to 85. The golden test pins all three. Evidence: `docs/img/realism/drone-lens-effects.jpg` (our own render; original, vignette, fringe, compression, all three, then a corner crop of the fringe and a crop of the compression blocks, at exaggerated settings). Not done: it is for the batch sensor chain only, not yet for video frames or the 2.5D renderer.
- **Compound wall outlines (FRK-106):** walls were always the lot's rectangle. About 70% of compounds now get one to three inward cuts (a slanted corner, a notch, a leaning side, a bent wedge), skipped when they would touch the house. The gate moves to the longest side nearest the road. A cut can also be a curved bite (a bowed wall) and about a third of reshaped compounds get rounded corners. Dense urban lots now take slanted cuts between neighbours (`skew_deg` 16, as informal settlement already had); Khartoum's strip lots are still square. Plan view: `docs/img/realism/compound-outlines.jpg`.
- **Street props (FRK-112):** six new code-written generators in `propgen`: `motorbike` and `minibus` (class `vehicle` 4), `pedestrian` (`person` 6), and `dumpster`, `barrel_stack` and `rubbish_pile` (`yard_object` 9). Tuk-tuks and market stalls already existed (FRK-130). They are placed by a second pass, `props_scene.sample(..., mix=STREET_MIX, stream="props:street")`, at 1.5 times the scene prop density: motorbikes, vans and people stand on the road raster, bins and litter on open ground. The first pass is untouched, so existing tiles keep their props. Like every code-written prop they are generated in the scene, not shipped as files, so there are no manifest rows (the catalogue is `propgen.GENERATORS`). Preview: `docs/img/realism/street-props.jpg`. Not done: vehicles and people face a random direction (the road raster has no heading), pedestrians do not walk, props are box shapes, and nothing is rendered in Blender yet.

