# Realism gap analysis (FRK-102, ADR-0012)

Written 2026-11-07. Question: why do our renders look fake next to real tropical imagery, and what
should we build first? Evidence is the review sheets of FRK-101 (our renders beside real chips at the
same ground size) plus native-pixel crops of both. Real chips are not in the repo (eval only); they are
named here and can be viewed with `fraktal review` (see [reference-board.md](reference-board.md)). Our own
renders are in [docs/img/gap-synthetic-native.jpg](../img/gap-synthetic-native.jpg) (dense_urban,
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
  before and after, from `fraktal scene`: `docs/img/soil-ground-before-after.jpg`. Not yet looked at as a full
  Blender render beside the real chips; paddy and mangrove tiles change little because their classes are not
  soil-capable. The 2.5D benchmark path is untouched.
- **FRK-120 (Khartoum tan, 2026-11-08):** the `khartoum` biome now gets a tan soil style (80% of its soil-capable
  ground) and, in the 3D batch path only, near-neutral `ARID` sensor ranges (`randomise.BIOME_RANGES`) instead of
  the humid-tropics tone. The 2.5D benchmark path keeps `KHARTOUM_TONE`, and `palette.json` and world ids are
  untouched. Ground colour before and after: `docs/img/khartoum-tan-before-after.jpg`. The standing water in the
  generated world still looks odd in a dry biome; that is the world generator, not this card.
- **FRK-109 (corrugated roofs, 2026-11-08):** the four tin roof materials now carry procedural ribs (a 0.1 m
  pitch, rib direction chosen per ~14 m cell), rust patches and a warm neutral tone in place of the pale blue
  (`add_corrugation` in `blender_scene.py`; the manifest hashes were updated). `docs/img/roofs-drone-scale.jpg` is
  two informal-settlement tiles at 3 cm per pixel: ribs are faint but visible and rust spots show; at 0.1 m per
  pixel (one rib is about one pixel) they wash out, as intended. Weak spots: rib contrast is low, the clay-tile roofs
  show a dimple pattern from their texture, and the ground looks soft at 3 cm because its colour map is not that fine.
- **FRK-110 (roof colours and clutter, 2026-11-08):** tin roofs now draw from a per-biome mix of eight colours
  (silver, red, rust, dark, plus blue, green, white and pink; `building_grammar.TIN_MIXES`, shares set by eye from
  the board), and the silver and dark tints no longer read teal (the texture is blue-grey, so the tint divides it
  out). Flat roofs get solar panels, brick stacks and black or blue tanks beside the existing tank, on their own
  random stream so earlier draws are unchanged (`_more_clutter`; shares are assumed, 20, 15 and 20 percent).
  `docs/img/roof-colours-clutter.jpg` is dense-urban and informal-settlement tiles at 0.12 m per pixel. Not done:
  clutter on tin and tile roofs, laundry, and checking the shares against the board by counting.
- **FRK-124 (props in the Blender scene, 2026-11-08):** the code-written props (`propgen`) are now placed by
  `props_scene.sample` (open ground clear of buildings and roads; tuk-tuks on streets; 60 per km2 by default,
  `--prop-density` on `render3d` and `render3d-batch`), packed as arrays and imported as one mesh. Class comes from
  the catalogue (`yard_object` 9, tuk-tuks `vehicle` 4), and each prop is a labelled instance numbered from 20001 so
  a tuk-tuk never shares an id with a parked car. Checked on a pod at 1500 per km2 and 0.15 m: 135 props, 130 and
  126 of them visible in the two tiles, with exact class and instance in `labels.png` (`docs/img/props-in-scene.jpg`).
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
  A/B on a pod, same two informal-settlement tiles at 4 cm per pixel (`docs/img/pbr-ab.jpg`, top row without, bottom
  with): the maps load without errors and change the buildings by 5 to 9 grey levels of 255 on average, so the
  effect is real but subtle under this soft lighting. Not done: the card's first goal, new diffusion textures made
  with their own maps (this derives maps for the existing ones), a comparison with a learned estimator, and a
  look under a low sun, where relief shows most.
