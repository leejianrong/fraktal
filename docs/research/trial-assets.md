# Trial: leaf-card trees, a CC0 vehicle pack and Infinigen (FRK-169)

Run 2026-11-15 on the local machine (CPU renders; the GPU was in use by another job). Strips: Manila-Quezon,
18 frames at 512 px, 32 samples, seed 7, route and settings of `scripts/local/city_strips.py`. Licence pages were
read on the same day. Everything here is opt-in and off by default.

## Verdicts

| Option | Verdict | Why |
|---|---|---|
| Quaternius CC0 car meshes, six models (`--car-meshes`) | **keep**, opt-in until a pilot cycle checks it | A real change in what a car looks like, labels exact, +23% strip time. Only six models, no jeepney, matatu or pickup |
| Leaf-card crowns (`--leaf-cards`) | **keep, now the default for street views** (FRK-172) | A modest gain: ragged edges on near and overhead crowns, little change on distant trees. +4% time. Labels and depth now follow the leaf shapes (see below) |
| Poly Haven models | **drop** for vehicles and trees | One vehicle (a car under a tarp, 12.6k triangles) and trees of 0.3 to 4.7 million triangles. Keep it as a CC0 source for HDRIs and textures |
| Infinigen | **defer** | The trial was not run: the install was refused by the session's permission check. Licence read, see below |

## Vehicles

The pack is "Realistic Car Pack - Nov 2018" by Quaternius, from `quaternius.itch.io/lowpoly-cars`. Its own
`License.txt` in the zip is CC0 1.0 Universal, and so is the licence on the itch.io page. The author's site,
`quaternius.com`, now describes a custom Quaternius Asset License (QAL v1.0) for its packs: commercial use allowed, no
attribution, but "resell or redistribute the Assets themselves" is not allowed. That page does not say it replaces the
CC0 grant of a pack published in 2018, and CC0 cannot be withdrawn, but the two do not agree, so the manifest rows
quote both. Our datasets contain renders, not the meshes. The Cop model is left out (the itch page says it is for
Patreon supporters). The model files were not edited.

| | Boxes (FRK-163) | CC0 meshes |
|---|---|---|
| Faces per car | 66 | 1,480 to 1,700 (six models) |
| Strip time, Manila (68 cars in the tile) | 254 s | 312 s |
| Moving-car label pixels, Nairobi 11-frame strip | 73,009 | 71,741 |
| Car detections in `tracks.json`, same strip | 59 | 59 |

Cars take their paint from the existing five-colour palette; windows, lights, tyres and trim use the colours in the
pack's MTL files. The model is chosen by car index, so the same car has the same model every run. Instances come from
`face_owner` (one number per car). The sheet is `img/frk169-cars.jpg`: top row boxes, bottom row meshes, same frames.

What it does not fix: the Manila strip has no moving car in view at all (its 68 cars are elsewhere on the tile), so
that city's cars are mostly props from `propgen`; and a mesh pack has no jeepney, matatu or pickup.

## Trees

`fraktal.leafcards.leaf_cluster` draws a grey 256 px cluster of 34 leaves with alpha (pure numpy, seed 0, ours,
licence `owned`, row `tex-leaf-cluster`). For each broadleaf and canopy crown the Blender script adds 14 to 55 quads on
and just outside the crown's ellipsoid, facing outward, each rolled at random, tinted with the tree's own green.
The cards are a separate object, `treecards`, hidden in the label and depth passes, so labels and depth are the
crown's, exactly as before.

- Poly count: 157,968 card quads on the Manila strip's trees (the crowns are unchanged).
- Time: 254 s to 264 s (+4%).
- Look (`img/frk169-trees.jpg`, top smooth crowns, bottom with cards): ragged leaf edges against the sky on overhead and
  near crowns; the distant crowns, which are most of the trees, change little.
- Labels: the beauty frames change in 2.71% of pixels, and 99% of those are outside the tree label, because a card
  sticks out of the crown and the label is the crown. So with cards on, labels are not exact at the leaf edges. Making
  the label pass alpha-aware (paint the cards as tree, cut by the texture's alpha) is the work before this can be a
  default. I did not do it.

### Update, FRK-172: alpha-aware labels and depth

The label and depth passes now show the cards, cut by the leaf texture's alpha at 0.5 with one unfiltered sample, labelled
as the tree they sit on (`img/frk172-card-labels.jpg`: beauty on the left, the tree label over it on the right). Measured on
the same Manila strip (18 frames, CPU, seed 7) with cards off and on:

- Pixels where the beauty frame shows a leaf and the label says something else: 0.317% of the frame at a beauty difference
  of 40 (it was 2.71%), 0.106% at 80 and 0.032% at 120. The remainder is soft-alpha pixels at the edge of a leaf and
  denoiser differences between two renders; the card's target was under 0.3%.
- Depth where a card is over a crown: closer in 13,862 pixels, equal in 2,353, farther in none.
- Label images still hold only the same classes as without cards: flat colours, no blending.
- Tree-class pixels per frame rose from 58,627 to 64,527, the leaves that stick out of the crown.
- Cost: 349 s to 282 s is machine noise (the cards-off run was slower); the earlier measurement was +4%.

`--leaf-cards` is now on by default for street strips (`--no-leaf-cards` turns it off) and off for other views.

## Infinigen

Not run. The clone worked (`princeton-vl/infinigen`, commit `53033dd`, 2026-11-10) but the minimal install into a Python
3.11 scratch venv was refused by the session's permission check, the same as in `infinigen-spike.md`. What the clone
shows: the code is BSD-3-Clause (Princeton University); the README says some newer procedural code is CC0 and marked in
comments; `mesh_to_sdf` carries its own MIT licence; there are `trees`, `tropic_plants` and `leaves` generators; it needs
Python 3.11 and `bpy==4.2.0`, so it cannot share our Blender 5 environment. Still not read: the licence of anything
bundled that a generator loads. The trial plan in `infinigen-spike.md` stands.

## Vehicle sources, second pass (FRK-173, 2026-11-15)

The question: a CC0 or owned jeepney, matatu, pickup or tuk-tuk. Each source's own licence was read.

| Source | Licence read | Has | Verdict |
|---|---|---|---|
| Kenney Car Kit v2.0 (`kenney-assets.itch.io/car-kit`, 45 models, OBJ, FBX and GLB) | its own `License.txt` in the zip: "Creative Commons Zero, CC0"; the itch page also says CC0 1.0 | sedan, hatchback, SUV, taxi, **van**, **truck** (an open-bed pickup), **truck-flat** (a flatbed), **delivery** (a box lorry), ambulance, firetruck, garbage truck, tractors, wheels. 2,082 to 2,488 faces a vehicle | **keep** as stand-ins: `truck` for a pickup, `van` for a matatu, `delivery` for a box lorry. Chunky and stylised, so it helps shape, not photo realism. Not added to `assets/` yet (see below) |
| Quaternius cars (FRK-169) | see above: CC0 in the pack, a custom licence on the site | saloons, SUV, taxi | in use, opt-in |
| Poly Haven | CC0, no bulk scraping | one covered car | drop, as above |
| CGTrader, TurboSquid, Sketchfab | per-listing licences; the jeepney, tuk-tuk, minibus and matatu listings found were paid or royalty-free, not CC0 | jeepneys, matatus, tuk-tuks | **drop** (ADR-0010: owned or CC0 only; a paid royalty-free licence is not CC0) |
| OpenGameArt "free low poly vehicles pack" (rgsdev, listed as CC0, with a pickup and a van) | not read: the site refused the fetch (403) | | **defer**, read the page before use |

There is no CC0 jeepney or tuk-tuk in anything I found. The jeepney and tuk-tuk stay code-made props from `propgen.py`;
the realistic way to get better ones is to model them ourselves, which is a larger job than this card.

Kenney's models use one shared palette image (`colormap.png`) with a UV per face, not a colour per material like the
Quaternius ones. The car loader in `blender_scene.py` reads a colour per material, so using them needs it to sample
the palette by UV. That is the follow-up card; no Kenney file is in `assets/` until then.

## Poly Haven licence (`polyhaven.com/license`)

All HDRIs, textures and models are CC0 ("effectively public domain"), no attribution, commercial use and
redistribution allowed. The site's terms prohibit "web scraping or data mining without express permission", so take
specific assets by hand or through their API, not in bulk. It says nothing on AI training either way. No Poly Haven
asset was added.

## Next

1. Use the car meshes in a pilot cycle (FRK-148 to 150) and judge them next to the photos.
2. A jeepney and tuk-tuk are still code-made props; no CC0 source exists. Kenney's `truck`, `van` and `delivery` could stand in for a pickup, a matatu and a box lorry once the loader reads palette textures.
