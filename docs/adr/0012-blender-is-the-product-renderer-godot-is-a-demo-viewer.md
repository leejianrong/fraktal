# ADR-0012: Blender is the product renderer; the 2.5D renderer is the fast nadir path; Godot is a light demo viewer

- Status: Accepted (2026-11-07, Jian); amends [ADR-0010](0010-the-3d-path-blender-renderer-and-an-owned-asset-library.md) and the phase 4 plan of [ADR-0011](0011-a-data-generation-studio-as-the-product.md)
- Date: 2026-11-07
- Deciders: Jian

## Context

Three things now draw the world: the 2.5D renderer (`render.py`), Blender with Cycles (`blender_scene.py`,
`render3d*.py`) and, since FRK-91, a Godot viewer (`godot/`). The repo shows what each does today:

- The 2.5D renderer feeds the whole benchmark spine (`synth`, `ablation`, `randomise`). Realism work lifted
  zero-shot IoU from about 0 to 0.3 (ADR-0006), but fine-tuned scores never beat real data alone.
- Blender produces everything with labels, depth, poses, tracks and video, at about 26 s a 500 m tile.
  With the sensor chain it is as close to real chips as the 2.5D path on 9 of 10 gauge pairs (ADR-0010).
  No training run has ever used 3D data; the per-biome benchmark (FRK-45) is still open.
- Godot loads an exported tile and lets a person drive. It has no labels, depth or capture, and a different
  look (plan: [godot-prototype-plan.md](../research/godot-prototype-plan.md)).

Looking at real chips next to our renders on 2026-11-07 (Open Cities drone chips, SpaceNet satellite chips,
our 3D and 2.5D output) shows the renders are still obviously artificial: straight roads and rows of
similar boxes, flat-colour roofs, blotchy green ground with blocky cells, round blob trees, no walls,
compounds or large industrial buildings, and almost no street life. Jian's direction: make the renders more
realistic first, put the investment in Blender, keep Godot light.

## Decision

1. **Blender is the product renderer.** Every dataset or output that needs exact labels, depth, poses,
   tracks or video is captured by Blender, offline, on rented GPUs. Realism investment (layout, assets,
   materials, light, lens) goes here first.
2. **The Python world model stays the source of truth** (ADR-0003). Layout realism (roads, parcels,
   footprints, compounds, industrial buildings, land cover) is done in the generators, so it benefits all
   three views. The exported scene files are the contract between the world and any renderer.
3. **The 2.5D renderer is kept for fast nadir data and the benchmark spine, in maintenance.** It gets
   coverage knobs and the generator improvements for free; it gets no new geometry features.
4. **Godot is a light demo viewer** for demos and marketing: drive a car, fly a drone, see the simulated
   agents move, share a small Windows build. It does not capture products. The capture cards (FRK-96, 97,
   98) are parked. They return only if a customer needs closed-loop robotics simulation.
5. **No 3D-data training until the qualitative review and the chosen improvements are done.** The first
   3D-data benchmark arm (FRK-45) waits for the realism gap analysis and the gaps Jian picks (epic FRK-E9).
   Day to day, realism is judged by looking at samples next to the real reference board, with the gauge as a
   secondary check (the gauge once reported better colour while roofs were pale blue).
6. **Real imagery is for looking and for evaluation only.** The reference board uses Open Cities AI (ODbL-1.0),
   SpaceNet 2 and Maxar Open Data (CC BY-NC 4.0, cite as "Maxar Open Data Program was accessed on DATE from
   https://registry.opendata.aws/maxar-open-data"), kept under `data/eval/`, never trained on,
   redistributed, committed or used in sales material. Only our own synthetic strips go in the repo.

## Consequences

- ADR-0010 decision 3 stands in spirit (2.5D kept) but is narrowed to maintenance; ADR-0011 phase 4
  becomes a demo, not a capture path, and its open question 3 is answered: interactive driving is a demo
  feature, authored and simulated capture is the product.
- Epic FRK-E8 is re-scoped to the light viewer; epic FRK-E9 holds the reference board, the gap analysis and
  the Blender realism work, and blocks FRK-45.
- Walled compounds and large industrial buildings are in scope for the layout work, which means a decision on
  their label classes.
- Any realism asset must be owned, CC0 or generated, with a manifest row (ADR-0010 decision 5).

## Alternatives considered

| Option | Why not |
|---|---|
| Capture in Godot too (FRK-96 to 98) | A second implementation of exact labels, the product's central promise; Blender already has it |
| Run the 3D benchmark arm first (FRK-45) | The renders still look artificial; a flat result would not tell us whether realism or 3D is the cause |
| Retire the 2.5D renderer | It drives the benchmark spine and bulk nadir data, and is cheap |
| Put realism work in the renderers only | Layout lives in the generators; fixing it there helps every view |

## Addendum 2026-11-07: label classes and the SpaceNet colour

Recommendations adopted by Jian on 2026-11-07 ("make a recommendation and we'll go with that").

**Label classes (3D path, `labels3d.CLASSES`; ids are append-only, existing ones do not change)**

- **Walls and fences are a new class, `wall` (8).** One instance per compound. Gates are gaps (class `none`
  or ground). The benchmark target is `building`; a wall merged into it would add thin linear false positives
  that real footprint labels (OSM, Open Cities) never mark, and a separate class lets a customer drop it.
- **Large industrial buildings stay `building`.** The task is footprints and real labels include warehouses.
  The scene's per-building `typology` (for example `industrial_hall`) is the attribute that tells them apart,
  so "industrial only" is a filter on instance ids, not a second class.
- **Yard objects (storage tanks, containers, silos) are a new class, `yard_object` (9).** They are not
  buildings in OSM or in the real labels, and from above they must not count as building pixels. Tanks and
  panels on a roof stay part of the building, since from above they are roof.
- The 2.5D renderer keeps `building`, land cover and road only. Walls and yard objects exist in the 3D path.

**SpaceNet colour**

- The band order is correct: the files are `RGB-PanSharpen`, and a per-band stretch gives a natural
  reddish-brown desert city. The green cast is an artefact of our own stretch: `chips.stretch_range` uses one
  2nd to 98th percentile range for all three bands, while the raw blue band is far lower (medians 534, 592
  and 409 for R, G, B).
- `KHARTOUM_TONE` (0.66, 0.94, 0.31), `palette.json` and the synthetic Khartoum chips were fitted to that
  cast, so the benchmark pair (synthetic against real chips) is internally consistent and the ADR-0006 and
  ADR-0009 numbers stay valid. We do not change the chips, palette or tone now: it would invalidate those
  results and need GPU time, and training waits for the milestone.
- SpaceNet is not a colour target for realism. Colour decisions use Maxar and Open Cities, which are
  colour-correct. The reference board shows SpaceNet with a per-band stretch.
- At the FRK-45 milestone run, the first step is to rebuild the SpaceNet chips with a per-band stretch, refit
  the palette, drop the cast-matching default tone, and record a new baseline.
