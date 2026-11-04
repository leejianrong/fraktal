# ADR-0010: The 3D path: a headless Blender renderer fed by our world files, with an asset library we own or that is CC0

- Status: Accepted 2026-11-04 (answers Q18)
- Date: 2026-11-04
- Deciders: Jian

## Context

The goal (Q18, PLAN "Later bets" item 0, epic FRK-E6) is whole procedural 3D worlds, a reusable
asset library, and imagery and video from any viewpoint (satellite, aerial, drone, ground) for
dataset building. The current renderer is a 2.5D heightfield with per-class shading and sensor
effects (ADR-0004). Its measured ceiling is visible in ADR-0006: scenes are smooth, textures are
flat, and strong-edge share is the largest remaining gap. Off-nadir views, believable roofs and
real materials need real geometry, materials and light transport.

A research spike (`docs/research/3d-assets-and-realism.md`, FRK-35) compared about 25 options. Its
findings that decide this ADR: licence terms for selling a derived dataset matter more than
render quality; scraped and marketplace assets and several generative-3D model licences are the
weakest on that; Blender with Cycles is free, scriptable and headless; CC0 materials (Poly Haven,
ambientCG) are available; and no source shows that any approach closes the sim-to-real gap for
building segmentation by itself (Synthinel-1 helped only when mixed with real data), which is
consistent with ADR-0009. Many licence points in the note are marked unverified.

## Decision

1. **Renderer: Blender (Cycles) run headless as a subprocess.** It reads our files (GeoParquet
   layers, rasters and an asset manifest) and writes Cloud-Optimized GeoTIFFs plus label rasters,
   or video frames. Stages still talk through files (AGENTS.md), so Blender stays a replaceable
   consumer. Mitsuba 3 stays the fallback; Unreal, Unity and Omniverse are out until their
   licences are confirmed.
2. **The world model is the source of truth, not the renderer.** The stable-ID, per-feature
   provenance, layered files and the procedural generators (terrain, land cover, roads, parcels,
   buildings, biomes) carry over unchanged. The 3D renderer consumes the same layers the 2.5D one
   does and adds geometry and materials from the building and asset specifications.
3. **Keep the 2.5D renderer.** It is cheap, deterministic to the pixel and good for bulk nadir
   data and fast iteration. 3D is the high-fidelity and multi-view path, not a replacement.
4. **Buildings come from our own Python grammar**, not a closed tool: typologies for each biome
   (for Khartoum, courtyard houses, walled compounds, flat-roof blocks), roof types, parapets and
   roof clutter, with every choice drawn from `fraktal.ids.substream`. CityEngine and Houdini are
   avoided (closed, recurring cost, lock-in).
5. **Assets and materials are owned or CC0.** The library has a manifest with a per-row origin,
   licence and an owned or CC0 flag; CI rejects any asset or material without a row, and attribution
   obligations (CC-BY) are not allowed into the default library because they would flow to dataset
   buyers. Poly Haven and ambientCG are the first material sources, after their primary licence
   pages are checked. Objaverse bulk, Sketchfab and marketplace downloads are not used.
6. **Generative assets are an experiment, not a foundation.** Image-to-3D models (TRELLIS.2, SF3D)
   and diffusion texture or style refinement may be tried after the baseline, per model licence,
   with provenance logged and label-preservation checked. Hunyuan3D is excluded by default for its
   territory carve-out.
7. **Determinism is defined as seeded scene equality**: the same seed and config give the same
   scene description (geometry, materials, camera, sun) byte for byte. Pixel equality is not
   promised for path-traced output, since GPU denoising is not bit-exact across drivers.
8. **Measure at each stage, train at milestones.** Stage 0 is a two-day experiment before any
   build: one 1 km tile from existing footprints, extruded, with random CC0 materials, a sky and a
   few instanced props, a nadir camera at about 0.5 m GSD, rendered on one GPU pod. Record seconds
   and cost per tile, and compare side by side with a real Khartoum tile and the 2.5D render using
   the realism gauge and plots. A downstream training run waits for a milestone (AGENTS.md).

## Alternatives considered

- **Unreal or Unity headless:** big asset ecosystems and fast, but EULAs and marketplace terms
  complicate selling derived data, and the toolchain is heavy.
- **Omniverse, Isaac or Replicator:** strong synthetic-data tooling, but the 2026 licence and cloud
  terms are unclear and it needs RT-capable GPUs.
- **Mitsuba 3:** physically exact and pip-installable, but a weaker scene-authoring ecosystem. Kept
  as a fallback and for sensor research.
- **CityEngine, Houdini:** best procedural tools, but closed and costly. Reconsider only if a
  technical artist joins.
- **Scraped or marketplace assets, Objaverse bulk:** fast variety, but per-object licences and
  no-AI disputes could taint a sold dataset.
- **Generative-first (image-to-3D, splats from real imagery):** unproven for top-down fidelity, with
  weaker licence stories and restricted source imagery.
- **Stay 2.5D and improve textures only:** cheaper, and still worth doing for bulk data, but it
  cannot give off-nadir or ground views.

## Consequences

- New work, filed as FRK-E6 stories once this ADR is accepted: the Stage 0 experiment; an asset
  manifest and licence CI check; the renderer file contract and Blender subprocess runner; the
  building grammar (one typology first); a sensor-effects kernel shared with 2.5D; camera
  trajectories (oblique, drone, ground) after nadir is stable; and an Infinigen spike for
  vegetation.
- Add Blender (GPL tool, our scripts private, outputs ours; to be confirmed) to the pod image for
  GPU rendering. RunPod cost per image becomes a tracked number.
- Licence review stays a gate before the first sale (ADR-0007), now including the asset manifest
  and the OSM and Overture question (whether renders from ODbL footprints are produced works is
  unsettled).
- Domain gap is not promised away. Realism work continues in both renderers, and the benchmark keeps
  steering what to build (ADR-0009).

## Open questions

- Output licence statement for Blender and the exact terms of Poly Haven, ambientCG and any add-ons.
- Cost and throughput per image at 1 to 4k pixels, and for video.
- Whether a Blender add-on or geometry nodes is worth it against pure Python mesh building.
- How labels are produced in 3D (object-index passes against rasterised footprints) and how the
  existing label-agreement check ports across.
- Camera and sun trajectories for video, and how they are seeded.
