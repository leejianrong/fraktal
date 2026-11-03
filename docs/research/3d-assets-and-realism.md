# Research note: 3D assets, materials and realism for Fraktal

Date: 2026-11-04. Status: research spike, not a decision. Needs an ADR before any build (Q18).
Verification key: [V] seen in a source this session; [U] unverified, from prior knowledge or a
secondary source. Licence statements are NOT legal advice; ADR-0007 already requires a licence
review before first sale, and every [U] below must be checked against the primary text.

## Summary and recommendation

Realism will come mostly from three things, in this order: (1) a physically based path-traced
renderer with real sun/sky, camera and sensor models, (2) a small, hand-curated and
procedurally varied library of assets we own or that are CC0, and (3) PBR materials with
weathering, roof clutter and per-scene randomisation. Do not start from scraped or
generative-AI assets: the licence story for selling derived data is weakest there. Recommended
path: drive Blender (Cycles, GPL tool but outputs are ours [U: Blender states output is not
GPL-encumbered]) as a subprocess from our existing GeoParquet footprints; build buildings with
our own Python/geometry-nodes grammar (extrude footprint, roof type, parapet, clutter) using
CC0 PBR materials from Poly Haven and ambientCG; place vehicles and trees from a small CC0/own
library; then measure with the V3 benchmark, and only then consider a diffusion or GAN
refinement pass on top. AI image-to-3D (TRELLIS.2, Hunyuan3D, SF3D) is a later, optional
asset-multiplier whose outputs need per-model licence review; Hunyuan is excluded for us as
a default because of its territory carve-out. Honest uncertainty: no source found shows that
a given approach closes the sim-to-real gap for building segmentation; Synthinel-1 [V] helped
only when mixed with real data, which is consistent with ADR-0009 (build, measure, steer).

## Comparison table

GPU cost figures are rough planning estimates [U]; measure on RunPod before relying on them.

| Option | What it gives us | Licence for selling derived images | Effort | GPU cost | Realism ceiling | Verdict |
|---|---|---|---|---|---|---|
| Blender + Cycles, headless, own Python scene builder | Full 3D, any viewpoint, PBR, sun/sky, video | Blender is GPL; rendered output is ours [U]. Own scripts private (check add-on licences) | Medium-high | ~0.3-3 min/frame at 1-4k px on one 4090-class GPU [U] | High | PRIMARY renderer candidate |
| Mitsuba 3 (pip, differentiable) | Physically exact, scriptable, no Blender dep | BSD-style [U] | Medium; weaker ecosystem for scene authoring | Similar to Cycles | High | Secondary; good for sensor/BRDF research |
| Unreal / Unity headless | Fast, big asset ecosystems | Engine EULAs plus per-asset marketplace terms; marketplace assets often bar redistribution of raw assets, rendered images usually fine [U] | High, heavy toolchain | Moderate | High | Avoid for now (licence and ops weight) |
| NVIDIA Omniverse / Isaac / Replicator | Synthetic-data tooling, RTX path tracer | NVIDIA licence changed in 2026; a forum post says free for production [V, forum only]; cloud and GPU-count terms unclear [V] | High | RTX GPUs only (not A100/H100 for RT) [U] | High | Avoid until licence confirmed with NVIDIA |
| three.js / Open3D / raster | Cheap, simple | MIT/BSD [U] | Low | Near zero | Low-medium (no GI) | Previews only |
| Infinigen (terrain, plants, creatures, rocks) | Procedural nature in Blender, ground-truth labels | 3-Clause BSD [V] (check repo for 2026 changes); GPU scene gen needs CUDA [V] | Medium | Moderate | High for vegetation and terrain | TRY for trees and ground cover; not for dense urban |
| Infinigen-Indoors | Interiors | BSD [V] | n/a | n/a | n/a | Irrelevant for overhead imagery |
| BlenderProc | Scripted Blender data generation, annotations | GPL-3.0 [U] | Low-medium | As Cycles | High | Use as reference for annotation outputs; avoid linking its code into a private engine without review |
| Esri CityEngine (CGA) | Mature shape grammar | Commercial subscription; output rights per Esri terms [U] | Medium | none | High | Avoid; closed, recurring cost, lock-in |
| Houdini / Houdini Engine | Best procedural tooling | Commercial; Apprentice is non-commercial [U] | High | none | High | Avoid for now; consider later if a TD is hired |
| Blender geometry nodes + Buildify (free) | Modular procedural buildings [V, 2022, check Blender compat] | Check Buildify's licence [U] | Low-medium | none | Medium-high | Good for a 2-day look; own grammar later |
| Paid add-ons (GIS2BLEND, MBG, blender-osm) | OSM import, building generators [V] | Add-on EULA plus OSM ODbL; unclear [U] | Low | none | Medium | Evaluate only; do not depend on |
| Own Python grammar (footprint to massing, roofs, clutter) | Full control, deterministic, ADR-0003 compatible | Ours | Medium | none | Medium-high | BUILD |
| Poly Haven, ambientCG (PBR textures, HDRIs, models) | CC0 materials, skies, some models | CC0, sale of derived images fine [V via secondary guides; confirm on the sites] | Low | none | High | USE |
| Kenney | CC0 low-poly assets | CC0 [V secondary] | Low | none | Low (stylised) | Placeholder props only |
| Objaverse / Objaverse-XL | Millions of objects | Dataset ODC-By but each object has its own licence; many NC, SA, or scraped from Sketchfab with NoAI disputes [V] | Medium to filter | none | Medium, wild quality | AVOID as bulk source; at most CC0 subset with provenance |
| Sketchfab downloads | Quality models | Per-model; CC-BY needs attribution flowing to dataset consumers; NoAI tag and ToU [V]; Standard licence disallows AI data use [V] | Medium | none | High | AVOID for production; CC0 only |
| TurboSquid / marketplaces | Commercial models | Royalty-free licences usually allow rendered images but forbid redistributing the models; AI-training clauses vary [U, not verified] | Low | none | High | Possible paid fallback after reading licence for dataset resale |
| OSM-derived 3D / Overture buildings | Real footprints and heights | OSM is ODbL; Overture Buildings is ODbL, some heights CDLA-Permissive [V]; whether renders are produced works vs derivative databases is unsettled [V, no definitive analysis found] | Low | none | n/a (geometry only) | Already gated by ADR-0007; keep a provenance manifest |
| TRELLIS.2 (Microsoft, image-to-3D, 4B) | PBR meshes from one image | Weights MIT [V]; training-data and output provenance [U]; needs ~24 GB VRAM [V] | Medium | ~tens of seconds per asset [U] | Medium; top-down fidelity poor [U] | TRY later as asset multiplier |
| Hunyuan3D 2.1 / 2.0 | Image/text to 3D | Tencent community licence excludes EU, UK, South Korea [V]; output terms and MAU limit not confirmed [V partial] | Medium | Moderate | Medium-high | AVOID as default (territory risk for a dataset business) |
| Stable Fast 3D | Fast single-image mesh | Stability Community Licence: free under USD 1M revenue, enterprise above [V]; revenue test is org-wide [V secondary] | Low | Seconds | Medium-low | Possible while small; plan for the threshold |
| Gaussian splats / NeRF from real imagery | Photoreal scenes | Depends on source imagery licence; Maxar-type imagery is restrictive [U] | High | High | Photo | AVOID for asset creation; research on realism only |
| Material Anything (MIT [V secondary]), MaterialMVP (licence unconfirmed [V]) | PBR textures on meshes | Check each | Medium | Moderate | Medium | LATER; CC0 PBR first |
| Diffusion texture or style refinement of renders | Closes appearance gap | Model licence matters; SDXL/SD3.5 under Stability community terms [V]; FLUX dev is non-commercial [U] | Medium | Moderate | Potentially high, can hallucinate and break labels | TRY after baseline, with label-preservation checks |

## How aerial synthetic data projects got realism (what we can verify)

- Synthinel-1 [V]: virtual cities rendered in a game-style engine; gain only when mixed with real
  data, not stand-alone. SIMPL [V] frames the same issue as a visual domain gap.
- RarePlanes (AI.Reverie) [V as description only]: engine-rendered planes over real-world-like
  backgrounds, mixed with real data; synthetic helped most in the few-shot regime [U].
- SpaceNet, AIRSIM-style, GTA-style, and Omniverse projects: could not verify details this
  session; treat any claim as [U].
- Common pattern [U]: heavy domain randomisation (sun angle, haze, colour grading, texture swaps),
  mix with real data, and sensor modelling (PSF blur, noise, JPEG/pan-sharpening artefacts).
- Refinement: CycleGAN or diffusion img2img on renders; no source found showing it closes the
  gap for these benchmarks [V negative search]. Treat as an experiment, not a promise.

## Khartoum-like materials and clutter (plan, mostly [U])

- Roofs: corrugated metal (rust, streaks, bright specular glints, seam lines), flat concrete
  with parapet, tar or bitumen patches, clay tile. Walls rarely matter from nadir; prioritise roofs.
- Ground: dust and compacted earth, from ambientCG/Poly Haven soil and sand sets plus our own
  macro-variation noise (patchy tone, tyre tracks, rubbish).
- Roof clutter as instanced props: water tanks (black plastic cylinders), satellite dishes,
  solar panels, AC units, laundry, stored material. Cheap to model by hand; high visual payoff.
- Weathering: dirt in AO/edges, rain streaks, colour drift between neighbours; per-building
  random albedo/roughness jitter. Check Blender-native procedural nodes before adding AI.
- PBR from photos: scanning our own materials with a phone (photogrammetry or a flat-lay
  capture) gives owned, licence-clean textures; real Maxar tiles must NOT be used as texture
  sources (licence, plus benchmark-leakage rule, ADR-0006).
- Sensor effects: sun-elevation shadows, atmospheric haze, MTF blur, pan-sharpening halos,
  noise, off-nadir tilt, JPEG artefacts. These are cheap post-render steps in our kernel seam.

## Recommended staged path

### Stage 0: 2-day experiment (cheap, reversible)

Goal: see whether Blender Cycles with CC0 materials already looks closer to Khartoum than the 2.5D render.
1. Pick one 1 km tile from existing footprints. Script Blender (`blender -b -P`) to extrude
   footprints with flat or gabled roofs and a plane for ground.
2. Assign 6 to 10 CC0 roof/ground materials from Poly Haven and ambientCG, randomised per building.
3. Add a Nishita sky or Poly Haven HDRI, sun elevation sampled via `fraktal.ids.substream`
   (determinism, ADR-0003). Add 4 instanced props (tank, AC unit, car, tree) from CC0 or modelled by hand.
4. Orthographic nadir camera at about 0.5 m GSD, 64 to 128 samples with denoising, one RunPod
   GPU. Record seconds per tile and cost per image (this fills a gap in our numbers).
5. Compare side by side with a real Khartoum tile, with a human check and, if cheap, a
   downstream segmentation score. Record the result even if negative (ADR-0009).
Output: a short result note, and the licence list of every asset used.

### Stage 1: build (after an ADR for Q18)

- Renderer contract: stages exchange GeoParquet plus an asset manifest; Blender runs as a
  subprocess reading files, emitting COG plus label rasters (matches "stages talk through files").
- Own building grammar in Python (typologies for Khartoum: courtyard houses, walled compounds,
  flat-roof blocks), with roof clutter and material assignment all seeded.
- Asset library with a manifest recording origin, licence, and a CC0 or owned flag; a CI check
  that rejects any asset without a licence row.
- Infinigen as optional supplier of trees, shrubs, rock and terrain; only after a spike
  confirms current licence and that it runs headless on RunPod.
- Sensor-effects module in `src/fraktal/kernels/` (arrays in, arrays out, golden test).
- Multi-view: add oblique/drone camera trajectories only after nadir is stable.

### Stage 2: widen and refine

- Wider biome materials for South-East Asia (corrugated, tile, vegetation).
- Experiment with TRELLIS.2 and SF3D as asset multipliers on our own concept images; log
  per-asset provenance; human QA gate.
- Experiment with a diffusion or CycleGAN refinement pass, evaluated by label preservation.

### Avoid

Scraped marketplaces and Objaverse bulk; Hunyuan3D by default; CityEngine and Houdini lock-in;
Omniverse until NVIDIA confirms cloud terms; using real Maxar imagery as texture or style
source; polishing one look (ADR-0009: widen the distribution).

## Key risks

1. Licence contamination of a sold dataset: a single CC-BY-NC or SA asset or model-weight
   restriction can taint product. Mitigation: asset manifest with licence per row, CC0/owned default.
2. OSM/Overture ODbL: unsettled whether renders from OSM footprints are produced works [V]. Needs
   a lawyer before the first sale (already a gate in ADR-0007).
3. CC-BY attribution must flow to dataset consumers; cheaper to avoid.
4. Model licences change: Stability and NVIDIA both changed terms recently [V]; re-check before each sale.
5. Domain gap may stay large (Synthinel-1 lesson [V]); do not promise stand-alone training value.
6. Cost and throughput: path-traced video at scale is expensive; denoising and low sample
   counts are necessary [U numbers].
7. Benchmark leakage: held-out footprints must never enter a world (ADR-0006).
8. Blender and GPL tooling: scripts that import GPL add-ons (BlenderProc) may carry obligations if
   ever distributed; the engine is private, but check before open-sourcing (ADR-0007).
9. Determinism: Cycles with GPU denoising is not bit-exact across drivers [U]; define the
   contract as seeded scene equality, not pixel equality.

## Not verified this session

CityEngine and Houdini terms; Blender output licence; Mitsuba licence and pip availability;
RunPod and per-image cost; SpaceNet, RarePlanes and AirSim details; TurboSquid licence terms;
Hunyuan 2.1 output ownership and MAU threshold; MaterialMVP, FLUX and TRELLIS.2 training-data
provenance; Poly Haven and ambientCG primary licence pages; Infinigen's current release state.

## Sources

- Infinigen overview and licence: https://www.cgchannel.com/2025/06/open-source-tool-infinigen-indoors-generates-procedural-3d-interiors/
- Infinigen original release: https://www.cgchannel.com/2023/07/infinigen-a-free-generator-for-procedural-3d-environments
- Infinigen repo (check directly): https://github.com/princeton-vl/infinigen
- Hunyuan3D 2.1 licence mirror: https://gitserver.onethingai.com/ai-models/Hunyuan3D-2.1/raw/branch/main/LICENSE
- Hunyuan3D 2.0 licence (HF): https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE
- TRELLIS.2 model card: https://huggingface.co/microsoft/TRELLIS.2-4B
- Stability AI licence tiers: https://stability.ai/license
- Stability licence update: https://stability.ai/news/license-update
- Objaverse-XL: https://huggingface.co/datasets/allenai/objaverse-xl
- Objaverse docs: https://objaverse.allenai.org/docs/intro
- Sketchfab and AI: https://sketchfab.com/blogs/community/restricting-generative-ai-use-of-free-models
- Sketchfab NoAI tags: https://sketchfab.com/blogs/community/introducing-the-noai-createdwithai-tags/
- Sketchfab/Objaverse dispute: https://the-decoder.com/sketchfab-objaverse-ai-copyright-dispute-enters-third-dimension/
- Overture Buildings guide: https://docs.overturemaps.org/guides/buildings
- Synthinel-1: https://arxiv.org/pdf/2001.05130
- SIMPL: https://arxiv.org/pdf/2106.15681
- Material Anything: https://arxiv.org/pdf/2411.15138
- MaterialMVP: https://arxiv.org/pdf/2503.10289
- Omniverse licensing change (forum): https://forums.developer.nvidia.com/t/nvidia-omniverse-licensing-change/375141
- Omniverse product terms: https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/NVIDIA-Product-Specific-Terms-for-Omniverse-2025.01.30.pdf
- Buildify: https://www.cgchannel.com/2022/07/download-free-blender-3d-building-generator-buildify/
- GIS2BLEND: https://superhivemarket.com/products/gis2blend
- CC0 asset guide (secondary, confirm on sites): https://app.cinevva.com/guides/free-textures-hdris-materials
- Poly Haven licence (primary, to check): https://polyhaven.com/license
- ambientCG licence (primary, to check): https://docs.ambientcg.com/license/
