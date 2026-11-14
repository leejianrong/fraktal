# Research spike: how people make hyperrealistic synthetic worlds, and what we can use

Written 2026-11-14. Question: what techniques, methods and tools do people use to make hyperrealistic worlds,
and can Fraktal use the same ones? Builds on [realism-gap-analysis.md](realism-gap-analysis.md),
[3d-assets-and-realism.md](3d-assets-and-realism.md), [generative-assets-plan.md](generative-assets-plan.md) and
[ADR-0013](../adr/0013-which-generative-models-may-make-assets-for-sold-datasets.md). Judged against the rules in
[AGENTS.md](../../AGENTS.md) and ADR-0007, 0009, 0010 and 0012.

## Short answer

- People take four routes: physically based path tracing of procedural scenes (film, archviz, Synscapes),
  procedural worlds built from scanned or PBR assets (the Matrix Awakens city, Infinigen), domain randomisation
  (variety over polish), and neural enhancement of rendered frames.
- The studio tricks are mostly not about the renderer. Cycles already does the light. What separates a
  studio frame from ours is surface imperfection (wear, dirt, cracks, decals), natural-looking detail at every
  scale (trees, people, vehicles, clutter), a believable camera (exposure, lens, noise, compression) and varied,
  correct lighting.
- The evidence on whether realism pays for training is mixed. One careful street-scene study says yes. Several
  object-detection studies say variety matters more, with lighting realism the part worth the effort. That fits
  ADR-0009: keep widening coverage, and spend realism effort where it is cheap or where lighting is involved.
- Most of what is worth copying fits our rules. The two routes that do not fit are neural enhancement trained on
  real driving datasets and anything built from the real reference photos.
- Our eye-level strips have no camera model at all. That is the cheapest big step and I would do it first.

## How far to trust this

One pass of web searches on 2026-11-14, read as search summaries, not as full papers. Verified from the sources
linked at the end: the Synscapes, Infinigen and Enhancing Photorealism Enhancement claims at abstract level, and
the Matrix Awakens procedural pipeline. Practitioner tips (grain, chromatic aberration, weathering) come from
forum and blog posts, so they are opinion. Not read: licences of the tools named below unless stated, the full
text of the studies on realism versus randomisation, anything about MakeHuman's output licence. Where a row says
"unverified", check before use. Nothing was run; the eye-level observations come from looking at our own sheets.

## The four routes

**1. Path-traced physical realism, end to end.** Film and archviz do this, and so does Synscapes, a 25,000 image
street-scene dataset from procedurally generated scenes. Its authors model sun and sky, scene geometry and
materials, and the camera's optics, sensor and processing, rendered with unbiased path tracing. They conclude that
networks show no sign of abstracting away the domain shift, so the sensor chain is part of the realism, not an afterthought.

**2. Procedural world plus real assets.** The Matrix Awakens city was built by a small team with procedural tools,
Houdini among them, rule sets for layout, and AI systems for traffic and crowds. Infinigen (Princeton, CVPR 2023)
makes natural scenes entirely from rules inside Blender: terrain from signed distance fields and noise, plants from
growth algorithms, and 50 procedural materials, rendered with Cycles. Both put the effort into variety and detail
at every scale, not into a cleverer renderer.

**3. Domain randomisation.** Vary lighting, textures, object placement and nuisance factors widely, and let the
network learn what does not matter. Studies on industrial and household objects report it matching or beating
photoreal data, and a hybrid of both is common.

**4. Neural enhancement.** Enhancing Photorealism Enhancement (Intel, 2021) takes the G-buffers of a game render and
restyles the frame toward real driving datasets (Cityscapes, KITTI, Mapillary Vistas) with an adversarial network.
It looks strikingly real and ran at about 2 frames per second on an RTX 3090. Diffusion refinement of renders is
the newer version of the same idea.

## Does realism pay for training?

- **For:** Synscapes beat the Synthia and GTA V based datasets on segmentation and detection, with the authors
  concluding that high realism, including the sensor, is worth aiming for.
- **Against, or mixed:** detection studies lean toward randomisation, one finding BlendTorch randomised data beating
  photoreal and even real training sets. A neural domain randomisation paper reports that realistic lighting is the
  part that matters most for generalising.
- **Cost:** a 2024 paper notes that high realism can cost more than collecting real data.

The sources disagree because the tasks differ. For us the useful reading is: make lighting and the camera right,
keep adding variety, and treat surface polish as a steer, which is what ADR-0009 already says.

## Where we stand

Already built: a Nishita sky and per-city light (FRK-152, 155), normal and roughness maps derived from albedo and
wired into building materials (FRK-126), a sensor chain with blur, noise, grade and drone lens effects
(FRK-16, 50, 57, 114) used for aerial renders only, procedural buildings, trees, cars, people and cables.

Visible gaps in our own eye-level sheets:

- Crowns, vehicles and people are low-poly, and vehicles are boxes (FRK-163).
- Walls show fine speckle at a distance where plaster should read as a flat tone.
- Roads are a uniform grey with painted lines. Real roads have patches, cracks, stains, polished tyre lanes and
  dirty kerb edges.
- Frames are clean. The reference photos are phone and dashcam images with noise, compression, lens softness and
  often a dashboard or windscreen in view.

## Technique by technique

Verdicts: **do next**, **try**, **later**, **no**. "Fits" means it passes the rules: owned or CC0 assets with a
manifest row, determinism through `fraktal.ids.substream`, exact labels, no real photo feeding a generator.

| Technique | What it buys | Our status | Fits? | Effort | Verdict |
|---|---|---|---|---|---|
| Eye-level camera chain: exposure and tone mapping, lens softness and tiny aberration, vignette, sensor noise, JPEG or video compression, optional dashboard and windscreen dirt | The gap between a clean CG frame and a phone or dashcam frame. Synscapes argues it matters | Aerial `--sensor` exists, eye-level has none | Yes. Applied after the labels, with any geometric change applied to the labels too | Small, the kernel exists | **do next** |
| Fix the wall speckle: texture filtering, lower-frequency noise, a denoise pass | Removes a visible CG tell on most pixels of a street frame | 32 samples, denoise off | Yes. Check labels do not change (they use their own pass) | Small | **do next** |
| Weathering and decals: edge wear and cavity dirt from curvature and AO masks, grime at wall bases, rain streaks, road cracks, patches, manholes, stains | Surface imperfection, the main studio realism trick | Wall texture has stains and a cap line (FRK-132), roads none | Yes if the decals are procedural or CC0 | Medium | **do next** |
| Lighting variety: sky parameters, cloud shadows, CC0 HDRIs, low sun, haze volume for dust | Lighting realism is the part the studies agree on | One sky model, per-city tweaks | Yes. Poly Haven HDRIs are CC0, licence page to be read | Medium | **try** |
| Better vegetation: leaf-card clusters with alpha, or Infinigen trees (BSD-3 code) | Silhouette and light through leaves, the next step after FRK-162 | Smooth blob crowns | Yes if cards use our own or CC0 leaf textures. Infinigen needs its own venv and a trial (infinigen-spike.md) | Medium to large | **try** |
| Vehicles and people from meshes: Poly Haven and Quaternius CC0 packs, Step1X-3D geometry (allowed in ADR-0013), more code-written props | Real shapes, wheels, bodies, humanoid figures | Boxes | CC0 yes. Quaternius is stylised low poly, so it helps shape but not photo realism. Realistic people with a clean licence are hard: MakeHuman's output licence is unverified, SMPL-based sets are usually non-commercial | Medium to large | **try**, vehicles first |
| Rules-based layout richness (Matrix style): mixed uses, irregular lots, varied building types | Variety at the scale of a block | Grammar and city biomes exist | Yes | Large | **later** |
| Photogrammetry or Gaussian splats from our own captures | Real fine detail of facades, props and ground | None | Yes only from our own photos (owned). Never from the reference photos (guardrail 1) | Large, needs capture and a new asset kind | **later** |
| Neural enhancement of renders (Enhancing Photorealism Enhancement) | Very high perceptual realism | None | **No.** It learns style from real driving datasets, which breaks guardrail 1 and needs their licences read; it moves pixels, so exact labels are lost; 2 frames per second | n/a | **no** |
| Diffusion refinement of our renders | Possibly the same effect | None | Unclear. SDXL is allowed (ADR-0013), but the guardrail bans image conditioning, and the output drifts from the labels. ControlNet licences not read | n/a | **no, until Jian rules** (see below) |
| More domain randomisation (palette, clutter, time of day, weather) | Coverage, which the evidence rates highly | Per-city profiles, per-seed light | Yes | Small to medium, rolling | **keep doing** |

## What I would do, in order

1. **Eye-level camera chain.** Reuse the sensor kernel, add exposure and tone mapping, a small lens model, noise
   and compression, and an optional dashboard or windscreen layer for dashcam rigs. Measure with
   `fraktal streets` (brightness, contrast, edge energy) and by looking at the sheets.
2. **Wall speckle and denoise.** One short card; look at a crop before and after.
3. **Weathering and decals** for roads first (it is the largest area in every frame), then wall bases.
4. **Lighting variety**, including cloud shadows and a haze volume for Khartoum.
5. **A vegetation and vehicle trial** against the current crowns and boxes: leaf-card clusters, one CC0 vehicle
   pack, and the Infinigen trial already planned.

Each step should show up in our own checks: the class shares, the brightness and contrast statistics, and a sheet
next to the photos. Per ADR-0009 a step that does not move any of them is still kept if it adds variety.

## Open questions for Jian

- **Image-to-image on our own renders.** The guardrail says reference imagery never feeds a generator. Does that
  also rule out SDXL or FLUX image-to-image on our own synthetic frames? If not, a structure-preserving refinement
  could be tried as an experiment, with a label check, but it is a large step and I would not start it first.
- **Our own photographs.** Can anyone capture street photos or short videos that the project owns? That is the
  only legal way into photogrammetry and the one thing no procedural tool replaces.

## Sources

- Wrenninge and Unger, [Synscapes: a photorealistic synthetic dataset for street scene parsing](https://arxiv.org/pdf/1810.08705)
- Raistrick et al., [Infinite photorealistic worlds using procedural generation (Infinigen, CVPR 2023)](https://arxiv.org/pdf/2306.09310)
- Richter, Abu AlHaija and Koltun, [Enhancing photorealism enhancement](https://arxiv.org/pdf/2105.04619)
- 80.lv, [Breakdown: creating the Matrix Awakens in Houdini and Unreal Engine 5](https://80.lv/articles/breakdown-creating-the-matrix-awakens-in-houdini-unreal-engine-5), and Epic's [City Sample documentation](https://dev.epicgames.com/documentation/unreal-engine/city-sample-project-unreal-engine-demonstration)
- Tremblay et al., [Training deep networks with synthetic data: bridging the reality gap by domain randomization](https://arxiv.org/pdf/1804.06516); Eversberg and Lambrecht, [Realism versus domain randomization for an industrial detection task](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659618/); [BlendTorch](https://arxiv.org/pdf/2010.11696); [Photo-realistic neural domain randomization](https://arxiv.org/pdf/2210.12682)
- Practitioner tips: BlenderGuru, [Achieving photorealism in Blender](https://www.blenderguru.com/posts/achieving-photorealism-in-blender); Blender Artists, [Who has tips for photo-realism in Cycles](https://blenderartists.org/t/who-has-tips-for-photo-realism-in-cycles/1132314); StraySpark, [procedural weathering with Geometry Nodes](https://www.strayspark.studio/blog/procedural-weathering-blender-geometry-nodes) and [film look](https://www.strayspark.studio/blog/blender-film-look-halation-grain-gate-weave)
- Free assets: Poly Haven and Quaternius, as summarised by [Cinevva's guide to free 3D model sites](https://app.cinevva.com/guides/free-3d-model-sites); licence pages themselves not read
