# Spike: real-time engine licences and feasibility (FRK-83)

Status: terms read on 2026-11-05 as far as the sources allowed. This is the engine spike that
[ADR-0011](../adr/0011-a-data-generation-studio-as-the-product.md) calls for in phase 4. Nothing was
installed and no engine was run, so every feasibility statement below is documentation or forum evidence,
not a test. Not legal advice.

How to read the evidence: "primary" means the vendor's own page, fetched on 2026-11-05. The fetch tool
summarises pages, so quotes marked primary are as the tool returned them, not re-read in the browser.
"Secondary" means a third party. **Epic's own licence pages (unrealengine.com) returned HTTP 403 to every
fetch, so all Unreal terms below are secondary and must be re-read by hand before any decision.**

The question: which engines can we build on if we sell a studio (users drive vehicles and fly drones in
generated worlds, we render on rented GPUs, users capture images, video and labels) and may also ship a
desktop or web client?

## Summary table

| | Godot 4 | Unreal Engine 5 | Unity 6 | Blender (EEVEE) |
|---|---|---|---|---|
| Licence | MIT | Epic EULA (proprietary) | Unity Terms of Service (proprietary) | GPL v2+ (Cycles Apache-2.0) |
| Royalty | none | 5% above $1M per product (secondary) | none (runtime fee cancelled) | none |
| Per-seat fee | none | not found for our use | $2,310/seat/yr Pro above $200k revenue or funding | none |
| Hosted or SaaS use | allowed | unclear, secondary hints at royalty on service revenue | restriction wording needs a lawyer | allowed for the tool, GPL duties if we distribute it |
| Ship runtime to users | yes, with notice | yes, royalty applies | restricted, see below | yes, GPL source duties |
| Headless GPU on Linux | partial, unverified | works in Docker per forums | not researched | EEVEE needs OpenGL, awkward |
| glTF in | yes, runtime | via plugin or Datasmith, not checked | via package, not checked | yes, and USD |
| Label and depth passes | custom shaders | built in (stencil, depth), not verified here | built in, not verified here | built in, offline only |

## Godot

Primary, https://godotengine.org/license/, checked 2026-11-05:

- "free and open source software released under the permissive MIT license", usable "for any purpose".
- "you need to distribute the copyright notice and license statement of Godot Engine whenever you
  redistribute it."
- Users may "distribute unmodified and changed versions of Godot Engine, even commercially."
- No royalty is mentioned, and the terms do not apply to content made with Godot.

Verdict on terms: the cleanest of the four. SaaS, desktop client and web client are all fine; the only
duty is carrying the notice. Contributors' third-party components (for example physics or audio libraries)
carry their own notices; the engine's third-party licence file was not read.

Feasibility, from https://docs.godotengine.org/en/stable/tutorials/editor/command_line_tutorial.html:

- `--headless` is "Enable headless mode (`--display-driver headless --audio-driver Dummy`). Useful for
  servers and with `--script`." The same page says it is "required on platforms that do not have GPU
  access". A headless display driver uses a dummy renderer, so it does not by itself produce frames.
  Rendering on a GPU pod probably needs a virtual display (Xvfb) or a GPU-capable driver path. The page
  does not say, and I did not test it. Open question.
- `--write-movie` writes AVI or PNG sequences with fixed fps, which fits deterministic capture.
- glTF: "can load and save glTF files in exported projects, in both text (.gltf) and binary (.glb)
  formats" (https://docs.godotengine.org/en/stable/tutorials/io/runtime_file_loading_and_saving.html).
  USD is not mentioned there; assume no native USD.
- Labels and depth: the docs do not describe a built-in segmentation or object-ID pass, and depth is only
  reachable through a shader (`hint_depth_texture`). A label pass would be our own: a second viewport or
  material override that draws each instance in a flat ID colour. That is doable but is work we own.
  (The override approach is general knowledge, not something I verified in the docs.)

## Unreal Engine

Primary sources were blocked (403) for https://www.unrealengine.com/en-US/eula/unreal, `/license`,
`/eula/publishing` and the EULA change log. What I have is secondary:

- ConductAtlas, a third-party EULA tracker (https://conductatlas.com/platform/unreal-engine/unreal-engine-eula/),
  summarises: a 5% royalty, with the first $1,000,000 lifetime gross revenue per Royalty Product excluded;
  a 3.5% rate for "Launch Everywhere with Epic"; and that Epic may require seat subscription fees for
  non-royalty commercial use, with no amount given. Its own pages disagree on whether 5% applies to all
  revenue or only the part above $1M, and it notes a newer version may exist. Treat as unverified.
- The same search surfaced a statement, attributed to the EULA change log, that a recent revision says
  revenue from streaming, subscription or other game-delivery services is subject to royalty. Secondary and
  second-hand.
- Epic staff on forums (secondary) say linear non-interactive video rendered with UE is not royalty-bearing,
  and that the test is whether the rendered media is interactive. Our product is interactive (drive and fly)
  and sells a service, so that carve-out likely does not cover it.
- Epic staff on forums also say standard terms are non-negotiable but a custom licence can be requested.

Feasibility (forum and vendor-blog evidence, secondary): Epic publishes Linux container images for Pixel
Streaming (`ghcr.io/epicgames/unreal-engine:runtime-pixel-streaming`), and several threads run UE in
Docker on NVIDIA hosts, with Vulkan driver version problems reported. Source access also needs an Epic
account linked to GitHub. Segmentation (custom stencil values) and depth are well known UE features, but
I did not verify them in the docs, nor the glTF and USD import status.

Verdict on terms: cannot be settled from what I could read. A sold, interactive, cloud-rendered tool is
the case most likely to hit royalty or need a custom licence. Highest quality and best streaming story,
highest legal uncertainty.

## Unity

Primary, https://unity.com/products/pricing-updates (announced 2025-11-10, Pro and Enterprise changes
effective 2026-01-12):

- Personal: free, "available to customers with up to $200,000 in revenue and funding."
- Pro: $2,310 per seat per year prepaid, or $210 per seat per month; required above $200,000 in revenue or
  funding.
- Enterprise: custom, required above $25 million annual revenue.
- The runtime fee "was canceled as of September 12, 2024". The page says nothing on servers or cloud
  rendering.

Primary, https://unity.com/legal/terms-of-service (the tool read only the first 100,000 of 120,046
characters, and the section numbers are as the tool reported them):

- 17.2(b) bars making Offerings available to third parties through "any hosting, application services
  provider, service bureau, software-as-a-service (SaaS)". This probably targets reselling Unity itself,
  but whether a rendered-output service falls under it is not clear to me.
- 17.2(r): "Monetize any runtime version of Software, including Unity Materials, without a separate grant of
  rights from Unity". The text I saw does not say how this squares with ordinary shipped games, so I cannot
  tell what it means for a sold client without a lawyer.
- 12.1 limits access to "solely for your personal and internal business purposes."

Verdict on terms: no royalty, but per-seat cost once we have revenue or funding over $200k, and
ambiguous hosting and runtime wording. I did not research headless Linux rendering, glTF or label passes.

## Blender and EEVEE

Primary, https://www.blender.org/about/license/, checked 2026-11-05:

- "Blender is released under the GNU General Public License"; most code is GPL v2 or later, and Cycles is
  Apache 2.0.
- "What you create with Blender is your sole property." Renders and .blend files are ours.
- Python add-ons that use the Blender API must be GPL-compatible if published. Bundling with proprietary
  software is possible "under certain conditions" that the page does not list. Hosted-service use is not
  addressed (the GPL itself has no network clause, which is general knowledge, not from that page).
- Trademark: separate policy, not read.

Consequence: running Blender on our servers and selling the output is fine and is what Fraktal does today.
Shipping Blender inside a client brings GPL source and notice duties; our own code that links into it
would be pulled toward GPL. EEVEE is a real-time rasteriser but Blender is not a driving or game runtime:
no physics for vehicles, no game loop, no input layer. It is a renderer, not a real-time engine for this
purpose.

Headless: secondary only (devtalk.blender.org threads and bug #77339, 2020, older Blender). EEVEE needs
OpenGL, so `blender -b` with EEVEE needs an X server or Xvfb on NVIDIA hosts, and one user measured 50 s
against an expected 2 s under Xvfb. Cycles with CUDA does not need OpenGL. I could not find current
(5.x) status; the manual page for command-line rendering was too long for the tool. Our existing Cycles
RunPod flow works; EEVEE on a pod is untested.

Import and passes: Blender imports glTF and USD (general knowledge, docs not read here) and has object
index, material index and depth passes, offline only.

## What I could not establish

- Unreal's actual EULA text: royalty base, definition of Product, any SaaS or streaming clause, seat fees.
- Whether Unity 17.2(b) and 17.2(r) bar a hosted render service or a client we distribute.
- Godot rendering on a GPU pod without a physical display (Xvfb, EGL or Vulkan headless).
- Any engine's glTF or USD import fidelity for our scenes, and label-pass quality (nothing was run).
- Current EEVEE headless behaviour in Blender 5.x.

## Recommendation

For a later one-car-over-one-tile prototype, use **Godot 4**. It is the only candidate whose terms I could
read in full and that has no royalty, no seat fee, no hosting question and no runtime restriction, and it
loads glTF at runtime. Its weakness is that labels and depth are ours to build, and headless GPU rendering
is unproven. Unreal is the better-looking, better-instrumented engine, but I could not read its terms and
the secondary evidence points at royalty exposure for a sold interactive service. Unity adds per-seat
cost and ambiguous wording. Blender stays the offline renderer.

Before any of that, **stay offline through phases 1 to 3** as ADR-0011 already decides. Nothing here
changes that: the offline path needs none of these licences. Do the Godot prototype only when phase 4 is
next, and run it as a test of the two unknowns (headless rendering on a RunPod GPU and a flat-colour
instance-ID pass).

## Open questions

1. Can you fetch Epic's EULA in a browser and paste the royalty, Product and streaming sections? That
   would settle Unreal.
2. Do we want to ship an interactive client at all, or only a web view fed by cloud rendering? The second
   avoids most distribution terms but needs pixel-streaming-style infrastructure.
3. Is a quality gap between Godot and Unreal acceptable for the first demo, given ADR-0009's stance on
   coverage over polish?
4. Worth a short legal read of Unity's terms, or do we drop Unity unless Godot fails?
5. Does the studio need interactive driving at all in v1 (ADR-0011 open question 3)? If not, this spike
   can wait.

## Sources checked (2026-11-05)

- https://godotengine.org/license/ (primary)
- https://docs.godotengine.org/en/stable/tutorials/editor/command_line_tutorial.html (primary)
- https://docs.godotengine.org/en/stable/tutorials/io/runtime_file_loading_and_saving.html (primary)
- https://unity.com/products/pricing-updates (primary)
- https://unity.com/legal/terms-of-service (primary, first 100k characters)
- https://www.blender.org/about/license/ (primary)
- https://conductatlas.com/platform/unreal-engine/unreal-engine-eula/ and linked provision pages (secondary)
- Unreal Engine forum and Answers threads on non-interactive use and Pixel Streaming in Docker (secondary)
- https://devtalk.blender.org/ EEVEE background-rendering threads and
  https://projects.blender.org/blender/blender/issues/77339 (secondary, old)
- Blocked, not read: all of unrealengine.com (HTTP 403)
