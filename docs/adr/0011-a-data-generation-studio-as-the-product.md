# ADR-0011: A data generation studio is the product

- Status: Accepted (2026-11-05, Jian); amends ADR-0007
- Date: 2026-11-05
- Deciders: Jian

## Context

On 2026-11-05 Jian said the thing to sell is the tool, not only the data: a studio where a user builds
a world, drives or flies through it, simulates agents in it, and captures pictures, video and sensor
data, with full control. ADR-0007 said the opposite for the MVP (sell datasets and custom projects,
keep the engine private). The two companies in the notes show why the answer is not obvious:

- **Another Earth** (anotherearth.ai): synthetic Earth-observation data (satellite, drone, aerial)
  and a model fine-tuned to a customer's sensor, region and resolution. Its site lists sector
  solutions and "Custom Projects", the calls to action are "explore" and "contact us", and it states
  no pricing, no named customers and no self-serve tool. In short, sales-led data and custom work.
  Claims on the page, not checked by us: up to 95% less real data needed, about 20% better models.
- **Bifrost AI**: sells *Stardust*, a platform that lets an AI developer with no simulation team build
  simulated 3D environments and generate training data, mainly for robotics, vehicles, aerospace and
  defence. Press and aggregator profiles (secondary sources) describe annual platform subscriptions
  plus usage-based pricing for data volume plus services; it raised an $8M Series A in October 2024.
  No public price list.

So "copy their business model" means two different things. Another Earth is ADR-0007 (data and
projects). Bifrost is the studio. The studio Jian describes is Bifrost's shape applied to a world that
also does satellite imagery, which Bifrost does not.

## Decision

Build toward a **data generation studio**: one world model (ADR-0003) seen through many sensors and
actors, with the user in control of the world, the sensors, the actors and the outputs. Revenue is,
in order of when it can start: (1) custom data projects and sample datasets, which fund and prove the
engine and are Another Earth's model; (2) usage-based generation (priced per tile, frame or GPU-hour)
and subscriptions for the studio, which are Bifrost's model. ADR-0007's "engine stays private" is kept
until a studio exists; nothing in this ADR open-sources anything.

### What the studio does

| Capability | What the user can do | Where it stands today |
|---|---|---|
| **Worlds** | Pick a biome and region, set a seed and parameters, edit what is placed | Seeded worlds, six biomes, real-conditioned (ADR-0002); no editing |
| **Satellite imagery** | Choose sensor, GSD, sun, haze, cloud, noise; get tiles and masks | 2.5D renderer with a sensor model (ADR-0004); the 3D path matches it with `sensor_chain` |
| **Drone and aerial imagery** | Fly a path at an altitude and angle; get stills and video | Still, orbit and flight camera trajectories (`fraktal.cameras`), 3D Blender renders, labels for any camera; frames only, no video file, no flight dynamics |
| **Ground views and vehicles** | Drive or place a vehicle and capture from it | Parked boxes along roads; no vehicle motion, no ground-level camera rig |
| **Simulated agents** | Traffic, pedestrians, drones, other vehicles moving through the world on their own | Nothing; the road graph and building footprints exist |
| **Data capture** | RGB frames and video plus labels, depth, camera pose, and later flow, IMU, GPS, lidar | RGB and class and instance labels; no depth, pose log or video |
| **Control surface** | Config files and a CLI now; a web studio and an API later | CLI with JSON output (ADR-0005); no UI or API |

### The hard architectural fork: offline or interactive

Driving a vehicle through the world needs about 30 frames a second. Blender Cycles renders a tile in
seconds, so it cannot do that. Two different products follow:

1. **Scripted and simulated capture (offline).** The user (or a simulator) defines paths and agent
   behaviour; the studio renders every frame afterwards, in the cloud, with full quality and exact labels.
   "Driving" means a vehicle following a controlled or authored trajectory, recorded and rendered later.
   It reuses everything built so far and is the cheapest way to a demo.
2. **Interactive (real-time).** A user steers a car or flies a drone live. That needs a real-time engine
   (candidates: Godot, Unreal, Unity, or Blender's EEVEE) reading our scene files. Engine licences
   differ a lot for a product that is itself sold (royalties, per-seat fees, redistribution), so that is
   a spike with the terms read from the licence text, as FRK-59 did for models.

**Decision:** do 1 first, design the scene format so 2 can read the same worlds, and run the engine
spike before committing to a real-time stack. Interactive driving is a later phase, not a promise.

## Plan

Each phase ends in something a person can see. Phases 1 and 2 need no new engine. Slice sizes are rough.

| Phase | Outcome | Slices (Pandan cards) |
|---|---|---|
| **0. Decide** | This ADR accepted; ADR-0007 amended; competitor notes kept in `docs/research` | Accept or change this ADR; write the notes |
| **1. Drone studio, offline** | `fraktal fly WORLD --path ... --video` writes an MP4, per-frame labels, a camera pose log and depth | Video encoding of frames; depth pass; pose and intrinsics log; user-authored paths (waypoints, speed, altitude); a flight dynamics model (smooth, bounded accelerations); a sensor model per frame (blur, rolling shutter later) |
| **2. Moving things** | Vehicles, people and drones move on the world's roads and paths and are in the labels | Agent model on the road graph (a mesoscopic traffic and pedestrian sim, or an existing open simulator after a licence read); animated vehicle and person assets under the owned or CC0 rule; per-frame instance tracks; ground and dashcam camera rigs |
| **3. Capture breadth** | More sensors from the same capture | Optical flow, normal and instance depth; IMU and GPS logs; a simple lidar by ray casting; thermal and SAR later |
| **4. Real-time spike** | A written verdict on an engine for interactive driving and flying | Licence read per engine; a scene exporter (glTF or USD) from our scene files; one car driven in the chosen engine over one tile |
| **5. Studio shell** | A user configures a world and a mission in a browser and gets the dataset | Web UI over a job API; a render queue on rented GPUs (RunPod); billing and usage metering; auth; dataset hosting (HF repo for samples, R2 for customers) |

Cross-cutting, all phases: stable IDs so a vehicle in frame 40 is the same object in frame 41;
determinism (a mission is a seed plus a config); exact labels from any camera; every asset with a
manifest row; progress lines for long jobs; and per ADR-0009, build for coverage and measure to steer.

## Risks and what we do not know

- **No evidence that the data helps a model.** ADR-0006 still stands: the first training benchmark has
  not run. A studio with unproven data is a toy; a custom project to one real customer is the cheapest
  test of that. Keep the benchmark milestone (FRK-45) alive.
- **Quality is not competitive yet.** Terrain is flat, ground is a colour image, trees are lumpy balls,
  vehicles are boxes. The studio's value is control, but users will compare images with real ones.
- **Rendering cost per user.** About 3 to 15 s of CPU per 500 m tile at 16 samples today; video at
  30 frames a second multiplies that. Usage-based pricing has to cover GPU time with margin.
- **Real-time is a different product.** Phase 4 may show that interactive driving needs an engine we
  do not own the licence terms for. We learn that before building a UI around it.
- **Licensing for a sold tool.** OSM share-alike, WorldCover attribution, SDXL output terms and each
  engine's EULA all change when the tool, not only a dataset, is distributed (ADR-0007 gate stays).
- **Scope.** Satellite, drone, vehicles and agents in one product is a very large surface for one
  author. The order above is chosen so each phase stands alone if the next never happens.

## What this changes

- **ADR-0007:** marked "Amended by 0011". The first revenue model
  stays (projects and datasets) but the destination is the studio, and "no UI or API" in PLAN.md
  becomes a later phase, not an exclusion.
- **PLAN.md "Later bets" 0 and 1** (3D worlds, robotics and drone views) become the main line.
- **Hosting:** public sample datasets on Hugging Face are fine (they are marketing for the studio).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Stay with ADR-0007 only (data and projects) | Matches Another Earth but not the stated goal, and caps revenue at services margins |
| Build the real-time studio first | Largest risk and cost, and needs an engine decision we have not researched |
| Open-source the engine, sell hosting | Irreversible and premature; see ADR-0007 |

## Open questions for Jian

1. Is Bifrost's model (platform plus usage-based plus services) the target, with Another Earth's
   project work as the way to get there? Or do you want Another Earth's model alone?
2. Who is the first user: EO and geospatial ML teams, or robotics and drone teams? The same engine
   serves both, but the first demo and sample data differ.
3. Is interactive driving a must for v1 of the studio, or is authored and simulated capture enough?
4. Is there one real customer or design partner to build for? It changes the order more than anything here.

## After FRK-78 and FRK-83

- **Phase 1 started.** `fraktal render3d --path waypoints.json --video` flies a drone through
  user-given waypoints (tile metres) at constant speed, with the heading following a smoothed path,
  and writes `video.mp4`, a colour-class `video_labels.mp4` preview, and the exact `labels/frame_NNNN.png`.
  Pod demo (`MODE=fly`, dense_urban, 180 frames at 512 px, 16 samples): 156 s of Blender time on
  16 vCPUs, about 0.9 s a frame. The frames and labels line up; the demo path crossed open land,
  so a nicer path over the town is still to pick. Depth, pose and intrinsics are FRK-79; real
  flight dynamics replace the moving-average smoothing in FRK-80.
- **Engine spike written** ([realtime-engine-spike.md](../research/realtime-engine-spike.md)):
  Godot 4 (MIT, no royalty) is the candidate for a later prototype; Unreal terms could not be
  fetched from Epic and are secondary only. Nothing is decided: stay offline through phases 1 to 3.

## After FRK-79

`render3d --depth` writes `depth/frame_NNNN.png` (16-bit, z depth along the optical axis, 5 cm
units, 0 = sky), and every camera sequence writes `poses.json` with the intrinsics and a 4x4
`camera_to_world` per frame (OpenCV axes, tile frame east, north, up). Blender renders depth as an
unlit shader that splits the distance over two 8-bit channels, the way labels work, so it is exact
and needs no EXR reader. Check on a pod (dense_urban, 180 frames, 16 samples): the centre-pixel depth
was 195.65 m against 195.81 m predicted from the pose at frame 30, 60, 90 and 120, and 191 m at frame 0
where a roof stands at the centre. Blender time for the 180 frames was 491 s with depth, against 156 s without it on the earlier
run, but those were different pod hosts (the install stage also took twice as long), so the depth
pass cost is not measured. A same-host comparison is still to do.

## After FRK-80

Path flights now use a point-mass drone (`fraktal.kernels.flight`) instead of smoothing the
polyline. It starts and ends at rest, speeds up and brakes within `--max-accel` (default 3 m/s²),
slows for corners (a speed profile along the path, using 80% of the limit so tracking has room),
and turns its heading no faster than `--max-yaw-rate` (default 60 deg/s). On an L of 200 m legs at
20 m/s cruise it passes the corner at about 7 m/s, cuts it by under 10 m, and stops within 2 m of the
last waypoint. It is a kinematic model, with no wind, battery or thrust; the poses are the same
kind as before, so rendering did not change and no pod run was needed.
