# Plan: moving vehicles and people (FRK-81, ADR-0011 phase 2)

Written 2026-11-05, before any code. Goal: a drone flight or a drive shows cars and people moving
along the world's roads, each with a stable track ID, so labels and tracks line up across frames.

## What exists today

- **Roads:** a `road` layer of LineStrings with `road_class` (arterial 12 m wide, local 6 m), width
  and a bridge flag (`roads.py`). Roads are painted into the ground colour; there is no junction
  graph, no lanes and no traffic rules.
- **Cars:** `blender_export._vehicles` scatters parked boxy cars on road pixels (`objects.place`),
  stored in `scene.json` as `[x, y, heading, colour]`. They never move.
- **Labels:** instance number is the index in a per-class list plus one (`labels3d`); the vehicle
  class is 4. Sixteen bits, so up to 65,534 instances a frame.
- **Frame loop:** `blender_scene.py` poses the camera and renders frame by frame, three passes
  (beauty, labels, depth). Objects are static meshes; nothing is animated.
- **Dynamics:** `kernels/flight.py` (a drone) is the model for how a kernel like this looks.

## Design

**Where the simulation lives.** A new module `fraktal.agents` builds the road graph, picks routes and
returns *tracks*: for every agent, its position, heading and speed at every frame time. It is pure
Python and numpy, deterministic through `fraktal.ids.substream`, and needs no Blender. Tracks are written
to `scene/agents.json` and are part of `scene_sha256`, so a mission is still a seed plus a config
(ADR-0011). The time grid is the camera's frame times, so every pass sees the same world at the same instant.

**Road graph.** Node the road LineStrings where they meet (union, then split at shared points), giving
edges with class, width, length and direction. Two lanes per road, the drive side a parameter
(default left, as in most of the biomes' real counterparts; open question below). A lane is the
centreline offset by a quarter of the road width.

**Cars, first version.** Each car gets a start edge and a route: follow the edge, and at each junction
pick a next edge by seeded choice with no U-turn, until the route is long enough for the whole clip.
Speed per car comes from road class (about 11 m/s on arterials, 6 m/s on local roads) with a seeded
spread, and slows for sharp turns using the same corner rule as the drone kernel. No interaction at
first: cars are spawned with spacing and cannot overtake, but they can pass through each other at
junctions. A second slice adds car-following (the Intelligent Driver Model, a standard rule: speed
follows the gap to the car ahead) and junction yielding. The model is a kernel in
`kernels/traffic.py`: arrays in, arrays out, with a golden test.

**People, later.** Pedestrians walk on the verge of local roads (a strip beside the road edge) at about
1.4 m/s with seeded stops, and cross at junctions. A simple articulated figure built from boxes and
capsules (owned, no licence question) without a walk cycle at first.

**Rendering.** One vehicle mesh with fixed topology (two boxes per car, as now), whose vertex
positions are rewritten for each frame from the track, then lifted onto the terrain with
`lift_to_terrain`. Fixed topology means the label colours painted once stay valid in all passes, and
nothing is rebuilt. Agents outside the tile are parked far underground rather than removed, so
face counts and instance numbers never change. The same update runs in the beauty, label and depth
loops. Parked cars move to the kerb when traffic is on, so moving cars do not drive through them.

**Stable IDs.** Instance number is `parked_count + track_index + 1` for moving cars, so it is the same in
every frame and never collides with a parked car. `tracks.json` maps instance to track and to
agent kind. People get their own class (6) and a block of their own instances.

**Ground-truth tracks.** After rendering, a pass over `labels/frame_NNNN.png` writes `tracks.json`: for every
frame and track, the 2D box of its visible pixels, its pixel count, and the 3D position, heading and
speed from the simulation. That is the MOT-style output a customer expects, and it uses the exact
labels, so occlusion is handled for free.

## Slices

| Card | Slice | Points | Done when |
|---|---|---|---|
| FRK-81 | Road graph and moving-car tracks (no Blender) | 3 | Cars stay on their lane, follow routes through junctions, respect speed limits, are deterministic; a plot of tracks on a road map |
| FRK-85 | Render moving cars in flights, with track IDs and `tracks.json` | 3 | A pod flight shows moving cars, labels follow them, `tracks.json` matches the labels |
| FRK-86 | Car-following and junction yielding | 2 | No overlaps in a busy clip; queues form behind a slow car |
| FRK-87 | Pedestrians on verges and crossings | 3 | People walk beside roads in a flight, labelled as a class of their own |

FRK-82 (ground and dashcam cameras) follows FRK-85: a camera that rides on an agent needs the tracks. Its
rig code can start first, since a camera glued to a moving pose is the same machinery as the drone's.

## Checks

- Unit: tracks stay within half a lane of the centreline; speeds within the class limits; no teleports
  (consecutive positions within speed times time step); same seed, same tracks; different seed,
  different tracks; a car never leaves the graph.
- Kernel golden test for the car model; mypy strict; no Python loops over pixels.
- Pod check each rendering slice: look at the frames, then verify from the files that a car's
  instance number is the same in frame 0 and frame 90 and that `tracks.json` agrees with the label image.

## Risks and unknowns

- **Procedural roads are simple.** Branching and crossings are sparse in the generated worlds, so traffic
  may look thin. Dense urban is the best test; some biomes have little traffic by nature.
- **Bridges.** Roads on bridges would put cars at water level because the ground mesh is the terrain.
  First version skips bridge segments; raising cars to a deck height is a later fix.
- **Quality.** Cars are boxes. Real traffic in these regions is mostly motorbikes and tuk-tuks, which
  matters more for realism than car-following does. A motorbike with a rider is a good follow-up asset
  (built in code, so owned).
- **Pods.** Per-frame mesh rewrites are cheap next to Cycles, but the number of agents times frames
  times three passes is worth timing on a pod before going big.

## Open questions for Jian

1. **Drive side:** left or right? Khartoum is right-hand; most of the tropical SE Asia biomes are left-hand.
   Proposal: a biome setting, left for the SE Asia ones and right for Khartoum.
2. **Which first:** more realistic traffic (car-following, FRK-86) or pedestrians (FRK-87) after the
   first moving cars?
3. **Motorbikes and tuk-tuks:** add them as a vehicle type in this epic, or keep that for a realism pass?

## Status after FRK-81 (2026-11-05)

Answers from Jian: drive side is a biome setting (and can be overridden); car following (FRK-86) comes
before pedestrians (FRK-87); motorbikes and tuk-tuks wait for a realism pass.

Built: `fraktal.agents` (junction graph, two lanes per road on the drive side, seeded routes, per-frame
`Tracks`), `kernels/traffic.py` (speed caps, braking, stepping), `RoadConfig.drive_side` (left for the
five tropical biomes, right for Khartoum; not part of the world id, so existing ids are unchanged),
and `fraktal traffic WORLD --center LON LAT --out DIR`, which writes `agents.json` and a map. On a
2 km tile of the dense_urban world (30.6 km of road, 184 cars, 40 s) the map below shows every car's
path from its start dot: cars follow the roads, turn at junctions and U-turn at dead ends and tile edges.

![Car paths on a dense_urban tile](../img/traffic-map.jpg)

Known limits, as planned: cars ignore each other, bridges are left out, the generated local roads are
mostly spurs off arterials so the network is sparse, and the tile centre of a world is not always where
the roads are (a 500 m tile at the dense_urban centre held 0.3 km of road). Rendering these tracks is FRK-85.
