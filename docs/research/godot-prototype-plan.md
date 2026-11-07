# Plan: a Godot 4 prototype you can drive, walk and fly (ADR-0011 phase 4)

Written 2026-11-07, before engine code. Epic FRK-E8; slice 1 is FRK-91. The engine choice and its
licence read are in [realtime-engine-spike.md](realtime-engine-spike.md); this note is the build plan.

## What the machine has

| Check | Result |
|---|---|
| Godot on WSL | not installed; 4.7.2-stable Linux x86_64 (78 MB zip) downloaded from the official GitHub release, SHA-512 checked, unpacked to `~/.local/opt/godot-4.7.2` (outside the repo) |
| Godot on Windows | not installed (`where godot`, Program Files and winget all empty) |
| GPU in WSL | `/dev/dxg` present, so WSL has a GPU path; `nvidia-smi` absent (the GPU is an Intel UHD) |
| Renderer, Vulkan under WSLg | Godot picks `llvmpipe` (CPU rasteriser), too slow to drive |
| Renderer, OpenGL (`--rendering-driver opengl3`) | Mesa D3D12 on the Intel UHD, hardware accelerated |
| Display | WSLg (`DISPLAY=:0`, Wayland) works for windows |

So on WSL run the Compatibility renderer with `--rendering-driver opengl3`. On Windows nothing is
needed beyond the Godot zip (one `.exe`, no installer; the standard, non-.NET build) and a copy of the
exported tile folder: Godot uses Vulkan natively there and the Forward+ renderer is fine.

## Smallest vertical slice (FRK-91)

Load one tile and drive a car in it:

1. `fraktal scene WORLD --out DIR` runs `export_scene` and nothing else (no Blender, 3 s for a 500 m
   tile at 0.5 m). It is a thin CLI over the existing exporter.
2. A Godot project in `godot/` reads `DIR/scene.json` and the `.npy` files:
   - ground: `ground_height.npy` (relief, 1 m grid) as a grid mesh, `ground.npy` as its colour texture;
   - buildings: `mesh_*.npy` (polygons, fan-triangulated) with a surface per material, each material
     read from `assets/materials/*.json` (a colour, or a texture mapped in world metres with the tint);
   - trees and poles from `scene.json`;
   - a car driven by a port of the browser demo's model, with building collision.
3. A scripted drive (`--autodrive`) and a headless self-test make it checkable without a person.

Follow-on slices, in order, each its own branch and PR:

| Card | Slice |
|---|---|
| FRK-92 | Walk a person (terrain following, collision) |
| FRK-93 | Fly a drone (altitude, roofs by height, camera pose per frame) |
| FRK-95 | `roads.json` in the export; lane markings in the engine |
| FRK-94 | Replay `agents.json`, `people.json`, `drones.json` |
| FRK-96 | Capture RGB, depth and `poses.json` at fixed steps |
| FRK-97 | Class and instance label pass |
| FRK-98 | The same capture on a RunPod GPU, headless |

## Data hand-off

The engine only loads files; it never generates or simulates the world (ADR-0003, ADR-0004's
"stages talk through files").

| File | Made by | Read by Godot | New exporter? |
|---|---|---|---|
| `scene.json`, `mesh_*.npy`, `ground.npy`, `ground_height.npy` | `export_scene` | yes, directly | no, only the `fraktal scene` CLI |
| `assets/materials/*.json` and textures | repo | yes | no |
| `agents.json`, `people.json`, `drones.json` | `fraktal traffic` / the render3d code | later (FRK-94) | a CLI to write all three beside the tile |
| roads (centrelines, width, class, drive side) | not in the export; roads are painted in `ground.npy` | later (FRK-95) | yes, `roads.json` |

Why `.npy` read directly and not glTF: the arrays are already the exact data Blender builds from, and an
`.npy` header is a few lines of GDScript, so there is no second format to keep in step and no float
rounding on the way. glTF stays the choice if the studio ever needs to hand scenes to Unreal or Unity.

Coordinate map, tile frame (x east, y north, z up) to Godot (x east, y up, z south): `(x, y, z) ->
(x, z, -y)`. It is a rotation (determinant 1), so polygon winding carries over unchanged.

The car does not use Godot physics. It is the demo's kinematic bicycle model at a fixed 60 Hz step with
polygon tests against the building footprints, and ground height from the same bilinear rule as
`blender_export.terrain_height`. That keeps a scripted drive reproducible to the bit on one machine
(ADR-0003) and is the same behaviour the browser demo already shows.

## What the engine will not reproduce

Do not claim parity with the Blender renders.

- **Look.** Cycles is a path tracer (soft shadows, bounce light, 16-sample noise, the sensor chain). Godot
  gives direct light, a shadow map and ambient sky light. Colours will be near, not equal.
- **Textures.** Blender reads texture images as non-colour data; Godot treats a colour texture as sRGB.
  The same JPEG will be a little different in tone.
- **Trees.** Blender builds lumpy, per-tree-seeded crowns; slice 1 uses instanced stand-ins (a trunk and
  a rounded crown), so tree shapes differ.
- **Parked cars** (`vehicles` in `scene.json`) are left out of slice 1.
- **Sensor model, labels, depth:** none in slice 1. Depth and labels are FRK-96 and FRK-97 and are ours
  to build, as the spike said.
- **Triangulation.** Faces of 4 and 8 vertices are fan-triangulated, which is right for convex faces.
  Slice 1 checks this on the tile used and records any face that fails.

## Decisions I made, and one that is yours

Made, and easy to reverse: a new epic (FRK-E7 is closed); `.npy` read directly; a kinematic car instead of
Godot's vehicle physics; no new asset (the materials come from the manifest already). The questions for
Jian are at the end of the slice-1 report, not here, so they stay current.

## Slice 1 measurements (FRK-91, 2026-11-07)

Machine: the laptop's Intel UHD graphics, 1280x720 window, Godot 4.7.2 Compatibility renderer, a 500 m tile
at 0.5 m exported by `fraktal scene` (3 s). The car drives a scripted 20 s (`--bench 20`); the first 10 frames
are dropped. "Khartoum" is `data/proc_review/proc_v8_objects` world_00 (58 buildings, 10.7 m of relief);
"dense" is a procedural dense_urban world, seed 3 (236 buildings, 107k building faces, flat).

| Run | Frame mean | p95 | Worst | Notes |
|---|---|---|---|---|
| WSL, OpenGL on D3D12, khartoum | 21.6 ms (46 fps) | 29.8 ms | 36.8 ms | not vsynced (the driver refuses to set it), so GPU-bound |
| WSL, OpenGL on D3D12, dense | 24.2 ms (41 fps) | 32.6 ms | 38.4 ms | |
| Windows, OpenGL on Intel, khartoum | 16.7 ms | 16.7 ms | 26.1 ms | held at the 60 fps vsync cap |
| Windows, OpenGL on Intel, dense | 16.7 ms | 16.7 ms | 36.8 ms | held at the 60 fps vsync cap |
| WSL, Vulkan | not timed | | | Godot chose llvmpipe, a CPU rasteriser |

Per frame the scene draws about 113 calls and 2.9 M primitives (khartoum) or 3.8 M (dense), shadow pass
included. 500k of those are the ground mesh at 1 m cells, so a coarser or LOD ground is the first thing to
try if a weaker GPU is a target. I did not test one, nor a discrete GPU.

Load, in milliseconds, headless on Linux: terrain 15 to 32, ground mesh 200 to 310, buildings 160 (khartoum)
and 890 (dense), trees and poles under 10. These are the GDScript loops only; texture upload is not counted.
Whole-process start to first frame was not timed.

Determinism: the headless self-test (60 s of scripted driving at a fixed 60 Hz step, then a head-on run at a
building) gave the same final state to four decimals on every run, and the same state on Linux and on Windows
(`215.9986 373.1683 2.1589 -0.0111 2`, x, y, heading, speed, wall hits). Same machine and CPU, so this is not
proof across CPUs. The test is in `tests/integration/test_godot_viewer.py` (skipped without Godot).

What did not work or surprised me:

- **Fortran-order arrays.** `ground_height.npy` is stored column-major (NumPy writes that for arrays that came
  out of transposed indexing). Blender never notices. The first load failed; the `.npy` reader now transposes.
  Any other exporter output could change order the same way, so the reader handles it rather than the exporter
  forcing C order.
- **Inside-out buildings.** 2 of the 58 Khartoum buildings have faces wound inwards (negative signed volume).
  Cycles ignores winding. Godot culls back faces, so the loader flips a building when its volume is negative.
  The dense tile had none.
- **A false pass.** The first self-test printed `ok: true` after Godot script errors, because Godot carries on
  after a failed assert. The self-test now checks that the tile loaded, and the pytest wrapper fails on any
  `SCRIPT ERROR` in the output.
- **A fresh checkout needs `godot --headless --path godot --import` once**, or `class_name` scripts are not found.
- **Vulkan under WSL is software.** Use `--rendering-driver opengl3` there.
- **My first scripted drive was too ambitious**: on the dense tile it hit a wall after 11 m, which the headless
  test caught. The script now drives straight along the road for 5 s before it weaves.
- Not checked: how the Godot ground and facades compare with a Blender render of the same tile, since there is
  no Blender here. Textures and tints are loaded from the same manifest files, so they should be near, not equal.
