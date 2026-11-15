# Fraktal viewer (Godot 4)

A real-time viewer for one exported tile (ADR-0011 phase 4, plan in
[docs/research/godot-prototype-plan.md](../docs/research/godot-prototype-plan.md)). Slice 1: drive a
car over the tile. The world is read from the files `fraktal scene` writes; the viewer generates and
simulates nothing.

## 1. Export a tile (no Blender)

```bash
uv run fraktal generate --seed 0 --fallback procedural --biome dense_urban --out data/worlds/dense
uv run fraktal scene data/worlds/dense --out data/scene_dense --size-m 500
```

`scene` writes `scene.json`, `mesh_*.npy`, `ground*.npy`, `ground_height.npy` and `start.json` (a spot
on the longest road). About 17 MB for a 500 m tile at 0.5 m.

## 2. Install Godot 4.7.2 (standard build, not .NET)

MIT licence, one executable, no installer. Verify the zip against `SHA512-SUMS.txt` on the release page.

- **WSL:** download `Godot_v4.7.2-stable_linux.x86_64.zip` from
  <https://github.com/godotengine/godot/releases/tag/4.7.2-stable>, unzip it somewhere on disk.
- **Windows:** download `Godot_v4.7.2-stable_win64.exe.zip`, unzip it, and copy the tile folder to a
  Windows path (reading `\\wsl.localhost\...` works but is slow to load). The project is
  `godot/project.godot`; the `godot/` folder is all it needs, plus `assets/` next to it (pass
  `--assets DIR` if you move it).

## 1b. Cars, people and drones that move (optional)

```bash
uv run fraktal scene data/worlds/dense --out data/scene_dense --size-m 500 --agents
```

`--agents` also writes `agents.json`, `people.json` and `drones.json` (a 120 s loop by default; `--duration-s`,
`--cars-per-km`, `--people-per-km`, `--drones`, `--seed`). The viewer replays them in a loop: the same rounded cars and
figures as the Blender scene (`godot/data/agent_shapes.json`, written by `scripts/godot/export_shapes.py` from
`blender_scene.py`), interpolated between samples. The simulation covers the whole world, so cars that are off the tile
are hidden until they drive back in. They do not react to you and you do not collide with them. `--no-agents` skips them.

## 3. Run

The first time, once per checkout, build Godot's script cache (without it `class_name` scripts are not
found):

```bash
godot --headless --path godot --import
```

Then:

```bash
# WSL: use OpenGL. Vulkan under WSLg falls back to a CPU rasteriser (llvmpipe), far too slow.
godot --path godot --rendering-driver opengl3 -- --scene data/scene_dense

# Windows (PowerShell), same project, native Vulkan or OpenGL:
.\Godot_v4.7.2-stable_win64.exe --path godot -- --scene C:\fraktal\scene_dense
```

What the viewer loads: the relief ground and its colours, buildings with their materials, trees and poles
(stand-ins), road markings and wear, props (jeepneys, stalls, parked cars from `propgen`), compound walls with
their plasters, overhead cables, and the box cars parked along the road. Stored colours are display values
(Blender renders with a raw view), so they are used as sRGB colours here, like the ground. It does not
simulate or generate anything, and has no labels or depth.

**This machine (WSL2 on Windows, checked 2026-11-15):** WSLg works (`DISPLAY=:0`, 1920x1200), and OpenGL is hardware
accelerated through Mesa D3D12 on the Intel Iris Xe, so the NVIDIA card stays free for other jobs. Godot 4.7.2 is
unpacked to `~/.local/opt/godot-4.7.2/` (zip checked against `SHA512-SUMS.txt`). A 500 m Manila-Quezon tile, with
about 5.5 million primitives, runs at 35 to 65 fps. For a screenshot without driving:
`godot --path godot --rendering-driver opengl3 -- --scene DIR --bench 6 --shot out.png`.

Three modes; keys 1, 2 and 3 switch, and the new one starts where you are.
- **1, car:** W/S throttle and brake, A/D steer, Space handbrake. A chase camera.
- **2, person:** W/S/A/D walk (Shift runs), eye height 1.6 m, first person. Click to capture the mouse for looking, Esc to
  release it; the arrow keys also turn and tilt. You slide along buildings, compound walls, trunks and poles.
- **3, drone:** W/S/A/D fly along where you look, E up, Q down, Shift fast, up to 400 m. It stops at the ground and at a roof,
  can fly over a building but not into its side.
- R resets to the start. Options after `--`:
`--mode car|person|drone` (start in that mode), `--autodrive` (a scripted run of the current mode), `--bench SECONDS [--shot out.png]` (frame statistics),
`--selftest` (headless check, exit code 0 or 1), `--assets DIR`.

## Tests

`pytest -m integration tests/integration/test_godot_viewer.py` runs the headless self-test twice on a
generated tile and requires the same final state. It finds Godot through `FRAKTAL_GODOT`, `godot` on
PATH or `~/.local/opt/godot-*/`, and is skipped without one. CI does not install Godot yet.

## Limits

Not the Cycles look, no labels or depth yet, stand-in trees, no agents, no people or drone. See the plan
doc for what the engine will not reproduce.
