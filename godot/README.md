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

Keys: W/S throttle and brake, A/D steer, Space handbrake, R reset. Options after `--`:
`--autodrive` (scripted drive), `--bench SECONDS [--shot out.png]` (frame statistics),
`--selftest` (headless check, exit code 0 or 1), `--assets DIR`.

## Tests

`pytest -m integration tests/integration/test_godot_viewer.py` runs the headless self-test twice on a
generated tile and requires the same final state. It finds Godot through `FRAKTAL_GODOT`, `godot` on
PATH or `~/.local/opt/godot-*/`, and is skipped without one. CI does not install Godot yet.

## Limits

Not the Cycles look, no labels or depth yet, stand-in trees, no agents, no people or drone. See the plan
doc for what the engine will not reproduce.
