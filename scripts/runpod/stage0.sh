#!/usr/bin/env bash
# CPU pod job (FRK-46, ADR-0010 Stage 0): time Blender Cycles on one 1 km tile at 0.5 m, nadir,
# at two sample counts, next to the 2.5D renderer. Timing only: no real chips on the pod.
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper. Needs COMMIT (a pushed sha).
# Optional SIZE_M (1000), SAMPLES ("16 64").
# Nothing here reads or prints a secret. Watch for: STAGE failed, ERROR, WARN, hb lines.
echo "RP_BOOT $(date +%s) host=$(hostname) nproc=$(nproc)"  # the driver's boot check looks for this
set -uo pipefail
T0=$(date +%s)
: "${COMMIT:?set COMMIT to the sha to run}"
SIZE_M="${SIZE_M:-1000}"
SAMPLES="${SAMPLES:-16 64}"
MODE="${MODE:-stage0}"  # stage0: timing loop; render3d: the CLI end to end plus a scene equality check; batch: render3d-batch
COUNT="${COUNT:-50}"; TILE_M="${TILE_M:-500}"; BATCH_BIOME="${BATCH_BIOME:-dense_urban}"  # MODE=batch
EXTRA="${EXTRA:-}"  # MODE=batch: more render3d-batch flags, e.g. --no-wall-textures
PUBLISH_ALL="${PUBLISH_ALL:-0}"  # MODE=batch: 1 sends every tile, 0 only two per biome and batch.json
# MODE=fly: a drone path over one world, written as video.mp4 and video_labels.mp4 (FRK-78)
# MODE=batch: BATCH_BIOME may list several biomes; SENSOR=1 also writes sensor.png per tile
RUN=/workspace/run
mkdir -p "$RUN" && cd "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
: > "$RUN/status"

stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1 t=$(( $(date +%s) - T0 ))s"; }
fail() { echo "STAGE failed ($1; was: $(cat "$RUN/stage" 2>/dev/null))"; echo failed > "$RUN/status"; }
trap 'fail "line $LINENO"' ERR
set -e

(
  while true; do
    disk=$(df /workspace | awk 'NR==2{gsub("%","",$5); print $5}')
    ram=$(free | awk 'NR==2{printf "%d", $3*100/$2}')
    quiet=$(( $(date +%s) - $(stat -c %Y "$RUN/job.log") ))
    load=$(cut -d' ' -f1 /proc/loadavg)
    echo "hb stage=$(cat "$RUN/stage" 2>/dev/null || echo boot) load=$load disk=${disk}% ram=${ram}% quiet=${quiet}s"
    [ "$quiet" -gt 900 ] && echo "WARN no output for ${quiet}s"
    [ "$disk" -gt 90 ] && echo "WARN disk ${disk}% full"
    [ "$ram" -gt 90 ] && echo "WARN ram ${ram}% used"
    sleep 60
  done
) &
HB=$!

publish() {
  stage publish
  cd "$RUN"
  tar czf review.tgz review job.log 2>/dev/null || true
  BASE="fraktal-$RANDOM-$RANDOM"
  timeout 900 runpodctl send --code "$BASE" review.tgz > send.log 2>&1 &
  SEND=$!
  sleep 8
  echo "RP_ARTIFACT_CODE=$(grep -oE "${BASE}[-0-9]+" send.log | head -1)"
  wait "$SEND" || true
}
finish() {
  code=$?
  trap - ERR EXIT
  set +e
  [ "$code" -ne 0 ] && [ "$(cat "$RUN/status")" != failed ] && fail "exit $code"
  publish
  kill "$HB" 2>/dev/null
  [ "$(cat "$RUN/status")" = failed ] && { echo "STAGE finished with FAILURE"; exit 1; }
  stage done
}
trap finish EXIT

stage install
git clone -q https://github.com/leejianrong/fraktal.git src
git -C src checkout -q "$COMMIT"
echo "commit $COMMIT nproc $(nproc)"
# bpy is a headless wheel but still links X11 and GL libraries the image lacks
timeout 300 apt-get update -qq > /dev/null
timeout 600 apt-get install -y -qq libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libsm6 libxkbcommon0 libegl1 > /dev/null
if [ "$MODE" = fly ]; then  # video needs ffmpeg: the package, or the pip wheel that bundles a binary
  timeout 240 apt-get install -y -qq --no-install-recommends ffmpeg > /dev/null 2>&1 || {
    echo "WARN apt ffmpeg failed, using imageio-ffmpeg"
    pip install -q --root-user-action=ignore imageio-ffmpeg
    ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" /usr/local/bin/ffmpeg
  }
fi
pip install -q --root-user-action=ignore ./src matplotlib
# bpy pins its own numpy, which breaks pyarrow in the main environment: give it a venv
python -m venv /workspace/bpyenv
/workspace/bpyenv/bin/pip install -q bpy numpy
/workspace/bpyenv/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"
mkdir -p review
if [ "$MODE" = batch ]; then
  export FRAKTAL_BPY_PYTHON=/workspace/bpyenv/bin/python
  export FRAKTAL_ASSETS="$RUN/src/assets"
  SENSOR_FLAG=""; [ "${SENSOR:-0}" = 1 ] && SENSOR_FLAG="--sensor"
  n=0
  for biome in $BATCH_BIOME; do  # one or more biomes, space separated, a world each
    n=$((n + 1))
    stage "generate_$biome"
    BATCH_BIOME="$biome" python - <<'PY'
import os
from pathlib import Path
from fraktal import biomes
from fraktal.procedural import generate_world
generate_world(Path("review/world"), (32.55, 15.64), 3, biomes.resolve(os.environ["BATCH_BIOME"], {"size_m": 3000.0}))
PY
    stage "batch_$biome"
    T1=$(date +%s)
    fraktal render3d-batch review/world --out "review/batch_$biome" --count "$COUNT" --size-m "$TILE_M" --gsd-m ${GSD:-0.5} --prop-density ${PROPS:-60} --sun-elevation-range ${SUN_ELEV:-35 70} \
      --samples "${SAMPLES%% *}" --threads "$(nproc)" --seed $((10 + n)) $SENSOR_FLAG ${EXTRA:-} \
      2> >(tee "review/batch_$biome.err" | grep --line-buffered render3d_batch_tile) > "review/batch_$biome.out" \
      || echo "ERROR render3d-batch $biome exited non-zero"
    echo "RP_BATCH_SECONDS $biome $(( $(date +%s) - T1 ))"
    python - "$biome" <<'PY'
import json, sys
d = json.load(open(f"review/batch_{sys.argv[1]}/batch.json"))
print("RP_BATCH_SUMMARY", sys.argv[1], json.dumps({k: d[k] for k in ("counts", "wall_seconds", "mean_seconds_per_rendered_tile", "samples")}))
PY
    rm -rf review/world
    if [ "$PUBLISH_ALL" != 1 ]; then
      ls -d review/batch_$biome/tile_* | tail -n +3 | xargs rm -rf
    fi
  done
  du -sh review | sed 's/^/RP_ARTIFACT_SIZE /'
  exit 0
fi
if [ "$MODE" = fly ]; then
  export FRAKTAL_BPY_PYTHON=/workspace/bpyenv/bin/python
  export FRAKTAL_ASSETS="$RUN/src/assets"
  python - <<'PY'
import json
import os
from pathlib import Path
from fraktal import agents, biomes, procedural
from fraktal.procedural import generate_world
generate_world(Path("review/world"), (32.55, 15.64), 3, biomes.resolve(os.environ["BATCH_BIOME"].split()[0], {"size_m": 3000.0}))
center, route = (32.55, 15.64), [[200, 200], [400, 260], [560, 420], [480, 560]]
if os.environ.get("TRAFFIC") == "1":  # fly along a real road in the road-dense part of the world
    from shapely.geometry import box
    from shapely.ops import substring
    roads = procedural.read_roads(Path("review/world")).to_crs(4326)
    weight = roads.length
    center = (float((roads.geometry.centroid.x * weight).sum() / weight.sum()),
              float((roads.geometry.centroid.y * weight).sum() / weight.sum()))
    local, bounds, side = agents.tile_roads(Path("review/world"), center, 800.0)
    inner = box(160, 160, 640, 640)
    pieces = [p for g in local.geometry for p in getattr(g.intersection(inner), "geoms", [g.intersection(inner)]) if p.geom_type == "LineString"]
    best = max(pieces, key=lambda p: p.length)
    best = substring(best, 0, min(best.length, 420.0)).simplify(10.0)
    route = [[x, y] for x, y in best.coords]
    print("RP_ROUTE", json.dumps({"center": center, "drive_side": side, "length_m": round(best.length), "points": len(route)}))
json.dump({"waypoints": route}, open("review/path.json", "w"))
open("review/center.txt", "w").write("%f %f" % center)
PY
  stage fly
  EXTRA=""
  [ "${TRAFFIC:-0}" = 1 ] && EXTRA="--traffic --cars-per-km 25 --people-per-km ${PEOPLE_PER_KM:-0} --drones ${DRONES:-0} --drone-scale ${DRONE_SCALE:-1} --altitude-m ${ALT:-90} --off-nadir 35 --fps 8 --speed ${SPEED:-20}"
  [ "${TRAFFIC:-0}" = 1 ] || EXTRA="--altitude-m 150 --off-nadir 40 --fps 12 --depth"
  [ "${FLOW:-0}" = 1 ] && EXTRA="$EXTRA --flow"
  [ -n "${FOLLOW:-}" ] && EXTRA="$EXTRA --follow $FOLLOW --fps 10 --duration-s ${DURATION:-24}"  # ride on a car or person
  fraktal render3d review/world --out review/fly --center $(cat review/center.txt) --size-m 800 --gsd-m ${GSD:-0.5} \
    --path review/path.json --fov 60 --frame-px 512 $EXTRA \
    --samples "${SAMPLES%% *}" --seed 7 --threads "$(nproc)" --video > review/fly.out 2> review/fly.err \
    || { echo "ERROR fly failed"; tail -n 20 review/fly.err; }
  echo "RP_FLY $(python -c "import json;d=json.load(open('review/fly/render.json'));print(len(d['frames']), d['files'], d['seconds'], d.get('traffic'))" 2>&1)"
  if [ "${TRAFFIC:-0}" = 1 ]; then
  python - <<'PY'
import json
import numpy as np
import rasterio
from fraktal import labels3d
d = "review/fly"
table = json.load(open(f"{d}/tracks.json"))
first = table["first_instance"]
seen = [t for t in table["tracks"] if t["detections"]]
per_frame = np.zeros(table["frames"], int)
bad = 0
for t in seen:
    for det in t["detections"]:
        per_frame[det["frame"]] += 1
for t in seen[:25] + [p for p in seen if p["kind"] != "car"][:25]:  # recount pixels
    klass = labels3d.CLASSES[{"person": "person", "drone": "drone"}.get(t["kind"], "vehicle")]
    for det in t["detections"][::7]:
        with rasterio.open(f"{d}/labels/frame_{det['frame']:04d}.png") as src:
            sem, inst = labels3d.decode(src.read()[:3])
        x0, y0, x1, y1 = det["bbox"]
        mine = (sem == klass) & (inst == t["instance"])
        ys, xs = np.nonzero(mine)
        if int(mine.sum()) != det["pixels"] or (xs.min(), ys.min(), xs.max(), ys.max()) != (x0, y0, x1, y1):
            bad += 1
long_lived = max(seen, key=lambda t: t["detections"][-1]["frame"] - t["detections"][0]["frame"]) if seen else None
kinds = {k: (sum(1 for t in table["tracks"] if t["kind"] == k), sum(1 for t in seen if t["kind"] == k)) for k in ("car", "person", "drone")}
print("RP_TRACK_KINDS", kinds)
print("RP_TRACK_CHECK", "tracks", len(table["tracks"]), "seen", len(seen), "mean_visible_per_frame", round(float(per_frame.mean()), 1), "max_visible", int(per_frame.max()), "mismatches", bad,
      "longest_track", long_lived and (long_lived["id"], long_lived["instance"], long_lived["detections"][0]["frame"], long_lived["detections"][-1]["frame"], len(long_lived["detections"])))
PY
  fi
  if [ "${TRAFFIC:-0}" != 1 ]; then
  python - <<'PY'
import json
import numpy as np
import rasterio
from fraktal import depth
d = "review/fly"
meta = json.load(open(f"{d}/render.json"))
k = meta["intrinsics"]
poses = json.load(open(f"{d}/poses.json"))["frames"]
for i in range(0, len(poses), 30):
    with rasterio.open(f"{d}/depth/frame_{i:04d}.png") as src:
        raw = src.read(1)
    m = np.array(poses[i]["camera_to_world"])
    p = meta["frames"][i]
    expect = depth.project((p["target_x"], p["target_y"], 0.0), m, k)[2]
    centre = depth.to_metres(raw[252:261, 252:261])
    print("RP_DEPTH_CHECK", i, "centre_median_m", round(float(np.median(centre)), 2), "expected_ground_m", round(expect, 2))
PY
  fi
  if [ "${FLOW:-0}" = 1 ]; then
  python - <<'PY'
import numpy as np
import rasterio
d = "review/fly"
def img(i):
    with rasterio.open(f"{d}/frames/frame_{i:04d}.png") as src:
        return src.read()[:3].astype("float32").mean(axis=0)
errs = []
for i in range(0, 300, 12):
    try:
        a, b = img(i), img(i + 1)
    except Exception:
        break
    f = np.load(f"{d}/flow/frame_{i:04d}.npy").astype("float32")
    v, u = np.mgrid[0:a.shape[0], 0:a.shape[1]] + 0.5
    ok = np.isfinite(f).all(axis=-1)
    f = np.nan_to_num(f)
    r = np.clip(np.rint(v - 0.5 + f[..., 1]), 0, a.shape[0] - 1).astype(int)
    c = np.clip(np.rint(u - 0.5 + f[..., 0]), 0, a.shape[1] - 1).astype(int)
    warped = b[r, c]
    errs.append((i, round(float(np.abs(warped - a)[ok].mean()), 2), round(float(np.abs(b - a)[ok].mean()), 2), round(float(np.hypot(f[..., 0], f[..., 1])[ok].mean()), 2)))
print("RP_FLOW_CHECK (frame, warp_err, raw_err, mean_flow_px)", errs)
PY
  fi
  rm -rf review/world review/fly/scene/ground* review/fly/scene/mesh_*
  ls -la review/fly | head; du -sh review | sed 's/^/RP_ARTIFACT_SIZE /'
  exit 0
fi
if [ "$MODE" = render3d ]; then
  stage generate
  export FRAKTAL_BPY_PYTHON=/workspace/bpyenv/bin/python
  export FRAKTAL_ASSETS="$RUN/src/assets"  # the installed package has no assets/ beside it
  fraktal generate --help > /dev/null  # fail early if the CLI did not install
  python - <<'PY'
from pathlib import Path
from fraktal import biomes
from fraktal.procedural import generate_world
generate_world(Path("review/world"), (32.55, 15.64), 3, biomes.resolve("dense_urban"))
PY
  for run in a b; do
    stage "render3d_$run"
    fraktal render3d review/world --out "review/r$run" --center 32.55 15.64 --size-m 300 --gsd-m ${GSD:-0.5} \
      --samples 16 --seed 7 --threads "$(nproc)" > "review/r$run.out" 2> "review/r$run.err" \
      || { echo "ERROR render3d $run failed"; tail -n 15 "review/r$run.err"; }
    echo "RP_RENDER3D_$run $(tr -d '\n ' < "review/r$run/render.json" 2>/dev/null)"
  done
  if [ "$(python -c "import json;print(json.load(open('review/ra/render.json'))['scene_sha256'])")" = "$(python -c "import json;print(json.load(open('review/rb/render.json'))['scene_sha256'])")" ]; then
    echo "RP_SCENE_EQUAL yes"
  else
    echo "RP_SCENE_EQUAL NO"
  fi
  rm -rf review/world
  exit 0
fi
for s in $SAMPLES; do
  stage "render_$s"
  timeout 2400 python src/scripts/blender/stage0_compare.py "review/s$s" --size-m "$SIZE_M" --gsd-m 0.5 --samples "$s" \
    --threads "$(nproc)" --bpy-python /workspace/bpyenv/bin/python > "review/s$s.out" 2> "review/s$s.err" || { echo "ERROR samples=$s failed"; tail -n 15 "review/s$s.err"; }
  echo "RP_STAGE0_BEGIN $s"; tr -d '\n ' < "review/s$s/stage0.json" 2>/dev/null; echo; echo "RP_STAGE0_END"
done
rm -rf review/s*/world review/s*/r25
