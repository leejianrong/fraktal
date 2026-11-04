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
MODE="${MODE:-stage0}"  # stage0: timing loop; render3d: the CLI end to end plus a scene equality check
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
pip install -q --root-user-action=ignore ./src matplotlib
# bpy pins its own numpy, which breaks pyarrow in the main environment: give it a venv
python -m venv /workspace/bpyenv
/workspace/bpyenv/bin/pip install -q bpy numpy
/workspace/bpyenv/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"
mkdir -p review
if [ "$MODE" = render3d ]; then
  stage generate
  export FRAKTAL_BPY_PYTHON=/workspace/bpyenv/bin/python
  fraktal generate --help > /dev/null  # fail early if the CLI did not install
  python - <<'PY'
from pathlib import Path
from fraktal import biomes
from fraktal.procedural import generate_world
generate_world(Path("review/world"), (32.55, 15.64), 3, biomes.resolve("dense_urban"))
PY
  for run in a b; do
    stage "render3d_$run"
    fraktal render3d review/world --out "review/r$run" --center 32.55 15.64 --size-m 300 --gsd-m 0.5 \
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
