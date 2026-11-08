#!/usr/bin/env bash
# CPU pod job (FRK-130): render the code-written props (src/fraktal/propgen.py) in Blender Cycles,
# ground and aerial views plus a GLB each, and time it. Needs COMMIT (a pushed sha).
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper (see propgen_run.sh). No secret is used.
echo "RP_BOOT $(date +%s) host=$(hostname) nproc=$(nproc)"
set -uo pipefail
T0=$(date +%s)
: "${COMMIT:?set COMMIT to the sha to run}"
VARIANTS="${VARIANTS:-10}"; PX="${PX:-320}"; SAMPLES="${SAMPLES:-16}"
RUN=/workspace/run
mkdir -p "$RUN" && cd "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
echo install > "$RUN/stage"
stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1 t=$(( $(date +%s) - T0 ))s"; }
(
  while true; do
    echo "hb stage=$(cat "$RUN/stage") load=$(cut -d' ' -f1 /proc/loadavg) disk=$(df -h /workspace | awk 'NR==2{print $3"/"$2}')"
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
trap 'publish; kill $HB 2>/dev/null; stage done' EXIT
mkdir -p review

stage install
git clone -q https://github.com/leejianrong/fraktal.git src && git -C src checkout -q "$COMMIT"
timeout 300 apt-get update -qq > /dev/null
timeout 600 apt-get install -y -qq libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libsm6 libxkbcommon0 libegl1 > /dev/null
pip install -q --root-user-action=ignore ./src pillow
python -m venv /workspace/bpyenv
/workspace/bpyenv/bin/pip install -q bpy numpy
/workspace/bpyenv/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"

stage generate
python src/scripts/props/propgen_dump.py review/props.json --variants "$VARIANTS" || echo "ERROR propgen_dump exited $?"

stage render
timeout 2400 /workspace/bpyenv/bin/python src/scripts/props/bpy_build.py review/props.json review --px "$PX" --samples "$SAMPLES" \
  || echo "ERROR bpy_build exited $?"
du -sh review | sed 's/^/RP_ARTIFACT_SIZE /'
