#!/usr/bin/env bash
# CPU pod job (FRK-46, ADR-0010 Stage 0): time Blender Cycles on one 1 km tile at 0.5 m, nadir,
# at two sample counts, next to the 2.5D renderer. Timing only: no real chips on the pod.
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper. Needs COMMIT (a pushed sha).
# Optional SIZE_M (1000), SAMPLES ("16 64").
# Nothing here reads or prints a secret. Watch for: STAGE failed, ERROR, WARN, hb lines.
set -uo pipefail
: "${COMMIT:?set COMMIT to the sha to run}"
SIZE_M="${SIZE_M:-1000}"
SAMPLES="${SAMPLES:-16 64}"
RUN=/workspace/run
mkdir -p "$RUN" && cd "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
: > "$RUN/status"

stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1"; }
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
pip install -q --root-user-action=ignore ./src bpy matplotlib
python -c "import bpy; print('bpy', bpy.app.version_string)"
mkdir -p review
for s in $SAMPLES; do
  stage "render_$s"
  timeout 2400 python src/scripts/blender/stage0_compare.py "review/s$s" --size-m "$SIZE_M" --gsd-m 0.5 --samples "$s" \
    --threads "$(nproc)" > "review/s$s.out" 2> "review/s$s.err" || { echo "ERROR samples=$s failed"; tail -n 15 "review/s$s.err"; }
  echo "RP_STAGE0_BEGIN $s"; tr -d '\n ' < "review/s$s/stage0.json" 2>/dev/null; echo; echo "RP_STAGE0_END"
done
rm -rf review/s*/world review/s*/r25
