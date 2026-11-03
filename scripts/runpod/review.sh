#!/usr/bin/env bash
# CPU pod job: render procedural and skeleton synthetic sets, then produce the gauge numbers,
# distribution plots and chip strips against the real chips (FRK-30). No training, no GPU.
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper. Needs COMMIT (a pushed sha).
# Optional REGIONS (default 6), MAX_CHIPS (60), VARIANTS ("v8_objects v6_randomised").
# Nothing here reads or prints a secret. Watch for: STAGE failed, ERROR, WARN, hb lines.
set -uo pipefail
: "${COMMIT:?set COMMIT to the sha to run}"
REGIONS="${REGIONS:-6}"
MAX_CHIPS="${MAX_CHIPS:-60}"
VARIANTS="${VARIANTS:-v8_objects v6_randomised}"
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
pip install -q --root-user-action=ignore ./src matplotlib
export FRAKTAL_DATA_DIR=/workspace/run/data
stage ingest;  fraktal ingest
stage split;   fraktal split --seed 0
stage chips
python - <<'PY'
from pathlib import Path
from fraktal.chips import build_chips
d = Path("data")
print("chips at", build_chips(d, d / "spacenet2" / "khartoum-split.json"))
PY
stage palette; fraktal palette --seed 0
mkdir -p review

stage render
# Two renders at a time keeps the CPU pod busy without thrashing; each uses several threads.
render() {  # name, extra flags
  local name=$1; shift
  if timeout 900 fraktal synth --regions "$REGIONS" --seed 0 --max-chips "$MAX_CHIPS" \
      --palette data/spacenet2/palette.json "$@" --out "data/review/$name" > "data/review_$name.log" 2>&1; then
    rm -rf "data/review/$name"/region_*
    echo "render_done $name"
  else
    echo "ERROR render of $name failed or timed out (900 s); last lines:"
    tail -n 15 "data/review_$name.log"
    return 0
  fi
}
mkdir -p data/review
# Stream each render's progress into the job log, so a slow region is visible, not silent.
( while true; do for f in data/review_*.log; do [ -f "$f" ] && echo "[$(basename "$f" .log)] $(tail -n 1 "$f" | cut -c1-160)"; done; sleep 90; done ) &
for v in $VARIANTS; do
  render "proc_$v" --variant "$v" --procedural &
  render "skel_$v" --variant "$v" &
  wait -n; wait -n || true
  wait
done

stage figures
ARGS=()
for v in $VARIANTS; do
  for kind in skel proc; do
    if [ -s "data/review/${kind}_$v/chips/chips.json" ]; then ARGS+=("${kind}_$v=data/review/${kind}_$v/chips")
    else echo "WARN no chips for ${kind}_$v, left out of the figures"; fi
  done
done
python src/scripts/review_figs.py data/spacenet2/chips review "${ARGS[@]}"
echo "RP_GAUGE_BEGIN"; tr -d '\n ' < review/gauge.json; echo; echo "RP_GAUGE_END"
for v in $VARIANTS; do cp -r "data/review/proc_$v/samples" "review/samples_proc_$v" 2>/dev/null || true; done
