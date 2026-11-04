#!/usr/bin/env bash
# CPU pod job (FRK-44): render one procedural set per biome, then produce the biome sheet, gauge
# numbers and distribution plots against the real Khartoum chips. No training, no GPU.
# The real chips are the only real reference we have; they are arid, so gauge numbers for the
# tropical biomes show distance from Khartoum, not from the tropics (Q15 is open).
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper. Needs COMMIT (a pushed sha).
# Optional REGIONS (default 3), MAX_CHIPS (30), SETS ("label:biome:variant ...").
# Nothing here reads or prints a secret. Watch for: STAGE failed, ERROR, WARN, hb lines.
set -uo pipefail
: "${COMMIT:?set COMMIT to the sha to run}"
REGIONS="${REGIONS:-3}"
MAX_CHIPS="${MAX_CHIPS:-30}"
SETS="${SETS:-khartoum_v9:khartoum:v9_randomised_wide dense_urban:dense_urban:v10_tropical informal_settlement:informal_settlement:v10_tropical oil_palm_plantation:oil_palm_plantation:v10_tropical paddy_terraces:paddy_terraces:v10_tropical mangrove_coast:mangrove_coast:v10_tropical}"
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
mkdir -p data/review
render() {  # label, biome, variant, regions, max_chips
  local label=$1 biome=$2 variant=$3 regions=$4 maxc=$5
  if timeout 1500 fraktal synth --regions "$regions" --seed 0 --max-chips "$maxc" --procedural \
      --biome "$biome" --variant "$variant" --palette data/spacenet2/palette.json \
      --out "data/review/$label" > "data/review_$label.log" 2>&1; then
    rm -rf "data/review/$label"/region_* "data/review/$label"/worlds
    echo "render_done $label"
  else
    echo "ERROR render of $label failed or timed out (1500 s); last lines:"
    tail -n 15 "data/review_$label.log"
    return 0
  fi
}
# Smoke first: one tiny render, so a broken flag or biome fails in minutes, not after the batch.
first=${SETS%% *}
IFS=: read -r l b v <<< "$first"
render "smoke_$l" "$b" "$v" 1 2
[ -s "data/review/smoke_$l/chips/chips.json" ] || { echo "ERROR smoke render produced no chips"; false; }
rm -rf "data/review/smoke_$l"
# Stream each render's progress into the job log, so a slow region is visible, not silent.
( while true; do for f in data/review_*.log; do [ -f "$f" ] && echo "[$(basename "$f" .log)] $(tail -n 1 "$f" | cut -c1-160)"; done; sleep 90; done ) &
n=0; pids=()
for spec in $SETS; do
  IFS=: read -r l b v <<< "$spec"
  render "$l" "$b" "$v" "$REGIONS" "$MAX_CHIPS" &
  pids+=($!)
  n=$((n + 1))
  if [ $((n % 3)) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi  # never a bare `wait`
done
[ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}"

stage figures
ARGS=()
for spec in $SETS; do
  IFS=: read -r l b v <<< "$spec"
  if [ -s "data/review/$l/chips/chips.json" ]; then ARGS+=("$l=data/review/$l/chips")
  else echo "WARN no chips for $l, left out of the figures"; fi
done
python src/scripts/review_figs.py data/spacenet2/chips review "${ARGS[@]}"
cp review/chips.png review/biome_sheet.png
echo "RP_GAUGE_BEGIN"; tr -d '\n ' < review/gauge.json; echo; echo "RP_GAUGE_END"
