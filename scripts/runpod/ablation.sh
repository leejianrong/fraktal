#!/usr/bin/env bash
# Pod-side job for the V3 realism ablation (FRK-18). Run it as RP_JOB_CMD under the runpod-jobs
# `rp` wrapper. Needs COMMIT (a pushed sha); optional SEEDS (default "0 1"), STEPS, VARIANTS
# (space-separated names, default all) and FRACTION (real share for arms 1 and 3, default 0.1).
# Nothing here reads or prints a secret.
#
# Watch it with scripts/runpod/status.sh <pod-id>. Failure signals, all grep-able in the logs:
#   STAGE failed ...     a step exited non-zero (with the line and the stage it was in)
#   ERROR <what>         a background job died (synth of a variant)
#   WARN <what>          no output for a while, an idle GPU, a nearly full disk or RAM
#   hb ...               a heartbeat every 60 s: stage, GPU, disk, RAM, seconds since output
# Results are logged as they finish (fraktal-log ablation_result lines) and once more at the end, so the numbers
# are in the logs even if the relay transfer fails. A failed run still publishes what it has.
set -uo pipefail
: "${COMMIT:?set COMMIT to the sha to run}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-}"
FRACTION="${FRACTION:-0.1}"
VARIANTS="${VARIANTS:-}"
RUN=/workspace/run
mkdir -p "$RUN" && cd "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
: > "$RUN/status"

stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1"; }
fail() { echo "STAGE failed ($1; was: $(cat "$RUN/stage" 2>/dev/null))"; echo failed > "$RUN/status"; }
trap 'fail "line $LINENO"' ERR
set -e

# Heartbeat and watchdog. It also keeps the pod's idle watchdog quiet during a long silent step;
# the lifetime ceiling stays the hard stop.
(
  idle=0
  while true; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0")
    util=${gpu%%,*}
    disk_pct=$(df /workspace | awk 'NR==2{gsub("%","",$5); print $5}')
    ram_pct=$(free | awk 'NR==2{printf "%d", $3*100/$2}')
    quiet=$(( $(date +%s) - $(stat -c %Y "$RUN/job.log") ))
    st=$(cat "$RUN/stage" 2>/dev/null || echo boot)
    echo "hb stage=$st gpu=[util ${util}% mem,temp ${gpu#*,}] disk=${disk_pct}% ram=${ram_pct}% quiet=${quiet}s"
    [ "$quiet" -gt 600 ] && echo "WARN no output for ${quiet}s in stage $st"
    [ "$disk_pct" -gt 90 ] && echo "WARN disk ${disk_pct}% full"
    [ "$ram_pct" -gt 90 ] && echo "WARN ram ${ram_pct}% used"
    case "$st" in ablate*)
      if [ "$util" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
      [ "$idle" -ge 6 ] && echo "WARN gpu idle for ${idle} min during $st" ;;
    *) idle=0 ;;
    esac
    sleep 60
  done
) &
HB=$!

publish() {
  stage publish
  cd "$RUN"
  [ -s results/ablation.json ] || { echo "no results to publish"; return 0; }
  echo "RP_REPORT_BEGIN"; tr -d '\n ' < results/ablation.json; echo; echo "RP_REPORT_END"
  tar czf results.tgz results job.log data/ablation/*/chips/chips.json data/ablation/*/samples \
    data/spacenet2/khartoum-split.json 2>/dev/null || true
  BASE="fraktal-$RANDOM-$RANDOM"
  timeout 1500 runpodctl send --code "$BASE" results.tgz > send.log 2>&1 &
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
echo "commit $COMMIT"
pip install -q --root-user-action=ignore ./src
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export FRAKTAL_DATA_DIR=/workspace/run/data
mkdir -p results data/ablation
stage ingest; fraktal ingest
stage split;  fraktal split --seed 0
fraktal ablate --help > /dev/null   # fail now, not after the data is built, if the command is broken

# Render every variant on the CPU in the background while the GPU trains the earlier ones.
VARS="${VARIANTS:-$(python -c "from fraktal.ablation import VARIANTS; print(' '.join(VARIANTS))")}"
VARLIST=($VARS)
(
  for v in $VARS; do
    echo "synth_start $v"
    if fraktal synth --regions 8 --seed 0 --variant "$v" --out "data/ablation/$v" > "data/ablation/$v.log" 2>&1; then
      rm -rf data/ablation/"$v"/region_*   # the per-region tiles are bulky and unused after chips
      echo "synth_done $v"
    else
      mkdir -p "data/ablation/$v"; touch "data/ablation/$v/FAILED"
      echo "ERROR synth of $v failed; last lines:"; tail -n 15 "data/ablation/$v.log"
      break
    fi
  done
) &
SYNTH=$!

# A short end-to-end smoke test of the real code path on the GPU as soon as the first variant
# exists: if anything is broken it shows here, minutes in, not an hour in.
stage "smoke (30 steps)"
fraktal ablate --seeds 0 --variants "${VARLIST[0]}" --fraction "$FRACTION" --steps 30 --device cuda --wait-secs 1800 \
  --out results/smoke.json
echo "smoke ok"

stage "ablate (seeds $SEEDS)"
EXTRA=()
[ -n "$STEPS" ] && EXTRA=(--steps "$STEPS")
timeout 11000 fraktal ablate --seeds $SEEDS --variants "${VARLIST[@]}" --fraction "$FRACTION" --device cuda --wait-secs 3600 \
  --out results/ablation.json "${EXTRA[@]}" 2>&1 | tee ablate.log || true
test "$(python -c "import json;print(json.load(open('results/ablation.json'))['status'])")" = complete
wait "$SYNTH" || true
