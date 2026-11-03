#!/usr/bin/env bash
# Pod-side job for the V1 benchmark. Run it as RP_JOB_CMD under the runpod-jobs `rp` wrapper.
# Needs COMMIT (a full sha of a pushed commit). Nothing here reads or prints a secret.
set -euo pipefail
: "${COMMIT:?set COMMIT to the sha to benchmark}"
SEEDS="${SEEDS:-0 0 1}"
mkdir -p /workspace/run && cd /workspace/run

stage() { echo "$1" > /workspace/run/stage; echo "STAGE $1"; }
trap 'echo "STAGE failed (line $LINENO, was: $(cat /workspace/run/stage 2>/dev/null))"' ERR

# Heartbeat: stage, GPU load and memory, disk and RAM, once a minute. It also keeps the pod's
# idle watchdog from reaping a long silent step, so the lifetime ceiling is the only hard stop.
(
  while true; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no-gpu")
    disk=$(df -h /workspace | awk 'NR==2{print $3"/"$2}')
    mem=$(free -m | awk 'NR==2{print $3"/"$2"MB"}')
    echo "hb stage=$(cat /workspace/run/stage 2>/dev/null || echo boot) gpu=[$gpu] disk=$disk ram=$mem"
    sleep 60
  done
) &

stage install
git clone -q https://github.com/leejianrong/fraktal.git src
git -C src checkout -q "$COMMIT"
echo "commit $COMMIT"
pip install -q --root-user-action=ignore ./src
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export FRAKTAL_DATA_DIR=/workspace/run/data
stage ingest;  fraktal ingest
stage split;   fraktal split --seed 0
stage synth;   fraktal synth --regions 8 --seed 0
mkdir -p results
n=0
for seed in $SEEDS; do
  n=$((n + 1))
  stage "benchmark $n seed $seed"
  fraktal benchmark --seed "$seed" --device cuda --synthetic data/synthetic/chips \
    --out "results/report_${n}_seed${seed}.json"
  echo "RP_REPORT_BEGIN $n seed $seed"; tr -d '\n ' < "results/report_${n}_seed${seed}.json"; echo
  echo "RP_REPORT_END"
done

stage publish
tar czf results.tgz results data/spacenet2/khartoum-split.json data/synthetic/samples \
  data/synthetic/chips/chips.json
BASE="fraktal-$RANDOM-$RANDOM"
runpodctl send --code "$BASE" results.tgz > send.log 2>&1 &
sleep 8
echo "RP_ARTIFACT_CODE=$(grep -oE "${BASE}[-0-9]+" send.log | head -1)"
wait
stage done
