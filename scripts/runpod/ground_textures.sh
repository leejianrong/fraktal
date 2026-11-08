#!/usr/bin/env bash
# GPU pod job (FRK-126): SDXL ground and plaster textures, derived maps and a Marigold normals comparison. Needs COMMIT (a pushed sha). Nothing here reads
# or prints a secret; the model is ungated so no Hugging Face token is used.
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper (see textures_run.sh).
echo "RP_BOOT $(date +%s) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
set -uo pipefail
T0=$(date +%s)
: "${COMMIT:?set COMMIT to the sha to run}"
RUN=/workspace/run
mkdir -p "$RUN" && cd "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
echo install > "$RUN/stage"; echo ok > "$RUN/status"
stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1 t=$(( $(date +%s) - T0 ))s"; }
(
  while true; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo no-gpu)
    echo "hb stage=$(cat "$RUN/stage") gpu=[$gpu] disk=$(df -h /workspace | awk 'NR==2{print $3"/"$2}')"
    sleep 60
  done
) &
HB=$!
publish() {
  stage publish
  tar czf review.tgz review job.log 2>/dev/null || true
  BASE="fraktal-$RANDOM-$RANDOM"
  timeout 600 runpodctl send --code "$BASE" review.tgz > send.log 2>&1 &
  SEND=$!
  sleep 8
  echo "RP_ARTIFACT_CODE=$(grep -oE "${BASE}[-0-9]+" send.log | head -1)"
  wait "$SEND" || true
}
trap 'publish; kill $HB 2>/dev/null; stage done' EXIT
stage install
git clone -q https://github.com/leejianrong/fraktal.git src && git -C src checkout -q "$COMMIT"
# Pinned PyPI releases, in the pod's own container only. The image already has torch.
python -m venv --system-site-packages /workspace/tex
/workspace/tex/bin/pip install -q diffusers==0.35.1 transformers==4.56.1 accelerate==1.10.1 safetensors pillow
/workspace/tex/bin/python -c "import torch,diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__)"
stage generate
mkdir -p review
/workspace/tex/bin/python src/scripts/textures/sdxl_ground.py review --seeds "${SEEDS:-4}" --steps "${STEPS:-20}" \
  || echo "ERROR sdxl_ground failed"
du -sh review | sed 's/^/RP_ARTIFACT_SIZE /'
