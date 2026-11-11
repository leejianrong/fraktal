#!/usr/bin/env bash
# GPU pod job (FRK-122): the image-to-3D props trial. Needs COMMIT (a pushed sha).
# Stack (ADR-0013): SDXL base 1.0 -> BiRefNet -> Step1X-3D geometry. Nothing here reads or prints a
# secret: every model is ungated, so no Hugging Face token is used.
# Run as RP_JOB_CMD under the runpod-jobs `rp` wrapper (see props_run.sh).
echo "RP_BOOT $(date +%s) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
set -uo pipefail
T0=$(date +%s)
: "${COMMIT:?set COMMIT to the sha to run}"
export STEP1X_SHA="${STEP1X_SHA:-cb5ac944709c6c913109070c7b90c3447f57f3d4}"  # exported: text_to_3d.py records it
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
  tar czf review.tgz --exclude=review/meshes_raw review job.log 2>/dev/null || true  # raw meshes are 1M faces each
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
git clone -q https://github.com/stepfun-ai/Step1X-3D.git /workspace/step1x && git -C /workspace/step1x checkout -q "$STEP1X_SHA"
python -m venv --system-site-packages /workspace/props
PIP=/workspace/props/bin/pip
timeout 900 $PIP install -q diffusers==0.32.2 transformers==4.48.0 accelerate safetensors einops jaxtyping omegaconf \
  timm trimesh scikit-image scipy PyMCubes pymeshlab rembg onnxruntime opencv-python-headless pillow \
  matplotlib fast-simplification easydict kornia huggingface-hub typeguard pytorch-lightning \
  mosaicml-streaming datasets plyfile imageio pyyaml || echo "ERROR pip install returned $?"
/workspace/props/bin/python -c "import torch,diffusers,transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
/workspace/props/bin/pip freeze > review/pip-freeze.txt
nvidia-smi > review/nvidia-smi.txt 2>&1 || true

# Third-party research code runs here and nowhere else. Import it first, before any big download:
# fix a missing module a few times, then give up and say which one.
stage import_check
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  err=$(cd /workspace/step1x && /workspace/props/bin/python -c "
import sys; sys.path.insert(0, '.')
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
print('import ok')" 2>&1)
  if echo "$err" | grep -q "import ok"; then echo "import_check ok after $try tries"; break; fi
  echo "import_check try $try failed: $(echo "$err" | tail -n 3 | tr '\n' ' ')"
  mod=$(echo "$err" | grep -oE "No module named '[A-Za-z0-9_]+" | tail -1 | sed "s/No module named '//")
  [ -z "$mod" ] && break
  case "$mod" in cv2) mod=opencv-python-headless;; skimage) mod=scikit-image;; mcubes) mod=PyMCubes;;
    streaming) mod=mosaicml-streaming;; pytorch_lightning) mod=pytorch-lightning;; yaml) mod=pyyaml;; sklearn) mod=scikit-learn;; PIL) mod=pillow;; esac
  timeout 300 $PIP install -q "$mod" || echo "WARN could not install $mod"
done

if [ "${MODE:-trial}" = text3d ]; then  # FRK-123: a prompt file in, meshes with sidecars out, then a Blender sheet
  stage text3d
  timeout 2700 /workspace/props/bin/python src/scripts/props/text_to_3d.py review --step1x /workspace/step1x \
    --prompts "src/scripts/props/${PROMPTS:-prompts-20.json}" --seed "${SEED:-0}" --usd-per-hour "${USD_PER_HOUR:-0.59}" \
    || echo "ERROR text_to_3d exited $?"
  stage blender_sheet
  timeout 300 apt-get update -qq > /dev/null
  for attempt in 1 2; do  # bpy is a headless wheel that still links X11 and GL libraries the image lacks
    timeout 300 apt-get install -y -qq libxrender1 libxi6 libxxf86vm1 libxfixes3 libgl1 libsm6 libxkbcommon0 libegl1 > /dev/null && break
  done
  python -m venv /workspace/bpyenv && /workspace/bpyenv/bin/pip install -q bpy numpy \
    && timeout 1500 /workspace/bpyenv/bin/python src/scripts/props/bpy_glb_sheet.py review/meshes review/blender --px 320 --samples 16 \
    || echo "ERROR blender sheet exited $?"
else
  stage trial
  timeout 2700 /workspace/props/bin/python src/scripts/props/props_trial.py review --step1x /workspace/step1x ${TRIAL_ARGS:-} \
    || echo "ERROR props_trial exited $?"
fi
du -sh review | sed 's/^/RP_ARTIFACT_SIZE /'
