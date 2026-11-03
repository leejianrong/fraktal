# Running the benchmark on RunPod

The local machine has no GPU, so the full three-arm benchmark runs on a rented pod. The pod
installs a pushed commit from the public repo, downloads SpaceNet itself and rebuilds the data,
so nothing derived from SpaceNet leaves the laptop (ADR-0008).

## Launch

Use the `runpod-jobs` skill's `rp` wrapper. Set the usual `RP_*` variables (GPU types, image,
`RP_MAX_HOURLY_USD`, `RP_MAX_LIFETIME_SECS`) and run the pod-side job:

```bash
export RP_JOB_CMD="COMMIT=<full sha> bash <(echo '<base64 of scripts/runpod/job.sh>' | base64 -d)"
```

The key is read from the environment by reference and never printed. The job takes `COMMIT`
(a pushed sha) and optional `SEEDS` (default `0 0 1`: seed 0 twice for the determinism check, then
seed 1). A 3000-step arm takes about 3 minutes on an A40.

## Watch

Every long step writes structured progress lines to stderr, prefixed `fraktal-log` (see
`src/fraktal/log.py`): download bytes, chip and region counts, training step with loss, steps per
second and ETA, evaluation progress, and a line per finished arm. The pod also writes `STAGE <name>`
markers and a once-a-minute heartbeat with GPU load and memory, disk and RAM.

```bash
scripts/runpod/status.sh <pod-id>      # stage, heartbeat, latest progress, any problem
```

Quiet for more than a few minutes, or no heartbeat, means the pod is stuck: look at
`runpodctl pod logs <id> --source system` and terminate it.

## Collect and tear down

The job prints each finished report between `RP_REPORT_BEGIN` and `RP_REPORT_END`, so the numbers
are readable from the logs even if the relay transfer fails. Afterwards confirm
`runpodctl pod list` no longer shows the pod.

## The V3 ablation (`scripts/runpod/ablation.sh`)

`fraktal ablate` runs arms 1 and 2 once per seed, then arm 3 for each realism variant
(`fraktal.ablation.VARIANTS`, cumulative from the flat V1 render to the randomised one). The pod
job renders the variants on the CPU in the background while the GPU trains earlier ones, runs a
30-step smoke test as soon as the first variant exists, and publishes whatever it has even if a
step fails. Launch it like the V1 job, with `COMMIT` and optionally `SEEDS` (default `0 1`).

Failure signals in the logs: `STAGE failed`, `ERROR <what>` (a background synth died), `WARN`
(no output for 10 minutes, GPU idle for 6 minutes during training, disk or RAM above 90%), and a
heartbeat that stops. `scripts/runpod/status.sh <pod-id>` lists them along with each finished
`ablation_result`.
