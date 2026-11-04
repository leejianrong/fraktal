#!/usr/bin/env bash
# Driver for the Stage 0 pod job (scripts/runpod/stage0.sh) with a boot check.
# Usage: scripts/runpod/stage0_run.sh COMMIT OUT_DIR     (COMMIT must be pushed to GitHub)
#
# A pod whose container never starts sits "initializing" and prints nothing, so the job's first
# line is RP_BOOT. If it is not in the pod logs within BOOT_TIMEOUT seconds this saves the
# platform's system log, terminates the pod and tries again (ATTEMPTS times). Once booted it waits
# for the artifact with `rp relay-get`. The pod is terminated on every exit path (trap), and the
# account is checked for leftovers at the end. The API key is read by reference from
# ~/.runpod/config.toml and is never printed.
set -uo pipefail
COMMIT="${1:?commit sha}"; OUT="${2:?output dir}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-300}"; ATTEMPTS="${ATTEMPTS:-3}"; POLL="${POLL:-15}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"; LOG="$OUT/driver.log"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

RUNPOD_API_KEY=$(python3 -c "import tomllib,os;print(tomllib.load(open(os.path.expanduser('~/.runpod/config.toml'),'rb'))['apikey'])")
export RUNPOD_API_KEY
export PATH="$HOME/.claude/skills/runpod-jobs/scripts:$PATH"
export RP_POD_NAME=fraktal-stage0 RP_POD_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
export RP_COMPUTE_TYPE=CPU RP_CLOUD_TYPE=SECURE RP_INTERRUPTIBLE=false RP_CPU_FLAVORS=cpu3c,cpu5c
export RP_VCPU_COUNT=16 RP_CONTAINER_DISK_GB=40 RP_MAX_HOURLY_USD=0.60
export RP_MAX_LIFETIME_SECS=3000 RP_IDLE_TIMEOUT_SECS=1800 RP_EXIT_GRACE_SECS=600
export RP_STATE_FILE="$OUT/.rp-state"
export RP_JOB_CMD="echo $(base64 -w0 "$HERE/stage0.sh") | base64 -d > /tmp/job.sh && COMMIT=$COMMIT bash /tmp/job.sh"

cleanup() { say "teardown"; rp down >> "$LOG" 2>&1; say "pods left: $(runpodctl pod list 2>/dev/null | grep -c fraktal-stage0 || true)"; }
trap cleanup EXIT INT TERM

booted=0
for n in $(seq 1 "$ATTEMPTS"); do
  say "attempt $n: launching"
  rp up >> "$LOG" 2>&1 || { say "launch failed"; rp down >> "$LOG" 2>&1; continue; }
  start=$(date +%s)
  while :; do
    elapsed=$(( $(date +%s) - start ))
    rp logs --source container --tail 200 --max-wait 5s > "$OUT/pod.log" 2>&1
    if grep -q RP_BOOT "$OUT/pod.log"; then booted=1; say "booted after ${elapsed}s: $(grep -m1 RP_BOOT "$OUT/pod.log")"; break; fi
    say "waiting for RP_BOOT (${elapsed}s of ${BOOT_TIMEOUT}s), status: $(rp status 2>&1 | tr '\n' ' ' | cut -c1-120)"
    [ "$elapsed" -ge "$BOOT_TIMEOUT" ] && break
    sleep "$POLL"
  done
  [ "$booted" = 1 ] && break
  say "no boot output in ${BOOT_TIMEOUT}s; saving the system log and terminating"
  rp logs --source system --tail 200 --max-wait 5s > "$OUT/system-attempt$n.log" 2>&1
  rp down >> "$LOG" 2>&1
done
[ "$booted" = 1 ] || { say "pod never booted in $ATTEMPTS attempts"; exit 2; }

say "waiting for the artifact (progress below, once a minute)"
rp relay-get "$OUT/artifact" >> "$LOG" 2>&1 &
RG=$!
while kill -0 "$RG" 2>/dev/null; do
  rp logs --source container --tail 400 --max-wait 5s 2>/dev/null | grep -oE '(STAGE [^"]*|hb [^"]*|ERROR[^"]*|WARN[^"]*)' | tail -n 2 | sed 's/^/  pod: /' | tee -a "$LOG"
  sleep 60
done
wait "$RG"; say "relay-get exit $?"
cp "$OUT/artifact/pod.log" "$OUT/job-pod.log" 2>/dev/null || true
