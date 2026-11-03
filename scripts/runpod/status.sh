#!/usr/bin/env bash
# One-screen status of a running benchmark pod. Usage: scripts/runpod/status.sh <pod-id>
# runpodctl authenticates itself (its own config, or RUNPOD_API_KEY in the environment); this
# script never touches the key.
set -euo pipefail
id="${1:?usage: status.sh <pod-id>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runpodctl pod logs "$id" --tail 5000 --max-wait 8s 2>&1 | python3 -I "$here/status.py"
