#!/usr/bin/env bash
# Render eye-level street strips for one or more city biomes on a RunPod CPU pod, one at a time,
# then unpack them. Usage: scripts/runpod/city_strips.sh <pushed commit sha> <tag> <biome>...
#   e.g. scripts/runpod/city_strips.sh 433eaac... 11 manila_quezon nairobi_centre khartoum_city
# Output: data/review/street<tag>_<biome>/out/review/fly/frames/*.png, and a progress line per city
# in data/review/cities<tag>_progress.txt (the last line is ALL_DONE). Pods render pushed commits
# only, so push first. Each city checks that the driver log says "pods left: 0".
# Compare with: fraktal streets <frames dir> --city '<glob>' --data-dir data --limit 30 --segment --sheet <png>
set -u
cd "$(dirname "$0")/../.."
COMMIT=$1 TAG=$2; shift 2
for city in "$@"; do
  OUT=data/review/street${TAG}_$city; rm -rf "$OUT"; mkdir -p "$OUT"
  MODE=fly STREET=1 SPEED=8 FOV=75 PEOPLE_PER_KM=30 BATCH_BIOME=$city SAMPLES=32 \
    scripts/runpod/stage0_run.sh "$COMMIT" "$OUT" > "$OUT/launch.out" 2>&1
  mkdir -p "$OUT/out" && tar xzf "$OUT/artifact/review.tgz" -C "$OUT/out" 2>/dev/null
  echo "$city done: $(grep -c 'pods left: 0' "$OUT/driver.log") pods-left-0, frames: $(ls "$OUT/out/review/fly/frames" 2>/dev/null | wc -l)" >> data/review/cities${TAG}_progress.txt
done
echo ALL_DONE >> data/review/cities${TAG}_progress.txt
