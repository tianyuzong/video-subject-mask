#!/usr/bin/env bash
# Submit subject masking to the a6000 cluster.
#
#   ./run_gpu_job.sh --dry-run  video1.mp4 [video2.mp4 ...]
#   ./run_gpu_job.sh            video1.mp4 [video2.mp4 ...]
#
# The a6000 queue allocates a whole 8-GPU node, so the worker script shards the
# video list across the 8 GPUs (one process per GPU, whole videos per process).
# A single video therefore uses one GPU and leaves the rest idle - batch several
# videos per submission when you can.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
  shift
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--dry-run] <video> [video ...]" >&2
  exit 2
fi

OUT="${OUT:-$HERE/outputs/subject_mask_gpu}"
DETECTOR="${DETECTOR:-$HERE/models/mm-gdino-swinb-hf}"
SAM="${SAM:-$HERE/models/sam2.1-hiera-large}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT"
LIST="$OUT/videos.txt"
: > "$LIST"
for v in "$@"; do
  readlink -f "$v" >> "$LIST"
done
echo "queued $(wc -l < "$LIST") video(s) -> $LIST"

cd "$HERE"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# shellcheck disable=SC2086
sslaunch submit \
  -c a6000 -q a6000 \
  -j subj-mask \
  -n 1 \
  -w "$HERE" \
  --no-log \
  $DRY_RUN \
  -- bash "$HERE/gpu_worker.sh" "$LIST" "$OUT" "$DETECTOR" "$SAM" $EXTRA_ARGS
