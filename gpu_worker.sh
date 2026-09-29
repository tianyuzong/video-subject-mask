#!/usr/bin/env bash
# In-container entrypoint: fan the video list out over the node's GPUs.
# Usage: gpu_worker.sh <list.txt> <outdir> <detector> <sam> [extra subject_mask.py args...]
set -euo pipefail

LIST="$1"; OUT="$2"; DETECTOR="$3"; SAM="$4"; shift 4
EXTRA=("$@")

PY=/miniconda3/envs/ss-vidu/bin/python
command -v "$PY" >/dev/null || PY=python3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "host=$(hostname) python=$PY"
"$PY" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"
command -v ffmpeg >/dev/null || { echo "ffmpeg missing in image" >&2; exit 1; }

NGPU="$("$PY" -c 'import torch;print(max(1,torch.cuda.device_count()))')"
mkdir -p "$OUT/logs"

shard=0
pids=()
while [[ $shard -lt $NGPU ]]; do
  (
    idx=0
    while IFS= read -r video; do
      [[ -z "$video" ]] && continue
      if (( idx % NGPU == shard )); then
        name="$(basename "${video%.*}")"
        echo "[gpu$shard] $video"
        CUDA_VISIBLE_DEVICES="$shard" "$PY" -u "$HERE/subject_mask.py" \
          --video "$video" \
          --detector "$DETECTOR" \
          --sam "$SAM" \
          --out "$OUT" \
          --device cuda \
          "${EXTRA[@]}" \
          > "$OUT/logs/${name}.gpu${shard}.log" 2>&1 \
          || echo "[gpu$shard] FAILED $video" >&2
      fi
      idx=$((idx + 1))
    done < "$LIST"
  ) &
  pids+=($!)
  shard=$((shard + 1))
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

echo "=== results in $OUT ==="
ls -l "$OUT"
exit "$fail"
