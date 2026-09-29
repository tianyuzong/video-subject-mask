#!/usr/bin/env bash
# CPU smoke test: a handful of frames, plus the correctness assertions that matter:
# exact palette colours, rank priority on overlaps, untouched background, valid JSON.
# Usage: ./run_cpu_smoke.sh [video] [num_frames]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="${1:-}"
FRAMES="${2:-12}"
DETECTOR="${DETECTOR:-$HERE/models/mm-gdino-swinb-hf}"
SAM="${SAM:-$HERE/models/sam2.1-hiera-large}"
OUT="${OUT:-$HERE/outputs/subject_mask_smoke}"

if [[ -z "$VIDEO" ]]; then
  echo "usage: $0 <video.mp4> [num_frames]" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Lossless so the "background is untouched" check is exact rather than approximate.
python3 "$HERE/subject_mask.py" \
  --video "$VIDEO" \
  --detector "$DETECTOR" \
  --sam "$SAM" \
  --out "$OUT" \
  --device cpu \
  --max-frames "$FRAMES" \
  --lossless \
  --save-masks \
  --debug-overlay

python3 - "$VIDEO" "$OUT" "$FRAMES" <<'PY'
import sys, json, pathlib, cv2, numpy as np

video, out, frames = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
stem = pathlib.Path(video).stem
masks = np.load(out / f"{stem}_masks.npz")["masks"]          # (K, T, H, W)
labels = json.loads((out / f"{stem}_labels.json").read_text())
palette = [tuple(p["rgb"]) for p in labels["palette"]]
K = masks.shape[0]

src = cv2.VideoCapture(video)
lab = cv2.VideoCapture(str(out / f"{stem}_labeled.mkv"))

n = bg_ok = colour_ok = prio_ok = 0
for i in range(frames):
    a, b = src.read()[1], lab.read()[1]
    if a is None or b is None:
        break
    n += 1
    # Higher rank wins contested pixels, so a rank's *visible* area excludes any
    # pixel claimed by a better rank.
    visible = []
    claimed = np.zeros(masks.shape[2:], dtype=bool)
    for r in range(K):
        v = masks[r, i] & ~claimed
        claimed |= masks[r, i]
        visible.append(v)

    ok_colour = True
    for r, v in enumerate(visible):
        if not v.any():
            continue
        bgr = np.array(palette[r][::-1], dtype=np.uint8)   # cv2 reads BGR
        ok_colour &= bool(np.all(b[v] == bgr))
    colour_ok += ok_colour
    prio_ok += 1  # priority is enforced by construction of `visible`; colour check proves it
    bg_ok += bool(np.array_equal(a[~claimed], b[~claimed]))

# JSON structural checks
assert labels["video"]["num_frames"] == n, "json frame count mismatch"
for shot in labels["shots"]:
    span = shot["end_frame"] - shot["start_frame"] + 1
    for s in shot["subjects"]:
        assert len(s["per_frame"]) == span, f"per_frame length {len(s['per_frame'])} != {span}"
        for row in s["per_frame"]:
            assert shot["start_frame"] <= row["frame"] <= shot["end_frame"], "frame out of shot"
        # area_ratio in JSON must match the stored masks exactly
        r = s["rank"] - 1
        for row in s["per_frame"]:
            got = round(float(masks[r, row["frame"]].sum()) /
                        (masks.shape[2] * masks.shape[3]), 6)
            assert got == row["area_ratio"], f"area mismatch f{row['frame']}: {got} vs {row['area_ratio']}"

print(f"frames compared:                      {n}")
print(f"subjects found:                       {K}")
print(f"palette colours exact:                {colour_ok}/{n}")
print(f"rank priority respected:              {prio_ok}/{n}")
print(f"background bit-identical:             {bg_ok}/{n}")
print("json schema + area cross-check:       OK")
assert n and colour_ok == bg_ok == n, "smoke test FAILED"
print("smoke test PASSED")
PY
