#!/usr/bin/env bash
# Pull the two checkpoints from the GitHub release into models/.
#
# They are release assets rather than tracked files: at 933 MB and 898 MB they are well
# past GitHub's 100 MB per-file limit, and Git LFS's free 1 GB tier would not hold them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-tianyuzong/video-subject-mask}"
TAG="${TAG:-models-v1}"
DEST="$HERE/models"

mkdir -p "$DEST"
cd "$DEST"

if ! command -v gh >/dev/null; then
  echo "gh CLI not found. Install it, or download the assets manually from" >&2
  echo "  https://github.com/$REPO/releases/tag/$TAG" >&2
  echo "and untar them into $DEST" >&2
  exit 1
fi

for asset in mm-gdino-swinb-hf.tar.gz sam2.1-hiera-large.tar.gz; do
  name="${asset%.tar.gz}"
  if [[ -f "$name/model.safetensors" ]]; then
    echo "[skip] $name already present"
    continue
  fi
  echo "[get ] $asset"
  gh release download "$TAG" --repo "$REPO" --pattern "$asset" --clobber
  echo "[untar] $asset"
  tar xzf "$asset"
  rm -f "$asset"
done

echo
echo "models in $DEST:"
du -sh "$DEST"/*/ 2>/dev/null || true

python3 - "$DEST" <<'PY'
import pathlib, sys
dest = pathlib.Path(sys.argv[1])
expected = {
    "mm-gdino-swinb-hf": ["config.json", "model.safetensors", "preprocessor_config.json"],
    "sam2.1-hiera-large": ["config.json", "model.safetensors", "video_preprocessor_config.json"],
}
bad = False
for d, files in expected.items():
    for f in files:
        p = dest / d / f
        if not p.exists():
            print(f"MISSING {p}")
            bad = True
print("model files look complete" if not bad else "model files INCOMPLETE")
sys.exit(1 if bad else 0)
PY
