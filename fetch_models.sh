#!/usr/bin/env bash
# Pull the two checkpoints from the GitHub release into models/.
#
# They are release assets rather than tracked files: at 933 MB and 898 MB they are well
# past GitHub's 100 MB per-file limit, and Git LFS's free 1 GB tier would not hold them.
#
# Downloads resume and are checksum-verified. A partial or corrupt transfer must fail
# loudly - a half-written tarball that silently "succeeds" wastes far more time later.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-tianyuzong/video-subject-mask}"
TAG="${TAG:-models-v1}"
DEST="$HERE/models"
BASE="https://github.com/$REPO/releases/download/$TAG"
RETRIES="${RETRIES:-6}"

# asset:sha256
ASSETS=(
  "mm-gdino-swinb-hf.tar.gz:46197ccd5db465caafd996dbe5c2aca84b4be3f96917c88f4431d4a78073359e"
  "sam2.1-hiera-large.tar.gz:21f559de4646960d6058b674baa7e981f9ffea8fd60f37bb8797598bc05d2c96"
)

sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "$DEST"
cd "$DEST"

for entry in "${ASSETS[@]}"; do
  asset="${entry%%:*}"
  want="${entry##*:}"
  name="${asset%.tar.gz}"

  if [[ -f "$name/model.safetensors" ]]; then
    echo "[skip ] $name already present"
    continue
  fi

  ok=0
  for attempt in $(seq 1 "$RETRIES"); do
    if [[ -f "$asset" ]] && [[ "$(sha256_of "$asset")" == "$want" ]]; then
      ok=1
      break
    fi
    echo "[get  ] $asset (attempt $attempt/$RETRIES, resuming if partial)"
    # -C - resumes; --retry covers transient 5xx; a dropped connection just means the
    # next attempt picks up where this one stopped.
    curl -fL --progress-bar -C - --retry 3 --retry-delay 3 \
         --connect-timeout 20 --speed-time 60 --speed-limit 10240 \
         -o "$asset" "$BASE/$asset" || true
    if [[ -f "$asset" ]] && [[ "$(sha256_of "$asset")" == "$want" ]]; then
      ok=1
      break
    fi
    echo "[warn ] $asset incomplete or checksum mismatch, retrying"
  done

  if [[ "$ok" -ne 1 ]]; then
    echo "[FAIL ] could not fetch a complete $asset after $RETRIES attempts" >&2
    echo "        expected sha256 $want" >&2
    echo "        download it manually from $BASE/$asset and re-run" >&2
    exit 1
  fi

  echo "[untar] $asset"
  tar xzf "$asset"
  rm -f "$asset"
done

missing=0
for d in "mm-gdino-swinb-hf:config.json model.safetensors preprocessor_config.json" \
         "sam2.1-hiera-large:config.json model.safetensors video_preprocessor_config.json"; do
  dir="${d%%:*}"
  for f in ${d##*:}; do
    if [[ ! -f "$dir/$f" ]]; then
      echo "[FAIL ] missing $DEST/$dir/$f" >&2
      missing=1
    fi
  done
done
[[ "$missing" -eq 0 ]] || exit 1

echo
du -sh "$DEST"/*/
echo "models complete"
