#!/usr/bin/env bash
# Pull the two checkpoints into models/.
#
# Two sources, tried in order:
#   1. BOS  - internal object storage, much faster from inside the cluster
#             (measured ~150 KB/s from GitHub's CDN on the same machine)
#   2. GitHub - release assets, for anyone without BOS access
#
# Downloads resume and are checksum-verified. A partial or corrupt transfer must fail
# loudly: a half-written tarball that silently "succeeds" wastes far more time later.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/models"
RETRIES="${RETRIES:-6}"

BOS_PREFIX="${BOS_PREFIX:-bos:/ss-base/zongtianyu/models/video-subject-mask/v1}"
GH_REPO="${REPO:-tianyuzong/video-subject-mask}"
GH_TAG="${TAG:-models-v1}"

# archive:sha256
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

have_bos() { command -v bcecmd >/dev/null 2>&1; }
have_gh()  { command -v curl   >/dev/null 2>&1; }

fetch_from_bos() {   # $1 = archive  -> 0 on success
  local asset="$1"
  have_bos || return 1
  echo "  [bos   ] $BOS_PREFIX/$asset"
  # bcecmd has no resume; a failed run just gets retried whole
  bcecmd bos cp "$BOS_PREFIX/$asset" "$asset" >/dev/null 2>&1 || return 1
  [[ -f "$asset" ]] || return 1
}

fetch_from_github() {  # $1 = archive  -> 0 on success
  local asset="$1" attempt
  have_gh || return 1
  local url="https://github.com/$GH_REPO/releases/download/$GH_TAG/$asset"
  for attempt in $(seq 1 "$RETRIES"); do
    echo "  [github] $asset (attempt $attempt/$RETRIES, resuming if partial)"
    # -C - resumes; a dropped connection just means the next attempt continues
    curl -fL --progress-bar -C - --retry 3 --retry-delay 3 \
         --connect-timeout 20 --speed-time 60 --speed-limit 10240 \
         -o "$asset" "$url" || true
    if [[ -f "$asset" ]] && [[ "$(sha256_of "$asset")" == "${2:-}" ]]; then
      return 0
    fi
  done
  return 1
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
  # whatever partial file may be lying around would fail the checksum anyway
  rm -f "$asset"

  if fetch_from_bos "$asset" && [[ "$(sha256_of "$asset")" == "$want" ]]; then
    ok=1
  else
    [[ -f "$asset" ]] && echo "  [bos   ] unusable, falling back"
    rm -f "$asset"
    if fetch_from_github "$asset" "$want"; then
      ok=1
    fi
  fi

  if [[ "$ok" -ne 1 ]]; then
    echo "[FAIL ] could not fetch a complete $asset" >&2
    echo "        expected sha256 $want" >&2
    echo "        tried BOS:    $BOS_PREFIX/$asset" >&2
    echo "        tried GitHub: https://github.com/$GH_REPO/releases/download/$GH_TAG/$asset" >&2
    exit 1
  fi

  echo "[untar] $asset"
  tar xzf "$asset"
  rm -f "$asset"
done

missing=0
# The detector has processor_config.json, not preprocessor_config.json - the two
# archives differ, and asserting the wrong name fails a perfectly good download.
for d in "mm-gdino-swinb-hf:config.json model.safetensors processor_config.json" \
         "sam2.1-hiera-large:config.json model.safetensors preprocessor_config.json"; do
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
