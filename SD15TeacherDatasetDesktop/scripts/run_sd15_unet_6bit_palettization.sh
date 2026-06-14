#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
MODEL_DIR="$ROOT_DIR/artifacts/ios-models/sd15/512"
SOURCE_MLPACKAGE="$MODEL_DIR/Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage"
OUTPUT_MLPACKAGE="$MODEL_DIR/Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet_6bit.mlpackage"
RESOURCES_DIR="$MODEL_DIR/Resources"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$SOURCE_MLPACKAGE" ]; then
  echo "UNet mlpackage not found at: $SOURCE_MLPACKAGE" >&2
  echo "Run ./scripts/run_first_sd15_conversion.sh first." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/apply_fixed_bit_palettization.py" \
  --mlpackage-path "$SOURCE_MLPACKAGE" \
  --output-mlpackage-path "$OUTPUT_MLPACKAGE" \
  --nbits 6 \
  --compile-to "$RESOURCES_DIR" \
  --final-name "Unet"
