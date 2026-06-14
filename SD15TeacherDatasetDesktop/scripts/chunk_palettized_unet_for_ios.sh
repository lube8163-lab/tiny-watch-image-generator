#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
APPLE_REPO="$ROOT_DIR/.build/ml-stable-diffusion-1.1.1"
MODEL_DIR="$ROOT_DIR/artifacts/sdxl-base-ios"
RESOURCES_DIR="$MODEL_DIR/Resources"
SOURCE_MLPACKAGE="$MODEL_DIR/Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet_6bit.mlpackage"
CHUNK1_MLPACKAGE="$MODEL_DIR/Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet_6bit_chunk1.mlpackage"
CHUNK2_MLPACKAGE="$MODEL_DIR/Stable_Diffusion_version_stabilityai_stable-diffusion-xl-base-1.0_unet_6bit_chunk2.mlpackage"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$APPLE_REPO/python_coreml_stable_diffusion" ]; then
  echo "Apple repo not found at: $APPLE_REPO" >&2
  exit 1
fi

if [ ! -d "$SOURCE_MLPACKAGE" ]; then
  echo "Palettized UNet mlpackage not found at: $SOURCE_MLPACKAGE" >&2
  echo "Run ./scripts/run_unet_6bit_palettization.sh first." >&2
  exit 1
fi

(
  cd "$APPLE_REPO"
  "$PYTHON_BIN" -m python_coreml_stable_diffusion.chunk_mlprogram \
    --mlpackage-path "$SOURCE_MLPACKAGE" \
    -o "$MODEL_DIR"
)

rm -rf "$RESOURCES_DIR/UnetChunk1.mlmodelc" "$RESOURCES_DIR/UnetChunk2.mlmodelc"
xcrun coremlcompiler compile "$CHUNK1_MLPACKAGE" "$RESOURCES_DIR"
xcrun coremlcompiler compile "$CHUNK2_MLPACKAGE" "$RESOURCES_DIR"
mv "$RESOURCES_DIR/$(basename "$CHUNK1_MLPACKAGE" .mlpackage).mlmodelc" "$RESOURCES_DIR/UnetChunk1.mlmodelc"
mv "$RESOURCES_DIR/$(basename "$CHUNK2_MLPACKAGE" .mlpackage).mlmodelc" "$RESOURCES_DIR/UnetChunk2.mlmodelc"

rm -rf "$RESOURCES_DIR/Unet.mlmodelc"

echo "Chunked UNet compiled to:"
echo "  $RESOURCES_DIR/UnetChunk1.mlmodelc"
echo "  $RESOURCES_DIR/UnetChunk2.mlmodelc"
