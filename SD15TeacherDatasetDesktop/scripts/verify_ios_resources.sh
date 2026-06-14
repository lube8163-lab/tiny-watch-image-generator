#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="${1:-$ROOT_DIR/artifacts/sdxl-base-ios/Resources}"

require_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

require_dir "$RESOURCES_DIR"
require_dir "$RESOURCES_DIR/TextEncoder.mlmodelc"
require_dir "$RESOURCES_DIR/TextEncoder2.mlmodelc"
require_dir "$RESOURCES_DIR/VAEDecoder.mlmodelc"
require_file "$RESOURCES_DIR/vocab.json"
require_file "$RESOURCES_DIR/merges.txt"

if [ -d "$RESOURCES_DIR/Unet.mlmodelc" ]; then
  :
elif [ -d "$RESOURCES_DIR/UnetChunk1.mlmodelc" ] && [ -d "$RESOURCES_DIR/UnetChunk2.mlmodelc" ]; then
  :
else
  echo "Missing Unet.mlmodelc or UnetChunk1/UnetChunk2.mlmodelc in $RESOURCES_DIR" >&2
  exit 1
fi

echo "Resources look complete:"
echo "  $RESOURCES_DIR"
du -sh "$RESOURCES_DIR" \
  "$RESOURCES_DIR/Unet.mlmodelc" \
  "$RESOURCES_DIR/UnetChunk1.mlmodelc" \
  "$RESOURCES_DIR/UnetChunk2.mlmodelc" 2>/dev/null || true
