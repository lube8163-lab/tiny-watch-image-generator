#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="${1:-$ROOT_DIR/artifacts/sdxl-base-ios/Resources}"
OUTPUT_ZIP="${2:-$ROOT_DIR/artifacts/sdxl-base-ios/SDXLBaseResources.zip}"

"$ROOT_DIR/scripts/verify_ios_resources.sh" "$RESOURCES_DIR"
rm -f "$OUTPUT_ZIP"

cd "$(dirname "$RESOURCES_DIR")"
/usr/bin/zip -r "$OUTPUT_ZIP" "$(basename "$RESOURCES_DIR")" >/dev/null

echo "Created:"
echo "  $OUTPUT_ZIP"
