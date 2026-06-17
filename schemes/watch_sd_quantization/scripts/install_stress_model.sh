#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage:
  schemes/watch_sd_quantization/scripts/install_stress_model.sh path/to/model.mlpackage

Compiles a Core ML .mlpackage into watchos_example/WatchStressTestApp/Models
so the WatchStressTestApp scheme can bundle and test it.
USAGE
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_MODEL="$1"
DEST_DIR="$ROOT/watchos_example/WatchStressTestApp/Models"

mkdir -p "$DEST_DIR"
xcrun coremlcompiler compile "$SOURCE_MODEL" "$DEST_DIR"

echo "Installed compiled model under:"
echo "$DEST_DIR"
