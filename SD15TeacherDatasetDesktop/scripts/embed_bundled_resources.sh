#!/bin/sh
set -eu

ROOT_SRC="${SRCROOT}/artifacts/sdxl-base-ios"
COMPILED_SRC="${ROOT_SRC}/Resources"
DST_ROOT="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/BundledResources"
DST="${DST_ROOT}/sdxl/768/Resources"

rm -rf "$DST_ROOT"
mkdir -p "$DST"

for model_dir in \
  TextEncoder.mlmodelc \
  TextEncoder2.mlmodelc \
  UnetChunk1.mlmodelc \
  UnetChunk2.mlmodelc \
  VAEDecoder.mlmodelc
do
  SRC_PATH="${COMPILED_SRC}/${model_dir}"
  if [ ! -d "$SRC_PATH" ]; then
    echo "Missing compiled bundled model: $SRC_PATH" >&2
    exit 1
  fi
  if [ ! -f "${SRC_PATH}/weights/weight.bin" ]; then
    echo "Missing compiled model weights: ${SRC_PATH}/weights/weight.bin" >&2
    exit 1
  fi
  /usr/bin/ditto --noextattr --noqtn "$SRC_PATH" "$DST/$model_dir"
done

for token_file in vocab.json merges.txt; do
  SRC_PATH="${COMPILED_SRC}/${token_file}"
  if [ ! -f "$SRC_PATH" ]; then
    echo "Missing tokenizer file: $SRC_PATH" >&2
    exit 1
  fi
  /usr/bin/ditto --noextattr --noqtn "$SRC_PATH" "$DST/$token_file"
done

APP_ROOT="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"

/usr/bin/find "$APP_ROOT" -name ".DS_Store" -delete
/usr/bin/xattr -cr "$DST_ROOT" 2>/dev/null || true
/usr/bin/find "$APP_ROOT" -exec /usr/bin/xattr -d com.apple.FinderInfo {} \; 2>/dev/null || true
/usr/bin/find "$APP_ROOT" -exec /usr/bin/xattr -d 'com.apple.fileprovider.fpfs#P' {} \; 2>/dev/null || true
