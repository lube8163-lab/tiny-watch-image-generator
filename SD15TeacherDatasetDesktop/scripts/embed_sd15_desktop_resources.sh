#!/bin/sh
set -eu

DST_ROOT="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/BundledResources"
rm -rf "$DST_ROOT"

copy_sd15_resources() {
  local resolution="$1"
  local compiled_src="${SRCROOT}/artifacts/ios-models/sd15/${resolution}/Resources"
  local dst="${DST_ROOT}/sd15/${resolution}/Resources"

  if [ ! -d "$compiled_src" ]; then
    return 0
  fi

  mkdir -p "$dst"

  for model_dir in TextEncoder.mlmodelc Unet.mlmodelc VAEDecoder.mlmodelc
  do
    local src_path="${compiled_src}/${model_dir}"
    if [ ! -d "$src_path" ]; then
      echo "Missing compiled bundled model: $src_path" >&2
      exit 1
    fi
    if [ ! -f "${src_path}/weights/weight.bin" ]; then
      echo "Missing compiled model weights: ${src_path}/weights/weight.bin" >&2
      exit 1
    fi
    /usr/bin/ditto --noextattr --noqtn "$src_path" "$dst/$model_dir"
  done

  for token_file in vocab.json merges.txt
  do
    local src_path="${compiled_src}/${token_file}"
    if [ ! -f "$src_path" ]; then
      echo "Missing tokenizer file: $src_path" >&2
      exit 1
    fi
    /usr/bin/ditto --noextattr --noqtn "$src_path" "$dst/$token_file"
  done
}

copy_sd15_resources "512"
copy_sd15_resources "256"

APP_ROOT="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"

if [ ! -d "$DST_ROOT" ]; then
  echo "No SD 1.5 desktop resources were copied. Expected at least one of:" >&2
  echo "  ${SRCROOT}/artifacts/ios-models/sd15/512/Resources" >&2
  echo "  ${SRCROOT}/artifacts/ios-models/sd15/256/Resources" >&2
  exit 1
fi

echo "Embedded SD 1.5 desktop resources into ${APP_ROOT}/BundledResources"
