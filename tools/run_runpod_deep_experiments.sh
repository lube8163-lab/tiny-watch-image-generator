#!/usr/bin/env bash
set -euo pipefail

cd /workspace/tiny-image-model

while pgrep -f run_smooth_experiments.sh >/dev/null; do
  sleep 10
done

source /tmp/tiny-image-model-train-venv/bin/activate

COMMON=(
  --teacher-root datasets/sdxl_cloud_teacher_watch46_v3_mixed_procedural_low6_balanced24
  --image-size 128
  --latent 48
  --coord-frequencies 1,2,4,8
  --prompt-encoder compositional_v1
  --batch-size 8192
  --device cuda
  --progress-every 5000
  --preview-prompts-file configs/prompt_eval_suite.json
)

run_train() {
  local name="$1"
  shift
  echo "== ${name} $(date --iso-8601=seconds) =="
  python tools/train_tiny_coordinate_mlp.py \
    "${COMMON[@]}" \
    "$@" \
    --out-dir "out/${name}" \
    --out-json "out/${name}/tiny_weights.json" \
    --out-swift "out/${name}/TinyWeights.swift" \
    --out-bin "out/${name}/TinyWeights.bin"
}

run_train tiny_train_watch46_deep3_h1024_smooth64_blur035_l48_128 \
  --hidden 1024 --hidden-layers 3 --steps 40000 --lr 0.0015 \
  --target-downsample-size 64 --target-blur-radius 0.35

run_train tiny_train_watch46_deep3_h1024_smooth80_blur025_l48_128 \
  --hidden 1024 --hidden-layers 3 --steps 40000 --lr 0.0015 \
  --target-downsample-size 80 --target-blur-radius 0.25

run_train tiny_train_watch46_deep3_h1280_smooth64_blur035_l48_128 \
  --hidden 1280 --hidden-layers 3 --steps 40000 --lr 0.0012 \
  --target-downsample-size 64 --target-blur-radius 0.35

run_train tiny_train_watch46_deep3_h1280_smooth80_blur025_l48_128 \
  --hidden 1280 --hidden-layers 3 --steps 40000 --lr 0.0012 \
  --target-downsample-size 80 --target-blur-radius 0.25

run_train tiny_train_watch46_deep3_h1536_smooth64_blur035_l48_128 \
  --hidden 1536 --hidden-layers 3 --steps 40000 --lr 0.001 \
  --target-downsample-size 64 --target-blur-radius 0.35

run_train tiny_train_watch46_deep3_h1536_smooth80_blur025_l48_128 \
  --hidden 1536 --hidden-layers 3 --steps 40000 --lr 0.001 \
  --target-downsample-size 80 --target-blur-radius 0.25
