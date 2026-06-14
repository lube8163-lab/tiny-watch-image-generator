#!/usr/bin/env bash
set -euo pipefail

cd /workspace/tiny-image-model

while pgrep -f run_runpod_deep_experiments.sh >/dev/null; do
  sleep 10
done

source /tmp/tiny-image-model-train-venv/bin/activate

COMMON=(
  --teacher-root datasets/sdxl_cloud_teacher_watch46_v3_mixed_procedural_low6_balanced24
  --image-size 128
  --latent 48
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

run_train tiny_train_watch46_h1536_lowfreq_smooth64_l48_128 \
  --hidden 1536 --hidden-layers 2 --steps 30000 --lr 0.002 \
  --coord-frequencies 1,2,4 \
  --target-downsample-size 64 --target-blur-radius 0.35

run_train tiny_train_watch46_h1536_lowfreq_smooth64_tv003_l48_128 \
  --hidden 1536 --hidden-layers 2 --steps 30000 --lr 0.002 \
  --coord-frequencies 1,2,4 \
  --target-downsample-size 64 --target-blur-radius 0.35 \
  --smoothness-loss-weight 0.03 --smoothness-step-pixels 1

run_train tiny_train_watch46_h1536_allfreq_smooth64_tv003_l48_128 \
  --hidden 1536 --hidden-layers 2 --steps 30000 --lr 0.002 \
  --coord-frequencies 1,2,4,8 \
  --target-downsample-size 64 --target-blur-radius 0.35 \
  --smoothness-loss-weight 0.03 --smoothness-step-pixels 1

run_train tiny_train_watch46_deep3_h1024_lowfreq_smooth64_tv003_l48_128 \
  --hidden 1024 --hidden-layers 3 --steps 40000 --lr 0.0015 \
  --coord-frequencies 1,2,4 \
  --target-downsample-size 64 --target-blur-radius 0.35 \
  --smoothness-loss-weight 0.03 --smoothness-step-pixels 1

run_train tiny_train_watch46_deep3_h1280_lowfreq_smooth64_tv003_l48_128 \
  --hidden 1280 --hidden-layers 3 --steps 40000 --lr 0.0012 \
  --coord-frequencies 1,2,4 \
  --target-downsample-size 64 --target-blur-radius 0.35 \
  --smoothness-loss-weight 0.03 --smoothness-step-pixels 1
