#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/tiny-image-model-cloud.tar.gz}"

cd "$ROOT"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
  README.md \
  requirements/cloud_sdxl.txt \
  configs/sdxl_tiny_teacher_prompts.json \
  configs/sdxl_tiny_teacher_prompts_v2.json \
  configs/sdxl_tiny_teacher_prompts_v3_watch.json \
  configs/sdxl_tiny_teacher_prompts_v4_watch_freeprompt.json \
  configs/sdxl_tiny_teacher_prompts_v5_watch_freeprompt.json \
  configs/sdxl_tiny_teacher_prompts_v6_watch_freeprompt.json \
  configs/sdxl_tiny_teacher_prompts_v7_focus_watch_freeprompt.json \
  configs/lcm64_watch_presets.json \
  configs/lcm64_cat_quality_presets.json \
  configs/lcm128_watch_plus_presets.json \
  configs/prompt_eval_suite.json \
  datasets/sdxl_cloud_teacher_watch46_v3_problem3_curated \
  tools/prompt_normalization.py \
  tools/research_common.py \
  tools/generate_sdxl_teacher_dataset.py \
  tools/validate_teacher_dataset.py \
  tools/filter_teacher_dataset.py \
  tools/make_teacher_category_contact_sheet.py \
  tools/build_watch_v4_prompt_config.py \
  tools/build_watch_v5_prompt_config.py \
  tools/build_watch_v6_prompt_config.py \
  tools/build_watch_v7_focus_prompt_config.py \
  tools/train_tiny_coordinate_mlp.py \
  tools/cloud_setup_sdxl.sh \
  tools/cloud_generate_sdxl_fixed16.sh \
  tools/cloud_generate_sdxl_expanded_v2.sh \
  tools/cloud_generate_sdxl_watch_v3.sh \
  tools/cloud_generate_sdxl_watch_v4.sh \
  tools/cloud_generate_sdxl_watch_v5.sh \
  tools/cloud_generate_sdxl_watch_v6.sh \
  tools/cloud_generate_sdxl_watch_v7_focus.sh \
  tools/cloud_lcm_resolution_probe.py \
  tools/cloud_lcm_watch_style_probe.py \
  tools/cloud_train_lcm_sr_probe.py \
  tools/make_watch_preview_comparison.py \
  tools/train_watch_v4_mlp.sh \
  tools/run_cloud_watch_v5_pipeline.sh \
  tools/run_cloud_watch_v6_pipeline.sh \
  tools/run_cloud_watch_v7_pipeline.sh \
  docs/cloud_gpu_teacher_dataset.md \
  docs/model_improvement_plan.md

echo "$OUT"
