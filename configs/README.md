# Configs

## Watch App And Evaluation

- `watch_txt2img.json`: app-facing prompt preset config for the pure Swift demo.
- `prompt_eval_suite.json`: fixed evaluation prompt groups.
- `sd_watch_presets.json`: earlier Stable Diffusion watch preset set.

## LCM Watch Pipeline

- `lcm64_watch_presets.json`: LCM64 prompt set.
- `lcm64_cat_quality_presets.json`: narrow cat-quality sweep prompts.
- `lcm128_watch_plus_presets.json`: current LCM128 WatchPipeline prompt asset source.

## SDXL Teacher Data

- `sdxl_tiny_teacher_prompts*.json`: teacher prompt sets by iteration.
- `mining_prompts.json`: prompt list for dataset mining.
- `model_candidates.json`: candidate model metadata.

Generated datasets, model packages, reports, and run outputs should stay under
ignored directories such as `datasets/`, `dist/`, `reports/`, and `out/`.
