# Tools

The tools are kept flat so older commands and imports continue to work. Use this
map to find the right entry point.

## Diffusion / LCM256 Evaluation

- `audit_watch_prompt_coverage.py`: prompt alias and UI coverage audit.
- `verify_watch_pipeline_smoke.py`: bundle verifier for `WatchPipelineSmokeApp`.
- `watch_lcm256_quality_eval.py`: Mac-side LCM256 free-prompt quality sweeps,
  contact sheets, and Markdown summaries.

## Original MLP Evaluation

- `make_watch_eval_contact_sheet.py`: contact sheets for fixed prompt/seed eval.
- `make_watch_postprocess_compare.py`: raw vs watch postprocess comparison.
- `make_watch_preview_comparison.py`: non-neural preview comparison.

## Watch Prompt And Teacher Data

- `build_watch_v*_prompt_config.py`: prompt config generation for each training iteration.
- `cloud_generate_sdxl_watch_*.sh`: cloud SDXL teacher image generation.
- `filter_teacher_dataset.py`: teacher dataset filtering.
- `generate_sdxl_teacher_dataset.py`: teacher dataset generation helpers.

## Core ML Watch Pipeline

- `phase4_export_watch_prompt_assets.py`: prompt/scheduler asset export.
- `chunk_coreml_model.py`: split large Core ML packages into streamed chunks.
- `export_clip_text_encoder_coreml.py`: CLIP text encoder export.
- `generate_lcm64_reference.py`: LCM reference generation.
- `sweep_lcm64_coreml.py`: LCM64 Core ML parameter sweeps.
- `cloud_lcm_resolution_probe.py`: resolution feasibility probes.
- `cloud_lcm_watch_style_probe.py`: watch-style LCM probes.
- `cloud_train_lcm_sr_probe.py`: small SR probe training.

## Original Tiny Generator

- `prompt_normalization.py`: shared prompt aliases and normalization.
- `train_tiny_coordinate_mlp.py`: pure Swift tiny generator training path.
- `generate_weights.py`: export tiny generator weights.
- `preview.py`: quick preview helpers.
