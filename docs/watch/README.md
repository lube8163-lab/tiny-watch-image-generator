# Watch Notes

This folder collects Apple Watch runtime notes, stress probes, and LCM256
quality-evaluation references.

## Track Index

- Original MLP demo: `TinyImageWatchApp`, `TinyWatchGenerator`, and the older
  Swift contact-sheet baseline.
- Core ML stress/probe path: `WatchStressTestApp` and
  `WatchTextEncoderSmokeApp`.
- Diffusion baseline: `WatchPipelineSmokeApp` with LCM256 6-bit streamed UNet,
  transient text encoder, and 256px decoder.

For the repository-wide map, see [../project_tracks.md](../project_tracks.md).

## Current Diffusion Runtime

- [txt2img_plan.md](txt2img_plan.md): overall direction, constraints, and current LCM256 baseline.
- [pipeline_smoke_current.md](pipeline_smoke_current.md): current `WatchPipelineSmokeApp` setup, including the 256px smoke branch bundle shape and device checklist.
- [pipeline_quality_notes.md](pipeline_quality_notes.md): current quality observations and postprocess/SR notes.
- [watch_256_baseline_summary_2026-06-23.md](watch_256_baseline_summary_2026-06-23.md): adopted 256px baseline, device evidence, and Mac/watch evaluation split.
- [mac_quality_eval.md](mac_quality_eval.md): Mac-side LCM256 quality sweep script, prompt suite, and report workflow.
- [mac_quality_eval_seed1_summary_2026-06-24.md](mac_quality_eval_seed1_summary_2026-06-24.md): 74-prompt seed-1 Mac eval summary.
- [mac_quality_eval_full_summary_2026-06-24.md](mac_quality_eval_full_summary_2026-06-24.md): 296-image full Mac eval summary.
- [future_quality_breakthroughs.md](future_quality_breakthroughs.md): larger training/distillation tracks needed for a major quality jump.

## Stress / Probe Notes

- [text_encoder_smoke.md](text_encoder_smoke.md): separated text encoder probe notes retained for diagnostics.

## Earlier Watch Demo

- [device_build.md](device_build.md): watchOS build and installation notes.
- [eval_baseline.md](eval_baseline.md): pure Swift tiny-generator evaluation baseline.

## Local Artifacts

Large generated outputs remain outside Git:

- `dist/`: converted or compressed model packages.
- `reports/`: contact sheets, reference images, and verifier reference outputs.
- `datasets/`: teacher and mined datasets.
- `models/`: downloaded or intermediate model files.

Compiled `.mlmodelc` directories are also ignored. Keep only small manifests,
prompt assets, tokenizer metadata, source files, and docs in normal commits.
