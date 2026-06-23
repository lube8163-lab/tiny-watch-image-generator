# Watch Notes

This folder collects Apple Watch runtime notes and smoke-test references.

## Current Runtime

- [txt2img_plan.md](txt2img_plan.md): overall direction, constraints, and current LCM256 baseline.
- [pipeline_smoke_current.md](pipeline_smoke_current.md): current `WatchPipelineSmokeApp` setup, including the 256px smoke branch bundle shape and device checklist.
- [text_encoder_smoke.md](text_encoder_smoke.md): separated text encoder probe notes.
- [pipeline_quality_notes.md](pipeline_quality_notes.md): current quality observations and postprocess/SR notes.
- [future_quality_breakthroughs.md](future_quality_breakthroughs.md): larger training/distillation tracks needed for a major quality jump.

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
