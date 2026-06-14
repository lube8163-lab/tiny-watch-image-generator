# Model Selection

The current implementation keeps the pure Swift toy generator as the device integration baseline. The real image-quality path should use existing distilled diffusion models, then split and compress them.

## First Target

Use `SimianLuo/LCM_Dreamshaper_v7` first.

Reasons:

- It is designed for low step counts.
- It works with Diffusers.
- It gives a practical quality baseline before we start destroying the model with quantization.
- It lets us test a realistic Stable Diffusion style component split: text encoder, denoiser, VAE decoder.

This is not expected to fit Apple Watch directly. It is the Mac-side reference model for Phase 1 and an export source for Phase 2.

## Comparison Targets

Use `segmind/tiny-sd` and `segmind/small-sd` as comparison targets for size and quality. They may be smaller, but they are not inherently as attractive as LCM for Watch because they usually need more denoising steps.

## Decision Rule

For each model, record:

- generated image quality at 256x256,
- generated image quality at 64x64 or 96x96,
- component parameter counts,
- VAE decoder Core ML size before and after compression,
- denoiser Core ML size before and after compression,
- Watch runtime memory and latency once a component can be deployed.

The first model that produces acceptable 64x64 results with a deployable decoder becomes the main branch for denoiser work.
