# Apple Watch txt2img Plan

Goal: build a technical demo that accepts text on Apple Watch and produces a tiny image locally.

This repo now has two tracks:

1. `TinyWatchGenerator`: toy txt2img API, pure Swift, useful for UI and watchOS integration tests.
2. `Core ML distillation/quantization`: the future path for a real-ish model.

## Hard Constraints

- Apple Watch has a small battery and a much tighter thermal envelope than iPhone or Mac.
- App bundle size is not the main problem. Runtime memory and intermediate activation buffers are.
- A 350M parameter model can be small on disk at 4-bit, but it can still allocate much more at runtime.
- A normal Stable Diffusion stack is too large because it includes a text encoder, denoiser, scheduler, and VAE.

## Target Budget

For a first serious demo, target this instead of a full SD model:

| Component | Target | Notes |
| --- | ---: | --- |
| Text encoder | 5M-30M params | Prefer frozen tiny CLIP/T5 or prompt hashing for first version |
| Denoiser | 80M-250M params | Latent diffusion UNet or DiT distilled to 4-8 steps |
| Decoder | 1M-15M params | Use a tiny autoencoder decoder, not full SD VAE |
| Resolution | 64x64 first | Upscale in UI if needed |
| Steps | 1-8 | More steps will feel bad on watch |
| Weight format | 4-bit or 6-bit where possible | Use 8-bit fallback for ops Core ML cannot compress well |
| Runtime | Core ML `mlprogram` | Pure Swift matmul is only for toy models |

## Practical Model Strategy

Start from an existing small latent diffusion model, but do not try to deploy the full pipeline unchanged.

Recommended path:

1. Pick a small SD-compatible base for experimentation on Mac.
2. Replace the full VAE decoder with a tiny decoder.
3. Distill the denoiser to very few steps.
4. Freeze or replace the text encoder.
5. Export each component as separate Core ML packages.
6. Quantize separately and profile runtime memory on watchOS.

The most important split is:

- Keep the text encoder tiny or compute text embeddings outside the watch demo at first.
- Make the denoiser the only large model.
- Decode at 64x64 or 96x96, then display with nearest/bilinear upscale.

## Quantization Ladder

Use this sequence so failures are diagnosable:

1. FP16 Core ML conversion.
2. 8-bit linear weight quantization.
3. 6-bit or 4-bit palettization for large linear/conv weights.
4. Activation quantization only after the model works and only if supported by the deployment target.
5. Mixed precision: leave sensitive layers FP16, quantize bulk layers aggressively.

Do not start with the most compressed model. Get one watchOS-runable pipeline first.

## Repo Milestones

### M0: Toy prompt path

Current state. `TinyImageGenerator.generate(prompt:seed:)` runs on Swift/watchOS and proves the UI path.

### M1: Core ML packaging scaffold

Add conversion scripts that can:

- load a PyTorch module,
- trace/export a fixed-shape component,
- convert to Core ML,
- apply post-training compression,
- emit a manifest with size and component metadata.

### M2: Tiny decoder

Deploy only a latent-to-RGB decoder first. Feed random or saved latents from Swift.

### M3: Denoiser

Deploy a very small denoiser and run 1-4 inference steps.

### M4: Text conditioning

Add tiny text encoder or a learned prompt embedding table. For the Apple Watch demo, a small vocabulary is acceptable.

### M5: End-to-end watch demo

Use this UI flow:

1. prompt input or preset prompt picker,
2. seed shuffle,
3. generate button,
4. progressive preview,
5. final 64x64/96x96 image upscaled for display.

## Open Risk

The 350M-parameter target is plausible on disk with 4-bit weights, but not guaranteed to run acceptably on Apple Watch because intermediate activations and Core ML execution planning can dominate memory. The real go/no-go check is a watchOS Instruments profile, not the file size.
