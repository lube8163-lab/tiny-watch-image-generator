# Future Quality Breakthroughs

Updated: 2026-06-23

The adopted Watch baseline is `LCM256 6b`: streamed 16-part UNet, 256px 4-bit
decoder, transient CLIP text encoder, 4 LCM steps, CPU-only Core ML. It now
runs on device with roughly `140MB` observed peak memory. Further resolution
increases may still work, but are unlikely to produce a dramatic quality jump
relative to their runtime and memory cost.

This note lists the larger steps required for a real quality breakthrough.
These are research/training tracks, not quick watchOS integration tweaks.

## Target

Aim for a Watch-specific generator rather than a generic Stable Diffusion stack:

- Native `256x256` output.
- 1-4 denoising steps.
- Short prompt focus: a few words or a compact phrase.
- Strong object/scene composition for common prompts.
- Core ML fixed-shape export with CPU-only runtime.
- Peak watchOS memory not materially above the current 256px path.

## Training Track

1. Build a small prompt/image corpus for the intended product domain.
   Keep prompts short and concrete: `tabby cat`, `dog logo`, `snowy mountain`,
   `blue bird`, `astronaut riding a horse`.
2. Generate teacher images offline at higher quality with a stronger model.
   Save the exact prompt, seed, style suffix, and any negative prompt used.
3. Train or fine-tune a small LCM-compatible student for native `256x256`.
   Prefer a model that already supports few-step generation over adapting a
   normal many-step diffusion model late in the process.
4. Distill the text-conditioning behavior for short prompts.
   The goal is not full SDXL-style prompt grammar; it is reliable semantics for
   a compact vocabulary and common two-to-five-word phrases.
5. Train or replace the decoder only if it improves actual Watch outputs.
   A better tiny decoder could matter more than non-neural postprocess once the
   latent model is stable.

## Evaluation Gates

Every candidate should pass these gates before watchOS integration work:

1. Offline contact sheets across a fixed prompt/seed suite.
2. Comparison against the current `LCM256 6b` branch at the same prompts.
3. Core ML conversion at fixed shapes: text encoder, UNet, decoder.
4. 6-bit or better compression of the denoiser without obvious semantic loss.
5. Chunked Core ML prediction parity against the unsplit model.
6. watchOS bundle verification with only the intended model family included.
7. Physical device run with Xcode logs for memory, step timings, decoder timing,
   preview timing, and failure modes.

## Practical Stop Rules

Do not move a candidate into the Watch app unless it clearly beats the current
256px baseline in at least one of these ways:

- noticeably better object structure,
- better short-prompt following,
- lower runtime at comparable quality,
- lower memory at comparable quality,
- or a much simpler product path.

Small edge sharpening, color changes, or seed-specific wins are not enough. The
current pipeline already benefits more from seed reroll than from most small
postprocess changes.

## Likely High-Leverage Experiments

- A Watch-domain LoRA or small fine-tune focused on icons, mascots, animals,
  simple scenes, and centered compositions.
- A compact text encoder distilled specifically for short prompts, replacing
  the full CLIP probe once quality parity is confirmed.
- A smaller UNet or DiT-style denoiser trained directly for `32x32` latents and
  1-4 steps, rather than compressing a larger generic UNet.
- A tiny decoder trained for the student's latent distribution, evaluated
  against the current 4-bit VAE decoder.
- Quantization-aware training for the denoiser if post-training 6-bit
  palettization becomes the main quality bottleneck.
