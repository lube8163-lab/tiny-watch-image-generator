# WatchPipelineSmokeApp Quality Notes

Updated: 2026-06-20

## Current Best Default

Keep the default run as:

- Prompt key: `cat_mascot`
- Seed: `1`
- Guidance: `6`
- Preview: `Sharp x2`

The 6-bit 16-part Core ML shortlist still ranks this as one of the strongest
general default images. It reads as a small white cat mascot and survives the
quantized/chunked path.

Reference:

`reports/watch_pipeline_reference/final_default_cat_mascot_s1_g6_coreml_16p/coreml_preview_sharp2x.png`

## RunPod Sweep

RunPod RTX 4090 was used for a broad Watch-style PyTorch sweep:

- Presets: 12 cat-focused presets
- Seeds: `0...31`
- Guidance: `6`, `8`
- Size: `128`
- Steps: `4`
- Total images: `768`

Artifacts:

- `reports/runpod_lcm128_watch_plus_gpu_sweeps/cat12_seed0_31_g6_g8_raw/contact_sheet.png`
- `reports/runpod_lcm128_watch_plus_gpu_sweeps/cat12_seed0_31_g6_g8_raw/manifest.json`

Guidance `6` generally looked more stable than `8` for Watch defaults. Guidance
`8` often increased contrast, but white cats blew out more easily and face crops
had more extreme failures.

## Core ML Shortlist

Promising RunPod candidates were regenerated through the local 6-bit 16-part
Core ML chunks and 128px decoder.

Artifacts:

- `reports/watch_pipeline_reference/lcm128_coreml_candidate_shortlist_g6_raw/contact_sheet.png`
- `reports/watch_pipeline_reference/lcm128_coreml_candidate_shortlist_g6_raw/manifest.json`
- `reports/watch_pipeline_reference/lcm128_coreml_candidate_shortlist_g6_preview_comparison/contact_sheet.png`

Good visual candidates after Core ML quantization/chunking:

- `cat_mascot:1`
- `cat_mascot:2`
- `lucky_cat:1`
- `cat_sticker:16`
- `white_mascot:14`
- `tabby_icon:30`
- `orange_cat:30`
- `cat_logo:3`

The Watch app curated seed map was updated to use these stronger per-preset
seeds while keeping the default unchanged.

The Watch UI now exposes seed `0...31` plus the older fixed seeds so these
shortlist candidates can be checked directly on device. The settings area also
shows the current preset key, resolved seed, curated seed, and expected run id
before generation.

The app also has one-tap candidate buttons for the current best Core ML
shortlist:

- `Mascot 1`: `cat_mascot:1 g6`
- `Mascot 2`: `cat_mascot:2 g6`
- `Lucky 1`: `lucky_cat:1 g6`
- `Sticker 16`: `cat_sticker:16 g6`
- `White 14`: `white_mascot:14 g6`
- `Tabby 30`: `tabby_icon:30 g6`
- `Orange 30`: `orange_cat:30 g6`
- `Logo 3`: `cat_logo:3 g6`

## Resolution / Model Direction

RunPod resolution probes confirm the broad trend:

- `64px` is too compressed for reliable cat structure.
- `128px` is the current practical sweet spot.
- Larger outputs can look better on GPU, but a Watch deployment would require
  `24x24` or `32x32` latents, larger intermediate tensors, larger chunk outputs,
  and much longer streamed inference.

Artifacts:

- `reports/runpod_lcm_resolution_probe_cat_grid/contact_sheet.png`
- `reports/runpod_lcm_resolution_probe_cat_grid/contact_sheet_downsampled64.png`

Trying a completely different text-to-image family is useful only if it can
still satisfy the Watch constraints:

- 4-ish denoising steps
- fixed 128px output
- Core ML exportable UNet/VAE
- chunkable without holding multiple heavy models
- acceptable CPU-only runtime

SDXL/Turbo-class models may produce nicer GPU images, but they are much less
aligned with the current watchOS memory and bundle constraints. The near-term
quality path is therefore:

1. Keep LCM128 6-bit 16-part streaming.
2. Promote strong preset/seed pairs.
3. Use non-neural `Sharp x2` as the default display path.
4. Consider a tiny fixed-shape SR CNN only after it clearly beats `Sharp x2`.

## Upscaling / SR

The current non-neural `Sharp x2` preview is a good first choice:

- Bicubic Catmull-Rom 2x
- Unsharp amount `0.45`
- No extra model memory
- Runs inside the current Watch app

SwinIR x2 was compared on top candidates:

`reports/watch_pipeline_reference/lcm128_coreml_candidate_swinir_comparison/contact_sheet.png`

SwinIR improves edges slightly, but it does not recover missing semantics. It
also adds a second neural model and conversion/runtime risk. The local 128px
Core ML conversion probe currently fails in coremltools on dynamic `int` ops
inside SwinIR attention/window logic, so shipping SwinIR on watchOS is not a
quick drop-in.

Practical next SR path, if needed:

- Prefer a tiny CNN/ESPCN-style x2 model trained specifically on these 128px
  LCM outputs, not transformer SwinIR.
- Keep the model fixed-shape `128 -> 256`.
- Compare against `Sharp x2` and only include it if it visibly beats the
  non-neural preview under watch memory/time limits.

## Tiny SR Probe

A small RunPod smoke test trained a lightweight PixelShuffle CNN on LCM
`128 -> 256` pairs:

- Presets: `cat_mascot`, `lucky_cat`, `white_mascot`
- Seeds: `0...7`
- Pairs: `24`
- Training: `500` steps, channels `40`, blocks `2`

Artifacts:

- `reports/runpod_lcm_sr_probe_smoke/contact_sheet.png`
- `reports/runpod_lcm_sr_probe_smoke/manifest.json`

Result:

- Tiny SR validation PSNR: `13.34dB`
- Non-neural `Sharp x2` validation PSNR: `34.62dB`

The visual comparison also favors `Sharp x2`; the tiny CNN creates cyan/blurred
averages and does not preserve details. This is enough evidence to avoid adding
a neural SR model to WatchPipelineSmokeApp for now. A future SR attempt would
need either a much better architecture/training setup or a very specific
postprocess goal, and should beat the current `Sharp x2` sheet before any watchOS
integration work.
