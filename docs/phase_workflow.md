# Phase Workflow

## Setup

```sh
bash tools/bootstrap_research_env.sh
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
```

If a model requires Hugging Face auth:

```sh
huggingface-cli login
```

## Phase 1: Mac Reference Generation

Download only the necessary files first:

```sh
python3 tools/phase0_download.py --candidate segmind_tiny_sd
```

Generate a baseline image with the default LCM candidate:

```sh
python3 tools/phase1_generate.py \
  --candidate segmind_tiny_sd \
  --local-files-only \
  --prompt "a small watercolor landscape, crisp details" \
  --seed 7 \
  --width 256 \
  --height 256 \
  --steps 20
```

Inspect component parameter counts:

```sh
python3 tools/phase1_inspect.py --candidate segmind_tiny_sd --local-files-only
```

## Phase 2: Export Decoder First

Export only the VAE decoder at 64x64 output size:

```sh
python3 tools/phase2_export_vae_decoder.py \
  --candidate segmind_tiny_sd \
  --local-files-only \
  --output-width 64 \
  --output-height 64 \
  --drop-mid-attention \
  --out dist/segmind_tiny_sd/vae_decoder_64x64_noattn.mlpackage
```

Compress it:

```sh
python3 tools/coreml_quantize.py \
  dist/segmind_tiny_sd/vae_decoder_64x64_noattn.mlpackage \
  --mode palettize4 \
  --out dist/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.mlpackage
```

Smoke-test the decoder on Mac:

```sh
python3 tools/phase2_smoke_coreml.py \
  dist/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.mlpackage \
  --latent-height 8 \
  --latent-width 8
```

## Next Gate

Only after the decoder can be exported, compressed, and run should we start exporting the denoiser. This avoids mixing Core ML conversion problems with diffusion scheduler and text-conditioning problems.

## Phase 3: UNet Probe

Export a fixed-shape LCM UNet for 64x64 output scale:

```sh
python3 tools/phase3_export_unet.py \
  --candidate lcm_dreamshaper_v7 \
  --local-files-only \
  --latent-height 8 \
  --latent-width 8 \
  --attention-processor eager \
  --out dist/lcm_dreamshaper_v7/unet_8x8.mlpackage
```

Compress it:

```sh
python3 tools/coreml_quantize.py \
  dist/lcm_dreamshaper_v7/unet_8x8.mlpackage \
  --mode palettize4 \
  --out dist/lcm_dreamshaper_v7/unet_8x8_4bit.mlpackage
```

Smoke-test it:

```sh
python3 tools/phase3_smoke_unet_coreml.py \
  dist/lcm_dreamshaper_v7/unet_8x8_4bit.mlpackage \
  --latent-height 8 \
  --latent-width 8 \
  --manifest reports/phase3/lcm_dreamshaper_v7/unet_8x8_4bit_smoke.json
```

Current result:

- FP16 Core ML UNet 8x8: about 1.6GB.
- 4-bit palettized UNet 8x8: about 411MB.
- Mac Core ML smoke test passes for both.
- This is still likely too large for comfortable Apple Watch deployment, but it proves the conversion path.

## Notes

- `HF_HUB_DISABLE_XET=1` avoids stalls seen with unauthenticated Xet downloads.
- `--drop-mid-attention` replaces the VAE decoder mid-block attention with identity for the first Core ML export. This is an engineering probe, not the final quality path.
- `phase3_export_unet.py` patches a Core ML Tools Torch frontend edge case where `int` conversion of length-1 ndarray constants fails.
- `--attention-processor eager` avoids PyTorch 2 fused attention during tracing.
- Direct 64x64 generation with `segmind/tiny-sd` was low quality. The better Watch path may be generating larger latents and downscaling, or using LCM after a minimal download succeeds.
