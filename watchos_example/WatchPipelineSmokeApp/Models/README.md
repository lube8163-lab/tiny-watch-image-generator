# WatchPipelineSmokeApp Models

Drop compiled Core ML pipeline smoke models here. Compiled `.mlmodelc`
directories are intentionally ignored by Git.

## Current LCM256 Baseline

Expected local files for the adopted 256px path:

- `lcm_unet_32x32_6bit_16p_part1.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part2.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part3.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part4.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part5.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part6.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part7.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part8.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part9.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part10.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part11.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part12.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part13.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part14.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part15.mlmodelc`
- `lcm_unet_32x32_6bit_16p_part16.mlmodelc`
- `vae_decoder_256x256_noattn_4bit.mlmodelc`

The text encoder assets live one level up in `TextEncoderAssets`:

- `clip_text_encoder_77.mlmodelc`
- `clip_vocab.json`
- `clip_merges.txt`

`WatchPipelineSmokeApp` loads the text encoder transiently, creates one
`hidden_states` embedding for the typed prompt, releases the model, purges the
Core ML cache, and then starts streamed LCM generation.

## Older Probe Files

Older 64px, 128px, and 192px compiled models may exist locally for comparison,
but they should not be bundled into the current LCM256 app build. Use
`tools/verify_watch_pipeline_smoke.py --family lcm256` against the built app to
catch stale model families.
