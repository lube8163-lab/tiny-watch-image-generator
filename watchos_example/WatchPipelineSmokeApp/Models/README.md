Drop compiled Core ML pipeline smoke models here.

Expected local files:
- `unet_sd_16x16_4bit.mlmodelc` is reused from `WatchStressTestApp/Models`.
- `vae_decoder_128x128_noattn_4bit.mlmodelc` can be compiled from `dist/segmind_tiny_sd/vae_decoder_128x128_noattn_4bit.mlpackage`.
- `vae_decoder_128x128_noattn.mlmodelc` can be compiled from `dist/segmind_tiny_sd/vae_decoder_128x128_noattn.mlpackage`.

The compiled model folders are intentionally ignored by Git.

The prompt text encoder assets live one level up in `TextEncoderAssets`:

- `clip_text_encoder_77.mlmodelc`
- `clip_vocab.json`
- `clip_merges.txt`

`WatchPipelineSmokeApp` loads the text encoder transiently, creates one
`hidden_states` embedding for the typed prompt, releases the model, purges the
Core ML cache, and then starts streamed LCM generation.
