# WatchStressTestApp Models

Drop compiled `.mlmodelc` directories here to bundle them into the stress test app.

Example:

```sh
xcrun coremlcompiler compile \
  dist/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.mlpackage \
  watchos_example/WatchStressTestApp/Models
```

The app scans the bundle recursively for `.mlmodelc` directories and logs load/prediction results to both the on-screen log and the Xcode console with the `[WatchStress]` prefix.
