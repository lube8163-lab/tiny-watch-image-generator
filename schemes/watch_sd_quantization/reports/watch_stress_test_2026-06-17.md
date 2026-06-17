# Watch Stress Test Report 2026-06-17

## Purpose

既存の `TinyImageWatchApp` を壊さず、別 target/scheme で Apple Watch 実機の Core ML 実行上限を測るためのストレステストを行った。

目的は、公開済み Stable Diffusion 系モデルを強く軽量化/量子化した場合に、Apple Watch 上でどこまで現実的に動かせるかを判断すること。

## Added App

- Xcode project: `watchos_example/TinyImageWatchApp.xcodeproj`
- Existing scheme: `TinyImageWatchApp`
- Added scheme: `WatchStressTestApp`
- Added target: `WatchStressTestApp`
- App source:
  - `watchos_example/WatchStressTestApp/WatchStressTestApp.swift`
  - `watchos_example/WatchStressTestApp/StressTestView.swift`
- Bundled model directory:
  - `watchos_example/WatchStressTestApp/Models`

`xcodebuild -list` で `TinyImageWatchApp` と `WatchStressTestApp` の両方が見える状態。

## Model Install

Core ML `.mlpackage` を stress app に入れるには、以下の helper を使う。

```sh
schemes/watch_sd_quantization/scripts/install_stress_model.sh path/to/model.mlpackage
```

実行すると `xcrun coremlcompiler compile` で `.mlmodelc` に変換し、`watchos_example/WatchStressTestApp/Models` に配置する。

今回入れたモデル:

```text
155M  watchos_example/WatchStressTestApp/Models/unet_sd_16x16_4bit.mlmodelc
 23M  watchos_example/WatchStressTestApp/Models/vae_decoder_64x64_noattn_4bit.mlmodelc
```

## Stress App Controls

- `Compute`: `CPU` or `All`
  - 現状は `CPU` を使う。
  - `All` は UNet で ANE compile 失敗とメモリ急増が出た。
- `Scan Models`: bundle 内の `.mlmodelc` を列挙。
- `+4 MB`, `+8 MB`, `+16 MB`, `+32 MB`: retained buffer を追加してメモリ余白を探る。
- `Fine Ladder`: 16 MB ずつ段階的に確保。
- `Aggressive Ladder`: 大きめの段階で確保。
- `Release Memory`: retained buffer を解放。
- `Load Only`: モデルを load してすぐ解放。
- `Load & Retain`: モデルを load して保持。
- `Release Models`: retained model を解放。
- `Predict Once`: bundle モデルを load して 1 回 predict。
- `Predict x4`: bundle モデルを load して各モデル 4 回 predict。
- `Retained x4`: retained model を各モデル 4 回 predict。
- `Pipeline 4+Decode`: retained UNet を 4 回、retained decoder を 1 回 predict。

ログは Xcode console とアプリ内 list に出る。Xcode console では `print` と `os.Logger` の両方に出しているため、同じ行が重複して見える。

## Device Measurements

### Memory-only

Fine ladder:

- 16 MB 刻みで `stable=272 MB` までは成功。
- `target=288 MB` の attempt 後にクラッシュ。
- 以前の aggressive ladder でも 300 MB 前後でクラッシュ。

解釈:

- 単純な retained buffer の安定域はおおむね 270 MB 前後。
- Core ML load/predict 用には、この数字をそのまま使わず、かなり余白を残す必要がある。

### Decoder-only

Model:

- `vae_decoder_64x64_noattn_4bit.mlmodelc`
- Size: about 23 MB
- Input: `latents`, shape `1x4x8x8`
- Output: `decoded`

実測:

- 初回 load: `4.325s`
- warm load: about `0.150s`
- predict: about `0.048s` to `0.066s`
- `Predict x4` と memory buffer 追加後の再実行も問題なし。

解釈:

- decoder は watchOS 実機上で十分軽い。
- VAE decoder 単体はボトルネックではない。

### UNet-only, Compute All

Model:

- `unet_sd_16x16_4bit.mlmodelc`
- Size: about 155 MB
- Inputs: `encoder_hidden_states`, `sample`, `timestep`
- Output: `noise_pred`

結果:

```text
MILCompilerForANE error: failed to compile ANE model using ANEF.
_ANECompiler : ANECCompile() FAILED
Espresso compiled without MPSGraph engine.
```

その後、メモリが 300 MB 付近まで急増してクラッシュ。

解釈:

- この UNet では `MLComputeUnits.all` は避ける。
- 現状の Watch 実機ルートは `CPU` 固定が妥当。

### UNet-only, CPU

実測:

- load-only: about `7.184s`
- predict once:
  - load: `7.143s`
  - predict: `0.830s`
  - total: `8.324s`
- observed memory peak: about `47.7 MB`
- `Predict x4` 成功。
- `+32 MB` 後の `Predict x4` も成功。

解釈:

- CPU-only なら 155 MB 4bit UNet は load/predict できる。
- 初回 load は重いが、predict 自体は想定より軽い。

### UNet + Decoder, CPU, Retained

Models:

- `unet_sd_16x16_4bit.mlmodelc`
- `vae_decoder_64x64_noattn_4bit.mlmodelc`

`Load & Retain`:

- UNet load: `6.455s`
- decoder load: `1.125s`
- total: `8.217s`

`Pipeline 4+Decode` first run:

- UNet #1: `1.223s`
- UNet #2: `0.324s`
- UNet #3: `0.187s`
- UNet #4: `0.161s`
- decoder: `0.171s`
- model_time: `2.064s`
- total: `2.163s`

After `+32 MB`, `Pipeline 4+Decode`:

- UNet #1: `0.283s`
- UNet #2: `0.166s`
- UNet #3: `0.161s`
- UNet #4: `0.159s`
- decoder: `0.219s`
- model_time: `0.989s`
- total: `1.065s`

Observed memory peak:

- about `61.2 MB`
- peak は推論中ではなく `Load & Retain` 時。

解釈:

- CPU-only, retained model, 4-step + decode は Apple Watch 実機で現実的。
- 生成時より load 時の一時ピークを重視するべき。
- 起動直後または生成前に model warmup/load を済ませて保持する設計が有力。

## Current Conclusion

当初の「direct quantized SD はほぼ無理かもしれない」という見立てから、かなり前進した。

現時点の有力条件:

- 128x128 output
- latent `16x16`
- UNet 4bit, CPU-only
- decoder 4bit, CPU-only
- text encoder は Watch 上で動かさない
- prompt は preset embedding または短い slot/preset 方式
- model は起動後に load and retain
- generation は UNet few-step + decoder

この条件なら、Apple Watch 実機で技術デモとして成立する可能性が高い。

## Important Limits

- 現在の stress app は zero-filled `MLMultiArray` を入力している。
- 実際の scheduler 更新、prompt embedding、latent 更新、画像表示はまだ入っていない。
- 4 step の品質は未検証。
- 長時間連続生成時の thermal/電池/Watchdog は未検証。
- `MLComputeUnits.all` は現状避ける。

## Recommended Next Test

次は stress test ではなく、別 target/scheme で最小 pipeline smoke test に進む。

最小要件:

1. `WatchPipelineSmokeApp` のような別 target/scheme を作る。
2. `TinyImageWatchApp` は触らない。
3. `computeUnits = .cpuOnly` 固定。
4. UNet と decoder を load and retain。
5. Watch 上では text encoder を動かさず、事前生成済み embedding を bundle する。
6. 4-step UNet loop と decoder 1 回を実行する。
7. 出力を `CGImage` または SwiftUI `Image` に変換して Watch 画面に表示する。
8. Xcode console に `[WatchPipeline]` prefix で load/predict/decode/image timing を出す。

合格条件:

- Apple Watch 実機でクラッシュしない。
- 初回 load 後、4-step + decode が数秒以内に終わる。
- 画像表示まで到達する。
- peak memory が load 時中心で、生成ごとに増え続けない。

## GitHub State

2026-06-17 時点の確認:

- remote: `origin https://github.com/lube8163-lab/tiny-watch-image-generator.git`
- remote branch: `origin/main`
- current local branch: `codex/watch-postprocess-followup`
- current branch は remote に push されていない。
- stress test 関連ファイルは未コミット差分。
