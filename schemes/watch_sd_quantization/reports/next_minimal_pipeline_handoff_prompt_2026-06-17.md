# Handoff Prompt: Watch Minimal Pipeline Smoke Test

以下を次のチャットに貼って開始する。

```text
リポジトリ: /Users/tasuku/Documents/ちっちゃい画像生成モデル

目的:
Apple Watch 実機で、既存 TinyImageWatchApp を壊さずに、別 target/scheme として Stable Diffusion 系の最小 pipeline smoke test を作りたいです。

重要:
- 既存の TinyImageWatchApp target/source は壊さないでください。
- 既存の WatchStressTestApp は stress test 用として残してください。
- 続きは別 scheme/target で進めたいです。候補名は WatchPipelineSmokeApp です。
- Core ML は CPU-only 固定で進めてください。MLComputeUnits.all は使わないでください。
- Watch 上で text encoder は動かさないでください。prompt embedding は事前生成済み asset/preset として bundle する方針です。
- Xcode console にログが出るようにしてください。prefix は [WatchPipeline] などにしてください。

現在ある stress test の成果:
- Xcode project: watchos_example/TinyImageWatchApp.xcodeproj
- 既存 scheme: TinyImageWatchApp
- stress scheme: WatchStressTestApp
- stress app: watchos_example/WatchStressTestApp
- stress model directory: watchos_example/WatchStressTestApp/Models
- helper: schemes/watch_sd_quantization/scripts/install_stress_model.sh
- 詳細レポート: schemes/watch_sd_quantization/reports/watch_stress_test_2026-06-17.md

実機ストレステスト結果:
- memory-only は 16 MB 刻みで stable=272 MB まで成功し、target=288 MB attempt 後にクラッシュしました。
- decoder 4bit 23 MB は問題なく動き、predict はおおむね 0.05-0.07s でした。
- UNet 4bit 155 MB は Compute=All だと ANE compile error とメモリ急増でクラッシュしました。
- 同じ UNet は Compute=CPU なら load/predict できました。
- UNet CPU predict once は predict 部分が 0.830s、メモリピークは約 47.7 MB でした。
- UNet + decoder を Load & Retain して Pipeline 4+Decode を試したところ、初回 total 2.163s、+32 MB 後の warm run total 1.065s でした。
- UNet + decoder retained 時のメモリピークは約 61.2 MB で、推論中ではなく Load & Retain 時が最大でした。

現在 bundle 済みの stress models:
- watchos_example/WatchStressTestApp/Models/unet_sd_16x16_4bit.mlmodelc 約 155 MB
- watchos_example/WatchStressTestApp/Models/vae_decoder_64x64_noattn_4bit.mlmodelc 約 23 MB

次に作りたいもの:
1. WatchPipelineSmokeApp target/scheme を追加してください。
2. TinyImageWatchApp と WatchStressTestApp の挙動を変えないでください。
3. UNet と decoder を CPU-only で load and retain してください。
4. prompt embedding と scheduler/latent 初期値を bundle asset として読み込める構成にしてください。
5. 最初は品質よりも実機 pipeline 統合を優先し、4-step UNet loop + decoder 1 回 + 画像表示まで到達してください。
6. 可能なら existing asset を調べてください:
   - ios_example/TinyImageIOSApp/Resources/PromptAssets/
   - dist/segmind_tiny_sd/ が存在する場合はその中の watch_sd_txt2img_128 など
7. 画像は SwiftUI 上に表示し、ボタンで Generate できるようにしてください。
8. load time, each UNet step time, decoder time, total time を Xcode console に出してください。
9. simulator build で xcodebuild が通るところまで確認してください。

想定する判断:
- 本命は 128x128 / latent 16x16 / 4-step / preset prompt embedding / CPU-only です。
- load の一時ピークが最大リスクなので、起動後に load and retain し、生成時に毎回 load しない設計がよさそうです。
- もし実 asset の scheduler/embedding 接続が重ければ、第一段階は固定 embedding + 固定/ランダム latent の smoke test で構いません。

GitHub 状態:
- remote はあります: https://github.com/lube8163-lab/tiny-watch-image-generator.git
- remote branch は origin/main のみ確認済みです。
- 作業ブランチ codex/watch-postprocess-followup は未 push です。
- stress test 関連差分は未コミットです。
```
