# Tiny Watch Image Generator

Apple Watch 上で動くことを優先した、小型の画像生成デモです。

現在の watchOS 版は diffusion を Watch 上で動かす構成ではありません。プロンプトを小さな latent に変換し、座標条件付き MLP が 128x128 RGB 画像を直接生成します。学習済み重み `TinyWeights.bin` はリポジトリに含めているため、研究用データセットや RunPod 環境がなくても Xcode からすぐにビルドできます。

## Xcode Quick Start

必要なもの:

- macOS
- Xcode with watchOS SDK and Swift 6 support
- Apple Watch Simulator, or a paired Apple Watch for device testing

Watch Simulator で動かす場合:

```sh
open watchos_example/TinyImageWatchApp.xcodeproj
```

1. Scheme に `TinyImageWatchApp` を選びます。
2. Destination に任意の Apple Watch Simulator を選びます。
3. Run を押します。

Simulator build は署名設定なしで動きます。実機の Apple Watch に入れる場合だけ、Xcode の `Signing & Capabilities` で自分の Team を選び、必要に応じて `Bundle Identifier` を自分用に変更してください。

CLI で Simulator 向けにビルド確認する場合:

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

実機アーキテクチャ向けに compile-only で確認する場合:

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

実機へインストールする CLI build は署名設定が必要です。Xcode で Team と Bundle Identifier を設定した後に実行してください。

```sh
xcodebuild \
  -project watchos_example/TinyImageWatchApp.xcodeproj \
  -scheme TinyImageWatchApp \
  -destination 'generic/platform=watchOS' \
  build
```

## What Is Included

- `watchos_example/TinyImageWatchApp.xcodeproj`: すぐ開ける watchOS サンプルアプリ
- `watchos_example/TinyImageWatchApp/TinyWeights.bin`: Watch アプリに同梱する学習済み int8 重み
- `WatchPipelineSmokeApp` scheme: LCM128 + transient text encoder の実機確認用 Core ML smoke target
- `WatchStressTestApp` scheme: Watch 上の Core ML load/predict/memory ceiling を測る stress target
- `Sources/TinyWatchGenerator`: 純 Swift の tiny generator 実装
- `Sources/TinyPreview`: macOS CLI で PPM を出す軽量プレビュー
- `tools/`: 学習、教師画像生成、prompt normalization、重み export 用スクリプト
- `docs/`: 研究ログと追加学習・モデル改善メモ

`datasets/`, `out/`, `models/`, `dist/`, `reports/` はローカルの研究成果物置き場です。GitHub には入れず、Watch サンプルのビルドにも不要です。

## SwiftPM Preview

Xcode を開かずに generator だけ確認できます。

```sh
swift run TinyPreview 7 cat > /tmp/cat.ppm
swift run TinyPreview --raw 7 cat > /tmp/cat_raw.ppm
```

`--raw` を付けない場合は watchOS 側と同じ軽量 postprocess を通します。

## Watch Eval Contact Sheets

モデル変更や prompt normalization 変更の前後比較用に、固定 prompt / seed の評価画像をまとめて生成できます。これは Watch アプリと同じ Swift generator を使うため、Python 側の近似実装ではなく実機コードに近い結果を見られます。

軽い smoke eval:

```sh
python3 tools/make_watch_eval_contact_sheet.py \
  --groups core_nouns,adjectives,actions,styles,japanese_aliases \
  --prompts-per-group 2 \
  --seeds 0
```

raw と watchOS postprocess 後を横並びで比較する場合:

```sh
python3 tools/make_watch_postprocess_compare.py \
  --groups core_nouns,adjectives,actions,styles,japanese_aliases \
  --prompts-per-group 2 \
  --seeds 0
```

prompt alias / slot / Watch UI preset のカバレッジを確認する場合:

```sh
python3 tools/audit_watch_prompt_coverage.py --fail-on-missing-ui --fail-on-unknown
```

通常 eval の出力先はデフォルトで `reports/watch_eval/YYYYMMDD_HHMMSS/`、postprocess 比較の出力先は `reports/watch_postprocess_compare/YYYYMMDD_HHMMSS/` です。`reports/` は Git 管理外なので、生成画像を誤ってコミットしにくい構成です。Swift evaluator はデフォルトで release build を使います。

Swift evaluator だけを直接使う場合:

```sh
swift run -c release TinyWatchEval \
  --config configs/prompt_eval_suite.json \
  --out-dir reports/watch_eval/current \
  --groups core_nouns \
  --prompts-per-group 4 \
  --seeds 0,7
```

PNG contact sheet 作成には Pillow が必要です。

```sh
python3 -m pip install pillow
```

## Current Watch App

- 128x128 RGBA output
- Prompt preset picker
- Slot/chip input: subject, color, action, view, style
- Advanced text input
- Lightweight postprocess for background denoise/matting
- No network access at runtime
- No Core ML model required for the current watchOS demo

## Core ML Watch Smoke Targets

The Core ML watch experiments live in the same Xcode project but are separate
schemes so the shipping tiny demo stays easy to run:

- `WatchStressTestApp`: transient text encoder and component memory probes.
- `WatchPipelineSmokeApp`: prompt input, transient CLIP text encoding, streamed
  LCM128 6-bit UNet chunks, 128px 4-bit decoder, and `Sharp x2` preview.

Compiled `.mlmodelc` bundles are intentionally ignored by Git. The repository
tracks the Swift targets, prompt/scheduler assets, tokenizer files, docs, and
verification scripts; large model packages should be restored locally under
`watchos_example/WatchPipelineSmokeApp/Models/` and
`watchos_example/WatchPipelineSmokeApp/TextEncoderAssets/`.

The main smoke-test notes are in [docs/watch/README.md](docs/watch/README.md).

## Training And Research Notes

追加学習や重み再生成は任意です。通常の Xcode build には不要です。

主な入口:

```sh
bash tools/run_cloud_watch_v7_pipeline.sh
python3 tools/build_watch_v8_slot_prompt_config.py
bash tools/cloud_generate_sdxl_watch_v8_slot.sh
python3 tools/prompt_normalization.py --help
python3 tools/train_tiny_coordinate_mlp.py --help
```

進捗メモ:

- [docs/watch/README.md](docs/watch/README.md)
- [docs/articles/tiny_watch_image_generator_progress_2026-06-14.md](docs/articles/tiny_watch_image_generator_progress_2026-06-14.md)
- [docs/model_improvement_plan.md](docs/model_improvement_plan.md)

## Troubleshooting

If Xcode asks for a development team when using Simulator:

- Make sure the selected destination is an Apple Watch Simulator, not a physical device or `Any watchOS Device`.
- Clean build folder if Xcode reused a previous device destination.

If a physical Watch build fails with signing errors:

- Select your own Team in `Signing & Capabilities`.
- Change `Bundle Identifier` from `dev.local.TinyImageWatchApp` to a unique identifier you own.
- Ensure the Watch is paired and enabled for development.

If the app crashes with `TinyWeights.bin is missing from the app bundle`:

- Confirm `watchos_example/TinyImageWatchApp/TinyWeights.bin` exists after clone.
- Confirm it appears in the target's Copy Bundle Resources phase.
