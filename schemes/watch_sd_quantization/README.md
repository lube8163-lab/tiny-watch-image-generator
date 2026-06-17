# Apple Watch Stable Diffusion Quantization Scheme

このスキームは、既存のスクラッチ/半スクラッチ系成果物を触らずに、公開済み Stable Diffusion 系モデルや公開重みを徹底的に軽量化して Apple Watch ローカル実行を狙うための隔離ルートです。

## Boundary

- 既存の `TinyImageWatchApp` target/source は壊さない。
- watchOS 実機検証に必要な場合だけ、隔離された追加 target/scheme を `watchos_example/` 配下に作る。
- このスキームの中間生成物とレポートは `schemes/watch_sd_quantization/artifacts/` と `schemes/watch_sd_quantization/reports/` に置く。
- 既存の `dist/` と `models/` は読み取り専用の入力として扱う。
- 既存スクリプトを使う場合も、`--out` と `--manifest` は必ずこのスキーム配下を指定する。

## Current Status

2026-06-17 時点では、隔離 target/scheme `WatchStressTestApp` を追加し、Apple Watch 実機で Core ML load/predict/memory のストレステストを実施済みです。

- 実測まとめ: `schemes/watch_sd_quantization/reports/watch_stress_test_2026-06-17.md`
- 次スキーム引き継ぎプロンプト: `schemes/watch_sd_quantization/reports/next_minimal_pipeline_handoff_prompt_2026-06-17.md`

## Feasibility

結論として、公開済み Stable Diffusion/SDXL を単純に 4bit 化して Apple Watch でそのまま txt2img するのはかなり厳しいです。理由はモデルファイルサイズではなく、UNet/Transformer の中間 activation、Core ML の実行計画、複数 step 実行時のメモリ/発熱が支配的になるためです。

一方で、以下の条件まで削れば検証価値があります。

- 解像度は最初 64x64、次に 96x96。
- step 数は 1-4。通常 SD の 20-30 step は Watch では避ける。
- text encoder は Watch から外し、最初は prompt embedding table にする。
- denoiser は公開モデルからの直接量子化だけでなく、公開モデルを teacher にした蒸留/剪定を前提にする。
- VAE はフル SD VAE ではなく tiny decoder または aggressive no-attention decoder から始める。

## Candidate Matrix

| Candidate | Role | Initial read | Risk |
| --- | --- | --- | --- |
| `segmind/tiny-sd` | Direct SD-compatible baseline | 4bit UNet が既存計測で約 155 MB、4bit decoder が約 23 MB | SD scheduler だと step 数が重い。LCM 化か student 蒸留が必要 |
| `SimianLuo/LCM_Dreamshaper_v7` | Few-step baseline | 4 step は魅力だが、4bit UNet が既存計測で約 411 MB | Watch 直載せには大きすぎる可能性が高い |
| `apple/coreml-stable-diffusion-xl-base-ios` | Compression reference | iOS/iPadOS 向け mixed-bit Core ML の参考 | SDXL は Watch には大きすぎる。レシピ参考用 |
| Custom distilled tiny denoiser | Probable Watch path | 公開 SD/LCM を teacher にして、固定 latent size の student を作る | 学習工程が必要。ただし direct quantization より成功確率が高い |

## Milestones

### Q0: Artifact audit

既存の Core ML package を読み取り専用で棚卸しし、Watch に載せる候補をサイズで分ける。

```sh
python3 schemes/watch_sd_quantization/scripts/audit_coreml_artifacts.py
```

出力:

```text
schemes/watch_sd_quantization/reports/coreml_artifact_report.json
```

### Q1: Decoder-only Watch probe

まず denoiser なしで 64x64 decoder だけを watchOS アプリに載せ、ランダム latent または固定 latent を decode できるかを見る。ここで Core ML load time、peak memory、thermal を測る。

既存の export script を使う場合は、出力をこのスキーム配下に限定する。

```sh
python3 tools/phase2_export_vae_decoder.py \
  --candidate segmind_tiny_sd \
  --local-files-only \
  --output-height 64 \
  --output-width 64 \
  --drop-mid-attention \
  --out schemes/watch_sd_quantization/artifacts/segmind_tiny_sd/vae_decoder_64x64_noattn.mlpackage \
  --manifest schemes/watch_sd_quantization/artifacts/segmind_tiny_sd/vae_decoder_64x64_noattn.json

python3 tools/coreml_quantize.py \
  schemes/watch_sd_quantization/artifacts/segmind_tiny_sd/vae_decoder_64x64_noattn.mlpackage \
  --mode palettize4 \
  --out schemes/watch_sd_quantization/artifacts/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.mlpackage \
  --manifest schemes/watch_sd_quantization/artifacts/segmind_tiny_sd/vae_decoder_64x64_noattn_4bit.json
```

### Q2: One-step denoiser probe

`segmind/tiny-sd` の 16x16 latent UNet を 4bit/6bit で固定 shape 化し、1 step だけ Watch で動くかを見る。prompt は自由入力ではなく、事前計算済み embedding table を使う。

この段階の合格条件:

- アプリが Watch 実機で model load できる。
- 1 step の Core ML prediction がメモリ警告/クラッシュなしで完了する。
- 10 秒以内に UI が戻る。
- Instruments で継続実行不能な発熱にならない。

### Q3: Few-step student

Q2 が重い場合、公開 SD/LCM を teacher にして 64x64/96x96 固定の小型 denoiser student を作る。ここが本命です。これはスクラッチではなく、公開重み/公開モデルの知識を圧縮して Watch 用に落とすルートです。

## Quantization Ladder

1. FP16 mlprogram で conversion が通ることを確認する。
2. 8bit linear quantization を試す。
3. 6bit palettization を試す。
4. 4bit palettization を試す。
5. layer 別 mixed-bit recipe を作る。
6. それでも重い層は student 化、層削減、attention 削除、latent size 固定で削る。

## Go/No-Go

Direct quantization ルートの暫定ライン:

- decoder-only 4bit が Watch で動く: 継続。
- 150 MB 級 UNet 4bit が 1 step でも Watch で落ちる: direct SD route は中止し、student 蒸留へ移行。
- 150 MB 級 UNet が 1 step 動くが 4 step が厳しい: LCM/student 化を優先。
- LCM Dreamshaper 411 MB 4bit が load できない: この系統は teacher 専用。

## References Checked

- [Apple Core ML documentation](https://developer.apple.com/documentation/coreml)
- [Core ML Tools optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [Apple ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)
- [Apple Stable Diffusion with Core ML on Apple Silicon](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)
- [segmind/tiny-sd model card](https://huggingface.co/segmind/tiny-sd)
- [SimianLuo/LCM_Dreamshaper_v7 model card](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- [apple/coreml-stable-diffusion-xl-base-ios model card](https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios)
