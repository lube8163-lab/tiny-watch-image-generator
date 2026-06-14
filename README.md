# Tiny Watch Image Generator

Apple Watch で動かすことを最優先にした、極小の画像生成モデルです。実用性や品質は捨て、技術デモとして「モデル推論で画像を生成する」ことだけに絞っています。

今後は 2 つのルートで進めます。

- `TinyWatchGenerator`: 純 Swift の玩具 txt2img。Watch UI と推論パスの検証用。
- Core ML 量子化ルート: 既存の小型画像生成モデルを分解、蒸留、量子化して Apple Watch を狙う研究用。

## 方針

- Diffusion / Transformer / VAE は使わない
- Core ML 変換も必須にしない
- 32x32 RGB を生成する座標条件付き MLP
- 重みは int8 配列として Swift に直接埋め込み
- 入力は `prompt, seed, x, y, radius, sin_feature, cos_feature, bias, latent[8]`
- 出力は各ピクセルの RGB

現在のモデルは 293 パラメータです。うち重みは 270 個の int8 で、残りは Float bias です。

## 生成

```sh
python3 tools/generate_weights.py
python3 tools/preview.py --prompt "sunset" --seed 7 --size 32 --out out/seed7.png
```

Swift 側の確認:

```sh
swift run TinyPreview 7 sunset > out/seed7.ppm
```

## Apple Watch への組み込み

`TinyWatchGenerator` を watchOS ターゲットに追加し、`watch_example/ContentView.swift` のように `TinyImageGenerator().generate(seed:)` の RGBA バッファを `CGImage` に変換して表示します。

この構成は画質ではなくサイズを優先しています。より小さくするなら hidden を 6 程度まで落とせます。少し見た目を改善するなら hidden を 16-24 に増やすのが現実的です。

## 350M 級 txt2img への方針

350M パラメータ級を目指すなら、純 Swift 配列ではなく Core ML の `mlpackage` と圧縮を使います。詳細は [docs/watch_txt2img_plan.md](docs/watch_txt2img_plan.md) にまとめています。

実際の作業順は [docs/phase_workflow.md](docs/phase_workflow.md) にまとめています。最初の本命候補は `SimianLuo/LCM_Dreamshaper_v7` です。これは Watch へ直接載せるモデルではなく、Mac 側で品質と component 分割を検証するための基準モデルです。

iPhone 実機検証後の品質改善方針は [docs/model_improvement_plan.md](docs/model_improvement_plan.md) にまとめています。現行小型モデルのMac基準画像生成と、`SDXL_test` のローカルSDXLキャッシュを使った教師画像セット生成は以下のスクリプトから始めます。

```sh
.venv/bin/python tools/generate_student_reference_grid.py --candidate segmind_tiny_sd --local-files-only
.venv/bin/python tools/generate_sdxl_teacher_dataset.py --limit 1 --variants-per-prompt 1 --seeds 0 --steps 1 --target-sizes 128,64
```

重みサイズの概算:

```sh
python3 tools/model_budget.py --params 350
```

既存 Core ML モデルの圧縮スクリプト雛形:

```sh
python3 tools/coreml_quantize.py path/to/component.mlpackage --mode palettize4 --out dist/component_4bit.mlpackage
```

現実的な最初の目標は、フル Stable Diffusion ではなく、64x64 の latent decoder + 小型 denoiser + 簡易 text conditioning です。Watch ではファイルサイズより実行時メモリと発熱が支配的なので、段階的に FP16 -> int8 -> 4bit の順で落としていきます。

研究環境:

```sh
bash tools/bootstrap_research_env.sh
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
```

Phase 1:

```sh
python3 tools/phase0_download.py --candidate segmind_tiny_sd
python3 tools/phase1_generate.py --candidate segmind_tiny_sd --local-files-only --prompt "a small watercolor landscape, crisp details"
python3 tools/phase1_inspect.py --candidate segmind_tiny_sd --local-files-only
```

Phase 2:

```sh
python3 tools/phase2_export_vae_decoder.py --candidate segmind_tiny_sd --local-files-only --output-width 64 --output-height 64 --drop-mid-attention
python3 tools/coreml_quantize.py dist/segmind_tiny_sd/vae_decoder_64x64.mlpackage --mode palettize4 --out dist/segmind_tiny_sd/vae_decoder_64x64_4bit.mlpackage
```

Phase 3 UNet probe:

```sh
python3 tools/phase3_export_unet.py --candidate lcm_dreamshaper_v7 --local-files-only --latent-height 8 --latent-width 8 --attention-processor eager
python3 tools/coreml_quantize.py dist/lcm_dreamshaper_v7/unet_8x8.mlpackage --mode palettize4 --out dist/lcm_dreamshaper_v7/unet_8x8_4bit.mlpackage
python3 tools/phase3_smoke_unet_coreml.py dist/lcm_dreamshaper_v7/unet_8x8_4bit.mlpackage --latent-height 8 --latent-width 8
```
