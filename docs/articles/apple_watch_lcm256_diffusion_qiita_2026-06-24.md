# Apple WatchでLCM Diffusionを動かして256x256画像生成まで持っていった話

## はじめに

Apple Watch上で、ネットワーク接続なしに画像生成を動かす実験を続けています。

前回は、Core MLを使わずに純Swiftの小さなMLPで画像を直接生成するところまでをまとめました。

- [Apple Watchで動く小型画像生成モデルを育てている話](https://qiita.com/tkkt97/items/1888f632b7dd4a16f6a0)

この記事はその続きです。

今回は、MLP方式ではなく、Apple Watch上でLCM系の小さなdiffusion pipelineを動かし、短い自由入力プロンプトから256x256画像を生成するところまで持っていった話をまとめます。

結論から言うと、SDXLのような完全自由入力ではありませんが、短いプロンプトなら「画像だけ見てもだいたい何を入力したか分かる」くらいまでは来ました。

## できたもの

現在のwatchOS実機向けbaselineは次の構成です。

| 項目 | 内容 |
|---|---|
| scheme | `WatchPipelineSmokeApp` |
| pipeline | `LCM256 6b` |
| 出力 | 256x256 RGB |
| text encoder | CLIP text encoderを一時的にロードして実行 |
| denoiser | LCM UNet、6bit、16分割streaming |
| decoder | 256px VAE decoder、4bit |
| steps | 4 |
| guidance | 6 |
| seed | ランダム、同じpromptでreroll可能 |
| compute | CPU-only Core ML |
| UI | prompt入力欄、Generate/Rerollボタン、生成結果のみ |

パイプラインはざっくり次のような流れです。

![LCM256 pipeline](https://raw.githubusercontent.com/lube8163-lab/tiny-watch-image-generator/main/docs/articles/assets/watch_lcm256_qiita_2026-06-24/01_lcm256_pipeline.png)

Apple Watch実機では、text encoderを生成パイプラインと同時に保持しないようにしています。

1. promptをtokenizeする
2. CLIP text encoderをロードする
3. `[1, 77, 768]` の `hidden_states` を作る
4. text encoderを解放する
5. Core ML cacheをpurgeする
6. LCM UNet chunkを順番にロード/推論する
7. decoderで256x256画像に戻す

この「分離実行」がかなり重要でした。

## なぜMLPからdiffusionに戻ったのか

前回のMLP方式は、Apple Watchで動かすという意味ではかなり扱いやすい方式でした。

- 依存モデルなし
- 実装が小さい
- メモリ使用量が小さい
- Xcodeで即ビルドできる

一方で、品質面では限界も見えていました。

- 学習済み語彙から外れると弱い
- 構図や形状がぼやけやすい
- 画像の多様性が低い
- 短い自由入力promptへの対応が難しい

そこで、MLPは「軽量demo / 比較baseline」として残しつつ、品質側の本命としてLCM系のdiffusion pipelineを試すことにしました。

## まずCore ML Stressで限界を見る

いきなり生成pipelineを組む前に、別schemeでCore MLのload/predict/memory限界を測りました。

使った隔離スキームはこちらです。

- [schemes/watch_sd_quantization](https://github.com/lube8163-lab/tiny-watch-image-generator/tree/main/schemes/watch_sd_quantization)
- [WatchStressTestApp.xcscheme](https://github.com/lube8163-lab/tiny-watch-image-generator/blob/main/watchos_example/TinyImageWatchApp.xcodeproj/xcshareddata/xcschemes/WatchStressTestApp.xcscheme)
- [stress test report](https://github.com/lube8163-lab/tiny-watch-image-generator/blob/main/schemes/watch_sd_quantization/reports/watch_stress_test_2026-06-17.md)

この段階では、既存の小さいStable Diffusion系Core MLモデルをwatchOS実機に載せて、単体ロードや推論を試しています。

メモリだけを見るFine ladderでは、16MB刻みで `272MB` までは安定、`288MB` 付近でクラッシュしました。以前のaggressive ladderでも300MB前後で落ちています。

つまり、ざっくり言えばApple Watch上では300MB程度がかなり強い壁になっていました。

また、`MLComputeUnits.all` ではUNetのANE compile失敗やメモリ急増がありました。

```text
MILCompilerForANE error
_ANECompiler : ANECCompile() FAILED
```

そのため、以降のWatch実機ルートは基本的にCPU-onlyで考えることにしました。

このstress testで分かったことは大きく3つです。

- ファイルサイズよりも、load時の一時メモリとCore ML実行計画が怖い
- decoder単体はかなり軽い
- 大きなUNetを一括で持つより、分割して一時ロードする方が現実的

## LCMを選んだ理由

普通のStable Diffusionは20〜30step程度を前提にすることが多く、Apple Watchには重すぎます。

一方でLCMは少ないstepで画像を出せます。今回のbaselineでは4stepです。

もちろん、LCMにしたからといって何でも解決するわけではありません。UNetはまだ大きいですし、text encoderも重いです。

そこで、次のように分解しました。

- text encoderは一時的にだけロードする
- UNetは16個のCore ML chunkに分割する
- decoderは4bit化する
- すべてCPU-onlyで動かす
- 画像は256x256に固定する
- UIはprompt入力と生成結果だけに寄せる

この構成で、Apple Watch実機上でも生成まで到達できました。

## 128px、192px、256px

最初は64pxや128pxから確認していました。

128pxでも動きましたが、見た目としてはまだかなり小さく、何が描かれているか分かるものと分からないものの差が大きい状態でした。

その後、192pxを試すと品質がかなり上がりました。

さらに256pxも試したところ、実機メモリピークは最大で約140MB程度に収まり、品質ももう一段上がりました。

生成時間はWatch実機でだいたい1分前後です。代表的な実機ログでは、promptによっておおむね58〜63秒程度でした。

256pxにしたことで、次のようなカテゴリがかなり読みやすくなりました。

- 動物
- 食べ物
- 自然風景
- 単体オブジェクト
- 一部のロゴ/アイコン
- 一部の関係prompt

## Mac上で296枚の品質評価

品質評価は、Apple Watch実機ではなくMac上で行っています。

理由は、モデル品質の比較だけならWatchのANEや実機メモリは不要で、同じscheduler、tokenizer、text encoder、UNet、decoder、seed規則を使えば、傾向を見るには十分だからです。

評価用スクリプトはこちらです。

- [tools/watch_lcm256_quality_eval.py](https://github.com/lube8163-lab/tiny-watch-image-generator/blob/main/tools/watch_lcm256_quality_eval.py)

実行コマンドは次のような形です。

```sh
.venv/bin/python tools/watch_lcm256_quality_eval.py \
  --out-dir reports/watch_lcm256_quality/full_lcm256_g6
```

評価は、74 prompts x 4 seeds = 296 imagesで行いました。

結果は次の通りです。

| 項目 | 結果 |
|---|---:|
| 生成枚数 | 296 |
| 失敗 | 0 |
| 平均生成時間 | 7.241s/image |
| 合計時間 | 35.7分 |
| 出力サイズ | 約44MB |

詳細な評価メモはこちらに置いています。

- [Mac LCM256 Full Quality Eval Summary](https://github.com/lube8163-lab/tiny-watch-image-generator/blob/main/docs/watch/mac_quality_eval_full_summary_2026-06-24.md)

## 生成例

以下はMac評価で生成した画像からの抜粋です。

Apple Watch実機スクリーンショットではなく、同じCore ML packageをMacでCPU-only実行した品質評価画像です。実機側では最終的なメモリ、時間、熱、UIを確認しています。

![readable examples](https://raw.githubusercontent.com/lube8163-lab/tiny-watch-image-generator/main/docs/articles/assets/watch_lcm256_qiita_2026-06-24/02_lcm256_good_examples.png)

個人的には、このあたりまで出るなら「Apple Watch上のローカル画像生成demo」としてはかなり面白いところまで来たと思っています。

特に、次のようなpromptはかなり読めます。

- `snowy mountain`
- `strawberry cake`
- `blue bird`
- `dragon`
- `astronaut riding a horse`
- `cat in a spaceship`
- `Apple logo`
- `steam train`

## seed rerollが大事

Watch UIでは、同じpromptで再生成するとseedをrerollするようにしています。

これはかなり重要でした。

同じpromptでもseedによってかなり結果が変わります。

![seed variation](https://raw.githubusercontent.com/lube8163-lab/tiny-watch-image-generator/main/docs/articles/assets/watch_lcm256_qiita_2026-06-24/03_seed_variation.png)

たとえば `cat in a spaceship` は、seedによって「何か青く光る猫っぽいもの」から「かなり猫と宇宙船っぽいもの」まで変わります。

`bicycle` のような細い構造は苦手ですが、それでもseedによっては読める形になります。

このモデルでは、1回で完璧な画像を出すよりも、短いpromptを入れて何回かrerollする体験の方が自然です。

## 苦手なもの

もちろん、まだ何でも生成できるわけではありません。

苦手な例もあります。

![weaker examples](https://raw.githubusercontent.com/lube8163-lab/tiny-watch-image-generator/main/docs/articles/assets/watch_lcm256_qiita_2026-06-24/04_lcm256_limit_examples.png)

弱いのはだいたい次のあたりです。

- 細い構造
  - bicycle
  - 指、腕、脚
  - 楽器の弦や細部
- pose依存の人物
  - ballerina
  - superhero
- 抽象的なstyle語
  - neon spiral
  - origami crane
- 厳密な関係表現
  - holding
  - playing
  - wearing
- 正確な文字やロゴ

ただし、数値上 `very_soft` と判定された画像が、目視では悪くないこともあります。

たとえば `green apple` や `moonlit lake` は、エッジが少ないので自動評価では弱く出ますが、画像としてはそこまで悪くありません。

そのため、現時点の評価では自動スコアはあくまで異常値検出に使い、最終判断はcontact sheetの目視にしています。

## 296枚評価で見えた傾向

ジャンル別には、だいたい次のような印象です。

| ジャンル | 読み |
|---|---|
| nature | 最も安定。mountain、ocean、beach、waterfallなどが強い |
| food | cake、ramen、sushi、hamburgerがかなり良い |
| fantasy | dragon、phoenix、crystal castleは良い。複合概念はseed依存 |
| animals | soft判定は多いが、目視ではかなり読める |
| objects | teapot、boot、cameraは良い。bicycle、smartphoneは弱い |
| logos/icons | Apple logoやpaw printは良いseedがある。文字ロゴは弱い |
| characters | astronaut、robot、knight、chefは出る。細い人物poseは弱い |
| relations | 想定より良いが、厳密な関係理解はまだ限定的 |

ゼロフラグで安定していたpromptには、次のようなものがあります。

- `dolphin`
- `tiger`
- `dragon`
- `phoenix bird`
- `hamburger`
- `ramen bowl`
- `strawberry cake`
- `sushi plate`
- `cat badge`
- `snowy mountain`
- `ocean wave`
- `waterfall`
- `leather boot`
- `astronaut riding a horse`
- `steam train`
- `yellow bus`

一方で、4seedすべてで弱めだったpromptもありました。

- `ballerina`
- `green apple`
- `moonlit lake`
- `neon spiral`

ここは今後の学習データやprompt展開の改善対象になりそうです。

## 実機側で気をつけたこと

Apple Watch実機では、モデルが動くかどうかだけでなく、次の点をかなり気にしました。

### text encoderを保持しない

CLIP text encoderは大きいので、生成中にUNetと一緒に持つのは避けました。

prompt embeddingを作ったらすぐに解放します。

### UNetを分割する

UNetを一括で持つのではなく、16 chunkに分けて順番に通します。

これにより、load時のメモリピークを抑えられます。

### CPU-onlyにする

ANEやGPUを使えば速くなる可能性はありますが、watchOS実機ではCore ML compileやメモリ挙動が不安定になりやすいです。

現状はCPU-onlyの方が読みやすく、制御しやすいです。

### UIを小さくする

最初はデバッグ用にpreset、seed、guidance、preview modeなどをUIに出していました。

最終的には、Watch上では次だけに寄せました。

- prompt入力欄
- Generate / Reroll Seed ボタン
- 生成結果

詳細はXcode consoleにログとして出します。

## 今後やりたいこと

ここから大きく品質を上げるには、単に解像度を上げるだけでは厳しそうです。

256px化でかなり改善しましたが、これ以上は時間やactivation memoryの割に伸びが小さそうです。

次の改善候補はこのあたりです。

- LCM256向けの追加学習/蒸留
- Watch向けdecoderの改善
- 短いpromptに特化した軽量text encoder
- prompt expansionの改善
- seed候補の複数生成/選択UI
- thermalや連続生成時の実機評価

特に、`bicycle`、`ballerina`、`holding` のような弱点は、解像度だけではなく学習データ側の改善が必要そうです。

## まとめ

Apple Watch上でLCM系diffusionを動かし、短い自由入力promptから256x256画像を生成できるところまで来ました。

現時点のポイントは次の通りです。

- Apple Watch実機でLCM256 6bit pipelineが動いた
- CLIP text encoderも分離実行なら使えた
- UNetは16 chunkに分けてstreamingした
- CPU-only Core MLで安定性を優先した
- 実機メモリピークは約140MB程度に収まった
- Mac品質評価では296枚生成して失敗0だった
- 短いpromptなら、画像だけでも意図が分かることが増えた

まだSDXLのような自由入力ではありません。

ただ、Apple Watch単体でここまで画像が出るなら、技術demoとしてはかなり面白い段階に入ったと思います。

リポジトリはこちらです。

- [tiny-watch-image-generator](https://github.com/lube8163-lab/tiny-watch-image-generator)
