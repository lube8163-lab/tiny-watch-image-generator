# SD1.5 Teacher Dataset Instructions

この指示書は、PC上でStable Diffusion 1.5を使って、tiny/watch向け画像生成モデルの教師データを作るためのものです。目的は高品質な大規模txt2imgではなく、Apple Watchでも扱える小型モデルへ蒸留しやすい、短いプロンプトと単純な構図の画像セットを作ることです。

## 目的

- 64x64から128x128程度の小型生成モデルに蒸留しやすい教師画像を作る。
- Apple Watchで入力しやすい、1から3語程度の短いプロンプトに対応する。
- まずは固定カテゴリで構造を学ばせ、その後に簡単な属性語を追加する。
- SDXL教師データ、OpenImages採掘データと同じように `metadata.jsonl` で混ぜられる形にする。

## 最初に作るべきデータ

最初の本命は、固定16カテゴリを厚くするデータセットです。

対象カテゴリ:

```text
apple
bird
car
castle
cat
dog
face
fish
flower
house
moon
robot
star
sun
train
tree
```

目標枚数:

- smoke: 各カテゴリ8枚、合計128枚
- pilot: 各カテゴリ64枚、合計1,024枚
- first train: 各カテゴリ256枚、合計4,096枚
- stronger train: 各カテゴリ512から1,024枚、合計8,192から16,384枚

今の小型MLP/小型Core ML向けには、まず4,096枚あれば十分に次の比較ができます。最終的には16カテゴリだけでも8,000から16,000枚程度あると安定しやすいです。

## 画像仕様

SD1.5の生成は512x512を基本にします。保存時に以下をすべて作ります。

- `images_512/*.png`: 元の教師画像
- `images_256/*.png`: 確認・将来用
- `images_128/*.png`: 現在の学習候補
- `images_64/*.png`: tiny MLP/Watch向け第一候補

リサイズは必ずRGBで、中央の物体が潰れすぎないようにします。単純な正方形画像で統一してください。

## プロンプト設計

プロンプトは「短い入力」と「教師生成用の詳細プロンプト」を分けます。

- `key`: 学習時に使う短い条件。例: `cat`
- `prompt`: SD1.5に渡す詳細プロンプト。
- `title`: UI表示用。例: `Cat`

重要なのは、小型モデルへ渡す条件は基本的に `key` に寄せることです。`prompt` 全文を条件にすると、Apple Watchで入力する短い単語と一致しません。

基本テンプレート:

```text
{subject}, centered subject, single subject, simple clean background, readable silhouette, clean illustration, no text, no logo, no watermark
```

negative prompt:

```text
text, logo, watermark, caption, low quality, blurry, distorted, deformed, cluttered background, multiple subjects, cropped subject, extra limbs
```

カテゴリごとに4から8個のvariantを作ります。例:

```json
{
  "key": "cat",
  "title": "Cat",
  "variants": [
    "cute cat sitting",
    "single orange cat face",
    "black cat silhouette",
    "small cat standing side view",
    "white cat sitting front view",
    "simple cartoon cat"
  ]
}
```

seedはvariantごとに複数使います。例: 8 variants x 32 seeds = 256枚/カテゴリ。

## 追加属性の段階

固定16カテゴリが安定してから、属性語を足します。

優先属性:

```text
red
blue
green
yellow
black
white
small
big
round
simple
cartoon
pixel art
```

最初は全カテゴリに属性を掛け合わせすぎないでください。データが薄くなります。

推奨:

- 16カテゴリ単体を厚くする。
- 次に `red apple`, `blue car`, `black cat`, `white dog`, `small bird` のような自然な組み合わせだけ追加する。
- 不自然な組み合わせを大量に作らない。

## 生成設定

SD1.5の推奨初期設定:

```text
resolution: 512x512
steps: 20 to 30
scheduler: DPM++ 2M Karras or Euler a
guidance_scale: 6.5 to 8.0
seed: deterministic sequence
batch size: 環境に合わせる
```

品質が荒い場合:

- stepsを25から30へ上げる。
- guidanceを7前後にする。
- negative promptを強める。

構図が複雑になりすぎる場合:

- `single subject`
- `plain background`
- `icon-like composition`
- `centered`
- `full object visible`

を強めます。

## metadata.jsonl形式

各画像につき1行のJSONLにします。既存資源と混ぜやすいように、最低限この形式にしてください。

```json
{
  "accepted": true,
  "id": "cat_v00_seed000000",
  "key": "cat",
  "title": "Cat",
  "prompt": "cute cat sitting, centered subject, single subject, simple clean background, readable silhouette, clean illustration, no text, no logo, no watermark",
  "negative_prompt": "text, logo, watermark, caption, low quality, blurry, distorted, deformed, cluttered background, multiple subjects, cropped subject, extra limbs",
  "variant": "v00",
  "seed": 0,
  "model_family": "sd15",
  "scheduler": "dpmpp_2m_karras",
  "steps": 25,
  "guidance_scale": 7.0,
  "source_width": 512,
  "source_height": 512,
  "saved_images": {
    "64": "images_64/cat_v00_seed000000.png",
    "128": "images_128/cat_v00_seed000000.png",
    "256": "images_256/cat_v00_seed000000.png",
    "512": "images_512/cat_v00_seed000000.png"
  },
  "qc_flags": []
}
```

失敗画像も可能なら残し、`accepted: false` と `reject_reason` を入れます。ただし学習では `accepted: true` のみ使います。

## manifest.json形式

データセットルートに `manifest.json` を置きます。

```json
{
  "run_id": "sd15_teacher_YYYYMMDD_HHMMSS",
  "model_family": "sd15",
  "status": "complete",
  "prompt_set_version": "fixed16_v1",
  "completed_jobs": 4096,
  "failed_jobs": 0,
  "source_resolution": 512,
  "generation_settings": {
    "steps": 25,
    "guidance_scale": 7.0,
    "scheduler": "dpmpp_2m_karras",
    "negative_prompt": "text, logo, watermark, caption, low quality, blurry, distorted, deformed, cluttered background, multiple subjects, cropped subject, extra limbs"
  }
}
```

## 品質チェック

生成後に必ずcontact sheetを作ります。

- `reports/contact_sheet_64_all.png`
- `reports/contact_sheet_128_labeled.png`
- `reports/summary.json`

目視で落とす基準:

- 文字、ロゴ、透かしがある。
- 主体が複数ある。
- 主体が画面外で切れている。
- 背景が複雑すぎて、64x64で主体が読めない。
- 顔や動物が大きく崩れている。
- keyと画像内容が明らかに違う。

ただし、小型モデル向けなので「細部が完璧か」より「64x64で読めるシルエットか」を優先します。

## 学習への混ぜ方

最初の学習では、SD1.5教師データを主軸にします。

推奨比率:

- SD1.5 fixed16: 70%
- iPhone SDXL pilot: 20%
- OpenImages selected: 10%

OpenImagesは実写寄りなので、入れすぎると小型MLPでは平均化してぼやけやすいです。まずはSD1.5/SDXLの生成画像を中心にしてください。

OpenImagesを混ぜる場合:

- 固定16カテゴリに完全一致するkeyだけ使う。
- `accepted: true` のみ使う。
- 1カテゴリあたり最大32から64枚程度に制限する。
- `moon`, `robot`, `star`, `sun` などOpenImagesで弱いカテゴリはSD1.5で補う。

## 優先順位

1. 固定16カテゴリ x 64枚 = 1,024枚を作る。
2. contact sheetで破綻を確認する。
3. 固定16カテゴリ x 256枚 = 4,096枚へ増やす。
4. tinyモデルで学習し、カテゴリごとの構造が出るか確認する。
5. 良ければ属性語つきの短いプロンプトへ拡張する。
6. 最後にOpenImagesを少量混ぜて汎化を確認する。

## 注意

自由入力対応を急がないでください。今の目的では、まず `cat`, `dog`, `moon`, `robot` のような短い単語で構造が出ることが重要です。

数単語対応は、固定カテゴリが安定した後に追加します。最初から数万プロンプトへ広げると、1カテゴリあたりの密度が下がり、小型モデルでは意味も見た目も薄くなります。
