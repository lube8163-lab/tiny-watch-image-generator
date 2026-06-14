# Cloud GPU teacher dataset runbook

This runbook is for generating SDXL teacher images on a cloud GPU instead of the
local M2/MPS path.

## Why cloud GPU

The local M2 / 8 GB PyTorch/MPS path is reliable only for tiny probes:

- `enable_attention_slicing()` caused non-finite SDXL latents and black images.
- With attention slicing disabled, `384x384`, `1` step still took about 158
  seconds for one image.

Use cloud GPU for any dataset larger than a small smoke test.

## Browser setup

Create a GPU pod manually in the browser. Do not share account passwords or
payment details with the agent.

Recommended starting point:

- Provider: RunPod
- Template: PyTorch / CUDA / Jupyter or SSH template
- GPU: RTX 4090 24 GB, L40S 48 GB, RTX A6000 48 GB, or A5000 24 GB
- Disk: at least 80 GB container/volume disk for SDXL cache and generated images

After the pod starts, copy the SSH command or Jupyter terminal URL. The agent can
continue from there.

## Upload project

From the Mac:

```sh
cd "/Users/tasuku/Documents/ちっちゃい画像生成モデル"
tools/make_cloud_bundle.sh /tmp/tiny-image-model-cloud.tar.gz
```

Upload to the pod. Example:

```sh
scp -P <PORT> /tmp/tiny-image-model-cloud.tar.gz root@<HOST>:/workspace/
```

On the pod:

```sh
cd /workspace
mkdir -p tiny-image-model
tar --no-same-owner -xzf tiny-image-model-cloud.tar.gz -C tiny-image-model
cd tiny-image-model
```

## Setup

```sh
bash tools/cloud_setup_sdxl.sh
```

This creates `.venv` with `--system-site-packages` so the PyTorch/CUDA install
from the cloud image remains visible.

## Smoke test

```sh
source .venv/bin/activate
python3 tools/generate_sdxl_teacher_dataset.py \
  --allow-downloads \
  --limit 1 \
  --variants-per-prompt 1 \
  --seeds 0 \
  --steps 4 \
  --width 384 \
  --height 384 \
  --target-sizes 128,64 \
  --batch-size 1 \
  --out-dir datasets/sdxl_cloud_smoke \
  --save-source

python3 tools/validate_teacher_dataset.py datasets/sdxl_cloud_smoke --image-size 128
```

Check:

- `datasets/sdxl_cloud_smoke/contact_sheet.png`
- `datasets/sdxl_cloud_smoke/quality_report.json`

## Fixed16 run

Start with this before broad prompt generation:

```sh
bash tools/cloud_generate_sdxl_fixed16.sh
```

Defaults:

- 16 categories
- 8 variants
- 8 seeds
- 20 steps
- 768x768 source
- 256/128/64 saved sizes
- batch size 2

Override defaults if needed:

```sh
RUN_NAME=sdxl_cloud_teacher_fixed16_v1 \
STEPS=20 \
BATCH_SIZE=2 \
bash tools/cloud_generate_sdxl_fixed16.sh
```

On a 24 GB RTX 4090, `BATCH_SIZE=2` is the conservative default for 768px SDXL.

## Expanded32 v2 run

This is the next dataset after the fixed16 pilot. It expands the controlled
teacher set to 32 categories and adds action/state, modifier, and composition
variants while using stricter plain-background prompts.

Run a small pilot first:

```sh
RUN_NAME=sdxl_cloud_teacher_expanded32_v2_pilot \
LIMIT=8 \
VARIANTS_PER_PROMPT=12 \
SEEDS=0,1 \
bash tools/cloud_generate_sdxl_expanded_v2.sh
```

Inspect:

- `datasets/sdxl_cloud_teacher_expanded32_v2_pilot/contact_sheet.png`
- `datasets/sdxl_cloud_teacher_expanded32_v2_pilot/contact_sheet_validation_128.png`
- `datasets/sdxl_cloud_teacher_expanded32_v2_pilot/contact_sheet_category_sample_128.png`
- `datasets/sdxl_cloud_teacher_expanded32_v2_pilot/quality_report.json`

If the pilot looks good, run the full default:

```sh
bash tools/cloud_generate_sdxl_expanded_v2.sh
```

Defaults:

- 32 categories
- 12 variants per category
- 8 seeds
- 3072 planned images before quality rejects
- 20 steps
- 768x768 source
- 256/128/64 saved sizes
- batch size 2
- `MAX_BORDER_EDGE_DENSITY=0.45`
- `MAX_FOREGROUND_COMPONENTS=8`
- `MIN_LARGEST_FOREGROUND_COMPONENT_RATIO=0.45`

If too many good images are rejected, loosen the background texture gate:

```sh
MAX_BORDER_EDGE_DENSITY=0.55 bash tools/cloud_generate_sdxl_expanded_v2.sh
```

If too many single-subject images are rejected as fragmented foregrounds,
increase `MAX_FOREGROUND_COMPONENTS` to `10` before lowering the largest-component
ratio.

### Current expanded32 result

The first full RunPod run produced:

- Dataset: `datasets/sdxl_cloud_teacher_expanded32_v2`
- Planned rows: 3072
- Accepted rows: 2576
- Rejected rows: 496
- Local validation: 2576 valid images, 0 missing images

The automatic filters removed many patterned-background failures, but visual
review still found a small number of contact-sheet / repeated-object images.
For training, prefer a filtered subset rather than the full accepted set:

```sh
.venv/bin/python tools/filter_teacher_dataset.py \
  datasets/sdxl_cloud_teacher_expanded32_v2 \
  datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict \
  --exclude-seeds 7 \
  --exclude-variants v03,c02,a02 \
  --link-mode hardlink

.venv/bin/python tools/validate_teacher_dataset.py \
  datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict \
  --image-size 128 \
  --max-border-edge-density 0.45 \
  --max-foreground-components 8 \
  --min-largest-foreground-component-ratio 0.45 \
  --allow-invalid

.venv/bin/python tools/make_teacher_category_contact_sheet.py \
  datasets/sdxl_cloud_teacher_expanded32_v2_curated_strict \
  --image-size 128 \
  --samples-per-key 6
```

Current curated-strict result:

- Selected rows: 1730
- Categories: 32
- Min per category: 33
- Validation: 1730 valid images

Use `contact_sheet_category_sample_128.png` for final visual review before a
training run. If quality matters more than coverage, add explicit bad IDs to a
new filtered subset or regenerate only the affected category/variant pairs.

## Watch46 v3 run

Use this when the Watch UI presets need full controlled-teacher coverage. It
adds the categories missing from expanded32:

```text
fox, owl, banana, orange, strawberry, cake, pizza, bread,
mountain, cloud, clock, ball, guitar, shoe
```

Run a small pilot:

```sh
RUN_NAME=sdxl_cloud_teacher_watch46_v3_pilot \
LIMIT=12 \
SEEDS=0,1 \
bash tools/cloud_generate_sdxl_watch_v3.sh
```

If the pilot looks good, run the default:

```sh
bash tools/cloud_generate_sdxl_watch_v3.sh
```

Defaults:

- 46 categories
- 12 variants per category
- seeds `0,1,2,5`
- 2208 planned images before quality rejects
- 20 steps
- 768x768 source
- 256/128/64 saved sizes
- batch size 2

After downloading, create a balanced subset before training. Start with 24
images per key for the current hidden-1536 coordinate MLP:

```sh
.venv/bin/python tools/filter_teacher_dataset.py \
  datasets/sdxl_cloud_teacher_watch46_v3 \
  datasets/sdxl_cloud_teacher_watch46_v3_balanced24 \
  --max-per-key 24 \
  --max-per-key-strategy even \
  --link-mode hardlink
```

## Download results

From the Mac:

```sh
mkdir -p "/Users/tasuku/Documents/ちっちゃい画像生成モデル/datasets"
rsync -avz -e "ssh -p <PORT>" \
  root@<HOST>:/workspace/tiny-image-model/datasets/sdxl_cloud_teacher_fixed16_v1 \
  "/Users/tasuku/Documents/ちっちゃい画像生成モデル/datasets/"
```

Validate again locally:

```sh
cd "/Users/tasuku/Documents/ちっちゃい画像生成モデル"
.venv/bin/python tools/validate_teacher_dataset.py \
  datasets/sdxl_cloud_teacher_fixed16_v1 \
  --image-size 128
```

## Cost control

Always stop or terminate the pod after downloading results. Persistent storage
can continue billing even when the GPU is stopped, depending on provider.
