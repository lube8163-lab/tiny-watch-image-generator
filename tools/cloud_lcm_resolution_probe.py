#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_presets(path: Path, keys: list[str] | None) -> list[dict[str, Any]]:
    presets = json.loads(path.read_text())
    if not keys:
        return presets
    by_key = {item["key"]: item for item in presets}
    missing = [key for key in keys if key not in by_key]
    if missing:
        known = ", ".join(by_key)
        raise SystemExit(f"unknown prompt keys {missing}; known keys: {known}")
    return [by_key[key] for key in keys]


def load_pipeline(model_id: str, dtype: str, local_files_only: bool):
    import torch
    from diffusers import LatentConsistencyModelPipeline

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    pipe = LatentConsistencyModelPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def make_contact_sheet(results: list[dict[str, Any]], columns: int, scale: int, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not results:
        return
    max_size = max(int(result["size"]) for result in results)
    tile = max_size * scale
    label_h = 36
    pad = 8
    columns = max(1, min(columns, len(results)))
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * pad, rows * (tile + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    for index, result in enumerate(results):
        col = index % columns
        row = index // columns
        x = pad + col * (tile + pad)
        y = pad + row * (tile + label_h + pad)
        label = f"{result['key']} s{result['seed']} {result['size']}px"
        draw.text((x, y), label[:34], fill=(0, 0, 0), font=font)
        draw.text((x, y + 13), f"{result['steps']}st g{result['guidance_scale']:g} {result['elapsed_seconds']:.1f}s", fill=(80, 80, 80), font=font)
        image = Image.open(result["path"]).convert("RGB")
        image = image.resize((int(result["size"]) * scale, int(result["size"]) * scale), resampling)
        dx = x + (tile - image.width) // 2
        dy = y + label_h + (tile - image.height) // 2
        sheet.paste(image, (dx, dy))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_downsample_sheet(results: list[dict[str, Any]], columns: int, scale: int, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not results:
        return
    tile = 64 * scale
    label_h = 36
    pad = 8
    columns = max(1, min(columns, len(results)))
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * pad, rows * (tile + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for index, result in enumerate(results):
        col = index % columns
        row = index // columns
        x = pad + col * (tile + pad)
        y = pad + row * (tile + label_h + pad)
        label = f"{result['key']} s{result['seed']} {result['size']}->64"
        draw.text((x, y), label[:34], fill=(0, 0, 0), font=font)
        draw.text((x, y + 13), f"{result['steps']}st g{result['guidance_scale']:g}", fill=(80, 80, 80), font=font)
        image = Image.open(result["path"]).convert("RGB")
        image = image.resize((64, 64), lanczos).resize((tile, tile), nearest)
        sheet.paste(image, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LCM Dreamshaper quality probes across output sizes on CUDA.")
    parser.add_argument("--model-id", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--presets", type=Path, default=ROOT / "configs" / "lcm64_watch_presets.json")
    parser.add_argument("--prompt-keys", nargs="*", default=["cat", "cat_face", "cat_logo", "orange_cat", "sitting_cat", "white_cat"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 22, 24, 25, 30, 31])
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 96, 128])
    parser.add_argument("--steps", nargs="+", type=int, default=[4])
    parser.add_argument("--guidance-scales", nargs="+", type=float, default=[8.0])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import torch

    presets = load_presets(args.presets, args.prompt_keys)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model_id, args.dtype, args.local_files_only)
    results: list[dict[str, Any]] = []
    for size in args.sizes:
        if size % 8 != 0:
            raise SystemExit(f"size must be divisible by 8, got {size}")
        for steps in args.steps:
            for guidance_scale in args.guidance_scales:
                for preset in presets:
                    for seed in args.seeds:
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        start = time.perf_counter()
                        with torch.inference_mode():
                            image = pipe(
                                prompt=preset["prompt"],
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                width=size,
                                height=size,
                                generator=generator,
                                output_type="pil",
                            ).images[0]
                        elapsed = time.perf_counter() - start
                        guidance_text = f"{guidance_scale:g}".replace(".", "p")
                        filename = f"{preset['key']}_seed{seed}_{size}px_{steps}st_g{guidance_text}.png"
                        path = image_dir / filename
                        image.save(path)
                        result = {
                            "key": preset["key"],
                            "title": preset["title"],
                            "prompt": preset["prompt"],
                            "seed": seed,
                            "size": size,
                            "steps": steps,
                            "guidance_scale": guidance_scale,
                            "elapsed_seconds": elapsed,
                            "path": str(path),
                        }
                        results.append(result)
                        print(
                            f"{preset['key']} seed={seed} size={size} steps={steps} "
                            f"guidance={guidance_scale:g} elapsed={elapsed:.2f}s",
                            flush=True,
                        )

    contact_sheet = args.out_dir / "contact_sheet.png"
    make_contact_sheet(results, args.columns, args.scale, contact_sheet)
    downsample_sheet = args.out_dir / "contact_sheet_downsampled64.png"
    save_downsample_sheet(results, args.columns, args.scale, downsample_sheet)
    manifest = {
        "phase": "cloud_lcm_resolution_probe",
        "model_id": args.model_id,
        "dtype": args.dtype,
        "presets": str(args.presets),
        "prompt_keys": [preset["key"] for preset in presets],
        "seeds": args.seeds,
        "sizes": args.sizes,
        "steps": args.steps,
        "guidance_scales": args.guidance_scales,
        "contact_sheet": str(contact_sheet),
        "downsample_sheet": str(downsample_sheet),
        "results": results,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(contact_sheet)
    print(downsample_sheet)


if __name__ == "__main__":
    main()
