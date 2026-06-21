#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_watch_preview_comparison import upscale_2x_bicubic_like_watch, unsharp_like_watch


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


def safe_token(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "item"


def pil_to_tensor(torch, image: Image.Image, device) -> Any:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).to(device=device)


def tensor_to_pil(torch, tensor) -> Image.Image:
    arr = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.rint(arr * 255).astype(np.uint8), "RGB")


def psnr(mse: float) -> float:
    return 99.0 if mse <= 0 else float(-10.0 * math.log10(mse))


class TinySRNet:
    @staticmethod
    def build(torch, channels: int, blocks: int):
        import torch.nn as nn

        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(channels, 3 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            ]
        )
        return nn.Sequential(*layers)


def generate_pairs(args, presets: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    import torch
    from diffusers import LatentConsistencyModelPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = LatentConsistencyModelPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    target_dir = out_dir / "targets_256"
    input_dir = out_dir / "inputs_128"
    target_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    for preset in presets:
        for seed in args.seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(
                    prompt=preset["prompt"],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    width=args.target_size,
                    height=args.target_size,
                    generator=generator,
                    output_type="pil",
                ).images[0].convert("RGB")
            elapsed = time.perf_counter() - start
            stem = f"{safe_token(preset['key'])}_seed{seed:04d}"
            target_path = target_dir / f"{stem}.png"
            input_path = input_dir / f"{stem}.png"
            image.save(target_path)
            image.resize((args.input_size, args.input_size), Image.Resampling.LANCZOS).save(input_path)
            pairs.append(
                {
                    "key": preset["key"],
                    "title": preset.get("title") or preset["key"],
                    "prompt": preset["prompt"],
                    "seed": seed,
                    "input": str(input_path),
                    "target": str(target_path),
                    "elapsed_seconds": elapsed,
                }
            )
            print(f"pair {preset['key']} seed={seed} elapsed={elapsed:.2f}s", flush=True)
    return pairs


def train_sr(args, pairs: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    rng = random.Random(args.split_seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_fraction)))
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])

    inputs = torch.stack([pil_to_tensor(torch, Image.open(item["input"]), device) for item in pairs])
    targets = torch.stack([pil_to_tensor(torch, Image.open(item["target"]), device) for item in pairs])

    model = TinySRNet.build(torch, args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_tensor_indices = torch.tensor(train_indices, device=device)
    val_tensor_indices = torch.tensor(val_indices, device=device)
    start = time.perf_counter()
    losses: list[dict[str, float]] = []
    for step in range(1, args.train_steps + 1):
        batch_positions = torch.randint(0, len(train_indices), (args.batch_size,), device=device)
        batch_indices = train_tensor_indices[batch_positions]
        prediction = model(inputs[batch_indices]).clamp(0, 1)
        loss = F.l1_loss(prediction, targets[batch_indices]) + args.mse_weight * F.mse_loss(prediction, targets[batch_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.progress_every == 0 or step == args.train_steps:
            with torch.no_grad():
                val_prediction = model(inputs[val_tensor_indices]).clamp(0, 1)
                val_mse = F.mse_loss(val_prediction, targets[val_tensor_indices]).item()
                sharp_images = [
                    pil_to_tensor(
                        torch,
                        unsharp_like_watch(upscale_2x_bicubic_like_watch(Image.open(pairs[index]["input"])), amount=0.45).convert("RGB"),
                        device,
                    )
                    for index in val_indices
                ]
                sharp = torch.stack(sharp_images)
                sharp_mse = F.mse_loss(sharp, targets[val_tensor_indices]).item()
            report = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "val_mse": float(val_mse),
                "val_psnr": psnr(val_mse),
                "sharp_mse": float(sharp_mse),
                "sharp_psnr": psnr(sharp_mse),
            }
            losses.append(report)
            print(
                f"step={step} loss={loss.item():.5f} "
                f"val_psnr={report['val_psnr']:.2f} sharp_psnr={report['sharp_psnr']:.2f}",
                flush=True,
            )

    model_path = out_dir / "tiny_sr_probe.pt"
    torch.save(
        {
            "model": model.cpu().eval().state_dict(),
            "channels": args.channels,
            "blocks": args.blocks,
            "input_size": args.input_size,
            "target_size": args.target_size,
        },
        model_path,
    )
    return {
        "model_path": str(model_path),
        "elapsed_seconds": time.perf_counter() - start,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "losses": losses,
    }


def render_comparison(args, pairs: list[dict[str, Any]], train_report: dict[str, Any], out_dir: Path) -> Path:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(train_report["model_path"], map_location=device)
    model = TinySRNet.build(torch, checkpoint["channels"], checkpoint["blocks"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sample_indices = train_report["val_indices"][: args.sheet_samples]
    if len(sample_indices) < args.sheet_samples:
        sample_indices += train_report["train_indices"][: args.sheet_samples - len(sample_indices)]

    columns = 4
    tile = args.target_size
    label_h = 34
    pad = 8
    rows = len(sample_indices)
    sheet = Image.new("RGB", (columns * tile + (columns + 1) * pad, rows * (tile + label_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    with torch.no_grad():
        for row, index in enumerate(sample_indices):
            item = pairs[index]
            low = Image.open(item["input"]).convert("RGB")
            target = Image.open(item["target"]).convert("RGB")
            sharp = unsharp_like_watch(upscale_2x_bicubic_like_watch(low), amount=0.45).convert("RGB")
            pred = tensor_to_pil(torch, model(pil_to_tensor(torch, low, device).unsqueeze(0))[0])
            variants = [
                ("input sharp", sharp),
                ("tiny SR", pred),
                ("target 256", target),
                (f"{item['key']} s{item['seed']}", low.resize((tile, tile), Image.Resampling.NEAREST)),
            ]
            for col, (label, image) in enumerate(variants):
                x = pad + col * (tile + pad)
                y = pad + row * (tile + label_h + pad)
                draw.text((x, y), label[:28], fill=(0, 0, 0), font=font)
                sheet.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, y + label_h))

    path = out_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny 128->256 SR probe on LCM outputs.")
    parser.add_argument("--model-id", default="SimianLuo/LCM_Dreamshaper_v7")
    parser.add_argument("--presets", type=Path, default=ROOT / "configs" / "lcm128_watch_plus_presets.json")
    parser.add_argument("--prompt-keys", nargs="*", default=["cat_mascot", "lucky_cat", "cat_sticker", "white_mascot", "tabby_icon", "orange_cat", "cat_logo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--split-seed", type=int, default=260620)
    parser.add_argument("--progress-every", type=int, default=150)
    parser.add_argument("--sheet-samples", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    presets = load_presets(args.presets, args.prompt_keys)
    pairs = generate_pairs(args, presets, args.out_dir)
    train_report = train_sr(args, pairs, args.out_dir)
    contact_sheet = render_comparison(args, pairs, train_report, args.out_dir)
    manifest = {
        "phase": "cloud_train_lcm_sr_probe",
        "model_id": args.model_id,
        "presets": str(args.presets),
        "prompt_keys": [preset["key"] for preset in presets],
        "seeds": args.seeds,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "input_size": args.input_size,
        "target_size": args.target_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "train_steps": args.train_steps,
        "pairs": pairs,
        "train_report": train_report,
        "contact_sheet": str(contact_sheet),
    }
    write_path = args.out_dir / "manifest.json"
    write_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(contact_sheet)
    print(write_path)


if __name__ == "__main__":
    main()
