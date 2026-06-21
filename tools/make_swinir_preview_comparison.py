#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from make_watch_preview_comparison import unsharp_like_watch, upscale_2x_bicubic_like_watch


ROOT = Path(__file__).resolve().parents[1]
SD_DESKTOP = ROOT / "SD15TeacherDatasetDesktop"
sys.path.insert(0, str(SD_DESKTOP / "scripts"))

from convert_swinir_sr import load_model  # noqa: E402


def load_swinir() -> torch.nn.Module:
    repo = SD_DESKTOP / ".build" / "SwinIR"
    weights = SD_DESKTOP / "artifacts" / "upscalers" / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    return load_model(repo, weights).eval()


def apply_swinir(model: torch.nn.Module, image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(output * 255).astype(np.uint8), "RGB")


def make_variants(model: torch.nn.Module, path: Path) -> dict[str, Image.Image]:
    image = Image.open(path).convert("RGB")
    return {
        "raw 128": image,
        "smooth 2x": image.resize((image.width * 2, image.height * 2), Image.Resampling.BILINEAR),
        "sharp x2": unsharp_like_watch(upscale_2x_bicubic_like_watch(image), amount=0.45).convert("RGB"),
        "swinir x2": apply_swinir(model, image),
    }


def draw_sheet(inputs: list[Path], output_path: Path) -> None:
    model = load_swinir()
    variants_by_input = [(path, make_variants(model, path)) for path in inputs]
    columns = 4
    tile = 256
    label_h = 32
    pad = 10
    width = columns * tile + (columns + 1) * pad
    height = len(inputs) * (tile + label_h) + (len(inputs) + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row, (path, variants) in enumerate(variants_by_input):
        y = pad + row * (tile + label_h + pad)
        for col, (label, image) in enumerate(variants.items()):
            x = pad + col * (tile + pad)
            draw.text((x, y), f"{path.parent.name}/{path.name}", fill=(0, 0, 0))
            draw.text((x, y + 14), label, fill=(80, 80, 80))
            preview = image.convert("RGB")
            if preview.size != (tile, tile):
                preview = preview.resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(preview, (x, y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Watch Sharp x2 preview with PyTorch SwinIR x2.")
    parser.add_argument("--input", nargs="+", required=True, help="Input PNG images.")
    parser.add_argument("--out", required=True, help="Output contact sheet path.")
    args = parser.parse_args()

    draw_sheet([Path(item) for item in args.input], Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
