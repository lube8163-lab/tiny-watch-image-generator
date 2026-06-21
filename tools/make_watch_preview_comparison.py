#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def upscale_2x_bilinear_like_watch(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height, width, _ = rgba.shape
    output = np.empty((height * 2, width * 2, 4), dtype=np.uint8)

    for y in range(height * 2):
        source_y = min(height - 1, y // 2)
        next_y = min(height - 1, source_y + 1)
        fy = 0.0 if y % 2 == 0 else 0.5
        for x in range(width * 2):
            source_x = min(width - 1, x // 2)
            next_x = min(width - 1, source_x + 1)
            fx = 0.0 if x % 2 == 0 else 0.5
            top = rgba[source_y, source_x] + (rgba[source_y, next_x] - rgba[source_y, source_x]) * fx
            bottom = rgba[next_y, source_x] + (rgba[next_y, next_x] - rgba[next_y, source_x]) * fx
            output[y, x] = np.clip(np.rint(top + (bottom - top) * fy), 0, 255).astype(np.uint8)

    output[:, :, 3] = 255
    return Image.fromarray(output, "RGBA")


def unsharp_like_watch(image: Image.Image, amount: float = 0.65) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height, width, _ = rgba.shape
    output = rgba.copy()

    for y in range(height):
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        for x in range(width):
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)
            blurred = rgba[y0:y1, x0:x1, :3].mean(axis=(0, 1))
            output[y, x, :3] = rgba[y, x, :3] + (rgba[y, x, :3] - blurred) * amount

    output[:, :, :3] = np.clip(np.rint(output[:, :, :3]), 0, 255)
    output[:, :, 3] = 255
    return Image.fromarray(output.astype(np.uint8), "RGBA")


def catmull_rom(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * b
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
        + (-a + 3.0 * b - 3.0 * c + d) * t3
    )


def upscale_2x_bicubic_like_watch(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    height, width, _ = rgba.shape
    output = np.empty((height * 2, width * 2, 4), dtype=np.uint8)

    for y in range(height * 2):
        source_y = y / 2.0
        base_y = int(np.floor(source_y))
        fy = source_y - base_y
        for x in range(width * 2):
            source_x = x / 2.0
            base_x = int(np.floor(source_x))
            fx = source_x - base_x
            rows = []
            for yy in range(-1, 3):
                py = min(height - 1, max(0, base_y + yy))
                samples = [
                    rgba[py, min(width - 1, max(0, base_x + xx)), :3]
                    for xx in range(-1, 3)
                ]
                rows.append(catmull_rom(samples[0], samples[1], samples[2], samples[3], fx))
            output[y, x, :3] = np.clip(np.rint(catmull_rom(rows[0], rows[1], rows[2], rows[3], fy)), 0, 255)
            output[y, x, 3] = 255

    return Image.fromarray(output, "RGBA")


def make_variants(path: Path) -> dict[str, Image.Image]:
    image = Image.open(path).convert("RGBA")
    return {
        "raw 128": image,
        "crisp 2x": image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST),
        "smooth 2x": image.resize((image.width * 2, image.height * 2), Image.Resampling.BILINEAR),
        "sharp old": unsharp_like_watch(upscale_2x_bilinear_like_watch(image), amount=0.65),
        "sharp new": unsharp_like_watch(upscale_2x_bicubic_like_watch(image), amount=0.45),
    }


def draw_sheet(inputs: list[Path], output_path: Path) -> None:
    variants_by_input = [(path, make_variants(path)) for path in inputs]
    columns = 5
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
    parser = argparse.ArgumentParser(description="Compare Watch preview interpolation/postprocess modes.")
    parser.add_argument("--input", nargs="+", required=True, help="Input PNG images.")
    parser.add_argument("--out", required=True, help="Output contact sheet path.")
    args = parser.parse_args()

    draw_sheet([Path(item) for item in args.input], Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
