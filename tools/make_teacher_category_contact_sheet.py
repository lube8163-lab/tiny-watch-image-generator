#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def pick_indices(count: int, samples_per_key: int) -> list[int]:
    if count <= 0:
        return []
    if samples_per_key <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (samples_per_key - 1)) for i in range(samples_per_key)})


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a category-balanced contact sheet for a teacher dataset.")
    parser.add_argument("dataset_root")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--samples-per-key", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    rows = [row for row in load_jsonl(root / "metadata.jsonl") if row.get("accepted", True) is True]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get("key") or "unknown")].append(row)

    keys = sorted(by_key)
    if not keys:
        raise SystemExit("no accepted rows found")

    label_height = 18
    font = load_font(11)
    width = args.samples_per_key * args.thumb_size
    height = len(keys) * (args.thumb_size + label_height)
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    for row_index, key in enumerate(keys):
        items = by_key[key]
        for col_index, item_index in enumerate(pick_indices(len(items), args.samples_per_key)):
            row = items[item_index]
            saved = row.get("saved_images") or {}
            image_path = resolve_path(root, saved.get(str(args.image_size)) or row.get("image"))
            if image_path is None or not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB").resize((args.thumb_size, args.thumb_size), Image.Resampling.LANCZOS)
            x = col_index * args.thumb_size
            y = row_index * (args.thumb_size + label_height)
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + args.thumb_size + 2), f"{key} #{item_index}", fill=(20, 20, 20), font=font)

    out_path = Path(args.out) if args.out else root / f"contact_sheet_category_sample_{args.image_size}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
