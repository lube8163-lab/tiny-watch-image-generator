#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import make_watch_eval_contact_sheet as eval_sheet


ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_eval_sheet(args: argparse.Namespace, out_dir: Path, raw: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "make_watch_eval_contact_sheet.py"),
        "--config",
        str(args.config),
        "--out-dir",
        str(out_dir),
        "--seeds",
        args.seeds,
        "--size",
        str(args.size),
        "--groups",
        args.groups,
        "--prompts-per-group",
        str(args.prompts_per_group),
        "--columns",
        str(args.columns),
        "--label-height",
        str(args.label_height),
        "--configuration",
        args.configuration,
    ]
    if args.cell is not None:
        cmd.extend(["--cell", str(args.cell)])
    if raw:
        cmd.append("--raw")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def entry_key(entry: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(entry["group"]),
        int(entry["promptIndex"]),
        str(entry["prompt"]),
        int(entry["seed"]),
    )


def paired_entries(raw_manifest: dict[str, Any], processed_manifest: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    processed_by_key = {entry_key(entry): entry for entry in eval_sheet.flatten_entries(processed_manifest)}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_entry in eval_sheet.flatten_entries(raw_manifest):
        key = entry_key(raw_entry)
        processed_entry = processed_by_key.get(key)
        if processed_entry is None:
            raise SystemExit(f"Missing processed entry for {key}")
        pairs.append((raw_entry, processed_entry))
    return pairs


def border_stats(image) -> dict[str, float]:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    border_width = max(2, min(8, min(width, height) // 16))
    pixels = rgb_image.load()

    sums = [0.0, 0.0, 0.0]
    count = 0
    for y in range(height):
        for x in range(width):
            if x < border_width or y < border_width or x >= width - border_width or y >= height - border_width:
                red, green, blue = pixels[x, y]
                sums[0] += red
                sums[1] += green
                sums[2] += blue
                count += 1

    if count == 0:
        return {"borderStd": 0.0, "borderEdgeDensity": 0.0}

    means = [value / count for value in sums]
    square_sum = 0.0
    edge_pairs = 0
    edge_hits = 0

    def luma(x: int, y: int) -> float:
        red, green, blue = pixels[x, y]
        return 0.299 * red + 0.587 * green + 0.114 * blue

    for y in range(height):
        for x in range(width):
            if not (x < border_width or y < border_width or x >= width - border_width or y >= height - border_width):
                continue
            red, green, blue = pixels[x, y]
            square_sum += (red - means[0]) ** 2 + (green - means[1]) ** 2 + (blue - means[2]) ** 2
            current_luma = luma(x, y)
            if x + 1 < width:
                edge_pairs += 1
                edge_hits += int(abs(current_luma - luma(x + 1, y)) >= 16.0)
            if y + 1 < height:
                edge_pairs += 1
                edge_hits += int(abs(current_luma - luma(x, y + 1)) >= 16.0)

    return {
        "borderStd": math.sqrt(square_sum / (count * 3.0)),
        "borderEdgeDensity": edge_hits / edge_pairs if edge_pairs else 0.0,
    }


def compute_metrics(raw_dir: Path, processed_dir: Path, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    Image, _, _ = eval_sheet.require_pillow()
    entries: list[dict[str, Any]] = []
    for raw_entry, processed_entry in pairs:
        raw_image = Image.open(eval_sheet.entry_image_path(raw_dir, raw_entry))
        processed_image = Image.open(eval_sheet.entry_image_path(processed_dir, processed_entry))
        raw_stats = border_stats(raw_image)
        processed_stats = border_stats(processed_image)
        raw_std = raw_stats["borderStd"]
        processed_std = processed_stats["borderStd"]
        entries.append(
            {
                "group": raw_entry["group"],
                "prompt": raw_entry["prompt"],
                "seed": raw_entry["seed"],
                "rawElapsedMs": raw_entry["elapsedMs"],
                "processedElapsedMs": processed_entry["elapsedMs"],
                "rawBorderStd": raw_std,
                "processedBorderStd": processed_std,
                "borderStdDelta": processed_std - raw_std,
                "rawBorderEdgeDensity": raw_stats["borderEdgeDensity"],
                "processedBorderEdgeDensity": processed_stats["borderEdgeDensity"],
            }
        )

    raw_mean = mean([entry["rawBorderStd"] for entry in entries])
    processed_mean = mean([entry["processedBorderStd"] for entry in entries])
    return {
        "summary": {
            "count": len(entries),
            "rawBorderStdMean": raw_mean,
            "processedBorderStdMean": processed_mean,
            "borderStdMeanDelta": processed_mean - raw_mean,
            "rawElapsedMsMean": mean([entry["rawElapsedMs"] for entry in entries]),
            "processedElapsedMsMean": mean([entry["processedElapsedMs"] for entry in entries]),
        },
        "entries": entries,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def draw_compare_sheet(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    raw_dir: Path,
    processed_dir: Path,
    out_path: Path,
    columns: int,
    cell: int,
    label_height: int,
) -> None:
    Image, ImageDraw, ImageFont = eval_sheet.require_pillow()
    if not pairs:
        return

    label_font = eval_sheet.load_label_font(ImageFont, size=11)
    columns = max(1, min(columns, len(pairs)))
    header_height = 16
    pair_width = cell * 2
    row_height = header_height + cell + label_height
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * pair_width, rows * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")

    for index, (raw_entry, processed_entry) in enumerate(pairs):
        x = (index % columns) * pair_width
        y = (index // columns) * row_height

        raw_image = Image.open(eval_sheet.entry_image_path(raw_dir, raw_entry)).convert("RGB")
        processed_image = Image.open(eval_sheet.entry_image_path(processed_dir, processed_entry)).convert("RGB")
        raw_image = raw_image.resize((cell, cell), resampling)
        processed_image = processed_image.resize((cell, cell), resampling)

        eval_sheet.draw_text_safe(draw, (x + 4, y + 2), "raw", fill=(80, 80, 80), font=label_font)
        eval_sheet.draw_text_safe(draw, (x + cell + 4, y + 2), "watchDenoise", fill=(80, 80, 80), font=label_font)
        sheet.paste(raw_image, (x, y + header_height))
        sheet.paste(processed_image, (x + cell, y + header_height))

        label_y = y + header_height + cell + 4
        label = eval_sheet.shorten_label(raw_entry["prompt"], 34)
        seed = f's{raw_entry["seed"]}'
        elapsed = f'{raw_entry["elapsedMs"]}->{processed_entry["elapsedMs"]}ms'
        eval_sheet.draw_text_safe(draw, (x + 4, label_y), label, fill=(0, 0, 0), font=label_font)
        eval_sheet.draw_text_safe(draw, (x + 4, label_y + 16), f"{seed} {elapsed}", fill=(80, 80, 80), font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_compare_sheets(args: argparse.Namespace, out_dir: Path, raw_dir: Path, processed_dir: Path) -> list[Path]:
    raw_manifest = eval_sheet.load_manifest(raw_dir)
    processed_manifest = eval_sheet.load_manifest(processed_dir)
    pairs = paired_entries(raw_manifest, processed_manifest)
    cell = args.cell or int(raw_manifest.get("size", 128))
    sheet_paths: list[Path] = []

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair[0]["group"]), []).append(pair)

    for group_key, group_pairs in grouped.items():
        path = out_dir / f"compare_{group_key}.png"
        draw_compare_sheet(group_pairs, raw_dir, processed_dir, path, args.columns, cell, args.label_height)
        sheet_paths.append(path)

    all_path = out_dir / "compare_all.png"
    draw_compare_sheet(pairs, raw_dir, processed_dir, all_path, args.columns, cell, args.label_height)
    sheet_paths.append(all_path)

    metrics = compute_metrics(raw_dir, processed_dir, pairs)
    metrics_path = out_dir / "compare_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    index_path = out_dir / "compare_index.json"
    index_path.write_text(
        json.dumps(
            {
                "rawManifest": str(raw_dir / "manifest.json"),
                "processedManifest": str(processed_dir / "manifest.json"),
                "metrics": str(metrics_path),
                "sheets": [str(path) for path in sheet_paths],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return sheet_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw vs watchDenoise comparison sheets.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "prompt_eval_suite.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--groups",
        default="core_nouns,adjectives,actions,styles,japanese_aliases,v6_new_subjects",
        help="Comma-separated eval group keys, or all.",
    )
    parser.add_argument("--prompts-per-group", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--cell", type=int, default=None)
    parser.add_argument("--label-height", type=int, default=38)
    parser.add_argument(
        "--configuration",
        choices=("release", "debug"),
        default="release",
        help="SwiftPM configuration for TinyWatchEval. Release is much faster.",
    )
    parser.add_argument("--skip-generate", action="store_true", help="Use existing raw/processed manifests in --out-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or ROOT / "reports" / "watch_postprocess_compare" / timestamp()
    out_dir = out_dir.resolve()
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "watchDenoise"

    if not args.skip_generate:
        run_eval_sheet(args, raw_dir, raw=True)
        run_eval_sheet(args, processed_dir, raw=False)

    sheet_paths = write_compare_sheets(args, out_dir, raw_dir, processed_dir)
    for path in sheet_paths:
        print(path)
    print(out_dir / "compare_metrics.json")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
