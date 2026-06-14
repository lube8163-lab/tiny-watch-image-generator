#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
    return rows, errors


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def image_stats(
    path: Path,
    border_margin_fraction: float,
    border_edge_threshold: float,
    foreground_threshold: float,
    foreground_min_component_area: int,
    foreground_sample_size: int,
) -> dict[str, float | int | list[int]]:
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)
    height, width, _ = pixels.shape
    margin = max(1, int(min(width, height) * border_margin_fraction))
    border_pixels = np.concatenate(
        [
            pixels[:margin, :, :].reshape(-1, 3),
            pixels[-margin:, :, :].reshape(-1, 3),
            pixels[:, :margin, :].reshape(-1, 3),
            pixels[:, -margin:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    gray = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    edge_x = np.abs(np.diff(gray, axis=1))
    edge_y = np.abs(np.diff(gray, axis=0))
    border_edges = np.concatenate(
        [
            edge_x[:margin, :].ravel(),
            edge_x[-margin:, :].ravel(),
            edge_x[:, :margin].ravel(),
            edge_x[:, -margin:].ravel(),
            edge_y[:margin, :].ravel(),
            edge_y[-margin:, :].ravel(),
            edge_y[:, :margin].ravel(),
            edge_y[:, -margin:].ravel(),
        ]
    )
    stats = {
        "size": list(image.size),
        "min": int(pixels.min()),
        "max": int(pixels.max()),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "border_margin": margin,
        "border_std": float(border_pixels.std()),
        "border_edge_mean": float(border_edges.mean()),
        "border_edge_density": float((border_edges > border_edge_threshold).mean()),
    }
    stats.update(
        compute_foreground_stats(
            image,
            foreground_threshold,
            foreground_min_component_area,
            foreground_sample_size,
        )
    )
    return stats


def compute_foreground_stats(
    image: Image.Image,
    threshold: float,
    min_component_area: int,
    sample_size: int,
) -> dict[str, float | int]:
    sample = image.convert("RGB").resize((sample_size, sample_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.float32)
    margin = max(1, int(sample_size * 0.12))
    border = np.concatenate(
        [
            pixels[:margin, :, :].reshape(-1, 3),
            pixels[-margin:, :, :].reshape(-1, 3),
            pixels[:, :margin, :].reshape(-1, 3),
            pixels[:, -margin:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2)
    mask = distance > threshold
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    component_areas: list[int] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                current_y, current_x = stack.pop()
                area += 1
                for next_y, next_x in (
                    (current_y + 1, current_x),
                    (current_y - 1, current_x),
                    (current_y, current_x + 1),
                    (current_y, current_x - 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            if area >= min_component_area:
                component_areas.append(area)

    foreground_area = int(sum(component_areas))
    largest_area = int(max(component_areas, default=0))
    largest_ratio = float(largest_area / foreground_area) if foreground_area else 1.0
    return {
        "foreground_sample_size": sample_size,
        "foreground_threshold": float(threshold),
        "foreground_component_count": len(component_areas),
        "foreground_area": foreground_area,
        "largest_foreground_component_area": largest_area,
        "largest_foreground_component_ratio": largest_ratio,
    }


def reject_reason(
    stats: dict[str, Any],
    expected_size: int,
    min_range: int,
    min_std: float,
    max_border_std: float,
    max_border_edge_density: float,
    max_foreground_components: int,
    min_largest_foreground_component_ratio: float,
) -> str | None:
    if stats["size"] != [expected_size, expected_size]:
        return f"bad_size:{stats['size']}!=[{expected_size},{expected_size}]"
    image_range = int(stats["max"]) - int(stats["min"])
    if image_range < min_range:
        return f"low_dynamic_range:{image_range}<min_range:{min_range}"
    if float(stats["std"]) < min_std:
        return f"low_std:{stats['std']:.4f}<min_std:{min_std:.4f}"
    if max_border_std > 0 and float(stats["border_std"]) > max_border_std:
        return f"high_border_std:{stats['border_std']:.4f}>max_border_std:{max_border_std:.4f}"
    if max_border_edge_density > 0 and float(stats["border_edge_density"]) > max_border_edge_density:
        return (
            "high_border_edge_density:"
            f"{stats['border_edge_density']:.4f}>max_border_edge_density:{max_border_edge_density:.4f}"
        )
    if max_foreground_components > 0 and int(stats["foreground_component_count"]) > max_foreground_components:
        return (
            "too_many_foreground_components:"
            f"{stats['foreground_component_count']}>max_foreground_components:{max_foreground_components}"
        )
    if (
        min_largest_foreground_component_ratio > 0
        and float(stats["largest_foreground_component_ratio"]) < min_largest_foreground_component_ratio
    ):
        return (
            "low_largest_foreground_component_ratio:"
            f"{stats['largest_foreground_component_ratio']:.4f}"
            f"<min_largest_foreground_component_ratio:{min_largest_foreground_component_ratio:.4f}"
        )
    return None


def make_contact_sheet(rows: list[dict[str, Any]], root: Path, out_path: Path, image_size: int, limit: int) -> None:
    selected = rows[:limit] if limit > 0 else rows
    if not selected:
        return
    columns = min(8, len(selected))
    label_height = 34
    row_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * image_size, row_count * (image_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        path = resolve_path(root, row.get("validation_image") or row.get("image"))
        if path is None or not path.exists():
            continue
        image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
        x = (index % columns) * image_size
        y = (index // columns) * (image_size + label_height)
        sheet.paste(image, (x, y))
        label = f'{row.get("key")} {row.get("variant")} s{row.get("seed")}'
        draw.text((x + 4, y + image_size + 6), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * fraction))))
        return sorted_values[index]

    return {
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": sorted_values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate teacher dataset metadata and saved images.")
    parser.add_argument("dataset_root")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-image-range", type=int, default=8)
    parser.add_argument("--min-image-std", type=float, default=2.0)
    parser.add_argument("--border-margin-fraction", type=float, default=0.12)
    parser.add_argument("--border-edge-threshold", type=float, default=20.0)
    parser.add_argument("--max-border-std", type=float, default=0.0)
    parser.add_argument("--max-border-edge-density", type=float, default=0.0)
    parser.add_argument("--foreground-threshold", type=float, default=30.0)
    parser.add_argument("--foreground-min-component-area", type=int, default=20)
    parser.add_argument("--foreground-sample-size", type=int, default=128)
    parser.add_argument("--max-foreground-components", type=int, default=0)
    parser.add_argument("--min-largest-foreground-component-ratio", type=float, default=0.0)
    parser.add_argument("--contact-limit", type=int, default=128)
    parser.add_argument("--contact-size", type=int, default=128)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    metadata_path = root / "metadata.jsonl"
    rows, json_errors = load_jsonl(metadata_path)
    accepted_rows = [row for row in rows if row.get("accepted", True) is True]
    rejected_rows = [row for row in rows if row.get("accepted", True) is not True]
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []

    for row in accepted_rows:
        saved = row.get("saved_images") or {}
        rel = saved.get(str(args.image_size)) or row.get("image")
        path = resolve_path(root, rel)
        if path is None or not path.exists():
            missing.append({"id": row.get("id"), "path": rel})
            continue
        stats = image_stats(
            path,
            args.border_margin_fraction,
            args.border_edge_threshold,
            args.foreground_threshold,
            args.foreground_min_component_area,
            args.foreground_sample_size,
        )
        reason = reject_reason(
            stats,
            args.image_size,
            args.min_image_range,
            args.min_image_std,
            args.max_border_std,
            args.max_border_edge_density,
            args.max_foreground_components,
            args.min_largest_foreground_component_ratio,
        )
        row = dict(row)
        row["validation_image"] = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        row["validation_stats"] = stats
        if reason:
            row["validation_reject_reason"] = reason
            invalid.append(row)
        else:
            valid_rows.append(row)

    border_edge_values = [
        float(row["validation_stats"]["border_edge_density"])
        for row in valid_rows
        if row.get("validation_stats")
    ]
    border_std_values = [
        float(row["validation_stats"]["border_std"])
        for row in valid_rows
        if row.get("validation_stats")
    ]
    foreground_component_values = [
        float(row["validation_stats"]["foreground_component_count"])
        for row in valid_rows
        if row.get("validation_stats")
    ]
    largest_foreground_ratio_values = [
        float(row["validation_stats"]["largest_foreground_component_ratio"])
        for row in valid_rows
        if row.get("validation_stats")
    ]

    report = {
        "dataset_root": str(root),
        "metadata_jsonl": str(metadata_path),
        "metadata_rows": len(rows),
        "accepted_metadata_rows": len(accepted_rows),
        "rejected_metadata_rows": len(rejected_rows),
        "json_errors": json_errors,
        "missing_images": missing,
        "invalid_images": [
            {
                "id": row.get("id"),
                "key": row.get("key"),
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "image": row.get("validation_image"),
                "reason": row.get("validation_reject_reason"),
                "stats": row.get("validation_stats"),
            }
            for row in invalid
        ],
        "valid_image_count": len(valid_rows),
        "border_edge_density": summarize_values(border_edge_values),
        "border_std": summarize_values(border_std_values),
        "foreground_component_count": summarize_values(foreground_component_values),
        "largest_foreground_component_ratio": summarize_values(largest_foreground_ratio_values),
        "keys": dict(sorted(Counter(row.get("key") for row in accepted_rows).items())),
        "variants": dict(sorted(Counter(row.get("variant") for row in accepted_rows).items())),
        "seeds": dict(sorted(Counter(str(row.get("seed")) for row in accepted_rows).items())),
    }

    report_path = root / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contact_path = root / f"contact_sheet_validation_{args.image_size}.png"
    make_contact_sheet(valid_rows, root, contact_path, args.contact_size, args.contact_limit)

    print(report_path)
    print(contact_path)
    print(json.dumps({k: report[k] for k in ["metadata_rows", "accepted_metadata_rows", "valid_image_count"]}, ensure_ascii=False))

    if not args.allow_invalid and (json_errors or missing or invalid):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
