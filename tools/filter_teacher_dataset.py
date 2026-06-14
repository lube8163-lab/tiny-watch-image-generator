#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_common import write_manifest


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_csv_int_set(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


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


def variant_id(row: dict[str, Any]) -> str:
    return str(row.get("variant_id") or row.get("variant") or "")


def row_paths(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["image", "validation_image", "source_image"]:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    saved_images = row.get("saved_images") or {}
    if isinstance(saved_images, dict):
        for value in saved_images.values():
            if isinstance(value, str) and value:
                values.append(value)
    return sorted(set(values))


def pick_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    indices = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    return [rows[index] for index in indices]


def image_quality_score(row: dict[str, Any]) -> float:
    stats = row.get("image_stats") or {}
    border_std = float(stats.get("border_std") or 0.0)
    border_edge_density = float(stats.get("border_edge_density") or 0.0)
    foreground_components = float(stats.get("foreground_component_count") or 1.0)
    largest_component_ratio = float(stats.get("largest_foreground_component_ratio") or 1.0)

    score = 0.0
    score -= border_std / 50.0
    score -= border_edge_density * 20.0
    score -= max(0.0, foreground_components - 1.0) * 0.15
    score -= max(0.0, 0.95 - largest_component_ratio) * 2.0
    return score


def apply_max_per_key(rows: list[dict[str, Any]], max_per_key: int, strategy: str) -> list[dict[str, Any]]:
    if max_per_key <= 0:
        return rows
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get("key") or "unknown")].append(row)

    selected_ids: set[int] = set()
    for items in by_key.values():
        if strategy == "first":
            picked = items[:max_per_key]
        elif strategy == "quality":
            picked = sorted(items, key=image_quality_score, reverse=True)[:max_per_key]
        elif strategy == "quality_diverse":
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in items:
                group_key = str(row.get("prompt") or variant_id(row) or row.get("id") or "unknown")
                grouped[group_key].append(row)
            for group_rows in grouped.values():
                group_rows.sort(key=image_quality_score, reverse=True)
            ordered_groups = sorted(
                grouped.values(),
                key=lambda group_rows: image_quality_score(group_rows[0]),
                reverse=True,
            )
            picked = []
            while ordered_groups and len(picked) < max_per_key:
                next_groups: list[list[dict[str, Any]]] = []
                for group_rows in ordered_groups:
                    if len(picked) >= max_per_key:
                        next_groups.append(group_rows)
                        continue
                    picked.append(group_rows.pop(0))
                    if group_rows:
                        next_groups.append(group_rows)
                ordered_groups = next_groups
        else:
            picked = pick_evenly(items, max_per_key)
        selected_ids.update(id(row) for row in picked)
    return [row for row in rows if id(row) in selected_ids]


def should_keep(
    row: dict[str, Any],
    include_rejected: bool,
    include_ids: set[str],
    exclude_ids: set[str],
    include_seeds: set[int],
    exclude_seeds: set[int],
    include_variants: set[str],
    exclude_variants: set[str],
    include_keys: set[str],
    exclude_keys: set[str],
) -> bool:
    if not include_rejected and row.get("accepted") is not True:
        return False

    seed = row.get("seed")
    row_id = str(row.get("id") or "")
    row_variant = variant_id(row)
    key = str(row.get("key") or "")

    if include_ids and row_id not in include_ids:
        return False
    if row_id in exclude_ids:
        return False
    if include_seeds and seed not in include_seeds:
        return False
    if seed in exclude_seeds:
        return False
    if include_variants and row_variant not in include_variants:
        return False
    if row_variant in exclude_variants:
        return False
    if include_keys and key not in include_keys:
        return False
    if key in exclude_keys:
        return False
    return True


def materialize_file(src_root: Path, out_root: Path, relpath: str, mode: str) -> None:
    src = resolve_path(src_root, relpath)
    if src is None or not src.exists():
        raise FileNotFoundError(f"missing source file for metadata path: {relpath}")

    dest = resolve_path(out_root, relpath)
    if dest is None:
        raise RuntimeError(f"invalid destination path: {relpath}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return
        dest.unlink()

    if mode == "copy":
        shutil.copy2(src, dest)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dest)
        return

    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a filtered teacher dataset subset from metadata.jsonl.")
    parser.add_argument("source_root")
    parser.add_argument("out_root")
    parser.add_argument("--exclude-ids", default="")
    parser.add_argument("--include-ids", default="")
    parser.add_argument("--exclude-seeds", default="")
    parser.add_argument("--include-seeds", default="")
    parser.add_argument("--exclude-variants", default="")
    parser.add_argument("--include-variants", default="")
    parser.add_argument("--exclude-keys", default="")
    parser.add_argument("--include-keys", default="")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--max-per-key", type=int, default=0)
    parser.add_argument("--max-per-key-strategy", choices=["first", "even", "quality", "quality_diverse"], default="even")
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    metadata_path = source_root / "metadata.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"metadata not found: {metadata_path}")
    if source_root == out_root:
        raise SystemExit("source_root and out_root must be different")
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty; pass --overwrite to reuse it: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    include_ids = parse_csv_set(args.include_ids)
    exclude_ids = parse_csv_set(args.exclude_ids)
    include_seeds = parse_csv_int_set(args.include_seeds)
    exclude_seeds = parse_csv_int_set(args.exclude_seeds)
    include_variants = parse_csv_set(args.include_variants)
    exclude_variants = parse_csv_set(args.exclude_variants)
    include_keys = parse_csv_set(args.include_keys)
    exclude_keys = parse_csv_set(args.exclude_keys)

    rows = load_jsonl(metadata_path)
    selected = [
        row
        for row in rows
        if should_keep(
            row,
            args.include_rejected,
            include_ids,
            exclude_ids,
            include_seeds,
            exclude_seeds,
            include_variants,
            exclude_variants,
            include_keys,
            exclude_keys,
        )
    ]
    selected = apply_max_per_key(selected, args.max_per_key, args.max_per_key_strategy)
    if not selected:
        raise SystemExit("no rows selected")

    unique_paths: set[str] = set()
    for row in selected:
        unique_paths.update(row_paths(row))
    for relpath in sorted(unique_paths):
        materialize_file(source_root, out_root, relpath, args.link_mode)

    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    key_counts = Counter(str(row.get("key") or "unknown") for row in selected)
    variant_counts = Counter(variant_id(row) for row in selected)
    seed_counts = Counter(str(row.get("seed")) for row in selected)
    write_manifest(
        out_root / "manifest.json",
        {
            "source_dataset": str(source_root),
            "source_metadata_rows": len(rows),
            "selected_rows": len(selected),
            "materialized_files": len(unique_paths),
            "link_mode": args.link_mode,
            "filters": {
                "include_rejected": args.include_rejected,
                "include_ids": sorted(include_ids),
                "exclude_ids": sorted(exclude_ids),
                "include_seeds": sorted(include_seeds),
                "exclude_seeds": sorted(exclude_seeds),
                "include_variants": sorted(include_variants),
                "exclude_variants": sorted(exclude_variants),
                "include_keys": sorted(include_keys),
                "exclude_keys": sorted(exclude_keys),
                "max_per_key": args.max_per_key,
                "max_per_key_strategy": args.max_per_key_strategy,
            },
            "keys": dict(sorted(key_counts.items())),
            "variants": dict(sorted(variant_counts.items())),
            "seeds": dict(sorted(seed_counts.items())),
        },
    )
    print(f"selected_rows={len(selected)}")
    print(f"materialized_files={len(unique_paths)}")
    print(f"keys={len(key_counts)} min_per_key={min(key_counts.values())} max_per_key={max(key_counts.values())}")
    print(out_root)


if __name__ == "__main__":
    main()
