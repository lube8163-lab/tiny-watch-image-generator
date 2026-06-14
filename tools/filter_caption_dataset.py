#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from prompt_normalization import (
    ACTION_ALIASES,
    COLOR_ALIASES,
    MODIFIER_ALIASES,
    PROMPT_ALIASES,
    STYLE_ALIASES,
    canonicalize_prompt,
    normalize_prompt_text,
    tokenize_prompt,
)
from research_common import ROOT, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter mined caption metadata without copying image files.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-count", type=int, default=0)
    parser.add_argument("--min-caption-score", type=float, default=None)
    parser.add_argument("--min-quality-score", type=float, default=None)
    parser.add_argument("--max-negative-score", type=float, default=None)
    parser.add_argument("--require-known-subject", action="store_true")
    parser.add_argument("--require-compositional-signal", action="store_true")
    parser.add_argument("--exclude-caption-fragments", default="")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_fragments(value: str) -> list[str]:
    return [normalize_prompt_text(item) for item in value.split(",") if normalize_prompt_text(item)]


def has_alias(caption: str, aliases: dict[str, tuple[str, ...]]) -> bool:
    tokens = set(tokenize_prompt(caption))
    phrase = " ".join(tokenize_prompt(caption))
    compact_phrase = phrase.replace(" ", "")
    for key, words in aliases.items():
        for alias in (key, *words):
            alias_tokens = tokenize_prompt(alias)
            if alias_tokens and all(token in tokens for token in alias_tokens):
                return True
            if any(not token.isascii() for token in alias_tokens):
                compact_alias = normalize_prompt_text(alias).replace(" ", "")
                if compact_alias and compact_alias in compact_phrase:
                    return True
    return False


def reject_reason(row: dict[str, Any], args: argparse.Namespace, fragments: list[str]) -> str | None:
    if row.get("accepted") is not True:
        return "source_rejected"
    caption = normalize_prompt_text(row.get("caption") or row.get("prompt") or row.get("key") or "")
    if not caption:
        return "missing_caption"
    if any(fragment in caption for fragment in fragments):
        return "excluded_caption_fragment"
    if args.min_caption_score is not None and float(row.get("image_caption_score") or row.get("top1_score") or 0) < args.min_caption_score:
        return "low_caption_score"
    if args.min_quality_score is not None and float(row.get("quality_score") or 0) < args.min_quality_score:
        return "low_quality_score"
    if args.max_negative_score is not None and float(row.get("negative_score_max") or 0) > args.max_negative_score:
        return "high_negative_score"
    if args.require_known_subject and canonicalize_prompt(caption) not in PROMPT_ALIASES:
        return "missing_known_subject"
    if args.require_compositional_signal and not (
        has_alias(caption, COLOR_ALIASES)
        or has_alias(caption, ACTION_ALIASES)
        or has_alias(caption, MODIFIER_ALIASES)
        or has_alias(caption, STYLE_ALIASES)
    ):
        return "missing_compositional_signal"
    return None


def resolve_image_path(path_value: str) -> Path | None:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path, cell_size: int = 128, label_height: int = 56) -> None:
    if not rows:
        return
    columns = min(5, len(rows))
    sheet_rows = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * cell_size, sheet_rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        saved = row.get("saved_images") or {}
        image_ref = saved.get("128") or saved.get("256")
        if not image_ref:
            continue
        image_path = resolve_image_path(image_ref)
        if image_path is None:
            continue
        x = (index % columns) * cell_size
        y = (index // columns) * (cell_size + label_height)
        image = Image.open(image_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        caption = str(row.get("caption") or row.get("prompt") or row.get("key") or "")[:24]
        draw.text((x + 4, y + cell_size + 4), caption, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_root / "metadata.jsonl")
    fragments = split_fragments(args.exclude_caption_fragments)
    kept: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    for row in rows:
        reason = reject_reason(row, args, fragments)
        if reason is not None:
            reject_counts[reason] += 1
            continue
        output = dict(row)
        output["accepted"] = True
        output["filter_source_root"] = str(args.input_root)
        kept.append(output)
        caption_counts[str(output.get("caption") or output.get("prompt") or output.get("key") or "")] += 1
        if args.max_count and len(kept) >= args.max_count:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports_dir = args.out_dir / "reports"
    contact_sheet_path = reports_dir / "contact_sheet_filtered.png"
    make_contact_sheet(kept[:40], contact_sheet_path)
    summary = {
        "input_root": str(args.input_root),
        "metadata_count": len(kept),
        "source_count": len(rows),
        "reject_reasons": dict(reject_counts),
        "top_captions": caption_counts.most_common(40),
        "exclude_caption_fragments": fragments,
        "require_known_subject": args.require_known_subject,
        "require_compositional_signal": args.require_compositional_signal,
    }
    summary_path = reports_dir / "filter_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "phase": "filtered_caption_dataset",
            "created_at_unix": int(time.time()),
            "metadata_jsonl": str(metadata_path),
            "reports": {
                "filter_summary": str(summary_path),
                "contact_sheet_filtered": str(contact_sheet_path),
            },
            **summary,
        },
    )
    print(metadata_path)
    print(summary_path)
    print(contact_sheet_path)
    print(args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
