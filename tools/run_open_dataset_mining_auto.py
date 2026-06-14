#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from research_common import ROOT, write_manifest


PRIMARY_KEYS = [
    "apple",
    "bird",
    "car",
    "castle",
    "cat",
    "dog",
    "face",
    "fish",
    "flower",
    "house",
    "moon",
    "robot",
    "star",
    "sun",
    "train",
    "tree",
]


DEFAULT_DATASET = "vikhyatk/openimages-bbox"
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_OUT_ROOT = ROOT / "datasets" / "open_mined_siglip2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Open Images + SigLIP2 mining shards and build one mixed-teacher-ready output."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-license", default="CC BY 2.0")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--target-per-class", type=int, default=20)
    parser.add_argument("--images-per-shard", type=int, default=512)
    parser.add_argument("--max-shards", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--min-center-score", type=float, default=0.02)
    parser.add_argument("--min-center-margin", type=float, default=0.005)
    parser.add_argument("--max-negative-score", type=float, default=0.09)
    parser.add_argument("--metadata-bonus", type=float, default=0.03)
    parser.add_argument("--metadata-prefilter", action="store_true")
    parser.add_argument("--max-source-rows-per-shard", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_id() -> str:
    return time.strftime("auto_openimages_bbox_%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mine_command(args: argparse.Namespace, shard_dir: Path, shard_index: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "mine_dataset_with_siglip2.py"),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--dataset-license",
        args.dataset_license,
        "--model",
        args.model,
        "--skip-images",
        str(shard_index * args.images_per_shard),
        "--max-images",
        str(args.images_per_shard),
        "--top-k-per-class",
        str(args.target_per_class),
        "--threshold",
        str(args.threshold),
        "--min-margin",
        str(args.min_margin),
        "--min-center-score",
        str(args.min_center_score),
        "--min-center-margin",
        str(args.min_center_margin),
        "--max-negative-score",
        str(args.max_negative_score),
        "--metadata-bonus",
        str(args.metadata_bonus),
        "--batch-size",
        str(args.batch_size),
        "--out-dir",
        str(shard_dir),
        "--force-exit",
    ]
    if args.metadata_prefilter:
        command.append("--metadata-prefilter")
    if args.max_source_rows_per_shard:
        command.extend(["--max-source-rows", str((shard_index + 1) * args.max_source_rows_per_shard)])
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def collect_shard_rows(shards_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(shards_dir.glob("shard_*/metadata.jsonl")):
        shard_name = metadata_path.parent.name
        for row in read_jsonl(metadata_path):
            row = dict(row)
            row["_shard"] = shard_name
            rows.append(row)
    return rows


def selected_ids(rows: list[dict[str, Any]], target_per_class: int) -> set[str]:
    selected: set[str] = set()
    for key in PRIMARY_KEYS:
        accepted = [row for row in rows if row.get("accepted") and row.get("key") == key]
        accepted.sort(key=lambda row: (row.get("top1_score", 0.0), row.get("score_margin", 0.0)), reverse=True)
        for row in accepted[:target_per_class]:
            selected.add(row["_aggregate_id"])
    return selected


def add_aggregate_ids(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for row in rows:
        base = f'{row.get("_shard", "shard")}_{row.get("id", "row")}'
        index = seen.get(base, 0)
        seen[base] = index + 1
        row["_aggregate_id"] = base if index == 0 else f"{base}_{index}"


def copy_selected_image(row: dict[str, Any], images_128_dir: Path, images_256_dir: Path) -> dict[str, str] | None:
    saved = row.get("saved_images") or {}
    src_128 = saved.get("128")
    src_256 = saved.get("256")
    if not src_128 or not src_256:
        return None
    src_128_path = Path(src_128)
    src_256_path = Path(src_256)
    if not src_128_path.is_absolute():
        src_128_path = ROOT / src_128_path
    if not src_256_path.is_absolute():
        src_256_path = ROOT / src_256_path
    key = row.get("key", "sample")
    stem = safe_name(f'{key}_{row["_aggregate_id"]}')
    dst_128 = images_128_dir / f"{stem}.png"
    dst_256 = images_256_dir / f"{stem}.png"
    images_128_dir.mkdir(parents=True, exist_ok=True)
    images_256_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_128_path, dst_128)
    shutil.copy2(src_256_path, dst_256)
    return {"128": str(dst_128), "256": str(dst_256)}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "sample"


def build_combined_output(run_dir: Path, target_per_class: int, args: argparse.Namespace) -> dict[str, Any]:
    shards_dir = run_dir / "_shards"
    reports_dir = run_dir / "reports"
    images_128_dir = run_dir / "images_128"
    images_256_dir = run_dir / "images_256"
    if images_128_dir.exists():
        shutil.rmtree(images_128_dir)
    if images_256_dir.exists():
        shutil.rmtree(images_256_dir)
    rows = collect_shard_rows(shards_dir)
    add_aggregate_ids(rows)
    chosen = selected_ids(rows, target_per_class)

    combined: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.get("accepted", False), item.get("top1_score", 0.0)), reverse=True):
        row = dict(row)
        aggregate_id = row.pop("_aggregate_id")
        row.pop("_shard", None)
        originally_accepted = bool(row.get("accepted"))
        row["accepted"] = originally_accepted and aggregate_id in chosen
        if originally_accepted and not row["accepted"]:
            existing = row.get("reject_reason")
            row["reject_reason"] = f"{existing},aggregate_class_target_overflow" if existing else "aggregate_class_target_overflow"
        if row["accepted"]:
            copied = copy_selected_image({"_aggregate_id": aggregate_id, **row}, images_128_dir, images_256_dir)
            if copied:
                row["saved_images"] = copied
        combined.append(row)

    metadata_path = run_dir / "metadata.jsonl"
    write_jsonl(metadata_path, combined)
    summary = summarize(combined)
    update_missing(summary, target_per_class)
    summary.update(
        {
            "target_per_class": target_per_class,
            "source_dataset": args.dataset,
            "shard_count": len(list(shards_dir.glob("shard_*"))),
        }
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_contact_sheet(
        [row for row in combined if row.get("accepted")][:32],
        reports_dir / "contact_sheet_top_matches.png",
    )
    make_contact_sheet(
        [row for row in combined if not row.get("accepted") and row.get("saved_images")][:32],
        reports_dir / "contact_sheet_rejected.png",
    )
    write_manifest(
        run_dir / "manifest.json",
        {
            "phase": "open_mined_siglip2_auto",
            "source_dataset": args.dataset,
            "split": args.split,
            "dataset_license": args.dataset_license,
            "model": args.model,
            "local_files_only": args.local_files_only,
            "target_per_class": target_per_class,
            "images_per_shard": args.images_per_shard,
            "max_shards": args.max_shards,
            "threshold": args.threshold,
            "min_margin": args.min_margin,
            "min_center_score": args.min_center_score,
            "min_center_margin": args.min_center_margin,
            "max_negative_score": args.max_negative_score,
            "metadata_bonus": args.metadata_bonus,
            "metadata_prefilter": args.metadata_prefilter,
            "max_source_rows_per_shard": args.max_source_rows_per_shard,
            "metadata_jsonl": str(metadata_path),
            "reports": {
                "score_summary": str(summary_path),
                "contact_sheet_top_matches": str(reports_dir / "contact_sheet_top_matches.png"),
                "contact_sheet_rejected": str(reports_dir / "contact_sheet_rejected.png"),
            },
            "summary": summary,
        },
    )
    return summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    rejected = [row for row in rows if not row.get("accepted")]
    per_class: dict[str, int] = {key: 0 for key in PRIMARY_KEYS}
    for row in accepted:
        key = row.get("key")
        if key in per_class:
            per_class[key] += 1
    reject_reasons: dict[str, int] = defaultdict(int)
    for row in rejected:
        for reason in str(row.get("reject_reason") or "unknown").split(","):
            reject_reasons[reason.strip() or "unknown"] += 1
    return {
        "metadata_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_per_class": per_class,
        "missing_per_class": {key: max(0, 0 - value) for key, value in per_class.items()},
        "reject_reasons": dict(reject_reasons),
    }


def update_missing(summary: dict[str, Any], target_per_class: int) -> dict[str, int]:
    per_class = summary.get("accepted_per_class", {})
    missing = {key: max(0, target_per_class - int(per_class.get(key, 0))) for key in PRIMARY_KEYS}
    summary["missing_per_class"] = missing
    return missing


def make_contact_sheet(records: list[dict[str, Any]], out_path: Path, cell_size: int = 128, label_height: int = 56) -> None:
    if not records:
        return
    columns = min(4, len(records))
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        saved = record.get("saved_images") or {}
        image_path = saved.get("128")
        if not image_path or not Path(image_path).exists():
            continue
        x = (index % columns) * cell_size
        y = (index // columns) * (cell_size + label_height)
        image = Image.open(image_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + cell_size + 4), f'{record.get("key")} {record.get("top1_score", 0):.3f}', fill=(0, 0, 0))
        draw.text((x + 4, y + cell_size + 20), f'm{record.get("score_margin", 0):.3f}', fill=(0, 0, 0))
        reason = str(record.get("reject_reason") or "")[:18]
        draw.text((x + 4, y + cell_size + 34), reason, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "automation_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_name = args.run_id or run_id()
    run_dir = Path(args.out_root) / run_name
    shards_dir = run_dir / "_shards"
    run_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        command = build_mine_command(args, shards_dir / "shard_000", 0)
        print(" ".join(command))
        return

    last_summary: dict[str, Any] = {}
    for shard_index in range(args.max_shards):
        shard_dir = shards_dir / f"shard_{shard_index:03d}"
        manifest_path = shard_dir / "manifest.json"
        if args.resume and manifest_path.exists():
            print(f"skip existing {shard_dir}")
        else:
            command = build_mine_command(args, shard_dir, shard_index)
            print(f"run shard {shard_index}: {' '.join(command)}")
            result = subprocess.run(command, cwd=str(ROOT))
            if result.returncode != 0:
                raise SystemExit(f"shard {shard_index} failed with exit code {result.returncode}")

        last_summary = build_combined_output(run_dir, args.target_per_class, args)
        missing = update_missing(last_summary, args.target_per_class)
        write_status(
            run_dir,
            {
                "run_dir": str(run_dir),
                "last_shard": shard_index,
                "complete": all(value == 0 for value in missing.values()),
                "summary": last_summary,
                "updated_at_unix": int(time.time()),
            },
        )
        print(json.dumps(last_summary["accepted_per_class"], ensure_ascii=False, sort_keys=True))
        if all(value == 0 for value in missing.values()):
            print("target reached")
            break
        if shard_index < args.max_shards - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(run_dir)


if __name__ == "__main__":
    main()
