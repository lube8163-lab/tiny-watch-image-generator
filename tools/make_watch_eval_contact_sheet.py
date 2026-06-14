#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow is required: python3 -m pip install pillow") from exc
    return Image, ImageDraw


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_eval(args: argparse.Namespace, out_dir: Path) -> None:
    cmd = [
        "swift",
        "run",
        "TinyWatchEval",
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
    ]
    if args.raw:
        cmd.append("--raw")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def shorten_label(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(1, max_chars - 1)] + "..."


def entry_image_path(out_dir: Path, entry: dict) -> Path:
    image_path = Path(entry["image"])
    if image_path.is_absolute():
        return image_path
    return out_dir / image_path


def draw_sheet(entries: list[dict], out_dir: Path, out_path: Path, columns: int, cell: int, label_height: int) -> None:
    Image, ImageDraw = require_pillow()
    if not entries:
        return

    columns = max(1, min(columns, len(entries)))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, entry in enumerate(entries):
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        image = Image.open(entry_image_path(out_dir, entry)).convert("RGB")
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((cell, cell), resampling)
        sheet.paste(image, (x, y))

        label = shorten_label(entry["prompt"], 22)
        seed = f's{entry["seed"]}'
        elapsed = f'{entry["elapsedMs"]}ms'
        draw_text_safe(draw, (x + 4, y + cell + 4), label, fill=(0, 0, 0))
        draw_text_safe(draw, (x + 4, y + cell + 20), f"{seed} {elapsed}", fill=(80, 80, 80))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def draw_text_safe(draw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    try:
        draw.text(xy, text, fill=fill)
    except UnicodeEncodeError:
        draw.text(xy, text.encode("ascii", "replace").decode("ascii"), fill=fill)


def load_manifest(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def flatten_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for group in manifest["groups"]:
        entries.extend(group["entries"])
    return entries


def write_contact_sheets(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    manifest = load_manifest(out_dir)
    sheet_paths: list[Path] = []
    cell = args.cell or int(manifest.get("size", 128))

    for group in manifest["groups"]:
        entries = group["entries"]
        path = out_dir / f'contact_sheet_{group["key"]}.png'
        draw_sheet(entries, out_dir, path, args.columns, cell, args.label_height)
        sheet_paths.append(path)

    all_path = out_dir / "contact_sheet_all.png"
    draw_sheet(flatten_entries(manifest), out_dir, all_path, args.columns, cell, args.label_height)
    sheet_paths.append(all_path)

    index_path = out_dir / "contact_sheets.json"
    index_path.write_text(
        json.dumps(
            {
                "manifest": str(out_dir / "manifest.json"),
                "sheets": [str(path) for path in sheet_paths],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return sheet_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact watch-generator eval images and PNG contact sheets."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "prompt_eval_suite.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--groups",
        default="core_nouns,adjectives,actions,styles,japanese_aliases,v6_new_subjects",
        help="Comma-separated eval group keys, or all.",
    )
    parser.add_argument(
        "--prompts-per-group",
        type=int,
        default=4,
        help="Prompts per selected group. Use 0 for all prompts.",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell", type=int, default=None)
    parser.add_argument("--label-height", type=int, default=38)
    parser.add_argument("--raw", action="store_true", help="Disable watch postprocess.")
    parser.add_argument("--skip-generate", action="store_true", help="Use an existing manifest in --out-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or ROOT / "reports" / "watch_eval" / timestamp()
    out_dir = out_dir.resolve()

    if not args.skip_generate:
        run_eval(args, out_dir)

    sheet_paths = write_contact_sheets(args, out_dir)
    for path in sheet_paths:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
