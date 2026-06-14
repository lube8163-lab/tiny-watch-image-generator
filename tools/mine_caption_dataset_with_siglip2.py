#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageDraw

from mine_dataset_with_siglip2 import (
    DEFAULT_MODEL,
    image_to_rgb,
    looks_like_image_url,
    prepare_text_features,
    require_mining_stack,
    resize_square,
    sanitize_id,
    score_batch,
)
from prompt_normalization import PROMPT_ALIASES, canonicalize_prompt
from research_common import ROOT, pick_torch_device, torch_dtype_for_device, write_manifest


DEFAULT_DATASET = "common-canvas/commoncatalog-cc-by"
DEFAULT_OUT_ROOT = ROOT / "datasets" / "open_mined_caption_siglip2"
DEFAULT_NEGATIVES = [
    "visible text",
    "logo",
    "watermark",
    "crowded scene",
    "multiple objects",
    "human group",
    "blurry image",
]
DEFAULT_QUALITY_PROMPTS = [
    "a clear centered subject",
    "one main object on a simple background",
]
TRAILING_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "sitting",
    "standing",
    "lying",
    "laying",
    "holding",
    "wearing",
    "eating",
    "drinking",
    "walking",
    "running",
    "flying",
    "parked",
    "painting",
    "crossing",
}
MULTI_SUBJECT_STARTERS = {
    "two",
    "three",
    "four",
    "five",
    "six",
    "several",
    "many",
    "multiple",
    "group",
    "crowd",
}
BAD_CAPTION_FRAGMENTS = {
    "all rights reserved",
    "copyright",
    "creative commons",
    "uploaded by",
    "screenshot",
    "screen shot",
    "untitled",
    "unknown",
    "img",
    "dsc",
    "sam ",
    "whatsapp",
    "once again",
}
TRIM_MARKERS = [
    " with ",
    " and ",
    " or ",
    " in front of ",
    " next to ",
    " near ",
    " beside ",
    " sitting ",
    " standing ",
    " lying ",
    " laying ",
    " holding ",
    " wearing ",
    " eating ",
    " drinking ",
    " walking ",
    " running ",
    " flying ",
    " parked ",
    " painting ",
    " crossing ",
]
PREFIXES = [
    "a photo of ",
    "a photograph of ",
    "a picture of ",
    "an image of ",
    "a close up of ",
    "a close-up of ",
    "close up of ",
    "close-up of ",
    "a blurry photo of ",
    "there is ",
    "there are ",
]


@dataclass
class CaptionSample:
    source_dataset: str
    source_id: str
    source_url: str | None
    source_path: str | None
    license: str | None
    raw_caption: str
    caption: str
    image_payload: Any
    width: int | None
    height: int | None
    extra: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine open image+caption datasets for tiny 128x128 teacher data using SigLIP2."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--image-dir", default=None, help="Local image directory. Sidecar .txt/.caption files are used when present.")
    parser.add_argument("--image-dir-license", default=None)
    parser.add_argument("--dataset-license", default=None, help="Fallback license when rows do not include one")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=1024, help="Maximum caption-filtered images to score")
    parser.add_argument("--max-source-rows", type=int, default=0, help="Maximum raw HF rows to read. 0 means unlimited.")
    parser.add_argument("--target-count", type=int, default=256, help="Stop after this many accepted images when --stop-at-target is enabled")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--min-quality-score", type=float, default=0.01)
    parser.add_argument("--max-negative-score", type=float, default=0.10)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument(
        "--preserve-modifiers",
        action="store_true",
        help="Keep short action/adjective phrases instead of trimming them to a noun-only prompt.",
    )
    parser.add_argument("--max-per-caption", type=int, default=24)
    parser.add_argument("--exclude-captions", default="", help="Comma-separated normalized captions to reject, e.g. person,man,woman")
    parser.add_argument(
        "--exclude-caption-fragments",
        default="",
        help="Comma-separated lowercase fragments. Captions containing any fragment are rejected.",
    )
    parser.add_argument(
        "--require-known-subject",
        action="store_true",
        help="Reject captions that do not map to one of the local prompt subject aliases.",
    )
    parser.add_argument("--caption-fields", default="blip2_caption,caption,usertags")
    parser.add_argument("--crop-bbox", action="store_true", help="Crop around the largest object bbox when the source row has Open Images-style objects.")
    parser.add_argument("--bbox-padding", type=float, default=0.08)
    parser.add_argument("--min-object-area", type=float, default=0.03)
    parser.add_argument("--negative-prompts", default="|".join(DEFAULT_NEGATIVES))
    parser.add_argument("--quality-prompts", default="|".join(DEFAULT_QUALITY_PROMPTS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.add_argument("--save-rejected-images", action="store_true", default=True)
    parser.add_argument("--no-save-rejected-images", action="store_false", dest="save_rejected_images")
    parser.add_argument("--max-rejected-images", type=int, default=64)
    parser.add_argument("--stop-at-target", action="store_true", default=True)
    parser.add_argument("--no-stop-at-target", action="store_false", dest="stop_at_target")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-exit", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def make_run_dir(out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)
    return DEFAULT_OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def split_fields(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_caption_set(value: str) -> set[str]:
    return {clean_caption_text(item) for item in value.split(",") if clean_caption_text(item)}


def split_caption_fragments(value: str) -> list[str]:
    return [clean_caption_text(item) for item in value.split(",") if clean_caption_text(item)]


def clean_caption_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("_", " ").replace("+", " ").replace("|", " ")
    text = re.sub(r"\.(jpg|jpeg|png|webp|gif|bmp|tiff)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def shorten_caption(value: Any, min_words: int, max_words: int, preserve_modifiers: bool = False) -> str | None:
    text = clean_caption_text(value)
    if not text:
        return None
    if any(fragment in f"{text} " for fragment in BAD_CAPTION_FRAGMENTS):
        return None
    for prefix in PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    if not preserve_modifiers:
        for marker in TRIM_MARKERS:
            index = text.find(marker)
            if index > 0:
                text = text[:index].strip()
    text = re.sub(r"\b\d{2,}\b", " ", text)
    text = re.sub(r"\b[a-z]{1}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -'")
    if not text:
        return None
    words = re.findall(r"[a-z][a-z0-9'-]*", text)
    if not preserve_modifiers:
        while words and words[-1] in TRAILING_WORDS:
            words.pop()
    if words and words[0] in MULTI_SUBJECT_STARTERS:
        return None
    if len(words) < min_words or len(words) > max_words:
        return None
    if len(set(words)) == 1 and len(words) > 2:
        return None
    return " ".join(words)


def object_caption_candidates(
    row: dict[str, Any],
    min_words: int,
    max_words: int,
    min_object_area: float,
    preserve_modifiers: bool,
) -> list[tuple[str, str, dict[str, Any]]]:
    objects = row.get("objects")
    if not isinstance(objects, list):
        return []
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    for item in objects:
        if not isinstance(item, dict) or item.get("is_group_of"):
            continue
        raw_label = item.get("label")
        caption = shorten_caption(raw_label, min_words, max_words, preserve_modifiers)
        if not caption:
            continue
        try:
            xmin = float(item["xmin"])
            xmax = float(item["xmax"])
            ymin = float(item["ymin"])
            ymax = float(item["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        if area < min_object_area:
            continue
        extra = {
            "object_label": raw_label,
            "object_bbox": [xmin, ymin, xmax, ymax],
            "object_area": area,
            "object_confidence": item.get("confidence"),
            "object_is_occluded": item.get("is_occluded"),
            "object_is_truncated": item.get("is_truncated"),
        }
        candidates.append((area, str(raw_label), caption, extra))
    candidates.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    output: list[tuple[str, str, dict[str, Any]]] = []
    for _, raw, caption, extra in candidates:
        if caption in seen:
            continue
        seen.add(caption)
        output.append((raw, caption, extra))
    return output


def candidate_captions(
    row: dict[str, Any],
    fields: list[str],
    min_words: int,
    max_words: int,
    min_object_area: float,
    preserve_modifiers: bool,
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for field in fields:
        if field == "objects":
            for raw, caption, extra in object_caption_candidates(
                row,
                min_words,
                max_words,
                min_object_area,
                preserve_modifiers,
            ):
                if caption not in seen:
                    seen.add(caption)
                    candidates.append((raw, caption, extra))
            continue
        value = row.get(field)
        if value in (None, ""):
            continue
        raw_items = [value]
        if field in {"usertags", "machinetags"} and isinstance(value, str):
            raw_items = re.split(r"[,;]", value)
        for raw in raw_items:
            caption = shorten_caption(raw, min_words, max_words, preserve_modifiers)
            if caption and caption not in seen:
                seen.add(caption)
                candidates.append((str(raw), caption, {}))
    return candidates


def detect_value(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def license_from_row(row: dict[str, Any], fallback: str | None) -> str | None:
    license_name = detect_value(row, ["license", "license_name", "licensename"])
    license_url = detect_value(row, ["licenseurl", "license_url"])
    if license_name and license_url:
        return f"{license_name} ({license_url})"
    return str(license_name or license_url or fallback) if (license_name or license_url or fallback) else None


def source_id_from_row(dataset_name: str, split: str, index: int, row: dict[str, Any]) -> str:
    value = detect_value(row, ["id", "image_id", "imageid", "source_id", "photoid", "key", "file_name", "filename"])
    return str(value or f"{dataset_name}-{split}-{index}")


def pick_image_key(features: Any, row: dict[str, Any]) -> str:
    try:
        from datasets import Image as DatasetsImage
    except Exception:
        DatasetsImage = None
    for key, feature in getattr(features, "items", lambda: [])():
        if DatasetsImage is not None and isinstance(feature, DatasetsImage):
            return key
    for key, value in row.items():
        if isinstance(value, Image.Image):
            return key
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return key
        if isinstance(value, str) and looks_like_image_url(value):
            return key
    raise SystemExit("could not detect an image column in the Hugging Face dataset")


def decode_image_payload(value: Any) -> Image.Image:
    import requests

    if isinstance(value, Image.Image):
        return image_to_rgb(value)
    if isinstance(value, dict):
        if value.get("bytes"):
            return image_to_rgb(Image.open(io.BytesIO(value["bytes"])))
        path = value.get("path")
        if path:
            if isinstance(path, str) and looks_like_image_url(path):
                response = requests.get(path, timeout=15)
                response.raise_for_status()
                return image_to_rgb(Image.open(io.BytesIO(response.content)))
            return image_to_rgb(Image.open(path))
    if isinstance(value, (bytes, bytearray)):
        return image_to_rgb(Image.open(io.BytesIO(value)))
    if isinstance(value, str) and looks_like_image_url(value):
        response = requests.get(value, timeout=15)
        response.raise_for_status()
        return image_to_rgb(Image.open(io.BytesIO(response.content)))
    raise ValueError(f"unsupported image payload: {type(value)!r}")


def crop_normalized_bbox(image: Image.Image, bbox: Any, padding: float) -> Image.Image:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return image
    width, height = image.size
    xmin, ymin, xmax, ymax = [float(value) for value in bbox]
    pad_x = (xmax - xmin) * padding
    pad_y = (ymax - ymin) * padding
    left = max(0, int((xmin - pad_x) * width))
    top = max(0, int((ymin - pad_y) * height))
    right = min(width, int((xmax + pad_x) * width))
    bottom = min(height, int((ymax + pad_y) * height))
    if right - left < 16 or bottom - top < 16:
        return image
    return image.crop((left, top, right, bottom))


def iter_hf_caption_samples(args: argparse.Namespace) -> Iterator[CaptionSample]:
    _, _, datasets = require_mining_stack()
    dataset = datasets.load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=args.streaming,
    )
    try:
        from datasets import Image as DatasetsImage

        for key, feature in getattr(dataset, "features", {}).items():
            if isinstance(feature, DatasetsImage):
                dataset = dataset.cast_column(key, DatasetsImage(decode=False))
                break
    except Exception:
        pass
    if args.shuffle_buffer:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    fields = split_fields(args.caption_fields)
    iterator = iter(dataset)
    first_row = next(iterator, None)
    if first_row is None:
        return
    image_key = pick_image_key(getattr(dataset, "features", {}), first_row)
    rows = itertools.chain([(0, first_row)], enumerate(iterator, start=1))
    for index, row in rows:
        if args.max_source_rows and index >= args.max_source_rows:
            break
        captions = candidate_captions(
            row,
            fields,
            args.min_words,
            args.max_words,
            args.min_object_area,
            args.preserve_modifiers,
        )
        if not captions:
            continue
        raw_caption, caption, caption_extra = captions[0]
        image_value = row.get(image_key)
        source_url = detect_value(row, ["source_url", "image_url", "url", "coco_url", "flickr_url", "downloadurl", "download_url"])
        source_path = detect_value(row, ["source_path", "path", "file_path", "filename"])
        if source_url is None and isinstance(image_value, str) and looks_like_image_url(image_value):
            source_url = image_value
        if source_path is None and isinstance(image_value, dict):
            source_path = image_value.get("path")
        yield CaptionSample(
            source_dataset=args.dataset,
            source_id=source_id_from_row(args.dataset, args.split, index, row),
            source_url=str(source_url) if source_url else None,
            source_path=str(source_path) if source_path else None,
            license=license_from_row(row, args.dataset_license),
            raw_caption=raw_caption,
            caption=caption,
            image_payload=image_value,
            width=detect_value(row, ["width", "original_width"]),
            height=detect_value(row, ["height", "original_height"]),
            extra={"row_index": index, "image_key": image_key, **caption_extra},
        )


def sidecar_caption(path: Path, min_words: int, max_words: int, preserve_modifiers: bool) -> tuple[str, str] | None:
    for suffix in [".caption", ".txt"]:
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            raw = sidecar.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            caption = shorten_caption(raw, min_words, max_words, preserve_modifiers)
            if caption:
                return raw, caption
    json_sidecar = path.with_suffix(".json")
    if json_sidecar.exists():
        payload = json.loads(json_sidecar.read_text(encoding="utf-8"))
        for key in ["caption", "prompt", "title"]:
            caption = shorten_caption(payload.get(key), min_words, max_words, preserve_modifiers)
            if caption:
                return str(payload.get(key)), caption
    caption = shorten_caption(path.stem, min_words, max_words, preserve_modifiers)
    if caption:
        return path.stem, caption
    return None


def iter_local_caption_samples(args: argparse.Namespace) -> Iterator[CaptionSample]:
    if not args.image_dir_license:
        raise SystemExit("--image-dir-license is required with --image-dir")
    image_root = Path(args.image_dir)
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        paths.extend(sorted(image_root.rglob(pattern)))
    for index, path in enumerate(paths):
        caption_pair = sidecar_caption(path, args.min_words, args.max_words, args.preserve_modifiers)
        if not caption_pair:
            continue
        raw_caption, caption = caption_pair
        yield CaptionSample(
            source_dataset="local_image_dir",
            source_id=path.stem,
            source_url=None,
            source_path=str(path),
            license=args.image_dir_license,
            raw_caption=raw_caption,
            caption=caption,
            image_payload=str(path),
            width=None,
            height=None,
            extra={"row_index": index},
        )


def batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_candidate_images(image: Image.Image, images_256_dir: Path, images_128_dir: Path, file_stem: str) -> dict[str, str]:
    image_256_path = images_256_dir / f"{file_stem}.png"
    image_128_path = images_128_dir / f"{file_stem}.png"
    resize_square(image, 256).save(image_256_path)
    resize_square(image, 128).save(image_128_path)
    return {"256": str(image_256_path), "128": str(image_128_path)}


def score_caption_pairs(
    model: Any,
    processor: Any,
    device: Any,
    dtype: Any,
    images: list[Image.Image],
    captions: list[str],
    negative_text_features: Any,
    quality_text_features: Any,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    import torch

    caption_features = prepare_text_features(model, processor, device, dtype, captions)
    text_features = torch.cat([caption_features, negative_text_features, quality_text_features], dim=0)
    scores = score_batch(model, processor, device, dtype, images, text_features).tolist()
    caption_scores: list[float] = []
    negative_scores: list[list[float]] = []
    quality_scores: list[list[float]] = []
    negative_start = len(captions)
    quality_start = negative_start + negative_text_features.shape[0]
    for index, row in enumerate(scores):
        caption_scores.append(float(row[index]))
        negative_scores.append([float(value) for value in row[negative_start:quality_start]])
        quality_scores.append([float(value) for value in row[quality_start:]])
    return caption_scores, negative_scores, quality_scores


def reject_reasons(
    *,
    sample: CaptionSample,
    caption_score: float,
    quality_score: float,
    max_negative_score: float,
    caption_count: int,
    excluded_captions: set[str],
    excluded_fragments: list[str],
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []
    if sample.license in (None, "", "unknown"):
        reasons.append("missing_license")
    if caption_score < args.threshold:
        reasons.append("low_caption_score")
    if quality_score < args.min_quality_score:
        reasons.append("low_quality_score")
    if max_negative_score > args.max_negative_score:
        reasons.append("high_negative_score")
    if caption_count >= args.max_per_caption:
        reasons.append("caption_overflow")
    if sample.caption in excluded_captions:
        reasons.append("excluded_caption")
    if any(fragment in sample.caption for fragment in excluded_fragments):
        reasons.append("excluded_caption_fragment")
    if args.require_known_subject and canonicalize_prompt(sample.caption) not in PROMPT_ALIASES:
        reasons.append("missing_known_subject")
    return reasons


def make_contact_sheet(records: list[dict[str, Any]], out_path: Path, cell_size: int = 128, label_height: int = 64) -> None:
    if not records:
        return
    columns = min(5, len(records))
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        x = (index % columns) * cell_size
        y = (index // columns) * (cell_size + label_height)
        saved = record.get("saved_images") or {}
        image_path = saved.get("128")
        if not image_path:
            continue
        image = Image.open(image_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        caption = str(record.get("caption") or record.get("prompt") or "")[:22]
        line1 = caption
        line2 = f's{record["image_caption_score"]:.3f} q{record["quality_score"]:.3f}'
        line3 = f'n{record["negative_score_max"]:.3f}'
        draw.text((x + 4, y + cell_size + 4), line1, fill=(0, 0, 0))
        draw.text((x + 4, y + cell_size + 22), line2, fill=(0, 0, 0))
        draw.text((x + 4, y + cell_size + 40), line3, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def compute_summary(records: list[dict[str, Any]], negative_prompts: list[str], quality_prompts: list[str]) -> dict[str, Any]:
    accepted = [row for row in records if row.get("accepted")]
    rejected = [row for row in records if not row.get("accepted")]
    reject_counts: Counter[str] = Counter()
    for row in rejected:
        for reason in str(row.get("reject_reason") or "unknown").split(","):
            reject_counts[reason.strip() or "unknown"] += 1
    scores = [row["image_caption_score"] for row in accepted]
    quality_scores = [row["quality_score"] for row in accepted]
    negative_scores = [row["negative_score_max"] for row in accepted]
    captions = Counter(row["caption"] for row in accepted)
    return {
        "total_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "reject_reasons": dict(reject_counts),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "quality_score_mean": sum(quality_scores) / len(quality_scores) if quality_scores else None,
        "negative_score_max_mean": sum(negative_scores) / len(negative_scores) if negative_scores else None,
        "unique_caption_count": len(captions),
        "top_captions": captions.most_common(40),
        "negative_prompts": negative_prompts,
        "quality_prompts": quality_prompts,
    }


def row_id(sample: CaptionSample, caption: str, index: int) -> str:
    base = sanitize_id(f"{caption}_{sample.source_id}")
    if len(base) > 80:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        base = f"{base[:68]}_{digest}"
    return f"{base}_{index:06d}"


def main() -> None:
    args = parse_args()
    if not args.dataset and not args.image_dir:
        raise SystemExit("either --dataset or --image-dir is required")

    out_dir = make_run_dir(args.out_dir)
    images_256_dir = out_dir / "images_256"
    images_128_dir = out_dir / "images_128"
    reports_dir = out_dir / "reports"
    images_256_dir.mkdir(parents=True, exist_ok=True)
    images_128_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"
    summary_path = reports_dir / "score_summary.json"
    top_sheet_path = reports_dir / "contact_sheet_top_matches.png"
    rejected_sheet_path = reports_dir / "contact_sheet_rejected.png"
    negative_prompts = split_pipe(args.negative_prompts)
    quality_prompts = split_pipe(args.quality_prompts)
    excluded_captions = split_caption_set(args.exclude_captions)
    excluded_fragments = split_caption_fragments(args.exclude_caption_fragments)

    samples = iter_local_caption_samples(args) if args.image_dir else iter_hf_caption_samples(args)
    if args.dry_run:
        first = next(iter(samples), None)
        print(
            json.dumps(
                {
                    "source_dataset": first.source_dataset if first else args.dataset,
                    "source_id": first.source_id if first else None,
                    "source_url": first.source_url if first else None,
                    "source_path": first.source_path if first else None,
                    "license": first.license if first else args.dataset_license,
                    "raw_caption": first.raw_caption if first else None,
                    "caption": first.caption if first else None,
                    "width": first.width if first else None,
                    "height": first.height if first else None,
                    "extra": first.extra if first else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.force_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return

    torch, transformers, _ = require_mining_stack()
    device = pick_torch_device(torch)
    dtype = torch_dtype_for_device(torch, device)
    processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = transformers.AutoModel.from_pretrained(args.model, dtype=dtype, local_files_only=args.local_files_only)
    model = model.to(device)
    model.eval()
    negative_text_features = prepare_text_features(model, processor, device, dtype, negative_prompts)
    quality_text_features = prepare_text_features(model, processor, device, dtype, quality_prompts)

    records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    rejected_sheet_records: list[dict[str, Any]] = []
    caption_counts: Counter[str] = Counter()
    scored_count = 0
    accepted_count = 0
    rejected_image_count = 0
    started_at = time.perf_counter()

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for batch_samples in batched(samples, args.batch_size):
            if scored_count >= args.max_images:
                break
            if args.stop_at_target and accepted_count >= args.target_count:
                break
            if scored_count + len(batch_samples) > args.max_images:
                batch_samples = batch_samples[: args.max_images - scored_count]
            decoded_samples: list[CaptionSample] = []
            images: list[Image.Image] = []
            for sample in batch_samples:
                try:
                    image = decode_image_payload(sample.image_payload)
                    if args.crop_bbox:
                        image = crop_normalized_bbox(image, sample.extra.get("object_bbox"), args.bbox_padding)
                except Exception as exc:
                    print(f"skip image {sample.source_id}: {exc}", file=sys.stderr)
                    continue
                decoded_samples.append(sample)
                images.append(image)
            if not decoded_samples:
                continue
            captions = [sample.caption for sample in decoded_samples]
            caption_scores, negative_rows, quality_rows = score_caption_pairs(
                model,
                processor,
                device,
                dtype,
                images,
                captions,
                negative_text_features,
                quality_text_features,
            )
            for sample, image, caption_score, negative_row, quality_row in zip(
                decoded_samples, images, caption_scores, negative_rows, quality_rows
            ):
                scored_count += 1
                negative_map = {prompt: float(score) for prompt, score in zip(negative_prompts, negative_row)}
                quality_map = {prompt: float(score) for prompt, score in zip(quality_prompts, quality_row)}
                negative_score_max = max(negative_map.values()) if negative_map else 0.0
                quality_score = max(quality_map.values()) if quality_map else 0.0
                caption_negative_margin = float(caption_score - negative_score_max)
                reasons = reject_reasons(
                    sample=sample,
                    caption_score=caption_score,
                    quality_score=quality_score,
                    max_negative_score=negative_score_max,
                    caption_count=caption_counts[sample.caption],
                    excluded_captions=excluded_captions,
                    excluded_fragments=excluded_fragments,
                    args=args,
                )
                accepted = not reasons
                if accepted and args.stop_at_target and accepted_count >= args.target_count:
                    accepted = False
                    reasons = ["target_count_reached"]
                if accepted:
                    caption_counts[sample.caption] += 1
                    accepted_count += 1
                sample_id = row_id(sample, sample.caption, scored_count - 1)
                should_save_rejected = (
                    args.save_rejected_images
                    and not accepted
                    and rejected_image_count < args.max_rejected_images
                )
                saved_images = None
                if accepted or should_save_rejected:
                    saved_images = save_candidate_images(image, images_256_dir, images_128_dir, sample_id)
                    if should_save_rejected:
                        rejected_image_count += 1
                record = {
                    "id": sample_id,
                    "source_type": "open_caption_dataset",
                    "source_dataset": sample.source_dataset,
                    "source_id": sample.source_id,
                    "source_url": sample.source_url,
                    "source_path": sample.source_path,
                    "license": sample.license,
                    "key": sample.caption,
                    "title": sample.caption,
                    "caption": sample.caption,
                    "raw_caption": sample.raw_caption,
                    "prompt": sample.caption,
                    "image_caption_score": float(caption_score),
                    "top1_score": float(caption_score),
                    "top2_score": float(negative_score_max),
                    "score_margin": caption_negative_margin,
                    "caption_negative_margin": caption_negative_margin,
                    "quality_score": float(quality_score),
                    "quality_scores": quality_map,
                    "negative_scores": negative_map,
                    "negative_score_max": float(negative_score_max),
                    "accepted": accepted,
                    "reject_reason": ",".join(reasons) if reasons else None,
                    "saved_images": saved_images,
                    "width": image.width,
                    "height": image.height,
                    "created_at_unix": int(time.time()),
                    "extra": sample.extra,
                }
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                if accepted:
                    accepted_records.append(record)
                elif saved_images:
                    rejected_sheet_records.append(record)
            metadata_file.flush()

    summary = compute_summary(records, negative_prompts, quality_prompts)
    summary.update(
        {
            "scored_count": scored_count,
            "target_count": args.target_count,
            "threshold": args.threshold,
            "min_quality_score": args.min_quality_score,
            "max_negative_score": args.max_negative_score,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "preserve_modifiers": args.preserve_modifiers,
            "max_per_caption": args.max_per_caption,
            "excluded_captions": sorted(excluded_captions),
            "excluded_caption_fragments": excluded_fragments,
            "require_known_subject": args.require_known_subject,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_contact_sheet(accepted_records[: min(40, len(accepted_records))], top_sheet_path)
    make_contact_sheet(rejected_sheet_records[: min(40, len(rejected_sheet_records))], rejected_sheet_path)
    write_manifest(
        out_dir / "manifest.json",
        {
            "phase": "open_mined_caption_siglip2",
            "model": args.model,
            "source": {
                "dataset": args.dataset,
                "split": args.split,
                "revision": args.revision,
                "image_dir": args.image_dir,
                "dataset_license": args.dataset_license,
                "image_dir_license": args.image_dir_license,
            },
            "caption_fields": split_fields(args.caption_fields),
            "preserve_modifiers": args.preserve_modifiers,
            "excluded_captions": sorted(excluded_captions),
            "excluded_caption_fragments": excluded_fragments,
            "require_known_subject": args.require_known_subject,
            "negative_prompts": negative_prompts,
            "quality_prompts": quality_prompts,
            "device": device.type,
            "dtype": str(dtype),
            "max_images": args.max_images,
            "max_source_rows": args.max_source_rows,
            "target_count": args.target_count,
            "threshold": args.threshold,
            "min_quality_score": args.min_quality_score,
            "max_negative_score": args.max_negative_score,
            "crop_bbox": args.crop_bbox,
            "bbox_padding": args.bbox_padding,
            "min_object_area": args.min_object_area,
            "accepted_count": len(accepted_records),
            "rejected_count": len(records) - len(accepted_records),
            "metadata_count": len(records),
            "elapsed_seconds": time.perf_counter() - started_at,
            "metadata_jsonl": str(metadata_path),
            "reports": {
                "score_summary": str(summary_path),
                "contact_sheet_top_matches": str(top_sheet_path),
                "contact_sheet_rejected": str(rejected_sheet_path),
            },
        },
    )
    print(metadata_path)
    print(summary_path)
    print(top_sheet_path)
    print(rejected_sheet_path)
    print(out_dir / "manifest.json")
    sys.stdout.flush()
    sys.stderr.flush()
    if args.force_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
