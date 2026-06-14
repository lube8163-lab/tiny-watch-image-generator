#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageDraw, ImageOps

from research_common import ROOT, pick_torch_device, torch_dtype_for_device, write_manifest


DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_PROMPTS = ROOT / "configs" / "mining_prompts.json"
DEFAULT_OUT_ROOT = ROOT / "datasets" / "open_mined_siglip2"


@dataclass
class PromptSpec:
    key: str
    prompt: str
    title: str | None = None
    aliases: list[str] = field(default_factory=list)
    metadata_keywords: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)


@dataclass
class Sample:
    source_dataset: str
    source_id: str
    source_url: str | None
    source_path: str | None
    license: str | None
    title: str | None
    text_blob: str | None
    image: Image.Image
    extra: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine open image datasets for small teacher-data candidates with SigLIP2."
    )
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-dir", default=None, help="Local image directory as an alternative input")
    parser.add_argument("--image-dir-license", default=None)
    parser.add_argument("--dataset-license", default=None, help="Fallback license when rows do not include one")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--config", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--top-k-per-class", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--max-negative-score", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metadata-bonus", type=float, default=0.03)
    parser.add_argument("--metadata-prefilter", action="store_true", help="Skip HF rows whose caption/tags/object labels do not mention target aliases before image decoding.")
    parser.add_argument("--max-source-rows", type=int, default=0, help="Stop after reading this many raw HF rows. 0 means no raw-row limit.")
    parser.add_argument("--center-crop-fraction", type=float, default=0.75)
    parser.add_argument("--min-center-score", type=float, default=0.0)
    parser.add_argument("--min-center-margin", type=float, default=0.005)
    parser.add_argument("--save-rejected-images", action="store_true", default=True)
    parser.add_argument("--no-save-rejected-images", action="store_false", dest="save_rejected_images")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-exit", action="store_true", help="Exit immediately after writing outputs; useful for HF streaming subprocesses.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def require_mining_stack():
    missing = []
    modules = {}
    for name in ["torch", "transformers", "datasets", "PIL"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        deps = ", ".join(missing)
        raise SystemExit(
            f"missing dependencies: {deps}\n"
            "Create a research environment first:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements/research.txt\n"
        )
    return modules["torch"], modules["transformers"], modules["datasets"]


def load_prompt_config(path: str) -> tuple[list[PromptSpec], list[PromptSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = [
        PromptSpec(
            key=item["key"],
            title=item.get("title"),
            prompt=item["prompt"],
            aliases=item.get("aliases", []),
            metadata_keywords=item.get("metadata_keywords", []),
            prompts=item.get("prompts", [item["prompt"]]),
        )
        for item in payload["targets"]
    ]
    negatives = [
        PromptSpec(
            key=item["key"],
            title=None,
            prompt=item["prompt"],
            prompts=item.get("prompts", [item["prompt"]]),
        )
        for item in payload.get("negatives", [])
    ]
    return targets, negatives


def sanitize_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "sample"


def make_run_dir(out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUT_ROOT / run_id


def image_to_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def center_focus_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    side = int(min(width, height) * fraction)
    side = max(16, min(side, min(width, height)))
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def detect_value(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.lower().replace("+", " ")
    return " ".join(normalized.split())


def extract_object_labels(row: dict[str, Any]) -> list[str]:
    objects = row.get("objects")
    labels: list[str] = []
    if isinstance(objects, list):
        for item in objects:
            if isinstance(item, dict) and item.get("label"):
                labels.append(str(item["label"]))
    elif isinstance(objects, dict):
        raw_labels = objects.get("label") or objects.get("labels")
        if isinstance(raw_labels, list):
            labels.extend(str(label) for label in raw_labels if label)
        elif raw_labels:
            labels.append(str(raw_labels))
    return labels


def extract_row_text_blob(row: dict[str, Any]) -> str:
    text_fields = [
        detect_value(row, ["title"]),
        detect_value(row, ["caption"]),
        detect_value(row, ["blip2_caption"]),
        detect_value(row, ["description"]),
        detect_value(row, ["text"]),
        detect_value(row, ["label"]),
        detect_value(row, ["usertags"]),
        detect_value(row, ["machinetags"]),
        " ".join(extract_object_labels(row)),
    ]
    return normalize_text(" ".join(str(item) for item in text_fields if item not in (None, "")))


def build_prefilter_keywords(targets: list[PromptSpec]) -> list[str]:
    keywords: set[str] = set()
    for target in targets:
        for item in [target.key, target.title or "", *target.aliases, *target.metadata_keywords]:
            normalized = normalize_text(item)
            if normalized:
                keywords.add(normalized)
    return sorted(keywords)


def row_matches_prefilter(row: dict[str, Any], keywords: list[str] | None) -> bool:
    if not keywords:
        return True
    text_blob = extract_row_text_blob(row)
    return any(keyword in text_blob for keyword in keywords)


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


def looks_like_image_url(value: str) -> bool:
    lowered = value.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return False
    return any(token in lowered for token in [".jpg", ".jpeg", ".png", ".webp", ".bmp", "staticflickr", "images.cocodataset"])


def decode_image(value: Any) -> Image.Image:
    import requests

    if isinstance(value, Image.Image):
        return image_to_rgb(value)
    if isinstance(value, dict):
        path = value.get("path")
        if path:
            return image_to_rgb(Image.open(path))
    if isinstance(value, str) and looks_like_image_url(value):
        response = requests.get(value, timeout=10)
        response.raise_for_status()
        return image_to_rgb(Image.open(io.BytesIO(response.content)))
    raise ValueError(f"unsupported image payload: {type(value)!r}")


def iter_hf_samples(
    dataset_name: str,
    split: str,
    revision: str | None,
    streaming: bool,
    fallback_license: str | None,
    prefilter_keywords: list[str] | None,
    max_source_rows: int,
) -> Iterator[Sample]:
    _, _, datasets = require_mining_stack()
    dataset = datasets.load_dataset(dataset_name, split=split, revision=revision, streaming=streaming)
    iterator = iter(dataset)
    first_row = next(iterator, None)
    if first_row is None:
        return
    image_key = pick_image_key(getattr(dataset, "features", {}), first_row)

    def row_to_sample(index: int, row: dict[str, Any]) -> Sample:
        raw_image_value = row[image_key]
        image = decode_image(raw_image_value)
        source_id = str(
            detect_value(row, ["id", "image_id", "imageid", "source_id", "photoid", "key", "file_name", "filename"])
            or f"{dataset_name}-{split}-{index}"
        )
        source_url = detect_value(row, ["source_url", "image_url", "url", "coco_url", "flickr_url", "downloadurl", "download_url"])
        if source_url is None and isinstance(row.get(image_key), str) and looks_like_image_url(row[image_key]):
            source_url = row[image_key]
        source_path = detect_value(row, ["source_path", "path", "file_path", "filename"])
        if source_path is None and isinstance(raw_image_value, Image.Image):
            source_path = getattr(raw_image_value, "filename", None)
        if source_path is None and isinstance(raw_image_value, dict):
            source_path = raw_image_value.get("path")
        if source_path == "":
            source_path = None
        if source_url is None and source_path is None:
            source_path = f"hf://datasets/{dataset_name}/{split}/{source_id}"
        license_value = detect_value(row, ["license", "license_name", "licenseurl", "license_url", "licensename"]) or fallback_license
        title = detect_value(row, ["title", "caption", "text", "description", "label", "blip2_caption"])
        object_labels = extract_object_labels(row)
        text_blob = extract_row_text_blob(row)
        extra = {
            "image_key": image_key,
            "object_labels": object_labels,
            "row_keys": sorted(row.keys()),
        }
        return Sample(
            source_dataset=dataset_name,
            source_id=source_id,
            source_url=str(source_url) if source_url is not None else None,
            source_path=str(source_path) if source_path is not None else None,
            license=str(license_value) if license_value is not None else None,
            title=str(title) if title is not None else None,
            text_blob=text_blob or None,
            image=image,
            extra=extra,
        )

    if row_matches_prefilter(first_row, prefilter_keywords):
        first_sample = safe_row_to_sample(row_to_sample, 0, first_row)
        if first_sample is not None:
            yield first_sample
    for index, row in enumerate(iterator, start=1):
        if max_source_rows and index >= max_source_rows:
            break
        if not row_matches_prefilter(row, prefilter_keywords):
            continue
        sample = safe_row_to_sample(row_to_sample, index, row)
        if sample is not None:
            yield sample


def iter_local_samples(image_dir: str, license_value: str | None) -> Iterator[Sample]:
    if not license_value:
        raise SystemExit("--image-dir-license is required for local image folders")
    image_root = Path(image_dir)
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(image_root.rglob(pattern)))
    for path in paths:
        yield Sample(
            source_dataset="local_image_dir",
            source_id=path.stem,
            source_url=None,
            source_path=str(path),
            license=license_value,
            title=path.stem,
            text_blob=normalize_text(path.stem),
            image=image_to_rgb(Image.open(path)),
            extra={},
        )


def safe_row_to_sample(builder: Any, index: int, row: dict[str, Any]) -> Sample | None:
    try:
        return builder(index, row)
    except Exception as exc:
        print(f"skip row {index}: {exc}")
        return None


def batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_batch(
    model: Any,
    processor: Any,
    device: Any,
    dtype: Any,
    images: list[Image.Image],
    text_features: Any,
) -> Any:
    import torch

    inputs = processor(images=images, return_tensors="pt")
    prepared = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device=device)
        else:
            prepared[key] = value
    with torch.inference_mode():
        image_outputs = model.get_image_features(**prepared)
        image_features = extract_pooled_features(image_outputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = image_features @ text_features.T
    return scores.detach().to("cpu")


def extract_pooled_features(outputs: Any) -> Any:
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    return outputs


def prepare_text_features(model: Any, processor: Any, device: Any, dtype: Any, texts: list[str]) -> Any:
    import torch

    inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
    prepared = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device=device)
        else:
            prepared[key] = value
    with torch.inference_mode():
        text_outputs = model.get_text_features(**prepared)
        text_features = extract_pooled_features(text_outputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def flatten_target_prompts(targets: list[PromptSpec]) -> tuple[list[str], list[tuple[int, int]]]:
    flat_prompts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for spec in targets:
        prompts = spec.prompts or [spec.prompt]
        flat_prompts.extend(prompts)
        offsets.append((cursor, cursor + len(prompts)))
        cursor += len(prompts)
    return flat_prompts, offsets


def aggregate_scores(flat_scores: list[float], offsets: list[tuple[int, int]]) -> list[float]:
    aggregated: list[float] = []
    for start, end in offsets:
        aggregated.append(max(flat_scores[start:end]))
    return aggregated


def compute_metadata_bonus(sample: Sample, target: PromptSpec, bonus: float) -> tuple[float, bool]:
    if not sample.text_blob:
        return 0.0, False
    keywords = [target.key, *target.aliases, *target.metadata_keywords]
    for keyword in keywords:
        normalized = normalize_text(keyword)
        if normalized and normalized in sample.text_blob:
            return bonus, True
    return 0.0, False


def rejection_reasons(
    *,
    sample: Sample,
    top1_score: float,
    score_margin: float,
    center_best_index: int,
    best_index: int,
    center_score: float,
    center_margin: float,
    negative_score_max: float,
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []
    if sample.license in (None, "", "unknown"):
        reasons.append("missing_license")
    if top1_score < args.threshold:
        reasons.append("low_top1_score")
    if score_margin < args.min_margin:
        reasons.append("low_score_margin")
    if center_best_index != best_index:
        reasons.append("center_class_mismatch")
    if center_score < args.min_center_score:
        reasons.append("low_center_score")
    if center_margin < args.min_center_margin:
        reasons.append("low_center_margin")
    if negative_score_max > args.max_negative_score:
        reasons.append("high_negative_score")
    return reasons


def save_candidate_images(
    image: Image.Image,
    images_256_dir: Path,
    images_128_dir: Path,
    file_stem: str,
) -> dict[str, str]:
    image_256_path = images_256_dir / f"{file_stem}.png"
    image_128_path = images_128_dir / f"{file_stem}.png"
    resize_square(image, 256).save(image_256_path)
    resize_square(image, 128).save(image_128_path)
    return {
        "256": str(image_256_path),
        "128": str(image_128_path),
    }


def compute_summary(records: list[dict[str, Any]], target_keys: list[str]) -> dict[str, Any]:
    per_class = defaultdict(list)
    accepted_records = [record for record in records if record.get("accepted")]
    rejected_records = [record for record in records if not record.get("accepted")]
    for record in accepted_records:
        per_class[record["key"]].append(record)
    classes = {}
    for key in target_keys:
        values = [item["score"] for item in per_class.get(key, [])]
        raw_values = [item.get("raw_score") for item in per_class.get(key, [])]
        center_values = [item.get("center_score") for item in per_class.get(key, [])]
        margins = [item["score_margin"] for item in per_class.get(key, [])]
        negatives = [item["negative_score_max"] for item in per_class.get(key, [])]
        classes[key] = {
            "count": len(values),
            "score_mean": sum(values) / len(values) if values else None,
            "raw_score_mean": sum(raw_values) / len(raw_values) if raw_values else None,
            "center_score_mean": sum(center_values) / len(center_values) if center_values else None,
            "score_min": min(values) if values else None,
            "score_max": max(values) if values else None,
            "margin_mean": sum(margins) / len(margins) if margins else None,
            "negative_score_max_mean": sum(negatives) / len(negatives) if negatives else None,
        }
    return {
        "total_count": len(records),
        "accepted_count": len(accepted_records),
        "rejected_count": len(rejected_records),
        "reject_reasons": dict(count_reject_reasons(rejected_records)),
        "classes": classes,
    }


def count_reject_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in str(record.get("reject_reason") or "unknown").split(","):
            reason = reason.strip() or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def make_contact_sheet(records: list[dict[str, Any]], out_path: Path, cell_size: int = 128, label_height: int = 56) -> None:
    if not records:
        return
    columns = min(4, len(records))
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
        line1 = f'{record["key"]} {record["top1_score"]:.3f}'
        line2 = f'm{record["score_margin"]:.3f} c{record.get("center_score", 0.0):.3f}'
        line3 = f'n{record["negative_score_max"]:.3f} b{record.get("metadata_bonus", 0.0):.3f}'
        draw.text((x + 4, y + cell_size + 4), line1, fill=(0, 0, 0))
        draw.text((x + 4, y + cell_size + 20), line2, fill=(0, 0, 0))
        draw.text((x + 4, y + cell_size + 34), line3, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    if not args.dataset and not args.image_dir:
        raise SystemExit("either --dataset or --image-dir is required")

    torch, transformers, _ = require_mining_stack()
    targets, negatives = load_prompt_config(args.config)
    prefilter_keywords = build_prefilter_keywords(targets) if args.metadata_prefilter else None
    out_dir = make_run_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    samples = (
        iter_local_samples(args.image_dir, args.image_dir_license)
        if args.image_dir
        else iter_hf_samples(
            args.dataset,
            args.split,
            args.revision,
            args.streaming,
            args.dataset_license,
            prefilter_keywords,
            args.max_source_rows,
        )
    )

    if args.dry_run:
        first = next(iter(samples), None)
        payload = {
            "source_dataset": first.source_dataset if first else args.dataset,
            "source_id": first.source_id if first else None,
            "license": first.license if first else args.dataset_license,
            "source_url": first.source_url if first else None,
            "source_path": first.source_path if first else None,
            "title": first.title if first else None,
            "image_size": list(first.image.size) if first else None,
            "extra": first.extra if first else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    device = pick_torch_device(torch)
    dtype = torch_dtype_for_device(torch, device)
    processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = transformers.AutoModel.from_pretrained(args.model, dtype=dtype, local_files_only=args.local_files_only)
    model = model.to(device)
    model.eval()
    if hasattr(model, "set_progress_bar_config"):
        model.set_progress_bar_config(disable=True)

    positive_prompt_texts, positive_offsets = flatten_target_prompts(targets)
    negative_prompt_texts, negative_offsets = flatten_target_prompts(negatives)
    prompt_texts = positive_prompt_texts + negative_prompt_texts
    text_features = prepare_text_features(model, processor, device, dtype, prompt_texts)
    started_at = time.perf_counter()
    scanned_count = 0
    skipped_count = 0
    candidates: list[dict[str, Any]] = []
    counts_by_class = defaultdict(int)

    for batch in batched(samples, args.batch_size):
        if args.skip_images and skipped_count < args.skip_images:
            remaining_skip = args.skip_images - skipped_count
            if len(batch) <= remaining_skip:
                skipped_count += len(batch)
                continue
            batch = batch[remaining_skip:]
            skipped_count += remaining_skip
        if scanned_count >= args.max_images:
            break
        batch = batch[: max(0, args.max_images - scanned_count)]
        logits = score_batch(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            images=[sample.image for sample in batch],
            text_features=text_features,
        )
        center_logits = score_batch(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            images=[center_focus_crop(sample.image, args.center_crop_fraction) for sample in batch],
            text_features=text_features,
        )
        for row, center_row, sample in zip(logits.tolist(), center_logits.tolist(), batch):
            scanned_count += 1
            positive_scores = aggregate_scores(row[: len(positive_prompt_texts)], positive_offsets)
            negative_scores = aggregate_scores(row[len(positive_prompt_texts) :], negative_offsets)
            center_positive_scores = aggregate_scores(center_row[: len(positive_prompt_texts)], positive_offsets)
            ordered = sorted(enumerate(positive_scores), key=lambda item: item[1], reverse=True)
            center_ordered = sorted(enumerate(center_positive_scores), key=lambda item: item[1], reverse=True)
            best_index, best_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else -1.0
            margin = best_score - second_score
            best_target = targets[best_index]
            center_best_index, center_best_score = center_ordered[0]
            center_second_score = center_ordered[1][1] if len(center_ordered) > 1 else -1.0
            center_margin = center_best_score - center_second_score
            negative_map = {spec.key: float(score) for spec, score in zip(negatives, negative_scores)}
            negative_score_max = max(negative_map.values()) if negative_map else 0.0
            metadata_bonus, metadata_matched = compute_metadata_bonus(sample, best_target, args.metadata_bonus)
            final_score = best_score + metadata_bonus
            reasons = rejection_reasons(
                sample=sample,
                top1_score=final_score,
                score_margin=margin,
                center_best_index=center_best_index,
                best_index=best_index,
                center_score=center_best_score,
                center_margin=center_margin,
                negative_score_max=negative_score_max,
                args=args,
            )
            candidates.append(
                {
                    "sample": sample,
                    "accepted": not reasons,
                    "reject_reason": ",".join(reasons) if reasons else None,
                    "key": best_target.key,
                    "title": best_target.title or best_target.key,
                    "prompt": best_target.prompt,
                    "score": float(final_score),
                    "top1_score": float(final_score),
                    "top2_score": float(second_score),
                    "raw_score": float(best_score),
                    "center_score": float(center_best_score),
                    "metadata_bonus": float(metadata_bonus),
                    "metadata_matched": metadata_matched,
                    "score_margin": float(margin),
                    "center_score_margin": float(center_margin),
                    "top2_key": targets[ordered[1][0]].key if len(ordered) > 1 else None,
                    "second_best_key": targets[ordered[1][0]].key if len(ordered) > 1 else None,
                    "second_best_score": float(second_score),
                    "positive_scores": {spec.key: float(score) for spec, score in zip(targets, positive_scores)},
                    "center_positive_scores": {spec.key: float(score) for spec, score in zip(targets, center_positive_scores)},
                    "negative_scores": negative_map,
                    "negative_score_max": float(negative_score_max),
                    "width": sample.image.width,
                    "height": sample.image.height,
                }
            )

    selected: list[dict[str, Any]] = []
    for key in [item.key for item in targets]:
        class_records = [item for item in candidates if item["accepted"] and item["key"] == key]
        class_records.sort(key=lambda item: (item["score"], item["score_margin"]), reverse=True)
        selected.extend(class_records[: args.top_k_per_class])
        counts_by_class[key] = min(len(class_records), args.top_k_per_class)

    selected.sort(key=lambda item: (item["score"], item["score_margin"]), reverse=True)
    selected_ids = {id(item) for item in selected}
    for item in candidates:
        if item["accepted"] and id(item) not in selected_ids:
            item["accepted"] = False
            item["reject_reason"] = "class_topk_overflow"
    out_dir.mkdir(parents=True, exist_ok=True)

    serialized: list[dict[str, Any]] = []
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        ordered_candidates = sorted(candidates, key=lambda item: (item["accepted"], item["score"], item["score_margin"]), reverse=True)
        for index, item in enumerate(ordered_candidates):
            sample = item["sample"]
            slug_base = sanitize_id(f'{item["key"]}_{sample.source_id}')
            if not slug_base:
                slug_base = hashlib.sha1(f"{sample.source_dataset}-{index}".encode("utf-8")).hexdigest()[:16]
            file_stem = f"{slug_base}_{index:04d}"
            should_save_images = item["accepted"] or args.save_rejected_images
            saved_images = (
                save_candidate_images(sample.image, images_256_dir, images_128_dir, file_stem)
                if should_save_images
                else None
            )
            record = {
                "id": file_stem,
                "source_type": "open_dataset",
                "source_dataset": sample.source_dataset,
                "source_id": sample.source_id,
                "source_url": sample.source_url,
                "source_path": sample.source_path,
                "license": sample.license,
                "key": item["key"],
                "title": item["title"],
                "prompt": item["prompt"],
                "score": item["score"],
                "top1_score": item["top1_score"],
                "top2_score": item["top2_score"],
                "raw_score": item["raw_score"],
                "center_score": item["center_score"],
                "metadata_bonus": item["metadata_bonus"],
                "metadata_matched": item["metadata_matched"],
                "score_margin": item["score_margin"],
                "center_score_margin": item["center_score_margin"],
                "top2_key": item["top2_key"],
                "second_best_key": item["second_best_key"],
                "second_best_score": item["second_best_score"],
                "negative_score_max": item["negative_score_max"],
                "negative_scores": item["negative_scores"],
                "positive_scores": item["positive_scores"],
                "center_positive_scores": item["center_positive_scores"],
                "accepted": item["accepted"],
                "reject_reason": item["reject_reason"],
                "width": item["width"],
                "height": item["height"],
                "saved_images": saved_images,
                "created_at_unix": int(time.time()),
            }
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            serialized.append(record)

    score_summary = compute_summary(serialized, [item.key for item in targets])
    score_summary["skipped_count"] = skipped_count
    score_summary["scanned_count"] = scanned_count
    score_summary["qualified_count_before_topk"] = len([item for item in candidates if item["reject_reason"] is None or item["reject_reason"] == "class_topk_overflow"])
    score_summary["selected_per_class"] = dict(counts_by_class)
    score_summary["threshold"] = args.threshold
    score_summary["min_margin"] = args.min_margin
    score_summary["max_negative_score"] = args.max_negative_score
    score_summary["metadata_bonus"] = args.metadata_bonus
    score_summary["center_crop_fraction"] = args.center_crop_fraction
    score_summary["min_center_score"] = args.min_center_score
    score_summary["min_center_margin"] = args.min_center_margin
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(score_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    accepted_serialized = [record for record in serialized if record["accepted"]]
    rejected_serialized = [record for record in serialized if not record["accepted"] and record.get("saved_images")]
    make_contact_sheet(accepted_serialized[: min(32, len(accepted_serialized))], top_sheet_path)
    make_contact_sheet(rejected_serialized[: min(32, len(rejected_serialized))], rejected_sheet_path)

    manifest = {
        "phase": "open_mined_siglip2",
        "model": args.model,
        "prompt_config": str(Path(args.config)),
        "source": {
            "dataset": args.dataset,
            "split": args.split,
            "revision": args.revision,
            "image_dir": args.image_dir,
            "dataset_license": args.dataset_license,
            "image_dir_license": args.image_dir_license,
        },
        "metadata_prefilter": args.metadata_prefilter,
        "max_source_rows": args.max_source_rows,
        "device": device.type,
        "dtype": str(dtype),
        "max_images": args.max_images,
        "skip_images": args.skip_images,
        "top_k_per_class": args.top_k_per_class,
        "threshold": args.threshold,
        "min_margin": args.min_margin,
        "max_negative_score": args.max_negative_score,
        "metadata_bonus": args.metadata_bonus,
        "center_crop_fraction": args.center_crop_fraction,
        "min_center_score": args.min_center_score,
        "min_center_margin": args.min_center_margin,
        "save_rejected_images": args.save_rejected_images,
        "batch_size": args.batch_size,
        "skipped_count": skipped_count,
        "scanned_count": scanned_count,
        "metadata_count": len(serialized),
        "accepted_count": len(accepted_serialized),
        "rejected_count": len(serialized) - len(accepted_serialized),
        "selected_count": len(accepted_serialized),
        "elapsed_seconds": time.perf_counter() - started_at,
        "metadata_jsonl": str(metadata_path),
        "reports": {
            "score_summary": str(summary_path),
            "contact_sheet_top_matches": str(top_sheet_path),
            "contact_sheet_rejected": str(rejected_sheet_path),
        },
    }
    write_manifest(out_dir / "manifest.json", manifest)
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
