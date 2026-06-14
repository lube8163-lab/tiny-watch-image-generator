#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research_common import ROOT, pick_torch_device, require_diffusion_stack, torch_dtype_for_device, write_manifest


DEFAULT_SDXL_TEST_CANDIDATES = [
    Path("/Users/tasuku/Desktop/SDXL_test"),
    Path("/Users/tasuku/Desktop/XcodeProjects/SDXL_test"),
]
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_PROMPTS = ROOT / "configs" / "sdxl_tiny_teacher_prompts.json"
DEFAULT_NEGATIVE = (
    "text, logo, watermark, caption, low quality, blurry, noisy, distorted, deformed, "
    "cluttered background, multiple subjects, cropped subject, extra limbs"
)


STYLE_SUFFIXES = [
    "centered subject, simple clean background, readable silhouette",
    "small icon-like illustration, centered composition, simple background",
    "clear toy-like object, front view, simple lighting",
]


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def parse_size_list(value: str) -> list[int]:
    sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise argparse.ArgumentTypeError("image sizes must be positive")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("expected at least one image size")
    return sizes


def safe_token(value: Any) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return token.strip("_") or "item"


def seed_token(seed: int) -> str:
    if seed < 0:
        return f"neg{abs(seed):06d}"
    return f"{seed:06d}"


def resolve_sdxl_test_root(value: str | None) -> Path | None:
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            raise SystemExit(f"SDXL_test root does not exist: {path}")
        return path.resolve()

    for candidate in DEFAULT_SDXL_TEST_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    return None


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def render_template(template: str, preset: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    guard = str(preset.get("guard") or "").strip()
    context = {
        "key": preset["key"],
        "title": preset.get("title") or preset["key"],
        "subject": preset.get("subject") or preset.get("prompt") or preset["key"],
        "guard": guard,
        "guard_clause": f", {guard}" if guard else "",
    }
    for key, value in preset.items():
        if isinstance(value, (str, int, float, bool)):
            context.setdefault(key, str(value))
    if extra:
        context.update({key: str(value) for key, value in extra.items()})
    return " ".join(template.format(**context).split())


def load_prompt_schedule(path: str, limit: int, variants_per_prompt: int) -> tuple[str, str | None, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        presets = payload[:limit] if limit > 0 else payload
        return "legacy_presets", None, expand_legacy_prompts(presets, variants_per_prompt)

    if not isinstance(payload, dict):
        raise SystemExit(f"prompt config must be a JSON object or list: {path}")

    presets = payload.get("presets")
    if not isinstance(presets, list):
        raise SystemExit(f"prompt config is missing a presets list: {path}")

    selected = presets[:limit] if limit > 0 else presets
    global_variants = payload.get("variants") or []
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    prompt_variants: list[dict[str, Any]] = []

    for preset in selected:
        if "key" not in preset:
            raise SystemExit(f"prompt preset is missing key: {preset}")
        variants = preset.get("variants") or global_variants
        if not variants:
            variants = [{"variant": "v00", "conditioning_prompt": "{key}", "prompt": "{subject}"}]
        preset_flags = list(preset.get("qc_flags") or [])
        expanded_variants: list[dict[str, Any]] = []
        seen_variant_ids: set[str] = set()

        for template_index, variant in enumerate(variants):
            if isinstance(variant, str):
                variant_payload = {"prompt": variant}
            elif isinstance(variant, dict):
                variant_payload = variant
            else:
                raise SystemExit(f"variant must be a string or object: {variant}")

            slot_sets = expand_variant_slots(variant_payload, preset, defaults)
            for slot_extra in slot_sets:
                index = len(expanded_variants)
                base_variant_id = str(variant_payload.get("variant") or f"v{template_index:02d}")
                variant_id = safe_variant_id(render_template(base_variant_id, preset, slot_extra))
                if variant_id in seen_variant_ids:
                    variant_id = f"{variant_id}_{index:02d}"
                seen_variant_ids.add(variant_id)
                conditioning_template = str(variant_payload.get("conditioning_prompt") or "{key}")
                teacher_template = str(variant_payload.get("prompt") or "{subject}")
                expanded_variants.append(
                    {
                        "key": str(preset["key"]),
                        "title": str(preset.get("title") or preset["key"]),
                        "subject": str(preset.get("subject") or preset["key"]),
                        "variant": variant_id,
                        "variant_index": index,
                        "prompt": render_template(conditioning_template, preset, slot_extra),
                        "teacher_prompt": render_template(teacher_template, preset, slot_extra),
                        "qc_flags": preset_flags + list(variant_payload.get("qc_flags") or []),
                        "prompt_slots": slot_extra,
                    }
                )

        prompt_variants.extend(expanded_variants[: max(1, variants_per_prompt)])

    return (
        str(payload.get("version") or Path(path).stem),
        payload.get("default_negative_prompt"),
        prompt_variants,
    )


def safe_variant_id(value: str) -> str:
    return safe_token(value).lower()


def expand_variant_slots(
    variant_payload: dict[str, Any],
    preset: dict[str, Any],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    slot_sources = variant_payload.get("slot_source") or {}
    slot_values = variant_payload.get("slot_values") or {}
    if not isinstance(slot_sources, dict):
        raise SystemExit(f"slot_source must be an object: {variant_payload}")
    if not isinstance(slot_values, dict):
        raise SystemExit(f"slot_values must be an object: {variant_payload}")

    slot_options: list[tuple[str, list[tuple[int, str]]]] = []
    for slot_name in sorted(set(slot_sources) | set(slot_values)):
        values: list[str]
        if slot_name in slot_values:
            values = as_string_list(slot_values[slot_name])
        else:
            source_name = str(slot_sources[slot_name])
            values = as_string_list(preset.get(source_name))
            if not values:
                values = as_string_list(defaults.get(source_name))
        if not values:
            return []
        slot_options.append((str(slot_name), list(enumerate(values))))

    if not slot_options:
        return [{}]

    max_expansions = int(variant_payload.get("max_expansions") or 0)
    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*(values for _, values in slot_options)):
        extra: dict[str, Any] = {}
        for (slot_name, _), (slot_index, slot_value) in zip(slot_options, combo):
            extra[slot_name] = slot_value
            extra[f"{slot_name}_index"] = f"{slot_index:02d}"
            extra[f"{slot_name}_token"] = safe_token(slot_value).lower()
        expanded.append(extra)
        if max_expansions > 0 and len(expanded) >= max_expansions:
            break
    return expanded


def expand_legacy_prompts(presets: list[dict[str, Any]], variants_per_prompt: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for preset in presets:
        suffixes = STYLE_SUFFIXES[: max(1, variants_per_prompt)]
        while len(suffixes) < variants_per_prompt:
            suffixes.append(STYLE_SUFFIXES[-1])
        for index, suffix in enumerate(suffixes[:variants_per_prompt]):
            expanded.append(
                {
                    "key": str(preset["key"]),
                    "title": str(preset.get("title") or preset["key"]),
                    "subject": str(preset.get("prompt") or preset["key"]),
                    "variant": f"v{index:02d}",
                    "variant_index": index,
                    "prompt": str(preset.get("conditioning_prompt") or preset.get("key")),
                    "teacher_prompt": f'{preset["prompt"]}, {suffix}',
                    "qc_flags": list(preset.get("qc_flags") or []),
                }
            )
    return expanded


def output_relpaths(stem: str, target_sizes: list[int], width: int, height: int, save_source: bool) -> tuple[dict[str, str], str | None]:
    saved_images = {str(size): str(Path(f"images_{size}") / f"{stem}.png") for size in target_sizes}
    source_image = str(Path(f"source_{width}x{height}") / f"{stem}.png") if save_source else None
    return saved_images, source_image


def resolve_out_path(out_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else out_dir / path


def entry_outputs_exist(entry: dict[str, Any], out_dir: Path, require_source: bool) -> bool:
    saved = entry.get("saved_images") or {}
    if not saved:
        return False
    for rel in saved.values():
        path = resolve_out_path(out_dir, rel)
        if path is None or not path.exists():
            return False
    source_path = resolve_out_path(out_dir, entry.get("source_image"))
    if require_source and (source_path is None or not source_path.exists()):
        return False
    return True


def load_existing_entries(metadata_path: Path, out_dir: Path, require_source: bool) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry_id = str(entry.get("id") or "")
            if not entry_id or entry_id in seen:
                continue
            if entry.get("accepted", True) is True and entry_outputs_exist(entry, out_dir, require_source):
                entries.append(entry)
                seen.add(entry_id)
    return entries


def save_resized(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == (size, size):
        image.save(path)
        return
    image.resize((size, size), Image.Resampling.LANCZOS).save(path)


def compute_image_stats(
    image: Image.Image,
    border_margin_fraction: float,
    border_edge_threshold: float,
    foreground_threshold: float,
    foreground_min_component_area: int,
    foreground_sample_size: int,
) -> dict[str, float | int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
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


def image_reject_reason(
    stats: dict[str, float | int],
    min_range: int,
    min_std: float,
    max_border_std: float,
    max_border_edge_density: float,
    max_foreground_components: int,
    min_largest_foreground_component_ratio: float,
) -> str | None:
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


def save_entry_images(
    image: Image.Image,
    out_dir: Path,
    saved_images: dict[str, str],
    source_image: str | None,
    width: int,
    height: int,
) -> None:
    rgb = image.convert("RGB")
    if source_image:
        source_path = resolve_out_path(out_dir, source_image)
        if source_path is None:
            raise RuntimeError("source path was unexpectedly empty")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(source_path)
    for size_text, relpath in saved_images.items():
        save_resized(rgb, out_dir / relpath, int(size_text))


def make_contact_sheet(entries: list[dict[str, Any]], out_dir: Path, out_path: Path, image_size: int, label_height: int = 34) -> None:
    if not entries:
        return
    columns = min(4, len(entries))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * image_size, rows * (image_size + label_height)), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        x = (index % columns) * image_size
        y = (index // columns) * (image_size + label_height)
        image_path = resolve_out_path(out_dir, entry.get("image"))
        if image_path is None or not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        label = f'{entry["key"]} {entry["variant"]} s{entry["seed"]}'
        draw.text((x + 4, y + image_size + 8), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def build_entry(
    item: dict[str, Any],
    seed: int,
    target_sizes: list[int],
    primary_size: int,
    width: int,
    height: int,
    save_source: bool,
    prompt_set_version: str,
    args: argparse.Namespace,
    scheduler_name: str | None = None,
) -> dict[str, Any]:
    stem = "_".join(
        [
            safe_token(item["key"]),
            safe_token(item["variant"]),
            f"seed{seed_token(seed)}",
        ]
    )
    saved_images, source_image = output_relpaths(stem, target_sizes, width, height, save_source)
    return {
        "accepted": True,
        "id": stem,
        "key": item["key"],
        "title": item["title"],
        "subject": item.get("subject"),
        "prompt": item["prompt"],
        "conditioning_prompt": item["prompt"],
        "teacher_prompt": item["teacher_prompt"],
        "negative_prompt": args.negative_prompt,
        "variant": item["variant"],
        "variant_index": item["variant_index"],
        "seed": seed,
        "model_family": "sdxl",
        "source_type": "mac_diffusers_sdxl",
        "model": args.model,
        "vae": args.vae,
        "model_variant": args.variant,
        "scheduler": scheduler_name,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "source_width": width,
        "source_height": height,
        "target_sizes": target_sizes,
        "image": saved_images[str(primary_size)],
        "saved_images": saved_images,
        "source_image": source_image,
        "prompt_set_version": prompt_set_version,
        "qc_flags": item.get("qc_flags") or [],
    }


def empty_device_cache(torch: Any, device: Any) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local prompt/image pairs with SDXL for tiny model distillation experiments."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--sdxl-test-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--presets", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--out-dir", default=str(ROOT / "datasets" / "sdxl_mac_teacher"))
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--limit", type=int, default=8, help="Number of prompt categories to use. Set <=0 for all.")
    parser.add_argument("--variants-per-prompt", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--target-sizes", default="256,128,64")
    parser.add_argument("--target-size", type=int, default=None, help="Legacy single-size alias for --target-sizes.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Generate this many images per pipeline call. Keep 1 on 8 GB MPS; use 4+ on cloud GPUs.")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--save-source", action="store_true")
    parser.add_argument("--contact-size", type=int, default=128)
    parser.add_argument("--contact-limit", type=int, default=64)
    parser.add_argument("--min-image-range", type=int, default=8)
    parser.add_argument("--min-image-std", type=float, default=2.0)
    parser.add_argument("--border-margin-fraction", type=float, default=0.12)
    parser.add_argument("--border-edge-threshold", type=float, default=20.0)
    parser.add_argument("--max-border-std", type=float, default=0.0, help="Reject images whose border pixel std exceeds this value. Set <=0 to disable.")
    parser.add_argument("--max-border-edge-density", type=float, default=0.0, help="Reject images whose border edge density exceeds this value. Set <=0 to disable.")
    parser.add_argument("--foreground-threshold", type=float, default=30.0)
    parser.add_argument("--foreground-min-component-area", type=int, default=20)
    parser.add_argument("--foreground-sample-size", type=int, default=128)
    parser.add_argument("--max-foreground-components", type=int, default=0, help="Reject images with too many foreground components. Set <=0 to disable.")
    parser.add_argument(
        "--min-largest-foreground-component-ratio",
        type=float,
        default=0.0,
        help="Reject images where the largest foreground component is too small a share of all foreground. Set <=0 to disable.",
    )
    parser.add_argument("--keep-invalid-images", action="store_true")
    parser.add_argument("--no-abort-on-invalid-image", action="store_false", dest="abort_on_invalid_image")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--no-attention-slicing", action="store_false", dest="attention_slicing")
    parser.add_argument("--no-vae-slicing", action="store_false", dest="vae_slicing")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.set_defaults(resume=True, attention_slicing=False, vae_slicing=True, abort_on_invalid_image=True)
    args = parser.parse_args()

    sdxl_test_root = resolve_sdxl_test_root(args.sdxl_test_root)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser()
    elif sdxl_test_root:
        cache_dir = sdxl_test_root / ".cache" / "huggingface" / "hub"
    else:
        cache_dir = ROOT / ".cache" / "huggingface" / "hub"
    target_sizes = [args.target_size] if args.target_size else parse_size_list(args.target_sizes)
    primary_size = target_sizes[0]
    prompt_set_version, config_negative_prompt, prompt_variants = load_prompt_schedule(
        args.presets,
        args.limit,
        args.variants_per_prompt,
    )
    args.negative_prompt = args.negative_prompt or config_negative_prompt or DEFAULT_NEGATIVE
    seeds = parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "metadata.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    torch, diffusers = require_diffusion_stack()
    device = pick_torch_device(torch)
    dtype = torch_dtype_for_device(torch, device)

    vae = None
    if args.vae:
        vae = diffusers.AutoencoderKL.from_pretrained(
            args.vae,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )

    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        vae=vae,
        torch_dtype=dtype,
        variant=args.variant or None,
        local_files_only=args.local_files_only,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if args.vae_slicing:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        elif hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    scheduler_name = pipe.scheduler.__class__.__name__
    total_jobs = len(prompt_variants) * len(seeds)
    existing_entries = [] if args.overwrite or not args.resume else load_existing_entries(metadata_path, out_dir, args.save_source)
    completed_ids = {str(entry["id"]) for entry in existing_entries}
    manifest_entries = list(existing_entries)
    started_at = time.perf_counter()
    generated_count = 0
    reused_count = 0
    abort_reason = None

    generator_device = "cpu" if device.type == "mps" else device.type

    def flush_generation_batch(batch: list[dict[str, Any]], metadata_file: Any) -> None:
        nonlocal generated_count, abort_reason
        if not batch or abort_reason:
            return

        prompts = [entry["teacher_prompt"] for entry in batch]
        negatives = [args.negative_prompt for _ in batch]
        generators = [torch.Generator(device=generator_device).manual_seed(int(entry["seed"])) for entry in batch]
        with torch.inference_mode():
            result = pipe(
                prompt=prompts[0] if len(prompts) == 1 else prompts,
                negative_prompt=negatives[0] if len(negatives) == 1 else negatives,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generators[0] if len(generators) == 1 else generators,
            )

        for base_entry, generated_image in zip(batch, result.images):
            image = generated_image.convert("RGB")
            stats = compute_image_stats(
                image,
                args.border_margin_fraction,
                args.border_edge_threshold,
                args.foreground_threshold,
                args.foreground_min_component_area,
                args.foreground_sample_size,
            )
            base_entry["image_stats"] = stats
            reject_reason = image_reject_reason(
                stats,
                args.min_image_range,
                args.min_image_std,
                args.max_border_std,
                args.max_border_edge_density,
                args.max_foreground_components,
                args.min_largest_foreground_component_ratio,
            )
            if reject_reason:
                base_entry["accepted"] = False
                base_entry["reject_reason"] = reject_reason
                if args.keep_invalid_images:
                    save_entry_images(
                        image,
                        out_dir,
                        base_entry["saved_images"],
                        base_entry["source_image"],
                        args.width,
                        args.height,
                    )
                base_entry["elapsed_seconds"] = time.perf_counter() - started_at
                metadata_file.write(json.dumps(base_entry, ensure_ascii=False) + "\n")
                metadata_file.flush()
                manifest_entries.append(base_entry)
                print(f"[{base_entry['job_index']}/{total_jobs}] rejected {base_entry['id']} {reject_reason}")
                if args.abort_on_invalid_image:
                    abort_reason = f"invalid generated image: {base_entry['id']} ({reject_reason})"
                    return
                continue

            save_entry_images(
                image,
                out_dir,
                base_entry["saved_images"],
                base_entry["source_image"],
                args.width,
                args.height,
            )
            base_entry["elapsed_seconds"] = time.perf_counter() - started_at
            metadata_file.write(json.dumps(base_entry, ensure_ascii=False) + "\n")
            metadata_file.flush()
            manifest_entries.append(base_entry)
            completed_ids.add(base_entry["id"])
            generated_count += 1
            print(f"[{base_entry['job_index']}/{total_jobs}] generated {base_entry['id']}")

        empty_device_cache(torch, device)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for entry in existing_entries:
            metadata_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        completed_or_reused = 0
        pending_batch: list[dict[str, Any]] = []
        for item in prompt_variants:
            for seed in seeds:
                base_entry = build_entry(
                    item,
                    seed,
                    target_sizes,
                    primary_size,
                    args.width,
                    args.height,
                    args.save_source,
                    prompt_set_version,
                    args,
                    scheduler_name,
                )
                completed_or_reused += 1
                base_entry["job_index"] = completed_or_reused
                if base_entry["id"] in completed_ids:
                    print(f"[{completed_or_reused}/{total_jobs}] skip {base_entry['id']}")
                    continue

                if args.resume and not args.overwrite and entry_outputs_exist(base_entry, out_dir, args.save_source):
                    base_entry["reused_existing_images"] = True
                    metadata_file.write(json.dumps(base_entry, ensure_ascii=False) + "\n")
                    metadata_file.flush()
                    manifest_entries.append(base_entry)
                    completed_ids.add(base_entry["id"])
                    reused_count += 1
                    print(f"[{completed_or_reused}/{total_jobs}] reuse {base_entry['id']}")
                    continue

                pending_batch.append(base_entry)
                if len(pending_batch) >= max(1, args.batch_size):
                    flush_generation_batch(pending_batch, metadata_file)
                    pending_batch = []
                if abort_reason:
                    break
            if abort_reason:
                break
        if pending_batch and not abort_reason:
            flush_generation_batch(pending_batch, metadata_file)

    contact_sheet_path = out_dir / "contact_sheet.png"
    accepted_entries = [entry for entry in manifest_entries if entry.get("accepted") is True]
    contact_entries = accepted_entries[: args.contact_limit] if args.contact_limit > 0 else accepted_entries
    make_contact_sheet(contact_entries, out_dir, contact_sheet_path, args.contact_size)
    manifest_path = out_dir / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "phase": "sdxl_teacher_dataset",
            "run_id": out_dir.name,
            "model_family": "sdxl",
            "source_type": "mac_diffusers_sdxl",
            "status": "aborted" if abort_reason else "complete",
            "abort_reason": abort_reason,
            "sdxl_test_root": str(sdxl_test_root) if sdxl_test_root else None,
            "model": args.model,
            "vae": args.vae,
            "variant": args.variant,
            "cache_dir": str(cache_dir),
            "device": device.type,
            "dtype": str(dtype),
            "prompt_set_version": prompt_set_version,
            "preset_limit": args.limit,
            "prompt_variant_count": len(prompt_variants),
            "variants_per_prompt": args.variants_per_prompt,
            "seeds": seeds,
            "planned_jobs": total_jobs,
            "completed_jobs": len(manifest_entries),
            "accepted_jobs": len(accepted_entries),
            "rejected_jobs": len(manifest_entries) - len(accepted_entries),
            "generated_jobs": generated_count,
            "reused_jobs": reused_count,
            "width": args.width,
            "height": args.height,
            "target_sizes": target_sizes,
            "primary_size": primary_size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "batch_size": args.batch_size,
            "scheduler": scheduler_name,
            "negative_prompt": args.negative_prompt,
            "attention_slicing": args.attention_slicing,
            "vae_slicing": args.vae_slicing,
            "min_image_range": args.min_image_range,
            "min_image_std": args.min_image_std,
            "border_margin_fraction": args.border_margin_fraction,
            "border_edge_threshold": args.border_edge_threshold,
            "max_border_std": args.max_border_std,
            "max_border_edge_density": args.max_border_edge_density,
            "foreground_threshold": args.foreground_threshold,
            "foreground_min_component_area": args.foreground_min_component_area,
            "foreground_sample_size": args.foreground_sample_size,
            "max_foreground_components": args.max_foreground_components,
            "min_largest_foreground_component_ratio": args.min_largest_foreground_component_ratio,
            "abort_on_invalid_image": args.abort_on_invalid_image,
            "elapsed_seconds": time.perf_counter() - started_at,
            "metadata_jsonl": str(metadata_path),
            "contact_sheet": str(contact_sheet_path),
            "entries": manifest_entries,
        },
    )
    print(metadata_path)
    print(contact_sheet_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
