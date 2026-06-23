#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from generate_lcm64_reference import (
    decoded_to_image,
    format_stats,
    make_watch_preview,
    run_watch_loop,
)
from research_common import ROOT, write_manifest


DEFAULT_MODEL_DIR = ROOT / "dist" / "lcm_dreamshaper_v7"
DEFAULT_ASSET_DIR = ROOT / "watchos_example" / "WatchPipelineSmokeApp" / "LCM256Assets"
DEFAULT_TEXT_ASSET_DIR = ROOT / "watchos_example" / "WatchPipelineSmokeApp" / "TextEncoderAssets"
DEFAULT_PROMPT_SUITE = ROOT / "configs" / "watch_lcm256_quality_prompts.json"
DEFAULT_GUIDANCE_SCALE = 6.0
OVERVIEW_CONTACT_SHEET_LIMIT = 96


class CLIPBPETokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path) -> None:
        self.vocab = json.loads(vocab_path.read_text())
        self.start_token_id = int(self.vocab["<|startoftext|>"])
        self.end_token_id = int(self.vocab["<|endoftext|>"])
        self.byte_encoder = self._make_byte_encoder()
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        rank = 0
        for raw_line in merges_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            self.bpe_ranks[(parts[0], parts[1])] = rank
            rank += 1
        if not self.bpe_ranks:
            raise ValueError(f"empty BPE merges: {merges_path}")
        self.cache: dict[str, str] = {}

    def tokenize(self, text: str, max_length: int = 77) -> tuple[np.ndarray, int]:
        ids = [self.start_token_id]
        for piece in self._tokenize_to_pieces(text):
            ids.append(int(self.vocab[piece]))
        ids.append(self.end_token_id)
        if len(ids) > max_length:
            ids = ids[:max_length]
            ids[max_length - 1] = self.end_token_id
        token_count = len(ids)
        ids.extend([self.end_token_id] * (max_length - len(ids)))
        return np.asarray(ids, dtype=np.int32).reshape(1, max_length), token_count

    def _tokenize_to_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for token in self._split_for_clip(text):
            encoded = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            bpe_text = self._bpe(encoded)
            if bpe_text:
                pieces.extend(bpe_text.split(" "))
        return pieces

    def _split_for_clip(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        tokens: list[str] = []
        scalars: list[str] = []
        current_kind: str | None = None

        def kind(char: str) -> str:
            if char.isalpha():
                return "letters"
            if char.isdecimal():
                return "digits"
            return "symbols"

        def flush() -> None:
            nonlocal current_kind
            if scalars:
                tokens.append("".join(scalars))
                scalars.clear()
            current_kind = None

        for char in normalized:
            if char.isspace():
                flush()
                continue
            next_kind = kind(char)
            if current_kind is not None and current_kind != next_kind:
                flush()
            current_kind = next_kind
            scalars.append(char)
        flush()
        return tokens

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        if not word:
            self.cache[token] = ""
            return ""
        word[-1] = word[-1] + "</w>"
        while len(word) > 1:
            pairs = {(word[index], word[index + 1]) for index in range(len(word) - 1)}
            best = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, math.inf))
            if best not in self.bpe_ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(word):
                if index < len(word) - 1 and word[index] == best[0] and word[index + 1] == best[1]:
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = merged
        result = " ".join(word)
        self.cache[token] = result
        return result

    @staticmethod
    def _make_byte_encoder() -> list[str]:
        bytes_ = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        code_points = list(bytes_)
        next_code = 0
        for byte in range(256):
            if byte in bytes_:
                continue
            bytes_.append(byte)
            code_points.append(256 + next_code)
            next_code += 1
        output = [""] * 256
        for byte, code_point in zip(bytes_, code_points):
            output[byte] = chr(code_point)
        return output


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalized_search_text(text: str) -> str:
    output: list[str] = []
    previous_was_space = True
    for char in text.lower():
        if char.isalnum():
            output.append(char)
            previous_was_space = False
        elif not previous_was_space:
            output.append(" ")
            previous_was_space = True
    return "".join(output).strip()


def token_set(text: str) -> set[str]:
    return {token for token in text.split(" ") if len(token) > 1}


def contains_any_normalized_term(normalized_text: str, words: list[str]) -> bool:
    tokens = token_set(normalized_text)
    for word in words:
        normalized_word = normalized_search_text(word)
        if not normalized_word:
            continue
        if " " in normalized_word and normalized_word in normalized_text:
            return True
        if normalized_word in tokens:
            return True
    return False


def prompt_run_key(prompt: str) -> str:
    slug = "_".join(normalized_search_text(prompt).split())
    return slug or "prompt"


def expanded_text_conditioning_prompt(prompt: str) -> str:
    trimmed = prompt.strip()
    style_clauses = ["clean anime illustration", "simple background"]
    if not trimmed:
        return "cat mascot, single subject, " + ", ".join(style_clauses)

    normalized = normalized_search_text(trimmed)
    preserves_plural_intent = contains_any_normalized_term(
        normalized,
        ["two", "three", "multiple", "many", "group", "crowd", "pair"],
    )
    has_compositional_relation = contains_any_normalized_term(
        normalized,
        ["in", "on", "with", "riding", "holding", "wearing", "inside", "under", "over", "near"],
    )
    is_closeup_prompt = contains_any_normalized_term(
        normalized,
        ["face", "head", "portrait", "closeup", "close"],
    )
    is_scene_prompt = contains_any_normalized_term(
        normalized,
        ["landscape", "scene", "mountain", "snowy", "forest", "ocean", "sky", "city", "desert"],
    )
    clauses = [trimmed]
    if not preserves_plural_intent and not contains_any_normalized_term(normalized, ["single", "one", "solo"]):
        clauses.append("centered composition" if has_compositional_relation or is_scene_prompt else "single subject")
    if not contains_any_normalized_term(normalized, ["center", "centered"]):
        clauses.append("centered")
    if not is_closeup_prompt and not is_scene_prompt and not contains_any_normalized_term(
        normalized,
        ["full", "visible", "body"],
    ):
        clauses.append("full object visible")
    if not contains_any_normalized_term(
        normalized,
        [
            "anime",
            "illustration",
            "illustrated",
            "icon",
            "logo",
            "sticker",
            "photo",
            "photograph",
            "photorealistic",
            "pixel",
            "3d",
        ],
    ):
        clauses.append("clean anime illustration")
    elif not contains_any_normalized_term(normalized, ["clean", "simple"]):
        clauses.append("clean illustration")
    if not contains_any_normalized_term(normalized, ["background", "scene", "landscape"]):
        clauses.append("simple background")
    return ", ".join(clauses)


def make_timestep_cond(guidance_scale: float, shape: list[int]) -> np.ndarray:
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError(f"unexpected timestep_cond shape: {shape}")
    embedding_dim = int(shape[1])
    half_dim = embedding_dim // 2
    if half_dim <= 1:
        raise ValueError(f"unexpected timestep_cond dim: {embedding_dim}")
    scale = (float(guidance_scale) - 1.0) * 1000.0
    exponent_base = math.log(10000.0) / float(half_dim - 1)
    values = []
    for index in range(half_dim):
        frequency = math.exp(float(index) * -exponent_base)
        values.append(math.sin(scale * frequency))
    for index in range(half_dim):
        frequency = math.exp(float(index) * -exponent_base)
        values.append(math.cos(scale * frequency))
    if embedding_dim % 2 == 1:
        values.append(0.0)
    return np.asarray(values, dtype=np.float16).reshape(shape)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


def compute_image_metrics(image: Image.Image, clipped_channels: int, total_channels: int) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    max_channel = np.max(arr, axis=2)
    min_channel = np.min(arr, axis=2)
    saturation = np.where(max_channel > 0.0, (max_channel - min_channel) / np.maximum(max_channel, 1e-6), 0.0)
    dx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    return {
        "brightness_mean": float(np.mean(luma)),
        "contrast_std": float(np.std(luma)),
        "saturation_mean": float(np.mean(saturation)),
        "edge_energy": float((dx + dy) / 2.0),
        "dark_fraction": float(np.mean(luma < 0.08)),
        "light_fraction": float(np.mean(luma > 0.92)),
        "clipped_fraction": float(clipped_channels / max(1, total_channels)),
    }


def metric_flags(metrics: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if metrics["contrast_std"] < 0.045:
        flags.append("low_contrast")
    if metrics["brightness_mean"] < 0.12:
        flags.append("very_dark")
    if metrics["brightness_mean"] > 0.88:
        flags.append("very_light")
    if metrics["clipped_fraction"] > 0.02:
        flags.append("high_clip")
    if metrics["edge_energy"] < 0.010:
        flags.append("very_soft")
    return flags


def load_prompt_suite(path: Path, prompt_ids: set[str] | None, genres: set[str] | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    suite = load_json(path)
    prompts = list(suite["prompts"])
    if prompt_ids:
        prompts = [item for item in prompts if item["id"] in prompt_ids]
    if genres:
        prompts = [item for item in prompts if item["genre"] in genres]
    if not prompts:
        raise SystemExit("prompt selection is empty")
    return suite, prompts


def compute_units_from_arg(value: str):
    import coremltools as ct

    mapping = {
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }
    return mapping[value]


def load_coreml_models(args: argparse.Namespace):
    import coremltools as ct

    compute_units = compute_units_from_arg(args.compute_units)
    text_encoder = ct.models.MLModel(str(args.text_encoder_package), compute_units=compute_units)
    unet = ct.models.MLModel(str(args.unet_package), compute_units=compute_units)
    decoder = ct.models.MLModel(str(args.decoder_package), compute_units=compute_units)
    unet_input_names = {item.name for item in unet.get_spec().description.input}
    return text_encoder, unet, decoder, unet_input_names


def make_prompt_embedding(
    *,
    text_encoder,
    tokenizer: CLIPBPETokenizer,
    prompt: str,
) -> tuple[np.ndarray, int, float]:
    input_ids, token_count = tokenizer.tokenize(prompt, max_length=77)
    started = time.perf_counter()
    output = text_encoder.predict({"input_ids": input_ids})
    elapsed = time.perf_counter() - started
    hidden_states = np.asarray(output["hidden_states"], dtype=np.float16)
    if list(hidden_states.shape) != [1, 77, 768]:
        raise ValueError(f"unexpected hidden_states shape: {hidden_states.shape}")
    return hidden_states, token_count, elapsed


def generate_one(
    *,
    prompt_item: dict[str, str],
    seed: int,
    scheduler: dict[str, Any],
    text_encoder,
    tokenizer: CLIPBPETokenizer,
    unet,
    unet_input_names: set[str],
    decoder,
    timestep_cond: np.ndarray,
    expand_prompts: bool,
) -> dict[str, Any]:
    raw_prompt = prompt_item["prompt"]
    conditioning_prompt = expanded_text_conditioning_prompt(raw_prompt) if expand_prompts else raw_prompt
    prompt_embedding, token_count, text_encoder_seconds = make_prompt_embedding(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=conditioning_prompt,
    )
    seed_key = prompt_run_key(raw_prompt)

    def predict_noise(latents: np.ndarray, step: dict[str, float]) -> np.ndarray:
        values = {
            "sample": latents.astype(np.float16),
            "timestep": np.array([float(step["timestep"])], dtype=np.float16),
            "encoder_hidden_states": prompt_embedding,
            "timestep_cond": timestep_cond,
        }
        filtered = {name: value for name, value in values.items() if name in unet_input_names}
        output = unet.predict(filtered)
        return np.asarray(output["noise_pred"], dtype=np.float32)

    def decode_latents(latents: np.ndarray) -> np.ndarray:
        output = decoder.predict({"latents": latents.astype(np.float16)})
        return np.asarray(output["decoded"], dtype=np.float32)

    result = run_watch_loop(
        scheduler=scheduler,
        prompt_key=seed_key,
        seed=seed,
        predict_noise=predict_noise,
        decode_latents=decode_latents,
    )
    image = decoded_to_image(result["decoded"])
    metrics = compute_image_metrics(image, result["clipped_channels"], result["total_channels"])
    return {
        "prompt_id": prompt_item["id"],
        "genre": prompt_item["genre"],
        "prompt": raw_prompt,
        "conditioning_prompt": conditioning_prompt,
        "run_key": seed_key,
        "seed": int(seed),
        "token_count": int(token_count),
        "text_encoder_seconds": text_encoder_seconds,
        "total_elapsed_seconds": result["total_elapsed_seconds"] + text_encoder_seconds,
        "denoise_decode_seconds": result["total_elapsed_seconds"],
        "decoder_elapsed_seconds": result["decoder_elapsed_seconds"],
        "final_stats": result["final_stats"],
        "decoded_stats": result["decoded_stats"],
        "clipped_channels": result["clipped_channels"],
        "total_channels": result["total_channels"],
        "image_metrics": metrics,
        "flags": metric_flags(metrics),
        "image": image,
    }


def save_result_image(result: dict[str, Any], out_dir: Path, preview_mode: str) -> None:
    genre_dir = out_dir / "images" / slugify(result["genre"])
    genre_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slugify(result['prompt_id'])}_seed{result['seed']}.png"
    path = genre_dir / filename
    result["image"].save(path)
    result["image_path"] = str(path)
    result["preview_mode"] = preview_mode
    result["preview_path"] = None
    result["display_image"] = result["image"]

    if preview_mode == "sharp2x":
        preview = make_watch_preview(result["image"], "sharp2x")
        if preview is not None:
            preview_dir = out_dir / "previews" / slugify(result["genre"])
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / filename
            preview.save(preview_path)
            result["preview_path"] = str(preview_path)
            result["display_image"] = preview


def make_contact_sheet(results: list[dict[str, Any]], path: Path, columns: int = 6, thumb_size: int = 160) -> None:
    if not results:
        return
    label_h = 48
    pad = 10
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_size + (columns + 1) * pad, rows * (thumb_size + label_h) + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        row = index // columns
        col = index % columns
        x = pad + col * (thumb_size + pad)
        y = pad + row * (thumb_size + label_h + pad)
        image = result["display_image"].resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
        sheet.paste(image, (x, y + label_h))
        label = f"{result['prompt'][:28]} s{result['seed']}"
        metrics = result["image_metrics"]
        draw.text((x, y), label, fill=(0, 0, 0))
        draw.text(
            (x, y + 14),
            f"b {metrics['brightness_mean']:.2f} c {metrics['contrast_std']:.2f} e {metrics['edge_energy']:.2f}",
            fill=(70, 70, 70),
        )
        if result["flags"]:
            draw.text((x, y + 28), ",".join(result["flags"])[:32], fill=(150, 60, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(results),
        "failed_count": 0,
        "elapsed_seconds_sum": float(sum(item["total_elapsed_seconds"] for item in results)),
        "elapsed_seconds_mean": float(np.mean([item["total_elapsed_seconds"] for item in results])) if results else 0.0,
        "flag_counts": {},
        "by_genre": {},
    }
    flag_counts: dict[str, int] = defaultdict(int)
    by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_genre[result["genre"]].append(result)
        for flag in result["flags"]:
            flag_counts[flag] += 1
    summary["flag_counts"] = dict(sorted(flag_counts.items()))
    for genre, genre_results in sorted(by_genre.items()):
        summary["by_genre"][genre] = {
            "count": len(genre_results),
            "elapsed_seconds_mean": float(np.mean([item["total_elapsed_seconds"] for item in genre_results])),
            "brightness_mean": float(np.mean([item["image_metrics"]["brightness_mean"] for item in genre_results])),
            "contrast_std_mean": float(np.mean([item["image_metrics"]["contrast_std"] for item in genre_results])),
            "edge_energy_mean": float(np.mean([item["image_metrics"]["edge_energy"] for item in genre_results])),
            "flagged": int(sum(1 for item in genre_results if item["flags"])),
        }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_safe(result: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    output.pop("image", None)
    output.pop("display_image", None)
    return output


def write_report(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    started_at: str,
) -> None:
    lines = [
        "# Watch LCM256 Quality Eval",
        "",
        f"- Started: `{started_at}`",
        f"- Prompt suite: `{args.prompt_suite}`",
        f"- Images completed: `{len(results)}`",
        f"- Failures: `{len(failures)}`",
        f"- Compute units: `{args.compute_units}`",
        f"- Guidance: `{args.guidance_scale}`",
        f"- Prompt expansion: `{not args.no_expand_prompts}`",
        f"- Preview mode: `{args.preview_mode}`",
        f"- Mean elapsed/image: `{summary['elapsed_seconds_mean']:.3f}s`",
        f"- Total measured elapsed: `{summary['elapsed_seconds_sum']:.3f}s`",
        "",
        "## Inputs",
        "",
        f"- Suite name: `{suite.get('name', 'unknown')}`",
        f"- Scheduler: `{args.asset_dir / 'lcm_scheduler.json'}`",
        f"- Text encoder: `{args.text_encoder_package}`",
        f"- UNet: `{args.unet_package}`",
        f"- Decoder: `{args.decoder_package}`",
        "",
        "## Genre Summary",
        "",
        "| Genre | Count | Mean Time | Brightness | Contrast | Edge | Flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for genre, item in summary["by_genre"].items():
        lines.append(
            f"| `{genre}` | {item['count']} | {item['elapsed_seconds_mean']:.2f}s | "
            f"{item['brightness_mean']:.2f} | {item['contrast_std_mean']:.2f} | "
            f"{item['edge_energy_mean']:.2f} | {item['flagged']} |"
        )
    lines.extend(["", "## Heuristic Flags", ""])
    if summary["flag_counts"]:
        for flag, count in summary["flag_counts"].items():
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- No heuristic flags.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The automatic metrics only catch obvious numeric outliers such as very low contrast or clipping.",
            "- Semantic prompt fit still needs visual review of the contact sheets.",
            "- This uses Mac Core ML `.mlpackage` assets with CPU-only execution. Treat it as quality-equivalent for ranking, not as a replacement for final Apple Watch memory/runtime checks.",
            "",
            "## Files",
            "",
            "- `manifest.json`: run metadata and aggregate results.",
            "- `manifest.jsonl`: one JSON record per generated image.",
            f"- `contact_sheets/overview.png`: up to the first {OVERVIEW_CONTACT_SHEET_LIMIT} generated images.",
            "- `contact_sheets/by_genre/*.png`: per-genre visual review sheets.",
            "- `images/`: raw generated 256px PNGs.",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:20]:
            lines.append(f"- `{failure['prompt_id']}` seed `{failure['seed']}`: {failure['error']}")
        if len(failures) > 20:
            lines.append(f"- ...and {len(failures) - 20} more")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and summarize Mac-side Watch LCM256 quality sweeps.")
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--text-asset-dir", type=Path, default=DEFAULT_TEXT_ASSET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-encoder-package", type=Path, default=None)
    parser.add_argument("--unet-package", type=Path, default=None)
    parser.add_argument("--decoder-package", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument("--genres", nargs="*", default=None)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--compute-units", choices=["cpuOnly", "cpuAndGPU", "all"], default="cpuOnly")
    parser.add_argument("--preview-mode", choices=["none", "smooth", "sharp2x"], default="smooth")
    parser.add_argument("--no-expand-prompts", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.text_encoder_package = args.text_encoder_package or (args.model_dir / "text_encoder_probe" / "clip_text_encoder_77.mlpackage")
    args.unet_package = args.unet_package or (args.model_dir / "unet_32x32_6bit.mlpackage")
    args.decoder_package = args.decoder_package or (args.model_dir / "vae_decoder_256x256_noattn_4bit.mlpackage")
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_slug = datetime.now().strftime("lcm256_quality_%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "reports" / "watch_lcm256_quality" / run_slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = set(args.prompt_ids) if args.prompt_ids else None
    genres = set(args.genres) if args.genres else None
    suite, prompts = load_prompt_suite(args.prompt_suite, prompt_ids, genres)
    seeds = args.seeds or suite.get("defaultSeeds") or [1, 7, 24, 42]

    scheduler = load_json(args.asset_dir / "lcm_scheduler.json")
    scheduler = dict(scheduler)
    scheduler["guidanceScale"] = float(args.guidance_scale)
    timestep_cond = make_timestep_cond(float(args.guidance_scale), scheduler["timestepCondShape"])

    tokenizer = CLIPBPETokenizer(args.text_asset_dir / "clip_vocab.json", args.text_asset_dir / "clip_merges.txt")
    print(f"load models compute={args.compute_units}", flush=True)
    text_encoder, unet, decoder, unet_input_names = load_coreml_models(args)

    jobs = [(prompt, int(seed)) for prompt in prompts for seed in seeds]
    if args.max_images is not None:
        jobs = jobs[: args.max_images]
    print(f"jobs={len(jobs)} prompts={len(prompts)} seeds={seeds} out={out_dir}", flush=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (prompt_item, seed) in enumerate(jobs, start=1):
        label = f"{prompt_item['id']} seed={seed}"
        item_started = time.perf_counter()
        try:
            result = generate_one(
                prompt_item=prompt_item,
                seed=seed,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                unet_input_names=unet_input_names,
                decoder=decoder,
                timestep_cond=timestep_cond,
                expand_prompts=not args.no_expand_prompts,
            )
            save_result_image(result, out_dir, args.preview_mode)
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {label} "
                f"time={result['total_elapsed_seconds']:.2f}s "
                f"decoded {format_stats(result['decoded_stats'])} "
                f"flags={','.join(result['flags']) or '-'}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "prompt_id": prompt_item["id"],
                "prompt": prompt_item["prompt"],
                "genre": prompt_item["genre"],
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - item_started,
            }
            failures.append(failure)
            print(f"[{index}/{len(jobs)}] FAILED {label}: {failure['error']}", flush=True)
            if args.stop_on_error:
                raise

    contact_dir = out_dir / "contact_sheets"
    make_contact_sheet(results[:OVERVIEW_CONTACT_SHEET_LIMIT], contact_dir / "overview.png")
    by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_genre[result["genre"]].append(result)
    for genre, genre_results in sorted(by_genre.items()):
        make_contact_sheet(genre_results, contact_dir / "by_genre" / f"{slugify(genre)}.png")

    safe_results = [manifest_safe(result) for result in results]
    summary = summarize(results)
    summary["failed_count"] = len(failures)
    manifest = {
        "phase": "watch_lcm256_quality_eval",
        "started_at": started_at,
        "prompt_suite": str(args.prompt_suite),
        "asset_dir": str(args.asset_dir),
        "text_asset_dir": str(args.text_asset_dir),
        "text_encoder_package": str(args.text_encoder_package),
        "unet_package": str(args.unet_package),
        "decoder_package": str(args.decoder_package),
        "compute_units": args.compute_units,
        "guidance_scale": float(args.guidance_scale),
        "seeds": seeds,
        "max_images": args.max_images,
        "preview_mode": args.preview_mode,
        "expanded_prompts": not args.no_expand_prompts,
        "summary": summary,
        "results": safe_results,
        "failures": failures,
    }
    write_manifest(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "manifest.jsonl", safe_results)
    if failures:
        write_jsonl(out_dir / "failures.jsonl", failures)
    write_report(
        out_dir=out_dir,
        args=args,
        suite=suite,
        results=results,
        failures=failures,
        summary=summary,
        started_at=started_at,
    )
    print(out_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
