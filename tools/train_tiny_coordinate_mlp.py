#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from prompt_normalization import (
    PROMPT_ENCODER_COMPOSITIONAL,
    PROMPT_ENCODER_HASH,
    PROMPT_ENCODERS,
    canonicalize_prompt,
    fnv1a_64,
    make_prompt_latent,
)


ROOT = Path(__file__).resolve().parents[1]
SWIFT_OUT = ROOT / "Sources" / "TinyWatchGenerator" / "TinyWeights.swift"
JSON_OUT = ROOT / "weights" / "tiny_weights.json"
BIN_OUT = ROOT / "watchos_example" / "TinyImageWatchApp" / "TinyWeights.bin"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    prompt: str
    seed: int
    key: str
    source: str


def make_latent(prompt: str, seed: int, count: int, prompt_encoder: str) -> list[float]:
    return make_prompt_latent(prompt, seed, count, prompt_encoder)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_prompt(row: dict, preserve_prompt: bool) -> str:
    raw = str(
        row.get("conditioning_prompt")
        or row.get("prompt")
        or row.get("caption")
        or row.get("key")
        or row.get("title")
        or "image"
    )
    return raw if preserve_prompt else canonicalize_prompt(raw)


def row_key(row: dict) -> str:
    return canonicalize_prompt(str(row.get("key") or row.get("prompt") or row.get("caption") or row.get("title") or "image"))


def resolve_image_path(root: Path, rel: str | Path) -> Path | None:
    path = Path(rel)
    candidates = [path] if path.is_absolute() else [root / path, ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def row_image_path(root: Path, row: dict, image_size: int) -> Path | None:
    saved = row.get("saved_images") or {}
    rel = saved.get(str(image_size)) or saved.get("128") or saved.get("256") or row.get("image")
    if not rel:
        return None
    return resolve_image_path(root, rel)


def load_teacher_root(
    path: Path,
    image_size: int,
    source_name: str,
    keys: set[str] | None = None,
    preserve_prompt: bool = False,
    max_per_key: int = 0,
) -> list[Sample]:
    samples: list[Sample] = []
    per_key_counts: Counter[str] = Counter()
    for row in load_jsonl(path / "metadata.jsonl"):
        if row.get("accepted", True) is not True:
            continue
        image_path = row_image_path(path, row, image_size)
        if image_path is None:
            continue
        key = row_key(row)
        if keys is not None and key not in keys:
            continue
        if max_per_key and per_key_counts[key] >= max_per_key:
            continue
        seed = per_key_counts[key]
        per_key_counts[key] += 1
        samples.append(
            Sample(
                image_path=image_path,
                prompt=row_prompt(row, preserve_prompt),
                seed=seed,
                key=key,
                source=source_name,
            )
        )
    return samples


def load_openimages_root(
    path: Path,
    image_size: int,
    limit: int,
    seed: int,
    keys: set[str] | None,
    max_per_key: int,
    preserve_prompt: bool = False,
) -> list[Sample]:
    rows = [r for r in load_jsonl(path / "metadata.jsonl") if r.get("accepted") is True]
    rng = random.Random(seed)
    rng.shuffle(rows)
    samples: list[Sample] = []
    per_key_counts: Counter[str] = Counter()
    for row in rows:
        image_path = row_image_path(path, row, image_size)
        if image_path is None:
            continue
        key = row_key(row)
        if keys is not None and key not in keys:
            continue
        if max_per_key and per_key_counts[key] >= max_per_key:
            continue
        # Open Images has many examples for the same prompt. Use a stable per-row
        # seed so the tiny model can learn prompt + seed as a compact image index.
        row_seed = int(fnv1a_64(str(row.get("source_id") or row.get("source_path") or len(samples))) & 0x7FFFFFFF)
        samples.append(
            Sample(
                image_path=image_path,
                prompt=row_prompt(row, preserve_prompt),
                seed=row_seed,
                key=key,
                source="openimages",
            )
        )
        per_key_counts[key] += 1
        if limit and len(samples) >= limit:
            break
    return samples


def normalize_teacher_seeds(samples: list[Sample]) -> list[Sample]:
    counts: Counter[str] = Counter()
    normalized: list[Sample] = []
    for sample in samples:
        if sample.source == "openimages":
            normalized.append(sample)
            continue
        seed = counts[sample.key]
        counts[sample.key] += 1
        normalized.append(replace(sample, seed=seed))
    return normalized


def parse_key_filter(spec: str) -> set[str] | None:
    if not spec.strip() or spec.strip() in {"*", "all"}:
        return None
    return {canonicalize_prompt(k) for k in spec.split(",") if k.strip()}


def load_image_tensor(
    samples: list[Sample],
    size: int,
    target_downsample_size: int,
    target_blur_radius: float,
) -> torch.Tensor:
    images = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        if 0 < target_downsample_size < size:
            image = image.resize(
                (target_downsample_size, target_downsample_size),
                Image.Resampling.LANCZOS,
            ).resize((size, size), Image.Resampling.BICUBIC)
        if target_blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=target_blur_radius))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images.append(arr)
    return torch.from_numpy(np.stack(images, axis=0))


def make_feature_table(samples: list[Sample], latent_count: int, prompt_encoder: str) -> torch.Tensor:
    latents = [make_latent(s.prompt, s.seed, latent_count, prompt_encoder) for s in samples]
    return torch.tensor(latents, dtype=torch.float32)


class TinyCoordinateMLP(torch.nn.Module):
    def __init__(
        self,
        latent_count: int,
        hidden_count: int,
        coord_frequencies: list[float],
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        self.latent_count = latent_count
        self.coord_frequencies = coord_frequencies
        self.coord_count = 4 + len(coord_frequencies) * 4
        self.input_count = self.coord_count + latent_count
        self.hidden_count = hidden_count
        self.hidden_layers = hidden_layers
        layers = [torch.nn.Linear(self.input_count, hidden_count)]
        layers.extend(torch.nn.Linear(hidden_count, hidden_count) for _ in range(hidden_layers - 1))
        self.layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(hidden_count, 3)

    def forward(self, xy: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        fx = xy[:, 0:1]
        fy = xy[:, 1:2]
        radius = torch.clamp(torch.sqrt(fx * fx + fy * fy), max=1.0)
        bias = torch.ones_like(fx)
        features = [fx, fy, radius, bias]
        for frequency in self.coord_frequencies:
            phase_x = latent[:, 0:1] * (0.35 * frequency)
            phase_y = latent[:, 1:2] * (0.35 * frequency)
            scaled_x = fx * (math.pi * frequency)
            scaled_y = fy * (math.pi * frequency)
            features.extend(
                [
                    torch.sin(scaled_x + phase_x),
                    torch.cos(scaled_x + phase_x),
                    torch.sin(scaled_y + phase_y),
                    torch.cos(scaled_y + phase_y),
                ]
            )
        x = torch.cat(features + [latent], dim=1)
        for layer in self.layers:
            x = torch.tanh(layer(x).clamp(-3.0, 3.0))
        return torch.sigmoid(self.output_layer(x) * 1.8)


def quantize_weight(tensor: torch.Tensor) -> tuple[list[int], float]:
    values = tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
    scale = max(float(np.max(np.abs(values))) / 127.0, 1e-6)
    q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return [int(v) for v in q.tolist()], scale


def swift_array_int(name: str, values: list[int]) -> str:
    return f"    static let {name}: [Int8] = [{', '.join(str(v) for v in values)}]"


def swift_array_float(name: str, values: list[float]) -> str:
    return f"    static let {name}: [Float] = [{', '.join(f'{v:.8} as Float' for v in values)}]"


def write_binary_weights(payload: dict, bin_out: Path) -> None:
    bin_out.parent.mkdir(parents=True, exist_ok=True)
    output_index = int(payload.get("hiddenLayerCount", 2)) + 1
    with bin_out.open("wb") as f:
        for index in range(1, output_index + 1):
            for key in (f"w{index}", f"b{index}"):
                values = payload[key]
                if key.startswith("w"):
                    f.write(bytes((int(v) & 0xFF for v in values)))
                else:
                    f.write(struct.pack(f"<{len(values)}f", *values))


def swift_weight_source(
    model: TinyCoordinateMLP,
    scales: list[float],
    trained_seed_count: int,
    prompt_encoder: str,
) -> str:
    output_index = model.hidden_layers + 1
    lines = [
        "// Generated by tools/train_tiny_coordinate_mlp.py",
        "import Foundation",
        "",
        "enum TinyWeights {",
        f"    static let latentCount = {model.latent_count}",
        f"    static let inputCount = {model.input_count}",
        f"    static let hiddenCount = {model.hidden_count}",
        f"    static let hiddenLayerCount = {model.hidden_layers}",
        "    static let outputCount = 3",
        f"    static let trainedSeedCount = {trained_seed_count}",
        f"    static let promptEncoder = \"{prompt_encoder}\"",
        swift_array_float("coordFrequencies", model.coord_frequencies),
    ]
    for index, scale in enumerate(scales, start=1):
        lines.append(f"    static let w{index}Scale = {scale:.10} as Float")
    if output_index == 3:
        lines.append("    static let w4Scale = 1.0 as Float")
    for index in range(1, output_index + 1):
        lines.append(f"    static var w{index}: [Float] {{ storage.w{index} }}")
        lines.append(f"    static var b{index}: [Float] {{ storage.b{index} }}")
    if output_index == 3:
        lines.append("    static var w4: [Float] { [] }")
        lines.append("    static var b4: [Float] { [] }")
    lines.extend(
        [
            "",
            "    private static let storage = TinyRuntimeWeights.load()",
            "}",
            "",
            "private struct TinyRuntimeWeights {",
        ]
    )
    for index in range(1, output_index + 1):
        lines.append(f"    let w{index}: [Float]")
        lines.append(f"    let b{index}: [Float]")
    lines.extend(
        [
            "",
            "    static func load() -> TinyRuntimeWeights {",
            "        guard let url = Bundle.main.url(forResource: \"TinyWeights\", withExtension: \"bin\") ?? developmentWeightURL() else {",
            "            fatalError(\"TinyWeights.bin is missing from the app bundle\")",
            "        }",
            "        let data: Data",
            "        do {",
            "            data = try Data(contentsOf: url, options: [.mappedIfSafe])",
            "        } catch {",
            "            fatalError(\"TinyWeights.bin could not be loaded: \\(error)\")",
            "        }",
            "        var reader = TinyWeightReader(data: data)",
            "        let w1 = reader.readScaledInt8(count: TinyWeights.inputCount * TinyWeights.hiddenCount, scale: TinyWeights.w1Scale)",
            "        let b1 = reader.readFloat32(count: TinyWeights.hiddenCount)",
        ]
    )
    for index in range(2, output_index):
        lines.append(f"        let w{index} = reader.readScaledInt8(count: TinyWeights.hiddenCount * TinyWeights.hiddenCount, scale: TinyWeights.w{index}Scale)")
        lines.append(f"        let b{index} = reader.readFloat32(count: TinyWeights.hiddenCount)")
    lines.extend(
        [
            f"        let w{output_index} = reader.readScaledInt8(count: TinyWeights.hiddenCount * TinyWeights.outputCount, scale: TinyWeights.w{output_index}Scale)",
            f"        let b{output_index} = reader.readFloat32(count: TinyWeights.outputCount)",
            "        reader.assertFinished()",
        ]
    )
    fields = ", ".join(f"w{index}: w{index}, b{index}: b{index}" for index in range(1, output_index + 1))
    lines.append(f"        return TinyRuntimeWeights({fields})")
    lines.extend(
        [
            "    }",
            "",
            "    private static func developmentWeightURL() -> URL? {",
            "        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)",
            "        let candidates = [",
            "            cwd.appendingPathComponent(\"TinyWeights.bin\"),",
            "            cwd.appendingPathComponent(\"watchos_example/TinyImageWatchApp/TinyWeights.bin\")",
            "        ]",
            "        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }",
            "    }",
            "}",
            "",
            "private struct TinyWeightReader {",
            "    let data: Data",
            "    var offset = 0",
            "",
            "    mutating func readScaledInt8(count: Int, scale: Float) -> [Float] {",
            "        precondition(offset + count <= data.count, \"TinyWeights.bin is truncated\")",
            "        var output = [Float]()",
            "        output.reserveCapacity(count)",
            "        for index in offset..<(offset + count) {",
            "            output.append(Float(Int8(bitPattern: data[index])) * scale)",
            "        }",
            "        offset += count",
            "        return output",
            "    }",
            "",
            "    mutating func readFloat32(count: Int) -> [Float] {",
            "        var output = [Float]()",
            "        output.reserveCapacity(count)",
            "        for _ in 0..<count {",
            "            output.append(readFloat32())",
            "        }",
            "        return output",
            "    }",
            "",
            "    mutating func readFloat32() -> Float {",
            "        precondition(offset + 4 <= data.count, \"TinyWeights.bin is truncated\")",
            "        var bits: UInt32 = 0",
            "        bits |= UInt32(data[offset])",
            "        bits |= UInt32(data[offset + 1]) << 8",
            "        bits |= UInt32(data[offset + 2]) << 16",
            "        bits |= UInt32(data[offset + 3]) << 24",
            "        offset += 4",
            "        return Float(bitPattern: bits)",
            "    }",
            "",
            "    func assertFinished() {",
            "        precondition(offset == data.count, \"TinyWeights.bin has unexpected trailing data\")",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def export_weights(
    model: TinyCoordinateMLP,
    json_out: Path,
    swift_out: Path,
    bin_out: Path,
    trained_seed_count: int,
    prompt_encoder: str,
) -> None:
    model = model.cpu().eval()
    weight_layers = list(model.layers) + [model.output_layer]
    quantized_weights: list[list[int]] = []
    scales: list[float] = []
    biases: list[list[float]] = []
    for layer in weight_layers:
        weights, scale = quantize_weight(layer.weight)
        quantized_weights.append(weights)
        scales.append(scale)
        biases.append([float(v) for v in layer.bias.detach().cpu().tolist()])
    payload = {
        "latent": model.latent_count,
        "input": model.input_count,
        "coordFrequencies": model.coord_frequencies,
        "trainedSeedCount": trained_seed_count,
        "hidden": model.hidden_count,
        "hiddenLayerCount": model.hidden_layers,
        "output": 3,
        "promptEncoder": prompt_encoder,
    }
    for index, (weights, scale, bias) in enumerate(zip(quantized_weights, scales, biases), start=1):
        payload[f"w{index}"] = weights
        payload[f"w{index}Scale"] = scale
        payload[f"b{index}"] = bias
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n")
    write_binary_weights(payload, bin_out)
    source = swift_weight_source(model, scales, trained_seed_count, prompt_encoder)
    swift_out.write_text(source)


def load_exported_weights(
    model: TinyCoordinateMLP,
    weights_path: Path,
    prompt_encoder: str,
) -> None:
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    expected = {
        "latent": model.latent_count,
        "input": model.input_count,
        "hidden": model.hidden_count,
        "hiddenLayerCount": model.hidden_layers,
        "output": 3,
        "promptEncoder": prompt_encoder,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"init weights are incompatible: {key}={payload.get(key)!r} expected {value!r}"
            )
    payload_frequencies = [float(v) for v in payload.get("coordFrequencies", [])]
    if payload_frequencies != [float(v) for v in model.coord_frequencies]:
        raise SystemExit(
            "init weights are incompatible: "
            f"coordFrequencies={payload_frequencies!r} expected {model.coord_frequencies!r}"
        )

    for index, layer in enumerate(list(model.layers) + [model.output_layer], start=1):
        scale = float(payload[f"w{index}Scale"])
        weight_values = np.asarray(payload[f"w{index}"], dtype=np.float32) * scale
        bias_values = np.asarray(payload[f"b{index}"], dtype=np.float32)
        weight = torch.from_numpy(weight_values.reshape(tuple(layer.weight.shape))).to(
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        bias = torch.from_numpy(bias_values.reshape(tuple(layer.bias.shape))).to(
            device=layer.bias.device,
            dtype=layer.bias.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)


@torch.no_grad()
def render_preview(
    model: TinyCoordinateMLP,
    prompt: str,
    seed: int,
    size: int,
    device: torch.device,
    prompt_encoder: str,
) -> Image.Image:
    latent = torch.tensor(
        [make_latent(prompt, seed, model.latent_count, prompt_encoder)],
        dtype=torch.float32,
        device=device,
    )
    coords = []
    for y in range(size):
        for x in range(size):
            fx = (x / max(size - 1, 1)) * 2.0 - 1.0
            fy = (y / max(size - 1, 1)) * 2.0 - 1.0
            coords.append((fx, fy))
    xy = torch.tensor(coords, dtype=torch.float32, device=device)
    lat = latent.repeat(size * size, 1)
    rgb = model(xy, lat).detach().cpu().numpy().reshape(size, size, 3)
    return Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8), "RGB")


def save_preview_sheet(
    model: TinyCoordinateMLP,
    prompts: list[str],
    out: Path,
    size: int,
    device: torch.device,
    prompt_encoder: str,
) -> None:
    cols = min(8, len(prompts))
    cell = max(64, size * 2)
    label_h = 18
    rows = math.ceil(len(prompts) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, prompt in enumerate(prompts):
        image = render_preview(model, prompt, 0, size, device, prompt_encoder).resize(
            (cell, cell),
            Image.Resampling.NEAREST,
        )
        x = (i % cols) * cell
        y = (i // cols) * (cell + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + cell + 2), prompt[:14], fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def load_preview_prompts(value: str, path: Path | None) -> list[str]:
    if path is None:
        return [p.strip() for p in value.split(",") if p.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    prompts: list[str] = []
    for group in payload.get("groups", []):
        if isinstance(group, dict):
            prompts.extend(str(item).strip() for item in group.get("prompts", []) if str(item).strip())
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--teacher-root-keys",
        action="append",
        default=[],
        help="Optional comma-separated category filter for each --teacher-root. Use '*' for all.",
    )
    parser.add_argument("--teacher-repeat", type=int, default=1)
    parser.add_argument(
        "--teacher-root-repeat",
        action="append",
        type=int,
        default=[],
        help="Optional repeat count for each --teacher-root. Falls back to --teacher-repeat.",
    )
    parser.add_argument(
        "--teacher-root-max-per-key",
        action="append",
        type=int,
        default=[],
        help="Optional per-key sample cap for each --teacher-root. 0 means uncapped.",
    )
    parser.add_argument("--openimages-root", type=Path)
    parser.add_argument("--openimages-limit", type=int, default=0)
    parser.add_argument("--openimages-keys", default="")
    parser.add_argument("--openimages-max-per-key", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--target-downsample-size",
        type=int,
        default=0,
        help="Optionally downsample teacher images to this size and upsample back before training to suppress texture noise.",
    )
    parser.add_argument(
        "--target-blur-radius",
        type=float,
        default=0.0,
        help="Optional Gaussian blur radius applied to teacher images before training.",
    )
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--coord-frequencies", default="1,2,4,8")
    parser.add_argument("--prompt-encoder", choices=PROMPT_ENCODERS, default=PROMPT_ENCODER_HASH)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--smoothness-loss-weight",
        type=float,
        default=0.0,
        help="Optional neighbor-pixel consistency loss to suppress speckled coordinate artifacts.",
    )
    parser.add_argument(
        "--smoothness-step-pixels",
        type=int,
        default=1,
        help="Pixel distance used by --smoothness-loss-weight.",
    )
    parser.add_argument("--seed", type=int, default=260607)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--preview-prompts", default="cat,dog,apple,robot,star,sun,moon,car,tree,flower,house,bird,fish,train,castle,face")
    parser.add_argument("--preview-prompts-file", type=Path)
    parser.add_argument(
        "--init-json",
        type=Path,
        help="Optional exported tiny_weights.json used to initialize the model before training.",
    )
    parser.add_argument("--out-json", type=Path, default=JSON_OUT)
    parser.add_argument("--out-swift", type=Path, default=SWIFT_OUT)
    parser.add_argument("--out-bin", type=Path, default=BIN_OUT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "tiny_train")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    preserve_prompt = args.prompt_encoder == PROMPT_ENCODER_COMPOSITIONAL

    samples: list[Sample] = []
    for index, teacher_root in enumerate(args.teacher_root):
        key_filter = None
        if index < len(args.teacher_root_keys):
            key_filter = parse_key_filter(args.teacher_root_keys[index])
        max_per_key = 0
        if index < len(args.teacher_root_max_per_key):
            max_per_key = args.teacher_root_max_per_key[index]
        teacher_samples = load_teacher_root(
            teacher_root,
            args.image_size,
            teacher_root.name,
            key_filter,
            preserve_prompt=preserve_prompt,
            max_per_key=max_per_key,
        )
        repeat = args.teacher_repeat
        if index < len(args.teacher_root_repeat):
            repeat = args.teacher_root_repeat[index]
        for _ in range(max(1, repeat)):
            samples.extend(teacher_samples)
    if args.openimages_root and args.openimages_limit != 0:
        openimages_keys = None
        if args.openimages_keys.strip():
            openimages_keys = {
                canonicalize_prompt(k)
                for k in args.openimages_keys.split(",")
                if k.strip()
            }
        samples.extend(
            load_openimages_root(
                args.openimages_root,
                args.image_size,
                args.openimages_limit,
                args.seed,
                openimages_keys,
                args.openimages_max_per_key,
                preserve_prompt=preserve_prompt,
            )
        )
    samples = normalize_teacher_seeds(samples)
    if not samples:
        raise SystemExit("no samples loaded")

    key_counts = Counter(s.key for s in samples)
    print(f"loaded samples={len(samples)} prompts={len(key_counts)}", flush=True)
    print("top prompts:", key_counts.most_common(20), flush=True)

    images = load_image_tensor(
        samples,
        args.image_size,
        args.target_downsample_size,
        args.target_blur_radius,
    )
    latents = make_feature_table(samples, args.latent, args.prompt_encoder)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA device was requested but is not available")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS device was requested but is not available")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(
        f"device={device} image_size={args.image_size} "
        f"hidden={args.hidden} hidden_layers={args.hidden_layers} latent={args.latent}",
        flush=True,
    )
    images = images.to(device)
    latents = latents.to(device)
    coord_frequencies = [float(v.strip()) for v in args.coord_frequencies.split(",") if v.strip()]
    model = TinyCoordinateMLP(args.latent, args.hidden, coord_frequencies, args.hidden_layers).to(device)
    if args.init_json:
        load_exported_weights(model, args.init_json, args.prompt_encoder)
        print(f"loaded init weights: {args.init_json}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    n = len(samples)
    size = args.image_size
    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        sample_idx = torch.randint(0, n, (args.batch_size,), device=device)
        px = torch.randint(0, size, (args.batch_size,), device=device)
        py = torch.randint(0, size, (args.batch_size,), device=device)
        fx = (px.float() / max(size - 1, 1)) * 2.0 - 1.0
        fy = (py.float() / max(size - 1, 1)) * 2.0 - 1.0
        xy = torch.stack([fx, fy], dim=1)
        target = images[sample_idx, py, px, :]
        batch_latents = latents[sample_idx]
        pred = model(xy, batch_latents)
        loss = loss_fn(pred, target)
        if args.smoothness_loss_weight > 0:
            direction = torch.randint(0, 4, (args.batch_size,), device=device)
            step_pixels = max(1, args.smoothness_step_pixels)
            nx = px + torch.where(direction == 0, step_pixels, torch.where(direction == 1, -step_pixels, 0))
            ny = py + torch.where(direction == 2, step_pixels, torch.where(direction == 3, -step_pixels, 0))
            nx = torch.clamp(nx, 0, size - 1)
            ny = torch.clamp(ny, 0, size - 1)
            nfx = (nx.float() / max(size - 1, 1)) * 2.0 - 1.0
            nfy = (ny.float() / max(size - 1, 1)) * 2.0 - 1.0
            nxy = torch.stack([nfx, nfy], dim=1)
            neighbor_pred = model(nxy, batch_latents)
            smoothness_loss = torch.mean((pred - neighbor_pred) * (pred - neighbor_pred))
            loss = loss + args.smoothness_loss_weight * smoothness_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.progress_every == 0 or step == args.steps:
            elapsed = time.monotonic() - start_time
            steps_per_second = step / elapsed if elapsed > 0 else 0.0
            print(
                f"step={step} loss={float(loss.detach().cpu()):.6f} "
                f"elapsed={elapsed:.1f}s steps_per_second={steps_per_second:.2f}",
                flush=True,
            )

    prompts = load_preview_prompts(args.preview_prompts, args.preview_prompts_file)
    save_preview_sheet(model, prompts, args.out_dir / "preview_sheet.png", args.image_size, device, args.prompt_encoder)
    # Use the minimum count so app-side random seeds stay valid for every prompt
    # even when combined datasets have different coverage per key.
    trained_seed_count = min(key_counts.values())
    export_weights(model, args.out_json, args.out_swift, args.out_bin, trained_seed_count, args.prompt_encoder)
    manifest = {
        "samples": len(samples),
        "prompt_counts": dict(sorted(key_counts.items())),
        "image_size": args.image_size,
        "target_downsample_size": args.target_downsample_size,
        "target_blur_radius": args.target_blur_radius,
        "latent": args.latent,
        "hidden": args.hidden,
        "hidden_layers": args.hidden_layers,
        "coord_frequencies": coord_frequencies,
        "trained_seed_count": trained_seed_count,
        "prompt_encoder": args.prompt_encoder,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "smoothness_loss_weight": args.smoothness_loss_weight,
        "smoothness_step_pixels": args.smoothness_step_pixels,
        "init_json": str(args.init_json) if args.init_json else None,
        "device": str(device),
        "json_out": str(args.out_json),
        "swift_out": str(args.out_swift),
        "bin_out": str(args.out_bin),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    print(f"wrote {args.out_swift}", flush=True)
    print(f"wrote {args.out_bin}", flush=True)
    print(f"wrote {args.out_dir / 'preview_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
