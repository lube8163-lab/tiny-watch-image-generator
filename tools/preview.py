#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from prompt_normalization import PROMPT_ENCODER_HASH, make_prompt_latent

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "tiny_weights.json"


def dense(inputs: list[float], weights: list[int], scale: float, bias: list[float]) -> list[float]:
    out = []
    width = len(inputs)
    for o, b in enumerate(bias):
        row = o * width
        s = b
        for i, x in enumerate(inputs):
            s += weights[row + i] * scale * x
        out.append(s)
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate(prompt: str, seed: int, size: int, weights: dict) -> Image.Image:
    prompt_encoder = weights.get("promptEncoder") or PROMPT_ENCODER_HASH
    latent = np.asarray(make_prompt_latent(prompt, seed, weights["latent"], prompt_encoder), dtype=np.float32)
    coord_frequencies = weights.get("coordFrequencies")

    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    fx, fy = np.meshgrid(axis, axis)
    flat_x = fx.reshape(-1, 1)
    flat_y = fy.reshape(-1, 1)
    radius = np.minimum(np.sqrt(flat_x * flat_x + flat_y * flat_y), 1.0)
    features = [flat_x, flat_y, radius, np.ones_like(flat_x)]
    if coord_frequencies:
        for frequency in coord_frequencies:
            phase_x = latent[0] * (0.35 * float(frequency))
            phase_y = latent[1] * (0.35 * float(frequency))
            scaled_x = flat_x * (math.pi * float(frequency))
            scaled_y = flat_y * (math.pi * float(frequency))
            features.extend(
                [
                    np.sin(scaled_x + phase_x),
                    np.cos(scaled_x + phase_x),
                    np.sin(scaled_y + phase_y),
                    np.cos(scaled_y + phase_y),
                ]
            )
    else:
        features = [
            flat_x,
            flat_y,
            radius,
            np.sin(flat_x * 6.0 + latent[0] * 3.0),
            np.cos(flat_y * 6.0 + latent[1] * 3.0),
            np.ones_like(flat_x),
        ]

    latent_features = np.repeat(latent.reshape(1, -1), size * size, axis=0)
    inputs = np.concatenate(features + [latent_features], axis=1).astype(np.float32)

    hidden_layers = int(weights.get("hiddenLayerCount", 2))
    x = inputs
    input_width = inputs.shape[1]
    for index in range(1, hidden_layers + 1):
        bias = np.asarray(weights[f"b{index}"], dtype=np.float32)
        layer = (
            np.asarray(weights[f"w{index}"], dtype=np.float32).reshape(len(bias), input_width)
            * float(weights[f"w{index}Scale"])
        )
        x = np.tanh(np.clip(x @ layer.T + bias, -3.0, 3.0))
        input_width = len(bias)

    output_index = hidden_layers + 1
    output_bias = np.asarray(weights[f"b{output_index}"], dtype=np.float32)
    output_layer = (
        np.asarray(weights[f"w{output_index}"], dtype=np.float32).reshape(len(output_bias), input_width)
        * float(weights[f"w{output_index}Scale"])
    )
    logits = (x @ output_layer.T + output_bias) * 1.8
    rgb = 1.0 / (1.0 + np.exp(-logits))
    arr = np.clip(np.rint(rgb.reshape(size, size, 3) * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--out", default="out/preview.png")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()

    weights_path = args.weights if args.weights.is_absolute() else ROOT / args.weights
    weights = json.loads(weights_path.read_text())
    image = generate(args.prompt, args.seed, args.size, weights)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
