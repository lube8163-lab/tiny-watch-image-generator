#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from prompt_normalization import (
    PROMPT_ENCODER_COMPOSITIONAL,
    PROMPT_ENCODERS,
    make_prompt_latent,
)
from train_tiny_coordinate_mlp import (
    ROOT,
    Sample,
    load_preview_prompts,
    load_teacher_root,
    normalize_teacher_seeds,
    parse_key_filter,
)


DEFAULT_OUT_DIR = ROOT / "out" / "tiny_cnn_decoder"


def parse_channels(value: str) -> list[int]:
    channels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(channels) < 2:
        raise argparse.ArgumentTypeError("expected at least two channel counts")
    if any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError("channels must be positive")
    return channels


def load_target_tensor(
    samples: list[Sample],
    size: int,
    target_downsample_size: int,
    target_blur_radius: float,
    posterize_bits: int,
    flatten_background: bool,
    foreground_threshold: float,
    mask_blur_radius: float,
) -> torch.Tensor:
    images = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        if flatten_background:
            image = flatten_to_border_background(
                image,
                foreground_threshold=foreground_threshold,
                mask_blur_radius=mask_blur_radius,
            )
        if 0 < target_downsample_size < size:
            image = image.resize(
                (target_downsample_size, target_downsample_size),
                Image.Resampling.LANCZOS,
            ).resize((size, size), Image.Resampling.BICUBIC)
        if target_blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=target_blur_radius))
        if 1 <= posterize_bits < 8:
            image = ImageOps.posterize(image, posterize_bits)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images.append(arr.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(images, axis=0))


def flatten_to_border_background(
    image: Image.Image,
    foreground_threshold: float,
    mask_blur_radius: float,
) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32)
    height, width, _ = pixels.shape
    margin = max(2, int(min(width, height) * 0.12))
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
    distance = np.linalg.norm(pixels - background.reshape(1, 1, 3), axis=2)
    mask = np.clip((distance - foreground_threshold) / max(foreground_threshold, 1.0), 0.0, 1.0)
    mask_image = Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L")
    if mask_blur_radius > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))
    flat = Image.new("RGB", image.size, tuple(int(round(v)) for v in background))
    return Image.composite(image, flat, mask_image)


def make_latent_tensor(samples: list[Sample], latent_count: int, prompt_encoder: str) -> torch.Tensor:
    latents = [make_prompt_latent(sample.prompt, sample.seed, latent_count, prompt_encoder) for sample in samples]
    return torch.tensor(latents, dtype=torch.float32)


class TinyCNNDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_count: int,
        image_size: int,
        base_size: int,
        channels: list[int],
        upsample_mode: str,
    ) -> None:
        super().__init__()
        if image_size % base_size != 0:
            raise ValueError("image_size must be divisible by base_size")
        upsample_count = int(math.log2(image_size // base_size))
        if 2**upsample_count * base_size != image_size:
            raise ValueError("image_size / base_size must be a power of two")
        if len(channels) != upsample_count + 1:
            raise ValueError(
                f"channels must contain base channel plus {upsample_count} upsample channels "
                f"for image_size={image_size} base_size={base_size}"
            )
        self.latent_count = latent_count
        self.image_size = image_size
        self.base_size = base_size
        self.channels = channels
        self.upsample_mode = upsample_mode
        self.fc = torch.nn.Linear(latent_count, channels[0] * base_size * base_size)
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        )
        self.out = torch.nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent).clamp(-3.0, 3.0)
        x = torch.tanh(x).view(latent.shape[0], self.channels[0], self.base_size, self.base_size)
        for conv in self.convs:
            if self.upsample_mode == "bilinear":
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.tanh(conv(x).clamp(-3.0, 3.0))
        return torch.sigmoid(self.out(x) * 1.8)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return dx + dy


@torch.no_grad()
def render_preview(
    model: TinyCNNDecoder,
    prompt: str,
    seed: int,
    device: torch.device,
    prompt_encoder: str,
) -> Image.Image:
    latent = torch.tensor(
        [make_prompt_latent(prompt, seed, model.latent_count, prompt_encoder)],
        dtype=torch.float32,
        device=device,
    )
    rgb = model(latent).detach().cpu().numpy()[0].transpose(1, 2, 0)
    return Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8), "RGB")


def save_preview_sheet(
    model: TinyCNNDecoder,
    prompts: list[str],
    out: Path,
    device: torch.device,
    prompt_encoder: str,
) -> None:
    cols = min(8, len(prompts))
    cell = max(64, model.image_size * 2)
    label_h = 18
    rows = math.ceil(len(prompts) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, prompt in enumerate(prompts):
        image = render_preview(model, prompt, 0, device, prompt_encoder).resize(
            (cell, cell),
            Image.Resampling.NEAREST,
        )
        x = (i % cols) * cell
        y = (i // cols) * (cell + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + cell + 2), prompt[:14], fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def select_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA device was requested but is not available")
        return torch.device("cuda")
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS device was requested but is not available")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_state(model: TinyCNNDecoder, out: Path, manifest: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest": manifest,
            "state_dict": model.cpu().eval().state_dict(),
        },
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny latent-to-image CNN decoder prototype.")
    parser.add_argument("--teacher-root", action="append", type=Path, default=[])
    parser.add_argument("--teacher-root-keys", action="append", default=[])
    parser.add_argument("--teacher-root-max-per-key", action="append", type=int, default=[])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--target-downsample-size", type=int, default=64)
    parser.add_argument("--target-blur-radius", type=float, default=0.25)
    parser.add_argument("--posterize-bits", type=int, default=0)
    parser.add_argument("--flatten-background", action="store_true")
    parser.add_argument("--foreground-threshold", type=float, default=32.0)
    parser.add_argument("--mask-blur-radius", type=float, default=2.0)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--base-size", type=int, default=8)
    parser.add_argument("--channels", type=parse_channels, default=parse_channels("64,48,32,24,16"))
    parser.add_argument("--upsample-mode", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--prompt-encoder", choices=PROMPT_ENCODERS, default=PROMPT_ENCODER_COMPOSITIONAL)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--tv-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=260612)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--preview-prompts", default="cat,dog,apple,robot,star,sun,moon,car,tree,flower,house,bird,fish,train,castle,face")
    parser.add_argument("--preview-prompts-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
        samples.extend(
            load_teacher_root(
                teacher_root,
                args.image_size,
                teacher_root.name,
                key_filter,
                preserve_prompt=preserve_prompt,
                max_per_key=max_per_key,
            )
        )
    samples = normalize_teacher_seeds(samples)
    if not samples:
        raise SystemExit("no samples loaded")

    key_counts = Counter(sample.key for sample in samples)
    print(f"loaded samples={len(samples)} prompts={len(key_counts)}", flush=True)
    print("top prompts:", key_counts.most_common(20), flush=True)

    images = load_target_tensor(
        samples,
        args.image_size,
        args.target_downsample_size,
        args.target_blur_radius,
        args.posterize_bits,
        args.flatten_background,
        args.foreground_threshold,
        args.mask_blur_radius,
    )
    latents = make_latent_tensor(samples, args.latent, args.prompt_encoder)
    device = select_device(args.device)
    print(
        f"device={device} image_size={args.image_size} latent={args.latent} "
        f"base_size={args.base_size} channels={args.channels} upsample_mode={args.upsample_mode}",
        flush=True,
    )
    images = images.to(device)
    latents = latents.to(device)
    model = TinyCNNDecoder(args.latent, args.image_size, args.base_size, args.channels, args.upsample_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    n = len(samples)
    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        sample_idx = torch.randint(0, n, (args.batch_size,), device=device)
        pred = model(latents[sample_idx])
        loss = loss_fn(pred, images[sample_idx])
        if args.tv_loss_weight > 0:
            loss = loss + args.tv_loss_weight * total_variation_loss(pred)
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_preview_prompts(args.preview_prompts, args.preview_prompts_file)
    save_preview_sheet(model, prompts, args.out_dir / "preview_sheet.png", device, args.prompt_encoder)
    trained_seed_count = min(key_counts.values())
    manifest = {
        "architecture": "tiny_cnn_decoder_v1",
        "samples": len(samples),
        "prompt_counts": dict(sorted(key_counts.items())),
        "image_size": args.image_size,
        "target_downsample_size": args.target_downsample_size,
        "target_blur_radius": args.target_blur_radius,
        "posterize_bits": args.posterize_bits,
        "flatten_background": args.flatten_background,
        "foreground_threshold": args.foreground_threshold,
        "mask_blur_radius": args.mask_blur_radius,
        "latent": args.latent,
        "base_size": args.base_size,
        "channels": args.channels,
        "upsample_mode": args.upsample_mode,
        "trained_seed_count": trained_seed_count,
        "prompt_encoder": args.prompt_encoder,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tv_loss_weight": args.tv_loss_weight,
        "device": str(device),
    }
    (args.out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    export_state(model, args.out_dir / "tiny_cnn_decoder.pt", manifest)
    print(f"wrote {args.out_dir / 'tiny_cnn_decoder.pt'}", flush=True)
    print(f"wrote {args.out_dir / 'preview_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
