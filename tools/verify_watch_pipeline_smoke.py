#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research_common import ROOT, directory_size


EXPECTED_UNET_COUNT = 16
EXPECTED_DEFAULT_KEY = "cat_mascot"
EXPECTED_MIN_PROMPT_COUNT = 31
EXPECTED_EMBEDDING_TRAILING_SHAPE = [77, 768]
EXPECTED_TEXT_ENCODER = "TextEncoderAssets/clip_text_encoder_77.mlmodelc"
EXPECTED_TOKENIZER_FILES = [
    "TextEncoderAssets/clip_vocab.json",
    "TextEncoderAssets/clip_merges.txt",
]
EXPECTED_REFERENCE = ROOT / "reports" / "watch_pipeline_reference" / "final_default_cat_mascot_s1_g6_coreml_16p" / "coreml.json"
FAMILY_CONFIGS = {
    "lcm128": {
        "unet_prefix": "lcm_unet_16x16_6bit_16p_part",
        "decoder": "vae_decoder_128x128_noattn_4bit.mlmodelc",
        "latent_shape": [1, 4, 16, 16],
        "decoded_shape": [1, 3, 128, 128],
        "scheduler_dir": "LCMAssets",
        "forbidden_patterns": ["*8x8*.mlmodelc", "*16x16_4bit*.mlmodelc", "*24x24*.mlmodelc"],
        "run_id": "cat_mascot-s1-g6-sharp2x",
    },
    "lcm192": {
        "unet_prefix": "lcm_unet_24x24_6bit_16p_part",
        "decoder": "vae_decoder_192x192_noattn_4bit.mlmodelc",
        "latent_shape": [1, 4, 24, 24],
        "decoded_shape": [1, 3, 192, 192],
        "scheduler_dir": "LCM192Assets",
        "forbidden_patterns": ["*8x8*.mlmodelc", "*16x16*.mlmodelc"],
        "run_id": "lcm192-smoke",
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        fail(f"missing JSON file: {path}")
        raise AssertionError from exc


def check_models(app: Path, config: dict) -> list[str]:
    unets = sorted((path.name for path in app.glob("lcm_unet_*.mlmodelc")), key=unet_sort_key)
    expected = [f"{config['unet_prefix']}{index}.mlmodelc" for index in range(1, EXPECTED_UNET_COUNT + 1)]
    if unets != expected:
        fail(f"unexpected UNet bundle entries:\n  got={unets}\n  expected={expected}")

    decoder_path = app / config["decoder"]
    if not decoder_path.is_dir():
        fail(f"missing decoder: {decoder_path}")

    forbidden = [
        path.name
        for pattern in config["forbidden_patterns"]
        for path in app.glob(pattern)
    ]
    if forbidden:
        fail(f"unexpected old model entries: {sorted(forbidden)}")
    return unets


def unet_sort_key(name: str) -> int:
    match = re.search(r"_part(\d+)\.mlmodelc$", name)
    return int(match.group(1)) if match else 10_000


def check_lcm_assets(app: Path, config: dict) -> None:
    lcm_dir = app / "LCMAssets"
    if not lcm_dir.is_dir():
        fail(f"missing LCMAssets directory: {lcm_dir}")
    scheduler_dir = app / config["scheduler_dir"]
    if not scheduler_dir.is_dir():
        fail(f"missing scheduler assets directory: {scheduler_dir}")

    preset_file = load_json(lcm_dir / "prompt_presets.json")
    scheduler = load_json(scheduler_dir / "lcm_scheduler.json")
    presets = preset_file.get("presets", [])
    embedding_shape = preset_file.get("embeddingShape")
    if len(presets) < EXPECTED_MIN_PROMPT_COUNT:
        fail(f"unexpected LCM prompt count: got {len(presets)}, expected at least {EXPECTED_MIN_PROMPT_COUNT}")
    if embedding_shape != [len(presets), *EXPECTED_EMBEDDING_TRAILING_SHAPE]:
        fail(
            "unexpected LCM embedding shape: "
            f"got {embedding_shape}, expected {[len(presets), *EXPECTED_EMBEDDING_TRAILING_SHAPE]}"
        )
    if not any(item.get("key") == EXPECTED_DEFAULT_KEY for item in presets):
        fail(f"default prompt key not found: {EXPECTED_DEFAULT_KEY}")
    if scheduler.get("latentShape") != config["latent_shape"]:
        fail(f"unexpected LCM latent shape: got {scheduler.get('latentShape')}, expected {config['latent_shape']}")
    if scheduler.get("decodedShape") != config["decoded_shape"]:
        fail(f"unexpected LCM decoded shape: got {scheduler.get('decodedShape')}, expected {config['decoded_shape']}")

    embedding_bytes = (lcm_dir / "prompt_embeddings_f16.bin").stat().st_size
    expected_embedding_bytes = 2
    for dimension in embedding_shape:
        expected_embedding_bytes *= dimension
    if embedding_bytes != expected_embedding_bytes:
        fail(f"unexpected LCM embedding bytes: got {embedding_bytes}, expected {expected_embedding_bytes}")

    timestep_shape = scheduler.get("timestepCondShape")
    if not isinstance(timestep_shape, list):
        fail(f"unexpected timestep cond shape: {timestep_shape}")
    timestep_bytes = (lcm_dir / "timestep_cond_f16.bin").stat().st_size
    expected_timestep_bytes = 2
    for dimension in timestep_shape:
        expected_timestep_bytes *= dimension
    if timestep_bytes != expected_timestep_bytes:
        fail(f"unexpected timestep cond bytes: got {timestep_bytes}, expected {expected_timestep_bytes}")


def check_text_encoder_assets(app: Path) -> None:
    text_encoder = app / EXPECTED_TEXT_ENCODER
    if not text_encoder.is_dir():
        fail(f"missing text encoder: {text_encoder}")
    for relative_path in EXPECTED_TOKENIZER_FILES:
        path = app / relative_path
        if not path.is_file():
            fail(f"missing tokenizer asset: {path}")
        if path.stat().st_size <= 0:
            fail(f"empty tokenizer asset: {path}")


def check_reference(path: Path) -> None:
    reference = load_json(path)
    expected = {
        "prompt_key": EXPECTED_DEFAULT_KEY,
        "seed": 1,
        "guidance_scale": 6.0,
        "steps": 4,
        "latent_shape": [1, 4, 16, 16],
        "decoded_shape": [1, 3, 128, 128],
        "clipped_channels": 4,
        "total_channels": 49152,
        "preview_mode": "sharp2x",
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            fail(f"reference {key} mismatch: got {reference.get(key)!r}, expected {value!r}")

    final_rms = float(reference.get("final_stats", {}).get("rms", -1))
    decoded_rms = float(reference.get("decoded_stats", {}).get("rms", -1))
    if abs(final_rms - 0.7768450379) > 0.0001:
        fail(f"reference final RMS mismatch: {final_rms}")
    if abs(decoded_rms - 0.3998658061) > 0.0001:
        fail(f"reference decoded RMS mismatch: {decoded_rms}")

    image_path = ROOT / reference.get("image", "")
    if not image_path.is_file():
        fail(f"reference image missing: {image_path}")
    preview_path = ROOT / reference.get("preview_image", "")
    if not preview_path.is_file():
        fail(f"reference preview image missing: {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify WatchPipelineSmokeApp final smoke bundle.")
    parser.add_argument("--app", required=True, help="Path to built WatchPipelineSmokeApp.app")
    parser.add_argument("--family", choices=sorted(FAMILY_CONFIGS), default="lcm128", help="Expected bundled LCM family")
    parser.add_argument("--reference", default=str(EXPECTED_REFERENCE), help="Expected Core ML reference JSON")
    parser.add_argument("--skip-reference", action="store_true", help="Skip the 128px golden reference check")
    args = parser.parse_args()

    app = Path(args.app)
    if not app.is_dir():
        fail(f"app bundle not found: {app}")

    config = FAMILY_CONFIGS[args.family]
    unets = check_models(app, config)
    check_lcm_assets(app, config)
    check_text_encoder_assets(app)
    checked_reference = False
    if not args.skip_reference and args.family == "lcm128":
        check_reference(Path(args.reference))
        checked_reference = True
    print("watch-pipeline-smoke: ok")
    print(f"  app: {app}")
    print(f"  family: {args.family}")
    print(f"  size: {directory_size(app) / 1024 / 1024:.1f}MB")
    print(f"  unet_chunks: {len(unets)}")
    print(f"  decoder: {config['decoder']}")
    print(f"  latent_shape: {'x'.join(map(str, config['latent_shape']))}")
    print(f"  decoded_shape: {'x'.join(map(str, config['decoded_shape']))}")
    print(f"  text_encoder: {EXPECTED_TEXT_ENCODER}")
    print(f"  reference_checked: {str(checked_reference).lower()}")
    print(f"  run_id: {config['run_id']}")


if __name__ == "__main__":
    main()
