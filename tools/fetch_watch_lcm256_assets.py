#!/usr/bin/env python3
"""Download and install the external LCM256 Core ML artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "lube8163/tiny-watch-image-generator-lcm256-coreml"
DEFAULT_VERSION = "v0.1.0"
WATCH_APP_DIR = ROOT / "watchos_example" / "WatchPipelineSmokeApp"


def hub_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest
    return checksums


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"reuse {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"download {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_mb = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                current_mb = done // (32 * 1024 * 1024)
                if current_mb != last_mb:
                    print(f"  {done / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MiB")
                    last_mb = current_mb
    tmp.replace(destination)


def verify(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")
    print(f"verified {path.name}")


def copy_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    def ignore_appledouble(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.startswith("._") or name == "__MACOSX"}

    for item in sorted(src.iterdir()):
        if item.name.startswith("._") or item.name == "__MACOSX":
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, symlinks=True, ignore=ignore_appledouble)
        else:
            shutil.copy2(item, target)


def install_watch_assets(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="watch-lcm256-assets-") as temp:
        temp_dir = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)

        base = temp_dir / "watchos_app_assets" / "WatchPipelineSmokeApp"
        if not base.exists():
            raise SystemExit(f"unexpected watch asset zip layout: {zip_path}")

        copy_contents(base / "Models", WATCH_APP_DIR / "Models")
        copy_contents(base / "TextEncoderAssets", WATCH_APP_DIR / "TextEncoderAssets")
        copy_contents(base / "LCM256Assets", WATCH_APP_DIR / "LCM256Assets")
    print("installed WatchPipelineSmokeApp LCM256 assets")


def install_coreml_packages(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="watch-lcm256-packages-") as temp:
        temp_dir = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)

        base = temp_dir / "coreml_packages"
        if not base.exists():
            raise SystemExit(f"unexpected Core ML package zip layout: {zip_path}")

        copy_contents(base, ROOT / "dist" / "lcm_dreamshaper_v7")
    print("installed Mac-side Core ML packages")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "out" / "hf_downloads")
    parser.add_argument("--packages", action="store_true", help="Also install .mlpackage files for Mac quality eval.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()

    release_dir = f"watch_lcm256_{args.version}"
    cache = args.cache_dir / args.repo_id.replace("/", "__") / args.version
    cache.mkdir(parents=True, exist_ok=True)

    files = ["manifest.json", "sha256sums.txt", "watchos_app_assets.zip"]
    if args.packages:
        files.append("coreml_packages.zip")

    for name in files:
        download(hub_url(args.repo_id, f"{release_dir}/{name}"), cache / name, force=args.force)

    checksums = {} if args.skip_checksum else load_checksums(cache / "sha256sums.txt")
    if not args.skip_checksum:
        for name in files:
            if name != "sha256sums.txt":
                verify(cache / name, checksums.get(name))

    manifest = json.loads((cache / "manifest.json").read_text())
    print(f"asset version: {manifest.get('version')} from {manifest.get('source_commit')}")

    install_watch_assets(cache / "watchos_app_assets.zip")
    if args.packages:
        install_coreml_packages(cache / "coreml_packages.zip")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
