#!/usr/bin/env python3
from __future__ import annotations

import argparse

from research_common import directory_size, select_candidate, snapshot_download_candidate, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: download only the files needed for a candidate pipeline.")
    parser.add_argument("--candidate", default=None)
    args = parser.parse_args()

    key, candidate = select_candidate(args.candidate)
    path = snapshot_download_candidate(key, candidate)
    manifest = path / "download_manifest.json"
    write_manifest(
        manifest,
        {
            "phase": "phase0_download",
            "candidate": key,
            "source_repo": candidate["repo"],
            "local_path": str(path),
            "bytes": directory_size(path),
            "allow_patterns": candidate.get("download_allow_patterns"),
            "ignore_patterns": candidate.get("download_ignore_patterns"),
        },
    )
    print(path)
    print(manifest)


if __name__ == "__main__":
    main()
