#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://raw.githubusercontent.com/CZhou-24/TMD-CUDA-main/main/"
    "TMD-Fits-Minimal/Fits/Bayes_Data"
)

KNOWN_FILES = {
    "emcee_gated_global_6d.h5",
    "emcee_gated_global_9d.h5",
    "emcee_gated_local_6d_uc0.020_b0.1.h5",
    "emcee_gated_local_smoke_6d_uc0.020_b0.1.h5",
    "emcee_truth_global_9d.h5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download source Bayes_Data HDF5 backends from the reference repo "
            "into Bayesian Analysis/source_data/."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific Bayes_Data filenames to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every known Bayes_Data backend.",
    )
    parser.add_argument(
        "--dest-dir",
        default=str(Path(__file__).resolve().parent / "source_data"),
        help="Destination directory. Default: Bayesian Analysis/source_data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known Bayes_Data filenames and exit.",
    )
    return parser.parse_args()


def download_file(name: str, dest_dir: Path, overwrite: bool) -> Path:
    if name not in KNOWN_FILES:
        raise ValueError(
            f"Unknown Bayes_Data file {name!r}. Use --list to see supported names."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and not overwrite:
        print(f"Skipping existing file: {dest}")
        return dest

    url = f"{BASE_URL}/{name}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest}")
    return dest


def main() -> int:
    args = parse_args()

    if args.list:
        for name in sorted(KNOWN_FILES):
            print(name)
        return 0

    selected = sorted(KNOWN_FILES) if args.all else args.files
    if not selected:
        print(
            "No files selected. Pass one or more filenames or use --all.",
            file=sys.stderr,
        )
        return 2

    dest_dir = Path(args.dest_dir).resolve()
    try:
        for name in selected:
            download_file(name, dest_dir, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
