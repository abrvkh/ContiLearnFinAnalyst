#!/usr/bin/env python3
"""Download JSONL files from the `large` split of financial-reports-sec.

Source:
  https://huggingface.co/datasets/JanosAudran/financial-reports-sec/tree/main/data

Example:
  uv run python src/data/build_10k_hf.py \
    --dataset JanosAudran/financial-reports-sec \
    --revision main \
    --out-dir data/financial-reports-sec
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all .jsonl files from data/large/{train,validate,test} in a HF dataset repo."
    )
    parser.add_argument("--dataset", type=str, default="JanosAudran/financial-reports-sec", help="HF dataset id")
    parser.add_argument("--revision", type=str, default="main", help="Branch/tag/commit")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/financial-reports-sec"),
        help="Destination directory for downloaded files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore

    repo_files = list_repo_files(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
    )

    wanted_prefixes = (
        "data/large/train/",
        "data/large/validate/",
        "data/large/test/",
    )
    jsonl_files = sorted(
        f
        for f in repo_files
        if f.lower().endswith(".jsonl") and any(f.startswith(prefix) for prefix in wanted_prefixes)
    )

    if not jsonl_files:
        raise RuntimeError(
            "No .jsonl files found under data/large/{train,val,test} "
            f"in dataset repo: {args.dataset}@{args.revision}"
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for rel_path in jsonl_files:
        cached_path = hf_hub_download(
            repo_id=args.dataset,
            repo_type="dataset",
            filename=rel_path,
            revision=args.revision,
        )

        # Preserve split structure as large/{train,validate,test}/... under out-dir.
        large_rel_path = Path(rel_path).relative_to("data")
        target_path = out_dir / large_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target_path)
        downloaded += 1

    print(f"Dataset: {args.dataset}@{args.revision}")
    print(f"Downloaded jsonl files from data/large: {downloaded}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
