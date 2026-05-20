#!/usr/bin/env python3
"""Export Hugging Face ImageNet-1K train parquet shards to unMORE's layout.

Hugging Face stores ILSVRC/imagenet-1k as parquet shards with an Image feature.
unMORE expects raw files under:

    datasets/unmore_imagenet/train/<wnid>/<wnid>_<id>.JPEG

The parquet image path uses a flattened form such as:

    n03954731_53652_n03954731.JPEG

This script restores the original ImageNet train path:

    n03954731/n03954731_53652.JPEG
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


PATH_RE = re.compile(r"^(?P<base>.+)_(?P<wnid>n\d{8})\.JPEG$")


def convert_hf_path(path: str) -> Path:
    name = Path(path).name
    match = PATH_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected ImageNet parquet path: {path}")
    base = match.group("base")
    wnid = match.group("wnid")
    return Path(wnid) / f"{base}.JPEG"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_status(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "completed_shards": [],
        "written": 0,
        "skipped": 0,
        "failed": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": None,
    }


def save_status(path: Path, status: dict) -> None:
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True))


def iter_train_shards(repo_id: str) -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=True)
    shards = [f for f in files if f.startswith("data/train-") and f.endswith(".parquet")]
    return sorted(shards)


def export_shard(shard_path: Path, output_root: Path, max_new_images: int | None) -> tuple[int, int, list[str]]:
    table = pq.read_table(shard_path, columns=["image"])
    images = table.column("image").to_pylist()
    written = 0
    skipped = 0
    failed: list[str] = []

    for idx, image in enumerate(images, 1):
        try:
            rel_path = convert_hf_path(image["path"])
            out_path = output_root / rel_path
            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
            else:
                data = image["bytes"]
                if not data:
                    raise ValueError(f"Missing image bytes for {image['path']}")
                atomic_write(out_path, data)
                written += 1
            if max_new_images is not None and written >= max_new_images:
                break
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{image.get('path')}: {exc}")
        if idx % 500 == 0 or idx == len(images):
            print(
                f"  progress {shard_path.name}: {idx}/{len(images)} "
                f"written={written} skipped={skipped} failed={len(failed)}",
                flush=True,
            )

    return written, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--output-root", default="datasets/unmore_imagenet/train")
    parser.add_argument("--tmp-dir", default="datasets/unmore_imagenet/downloads/hf_parquet_tmp")
    parser.add_argument("--status-path", default="datasets/unmore_imagenet/export_hf_imagenet_train_status.json")
    parser.add_argument("--shard-limit", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    tmp_dir = Path(args.tmp_dir)
    status_path = Path(args.status_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    status = load_status(status_path)
    completed = set(status.get("completed_shards", []))
    shards = iter_train_shards(args.repo_id)
    if args.shard_limit is not None:
        shards = shards[: args.shard_limit]

    print(f"Found {len(shards)} train shards", flush=True)
    print(f"Output root: {output_root}", flush=True)

    total_new = 0
    for shard in shards:
        if shard in completed:
            print(f"SKIP completed {shard}", flush=True)
            continue

        print(f"DOWNLOAD {shard}", flush=True)
        local_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=shard,
                local_dir=tmp_dir,
                token=True,
            )
        )

        print(f"EXPORT {local_path}", flush=True)
        remaining = None if args.max_images is None else max(args.max_images - total_new, 0)
        if remaining == 0:
            break
        written, skipped, failed = export_shard(local_path, output_root, remaining)
        total_new += written
        status["written"] = int(status.get("written", 0)) + written
        status["skipped"] = int(status.get("skipped", 0)) + skipped
        status.setdefault("failed", []).extend(failed)

        if args.max_images is None:
            status.setdefault("completed_shards", []).append(shard)

        save_status(status_path, status)
        print(
            f"DONE {shard}: written={written} skipped={skipped} failed={len(failed)} total_new={total_new}",
            flush=True,
        )

        if not args.keep_temp:
            try:
                local_path.unlink()
            except FileNotFoundError:
                pass

        if args.max_images is not None and total_new >= args.max_images:
            break

    if not args.keep_temp:
        data_dir = tmp_dir / "data"
        if data_dir.exists() and not any(data_dir.iterdir()):
            shutil.rmtree(data_dir)

    save_status(status_path, status)
    print("FINISHED", json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
