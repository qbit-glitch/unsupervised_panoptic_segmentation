#!/usr/bin/env python3
"""Create unMORE COCO20K filename symlinks from a local COCO train2017 folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="datasets/coco/annotation/coco20k_trainval_gt.json")
    parser.add_argument("--source-root", default="/Users/qbit-glitch/Desktop/datasets/coco/train2017")
    parser.add_argument("--dest-root", default="datasets/COCO/train2014")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)

    data = json.loads(annotation_path.read_text())
    images = data["images"]

    missing = []
    created = existing = replaced = 0
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    for image in images:
        official_name = image["file_name"]
        numeric_name = (
            official_name.replace("COCO_train2014_", "")
            .replace("COCO_val2014_", "")
        )
        source = source_root / numeric_name
        dest = dest_root / official_name
        if not source.exists():
            missing.append((official_name, str(source)))
            continue

        if args.dry_run:
            continue

        if dest.is_symlink():
            if dest.resolve() == source.resolve():
                existing += 1
                continue
            dest.unlink()
            replaced += 1
        elif dest.exists():
            existing += 1
            continue

        dest.symlink_to(source)
        created += 1

    print(f"annotation_images={len(images)}")
    print(f"source_root={source_root}")
    print(f"dest_root={dest_root}")
    print(f"missing={len(missing)}")
    print(f"created={created}")
    print(f"existing={existing}")
    print(f"replaced={replaced}")
    if missing[:10]:
        print("missing_sample=" + json.dumps(missing[:10], indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
