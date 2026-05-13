#!/usr/bin/env python3
"""Compute per-image instance count statistics for a CUPS pseudo-label directory.

Used as a proxy for over-fragmentation when comparing SIMCF step variants.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-image instance count stats")
    parser.add_argument("--variant", type=str, required=True,
                        help="Variant label (printed only)")
    parser.add_argument("--dir", type=str, required=True,
                        help="CUPS flat label directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    variant = args.variant
    label_dir = Path(args.dir).expanduser()
    inst_paths = sorted(label_dir.glob("*_instance.png"))
    if not inst_paths:
        logger.warning("No instance PNGs in %s", label_dir)
        return

    counts = []
    sizes_per_img = []
    for p in tqdm(inst_paths, desc=f"inst-stats {variant}"):
        inst = np.array(Image.open(p))
        ids, sizes = np.unique(inst, return_counts=True)
        # Drop background 0
        valid = ids != 0
        n_inst = int(valid.sum())
        counts.append(n_inst)
        if n_inst > 0:
            sizes_per_img.append(int(np.median(sizes[valid])))

    counts_arr = np.asarray(counts)
    sizes_arr = np.asarray(sizes_per_img) if sizes_per_img else np.asarray([0])

    result = {
        "variant": variant,
        "n_images": int(len(counts_arr)),
        "instances_per_image": {
            "mean": float(counts_arr.mean()),
            "median": float(np.median(counts_arr)),
            "std": float(counts_arr.std()),
            "p10": float(np.percentile(counts_arr, 10)),
            "p90": float(np.percentile(counts_arr, 90)),
            "min": int(counts_arr.min()),
            "max": int(counts_arr.max()),
        },
        "median_instance_size_per_image": {
            "mean": float(sizes_arr.mean()),
            "median": float(np.median(sizes_arr)),
        },
    }
    print("=" * 72)
    print(f"INSTANCE STATS [{variant}]")
    print(f"  N images:                    {result['n_images']}")
    print(f"  Instances/image  mean:       {result['instances_per_image']['mean']:.2f}")
    print(f"  Instances/image  median:     {result['instances_per_image']['median']:.0f}")
    print(f"  Instances/image  std:        {result['instances_per_image']['std']:.2f}")
    print(f"  Instances/image  [p10, p90]: [{result['instances_per_image']['p10']:.0f}, "
          f"{result['instances_per_image']['p90']:.0f}]")
    print(f"  Median inst size mean (px):  {result['median_instance_size_per_image']['mean']:.0f}")
    print("=" * 72)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
