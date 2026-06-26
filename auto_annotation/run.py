#!/usr/bin/env python3
"""CLI entry point for the auto-annotation pipeline.

Examples:
    # Smoke test with no model weights (exercises merge/QA/IO end-to-end):
    python -m auto_annotation.run --video auto_annotation/data/clip.mp4 \
        --semantic dummy --instance dummy --split train

    # Real run once INSID3 + SAM3 are wired:
    python -m auto_annotation.run --video clips/drive_01.mp4 \
        --semantic insid3 --instance sam3 --split train --device mps
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .pipeline import run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monocular-video panoptic auto-annotation")
    p.add_argument("--video", type=Path, default=None, help="path to input video")
    p.add_argument("--output", type=Path, default=None, help="output dir")
    p.add_argument("--semantic", default="dummy", help="semantic backend key")
    p.add_argument("--instance", default="dummy", help="instance backend key")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=5, help="keyframe stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _set_seed(args.seed)

    base = PipelineConfig(
        semantic_backend=args.semantic,
        instance_backend=args.instance,
        split=args.split,
        keyframe_stride=args.stride,
        max_frames=args.max_frames,
        device=args.device,
        seed=args.seed,
        log_level=args.log_level,
        **({"video_path": args.video} if args.video else {}),
        **({"output_dir": args.output, "frames_dir": args.output / "frames"}
           if args.output else {}),
    )
    cfg = base.resolved(PROJECT_ROOT)
    run(cfg)


if __name__ == "__main__":
    main()
