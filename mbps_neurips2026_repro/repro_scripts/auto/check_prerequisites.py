#!/usr/bin/env python3
"""Check prerequisites before running unattended experiments."""

import argparse
import shutil
import sys
from pathlib import Path


def check_disk_space(path: Path, min_gb: float = 50.0) -> bool:
    """Check available disk space."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"ERROR: Only {free_gb:.1f}GB free at {path}. Need {min_gb}GB.")
        return False
    print(f"OK: {free_gb:.1f}GB free at {path}")
    return True


def check_dataset(root: Path) -> bool:
    """Verify Cityscapes dataset structure."""
    required = [
        root / "leftImg8bit" / "train",
        root / "leftImg8bit" / "val",
        root / "gtFine" / "val",
    ]
    ok = True
    for p in required:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            ok = False
    if ok:
        train_imgs = list((root / "leftImg8bit" / "train").rglob("*.png"))
        print(f"OK: Cityscapes found. Train images: {len(train_imgs)}")
    return ok


def check_checkpoints(project_root: Path) -> bool:
    """Verify required model checkpoints exist."""
    ok = True
    cause_dir = project_root / "unsupervised-panoptic-segmentation" / "refs" / "cause"
    if not cause_dir.exists():
        print(f"WARNING: CAUSE-TR directory not found at {cause_dir}")
    depthg_ckpt = project_root / "weights" / "depthg.ckpt"
    if depthg_ckpt.exists():
        print(f"OK: DepthG checkpoint found: {depthg_ckpt}")
    else:
        print(f"WARNING: DepthG checkpoint not found: {depthg_ckpt}")
    dcfa_ckpt = project_root / "results" / "depth_adapter" / "best.pt"
    if dcfa_ckpt.exists():
        print(f"OK: DCFA checkpoint found: {dcfa_ckpt}")
    else:
        print(f"WARNING: DCFA checkpoint not found: {dcfa_ckpt}")
    return ok


def check_python_deps() -> bool:
    """Verify required Python packages."""
    deps = ["torch", "numpy", "PIL", "torchvision", "tqdm"]
    ok = True
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            print(f"ERROR: Missing Python dependency: {dep}")
            ok = False
    if ok:
        print("OK: All Python dependencies available")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Check experiment prerequisites")
    parser.add_argument("--cityscapes_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_disk_gb", type=float, default=50.0)
    args = parser.parse_args()

    print("=" * 60)
    print("PREREQUISITE CHECK")
    print("=" * 60)

    checks = [
        ("Disk space", check_disk_space(args.output_dir, args.min_disk_gb)),
        ("Cityscapes dataset", check_dataset(args.cityscapes_root)),
        ("Checkpoints", check_checkpoints(args.project_root)),
        ("Python dependencies", check_python_deps()),
    ]

    print("=" * 60)
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("ALL PREREQUISITES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
