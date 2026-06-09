"""Semantic-only CUPS pseudo-label generation using our retrained DepthG-DepthPro-monocular ckpt.

For each Cityscapes train image:
  1. Load RGB at the DepthG inference resolution (640 × 1280).
  2. Load matching DepthPro depth (.npy) and renormalize to a CUPS-style depth_weight.
  3. Forward through DepthG.depth_guided_sliding_window — same code path the upstream
     gen_pseudo_labels.py uses.
  4. CRF-refine with cups.crf.batched_crf.
  5. argmax → save as <out>/<city>/<frame>_leftImg8bit_semantic.png (uint8).

No optical flow, no stereo, no SF2SE3 — purely the semantic branch. Instance maps come from
the existing /Volumes/code_files/.../cups_pseudo_labels_dcfa_simcf_v3depthpro cache (paired
later by mbps_pytorch/assemble_depthg_depthpro_pseudolabels.py).

Run:
  PYTHONPATH=refs/cups/external/depthg \
    .venv_cups_cpu/bin/python mbps_pytorch/gen_semantic_only_depthg_depthpro.py \
      --cities aachen \
      --limit 5

  PYTHONPATH=refs/cups/external/depthg \
    nohup .venv_cups_cpu/bin/python -u mbps_pytorch/gen_semantic_only_depthg_depthpro.py \
      > logs/gen_semantic_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg" / "src"))

from cups.crf import batched_crf  # noqa: E402
from cups.semantics.model import DepthG  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEFAULT_IMG_ROOT = Path("/Volumes/code_files/datasets/cityscapes/leftImg8bit/train")
DEFAULT_DEPTH_ROOT = Path("/Volumes/code_files/datasets/cityscapes/depth_depthpro/train")
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "depthg_depthpro_monocular" / "epoch6_step1680.ckpt"
DEFAULT_OUT = Path("/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_depthg_depthpro_monocular/train")

# DepthG was trained at res=224 with crop_type=null; inference uses sliding-window over (320,320)
# tiles on the standard 640×1280 half-Cityscapes resolution (matches gen_pseudo_labels.py:151).
IMG_SHAPE = (640, 1280)
STRIDE = (160, 160)
CROP = (320, 320)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--img_root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--depth_root", type=Path, default=DEFAULT_DEPTH_ROOT)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cities", nargs="*", default=None, help="city subset; omit for all")
    p.add_argument("--limit", type=int, default=None, help="cap images per city (smoke)")
    p.add_argument("--device", default=None, choices=("cpu", "mps", "cuda"))
    p.add_argument("--num_crf_workers", type=int, default=2)
    p.add_argument("--no_crf", action="store_true", help="skip CRF (faster smoke)")
    return p.parse_args()


def list_frames(img_root: Path, cities: list[str] | None, limit: int | None) -> list[tuple[Path, Path, str]]:
    """Return [(img_path, depth_path, frame_stem), ...] for each leftImg8bit image."""
    cities = cities or sorted([d.name for d in img_root.iterdir() if d.is_dir()])
    out: list[tuple[Path, Path, str]] = []
    for city in cities:
        city_imgs = sorted((img_root / city).glob("*_leftImg8bit.png"))
        if limit is not None:
            city_imgs = city_imgs[:limit]
        for img in city_imgs:
            stem = img.stem.replace("_leftImg8bit", "")
            out.append((img, DEFAULT_DEPTH_ROOT.parent / "train" / city / f"{stem}.npy", img.stem))
    return out


def load_image(img_path: Path, device: str) -> torch.Tensor:
    """Load + resize to (1, 3, 640, 1280), ImageNet-normalize, push to device."""
    rgb = Image.open(img_path).convert("RGB").resize((IMG_SHAPE[1], IMG_SHAPE[0]), Image.BILINEAR)
    x = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def load_depth_weight(depth_path: Path, device: str) -> torch.Tensor:
    """Load DepthPro npy (assumed normalized [0,1]), compute CUPS-style depth_weight = 1/(d+1).

    Shape returned: (1, 1, 640, 1280) matching the image. depth_weight is interpolated up from
    the native (512, 1024) DepthPro cache.
    """
    d = np.load(depth_path).astype(np.float32)
    if d.ndim == 3:
        d = d[0] if d.shape[0] == 1 else d[..., 0]
    # The santosh cache is already [0,1]; if upstream stored metric depth we re-normalize so
    # depth_weight stays well-scaled.
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > 1.5 or d_min < 0.0:
        d = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    depth = torch.from_numpy(d).view(1, 1, *d.shape)
    depth = F.interpolate(depth, size=IMG_SHAPE, mode="bilinear", align_corners=False)
    depth_weight = 1.0 / (depth + 1.0)
    return depth_weight.to(device)


def write_semantic_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[gen] device={device}  ckpt={args.ckpt.name}  out={args.out}")
    assert args.ckpt.exists(), f"missing ckpt: {args.ckpt}"
    assert args.img_root.exists(), f"missing image root: {args.img_root}"
    assert args.depth_root.exists(), f"missing depth root: {args.depth_root}"

    args.out.mkdir(parents=True, exist_ok=True)

    print("[gen] loading DepthG…")
    model = DepthG(
        device=device,
        checkpoint_root=str(args.ckpt),
        img_shape=IMG_SHAPE,
        stride=STRIDE,
        crop=CROP,
    )
    print(f"[gen] n_classes={model.model.cluster_probe.n_classes}")

    frames = list_frames(args.img_root, args.cities, args.limit)
    print(f"[gen] frames to process: {len(frames)}")

    pool_ctx = nullcontext(None) if args.no_crf or args.num_crf_workers <= 0 else Pool(args.num_crf_workers)
    times: list[float] = []
    with pool_ctx as pool:
        for i, (img_path, depth_path, stem) in enumerate(frames):
            city = img_path.parent.name
            out_path = args.out / city / f"{stem}_semantic.png"
            if out_path.exists():
                continue
            if not depth_path.exists():
                print(f"[gen] SKIP {stem}: missing depth {depth_path}")
                continue
            t0 = time.time()
            img = load_image(img_path, device)
            dw = load_depth_weight(depth_path, device)
            logits = model.depth_guided_sliding_window(img, dw)  # (1, 27, 640, 1280)
            if args.no_crf:
                pred = logits.argmax(dim=1).long().squeeze(0)
            else:
                pred = batched_crf(pool, img, logits).argmax(dim=1).long().squeeze(0)
            arr = pred.cpu().numpy()
            assert arr.dtype == np.int64 and arr.shape == IMG_SHAPE, f"bad pred {arr.dtype}/{arr.shape}"
            write_semantic_png(arr, out_path)
            times.append(time.time() - t0)
            if (i + 1) % 10 == 0 or i + 1 == len(frames):
                avg = sum(times[-50:]) / max(1, len(times[-50:]))
                remaining_min = (len(frames) - i - 1) * avg / 60
                print(
                    f"[gen] {i+1:5d}/{len(frames):5d}  {city}/{stem}  unique={len(np.unique(arr))}  "
                    f"avg={avg:.2f}s/img  ETA≈{remaining_min:.0f}m"
                )

    print(f"[gen] done. wrote {sum(1 for _ in (args.out.rglob('*_semantic.png')))} files under {args.out}")


if __name__ == "__main__":
    main()
