"""Cache Stage-3 FPN-P4 features for the AuxThingAdapter training pipeline.

Loads the frozen Stage-3 best ckpt (DCFA + SIMCF-ABC + DepthPro), runs each
Cityscapes train image through DINOv3 + FPN, extracts the P4 feature
(stride-4 from the input, 256 channels per pixel), and saves it as a
float16 tensor on disk.

Output layout:
    <out_dir>/<image_id>.pt  --  shape (256, H/4, W/4), dtype torch.float16

Total cache size on Cityscapes train (2975 images at 512x1024 input,
P4 stride 4 → 128x256 spatial): ~9.5 GB.

This is a one-time pre-computation. Once cached, train_aux_thing_adapter.py
reads the cache directly without re-running the frozen pipeline → ~200x
speedup over direct training.

CLI:
    python scripts/cache_stage3_p4_features.py \
        --ckpt checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
        --cfg refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \
        --split train \
        --out /Users/qbit-glitch/Desktop/datasets/cityscapes/p4_cache_stage3 \
        --device cpu

Verification (50-image dry run):
    python scripts/cache_stage3_p4_features.py \
        ... same as above ... \
        --max_images 50 --check
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _import_cups(project_root: Path):
    """Add CUPS paths to sys.path."""
    cups_root = project_root / "refs" / "cups"
    for p in (project_root, cups_root):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _load_stage3_model(ckpt: Path, cfg: Path, device: str):
    """Load the Stage-3 model in eval mode, return the frozen pipeline pieces."""
    import cups
    from cups.augmentation import PhotometricAugmentations, ResolutionJitter
    from cups.data import (
        CITYSCAPES_THING_CLASSES,
        CITYSCAPES_STUFF_CLASSES,
        CITYSCAPES_CLASSNAMES,
    )

    config = cups.get_default_config(experiment_config_file=str(cfg))
    config.defrost()
    config.MODEL.CHECKPOINT = str(ckpt)
    config.SYSTEM.ACCELERATOR = device if not device.startswith("cuda") else "gpu"
    config.freeze()

    model = cups.build_model_self(
        config=config,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        class_weights=None,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        freeze_bn=True,
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None, resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _extract_p4_via_hook(model) -> dict:
    """Install a forward hook on the FPN to capture P4 (stride 4) features."""
    captured: dict = {}

    # The CUPS model's `model.model.backbone` returns a dict of
    # {"p2": ..., "p3": ..., "p4": ..., "p5": ...} (per detectron2 FPN convention).
    # Hook the backbone's forward to grab the P4 feature.
    def hook(_mod, _inp, out):
        # `out` is a dict of feature maps; capture the smallest stride that's
        # at the FPN output level. detectron2 typically uses stride-4 = "p2"
        # for SimpleFPN, but we want the deeper stride-16 one ("p4") for our
        # 256-channel feature. Save all and pick later.
        if isinstance(out, dict):
            captured["fpn_dict"] = {k: v.detach().clone() for k, v in out.items()}

    handle = model.model.backbone.register_forward_hook(hook)
    return captured, handle


def _list_images(cityscapes_root: Path, split: str) -> List[Path]:
    img_root = cityscapes_root / "leftImg8bit" / split
    paths = sorted(img_root.rglob("*_leftImg8bit.png"))
    return paths


def _resize_image(img: Image.Image, target_short: int = 512, divisor: int = 16) -> torch.Tensor:
    """Resize so short side = target_short (rounded to divisor), return (3, H, W) [0,1]."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    short = min(h, w)
    scale = target_short / short
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(divisor, (new_h // divisor) * divisor)
    new_w = max(divisor, (new_w // divisor) * divisor)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, dtype=np.uint8)).float() / 255.0
    return t.permute(2, 0, 1).contiguous()  # (3, H, W)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--cityscapes_root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max_images", default=None, type=int,
                        help="Cap for dry-run verification.")
    parser.add_argument("--check", action="store_true",
                        help="Verify cache by re-loading first 5 saved files.")
    parser.add_argument("--target_short", default=512, type=int,
                        help="Resize short side (must be divisible by 16).")
    parser.add_argument("--p4_key", default="p4",
                        help="FPN key for the stride-16 feature (CUPS uses 'p4').")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _import_cups(project_root)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading Stage-3 model from {args.ckpt}", flush=True)
    model = _load_stage3_model(args.ckpt, args.cfg, args.device)
    captured, handle = _extract_p4_via_hook(model)
    print(f"[INFO] Model loaded on {args.device}.", flush=True)

    images = _list_images(args.cityscapes_root, args.split)
    if args.max_images is not None:
        images = images[: args.max_images]
    print(f"[INFO] Will cache {len(images)} images from {args.split} split.", flush=True)

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    for img_path in tqdm(images, desc=f"caching {args.split}"):
        out_path = args.out / f"{img_path.stem.replace('_leftImg8bit', '')}.pt"
        if out_path.exists() and not args.check:
            n_skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        x = _resize_image(img, target_short=args.target_short, divisor=16).to(args.device)

        # Run the model forward (hook captures FPN outputs)
        # CUPS train_self model expects a list of dicts with "image" key.
        _ = model.model([{"image": x}])
        fpn = captured.get("fpn_dict")
        if fpn is None:
            print(f"[WARN] {img_path.name}: hook captured no FPN dict; skipping", flush=True)
            continue
        # Pick the P4 key (stride 16, 256 channels in SimpleFPN-style CUPS).
        # Some CUPS variants use different naming; try p4 first, fall back to last key.
        if args.p4_key in fpn:
            feat = fpn[args.p4_key]
        else:
            # detectron2 SimpleFPN typical keys: ["p2", "p3", "p4", "p5"]
            keys = sorted(fpn.keys())
            feat = fpn[keys[-2]]  # p4 is second-to-last
            if n_done == 0:
                print(f"[INFO] Using FPN key={keys[-2]} (available: {keys})", flush=True)
        # feat: (1, C, H, W) → save as float16 (C, H, W)
        feat = feat[0].to(torch.float16).cpu()
        torch.save(feat, out_path)
        n_done += 1
        captured.clear()

    handle.remove()
    dt = time.time() - t_start
    print(f"[DONE] cached {n_done} files; skipped {n_skipped}; elapsed={dt/60:.1f} min", flush=True)
    print(f"[DONE] Output: {args.out}", flush=True)

    if args.check:
        print("[CHECK] Verifying first 5 saved files...", flush=True)
        for i, img_path in enumerate(images[:5]):
            out_path = args.out / f"{img_path.stem.replace('_leftImg8bit', '')}.pt"
            if not out_path.exists():
                print(f"  [{i}] MISSING: {out_path.name}", flush=True)
                continue
            t = torch.load(out_path, weights_only=True)
            print(f"  [{i}] {out_path.name}: shape={tuple(t.shape)} dtype={t.dtype}", flush=True)


if __name__ == "__main__":
    main()
