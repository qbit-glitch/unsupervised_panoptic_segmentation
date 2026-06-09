"""Local probe of the DepthG-DepthPro-monocular retrained ckpt (step 1680, best by cluster mIoU).

Forward 6 Cityscapes val images through `model.single_image_prediction` and render a 2-column
grid (RGB | cluster_probe argmax overlay). Tells us in ~2 min whether cluster_mIoU = 14.8 reflects
"weak but structured" pseudo-labels vs. noise.

Run from the project root:

    cd refs/cups
    PYTHONPATH=external/depthg \
      ../../.venv_cups_cpu/bin/python ../../mbps_pytorch/probe_depthg_depthpro_monocular.py

Output: reports/depthg_depthpro_probe_<timestamp>.png
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from PIL import Image

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg"))
sys.path.insert(0, str(PROJECT_ROOT / "refs" / "cups" / "external" / "depthg" / "src"))

from cups.semantics.model import DepthG  # noqa: E402

CKPT = PROJECT_ROOT / "checkpoints" / "depthg_depthpro_monocular" / "epoch6_step1680.ckpt"
VAL_ROOT = Path("/Volumes/code_files/datasets/cityscapes/leftImg8bit/val")
OUT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _pick_six() -> list[Path]:
    """Grab two existing val images per city (frankfurt, lindau, munster)."""
    picks: list[Path] = []
    for city in ("frankfurt", "lindau", "munster"):
        city_dir = VAL_ROOT / city
        files = sorted(city_dir.glob("*_leftImg8bit.png"))
        picks.extend(files[:2])
    return picks


PICKS = _pick_six()


def make_palette(n: int = 27, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 240, size=(n, 3), dtype=np.uint8)


def overlay(rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    a, b = rgb.astype(np.float32), mask_rgb.astype(np.float32)
    return np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[probe] device={device}")
    print(f"[probe] ckpt={CKPT}  size={CKPT.stat().st_size/1024/1024:.1f} MB")
    print(f"[probe] loading ckpt…")
    model = DepthG(device=device, checkpoint_root=str(CKPT), img_shape=(640, 1280), stride=(160, 160), crop=(320, 320))
    palette = make_palette(model.model.cluster_probe.n_classes)
    print(f"[probe] n_classes={model.model.cluster_probe.n_classes}")

    rows = len(PICKS)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes[None, :]

    for r, img_path in enumerate(PICKS):
        rel = img_path.relative_to(VAL_ROOT)
        rgb = Image.open(img_path).convert("RGB").resize((1280, 640), Image.BILINEAR)
        rgb_np = np.array(rgb)
        x = torch.from_numpy(rgb_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        # ImageNet normalize (matches inline `normalize` in patched data.py)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.to(device)
        with torch.no_grad():
            pred, _feats, _logits = model.single_image_prediction(x)
        pred = pred.squeeze(0).cpu().numpy().astype(np.int64)
        pred = np.clip(pred, 0, palette.shape[0] - 1)
        sem_rgb = palette[pred]
        ov = overlay(rgb_np, sem_rgb, alpha=0.55)

        axes[r, 0].imshow(rgb_np)
        axes[r, 0].set_title(img_path.stem, fontsize=10)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(ov)
        axes[r, 1].set_title(f"cluster_probe argmax ({len(np.unique(pred))} unique ids)", fontsize=10)
        axes[r, 1].axis("off")
        print(f"[probe] {rel}  unique={len(np.unique(pred))}  most-common={int(np.bincount(pred.flatten()).argmax())}")
        del x, pred, sem_rgb, ov

    out = OUT_DIR / f"depthg_depthpro_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
