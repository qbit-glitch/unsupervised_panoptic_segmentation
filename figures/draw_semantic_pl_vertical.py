#!/usr/bin/env python3
"""
draw_semantic_pl_vertical.py

Paper-ready VERTICAL flowchart for the semantic pseudo-label generation
pipeline. Runs the real CAUSE pipeline and embeds actual intermediate
visualizations at each step.

Layout matches the pipeline:
  Input → Resize → 3 Sliding-Window Crops → DINOv2 → CAUSE → Bilinear →
  Stitch → L2 Norm → K-Means → NN Upsample → Pseudo-Label → Majority Vote

Output: figures/semantic_pl_vertical.pdf + .png
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress macOS Accelerate BLAS false-positive overflow warnings in PCA
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
from pathlib import Path
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CS_ROOT = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
CAUSE_DIR = PROJECT_ROOT / "refs" / "cause"
sys.path.insert(0, str(CAUSE_DIR))

STEM = "frankfurt_000000_002963"
CITY = "frankfurt"
RAW_IMG = str(CS_ROOT / "leftImg8bit" / "val" / CITY / f"{STEM}_leftImg8bit.png")
K80_PNG = str(CS_ROOT / "pseudo_semantic_raw_k80" / "val" / CITY / f"{STEM}.png")
MAP_PNG = str(CS_ROOT / "pseudo_semantic_mapped_k80" / "val" / CITY / f"{STEM}.png")

CROP_SIZE = 322
PATCH_SIZE = 14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Cityscapes palette ───────────────────────────────────────────────────────
CS19 = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# ── Visual constants ─────────────────────────────────────────────────────────
BG = "#FFFFFF"
FROZEN_FILL = "#D6EAF8"
FROZEN_EDGE = "#2980B9"
ALGO_FILL = "#FFF8E1"
ALGO_EDGE = "#F57F17"
DATA_FILL = "#F0F0F0"
DATA_EDGE = "#2C3E50"
ARROW_COL = "#4A4A4A"
DIM_COL = "#666666"
LABEL_COL = "#1A1A2E"


# ── CAUSE pipeline ───────────────────────────────────────────────────────────

def resize_for_cause(pil_img: Image.Image) -> tuple:
    """Resize: short side = 322, both dims divisible by 14."""
    orig_w, orig_h = pil_img.size
    scale = CROP_SIZE / min(orig_h, orig_w)
    new_h = int(round(orig_h * scale / PATCH_SIZE)) * PATCH_SIZE
    new_w = int(round(orig_w * scale / PATCH_SIZE)) * PATCH_SIZE
    return pil_img.resize((new_w, new_h), Image.BILINEAR), (new_h, new_w)


def load_cause_models(device: torch.device) -> tuple:
    """Load frozen DINOv2 backbone + CAUSE Segment_TR decoder."""
    from types import SimpleNamespace
    from models.dinov2vit import dinov2_vit_base_14
    from modules.segment import Segment_TR

    args = SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )
    net = dinov2_vit_base_14()
    state = torch.load(str(CAUSE_DIR / "checkpoint" / "dinov2_vit_base_14.pth"),
                       map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    segment = Segment_TR(args).to(device)
    seg_state = torch.load(
        str(CAUSE_DIR / "CAUSE" / "cityscapes" / "dinov2_vit_base_14"
            / "2048" / "segment_tr.pth"),
        map_location="cpu", weights_only=True,
    )
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()
    cb = torch.from_numpy(np.load(
        str(CAUSE_DIR / "CAUSE" / "cityscapes" / "modularity"
            / "dinov2_vit_base_14" / "2048" / "modular.npy")
    )).to(device)
    segment.head.codebook = cb
    segment.head_ema.codebook = cb
    return net, segment


def run_pipeline(device: torch.device) -> dict:
    """Run full CAUSE pipeline, return all intermediate results."""
    from modules.segment_module import transform as cause_transform

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    print("  Loading models...")
    net, segment = load_cause_models(device)

    pil_img = Image.open(RAW_IMG).convert("RGB")
    raw_np = np.array(pil_img)

    resized_pil, (rh, rw) = resize_for_cause(pil_img)
    resized_np = np.array(resized_pil)
    img_tensor = normalize(transforms.ToTensor()(resized_pil)).unsqueeze(0).to(device)

    stride = CROP_SIZE // 2
    x_positions = sorted(set(
        list(range(0, rw - CROP_SIZE, stride)) + [rw - CROP_SIZE]
    ))

    # Per-crop intermediates
    crop_images = []
    dino_pca_crops = []
    cause_pca_crops = []
    cause_up_crops = []

    feat_sum = torch.zeros(90, rh, rw)
    count = torch.zeros(1, rh, rw)

    print("  Running 3-crop sliding window...")
    for x_pos in x_positions:
        crop = img_tensor[:, :, :CROP_SIZE, x_pos:x_pos + CROP_SIZE]
        crop_images.append(resized_np[:CROP_SIZE, x_pos:x_pos + CROP_SIZE])

        with torch.no_grad():
            feat = net(crop)[:, 1:, :]  # (1, 529, 768)
            feat_flip = net(crop.flip(dims=[3]))[:, 1:, :]
            seg_feat = cause_transform(segment.head_ema(feat))  # (1,90,23,23)
            seg_feat_flip = cause_transform(segment.head_ema(feat_flip))
            seg_feat = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2

        # DINOv2 PCA (per-crop, for visualization)
        dino_np = feat[0].cpu().float().numpy()  # (529, 768)
        dino_np = np.nan_to_num(dino_np, nan=0.0, posinf=0.0, neginf=0.0)
        dino_np = np.clip(dino_np, -100, 100)
        dino_pca_crops.append(dino_np)

        # CAUSE features (per-crop, raw 23x23)
        cause_np = seg_feat[0].cpu().float().numpy()  # (90, 23, 23)
        cause_np = np.nan_to_num(cause_np, nan=0.0, posinf=0.0, neginf=0.0)
        cause_np = np.clip(cause_np, -100, 100)
        cause_pca_crops.append(cause_np)

        # Bilinear upsample
        feat_up = F.interpolate(
            seg_feat, size=(CROP_SIZE, CROP_SIZE),
            mode="bilinear", align_corners=False,
        )
        cause_up_np = feat_up[0].cpu().float().numpy()  # (90, 322, 322)
        cause_up_np = np.nan_to_num(cause_up_np, nan=0.0, posinf=0.0, neginf=0.0)
        cause_up_np = np.clip(cause_up_np, -100, 100)
        cause_up_crops.append(cause_up_np)

        ch = min(CROP_SIZE, rh)
        cw = min(CROP_SIZE, rw - x_pos)
        feat_sum[:, :ch, x_pos:x_pos + cw] += feat_up[0, :, :ch, :cw].cpu()
        count[:, :ch, x_pos:x_pos + cw] += 1

    stitched = feat_sum / count.clamp(min=1)
    stitched = torch.nan_to_num(stitched, nan=0.0, posinf=0.0, neginf=0.0)
    stitched = stitched.clamp(-100, 100)

    return {
        "raw_np": raw_np,
        "resized_np": resized_np,
        "resized_hw": (rh, rw),
        "x_positions": x_positions,
        "crop_images": crop_images,
        "dino_pca_crops": dino_pca_crops,
        "cause_pca_crops": cause_pca_crops,
        "cause_up_crops": cause_up_crops,
        "stitched": stitched.numpy(),
    }


# ── PCA visualization ────────────────────────────────────────────────────────

def pca_rgb(feat_flat: np.ndarray, hw: tuple,
            max_fit: int = 30000) -> np.ndarray:
    """PCA 3-comp RGB visualization using sklearn (stable on MPS outputs)."""
    from sklearn.decomposition import PCA as SkPCA
    data = np.nan_to_num(feat_flat.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, -100, 100)
    n = data.shape[0]
    pca = SkPCA(n_components=3, svd_solver="full")
    if n > max_fit:
        rng = np.random.RandomState(42)
        pca.fit(data[rng.choice(n, max_fit, replace=False)])
    else:
        pca.fit(data)
    proj = pca.transform(data)
    proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1)
    return proj.reshape(*hw, 3).astype(np.float32)


def colorize_k80(label: np.ndarray) -> np.ndarray:
    """Colorize k=80 cluster map."""
    tab20 = plt.get_cmap("tab20")
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for cid in np.unique(label):
        rgb[label == cid] = np.array(tab20((cid % 20) / 20.0)[:3])
    return rgb


def colorize_19cls(label: np.ndarray) -> np.ndarray:
    """Colorize 19-class pseudo-label."""
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for cid in np.unique(label):
        if cid < len(CS19):
            rgb[label == cid] = CS19[cid].astype(np.float32) / 255.0
    return rgb


# ── Build all thumbnails ─────────────────────────────────────────────────────

def build_thumbnails(results: dict) -> dict:
    """Generate all thumbnail images for the diagram."""
    raw = results["raw_np"]
    resized = results["resized_np"]

    # Thumbnail sizes (width-proportional for landscape images)
    TH_W, TH_H = 200, 100  # landscape for Cityscapes aspect

    def to_thumb(img, w=TH_W, h=TH_H, nearest=False):
        mode = Image.NEAREST if nearest else Image.BILINEAR
        return np.array(Image.fromarray(
            (img * 255).astype(np.uint8) if img.dtype == np.float32
            else img
        ).resize((w, h), mode)).astype(np.float32) / 255.0

    def to_thumb_sq(img, s=100, nearest=False):
        mode = Image.NEAREST if nearest else Image.BILINEAR
        return np.array(Image.fromarray(
            (img * 255).astype(np.uint8) if img.dtype == np.float32
            else img
        ).resize((s, s), mode)).astype(np.float32) / 255.0

    thumbs = {}

    # 1. Raw input
    thumbs["raw"] = to_thumb(raw)

    # 2. Resized
    thumbs["resized"] = to_thumb(resized)

    # 3. Three crops
    for i, crop_img in enumerate(results["crop_images"]):
        thumbs[f"crop_{i}"] = to_thumb_sq(crop_img, s=80)

    # Helper: joint PCA for per-crop visualization
    from sklearn.decomposition import PCA as SkPCA

    def joint_pca_crops(crop_feats, spatial_hw, thumb_size=80, nearest=True):
        """Fit PCA on concatenated crops, transform each individually."""
        all_flat = np.concatenate(crop_feats, axis=0).astype(np.float32)
        all_flat = np.nan_to_num(all_flat, nan=0.0, posinf=0.0, neginf=0.0)
        all_flat = np.clip(all_flat, -100, 100)
        pca = SkPCA(n_components=3, svd_solver="full")
        pca.fit(all_flat)
        result = []
        for feat in crop_feats:
            flat = np.clip(np.nan_to_num(feat.astype(np.float32)), -100, 100)
            proj = pca.transform(flat)
            proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.percentile(proj, 2, axis=0)
            hi = np.percentile(proj, 98, axis=0)
            proj = np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1)
            img = proj.reshape(*spatial_hw, 3).astype(np.float32)
            result.append(to_thumb_sq(img, s=thumb_size, nearest=nearest))
        return result

    # 4. DINOv2 PCA per crop (23x23)
    dino_flats = [d for d in results["dino_pca_crops"]]  # each (529, 768)
    dino_thumbs = joint_pca_crops(dino_flats, (23, 23), nearest=True)
    for i, t in enumerate(dino_thumbs):
        thumbs[f"dino_{i}"] = t

    # 5. CAUSE PCA per crop (23x23)
    cause_flats = [c.reshape(90, -1).T for c in results["cause_pca_crops"]]
    cause_thumbs = joint_pca_crops(cause_flats, (23, 23), nearest=True)
    for i, t in enumerate(cause_thumbs):
        thumbs[f"cause_{i}"] = t

    # 6. Bilinear upsampled CAUSE per crop (322x322)
    up_flats = [u.reshape(90, -1).T for u in results["cause_up_crops"]]
    up_thumbs = joint_pca_crops(up_flats, (CROP_SIZE, CROP_SIZE), nearest=False)
    for i, t in enumerate(up_thumbs):
        thumbs[f"up_{i}"] = t

    # 7. Stitched features PCA (322x644)
    stitched = results["stitched"]  # (90, H, W)
    rh, rw = stitched.shape[1], stitched.shape[2]
    thumbs["stitched"] = to_thumb(
        pca_rgb(stitched.reshape(90, -1).T, hw=(rh, rw)),
    )

    # 8. k=80 cluster map
    k80 = np.array(Image.open(K80_PNG))
    thumbs["k80"] = to_thumb(colorize_k80(k80), nearest=True)

    # 9. 19-class mapped
    mapped = np.array(Image.open(MAP_PNG))
    thumbs["mapped"] = to_thumb(colorize_19cls(mapped), nearest=True)

    return thumbs


# ── Figure rendering ─────────────────────────────────────────────────────────

def draw_img(ax, img_arr, x, y, w, h, border_color=DATA_EDGE):
    """Draw an image thumbnail at (x, y) with given width/height in axes coords."""
    ax.imshow(
        img_arr, extent=[x, x + w, y - h, y],
        aspect="auto", zorder=3, interpolation="bilinear",
    )
    ax.add_patch(Rectangle(
        (x, y - h), w, h,
        linewidth=1.2, edgecolor=border_color, facecolor="none", zorder=5,
    ))


def draw_box(ax, x, y, w, h, text, fill, edge, fontsize=8):
    """Draw a labeled rounded box."""
    ax.add_patch(FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0.008", linewidth=1.3,
        edgecolor=edge, facecolor=fill, zorder=4,
    ))
    ax.text(
        x + w / 2, y - h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color=LABEL_COL, zorder=6,
        multialignment="center",
    )


def draw_varrow(ax, x, y_start, y_end, label=None, label_side="right"):
    """Draw a vertical arrow with optional label."""
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COL, lw=1.2,
                        mutation_scale=10),
        zorder=4,
    )
    if label:
        offset = 0.008 if label_side == "right" else -0.008
        ha = "left" if label_side == "right" else "right"
        ax.text(
            x + offset, (y_start + y_end) / 2, label,
            ha=ha, va="center", fontsize=6.5, color=DIM_COL,
            style="italic", zorder=6,
        )


def draw_merge_lines(ax, x_sources, y_source, x_target, y_target):
    """Draw lines from multiple sources merging into one target."""
    y_mid = (y_source + y_target) / 2
    for xs in x_sources:
        ax.plot([xs, xs], [y_source, y_mid], color=ARROW_COL, lw=1.0, zorder=3)
    ax.plot(
        [min(x_sources), max(x_sources)], [y_mid, y_mid],
        color=ARROW_COL, lw=1.0, zorder=3,
    )
    ax.annotate(
        "", xy=(x_target, y_target), xytext=(x_target, y_mid),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COL, lw=1.2,
                        mutation_scale=10),
        zorder=4,
    )


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print("Running CAUSE pipeline...")
    results = run_pipeline(device)
    print("  Building thumbnails...")
    thumbs = build_thumbnails(results)
    print("  Rendering figure...")

    # ── Layout constants ─────────────────────────────────────────────────────
    FIG_W, FIG_H = 10, 18
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Image thumbnail dimensions in axes coordinates
    IW = 0.28   # landscape thumb width
    IH = 0.045  # landscape thumb height
    SQ = 0.065  # square thumb size
    COL_GAP = 0.025  # gap between 3-column items

    # Center x for single-column items
    CX = 0.5
    CX_IMG = CX - IW / 2

    # 3-column x positions (centered)
    total_3col = 3 * SQ + 2 * COL_GAP
    C3_X = [CX - total_3col / 2 + i * (SQ + COL_GAP) for i in range(3)]
    C3_CX = [x + SQ / 2 for x in C3_X]  # center of each column

    # ── Y positions (top to bottom, y decreases) ────────────────────────────
    Y = 0.96  # start near top

    # Row 1: Input Image
    draw_img(ax, thumbs["raw"], CX_IMG, Y, IW, IH)
    ax.text(CX_IMG - 0.01, Y - IH / 2, "Cityscapes Image\nH=1024, W=2048",
            ha="right", va="center", fontsize=7, color=LABEL_COL, fontweight="bold")
    Y -= IH

    # Arrow: resize
    draw_varrow(ax, CX, Y, Y - 0.025, "bilinear resize\nshort side → 322, ÷14")
    Y -= 0.025

    # Row 2: Resized Image
    draw_img(ax, thumbs["resized"], CX_IMG, Y, IW, IH)
    ax.text(CX_IMG - 0.01, Y - IH / 2, "Resized Image\nH=322, W=644",
            ha="right", va="center", fontsize=7, color=LABEL_COL, fontweight="bold")
    Y -= IH

    # Arrow: sliding window (fan-out to 3 columns)
    y_fan = Y - 0.018
    ax.annotate(
        "", xy=(CX, y_fan), xytext=(CX, Y),
        arrowprops=dict(arrowstyle="-", color=ARROW_COL, lw=1.0), zorder=3,
    )
    ax.text(CX + 0.005, (Y + y_fan) / 2, "sliding window\n322×322, stride=161",
            ha="left", va="center", fontsize=6.5, color=DIM_COL, style="italic")
    # Fan-out lines
    for cx in C3_CX:
        ax.plot([CX, cx], [y_fan, y_fan - 0.012], color=ARROW_COL, lw=1.0, zorder=3)
    Y = y_fan - 0.012

    # Row 3: Three crops
    crop_labels = ["crop @ x=0", "crop @ x=161", "crop @ x=322"]
    for i in range(3):
        draw_img(ax, thumbs[f"crop_{i}"], C3_X[i], Y, SQ, SQ)
        ax.text(C3_CX[i], Y + 0.006, crop_labels[i],
                ha="center", va="bottom", fontsize=6, color=DIM_COL)
    Y -= SQ

    # Arrows to DINOv2
    for cx in C3_CX:
        draw_varrow(ax, cx, Y, Y - 0.015)
    Y -= 0.015

    # Row 4: DINOv2 boxes (frozen)
    BOX_W, BOX_H = SQ, 0.032
    for i in range(3):
        draw_box(ax, C3_X[i], Y, BOX_W, BOX_H,
                 "DINOv2 ViT-B/14\nfrozen, 768-dim", FROZEN_FILL, FROZEN_EDGE,
                 fontsize=5.5)
    Y -= BOX_H

    # Arrows with DINOv2 PCA thumbnails
    for cx in C3_CX:
        draw_varrow(ax, cx, Y, Y - 0.012)
    Y -= 0.012

    # Row 5: DINOv2 PCA visualizations (23x23)
    for i in range(3):
        draw_img(ax, thumbs[f"dino_{i}"], C3_X[i], Y, SQ, SQ,
                 border_color=FROZEN_EDGE)
        ax.text(C3_CX[i], Y - SQ - 0.003,
                "23×23, 768-dim\n(PCA→RGB)", ha="center", va="top",
                fontsize=5, color=DIM_COL)
    Y -= SQ + 0.015

    # Arrows to CAUSE
    for cx in C3_CX:
        draw_varrow(ax, cx, Y, Y - 0.012)
    Y -= 0.012

    # Row 6: CAUSE Segment_TR boxes (frozen)
    for i in range(3):
        draw_box(ax, C3_X[i], Y, BOX_W, BOX_H,
                 "CAUSE Segment_TR\nfrozen EMA, 90-dim", FROZEN_FILL, FROZEN_EDGE,
                 fontsize=5.5)
    Y -= BOX_H

    # Arrows with CAUSE PCA thumbnails
    for cx in C3_CX:
        draw_varrow(ax, cx, Y, Y - 0.012)
    Y -= 0.012

    # Row 7: CAUSE PCA visualizations (23x23)
    for i in range(3):
        draw_img(ax, thumbs[f"cause_{i}"], C3_X[i], Y, SQ, SQ,
                 border_color=FROZEN_EDGE)
        ax.text(C3_CX[i], Y - SQ - 0.003,
                "23×23, 90-dim\n(PCA→RGB)", ha="center", va="top",
                fontsize=5, color=DIM_COL)
    Y -= SQ + 0.015

    # Arrows: bilinear upsample
    for cx in C3_CX:
        draw_varrow(ax, cx, Y, Y - 0.012)
    Y -= 0.012

    # Row 8: Bilinear upsampled (322x322)
    for i in range(3):
        draw_img(ax, thumbs[f"up_{i}"], C3_X[i], Y, SQ, SQ)
    ax.text(C3_CX[0] - SQ / 2 - 0.01, Y - SQ / 2,
            "bilinear\n23×23 → 322×322", ha="right", va="center",
            fontsize=5.5, color=DIM_COL, style="italic")
    Y -= SQ

    # Merge lines: 3 columns → single column
    y_merge_top = Y - 0.005
    y_merge_bot = y_merge_top - 0.025
    draw_merge_lines(ax, C3_CX, y_merge_top, CX, y_merge_bot)
    ax.text(CX + 0.005, (y_merge_top + y_merge_bot) / 2 + 0.005,
            "stitch + average\noverlapping regions",
            ha="left", va="center", fontsize=6.5, color=DIM_COL, style="italic")
    Y = y_merge_bot

    # Row 9: Stitched feature map
    draw_img(ax, thumbs["stitched"], CX_IMG, Y, IW, IH)
    ax.text(CX_IMG - 0.01, Y - IH / 2, "Stitched Features\n(90, 322, 644)",
            ha="right", va="center", fontsize=7, color=LABEL_COL, fontweight="bold")
    Y -= IH

    # Arrow: L2 norm + K-Means
    draw_varrow(ax, CX, Y, Y - 0.020,
                "L2 normalize per pixel\n→ K-Means k=80 (cosine sim.)")
    Y -= 0.020

    # Row 10: K-Means box
    KM_W, KM_H = 0.22, 0.028
    draw_box(ax, CX - KM_W / 2, Y, KM_W, KM_H,
             "K-Means  k=80  (cosine similarity)", ALGO_FILL, ALGO_EDGE,
             fontsize=7)
    Y -= KM_H

    # Arrow
    draw_varrow(ax, CX, Y, Y - 0.015, "cluster ID per pixel\nNN upsample → 1024×2048")
    Y -= 0.015

    # Row 11: k=80 cluster map
    draw_img(ax, thumbs["k80"], CX_IMG, Y, IW, IH)
    ax.text(CX_IMG - 0.01, Y - IH / 2,
            "Pseudo-Semantic Label\n80-class, 1024×2048",
            ha="right", va="center", fontsize=7, color=LABEL_COL, fontweight="bold")
    Y -= IH

    # Arrow: majority vote
    draw_varrow(ax, CX, Y, Y - 0.020, "majority vote mapping\n(eval only, 80→19 cls)")
    Y -= 0.020

    # Row 12: 19-class mapped
    draw_img(ax, thumbs["mapped"], CX_IMG, Y, IW, IH)
    ax.text(CX_IMG - 0.01, Y - IH / 2,
            "19-Class Mapped\n(eval only)",
            ha="right", va="center", fontsize=7, color=LABEL_COL, fontweight="bold")
    Y -= IH

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(CX, 0.99, "Semantic Pseudo-Label Generation Pipeline",
            ha="center", va="top", fontsize=13, fontweight="bold", color=LABEL_COL)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_y = Y - 0.020
    items = [
        (FROZEN_FILL, FROZEN_EDGE, "Frozen pretrained model"),
        (ALGO_FILL, ALGO_EDGE, "Algorithmic (no parameters)"),
        (None, DATA_EDGE, "Real data / PCA visualization"),
    ]
    leg_x = 0.18
    for fc, ec, txt in items:
        sw, sh = 0.020, 0.012
        ax.add_patch(Rectangle(
            (leg_x, leg_y - sh / 2), sw, sh,
            linewidth=1.2, edgecolor=ec,
            facecolor=fc if fc else "#F5F5F5", zorder=3,
        ))
        ax.text(leg_x + sw + 0.008, leg_y, txt,
                va="center", fontsize=7, color=LABEL_COL)
        leg_x += 0.28

    # ── Save ─────────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "semantic_pl_vertical")
    plt.savefig(out + ".pdf", bbox_inches="tight", dpi=300, facecolor=BG)
    plt.savefig(out + ".png", bbox_inches="tight", dpi=300, facecolor=BG)
    print(f"\nSaved:\n  {out}.pdf\n  {out}.png")


if __name__ == "__main__":
    main()
