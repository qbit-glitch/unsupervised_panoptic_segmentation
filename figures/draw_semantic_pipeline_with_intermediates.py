#!/usr/bin/env python3
"""
draw_semantic_pipeline_with_intermediates.py

Paper-ready pipeline figure for semantic pseudo-label generation.
Runs the ACTUAL CAUSE pipeline on one Cityscapes image and embeds
real intermediate visualizations (PCA of DINOv2, PCA of CAUSE,
stitched features, k=80 clusters, final pseudo-labels).

Panels (left to right):
  1. Raw Cityscapes image (1024x2048)
  2. Resized (322x644) with sliding window crop boundaries
  3. DINOv2 ViT-B/14 patch tokens (PCA 3-comp RGB, 23x23)
  4. CAUSE 90-dim features — stitched + L2-normalized (PCA 3-comp, 322x644)
  5. k=80 cluster assignment map (1024x2048)
  6. 19-class pseudo-semantic label (1024x2048)

Output: figures/semantic_pipeline_intermediates.pdf + .png
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CS_ROOT = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
CAUSE_DIR = PROJECT_ROOT / "refs" / "cause"

# Add CAUSE to path
sys.path.insert(0, str(CAUSE_DIR))

STEM = "frankfurt_000000_002963"
CITY = "frankfurt"
RAW_IMG = str(CS_ROOT / "leftImg8bit" / "val" / CITY / f"{STEM}_leftImg8bit.png")
K80_PNG = str(CS_ROOT / "pseudo_semantic_raw_k80" / "val" / CITY / f"{STEM}.png")
MAP_PNG = str(CS_ROOT / "pseudo_semantic_mapped_k80" / "val" / CITY / f"{STEM}.png")
CENTROIDS = str(CS_ROOT / "pseudo_semantic_raw_k80" / "kmeans_centroids.npz")

CROP_SIZE = 322
PATCH_SIZE = 14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Cityscapes palette ───────────────────────────────────────────────────────
CS19_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# ── Visual constants ─────────────────────────────────────────────────────────
BG = "#FFFFFF"
PANEL_BORDER = "#2C3E50"
ARROW_COL = "#4A4A4A"
DIM_COL = "#555555"
STAGE_COL = "#1A1A2E"
OP_COL = "#2471A3"
FROZEN_FILL = "#D6EAF8"
FROZEN_EDGE = "#2980B9"


# ── CAUSE pipeline (same code as notebook) ───────────────────────────────────

def resize_for_cause(pil_img: Image.Image) -> tuple:
    """Resize so short side = 322, both dims divisible by 14."""
    orig_w, orig_h = pil_img.size
    scale = CROP_SIZE / min(orig_h, orig_w)
    new_h = int(round(orig_h * scale / PATCH_SIZE)) * PATCH_SIZE
    new_w = int(round(orig_w * scale / PATCH_SIZE)) * PATCH_SIZE
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return resized, (orig_h, orig_w), (new_h, new_w)


def load_cause_models(device: torch.device) -> tuple:
    """Load DINOv2 backbone + CAUSE Segment_TR decoder."""
    from types import SimpleNamespace
    from models.dinov2vit import dinov2_vit_base_14
    from modules.segment import Segment_TR

    cause_args = SimpleNamespace(
        dim=768, reduced_dim=90, projection_dim=2048,
        num_codebook=2048, n_classes=27,
        num_queries=23 * 23, crop_size=322, patch_size=14,
    )

    # DINOv2 backbone
    net = dinov2_vit_base_14()
    state = torch.load(
        str(CAUSE_DIR / "checkpoint" / "dinov2_vit_base_14.pth"),
        map_location="cpu", weights_only=True,
    )
    net.load_state_dict(state, strict=False)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    # Segment_TR decoder
    segment = Segment_TR(cause_args).to(device)
    seg_state = torch.load(
        str(CAUSE_DIR / "CAUSE" / "cityscapes" / "dinov2_vit_base_14"
            / "2048" / "segment_tr.pth"),
        map_location="cpu", weights_only=True,
    )
    segment.load_state_dict(seg_state, strict=False)
    segment.eval()

    # Modularity codebook
    cb = torch.from_numpy(np.load(
        str(CAUSE_DIR / "CAUSE" / "cityscapes" / "modularity"
            / "dinov2_vit_base_14" / "2048" / "modular.npy")
    )).to(device)
    segment.head.codebook = cb
    segment.head_ema.codebook = cb

    return net, segment


def extract_cause_features_crop(
    net: torch.nn.Module,
    segment: torch.nn.Module,
    img_tensor: torch.Tensor,
) -> tuple:
    """Extract features for one 322x322 crop.

    Returns:
        dino_tokens: (1, 529, 768) DINOv2 patch tokens
        cause_feat: (1, 90, 23, 23) CAUSE 90-dim features
    """
    from modules.segment_module import transform

    with torch.no_grad():
        feat = net(img_tensor)[:, 1:, :]
        feat_flip = net(img_tensor.flip(dims=[3]))[:, 1:, :]
        seg_feat = transform(segment.head_ema(feat))
        seg_feat_flip = transform(segment.head_ema(feat_flip))
        seg_feat = (seg_feat + seg_feat_flip.flip(dims=[3])) / 2
    return feat, seg_feat


def run_pipeline(device: torch.device) -> dict:
    """Run full CAUSE pipeline, return intermediate results for visualization."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    print("  Loading CAUSE models...")
    net, segment = load_cause_models(device)

    # Step 1: Load raw image
    pil_img = Image.open(RAW_IMG).convert("RGB")
    raw_img = np.array(pil_img)

    # Step 2: Resize
    resized_pil, (orig_h, orig_w), (rh, rw) = resize_for_cause(pil_img)
    resized_np = np.array(resized_pil)
    img_tensor = normalize(transforms.ToTensor()(resized_pil)).unsqueeze(0).to(device)

    # Step 3: Sliding window crops
    stride = CROP_SIZE // 2
    x_positions = sorted(set(
        list(range(0, rw - CROP_SIZE, stride)) + [rw - CROP_SIZE]
    ))

    # Step 4+5: DINOv2 + CAUSE features per crop, then stitch
    feat_sum = torch.zeros(90, rh, rw)
    count = torch.zeros(1, rh, rw)
    first_crop_dino = None
    first_crop_cause = None

    print("  Running sliding window inference...")
    for x_pos in x_positions:
        crop = img_tensor[:, :, :CROP_SIZE, x_pos:x_pos + CROP_SIZE]
        dino_tokens, cause_feat = extract_cause_features_crop(net, segment, crop)

        if first_crop_dino is None:
            first_crop_dino = dino_tokens.cpu().numpy()
            first_crop_cause = cause_feat[0].cpu().numpy()

        feat_up = F.interpolate(
            cause_feat, size=(CROP_SIZE, CROP_SIZE),
            mode="bilinear", align_corners=False,
        )
        ch = min(CROP_SIZE, rh)
        cw = min(CROP_SIZE, rw - x_pos)
        feat_sum[:, :ch, x_pos:x_pos + cw] += feat_up[0, :, :ch, :cw].cpu()
        count[:, :ch, x_pos:x_pos + cw] += 1

    stitched = feat_sum / count.clamp(min=1)

    # Sanitize: MPS can produce inf/NaN in some edge pixels
    stitched = torch.nan_to_num(stitched, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 6: L2 normalize
    normalized = F.normalize(stitched, dim=0, p=2)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "raw_img": raw_img,
        "resized_np": resized_np,
        "resized_hw": (rh, rw),
        "x_positions": x_positions,
        "first_crop_dino": first_crop_dino,   # (1, 529, 768)
        "first_crop_cause": first_crop_cause,  # (90, 23, 23)
        "stitched": stitched.numpy(),          # (90, H, W) — pre-L2-norm
    }


# ── PCA visualization ────────────────────────────────────────────────────────

def pca_rgb(feat_flat: np.ndarray, hw: tuple,
            max_samples: int = 50000) -> np.ndarray:
    """PCA 3-component RGB visualization with percentile normalization.

    Args:
        feat_flat: (N, D) feature vectors.
        hw: (H, W) spatial dimensions.
        max_samples: Subsample for PCA fitting to avoid numerical issues.

    Returns:
        (H, W, 3) float32 in [0, 1].
    """
    data = feat_flat.astype(np.float64)
    # Sanitize any inf/NaN from MPS computation
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Center the data first (prevents overflow in matmul)
    mean = data.mean(axis=0)
    data_centered = data - mean
    # SVD on subsampled centered data
    n = data_centered.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        _, _, Vt = np.linalg.svd(data_centered[idx], full_matrices=False)
    else:
        _, _, Vt = np.linalg.svd(data_centered, full_matrices=False)
    # Project onto top 3 components
    proj = data_centered @ Vt[:3].T
    proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1)
    return proj.reshape(*hw, 3).astype(np.float32)


def colorize_k80(label: np.ndarray) -> np.ndarray:
    """Colorize k=80 cluster map using tab20."""
    tab20 = plt.get_cmap("tab20")
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for cid in np.unique(label):
        rgb[label == cid] = np.array(tab20((cid % 20) / 20.0)[:3])
    return rgb


def colorize_19cls(label: np.ndarray) -> np.ndarray:
    """Colorize 19-class mapped pseudo-label."""
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for cid in np.unique(label):
        if cid < len(CS19_PALETTE):
            rgb[label == cid] = CS19_PALETTE[cid].astype(np.float32) / 255.0
    return rgb


# ── Build panels ─────────────────────────────────────────────────────────────

def build_panels(results: dict) -> list:
    """Build 6 panel images from pipeline results."""
    S = 256  # thumbnail render size

    # Panel 1: Raw image (aspect-preserving centre crop to square)
    raw = results["raw_img"]
    h, w = raw.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    p1 = np.array(Image.fromarray(
        raw[y0:y0 + side, x0:x0 + side]
    ).resize((S, S), Image.BILINEAR)).astype(np.float32) / 255.0

    # Panel 2: Resized with crop boundary overlay
    resized = results["resized_np"].copy()
    rh, rw = results["resized_hw"]
    # Draw crop boundaries on the resized image
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
    thickness = 3
    for i, x_pos in enumerate(results["x_positions"]):
        c = colors[i % len(colors)]
        # Top and bottom edges
        resized[:thickness, x_pos:x_pos + CROP_SIZE] = c
        resized[-thickness:, x_pos:x_pos + CROP_SIZE] = c
        # Left and right edges
        resized[:, x_pos:x_pos + thickness] = c
        end_x = min(x_pos + CROP_SIZE, rw)
        resized[:, end_x - thickness:end_x] = c
    # Resize to square for panel
    p2 = np.array(Image.fromarray(resized).resize(
        (S, S), Image.BILINEAR
    )).astype(np.float32) / 255.0

    # Panel 3: DINOv2 PCA (23x23, from first crop)
    dino_flat = results["first_crop_dino"].reshape(-1, 768)  # (529, 768)
    p3_small = pca_rgb(dino_flat, hw=(23, 23))
    p3 = np.array(Image.fromarray(
        (p3_small * 255).astype(np.uint8)
    ).resize((S, S), Image.NEAREST)).astype(np.float32) / 255.0

    # Panel 4: CAUSE stitched + L2-normalized PCA (full 322x644)
    norm_feat = results["stitched_normalized"]  # (90, H, W)
    rh, rw = norm_feat.shape[1], norm_feat.shape[2]
    norm_flat = norm_feat.reshape(90, -1).T  # (H*W, 90)
    p4_full = pca_rgb(norm_flat, hw=(rh, rw))
    p4 = np.array(Image.fromarray(
        (p4_full * 255).astype(np.uint8)
    ).resize((S, S), Image.BILINEAR)).astype(np.float32) / 255.0

    # Panel 5: k=80 cluster map (full resolution)
    k80_label = np.array(Image.open(K80_PNG))
    k80_rgb = colorize_k80(k80_label)
    side_k = min(k80_label.shape)
    y0 = (k80_label.shape[0] - side_k) // 2
    x0 = (k80_label.shape[1] - side_k) // 2
    p5 = np.array(Image.fromarray(
        (k80_rgb[y0:y0 + side_k, x0:x0 + side_k] * 255).astype(np.uint8)
    ).resize((S, S), Image.NEAREST)).astype(np.float32) / 255.0

    # Panel 6: 19-class pseudo-label (full resolution)
    map_label = np.array(Image.open(MAP_PNG))
    map_rgb = colorize_19cls(map_label)
    p6 = np.array(Image.fromarray(
        (map_rgb[y0:y0 + side_k, x0:x0 + side_k] * 255).astype(np.uint8)
    ).resize((S, S), Image.NEAREST)).astype(np.float32) / 255.0

    return [p1, p2, p3, p4, p5, p6]


# ── Figure layout ────────────────────────────────────────────────────────────
FIGSIZE = (22, 5.6)
N_PANELS = 6
PW = 0.115
GAP = 0.048
Y_C = 0.555
PH = 0.46

X_START = 0.025
PX = [X_START + i * (PW + GAP) for i in range(N_PANELS)]
ARROW_X = [(PX[i] + PW + 0.003, PX[i + 1] - 0.003) for i in range(N_PANELS - 1)]
TOP_Y = Y_C - PH / 2
BOT_Y = Y_C + PH / 2


# ── Rendering ────────────────────────────────────────────────────────────────

def draw_panel(ax, x, y_top, pw, ph, rgb_arr):
    """Draw one image panel with border."""
    ax.imshow(
        rgb_arr, extent=[x, x + pw, y_top, y_top + ph],
        aspect="auto", zorder=2, interpolation="bilinear",
    )
    ax.add_patch(Rectangle(
        (x, y_top), pw, ph,
        linewidth=1.6, edgecolor=PANEL_BORDER, facecolor="none", zorder=5,
    ))


def draw_arrow_with_label(ax, x0, x1, y, op_name, shape_str):
    """Draw arrow between panels with operation + shape labels."""
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>", color=ARROW_COL, lw=1.5, mutation_scale=13,
        ),
        zorder=4,
    )
    mid = (x0 + x1) / 2
    ax.text(
        mid, y + 0.025, op_name,
        ha="center", va="bottom",
        fontsize=7.8, fontweight="bold", color=OP_COL,
        multialignment="center", zorder=6,
    )
    ax.text(
        mid, y - 0.015, shape_str,
        ha="center", va="top",
        fontsize=6.5, color=DIM_COL, fontfamily="monospace", zorder=6,
    )


def draw_panel_annotation(ax, x, pw, y_top, ph, stage_num, name, shape,
                          frozen_tag=None):
    """Draw step number, name, shape, and optional frozen badge."""
    xc = x + pw / 2

    # Step badge (top-left corner)
    ax.text(
        x + 0.003, y_top + ph - 0.003, str(stage_num),
        ha="left", va="top",
        fontsize=7, fontweight="bold", color="white",
        bbox=dict(fc="#2C3E50", ec="none", pad=1.5, boxstyle="round,pad=0.15"),
        zorder=7,
    )

    # Name above panel
    ax.text(
        xc, y_top + ph + 0.010, name,
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color=STAGE_COL, zorder=6,
    )

    # Shape below panel
    ax.text(
        xc, y_top - 0.012, shape,
        ha="center", va="top",
        fontsize=6.5, color=DIM_COL, fontfamily="monospace", zorder=6,
    )

    # Frozen tag
    if frozen_tag:
        ax.text(
            xc, y_top - 0.030, frozen_tag,
            ha="center", va="top",
            fontsize=6.0, color="#1A5276",
            bbox=dict(fc=FROZEN_FILL, ec=FROZEN_EDGE, pad=1.5,
                      boxstyle="round,pad=0.15"),
            zorder=7,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Running CAUSE pipeline for real intermediate visualizations...")
    results = run_pipeline(device)
    print("  Pipeline complete. Building panels...")
    panels = build_panels(results)
    print("  Panels built. Rendering figure...")

    # ── Create figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # ── Frozen model background band (panels 3-4) ───────────────────────────
    band_x = PX[2] - 0.012
    band_w = (PX[3] + PW + 0.012) - band_x
    ax.add_patch(Rectangle(
        (band_x, TOP_Y - 0.005), band_w, PH + 0.010,
        linewidth=0, facecolor="#EBF5FB", zorder=0,
    ))

    # ── Draw panels ──────────────────────────────────────────────────────────
    panel_meta = [
        (1, "Input Image",            "1024 x 2048 x 3",          None),
        (2, "Resize + Crops",         "322 x 644 (3 crops)",      None),
        (3, "DINOv2 Features",        "23 x 23 x 768",            "frozen ViT-B/14"),
        (4, "CAUSE + Stitch + L2",    "322 x 644 x 90",           "frozen EMA head"),
        (5, "k=80 Cluster Map",       "1024 x 2048  [0-79]",      None),
        (6, "Pseudo-Label (19 cls)",  "1024 x 2048  [0-18]",      None),
    ]

    for i, (img, (num, name, shape, tag)) in enumerate(zip(panels, panel_meta)):
        draw_panel(ax, PX[i], TOP_Y, PW, PH, img)
        draw_panel_annotation(ax, PX[i], PW, TOP_Y, PH, num, name, shape, tag)

    # ── Patch grid overlay on input image ────────────────────────────────────
    n_grid = 8
    for k in range(n_grid + 1):
        frac = k / n_grid
        ax.plot(
            [PX[0] + frac * PW, PX[0] + frac * PW], [TOP_Y, TOP_Y + PH],
            color="white", lw=0.45, alpha=0.5, zorder=3,
        )
        ax.plot(
            [PX[0], PX[0] + PW], [TOP_Y + frac * PH, TOP_Y + frac * PH],
            color="white", lw=0.45, alpha=0.5, zorder=3,
        )

    # ── Arrows with operation labels ─────────────────────────────────────────
    ops = [
        ("Resize + Slide\nstride=161",       "short side -> 322"),
        ("DINOv2 ViT-B/14\n[frozen]",        "patch 14 -> 23x23x768"),
        ("Segment_TR EMA\n+ Bilinear + L2",   "768 -> 90-dim, stitch"),
        ("k-Means  k=80\ncosine sim.",        "nearest centroid"),
        ("Majority Vote\nRemap",              "80 cls -> 19 cls"),
    ]
    for i, ((x0, x1), (op, shp)) in enumerate(zip(ARROW_X, ops)):
        draw_arrow_with_label(ax, x0, x1, Y_C, op, shp)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (FROZEN_FILL, FROZEN_EDGE,
         "Frozen pretrained (no gradient)"),
        (None, PANEL_BORDER,
         "Real data / PCA visualization"),
    ]
    lx_start = 0.30
    for j, (fc, ec, txt) in enumerate(legend_items):
        lx = lx_start + j * 0.22
        ly = 0.075
        sw, sh = 0.018, 0.020
        ax.add_patch(Rectangle(
            (lx, ly - sh / 2), sw, sh,
            linewidth=1.4, edgecolor=ec,
            facecolor=fc if fc else "none", zorder=3,
        ))
        ax.text(
            lx + sw + 0.008, ly, txt,
            va="center", fontsize=7.5, color="#1A1A2E",
        )

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.97,
        "Semantic Pseudo-Label Generation Pipeline",
        ha="center", va="top",
        fontsize=12, fontweight="bold", color="#1A1A2E",
    )

    # ── Caption ──────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.003,
        f"Cityscapes val image {STEM}.  "
        "Panels 3-4 show PCA 3-component RGB of real features.  "
        "Panel 2 shows crop boundaries (R/G/B = crops 1/2/3, stride=161px).  "
        "Panel 5: TAB-20 cluster colors.  Panel 6: Cityscapes trainID palette.",
        ha="center", va="bottom",
        fontsize=7, color="#333333", style="italic",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "semantic_pipeline_intermediates",
    )
    plt.savefig(out_base + ".pdf", bbox_inches="tight", dpi=300, facecolor=BG)
    plt.savefig(out_base + ".png", bbox_inches="tight", dpi=300, facecolor=BG)
    print(f"\nSaved:\n  {out_base}.pdf\n  {out_base}.png")


if __name__ == "__main__":
    main()
