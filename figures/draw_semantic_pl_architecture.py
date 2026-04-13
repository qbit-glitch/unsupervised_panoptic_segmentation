#!/usr/bin/env python3
"""
Semantic Pseudo-Label Generation — architecture diagram.

Visual grammar (from diagram_style.py):
  Panel 1: Input image         — Idiom A (real data)
  Panel 2: DINOv2 ViT-B/14    — Idiom B (flat frozen box, cyan)
  Panel 3: CAUSE Segment_TR   — Idiom B (flat frozen box, cyan)
  Panel 4: Sliding Window + L2 — Idiom C (schematic feature grid)
  Panel 5: K-Means k=80        — Idiom B (flat algorithmic box, amber)
  Panel 6: Pseudo-semantic PL  — Idiom A (real data)

Outputs: figures/semantic_pseudolabel_architecture.pdf + .png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm
from PIL import Image as PILImage

import diagram_style as ds

# ── Real Cityscapes assets ────────────────────────────────────────────────────
STEM    = "frankfurt_000000_002963"
CITY    = "frankfurt"
CS_ROOT = "/Users/qbit-glitch/Desktop/datasets/cityscapes"
REAL_IMG = f"{CS_ROOT}/leftImg8bit/val/{CITY}/{STEM}_leftImg8bit.png"
REAL_PL  = f"{CS_ROOT}/pseudo_semantic_raw_k80/val/{CITY}/{STEM}.png"

TAB20 = plt.get_cmap("tab20")


def load_square_rgb(path, size=256, is_label=False):
    """Load image, crop centre square, resize, return float32 HxWx3."""
    img = PILImage.open(path)
    if not is_label:
        img = img.convert("RGB")
    iw, ih = img.size
    side = min(iw, ih)
    left = (iw - side) // 2
    top  = max(0, ih // 2 - side // 2)
    img  = img.crop((left, top, left + side, top + side))
    resample = PILImage.NEAREST if is_label else PILImage.BILINEAR
    return np.array(img.resize((size, size), resample=resample))


def make_pseudolabel_rgb(pl_path, size=128):
    """Map k=80 cluster IDs to TAB20 RGB."""
    pl = load_square_rgb(pl_path, size=size, is_label=True)
    rgb = np.zeros((*pl.shape, 3), dtype=np.float32)
    for cid in np.unique(pl):
        mask = pl == cid
        rgb[mask] = np.array(TAB20((cid % 20) / 20.0)[:3])
    return rgb


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = ds.setup_figure()

# Section band behind panels 2–5
ds.draw_section_band(ax)

# ── Panel 1: Input image ──────────────────────────────────────────────────────
img_rgb = load_square_rgb(REAL_IMG, size=256).astype(np.float32) / 255.0
ds.draw_data_panel(ax, ds.PX[0], ds.TOP_Y, ds.PW, ds.ITEM_H, img_rgb)
ds.draw_patch_grid_overlay(ax, ds.PX[0], ds.TOP_Y, ds.PW, ds.ITEM_H, n=7)
ds.draw_panel_labels(ax, ds.PX[0], ds.PW, ds.TOP_Y,
                     name_line1="Input Image",
                     dim_text="1024 × 2048")

# ── Arrow 1 → 2 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[0][0], ds.Y_CENTER, ds.ARROWS_X[0][1], ds.Y_CENTER,
              label="resize")

# ── Panel 2: DINOv2 ViT-B/14 (frozen) ────────────────────────────────────────
ds.draw_flat_module_box(ax, ds.PX[1], ds.TOP_Y, ds.PW, ds.ITEM_H,
                        name="DINOv2\nViT-B/14",
                        fill=ds.FROZEN_FILL, edge=ds.FROZEN_EDGE)
ds.draw_panel_labels(ax, ds.PX[1], ds.PW, ds.TOP_Y,
                     dim_text="768-dim  •  frozen")

# ── Arrow 2 → 3 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[1][0], ds.Y_CENTER, ds.ARROWS_X[1][1], ds.Y_CENTER,
              label="529 tokens")

# ── Panel 3: CAUSE Segment_TR (frozen EMA) ────────────────────────────────────
ds.draw_flat_module_box(ax, ds.PX[2], ds.TOP_Y, ds.PW, ds.ITEM_H,
                        name="CAUSE\nSegment_TR",
                        fill=ds.FROZEN_FILL, edge=ds.FROZEN_EDGE)
ds.draw_panel_labels(ax, ds.PX[2], ds.PW, ds.TOP_Y,
                     dim_text="90-dim  •  frozen EMA")

# ── Arrow 3 → 4 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[2][0], ds.Y_CENTER, ds.ARROWS_X[2][1], ds.Y_CENTER,
              label="90-dim feats")

# ── Panel 4: Sliding Window + L2 (schematic feature grid) ────────────────────
ds.draw_feature_grid(ax, ds.PX[3], ds.TOP_Y, ds.PW, ds.ITEM_H,
                     n_cols=6, n_rows=6, seed=42)
ds.draw_panel_labels(ax, ds.PX[3], ds.PW, ds.TOP_Y,
                     name_line1="Sliding Window",
                     name_line2="+ L2 Normalize",
                     dim_text="stride = 161 px")

# ── Arrow 4 → 5 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[3][0], ds.Y_CENTER, ds.ARROWS_X[3][1], ds.Y_CENTER,
              label="cosine feats")

# ── Panel 5: K-Means k=80 ────────────────────────────────────────────────────
ds.draw_flat_module_box(ax, ds.PX[4], ds.TOP_Y, ds.PW, ds.ITEM_H,
                        name="K-Means\n(k=80)",
                        fill=ds.KMEANS_FILL, edge=ds.KMEANS_EDGE)
ds.draw_panel_labels(ax, ds.PX[4], ds.PW, ds.TOP_Y,
                     dim_text="90-dim  •  cosine sim")
ds.draw_annotation_badge(ax,
                         x_center=ds.PX[4] + ds.PW / 2,
                         y_center=ds.BADGE_Y,
                         text="majority-vote \u2192 19 cls",
                         fill=ds.AREA_BADGE_FILL,
                         edge=ds.AREA_BADGE_EDGE)

# ── Arrow 5 → 6 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[4][0], ds.Y_CENTER, ds.ARROWS_X[4][1], ds.Y_CENTER,
              label="cluster ID\n(0\u201379)")

# ── Panel 6: Pseudo-semantic label ────────────────────────────────────────────
pl_rgb = make_pseudolabel_rgb(REAL_PL, size=128)
ds.draw_data_panel(ax, ds.PX[5], ds.TOP_Y, ds.PW, ds.ITEM_H, pl_rgb)
ds.draw_panel_labels(ax, ds.PX[5], ds.PW, ds.TOP_Y,
                     name_line1="Pseudo-Semantic Label",
                     dim_text="1024 \u00d7 2048  \u2022  uint8 PNG")

# ── Legend ────────────────────────────────────────────────────────────────────
ds.draw_legend(ax, [
    (ds.FROZEN_FILL, ds.FROZEN_EDGE, "Frozen encoder"),
    (ds.KMEANS_FILL, ds.KMEANS_EDGE, "Clustering module"),
    (None,           ds.PANEL_BORDER, "Real data panel"),
])

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.002,
    "Figure 2.  Semantic pseudo-label generation.  "
    "A frozen DINOv2 ViT-B/14 backbone extracts patch tokens (768-dim), compressed to 90-dim by a frozen CAUSE Segment\u2009TR EMA head.  "
    "A stride-161 sliding window with L2 normalisation feeds K-Means (k=80) clustering.  "
    "Cluster IDs (0\u201379) are mapped to 19 Cityscapes semantic classes by majority vote on train patches.",
    ha="center", va="bottom",
    fontsize=7.5, color="#333333", style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__),
                   "semantic_pseudolabel_architecture")
plt.savefig(OUT + ".pdf", bbox_inches="tight", dpi=300, facecolor=ds.BG)
plt.savefig(OUT + ".png", bbox_inches="tight", dpi=300, facecolor=ds.BG)
print(f"Saved:\n  {OUT}.pdf\n  {OUT}.png")
