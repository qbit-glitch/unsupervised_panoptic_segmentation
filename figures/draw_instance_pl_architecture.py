#!/usr/bin/env python3
"""
Instance Pseudo-Label Generation — architecture diagram.

Visual grammar (from diagram_style.py):
  Panel 1: Input image          — Idiom A (real data)
  Panel 2: Depth Anything v3    — Idiom B (flat frozen box, cyan)
  Panel 3: Dense depth map      — Idiom A (real data, plasma)
  Panel 4: Sobel edge magnitude — Idiom A (real data, hot) + tau badge below
  Panel 5: Binary thing mask    — Idiom A (real data) + A_min badge below
  Panel 6: Final instance masks — Idiom A (real data, TAB10)

Outputs: figures/instance_pseudolabel_architecture.pdf + .png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm
from PIL import Image as PILImage
from scipy.ndimage import sobel as scipy_sobel, label as scipy_label
from scipy.ndimage import binary_dilation

import diagram_style as ds

# ── Real Cityscapes assets ────────────────────────────────────────────────────
STEM      = "frankfurt_000000_002963"
CITY      = "frankfurt"
CS_ROOT   = "/Users/qbit-glitch/Desktop/datasets/cityscapes"
REAL_IMG  = f"{CS_ROOT}/leftImg8bit/val/{CITY}/{STEM}_leftImg8bit.png"
DEPTH_NPY = f"{CS_ROOT}/depth_dav3/val/{CITY}/{STEM}_leftImg8bit.npy"
INST_NPZ  = f"{CS_ROOT}/pseudo_instance_spidepth/val/{CITY}/{STEM}.npz"

TAB10 = plt.get_cmap("tab10")


# ── Pipeline computation (real data, unchanged from original) ─────────────────
def compute_pipeline():
    depth = np.load(DEPTH_NPY)                    # (512, 1024) float32 0-1
    sx    = scipy_sobel(depth, axis=1)
    sy    = scipy_sobel(depth, axis=0)
    grad  = np.hypot(sx, sy)

    data        = np.load(INST_NPZ, allow_pickle=True)
    masks       = data["masks"]                   # (5, 512*1024) bool
    h, w        = int(data["h_patches"]), int(data["w_patches"])
    thing_union = masks.any(axis=0).reshape(h, w)

    TAU   = 0.03
    A_MIN = 1000
    masked   = thing_union & ~(grad > TAU)
    labeled, n_comps = scipy_label(masked)

    filtered = np.zeros_like(labeled)
    inst_id  = 1
    for c in range(1, n_comps + 1):
        m = labeled == c
        if m.sum() >= A_MIN:
            filtered[m] = inst_id
            inst_id += 1

    struct   = np.ones((7, 7), dtype=bool)
    final    = np.zeros_like(filtered)
    for iid in range(1, inst_id):
        m       = filtered == iid
        m_dil   = binary_dilation(m, structure=struct)
        final[m_dil & (final == 0)] = iid

    return depth, grad, masked, final, inst_id - 1


DEPTH, GRAD, MASKED, FINAL, N_FINAL = compute_pipeline()


# ── Array-to-RGB helpers ──────────────────────────────────────────────────────
def _square_crop_array(arr, size=128):
    """Crop float32 2D array to square and resize."""
    h, w = arr.shape
    side = min(h, w)
    r0   = (h - side) // 2
    c0   = (w - side) // 2
    crop = arr[r0: r0 + side, c0: c0 + side]
    img  = PILImage.fromarray(
        (crop * 255).clip(0, 255).astype(np.uint8)
    ).resize((size, size), PILImage.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def _square_crop_label(arr, size=128):
    """Crop int array to square and resize with nearest-neighbor."""
    h, w = arr.shape
    side = min(h, w)
    r0   = (h - side) // 2
    c0   = (w - side) // 2
    crop = arr[r0: r0 + side, c0: c0 + side]
    img  = PILImage.fromarray(crop.astype(np.uint8) if crop.max() < 256
                               else (crop / max(crop.max(), 1) * 255).astype(np.uint8))
    return np.array(img.resize((size, size), PILImage.NEAREST))


def make_depth_rgb(depth, size=128):
    gray = _square_crop_array(depth, size)
    return plt.cm.plasma(gray)[:, :, :3].astype(np.float32)


def make_sobel_rgb(grad, size=128):
    g    = _square_crop_array(grad / (grad.max() + 1e-8), size)
    return plt.cm.hot(g)[:, :, :3].astype(np.float32)


def make_thing_mask_rgb(masked, size=128):
    sq = _square_crop_label(masked.astype(np.uint8), size)
    rgb = np.ones((*sq.shape, 3), dtype=np.float32) * 0.95
    rgb[sq > 0] = np.array([0.157, 0.333, 0.784])
    return rgb


def make_instance_rgb(final, n_inst, size=128):
    sq  = _square_crop_label(final, size)
    sq  = sq.astype(float) / max(final.max(), 1) * n_inst
    sq  = np.round(sq).astype(int).clip(0, n_inst)
    rgb = np.ones((*sq.shape, 3), dtype=np.float32) * 0.97
    for iid in range(1, n_inst + 1):
        m   = sq == iid
        rgb[m] = np.array(TAB10(((iid - 1) % 10) / 10.0)[:3])
    return rgb


def load_square_rgb(path, size=256):
    img  = PILImage.open(path).convert("RGB")
    iw, ih = img.size
    side = min(iw, ih)
    left = (iw - side) // 2
    top  = max(0, ih // 2 - side // 2)
    img  = img.crop((left, top, left + side, top + side))
    return np.array(img.resize((size, size), PILImage.BILINEAR)).astype(np.float32) / 255.0


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = ds.setup_figure()

# Section band behind panels 2–5
ds.draw_section_band(ax)

# ── Panel 1: Input image ──────────────────────────────────────────────────────
img_rgb = load_square_rgb(REAL_IMG, size=256)
ds.draw_data_panel(ax, ds.PX[0], ds.TOP_Y, ds.PW, ds.ITEM_H, img_rgb)
ds.draw_patch_grid_overlay(ax, ds.PX[0], ds.TOP_Y, ds.PW, ds.ITEM_H, n=7)
ds.draw_panel_labels(ax, ds.PX[0], ds.PW, ds.TOP_Y,
                     name_line1="Input Image",
                     dim_text="1024 \u00d7 2048")

# ── Arrow 1 → 2 ──────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[0][0], ds.Y_CENTER, ds.ARROWS_X[0][1], ds.Y_CENTER,
              label="infer")

# ── Panel 2: Depth Anything v3 (frozen) ───────────────────────────────────────
ds.draw_flat_module_box(ax, ds.PX[1], ds.TOP_Y, ds.PW, ds.ITEM_H,
                        name="Depth Anything v3",
                        fill=ds.FROZEN_FILL, edge=ds.FROZEN_EDGE)
ds.draw_panel_labels(ax, ds.PX[1], ds.PW, ds.TOP_Y,
                     dim_text="ViT-L  \u2022  frozen")

# ── Arrow 2 → 3 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[1][0], ds.Y_CENTER, ds.ARROWS_X[1][1], ds.Y_CENTER,
              label="depth\n(0\u20131)")

# ── Panel 3: Dense depth map ──────────────────────────────────────────────────
depth_rgb = make_depth_rgb(DEPTH, size=128)
ds.draw_data_panel(ax, ds.PX[2], ds.TOP_Y, ds.PW, ds.ITEM_H, depth_rgb)
ds.draw_panel_labels(ax, ds.PX[2], ds.PW, ds.TOP_Y,
                     name_line1="Dense Depth Map",
                     dim_text="512 \u00d7 1024  \u2022  plasma")

# ── Arrow 3 → 4 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[2][0], ds.Y_CENTER, ds.ARROWS_X[2][1], ds.Y_CENTER,
              label="Sobel filter")

# ── Panel 4: Sobel edge magnitude ─────────────────────────────────────────────
sobel_rgb = make_sobel_rgb(GRAD, size=128)
ds.draw_data_panel(ax, ds.PX[3], ds.TOP_Y, ds.PW, ds.ITEM_H, sobel_rgb)
ds.draw_panel_labels(ax, ds.PX[3], ds.PW, ds.TOP_Y,
                     name_line1="Depth Edge Magnitude",
                     dim_text="Sobel \u2016\u2207x\u2016 + \u2016\u2207y\u2016")
ds.draw_annotation_badge(ax,
                         x_center=ds.PX[3] + ds.PW / 2,
                         y_center=ds.BADGE_Y,
                         text="\u03c4 = 0.03",
                         fill=ds.TAU_BADGE_FILL,
                         edge=ds.TAU_BADGE_EDGE)

# ── Arrow 4 → 5 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[3][0], ds.Y_CENTER, ds.ARROWS_X[3][1], ds.Y_CENTER,
              label="threshold\n& AND")

# ── Panel 5: Binary thing mask ────────────────────────────────────────────────
thing_rgb = make_thing_mask_rgb(MASKED, size=128)
ds.draw_data_panel(ax, ds.PX[4], ds.TOP_Y, ds.PW, ds.ITEM_H, thing_rgb)
ds.draw_panel_labels(ax, ds.PX[4], ds.PW, ds.TOP_Y,
                     name_line1="Thing Mask",
                     dim_text="edge pixels removed")
ds.draw_annotation_badge(ax,
                         x_center=ds.PX[4] + ds.PW / 2,
                         y_center=ds.BADGE_Y,
                         text="A_min = 1000 px\n\u2022  3 px dilation",
                         fill=ds.AREA_BADGE_FILL,
                         edge=ds.AREA_BADGE_EDGE)

# ── Arrow 5 → 6 ───────────────────────────────────────────────────────────────
ds.draw_arrow(ax, ds.ARROWS_X[4][0], ds.Y_CENTER, ds.ARROWS_X[4][1], ds.Y_CENTER,
              label="CC + filter")

# ── Panel 6: Final instance masks ─────────────────────────────────────────────
inst_rgb = make_instance_rgb(FINAL, N_FINAL, size=128)
ds.draw_data_panel(ax, ds.PX[5], ds.TOP_Y, ds.PW, ds.ITEM_H, inst_rgb)
ds.draw_panel_labels(ax, ds.PX[5], ds.PW, ds.TOP_Y,
                     name_line1="Instance Pseudo-Labels",
                     dim_text="panoptic_id = cls\u00d71000 + iid")

# ── Legend ────────────────────────────────────────────────────────────────────
ds.draw_legend(ax, [
    (ds.FROZEN_FILL,      ds.FROZEN_EDGE,      "Frozen model"),
    (None,                ds.PANEL_BORDER,      "Real data panel"),
    (ds.TAU_BADGE_FILL,   ds.TAU_BADGE_EDGE,    "Hyperparameter threshold"),
])

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.002,
    "Figure 3.  Instance pseudo-label generation.  "
    "A frozen Depth Anything v3 (ViT-L) model infers a dense relative depth map.  "
    "Sobel gradient magnitudes identify depth discontinuities; pixels below \u03c4=0.03 within thing-class regions "
    "form connected components filtered by A_min=1000\u2009px and dilated 3\u2009px to produce instance pseudo-labels.",
    ha="center", va="bottom",
    fontsize=7.5, color="#333333", style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__),
                   "instance_pseudolabel_architecture")
plt.savefig(OUT + ".pdf", bbox_inches="tight", dpi=300, facecolor=ds.BG)
plt.savefig(OUT + ".png", bbox_inches="tight", dpi=300, facecolor=ds.BG)
print(f"Saved:\n  {OUT}.pdf\n  {OUT}.png")
