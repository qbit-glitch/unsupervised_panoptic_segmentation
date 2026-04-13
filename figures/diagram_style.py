"""
Shared visual constants and helper functions for MBPS paper architecture diagrams.
Both draw_semantic_pl_architecture.py and draw_instance_pl_architecture.py
import from this module exclusively — no colors or drawing primitives are
defined in the diagram scripts themselves.

3-idiom visual grammar:
  A: Real data panel      — photo, depth map, pseudo-label, computed maps
  B: Flat module box      — model block (frozen=cyan, algorithmic=amber)
  C: Schematic feature grid — tensor representation, replaces random-noise imshow
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# ── Color vocabulary ──────────────────────────────────────────────────────────
BG           = "#FFFFFF"
BAND_FILL    = "#EBEBEB"    # light gray band behind algorithmic panels
PANEL_BORDER = "#2C3E50"    # dark border for real data panels (Idiom A)

FROZEN_FILL  = "#D6EAF8"    # light cyan — frozen / pretrained module
FROZEN_EDGE  = "#2980B9"    # medium blue border

MODULE_FILL  = "#EBF5FB"    # generic processing module fill
MODULE_EDGE  = "#1A5276"    # dark navy border

KMEANS_FILL  = "#FFF8E1"    # warm amber — clustering / algorithmic module
KMEANS_EDGE  = "#F57F17"    # dark amber border

ARROW_COL    = "#4A4A4A"    # dark gray arrows (not pure black — softer)
DIM_COL      = "#666666"    # secondary gray for dimension and subtitle text

TAU_BADGE_FILL  = "#FDFEFE"
TAU_BADGE_EDGE  = "#922B21"  # dark red — muted, not aggressive #E74C3C

AREA_BADGE_FILL = "#F9FBE7"
AREA_BADGE_EDGE = "#558B2F"  # muted olive green

# Pastel feature-grid cell colors (6 colors, cycling)
FEAT_PALETTE = [
    "#AED6F1",   # pale blue
    "#A9DFBF",   # pale green
    "#F9E79F",   # pale yellow
    "#F5CBA7",   # pale orange
    "#D2B4DE",   # pale purple
    "#FADBD8",   # pale pink
]

# ── Typography ────────────────────────────────────────────────────────────────
F_MODULE = 11    # module name bold (inside or below box)
F_DIM    = 7     # dimension / subtitle text (gray, below module name)
F_ARROW  = 9     # arrow midpoint label
F_BADGE  = 7     # annotation badge text
F_LEGEND = 7.5   # legend label text

# ── Layout constants (both diagrams use identical geometry) ───────────────────
FIGSIZE  = (18, 4.5)
PW       = 0.1175   # panel width — all 6 panels identical
GAP      = 0.047    # inter-panel gap
Y_CENTER = 0.56
ITEM_H   = 0.42
TOP_Y    = Y_CENTER - ITEM_H / 2    # 0.35
BOT_Y    = TOP_Y + ITEM_H           # 0.77

# Panel left edges
PX = [0.0300 + i * (PW + GAP) for i in range(6)]

# Arrow x-endpoints between consecutive panels
ARROWS_X = [(PX[i] + PW + 0.005, PX[i + 1] - 0.005) for i in range(5)]

# Section band covers panels 2–5 (the algorithmic / processing section)
BAND_X = PX[1] - 0.015
BAND_Y = TOP_Y
BAND_W = (PX[4] + PW + 0.015) - BAND_X
BAND_H = ITEM_H

# Annotation zones
BADGE_Y  = 0.22   # horizontal badge strip below panel labels
LEGEND_Y = 0.075  # legend row


# ── Helper functions ──────────────────────────────────────────────────────────

def setup_figure():
    """Create and return (fig, ax) with standard layout."""
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def draw_section_band(ax, x=BAND_X, y=BAND_Y, w=BAND_W, h=BAND_H):
    """Draw a flat gray background band (zorder=1, behind all panels)."""
    ax.add_patch(Rectangle((x, y), w, h,
                            linewidth=0, facecolor=BAND_FILL, zorder=1))


def draw_flat_module_box(ax, x, y, w, h, name, fill, edge):
    """
    Idiom B: Single flat module box.
    Sharp corners, solid border 1.5pt, module name bold centered inside.
    Nothing else inside the box — no nested layers, no sub-labels.
    """
    ax.add_patch(Rectangle((x, y), w, h,
                            linewidth=1.5, edgecolor=edge, facecolor=fill,
                            zorder=3))
    ax.text(x + w / 2, y + h / 2, name,
            ha="center", va="center",
            fontsize=F_MODULE, fontweight="bold",
            color="#1A1A2E", zorder=4,
            multialignment="center")


def draw_data_panel(ax, x, y, w, h, rgb_array):
    """
    Idiom A: Real data panel (photo, depth map, pseudo-label, computed map).
    Displays rgb_array with a dark PANEL_BORDER border.
    """
    ax.imshow(rgb_array,
              extent=[x, x + w, y, y + h],
              aspect="auto", zorder=2, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h,
                            linewidth=1.5, edgecolor=PANEL_BORDER,
                            facecolor="none", zorder=5))


def draw_patch_grid_overlay(ax, x, y, w, h, n=7):
    """
    Draw a white grid overlay on an existing data panel (input image only).
    No colored patch highlights — just the grid lines.
    """
    for i in range(n + 1):
        frac = i / n
        ax.plot([x + frac * w, x + frac * w], [y, y + h],
                color="white", lw=0.5, alpha=0.55, zorder=3)
        ax.plot([x, x + w], [y + frac * h, y + frac * h],
                color="white", lw=0.5, alpha=0.55, zorder=3)


def draw_feature_grid(ax, x, y, w, h, n_cols=6, n_rows=6, seed=42):
    """
    Idiom C: High-dimensional feature vector representation.
    Vertical bars of varying heights — each bar = one feature channel.
    Conveys "90-dim feature descriptor" rather than a spatial feature map.
    n_cols/n_rows/seed kept for API compatibility but not used.
    """
    import math
    n_bars = 12
    gap = 0.001
    bar_w = (w - (n_bars - 1) * gap) / n_bars
    for i in range(n_bars):
        bx = x + i * (bar_w + gap)
        height_frac = 0.30 + 0.60 * abs(math.sin(i * 1.7 + 0.5))
        bar_h = h * height_frac
        color = FEAT_PALETTE[i % len(FEAT_PALETTE)]
        ax.add_patch(Rectangle((bx, y), bar_w, bar_h,
                                linewidth=0.4, edgecolor="white",
                                facecolor=color, zorder=2))
    # Outer border
    ax.add_patch(Rectangle((x, y), w, h,
                            linewidth=1.5, edgecolor=MODULE_EDGE,
                            facecolor="none", zorder=5))


def draw_arrow(ax, x0, y0, x1, y1, label=None):
    """
    Standard pipeline arrow.
    Plain text label above midpoint — no pill badge, no bbox.
    """
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COL,
                                lw=1.5, mutation_scale=12),
                zorder=5)
    if label:
        mx = (x0 + x1) / 2
        ax.text(mx, y0 + 0.02, label,
                ha="center", va="bottom",
                fontsize=F_ARROW, color=DIM_COL,
                multialignment="center", zorder=6)


def draw_panel_labels(ax, x, w, top_y, name_line1=None,
                      name_line2=None, dim_text=None):
    """
    Two-level label system below each panel:
      • name_line1 (+ optional name_line2): bold F_MODULE, dark
      • dim_text: F_DIM, gray

    name_line1 is optional — pass None for Idiom B (module box) panels
    where the name already appears inside the box.
    All labels are va='top' to align cleanly at a fixed y-coordinate.
    """
    y1 = top_y - 0.045
    y_dim_base = y1

    if name_line1:
        ax.text(x + w / 2, y1, name_line1,
                ha="center", va="top",
                fontsize=F_MODULE, fontweight="bold", color="#1A1A2E")

    if name_line2:
        y2 = y_dim_base - 0.030
        ax.text(x + w / 2, y2, name_line2,
                ha="center", va="top",
                fontsize=F_MODULE, fontweight="bold", color="#1A1A2E")
        y_dim_base = y2

    if dim_text:
        y_off = 0.028 if name_line1 or name_line2 else 0.0
        ax.text(x + w / 2, y_dim_base - y_off, dim_text,
                ha="center", va="top",
                fontsize=F_DIM, color=DIM_COL)


def draw_annotation_badge(ax, x_center, y_center, text, fill, edge):
    """
    Small rounded badge placed below a panel at a fixed y coordinate.
    Static position only — never uses ax.annotate with an arrow.
    """
    ax.text(x_center, y_center, text,
            ha="center", va="center",
            fontsize=F_BADGE, color=edge,
            bbox=dict(fc=fill, ec=edge, pad=2.5,
                      boxstyle="round,pad=0.2", lw=1.0),
            zorder=6)


def draw_legend(ax, items, x_start=0.22, y=LEGEND_Y, step=0.20):
    """
    Draw a 3-item legend row with flat Rectangle swatches.
    items: list of (fill, edge, label) tuples.
    """
    sw = 0.018   # swatch width
    sh = 0.020   # swatch height
    for i, (fc, ec, txt) in enumerate(items):
        lx = x_start + i * step
        ax.add_patch(Rectangle((lx, y - sh / 2), sw, sh,
                                linewidth=1.4, edgecolor=ec,
                                facecolor=fc if fc else "none",
                                zorder=3))
        ax.text(lx + sw + 0.008, y, txt,
                va="center", fontsize=F_LEGEND, color="#1A1A2E")
