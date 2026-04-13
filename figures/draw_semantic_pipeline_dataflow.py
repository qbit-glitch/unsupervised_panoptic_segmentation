"""
draw_semantic_pipeline_dataflow.py

Data flow diagram for the semantic pseudo-label generation pipeline.
Content grounded in NotebookLM query against project sources.

Pipeline:
  Raw Image → Sliding Window Crops → DINOv2 ViT-B/14 → CAUSE-TR EMA Head
  → Accumulate & Upsample → k=80 k-Means → Label Remap → Pseudo-Labels

Output: figures/semantic_pipeline_dataflow.pdf + .png
"""

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Style constants ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10.0, 14.0

# Box colours (dark fill, white text)
C_INPUT   = "#2c3e50"   # dark navy   — data stages
C_FROZEN  = "#1a5276"   # deep blue   — frozen models
C_PROC    = "#117a65"   # teal        — processing
C_CLUSTER = "#784212"   # brown       — clustering
C_OUTPUT  = "#6c3483"   # purple      — output / labels

ARROW_COL = "#555555"
SHAPE_COL = "#e8f4f8"    # light blue for tensor shape tags

BOX_W = 7.2
BOX_H = 1.15
X_CENTER = 5.0
X_LEFT = X_CENTER - BOX_W / 2

STEP_GAP = 1.75          # vertical gap between box centres


def rounded_box(ax, x, y, w, h, color, label, sublabel="", shape_tag=""):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color,
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(box)

    # Main label
    ax.text(
        x + w / 2, y + (0.12 if sublabel else 0),
        label,
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="white",
        zorder=4,
    )

    # Sub-label (hyperparams / description)
    if sublabel:
        ax.text(
            x + w / 2, y - 0.28,
            sublabel,
            ha="center", va="center",
            fontsize=8.5, color="#d5d8dc",
            zorder=4,
        )

    # Tensor shape tag (right side)
    if shape_tag:
        tag_x = x + w + 0.15
        tag_box = FancyBboxPatch(
            (tag_x, y - 0.22), len(shape_tag) * 0.115 + 0.2, 0.44,
            boxstyle="round,pad=0.04",
            facecolor=SHAPE_COL,
            edgecolor="#2980b9",
            linewidth=0.9,
            zorder=5,
        )
        ax.add_patch(tag_box)
        ax.text(
            tag_x + (len(shape_tag) * 0.115 + 0.2) / 2,
            y,
            shape_tag,
            ha="center", va="center",
            fontsize=7.5, color="#1a5276",
            fontfamily="monospace",
            zorder=6,
        )


def arrow(ax, y_top, y_bottom, label=""):
    """Draw a downward arrow between two boxes."""
    ax.annotate(
        "",
        xy=(X_CENTER, y_bottom),
        xytext=(X_CENTER, y_top),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COL,
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=2,
    )
    if label:
        ax.text(
            X_CENTER + 0.25, (y_top + y_bottom) / 2,
            label,
            ha="left", va="center",
            fontsize=7.8, color="#555555", style="italic",
        )


def section_bar(ax, y, label, color):
    """Thin coloured bar as section divider."""
    ax.axhline(y, xmin=0.05, xmax=0.95, color=color, lw=0.6, alpha=0.4, zorder=1)
    ax.text(0.3, y + 0.07, label, fontsize=7, color=color, alpha=0.7)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # Title
    ax.text(
        FIG_W / 2, FIG_H - 0.45,
        "Semantic Pseudo-Label Generation — Data Flow",
        ha="center", va="center",
        fontsize=13.5, fontweight="bold", color="#1c2833",
    )
    ax.text(
        FIG_W / 2, FIG_H - 0.85,
        "Cityscapes (1024 × 2048) → 19-class pseudo-semantic labels",
        ha="center", va="center",
        fontsize=9, color="#555555",
    )

    # ── Define steps (y decreases top → bottom) ──────────────────────────────
    tops = [FIG_H - 1.5 - i * STEP_GAP for i in range(7)]

    steps = [
        # (y, color, label, sublabel, shape_tag)
        (tops[0], C_INPUT,
         "Raw RGB Image",
         "Cityscapes training / val split",
         "H×W×3"),

        (tops[1], C_INPUT,
         "Sliding Window Crops  (+  Horizontal Flip)",
         "Shortest side = 322 px  |  stride = 161 (50% overlap)",
         "322×322×3"),

        (tops[2], C_FROZEN,
         "DINOv2  ViT-B/14  [frozen]",
         "Patch size = 14  |  embed dim = 768  |  forward_features()['x_norm_patchtokens']",
         "23×23×768"),

        (tops[3], C_FROZEN,
         "CAUSE-TR  EMA Head  [frozen]",
         "Project 768 → 90-dim  |  avg mirror crops  |  accumulate windows",
         "H×W×90"),

        (tops[4], C_PROC,
         "Bilinear Upsample  →  L2 Normalise",
         "Restore full 1024×2048 resolution  |  ‖f‖₂ = 1 per pixel",
         "1024×2048×90"),

        (tops[5], C_CLUSTER,
         "k-Means  Overclustering  (k = 80)",
         "Cosine nearest-centroid assignment  |  centroids saved to kmeans_centroids.npz",
         "1024×2048 [0–79]"),

        (tops[6], C_OUTPUT,
         "Majority-Vote Remap  →  19-Class Labels",
         "Each centroid → plurality GT class  |  saved as pseudo_semantic_raw_k80/",
         "1024×2048 [0–18]"),
    ]

    for y, col, lbl, sub, tag in steps:
        rounded_box(ax, X_LEFT, y, BOX_W, BOX_H, col, lbl, sub, tag)

    # ── Arrows ────────────────────────────────────────────────────────────────
    arrow_labels = [
        "crops ×N",
        "patch tokens",
        "projected feats",
        "accumulated + flipped avg",
        "cluster ids",
        "pseudo-semantic map",
    ]
    for i in range(len(tops) - 1):
        arrow(
            ax,
            tops[i] - BOX_H / 2 - 0.04,
            tops[i + 1] + BOX_H / 2 + 0.04,
            arrow_labels[i],
        )

    # ── Downstream usage note ─────────────────────────────────────────────────
    note_y = tops[6] - BOX_H / 2 - 0.55
    note_box = FancyBboxPatch(
        (X_LEFT, note_y - 0.38), BOX_W, 0.76,
        boxstyle="round,pad=0.06",
        facecolor="#f9f9f9",
        edgecolor="#aab7b8",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(note_box)
    ax.text(
        X_CENTER, note_y + 0.12,
        "Downstream usage:",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#2c3e50",
    )
    ax.text(
        X_CENTER, note_y - 0.18,
        "raw k=80 map → CUPS Cascade Mask R-CNN training  |  "
        "19-class map → panoptic assembly (instance-first merge)",
        ha="center", va="center",
        fontsize=7.8, color="#555555",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C_INPUT,   label="Data / preprocessing"),
        mpatches.Patch(facecolor=C_FROZEN,  label="Frozen pretrained model"),
        mpatches.Patch(facecolor=C_PROC,    label="Feature processing"),
        mpatches.Patch(facecolor=C_CLUSTER, label="Clustering"),
        mpatches.Patch(facecolor=C_OUTPUT,  label="Output pseudo-labels"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        fontsize=7.5,
        framealpha=0.85,
        edgecolor="#aab7b8",
    )

    plt.tight_layout(pad=0.3)

    out_base = "figures/semantic_pipeline_dataflow"
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_base}.png", bbox_inches="tight", dpi=200)
    print(f"Saved: {out_base}.pdf  +  {out_base}.png")
    plt.show()


if __name__ == "__main__":
    main()
