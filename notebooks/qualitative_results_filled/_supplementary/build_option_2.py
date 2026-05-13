"""Build Option 2 supplementary figure: Ablation Visualization.

Originally requested as a 4-row x 4-col grid showing the same image rendered by:
    (a) Baseline CUPS (weights/cups.ckpt)
    (b) +DCFA only
    (c) +DCFA+SIMCF-ABC (full Ours)

CHECKPOINT AUDIT (2026-05-03):
    (a) Baseline CUPS:     weights/cups.ckpt                                              [OK]
    (c) Full Ours:          checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt    [OK]
    (b) DCFA-only:          NOT FOUND. No checkpoint trained on DCFA-derived pseudo-labels
                            (cups_pseudo_labels_dcfa_*) without SIMCF-ABC exists in the
                            local checkpoints/ tree. The closest siblings
                            (dinov3_vitb_depthpro_tau020_stage2/, cups_da3_causetr/,
                            dinov3_vitb_da3_causetr_stage3_*) all use raw pseudo-labels,
                            not the DCFA-corrected variant, so they would isolate the
                            wrong axis (pseudo-label-source rather than +DCFA).

Per the task's "Step 2 alternative" instructions, this script produces a 2-column
ablation figure (Original | CUPS Baseline | Ours Full) for 4 images, one per dataset.

We re-use the pre-rendered overlays from qualitative_results_filled/ which were
produced by the same scripts/generate_qualitative_ours_vs_cups.py pipeline using
exactly the (a) and (c) checkpoints listed above. No fresh inference is required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

BASE = Path(
    "/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/notebooks/qualitative_results_filled"
)
OUT_PATH = BASE / "_supplementary" / "option_2_ablation_viz.png"

# 4 images: one from each dataset. Selections mirror the first pick of Option 1
# so the supplementary figures share a coherent visual narrative.
SELECTIONS: List[Tuple[str, str, str]] = [
    ("cityscapes", "frankfurt_000001_024927", "Cityscapes"),
    ("kitti", "000028_10", "KITTI"),
    ("mapillary", "aQO9K3GrRGfsPZ9pGK1E2Q", "Mapillary"),
    ("waymo", "13982731384839979987_1680_000_1700_000_1553744730524089_cam1_image", "Waymo"),
]

COL_TITLES = ("Input Image", "CUPS baseline (Hahn '25)", "Ours (DCFA + SIMCF-ABC)")


def load_triplet(dataset: str, image_id: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Load (original, cups_overlay, ours_overlay). Raise loudly if anything is missing."""
    folder = BASE / dataset / image_id
    p_orig = folder / "original.png"
    p_cups = folder / "cups_overlay.png"
    p_ours = folder / "ours_overlay.png"
    for p in (p_orig, p_cups, p_ours):
        if not p.exists():
            raise FileNotFoundError(f"Missing required panel: {p}")
    return (
        Image.open(p_orig).convert("RGB"),
        Image.open(p_cups).convert("RGB"),
        Image.open(p_ours).convert("RGB"),
    )


def get_dataset_aspect(dataset: str, image_id: str) -> float:
    """Return W/H for one image."""
    with Image.open(BASE / dataset / image_id / "original.png") as img:
        w, h = img.size
    return w / h


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(SELECTIONS)
    n_cols = 3  # Original | CUPS | Ours

    # ----- Layout knobs -----
    label_w_ratio = 0.07    # left row-label margin width relative to one image-col width
    title_h_ratio = 0.18    # top column-titles strip height relative to one image-row height
    fig_title_h = 0.22      # figure title strip height relative to one image-row height
    row_gap_ratio = 0.04    # gap between image rows
    dpi = 200

    # Per-row height proportional to image aspect (so overlays fill cells uniformly).
    aspects: Dict[str, float] = {ds: get_dataset_aspect(ds, iid) for ds, iid, _ in SELECTIONS}
    row_h_units: Dict[str, float] = {ds: 1.0 / aspects[ds] for ds in aspects}

    # Build height_ratios: [fig_title, col_titles, row1, gap, row2, gap, row3, gap, row4]
    height_ratios: List[float] = [fig_title_h, title_h_ratio]
    image_row_indices: List[int] = []
    for i, (ds, _iid, _label) in enumerate(SELECTIONS):
        image_row_indices.append(len(height_ratios))
        height_ratios.append(row_h_units[ds])
        if i != n_rows - 1:
            height_ratios.append(row_gap_ratio)

    width_ratios = [label_w_ratio] + [1.0] * n_cols

    fig_w_in = 16.0
    total_w_units = sum(width_ratios)
    total_h_units = sum(height_ratios)
    fig_h_in = fig_w_in * (total_h_units / total_w_units)

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor="white")

    gs = GridSpec(
        nrows=len(height_ratios),
        ncols=1 + n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.04,
        wspace=0.025,
        left=0.005,
        right=0.995,
        top=0.985,
        bottom=0.005,
    )

    # ----- Figure title (row 0, full width) -----
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        "Ablation: CUPS Baseline vs. Ours (DCFA + SIMCF-ABC)",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # ----- Column titles (row 1) -----
    for c, ct in enumerate(COL_TITLES):
        ax_ct = fig.add_subplot(gs[1, 1 + c])
        ax_ct.axis("off")
        ax_ct.text(
            0.5,
            0.4,
            ct,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            family="DejaVu Sans",
        )

    # ----- Image rows -----
    for r, (ds, iid, label) in enumerate(SELECTIONS):
        gs_row = image_row_indices[r]

        ax_lab = fig.add_subplot(gs[gs_row, 0])
        ax_lab.axis("off")
        ax_lab.text(
            0.55,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            rotation=90,
            family="DejaVu Sans",
        )

        orig, cups_ov, ours_ov = load_triplet(ds, iid)
        panels = [orig, cups_ov, ours_ov]
        for c, panel in enumerate(panels):
            ax = fig.add_subplot(gs[gs_row, 1 + c])
            ax.imshow(panel, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.savefig(OUT_PATH, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"WROTE: {OUT_PATH} ({size_kb:.1f} KB)")
    with Image.open(OUT_PATH) as img:
        print(f"PIXELS: {img.size}")
        print(f"FIG_INCHES: {fig_w_in:.2f} x {fig_h_in:.2f} (target dpi={dpi})")


if __name__ == "__main__":
    main()
