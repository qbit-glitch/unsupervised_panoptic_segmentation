"""Build Option 1 supplementary figure: MBPS Qualitative Behavior Across Driving Datasets.

4 rows (datasets) x 3 cols (selected images). Each cell stacks:
    [original.png]
    [ours_overlay.png]
vertically. Row label on the left margin. Title at top.

Implementation note: each dataset has a DIFFERENT aspect ratio
(Cityscapes 2:1, KITTI 3.8:1, Mapillary 1.36:1, Waymo 2.18:1). To avoid the
extreme letterboxing that matplotlib produces with uniform cells, we size each
dataset block's row heights proportionally to its image aspect ratio so that
images fill their cells uniformly (modulo a single shared column width).

Output: option_1_cross_dataset_qualitative.png at ~200 DPI / ~3000x2400 px.
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
OUT_PATH = BASE / "_supplementary" / "option_1_cross_dataset_qualitative.png"

# Selections — see notes in conversation.
SELECTIONS: Dict[str, List[str]] = {
    "cityscapes": [
        "frankfurt_000001_024927",
        "frankfurt_000001_004327",
        "munster_000053_000019",
    ],
    "kitti": [
        "000028_10",
        "000163_10",
        "000006_10",
    ],
    "mapillary": [
        "aQO9K3GrRGfsPZ9pGK1E2Q",
        "4XNyiO49GE1W6chmXeP3ig",
        "1Q7hEw4K52MUqo3W8-ygoA",
    ],
    "waymo": [
        "13982731384839979987_1680_000_1700_000_1553744730524089_cam1_image",
        "9243656068381062947_1297_428_1317_428_1508793809111496_cam1_image",
        "4575389405178805994_4900_000_4920_000_1544724528697577_cam3_image",
    ],
}

# Pretty row labels.
ROW_LABELS: Dict[str, str] = {
    "cityscapes": "Cityscapes",
    "kitti": "KITTI",
    "mapillary": "Mapillary",
    "waymo": "Waymo",
}


def load_pair(dataset: str, image_id: str) -> Tuple[Image.Image, Image.Image]:
    """Load (original, ours_overlay) for one image; raise loudly if missing."""
    folder = BASE / dataset / image_id
    p_orig = folder / "original.png"
    p_overlay = folder / "ours_overlay.png"
    if not p_orig.exists():
        raise FileNotFoundError(f"Missing original.png at {p_orig}")
    if not p_overlay.exists():
        raise FileNotFoundError(f"Missing ours_overlay.png at {p_overlay}")
    return Image.open(p_orig).convert("RGB"), Image.open(p_overlay).convert("RGB")


def get_dataset_aspect(dataset: str) -> float:
    """Return W/H of this dataset's images (assumed uniform within a dataset)."""
    first_id = SELECTIONS[dataset][0]
    p = BASE / dataset / first_id / "original.png"
    with Image.open(p) as img:
        w, h = img.size
    return w / h


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_rows_ds = len(SELECTIONS)  # 4 datasets
    n_cols = 3                   # 3 picks per dataset

    # ----- Layout knobs (in figure-relative units = matplotlib row heights) -----
    label_w_ratio = 0.10         # left label margin width relative to image-col width
    block_gap_ratio = 0.10       # vertical gap between dataset blocks (relative to one image-row height)
    title_height_ratio = 0.30    # title strip height relative to one image-row height
    dpi = 200

    # Per-dataset image-row height (relative units): we set each image's aspect to
    # 1.0 wide x (1/aspect) tall when the column is unit wide. matplotlib normalizes
    # height_ratios to sum, so we use absolute relative numbers.
    aspects: Dict[str, float] = {ds: get_dataset_aspect(ds) for ds in SELECTIONS}
    # Per-dataset row height in "column-width" units:
    row_h_units: Dict[str, float] = {ds: 1.0 / aspects[ds] for ds in SELECTIONS}

    # Build height_ratios array.
    height_ratios: List[float] = [title_height_ratio]
    block_starts: List[int] = []
    for i, ds in enumerate(SELECTIONS):
        h = row_h_units[ds]
        block_starts.append(len(height_ratios))
        height_ratios.append(h)        # original
        height_ratios.append(h)        # overlay
        if i != n_rows_ds - 1:
            height_ratios.append(block_gap_ratio)

    width_ratios = [label_w_ratio] + [1.0] * n_cols

    # Total relative size: figure aspect = total_w / total_h. Each dataset row's
    # height is sized to its own image aspect, so the figure naturally comes out
    # taller than wide (aspect ~0.6). We target ~3000 px wide @ 200 DPI = 15 in
    # wide and let the height follow.
    total_w_units = sum(width_ratios)
    total_h_units = sum(height_ratios)
    aspect_fig = total_w_units / total_h_units  # natural width/height ratio

    fig_w_in = 15.0
    fig_h_in = fig_w_in / aspect_fig

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor="white")

    gs = GridSpec(
        nrows=len(height_ratios),
        ncols=1 + n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.03,
        left=0.005,
        right=0.995,
        top=0.985,
        bottom=0.005,
    )

    # ----- Title -----
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        "MBPS Qualitative Behavior Across Driving Datasets",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # ----- Per-dataset blocks -----
    for i, (ds, picks) in enumerate(SELECTIONS.items()):
        start = block_starts[i]

        # Left-margin row label spanning the 2 image rows of this block.
        ax_label = fig.add_subplot(gs[start:start + 2, 0])
        ax_label.axis("off")
        ax_label.text(
            0.55,
            0.5,
            ROW_LABELS[ds],
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            rotation=90,
            family="DejaVu Sans",
        )

        for c, image_id in enumerate(picks):
            orig, overlay = load_pair(ds, image_id)

            # Row A: original
            ax_o = fig.add_subplot(gs[start, 1 + c])
            ax_o.imshow(orig, aspect="auto")
            ax_o.set_xticks([])
            ax_o.set_yticks([])
            for spine in ax_o.spines.values():
                spine.set_visible(False)

            # Row B: overlay
            ax_v = fig.add_subplot(gs[start + 1, 1 + c])
            ax_v.imshow(overlay, aspect="auto")
            ax_v.set_xticks([])
            ax_v.set_yticks([])
            for spine in ax_v.spines.values():
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
