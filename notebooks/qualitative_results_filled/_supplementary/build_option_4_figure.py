"""Assemble the Option-4 supplementary figure: per-class qualitative crops.

Layout: 2 rows x 3 cols of CELLS. Each cell shows
    [original crop | ours overlay crop]
side-by-side, with a transition-type label above. 12 sub-images total.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/notebooks/qualitative_results_filled")
OUT_PATH = ROOT / "_supplementary" / "option_4_per_class_detail.png"
CROP = 256


@dataclass(frozen=True)
class Pick:
    label: str
    dataset: str
    image_id: str
    x: int
    y: int


# Hand-picked 6 crops covering the four target transition types
PICKS: List[Pick] = [
    Pick(
        label="Building / Wall / Window",
        dataset="cityscapes",
        image_id="frankfurt_000000_008206",
        x=896,
        y=128,
    ),
    Pick(
        label="Road / Sidewalk / Pole",
        dataset="cityscapes",
        image_id="frankfurt_000000_008206",
        x=256,
        y=128,
    ),
    Pick(
        label="Pedestrian Cluster",
        dataset="cityscapes",
        image_id="frankfurt_000000_020321",
        x=1024,
        y=256,
    ),
    Pick(
        label="Vegetation / Vehicles / Road",
        dataset="kitti",
        image_id="000163_10",
        x=256,
        y=0,
    ),
    Pick(
        label="Vehicle Cluster + Vegetation",
        dataset="cityscapes",
        image_id="frankfurt_000000_005898",
        x=384,
        y=128,
    ),
    Pick(
        label="Sky / Vegetation / Poles",
        dataset="mapillary",
        image_id="PxmqFeA1n_avfakO17JaZg",
        x=256,
        y=256,
    ),
]


def _load_crop(pick: Pick, name: str) -> np.ndarray:
    img_path = ROOT / pick.dataset / pick.image_id / name
    img = np.array(Image.open(img_path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img[pick.y : pick.y + CROP, pick.x : pick.x + CROP]


def main() -> None:
    fig = plt.figure(figsize=(15.0, 9.5), dpi=220)
    fig.suptitle(
        "MBPS Per-Class Qualitative Detail — fine-grained transitions from k=80 overclustering",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    # Outer grid: 2 rows x 3 cols of cells. Each cell holds: title row + image row of (orig | overlay).
    outer = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        left=0.02,
        right=0.985,
        top=0.93,
        bottom=0.05,
        wspace=0.07,
        hspace=0.20,
    )

    for idx, pick in enumerate(PICKS):
        row, col = idx // 3, idx % 3
        # Inner gridspec: 1 row, 2 cols (original | overlay)
        inner = outer[row, col].subgridspec(nrows=1, ncols=2, wspace=0.04)

        ax_o = fig.add_subplot(inner[0, 0])
        ax_p = fig.add_subplot(inner[0, 1])

        crop_orig = _load_crop(pick, "original.png")
        crop_pred = _load_crop(pick, "ours_overlay.png")

        ax_o.imshow(crop_orig)
        ax_o.set_xticks([])
        ax_o.set_yticks([])
        ax_o.set_title("Input", fontsize=10, pad=3)
        for spine in ax_o.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

        ax_p.imshow(crop_pred)
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_title("MBPS overlay", fontsize=10, pad=3)
        for spine in ax_p.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

        # Cell-level title: transition type + dataset/image_id snippet
        meta_str = f"{pick.dataset} / {pick.image_id[:30]}{'...' if len(pick.image_id) > 30 else ''}  (x={pick.x}, y={pick.y})"
        # Place a suptitle for the inner cell using the outer cell bounding box
        bbox = outer[row, col].get_position(fig)
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y1 + 0.012,
            pick.label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - 0.012,
            meta_str,
            ha="center",
            va="top",
            fontsize=7,
            color="#666666",
            family="monospace",
        )

    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote figure to %s", OUT_PATH)

    # Save the picks for record
    picks_path = ROOT / "_supplementary" / "option_4_picks.json"
    picks_path.write_text(
        json.dumps(
            [
                {"label": p.label, "dataset": p.dataset, "image_id": p.image_id, "x": p.x, "y": p.y}
                for p in PICKS
            ],
            indent=2,
        )
    )
    logger.info("Wrote picks to %s", picks_path)


if __name__ == "__main__":
    main()
