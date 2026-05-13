"""Render the redesigned Fig. 5 for the paper.

Composite: 4 rows (diverse Cityscapes train frames) x 6 columns:
RGB | DepthPro depth | CAUSE-TR (raw K=80) | DCFA K=80 | Depth-CC instances |
SIMCF panoptic overlay.

Outputs: figures/paper_ready/fig5_stage1_trace_composite.png
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
OUT_DIR = PROJECT_ROOT / "figures" / "paper_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEMS = [
    "bochum_000000_016758",
    "hamburg_000000_046510",
    "monchengladbach_000000_019500",
    "erfurt_000049_000019",
]


def city(stem):
    return stem.split("_")[0]


def load_rgb(stem):
    p = CR / "leftImg8bit" / "train" / city(stem) / f"{stem}_leftImg8bit.png"
    return np.array(Image.open(p).convert("RGB"))


def load_depth(stem):
    p = CR / "depth_depthpro" / "train" / city(stem) / f"{stem}.npy"
    d = np.load(p)
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def load_label(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def load_raw_sem(stem):
    return load_label(CR / "pseudo_semantic_raw_k80" / "train" / city(stem) / f"{stem}.png")


def load_dcfa_sem(stem):
    return load_label(CR / "pseudo_semantic_adapter_V3_k80" / "train" / city(stem) / f"{stem}.png")


def load_depthcc(stem):
    p = CR / "pseudo_instance_depthpro" / "train" / city(stem) / f"{stem}_instance.png"
    return load_label(p)


def load_simcf_sem(stem):
    return load_label(CR / "cups_pseudo_labels_dcfa_simcf_abc" / f"{stem}_leftImg8bit_semantic.png")


def load_simcf_inst(stem):
    return load_label(CR / "cups_pseudo_labels_dcfa_simcf_abc" / f"{stem}_leftImg8bit_instance.png")


# Distinct color map for K=80 over-cluster labels (deterministic per id)
_RNG = np.random.default_rng(7)
_K80_COLORS = (_RNG.random((256, 3)) * 0.85 + 0.10).astype(np.float32)
_K80_COLORS[0] = (0.05, 0.05, 0.05)  # void/0


def colorize_labels(arr, palette=_K80_COLORS):
    out = palette[arr % palette.shape[0]]
    return (out * 255).astype(np.uint8)


def colorize_instances(inst):
    rng = np.random.default_rng(13)
    palette = (rng.random((256, 3)) * 0.85 + 0.10).astype(np.float32)
    palette[0] = (0.0, 0.0, 0.0)
    out = palette[inst % palette.shape[0]]
    return (out * 255).astype(np.uint8)


def panoptic_overlay(rgb, sem, inst, alpha=0.55):
    sem_color = colorize_labels(sem)
    inst_color = colorize_instances(inst)
    has_inst = inst > 0
    base = sem_color.astype(np.float32)
    base[has_inst] = inst_color[has_inst].astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    out = (1.0 - alpha) * rgb_f + alpha * base
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_instances_on_rgb(rgb, inst, alpha=0.65):
    """Overlay sparse instance map on RGB so instances pop against context."""
    inst_color = colorize_instances(inst)
    has_inst = inst > 0
    out = rgb.astype(np.float32).copy()
    out[has_inst] = (1.0 - alpha) * out[has_inst] + alpha * inst_color[has_inst].astype(np.float32)
    # darken non-instance regions slightly
    out[~has_inst] *= 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_to_rgb(d):
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    cm = plt.get_cmap("magma")
    return (cm(d)[..., :3] * 255).astype(np.uint8)


def resize_to(arr, target_hw):
    if arr.shape[:2] == target_hw:
        return arr
    pil = Image.fromarray(arr)
    return np.array(pil.resize((target_hw[1], target_hw[0]), Image.NEAREST))


def main():
    fig, axes = plt.subplots(len(STEMS), 6, figsize=(18, 3.0 * len(STEMS)))
    col_titles = [
        "RGB input",
        "DepthPro",
        "CAUSE-TR (raw K=80)",
        "DCFA K=80",
        "Depth-CC instances",
        "SIMCF panoptic",
    ]

    for r, stem in enumerate(STEMS):
        rgb = load_rgb(stem)
        H, W = rgb.shape[:2]
        # downscale for figure size
        target = (H // 2, W // 2)
        rgb_s = resize_to(rgb, target)
        d = load_depth(stem)
        depth_s = resize_to(depth_to_rgb(d), target)
        raw = load_raw_sem(stem)
        dcfa = load_dcfa_sem(stem)
        depthcc = load_depthcc(stem)
        simcf_sem = load_simcf_sem(stem)
        simcf_inst = load_simcf_inst(stem)
        raw_c = resize_to(colorize_labels(raw), target)
        dcfa_c = resize_to(colorize_labels(dcfa), target)
        depthcc_overlay = overlay_instances_on_rgb(rgb_s, resize_to(depthcc, target))
        pano = panoptic_overlay(rgb_s, resize_to(simcf_sem, target), resize_to(simcf_inst, target))

        for c, img in enumerate([rgb_s, depth_s, raw_c, dcfa_c, depthcc_overlay, pano]):
            ax = axes[r, c] if len(STEMS) > 1 else axes[c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=4)
        if len(STEMS) > 1:
            axes[r, 0].set_ylabel(stem.replace("_leftImg8bit", "").replace("_", "\\_"),
                                  fontsize=8, rotation=90, labelpad=4)

    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03, hspace=0.04)
    out_path = OUT_DIR / "fig5_stage1_trace_composite.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
