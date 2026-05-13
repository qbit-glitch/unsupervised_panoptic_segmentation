"""Generate the paper qualitative-results figure.

For each of N Cityscapes val images, runs Stage-3 panoptic inference and emits a
single 3-column row: RGB input | panoptic prediction | RGB overlay. All rows
are stacked into one composite figure saved at
figures/paper_ready/qualitative_stage3.png.

Usage:
    /Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python \\
        scripts/generate_qualitative_figure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# ----------------------------------------------------------------------
# Paths & imports — mirror the supplementary notebook's setup
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CITYSCAPES_ROOT = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
CUPS_ROOT = PROJECT_ROOT / "refs" / "cups"
for path in [PROJECT_ROOT, CUPS_ROOT]:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
os.environ.setdefault("WANDB_MODE", "disabled")

import cups  # noqa: E402
from cups.augmentation import PhotometricAugmentations, ResolutionJitter  # noqa: E402
from cups.data import (  # noqa: E402
    CITYSCAPES_THING_CLASSES,
    CITYSCAPES_STUFF_CLASSES,
    CITYSCAPES_CLASSNAMES,
)
from cups.model.model import prediction_to_standard_format  # noqa: E402

CFG_PATH = PROJECT_ROOT / "refs/cups/configs/val_stage3_dcfa_simcf_abc_local.yaml"
CKPT_PATH = (
    PROJECT_ROOT / "checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt"
)
OUT_FIG = PROJECT_ROOT / "figures/paper_ready/qualitative_stage3.png"
CACHE_DIR = PROJECT_ROOT / "results/stage3_visualization_cache"

# 10 representative val images (mix of frankfurt, lindau, munster + the
# canonical zurich sample that already has a cached prediction).
SAMPLES: List[Tuple[str, str]] = [
    # (city, stem) — 9 representative Cityscapes val images, sized to fit
    # on a single CVPR page. Mix of frankfurt (urban dense), lindau (mid),
    # munster (open). lindau_000019 was dropped (model emitted 0 things).
    ("frankfurt", "frankfurt_000000_000294"),
    ("frankfurt", "frankfurt_000000_001016"),
    ("frankfurt", "frankfurt_000000_009291"),
    ("frankfurt", "frankfurt_000001_007973"),
    ("frankfurt", "frankfurt_000001_014565"),
    ("frankfurt", "frankfurt_000001_083029"),
    ("lindau",    "lindau_000003_000019"),
    ("munster",   "munster_000031_000019"),
    ("munster",   "munster_000124_000019"),
]


# ----------------------------------------------------------------------
# Cityscapes 27-class palette (CUPS protocol).
# Index 0..26 follows refs/cups evaluate_kitti.py CS_NAMES_27 ordering:
# road, sidewalk, parking, rail track, building, wall, fence, guard rail,
# bridge, tunnel, pole, polegroup, traffic light, traffic sign, vegetation,
# terrain, sky, person, rider, car, truck, bus, caravan, trailer, train,
# motorcycle, bicycle.
# ----------------------------------------------------------------------
CS27_PALETTE = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [250, 170, 160],  # parking
    [230, 150, 140],  # rail track
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [180, 165, 180],  # guard rail
    [150, 100, 100],  # bridge
    [150, 120,  90],  # tunnel
    [153, 153, 153],  # pole
    [153, 153, 153],  # polegroup
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,   0,  90],  # caravan
    [  0,   0, 110],  # trailer
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
], dtype=np.uint8)


def _make_pseudo_palette(n: int = 80, seed: int = 42) -> np.ndarray:
    """Generate n visually distinct colors for pseudo-class IDs 0..n-1.

    Anchor the first 27 entries to the Cityscapes-19 colors (so when the
    Hungarian mapping happens to align certain pseudo-IDs to GT classes,
    the visualization remains close to standard Cityscapes coloring).
    Remaining slots use a perceptually-spaced HSV sweep with deterministic
    saturation/value jitter.
    """
    palette = np.zeros((n, 3), dtype=np.uint8)
    palette[: len(CS27_PALETTE)] = CS27_PALETTE
    if n > len(CS27_PALETTE):
        rng = np.random.default_rng(seed)
        rest = n - len(CS27_PALETTE)
        # Distribute hues evenly, vary saturation+value to get visually distinct colors.
        hues = (np.arange(rest) / rest + rng.uniform(0, 1)) % 1.0
        sats = rng.uniform(0.55, 1.0, size=rest)
        vals = rng.uniform(0.55, 1.0, size=rest)
        import colorsys
        for i, (h, s, v) in enumerate(zip(hues, sats, vals)):
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette[len(CS27_PALETTE) + i] = (
                int(r * 255), int(g * 255), int(b * 255)
            )
    return palette


PSEUDO_PALETTE_80 = _make_pseudo_palette(80)


def colorize_panoptic(sem: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """Colorize a (sem, inst) panoptic prediction.

    Void (sem=255 or out-of-range): light grey (200, 200, 200).
    Pseudo-class IDs 0..79: indexed into PSEUDO_PALETTE_80; first 27 colors
    are anchored to the Cityscapes palette so road, sky, vegetation, etc.
    show in their conventional colors when the model emits those IDs.
    Things: per-instance hue jitter on top of the class color so adjacent
    instances of the same class are visually separable.
    """
    h, w = sem.shape
    out = np.full((h, w, 3), 200, dtype=np.uint8)
    n_classes = len(PSEUDO_PALETTE_80)
    for c in range(n_classes):
        m = sem == c
        if not m.any():
            continue
        out[m] = PSEUDO_PALETTE_80[c]
    # Per-instance jitter (hash-based, deterministic) for thing IDs > 0.
    things_mask = (inst > 0)
    if things_mask.any():
        ids = np.unique(inst[things_mask])
        for iid in ids:
            seg = inst == iid
            if not seg.any():
                continue
            rng = np.random.default_rng(int(iid) * 9173 + 17)
            jitter = rng.integers(-40, 41, size=3)
            base = out[seg].mean(axis=0).astype(np.int32) + jitter
            base = np.clip(base, 0, 255).astype(np.uint8)
            out[seg] = base
    return out


def overlay(rgb: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    color_f = color.astype(np.float32) / 255.0
    blended = (1 - alpha) * rgb_f + alpha * color_f
    return (blended * 255.0).clip(0, 255).astype(np.uint8)


def load_rgb(stem: str, city: str) -> np.ndarray:
    p = CITYSCAPES_ROOT / "leftImg8bit" / "val" / city / f"{stem}_leftImg8bit.png"
    return np.array(Image.open(p).convert("RGB"))


def load_cached_prediction(stem: str) -> Tuple[np.ndarray, np.ndarray] | None:
    sem_p = CACHE_DIR / f"{stem}_semantic.png"
    inst_p = CACHE_DIR / f"{stem}_instance.png"
    if sem_p.exists() and inst_p.exists():
        return (
            np.array(Image.open(sem_p)).astype(np.int64),
            np.array(Image.open(inst_p)).astype(np.int64),
        )
    return None


def cups_input_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    # Detectron2 panoptic FPN expects an image tensor scaled to [0, 1] in BGR
    # layout. CUPS' build_model_self pre-normalizes internally; we just need
    # CHW float in [0, 1] order matching how the notebook uses it.
    arr = rgb.astype(np.float32) / 255.0
    # Convert RGB -> BGR for Detectron2 convention.
    arr = arr[..., ::-1].copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.to(device)


def build_model(device: str = "cpu"):
    print(f"Loading config: {CFG_PATH}")
    config = cups.get_default_config(
        experiment_config_file=str(CFG_PATH), command_line_arguments=[]
    )
    config.defrost()
    config.MODEL.CHECKPOINT = str(CKPT_PATH)
    config.SYSTEM.ACCELERATOR = device
    config.DATA.ROOT = str(CITYSCAPES_ROOT)
    config.DATA.ROOT_VAL = str(CITYSCAPES_ROOT)
    config.freeze()
    print("Building Stage-3 model on", device)
    model = cups.build_model_self(
        config=config,
        thing_pseudo_classes=None,
        stuff_pseudo_classes=None,
        thing_classes=CITYSCAPES_THING_CLASSES,
        stuff_classes=CITYSCAPES_STUFF_CLASSES,
        class_names=CITYSCAPES_CLASSNAMES,
        photometric_augmentation=PhotometricAugmentations(),
        resolution_jitter_augmentation=ResolutionJitter(
            scales=None,
            resolutions=config.AUGMENTATION.RESOLUTIONS,
        ),
        freeze_bn=True,
    )
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, rgb: np.ndarray, stem: str, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Run Stage-3 inference and return (sem, inst). Caches to disk."""
    cached = load_cached_prediction(stem)
    if cached is not None:
        print(f"  cache hit: {stem}")
        return cached
    image = cups_input_tensor(rgb, device)
    with torch.no_grad():
        prediction = model([{"image": image}])[0]
        panoptic = prediction_to_standard_format(
            prediction["panoptic_seg"],
            stuff_classes=model.hparams.stuff_pseudo_classes,
            thing_classes=model.hparams.thing_pseudo_classes,
        ).detach().cpu().numpy()
    sem = panoptic[..., 0].astype(np.uint16)
    inst = panoptic[..., 1].astype(np.uint16)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sem).save(CACHE_DIR / f"{stem}_semantic.png")
    Image.fromarray(inst).save(CACHE_DIR / f"{stem}_instance.png")
    print(f"  cached: {stem} (sem max={sem.max()}, inst max={inst.max()})")
    return sem.astype(np.int64), inst.astype(np.int64)


def main():
    device = "cpu"
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output figure → {OUT_FIG}")
    print(f"# samples: {len(SAMPLES)}")

    # Try to load model only if at least one sample is missing a cache.
    needs_inference = any(
        load_cached_prediction(stem) is None for _, stem in SAMPLES
    )
    model = build_model(device) if needs_inference else None

    rows = []
    for city, stem in SAMPLES:
        print(f"\n[{stem}] city={city}")
        rgb = load_rgb(stem, city)
        sem, inst = run_inference(model, rgb, stem, device) if model else (
            *load_cached_prediction(stem),
        )
        color = colorize_panoptic(sem, inst)
        ovl = overlay(rgb, color, alpha=0.55)
        rows.append((rgb, color, ovl, stem))

    # Composite figure: 9 rows × 3 cols, sized to fit on a single CVPR page.
    # Per-row height reduced from 2.5 to 1.4 inches; tighter padding.
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13, 1.4 * n), dpi=150)
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Input RGB", "Panoptic Prediction", "Overlay"]
    for r, (rgb, color, ovl, stem) in enumerate(rows):
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(color)
        axes[r, 2].imshow(ovl)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        # Compact left-side label.
        axes[r, 0].set_ylabel(
            stem.replace("_leftImg8bit", "").replace("_000019", ""),
            rotation=0, ha="right", va="center", fontsize=7,
            labelpad=58,
        )
        if r == 0:
            for c, title in enumerate(col_titles):
                axes[r, c].set_title(title, fontsize=10)
    plt.tight_layout(pad=0.2, h_pad=0.15, w_pad=0.15)
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"\nSaved {OUT_FIG} ({OUT_FIG.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
