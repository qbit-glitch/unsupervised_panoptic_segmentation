"""Evaluation Visualizer.

Creates visualizations of panoptic segmentation predictions:
    - Semantic segmentation overlay
    - Instance segmentation with random colors
    - Panoptic segmentation combined view
    - Side-by-side comparison with ground truth
    - Depth map visualization
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# Fixed color palette for semantic classes (27 classes from COCO-Stuff)
COCO_STUFF_27_PALETTE = np.array([
    [0, 0, 0],       # 0: background
    [128, 0, 0],     # 1
    [0, 128, 0],     # 2
    [128, 128, 0],   # 3
    [0, 0, 128],     # 4
    [128, 0, 128],   # 5
    [0, 128, 128],   # 6
    [128, 128, 128], # 7
    [64, 0, 0],      # 8
    [192, 0, 0],     # 9
    [64, 128, 0],    # 10
    [192, 128, 0],   # 11
    [64, 0, 128],    # 12
    [192, 0, 128],   # 13
    [64, 128, 128],  # 14
    [192, 128, 128], # 15
    [0, 64, 0],      # 16
    [128, 64, 0],    # 17
    [0, 192, 0],     # 18
    [128, 192, 0],   # 19
    [0, 64, 128],    # 20
    [128, 64, 128],  # 21
    [0, 192, 128],   # 22
    [128, 192, 128], # 23
    [64, 64, 0],     # 24
    [192, 64, 0],    # 25
    [64, 192, 0],    # 26
], dtype=np.uint8)


def colorize_semantic(
    semantic_map: np.ndarray,
    palette: Optional[np.ndarray] = None,
    num_classes: int = 27,
) -> np.ndarray:
    """Colorize semantic segmentation map.

    Args:
        semantic_map: Class IDs of shape (H, W).
        palette: Color palette (K, 3). If None, uses default.
        num_classes: Number of classes.

    Returns:
        RGB image of shape (H, W, 3).
    """
    if palette is None:
        if num_classes <= 27:
            palette = COCO_STUFF_27_PALETTE
        else:
            # Generate random palette
            np.random.seed(0)
            palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)

    h, w = semantic_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(num_classes):
        mask = semantic_map == c
        if palette.shape[0] > c:
            color_map[mask] = palette[c]

    return color_map


def colorize_instances(
    instance_map: np.ndarray,
    max_instances: int = 100,
) -> np.ndarray:
    """Colorize instance segmentation map with random colors.

    Args:
        instance_map: Instance IDs of shape (H, W). 0 = background.
        max_instances: Max number of instances for palette.

    Returns:
        RGB image of shape (H, W, 3).
    """
    h, w = instance_map.shape
    np.random.seed(42)
    palette = np.random.randint(50, 255, (max_instances + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background is black

    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for inst_id in np.unique(instance_map):
        mask = instance_map == inst_id
        idx = int(inst_id) % (max_instances + 1)
        color_map[mask] = palette[idx]

    return color_map


def colorize_panoptic(
    panoptic_ids: np.ndarray,
    label_divisor: int = 1000,
    num_classes: int = 27,
) -> np.ndarray:
    """Colorize panoptic segmentation.

    Things get instance-colored, stuff gets semantic-colored.

    Args:
        panoptic_ids: Encoded panoptic IDs (H, W).
        label_divisor: Divisor for encoding.
        num_classes: Number of semantic classes.

    Returns:
        RGB image of shape (H, W, 3).
    """
    instance_ids = panoptic_ids // label_divisor
    semantic_ids = panoptic_ids % label_divisor

    h, w = panoptic_ids.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Stuff pixels: use semantic palette
    stuff_mask = instance_ids == 0
    sem_colors = colorize_semantic(semantic_ids.astype(np.int32), num_classes=num_classes)
    color_map[stuff_mask] = sem_colors[stuff_mask]

    # Thing pixels: use instance palette
    thing_mask = instance_ids > 0
    inst_colors = colorize_instances(instance_ids.astype(np.int32))
    color_map[thing_mask] = inst_colors[thing_mask]

    return color_map


def overlay_segmentation(
    image: np.ndarray,
    seg_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay segmentation colors on original image.

    Args:
        image: Original RGB image (H, W, 3), uint8.
        seg_map: Colorized segmentation (H, W, 3), uint8.
        alpha: Blend factor.

    Returns:
        Blended image (H, W, 3).
    """
    return (
        alpha * seg_map.astype(np.float32)
        + (1 - alpha) * image.astype(np.float32)
    ).astype(np.uint8)


def colorize_depth(
    depth: np.ndarray,
    colormap: str = "viridis",
) -> np.ndarray:
    """Colorize depth map using matplotlib colormap.

    Args:
        depth: Depth values (H, W).
        colormap: Matplotlib colormap name.

    Returns:
        RGB image (H, W, 3), uint8.
    """
    # Normalize to [0, 1]
    d_min, d_max = np.min(depth), np.max(depth)
    if d_max - d_min > 1e-8:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth)

    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_norm)[:, :, :3]
        return (colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: grayscale
        gray = (depth_norm * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def create_comparison_grid(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Create comparison grid visualization.

    Args:
        image: Original image (H, W, 3).
        predictions: Dict with 'semantic', 'instance', 'panoptic'.
        ground_truth: Optional GT Dict.

    Returns:
        Grid image.
    """
    h, w = image.shape[:2]
    panels = [image]

    # Prediction panels
    if "semantic" in predictions:
        sem_color = colorize_semantic(predictions["semantic"])
        panels.append(overlay_segmentation(image, sem_color))

    if "instance" in predictions:
        inst_color = colorize_instances(predictions["instance"])
        panels.append(overlay_segmentation(image, inst_color))

    if "panoptic" in predictions:
        pan_color = colorize_panoptic(predictions["panoptic"])
        panels.append(overlay_segmentation(image, pan_color))

    if "depth" in predictions:
        depth_color = colorize_depth(predictions["depth"])
        panels.append(depth_color)

    # GT panels
    if ground_truth:
        if "semantic" in ground_truth:
            gt_sem = colorize_semantic(ground_truth["semantic"])
            panels.append(overlay_segmentation(image, gt_sem))

    # Arrange in grid (2 rows if > 4 panels)
    n = len(panels)
    if n <= 4:
        grid = np.concatenate(panels, axis=1)
    else:
        row1 = np.concatenate(panels[:4], axis=1)
        row2_panels = panels[4:]
        while len(row2_panels) < 4:
            row2_panels.append(np.zeros((h, w, 3), dtype=np.uint8))
        row2 = np.concatenate(row2_panels, axis=1)
        grid = np.concatenate([row1, row2], axis=0)

    return grid


def save_visualization(
    grid: np.ndarray,
    output_path: str,
):
    """Save visualization to file.

    Args:
        grid: Grid image array.
        output_path: Output file path.
    """
    try:
        from PIL import Image
        Image.fromarray(grid).save(output_path)
    except ImportError:
        # Fallback using raw bytes
        np.save(output_path.replace(".png", ".npy"), grid)
