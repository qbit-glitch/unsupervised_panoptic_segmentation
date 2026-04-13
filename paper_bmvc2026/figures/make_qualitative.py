#!/usr/bin/env python3
"""Generate Figure 5: Full pipeline qualitative comparison with all intermediate steps.

7 rows × 3 images:
  Row 1: RGB Input
  Row 2: Depth Map (SPIdepth viridis)
  Row 3: Depth Edges (Sobel threshold binary)
  Row 4: Semantic Pseudo-Labels (k=80 colorized)
  Row 5: Instance Pseudo-Labels (depth-guided, colorized overlay)
  Row 6: Stage-2 Panoptic Prediction
  Row 7: Stage-3 Panoptic Prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# Directories
FI = Path("../notebooks/visualizations/figures_instance_pipeline")
FS2 = Path("../notebooks/visualizations/figures_stage2_pipeline")
FS3 = Path("../notebooks/visualizations/figures_stage3_pipeline")
RGB_DIR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit/val")

SAMPLES = [
    ("frankfurt", "frankfurt_000000_002963"),
    ("munster", "munster_000069_000019"),
    ("lindau", "lindau_000042_000019"),
]

def load(path, w=400, h=200):
    if not path.exists():
        print(f"  MISSING: {path}")
        return np.ones((h, w, 3), dtype=np.uint8) * 200
    return np.array(Image.open(path).resize((w, h), Image.LANCZOS))

# 7 rows × 3 columns
fig, axes = plt.subplots(7, 3, figsize=(14, 18))

row_labels = [
    "(a) RGB Input",
    "(b) Depth Map",
    "(c) Depth Edges (τ=0.20)",
    "(d) Semantic PLs (k=80)",
    "(e) Instance PLs",
    "(f) Stage-2 (PQ=27.87%)",
    "(g) Stage-3 (PQ=32.76%)",
]

for col, (city, stem) in enumerate(SAMPLES):
    print(f"Processing {stem}...")

    # Row 0: RGB
    axes[0, col].imshow(load(RGB_DIR / city / f"{stem}_leftImg8bit.png"))
    axes[0, col].set_title(stem.split("_")[0], fontsize=11, fontweight="bold")

    # Row 1: Depth map
    axes[1, col].imshow(load(FI / f"step01_depth_{stem}.png"))

    # Row 2: Depth edges
    axes[2, col].imshow(load(FI / f"step04_depth_edges_{stem}.png"))

    # Row 3: Semantic pseudo-labels (raw colorized — every pixel has a class color)
    sem_path = FI / f"step05_semantic_{stem}.png"
    axes[3, col].imshow(load(sem_path))

    # Row 4: Instance pseudo-labels (overlay on RGB to show spatial context)
    inst_path = FI / f"step11_final_instances_overlay_{stem}.png"
    if not inst_path.exists():
        inst_path = FI / f"step11_final_instances_{stem}.png"  # raw colored instances
    axes[4, col].imshow(load(inst_path))

    # Row 5: Stage-2 panoptic
    s2_path = FS2 / f"step07_s2_panoptic_overlay_{stem}.png"
    if not s2_path.exists():
        s2_path = FS3 / f"step06_s2_panoptic_{stem}.png"
    axes[5, col].imshow(load(s2_path))

    # Row 6: Stage-3 panoptic
    s3_path = FS3 / f"step05_s3_panoptic_overlay_{stem}.png"
    if not s3_path.exists():
        s3_path = FS3 / f"step06_s3_panoptic_{stem}.png"
    axes[6, col].imshow(load(s3_path))

# Turn off all axes and add row labels
for row in range(7):
    for col in range(3):
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(row_labels[row], fontsize=9, fontweight="bold",
                            rotation=90, labelpad=12, va="center")

# Add phase annotations on the left
fig.text(0.01, 0.78, "Phase 1:\nPseudo-Label\nGeneration",
         fontsize=9, fontweight="bold", color="#4CAF50",
         ha="center", va="center", rotation=90)
fig.text(0.01, 0.22, "Phase 2:\nNetwork\nTraining",
         fontsize=9, fontweight="bold", color="#1565C0",
         ha="center", va="center", rotation=90)

# Add a thin separator between Phase 1 and Phase 2
line_y = 0.34  # between row 4 and row 5
fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y],
               transform=fig.transFigure, color="gray",
               linewidth=1.5, linestyle="--"))

plt.tight_layout(pad=0.3, rect=[0.03, 0, 1, 1])
fig.savefig("figures/fig5_qualitative.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
fig.savefig("figures/fig5_qualitative.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("\nSaved: figures/fig5_qualitative.pdf")
print("Saved: figures/fig5_qualitative.png")
plt.close()
