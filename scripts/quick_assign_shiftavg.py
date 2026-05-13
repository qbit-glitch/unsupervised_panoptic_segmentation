#!/usr/bin/env python3
"""Quick assignment: use baseline centroids on shift-avg val features.

Skips train feature extraction + re-clustering. Uses cosine similarity
to assign 64x128 upsampled features to existing 32x64 centroids (same
1024-dim DINOv2 space). Gives early signal on whether higher resolution
recovers dead classes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.generate_clustering_ablation import (
    PATCH_GRIDS,
    assign_clusters_cosine,
    find_feature_files,
)

CS = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
CENTROID_PATH = CS / "pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16" / "centroids.npz"
FEAT_SUBDIR = "dinov3_features_shiftavg_vitl16"
OUT_SUBDIR = "pseudo_semantic_raw_dinov3_k100_spherical_kmeans_vitl16_shiftavg64x128"

centers = np.load(str(CENTROID_PATH))["centers"]
print(f"Loaded centroids: {centers.shape}")

val_files = find_feature_files(CS, "val", FEAT_SUBDIR)
print(f"Found {len(val_files)} val images with shift-avg features")

out_dir = CS / OUT_SUBDIR
assign_clusters_cosine(val_files, centers, out_dir, "val")
print(f"Done -> {out_dir}/val/")
