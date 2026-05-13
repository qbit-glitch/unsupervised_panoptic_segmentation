"""Approach #5: Local Plane Decomposition (GeoDepth-inspired).

Fits local depth planes via SVD per patch, detects instance boundaries
where adjacent patches have different surface normals or high plane
fitting residuals. Inspired by GeoDepth (CVPR 2025).
"""

import numpy as np
from scipy import ndimage

from .utils import dilation_reclaim

THING_IDS = set(range(11, 19))


def _fit_planes(
    depth: np.ndarray,
    patch_size: int = 16,
) -> tuple:
    """Fit local planes on depth map via SVD.

    For each patch, fits z = ax + by + c using least squares (SVD).

    Args:
        depth: (H, W) float32 depth map.
        patch_size: size of each patch in pixels.

    Returns:
        normals: (ph, pw, 3) surface normals per patch.
        residuals: (ph, pw) mean fitting residual per patch.
    """
    H, W = depth.shape
    ph = H // patch_size
    pw = W // patch_size

    normals = np.zeros((ph, pw, 3), dtype=np.float64)
    residuals = np.zeros((ph, pw), dtype=np.float64)

    for py in range(ph):
        for px in range(pw):
            y0, y1 = py * patch_size, (py + 1) * patch_size
            x0, x1 = px * patch_size, (px + 1) * patch_size
            patch = depth[y0:y1, x0:x1].astype(np.float64)

            # Build coordinate grid
            ys = np.arange(y0, y1)
            xs = np.arange(x0, x1)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            yy_flat = yy.ravel()
            xx_flat = xx.ravel()
            z_flat = patch.ravel()

            # Fit plane z = ax + by + c via least squares
            # A @ [a, b, c]^T = z
            A = np.column_stack([
                xx_flat, yy_flat, np.ones_like(xx_flat)
            ])

            try:
                coeffs, res, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
                a, b, c = coeffs

                # Surface normal: n = (-a, -b, 1) normalized
                normal = np.array([-a, -b, 1.0])
                normal /= np.linalg.norm(normal) + 1e-10
                normals[py, px] = normal

                # Mean residual
                z_pred = A @ coeffs
                residuals[py, px] = np.mean(np.abs(z_flat - z_pred))
            except np.linalg.LinAlgError:
                normals[py, px] = np.array([0.0, 0.0, 1.0])
                residuals[py, px] = 0.0

    return normals, residuals


def _compute_plane_boundaries(
    normals: np.ndarray,
    residuals: np.ndarray,
    normal_angle_threshold: float = 15.0,
    residual_threshold: float = 0.02,
    patch_size: int = 16,
    target_h: int = 512,
    target_w: int = 1024,
) -> np.ndarray:
    """Compute boundary map from plane discontinuities.

    Args:
        normals: (ph, pw, 3) surface normals per patch.
        residuals: (ph, pw) fitting residuals per patch.
        normal_angle_threshold: angle difference (degrees) for boundary.
        residual_threshold: residual threshold for boundary.
        patch_size: original patch size.
        target_h, target_w: output boundary map resolution.

    Returns:
        boundary: (target_h, target_w) bool boundary map.
    """
    ph, pw = normals.shape[:2]
    cos_threshold = np.cos(np.radians(normal_angle_threshold))

    # Compute patch-level boundary map
    patch_boundary = np.zeros((ph, pw), dtype=bool)

    for py in range(ph):
        for px in range(pw):
            n1 = normals[py, px]

            # Check right neighbor
            if px + 1 < pw:
                n2 = normals[py, px + 1]
                cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
                if cos_angle < cos_threshold:
                    patch_boundary[py, px] = True
                    patch_boundary[py, px + 1] = True

            # Check bottom neighbor
            if py + 1 < ph:
                n2 = normals[py + 1, px]
                cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
                if cos_angle < cos_threshold:
                    patch_boundary[py, px] = True
                    patch_boundary[py + 1, px] = True

            # High residual patches are also boundaries
            if residuals[py, px] > residual_threshold:
                patch_boundary[py, px] = True

    # Upsample patch boundary to pixel resolution
    boundary = np.zeros((target_h, target_w), dtype=bool)
    for py in range(ph):
        for px in range(pw):
            if patch_boundary[py, px]:
                y0 = py * patch_size
                y1 = min((py + 1) * patch_size, target_h)
                x0 = px * patch_size
                x1 = min((px + 1) * patch_size, target_w)
                # Mark boundary at patch edges (not fill entire patch)
                boundary[y0:y1, x0] = True
                boundary[y0:y1, min(x1, target_w - 1)] = True
                boundary[y0, x0:x1] = True
                boundary[min(y1, target_h - 1), x0:x1] = True

    return boundary


def plane_decomp_instances(
    semantic: np.ndarray,
    depth: np.ndarray,
    thing_ids: set = THING_IDS,
    patch_size: int = 16,
    normal_angle_threshold: float = 15.0,
    residual_threshold: float = 0.02,
    min_area: int = 1000,
    dilation_iters: int = 3,
    depth_blur_sigma: float = 1.0,
    features: np.ndarray = None,
) -> list:
    """Instance decomposition via local plane fitting on depth.

    Args:
        semantic: (H,W) uint8 trainID map.
        depth: (H,W) float32 [0,1].
        thing_ids: set of thing class trainIDs.
        patch_size: patch size for plane fitting (pixels).
        normal_angle_threshold: angle threshold (degrees) between adjacent
            patch normals to mark a boundary.
        residual_threshold: plane fitting residual threshold for boundary.
        min_area: minimum instance area.
        dilation_iters: boundary reclamation iterations.
        depth_blur_sigma: Gaussian blur on depth before plane fitting.
        features: unused (included for interface compatibility).

    Returns:
        List of (mask, class_id, score).
    """
    H, W = semantic.shape

    # Optional depth smoothing
    if depth_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        depth_smooth = gaussian_filter(
            depth.astype(np.float64), sigma=depth_blur_sigma
        )
    else:
        depth_smooth = depth.astype(np.float64)

    # Fit planes and compute boundaries
    normals, residuals = _fit_planes(depth_smooth, patch_size=patch_size)
    boundary = _compute_plane_boundaries(
        normals, residuals,
        normal_angle_threshold=normal_angle_threshold,
        residual_threshold=residual_threshold,
        patch_size=patch_size,
        target_h=H, target_w=W,
    )

    # Standard CC pipeline on plane boundaries
    instances = []
    for cls in sorted(thing_ids):
        cls_mask = semantic == cls
        if cls_mask.sum() < min_area:
            continue

        split_mask = cls_mask & ~boundary
        labeled, n_cc = ndimage.label(split_mask)

        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            area = float(cc_mask.sum())
            if area >= min_area:
                instances.append((cc_mask, cls, area))

    return dilation_reclaim(
        instances, semantic, thing_ids,
        min_area=min_area, dilation_iters=dilation_iters,
    )
