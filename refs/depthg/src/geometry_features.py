"""Gravity-aligned geometry features for GA-DepthG.

Turns a monocular (DepthPro) inverse-depth map into a per-pixel geometric descriptor
``g = [ĥ, n_x, n_y, n_z]`` — per-scene-standardized height-above-ground + unit surface normal —
and provides the geometric-affinity op that replaces DepthG's scalar depth-correlation.

Geometry primitives (`back_project`, `fit_ground_plane`, `surface_normals`) are the exact functions
validated in ``mbps_pytorch/premise_check_geometry_affinity.py`` (premise check, 150-image run).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


# ----------------------------------------------------------------- geometry (numpy, validated)
def back_project(depth_inv: np.ndarray, fx, fy, u0, v0, eps: float = 2e-3):
    """Normalized inverse-depth -> 3D point per pixel (relative units). (H,W,3) + valid mask."""
    H, W = depth_inv.shape
    Z = 1.0 / (depth_inv + eps)
    valid = depth_inv > 0.02  # drop sky / far-unreliable
    Zc = np.clip(Z, 0, np.percentile(Z[valid], 98))
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X = (us - u0) * Zc / fx
    Y = (vs - v0) * Zc / fy
    return np.stack([X, Y, Zc], -1).astype(np.float32), valid


def fit_ground_plane(pts: np.ndarray, region: np.ndarray, iters: int = 4):
    """Robust (IRLS) total-least-squares plane fit on the lower-image region. Returns (n, d) or None."""
    P = pts[region]
    if len(P) < 200:
        return None
    inl = np.ones(len(P), bool)
    n = np.array([0, 1, 0.0])
    d = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # robust IRLS on degenerate pts
        for _ in range(iters):
            Q = P[inl]
            if len(Q) < 3:
                break
            c = Q.mean(0)
            _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
            n2, d2 = Vt[-1], -float(Vt[-1] @ c)
            if not np.isfinite(n2).all() or not np.isfinite(d2):
                break                                        # keep last finite (n, d)
            n, d = n2, d2
            res = np.abs(P @ n + d)
            mad = np.median(np.abs(res - np.median(res))) + 1e-9
            inl = res < (np.median(res) + 3 * 1.4826 * mad)
    return n, d


def surface_normals(pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel unit surface normal from the (smoothed) 3D point grid. (H,W,3)."""
    Z = pts[..., 2]
    num = ndimage.uniform_filter(np.where(valid, Z, 0.0), 5)
    den = ndimage.uniform_filter(valid.astype(np.float32), 5) + 1e-6
    Zs = num / den
    P = pts.copy()
    P[..., 2] = Zs
    tu = np.stack([np.gradient(P[..., k], axis=1) for k in range(3)], -1)  # d/col
    tv = np.stack([np.gradient(P[..., k], axis=0) for k in range(3)], -1)  # d/row
    n = np.cross(tu, tv)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    return n.astype(np.float32)


def compute_geometry(depth_inv: np.ndarray, fx, fy, u0, v0, cam_h) -> np.ndarray:
    """DepthPro inverse-depth -> (4,H,W) float32 [ĥ, n_x, n_y, n_z]; ĥ per-scene-standardized."""
    pts, valid = back_project(depth_inv, fx, fy, u0, v0)
    H, W, _ = pts.shape
    region = np.zeros((H, W), bool)
    region[int(H * 0.62):, int(W * 0.2):int(W * 0.8)] = True
    region &= valid
    plane = fit_ground_plane(pts, region)
    if plane is None:
        n_g, d, scale = np.array([0, 1, 0.0]), 0.0, 1.0
    else:
        n_g, d = plane
        scale = cam_h / (abs(d) + 1e-9)                       # recover metric scale via camera height
    height = (pts @ n_g + d) * scale
    if height[:int(H * 0.4)][valid[:int(H * 0.4)]].mean() < 0:  # orient: objects up, ground ~0
        height = -height
        n_g = -n_g
    normal = surface_normals(pts * scale, valid)             # (H,W,3) unit
    med = float(np.median(height[valid])) if valid.any() else 0.0
    s90 = float(np.percentile(np.abs(height[valid] - med), 90)) + 1e-6 if valid.any() else 1.0
    h_hat = (height - med) / s90                             # per-scene standardized, scale-free
    g = np.concatenate([h_hat[None], normal.transpose(2, 0, 1)], 0).astype(np.float32)
    g[:, ~valid] = 0.0
    return g


# ----------------------------------------------------------------- affinity (torch)
def geometric_affinity(g1: torch.Tensor, g2: torch.Tensor, mode: str, w_h: float, w_n: float):
    """Pairwise geometric affinity. g: (n,4,h,w) with ch 0 = ĥ, ch 1:4 = unit normal.

    Returns (n,h,w,h,w). mode: 'height' (ĥ_i·ĥ_j) | 'normal' (n_i·n_j cosine) | 'both' (weighted).
    """
    h1, n1 = g1[:, :1], g1[:, 1:]
    h2, n2 = g2[:, :1], g2[:, 1:]
    height_corr = torch.einsum("nchw,ncij->nhwij", h1, h2)
    normal_corr = torch.einsum("nchw,ncij->nhwij", n1, n2)
    if mode == "height":
        return height_corr
    if mode == "normal":
        return normal_corr
    return w_n * normal_corr + w_h * height_corr             # "both"
