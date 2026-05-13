"""Geometric feature computation from monocular depth maps.

Computes surface normals, Sobel gradients, and sinusoidal depth encoding
from precomputed depth maps. All operations work at patch level (32x64).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_depth_gradients(depth_2d: torch.Tensor) -> torch.Tensor:
    """Compute Sobel depth gradients.

    Args:
        depth_2d: Depth map of shape (H, W) or (B, H, W).

    Returns:
        Gradients of shape (..., H, W, 2) — [dD/dx, dD/dy].
    """
    squeeze = False
    if depth_2d.ndim == 2:
        depth_2d = depth_2d.unsqueeze(0)
        squeeze = True

    # (B, 1, H, W) for conv2d
    d = depth_2d.unsqueeze(1).float()

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=d.device,
    ).reshape(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=d.device,
    ).reshape(1, 1, 3, 3) / 8.0

    grad_x = F.conv2d(d, sobel_x, padding=1)  # (B, 1, H, W)
    grad_y = F.conv2d(d, sobel_y, padding=1)  # (B, 1, H, W)

    # Stack to (B, H, W, 2)
    grads = torch.cat([grad_x, grad_y], dim=1).permute(0, 2, 3, 1)

    if squeeze:
        grads = grads.squeeze(0)
    return grads


def compute_surface_normals(depth_2d: torch.Tensor) -> torch.Tensor:
    """Compute surface normals from depth map.

    Normal = (-dD/dx, -dD/dy, 1) / ||(-dD/dx, -dD/dy, 1)||

    Args:
        depth_2d: Depth map of shape (H, W) or (B, H, W).

    Returns:
        Normals of shape (..., H, W, 3) — unit vectors.
    """
    grads = compute_depth_gradients(depth_2d)  # (..., H, W, 2)

    # Normal = (-dD/dx, -dD/dy, 1)
    ones = torch.ones_like(grads[..., :1])
    normals = torch.cat([-grads, ones], dim=-1)  # (..., H, W, 3)

    # Normalize to unit length
    normals = F.normalize(normals, dim=-1, eps=1e-6)
    return normals


def sinusoidal_depth_encoding(
    depth: torch.Tensor,
    freq_bands: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> torch.Tensor:
    """Sinusoidal positional encoding of depth values.

    Args:
        depth: Depth values of shape (...,) in [0, 1].
        freq_bands: Frequency bands for encoding.

    Returns:
        Encoding of shape (..., 1 + 2*len(freq_bands)).
        First channel is raw depth, then [sin, cos] pairs.
    """
    encodings = [depth.unsqueeze(-1)]  # raw depth

    for freq in freq_bands:
        encodings.append(torch.sin(freq * torch.pi * depth).unsqueeze(-1))
        encodings.append(torch.cos(freq * torch.pi * depth).unsqueeze(-1))

    return torch.cat(encodings, dim=-1)  # (..., 13)


def compute_geometric_features(
    depth_2d: torch.Tensor,
    target_h: int = 32,
    target_w: int = 64,
) -> torch.Tensor:
    """Compute all geometric features from a depth map at patch level.

    Computes surface normals (3), depth gradients (2), and sinusoidal
    depth encoding (13) at the target spatial resolution.

    Args:
        depth_2d: Depth map of shape (H, W) in [0, 1].
        target_h: Target patch grid height (default 32).
        target_w: Target patch grid width (default 64).

    Returns:
        Geometric features of shape (target_h * target_w, 18).
        Channels: [depth_sin_cos (13), grad_x, grad_y (2), nx, ny, nz (3)].
    """
    # Downsample depth to patch level
    depth_patch = F.interpolate(
        depth_2d.float().unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)  # (target_h, target_w)

    # Sinusoidal encoding of depth — (H, W, 13)
    depth_enc = sinusoidal_depth_encoding(depth_patch)

    # Depth gradients at patch level — (H, W, 2)
    grads = compute_depth_gradients(depth_patch)

    # Surface normals at patch level — (H, W, 3)
    normals = compute_surface_normals(depth_patch)

    # Concatenate: (H, W, 18)
    geo = torch.cat([depth_enc, grads, normals], dim=-1)

    # Flatten to (N, 18)
    return geo.reshape(-1, 18)
