import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from mbps_pytorch.premise_check_geometry_affinity import geometry as _scene_geometry
from mbps_pytorch.ga_uniap.config import Phase0Config


def pool_to_grid(arr: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Area-average pool (H,W[,C]) -> (gh,gw[,C]). H,W must be multiples of gh,gw."""
    H, W = arr.shape[:2]
    assert H % gh == 0 and W % gw == 0, f"{(H, W)} not divisible by {(gh, gw)}"
    sh, sw = H // gh, W // gw
    if arr.ndim == 2:
        return arr.reshape(gh, sh, gw, sw).mean(axis=(1, 3))
    C = arr.shape[2]
    return arr.reshape(gh, sh, gw, sw, C).mean(axis=(1, 3))


def grid_geometry(stem: str, city: str, cfg: Phase0Config
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(normal_grid[gh,gw,3] unit, height_grid[gh,gw]) from DepthPro, or None."""
    g = _scene_geometry(stem, city, cfg.split, source="mono")
    if g is None:
        return None
    normal = pool_to_grid(g["normal"].astype(np.float32), cfg.grid_h, cfg.grid_w)
    height = pool_to_grid(g["height"].astype(np.float32), cfg.grid_h, cfg.grid_w)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (nrm + 1e-9)
    return normal.astype(np.float32), height.astype(np.float32)
