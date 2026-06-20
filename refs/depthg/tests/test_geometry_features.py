import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry_features import compute_geometry  # noqa: E402


def test_compute_geometry_shape_unit_normals_standardized_height():
    H, W = 64, 128
    # synthetic receding surface: inverse-depth in (0,1], smoothly varying over rows
    inv = np.linspace(0.9, 0.05, H)[:, None].repeat(W, 1).astype(np.float32)
    g = compute_geometry(inv, fx=500.0, fy=500.0, u0=W / 2, v0=H / 2, cam_h=1.22)

    assert g.shape == (4, H, W)
    n = g[1:4]
    assert np.median(np.linalg.norm(n, axis=0)) > 0.99      # normals are unit
    assert abs(np.median(g[0])) < 0.5                        # height standardized ~0 median
