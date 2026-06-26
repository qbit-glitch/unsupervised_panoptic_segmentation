import numpy as np

from mbps_pytorch.ga_uniap.config import Phase0Config
from mbps_pytorch.ga_uniap.preflight import list_val_stems
from mbps_pytorch.ga_uniap.geometry import grid_geometry, pool_to_grid


def test_preflight_finds_val_stems():
    cfg = Phase0Config(n_images=10)
    stems = list_val_stems(cfg)
    assert len(stems) >= 10, f"expected >=10 usable val stems, got {len(stems)}"
    stem, city = stems[0]
    assert city in {"frankfurt", "lindau", "munster"}
    assert isinstance(stem, str) and stem.startswith(city)


def test_pool_to_grid_shapes_and_average():
    arr = np.ones((512, 1024, 3), np.float32) * 2.0
    out = pool_to_grid(arr, 32, 64)
    assert out.shape == (32, 64, 3)
    assert np.allclose(out, 2.0)


def test_grid_geometry_on_real_stem():
    cfg = Phase0Config(n_images=5)
    stem, city = list_val_stems(cfg)[0]
    res = grid_geometry(stem, city, cfg)
    assert res is not None, "geometry returned None on a real stem"
    normal, height = res
    assert normal.shape == (32, 64, 3)
    assert height.shape == (32, 64)
    norms = np.linalg.norm(normal, axis=-1)
    assert np.all(np.isfinite(normal)) and np.all(np.isfinite(height))
    # normals approximately unit (pooling shrinks them slightly; re-normalized in module)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)
