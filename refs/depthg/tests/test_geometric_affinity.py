import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry_features import geometric_affinity  # noqa: E402


def test_normal_cosine_extremes():
    up = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    side = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    g_up = torch.cat([torch.zeros(1, 1, 1, 1), up], 1)
    g_side = torch.cat([torch.zeros(1, 1, 1, 1), side], 1)
    same = geometric_affinity(g_up, g_up, "normal", 0.4, 0.6)[0, 0, 0, 0, 0]
    orth = geometric_affinity(g_up, g_side, "normal", 0.4, 0.6)[0, 0, 0, 0, 0]
    assert torch.isclose(same, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(orth, torch.tensor(0.0), atol=1e-5)


def test_both_is_weighted_sum_of_height_and_normal():
    g = torch.randn(1, 4, 2, 2)
    g[:, 1:] = torch.nn.functional.normalize(g[:, 1:], dim=1)
    h = geometric_affinity(g, g, "height", 0.4, 0.6)
    nrm = geometric_affinity(g, g, "normal", 0.4, 0.6)
    both = geometric_affinity(g, g, "both", 0.4, 0.6)
    assert torch.allclose(both, 0.6 * nrm + 0.4 * h, atol=1e-5)
