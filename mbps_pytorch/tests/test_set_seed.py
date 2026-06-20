import importlib

import numpy as np
import torch


def test_set_seed_is_deterministic():
    trn = importlib.import_module("mbps_pytorch.train_refine_net")
    assert hasattr(trn, "set_seed"), "set_seed not defined"
    trn.set_seed(42)
    a_np, a_torch = np.random.rand(3), torch.rand(3)
    trn.set_seed(42)
    b_np, b_torch = np.random.rand(3), torch.rand(3)
    assert np.allclose(a_np, b_np) and torch.allclose(a_torch, b_torch)
    assert torch.backends.cudnn.deterministic is True
