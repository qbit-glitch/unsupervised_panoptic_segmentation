import importlib
import inspect

import torch

from mbps_pytorch.refine_net import DepthGuidedUNet


def test_unet_accepts_1024d_dinov3():
    m = DepthGuidedUNet(num_classes=27, feature_dim=1024, block_type="attention",
                        num_decoder_stages=2, gradient_checkpointing=False).eval()
    out = m(torch.randn(1, 1024, 32, 64), torch.rand(1, 1, 32, 64),
            torch.randn(1, 2, 32, 64), depth_full=torch.rand(1, 1, 512, 1024))
    assert out.shape == (1, 27, 128, 256)


def test_dataset_has_feature_subdir_param():
    trn = importlib.import_module("mbps_pytorch.train_refine_net")
    params = inspect.signature(trn.PseudoLabelDataset.__init__).parameters
    assert "feature_subdir" in params and "depth_subdir" in params
