import torch

from mbps_pytorch.mobile_panoptic_sup import smoke_eomt


def test_eomt_forward_shapes_cpu():
    model = smoke_eomt.build(img=224, num_classes=133, num_q=100)  # 224 = 14*16
    mask_l, cls_l = model(torch.rand(1, 3, 224, 224))
    assert cls_l[-1].shape == (1, 100, 134)        # num_classes + 1
    assert mask_l[-1].shape[1] == 100
