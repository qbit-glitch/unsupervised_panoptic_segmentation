import torch

from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent


def test_conv_student_semantic_shape():
    m = ConvStudent(num_classes=133, pretrained=False).eval()  # offline-safe
    y = m(torch.rand(1, 3, 512, 512))
    assert y.shape[0] == 1 and y.shape[1] == 133
    assert y.shape[2] == 128 and y.shape[3] == 128   # 1/4 resolution
