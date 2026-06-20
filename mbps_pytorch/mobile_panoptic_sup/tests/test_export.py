import os
import tempfile

from mbps_pytorch.mobile_panoptic_sup import export_onnx
from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent


def test_conv_export_parity_and_latency():
    m = ConvStudent(num_classes=133, pretrained=False).eval()
    path = os.path.join(tempfile.mkdtemp(), "conv.onnx")
    export_onnx.export(m, path, img=256)
    assert export_onnx.parity(m, path, img=256) < 1e-3
    assert export_onnx.latency(path, img=256, runs=3) > 0.0
