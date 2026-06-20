"""ONNX export + PyTorch-vs-onnxruntime parity + CPU latency, for both students."""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EoMTExportWrapper(nn.Module):
    """Return only the final-layer (mask_logits, class_logits) tensors.

    EoMT.forward returns Python lists of varying length; ONNX needs fixed tensor
    outputs. Build the wrapped EoMT with ``masked_attn=False`` so the forward is a
    single plain pass.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        mask_l, cls_l = self.model(x)
        return mask_l[-1], cls_l[-1]


def export(model: nn.Module, path: str, img: int = 640) -> None:
    model.eval()
    dummy = torch.rand(1, 3, img, img)
    # dynamo=False -> legacy TorchScript exporter (no onnxscript dep); fixed shape.
    torch.onnx.export(model, dummy, path, input_names=["image"],
                      output_names=["out"], opset_version=17, dynamo=False)
    logger.info("exported %s", path)


def _ref_first(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    return np.asarray(out)


def parity(model: nn.Module, path: str, img: int = 640) -> float:
    import onnxruntime as ort

    model.eval()
    x = torch.rand(1, 3, img, img)
    ref = _ref_first(model, x)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    return float(np.abs(ref - out).max())


def latency(path: str, img: int = 640, runs: int = 20) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img, img).astype(np.float32)
    sess.run(None, {"image": x})            # warmup
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"image": x})
    return (time.perf_counter() - t0) / runs * 1000.0


if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO)
    from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent

    out_dir = tempfile.mkdtemp()
    conv = ConvStudent(num_classes=133, pretrained=False).eval()
    cp = os.path.join(out_dir, "conv.onnx")
    export(conv, cp, img=512)
    print("CONV  parity=%.2e  latency=%.1f ms (CPU, 512)"
          % (parity(conv, cp, 512), latency(cp, 512, runs=10)))

    try:
        from mbps_pytorch.mobile_panoptic_sup.smoke_eomt import build
        eomt = EoMTExportWrapper(build(img=512, masked_attn=False).eval()).eval()
        ep = os.path.join(out_dir, "eomt.onnx")
        export(eomt, ep, img=512)
        print("EoMT  parity=%.2e  latency=%.1f ms (CPU, 512)"
              % (parity(eomt, ep, 512), latency(ep, 512, runs=10)))
    except Exception as exc:  # noqa: BLE001  (bench: report, don't crash)
        print("EoMT export FAILED:", repr(exc)[:300])
