"""G6 dense-CRF post-processing wrapper (pydensecrf)."""
from __future__ import annotations

import numpy as np
import torch

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_DCRF = True
except ImportError:       # graceful fallback
    _HAS_DCRF = False

__all__ = ["dense_crf_refine"]


def dense_crf_refine(
    image: torch.Tensor, probs: torch.Tensor, num_iter: int = 5, bi_w: float = 4.0, pos_w: float = 3.0,
) -> torch.Tensor:
    """Refine softmax probs using dense-CRF.

    Args:
        image: uint8 (H, W, 3) or float (3, H, W) in [0, 1].
        probs: (K, H, W) softmax probabilities.
        num_iter: number of CRF iterations.
    """
    if not _HAS_DCRF:
        return probs
    img_np = image.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    probs_np = probs.detach().cpu().numpy().astype(np.float32)
    # Sanitize tiny values for numerical stability.
    probs_np = np.maximum(probs_np, 1e-8)
    K, H, W = probs_np.shape
    unary = unary_from_softmax(probs_np.reshape(K, -1))
    d = dcrf.DenseCRF2D(W, H, K)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=pos_w)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.ascontiguousarray(img_np), compat=bi_w)
    Q = d.inference(num_iter)
    out = np.asarray(Q).reshape(K, H, W)
    return torch.from_numpy(out).to(probs.device)
