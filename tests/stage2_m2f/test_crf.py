from __future__ import annotations

import pytest

pytest.importorskip("pydensecrf", reason="dense-CRF requires pydensecrf")

import numpy as np
import torch

from cups.model.modeling.mask2former.dense_crf import dense_crf_refine


def test_dense_crf_refines_known_mask() -> None:
    # Input: 10x10 noisy probability field with a clean GT blob in the middle.
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.2, 0.4, size=(3, 10, 10)).astype(np.float32)
    probs[0, 3:7, 3:7] += 0.5        # class 0 dominant in center
    probs = probs / probs.sum(axis=0, keepdims=True)
    img = rng.integers(0, 255, size=(10, 10, 3), dtype=np.uint8)
    probs_t = torch.from_numpy(probs)
    img_t = torch.from_numpy(img)
    out = dense_crf_refine(img_t, probs_t, num_iter=3)
    assert out.shape == probs_t.shape
    # Refined argmax should have class-0 in the center block.
    lbl = out.argmax(0)
    assert lbl[5, 5].item() == 0
