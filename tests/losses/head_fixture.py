"""Test fixture for :class:`CustomSemSegFPNHead` aux-loss aggregation.

Instantiating the head via ``from_config`` requires a full CUPS-flavoured
config tree, an FPN backbone, and Detectron2 wiring.  For unit-testing the
loss aggregation logic we only need a minimal object that exposes the
attributes used inside ``losses()``. This fixture builds such an object by
allocating the class via ``__new__`` and hand-populating the fields.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch

from cups.model.modeling.roi_heads.semantic_seg import CustomSemSegFPNHead


_DEFAULT_AUX_PARAMS: Dict[str, object] = {
    "boundary_dilate_px": 3,
    "boundary_ce_mult": 2.0,
    "stego_temperature": 0.1,
    "stego_knn_k": 7,
    "stego_feature_source": "fpn_p2",
    "depth_smooth_alpha": 10.0,
    "gated_crf_kernel": 5,
    "gated_crf_rgb_sigma": 0.1,
    "neco_k": 5,
}


def make_head_with_aux(
    num_classes: int = 19,
    weights: Optional[Dict[str, float]] = None,
    aux_params: Optional[Dict[str, object]] = None,
    common_stride: int = 4,
    loss_weight: float = 1.0,
    ignore_value: int = 255,
    class_weight: Optional[Iterable[float]] = None,
) -> CustomSemSegFPNHead:
    """Build a minimal ``CustomSemSegFPNHead`` instance for loss-only tests.

    The returned object is a real :class:`CustomSemSegFPNHead` but skips the
    Detectron2 FPN construction path. Call ``head.losses(logits, targets,
    ctx=...)`` directly; do not invoke ``forward``/``layers`` — they require
    real FPN inputs that this fixture does not set up.
    """
    head = CustomSemSegFPNHead.__new__(CustomSemSegFPNHead)
    torch.nn.Module.__init__(head)

    head.ignore_value = ignore_value
    head.common_stride = common_stride
    head.loss_weight = loss_weight
    head.class_weight = list(class_weight) if class_weight is not None else None
    head.stuff_kd_weight = 0.0
    head.kd_temperature = 2.0

    head.aux_weights = {
        "lovasz": 0.0,
        "boundary": 0.0,
        "stego": 0.0,
        "depth_smooth": 0.0,
        "gated_crf": 0.0,
        "neco": 0.0,
    }
    if weights:
        # Accept both config-style keys (LOVASZ_WEIGHT) and internal keys
        # (lovasz) so tests can pass whichever is convenient.
        _alias = {
            "LOVASZ_WEIGHT": "lovasz",
            "BOUNDARY_WEIGHT": "boundary",
            "STEGO_WEIGHT": "stego",
            "DEPTH_SMOOTH_WEIGHT": "depth_smooth",
            "GATED_CRF_WEIGHT": "gated_crf",
            "NECO_WEIGHT": "neco",
        }
        for k, v in weights.items():
            head.aux_weights[_alias.get(k, k)] = float(v)

    head.aux_params = {**_DEFAULT_AUX_PARAMS, **(aux_params or {})}

    head._aux_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]] = {}
    if any(w > 0.0 for w in head.aux_weights.values()):
        from cups.losses import build_aux_losses

        head._aux_fns = build_aux_losses()

    # Minimal attributes used by ``num_classes`` consumers. We skip building
    # ``scale_heads`` / ``predictor`` because tests do not call ``layers``.
    head.num_classes = num_classes
    return head


__all__ = ["make_head_with_aux"]
