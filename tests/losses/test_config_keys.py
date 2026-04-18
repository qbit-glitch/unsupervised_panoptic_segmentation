"""Verify CUPS config exposes SEM_SEG_HEAD aux-loss weight keys."""
from __future__ import annotations

from cups.config import get_default_config


def test_sem_seg_head_aux_weights_exist() -> None:
    cfg = get_default_config()
    head = cfg.MODEL.SEM_SEG_HEAD

    # P1 -- LoCE
    assert head.LOVASZ_WEIGHT == 0.0
    assert head.BOUNDARY_WEIGHT == 0.0
    assert head.BOUNDARY_DILATE_PX == 3
    assert head.BOUNDARY_CE_MULT == 2.0

    # P2 -- FeatMirror
    assert head.STEGO_WEIGHT == 0.0
    assert head.STEGO_TEMPERATURE == 0.1
    assert head.STEGO_KNN_K == 7
    assert head.STEGO_FEATURE_SOURCE == "fpn_p2"

    # P3 -- DGLR
    assert head.DEPTH_SMOOTH_WEIGHT == 0.0
    assert head.DEPTH_SMOOTH_ALPHA == 10.0

    # P4 -- DAff
    assert head.GATED_CRF_WEIGHT == 0.0
    assert head.GATED_CRF_KERNEL == 5
    assert head.GATED_CRF_RGB_SIGMA == 0.1
    assert head.NECO_WEIGHT == 0.0
    assert head.NECO_K == 5
