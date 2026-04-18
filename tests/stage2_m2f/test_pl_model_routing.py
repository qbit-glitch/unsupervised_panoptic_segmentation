from __future__ import annotations

import pytest
from yacs.config import CfgNode

from cups.config import get_default_config


def test_build_model_pseudo_routes_mask2former(monkeypatch) -> None:
    from cups import pl_model_pseudo as plm

    called = {"built": False}

    def fake_build(config: CfgNode, *args, **kwargs):
        called["built"] = True
        import torch.nn as nn
        return nn.Identity()

    monkeypatch.setattr(plm, "_build_mask2former_model", fake_build)
    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg._NUM_STUFF_CLASSES = 12
    cfg._NUM_THING_CLASSES = 8
    cfg.freeze()
    model = plm.build_model_pseudo(cfg)
    assert called["built"] is True
