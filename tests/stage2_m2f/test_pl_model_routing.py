from __future__ import annotations

import pytest
from yacs.config import CfgNode

from cups.config import get_default_config


def test_build_model_pseudo_routes_mask2former(monkeypatch) -> None:
    from cups import pl_model_pseudo as plm

    called = {"built": False, "num_stuff": None, "num_thing": None}

    def fake_build(config: CfgNode, *args, **kwargs):
        called["built"] = True
        called["num_stuff"] = kwargs.get("num_stuff_classes")
        called["num_thing"] = kwargs.get("num_thing_classes")
        import torch.nn as nn
        return nn.Identity()

    monkeypatch.setattr(plm, "_build_mask2former_model", fake_build)
    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg.freeze()
    # Counts now flow from the dataloader tuples (prod pathway), not
    # cfg._NUM_* runtime-injected attrs (the broken legacy path).
    stuff_pseudo = tuple(range(12))
    thing_pseudo = tuple(range(12, 20))
    model = plm.build_model_pseudo(
        cfg,
        thing_classes=set(range(11, 19)),
        stuff_classes=set(range(11)),
        thing_pseudo_classes=thing_pseudo,
        stuff_pseudo_classes=stuff_pseudo,
    )
    assert called["built"] is True
    assert called["num_stuff"] == 12
    assert called["num_thing"] == 8


def test_build_model_pseudo_m2f_requires_pseudo_classes() -> None:
    """M2F branch must raise if dataloader tuples are missing."""
    from cups import pl_model_pseudo as plm

    cfg = get_default_config()
    cfg.defrost()
    cfg.MODEL.META_ARCH = "Mask2FormerPanoptic"
    cfg.freeze()
    with pytest.raises(ValueError, match="thing_pseudo_classes"):
        plm.build_model_pseudo(cfg)
