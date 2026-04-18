from __future__ import annotations

import torch
from yacs.config import CfgNode

from cups.model.model_mask2former import build_mask2former_vitb


def _minimal_config(num_stuff: int = 12, num_thing: int = 8) -> CfgNode:
    c = CfgNode()
    c.MODEL = CfgNode()
    c.MODEL.META_ARCH = "Mask2FormerPanoptic"
    c.MODEL.BACKBONE_TYPE = "dinov3_vitb"
    c.MODEL.DINOV2_FREEZE = True
    c.MODEL.TTA_SCALES = (1.0,)
    c.MODEL.MASK2FORMER = CfgNode()
    c.MODEL.MASK2FORMER.NUM_QUERIES = 10
    c.MODEL.MASK2FORMER.QUERY_POOL = "standard"
    c.MODEL.MASK2FORMER.NUM_DECODER_LAYERS = 2
    c.MODEL.MASK2FORMER.HIDDEN_DIM = 128
    c.MODEL.MASK2FORMER.NUM_HEADS = 4
    c.MODEL.MASK2FORMER.MASK_WEIGHT = 5.0
    c.MODEL.MASK2FORMER.DICE_WEIGHT = 5.0
    c.MODEL.MASK2FORMER.CLASS_WEIGHT = 2.0
    c.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT = 0.1
    c.MODEL.MASK2FORMER.NUM_POINTS = 64
    c.MODEL.MASK2FORMER.OBJECT_MASK_THRESHOLD = 0.4
    c.MODEL.MASK2FORMER.OVERLAP_THRESHOLD = 0.8
    c.MODEL.MASK2FORMER.PYRAMID_CHANNELS = 128
    c.MODEL.MASK2FORMER.ADAPTER_BLOCKS = 1
    c.MODEL.MASK2FORMER.ADAPTER_EMBED_DIM = 768
    c.MODEL.MASK2FORMER.PIXEL_DECODER_LAYERS = 2
    c.MODEL.MASK2FORMER.DROPPATH = 0.0
    c.MODEL.MASK2FORMER.QUERIES_STUFF = 150
    c.MODEL.MASK2FORMER.QUERIES_THING = 50
    # N3/N4 aux-loss weights default to 0.0 (off); tests that exercise the
    # hooks override these explicitly.
    c.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.0
    c.MODEL.MASK2FORMER.XQUERY_TEMPERATURE = 0.1
    c.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.0
    c.MODEL.MASK2FORMER.QUERY_CONSISTENCY_TEMPERATURE = 0.1
    c.DATA = CfgNode()
    c.DATA.NUM_PSEUDO_CLASSES = num_stuff + num_thing
    # Marker fields consumed by the builder:
    c._NUM_STUFF_CLASSES = num_stuff
    c._NUM_THING_CLASSES = num_thing
    return c


class _Dummy(torch.nn.Module):
    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _install_fake_backbone(monkeypatch) -> None:
    """Patch the dino-loading helper with a config-driven dummy.

    Critically, the dummy MUST respect ``cfg.MODEL.DINOV2_FREEZE`` so that
    flipping the flag actually exercises the freeze branch of the builder.
    """
    def _fake_build(cfg):
        return _Dummy(freeze=bool(cfg.MODEL.DINOV2_FREEZE))

    import cups.model.model_mask2former as mm

    monkeypatch.setattr(mm, "_build_dinov3_backbone", _fake_build)


def test_builder_returns_frozen_backbone(monkeypatch) -> None:
    _install_fake_backbone(monkeypatch)
    cfg = _minimal_config()
    cfg.MODEL.DINOV2_FREEZE = True
    model = build_mask2former_vitb(cfg)
    # ViTAdapter wraps the backbone as `self.backbone`; hence the double attr.
    for p in model.backbone.backbone.parameters():
        assert p.requires_grad is False


def test_builder_propagates_freeze_flag(monkeypatch) -> None:
    """Regression guard: the builder must forward cfg.MODEL.DINOV2_FREEZE.

    Note: the ViTAdapter wrapper re-freezes its backbone unconditionally
    (by design, because Phase 0 treats DINOv3 as frozen). We therefore
    validate propagation at the _build_dinov3_backbone call site rather
    than at post-adapter params, so a regression that drops the flag is
    still caught.
    """
    captured: dict = {}

    def _fake_build(cfg):
        captured["freeze"] = bool(cfg.MODEL.DINOV2_FREEZE)
        return _Dummy(freeze=bool(cfg.MODEL.DINOV2_FREEZE))

    import cups.model.model_mask2former as mm

    monkeypatch.setattr(mm, "_build_dinov3_backbone", _fake_build)

    cfg = _minimal_config()
    cfg.MODEL.DINOV2_FREEZE = False
    build_mask2former_vitb(cfg)
    assert captured["freeze"] is False

    captured.clear()
    cfg.MODEL.DINOV2_FREEZE = True
    build_mask2former_vitb(cfg)
    assert captured["freeze"] is True


def test_builder_wires_no_hooks_when_weights_zero(monkeypatch) -> None:
    _install_fake_backbone(monkeypatch)
    cfg = _minimal_config()
    cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.0
    cfg.MODEL.MASK2FORMER.XQUERY_TEMPERATURE = 0.1
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.0
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_TEMPERATURE = 0.1
    model = build_mask2former_vitb(cfg)
    assert "xquery" not in model.aux_loss_hooks
    assert "query_consistency" not in model.aux_loss_hooks
    assert not model.aux_loss_hooks.get("return_query_embeds", False)


def test_builder_wires_xquery_hook_when_weight_positive(monkeypatch, tiny_gt_panoptic) -> None:
    _install_fake_backbone(monkeypatch)
    cfg = _minimal_config()
    cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.1
    cfg.MODEL.MASK2FORMER.XQUERY_TEMPERATURE = 0.1
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.0
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_TEMPERATURE = 0.1
    model = build_mask2former_vitb(cfg).train()
    assert "xquery" in model.aux_loss_hooks
    assert callable(model.aux_loss_hooks["xquery"])
    assert model.aux_loss_hooks.get("return_query_embeds") is True

    torch.manual_seed(0)
    batch = []
    for gt in tiny_gt_panoptic:
        batch.append({
            "image": torch.randn(3, 64, 128),
            "_m2f_targets": {"labels": gt["labels"], "masks": gt["masks"]},
        })
    loss_dict = model(batch)
    assert "loss_xquery" in loss_dict
    assert torch.isfinite(loss_dict["loss_xquery"])


def test_builder_wires_query_consistency_hook_when_weight_positive(monkeypatch, tiny_gt_panoptic) -> None:
    _install_fake_backbone(monkeypatch)
    cfg = _minimal_config()
    cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT = 0.0
    cfg.MODEL.MASK2FORMER.XQUERY_TEMPERATURE = 0.1
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT = 0.1
    cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_TEMPERATURE = 0.1
    model = build_mask2former_vitb(cfg).train()
    assert "query_consistency" in model.aux_loss_hooks
    assert callable(model.aux_loss_hooks["query_consistency"])
    assert model.aux_loss_hooks.get("return_query_embeds") is True

    torch.manual_seed(0)
    batch = []
    for gt in tiny_gt_panoptic:
        batch.append({
            "image": torch.randn(3, 64, 128),
            "_m2f_targets": {"labels": gt["labels"], "masks": gt["masks"]},
        })
    loss_dict = model(batch)
    assert "loss_query_consistency" in loss_dict
    assert torch.isfinite(loss_dict["loss_query_consistency"])
