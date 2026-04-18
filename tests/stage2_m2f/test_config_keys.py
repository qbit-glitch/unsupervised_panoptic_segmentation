from __future__ import annotations

from cups.config import get_default_config


def test_mask2former_config_keys_exist() -> None:
    cfg = get_default_config()
    # Meta-arch switch.
    assert cfg.MODEL.META_ARCH == "Cascade"
    # M2F keys with defaults.
    assert cfg.MODEL.MASK2FORMER.NUM_QUERIES == 100
    assert cfg.MODEL.MASK2FORMER.QUERY_POOL == "standard"
    assert cfg.MODEL.MASK2FORMER.NUM_DECODER_LAYERS == 9
    assert cfg.MODEL.MASK2FORMER.XQUERY_WEIGHT == 0.0
    assert cfg.MODEL.MASK2FORMER.QUERY_CONSISTENCY_WEIGHT == 0.0
    # G-lever keys.
    assert cfg.MODEL.EMA.ENABLED is False
    assert cfg.MODEL.SWA.ENABLED is False
    assert cfg.AUGMENTATION.LSJ.ENABLED is False
    assert cfg.AUGMENTATION.COLOR_JITTER.ENABLED is False
    assert cfg.VALIDATION.USE_DENSE_CRF is False
