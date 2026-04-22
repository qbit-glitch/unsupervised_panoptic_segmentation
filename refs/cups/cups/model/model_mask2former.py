"""Builder for Mask2Former + ViT-Adapter with frozen DINOv3 ViT-B/16."""
from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from yacs.config import CfgNode

from cups.losses.query_consistency import query_consistency_loss
from cups.losses.xquery import xquery_loss
from cups.model.backbone_dinov3_vit import DINOv3ViTBackbone
from cups.model.modeling.mask2former.masked_attention_decoder import MaskedAttentionDecoder
from cups.model.modeling.mask2former.matcher import HungarianMatcher
from cups.model.modeling.mask2former.msdeform_pixel_decoder import MSDeformAttnPixelDecoder
from cups.model.modeling.mask2former.query_pool import build_query_pool
from cups.model.modeling.mask2former.set_criterion import SetCriterion
from cups.model.modeling.mask2former.vit_adapter import ViTAdapter
from cups.model.modeling.meta_arch.mask2former_panoptic import Mask2FormerPanoptic

log = logging.getLogger(__name__)

__all__ = ["build_mask2former_vitb"]


def _build_dinov3_backbone(cfg: CfgNode) -> nn.Module:
    """Wrapped for monkey-patching in tests.

    Reads LoRA config via ``getattr`` so the M2F path stays symmetric with
    the cascade builder in ``model_vitb.py``. Checks ``MODEL.LORA_CONFIG``
    first (legacy), then ``MODEL.LORA`` (canonical config key used by all
    existing stage-2/3 YAMLs) so DoRA is not silently dropped on M2F.
    """
    lora_config = getattr(cfg.MODEL, "LORA_CONFIG", None)
    if lora_config is None and hasattr(cfg.MODEL, "LORA"):
        lora_section = cfg.MODEL.LORA
        if getattr(lora_section, "ENABLED", False):
            lora_config = {
                "VARIANT": getattr(lora_section, "VARIANT", "dora"),
                "RANK": getattr(lora_section, "RANK", 4),
                "ALPHA": getattr(lora_section, "ALPHA", 4.0),
                "DROPOUT": getattr(lora_section, "DROPOUT", 0.05),
                "LATE_BLOCK_START": getattr(lora_section, "LATE_BLOCK_START", 6),
            }
    backbone = DINOv3ViTBackbone(
        freeze=cfg.MODEL.DINOV2_FREEZE,
        lora_config=lora_config,
    )
    return backbone


def build_mask2former_vitb(
    cfg: CfgNode,
    num_stuff_classes: int | None = None,
    num_thing_classes: int | None = None,
    class_weights: Sequence[float] | Tuple[float, ...] | None = None,
) -> Mask2FormerPanoptic:
    """Build M2F + ViT-Adapter from a yacs config.

    ``num_stuff_classes`` / ``num_thing_classes`` come from the dataloader
    (``training_dataset.stuff_classes`` / ``things_classes``) and are passed
    by ``build_model_pseudo``. The legacy ``cfg._NUM_STUFF_CLASSES`` /
    ``cfg._NUM_THING_CLASSES`` attributes remain as a fallback for callers
    that still inject those underscore-prefixed runtime keys.

    ``class_weights`` (if provided, length == num_stuff + num_thing) is folded
    into ``SetCriterion.empty_weight[:num_classes]`` so rare thing classes get
    upweighted in the CE loss. Needed because the Cascade Mask R-CNN branch
    consumes class_weights via the semantic-head path, but M2F's per-query
    classification has no equivalent plumbing.
    """
    m = cfg.MODEL.MASK2FORMER
    if num_stuff_classes is None:
        num_stuff_classes = int(getattr(cfg, "_NUM_STUFF_CLASSES", 0))
    if num_thing_classes is None:
        num_thing_classes = int(getattr(cfg, "_NUM_THING_CLASSES", 0))
    if num_stuff_classes <= 0 or num_thing_classes <= 0:
        raise ValueError(
            f"build_mask2former_vitb requires positive num_stuff_classes and "
            f"num_thing_classes; got num_stuff={num_stuff_classes}, "
            f"num_thing={num_thing_classes}. Pass them explicitly or set "
            f"cfg._NUM_STUFF_CLASSES / cfg._NUM_THING_CLASSES before calling."
        )
    num_stuff = int(num_stuff_classes)
    num_thing = int(num_thing_classes)
    num_classes = num_stuff + num_thing

    # Frozen backbone.
    dino = _build_dinov3_backbone(cfg)

    # ViT-Adapter.
    adapter = ViTAdapter(
        backbone=dino,
        embed_dim=m.ADAPTER_EMBED_DIM,
        num_blocks=m.ADAPTER_BLOCKS,
        pyramid_channels=m.PYRAMID_CHANNELS,
    )

    # Pixel decoder.
    pixel = MSDeformAttnPixelDecoder(
        in_channels=m.PYRAMID_CHANNELS,
        hidden_dim=m.HIDDEN_DIM,
        mask_dim=m.HIDDEN_DIM,
        num_layers=m.PIXEL_DECODER_LAYERS,
        num_heads=m.NUM_HEADS,
    )

    # QueryPool.
    if m.QUERY_POOL == "decoupled":
        pool = build_query_pool(
            kind="decoupled",
            num_queries_stuff=m.QUERIES_STUFF,
            num_queries_thing=m.QUERIES_THING,
            embed_dim=m.HIDDEN_DIM,
        )
        num_queries = m.QUERIES_STUFF + m.QUERIES_THING
    elif m.QUERY_POOL == "depth_bias":
        # The depth_bias pool silently returns un-modulated queries if the
        # dataloader does not supply a per-sample "depth" entry. Warn loudly
        # at build time so a misconfigured run is traceable from the log.
        log.warning(
            "QUERY_POOL='depth_bias' requires the dataloader to emit "
            "batch['depth']; missing depth falls back to standard behaviour."
        )
        pool = build_query_pool(kind="depth_bias", num_queries=m.NUM_QUERIES, embed_dim=m.HIDDEN_DIM)
        num_queries = m.NUM_QUERIES
    else:
        pool = build_query_pool(kind="standard", num_queries=m.NUM_QUERIES, embed_dim=m.HIDDEN_DIM)
        num_queries = m.NUM_QUERIES

    # Transformer decoder.
    use_decoupled = getattr(m, "DECOUPLED_CLASS_HEADS", False)
    if use_decoupled and m.QUERY_POOL != "decoupled":
        raise ValueError(
            "DECOUPLED_CLASS_HEADS=True requires QUERY_POOL='decoupled'"
        )
    dec = MaskedAttentionDecoder(
        hidden_dim=m.HIDDEN_DIM,
        num_queries=num_queries,
        num_classes=num_classes,
        num_layers=m.NUM_DECODER_LAYERS,
        num_heads=m.NUM_HEADS,
        query_pool=pool,
        droppath=m.DROPPATH,
        num_stuff_classes=num_stuff if use_decoupled else 0,
        num_thing_classes=num_thing if use_decoupled else 0,
    )

    matcher = HungarianMatcher(
        cost_class=m.CLASS_WEIGHT,
        cost_mask=m.MASK_WEIGHT,
        cost_dice=m.DICE_WEIGHT,
        num_points=m.NUM_POINTS,
        num_stuff_classes=num_stuff if use_decoupled else 0,
        num_queries_stuff=m.QUERIES_STUFF if use_decoupled else 0,
    )
    if class_weights is not None and len(class_weights) != num_classes:
        raise ValueError(
            f"class_weights length {len(class_weights)} != "
            f"num_stuff+num_thing {num_classes}"
        )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": m.CLASS_WEIGHT,
            "loss_mask": m.MASK_WEIGHT,
            "loss_dice": m.DICE_WEIGHT,
        },
        eos_coef=m.NO_OBJECT_WEIGHT,
        losses=("labels", "masks"),
        num_points=m.NUM_POINTS,
        class_weights=class_weights,
    )

    model = Mask2FormerPanoptic(
        backbone=adapter,
        pixel_decoder=pixel,
        transformer_decoder=dec,
        criterion=criterion,
        num_stuff_classes=num_stuff,
        num_thing_classes=num_thing,
        object_mask_threshold=m.OBJECT_MASK_THRESHOLD,
        overlap_threshold=m.OVERLAP_THRESHOLD,
    )

    aux_loss_hooks: Dict[str, Any] = {}
    if m.XQUERY_WEIGHT > 0.0:
        def _xquery_hook(dec_out, targets, ctx, _w=m.XQUERY_WEIGHT, _t=m.XQUERY_TEMPERATURE):
            return _w * xquery_loss(dec_out, targets, {"temperature": _t})
        aux_loss_hooks["xquery"] = _xquery_hook
    if m.QUERY_CONSISTENCY_WEIGHT > 0.0:
        def _qc_hook(dec_out, targets, ctx, _w=m.QUERY_CONSISTENCY_WEIGHT, _t=m.QUERY_CONSISTENCY_TEMPERATURE):
            return _w * query_consistency_loss(dec_out, targets, {**ctx, "temperature": _t})
        aux_loss_hooks["query_consistency"] = _qc_hook
    if aux_loss_hooks:
        aux_loss_hooks["return_query_embeds"] = True
        model.aux_loss_hooks = aux_loss_hooks

    log.info(
        "Built Mask2FormerPanoptic: queries=%d stuff=%d thing=%d "
        "adapter_blocks=%d dec_layers=%d class_weights=%s aux_hooks=%s",
        num_queries, num_stuff, num_thing, m.ADAPTER_BLOCKS, m.NUM_DECODER_LAYERS,
        "ON" if class_weights is not None else "OFF",
        sorted(k for k in aux_loss_hooks.keys() if k != "return_query_embeds"),
    )
    return model
