"""Builder for Mask2Former + ViT-Adapter with frozen DINOv3 ViT-B/16."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from yacs.config import CfgNode

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
    the cascade builder in ``model_vitb.py``. Future LoRA ablations can set
    ``cfg.MODEL.LORA_CONFIG`` without silently being dropped here.
    """
    lora_config = getattr(cfg.MODEL, "LORA_CONFIG", None)
    backbone = DINOv3ViTBackbone(
        freeze=cfg.MODEL.DINOV2_FREEZE,
        lora_config=lora_config,
    )
    return backbone


def build_mask2former_vitb(cfg: CfgNode) -> Mask2FormerPanoptic:
    """Build M2F + ViT-Adapter from a yacs config."""
    m = cfg.MODEL.MASK2FORMER
    num_stuff = int(cfg._NUM_STUFF_CLASSES)
    num_thing = int(cfg._NUM_THING_CLASSES)
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
    dec = MaskedAttentionDecoder(
        hidden_dim=m.HIDDEN_DIM,
        num_queries=num_queries,
        num_classes=num_classes,
        num_layers=m.NUM_DECODER_LAYERS,
        num_heads=m.NUM_HEADS,
        query_pool=pool,
        droppath=m.DROPPATH,
    )

    matcher = HungarianMatcher(
        cost_class=m.CLASS_WEIGHT,
        cost_mask=m.MASK_WEIGHT,
        cost_dice=m.DICE_WEIGHT,
        num_points=m.NUM_POINTS,
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
    log.info(
        "Built Mask2FormerPanoptic: queries=%d stuff=%d thing=%d adapter_blocks=%d dec_layers=%d",
        num_queries, num_stuff, num_thing, m.ADAPTER_BLOCKS, m.NUM_DECODER_LAYERS,
    )
    return model
