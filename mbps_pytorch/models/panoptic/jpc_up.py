"""JPC-Up: joint panoptic coupling with code and feature upsampling.

This module is the trainable core for single-image, fully unsupervised panoptic
pseudo-label generation. It consumes cached/frozen signals from DINO/CAUSE-DCFA
and DepthPro, then predicts semantic clusters and class-agnostic object masks.
Heavy pretrained backbones stay outside this module so it can be trained from
feature caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class JPCUpConfig:
    """Configuration for :class:`JointPanopticCouplerUp`."""

    appearance_dim: int = 768
    semantic_dim: int = 90
    fusion_dim: int = 256
    num_prototypes: int = 80
    num_queries: int = 50
    num_query_classes: int = 20
    query_dim: int = 256
    instance_embed_dim: int = 32
    low_hw: Tuple[int, int] = (32, 64)
    mid_hw: Tuple[int, int] = (64, 128)
    high_hw: Tuple[int, int] = (128, 256)
    depth_freqs: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    num_coupling_blocks: int = 4
    mid_num_coupling_blocks: int = 2
    num_heads: int = 8
    up_kernel_size: int = 3
    up_guidance_ch: int = 24
    residual_scale: float = 0.15
    proposal_anchor_logit: float = 4.0
    proposal_refine_residual_scale: float = 0.25
    dropout: float = 0.0


@dataclass
class JPCUpOutput:
    """Model outputs at high resolution plus intermediate codes."""

    refined_codes: Tensor
    semantic_logits: Tensor
    objectness_logits: Tensor
    center_logits: Tensor
    boundary_logits: Tensor
    instance_embeddings: Tensor
    mask_logits: Tensor
    query_scores: Tensor
    query_class_logits: Tensor
    query_embeddings: Tensor
    low_codes: Tensor
    mid_codes: Tensor
    mid_features: Tensor
    high_features: Tensor
    depth_maps: Dict[str, Tensor]
    depth_grads: Dict[str, Tensor]
    depth_encodings: Dict[str, Tensor]
    proposal_anchor_logits: Optional[Tensor] = None
    proposal_anchor_masks: Optional[Tensor] = None
    proposal_anchor_valid: Optional[Tensor] = None
    proposal_anchor_indices: Optional[Tensor] = None
    proposal_mask_delta: Optional[Tensor] = None


class ConvGNAct(nn.Module):
    """Small Conv-GN-GELU block used throughout the module."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DepthwiseResidualBlock(nn.Module):
    """Depthwise residual refinement for high-resolution feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pw(self.act(self.dw(self.norm(x))))


class DepthFiLM2d(nn.Module):
    """Per-pixel FiLM modulation from sinusoidal depth encodings."""

    def __init__(self, depth_ch: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Conv2d(depth_ch, 2 * channels, 1)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, depth_encoding: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(depth_encoding).chunk(2, dim=1)
        return x * (1.0 + scale) + shift


class DynamicKernelLifter(nn.Module):
    """AnyUp-style local reassembly for code or feature maps.

    Guidance predicts one local kernel per output pixel. The same kernel is
    shared across channels, preserving feature semantics while allowing
    RGB/depth boundaries to steer high-resolution reassembly.
    """

    def __init__(
        self,
        channels: int,
        guidance_ch: int,
        hidden_ch: int,
        kernel_size: int = 3,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.kernel_elems = kernel_size * kernel_size
        self.residual_scale = residual_scale

        self.guidance = nn.Sequential(
            ConvGNAct(guidance_ch, hidden_ch),
            DepthwiseResidualBlock(hidden_ch),
        )
        self.kernel_head = nn.Conv2d(hidden_ch, self.kernel_elems, 3, padding=1)
        nn.init.zeros_(self.kernel_head.weight)
        nn.init.zeros_(self.kernel_head.bias)
        with torch.no_grad():
            self.kernel_head.bias[self.kernel_elems // 2] = 4.0

        self.refine = nn.Sequential(
            ConvGNAct(channels + hidden_ch, max(channels, hidden_ch)),
            DepthwiseResidualBlock(max(channels, hidden_ch)),
            nn.Conv2d(max(channels, hidden_ch), channels, 3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x: Tensor, guidance: Tensor, output_size: Tuple[int, int]) -> Tensor:
        base = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        guide = self.guidance(guidance)
        weights = F.softmax(self.kernel_head(guide), dim=1)

        bsz, channels, height, width = base.shape
        patches = F.unfold(
            base,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        patches = patches.view(bsz, channels, self.kernel_elems, height, width)
        reassembled = (patches * weights.unsqueeze(1)).sum(dim=2)
        residual = self.refine(torch.cat([reassembled, guide], dim=1))
        return reassembled + self.residual_scale * residual


class PanopticCouplingBlock(nn.Module):
    """Bidirectional semantic-instance coupling at the low-resolution grid."""

    def __init__(
        self,
        channels: int,
        depth_ch: int,
        num_heads: int,
        num_prototypes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth_film = DepthFiLM2d(depth_ch, channels)
        self.sem_norm = nn.LayerNorm(channels)
        self.inst_norm = nn.LayerNorm(channels)
        self.sem_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.inst_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.sem_to_inst = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.inst_to_sem = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.proto_tokens = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)
        self.pixel_to_proto = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.proto_to_pixel = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.sem_mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.inst_mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, sem: Tensor, inst: Tensor, depth_encoding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bsz, channels, height, width = sem.shape
        sem = self.depth_film(sem, depth_encoding)
        inst = self.depth_film(inst, depth_encoding)
        sem_tok = sem.flatten(2).transpose(1, 2)
        inst_tok = inst.flatten(2).transpose(1, 2)

        sem_tok = sem_tok + self.sem_attn(self.sem_norm(sem_tok), self.sem_norm(sem_tok), self.sem_norm(sem_tok), need_weights=False)[0]
        inst_tok = inst_tok + self.inst_attn(self.inst_norm(inst_tok), self.inst_norm(inst_tok), self.inst_norm(inst_tok), need_weights=False)[0]
        inst_tok = inst_tok + self.sem_to_inst(self.inst_norm(inst_tok), self.sem_norm(sem_tok), self.sem_norm(sem_tok), need_weights=False)[0]
        sem_tok = sem_tok + self.inst_to_sem(self.sem_norm(sem_tok), self.inst_norm(inst_tok), self.inst_norm(inst_tok), need_weights=False)[0]

        proto = self.proto_tokens.unsqueeze(0).expand(bsz, -1, -1)
        proto = proto + self.pixel_to_proto(proto, sem_tok, sem_tok, need_weights=False)[0]
        sem_tok = sem_tok + self.proto_to_pixel(sem_tok, proto, proto, need_weights=False)[0]

        sem_tok = sem_tok + self.sem_mlp(sem_tok)
        inst_tok = inst_tok + self.inst_mlp(inst_tok)
        sem = sem_tok.transpose(1, 2).reshape(bsz, channels, height, width)
        inst = inst_tok.transpose(1, 2).reshape(bsz, channels, height, width)
        fused = self.fuse(torch.cat([sem, inst], dim=1))
        return sem, inst, fused


class EfficientPanopticCouplingBlock(nn.Module):
    """Scalable mid-resolution coupling with local mixing and K prototypes.

    The low-resolution block can afford full pixel-token attention.  At
    ``64 x 128`` that would be unnecessarily heavy, so this block keeps the
    same semantic/instance/prototype contract with depth FiLM, depthwise local
    mixing, bidirectional cross-branch gates, and pixel-to-prototype attention
    whose complexity is ``O(HWK)`` instead of ``O((HW)^2)``.
    """

    def __init__(self, channels: int, depth_ch: int, num_prototypes: int) -> None:
        super().__init__()
        self.depth_film = DepthFiLM2d(depth_ch, channels)
        self.sem_local = DepthwiseResidualBlock(channels)
        self.inst_local = DepthwiseResidualBlock(channels)
        self.sem_from_inst = nn.Sequential(
            ConvGNAct(2 * channels, channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.inst_from_sem = nn.Sequential(
            ConvGNAct(2 * channels, channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.proto_tokens = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)
        self.proto_norm = nn.LayerNorm(channels)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, sem: Tensor, inst: Tensor, depth_encoding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bsz, channels, height, width = sem.shape
        sem = self.sem_local(self.depth_film(sem, depth_encoding))
        inst = self.inst_local(self.depth_film(inst, depth_encoding))

        sem_gate = self.sem_from_inst(torch.cat([sem, inst], dim=1))
        inst_gate = self.inst_from_sem(torch.cat([inst, sem], dim=1))
        sem = sem + sem_gate * inst
        inst = inst + inst_gate * sem

        sem_tok = sem.flatten(2).transpose(1, 2)
        proto = self.proto_norm(self.proto_tokens).unsqueeze(0).expand(bsz, -1, -1)
        attn_pixel_to_proto = torch.softmax((sem_tok @ proto.transpose(1, 2)) / (channels ** 0.5), dim=-1)
        proto_update = torch.softmax(attn_pixel_to_proto.transpose(1, 2), dim=-1) @ sem_tok
        proto = proto + proto_update
        sem_tok = sem_tok + attn_pixel_to_proto @ proto
        sem = sem_tok.transpose(1, 2).reshape(bsz, channels, height, width)

        fused = self.fuse(torch.cat([sem, inst], dim=1))
        return sem, inst, fused


class ObjectQueryDecoder(nn.Module):
    """Object query decoder with optional proposal-conditioned anchors."""

    def __init__(
        self,
        channels: int,
        query_dim: int,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        *,
        anchor_logit: float = 4.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.num_queries = num_queries
        self.anchor_logit = float(anchor_logit)
        self.residual_scale = float(residual_scale)
        self.token_proj = nn.Linear(channels, query_dim)
        self.proposal_feature_proj = nn.Sequential(
            nn.Linear(channels, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.proposal_geometry_proj = nn.Sequential(
            nn.Linear(7, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.proposal_score_proj = nn.Linear(1, query_dim)
        self.query_attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(query_dim)
        self.mask_feature = nn.Conv2d(channels, query_dim, 1)
        self.score_head = nn.Linear(query_dim, 1)
        self.class_head = nn.Linear(query_dim, num_classes)
        self.embed_head = nn.Linear(query_dim, embed_dim)

    def _proposal_scores(
        self,
        proposal_masks: Tensor,
        proposal_scores: Optional[Tensor],
    ) -> Tensor:
        bsz, num_props = proposal_masks.shape[:2]
        if proposal_scores is None:
            scores = torch.ones((bsz, num_props), dtype=proposal_masks.dtype, device=proposal_masks.device)
        else:
            scores = proposal_scores.to(device=proposal_masks.device, dtype=proposal_masks.dtype)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0).expand(bsz, -1)
            if scores.shape[1] != num_props:
                aligned = torch.zeros((bsz, num_props), dtype=scores.dtype, device=scores.device)
                n = min(num_props, scores.shape[1])
                aligned[:, :n] = scores[:, :n]
                scores = aligned
        area = proposal_masks.flatten(2).sum(dim=-1)
        return torch.where(area > 0, scores.clamp(0.0, 1.0), torch.zeros_like(scores))

    def _select_proposal_anchors(
        self,
        proposal_masks: Optional[Tensor],
        proposal_scores: Optional[Tensor],
        output_hw: Tuple[int, int],
        reference: Tensor,
    ) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if proposal_masks is None or proposal_masks.numel() == 0:
            return None, None, None, None, None

        masks = proposal_masks.to(device=reference.device, dtype=reference.dtype)
        if masks.dim() == 3:
            masks = masks.unsqueeze(0)
        if masks.dim() != 4:
            raise ValueError(f"proposal_masks must be B x P x H x W or P x H x W, got {tuple(masks.shape)}")
        if masks.shape[-2:] != output_hw:
            masks = F.interpolate(masks, size=output_hw, mode="nearest")
        masks = masks.clamp(0.0, 1.0)
        scores = self._proposal_scores(masks, proposal_scores)

        bsz, _, height, width = masks.shape
        anchor_masks = masks.new_zeros((bsz, self.num_queries, height, width))
        anchor_scores = masks.new_zeros((bsz, self.num_queries))
        anchor_valid = torch.zeros((bsz, self.num_queries), dtype=torch.bool, device=masks.device)
        anchor_indices = torch.full((bsz, self.num_queries), -1, dtype=torch.long, device=masks.device)

        for b in range(bsz):
            valid = torch.nonzero(scores[b] > 0, as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            order = valid[torch.argsort(scores[b, valid], descending=True)]
            order = order[: self.num_queries]
            n = int(order.numel())
            anchor_masks[b, :n] = masks[b, order]
            anchor_scores[b, :n] = scores[b, order]
            anchor_valid[b, :n] = True
            anchor_indices[b, :n] = order

        anchor_logits = torch.where(
            anchor_masks > 0.5,
            anchor_masks.new_full((), self.anchor_logit),
            anchor_masks.new_full((), -self.anchor_logit),
        )
        return anchor_masks, anchor_logits, anchor_scores, anchor_valid, anchor_indices

    @staticmethod
    def _mask_geometry(masks: Tensor, valid: Tensor) -> Tensor:
        bsz, num_queries, height, width = masks.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=masks.device, dtype=masks.dtype),
            torch.linspace(0.0, 1.0, width, device=masks.device, dtype=masks.dtype),
            indexing="ij",
        )
        xx = xx.view(1, 1, height, width)
        yy = yy.view(1, 1, height, width)
        keep = masks > 0.5
        one = masks.new_ones(())
        zero = masks.new_zeros(())
        x_min = torch.where(keep, xx, one).amin(dim=(-2, -1))
        y_min = torch.where(keep, yy, one).amin(dim=(-2, -1))
        x_max = torch.where(keep, xx, zero).amax(dim=(-2, -1))
        y_max = torch.where(keep, yy, zero).amax(dim=(-2, -1))
        area = masks.flatten(2).mean(dim=-1).clamp(0.0, 1.0)
        width_box = (x_max - x_min).clamp(min=0.0)
        height_box = (y_max - y_min).clamp(min=0.0)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        geom = torch.stack([cx, cy, width_box, height_box, area, x_min, y_min], dim=-1)
        return torch.where(valid.unsqueeze(-1), geom, torch.zeros_like(geom))

    def forward(
        self,
        features: Tensor,
        proposal_masks: Optional[Tensor] = None,
        proposal_scores: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor]:
        bsz, _, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        queries = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        anchor_masks, anchor_logits, anchor_scores, anchor_valid, anchor_indices = self._select_proposal_anchors(
            proposal_masks,
            proposal_scores,
            (height, width),
            features,
        )
        if anchor_masks is not None and anchor_valid is not None and anchor_scores is not None:
            masks_flat = anchor_masks.flatten(2)
            features_flat = features.flatten(2)
            denom = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled = torch.einsum("bqn,bcn->bqc", masks_flat, features_flat) / denom
            geom = self._mask_geometry(anchor_masks, anchor_valid)
            proposal_query = (
                self.proposal_feature_proj(pooled)
                + self.proposal_geometry_proj(geom)
                + self.proposal_score_proj(anchor_scores.unsqueeze(-1))
            )
            queries = queries + proposal_query * anchor_valid.unsqueeze(-1).to(dtype=queries.dtype)
        queries = queries + self.query_attn(queries, tokens, tokens, need_weights=False)[0]
        queries = self.query_norm(queries)

        mask_features = self.mask_feature(features)
        mask_delta = torch.einsum("bqc,bchw->bqhw", queries, mask_features)
        mask_delta = mask_delta / (queries.shape[-1] ** 0.5)
        if anchor_logits is not None and anchor_valid is not None:
            anchored = anchor_logits + self.residual_scale * mask_delta
            mask_logits = torch.where(anchor_valid[:, :, None, None], anchored, mask_delta)
        else:
            mask_logits = mask_delta
        query_scores = self.score_head(queries).squeeze(-1)
        query_class_logits = self.class_head(queries)
        query_embeddings = F.normalize(self.embed_head(queries), dim=-1)
        return (
            mask_logits.reshape(bsz, -1, height, width),
            query_scores,
            query_class_logits,
            query_embeddings,
            anchor_logits,
            anchor_masks,
            anchor_valid,
            anchor_indices,
            mask_delta,
        )


def _resize_like(x: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
    if x.shape[-2:] == size:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def _sobel_grad(depth: Tensor) -> Tensor:
    dtype = depth.dtype
    device = depth.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(depth, kx, padding=1)
    gy = F.conv2d(depth, ky, padding=1)
    return torch.cat([gx, gy], dim=1)


def _coord_grid(batch: int, size: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
    height, width = size
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return xy.expand(batch, -1, -1, -1)


def sinusoidal_depth_encoding(depth: Tensor, freqs: Tuple[float, ...]) -> Tensor:
    """Return 16D default sin/cos depth encoding for ``B x 1 x H x W`` depth."""
    chans = []
    for freq in freqs:
        scaled = depth * (freq * torch.pi)
        chans.append(torch.sin(scaled))
        chans.append(torch.cos(scaled))
    return torch.cat(chans, dim=1)


class JointPanopticCouplerUp(nn.Module):
    """Joint semantic-instance pseudo-label generator.

    Forward inputs are frozen/cached signals:

    * ``appearance_features``: DINO-style features at ``config.low_hw``.
    * ``semantic_codes``: DCFA/CAUSE 90D codes at ``config.low_hw``.
    * ``depth``: DepthPro depth map at image or output resolution.
    * ``rgb``: optional RGB guidance at image resolution for upsampling.
    """

    def __init__(self, config: JPCUpConfig = JPCUpConfig()) -> None:
        super().__init__()
        self.config = config
        if config.up_guidance_ch != 24:
            raise ValueError("JPC-Up guidance is fixed to RGB(3)+depth(1)+grad(2)+depth_enc(16)+xy(2) = 24 channels")
        depth_ch = 2 * len(config.depth_freqs)

        self.appearance_proj = ConvGNAct(config.appearance_dim, config.fusion_dim, kernel_size=1)
        self.semantic_proj = ConvGNAct(config.semantic_dim, config.fusion_dim, kernel_size=1)
        self.low_input_proj = ConvGNAct(config.fusion_dim + config.semantic_dim + depth_ch + 2 + 1 + 2, config.fusion_dim, kernel_size=1)

        self.low_couplers = nn.ModuleList(
            [
                PanopticCouplingBlock(
                    channels=config.fusion_dim,
                    depth_ch=depth_ch,
                    num_heads=config.num_heads,
                    num_prototypes=config.num_prototypes,
                    dropout=config.dropout,
                )
                for _ in range(config.num_coupling_blocks)
            ]
        )
        self.mid_couplers = nn.ModuleList(
            [
                EfficientPanopticCouplingBlock(
                    channels=config.fusion_dim,
                    depth_ch=depth_ch,
                    num_prototypes=config.num_prototypes,
                )
                for _ in range(config.mid_num_coupling_blocks)
            ]
        )

        self.code_up1 = DynamicKernelLifter(
            config.semantic_dim, config.up_guidance_ch, hidden_ch=32,
            kernel_size=config.up_kernel_size, residual_scale=config.residual_scale,
        )
        self.code_up2 = DynamicKernelLifter(
            config.semantic_dim, config.up_guidance_ch, hidden_ch=32,
            kernel_size=config.up_kernel_size, residual_scale=config.residual_scale,
        )
        self.feat_up1 = DynamicKernelLifter(
            config.fusion_dim, config.up_guidance_ch, hidden_ch=64,
            kernel_size=config.up_kernel_size, residual_scale=config.residual_scale,
        )
        self.feat_up2 = DynamicKernelLifter(
            config.fusion_dim, config.up_guidance_ch, hidden_ch=64,
            kernel_size=config.up_kernel_size, residual_scale=config.residual_scale,
        )

        self.high_refine = nn.Sequential(
            DepthwiseResidualBlock(config.fusion_dim),
            DepthwiseResidualBlock(config.fusion_dim),
        )
        self.refined_code_head = nn.Sequential(
            ConvGNAct(config.fusion_dim + config.semantic_dim, config.fusion_dim),
            nn.Conv2d(config.fusion_dim, config.semantic_dim, 1),
        )
        self.semantic_proto_head = nn.Conv2d(config.semantic_dim, config.num_prototypes, 1)

        self.objectness_head = nn.Sequential(ConvGNAct(config.fusion_dim, 64), nn.Conv2d(64, 1, 1))
        self.center_head = nn.Sequential(ConvGNAct(config.fusion_dim, 64), nn.Conv2d(64, 1, 1))
        self.boundary_head = nn.Sequential(ConvGNAct(config.fusion_dim, 64), nn.Conv2d(64, 1, 1))
        self.instance_embed_head = nn.Sequential(
            ConvGNAct(config.fusion_dim, 64),
            nn.Conv2d(64, config.instance_embed_dim, 1),
        )
        self.query_decoder = ObjectQueryDecoder(
            channels=config.fusion_dim,
            query_dim=config.query_dim,
            num_queries=config.num_queries,
            embed_dim=config.instance_embed_dim,
            num_heads=config.num_heads,
            num_classes=config.num_query_classes,
            anchor_logit=config.proposal_anchor_logit,
            residual_scale=config.proposal_refine_residual_scale,
        )

    def _depth_pyramid(self, depth: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        cfg = self.config
        depth_maps = {
            "low": _resize_like(depth, cfg.low_hw),
            "mid": _resize_like(depth, cfg.mid_hw),
            "high": _resize_like(depth, cfg.high_hw),
        }
        grads = {key: _sobel_grad(val) for key, val in depth_maps.items()}
        enc = {key: sinusoidal_depth_encoding(val, cfg.depth_freqs) for key, val in depth_maps.items()}
        return depth_maps, grads, enc

    def _guidance(
        self,
        rgb: Optional[Tensor],
        depth_map: Tensor,
        grad: Tensor,
        depth_encoding: Tensor,
        size: Tuple[int, int],
    ) -> Tensor:
        bsz = depth_map.shape[0]
        if rgb is None:
            rgb_resized = torch.zeros(
                bsz, 3, size[0], size[1],
                device=depth_map.device,
                dtype=depth_map.dtype,
            )
        else:
            rgb_resized = _resize_like(rgb, size)
        xy = _coord_grid(bsz, size, depth_map.device, depth_map.dtype)
        return torch.cat([rgb_resized, depth_map, grad, depth_encoding, xy], dim=1)

    def forward(
        self,
        appearance_features: Tensor,
        semantic_codes: Tensor,
        depth: Tensor,
        rgb: Optional[Tensor] = None,
        proposal_masks: Optional[Tensor] = None,
        proposal_scores: Optional[Tensor] = None,
    ) -> JPCUpOutput:
        cfg = self.config
        if appearance_features.shape[-2:] != cfg.low_hw:
            raise ValueError(f"appearance_features must have spatial size {cfg.low_hw}, got {appearance_features.shape[-2:]}")
        if semantic_codes.shape[-2:] != cfg.low_hw:
            raise ValueError(f"semantic_codes must have spatial size {cfg.low_hw}, got {semantic_codes.shape[-2:]}")
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if rgb is not None and rgb.dim() != 4:
            raise ValueError("rgb must be B x 3 x H x W when provided")

        bsz = appearance_features.shape[0]
        depth_maps, depth_grads, depth_enc = self._depth_pyramid(depth)
        xy_low = _coord_grid(bsz, cfg.low_hw, appearance_features.device, appearance_features.dtype)

        appearance = self.appearance_proj(appearance_features)
        low_input = torch.cat(
            [
                appearance,
                semantic_codes,
                depth_enc["low"],
                depth_grads["low"],
                depth_maps["low"],
                xy_low,
            ],
            dim=1,
        )
        fused = self.low_input_proj(low_input)
        sem = fused
        inst = fused
        for block in self.low_couplers:
            sem, inst, fused = block(sem, inst, depth_enc["low"])

        mid_guidance = self._guidance(rgb, depth_maps["mid"], depth_grads["mid"], depth_enc["mid"], cfg.mid_hw)
        high_guidance = self._guidance(rgb, depth_maps["high"], depth_grads["high"], depth_enc["high"], cfg.high_hw)
        mid_codes = self.code_up1(semantic_codes, mid_guidance, cfg.mid_hw)
        mid_features = self.feat_up1(fused, mid_guidance, cfg.mid_hw)
        mid_sem = mid_features + self.semantic_proj(mid_codes)
        mid_inst = mid_features
        for block in self.mid_couplers:
            mid_sem, mid_inst, mid_features = block(mid_sem, mid_inst, depth_enc["mid"])

        high_codes = self.code_up2(mid_codes, high_guidance, cfg.high_hw)
        high_features = self.feat_up2(mid_features, high_guidance, cfg.high_hw)
        high_features = self.high_refine(high_features + self.semantic_proj(high_codes))

        refined_delta = self.refined_code_head(torch.cat([high_features, high_codes], dim=1))
        refined_codes = high_codes + refined_delta
        semantic_logits = self.semantic_proto_head(refined_codes)

        objectness_logits = self.objectness_head(high_features)
        center_logits = self.center_head(high_features)
        boundary_logits = self.boundary_head(high_features)
        instance_embeddings = F.normalize(self.instance_embed_head(high_features), dim=1)
        (
            mask_logits,
            query_scores,
            query_class_logits,
            query_embeddings,
            proposal_anchor_logits,
            proposal_anchor_masks,
            proposal_anchor_valid,
            proposal_anchor_indices,
            proposal_mask_delta,
        ) = self.query_decoder(high_features, proposal_masks=proposal_masks, proposal_scores=proposal_scores)

        return JPCUpOutput(
            refined_codes=refined_codes,
            semantic_logits=semantic_logits,
            objectness_logits=objectness_logits,
            center_logits=center_logits,
            boundary_logits=boundary_logits,
            instance_embeddings=instance_embeddings,
            mask_logits=mask_logits,
            query_scores=query_scores,
            query_class_logits=query_class_logits,
            query_embeddings=query_embeddings,
            low_codes=semantic_codes,
            mid_codes=mid_codes,
            mid_features=mid_features,
            high_features=high_features,
            depth_maps=depth_maps,
            depth_grads=depth_grads,
            depth_encodings=depth_enc,
            proposal_anchor_logits=proposal_anchor_logits,
            proposal_anchor_masks=proposal_anchor_masks,
            proposal_anchor_valid=proposal_anchor_valid,
            proposal_anchor_indices=proposal_anchor_indices,
            proposal_mask_delta=proposal_mask_delta,
        )
