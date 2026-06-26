#!/usr/bin/env python3
"""HF-Transformers DINOv3 encoder adapter for INSID3.

INSID3's torch.hub path expects the ORIGINAL DINOv3 `.pth`, but Meta only ships an
HF-format `model.safetensors` (gated, downloadable with an accepted-license HF token).
INSID3 calls exactly one encoder method — `get_intermediate_layers(x, n=1,
reshape=True)` returning a list of [B, C, h, w] feature maps — so we wrap the HF
`Dinov3Model` to expose that single method. Same weights, different framework wrapper.
"""

import logging

import einops
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

__all__ = ["DinoV3HFEncoder", "REPO_BY_SIZE"]

REPO_BY_SIZE = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class DinoV3HFEncoder(nn.Module):
    """Frozen DINOv3 (HF) exposing the INSID3 `get_intermediate_layers` contract."""

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        super().__init__()
        repo = REPO_BY_SIZE[model_size]
        self.model = AutoModel.from_pretrained(repo).eval()
        self.device = device
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        cfg = self.model.config
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        logger.info("DINOv3-HF %s loaded (patch=%d, dim=%d) on %s",
                    model_size, self.patch_size, self.hidden_size, device)

    @torch.no_grad()
    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1,
                                reshape: bool = True, **_) -> list:
        x = x.to(self.device)
        _, _, h_px, w_px = x.shape
        h, w = h_px // self.patch_size, w_px // self.patch_size
        tokens = self.model(pixel_values=x).last_hidden_state  # [B, T, C]
        # robustly drop ALL prefix tokens (CLS + register tokens), keep h*w patches
        prefix = tokens.shape[1] - h * w
        patch = tokens[:, prefix:, :]
        if not reshape:
            return [patch]
        return [einops.rearrange(patch, "b (h w) c -> b c h w", h=h, w=w)]

    def to(self, device):  # keep self.device in sync (INSID3 calls encoder.to(device))
        self.device = device
        self.model.to(device)
        return self
