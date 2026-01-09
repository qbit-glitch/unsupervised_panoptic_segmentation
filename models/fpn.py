#!/usr/bin/env python
"""
Feature Pyramid Network (FPN) for Multi-Scale Features

Based on:
- FPN (Lin et al., CVPR 2017)
- PANet (Liu et al., CVPR 2018)

Provides multi-scale feature extraction for better object discovery.
Expected improvement: +2-3 PQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale slot attention.
    
    Takes features from multiple backbone levels and creates
    a unified multi-scale feature pyramid.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],  # e.g., [256, 512, 1024, 2048]
        out_channels: int = 256,
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        # Output convolutions (3x3 to smooth after upsampling)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Build feature pyramid from backbone features.
        
        Args:
            features: List of [B, C_i, H_i, W_i] features from backbone
                      (from low resolution to high resolution)
                      
        Returns:
            pyramid: List of [B, out_channels, H_i, W_i] pyramid features
        """
        assert len(features) == self.num_levels
        
        # Lateral connections
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway with lateral connections
        # Start from highest level (smallest feature map)
        for i in range(self.num_levels - 1, 0, -1):
            # Upsample and add
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        # Output convolutions
        pyramid = [
            output_conv(lat)
            for lat, output_conv in zip(laterals, self.output_convs)
        ]
        
        return pyramid


class MultiscaleSlotAttention(nn.Module):
    """
    Multi-scale Slot Attention using FPN features.
    
    Applies slot attention at multiple scales and fuses results.
    """
    
    def __init__(
        self,
        slot_attention_module: nn.Module,
        num_scales: int = 3,
        fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
    ):
        super().__init__()
        self.slot_attention = slot_attention_module
        self.num_scales = num_scales
        self.fusion_method = fusion_method
        
        # Projection layers for each scale
        self.scale_projs = nn.ModuleList([
            nn.Linear(slot_attention_module.dim, slot_attention_module.dim)
            for _ in range(num_scales)
        ])
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = nn.Linear(
                slot_attention_module.dim * num_scales,
                slot_attention_module.dim
            )
        elif fusion_method == 'attention':
            self.fusion_attn = nn.MultiheadAttention(
                slot_attention_module.dim,
                num_heads=4,
                batch_first=True
            )
    
    def forward(
        self,
        features_pyramid: List[torch.Tensor],
        slots_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply slot attention at multiple scales.
        
        Args:
            features_pyramid: List of [B, H_i, W_i, D] features at different scales
            slots_init: [B, K, D] optional initial slots
            
        Returns:
            slots: [B, K, D] fused slot representations
            attn_maps: List of [B, N_i, K] attention maps per scale
        """
        all_slots = []
        all_attns = []
        
        for i, features in enumerate(features_pyramid[:self.num_scales]):
            # Flatten spatial dims
            B, H, W, D = features.shape
            features_flat = features.reshape(B, H * W, D)
            
            # Slot attention
            slots, attn = self.slot_attention(features_flat, slots_init)
            
            # Project
            slots = self.scale_projs[i](slots)
            
            all_slots.append(slots)
            all_attns.append(attn)
        
        # Fuse slots across scales
        if self.fusion_method == 'concat':
            # Concatenate and project
            fused = torch.cat(all_slots, dim=-1)  # [B, K, D * num_scales]
            fused = self.fusion(fused)  # [B, K, D]
        elif self.fusion_method == 'add':
            # Simple addition
            fused = sum(all_slots) / len(all_slots)
        elif self.fusion_method == 'attention':
            # Stack and use attention to aggregate
            stacked = torch.stack(all_slots, dim=2)  # [B, K, num_scales, D]
            B, K, S, D = stacked.shape
            stacked = stacked.reshape(B * K, S, D)
            fused, _ = self.fusion_attn(stacked, stacked, stacked)
            fused = fused.mean(dim=1).reshape(B, K, D)
        else:
            fused = all_slots[0]
        
        return fused, all_attns


class SimpleFPN(nn.Module):
    """
    Simple FPN that works with a single backbone output.
    
    Creates multi-scale features by pooling at different resolutions.
    Useful when you don't have access to intermediate backbone features.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        scales: List[float] = [1.0, 0.5, 0.25],
    ):
        super().__init__()
        self.scales = scales
        
        # Projection for each scale
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
            for _ in scales
        ])
    
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Create feature pyramid from single feature map.
        
        Args:
            features: [B, C, H, W] input features
            
        Returns:
            pyramid: List of [B, C, H_i, W_i] multi-scale features
        """
        B, C, H, W = features.shape
        pyramid = []
        
        for scale, proj in zip(self.scales, self.projs):
            if scale != 1.0:
                new_H, new_W = int(H * scale), int(W * scale)
                scaled = F.interpolate(
                    features, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
            else:
                scaled = features
            
            pyramid.append(proj(scaled))
        
        return pyramid
