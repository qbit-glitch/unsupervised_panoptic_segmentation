"""
SpectralDiffusion V2 - Complete Implementation

This version implements ALL critical components from icml_2027_solution.md:
1. DINOv2/v3 backbone (float32)
2. Multi-Scale Spectral Initialization  
3. MambaSlotAttention with GMM Prior
4. SlotConditionedDiffusion decoder (true diffusion)
5. Complete 5-component loss function
6. All float32 operations

Based on:
- icml_2027_solution.md (1773 lines)
- Implementation Gap Analysis findings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

# Force float32 globally
torch.set_default_dtype(torch.float32)


class SpectralDiffusionV2(nn.Module):
    """
    Complete SpectralDiffusion with ALL critical components.
    
    Key differences from SimplifiedSpectralDiffusion:
    - Uses SlotConditionedDiffusion decoder (not MLP)
    - Has true diffusion loss
    - GMM prior properly integrated
    - All float32
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        num_slots: int = 12,
        feature_dim: int = 256,
        use_dino: bool = False,
        use_mamba: bool = True,
        use_diffusion_decoder: bool = True,  # NEW: use diffusion decoder
        init_mode: str = "spectral",
        scales: List[int] = [8, 16, 32],
        num_power_iters: int = 20,
        diffusion_steps: int = 100,
        lambda_diff: float = 1.0,
        lambda_spec: float = 0.1,
        lambda_ident: float = 0.01,
        lambda_diversity: float = 0.5,
        lambda_recon: float = 10.0,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.init_mode = init_mode
        self.use_dino = use_dino
        self.use_diffusion_decoder = use_diffusion_decoder
        self.diffusion_steps = diffusion_steps
        
        # Loss weights
        self.lambda_diff = lambda_diff
        self.lambda_spec = lambda_spec
        self.lambda_ident = lambda_ident
        self.lambda_diversity = lambda_diversity
        self.lambda_recon = lambda_recon
        
        # Import components
        from models.spectral_init import MultiScaleSpectralInit
        from models.mamba_slot import create_slot_attention, MambaSlotAttention
        from models.diffusion import SlotConditionedDiffusion, MLPMaskDecoder
        from models.pruning import AdaptiveSlotPruning
        
        # ========== 1. Feature Encoder (float32) ==========
        if use_dino:
            self._init_dino_encoder()
        else:
            # Simple CNN encoder (lightweight)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, feature_dim, 3, stride=2, padding=1),
                nn.ReLU(),
            )
        
        # ========== 2. Multi-Scale Spectral Initialization ==========
        num_scales = len(scales)
        slots_per_scale = (num_slots + num_scales - 1) // num_scales
        self.actual_num_slots = slots_per_scale * num_scales
        self.scales = scales
        
        self.spectral_init = MultiScaleSpectralInit(
            scales=scales,
            slots_per_scale=slots_per_scale,
            feature_dim=feature_dim,
            num_power_iters=num_power_iters,
        )
        
        # Learnable slot fallback
        self.slot_mu = nn.Parameter(torch.randn(1, self.actual_num_slots, feature_dim) * 0.1)
        self.slot_sigma = nn.Parameter(torch.ones(1, self.actual_num_slots, feature_dim) * 0.1)
        
        # ========== 3. Mamba-Slot Attention with GMM Prior ==========
        self.slot_attention = create_slot_attention(
            attention_type="mamba" if use_mamba else "standard",
            dim=feature_dim,
            num_slots=self.actual_num_slots,
            num_iterations=3,
        )
        
        # ========== 4. Decoder - Diffusion or MLP ==========
        if use_diffusion_decoder:
            self.diffusion_decoder = SlotConditionedDiffusion(
                num_slots=self.actual_num_slots,
                slot_dim=feature_dim,
                latent_dim=64,
                image_size=image_size,
                num_timesteps=diffusion_steps,
            )
            self.mask_decoder = None
        else:
            self.diffusion_decoder = None
            self.mask_decoder = MLPMaskDecoder(
                slot_dim=feature_dim,
                num_slots=self.actual_num_slots,
                image_size=image_size,
            )
        
        # ========== 5. Spatial Broadcast RGB Decoder ==========
        H, W = image_size
        self.slot_decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.register_buffer('pos_grid', self._create_pos_grid(H, W))
        
        self.rgb_decoder = nn.Sequential(
            nn.Linear(256 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        
        # ========== 6. Adaptive Slot Pruning ==========
        self.pruning = AdaptiveSlotPruning(
            num_slots=self.actual_num_slots,
            threshold=0.05,
        )
        
        # ========== 7. Loss Modules ==========
        from losses.identifiable_loss import IdentifiabilityLoss, SlotDiversityLoss
        from losses.spectral_loss import SpectralConsistencyLoss
        
        self.ident_loss_fn = IdentifiabilityLoss(
            num_slots=self.actual_num_slots,
            slot_dim=feature_dim,
        )
        self.diversity_loss_fn = SlotDiversityLoss()
        self.spectral_loss_fn = SpectralConsistencyLoss()
        
        # Ensure float32
        self.float()
    
    def _init_dino_encoder(self):
        """Initialize DINOv2 encoder."""
        try:
            import os
            from transformers import AutoImageProcessor, AutoModel
            
            hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
            model_name = "facebook/dinov2-base"
            
            self.dino = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float32,  # Force float32
            )
            self.dino_processor = AutoImageProcessor.from_pretrained(
                model_name,
                token=hf_token,
            )
            
            # Freeze DINOv2
            for param in self.dino.parameters():
                param.requires_grad = False
            
            self.dino_proj = nn.Linear(768, self.feature_dim)
            
        except Exception as e:
            print(f"Failed to load DINOv2: {e}")
            print("Falling back to CNN encoder")
            
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.feature_dim, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.use_dino = False
    
    def _create_pos_grid(self, H: int, W: int) -> torch.Tensor:
        y = torch.linspace(-1, 1, H, dtype=torch.float32)
        x = torch.linspace(-1, 1, W, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([xx, yy], dim=-1)
        return pos.view(1, H * W, 2)
    
    def encode_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Encode images to features."""
        B = images.shape[0]
        device = images.device
        images = images.float()  # Ensure float32
        
        if self.use_dino and hasattr(self, 'dino'):
            with torch.no_grad():
                outputs = self.dino(images, output_hidden_states=True)
                features = outputs.last_hidden_state[:, 1:, :]  # Skip CLS
            features = self.dino_proj(features.float())
            
            # Reshape to spatial
            H_feat = W_feat = int(math.sqrt(features.shape[1]))
            features = features.transpose(1, 2).reshape(B, -1, H_feat, W_feat)
        else:
            features = self.encoder(images)
        
        _, C, H_feat, W_feat = features.shape
        
        # Multi-scale features - use CPU fallback for MPS compatibility
        multiscale = {}
        for scale in self.scales:
            # Move to CPU for adaptive_avg_pool2d if on MPS
            if features.device.type == 'mps':
                pooled = F.adaptive_avg_pool2d(features.cpu(), scale).to(device)
            else:
                pooled = F.adaptive_avg_pool2d(features, scale)
            multiscale[scale] = pooled.permute(0, 2, 3, 1)
        
        flat_features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return flat_features, multiscale
    
    def decode_slots_to_image(self, slots: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode slots to reconstructed image."""
        B, K, D = slots.shape
        H, W = self.image_size
        N = H * W
        
        slot_features = self.slot_decoder(slots.float())
        slot_broadcast = slot_features.unsqueeze(2).expand(-1, -1, N, -1)
        pos = self.pos_grid.expand(B, -1, -1).unsqueeze(1).expand(-1, K, -1, -1)
        combined = torch.cat([slot_broadcast, pos], dim=-1)
        
        rgb_per_slot = self.rgb_decoder(combined)
        rgb_per_slot = rgb_per_slot.view(B, K, H, W, 3).permute(0, 1, 4, 2, 3)
        
        masks_expanded = masks.unsqueeze(2)
        reconstructed = (rgb_per_slot * masks_expanded).sum(dim=1)
        
        return reconstructed, rgb_per_slot
    
    def forward(self, images: torch.Tensor, return_loss: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B = images.shape[0]
        device = images.device
        images = images.float()  # Ensure float32
        H, W = self.image_size
        
        # 1. Encode features
        flat_features, multiscale = self.encode_features(images)
        
        # 2. Spectral initialization
        if self.init_mode == "spectral":
            slots_init = self.spectral_init(multiscale, mode="spectral")
        elif self.init_mode == "random":
            slots_init = self.spectral_init(multiscale, mode="random")
        else:
            slots_init = self.slot_mu + self.slot_sigma * torch.randn_like(self.slot_mu)
            slots_init = slots_init.expand(B, -1, -1).to(device)
        
        # 3. Mamba-Slot Attention (with GMM prior update)
        slots, attn_masks = self.slot_attention(flat_features, slots_init)
        
        # 4. Adaptive pruning
        slots, attn_masks, prune_info = self.pruning(slots, attn_masks)
        
        # 5. Decode masks (diffusion or MLP)
        if self.use_diffusion_decoder and self.diffusion_decoder is not None:
            # Use diffusion decoder - needs images for training loss
            if return_loss:
                diffusion_outputs = self.diffusion_decoder(images, slots, return_loss=True)
                diffusion_loss = diffusion_outputs.get('loss', torch.tensor(0.0, device=device))
                # Also get masks for reconstruction via inference
                with torch.no_grad():
                    mask_outputs = self.diffusion_decoder(images, slots, return_loss=False)
                    masks = mask_outputs.get('masks')
            else:
                mask_outputs = self.diffusion_decoder(images, slots, return_loss=False)
                masks = mask_outputs.get('masks')
                diffusion_loss = torch.tensor(0.0, device=device)
        else:
            # Use MLP decoder
            masks = self.mask_decoder(slots)
            diffusion_loss = torch.tensor(0.0, device=device)
        
        # 6. Reconstruct image
        reconstructed, rgb_per_slot = self.decode_slots_to_image(slots, masks)
        
        outputs = {
            'slots': slots,
            'slots_init': slots_init,
            'masks': masks,
            'attention_masks': attn_masks,
            'reconstructed': reconstructed,
            'diffusion_loss': diffusion_loss,
        }
        
        if return_loss:
            loss, loss_dict = self._compute_loss(
                images, masks, slots, reconstructed, multiscale, slots_init, diffusion_loss
            )
            outputs['loss'] = loss
            outputs['loss_dict'] = loss_dict
        
        return outputs
    
    def _compute_loss(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        slots: torch.Tensor,
        reconstructed: torch.Tensor,
        multiscale: Dict[int, torch.Tensor],
        slots_init: torch.Tensor,
        diffusion_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the FULL 5-component loss as per icml_2027_solution.md:
        
        L_SD = λ_diff * L_diff + λ_spec * L_spec + λ_ident * L_ident + L_recon + L_aux
        """
        B, K, H, W = masks.shape
        loss_dict = {}
        
        # ========== 1. DIFFUSION LOSS (L_diff) ==========
        # This is the core diffusion objective
        loss_dict['diff'] = diffusion_loss.item()
        
        # ========== 2. RECONSTRUCTION LOSS ==========
        recon_loss = F.mse_loss(reconstructed, images)
        loss_dict['recon'] = recon_loss.item()
        
        # ========== 3. SPECTRAL CONSISTENCY LOSS (L_spec) ==========
        try:
            spectral_masks = self.spectral_init.get_spectral_masks(multiscale)
            spectral_masks_resized = F.interpolate(
                spectral_masks, size=(H, W), mode='bilinear', align_corners=False
            )
            spectral_masks_resized = F.softmax(spectral_masks_resized, dim=1)
            spec_loss = F.mse_loss(masks, spectral_masks_resized)
        except Exception:
            spec_loss = torch.tensor(0.0, device=masks.device)
        loss_dict['spectral'] = spec_loss.item()
        
        # ========== 4. IDENTIFIABILITY LOSS (L_ident) - GMM Prior ==========
        if hasattr(self.slot_attention, 'compute_identifiability_loss'):
            ident_loss = self.slot_attention.compute_identifiability_loss(slots)
        else:
            ident_loss = self.ident_loss_fn(slots, update_prior=self.training)
        loss_dict['ident'] = ident_loss.item()
        
        # ========== 5. DIVERSITY LOSS (prevent slot collapse) ==========
        slots_normalized = F.normalize(slots, dim=-1)
        similarity = torch.bmm(slots_normalized, slots_normalized.transpose(1, 2))
        eye = torch.eye(K, device=slots.device).unsqueeze(0)
        off_diag = similarity * (1 - eye)
        diversity_loss = off_diag.abs().mean()
        loss_dict['diversity'] = diversity_loss.item()
        
        # ========== 6. COVERAGE LOSS (masks should sum to 1) ==========
        mask_sum = masks.sum(dim=1)
        coverage_loss = ((mask_sum - 1) ** 2).mean()
        loss_dict['coverage'] = coverage_loss.item()
        
        # ========== 7. ANTI-COLLAPSE LOSS ==========
        slot_areas = masks.mean(dim=(2, 3))
        target_area = 1.0 / K
        area_deviation = (slot_areas - target_area).abs()
        dominant_penalty = F.relu(slot_areas - 2 * target_area).mean()
        collapse_loss = area_deviation.mean() + 2.0 * dominant_penalty
        loss_dict['collapse'] = collapse_loss.item()
        
        # ========== 8. SLOT REPULSION LOSS ==========
        masks_flat = masks.view(B, K, -1)
        mask_overlap = torch.bmm(masks_flat, masks_flat.transpose(1, 2))
        mask_areas = masks_flat.sum(dim=-1, keepdim=True)
        mask_overlap_normalized = mask_overlap / (mask_areas + 1e-8)
        repulsion_loss = (mask_overlap_normalized * (1 - eye)).mean()
        loss_dict['repulsion'] = repulsion_loss.item()
        
        # ========== TOTAL LOSS ==========
        total_loss = (
            self.lambda_diff * diffusion_loss +      # Diffusion objective
            self.lambda_recon * recon_loss +         # Reconstruction
            self.lambda_spec * spec_loss +           # Spectral consistency
            self.lambda_ident * ident_loss +         # GMM prior
            self.lambda_diversity * diversity_loss + # Slot diversity
            1.0 * coverage_loss +                    # Coverage
            5.0 * collapse_loss +                    # Anti-collapse
            10.0 * repulsion_loss                    # Repulsion
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def create_spectral_diffusion_v2(args) -> SpectralDiffusionV2:
    """Factory function to create SpectralDiffusionV2 model."""
    model = SpectralDiffusionV2(
        image_size=tuple(args.image_size),
        num_slots=args.num_slots,
        feature_dim=256,
        use_dino=getattr(args, 'use_dino', False),
        use_mamba=getattr(args, 'use_mamba', True),
        use_diffusion_decoder=getattr(args, 'use_diffusion_decoder', True),
        init_mode=getattr(args, 'init_mode', 'spectral'),
        diffusion_steps=getattr(args, 'diffusion_steps', 100),
        lambda_diff=getattr(args, 'lambda_diff', 1.0),
        lambda_spec=getattr(args, 'lambda_spec', 0.1),
        lambda_ident=getattr(args, 'lambda_ident', 0.01),
    )
    
    return model.to(args.device).float()
