"""
Full SpectralDiffusion Model

Complete implementation integrating all components from icml_2027_solution.md:
- DINOv2 frozen backbone (or simple CNN fallback)
- Multi-scale spectral initialization (3 scales: 8, 16, 32)
- Mamba-Slot Attention with identifiability
- Adaptive slot pruning
- Spatial broadcast decoder with reconstruction loss
- All auxiliary losses: spectral, identifiability, diversity

Based on:
- SlotDiffusion (NeurIPS 2023)
- Identifiable Object-Centric Learning (NeurIPS 2024)
- Mamba-2 (ICML 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .spectral_init import MultiScaleSpectralInit
from .mamba_slot import create_slot_attention, MambaSlotAttention
from .diffusion import MLPMaskDecoder
from .pruning import AdaptiveSlotPruning


class FullSpectralDiffusion(nn.Module):
    """
    Full SpectralDiffusion model with all components integrated.
    
    Architecture:
    1. Feature Encoder (CNN or DINOv2)
    2. Multi-scale Spectral Initialization
    3. Mamba-Slot Attention
    4. Adaptive Slot Pruning
    5. Spatial Broadcast Decoder
    
    Losses:
    - Reconstruction loss (MSE)
    - Coverage loss (masks sum to 1)
    - Overlap loss (masks don't overlap)
    - Slot diversity loss (slots are different)
    - Spectral consistency loss (slots align with spectral)
    - Identifiability loss (GMM prior regularization)
    
    Args:
        image_size: Tuple of (H, W)
        num_slots: Number of slots
        feature_dim: Feature dimension (default 256)
        use_dino: Whether to use DINOv2 backbone (slower but better)
        use_mamba: Whether to use Mamba (True) or standard attention
        init_mode: Slot initialization mode
        scales: Multi-scale spatial dimensions
        lambda_spec: Weight for spectral consistency loss
        lambda_ident: Weight for identifiability loss
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        num_slots: int = 12,
        feature_dim: int = 256,
        use_dino: bool = False,
        use_mamba: bool = True,
        init_mode: str = "spectral",
        scales: List[int] = [8, 16, 32],
        num_power_iters: int = 20,
        lambda_spec: float = 0.1,
        lambda_ident: float = 0.01,
        lambda_diversity: float = 0.5,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.init_mode = init_mode
        self.scales = scales
        self.lambda_spec = lambda_spec
        self.lambda_ident = lambda_ident
        self.lambda_diversity = lambda_diversity
        
        # ========== 1. Feature Encoder ==========
        if use_dino:
            # DINOv2 backbone (frozen)
            self._init_dino_encoder()
        else:
            # Simple CNN encoder (faster, for testing)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(),
            )
            self.use_dino = False
        
        # ========== 2. Multi-Scale Spectral Initialization ==========
        # Calculate slots per scale dynamically
        num_scales = len(scales)
        slots_per_scale = (num_slots + num_scales - 1) // num_scales
        self.actual_num_slots = slots_per_scale * num_scales
        
        self.spectral_init = MultiScaleSpectralInit(
            scales=scales,
            slots_per_scale=slots_per_scale,
            feature_dim=feature_dim,
            num_power_iters=num_power_iters,
        )
        
        # Learnable slots as fallback/alternative
        self.slot_mu = nn.Parameter(torch.randn(1, self.actual_num_slots, feature_dim) * 0.1)
        self.slot_sigma = nn.Parameter(torch.ones(1, self.actual_num_slots, feature_dim) * 0.1)
        
        # ========== 3. Mamba-Slot Attention ==========
        self.slot_attention = create_slot_attention(
            attention_type="mamba" if use_mamba else "standard",
            dim=feature_dim,
            num_slots=self.actual_num_slots,
            num_iterations=3,
        )
        
        # ========== 4. Mask Decoder ==========
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
        
        # Positional encoding
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
        # Identifiability loss (GMM prior)
        from losses.identifiable_loss import IdentifiabilityLoss, SlotDiversityLoss
        self.ident_loss_fn = IdentifiabilityLoss(
            num_slots=self.actual_num_slots,
            slot_dim=feature_dim,
        )
        self.diversity_loss_fn = SlotDiversityLoss()
        
        # Spectral consistency loss
        from losses.spectral_loss import SpectralConsistencyLoss
        self.spectral_loss_fn = SpectralConsistencyLoss()
        
    def _init_dino_encoder(self):
        """Initialize DINOv3 encoder from HuggingFace transformers."""
        try:
            import os
            from transformers import AutoImageProcessor, AutoModel
            
            model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            print(f"Loading DINOv3 from {model_name}...")
            
            # Get token from environment
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            self.dino_processor = AutoImageProcessor.from_pretrained(model_name, token=token)
            self.dino = AutoModel.from_pretrained(model_name, token=token, torch_dtype=torch.float32)
            
            # Freeze backbone
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
            
            # Get feature dim from model config (ViT-L/16 has 1024 dim)
            dino_dim = self.dino.config.hidden_size
            self.dino_proj = nn.Linear(dino_dim, self.feature_dim)
            self.use_dino = True
            self.dino_patch_size = 16  # vit-l16
            print(f"DINOv3 backbone loaded and frozen (dim={dino_dim})")
        except Exception as e:
            print(f"Could not load DINOv3: {e}")
            print("Falling back to CNN encoder")
            self.use_dino = False
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, self.feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(),
            )
    
    def _create_pos_grid(self, H: int, W: int) -> torch.Tensor:
        """Create normalized position grid."""
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([xx, yy], dim=-1)
        return pos.view(1, H * W, 2)
    
    def encode_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Encode images to features.
        
        Returns:
            flat_features: [B, N, D] flattened features
            multiscale: Dict[scale -> [B, H, W, D]] multi-scale features
        """
        if self.use_dino:
            # DINOv3 encoding via HuggingFace transformers
            B = images.shape[0]
            device = images.device
            
            with torch.no_grad():
                # Convert tensor images [B, C, H, W] to list of PIL images for processor
                # Processor expects PIL images or numpy arrays
                # Must convert from bfloat16 to float32 first
                import torchvision.transforms.functional as TF
                pil_images = [TF.to_pil_image(img.float().cpu().clamp(0, 1)) for img in images]
                
                # Process images
                inputs = self.dino_processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Handle bfloat16 for pixel_values
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.dino.dtype)
                
                outputs = self.dino(**inputs)
                # Get last hidden state (patch tokens), skip CLS token at position 0
                patch_tokens = outputs.last_hidden_state[:, 1:, :]
            
            # Project to our feature_dim  
            features = self.dino_proj(patch_tokens.float())  # [B, N, D]
            
            # Compute spatial dimensions from patch count
            # DINOv3 may output more patches than square (e.g., 200 for 14x~14)
            n_patches = patch_tokens.shape[1]
            h_feat = w_feat = int(n_patches ** 0.5)
            n_square = h_feat * w_feat  # 196
            
            # Truncate to square grid (drop extra patches)
            features_square = features[:, :n_square, :]
            features_spatial = features_square.view(B, h_feat, w_feat, -1)
            
            # Create multi-scale features from spatial
            multiscale = {}
            feat_chw = features_spatial.permute(0, 3, 1, 2)  # [B, D, H, W]
            for scale in self.scales:
                # MPS doesn't support adaptive_avg_pool2d for non-divisible sizes
                # Do pooling on CPU then move back
                feat_cpu = F.adaptive_avg_pool2d(feat_chw.cpu().float(), scale)
                feat_pooled = feat_cpu.to(device).to(features.dtype)
                feat_pooled = feat_pooled.permute(0, 2, 3, 1)  # [B, scale, scale, D]
                multiscale[scale] = feat_pooled
            
            # Flatten for slot attention
            flat = features_spatial.reshape(B, -1, self.feature_dim)
        else:
            # CNN encoding
            feat = self.encoder(images)  # [B, C, H, W]
            B, C, H_feat, W_feat = feat.shape
            features_spatial = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # Create multi-scale features
            multiscale = {}
            for scale in self.scales:
                feat_pooled = F.adaptive_avg_pool2d(feat, scale)
                feat_pooled = feat_pooled.permute(0, 2, 3, 1)  # [B, scale, scale, C]
                multiscale[scale] = feat_pooled
            
            # Flatten for slot attention
            flat = features_spatial.reshape(B, -1, self.feature_dim)
        
        return flat, multiscale
    
    def decode_slots_to_image(
        self, 
        slots: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode slots to reconstructed image via spatial broadcast."""
        B, K, D = slots.shape
        H, W = self.image_size
        N = H * W
        
        # Transform slots
        slot_features = self.slot_decoder(slots)  # [B, K, 256]
        
        # Spatial broadcast
        slot_broadcast = slot_features.unsqueeze(2).expand(-1, -1, N, -1)
        
        # Add positional encoding
        pos = self.pos_grid.expand(B, -1, -1).unsqueeze(1).expand(-1, K, -1, -1)
        
        # Concatenate and decode
        combined = torch.cat([slot_broadcast, pos], dim=-1)
        rgb_per_slot = self.rgb_decoder(combined)
        rgb_per_slot = rgb_per_slot.view(B, K, H, W, 3).permute(0, 1, 4, 2, 3)
        
        # Combine using masks
        masks_expanded = masks.unsqueeze(2)
        reconstructed = (rgb_per_slot * masks_expanded).sum(dim=1)
        
        return reconstructed, rgb_per_slot
    
    def forward(
        self, 
        images: torch.Tensor, 
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary with slots, masks, reconstructed, and optionally loss
        """
        B = images.shape[0]
        device = images.device
        
        # 1. Encode features
        flat_features, multiscale = self.encode_features(images)
        
        # 2. Slot initialization
        if self.init_mode == "spectral":
            slots_init = self.spectral_init(multiscale, mode="spectral")
        elif self.init_mode == "random":
            slots_init = self.spectral_init(multiscale, mode="random")
        else:  # learned
            slots_init = self.slot_mu + self.slot_sigma * torch.randn_like(self.slot_mu)
            slots_init = slots_init.expand(B, -1, -1).to(device)
        
        # 3. Mamba-Slot Attention
        slots, attn_masks = self.slot_attention(flat_features, slots_init)
        
        # 4. Pruning
        slots, attn_masks, prune_info = self.pruning(slots, attn_masks)
        
        # 5. Decode to masks
        output_masks = self.mask_decoder(slots)
        
        # 6. Decode to reconstructed image
        reconstructed, rgb_per_slot = self.decode_slots_to_image(slots, output_masks)
        
        outputs = {
            'slots': slots,
            'masks': output_masks,
            'attention_masks': attn_masks,
            'reconstructed': reconstructed,
            'slots_init': slots_init,
        }
        
        if return_loss:
            loss, loss_dict = self._compute_loss(
                images, output_masks, slots, reconstructed, 
                multiscale, slots_init
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute full training loss with all components."""
        B, K, H, W = masks.shape
        loss_dict = {}
        
        # 1. RECONSTRUCTION LOSS (Primary)
        recon_loss = F.mse_loss(reconstructed, images)
        loss_dict['recon'] = recon_loss.item()
        
        # 2. COVERAGE LOSS
        mask_sum = masks.sum(dim=1)
        coverage_loss = ((mask_sum - 1) ** 2).mean()
        loss_dict['coverage'] = coverage_loss.item()
        
        # 3. OVERLAP LOSS
        mask_sq_sum = (masks ** 2).sum(dim=1)
        overlap_loss = ((mask_sum ** 2 - mask_sq_sum) ** 2).mean()
        loss_dict['overlap'] = overlap_loss.item()
        
        # 4. SLOT DIVERSITY LOSS (from module)
        diversity_loss = self.diversity_loss_fn(slots)
        loss_dict['diversity'] = diversity_loss.item()
        
        # 5. ENTROPY LOSS
        entropy = -(masks * torch.log(masks + 1e-8)).sum(dim=1).mean()
        loss_dict['entropy'] = entropy.item()
        
        # 6. IDENTIFIABILITY LOSS (GMM prior - NeurIPS 2024)
        if self.training:
            ident_loss = self.ident_loss_fn(slots, update_prior=True)
        else:
            ident_loss = self.ident_loss_fn(slots, update_prior=False)
        loss_dict['ident'] = ident_loss.item()
        
        # 7. SPECTRAL CONSISTENCY LOSS
        # Get spectral masks and compare with slot masks
        try:
            spectral_masks = self.spectral_init.get_spectral_masks(multiscale)
            # Resize spectral masks to match output masks
            spectral_masks_resized = F.interpolate(
                spectral_masks, size=(H, W), mode='bilinear', align_corners=False
            )
            # Normalize
            spectral_masks_resized = F.softmax(spectral_masks_resized, dim=1)
            spec_loss = F.mse_loss(masks, spectral_masks_resized)
            loss_dict['spectral'] = spec_loss.item()
        except Exception:
            spec_loss = torch.tensor(0.0, device=masks.device)
            loss_dict['spectral'] = 0.0
        
        # Total loss with weights
        total_loss = (
            10.0 * recon_loss +
            1.0 * coverage_loss +
            0.5 * overlap_loss +
            self.lambda_diversity * diversity_loss +
            0.1 * entropy +
            self.lambda_ident * ident_loss +
            self.lambda_spec * spec_loss
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def create_full_model(args) -> nn.Module:
    """Factory function to create full SpectralDiffusion model."""
    model = FullSpectralDiffusion(
        image_size=tuple(args.image_size),
        num_slots=args.num_slots,
        feature_dim=256,
        use_dino=getattr(args, 'use_dino', False),  # Use DINOv3 if requested
        use_mamba=args.use_mamba,
        init_mode=args.init_mode,
        scales=[8, 16] if args.image_size[0] <= 128 else [8, 16, 32],
        lambda_spec=args.lambda_spec,
        lambda_ident=args.lambda_ident,
    )
    return model.to(args.device)
