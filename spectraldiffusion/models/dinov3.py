"""
DINOv3 Feature Extractor for SpectralDiffusion

Wrapper for DINOv3 models from Meta (released August 2025).
Supports small, base, and large variants for ablation studies.
Multi-scale feature extraction with frozen backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoImageProcessor
from einops import rearrange


class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 Feature Extractor with multi-scale output.
    
    Extracts dense features from DINOv3 ViT models at multiple scales
    using adaptive pooling. Backbone is frozen by default.
    
    Args:
        model_name: Hugging Face model identifier
            - "facebook/dinov3-small-patch14" (384 dim)
            - "facebook/dinov3-base-patch14" (768 dim)
            - "facebook/dinov3-large-patch14" (1024 dim)
        scales: List of spatial scales for multi-scale features
        freeze: Whether to freeze backbone weights
        use_cls_token: Whether to include CLS token in features
        dtype: Model dtype (bfloat16 supported on MPS)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "dinov3-small": {
            "model_id": "facebook/dinov3-small-patch14",
            "dim": 384,
            "patch_size": 14,
        },
        "dinov3-base": {
            "model_id": "facebook/dinov3-base-patch14",
            "dim": 768,
            "patch_size": 14,
        },
        "dinov3-large": {
            "model_id": "facebook/dinov3-large-patch14",
            "dim": 1024,
            "patch_size": 14,
        },
        # DINOv2 for ablation comparison
        "dinov2-base": {
            "model_id": "facebook/dinov2-base",
            "dim": 768,
            "patch_size": 14,
        },
    }
    
    def __init__(
        self,
        model_name: str = "dinov3-base",
        scales: List[int] = [8, 16, 32],
        freeze: bool = True,
        use_cls_token: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        # Get model config
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.scales = scales
        self.use_cls_token = use_cls_token
        self.dtype = dtype
        self.feature_dim = self.config["dim"]
        self.patch_size = self.config["patch_size"]
        
        # Load model and processor
        print(f"Loading {model_name} from {self.config['model_id']}...")
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_id"])
        self.model = AutoModel.from_pretrained(
            self.config["model_id"],
            torch_dtype=dtype,
        )
        
        # Freeze backbone
        if freeze:
            self._freeze_backbone()
        
        # Feature projection for each scale (optional, identity by default)
        self.scale_projections = nn.ModuleDict({
            str(s): nn.Identity() for s in scales
        })
        
        print(f"DINOv3 loaded: dim={self.feature_dim}, patch_size={self.patch_size}")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("Backbone frozen.")
    
    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze last N transformer blocks for fine-tuning ablation."""
        # First freeze all
        self._freeze_backbone()
        
        # Then unfreeze last N blocks
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {n} transformer blocks.")
        else:
            print("Warning: Could not find encoder layers to unfreeze.")
    
    def get_patch_features(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Extract raw patch features from DINOv3.
        
        Args:
            pixel_values: [B, 3, H, W] normalized images
            
        Returns:
            features: [B, N, D] patch features (excluding CLS if not used)
            h_patches: Number of patches in height
            w_patches: Number of patches in width
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Forward pass through DINOv3
            outputs = self.model(pixel_values, output_hidden_states=True)
            
            # Get last hidden state: [B, 1+N, D] where 1 is CLS token
            features = outputs.last_hidden_state
            
            # Compute spatial dimensions
            B, _, D = features.shape
            H, W = pixel_values.shape[2:]
            h_patches = H // self.patch_size
            w_patches = W // self.patch_size
            
            if self.use_cls_token:
                # Keep CLS token prepended
                return features, h_patches, w_patches
            else:
                # Remove CLS token: [B, N, D]
                patch_features = features[:, 1:, :]
                return patch_features, h_patches, w_patches
    
    def extract_multiscale_features(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract multi-scale features via adaptive pooling.
        
        Args:
            pixel_values: [B, 3, H, W] normalized images
            
        Returns:
            Dictionary mapping scale -> [B, scale, scale, D] features
        """
        # Get patch features
        features, h_patches, w_patches = self.get_patch_features(pixel_values)
        B, N, D = features.shape
        
        # Reshape to spatial: [B, H, W, D]
        features_spatial = rearrange(
            features, 'b (h w) d -> b h w d', h=h_patches, w=w_patches
        )
        
        # Extract multi-scale features
        multiscale_features = {}
        for scale in self.scales:
            # Adaptive pooling to target scale
            # Rearrange to [B, D, H, W] for pooling
            feat = rearrange(features_spatial, 'b h w d -> b d h w')
            feat_pooled = F.adaptive_avg_pool2d(feat, (scale, scale))
            # Back to [B, scale, scale, D]
            feat_pooled = rearrange(feat_pooled, 'b d h w -> b h w d')
            
            # Apply scale-specific projection (identity by default)
            feat_pooled = self.scale_projections[str(scale)](feat_pooled)
            
            multiscale_features[scale] = feat_pooled
        
        return multiscale_features
    
    def forward(
        self,
        images: torch.Tensor,
        return_multiscale: bool = True,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass for feature extraction.
        
        Args:
            images: [B, 3, H, W] input images (should be preprocessed)
            return_multiscale: If True, return multi-scale features dict
                             If False, return flattened features [B, N, D]
        
        Returns:
            Multi-scale features dict or flattened features
        """
        # Ensure correct dtype
        if images.dtype != self.dtype:
            images = images.to(self.dtype)
        
        if return_multiscale:
            return self.extract_multiscale_features(images)
        else:
            features, h, w = self.get_patch_features(images)
            return features
    
    def preprocess(
        self,
        images: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess images for DINOv3.
        
        Args:
            images: [B, 3, H, W] images in [0, 1] range
            size: Optional target size (H, W)
            
        Returns:
            Preprocessed images
        """
        # Resize if needed (must be divisible by patch_size)
        if size is not None:
            images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
        
        # Ensure dimensions are divisible by patch_size
        B, C, H, W = images.shape
        new_h = (H // self.patch_size) * self.patch_size
        new_w = (W // self.patch_size) * self.patch_size
        if new_h != H or new_w != W:
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Normalize using ImageNet stats (DINOv3 uses same normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
        images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        return images
    
    @property
    def output_dim(self) -> int:
        """Return the feature dimension."""
        return self.feature_dim
    
    @property
    def num_scales(self) -> int:
        """Return number of scales."""
        return len(self.scales)


def create_dinov3_extractor(
    variant: str = "base",
    scales: List[int] = [8, 16, 32],
    freeze: bool = True,
    device: str = "mps",
) -> DINOv3FeatureExtractor:
    """
    Factory function to create DINOv3 feature extractor.
    
    Args:
        variant: "small", "base", or "large"
        scales: Multi-scale spatial dimensions
        freeze: Whether to freeze backbone
        device: Target device
        
    Returns:
        DINOv3FeatureExtractor instance
    """
    model_name = f"dinov3-{variant}"
    extractor = DINOv3FeatureExtractor(
        model_name=model_name,
        scales=scales,
        freeze=freeze,
        dtype=torch.float32 if device == "mps" else torch.float32,
    )
    return extractor.to(device)


if __name__ == "__main__":
    # Test the feature extractor
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create extractor
    extractor = create_dinov3_extractor(
        variant="base",
        scales=[8, 16, 32],
        device=device,
    )
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 518, 518, device=device)
    dummy_input = extractor.preprocess(dummy_input)
    
    # Extract features
    with torch.no_grad():
        features = extractor(dummy_input, return_multiscale=True)
    
    print("\nMulti-scale features:")
    for scale, feat in features.items():
        print(f"  Scale {scale}: {feat.shape}")
    
    # Test flattened output
    with torch.no_grad():
        flat_features = extractor(dummy_input, return_multiscale=False)
    print(f"\nFlattened features: {flat_features.shape}")
