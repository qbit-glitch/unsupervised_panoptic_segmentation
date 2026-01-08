"""
Latent Diffusion Decoder for SpectralDiffusion

Slot-conditioned diffusion model for high-quality panoptic mask generation.

Based on:
- SlotDiffusion (Wu et al., NeurIPS 2023)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Improved DDPM (Nichol & Dhariwal, 2021)

Key innovations:
- Condition latent diffusion on slot representations
- Cross-attention between denoiser and slots
- DDIM fast sampling for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from einops import rearrange, repeat
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time and slot conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding (scale and shift)
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class SlotCrossAttention(nn.Module):
    """Cross-attention between U-Net features and slots."""
    
    def __init__(
        self,
        channels: int,
        slot_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(slot_dim, channels)
        self.to_v = nn.Linear(slot_dim, channels)
        
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
            slots: [B, K, D] slot representations
            
        Returns:
            out: [B, C, H, W] attended features
        """
        B, C, H, W = x.shape
        K = slots.shape[1]
        
        residual = x
        x = self.norm(x)
        
        # Query from spatial features
        q = self.to_q(x)  # [B, C, H, W]
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        
        # Key, Value from slots
        k = self.to_k(slots)  # [B, K, C]
        v = self.to_v(slots)  # [B, K, C]
        k = rearrange(k, 'b k (h d) -> b h k d', h=self.num_heads)
        v = rearrange(v, 'b k (h d) -> b h k d', h=self.num_heads)
        
        # Attention
        attn = torch.einsum('b h n d, b h k d -> b h n k', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Aggregate
        out = torch.einsum('b h n k, b h k d -> b h n d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        out = self.to_out(out)
        
        return out + residual


class DownBlock(nn.Module):
    """Downsampling block for U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        slot_dim: int,
        has_attn: bool = True,
    ):
        super().__init__()
        
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        
        if has_attn:
            self.attn = SlotCrossAttention(out_channels, slot_dim)
        else:
            self.attn = nn.Identity()
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        
        if isinstance(self.attn, SlotCrossAttention):
            x = self.attn(x, slots)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        slot_dim: int,
        has_attn: bool = True,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # Double channels due to skip connection
        self.res1 = ResBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        
        if has_attn:
            self.attn = SlotCrossAttention(out_channels, slot_dim)
        else:
            self.attn = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        
        if isinstance(self.attn, SlotCrossAttention):
            x = self.attn(x, slots)
        
        return x


class SlotConditionedUNet(nn.Module):
    """
    U-Net conditioned on slots for mask generation.
    
    Args:
        in_channels: Input channels (latent dim for VAE, or 3 for direct)
        out_channels: Output channels (same as in for denoising, or K for masks)
        slot_dim: Slot representation dimension
        num_slots: Number of slots
        base_channels: Base channel count
        channel_mults: Channel multipliers per level
        num_res_blocks: Residual blocks per level
        time_emb_dim: Time embedding dimension
        attn_levels: Which levels have cross-attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        slot_dim: int = 768,
        num_slots: int = 12,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_levels: Tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slots = num_slots
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Slot embedding projection
        self.slot_proj = nn.Linear(slot_dim, slot_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            has_attn = level in attn_levels
            
            self.down_blocks.append(
                DownBlock(ch, out_ch, time_emb_dim, slot_dim, has_attn)
            )
            ch = out_ch
            channels.append(ch)
        
        # Middle block
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim)
        self.mid_attn = SlotCrossAttention(ch, slot_dim)
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            has_attn = level in attn_levels
            
            # Skip from corresponding down block
            skip_ch = channels.pop()
            
            self.up_blocks.append(
                UpBlock(ch, out_ch, time_emb_dim, slot_dim, has_attn)
            )
            ch = out_ch
        
        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise x at timestep t conditioned on slots.
        
        Args:
            x: [B, C, H, W] noisy input
            t: [B] timestep indices
            slots: [B, K, D] slot representations
            
        Returns:
            noise_pred: [B, C, H, W] predicted noise
        """
        # Time embedding
        t_emb = self.time_emb(t)  # [B, time_emb_dim]
        
        # Slot projection
        slots = self.slot_proj(slots)  # [B, K, D]
        
        # Initial conv
        h = self.conv_in(x)
        
        # Downsample
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb, slots)
            skips.append(skip)
        
        # Middle
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, slots)
        h = self.mid_res2(h, t_emb)
        
        # Upsample
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb, slots)
        
        # Output
        out = self.conv_out(h)
        
        return out


class SimpleVAEEncoder(nn.Module):
    """Simple VAE encoder for image -> latent."""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),  # /4
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),  # /8
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1),  # mu and logvar
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SimpleVAEDecoder(nn.Module):
    """Simple VAE decoder for latent -> masks."""
    
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),  # x2
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),  # x4
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),  # x8
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SlotConditionedDiffusion(nn.Module):
    """
    Latent diffusion model conditioned on slots.
    
    Args:
        slot_dim: Slot representation dimension
        num_slots: Number of slots
        latent_channels: Latent space channels
        image_size: Input image size (H, W)
        num_timesteps: Number of diffusion steps
        use_vae: Whether to use VAE for latent space
    """
    
    def __init__(
        self,
        slot_dim: int = 768,
        num_slots: int = 12,
        latent_channels: int = 4,
        image_size: Tuple[int, int] = (256, 256),
        num_timesteps: int = 50,
        use_vae: bool = True,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.use_vae = use_vae
        
        # VAE encoder/decoder
        if use_vae:
            self.vae_encoder = SimpleVAEEncoder(
                in_channels=3,
                latent_channels=latent_channels,
            )
            self.vae_decoder = SimpleVAEDecoder(
                latent_channels=latent_channels,
                out_channels=num_slots,  # Output per-slot mask
            )
        
        # Denoising U-Net
        self.denoiser = SlotConditionedUNet(
            in_channels=latent_channels,
            out_channels=latent_channels,
            slot_dim=slot_dim,
            num_slots=num_slots,
        )
        
        # Cosine noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Precompute useful values
        self.register_buffer('sqrt_alphas_cumprod', 
            torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
            torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', 
            torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 
            torch.sqrt(1.0 / self.alphas_cumprod - 1))
    
    def _cosine_beta_schedule(
        self,
        timesteps: int,
        s: float = 0.008,
    ) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from x_0.
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def p_losses(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            x_0: [B, C, H, W] clean latent
            t: [B] timesteps
            slots: [B, K, D] slot representations
            
        Returns:
            loss: MSE loss between predicted and actual noise
        """
        # Forward diffusion
        x_t, noise = self.q_sample(x_0, t)
        
        # Predict noise
        noise_pred = self.denoiser(x_t, t, slots)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def encode_to_latent(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images to latent space using VAE."""
        if self.use_vae:
            mu, logvar = self.vae_encoder(images)
            z = self.vae_encoder.sample(mu, logvar)
            return z
        else:
            return images
    
    def decode_to_masks(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent to panoptic masks."""
        if self.use_vae:
            masks = self.vae_decoder(z)  # [B, K, H, W]
            masks = torch.sigmoid(masks)  # Normalize to [0, 1]
            return masks
        else:
            return z
    
    @torch.no_grad()
    def ddim_sample(
        self,
        slots: torch.Tensor,
        num_steps: int = 10,
        eta: float = 0.0,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling for fast inference.
        
        Args:
            slots: [B, K, D] slot representations
            num_steps: Number of DDIM steps (default 10 for 5x speedup)
            eta: DDIM eta parameter (0 = deterministic)
            return_all: Whether to return all intermediate samples
            
        Returns:
            samples: [B, K, H, W] generated masks
        """
        B = slots.shape[0]
        device = slots.device
        dtype = slots.dtype
        
        # Compute latent spatial size
        H, W = self.image_size
        latent_h, latent_w = H // 8, W // 8
        
        # Start from random noise
        x = torch.randn(B, self.latent_channels, latent_h, latent_w, 
                       device=device, dtype=dtype)
        
        # DDIM timestep schedule (evenly spaced)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, 
            device=device, dtype=torch.long
        )
        
        all_samples = [x] if return_all else []
        
        for i in range(num_steps):
            t = timesteps[i].expand(B)
            t_prev = timesteps[i + 1].expand(B)
            
            # Predict noise
            noise_pred = self.denoiser(x, t, slots)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)
            
            # DDIM update
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * \
                      torch.sqrt(1 - alpha_t / alpha_t_prev)
            
            # Sample x_{t-1}
            noise = torch.randn_like(x) if i < num_steps - 1 else 0
            x = torch.sqrt(alpha_t_prev) * x_0_pred + \
                torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * noise_pred + \
                sigma_t * noise
            
            if return_all:
                all_samples.append(x)
        
        # Decode to masks
        masks = self.decode_to_masks(x)
        
        if return_all:
            return masks, all_samples
        return masks
    
    def forward(
        self,
        images: torch.Tensor,
        slots: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W] input images
            slots: [B, K, D] slot representations
            return_loss: If True, compute training loss. If False, generate masks.
            
        Returns:
            Dictionary with 'loss' or 'masks'
        """
        B = images.shape[0]
        device = images.device
        
        if return_loss:
            # Training: compute diffusion loss
            # Encode to latent
            z_0 = self.encode_to_latent(images)
            
            # Sample random timesteps
            t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
            
            # Compute loss
            loss = self.p_losses(z_0, t, slots)
            
            return {'loss': loss}
        else:
            # Inference: generate masks
            masks = self.ddim_sample(slots, num_steps=10)
            
            return {'masks': masks}


# Ablation: Simple MLP mask decoder (A3)
class MLPMaskDecoder(nn.Module):
    """
    Simple MLP decoder for ablation (A3).
    
    Baseline for comparing diffusion vs direct mask prediction.
    
    IMPORTANT: Uses softmax over slots (not independent sigmoids) to force
    slot competition. Each pixel is assigned to exactly one slot.
    """
    
    def __init__(
        self,
        slot_dim: int = 768,
        num_slots: int = 12,
        hidden_dim: int = 256,
        image_size: Tuple[int, int] = (256, 256),
        use_softmax: bool = True,  # Force slot competition
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.image_size = image_size
        self.use_softmax = use_softmax
        
        # Slot to spatial features
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Spatial decoder
        H, W = image_size
        self.spatial_size = (H // 8, W // 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # x2
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),  # x4
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),  # x8
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 1, 3, padding=1),  # Per-slot logit (no activation!)
        )
    
    def forward(
        self,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate masks from slots.
        
        Args:
            slots: [B, K, D] slot representations
            
        Returns:
            masks: [B, K, H, W] per-slot masks (softmax normalized across slots)
        """
        B, K, D = slots.shape
        H, W = self.image_size
        
        # Project slots
        slot_features = self.slot_proj(slots)  # [B, K, hidden_dim]
        
        # Reshape for spatial decoding
        slot_features = slot_features.view(B * K, -1, 1, 1)
        slot_features = slot_features.expand(-1, -1, self.spatial_size[0], self.spatial_size[1])
        
        # Decode to mask logits (no activation in decoder)
        mask_logits = self.decoder(slot_features)  # [B*K, 1, H, W]
        mask_logits = mask_logits.view(B, K, H, W)
        
        if self.use_softmax:
            # Temperature scaling - lower temperature = sharper distributions
            # This amplifies small differences between slots
            temperature = 0.1  # Low temp makes softmax more "winner-take-all"
            masks = F.softmax(mask_logits / temperature, dim=1)
        else:
            # Fallback to sigmoid (for backward compatibility)
            masks = torch.sigmoid(mask_logits)
        
        return masks


if __name__ == "__main__":
    # Test diffusion decoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create diffusion model
    print("\n=== Testing SlotConditionedDiffusion ===")
    diffusion = SlotConditionedDiffusion(
        slot_dim=768,
        num_slots=12,
        latent_channels=4,
        image_size=(256, 256),
        num_timesteps=50,
        use_vae=True,
    ).to(device)
    
    # Test inputs
    B = 2
    images = torch.randn(B, 3, 256, 256, device=device)
    slots = torch.randn(B, 12, 768, device=device)
    
    # Training loss
    print("\nTraining forward pass...")
    output = diffusion(images, slots, return_loss=True)
    print(f"Diffusion loss: {output['loss'].item():.4f}")
    
    # Inference
    print("\nInference (DDIM sampling)...")
    with torch.no_grad():
        output = diffusion(images, slots, return_loss=False)
    print(f"Generated masks: {output['masks'].shape}")
    
    # Count parameters
    params = sum(p.numel() for p in diffusion.parameters())
    print(f"Total parameters: {params:,}")
    
    # Test MLP decoder for comparison
    print("\n=== Testing MLPMaskDecoder (ablation) ===")
    mlp_decoder = MLPMaskDecoder(
        slot_dim=768,
        num_slots=12,
        image_size=(256, 256),
    ).to(device)
    
    with torch.no_grad():
        mlp_masks = mlp_decoder(slots)
    print(f"MLP masks: {mlp_masks.shape}")
    
    mlp_params = sum(p.numel() for p in mlp_decoder.parameters())
    print(f"MLP parameters: {mlp_params:,}")
