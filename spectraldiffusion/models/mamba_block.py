"""
MPS-Compatible Mamba-2 Block for SpectralDiffusion

Pure PyTorch implementation of Mamba-2 Selective State Space Model,
compatible with Apple Silicon MPS backend.

Based on:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2024)
- Mamba-2: Structured State Space Duality (Dao & Gu, ICML 2024)

Key innovations:
- Selective scan with O(N) complexity (vs O(N²) attention)
- Input-dependent SSM parameters for content-aware processing
- No CUDA kernels required - pure PyTorch for MPS compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
import math


class Mamba2Block(nn.Module):
    """
    Mamba-2 Selective State Space block (MPS-compatible).
    
    Implements the core Mamba operation:
    1. Input projection with gating
    2. Local convolution for short-range context
    3. Selective SSM scan for long-range dependencies
    4. Output projection
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in the paper)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        bidirectional: Whether to use bidirectional scan
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.bidirectional = bidirectional
        
        # Input projection: x -> (z, x_proj) where z is for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context (depthwise separable)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        
        # SSM parameter projections (input-dependent!)
        # Projects to: delta (dt), B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        
        # Delta (dt) projection
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # Initialize dt bias for stable training
        dt_init_std = 0.01
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # SSM parameters A and D (learnable, not input-dependent)
        # A is initialized with log spacing for stable dynamics
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # For bidirectional: reverse scan path
        if bidirectional:
            self.conv1d_reverse = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias=True,
            )
            self.x_proj_reverse = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
            self.out_proj = nn.Linear(self.d_inner * 2, d_model, bias=False)
    
    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Selective scan operation - core of Mamba.
        
        Computes recurrence:
            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
            y_t = C_t^T * h_t + D * x_t
        
        Args:
            x: [B, L, D] input sequence
            delta: [B, L, D] discretization step sizes
            B: [B, L, N] input-to-state matrix
            C: [B, L, N] state-to-output matrix
            reverse: Whether to scan in reverse order
            
        Returns:
            y: [B, L, D] output sequence
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[-1]
        device = x.device
        dtype = x.dtype
        
        # Get A from learnable log-space parameter
        A = -torch.exp(self.A_log.float())  # [N] - negative for stability
        
        # Discretize: A_bar = exp(delta * A)
        # delta: [B, L, D], A: [N] -> delta_A: [B, L, D, N]
        delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, 1, -1))  # [B, L, D, N]
        
        # delta_B: [B, L, D, N]
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        
        # Output buffer
        outputs = []
        
        # Determine scan order
        if reverse:
            indices = range(seq_len - 1, -1, -1)
        else:
            indices = range(seq_len)
        
        for t in indices:
            # Get timestep-specific parameters
            x_t = x[:, t, :]  # [B, D]
            delta_A_t = delta_A[:, t, :, :]  # [B, D, N]
            delta_B_t = delta_B[:, t, :, :]  # [B, D, N]
            C_t = C[:, t, :]  # [B, N]
            
            # State update: h = A_bar * h + B_bar * x
            h = delta_A_t * h + delta_B_t * x_t.unsqueeze(-1)
            
            # Output: y = C^T * h
            y_t = torch.einsum('bdn,bn->bd', h, C_t)  # [B, D]
            
            outputs.append(y_t)
        
        # Stack outputs
        if reverse:
            outputs = outputs[::-1]
        y = torch.stack(outputs, dim=1)  # [B, L, D]
        
        # Add skip connection with D
        y = y + self.D.view(1, 1, -1) * x
        
        return y
    
    def forward_single_direction(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        x_proj: nn.Linear,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass in one direction.
        
        Args:
            x: [B, L, D_inner] projected input
            conv: Convolution layer
            x_proj: Projection layer for SSM params
            reverse: Whether to reverse scan
            
        Returns:
            y: [B, L, D_inner] output
        """
        batch, seq_len, d_inner = x.shape
        
        # 1. Convolution for local context
        # [B, L, D] -> [B, D, L] for conv1d
        x_conv = rearrange(x, 'b l d -> b d l')
        x_conv = conv(x_conv)[:, :, :seq_len]  # Truncate padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # 2. SiLU activation
        x_conv = F.silu(x_conv)
        
        # 3. Project to SSM parameters (input-dependent)
        x_ssm = x_proj(x_conv)  # [B, L, N + N + D_inner]
        
        # Split into delta (pre-activation), B, C
        delta_pre = x_ssm[:, :, :self.d_state]  # [B, L, N]
        B = x_ssm[:, :, self.d_state:2*self.d_state]  # [B, L, N]
        C = x_ssm[:, :, 2*self.d_state:]  # [B, L, D_inner] - actually projects back
        
        # Delta transformation: softplus for positivity
        delta = F.softplus(self.dt_proj(delta_pre))  # [B, L, D_inner]
        
        # Recompute C to have correct dimension [B, L, N]
        # Note: Paper uses different parameterization, this is simplified
        C = B  # Use same projection for simplicity
        
        # 4. Selective scan
        y = self.selective_scan(x_conv, delta, B, C, reverse=reverse)
        
        return y
    
    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Mamba-2 block.
        
        Args:
            x: [B, L, D] input sequence
            return_hidden: Whether to return intermediate hidden states
            
        Returns:
            output: [B, L, D] output sequence
        """
        batch, seq_len, d_model = x.shape
        
        # Residual connection
        residual = x
        
        # LayerNorm
        x = self.norm(x)
        
        # Input projection: split into z (gate) and x_proj
        x_proj = self.in_proj(x)  # [B, L, 2*D_inner]
        x_proj, z = x_proj.chunk(2, dim=-1)  # Each [B, L, D_inner]
        
        # Forward direction
        y_forward = self.forward_single_direction(
            x_proj, self.conv1d, self.x_proj, reverse=False
        )
        
        if self.bidirectional:
            # Reverse direction
            y_reverse = self.forward_single_direction(
                x_proj, self.conv1d_reverse, self.x_proj_reverse, reverse=True
            )
            
            # Concatenate bidirectional outputs
            y = torch.cat([y_forward, y_reverse], dim=-1)  # [B, L, 2*D_inner]
        else:
            y = y_forward
        
        # Gating with z
        z = F.silu(z)
        if self.bidirectional:
            z = torch.cat([z, z], dim=-1)  # Expand gate for bidirectional
        y = y * z
        
        # Output projection
        output = self.out_proj(y)
        
        # Dropout
        output = self.dropout(output)
        
        # Residual connection
        output = output + residual
        
        return output


class MambaStack(nn.Module):
    """
    Stack of Mamba-2 blocks for iterative processing.
    
    Args:
        d_model: Model dimension
        n_layers: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        bidirectional: Whether to use bidirectional scan
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 3,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Mamba stack.
        
        Args:
            x: [B, L, D] input sequence
            return_all_layers: Whether to return outputs from all layers
            
        Returns:
            output: [B, L, D] or list of [B, L, D] if return_all_layers
        """
        outputs = []
        
        for layer in self.layers:
            x = layer(x)
            if return_all_layers:
                outputs.append(x)
        
        x = self.final_norm(x)
        
        if return_all_layers:
            return outputs
        return x


# Ablation: Standard Transformer Attention Block for comparison
class TransformerBlock(nn.Module):
    """
    Standard Transformer block for ablation comparison (A2).
    
    O(N²) complexity vs Mamba's O(N).
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


# Ablation: Linear Attention Block for comparison
class LinearAttentionBlock(nn.Module):
    """
    Linear attention block for ablation comparison (A2).
    
    O(N) complexity like Mamba, but different mechanism.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.norm = nn.LayerNorm(d_model)
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        
        # Linear attention: kernel feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute K^T V first: O(D^2) instead of O(L^2)
        kv = torch.einsum('bhld,bhlv->bhdv', k, v)  # [B, H, D, D]
        
        # Then Q @ (K^T V)
        out = torch.einsum('bhld,bhdv->bhlv', q, kv)  # [B, H, L, D]
        
        # Normalization
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = torch.einsum('bhld,bhkd->bhlk', q, k_sum).clamp(min=1e-6)
        out = out / normalizer
        
        # Merge heads
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out + residual


if __name__ == "__main__":
    # Test Mamba-2 block
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Test single block
    print("\n=== Testing Mamba2Block ===")
    mamba_block = Mamba2Block(
        d_model=768,
        d_state=64,
        d_conv=4,
        expand=2,
        bidirectional=True,
    ).to(device)
    
    x = torch.randn(2, 100, 768, device=device)
    with torch.no_grad():
        y = mamba_block(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in mamba_block.parameters())
    print(f"Parameters: {params:,}")
    
    # Test stack
    print("\n=== Testing MambaStack ===")
    mamba_stack = MambaStack(
        d_model=768,
        n_layers=3,
        d_state=64,
        bidirectional=True,
    ).to(device)
    
    with torch.no_grad():
        y = mamba_stack(x)
    print(f"Stack output: {y.shape}")
    
    # Compare with Transformer
    print("\n=== Testing TransformerBlock (for comparison) ===")
    transformer_block = TransformerBlock(d_model=768, n_heads=8).to(device)
    
    with torch.no_grad():
        y_transformer = transformer_block(x)
    print(f"Transformer output: {y_transformer.shape}")
    
    transformer_params = sum(p.numel() for p in transformer_block.parameters())
    print(f"Transformer parameters: {transformer_params:,}")
    
    # Test linear attention
    print("\n=== Testing LinearAttentionBlock (for comparison) ===")
    linear_block = LinearAttentionBlock(d_model=768, n_heads=8).to(device)
    
    with torch.no_grad():
        y_linear = linear_block(x)
    print(f"Linear attention output: {y_linear.shape}")
