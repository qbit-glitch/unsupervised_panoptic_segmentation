"""Conv-DoRA: Weight-Decomposed Low-Rank Adaptation with Spatial Conv
for DINOv3 backbone.

Implements three complementary PEFT methods for task-adapting a frozen
DINOv3 ViT-B/16 backbone in the CUPS panoptic segmentation pipeline:

1. DoRA (Liu et al., ICML 2024 Oral, Eq. 5):
   W' = m * (W0 + BA) / ||W0 + BA||_c
   Weight-space magnitude-direction decomposition protects pretrained
   feature scales from noisy pseudo-label corruption.

2. Conv-DoRA (new combination):
   DoRA weight-space normalization + Conv-LoRA activation-space spatial
   refinement. DWConv3x3 in LoRA bottleneck injects spatial inductive
   bias that plain ViT linear layers lack for dense prediction.

3. LoRA (Hu et al., ICLR 2022): standard baseline for ablation.

References:
    - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024
    - Zhu et al., "Conv-LoRA: Convolution Meets LoRA for SAM", ICLR 2024
    - Hayou et al., "LoRA+: Efficient Low Rank Adaptation", ICML 2024
    - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DoRAConfig:
    """Configuration for DoRA injection into a ViT backbone.

    Attributes:
        rank: Low-rank dimension r. Default 4.
        alpha: Scaling factor alpha. Default 4.0 (alpha/r = 1.0).
        dropout: Dropout on LoRA path. Default 0.05.
        late_block_start: First block index for full (attn+MLP) adaptation.
            Blocks 0..late_block_start-1 get qkv-only DoRA.
            Blocks late_block_start..11 get qkv+proj+fc1+fc2. Default 6.
        lr_a: Learning rate for A matrices (down-projection). Default 1e-5.
        lr_b: Learning rate for B matrices + magnitude vectors. Default 5e-5.
        magnitude_wd: Weight decay for magnitude vector m. Default 1e-3.
        delayed_start_steps: Steps before DoRA params start training. Default 500.
    """

    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.05
    late_block_start: int = 6
    lr_a: float = 1e-5
    lr_b: float = 5e-5
    magnitude_wd: float = 1e-3
    delayed_start_steps: int = 500


@dataclass(frozen=True)
class MitigationConfig:
    """Configuration for noise-robustness mitigations (Exps 15-21).

    Each mitigation is independently toggleable. All default OFF to preserve
    backward compatibility with existing experiments.

    Attributes:
        cosine_warmup_enabled: M1 — ramp LoRA LR from 0 via cosine schedule.
        cosine_warmup_steps: Steps over which to ramp LR.
        magnitude_warmup_enabled: M2 — freeze magnitude vector m for N steps.
        magnitude_warmup_freeze_steps: Steps to freeze m per round.
        spectral_norm_ball_enabled: M3 — project m onto norm ball after each step.
        spectral_norm_ball_delta: Max fractional drift from initial norm.
        swa_enabled: M4 — average LoRA params over last fraction of round.
        swa_fraction: Fraction of round steps to average over.
        confidence_weighted_loss_enabled: M5 — weight sem loss by teacher confidence.
        confidence_weighted_loss_temperature: Softmax temperature for confidence.
        confidence_weighted_loss_min_weight: Floor weight (no pixel fully zeroed).
        adaptive_delayed_start_enabled: M6 — activate LoRA when head loss converges.
        adaptive_delayed_start_tau: Activate when loss < tau * initial_loss.
        adaptive_delayed_start_max_wait: Hard fallback step limit.
    """

    cosine_warmup_enabled: bool = False
    cosine_warmup_steps: int = 500
    magnitude_warmup_enabled: bool = False
    magnitude_warmup_freeze_steps: int = 200
    spectral_norm_ball_enabled: bool = False
    spectral_norm_ball_delta: float = 0.1
    swa_enabled: bool = False
    swa_fraction: float = 0.3
    confidence_weighted_loss_enabled: bool = False
    confidence_weighted_loss_temperature: float = 1.0
    confidence_weighted_loss_min_weight: float = 0.1
    adaptive_delayed_start_enabled: bool = False
    adaptive_delayed_start_tau: float = 0.7
    adaptive_delayed_start_max_wait: int = 1000

    @classmethod
    def from_cfg(cls, cfg) -> "MitigationConfig":
        """Construct from a yacs CfgNode (MODEL.LORA.MITIGATIONS)."""
        m = cfg.MODEL.LORA.MITIGATIONS
        return cls(
            cosine_warmup_enabled=m.COSINE_WARMUP.ENABLED,
            cosine_warmup_steps=m.COSINE_WARMUP.WARMUP_STEPS,
            magnitude_warmup_enabled=m.MAGNITUDE_WARMUP.ENABLED,
            magnitude_warmup_freeze_steps=m.MAGNITUDE_WARMUP.FREEZE_STEPS,
            spectral_norm_ball_enabled=m.SPECTRAL_NORM_BALL.ENABLED,
            spectral_norm_ball_delta=m.SPECTRAL_NORM_BALL.DELTA,
            swa_enabled=m.SWA.ENABLED,
            swa_fraction=m.SWA.FRACTION,
            confidence_weighted_loss_enabled=m.CONFIDENCE_WEIGHTED_LOSS.ENABLED,
            confidence_weighted_loss_temperature=m.CONFIDENCE_WEIGHTED_LOSS.TEMPERATURE,
            confidence_weighted_loss_min_weight=m.CONFIDENCE_WEIGHTED_LOSS.MIN_WEIGHT,
            adaptive_delayed_start_enabled=m.ADAPTIVE_DELAYED_START.ENABLED,
            adaptive_delayed_start_tau=m.ADAPTIVE_DELAYED_START.TAU,
            adaptive_delayed_start_max_wait=m.ADAPTIVE_DELAYED_START.MAX_WAIT_STEPS,
        )


@dataclass(frozen=True)
class ProgressiveDoRAConfig:
    """Per-round configuration for progressive LoRA in Stage-3 self-training.

    Inspired by Filatov & Kindulov (2023) who showed progressive rank
    expansion (32→32→64) + layer coverage expansion across self-training
    rounds achieves mIoU=0.515, beating EMA (0.513).

    Mathematical basis: as pseudo-label noise ε_t decreases across rounds
    (EMA teacher improves), the capacity-noise tradeoff shifts, allowing
    higher rank and broader layer coverage while maintaining stable SNR:

        SNR(r_t, ε_t) ≈ SNR_full × √(d_in / r_t) × √(ε_1 / ε_t)

    Attributes:
        ranks: Rank per round, e.g. (2, 4, 8) for 3 rounds.
        alphas: Alpha per round. Must match len(ranks). Typically alpha=rank.
        late_block_starts: late_block_start per round. Lower = more blocks.
            e.g. (9, 6, 0) expands from late-only to all-blocks.
        attn_only_rounds: Set of round indices (0-based) where only attn
            layers (qkv, proj) are adapted (no MLP). E.g. {0} for round 1.
        variant: Adapter variant ("dora", "conv_dora", "lora").
        dropout: Dropout on LoRA path.
        lr_a: Learning rate for A matrices.
        lr_b: Learning rate for B + magnitude.
        magnitude_wd: Weight decay for DoRA magnitude vector.
    """

    ranks: Tuple[int, ...] = (2, 4, 8)
    alphas: Tuple[float, ...] = (2.0, 4.0, 8.0)
    late_block_starts: Tuple[int, ...] = (6, 6, 0)
    attn_only_rounds: Tuple[int, ...] = ()
    variant: str = "conv_dora"
    dropout: float = 0.05
    lr_a: float = 1e-5
    lr_b: float = 5e-5
    magnitude_wd: float = 1e-3

    @property
    def num_rounds(self) -> int:
        return len(self.ranks)

    def get_dora_config(self, round_idx: int) -> DoRAConfig:
        """Build a DoRAConfig for a specific self-training round.

        Args:
            round_idx: 0-based round index.

        Returns:
            DoRAConfig for this round's rank/alpha/coverage.
        """
        if round_idx >= self.num_rounds:
            raise IndexError(
                f"Round {round_idx} out of range (have {self.num_rounds} rounds)"
            )
        return DoRAConfig(
            rank=self.ranks[round_idx],
            alpha=self.alphas[round_idx],
            dropout=self.dropout,
            late_block_start=self.late_block_starts[round_idx],
            lr_a=self.lr_a,
            lr_b=self.lr_b,
            magnitude_wd=self.magnitude_wd,
            delayed_start_steps=0,  # no delay in Stage-3 (warmed up from Stage-2)
        )


class DoRALinear(nn.Module):
    """DoRA-adapted linear layer (Liu et al., ICML 2024 Oral, Eq. 5).

    Wraps an existing nn.Linear by freezing its weight as the direction
    matrix V, adding a trainable magnitude vector m and low-rank LoRA
    matrices A, B for directional adaptation.

    Forward pass:
        V' = W0 + (alpha/r) * B @ A       (adapted direction)
        W' = m * V' / ||V'||_c             (magnitude-direction recombination)
        h  = W' @ x + bias

    Initialization ensures W' = W0 at step 0 (B initialized to zero).

    Args:
        wrapped: The original nn.Linear to wrap.
        rank: LoRA rank r.
        alpha: Scaling factor alpha.
        dropout: Dropout probability on the LoRA path.
    """

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight and bias
        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable magnitude vector m (Liu et al., Eq. 5):
        # m = ||W0||_c (column-wise L2 norms of W0)
        # Shape: (1, out_features) for row-wise norm since PyTorch Linear
        # stores weight as (out_features, in_features)
        self.lora_magnitude = nn.Parameter(
            self.weight.data.norm(dim=1, keepdim=True).clone()
        )  # (out_features, 1)

        # Store initial magnitude norm for M3 spectral norm ball projection
        self.register_buffer(
            "_m_init_norm", self.lora_magnitude.data.norm().clone()
        )

        # Low-rank direction update: delta_V = B @ A
        # A: down-projection (rank, in_features) — Gaussian init
        # B: up-projection (out_features, rank) — zero init
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directional update: V' = W0 + scaling * B @ A
        delta_V = self.lora_dropout(self.scaling * (self.lora_B @ self.lora_A))
        V_prime = self.weight + delta_V  # (out_features, in_features)

        # Row-wise normalization (each output neuron is a row in W)
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (out_features, 1)

        # Magnitude-direction recombination (Liu et al., Eq. 5)
        # Using the memory-efficient formulation (Liu et al., Eq. 11):
        # treat ||V'||_c as constant C during backward (detach)
        W_prime = self.lora_magnitude * (V_prime / V_norm.detach())

        return F.linear(x, W_prime, self.bias)

    def trainable_count(self) -> int:
        """Number of trainable parameters."""
        return (
            self.lora_A.numel()
            + self.lora_B.numel()
            + self.lora_magnitude.numel()
        )


class LoRALinear(nn.Module):
    """Standard LoRA-adapted linear layer (Hu et al., ICLR 2022, Eq. 3).

    For ablation: compare DoRA vs standard LoRA.

    Forward: h = W0 @ x + (alpha/r) * B @ A @ x + bias
    """

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(wrapped.weight.data.clone(), requires_grad=False)
        if wrapped.bias is not None:
            self.bias = nn.Parameter(wrapped.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(
            F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        )
        return base_out + lora_out

    def trainable_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


class ConvDoRALinear(DoRALinear):
    """Conv-DoRA: DoRA weight-space + DWConv activation-space refinement.

    Combines DoRA magnitude-direction decomposition (Liu et al., ICML 2024)
    with Conv-LoRA spatial inductive bias (Zhu et al., ICLR 2024).

    DoRA operates in weight-space (protects pretrained magnitudes).
    The conv path operates in activation-space (adds local spatial context
    that plain ViT linear layers lack for dense prediction).

    Forward:
        dora_out = DoRA(x)                               # weight-space
        z = A @ x → reshape(B,r,h,w) → DWConv3x3        # spatial conv
        conv_out = B @ flatten(z_conv) * scaling          # up-project
        output = dora_out + conv_gate * conv_out          # gated addition

    The conv path is zero-initialized (DWConv weights=0, conv_gate=0),
    so at step 0: output == DoRA(x) == W0(x). The conv refinement
    gradually activates during training.

    Spatial dimensions (h_p, w_p) must be set before each forward pass
    via the ``_spatial_dims`` attribute. When unavailable (e.g. CLS token
    or non-spatial input), falls back to standard DoRA.

    Args:
        wrapped: The original nn.Linear to wrap.
        rank: LoRA rank r.
        alpha: Scaling factor alpha.
        dropout: Dropout probability on the LoRA path.
    """

    def __init__(
        self,
        wrapped: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(wrapped, rank=rank, alpha=alpha, dropout=dropout)

        # Depthwise conv on r-channel intermediate features
        # Groups=rank → each rank channel gets its own 3x3 filter
        self.dwconv = nn.Conv2d(
            rank, rank, kernel_size=3, padding=1, groups=rank, bias=False,
        )
        nn.init.zeros_(self.dwconv.weight)  # zero-init for identity at step 0

        # Learnable gate — starts at 0 so conv path contributes nothing initially
        self.conv_gate = nn.Parameter(torch.zeros(1))

        # Spatial dims set by backbone before forward (h_patches, w_patches)
        self._spatial_dims: Optional[Tuple[int, int]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with DoRA weight-space + optional conv activation-space.

        Args:
            x: Input tensor (batch, seq_len, in_features).

        Returns:
            Output tensor (batch, seq_len, out_features).
        """
        # --- DoRA weight-space path (same as DoRALinear) ---
        delta_V = self.lora_dropout(self.scaling * (self.lora_B @ self.lora_A))
        V_prime = self.weight + delta_V
        V_norm = V_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_prime = self.lora_magnitude * (V_prime / V_norm.detach())
        dora_out = F.linear(x, W_prime, self.bias)

        # --- Conv activation-space path ---
        h_p, w_p = self._spatial_dims or (None, None)
        if h_p is not None and w_p is not None:
            B_size = x.shape[0]
            # Down-project: (B, N, in) → (B, N, r)
            z = F.linear(x, self.lora_A)
            # Reshape to spatial grid: (B, N, r) → (B, r, h_p, w_p)
            z_2d = z.transpose(1, 2).reshape(B_size, self.rank, h_p, w_p)
            # DWConv3x3 for local spatial context
            z_conv = self.dwconv(z_2d)
            # Flatten back: (B, r, h_p, w_p) → (B, N, r)
            z_flat = z_conv.reshape(B_size, self.rank, -1).transpose(1, 2)
            # Up-project: (B, N, r) → (B, N, out)
            conv_out = F.linear(z_flat, self.lora_B) * self.scaling
            dora_out = dora_out + self.conv_gate * conv_out

        return dora_out

    def trainable_count(self) -> int:
        """Number of trainable parameters (DoRA + conv)."""
        base = super().trainable_count()
        conv_params = self.dwconv.weight.numel() + self.conv_gate.numel()
        return base + conv_params


def inject_dora_into_model(
    model: nn.Module,
    config: DoRAConfig,
    variant: str = "dora",
) -> Dict[str, int]:
    """Inject DoRA/Conv-DoRA/LoRA adapters into a DINOv3 ViT backbone.

    Walks model.blocks[i].attn.{qkv, proj} and model.blocks[i].mlp.{fc1, fc2},
    replacing target nn.Linear layers with adapted wrappers.

    Tiered strategy:
        - Blocks 0..late_block_start-1: qkv only (minimal early-block steering)
        - Blocks late_block_start..N: qkv + proj + fc1 + fc2 (full adaptation)

    Args:
        model: The DINOv3 ViT model (e.g., DinoVisionTransformer).
        config: DoRA configuration.
        variant: "dora", "conv_dora", or "lora" (for ablation).

    Returns:
        Dict mapping adapted layer names to their trainable param counts.
    """
    _variant_map = {
        "dora": DoRALinear,
        "conv_dora": ConvDoRALinear,
        "lora": LoRALinear,
    }
    if variant not in _variant_map:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(_variant_map)}"
        )
    adapter_cls = _variant_map[variant]
    adapted: Dict[str, int] = {}

    # Find transformer blocks
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        blocks = model.encoder.layer
    else:
        logger.warning("Cannot find transformer blocks in model. No DoRA injected.")
        return adapted

    n_blocks = len(blocks)
    logger.info(
        "Injecting %s (r=%d, alpha=%.1f) into %d blocks "
        "(early[0:%d]=qkv-only, late[%d:%d]=full)",
        variant.upper(), config.rank, config.alpha,
        n_blocks, config.late_block_start,
        config.late_block_start, n_blocks,
    )

    for block_idx, block in enumerate(blocks):
        is_late = block_idx >= config.late_block_start

        # Target layers for this block
        if is_late:
            targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        else:
            targets = ["attn.qkv"]

        for target_path in targets:
            parts = target_path.split(".")
            # Navigate to parent
            parent = block
            for part in parts[:-1]:
                if hasattr(parent, part):
                    parent = getattr(parent, part)
                else:
                    logger.warning(
                        "Block %d: cannot find '%s' in path '%s'",
                        block_idx, part, target_path,
                    )
                    parent = None
                    break

            if parent is None:
                continue

            attr_name = parts[-1]
            original = getattr(parent, attr_name, None)
            if original is None or not isinstance(original, nn.Linear):
                logger.warning(
                    "Block %d: '%s' is not nn.Linear (got %s), skipping",
                    block_idx, target_path,
                    type(original).__name__ if original else "None",
                )
                continue

            # Create adapter wrapping the original layer
            adapter = adapter_cls(
                wrapped=original,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )

            # Replace in the model
            setattr(parent, attr_name, adapter)
            full_name = f"blocks.{block_idx}.{target_path}"
            adapted[full_name] = adapter.trainable_count()
            logger.debug(
                "  %s: %s (%d, %d) → %s r=%d [+%d params]",
                full_name,
                type(original).__name__,
                original.in_features,
                original.out_features,
                variant.upper(),
                config.rank,
                adapter.trainable_count(),
            )

    total_dora = sum(adapted.values())
    total_model = sum(p.numel() for p in model.parameters())
    logger.info(
        "%s injection complete: %d layers adapted, "
        "+%d trainable params (%.2f%% of %dM backbone)",
        variant.upper(),
        len(adapted),
        total_dora,
        total_dora / total_model * 100,
        total_model // 1_000_000,
    )

    return adapted


def get_dora_param_groups(
    model: nn.Module,
    config: DoRAConfig,
    head_lr: float = 1e-4,
    head_wd: float = 1e-5,
) -> List[Dict]:
    """Build optimizer parameter groups with differential LR for DoRA.

    Returns up to 6 groups:
        1. Non-norm head/FPN params — head_lr, head_wd
        2. Norm head/FPN params — head_lr, 0.0
        3. DoRA B — config.lr_b, 0.0
        4. DoRA magnitude — config.lr_b, magnitude_wd
        5. DoRA A — config.lr_a, 0.0
        6. Conv params (dwconv + conv_gate) — config.lr_a, 0.0
           (same LR as A: conv refines the down-projected space)

    Args:
        model: The full panoptic model (backbone + FPN + heads).
        config: DoRA configuration.
        head_lr: Learning rate for detection heads.
        head_wd: Weight decay for detection heads.

    Returns:
        List of param group dicts for torch.optim.AdamW.
    """
    dora_a_params: List[torch.Tensor] = []
    dora_b_params: List[torch.Tensor] = []
    dora_magnitude_params: List[torch.Tensor] = []
    dora_conv_params: List[torch.Tensor] = []
    head_norm_params: List[torch.Tensor] = []
    head_other_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_A" in name:
            dora_a_params.append(param)
        elif "lora_B" in name:
            dora_b_params.append(param)
        elif "lora_magnitude" in name:
            dora_magnitude_params.append(param)
        elif "dwconv" in name or "conv_gate" in name:
            dora_conv_params.append(param)
        elif "norm" in name.lower() or "bn" in name.lower():
            head_norm_params.append(param)
        else:
            head_other_params.append(param)

    groups = []
    if head_other_params:
        groups.append({
            "params": head_other_params,
            "lr": head_lr,
            "weight_decay": head_wd,
            "name": "head_other",
        })
    if head_norm_params:
        groups.append({
            "params": head_norm_params,
            "lr": head_lr,
            "weight_decay": 0.0,
            "name": "head_norm",
        })
    if dora_b_params:
        groups.append({
            "params": dora_b_params,
            "lr": config.lr_b,
            "weight_decay": 0.0,
            "name": "dora_B",
        })
    if dora_magnitude_params:
        groups.append({
            "params": dora_magnitude_params,
            "lr": config.lr_b,
            "weight_decay": config.magnitude_wd,
            "name": "dora_magnitude",
        })
    if dora_a_params:
        groups.append({
            "params": dora_a_params,
            "lr": config.lr_a,
            "weight_decay": 0.0,
            "name": "dora_A",
        })
    if dora_conv_params:
        groups.append({
            "params": dora_conv_params,
            "lr": config.lr_a,
            "weight_decay": 0.0,
            "name": "dora_conv",
        })

    # Log summary
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        logger.info(
            "Param group '%s': %d params, lr=%.1e, wd=%.1e",
            g["name"], n_params, g["lr"], g["weight_decay"],
        )

    return groups


# ---------------------------------------------------------------------------
# Noise-robustness mitigations (Exps 15-21)
# ---------------------------------------------------------------------------


@torch.no_grad()
def spectral_norm_project(model: nn.Module, delta: float) -> int:
    """M3: Project all DoRA magnitude vectors onto norm ball.

    For each DoRA adapter, ensures ||m||_2 <= ||m_init||_2 * (1 + delta).
    This bounds magnitude drift to delta fraction of the initial norm,
    preventing the full-rank magnitude vector from drifting under noisy
    pseudo-label gradients.

    Args:
        model: Model (or backbone) containing DoRALinear modules.
        delta: Max fractional drift from initial norm (e.g., 0.1 = 10%).

    Returns:
        Number of adapters that were actually projected (exceeded bound).
    """
    count = 0
    for module in model.modules():
        if isinstance(module, DoRALinear) and hasattr(module, "_m_init_norm"):
            m = module.lora_magnitude
            max_norm = module._m_init_norm * (1.0 + delta)
            current_norm = m.data.norm().clamp(min=1e-8)
            if current_norm > max_norm:
                m.data.mul_(max_norm / current_norm)
                count += 1
    return count


class SWAAccumulator:
    """M4: Stochastic Weight Averaging for LoRA parameters.

    Maintains a running sum of LoRA parameter snapshots. Call update()
    during the last fraction of each self-training round, then apply()
    at the round boundary to swap in averaged weights.

    Must be reset() after each round boundary (and before rank expansion).
    """

    def __init__(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._count: int = 0

    @property
    def count(self) -> int:
        return self._count

    def update(self, model: nn.Module) -> None:
        """Accumulate current LoRA param snapshot into running sum."""
        for name, module in model.named_modules():
            if not isinstance(module, (DoRALinear, LoRALinear)):
                continue
            for pname, param in module.named_parameters(recurse=False):
                key = f"{name}.{pname}"
                if key not in self._sum:
                    self._sum[key] = param.data.clone()
                else:
                    self._sum[key].add_(param.data)
        self._count += 1

    @torch.no_grad()
    def apply(self, model: nn.Module) -> int:
        """Write averaged params into model. Returns count of params updated."""
        if self._count == 0:
            return 0
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, (DoRALinear, LoRALinear)):
                continue
            for pname, param in module.named_parameters(recurse=False):
                key = f"{name}.{pname}"
                if key in self._sum:
                    param.data.copy_(self._sum[key] / self._count)
                    count += 1
        logger.info(
            "SWA applied: averaged %d params over %d snapshots",
            count, self._count,
        )
        return count

    def reset(self) -> None:
        """Clear accumulated state for next round."""
        self._sum.clear()
        self._count = 0


def cosine_warmup_lambda(
    step: int,
    warmup_steps: int,
    is_lora_group: bool,
) -> float:
    """M1: Cosine warmup LR multiplier for LambdaLR scheduler.

    Args:
        step: Current training step.
        warmup_steps: Steps over which to ramp from 0 to 1.
        is_lora_group: If True, apply warmup. If False, return 1.0 (head groups).

    Returns:
        LR multiplier in [0, 1].
    """
    if not is_lora_group:
        return 1.0
    if warmup_steps <= 0:
        return 1.0
    t = min(step, warmup_steps) / warmup_steps
    return 0.5 * (1.0 - math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Progressive LoRA: rank expansion + layer coverage expansion
# ---------------------------------------------------------------------------


def _expand_adapter_rank(
    adapter: nn.Module,
    new_rank: int,
    new_alpha: float,
) -> None:
    """Expand an existing LoRA/DoRA/ConvDoRA adapter to a higher rank in-place.

    Preserves learned weights: existing A rows and B columns are kept,
    new dimensions are Kaiming-init (A) / zero-init (B) so the model
    output is unchanged at the expansion boundary.

    Args:
        adapter: A DoRALinear, LoRALinear, or ConvDoRALinear instance.
        new_rank: Target rank (must be >= adapter.rank).
        new_alpha: New alpha value (typically new_alpha = new_rank).
    """
    old_rank = adapter.rank
    if new_rank <= old_rank:
        return  # nothing to expand

    device = adapter.lora_A.device
    dtype = adapter.lora_A.dtype

    # Expand A: (old_rank, in_features) → (new_rank, in_features)
    A_new = torch.empty(new_rank, adapter.in_features, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(A_new, a=math.sqrt(5))
    A_new[:old_rank, :] = adapter.lora_A.data
    adapter.lora_A = nn.Parameter(A_new)

    # Expand B: (out_features, old_rank) → (out_features, new_rank)
    B_new = torch.zeros(adapter.out_features, new_rank, device=device, dtype=dtype)
    B_new[:, :old_rank] = adapter.lora_B.data
    adapter.lora_B = nn.Parameter(B_new)

    # Update rank and scaling
    adapter.rank = new_rank
    adapter.scaling = new_alpha / new_rank

    # Expand ConvDoRA DWConv if applicable
    if isinstance(adapter, ConvDoRALinear) and hasattr(adapter, "dwconv"):
        old_conv = adapter.dwconv
        new_conv = nn.Conv2d(
            new_rank, new_rank, kernel_size=3, padding=1,
            groups=new_rank, bias=False,
        ).to(device=device, dtype=dtype)
        nn.init.zeros_(new_conv.weight)
        # Copy existing channels
        new_conv.weight.data[:old_rank] = old_conv.weight.data
        adapter.dwconv = new_conv

    logger.debug(
        "Expanded adapter r=%d→%d (alpha=%.1f), in=%d out=%d",
        old_rank, new_rank, new_alpha,
        adapter.in_features, adapter.out_features,
    )


def expand_lora_rank(
    model: nn.Module,
    new_rank: int,
    new_alpha: float,
) -> int:
    """Expand all LoRA/DoRA adapters in the model to a higher rank.

    Walks model.modules() to find all DoRALinear/LoRALinear/ConvDoRALinear
    instances and expands them in-place. Preserves learned weights.

    Args:
        model: Model containing adapted layers.
        new_rank: Target rank.
        new_alpha: New alpha (typically new_alpha = new_rank).

    Returns:
        Number of adapters expanded.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (DoRALinear, LoRALinear)):
            if module.rank < new_rank:
                _expand_adapter_rank(module, new_rank, new_alpha)
                count += 1
    logger.info("Expanded %d adapters to rank=%d, alpha=%.1f", count, new_rank, new_alpha)
    return count


def expand_lora_coverage(
    model: nn.Module,
    new_config: DoRAConfig,
    variant: str = "conv_dora",
) -> Dict[str, int]:
    """Expand LoRA layer coverage without touching existing adapters.

    Injects new adapters into layers that are currently plain nn.Linear
    (not yet adapted). Already-adapted layers are left unchanged.

    This is used when expanding from e.g. late-block-only to all-blocks:
    the new early-block layers get fresh zero-initialized adapters while
    late-block layers keep their learned weights.

    Args:
        model: Model with some layers already adapted.
        new_config: DoRAConfig with the expanded coverage settings.
        variant: Adapter variant.

    Returns:
        Dict of newly adapted layer names → trainable param counts.
    """
    _variant_map = {
        "dora": DoRALinear,
        "conv_dora": ConvDoRALinear,
        "lora": LoRALinear,
    }
    adapter_cls = _variant_map[variant]
    newly_adapted: Dict[str, int] = {}

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        encoder = getattr(model, "encoder", None)
        blocks = getattr(encoder, "layer", None) if encoder else None
    if blocks is None:
        logger.warning("Cannot find blocks for coverage expansion.")
        return newly_adapted

    for block_idx, block in enumerate(blocks):
        is_late = block_idx >= new_config.late_block_start
        targets = (
            ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
            if is_late else ["attn.qkv"]
        )

        for target_path in targets:
            parts = target_path.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break
            if parent is None:
                continue

            attr_name = parts[-1]
            current = getattr(parent, attr_name, None)

            # Skip if already adapted
            if isinstance(current, (DoRALinear, LoRALinear)):
                continue

            # Only adapt plain nn.Linear
            if not isinstance(current, nn.Linear):
                continue

            adapter = adapter_cls(
                wrapped=current,
                rank=new_config.rank,
                alpha=new_config.alpha,
                dropout=new_config.dropout,
            )
            setattr(parent, attr_name, adapter)
            full_name = f"blocks.{block_idx}.{target_path}"
            newly_adapted[full_name] = adapter.trainable_count()
            logger.debug("  Coverage expansion: %s → %s r=%d", full_name, variant, new_config.rank)

    if newly_adapted:
        total_new = sum(newly_adapted.values())
        logger.info(
            "Coverage expansion: +%d new layers, +%d params",
            len(newly_adapted), total_new,
        )
    return newly_adapted
