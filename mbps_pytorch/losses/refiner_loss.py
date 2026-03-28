"""Combined unsupervised loss for M2PR (Mamba2 Panoptic Refiner).

Combines existing losses (STEGO, DepthG, uniformity, boundary, DBC,
discriminative embeddings) with new losses (KL regularization, entropy,
affinity consistency).

All losses are fully unsupervised — no ground truth labels required.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbps_pytorch.models.semantic.stego_loss import (
    stego_loss,
    depth_guided_correlation_loss,
)
from mbps_pytorch.losses.consistency_loss import (
    uniformity_loss,
    boundary_alignment_loss,
    depth_boundary_coherence_loss,
)
from mbps_pytorch.losses.instance_embedding_loss import discriminative_loss


# --- New loss functions ---

def kl_regularization_loss(
    refined_logits: torch.Tensor,
    original_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between refined and original semantic distributions.

    Prevents the refiner from diverging too far from CAUSE predictions.

    Args:
        refined_logits: Refined semantic logits (B, N, C).
        original_logits: Original CAUSE logits (B, N, C).
        temperature: Softmax temperature.

    Returns:
        Scalar KL loss.
    """
    p = F.log_softmax(refined_logits / temperature, dim=-1)
    q = F.softmax(original_logits / temperature, dim=-1)
    kl = F.kl_div(p, q, reduction="batchmean") * (temperature ** 2)
    return kl


def entropy_loss(
    logits: torch.Tensor,
    mode: str = "local",
) -> torch.Tensor:
    """Entropy-based regularization.

    Args:
        logits: Semantic logits (B, N, C).
        mode: "local" = minimize per-patch entropy (encourage confidence),
              "global" = maximize global class distribution entropy (prevent collapse).

    Returns:
        Scalar entropy loss.
    """
    probs = F.softmax(logits, dim=-1)  # (B, N, C)

    if mode == "local":
        # Minimize per-patch entropy → encourage confident predictions
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, N)
        return entropy.mean()

    elif mode == "global":
        # Maximize global entropy → prevent class collapse
        # Average class distribution across all patches
        mean_probs = probs.mean(dim=(0, 1))  # (C,)
        mean_probs = torch.clamp(mean_probs, 1e-7, 1.0)
        global_entropy = -torch.sum(mean_probs * torch.log(mean_probs))
        # Negate because we want to maximize
        return -global_entropy

    else:
        raise ValueError(f"Unknown entropy mode: {mode}")


def affinity_consistency_loss(
    refined_sem: torch.Tensor,
    inst_embeddings: torch.Tensor,
    dino_features: torch.Tensor,
    num_samples: int = 256,
) -> torch.Tensor:
    """Affinity consistency between refined outputs and DINOv2 features.

    Forces the semantic and instance affinity structures to match DINOv2.

    Args:
        refined_sem: Refined semantic logits (B, N, C).
        inst_embeddings: Instance embeddings (B, N, D_inst).
        dino_features: DINOv2 features (B, N, D_dino).
        num_samples: Number of random pairs to sample per image.

    Returns:
        Scalar affinity consistency loss.
    """
    B, N, _ = dino_features.shape
    device = dino_features.device

    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        # Sample random indices
        idx = torch.randint(0, N, (num_samples,), device=device)

        # DINOv2 affinity (target)
        f_dino = F.normalize(dino_features[b, idx], dim=-1)
        a_dino = f_dino @ f_dino.T  # (S, S)

        # Semantic affinity
        f_sem = F.normalize(F.softmax(refined_sem[b, idx], dim=-1), dim=-1)
        a_sem = f_sem @ f_sem.T

        # Instance affinity
        f_inst = F.normalize(inst_embeddings[b, idx], dim=-1)
        a_inst = f_inst @ f_inst.T

        # MSE between affinities
        total_loss = total_loss + F.mse_loss(a_sem, a_dino.detach())
        total_loss = total_loss + F.mse_loss(a_inst, a_dino.detach())

    return total_loss / B


# --- Combined Refiner Loss ---

class RefinerLoss(nn.Module):
    """Combined unsupervised loss for M2PR.

    Args:
        num_classes: Number of semantic classes (27).
        spatial_h: Spatial height (32).
        spatial_w: Spatial width (64).
        lambda_stego: STEGO contrastive loss weight.
        lambda_depthg: DepthG correlation loss weight.
        lambda_uniformity: Uniformity loss weight.
        lambda_boundary: Boundary alignment loss weight.
        lambda_dbc: Depth-boundary coherence loss weight.
        lambda_inst_discrim: Instance discriminative loss weight.
        lambda_kl: KL regularization weight.
        lambda_entropy_local: Local entropy loss weight.
        lambda_entropy_global: Global entropy loss weight (negative = maximize).
        lambda_affinity: Affinity consistency loss weight.
        lambda_depth_cond: Depth conditioning loss weight.
    """

    def __init__(
        self,
        num_classes: int = 27,
        spatial_h: int = 32,
        spatial_w: int = 64,
        lambda_stego: float = 1.0,
        lambda_depthg: float = 0.5,
        lambda_uniformity: float = 0.3,
        lambda_boundary: float = 0.2,
        lambda_dbc: float = 0.2,
        lambda_inst_discrim: float = 1.0,
        lambda_kl: float = 0.3,
        lambda_entropy_local: float = 0.1,
        lambda_entropy_global: float = 0.05,
        lambda_affinity: float = 0.2,
        lambda_depth_cond: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.lambda_stego = lambda_stego
        self.lambda_depthg = lambda_depthg
        self.lambda_uniformity = lambda_uniformity
        self.lambda_boundary = lambda_boundary
        self.lambda_dbc = lambda_dbc
        self.lambda_inst_discrim = lambda_inst_discrim
        self.lambda_kl = lambda_kl
        self.lambda_entropy_local = lambda_entropy_local
        self.lambda_entropy_global = lambda_entropy_global
        self.lambda_affinity = lambda_affinity
        self.lambda_depth_cond = lambda_depth_cond

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        dino_features: torch.Tensor,
        depth: torch.Tensor,
        instance_labels: torch.Tensor,
        phase: str = "full",
    ) -> Dict[str, torch.Tensor]:
        """Compute combined refiner loss.

        Args:
            model_outputs: Dict from Mamba2PanopticRefiner.forward().
            dino_features: DINOv2 features (B, N, 768).
            depth: Depth values (B, N).
            instance_labels: Instance ID labels (B, N). 0 = background.
            phase: Training phase — "warmup", "rampup", or "full".

        Returns:
            Dict with all loss components and total.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = dino_features.device

        refined_sem = model_outputs["refined_sem"]       # (B, N, 27)
        cause_logits = model_outputs["cause_logits"]     # (B, N, 27)
        inst_embed = model_outputs["inst_embeddings"]    # (B, N, 64)
        boundary = model_outputs["boundary"]             # (B, N, 1)

        # Phase-dependent loss scaling
        if phase == "warmup":
            kl_scale = 1.0
            other_scale = 0.3
        elif phase == "rampup":
            kl_scale = 0.5
            other_scale = 1.0
        else:
            kl_scale = 1.0
            other_scale = 1.0

        # --- STEGO: contrastive on refined semantics ---
        l_stego = stego_loss(refined_sem, dino_features)
        losses["stego"] = l_stego

        # --- DepthG: depth-weighted correlation ---
        l_depthg = depth_guided_correlation_loss(refined_sem, depth)
        losses["depthg"] = l_depthg

        # --- Uniformity: semantic entropy within instances ---
        B, N, _ = refined_sem.shape
        sem_pred = refined_sem.argmax(dim=-1)  # (B, N)
        # Create soft instance masks from labels (top-1 for efficiency)
        unique_per_batch = []
        max_inst = 0
        for b in range(B):
            u = torch.unique(instance_labels[b])
            u = u[u > 0][:16]  # cap at 16 instances for speed
            unique_per_batch.append(u)
            max_inst = max(max_inst, len(u))

        if max_inst > 0:
            inst_masks = torch.zeros(B, max(max_inst, 1), N, device=device)
            for b in range(B):
                for m, iid in enumerate(unique_per_batch[b]):
                    inst_masks[b, m] = (instance_labels[b] == iid).float()
            l_uniform = uniformity_loss(inst_masks, sem_pred, self.num_classes)
        else:
            l_uniform = torch.tensor(0.0, device=device)
        losses["uniformity"] = l_uniform

        # --- Boundary alignment ---
        if max_inst > 0:
            l_boundary = boundary_alignment_loss(
                sem_pred, inst_masks, self.spatial_h, self.spatial_w
            )
        else:
            l_boundary = torch.tensor(0.0, device=device)
        losses["boundary"] = l_boundary

        # --- DBC: depth-boundary coherence ---
        if max_inst > 0:
            l_dbc = depth_boundary_coherence_loss(
                sem_pred, inst_masks, depth, self.spatial_h, self.spatial_w
            )
        else:
            l_dbc = torch.tensor(0.0, device=device)
        losses["dbc"] = l_dbc

        # --- Instance discriminative loss ---
        l_inst = discriminative_loss(inst_embed, instance_labels)
        losses["inst_pull"] = l_inst["pull"]
        losses["inst_push"] = l_inst["push"]
        losses["inst_discrim"] = l_inst["total"]

        # --- KL regularization ---
        l_kl = kl_regularization_loss(refined_sem, cause_logits)
        losses["kl"] = l_kl

        # --- Entropy losses ---
        l_ent_local = entropy_loss(refined_sem, mode="local")
        l_ent_global = entropy_loss(refined_sem, mode="global")
        losses["entropy_local"] = l_ent_local
        losses["entropy_global"] = l_ent_global

        # --- Affinity consistency ---
        l_affinity = affinity_consistency_loss(refined_sem, inst_embed, dino_features)
        losses["affinity"] = l_affinity

        # --- Depth conditioning loss (from model) ---
        l_depth_cond = model_outputs.get(
            "depth_cond_loss", torch.tensor(0.0, device=device)
        )
        losses["depth_cond"] = l_depth_cond

        # --- Total ---
        total = (
            self.lambda_stego * l_stego * other_scale
            + self.lambda_depthg * l_depthg * other_scale
            + self.lambda_uniformity * l_uniform * other_scale
            + self.lambda_boundary * l_boundary * other_scale
            + self.lambda_dbc * l_dbc * other_scale
            + self.lambda_inst_discrim * l_inst["total"] * other_scale
            + self.lambda_kl * l_kl * kl_scale
            + self.lambda_entropy_local * l_ent_local * other_scale
            + self.lambda_entropy_global * l_ent_global * other_scale
            + self.lambda_affinity * l_affinity * other_scale
            + self.lambda_depth_cond * l_depth_cond
        )
        losses["total"] = total

        return losses
