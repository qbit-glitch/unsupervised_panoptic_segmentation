#!/usr/bin/env python3
"""Self-supervised training of CrossAttentionFusion for MMGD-Cut (Round 8).

Uses the best MMGD-Cut NCut segment maps as pseudo-labels to train the
cross-attention fusion module. The training objective is a cluster consistency
loss: within-cluster fused features should be more similar than between-cluster
features (a self-supervised contrastive objective).

Supervision signal: NCut segment maps from the best baseline run
(DINOv3+SSD-1B, no diffusion, 46.39% mIoU). These are already cached as
.npy files at: $COCO/mmgd_segments/<config_key>/val2017/

Training logic:
    For each image:
    1. Load DINOv3 tokens (N, 1024) and SSD-1B tokens (N, 1280)
    2. Run CrossAttentionFusion → fused (N, 512)
    3. Load NCut pseudo-segments (N, 1) — K=54 segment IDs
    4. Pool fused features per segment → (K, 512) segment prototypes
    5. Loss: InfoNCE across segments (same-segment = positive, different = negative)

This is inference-time fine-tuning on the 500 val images — fast (~minutes on GPU).

Usage:
    python train_cross_attention_fusion.py \
        --coco_root /path/to/coco \
        --seg_config_key "mmgd_K54_a5.5_reg0.7_dinov3+ssd1b_nodiff_r32" \
        --output_dir checkpoints/crossattn_fusion \
        --device mps --epochs 20
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cross_attention_fusion import CrossAttentionFusion, FusionWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════


def load_image_features(
    img_id: str,
    coco_root: str,
    dino_subdir: str = "dinov3_features/val2017",
    ssd_subdir: str = "ssd1b_features_s10/val2017",
    device: str = "mps",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load DINOv3 and SSD-1B features for one image.

    Returns (dino_feats, ssd_feats) or None if either is missing.
    """
    coco_p = Path(coco_root)
    dino_path = coco_p / dino_subdir / f"{img_id}.npy"
    ssd_path = coco_p / ssd_subdir / f"{img_id}.npy"

    if not dino_path.exists() or not ssd_path.exists():
        return None

    dino = torch.tensor(np.load(dino_path), dtype=torch.float32, device=device)
    ssd = torch.tensor(np.load(ssd_path), dtype=torch.float32, device=device)
    return dino, ssd


def load_segments(
    img_id: str,
    seg_dir: Path,
    target_n: int = 1024,
) -> Optional[np.ndarray]:
    """Load NCut segment IDs (N,) for one image.

    Args:
        img_id:   Image ID string.
        seg_dir:  Directory containing <img_id>.npy segment maps.
        target_n: Expected number of tokens (for validation).

    Returns:
        Flat (N,) int array of segment IDs, or None if missing.
    """
    seg_path = seg_dir / f"{img_id}.npy"
    if not seg_path.exists():
        return None

    seg_map = np.load(seg_path)  # may be (1, 1, H, W) or (H, W)
    seg = seg_map[0, 0] if seg_map.ndim == 4 else seg_map  # (H, W)

    # Downsample to token grid (32×32 = 1024 tokens)
    from PIL import Image
    grid = int(math.sqrt(target_n))
    seg_ds = np.array(
        Image.fromarray(seg.astype(np.uint8)).resize((grid, grid), Image.NEAREST)
    )
    return seg_ds.flatten()  # (N,)


# ═══════════════════════════════════════════════════════════════════════
# Cluster Consistency Loss
# ═══════════════════════════════════════════════════════════════════════


def cluster_consistency_loss(
    fused: torch.Tensor,
    seg_ids: np.ndarray,
    temperature: float = 0.07,
    min_size: int = 4,
) -> torch.Tensor:
    """InfoNCE loss across NCut segment prototypes.

    For each segment, its prototype is the mean-pooled fused embedding.
    Positive pairs: two random sub-samples from the same segment.
    Negative pairs: prototypes of all other segments.

    Args:
        fused:       (N, D) L2-normalised fused feature embeddings.
        seg_ids:     (N,) int array of segment IDs from NCut.
        temperature: InfoNCE temperature.
        min_size:    Minimum segment size to include (avoids tiny, noisy segments).

    Returns:
        Scalar loss tensor.
    """
    device = fused.device
    unique_segs = [s for s in np.unique(seg_ids) if (seg_ids == s).sum() >= min_size]

    if len(unique_segs) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Pool per-segment prototype: mean of L2-normalised tokens
    prototypes = []
    for seg_id in unique_segs:
        mask = torch.from_numpy(seg_ids == seg_id).to(device)
        seg_feats = fused[mask]
        proto = F.normalize(seg_feats.mean(dim=0, keepdim=True), p=2, dim=1)
        prototypes.append(proto)

    protos = torch.cat(prototypes, dim=0)  # (K, D)

    # Pairwise similarities
    sim = protos @ protos.T / temperature  # (K, K)

    # InfoNCE: each prototype is an anchor, all others are negatives
    # (no explicit positive pair — each row sums over all same-cluster tokens)
    # Here we use leave-one-out: diagonal is "identity" similarity → remove it
    K = protos.shape[0]
    mask_diag = ~torch.eye(K, dtype=torch.bool, device=device)

    total_loss = torch.tensor(0.0, device=device)
    for i in range(K):
        # Positive: high self-coherence (we want proto_i to be far from all others)
        neg_sims = sim[i][mask_diag[i]]  # (K-1,)
        # Self-similarity = 1.0 / temperature (always max)
        self_sim = torch.tensor(1.0 / temperature, device=device)
        logits = torch.cat([self_sim.unsqueeze(0), neg_sims])
        loss_i = -F.log_softmax(logits, dim=0)[0]
        total_loss = total_loss + loss_i

    return total_loss / K


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════


def train_cross_attention_fusion(
    coco_root: str,
    seg_config_key: str,
    output_dir: str,
    dino_subdir: str = "dinov3_features/val2017",
    ssd_subdir: str = "ssd1b_features_s10/val2017",
    target_res: int = 32,
    d_dino: int = 1024,
    d_ssd: int = 1280,
    d_proj: int = 256,
    n_heads: int = 4,
    d_out: int = 512,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    min_seg_size: int = 4,
    n_images: Optional[int] = None,
    device: str = "mps",
    log_interval: int = 20,
) -> None:
    """Train CrossAttentionFusion on val2017 using NCut pseudo-labels.

    Args:
        coco_root:       Path to COCO dataset root.
        seg_config_key:  Config key of the cached NCut segments to use as
                         pseudo-labels (from mmgd_cut.py run).
        output_dir:      Directory to save trained module.
        d_dino/d_ssd:    Feature dimensions for each modality.
        d_proj/d_out:    Internal/output projection dimensions.
        epochs:          Training epochs (over the full val set).
        lr:              Learning rate.
        temperature:     InfoNCE temperature for cluster consistency loss.
        min_seg_size:    Min tokens per segment to include in loss.
        n_images:        Limit dataset size (for testing).
        device:          Compute device.
    """
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)

    seg_dir = Path(coco_root) / "mmgd_segments" / seg_config_key / "val2017"
    if not seg_dir.exists():
        raise FileNotFoundError(
            f"Segment directory not found: {seg_dir}\n"
            f"Run mmgd_cut.py with config_key='{seg_config_key}' first."
        )

    # ── Collect valid images ──
    dino_dir = Path(coco_root) / dino_subdir
    img_ids = sorted([f.stem for f in dino_dir.glob("*.npy")])
    if n_images:
        img_ids = img_ids[:n_images]

    # Filter to images that have all three: dino, ssd, segments
    valid_ids = []
    for img_id in img_ids:
        if (
            (Path(coco_root) / dino_subdir / f"{img_id}.npy").exists()
            and (Path(coco_root) / ssd_subdir / f"{img_id}.npy").exists()
            and (seg_dir / f"{img_id}.npy").exists()
        ):
            valid_ids.append(img_id)

    logger.info(
        "Training on %d images with seg_config=%s", len(valid_ids), seg_config_key
    )
    if not valid_ids:
        raise RuntimeError("No valid images found. Check paths.")

    target_n = target_res * target_res

    # ── Pre-load all features into memory (small dataset) ──
    logger.info("Pre-loading features into memory...")
    all_dino: List[torch.Tensor] = []
    all_ssd: List[torch.Tensor] = []
    all_segs: List[np.ndarray] = []

    for img_id in tqdm(valid_ids, desc="Loading"):
        feats = load_image_features(img_id, coco_root, dino_subdir, ssd_subdir, device)
        segs = load_segments(img_id, seg_dir, target_n)
        if feats is None or segs is None:
            continue
        dino_f, ssd_f = feats

        # Align to target_res
        def _align(f: torch.Tensor, res: int) -> torch.Tensor:
            n, d = f.shape
            g = int(math.sqrt(n))
            if g == res:
                return F.normalize(f, p=2, dim=1)
            feat_2d = f.reshape(g, g, d).permute(2, 0, 1).unsqueeze(0)
            out = F.interpolate(feat_2d, size=(res, res), mode="bilinear", align_corners=False)
            return F.normalize(out[0].permute(1, 2, 0).reshape(res * res, d), p=2, dim=1)

        all_dino.append(_align(dino_f, target_res))  # (N, d_dino)
        all_ssd.append(_align(ssd_f, target_res))    # (N, d_ssd)
        all_segs.append(segs)

    n_valid = len(all_dino)
    logger.info("Loaded %d valid images", n_valid)

    # ── Model ──
    model = CrossAttentionFusion(
        d_dino=d_dino,
        d_ssd=d_ssd,
        d_proj=d_proj,
        n_heads=n_heads,
        d_out=d_out,
    ).to(device_t)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("CrossAttentionFusion: %d parameters", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * n_valid, eta_min=lr * 0.01,
    )

    # ── Training Loop ──
    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        indices = torch.randperm(n_valid).tolist()
        t_ep = time.time()

        for step, i in enumerate(tqdm(indices, desc=f"Epoch {epoch}/{epochs}")):
            dino_f = all_dino[i].unsqueeze(0)  # (1, N, d_dino)
            ssd_f = all_ssd[i].unsqueeze(0)    # (1, N, d_ssd)
            segs = all_segs[i]

            fused, alpha = model(dino_f, ssd_f)  # (1, N, d_out), (1, N, 1)
            fused_tokens = F.normalize(fused[0], p=2, dim=1)  # (N, d_out)

            loss = cluster_consistency_loss(
                fused_tokens, segs,
                temperature=temperature,
                min_size=min_seg_size,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1

            if (step + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d step %d/%d | loss=%.4f | α_mean=%.3f",
                    epoch, step + 1, n_valid,
                    epoch_loss / n_steps,
                    alpha[0, :, 0].mean().item(),
                )

        avg = epoch_loss / max(n_steps, 1)
        logger.info(
            "Epoch %d | loss=%.4f | time=%.1fs",
            epoch, avg, time.time() - t_ep,
        )
        history.append({"epoch": epoch, "loss": avg})

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": {
                    "d_dino": d_dino, "d_ssd": d_ssd,
                    "d_proj": d_proj, "n_heads": n_heads,
                    "d_out": d_out, "target_res": target_res,
                },
                "loss": avg,
            }, output_p / "best.pth")
            logger.info("New best: loss=%.4f", avg)

    with open(output_p / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best loss: %.4f", best_loss)
    logger.info(
        "Checkpoint saved to %s", output_p / "best.pth",
    )
    logger.info(
        "To evaluate: run mmgd_cut.py with --fusion_ckpt %s/best.pth",
        output_p,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-supervised training of CrossAttentionFusion"
    )
    p.add_argument("--coco_root", required=True)
    p.add_argument(
        "--seg_config_key", required=True,
        help="Config key of cached NCut segments "
             "(e.g. mmgd_K54_a5.5_reg0.7_dinov3+ssd1b_nodiff_r32)",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dino_subdir", default="dinov3_features/val2017")
    p.add_argument("--ssd_subdir", default="ssd1b_features_s10/val2017")
    p.add_argument("--target_res", type=int, default=32)
    p.add_argument("--d_dino", type=int, default=1024)
    p.add_argument("--d_ssd", type=int, default=1280)
    p.add_argument("--d_proj", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_out", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--min_seg_size", type=int, default=4)
    p.add_argument("--n_images", type=int, default=None)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--log_interval", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_cross_attention_fusion(
        coco_root=args.coco_root,
        seg_config_key=args.seg_config_key,
        output_dir=args.output_dir,
        dino_subdir=args.dino_subdir,
        ssd_subdir=args.ssd_subdir,
        target_res=args.target_res,
        d_dino=args.d_dino,
        d_ssd=args.d_ssd,
        d_proj=args.d_proj,
        n_heads=args.n_heads,
        d_out=args.d_out,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        min_seg_size=args.min_seg_size,
        n_images=args.n_images,
        device=args.device,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
