"""CAUSE-TR Mode B trainer: Stage-B-only loop with frozen codebook + cluster_probe.

Differs from the original CAUSE Stage B (refs/cause/train_cause_tr_dinov2.py):
- Backbone is DINOv3 instead of DINOv2 (frozen).
- A new `DINOv3ToDINOv2Adapter` (Linear, identity-init) sits between backbone
  features and the codebook.
- `codebook` and `cluster_probe` are loaded from the official DINOv2 CAUSE-TR
  checkpoint and frozen (`requires_grad=False`).
- The centroid loss uses a custom non-detached variant — CAUSE's
  `Cluster.forward_centroid` does `.detach()` on input which would zero out
  gradients when `cluster_probe` is frozen.
"""

from __future__ import annotations

import gc
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Make CAUSE modules importable when invoked from project root.
_CAUSE_DIR = Path(__file__).resolve().parent.parent.parent / "refs" / "cause"
if str(_CAUSE_DIR) not in sys.path:
    sys.path.insert(0, str(_CAUSE_DIR))

from modules.segment_module import (  # noqa: E402
    ema_update,
    transform,
)

logger = logging.getLogger(__name__)


@dataclass
class CAUSEModeBConfig:
    """Trainer-side hyperparameters extracted from the YAML config."""
    epochs: int = 40
    batch_size: int = 8
    grad_accum_steps: int = 2
    head_lr: float = 5e-5
    adapter_lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.99
    contrastive_temp: float = 0.07
    pos_thresh: float = 0.3
    neg_thresh: float = 0.1
    bank_max_size: int = 100
    loss_weight_contrastive: float = 1.0
    loss_weight_centroid: float = 1.0
    log_every: int = 50
    save_every: int = 5
    output_dir: Path = Path("refs/cause/CAUSE_modeb_dinov3_vitb16")
    flip_tta: bool = True
    n_classes: int = 27          # GT class count (Cityscapes 27-class)


class CAUSEModeBTrainer:
    """Stage-B-only CAUSE trainer for the Mode B retraining experiment."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        adapter: nn.Module,
        segment: nn.Module,
        cluster: nn.Module,
        device: torch.device,
        cfg: CAUSEModeBConfig,
    ) -> None:
        self.backbone = backbone
        self.adapter = adapter
        self.segment = segment
        self.cluster = cluster
        self.device = device
        self.cfg = cfg
        self.cfg.output_dir = Path(self.cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer with two param groups (head LR + adapter LR).
        head_params = (
            list(self.segment.head.parameters())
            + list(self.segment.projection_head.parameters())
            + list(self.segment.linear.parameters())
        )
        # Filter out the codebook reference inside segment.head/head_ema (frozen).
        head_params = [p for p in head_params if p.requires_grad]
        adapter_params = [p for p in self.adapter.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(
            [
                {"params": head_params, "lr": self.cfg.head_lr, "weight_decay": self.cfg.weight_decay},
                {"params": adapter_params, "lr": self.cfg.adapter_lr, "weight_decay": self.cfg.weight_decay},
            ],
        )
        self._trainable_params = head_params + adapter_params

        self.global_step = 0
        self.best_miou = -1.0
        logger.info(
            "CAUSEModeBTrainer ready (head_params=%d adapter_params=%d)",
            sum(p.numel() for p in head_params),
            sum(p.numel() for p in adapter_params),
        )

    # ------------------------------------------------------------------ #
    # Forward pieces                                                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _backbone_features(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, 768) — CLS+registers already stripped by DINOv3ViTB."""
        feat = self.backbone(img)
        if feat.dim() != 3:
            raise RuntimeError(f"Backbone output must be (B, N, D); got {tuple(feat.shape)}")
        return feat

    def _centroid_loss_modeb(self, seg_feat: torch.Tensor) -> torch.Tensor:
        """Mode B centroid loss — cluster_probe is frozen so we MUST keep gradient
        flowing through `seg_feat` (the student head output).

        Equivalent to `Cluster.forward_centroid` minus the `.detach()` call at
        segment_module.py:221. With frozen cluster_probe, the original loss has
        zero gradient and nothing trains.
        """
        normed_features = F.normalize(transform(seg_feat), dim=1)               # NO .detach()
        normed_clusters = F.normalize(self.cluster.cluster_probe, dim=1)        # frozen
        inner = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        n_classes = self.cluster.cluster_probe.shape[0]
        with torch.no_grad():
            one_hot = F.one_hot(inner.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
        return -(one_hot * inner).sum(1).mean()

    # ------------------------------------------------------------------ #
    # Training step                                                       #
    # ------------------------------------------------------------------ #
    def _step(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_feat = self._backbone_features(img)                                # (B, N, 768)
        adapted_feat = self.adapter(raw_feat)                                  # (B, N, 768)

        # Student
        seg_feat = self.segment.head(adapted_feat, drop=self.segment.dropout)  # (B, N, 90)
        proj_feat = self.segment.projection_head(seg_feat)                     # (B, N, 2048)

        # Teacher
        with torch.no_grad():
            seg_feat_ema = self.segment.head_ema(adapted_feat)
            proj_feat_ema = self.segment.projection_head_ema(seg_feat_ema)

        # Losses
        self.cluster.bank_compute()
        loss_contrastive = self.cluster.contrastive_ema_with_codebook_bank(
            adapted_feat,
            proj_feat,
            proj_feat_ema,
            temp=self.cfg.contrastive_temp,
            pos_thresh=self.cfg.pos_thresh,
            neg_thresh=self.cfg.neg_thresh,
        )
        loss_centroid = self._centroid_loss_modeb(seg_feat)
        loss = (
            self.cfg.loss_weight_contrastive * loss_contrastive
            + self.cfg.loss_weight_centroid * loss_centroid
        )
        return {
            "loss": loss,
            "contrastive": loss_contrastive.detach(),
            "centroid": loss_centroid.detach(),
            "adapted_feat": adapted_feat.detach(),
            "proj_feat_ema": proj_feat_ema.detach(),
        }

    @torch.no_grad()
    def _ema_step(self) -> None:
        ema_update(self.segment.head, self.segment.head_ema, self.cfg.ema_decay)
        ema_update(self.segment.projection_head, self.segment.projection_head_ema, self.cfg.ema_decay)

    @torch.no_grad()
    def _bank_update(self, adapted_feat: torch.Tensor, proj_feat_ema: torch.Tensor) -> None:
        self.cluster.bank_update(adapted_feat, proj_feat_ema, max_num=self.cfg.bank_max_size)

    # ------------------------------------------------------------------ #
    # Loops                                                              #
    # ------------------------------------------------------------------ #
    def fit(self, train_loader, val_loader) -> None:
        self.cluster.bank_init()  # initialize per-epoch buffer once at start

        for epoch in range(1, self.cfg.epochs + 1):
            self._train_epoch(epoch, train_loader)
            metrics = self.validate(val_loader)
            logger.info(
                "[epoch %03d] mIoU=%.2f pAcc=%.2f best=%.2f",
                epoch, metrics["mIoU"], metrics["pAcc"], self.best_miou,
            )

            self._save_checkpoint(epoch, metrics)
            if metrics["mIoU"] > self.best_miou:
                self.best_miou = metrics["mIoU"]
                self._save_checkpoint(epoch, metrics, tag="best")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def _train_epoch(self, epoch: int, train_loader) -> None:
        self.segment.train()
        self.adapter.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_t0 = time.time()
        accum_loss = 0.0
        accum_count = 0
        accum_idx = 0
        pbar = tqdm(
            train_loader,
            desc=f"train ep{epoch:03d}",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for batch_idx, batch in enumerate(pbar):
            img = batch["img"].to(self.device, non_blocking=True)
            out = self._step(img)
            (out["loss"] / self.cfg.grad_accum_steps).backward()
            accum_idx += 1

            if accum_idx == self.cfg.grad_accum_steps:
                if self.cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._ema_step()
                accum_idx = 0

            self._bank_update(out["adapted_feat"], out["proj_feat_ema"])

            loss_val = float(out["loss"].detach())
            accum_loss += loss_val
            accum_count += 1
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.3f}",
                con=f"{float(out['contrastive']):.3f}",
                cen=f"{float(out['centroid']):.3f}",
                avg=f"{accum_loss / accum_count:.3f}",
            )

            if batch_idx % self.cfg.log_every == 0:
                logger.info(
                    "[epoch %03d step %05d batch %05d] loss=%.4f contrastive=%.4f centroid=%.4f",
                    epoch, self.global_step, batch_idx,
                    loss_val,
                    float(out["contrastive"]),
                    float(out["centroid"]),
                )

        # flush any partial accumulation at epoch end.
        if accum_idx > 0:
            if self.cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._ema_step()

        avg_loss = accum_loss / max(accum_count, 1)
        logger.info(
            "[epoch %03d] avg_loss=%.4f time=%.1fs",
            epoch, avg_loss, time.time() - epoch_t0,
        )

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """27-class Cityscapes mIoU with Hungarian matching."""
        self.segment.eval()
        self.adapter.eval()

        n_gt = self.cfg.n_classes
        n_cluster = self.cluster.cluster_probe.shape[0]
        histogram = torch.zeros(n_cluster, n_gt, dtype=torch.long)

        for batch in tqdm(val_loader, desc="  val", dynamic_ncols=True, leave=False, mininterval=1.0):
            imgs = batch["img"].to(self.device, non_blocking=True)
            labels = batch["label"]   # (B, H, W) on CPU, int64 in [-1..n_gt-1]

            feat = self._backbone_features(imgs)
            adapted = self.adapter(feat)
            seg = transform(self.segment.head_ema(adapted))    # (B, 90, sqrtN, sqrtN)

            if self.cfg.flip_tta:
                feat_flip = self._backbone_features(imgs.flip(dims=[3]))
                adapted_flip = self.adapter(feat_flip)
                seg_flip = transform(self.segment.head_ema(adapted_flip))
                seg = (seg + seg_flip.flip(dims=[3])) / 2.0

            spatial_up = F.interpolate(
                seg, size=labels.shape[-2:], mode="bilinear", align_corners=False,
            )
            normed_feats = F.normalize(spatial_up, dim=1)
            normed_probes = F.normalize(self.cluster.cluster_probe.detach(), dim=1)
            inner = torch.einsum("bchw,nc->bnhw", normed_feats, normed_probes)
            preds = inner.argmax(dim=1).cpu()

            lt = labels.reshape(-1)
            lp = preds.reshape(-1)
            mask = (lt >= 0) & (lt < n_gt) & (lp >= 0) & (lp < n_cluster)
            lt_valid = lt[mask]
            lp_valid = lp[mask]
            hist = torch.bincount(
                n_gt * lp_valid + lt_valid,
                minlength=n_gt * n_cluster,
            ).reshape(n_cluster, n_gt)
            histogram += hist

        conf_np = histogram.numpy()
        row_ind, col_ind = linear_sum_assignment(conf_np, maximize=True)
        tp = np.array([conf_np[row_ind[i], col_ind[i]] for i in range(len(col_ind))], dtype=np.float64)
        fp = np.array([conf_np[row_ind[i], :].sum() - tp[i] for i in range(len(col_ind))], dtype=np.float64)
        fn = np.array([conf_np[:, col_ind[i]].sum() - tp[i] for i in range(len(col_ind))], dtype=np.float64)
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, np.nan)

        valid = ~np.isnan(iou)
        miou = float(iou[valid].mean()) * 100 if valid.any() else 0.0
        pacc = float(tp.sum() / max(conf_np.sum(), 1)) * 100

        self.segment.train()
        self.adapter.train()
        return {"mIoU": miou, "pAcc": pacc}

    # ------------------------------------------------------------------ #
    # Checkpointing                                                       #
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], tag: Optional[str] = None) -> None:
        if tag is None and (epoch % self.cfg.save_every) != 0:
            return
        sub = tag if tag is not None else f"epoch_{epoch:03d}"
        ckpt_dir = self.cfg.output_dir / sub
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.segment.state_dict(), ckpt_dir / "segment_tr.pth")
        torch.save(
            {
                "codebook": self.cluster.codebook.detach().cpu(),
                "cluster_probe": self.cluster.cluster_probe.detach().cpu(),
            },
            ckpt_dir / "cluster_tr.pth",
        )
        torch.save(self.adapter.state_dict(), ckpt_dir / "adapter.pth")
        meta = {
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "best_miou": self.best_miou,
        }
        (ckpt_dir / "train_meta.json").write_text(_json_dumps(meta))
        logger.info("Saved checkpoint -> %s (mIoU=%.2f)", ckpt_dir, metrics.get("mIoU", -1.0))


def _json_dumps(d: Dict[str, Any]) -> str:
    import json
    return json.dumps(d, indent=2, default=str)


__all__ = ["CAUSEModeBConfig", "CAUSEModeBTrainer"]
