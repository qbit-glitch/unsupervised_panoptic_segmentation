"""MBPS v2 Trainer (PyTorch).

Simplified 2-phase training:
    Phase 1 (Bootstrap, epochs 1-25): Train heads + bridge (ramp-up)
    Phase 2 (Self-training, epochs 26-40): EMA teacher refines pseudo-labels

Uses cross-entropy + discriminative loss + bridge loss (3 losses vs v1's 12).
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from mbps_pytorch.losses.semantic_loss_v2 import semantic_cross_entropy
from mbps_pytorch.losses.instance_embedding_loss import discriminative_loss
from mbps_pytorch.losses.bridge_loss import BridgeLoss
from mbps_pytorch.training.checkpointing import CheckpointManager
from mbps_pytorch.training.ema import EMAState, create_ema

logger = logging.getLogger(__name__)


class MBPSv2Trainer:
    """MBPS v2 training orchestrator.

    Args:
        config: Training configuration dict.
        model: MBPSv2Model instance.
        device: Primary torch device.
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        self.config = config
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"

        logger.info(
            f"MBPSv2Trainer: device={self.device} AMP={'on' if self.use_amp else 'off'}"
        )

        # Optimizer
        tc = config["training"]
        lr = float(tc.get("learning_rate", 5e-5))
        weight_decay = float(tc.get("weight_decay", 0.05))
        self.gradient_clip = float(tc.get("gradient_clip", 1.0))

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        # Mixed-precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Loss weights
        lw = config.get("loss_weights", {})
        self.lambda_semantic = float(lw.get("lambda_semantic", 1.0))
        self.lambda_instance = float(lw.get("lambda_instance", 1.0))
        self.lambda_bridge = float(lw.get("lambda_bridge", 0.1))
        self.label_smoothing = float(lw.get("label_smoothing", 0.1))
        self.delta_v = float(lw.get("delta_v", 0.5))
        self.delta_d = float(lw.get("delta_d", 1.5))

        # Bridge loss
        self.bridge_loss_fn = BridgeLoss(
            lambda_recon=float(lw.get("lambda_recon", 0.5)),
            lambda_cka=float(lw.get("lambda_cka", 0.1)),
            lambda_state=0.0,  # No state reg in v2
        )

        # Curriculum
        self.bootstrap_end = int(tc.get("bootstrap_end", 25))
        self.bridge_enable_epoch = int(tc.get("bridge_enable_epoch", 5))
        self.total_epochs = int(tc.get("total_epochs", 40))
        self.self_training_rounds = int(tc.get("self_training_rounds", 3))
        self.self_training_epochs = int(tc.get("self_training_epochs_per_round", 5))

        # Checkpointing
        ckpt_cfg = config.get("checkpointing", {})
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=ckpt_cfg.get("checkpoint_dir", "checkpoints"),
            keep_top_k=ckpt_cfg.get("keep_top_k", 3),
            save_every_n_epochs=ckpt_cfg.get("save_every_n_epochs", 5),
        )

        # EMA
        self.ema: Optional[EMAState] = None

        # Training bookkeeping
        self.epoch: int = 0
        self.global_step: int = 0

    # ------------------------------------------------------------------
    # W&B integration
    # ------------------------------------------------------------------

    def init_wandb(
        self, vm_name: str = "local", experiment_name: str = "v2_default",
    ) -> None:
        if not HAS_WANDB:
            return
        log_cfg = self.config.get("logging", {})
        group = log_cfg.get("wandb_group") or experiment_name
        tags = list(log_cfg.get("wandb_tags") or [])
        tags.append(vm_name)
        wandb.init(
            project=log_cfg.get("wandb_project", "mbps"),
            entity=log_cfg.get("wandb_entity"),
            group=group,
            name=f"{experiment_name}/{vm_name}",
            tags=tags,
            config=self.config,
            reinit=True,
        )
        self._wandb_active = True

    @property
    def _use_wandb(self) -> bool:
        return HAS_WANDB and getattr(self, "_wandb_active", False)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def create_train_state(self) -> None:
        self.ema = create_ema(
            self.model,
            momentum=self.config["training"].get("ema_momentum", 0.999),
        )
        logger.info("Train state created (model + EMA on %s)", self.device)

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = self.ckpt_manager.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt.get("epoch", 0)
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.shadow = ckpt["ema_state_dict"]
        logger.info("Resumed from %s (epoch %d)", checkpoint_path, self.epoch)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute v2 loss for one batch.

        Args:
            batch: Dict with 'image', 'depth', 'pseudo_semantic', 'pseudo_instance'.
            epoch: Current epoch (1-indexed).

        Returns:
            (total_loss, loss_dict).
        """
        use_bridge = epoch >= self.bridge_enable_epoch

        model_output = self.model(
            image=batch["image"],
            depth=batch.get("depth"),
            use_bridge=use_bridge,
        )

        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. Semantic cross-entropy
        sem_loss = semantic_cross_entropy(
            logits=model_output["semantic_logits"],
            labels=batch["pseudo_semantic"],
            label_smoothing=self.label_smoothing,
        )
        loss_dict["L_semantic"] = sem_loss

        # 2. Discriminative instance loss
        inst_losses = discriminative_loss(
            embeddings=model_output["instance_embeddings"],
            instance_labels=batch["pseudo_instance"],
            delta_v=self.delta_v,
            delta_d=self.delta_d,
        )
        loss_dict["L_instance"] = inst_losses["total"]
        loss_dict["L_instance_pull"] = inst_losses["pull"]
        loss_dict["L_instance_push"] = inst_losses["push"]

        # 3. Bridge loss
        bridge_loss = torch.tensor(0.0, device=self.device)
        if use_bridge and "fused_semantic" in model_output:
            b_losses = self.bridge_loss_fn(
                original_semantic=model_output["semantic_logits"],
                original_features=model_output["instance_embeddings"],
                reconstructed_semantic=model_output.get(
                    "reconstructed_semantic", model_output["semantic_logits"]
                ),
                reconstructed_features=model_output.get(
                    "reconstructed_instance", model_output["instance_embeddings"]
                ),
                fused_semantic=model_output["fused_semantic"],
                fused_features=model_output["fused_instance"],
                align_loss=model_output.get(
                    "align_loss", torch.tensor(0.0, device=self.device)
                ),
            )
            bridge_loss = b_losses["total"]
        loss_dict["L_bridge"] = bridge_loss
        loss_dict["bridge_gate"] = model_output.get(
            "bridge_gate", torch.tensor(0.0, device=self.device)
        )

        # Total
        _safe = lambda x: torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        total = (
            self.lambda_semantic * _safe(sem_loss)
            + self.lambda_instance * _safe(inst_losses["total"])
            + self.lambda_bridge * _safe(bridge_loss)
        )
        loss_dict["L_total"] = total

        return total, loss_dict

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _train_step(
        self, batch: Dict[str, torch.Tensor], epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            total_loss, loss_dict = self.compute_loss(batch, epoch)

        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.gradient_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # NaN-safe gradient check
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        self.global_step += 1

        return {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_metrics: Dict[str, float] = {}
        num_steps = 0
        start_time = time.time()

        for batch in dataloader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                   if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            step_losses = self._train_step(batch, self.epoch)

            for k, v in step_losses.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            num_steps += 1

            log_every = self.config.get("logging", {}).get("log_every_n_steps", 50)
            if num_steps % log_every == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {num_steps}: "
                    f"L_total={step_losses.get('L_total', 0):.4f} "
                    f"L_sem={step_losses.get('L_semantic', 0):.4f} "
                    f"L_inst={step_losses.get('L_instance', 0):.4f} "
                    f"L_bridge={step_losses.get('L_bridge', 0):.4f} "
                    f"gate={step_losses.get('bridge_gate', 0):.4f}"
                )
                if self._use_wandb:
                    wandb.log(
                        {f"step/{k}": v for k, v in step_losses.items()
                         if isinstance(v, (int, float))},
                        step=self.global_step,
                    )

        epoch_time = time.time() - start_time
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_steps, 1)
        epoch_metrics["epoch_time_s"] = epoch_time
        epoch_metrics["num_steps"] = num_steps

        if self._use_wandb:
            wandb.log(
                {f"epoch/{k}": v for k, v in epoch_metrics.items()
                 if isinstance(v, (int, float))},
                step=self.global_step,
            )

        return epoch_metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run full v2 training (bootstrap + self-training)."""
        start_epoch = self.epoch + 1

        # Phase 1: Bootstrap
        for epoch in range(start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            phase = "bootstrap" if epoch <= self.bootstrap_end else "self_train"
            bridge_on = epoch >= self.bridge_enable_epoch

            logger.info(
                f"=== Epoch {epoch}/{self.total_epochs} "
                f"(phase={phase}, bridge={'ON' if bridge_on else 'OFF'}) ==="
            )

            metrics = self.train_epoch(train_loader)

            logger.info(
                f"Epoch {epoch} done: "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                            if isinstance(v, float))
            )

            # Checkpoint
            if epoch % self.config.get("checkpointing", {}).get("save_every_n_epochs", 5) == 0:
                ema_sd = self.ema.get_state_dict() if self.ema is not None else None
                self.ckpt_manager.save(
                    epoch,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    ema_state_dict=ema_sd,
                    metrics=metrics,
                )

        if self._use_wandb:
            wandb.finish()
