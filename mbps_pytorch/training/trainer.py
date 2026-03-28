"""Main MBPS Trainer.

Orchestrates the full training pipeline across all 4 phases:
    Phase A: Semantic-only training
    Phase B: Instance integration with gradient projection
    Phase C: Full model with bridge, consistency, PQ losses
    Phase D: Self-training refinement

Uses ``torch.nn.parallel.DistributedDataParallel`` (DDP) or
``DataParallel`` for multi-GPU training, and ``torch.cuda.amp``
for optional mixed-precision.
"""

from __future__ import annotations

import copy
import logging
import os
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

from mbps_pytorch.losses import (
    BridgeLoss,
    ConsistencyLoss,
    GradientBalancer,
    InstanceLoss,
    PQProxyLoss,
    SemanticLoss,
)
from mbps_pytorch.training.checkpointing import CheckpointManager
from mbps_pytorch.training.curriculum import TrainingCurriculum
from mbps_pytorch.training.ema import EMAState, create_ema
from mbps_pytorch.training.self_training import SelfTrainer


logger = logging.getLogger(__name__)


class MBPSTrainer:
    """Main MBPS training orchestrator with multi-GPU support.

    Replaces the JAX ``pmap``-based trainer with a standard PyTorch
    training loop using ``torch.cuda.amp.GradScaler`` for mixed
    precision and optional ``DistributedDataParallel`` wrapping.

    Args:
        config: Training configuration dict.
        model: MBPS ``nn.Module`` model instance.
        device: Primary torch device (e.g., ``torch.device('cuda:0')``).
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
            f"MBPSTrainer: using device {self.device} "
            f"(AMP={'on' if self.use_amp else 'off'})"
        )

        # Optimizer
        lr = float(config["training"].get("learning_rate", 1e-4))
        weight_decay = float(config["training"].get("weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed-precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Training components
        self.curriculum = TrainingCurriculum(
            phase_a_end=config["training"]["phase_a_end"],
            phase_b_end=config["training"]["phase_b_end"],
            total_epochs=config["training"]["total_epochs"],
        )

        # Loss functions
        self.semantic_loss = SemanticLoss(
            lambda_depthg=config["loss_weights"]["lambda_depthg"],
            stego_temperature=config["loss_weights"]["stego_temperature"],
            knn_k=config["loss_weights"]["stego_knn_k"],
            depth_sigma=config["loss_weights"]["depth_sigma"],
        )
        self.instance_loss = InstanceLoss(
            lambda_dice=1.0,
            lambda_bce=config["loss_weights"]["lambda_drop"],
            lambda_unsup=0.3,
        )
        self.bridge_loss = BridgeLoss(
            lambda_recon=config["loss_weights"]["lambda_recon"],
            lambda_cka=config["loss_weights"]["lambda_cka"],
            lambda_state=config["loss_weights"]["lambda_state"],
        )
        self.consistency_loss = ConsistencyLoss(
            lambda_uniform=config["loss_weights"]["lambda_uniform"],
            lambda_boundary=config["loss_weights"]["lambda_boundary"],
            lambda_dbc=config["loss_weights"]["lambda_dbc"],
        )
        self.pq_loss = PQProxyLoss()
        self.grad_balancer = GradientBalancer()

        # Checkpointing
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=config["checkpointing"]["checkpoint_dir"],
            keep_top_k=config["checkpointing"]["keep_top_k"],
            save_every_n_epochs=config["checkpointing"]["save_every_n_epochs"],
        )

        # Self-training
        self.self_trainer = SelfTrainer(
            num_rounds=config["training"]["self_training_rounds"],
            epochs_per_round=config["training"]["self_training_epochs_per_round"],
            initial_threshold=config["self_training"]["conf_threshold_init"],
            threshold_increment=config["self_training"]["conf_threshold_increment"],
        )

        # EMA (initialised in ``create_train_state``)
        self.ema: Optional[EMAState] = None

        # Training bookkeeping
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_pq: float = 0.0

    # ------------------------------------------------------------------
    # W&B integration
    # ------------------------------------------------------------------

    def init_wandb(
        self, vm_name: str = "local", experiment_name: str = "default",
    ) -> None:
        """Initialize Weights & Biases run.

        Args:
            vm_name: VM identifier (used as run name).
            experiment_name: Experiment name (used as W&B group).
        """
        if not HAS_WANDB:
            logger.warning("wandb not installed -- skipping W&B init")
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
        logger.info(
            f"W&B initialized: project={log_cfg.get('wandb_project')}, "
            f"group={group}"
        )

    @property
    def _use_wandb(self) -> bool:
        return HAS_WANDB and getattr(self, "_wandb_active", False)

    # ------------------------------------------------------------------
    # State creation
    # ------------------------------------------------------------------

    def create_train_state(self) -> None:
        """Initialise EMA and move everything to the target device.

        Call this after constructing the trainer and before ``train()``.
        """
        self.ema = create_ema(
            self.model,
            momentum=self.config["training"]["ema_momentum"],
        )
        logger.info("Train state created (model + EMA on %s)", self.device)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a saved checkpoint.

        Restores model weights, optimizer state, and epoch counter.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
        """
        ckpt = self.ckpt_manager.load(
            checkpoint_path, map_location=self.device
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt.get("epoch", 0)
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.shadow = ckpt["ema_state_dict"]
        logger.info(
            "Resumed from %s (epoch %d)", checkpoint_path, self.epoch
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total loss for one micro-batch.

        Args:
            batch: Dictionary with at least 'image' and 'depth' tensors,
                already on ``self.device``.
            epoch: Current epoch number (1-indexed).

        Returns:
            Tuple of (scalar total_loss, dict of named sub-losses).
        """
        phase_config = self.curriculum.get_config(epoch)
        weights = self.curriculum.get_loss_weights(epoch)

        model_output = self.model(
            image=batch["image"],
            depth=batch["depth"],
            use_bridge=phase_config.use_bridge,
        )

        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Semantic loss (always active)
        sem_losses = self.semantic_loss(
            semantic_codes=model_output["semantic_codes"],
            dino_features=model_output["dino_features"],
            depth=batch["depth"].reshape(batch["depth"].shape[0], -1),
        )
        loss_dict["L_semantic"] = sem_losses["total"]
        total_loss = total_loss + weights["alpha"] * sem_losses["total"]

        # Instance loss (Phase B onwards)
        if weights["beta"] > 0:
            inst_losses = self.instance_loss(
                pred_masks=model_output["instance_masks"],
                pred_scores=model_output["instance_scores"],
                features=model_output["dino_features"],
            )
            loss_dict["L_instance"] = inst_losses["total"]
            total_loss = total_loss + weights["beta"] * inst_losses["total"]
        else:
            loss_dict["L_instance"] = torch.tensor(0.0, device=self.device)

        # Bridge loss (Phase C onwards)
        if weights["gamma"] > 0 and "fused_semantic" in model_output:
            bridge_losses = self.bridge_loss(
                original_semantic=model_output["semantic_codes"],
                original_features=model_output["dino_features"],
                reconstructed_semantic=model_output.get(
                    "reconstructed_semantic",
                    model_output["semantic_codes"],
                ),
                reconstructed_features=model_output.get(
                    "reconstructed_features",
                    model_output["dino_features"],
                ),
                fused_semantic=model_output["fused_semantic"],
                fused_features=model_output["fused_features"],
                align_loss=model_output.get(
                    "align_loss",
                    torch.tensor(0.0, device=self.device),
                ),
            )
            loss_dict["L_bridge"] = bridge_losses["total"]
            total_loss = total_loss + weights["gamma"] * bridge_losses["total"]
        else:
            loss_dict["L_bridge"] = torch.tensor(0.0, device=self.device)

        # Consistency loss (Phase C onwards)
        if weights["delta"] > 0:
            sem_pred = torch.argmax(model_output["semantic_codes"], dim=-1)
            mask_probs = torch.sigmoid(model_output["instance_masks"])
            depth_flat = batch["depth"].reshape(batch["depth"].shape[0], -1)

            n = sem_pred.shape[-1]
            spatial_h = int(n ** 0.5)
            spatial_w = n // spatial_h

            cons_losses = self.consistency_loss(
                semantic_pred=sem_pred,
                instance_masks=mask_probs,
                depth=depth_flat,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            loss_dict["L_consistency"] = cons_losses["total"]
            total_loss = total_loss + weights["delta"] * cons_losses["total"]
        else:
            loss_dict["L_consistency"] = torch.tensor(0.0, device=self.device)

        # PQ proxy loss (Phase C onwards, requires EMA teacher)
        if weights["epsilon"] > 0 and self.ema is not None:
            # Build a temporary teacher model for forward pass
            teacher_model = copy.deepcopy(self.model)
            self.ema.apply_to_model(teacher_model)
            teacher_model.eval()

            with torch.no_grad():
                teacher_output = teacher_model(
                    image=batch["image"],
                    depth=batch["depth"],
                    use_bridge=True,
                )

            pq_losses = self.pq_loss(
                pred_masks=model_output["instance_masks"],
                pred_scores=model_output["instance_scores"],
                teacher_masks=teacher_output["instance_masks"],
                teacher_scores=teacher_output["instance_scores"],
            )
            loss_dict["L_pq"] = pq_losses["total"]
            total_loss = total_loss + weights["epsilon"] * pq_losses["total"]
        else:
            loss_dict["L_pq"] = torch.tensor(0.0, device=self.device)

        loss_dict["L_total"] = total_loss
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Single train step
    # ------------------------------------------------------------------

    def _train_step(
        self, batch: Dict[str, torch.Tensor], epoch: int
    ) -> Dict[str, float]:
        """Execute one optimizer step.

        Uses ``torch.cuda.amp`` for optional mixed precision.

        Args:
            batch: Batch dict with tensors on ``self.device``.
            epoch: Current epoch.

        Returns:
            Dict of scalar loss values (detached, on CPU).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            total_loss, loss_dict = self.compute_loss(batch, epoch)

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        self.global_step += 1

        return {k: v.detach().cpu().item() for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: A ``torch.utils.data.DataLoader`` yielding batch
                dicts.

        Returns:
            Dict of averaged epoch metrics.
        """
        epoch_metrics: Dict[str, float] = {}
        num_steps = 0
        start_time = time.time()

        for batch in dataloader:
            # Move batch to device
            batch = {
                k: v.to(self.device, non_blocking=True)
                   if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            step_losses = self._train_step(batch, self.epoch)

            for k, v in step_losses.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            num_steps += 1

            log_every = self.config["logging"]["log_every_n_steps"]
            if num_steps % log_every == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {num_steps}: "
                    f"L_total={step_losses.get('L_total', 0):.4f}"
                )
                if self._use_wandb:
                    step_metrics = {f"step/{k}": v for k, v in step_losses.items()}
                    step_metrics["step/global_step"] = self.global_step
                    wandb.log(step_metrics, step=self.global_step)

        # Average over steps
        epoch_time = time.time() - start_time
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_steps, 1)
        epoch_metrics["epoch_time_s"] = epoch_time
        epoch_metrics["phase"] = self.curriculum.get_phase(self.epoch)

        # W&B epoch-level logging
        if self._use_wandb:
            wandb_metrics = {
                f"epoch/{k}": v for k, v in epoch_metrics.items()
                if isinstance(v, (int, float))
            }
            wandb_metrics["epoch/epoch"] = self.epoch
            wandb.log(wandb_metrics, step=self.global_step)

        return epoch_metrics

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run full training pipeline (Phases A-D).

        Args:
            train_loader: Training ``DataLoader``.
            val_loader: Optional validation ``DataLoader``.
        """
        total_epochs = self.config["training"]["total_epochs"]
        start_epoch = self.epoch + 1  # resume-aware

        for epoch in range(start_epoch, total_epochs + 1):
            self.epoch = epoch
            phase = self.curriculum.get_phase(epoch)

            logger.info(
                f"=== Epoch {epoch}/{total_epochs} "
                f"(Phase {phase}) ==="
            )

            metrics = self.train_epoch(train_loader)

            # Log
            logger.info(
                f"Epoch {epoch} done: "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                    if isinstance(v, float)
                )
            )

            # Checkpoint
            if epoch % self.config["checkpointing"]["save_every_n_epochs"] == 0:
                ema_sd = self.ema.get_state_dict() if self.ema is not None else None
                self.ckpt_manager.save(
                    epoch,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    ema_state_dict=ema_sd,
                    metrics=metrics,
                )

        # Phase D: Self-training
        logger.info("=== Starting Phase D: Self-Training ===")
        self._run_self_training(train_loader)

        # Finish W&B run
        if self._use_wandb:
            wandb.finish()

    def _run_self_training(
        self,
        train_loader: DataLoader,
    ) -> None:
        """Run Phase D self-training rounds.

        Args:
            train_loader: Training ``DataLoader``.
        """
        for round_idx in range(self.self_trainer.num_rounds):
            logger.info(
                f"Self-training round {round_idx + 1}/"
                f"{self.self_trainer.num_rounds}"
            )

            epochs_per_round = self.config["training"][
                "self_training_epochs_per_round"
            ]
            for ep in range(epochs_per_round):
                self.epoch += 1
                metrics = self.train_epoch(train_loader)
                logger.info(
                    f"  Round {round_idx + 1}, Epoch {ep + 1}: "
                    f"L_total={metrics.get('L_total', 0):.4f}"
                )

            # Advance threshold
            self.self_trainer.advance_round()

            # Save checkpoint
            ema_sd = self.ema.get_state_dict() if self.ema is not None else None
            self.ckpt_manager.save(
                self.epoch,
                self.model.state_dict(),
                self.optimizer.state_dict(),
                ema_state_dict=ema_sd,
            )
