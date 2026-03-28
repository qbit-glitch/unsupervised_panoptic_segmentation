"""Main MBPS Trainer.

Orchestrates the full training pipeline across all 4 phases:
    Phase A: Semantic-only training
    Phase B: Instance integration with gradient projection
    Phase C: Full model with bridge, consistency, PQ losses
    Phase D: Self-training refinement

Uses jax.pmap for data-parallel training across all available TPU/GPU devices.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import yaml
from absl import logging
from flax import linen as nn

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from mbps.losses import (
    BridgeLoss,
    ConsistencyLoss,
    GradientBalancer,
    PQProxyLoss,
    SemanticLoss,
)
from mbps.models.instance.instance_loss import CutS3DInstanceLoss
from mbps.training.checkpointing import CheckpointManager
from mbps.training.curriculum import TrainingCurriculum
from mbps.training.ema import EMAState, create_ema
from mbps.training.self_training import SelfTrainer


# ---------------------------------------------------------------------------
# Utilities for multi-device data parallelism
# ---------------------------------------------------------------------------

def shard_batch(batch: Dict[str, jnp.ndarray], num_devices: int) -> Dict[str, jnp.ndarray]:
    """Reshape batch so the leading axis is ``num_devices``.

    (B, ...) → (num_devices, B // num_devices, ...)
    """
    def _shard(x):
        assert x.shape[0] % num_devices == 0, (
            f"Batch dim {x.shape[0]} not divisible by {num_devices}"
        )
        return x.reshape((num_devices, -1) + x.shape[1:])
    return jax.tree.map(_shard, batch)


def unreplicate(tree):
    """Take slice [0] of every leaf (device-0 copy)."""
    return jax.tree.map(lambda x: x[0], tree)


class TrainState:
    """Training state container.

    When using pmap the ``params``, ``opt_state`` and ``ema_params`` pytrees
    have an extra leading device axis of size ``num_devices``.

    Attributes:
        params: Model parameters (replicated across devices when using pmap).
        opt_state: Optimizer state (replicated).
        ema_params: EMA teacher parameters (replicated).
        epoch: Current epoch.
        global_step: Global training step.
        best_pq: Best PQ metric achieved.
    """

    def __init__(
        self,
        params: Any,
        opt_state: Any,
        ema_params: Any = None,
        epoch: int = 0,
        global_step: int = 0,
        best_pq: float = 0.0,
    ):
        self.params = params
        self.opt_state = opt_state
        self.ema_params = ema_params
        self.epoch = epoch
        self.global_step = global_step
        self.best_pq = best_pq


class MBPSTrainer:
    """Main MBPS training orchestrator with multi-device pmap support.

    Args:
        config: Training configuration dict.
        model: MBPS model instance.
        optimizer: Optax optimizer.
        num_devices: Number of devices to use (default: all available).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        num_devices: Optional[int] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.num_devices = num_devices or jax.local_device_count()
        self._is_coordinator = jax.process_index() == 0
        self._process_index = jax.process_index()
        self._num_processes = jax.process_count()

        logging.info(
            f"MBPSTrainer: process {self._process_index}/{self._num_processes}, "
            f"using {self.num_devices} local device(s) "
            f"({jax.default_backend()})"
        )

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
        self.instance_loss = CutS3DInstanceLoss(
            lambda_drop=config["loss_weights"]["lambda_drop"],
            lambda_box=config["loss_weights"]["lambda_box"],
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

        # Build the pmapped train step (done once, reused every step)
        self._p_train_step = jax.pmap(
            self._train_step_fn,
            axis_name="batch",
            static_broadcasted_argnums=(4,),  # ``epoch`` is static
        )

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
            logging.warning("wandb not installed — skipping W&B init")
            return
        if not self._is_coordinator:
            logging.info("Skipping W&B init on non-coordinator process")
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
        logging.info(f"W&B initialized: project={log_cfg.get('wandb_project')}, group={group}")

    @property
    def _use_wandb(self) -> bool:
        return HAS_WANDB and getattr(self, "_wandb_active", False)

    # ------------------------------------------------------------------
    # State creation
    # ------------------------------------------------------------------

    def create_train_state(
        self,
        rng: jax.Array,
        dummy_input: Dict[str, jnp.ndarray],
    ) -> TrainState:
        """Initialise model and replicate across devices.

        Args:
            rng: PRNG key.
            dummy_input: Dummy input for ``model.init``.

        Returns:
            A ``TrainState`` whose pytrees have a leading device axis.
        """
        params = self.model.init(rng, **dummy_input)
        opt_state = self.optimizer.init(params)
        ema_params = jax.tree.map(jnp.copy, params)

        devices = jax.local_devices()[: self.num_devices]
        params = jax.device_put_replicated(params, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)
        ema_params = jax.device_put_replicated(ema_params, devices)

        logging.info(f"Parameters replicated to {self.num_devices} devices")

        return TrainState(
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )

    # ------------------------------------------------------------------
    # Loss computation (runs per-device inside pmap)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        params: Any,
        batch: Dict[str, jnp.ndarray],
        epoch: int,
        rng: jax.Array,
        ema_params: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the total loss for one per-device micro-batch."""
        phase_config = self.curriculum.get_config(epoch)
        weights = self.curriculum.get_loss_weights(epoch)

        model_output = self.model.apply(
            params,
            image=batch["image"],
            depth=batch["depth"],
            use_bridge=phase_config.use_bridge,
            deterministic=False,
            rngs={"dropout": rng},
        )

        loss_dict: Dict[str, jnp.ndarray] = {}
        total_loss = jnp.array(0.0)

        # Semantic loss (always active)
        sem_losses = self.semantic_loss(
            semantic_codes=model_output["semantic_codes"],
            dino_features=model_output["dino_features"],
            depth=batch["depth"].reshape(batch["depth"].shape[0], -1),
            key=rng,
        )
        loss_dict["L_semantic"] = sem_losses["total"]
        l_sem_safe = jnp.where(
            jnp.isfinite(sem_losses["total"]), sem_losses["total"], 0.0
        )
        total_loss += weights["alpha"] * l_sem_safe

        # Collapse monitoring: track cluster usage diversity
        sem_pred = jnp.argmax(model_output["semantic_codes"], axis=-1)  # (B, N)
        num_classes = model_output["semantic_codes"].shape[-1]
        one_hot = jax.nn.one_hot(sem_pred, num_classes)  # (B, N, K)
        class_counts = one_hot.sum(axis=(0, 1))  # (K,)
        clusters_used = jnp.sum(class_counts > 0)
        max_cluster_frac = class_counts.max() / jnp.clip(class_counts.sum(), 1.0, None)
        loss_dict["monitor/clusters_used"] = clusters_used.astype(jnp.float32)
        loss_dict["monitor/max_cluster_frac"] = max_cluster_frac

        # Instance loss (Phase B onwards)
        if weights["beta"] > 0:
            # Extract pseudo masks from batch if available
            target_masks = batch.get("pseudo_masks")  # (B, M, K) or None
            spatial_confidence = batch.get("spatial_confidence")  # (B, M, K) or None
            num_valid = batch.get("num_valid_masks")  # (B,) or None

            # Build matched mask: True for valid instances
            matched = None
            if target_masks is not None and num_valid is not None:
                M = target_masks.shape[1]
                matched = jnp.arange(M)[None, :] < num_valid[:, None]  # (B, M)

            inst_losses = self.instance_loss(
                pred_masks=model_output["instance_masks"],
                pred_scores=model_output["instance_scores"],
                features=model_output["dino_features"],
                target_masks=target_masks,
                spatial_confidence=spatial_confidence,
                matched=matched,
                num_valid=num_valid,
            )
            loss_dict["L_instance"] = inst_losses["total"]
            l_inst_safe = jnp.where(
                jnp.isfinite(inst_losses["total"]), inst_losses["total"], 0.0
            )
            total_loss += weights["beta"] * l_inst_safe
        else:
            loss_dict["L_instance"] = jnp.array(0.0)

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
                align_loss=model_output.get("align_loss", jnp.array(0.0)),
            )
            loss_dict["L_bridge"] = bridge_losses["total"]
            l_bridge_safe = jnp.where(
                jnp.isfinite(bridge_losses["total"]),
                bridge_losses["total"],
                0.0,
            )
            total_loss += weights["gamma"] * l_bridge_safe
        else:
            loss_dict["L_bridge"] = jnp.array(0.0)

        # Consistency loss (Phase C onwards)
        if weights["delta"] > 0:
            sem_pred = jnp.argmax(model_output["semantic_codes"], axis=-1)
            mask_probs = jax.nn.sigmoid(model_output["instance_masks"])

            image_size = self.config["data"]["image_size"]
            spatial_h = image_size[0] // 8  # patch stride = 8
            spatial_w = image_size[1] // 8

            # Downsample depth to patch resolution (average pooling)
            b = batch["depth"].shape[0]
            depth_2d = batch["depth"].reshape(b, image_size[0], image_size[1])
            depth_patches = depth_2d.reshape(
                b, spatial_h, 8, spatial_w, 8
            ).mean(axis=(2, 4))  # (B, spatial_h, spatial_w)
            depth_flat = depth_patches.reshape(b, -1)  # (B, N)

            cons_losses = self.consistency_loss(
                semantic_pred=sem_pred,
                instance_masks=mask_probs,
                depth=depth_flat,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            loss_dict["L_consistency"] = cons_losses["total"]
            l_cons_safe = jnp.where(
                jnp.isfinite(cons_losses["total"]),
                cons_losses["total"],
                0.0,
            )
            total_loss += weights["delta"] * l_cons_safe
        else:
            loss_dict["L_consistency"] = jnp.array(0.0)

        # PQ proxy loss (Phase C onwards, requires EMA teacher)
        if weights["epsilon"] > 0 and ema_params is not None:
            # Teacher runs WITHOUT bridge — PQ loss only uses instance
            # outputs (masks, scores) which are computed before the bridge.
            # Running bridge in teacher wastes compute and risks NaN from
            # untrained EMA bridge params flowing into the trace.
            teacher_output = self.model.apply(
                ema_params,
                image=batch["image"],
                depth=batch["depth"],
                use_bridge=False,
                deterministic=True,
            )
            pq_losses = self.pq_loss(
                pred_masks=model_output["instance_masks"],
                pred_scores=model_output["instance_scores"],
                teacher_masks=teacher_output["instance_masks"],
                teacher_scores=teacher_output["instance_scores"],
            )
            loss_dict["L_pq"] = pq_losses["total"]
            l_pq_safe = jnp.where(
                jnp.isfinite(pq_losses["total"]),
                pq_losses["total"],
                0.0,
            )
            total_loss += weights["epsilon"] * l_pq_safe
        else:
            loss_dict["L_pq"] = jnp.array(0.0)

        loss_dict["L_total"] = total_loss

        # NaN monitoring: flag which loss components are NaN
        for key in ["L_semantic", "L_instance", "L_bridge", "L_consistency", "L_pq"]:
            if key in loss_dict:
                loss_dict[f"nan/{key}"] = (~jnp.isfinite(loss_dict[key])).astype(
                    jnp.float32
                )

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Single train step (runs inside jax.pmap)
    # ------------------------------------------------------------------

    def _train_step_fn(
        self,
        params: Any,
        opt_state: Any,
        ema_params: Any,
        batch: Dict[str, jnp.ndarray],
        epoch: int,
        rng: jax.Array,
    ) -> Tuple[Any, Any, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """One optimiser step executed on each device.

        This function is passed to ``jax.pmap`` with ``axis_name="batch"``
        so that ``lax.pmean`` averages gradients across devices.
        """

        def loss_fn(p):
            return self.compute_loss(p, batch, epoch, rng, ema_params)

        (loss_val, loss_dict), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        # Average gradients and losses across devices
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss_val = jax.lax.pmean(loss_val, axis_name="batch")
        loss_dict = jax.tree.map(
            lambda x: jax.lax.pmean(x, axis_name="batch"), loss_dict
        )

        # Optimiser update
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # EMA update (per-device, all identical after pmean)
        momentum = self.config["training"]["ema_momentum"]
        new_ema = jax.tree.map(
            lambda e, p: momentum * e + (1.0 - momentum) * p,
            ema_params, new_params,
        )

        return new_params, new_opt_state, new_ema, loss_val, loss_dict

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        state: TrainState,
        train_data,
        rng: jax.Array,
        steps_per_epoch: int = 0,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Train for one epoch across all devices.

        ``train_data`` must yield dicts whose batch dimension is divisible
        by ``self.num_devices``.

        Args:
            steps_per_epoch: If > 0, stop after this many steps (required
                for multi-host training with repeated datasets).
        """
        epoch_metrics: Dict[str, float] = {}
        num_steps = 0
        start_time = time.time()

        for batch in train_data:
            # Drop string fields (e.g. image_id) that can't go to JAX
            batch = {
                k: v for k, v in batch.items()
                if not (hasattr(v, 'dtype') and v.dtype == tf.string)
                and not (hasattr(v, 'dtype') and str(v.dtype) == 'string')
            }

            # Convert numpy/TF arrays to jax arrays if needed
            batch = jax.tree.map(
                lambda x: jnp.array(x) if not isinstance(x, jnp.ndarray) else x,
                batch,
            )

            # Skip batches that can't be evenly sharded
            if batch["image"].shape[0] % self.num_devices != 0:
                continue

            # Shard across devices
            batch = shard_batch(batch, self.num_devices)

            # Per-device PRNG keys
            rng, *step_rngs = jax.random.split(rng, self.num_devices + 1)
            step_rngs = jnp.array(step_rngs)

            new_params, new_opt_state, new_ema, loss_val, loss_dict = (
                self._p_train_step(
                    state.params,
                    state.opt_state,
                    state.ema_params,
                    batch,
                    state.epoch,
                    step_rngs,
                )
            )

            state.params = new_params
            state.opt_state = new_opt_state
            state.ema_params = new_ema
            state.global_step += 1

            # Collect metrics from device 0
            for k, v in loss_dict.items():
                val = float(v[0])
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + val
            num_steps += 1

            # For multi-host: break after fixed step count to keep workers in sync
            if steps_per_epoch > 0 and num_steps >= steps_per_epoch:
                break

            if num_steps % self.config["logging"]["log_every_n_steps"] == 0:
                step_loss = float(loss_val[0])
                parts = [f"L_total={step_loss:.4f}"]
                for lk in ["L_semantic", "L_instance", "L_bridge",
                           "L_consistency", "L_pq"]:
                    if lk in loss_dict:
                        parts.append(f"{lk}={float(loss_dict[lk][0]):.4f}")
                # Count NaN losses this step
                nan_count = sum(
                    1 for k in loss_dict
                    if k.startswith("nan/") and float(loss_dict[k][0]) > 0
                )
                if nan_count > 0:
                    parts.append(f"nan_losses={nan_count}")
                logging.info(
                    f"Epoch {state.epoch}, Step {num_steps}: "
                    + ", ".join(parts)
                )
                if self._use_wandb and self._is_coordinator:
                    step_metrics = {
                        f"step/{k}": float(v[0]) for k, v in loss_dict.items()
                    }
                    step_metrics["step/global_step"] = state.global_step
                    wandb.log(step_metrics, step=state.global_step)

        # Average over steps
        epoch_time = time.time() - start_time
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_steps, 1)
        epoch_metrics["epoch_time_s"] = epoch_time
        epoch_metrics["phase"] = self.curriculum.get_phase(state.epoch)

        # Collapse warning
        clusters_used = epoch_metrics.get("monitor/clusters_used", 0)
        max_frac = epoch_metrics.get("monitor/max_cluster_frac", 0)
        if clusters_used <= 3:
            logging.warning(
                f"COLLAPSE WARNING: only {clusters_used:.0f} clusters used, "
                f"max_cluster_frac={max_frac:.2%}"
            )

        # W&B epoch-level logging
        if self._use_wandb and self._is_coordinator:
            wandb_metrics = {
                f"epoch/{k}": v for k, v in epoch_metrics.items()
                if isinstance(v, (int, float))
            }
            wandb_metrics["epoch/epoch"] = state.epoch
            wandb.log(wandb_metrics, step=state.global_step)

        return state, epoch_metrics

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def train(
        self,
        state: TrainState,
        train_data,
        val_data=None,
        rng: jax.Array = None,
        start_epoch: int = 1,
        steps_per_epoch: int = 0,
    ) -> TrainState:
        """Run full training pipeline (Phases A-D).

        Args:
            state: Initial training state (replicated).
            train_data: Training data iterator.
            val_data: Optional validation data.
            rng: PRNG key.
            start_epoch: Epoch to start from (for resuming).

        Returns:
            Final training state.
        """
        if rng is None:
            rng = jax.random.PRNGKey(42)

        total_epochs = self.config["training"]["total_epochs"]

        for epoch in range(start_epoch, total_epochs + 1):
            state.epoch = epoch
            phase = self.curriculum.get_phase(epoch)

            logging.info(
                f"=== Epoch {epoch}/{total_epochs} "
                f"(Phase {phase}) ==="
            )

            rng, epoch_rng = jax.random.split(rng)
            state, metrics = self.train_epoch(
                state, train_data, epoch_rng, steps_per_epoch=steps_per_epoch,
            )

            # Log
            logging.info(
                f"Epoch {epoch} done: "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                            if isinstance(v, float))
            )

            # Checkpoint (unreplicate before saving, rank-0 only)
            if epoch % self.config["checkpointing"]["save_every_n_epochs"] == 0:
                if self._is_coordinator:
                    self.ckpt_manager.save(
                        epoch,
                        unreplicate(state.params),
                        unreplicate(state.opt_state),
                        ema_params=unreplicate(state.ema_params),
                        metrics=metrics,
                    )

        # Phase D: Self-training
        logging.info("=== Starting Phase D: Self-Training ===")
        state = self._run_self_training(
            state, train_data, rng, steps_per_epoch=steps_per_epoch
        )

        # Finish W&B run
        if self._use_wandb and self._is_coordinator:
            wandb.finish()

        return state

    def _run_self_training(
        self,
        state: TrainState,
        train_data,
        rng: jax.Array,
        steps_per_epoch: int = 0,
    ) -> TrainState:
        """Run Phase D self-training rounds."""
        for round_idx in range(self.self_trainer.num_rounds):
            logging.info(
                f"Self-training round {round_idx + 1}/"
                f"{self.self_trainer.num_rounds}"
            )

            epochs_per_round = self.config["training"][
                "self_training_epochs_per_round"
            ]
            for ep in range(epochs_per_round):
                rng, epoch_rng = jax.random.split(rng)
                state, metrics = self.train_epoch(
                    state, train_data, epoch_rng,
                    steps_per_epoch=steps_per_epoch,
                )
                logging.info(
                    f"  Round {round_idx + 1}, Epoch {ep + 1}: "
                    f"L_total={metrics.get('L_total', 0):.4f}"
                )

            # Advance threshold
            self.self_trainer.advance_round()

            # Save checkpoint (unreplicate first, rank-0 only)
            if self._is_coordinator:
                self.ckpt_manager.save(
                    state.epoch + round_idx * epochs_per_round,
                    unreplicate(state.params),
                    unreplicate(state.opt_state),
                    ema_params=unreplicate(state.ema_params),
                )

        return state
