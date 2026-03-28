"""Checkpointing utilities using PyTorch native save/load.

Handles saving and loading model checkpoints, including:
    - Model parameters (state_dict)
    - Optimizer state
    - EMA parameters
    - Training state (epoch, step, best metric)
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints.

    Saves and loads checkpoints using ``torch.save`` / ``torch.load``.
    Supports keeping only the top-k best checkpoints by metric value.

    Args:
        checkpoint_dir: Directory for checkpoints.
        keep_top_k: Number of best checkpoints to keep.
        save_every_n_epochs: Save frequency.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_top_k: int = 3,
        save_every_n_epochs: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self.save_every_n_epochs = save_every_n_epochs
        self.best_metrics: List[Tuple[float, str]] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Dict[str, Any],
        ema_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save checkpoint.

        Args:
            epoch: Current epoch.
            model_state_dict: Model ``state_dict()``.
            optimizer_state_dict: Optimizer ``state_dict()``.
            ema_state_dict: EMA parameter state dict.
            metrics: Training metrics.

        Returns:
            Path to saved checkpoint file.
        """
        ckpt_name = f"checkpoint_epoch_{epoch:04d}.pt"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "metrics": metrics or {},
        }

        if ema_state_dict is not None:
            checkpoint["ema_state_dict"] = ema_state_dict

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        return ckpt_path

    def save_best(
        self,
        epoch: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Dict[str, Any],
        metric_value: float,
        ema_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save checkpoint if metric is among top-k best.

        Args:
            epoch: Current epoch.
            model_state_dict: Model ``state_dict()``.
            optimizer_state_dict: Optimizer ``state_dict()``.
            metric_value: Metric value (higher = better).
            ema_state_dict: EMA parameter state dict.

        Returns:
            Path if saved, None if not in top-k.
        """
        if len(self.best_metrics) < self.keep_top_k or metric_value > min(
            m[0] for m in self.best_metrics
        ):
            path = self.save(
                epoch, model_state_dict, optimizer_state_dict,
                ema_state_dict,
                metrics={"pq": metric_value},
            )

            self.best_metrics.append((metric_value, path))
            self.best_metrics.sort(reverse=True)

            # Remove worst if over limit
            while len(self.best_metrics) > self.keep_top_k:
                _, old_path = self.best_metrics.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
                logger.info(f"Removing old checkpoint: {old_path}")

            return path
        return None

    def load(
        self,
        checkpoint_path: str,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint ``.pt`` file.
            map_location: Device mapping (e.g., ``'cpu'``,
                ``torch.device('cuda:0')``).  Passed directly to
                ``torch.load``.

        Returns:
            Dict with 'model_state_dict', 'optimizer_state_dict',
            optionally 'ema_state_dict', 'epoch', 'metrics'.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint.

        Returns:
            Path string or None if no checkpoints exist.
        """
        if not os.path.exists(self.checkpoint_dir):
            return None

        ckpts = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
        ])

        return os.path.join(self.checkpoint_dir, ckpts[-1]) if ckpts else None
