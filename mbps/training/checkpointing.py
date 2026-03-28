"""Checkpointing utilities using Orbax.

Handles saving and loading model checkpoints, including:
    - Model parameters
    - Optimizer state
    - EMA parameters
    - Training state (epoch, step, best metric)
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import logging

try:
    import orbax.checkpoint as ocp
    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False
    logging.warning("orbax not found. Using simple numpy checkpointing.")


class CheckpointManager:
    """Manage model checkpoints.

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
        self.best_metrics: list[Tuple[float, str]] = []

        tf.io.gfile.makedirs(checkpoint_dir)

    def save(
        self,
        epoch: int,
        params: Any,
        opt_state: Any,
        ema_params: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save checkpoint.

        Args:
            epoch: Current epoch.
            params: Model parameters.
            opt_state: Optimizer state.
            ema_params: EMA parameters.
            metrics: Training metrics.

        Returns:
            Path to saved checkpoint.
        """
        ckpt_name = f"checkpoint_epoch_{epoch:04d}"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        tf.io.gfile.makedirs(ckpt_path)

        # Save using numpy via BytesIO (works with both local and gs:// paths)
        flat_params = _flatten_pytree(params, "params")
        for key, val in flat_params.items():
            _save_npy(os.path.join(ckpt_path, f"{key}.npy"), np.array(val))

        # Save metadata
        metadata = {
            "epoch": epoch,
            "metrics": metrics or {},
        }
        _save_npy(os.path.join(ckpt_path, "metadata.npy"), metadata)

        if ema_params is not None:
            flat_ema = _flatten_pytree(ema_params, "ema")
            for key, val in flat_ema.items():
                _save_npy(
                    os.path.join(ckpt_path, f"{key}.npy"), np.array(val),
                )

        logging.info(f"Saved checkpoint: {ckpt_path}")
        return ckpt_path

    def save_best(
        self,
        epoch: int,
        params: Any,
        opt_state: Any,
        metric_value: float,
        ema_params: Optional[Any] = None,
    ) -> Optional[str]:
        """Save checkpoint if metric is among top-k best.

        Args:
            epoch: Current epoch.
            params: Model parameters.
            opt_state: Optimizer state.
            metric_value: Metric value (higher = better).
            ema_params: EMA parameters.

        Returns:
            Path if saved, None if not in top-k.
        """
        if len(self.best_metrics) < self.keep_top_k or metric_value > min(
            m[0] for m in self.best_metrics
        ):
            path = self.save(
                epoch, params, opt_state, ema_params,
                metrics={"pq": metric_value}
            )

            self.best_metrics.append((metric_value, path))
            self.best_metrics.sort(reverse=True)

            # Remove worst if over limit
            while len(self.best_metrics) > self.keep_top_k:
                _, old_path = self.best_metrics.pop()
                if tf.io.gfile.exists(old_path):
                    tf.io.gfile.rmtree(old_path)
                logging.info(f"Removing old checkpoint: {old_path}")

            return path
        return None

    def load(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Dict with 'params', 'ema_params', 'metadata'.
        """
        result = {"params": {}, "ema_params": {}, "metadata": {}}

        for fname in tf.io.gfile.listdir(checkpoint_path):
            fpath = os.path.join(checkpoint_path, fname)
            if fname == "metadata.npy":
                result["metadata"] = _load_npy(fpath, allow_pickle=True).item()
            elif fname.startswith("params_"):
                key = fname.replace(".npy", "")
                result["params"][key] = jnp.array(_load_npy(fpath))
            elif fname.startswith("ema_"):
                key = fname.replace(".npy", "")
                result["ema_params"][key] = jnp.array(_load_npy(fpath))

        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return result

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint.

        Returns:
            Path string or None if no checkpoints exist.
        """
        if not tf.io.gfile.exists(self.checkpoint_dir):
            return None

        ckpts = sorted([
            d for d in tf.io.gfile.listdir(self.checkpoint_dir)
            if d.startswith("checkpoint_epoch_")
        ])

        return os.path.join(self.checkpoint_dir, ckpts[-1]) if ckpts else None


def _save_npy(path: str, arr: Any) -> None:
    """Save numpy array to local or GCS path via BytesIO."""
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    with tf.io.gfile.GFile(path, "wb") as f:
        f.write(buf.read())


def _load_npy(path: str, allow_pickle: bool = False) -> np.ndarray:
    """Load numpy array from local or GCS path via BytesIO."""
    with tf.io.gfile.GFile(path, "rb") as f:
        buf = io.BytesIO(f.read())
    return np.load(buf, allow_pickle=allow_pickle)


def _unflatten_pytree(flat: Dict[str, Any], prefix: str = "params") -> Dict:
    """Unflatten a flat dict back into a nested pytree.

    Args:
        flat: Flat dict with keys like 'params_backbone_dense_kernel'.
        prefix: Top-level prefix to strip (e.g. 'params').

    Returns:
        Nested dict suitable for model.apply.
    """
    tree: Dict = {}
    for key, val in flat.items():
        # Strip the prefix (e.g. 'params_backbone_...' -> 'backbone_...')
        if key.startswith(prefix + "_"):
            parts = key[len(prefix) + 1:].split("_")
        else:
            parts = key.split("_")

        # Build nested dict
        node = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = val
    return tree


def _flatten_pytree(
    tree: Any, prefix: str = ""
) -> Dict[str, np.ndarray]:
    """Flatten a JAX pytree to a flat dict.

    Args:
        tree: JAX pytree (nested dict/list).
        prefix: Key prefix.

    Returns:
        Flat dict mapping string keys to numpy arrays.
    """
    flat = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            child = _flatten_pytree(v, f"{prefix}_{k}" if prefix else k)
            flat.update(child)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            child = _flatten_pytree(v, f"{prefix}_{i}" if prefix else str(i))
            flat.update(child)
    elif hasattr(tree, "__array__"):
        flat[prefix] = np.array(tree)
    else:
        flat[prefix] = tree
    return flat
