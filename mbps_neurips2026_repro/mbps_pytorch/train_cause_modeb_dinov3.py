#!/usr/bin/env python3
"""CAUSE-TR Mode B training: DINOv3 ViT-B/16 + frozen DINOv2 codebook & cluster_probe.

Usage:
    python mbps_pytorch/train_cause_modeb_dinov3.py \
        --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
        --device auto

    # Smoke run (1 epoch, 8 batches):
    python mbps_pytorch/train_cause_modeb_dinov3.py \
        --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
        --epochs 1 --max_batches 8 --device auto

See docs/plans/2026-04-27-cause-modeb-dinov3-vitb16.md for design rationale.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

# Project root must be importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Shared dataset and HF-backed DINOv3Backbone helpers from the v9 trainer.
from mbps_pytorch.train_cause_dinov3 import (  # noqa: E402
    CityscapesCAUSE,
    CityscapesCAUSEVal,
    DINOv3Backbone,
)
from mbps_pytorch.training.cause_modeb_cluster import ModeBCluster  # noqa: E402
from mbps_pytorch.models.adapters.dinov3_to_dinov2_adapter import DINOv3ToDINOv2Adapter  # noqa: E402
from mbps_pytorch.training.cause_modeb_freeze import (  # noqa: E402
    install_frozen_into_cluster,
    load_frozen_codebook_and_probe,
    verify_freeze,
    wire_codebook_into_segment,
)
from mbps_pytorch.training.cause_modeb_trainer import (  # noqa: E402
    CAUSEModeBConfig,
    CAUSEModeBTrainer,
)

# CAUSE module path is set up by train_cause_dinov3 import above; pull what we need.
from modules.segment import Segment_TR  # noqa: E402
from modules.segment_module import ema_init  # noqa: E402

logger = logging.getLogger(__name__)


# ---- Helpers ---------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---- Build pipeline --------------------------------------------------- #

class _DINOv3PatchOnly(nn.Module):
    """Wraps DINOv3Backbone to return only patch tokens (B, N, D), dropping CLS.

    DINOv3Backbone returns (B, 1+N, D) (CLS + patches; registers stripped). The
    Mode B trainer expects only patches, matching how CAUSE's `feat = net(img)[:, 1:, :]`
    convention is applied at refs/cause/train_cause_tr_dinov2.py.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.backbone = DINOv3Backbone(model_name=model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, 1:, :]


def build_model_and_trainer(cfg: Dict[str, Any], device: torch.device) -> CAUSEModeBTrainer:
    # 1) Backbone — frozen DINOv3 ViT-B/16
    logger.info("Loading DINOv3 backbone: %s", cfg["backbone"]["model_name"])
    backbone = _DINOv3PatchOnly(cfg["backbone"]["model_name"]).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # 2) Frozen artifacts
    npz_path = _PROJECT_ROOT / cfg["frozen_centroids_npz"]
    codebook, probe = load_frozen_codebook_and_probe(npz_path, device=device)

    # 3) CAUSE args namespace + cluster
    cause_args = SimpleNamespace(
        dim=cfg["cause"]["dim"],
        reduced_dim=cfg["cause"]["reduced_dim"],
        projection_dim=cfg["cause"]["projection_dim"],
        num_codebook=cfg["cause"]["num_codebook"],
        n_classes=cfg["cause"]["n_classes"],
        num_queries=cfg["cause"]["num_queries"],
    )
    cluster = ModeBCluster(cause_args, device).to(device)
    install_frozen_into_cluster(cluster, codebook, probe)
    cluster.bank_init()

    # 4) Segment_TR — built AFTER frozen install so codebook wiring uses the right Parameter
    segment = Segment_TR(cause_args).to(device)
    wire_codebook_into_segment(segment, cluster)
    ema_init(segment.head, segment.head_ema)
    ema_init(segment.projection_head, segment.projection_head_ema)
    # EMA copies must not be optimized.
    for p in segment.head_ema.parameters():
        p.requires_grad_(False)
    for p in segment.projection_head_ema.parameters():
        p.requires_grad_(False)

    # 5) Adapter — Linear(768,768) identity-init
    adapter = DINOv3ToDINOv2Adapter(
        dim=cfg["adapter"]["in_dim"],
        bias=cfg["adapter"].get("bias", True),
        init="identity",
    ).to(device)

    # 6) Verify
    verify_freeze(
        cluster,
        backbone=backbone,
        expected_codebook=codebook,
        expected_probe=probe,
    )

    # 7) Trainer
    tcfg = cfg["training"]
    trainer_cfg = CAUSEModeBConfig(
        epochs=tcfg["epochs"],
        batch_size=tcfg["batch_size"],
        grad_accum_steps=tcfg.get("grad_accum_steps", 1),
        head_lr=float(tcfg["head_lr"]),
        adapter_lr=float(tcfg["adapter_lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        grad_clip_norm=float(tcfg.get("grad_clip_norm", 1.0)),
        ema_decay=float(tcfg.get("ema_decay", 0.99)),
        contrastive_temp=float(tcfg.get("contrastive_temp", 0.07)),
        pos_thresh=float(tcfg.get("pos_thresh", 0.3)),
        neg_thresh=float(tcfg.get("neg_thresh", 0.1)),
        bank_max_size=int(tcfg.get("bank_max_size", 100)),
        loss_weight_contrastive=float(tcfg["loss_weights"]["contrastive"]),
        loss_weight_centroid=float(tcfg["loss_weights"]["centroid"]),
        log_every=int(cfg["logging"].get("log_every", 50)),
        save_every=int(cfg["logging"].get("save_every", 5)),
        output_dir=Path(cfg["logging"]["output_dir"]),
        flip_tta=bool(cfg["validation"].get("flip_tta", True)),
        n_classes=int(cause_args.n_classes),
    )
    return CAUSEModeBTrainer(
        backbone=backbone,
        adapter=adapter,
        segment=segment,
        cluster=cluster,
        device=device,
        cfg=trainer_cfg,
    )


def build_loaders(cfg: Dict[str, Any], max_batches: int | None = None):
    res = cfg["resolution"]
    cs_root = cfg["data"]["cityscapes_root"]
    train_ds = CityscapesCAUSE(cs_root, split="train", resolution=res, augment=True,
                                crop_strategy="rect_then_crop")
    val_ds = CityscapesCAUSEVal(cs_root, resolution=res)

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)

    if max_batches is not None and max_batches > 0:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), max_batches * bs))))
        val_ds = Subset(val_ds, list(range(min(len(val_ds), 8))))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, bs // 2), shuffle=False, num_workers=nw, pin_memory=False,
    )
    return train_loader, val_loader


# ---- Entrypoint ------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAUSE-TR Mode B training (DINOv3 + frozen codebook)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, cpu")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke test)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="limit number of training batches per epoch (smoke test)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(Path(args.config))
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device)
    logger.info("Device: %s", device)
    logger.info("Experiment: %s", cfg["experiment_name"])

    trainer = build_model_and_trainer(cfg, device)
    train_loader, val_loader = build_loaders(cfg, max_batches=args.max_batches)
    trainer.fit(train_loader, val_loader)
    logger.info("Training complete. Best mIoU=%.2f", trainer.best_miou)


if __name__ == "__main__":
    main()
