#!/usr/bin/env python3
"""Standalone evaluation for CAUSE-TR Mode B.

Loads a Mode B checkpoint (segment_tr.pth + cluster_tr.pth + adapter.pth) and
runs Cityscapes val with 27-class Hungarian-matched mIoU. Optionally exports
per-image prediction PNGs in CUPS-compatible format for downstream Cascade
Mask R-CNN training.

Usage:
    python mbps_pytorch/scripts/eval_cause_modeb.py \
        --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
        --checkpoint refs/cause/CAUSE_modeb_dinov3_vitb16/best

    # With pseudo-label export:
    python mbps_pytorch/scripts/eval_cause_modeb.py \
        --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
        --checkpoint refs/cause/CAUSE_modeb_dinov3_vitb16/best \
        --export_pseudolabels pseudo_semantic_modeb/val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mbps_pytorch.train_cause_modeb_dinov3 import (  # noqa: E402
    build_loaders,
    build_model_and_trainer,
    load_yaml,
    pick_device,
    setup_logging,
)

logger = logging.getLogger(__name__)


def load_modeb_checkpoint(trainer, ckpt_dir: Path) -> None:
    """Load segment_tr.pth, cluster_tr.pth, adapter.pth from ckpt_dir.

    Frozen tensors (codebook, cluster_probe) are validated to be identical
    to what we already loaded from the npz — drift here is a hard error.
    """
    seg_pth = ckpt_dir / "segment_tr.pth"
    clu_pth = ckpt_dir / "cluster_tr.pth"
    ada_pth = ckpt_dir / "adapter.pth"
    for p in (seg_pth, clu_pth, ada_pth):
        if not p.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {p}")

    seg_sd = torch.load(seg_pth, map_location=trainer.device, weights_only=False)
    trainer.segment.load_state_dict(seg_sd, strict=False)

    clu_sd = torch.load(clu_pth, map_location=trainer.device, weights_only=False)
    cb_loaded = clu_sd["codebook"].to(trainer.device)
    cp_loaded = clu_sd["cluster_probe"].to(trainer.device)
    # Tolerance = float32 ULP (~3e-8). Round-tripping through npz->torch->save->load can
    # introduce 1-ULP drift on a fraction of cells without changing semantics.
    if not torch.allclose(cb_loaded, trainer.cluster.codebook.detach(), atol=1e-7, rtol=0):
        max_d = (cb_loaded - trainer.cluster.codebook.detach()).abs().max().item()
        raise RuntimeError(
            f"Codebook in checkpoint differs from frozen codebook (max diff={max_d:.3e}). "
            "Mode B should NEVER train the codebook."
        )
    if not torch.allclose(cp_loaded, trainer.cluster.cluster_probe.detach(), atol=1e-7, rtol=0):
        max_d = (cp_loaded - trainer.cluster.cluster_probe.detach()).abs().max().item()
        raise RuntimeError(
            f"cluster_probe in checkpoint differs from frozen probe (max diff={max_d:.3e})."
        )

    ada_sd = torch.load(ada_pth, map_location=trainer.device, weights_only=False)
    trainer.adapter.load_state_dict(ada_sd, strict=True)
    logger.info("Loaded Mode B checkpoint from %s", ckpt_dir)


@torch.no_grad()
def export_pseudolabels(trainer, val_loader, out_dir: Path) -> None:
    """Run val inference and write per-image 27-class semantic PNGs (uint8).

    Hungarian matching is NOT applied — the raw cluster_probe argmax (in [0..26]) is
    written directly. Downstream training scripts already handle the Cityscapes
    labelID remap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.segment.eval()
    trainer.adapter.eval()

    img_paths = []
    underlying = val_loader.dataset
    if hasattr(underlying, "dataset"):  # Subset
        for i in underlying.indices:
            img_paths.append(underlying.dataset.image_paths[i])
    else:
        img_paths = list(underlying.image_paths)
    img_idx = 0

    for batch in val_loader:
        imgs = batch["img"].to(trainer.device, non_blocking=True)
        feat = trainer._backbone_features(imgs)
        adapted = trainer.adapter(feat)
        from modules.segment_module import transform  # noqa: WPS433
        seg = transform(trainer.segment.head_ema(adapted))

        if trainer.cfg.flip_tta:
            feat_flip = trainer._backbone_features(imgs.flip(dims=[3]))
            adapted_flip = trainer.adapter(feat_flip)
            seg_flip = transform(trainer.segment.head_ema(adapted_flip))
            seg = (seg + seg_flip.flip(dims=[3])) / 2.0

        h_lbl, w_lbl = batch["label"].shape[-2:]
        spatial_up = torch.nn.functional.interpolate(
            seg, size=(h_lbl, w_lbl), mode="bilinear", align_corners=False,
        )
        normed_feats = torch.nn.functional.normalize(spatial_up, dim=1)
        normed_probes = torch.nn.functional.normalize(
            trainer.cluster.cluster_probe.detach(), dim=1,
        )
        inner = torch.einsum("bchw,nc->bnhw", normed_feats, normed_probes)
        preds = inner.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W) in [0..26]

        for b in range(preds.shape[0]):
            img_path = img_paths[img_idx]
            img_idx += 1
            stem = Path(img_path).stem.replace("_leftImg8bit", "")
            Image.fromarray(preds[b]).save(out_dir / f"{stem}.png")
    logger.info("Exported %d pseudo-label PNGs to %s", img_idx, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval CAUSE-TR Mode B checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint directory containing segment_tr.pth, cluster_tr.pth, adapter.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--export_pseudolabels", type=str, default=None,
                        help="if set, export per-image 27-class PNGs to this dir")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(Path(args.config))
    device = pick_device(args.device)
    logger.info("Device: %s", device)

    # Build pipeline (same as training) so we can load weights into it.
    trainer = build_model_and_trainer(cfg, device)
    _, val_loader = build_loaders(cfg)

    ckpt_dir = Path(args.checkpoint)
    load_modeb_checkpoint(trainer, ckpt_dir)

    metrics = trainer.validate(val_loader)
    logger.info("Eval mIoU=%.2f pAcc=%.2f", metrics["mIoU"], metrics["pAcc"])
    out_json = ckpt_dir / "eval_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out_json)

    if args.export_pseudolabels is not None:
        export_pseudolabels(trainer, val_loader, Path(args.export_pseudolabels))


if __name__ == "__main__":
    main()
