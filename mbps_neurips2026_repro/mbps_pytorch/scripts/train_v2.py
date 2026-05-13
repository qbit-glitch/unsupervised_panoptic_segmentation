"""MBPS v2 Training Script (PyTorch).

Usage:
    python mbps_pytorch/scripts/train_v2.py --config configs/v2_cityscapes.yaml --seed 42

    # With ablation:
    python mbps_pytorch/scripts/train_v2.py --config configs/v2_cityscapes.yaml \
        --ablation configs/v2_ablations/no_mamba.yaml --seed 42

    # Resume:
    python mbps_pytorch/scripts/train_v2.py --config configs/v2_cityscapes.yaml \
        --resume checkpoints/v2_default/local/checkpoint_epoch_0020.pt

v2 training is a simplified 2-phase pipeline:
    Phase 1 (Bootstrap): Train on pseudo-labels with CE + discriminative loss
    Phase 2 (Self-training): EMA teacher generates refined pseudo-labels
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mbps_pytorch.models.mbps_v2_model import MBPSv2Model
from mbps_pytorch.training.trainer_v2 import MBPSv2Trainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge YAML config with v2 defaults."""
    default_path = os.path.join(os.path.dirname(config_path), "v2_default.yaml")
    with open(default_path) as f:
        config = yaml.safe_load(f)

    with open(config_path) as f:
        override = yaml.safe_load(f)

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    if override:
        deep_merge(config, override)
    return config


def create_v2_dataloader(config: Dict[str, Any]) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader for v2 training.

    Loads data from TFRecords-equivalent format or local files.
    Expects: image, depth, pseudo_semantic, pseudo_instance per sample.
    """
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    dc = config["data"]
    batch_size = dc.get("batch_size", 4)
    data_dir = dc.get("data_dir", "")
    image_size = tuple(dc.get("image_size", [512, 1024]))

    class V2Dataset(Dataset):
        """Loads pseudo-labeled Cityscapes data.

        Expects directory structure:
            data_dir/images/*.png
            data_dir/depth/*.npy
            data_dir/pseudo_semantic/*.png
            data_dir/pseudo_instance/*.npz
        """

        def __init__(self, data_dir: str, image_size: tuple):
            self.data_dir = data_dir
            self.image_size = image_size
            self.samples = []

            img_dir = os.path.join(data_dir, "images")
            if os.path.isdir(img_dir):
                self.samples = sorted([
                    f.replace(".png", "").replace(".jpg", "")
                    for f in os.listdir(img_dir)
                    if f.endswith((".png", ".jpg"))
                ])

            if not self.samples:
                logger.warning(f"No samples found in {data_dir}")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            name = self.samples[idx]
            h, w = self.image_size

            # Image
            from PIL import Image
            img_path = os.path.join(self.data_dir, "images", f"{name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, "images", f"{name}.jpg")
            img = Image.open(img_path).convert("RGB").resize((w, h))
            img = np.array(img, dtype=np.float32) / 255.0
            # ImageNet normalize
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

            # Depth
            depth_path = os.path.join(self.data_dir, "depth", f"{name}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.float32)
                depth = torch.from_numpy(depth)
                if depth.shape != (h, w):
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0).unsqueeze(0),
                        size=(h, w), mode="bilinear", align_corners=False,
                    ).squeeze()
            else:
                depth = torch.zeros(h, w)

            # Pseudo-semantic labels (patch-level)
            ps = dc.get("patch_size", 16)
            ph, pw = h // ps, w // ps
            sem_path = os.path.join(self.data_dir, "pseudo_semantic", f"{name}.npy")
            if os.path.exists(sem_path):
                sem = np.load(sem_path).astype(np.int64)
                sem = torch.from_numpy(sem).long()
                if sem.numel() != ph * pw:
                    # Pixel-level -> patch-level via mode
                    sem_img = sem.reshape(h, w) if sem.numel() == h * w else sem
                    sem = sem_img[:ph * ps, :pw * ps].reshape(ph, ps, pw, ps)
                    sem = sem.float().mean(dim=(1, 3)).round().long()
                sem = sem.reshape(ph * pw)
            else:
                sem = torch.zeros(ph * pw, dtype=torch.long)

            # Pseudo-instance labels (patch-level)
            inst_path = os.path.join(self.data_dir, "pseudo_instance", f"{name}.npy")
            if os.path.exists(inst_path):
                inst = np.load(inst_path).astype(np.int64)
                inst = torch.from_numpy(inst).long()
                if inst.numel() != ph * pw:
                    inst_img = inst.reshape(h, w) if inst.numel() == h * w else inst
                    inst = inst_img[:ph * ps, :pw * ps].reshape(ph, ps, pw, ps)
                    inst = inst.float().mean(dim=(1, 3)).round().long()
                inst = inst.reshape(ph * pw)
            else:
                inst = torch.zeros(ph * pw, dtype=torch.long)

            return {
                "image": img,
                "depth": depth,
                "pseudo_semantic": sem,
                "pseudo_instance": inst,
            }

    dataset = V2Dataset(data_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dc.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    return loader


def main():
    parser = argparse.ArgumentParser(description="MBPS v2 Training (PyTorch)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--vm_name", type=str,
                        default=os.environ.get("MBPS_VM_NAME", "local"))
    parser.add_argument("--experiment", type=str,
                        default=os.environ.get("MBPS_EXPERIMENT", "v2_default"))
    parser.add_argument("--ablation", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Merge ablation
    if args.ablation:
        with open(args.ablation) as f:
            ablation = yaml.safe_load(f)
        if ablation:
            def deep_merge(base, override):
                for k, v in override.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        deep_merge(base[k], v)
                    else:
                        base[k] = v
            deep_merge(config, ablation)
            logger.info(f"Applied ablation: {args.ablation}")

    # Device
    if torch.cuda.is_available():
        num_devices = args.num_devices or torch.cuda.device_count()
        device = torch.device("cuda")
    else:
        num_devices = 1
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Log
    ac = config["architecture"]
    mc = ac.get("mamba", {})
    logger.info(f"MBPS v2 Training (PyTorch)")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Experiment: {args.experiment}")
    logger.info(f"  Device: {device} x {num_devices}")
    logger.info(f"  Backbone: {ac.get('backbone', 'dinov3_vitb')} ({ac.get('backbone_dim', 768)}d)")
    logger.info(f"  Bridge: {ac.get('bridge_dim', 384)}d, mamba={ac.get('use_mamba_bridge', True)}")

    # Create model
    model = MBPSv2Model(
        num_classes=ac.get("num_classes", 19),
        backbone_dim=ac.get("backbone_dim", 768),
        instance_embed_dim=ac.get("instance_embed_dim", 64),
        bridge_dim=ac.get("bridge_dim", 384),
        mamba_layers=mc.get("num_layers", 4),
        mamba_state_dim=mc.get("state_dim", 64),
        chunk_size=mc.get("chunk_size", 128),
        use_depth_conditioning=ac.get("use_depth_conditioning", True),
        use_mamba_bridge=ac.get("use_mamba_bridge", True),
        use_bidirectional=mc.get("use_bidirectional", True),
        dropout_rate=config["training"].get("dropout_rate", 0.1),
    )

    # Load pretrained DINOv3 backbone
    try:
        from mbps_pytorch.models.backbone.dinov3_vitb import DINOv3ViTB
        pretrained_backbone = DINOv3ViTB.from_pretrained()
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        logger.info("Loaded pretrained DINOv3 backbone")
    except Exception as e:
        logger.warning(f"Could not load pretrained DINOv3: {e}")
        logger.warning("Using random init (will produce garbage results)")

    # Checkpoint dir
    ckpt_base = config.get("checkpointing", {}).get("checkpoint_dir", "checkpoints")
    config.setdefault("checkpointing", {})["checkpoint_dir"] = os.path.join(
        ckpt_base, args.experiment, args.vm_name
    )

    # Create data
    train_loader = create_v2_dataloader(config)

    # Create trainer
    trainer = MBPSv2Trainer(config=config, model=model, device=device)
    trainer.create_train_state()

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    if config.get("logging", {}).get("use_wandb", False):
        trainer.init_wandb(vm_name=args.vm_name, experiment_name=args.experiment)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} (trainable: {trainable:,})")

    # Train
    trainer.train(train_loader=train_loader)
    logger.info("v2 training complete!")


if __name__ == "__main__":
    main()
