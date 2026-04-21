"""Smoke test for Stage-2 M2F+ViT-Adapter training configs.

Runs a minimal training loop (config load → model build → 10 steps forward/backward)
to catch crashes before launching full multi-day training runs.

Usage on anydesk:
    module load gcc-9.3.0
    source ~/umesh/ups_env/bin/activate
    cd /home/cvpr_ug_5/umesh/mbps_panoptic_segmentation
    python scripts/smoke_test_stage2_m2f.py --config configs/stage2_m2f/M0_A_only_anydesk.yaml

Exit codes:
    0 = all checks passed
    1 = config load failed
    2 = model build failed
    3 = forward/backward failed
    4 = gradient check failed
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Add refs/cups to path before any detectron2 / cups imports
PROJ_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJ_ROOT / "refs" / "cups"))

import torch
from cups.config import get_default_config
from cups.augmentation import get_pseudo_label_augmentations
from cups.data.pseudo_label_dataset import PseudoLabelDataset
from cups.pl_model_pseudo import build_model_pseudo
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Smoke test Stage-2 M2F configs")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--steps", type=int, default=10, help="Number of train steps to run")
    ap.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke test")
    ap.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    ap.add_argument("--device", default="cuda:0", help="Torch device")
    ap.add_argument("--root", default="", help="Override DATA.ROOT (Cityscapes root)")
    ap.add_argument("--root-pseudo", default="", help="Override DATA.ROOT_PSEUDO")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    print(f"\n{'='*60}")
    print(f"Stage-2 M2F Smoke Test")
    print(f"Config: {cfg_path}")
    print(f"Steps:  {args.steps}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Config load
    # ------------------------------------------------------------------
    print("[1/5] Loading config...")
    try:
        config = get_default_config(experiment_config_file=str(cfg_path))
        config.defrost()
        # Override for smoke test
        config.TRAINING.BATCH_SIZE = args.batch_size
        config.SYSTEM.NUM_WORKERS = args.num_workers
        # Shorten crop resolution for speed (must be tuple)
        config.DATA.CROP_RESOLUTION = (512, 512)
        # Override data paths if provided
        if args.root:
            config.DATA.ROOT = args.root
            config.DATA.ROOT_VAL = args.root
        if args.root_pseudo:
            config.DATA.ROOT_PSEUDO = args.root_pseudo
        config.freeze()
        print(f"  META_ARCH:          {config.MODEL.META_ARCH}")
        print(f"  BACKBONE_TYPE:      {config.MODEL.BACKBONE_TYPE}")
        print(f"  QUERY_POOL:         {config.MODEL.MASK2FORMER.QUERY_POOL}")
        print(f"  NUM_QUERIES:        {config.MODEL.MASK2FORMER.NUM_QUERIES}")
        print(f"  DECOUPLED_HEADS:    {getattr(config.MODEL.MASK2FORMER, 'DECOUPLED_CLASS_HEADS', False)}")
        print(f"  BATCH_SIZE:         {config.TRAINING.BATCH_SIZE}")
        print("  ✓ Config loaded")
    except Exception as e:
        print(f"  ✗ Config load FAILED: {e}")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Dataloader
    # ------------------------------------------------------------------
    print("\n[2/5] Building dataloader...")
    try:
        training_dataset = PseudoLabelDataset(
            root=config.DATA.ROOT,
            root_pseudo=config.DATA.ROOT_PSEUDO,
            return_detectron2_format=True,
            ground_truth_scale=config.DATA.SCALE,
            crop_resolution=config.DATA.CROP_RESOLUTION,
            thing_stuff_threshold=config.DATA.THING_STUFF_THRESHOLD,
            ignore_unknown_thing_regions=config.DATA.IGNORE_UNKNOWN_THING_REGIONS,
            augmentations=get_pseudo_label_augmentations(
                config.DATA.CROP_RESOLUTION,
                scale=getattr(config.AUGMENTATION, "RANDOM_CROP_SCALE", (0.7, 1.0)),
            ),
            instance_aware_crop=getattr(config.AUGMENTATION, "INSTANCE_AWARE_CROP", False),
            instance_aware_crop_prob=getattr(config.AUGMENTATION, "INSTANCE_AWARE_CROP_PROB", 0.5),
            dataset=config.DATA.DATASET,
            only_use_non_empty_samples=True,
            num_pseudo_classes=getattr(config.DATA, "NUM_PSEUDO_CLASSES", config.DATA.NUM_CLASSES),
        )
        dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=config.TRAINING.BATCH_SIZE,
            shuffle=True,
            num_workers=config.SYSTEM.NUM_WORKERS,
            collate_fn=lambda x: x,
            drop_last=True,
            pin_memory=False,
            persistent_workers=False,
        )
        batch = next(iter(dataloader))
        print(f"  Dataset size:       {len(training_dataset)}")
        print(f"  Batch keys:         {list(batch[0].keys())}")
        print(f"  Image shape:        {batch[0]['image'].shape}")
        print(f"  Sem seg shape:      {batch[0]['sem_seg'].shape}")
        if "instances" in batch[0]:
            print(f"  Instances classes:  {batch[0]['instances'].gt_classes.tolist()}")
            print(f"  Instances masks:    {batch[0]['instances'].gt_masks.tensor.shape}")
        print("  ✓ Dataloader ready")
    except Exception as e:
        print(f"  ✗ Dataloader FAILED: {e}")
        traceback.print_exc()
        return 2

    # ------------------------------------------------------------------
    # 3. Model build
    # ------------------------------------------------------------------
    print("\n[3/5] Building model...")
    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # Infer thing/stuff split from first batch
        if "instances" in batch[0]:
            thing_ids = set(batch[0]["instances"].gt_classes.tolist())
        else:
            thing_ids = set()
        stuff_ids = set(int(c) for c in batch[0]["sem_seg"].unique().tolist()) - {0, 255, -1}
        stuff_ids -= thing_ids

        model = build_model_pseudo(
            config=config,
            thing_pseudo_classes=tuple(sorted(thing_ids)),
            stuff_pseudo_classes=tuple(sorted(stuff_ids)),
            thing_classes=thing_ids,
            stuff_classes=stuff_ids,
        )
        model = model.to(device)
        model.train()
        print(f"  Device:             {device}")
        print(f"  Stuff classes:      {len(stuff_ids)}")
        print(f"  Thing classes:      {len(thing_ids)}")
        print(f"  Total params:       {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print("  ✓ Model built")
    except Exception as e:
        print(f"  ✗ Model build FAILED: {e}")
        traceback.print_exc()
        return 2

    # ------------------------------------------------------------------
    # 4. Forward / backward loop
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running {args.steps} train steps...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dl_iter = iter(dataloader)
        for step in range(args.steps):
            try:
                batch_step = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch_step = next(dl_iter)

            # Move to device
            for sample in batch_step:
                sample["image"] = sample["image"].to(device)
                sample["sem_seg"] = sample["sem_seg"].to(device)
                if "instances" in sample:
                    sample["instances"] = sample["instances"].to(device)

            loss_dict = model(batch_step)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step == 0:
                print(f"  Step 0 loss:        {loss.item():.4f}")
                print(f"  Loss keys:          {list(loss_dict.keys())}")
            if (step + 1) % max(1, args.steps // 5) == 0:
                print(f"  Step {step + 1}/{args.steps} loss: {loss.item():.4f}")

        print("  ✓ Forward/backward OK")
    except Exception as e:
        print(f"  ✗ Train step FAILED at step {step}: {e}")
        traceback.print_exc()
        return 3

    # ------------------------------------------------------------------
    # 5. Gradient check
    # ------------------------------------------------------------------
    print("\n[5/5] Checking gradients...")
    try:
        has_grad = 0
        no_grad = 0
        grad_norms = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None and p.grad.abs().max().item() > 0:
                    has_grad += 1
                    grad_norms.append((name, p.grad.norm().item()))
                else:
                    no_grad += 1
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        print(f"  Params with grad:   {has_grad}")
        print(f"  Params w/o grad:    {no_grad}")
        print(f"  Top 5 grad norms:")
        for name, gnorm in grad_norms[:5]:
            print(f"    {gnorm:>10.4f}  {name}")

        # Sanity: at least 50% of trainable params should have non-zero grad
        if has_grad == 0:
            print("  ✗ NO parameters have gradients!")
            return 4
        if no_grad > has_grad * 2:
            print(f"  ⚠ Warning: {no_grad} params have zero grad vs {has_grad} with grad")
        else:
            print("  ✓ Gradients OK")
    except Exception as e:
        print(f"  ✗ Gradient check FAILED: {e}")
        traceback.print_exc()
        return 4

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED — safe to launch full training")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
