#!/usr/bin/env python3
"""Train a compact DINOv3-S Mask R-CNN student from unMORE teacher caches.

This is the controlled pure-distillation track before adding EMA/noisy-student
self-training. It uses teacher boxes and RLE masks from external-drive caches
produced by ``scripts/build_unmore_teacher_cache.py`` or
``scripts/build_unmore_imagenet_votecut_cache.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbps_pytorch.unmore_distill import (  # noqa: E402
    UnmoreTeacherCacheDataset,
    build_unmore_dinov3s_maskrcnn,
)
from mbps_pytorch.unmore_distill.model import count_parameters, count_trainable_parameters  # noqa: E402
from mbps_pytorch.unmore_distill.teacher_cache import collate_unmore_teacher_batch  # noqa: E402


LOSS_WEIGHT_ARG = {
    "loss_classifier": "loss_classifier_weight",
    "loss_box_reg": "loss_box_weight",
    "loss_mask": "loss_mask_weight",
    "loss_objectness": "loss_rpn_objectness_weight",
    "loss_rpn_box_reg": "loss_rpn_box_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/Volumes/code_files/mbps_datasets/unmore_teacher_cache/coco20k_official_unmore"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/unmore_dinov3s_student_pilot"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--score-min", type=float, default=0.10)
    parser.add_argument("--max-instances", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss-classifier-weight", type=float, default=1.0)
    parser.add_argument("--loss-box-weight", type=float, default=1.0)
    parser.add_argument("--loss-mask-weight", type=float, default=1.0)
    parser.add_argument("--loss-rpn-objectness-weight", type=float, default=1.0)
    parser.add_argument("--loss-rpn-box-weight", type=float, default=1.0)
    parser.add_argument("--teacher-score-weight-power", type=float, default=0.0)
    parser.add_argument("--teacher-score-weight-min", type=float, default=0.25)
    parser.add_argument("--teacher-score-weight-max", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-backbone-at-step", type=int, default=None)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-backbone-lr-mult", type=float, default=0.2)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=896)
    parser.add_argument("--fpn-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--val-every-steps", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one train batch and one val batch.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            try:
                torch.empty(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")
    if name == "mps":
        try:
            torch.empty(1, device="mps")
            return torch.device("mps")
        except Exception as exc:
            raise RuntimeError("MPS was requested but is not usable in this environment") from exc
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets(targets, device: torch.device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def loss_to_float(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def weighted_detection_loss(losses: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    total = None
    for name, value in losses.items():
        arg_name = LOSS_WEIGHT_ARG.get(name)
        weight = float(getattr(args, arg_name, 1.0)) if arg_name else 1.0
        term = value * weight
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Model returned no losses")
    return total


def batch_teacher_score_weight(targets, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.teacher_score_weight_power <= 0:
        return torch.ones((), device=device)
    weights = []
    for target in targets:
        scores = target.get("teacher_scores")
        if scores is None or scores.numel() == 0:
            weights.append(torch.tensor(args.teacher_score_weight_min, device=device))
            continue
        score = scores.float().mean().clamp(1e-6, 1.0)
        weights.append(score.pow(args.teacher_score_weight_power))
    stacked = torch.stack(weights).mean()
    return stacked.clamp(args.teacher_score_weight_min, args.teacher_score_weight_max)


def unfreeze_backbone_tail(model: torch.nn.Module, last_blocks: int) -> int:
    """Unfreeze the last N DINO/EVA blocks, falling back to the full body."""

    backbone = model.backbone
    body = getattr(backbone, "body", None)
    if body is None:
        return 0

    for param in body.parameters():
        param.requires_grad = False

    core = getattr(body, "model", body)
    blocks = getattr(core, "blocks", None)
    unfrozen_params = []
    if blocks is not None and last_blocks > 0:
        for block in list(blocks)[-last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_params.append(param)
        for attr in ("norm", "fc_norm"):
            module = getattr(core, attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_params.append(param)
    else:
        for param in body.parameters():
            param.requires_grad = True
            unfrozen_params.append(param)

    # Deduplicate because norm params may be visited more than once on some timm models.
    seen = set()
    unique = []
    for param in unfrozen_params:
        ident = id(param)
        if ident not in seen:
            seen.add(ident)
            unique.append(param)
    return sum(param.numel() for param in unique if param.requires_grad)


def newly_trainable_params(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    existing = {id(param) for group in optimizer.param_groups for param in group["params"]}
    return [param for param in model.parameters() if param.requires_grad and id(param) not in existing]


def append_history(path: Path, row: Dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    run_config: Dict,
) -> Path:
    ckpt_path = output_dir / name
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": run_config,
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}", flush=True)
    return ckpt_path


def run_validation(
    model,
    val_loader,
    device: torch.device,
    max_batches: int,
    global_step: int,
    history_path: Path,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses_total = []
    with torch.no_grad():
        for val_idx, (images, targets, _metas) in enumerate(tqdm(val_loader, desc=f"step {global_step} val")):
            images = [img.to(device) for img in images]
            targets = move_targets(targets, device)
            losses = model(images, targets)
            loss_dict = loss_to_float(losses)
            total_tensor = weighted_detection_loss(losses, args) * batch_teacher_score_weight(targets, args, device)
            total = float(total_tensor.detach().cpu())
            loss_dict["total"] = total
            loss_dict["total_unweighted"] = float(sum(losses.values()).detach().cpu())
            losses_total.append(total)
            append_history(
                history_path,
                {"step": global_step, "split": "val", "val_batch": val_idx + 1, **loss_dict},
            )
            if val_idx + 1 >= max_batches:
                break
    model.train()
    mean_total = float(np.mean(losses_total)) if losses_total else 0.0
    print(f"Validation at step {global_step}: total={mean_total:.4f}", flush=True)
    return mean_total


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    dataset = UnmoreTeacherCacheDataset(
        cache_dir=args.cache_dir,
        max_images=args.max_images,
        score_min=args.score_min,
        max_instances=args.max_instances,
        skip_empty=args.skip_empty,
    )
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_unmore_teacher_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_unmore_teacher_batch,
    )

    model = build_unmore_dinov3s_maskrcnn(
        pretrained_backbone=args.pretrained_backbone,
        train_backbone=not args.freeze_backbone,
        fpn_dim=args.fpn_dim,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    approx_fp32_mb = total_params * 4 / (1024**2)
    print(
        f"Model params: total={total_params:,}, trainable={trainable_params:,}, "
        f"fp32_size~{approx_fp32_mb:.1f} MB",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = vars(args).copy()
    run_config.update(
        {
            "device": str(device),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "approx_fp32_mb": approx_fp32_mb,
            "dataset_manifest": dataset.manifest,
        }
    )
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)
        f.write("\n")

    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"Resumed {args.resume} at epoch={start_epoch} global_step={global_step}", flush=True)

    history_path = args.output_dir / "history.jsonl"
    if args.resume is None and history_path.exists():
        history_path.unlink()

    stop_training = False
    backbone_unfrozen = False
    if args.unfreeze_backbone_at_step is not None and global_step >= args.unfreeze_backbone_at_step:
        num_unfrozen = unfreeze_backbone_tail(model, args.unfreeze_last_blocks)
        new_params = newly_trainable_params(model, optimizer)
        if new_params:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "lr": args.lr * args.unfreeze_backbone_lr_mult,
                    "weight_decay": args.weight_decay,
                }
            )
        backbone_unfrozen = True
        print(
            f"Backbone tail already unfrozen at resume: params={num_unfrozen:,} new_optimizer_params={len(new_params)}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        if epoch < start_epoch:
            continue
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        epoch_start = time.time()
        for images, targets, _metas in pbar:
            if (
                not backbone_unfrozen
                and args.unfreeze_backbone_at_step is not None
                and global_step >= args.unfreeze_backbone_at_step
            ):
                num_unfrozen = unfreeze_backbone_tail(model, args.unfreeze_last_blocks)
                new_params = newly_trainable_params(model, optimizer)
                if new_params:
                    optimizer.add_param_group(
                        {
                            "params": new_params,
                            "lr": args.lr * args.unfreeze_backbone_lr_mult,
                            "weight_decay": args.weight_decay,
                        }
                    )
                backbone_unfrozen = True
                print(
                    f"Unfroze DINOv3 backbone tail at step={global_step}: "
                    f"params={num_unfrozen:,} new_optimizer_params={len(new_params)}",
                    flush=True,
                )

            images = [img.to(device) for img in images]
            targets = move_targets(targets, device)
            losses = model(images, targets)
            score_weight = batch_teacher_score_weight(targets, args, device)
            weighted_total = weighted_detection_loss(losses, args) * score_weight
            loss = weighted_total / max(args.grad_accum_steps, 1)

            loss.backward()
            accum_count += 1

            loss_dict = loss_to_float(losses)
            loss_dict["total"] = float(weighted_total.detach().cpu())
            loss_dict["total_unweighted"] = float(sum(losses.values()).detach().cpu())
            loss_dict["teacher_score_weight"] = float(score_weight.detach().cpu())
            if accum_count >= max(args.grad_accum_steps, 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1

                elapsed = max(time.time() - epoch_start, 1e-6)
                examples_seen = global_step * args.batch_size * max(args.grad_accum_steps, 1)
                examples_per_sec = examples_seen / elapsed
                pbar.set_postfix(
                    {
                        k: f"{v:.3f}"
                        for k, v in loss_dict.items()
                        if k in {"total", "loss_mask", "loss_box_reg"}
                    }
                )
                if global_step == 1 or global_step % args.log_every == 0:
                    append_history(
                        history_path,
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "split": "train",
                            "examples_per_sec_epoch": examples_per_sec,
                            **loss_dict,
                        },
                    )
                    print(
                        f"step={global_step} total={loss_dict['total']:.4f} "
                        f"mask={loss_dict.get('loss_mask', 0.0):.4f} "
                        f"box={loss_dict.get('loss_box_reg', 0.0):.4f} "
                        f"ex/s={examples_per_sec:.3f}",
                        flush=True,
                    )

                if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                    run_validation(model, val_loader, device, args.val_batches, global_step, history_path, args)

                if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                    save_checkpoint(
                        args.output_dir,
                        f"student_step_{global_step:07d}.pth",
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        run_config,
                    )

                if args.smoke or (args.max_steps is not None and global_step >= args.max_steps):
                    stop_training = True
                    break

        if not stop_training:
            run_validation(model, val_loader, device, args.val_batches, global_step, history_path, args)

        if epoch % args.save_every == 0:
            save_checkpoint(
                args.output_dir,
                f"student_epoch_{epoch:03d}.pth",
                model,
                optimizer,
                epoch,
                global_step,
                run_config,
            )

        if stop_training:
            break

    save_checkpoint(
        args.output_dir,
        "student_latest.pth",
        model,
        optimizer,
        epoch,
        global_step,
        run_config,
    )
    print(f"Done. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
