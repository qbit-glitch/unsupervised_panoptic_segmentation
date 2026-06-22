"""Distillation trainer: frozen EoMT-ViT-L teacher → EoMT-ViT-S student.

Student: refs/eomt EoMT + timm vit_small_patch16_224 (Q=100, 133 cls, export-clean)
Teacher: frozen bf16 EoMT-ViT-L via TeacherWrapper (Q=200, 134 cls)
Data:    COCO-train-118k GT panoptic labels (CocoPanopticGTDS)
Loss:    λ_gt * GT_hard_loss + λ_kd * soft_KD_loss
  KD:   sorted-confidence query matching (top-Qmin from each),
        KL on class logits (τ=4) + sigmoid-MSE on masks

Usage:
    # Smoke (2 opt-steps, limit=8 images)
    python train_distill.py --limit 8 --steps 2 --device cuda
    # Full training (effective batch 16 via grad-accum)
    python train_distill.py --epochs 40 --bs 4 --accum 4 --device cuda
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs/eomt"))
from models.eomt import EoMT                                              # noqa: E402
from models.vit import ViT                                                # noqa: E402
from training.mask_classification_loss import MaskClassificationLoss      # noqa: E402

sys.path.insert(0, str(ROOT))
from mbps_pytorch.mobile_panoptic_sup.coco_distill_data import (          # noqa: E402
    CocoPanopticGTDS, collate as gt_collate,
)
from mbps_pytorch.mobile_panoptic_sup.teacher_wrapper import TeacherWrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_distill")

# ── loss weights ──────────────────────────────────────────────────────────────
_HARD_W = {"loss_mask": 5.0, "loss_dice": 5.0, "loss_cross_entropy": 2.0}
_KD_TAU = 4.0


def _build_student_targets(proc_batch: dict, size: int, dev: torch.device) -> list[dict]:
    """Convert HF-proc GT batch → refs/eomt target format [{masks, labels, is_crowd}]."""
    targets = []
    for ml, cl in zip(proc_batch["mask_labels"], proc_batch["class_labels"]):
        # ml: [Q, H, W] float32 {0,1}; cl: [Q] int64 0..132
        masks = (ml > 0.5).bool().to(dev)
        if size != ml.shape[-1]:
            # resize to model's native resolution
            masks = F.interpolate(
                masks.unsqueeze(0).float(), size=(size, size), mode="nearest"
            ).squeeze(0).bool()
        labels = cl.to(dev)
        targets.append({
            "masks": masks,
            "labels": labels,
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool, device=dev),
        })
    return targets


def _kd_loss(
    t_cls: torch.Tensor,   # [B, Qt, Ct]  teacher class logits (Ct=134)
    s_cls: torch.Tensor,   # [B, Qs, Cs]  student class logits (Cs=133)
    t_mask: torch.Tensor,  # [B, Qt, Ht, Wt]
    s_mask: torch.Tensor,  # [B, Qs, Hs, Ws]
    tau: float = _KD_TAU,
    lam_mask: float = 0.5,
) -> torch.Tensor:
    """Sorted-confidence query matching KD loss."""
    B, Qt = t_cls.shape[:2]
    _, Qs = s_cls.shape[:2]
    Qmin = min(Qt, Qs)
    C = min(t_cls.shape[-1], s_cls.shape[-1])  # match to smaller vocab (133)

    # confidence = max class prob over thing/stuff classes (exclude no-obj slot if present)
    t_conf = t_cls[:, :, :C].softmax(-1).max(-1).values   # [B, Qt]
    s_conf = s_cls[:, :, :C].softmax(-1).max(-1).values   # [B, Qs]

    t_idx = t_conf.argsort(dim=1, descending=True)[:, :Qmin]   # [B, Qmin]
    s_idx = s_conf.argsort(dim=1, descending=True)[:, :Qmin]

    # gather matched queries
    t_cls_m = t_cls[:, :, :C].gather(
        1, t_idx.unsqueeze(-1).expand(-1, -1, C))   # [B, Qmin, C]
    s_cls_m = s_cls[:, :, :C].gather(
        1, s_idx.unsqueeze(-1).expand(-1, -1, C))

    loss_cls = F.kl_div(
        F.log_softmax(s_cls_m / tau, dim=-1),
        F.softmax(t_cls_m.detach() / tau, dim=-1),
        reduction="batchmean",
    ) * (tau ** 2)

    # mask KD — resize student masks to teacher resolution
    Ht, Wt = t_mask.shape[-2:]
    s_mask_r = F.interpolate(
        s_mask.flatten(0, 1).unsqueeze(1).float(),
        size=(Ht, Wt), mode="bilinear", align_corners=False,
    ).squeeze(1).view(B, Qs, Ht, Wt)

    t_mask_m = t_mask.gather(
        1, t_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Ht, Wt))
    s_mask_m = s_mask_r.gather(
        1, s_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Ht, Wt))

    loss_mask = F.mse_loss(
        torch.sigmoid(s_mask_m),
        torch.sigmoid(t_mask_m.detach()),
    )

    return loss_cls + lam_mask * loss_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0, help="cap opt steps (smoke)")
    ap.add_argument("--bs", type=int, default=4, help="micro-batch size")
    ap.add_argument("--accum", type=int, default=4, help="grad-accum steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam_gt", type=float, default=1.0, help="hard GT loss weight")
    ap.add_argument("--lam_kd", type=float, default=0.5, help="soft KD loss weight")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "checkpoints/eomt_distill_vits")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()

    dev = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    eff_batch = a.bs * a.accum

    # ── dataset ───────────────────────────────────────────────────────────────
    from transformers import AutoImageProcessor
    REPO = "tue-mps/eomt-dinov3-coco-panoptic-large-640"
    proc = AutoImageProcessor.from_pretrained(REPO)
    proc.ignore_index = 255

    ds = CocoPanopticGTDS(proc, limit=a.limit)
    loader = DataLoader(
        ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
        collate_fn=gt_collate, drop_last=True,
        pin_memory=True, persistent_workers=a.workers > 0,
    )
    log.info("dataset=%d eff_batch=%d steps/epoch=%d",
             len(ds), eff_batch, len(loader) // a.accum)

    # ── student ───────────────────────────────────────────────────────────────
    student = EoMT(
        encoder=ViT(img_size=(a.img, a.img), backbone_name=a.backbone),
        num_classes=133, num_q=100, num_blocks=4,
        masked_attn_enabled=True,
    ).to(dev).train()
    crit = MaskClassificationLoss(
        num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=133, no_object_coefficient=0.1,
    ).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.05)

    start_epoch = 0
    if a.resume and a.resume.exists():
        sd = torch.load(a.resume, map_location=dev)
        student.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start_epoch = sd.get("epoch", 0) + 1
        log.info("resumed epoch %d from %s", start_epoch, a.resume)

    # ── teacher ───────────────────────────────────────────────────────────────
    teacher = TeacherWrapper(device=a.device)

    total_steps = a.steps or a.epochs * max(1, len(loader) // a.accum)
    opt_step, micro = 0, 0
    opt.zero_grad()

    for epoch in range(start_epoch, a.epochs):
        for batch in loader:
            pv = batch["pixel_values"].to(dev)  # [B, 3, 640, 640]

            # hard GT targets for student (refs/eomt format)
            gt_targets = _build_student_targets(batch, a.img, dev)

            # masked-attn annealing
            student.attn_mask_probs.fill_(max(0.0, 1.0 - opt_step / total_steps))

            # student forward → list of (mask_logits [B, Q, H, W], cls_logits [B, Q, C])
            mask_l, cls_l = student(pv)

            # hard GT loss (all decoder stages)
            loss_gt = sum(
                sum(_HARD_W.get(k, 1.0) * v for k, v in crit(m, gt_targets, c).items())
                for m, c in zip(mask_l, cls_l)
            )

            # soft KD loss (final decoder stage only)
            teacher_out = teacher(pv)
            loss_kd = _kd_loss(
                teacher_out.class_logits.to(dev),
                cls_l[-1],
                teacher_out.mask_logits.to(dev),
                mask_l[-1],
            )

            loss = (a.lam_gt * loss_gt + a.lam_kd * loss_kd) / a.accum
            loss.backward()
            micro += 1

            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                opt_step += 1

                if opt_step % 10 == 0:
                    log.info("epoch %d step %d/%d loss_gt=%.3f loss_kd=%.3f",
                             epoch, opt_step, total_steps,
                             float(loss_gt), float(loss_kd))
                if a.steps and opt_step >= a.steps:
                    torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                                "epoch": epoch},
                               a.out / "smoke.pt")
                    log.info("SMOKE_DONE step=%d", opt_step)
                    return

        ckpt = a.out / f"epoch_{epoch:03d}.pt"
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                   ckpt)
        log.info("saved %s", ckpt)


if __name__ == "__main__":
    main()
