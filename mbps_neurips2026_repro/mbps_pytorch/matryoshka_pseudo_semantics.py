#!/usr/bin/env python3
"""Approach 3 Stage 2: Matryoshka Multi-Granularity Clustering for COCO-Stuff-27.

Multi-head projector with Sinkhorn-Knopp equipartition at multiple granularity
levels (27, 54, 150, 300). Cross-entropy between sharpened teacher and student
assignments with hierarchical consistency across heads.

Operates on pre-extracted DINOv3 features (no live backbone needed).

Usage:
    # Single-head baseline (K=27 only)
    python mbps_pytorch/matryoshka_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --heads 27 --epochs 30

    # Multi-head Matryoshka
    python mbps_pytorch/matryoshka_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --heads 27,54,150 --epochs 30

    # Eval-only on saved checkpoint
    python mbps_pytorch/matryoshka_pseudo_semantics.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --eval_only --checkpoint path/to/best.pth
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── COCO-Stuff-27 class definitions ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27
THING_IDS = set(range(12))
STUFF_IDS = set(range(12, 27))

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


def load_coco_panoptic_gt(coco_root: str, image_id: int) -> Optional[np.ndarray]:
    """Load COCO panoptic GT and convert to 27-class semantic label map."""
    panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)
        cat_map = {cat["id"]: cat["supercategory"] for cat in data["categories"]}
        ann_map = {ann["image_id"]: ann for ann in data["annotations"]}
        load_coco_panoptic_gt._cache = (cat_map, ann_map, str(panoptic_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache
    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_img = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (pan_img[:, :, 0].astype(np.int32) +
              pan_img[:, :, 1].astype(np.int32) * 256 +
              pan_img[:, :, 2].astype(np.int32) * 256 * 256)

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        mask = pan_id == seg["id"]
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[mask] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


# ─── Model ───

class MatryoshkaHead(nn.Module):
    """Single classification head for one granularity level."""

    def __init__(self, input_dim: int, n_clusters: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.linear = nn.Linear(input_dim, n_clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> logits: (B, N, K)."""
        return self.linear(x)


class MatryoshkaClusterer(nn.Module):
    """Multi-head projector with shared backbone + multiple granularity heads.

    Architecture:
        features (1024) -> shared_proj (256) -> Head_K1 (27), Head_K2 (54), ...
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        proj_dim: int = 256,
        head_sizes: List[int] = None,
    ):
        super().__init__()
        if head_sizes is None:
            head_sizes = [27]

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        self.heads = nn.ModuleDict({
            str(k): MatryoshkaHead(proj_dim, k)
            for k in head_sizes
        })
        self.head_sizes = sorted(head_sizes)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """x: (B, N, D) -> dict of {K: logits (B, N, K)}."""
        z = self.proj(x)
        return {k: self.heads[str(k)](z) for k in self.head_sizes}


# ─── Sinkhorn-Knopp ───

@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    temperature: float = 0.1,
    n_iters: int = 3,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Sinkhorn-Knopp normalization for equipartitioned soft assignments.

    Args:
        logits: (N, K) raw logits.
        temperature: Sharpening temperature.
        n_iters: Number of SK iterations.
        epsilon: Numerical stability.

    Returns:
        Soft assignment matrix Q (N, K) — equipartitioned across clusters.
    """
    Q = torch.exp(logits / temperature)
    Q = Q / (Q.sum() + epsilon)

    N, K = Q.shape

    for _ in range(n_iters):
        # Column normalization (each cluster gets equal mass)
        Q = Q / (Q.sum(dim=0, keepdim=True) + epsilon)
        Q = Q / K
        # Row normalization (each sample assigns to 1.0 total)
        Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)
        Q = Q / N

    # Final row normalization to get probabilities
    Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)
    return Q


# ─── Loss Functions ───

def swav_cross_entropy_loss(
    student_logits: torch.Tensor,
    teacher_assignments: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy between student predictions and teacher SK assignments.

    Args:
        student_logits: (N, K) raw logits from student.
        teacher_assignments: (N, K) soft targets from SK on teacher.
        temperature: Student temperature for softmax.

    Returns:
        Scalar loss.
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = -(teacher_assignments * student_probs).sum(dim=-1).mean()
    return loss


def matryoshka_consistency_loss(
    coarse_logits: torch.Tensor,
    fine_assignments: torch.Tensor,
    mapping: torch.Tensor,
) -> torch.Tensor:
    """Hierarchical consistency: coarse head should agree with merged fine head.

    Args:
        coarse_logits: (N, K_coarse) logits from coarser head.
        fine_assignments: (N, K_fine) SK assignments from finer head.
        mapping: (K_fine, K_coarse) binary mapping from fine to coarse clusters.

    Returns:
        Scalar loss.
    """
    # Merge fine assignments to coarse level
    merged_fine = fine_assignments @ mapping  # (N, K_coarse)
    merged_fine = merged_fine / (merged_fine.sum(dim=-1, keepdim=True) + 1e-8)

    coarse_probs = F.log_softmax(coarse_logits / 0.1, dim=-1)
    loss = -(merged_fine * coarse_probs).sum(dim=-1).mean()
    return loss


def entropy_regularization(logits: torch.Tensor) -> torch.Tensor:
    """Maximize entropy of mean assignment to prevent cluster collapse.

    Args:
        logits: (N, K) raw logits.

    Returns:
        Negative entropy (minimize to maximize entropy).
    """
    probs = F.softmax(logits, dim=-1)
    avg = probs.mean(dim=0)
    avg = avg.clamp(min=1e-8)
    return -(avg * torch.log(avg)).sum()


# ─── Dataset ───

class COCOFeatureDataset(torch.utils.data.Dataset):
    """Load pre-extracted DINOv3 features with spatial crop augmentation.

    Returns two random spatial crops of the feature grid as views for
    self-distillation. The overlap region provides the learning signal.
    """

    def __init__(
        self,
        feat_dir: str,
        patch_grid: int = 32,
        crop_size: int = 16,
        augment: bool = True,
    ):
        self.feat_files = sorted(Path(feat_dir).glob("*.npy"))
        self.patch_grid = patch_grid
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.feat_files)

    def _random_crop(
        self, feat_grid: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Random crop of feat_grid (H, W, C). Returns (crop, (y0, x0))."""
        h, w = self.patch_grid, self.patch_grid
        cs = self.crop_size
        y0 = np.random.randint(0, h - cs + 1)
        x0 = np.random.randint(0, w - cs + 1)
        return feat_grid[y0:y0+cs, x0:x0+cs], (y0, x0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        feat = np.load(self.feat_files[idx])  # (N_patches, feat_dim)
        pg = self.patch_grid
        feat_grid = feat.reshape(pg, pg, -1)

        if self.augment:
            crop1, (y1, x1) = self._random_crop(feat_grid)
            crop2, (y2, x2) = self._random_crop(feat_grid)
            # Add small Gaussian noise to create distinct views
            noise_scale = 0.02
            crop1 = crop1 + np.random.randn(*crop1.shape).astype(np.float32) * noise_scale
            crop2 = crop2 + np.random.randn(*crop2.shape).astype(np.float32) * noise_scale
            t1 = torch.from_numpy(crop1.reshape(-1, crop1.shape[-1])).float()
            t2 = torch.from_numpy(crop2.reshape(-1, crop2.shape[-1])).float()
        else:
            t1 = torch.from_numpy(feat).float()
            t2 = t1

        image_id = int(self.feat_files[idx].stem)
        return t1, t2, image_id


# ─── Training ───

def train_matryoshka(args: argparse.Namespace) -> None:
    """Train Matryoshka multi-head clusterer with EMA teacher + spatial augmentation."""
    root = Path(args.coco_root)
    feat_dir = root / args.features_subdir
    head_sizes = [int(h) for h in args.heads.split(",")]

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Config string for output
    heads_str = "_".join(str(h) for h in head_sizes)
    config_name = f"matryoshka_h{heads_str}_e{args.epochs}"
    if args.output_subdir is None:
        args.output_subdir = f"pseudo_semantic_{config_name}"
    out_dir = root / args.output_subdir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / args.output_subdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MATRYOSHKA CLUSTERING")
    print(f"{'='*60}")
    print(f"  Heads: {head_sizes}")
    print(f"  Proj dim: {args.proj_dim}")
    print(f"  Crop size: {args.crop_size}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  SK temperature: {args.sk_temp}, SK iters: {args.sk_iters}")
    print(f"  EMA momentum: {args.ema_momentum}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")

    # Dataset with spatial augmentation
    dataset = COCOFeatureDataset(
        str(feat_dir), patch_grid=args.patch_grid,
        crop_size=args.crop_size, augment=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    # Full-resolution dataset for eval
    eval_dataset = COCOFeatureDataset(
        str(feat_dir), patch_grid=args.patch_grid, augment=False,
    )
    print(f"  Images: {len(dataset)}")

    # Student + Teacher (EMA)
    student = MatryoshkaClusterer(
        feat_dim=1024, proj_dim=args.proj_dim, head_sizes=head_sizes,
    ).to(device)
    teacher = MatryoshkaClusterer(
        feat_dim=1024, proj_dim=args.proj_dim, head_sizes=head_sizes,
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_miou = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for view1, view2, _ in dataloader:
            view1 = view1.to(device)  # (B, N_crop, feat_dim)
            view2 = view2.to(device)

            # Student on view1, Teacher on view2 (cross-prediction)
            student_out = student(view1)
            with torch.no_grad():
                teacher_out = teacher(view2)

            loss = torch.tensor(0.0, device=device)

            for k in head_sizes:
                s_logits = student_out[k]  # (B, N_crop, K)
                t_logits = teacher_out[k]

                B, N, K = s_logits.shape
                s_flat = s_logits.reshape(B * N, K)
                t_flat = t_logits.reshape(B * N, K)

                # Teacher SK assignments on view2
                t_assign = sinkhorn_knopp(
                    t_flat, temperature=args.sk_temp, n_iters=args.sk_iters,
                )

                # Student predicts teacher's assignments
                ce_loss = swav_cross_entropy_loss(s_flat, t_assign, temperature=0.1)
                loss = loss + ce_loss

                # Entropy regularization
                ent_loss = entropy_regularization(s_flat)
                loss = loss - args.lambda_entropy * ent_loss

            # Also do reverse: student on view2, teacher on view1
            student_out2 = student(view2)
            with torch.no_grad():
                teacher_out1 = teacher(view1)

            for k in head_sizes:
                s_logits2 = student_out2[k]
                t_logits1 = teacher_out1[k]

                B, N, K = s_logits2.shape
                s_flat2 = s_logits2.reshape(B * N, K)
                t_flat1 = t_logits1.reshape(B * N, K)

                t_assign1 = sinkhorn_knopp(
                    t_flat1, temperature=args.sk_temp, n_iters=args.sk_iters,
                )
                ce_loss2 = swav_cross_entropy_loss(s_flat2, t_assign1, temperature=0.1)
                loss = loss + ce_loss2

            # Matryoshka consistency (on view1 only)
            if len(head_sizes) > 1 and args.lambda_consistency > 0:
                for i in range(len(head_sizes) - 1):
                    k_coarse = head_sizes[i]
                    k_fine = head_sizes[i + 1]

                    with torch.no_grad():
                        t_coarse = teacher_out[k_coarse].reshape(-1, k_coarse)
                        t_fine = teacher_out[k_fine].reshape(-1, k_fine)
                        t_coarse_a = F.softmax(t_coarse / args.sk_temp, dim=-1)
                        t_fine_a = F.softmax(t_fine / args.sk_temp, dim=-1)
                        overlap = t_fine_a.T @ t_coarse_a
                        mapping = torch.zeros(k_fine, k_coarse, device=device)
                        best_coarse = overlap.argmax(dim=1)
                        mapping[torch.arange(k_fine, device=device), best_coarse] = 1.0

                    s_coarse = student_out[k_coarse].reshape(-1, k_coarse)
                    s_fine_assign = sinkhorn_knopp(
                        student_out[k_fine].reshape(-1, k_fine),
                        temperature=args.sk_temp, n_iters=args.sk_iters,
                    )
                    cons_loss = matryoshka_consistency_loss(
                        s_coarse, s_fine_assign, mapping,
                    )
                    loss = loss + args.lambda_consistency * cons_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                m = args.ema_momentum
                for sp, tp in zip(student.parameters(), teacher.parameters()):
                    tp.data.mul_(m).add_(sp.data, alpha=1 - m)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate every eval_every epochs
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            miou = evaluate_matryoshka(
                teacher, root, feat_dir, out_dir, head_sizes[0],
                device, args.patch_grid, args.eval_images,
            )
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
                  f"mIoU={miou:.1f}%, lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"time={elapsed:.0f}s")

            if miou > best_miou:
                best_miou = miou
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": teacher.state_dict(),
                    "head_sizes": head_sizes,
                    "proj_dim": args.proj_dim,
                    "miou": miou,
                }, str(ckpt_dir / "best.pth"))
        else:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}, time={elapsed:.0f}s")

    print(f"\nBest mIoU: {best_miou:.1f}%")
    print(f"Checkpoint: {ckpt_dir / 'best.pth'}")


def evaluate_matryoshka(
    model: nn.Module,
    coco_root: Path,
    feat_dir: Path,
    out_dir: Path,
    k_eval: int,
    device: str,
    patch_grid: int = 32,
    n_eval: Optional[int] = None,
) -> float:
    """Evaluate teacher model using Hungarian matching on the K=k_eval head.

    Returns mIoU percentage.
    """
    model.eval()
    feat_files = sorted(feat_dir.glob("*.npy"))

    # Step 1: Predict cluster assignments for all images
    all_preds = []
    image_ids = []

    for fp in feat_files:
        feat = torch.from_numpy(np.load(fp)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feat)
        logits = out[k_eval][0]  # (N, K)
        pred = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(pred)
        image_ids.append(int(fp.stem))

    # Step 2: Hungarian matching
    cost_matrix = np.zeros((k_eval, NUM_CLASSES), dtype=np.float64)

    for idx, img_id in enumerate(image_ids):
        gt_sem = load_coco_panoptic_gt(str(coco_root), img_id)
        if gt_sem is None:
            continue
        gt_resized = np.array(Image.fromarray(gt_sem).resize(
            (patch_grid, patch_grid), Image.NEAREST
        ))
        gt_flat = gt_resized.flatten()
        pred_flat = all_preds[idx]

        for p, g in zip(pred_flat, gt_flat):
            if g < NUM_CLASSES:
                cost_matrix[int(p), int(g)] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    for k_id in range(k_eval):
        if k_id not in cluster_to_class:
            cluster_to_class[k_id] = int(np.argmax(cost_matrix[k_id]))

    # Step 3: Compute mIoU
    n_imgs = n_eval or len(image_ids)
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for idx, img_id in enumerate(image_ids[:n_imgs]):
        gt_sem = load_coco_panoptic_gt(str(coco_root), img_id)
        if gt_sem is None:
            continue

        pred_flat = all_preds[idx]
        pred_mapped = np.vectorize(cluster_to_class.get)(
            pred_flat.reshape(patch_grid, patch_grid)
        ).astype(np.uint8)
        pred_full = np.array(Image.fromarray(pred_mapped).resize(
            (gt_sem.shape[1], gt_sem.shape[0]), Image.NEAREST
        ))

        for c in range(NUM_CLASSES):
            gt_mask = gt_sem == c
            pred_mask = pred_full == c
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid = count_per_class > 0
    miou = (iou_per_class[valid] / count_per_class[valid]).mean() * 100

    # Save labels from best head
    for idx, img_id in enumerate(image_ids):
        img_path = coco_root / "val2017" / f"{img_id:012d}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        w, h = img.size

        pred_flat = all_preds[idx]
        pred_mapped = np.vectorize(cluster_to_class.get)(
            pred_flat.reshape(patch_grid, patch_grid)
        ).astype(np.uint8)
        pred_full = np.array(Image.fromarray(pred_mapped).resize((w, h), Image.NEAREST))

        out_path = out_dir / f"{img_id:012d}.png"
        Image.fromarray(pred_full).save(out_path)

    return miou


def main() -> None:
    parser = argparse.ArgumentParser(description="Matryoshka Pseudo-Semantics")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--features_subdir", default="dinov3_features/val2017")
    parser.add_argument("--heads", default="27",
                        help="Comma-separated head sizes, e.g., '27,54,150'")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sk_temp", type=float, default=0.1,
                        help="Sinkhorn-Knopp temperature")
    parser.add_argument("--sk_iters", type=int, default=3,
                        help="Sinkhorn-Knopp iterations")
    parser.add_argument("--ema_momentum", type=float, default=0.996)
    parser.add_argument("--lambda_entropy", type=float, default=0.5,
                        help="Entropy regularization weight")
    parser.add_argument("--lambda_consistency", type=float, default=1.0,
                        help="Matryoshka consistency loss weight")
    parser.add_argument("--patch_grid", type=int, default=32)
    parser.add_argument("--crop_size", type=int, default=16,
                        help="Spatial crop size for augmentation (default: 16)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_subdir", default=None)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_images", type=int, default=None,
                        help="Number of images for mIoU eval (None=all)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.eval_only:
        if args.checkpoint is None:
            parser.error("--checkpoint required with --eval_only")
        device = args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        ckpt = torch.load(args.checkpoint, map_location=device)
        head_sizes = ckpt["head_sizes"]
        model = MatryoshkaClusterer(
            feat_dim=1024, proj_dim=ckpt["proj_dim"], head_sizes=head_sizes,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        root = Path(args.coco_root)
        feat_dir = root / args.features_subdir
        if args.output_subdir is None:
            args.output_subdir = "pseudo_semantic_matryoshka_eval"
        out_dir = root / args.output_subdir / "val2017"
        out_dir.mkdir(parents=True, exist_ok=True)

        miou = evaluate_matryoshka(
            model, root, feat_dir, out_dir, head_sizes[0],
            device, args.patch_grid, args.eval_images,
        )
        print(f"mIoU = {miou:.1f}%")
    else:
        train_matryoshka(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
