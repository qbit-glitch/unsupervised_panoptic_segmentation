#!/usr/bin/env python3
"""Train CUPS-style ClusterLookup on pre-extracted DINOv3 features for COCO-Stuff-27.

Path 3 of the COCO semantic ablation: lightweight projector + learnable cluster
centers trained with STEGO-style feature correlation loss. Operates entirely on
pre-extracted features — no live backbone inference needed.

Usage:
    # Default config
    python mbps_pytorch/train_coco_cluster_lookup.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --device mps

    # Sweep configs
    python mbps_pytorch/train_coco_cluster_lookup.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --k 150 --proj_dim 128 --lr 5e-4 --epochs 30 --shift 0.05

    # Use train+val features
    python mbps_pytorch/train_coco_cluster_lookup.py \
        --coco_root /Users/qbit-glitch/Desktop/datasets/coco \
        --use_train --k 80
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# ─── COCO-Stuff-27 Constants ───

COCOSTUFF27_CLASSNAMES = [
    "electronic", "appliance", "food", "furniture", "indoor",
    "kitchen", "accessory", "animal", "outdoor", "person",
    "sports", "vehicle",
    "ceiling", "floor", "food-stuff", "furniture-stuff", "raw-material",
    "textile", "wall", "window", "building", "ground",
    "plant", "sky", "solid", "structural", "water",
]
NUM_CLASSES = 27

SUPERCATEGORY_TO_COARSE = {
    "electronic": 0, "appliance": 1, "food": 2, "furniture": 3,
    "indoor": 4, "kitchen": 5, "accessory": 6, "animal": 7,
    "outdoor": 8, "person": 9, "sports": 10, "vehicle": 11,
    "ceiling": 12, "floor": 13, "food-stuff": 14, "furniture-stuff": 15,
    "raw-material": 16, "textile": 17, "wall": 18, "window": 19,
    "building": 20, "ground": 21, "plant": 22, "sky": 23,
    "solid": 24, "structural": 25, "water": 26,
}


# ─── Model Components ───

class Projector(nn.Module):
    """MLP projector: feat_dim -> proj_dim."""

    def __init__(self, feat_dim=1024, proj_dim=90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ClusterLookup(nn.Module):
    """Learnable cluster centers with cosine similarity assignment."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(n_classes, dim))

    def forward(self, x, alpha=2.0):
        """x: (B, C, H, W). Returns cluster_loss, cluster_probs."""
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        cluster_probs = F.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        return cluster_loss, cluster_probs

    def predict(self, x):
        """x: (B, C, H, W). Returns (B, H, W) argmax assignments."""
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        return inner_products.argmax(dim=1)


# ─── STEGO Correlation Loss (Full) ───

def _correlation_helper(feat1, feat2, code1, code2, shift, num_samples=100):
    """STEGO loss helper: -cd.clamp(0) * (fd - shift).

    When fd > shift → positive pair → encourage code similarity
    When fd < shift → negative pair → penalize code similarity
    """
    B, C_f, H, W = feat1.shape
    N = H * W

    feat1_flat = feat1.reshape(B, C_f, N).permute(0, 2, 1)
    code1_flat = code1.reshape(B, -1, N).permute(0, 2, 1)
    feat2_flat = feat2.reshape(B, C_f, N).permute(0, 2, 1)
    code2_flat = code2.reshape(B, -1, N).permute(0, 2, 1)

    n_sample = min(num_samples, N)
    idx1 = torch.randperm(N, device=feat1.device)[:n_sample]
    idx2 = torch.randperm(N, device=feat1.device)[:n_sample]

    feat1_s = F.normalize(feat1_flat[:, idx1], dim=-1)
    code1_s = F.normalize(code1_flat[:, idx1], dim=-1)
    feat2_s = F.normalize(feat2_flat[:, idx2], dim=-1)
    code2_s = F.normalize(code2_flat[:, idx2], dim=-1)

    with torch.no_grad():
        fd = torch.bmm(feat1_s, feat2_s.transpose(1, 2))
    cd = torch.bmm(code1_s, code2_s.transpose(1, 2))

    loss = -(cd.clamp(0) * (fd - shift))
    return loss.mean()


def stego_correlation_loss(feat, code, pos_intra_shift=0.18, neg_inter_shift=0.46,
                            num_samples=100, neg_samples=3):
    """Full STEGO loss with intra-image positive + inter-image negative terms.

    Args:
        feat: Backbone features (B, C_f, H, W).
        code: Projected codes (B, C_c, H, W).
        pos_intra_shift: Shift for same-image pairs (positive).
        neg_inter_shift: Shift for cross-image pairs (negative, larger → more repulsion).
        num_samples: Spatial samples per pair computation.
        neg_samples: Number of random negative permutations.
    """
    # Positive intra-image loss (same image, same features)
    pos_loss = _correlation_helper(feat, feat, code, code, pos_intra_shift, num_samples)

    # Negative inter-image loss (different images in batch)
    neg_loss = 0.0
    B = feat.shape[0]
    if B > 1:
        for _ in range(neg_samples):
            perm = torch.randperm(B, device=feat.device)
            # Ensure no image maps to itself
            while (perm == torch.arange(B, device=feat.device)).any():
                perm = torch.randperm(B, device=feat.device)
            neg_loss += _correlation_helper(
                feat, feat[perm], code, code[perm], neg_inter_shift, num_samples)
        neg_loss = neg_loss / neg_samples

    return pos_loss + neg_loss


def cluster_entropy_loss(cluster_probs):
    """Maximize entropy of average cluster assignment to prevent collapse.

    Args:
        cluster_probs: (B, K, H, W) softmax cluster probabilities.
    Returns:
        Negative entropy (to minimize).
    """
    # Average assignment across all spatial locations and batch
    avg_probs = cluster_probs.mean(dim=(0, 2, 3))  # (K,)
    avg_probs = avg_probs.clamp(min=1e-8)
    entropy = -(avg_probs * torch.log(avg_probs)).sum()
    # Maximize entropy → minimize negative entropy
    return -entropy


# ─── Dataset ───

class COCOFeatureDataset(torch.utils.data.Dataset):
    """Load pre-extracted DINOv3 features."""

    def __init__(self, feat_dir, patch_grid=32):
        self.feat_files = sorted(Path(feat_dir).glob("*.npy"))
        self.patch_grid = patch_grid

    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        feat = np.load(self.feat_files[idx])  # (n_patches, feat_dim)
        feat = torch.from_numpy(feat).float()
        # Reshape to spatial grid
        feat = feat.reshape(self.patch_grid, self.patch_grid, -1)
        feat = feat.permute(2, 0, 1)  # (C, H, W)
        image_id = int(self.feat_files[idx].stem)
        return feat, image_id


# ─── GT Loading for Evaluation ───

def load_coco_panoptic_gt(coco_root):
    """Load COCO panoptic GT for evaluation."""
    panoptic_json = os.path.join(coco_root, "annotations", "panoptic_val2017.json")
    panoptic_dir = os.path.join(coco_root, "annotations", "panoptic_val2017")

    with open(panoptic_json) as f:
        data = json.load(f)

    cat_map = {}
    for cat in data["categories"]:
        cat_map[cat["id"]] = cat["supercategory"]

    ann_map = {}
    for ann in data["annotations"]:
        ann_map[ann["image_id"]] = ann

    return cat_map, ann_map, panoptic_dir


def get_gt_semantic(cat_map, ann_map, panoptic_dir, image_id):
    """Get 27-class semantic GT for one image."""
    if image_id not in ann_map:
        return None
    ann = ann_map[image_id]
    pan_path = os.path.join(panoptic_dir, ann["file_name"])
    pan_img = np.array(Image.open(pan_path))
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


# ─── Hungarian Matching + mIoU ─��─

def hungarian_match_and_miou(predictions, gt_labels, n_clusters, n_classes=27, n_eval=100):
    """Match cluster IDs to GT classes via Hungarian + compute mIoU."""
    cost_matrix = np.zeros((n_clusters, n_classes), dtype=np.float64)

    for pred, gt in zip(predictions[:n_eval], gt_labels[:n_eval]):
        if gt is None:
            continue
        # Resize GT to match prediction grid
        gt_resized = np.array(Image.fromarray(gt).resize(
            (pred.shape[1], pred.shape[0]), Image.NEAREST))
        for p, g in zip(pred.flatten(), gt_resized.flatten()):
            if g < n_classes:
                cost_matrix[int(p), g] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    for k_id in range(n_clusters):
        if k_id not in cluster_to_class:
            best_class = int(np.argmax(cost_matrix[k_id]))
            cluster_to_class[k_id] = best_class

    # Compute mIoU on ALL images
    iou_per_class = np.zeros(n_classes)
    count_per_class = np.zeros(n_classes)

    for pred, gt in zip(predictions, gt_labels):
        if gt is None:
            continue
        # Map predictions to classes
        pred_mapped = np.vectorize(cluster_to_class.get)(pred).astype(np.uint8)
        pred_full = np.array(Image.fromarray(pred_mapped).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST))

        for c in range(n_classes):
            gt_mask = gt == c
            pred_mask = pred_full == c
            inter = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou_per_class[c] += inter / union
                count_per_class[c] += 1

    valid = count_per_class > 0
    miou = (iou_per_class[valid] / count_per_class[valid]).mean() * 100
    per_class = {}
    for c in range(n_classes):
        if count_per_class[c] > 0:
            per_class[COCOSTUFF27_CLASSNAMES[c]] = round(
                iou_per_class[c] / count_per_class[c] * 100, 1)
    return miou, per_class, cluster_to_class


# ─── Feature Augmentation ───

def augment_feature_grid(feat, crop_ratio=0.75):
    """Spatial augmentation on feature grids: random crop + flip.

    Args:
        feat: (B, C, H, W) feature grid.
        crop_ratio: Fraction of spatial extent to crop.
    Returns:
        Augmented feature grid of same shape.
    """
    B, C, H, W = feat.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)

    # Random crop origin
    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    feat_crop = feat[:, :, top:top+crop_h, left:left+crop_w]

    # Resize back to original size via bilinear interpolation
    feat_aug = F.interpolate(feat_crop, size=(H, W), mode='bilinear', align_corners=False)

    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        feat_aug = feat_aug.flip(dims=[3])

    return feat_aug


# ─── Training ───

def train(args):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    device = torch.device(args.device)
    root = Path(args.coco_root)

    # Load features
    feat_dirs = [root / "dinov3_features" / "val2017"]
    if args.use_train:
        feat_dirs.append(root / "dinov3_features" / "train2017")

    all_feat_files = []
    for fd in feat_dirs:
        all_feat_files.extend(sorted(fd.glob("*.npy")))

    print(f"\n{'='*60}")
    print(f"COCO CLUSTER LOOKUP TRAINING (Path 3)")
    print(f"{'='*60}")
    print(f"  Features: {len(all_feat_files)} images from {[str(d) for d in feat_dirs]}")
    print(f"  K:        {args.k}")
    print(f"  Proj dim: {args.proj_dim}")
    print(f"  LR:       {args.lr}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Pos shift:{args.pos_intra_shift}")
    print(f"  Neg shift:{args.neg_inter_shift}")
    print(f"  Alpha:    {args.alpha}")
    print(f"  λ_entropy:{args.lambda_entropy}")
    print(f"  Augment:  {args.augment}")
    print(f"  Device:   {device}")

    # Build dataset from val features only (for training on pre-extracted)
    dataset = COCOFeatureDataset(root / "dinov3_features" / "val2017")
    if args.use_train:
        train_ds = COCOFeatureDataset(root / "dinov3_features" / "train2017")
        dataset = torch.utils.data.ConcatDataset([dataset, train_ds])

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True)

    # Model
    projector = Projector(feat_dim=1024, proj_dim=args.proj_dim).to(device)
    cluster_lookup = ClusterLookup(dim=args.proj_dim, n_classes=args.k).to(device)

    # ─── K-means initialization ───
    if args.kmeans_init:
        print("  Initializing with k-means on projected features...")
        # Collect a sample of features
        sample_feats = []
        for i, (feat_batch, _) in enumerate(loader):
            if i >= 20:  # 20 batches should be enough
                break
            feat_batch = feat_batch.to(device)
            feat_normed = F.normalize(feat_batch, dim=1)
            with torch.no_grad():
                code = projector(feat_normed)
            # (B, proj_dim, H, W) -> (B*H*W, proj_dim)
            code_flat = code.permute(0, 2, 3, 1).reshape(-1, args.proj_dim)
            code_flat = F.normalize(code_flat, dim=1)
            sample_feats.append(code_flat.cpu().numpy())
        sample_feats = np.concatenate(sample_feats, axis=0)

        kmeans = MiniBatchKMeans(n_clusters=args.k, batch_size=5000, max_iter=100,
                                  random_state=42)
        kmeans.fit(sample_feats)
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        centroids = F.normalize(centroids, dim=1)
        cluster_lookup.clusters.data.copy_(centroids)
        print(f"  K-means init done (fit on {len(sample_feats)} samples)")

    params = list(projector.parameters()) + list(cluster_lookup.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    t0 = time.time()
    for epoch in range(args.epochs):
        projector.train()
        cluster_lookup.train()
        epoch_corr_loss = 0
        epoch_cluster_loss = 0
        epoch_entropy_loss = 0
        n_batches = 0

        for feat_batch, _ in loader:
            feat_batch = feat_batch.to(device)  # (B, 1024, 32, 32)

            # L2-normalize input features
            feat_normed = F.normalize(feat_batch, dim=1)

            # Project original
            code = projector(feat_normed)  # (B, proj_dim, 32, 32)

            # Augmentation: create positive pair
            if args.augment:
                feat_aug = augment_feature_grid(feat_normed)
                code_aug = projector(feat_aug)
                # Positive inter: original vs augmented (same image)
                pos_inter_loss = _correlation_helper(
                    feat_normed, feat_aug, code, code_aug,
                    args.pos_intra_shift, args.num_samples)
            else:
                pos_inter_loss = 0.0

            # STEGO loss (intra + neg inter)
            corr_loss = stego_correlation_loss(
                feat_normed, code,
                pos_intra_shift=args.pos_intra_shift,
                neg_inter_shift=args.neg_inter_shift,
                num_samples=args.num_samples,
                neg_samples=args.neg_samples)
            cluster_loss, cluster_probs = cluster_lookup(code, alpha=args.alpha)
            entropy_loss = cluster_entropy_loss(cluster_probs)

            loss = (corr_loss + pos_inter_loss
                    + args.lambda_cluster * cluster_loss
                    + args.lambda_entropy * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_corr_loss += (corr_loss.item() if torch.is_tensor(corr_loss) else corr_loss)
            epoch_cluster_loss += cluster_loss.item()
            epoch_entropy_loss += entropy_loss.item()
            n_batches += 1

        scheduler.step()
        avg_corr = epoch_corr_loss / max(n_batches, 1)
        avg_clust = epoch_cluster_loss / max(n_batches, 1)
        avg_ent = epoch_entropy_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
              f"corr={avg_corr:.4f}  cluster={avg_clust:.4f}  entropy={avg_ent:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s")

    # ─── Inference on val ───
    print(f"\n  Generating predictions for val2017...")
    val_dataset = COCOFeatureDataset(root / "dinov3_features" / "val2017")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=0)

    projector.eval()
    cluster_lookup.eval()

    all_preds = []
    all_image_ids = []
    with torch.no_grad():
        for feat_batch, img_ids in val_loader:
            feat_batch = feat_batch.to(device)
            feat_normed = F.normalize(feat_batch, dim=1)
            code = projector(feat_normed)
            preds = cluster_lookup.predict(code)  # (B, 32, 32)
            all_preds.append(preds.cpu().numpy())
            all_image_ids.extend(img_ids.tolist())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 32, 32)

    # ─── Evaluation ───
    print(f"\n  Evaluating mIoU...")
    cat_map, ann_map, panoptic_dir = load_coco_panoptic_gt(args.coco_root)

    gt_labels = []
    for img_id in all_image_ids:
        gt = get_gt_semantic(cat_map, ann_map, panoptic_dir, img_id)
        gt_labels.append(gt)

    miou, per_class, cluster_to_class = hungarian_match_and_miou(
        all_preds, gt_labels, args.k)

    print(f"\n{'='*60}")
    print(f"RESULTS: K={args.k}, proj={args.proj_dim}, lr={args.lr}, "
          f"pos_shift={args.pos_intra_shift}, neg_shift={args.neg_inter_shift}, "
          f"alpha={args.alpha}")
    print(f"{'='*60}")
    print(f"  mIoU: {miou:.1f}%")
    for name, iou in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}: {iou:.1f}%")

    # ─── Save pseudo-labels ───
    out_subdir = f"pseudo_semantic_cups_k{args.k}"
    out_dir = root / out_subdir / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Saving pseudo-labels to {out_dir}...")
    for idx, img_id in enumerate(all_image_ids):
        pred = all_preds[idx]  # (32, 32) cluster IDs
        class_map = np.vectorize(cluster_to_class.get)(pred).astype(np.uint8)

        # Get original image size
        img_path = root / "val2017" / f"{img_id:012d}.jpg"
        if img_path.exists():
            img = Image.open(img_path)
            W, H = img.size
            class_map_full = np.array(
                Image.fromarray(class_map).resize((W, H), Image.NEAREST))
        else:
            class_map_full = class_map

        Image.fromarray(class_map_full).save(out_dir / f"{img_id:012d}.png")

    # Save metadata
    meta = {
        "method": "cluster_lookup",
        "k": args.k,
        "proj_dim": args.proj_dim,
        "lr": args.lr,
        "pos_intra_shift": args.pos_intra_shift,
        "neg_inter_shift": args.neg_inter_shift,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "lambda_cluster": args.lambda_cluster,
        "lambda_entropy": args.lambda_entropy,
        "use_train": args.use_train,
        "miou": round(miou, 2),
        "per_class": per_class,
        "cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
        "training_time_s": round(elapsed, 1),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    # Save model
    ckpt_path = out_dir / "model.pth"
    torch.save({
        "projector": projector.state_dict(),
        "cluster_lookup": cluster_lookup.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"  Saved model to {ckpt_path}")

    return miou, per_class


def main():
    parser = argparse.ArgumentParser(description="Train ClusterLookup on COCO features")
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--k", type=int, default=80, help="Number of clusters")
    parser.add_argument("--proj_dim", type=int, default=90, help="Projector output dim")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pos_intra_shift", type=float, default=0.18,
                        help="STEGO shift for intra-image positive pairs")
    parser.add_argument("--neg_inter_shift", type=float, default=0.46,
                        help="STEGO shift for inter-image negative pairs")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Cluster softmax temperature")
    parser.add_argument("--lambda_cluster", type=float, default=1.0,
                        help="Weight for cluster loss")
    parser.add_argument("--lambda_entropy", type=float, default=2.0,
                        help="Weight for cluster entropy loss")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Spatial samples for correlation loss")
    parser.add_argument("--neg_samples", type=int, default=3,
                        help="Number of negative permutations per batch")
    parser.add_argument("--use_train", action="store_true",
                        help="Also use train2017 features for training")
    parser.add_argument("--augment", action="store_true",
                        help="Spatial augmentation on feature grids")
    parser.add_argument("--kmeans_init", action="store_true",
                        help="Initialize clusters from k-means on projected features")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
