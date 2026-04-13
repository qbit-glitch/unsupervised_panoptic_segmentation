#!/usr/bin/env python3
"""Train a pairwise fragment merge predictor.

Two-phase script:
  Phase 1: Extract training pairs from multi-threshold Sobel consensus
  Phase 2: Train MergePredictor MLP on pairwise descriptors

Self-supervised labels:
  - Run Sobel+CC at tau_low (0.10) -> over-segmented fragments
  - Run Sobel+CC at tau_high (0.20) -> "oracle" grouping
  - Adjacent same-class fragments at tau_low that belong to same CC at tau_high -> MERGE
  - Adjacent same-class fragments at tau_low that belong to different CCs at tau_high -> NO-MERGE

Usage:
    python mbps_pytorch/train_merge_predictor.py \
        --cityscapes_root /path/to/cityscapes \
        --epochs 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from instance_methods.learned_merge import (
    _extract_pairwise_descriptor,
    _find_adjacent_pairs,
    _oversegment_sobel_cc,
)
from instance_methods.utils import load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THING_IDS = set(range(11, 19))
WORK_H, WORK_W = 512, 1024


class MergePredictor(nn.Module):
    """Binary classifier: should two fragments merge?

    Args:
        input_dim: dimensionality of pairwise descriptor.
        hidden_dim: hidden layer size.
    """

    def __init__(self, input_dim: int = 200, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discover_train_files(
    cs_root: Path,
    split: str,
    semantic_subdir: str,
    depth_subdir: str,
    feature_subdir: str,
) -> list:
    """Find matching (semantic, depth, feature) tuples for training."""
    sem_dir = cs_root / semantic_subdir / split
    depth_dir = cs_root / depth_subdir / split
    feat_dir = cs_root / feature_subdir / split

    files = []
    if not sem_dir.exists():
        logger.error(f"Semantic dir not found: {sem_dir}")
        return files

    for city_dir in sorted(sem_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for sem_path in sorted(city_dir.glob("*.png")):
            stem = sem_path.stem

            depth_path = depth_dir / city / f"{stem}.npy"
            if not depth_path.exists():
                alt = stem.replace("_leftImg8bit", "")
                depth_path = depth_dir / city / f"{alt}.npy"
            if not depth_path.exists():
                depth_path = depth_dir / city / f"{stem}_leftImg8bit.npy"

            feat_path = feat_dir / city / f"{stem}.npy"
            if not feat_path.exists():
                alt = stem.replace("_leftImg8bit", "")
                feat_path = feat_dir / city / f"{alt}.npy"
            if not feat_path.exists():
                feat_path = feat_dir / city / f"{stem}_leftImg8bit.npy"

            if depth_path.exists() and feat_path.exists():
                files.append((str(sem_path), str(depth_path), str(feat_path)))

    return files


def extract_pairs_from_image(
    sem_path: str,
    depth_path: str,
    feat_path: str,
    k_to_trainid: np.ndarray,
    tau_low: float = 0.10,
    tau_high: float = 0.20,
    pca=None,
) -> tuple:
    """Extract merge/no-merge pairs from one image.

    Returns:
        descriptors: list of (D,) arrays.
        labels: list of 0/1 ints.
    """
    from PIL import Image as PILImage

    # Load
    pred_k = np.array(PILImage.open(sem_path))
    semantic = k_to_trainid[pred_k]
    if semantic.shape != (WORK_H, WORK_W):
        semantic = np.array(
            PILImage.fromarray(semantic).resize(
                (WORK_W, WORK_H), PILImage.NEAREST
            )
        )

    depth = np.load(depth_path)
    if depth.shape != (WORK_H, WORK_W):
        depth = np.array(
            PILImage.fromarray(depth).resize(
                (WORK_W, WORK_H), PILImage.BILINEAR
            )
        )

    features = load_features(feat_path)

    # Get fragments at tau_low
    frags_low = _oversegment_sobel_cc(
        semantic, depth, THING_IDS,
        grad_threshold=tau_low, min_area=100,
    )
    if len(frags_low) < 2:
        return [], []

    # Get fragments at tau_high (oracle grouping)
    frags_high = _oversegment_sobel_cc(
        semantic, depth, THING_IDS,
        grad_threshold=tau_high, min_area=100,
    )

    # Build a label map from tau_high fragments
    high_map = np.full((WORK_H, WORK_W), -1, dtype=np.int32)
    for idx, (mask, cls, area) in enumerate(frags_high):
        high_map[mask] = idx

    # Find adjacent pairs at tau_low
    pairs = _find_adjacent_pairs(frags_low)

    descriptors = []
    labels = []
    for i, j in pairs:
        mask_i = frags_low[i][0]
        mask_j = frags_low[j][0]

        # Determine if these fragments map to same CC at tau_high
        high_ids_i = high_map[mask_i]
        high_ids_i = high_ids_i[high_ids_i >= 0]
        high_ids_j = high_map[mask_j]
        high_ids_j = high_ids_j[high_ids_j >= 0]

        if len(high_ids_i) == 0 or len(high_ids_j) == 0:
            continue

        # Majority vote for which high-threshold CC each belongs to
        mode_i = int(np.bincount(high_ids_i).argmax())
        mode_j = int(np.bincount(high_ids_j).argmax())

        label = 1 if mode_i == mode_j else 0

        desc = _extract_pairwise_descriptor(
            frags_low[i], frags_low[j], features, depth, pca
        )
        descriptors.append(desc)
        labels.append(label)

    return descriptors, labels


def extract_all_pairs(
    files: list,
    k_to_trainid: np.ndarray,
    tau_low: float,
    tau_high: float,
    pca=None,
    max_images: int = None,
) -> tuple:
    """Extract pairs from all images.

    Returns:
        X: (N, D) float32 array.
        y: (N,) int array of 0/1 labels.
    """
    if max_images:
        files = files[:max_images]

    all_desc = []
    all_labels = []
    for sem_path, depth_path, feat_path in tqdm(
        files, desc="Extracting pairs", ncols=100
    ):
        descs, labels = extract_pairs_from_image(
            sem_path, depth_path, feat_path,
            k_to_trainid, tau_low, tau_high, pca
        )
        all_desc.extend(descs)
        all_labels.extend(labels)

    if not all_desc:
        return np.zeros((0, 1)), np.zeros(0, dtype=np.int64)

    X = np.stack(all_desc).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def fit_pca_on_features(
    files: list,
    k_to_trainid: np.ndarray,
    n_components: int = 64,
    max_images: int = 200,
) -> PCA:
    """Fit PCA on fragment-level mean DINOv2 features."""
    from PIL import Image as PILImage

    all_feats = []
    for sem_path, depth_path, feat_path in tqdm(
        files[:max_images], desc="Fitting PCA", ncols=100
    ):
        features = load_features(feat_path)
        pred_k = np.array(PILImage.open(sem_path))
        semantic = k_to_trainid[pred_k]
        if semantic.shape != (WORK_H, WORK_W):
            semantic = np.array(
                PILImage.fromarray(semantic).resize(
                    (WORK_W, WORK_H), PILImage.NEAREST
                )
            )

        fh, fw, C = features.shape
        sem_ds = np.array(
            PILImage.fromarray(semantic).resize((fw, fh), PILImage.NEAREST)
        )

        for cls in THING_IDS:
            cls_mask = sem_ds == cls
            if cls_mask.sum() < 5:
                continue
            mean_feat = features[cls_mask].mean(axis=0)
            all_feats.append(mean_feat)

    if len(all_feats) < n_components:
        logger.warning(f"Only {len(all_feats)} features for PCA "
                       f"(need >= {n_components})")
        n_components = max(2, len(all_feats) // 2)

    X = np.stack(all_feats).astype(np.float32)
    # Clean NaN/Inf before PCA fitting
    bad_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if bad_mask.sum() > 0:
        logger.warning(f"Removing {bad_mask.sum()} NaN/Inf samples from PCA data")
        X = X[~bad_mask]
    logger.info(f"Fitting PCA: {X.shape[0]} samples, "
                f"{X.shape[1]} -> {n_components} dims")

    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance: {explained:.3f}")
    return pca


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    device: str = "cpu",
) -> tuple:
    """Train MergePredictor.

    Returns:
        model: trained MergePredictor.
        history: dict with training metrics.
    """
    input_dim = X_train.shape[1]
    model = MergePredictor(input_dim=input_dim, hidden_dim=hidden_dim)
    model.to(device)

    # Class balance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    logger.info(f"Class balance: {n_pos} pos, {n_neg} neg, "
                f"pos_weight={pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

            # Precision/recall for positive class
            tp = ((val_preds == 1) & (y_val_t == 1)).sum().item()
            fp = ((val_preds == 1) & (y_val_t == 0)).sum().item()
            fn = ((val_preds == 0) & (y_val_t == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:2d}/{epochs}  "
            f"loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  P={precision:.3f}  R={recall:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Best val accuracy: {best_acc:.3f} at epoch {best_epoch}")

    return model, history, best_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train pairwise fragment merge predictor"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_val", type=str, default="val")
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_raw_k80")
    parser.add_argument("--depth_subdir", type=str, default="depth_spidepth")
    parser.add_argument("--feature_subdir", type=str,
                        default="dinov2_features")
    parser.add_argument("--centroids_path", type=str, default=None)
    parser.add_argument("--tau_low", type=float, default=0.10)
    parser.add_argument("--tau_high", type=float, default=0.20)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=200)
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/merge_predictor")
    parser.add_argument("--no_pca", action="store_true",
                        help="Skip PCA, use raw 768-dim features")
    args = parser.parse_args()

    cs_root = Path(args.cityscapes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load centroids for k->trainid mapping
    centroids_path = args.centroids_path or str(
        cs_root / args.semantic_subdir / "kmeans_centroids.npz"
    )
    cent_data = np.load(centroids_path)
    c2c = cent_data["cluster_to_class"]
    k_to_trainid = np.full(256, 255, dtype=np.uint8)
    for cid, tid in enumerate(c2c):
        k_to_trainid[cid] = int(tid)

    # Discover files
    logger.info("Discovering files...")
    train_files = discover_train_files(
        cs_root, args.split_train, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir
    )
    val_files = discover_train_files(
        cs_root, args.split_val, args.semantic_subdir,
        args.depth_subdir, args.feature_subdir
    )
    logger.info(f"Train: {len(train_files)} images, Val: {len(val_files)} images")

    # Phase 1a: Fit PCA
    pca = None
    if not args.no_pca:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 1a: Fitting PCA on DINOv2 features")
        logger.info(f"{'='*60}")
        pca = fit_pca_on_features(
            train_files, k_to_trainid,
            n_components=args.pca_dim,
            max_images=min(200, len(train_files)),
        )
        # Save PCA
        pca_path = output_dir / "pca.npz"
        np.savez(
            str(pca_path),
            components=pca.components_,
            mean=pca.mean_,
            explained_variance=pca.explained_variance_,
        )
        logger.info(f"PCA saved to {pca_path}")

    # Phase 1b: Extract training pairs
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1b: Extracting pairs (tau_low={args.tau_low}, "
                f"tau_high={args.tau_high})")
    logger.info(f"{'='*60}")

    X_train, y_train = extract_all_pairs(
        train_files, k_to_trainid,
        args.tau_low, args.tau_high, pca,
        max_images=args.max_train_images,
    )
    logger.info(f"Train: {len(X_train)} pairs, "
                f"{(y_train == 1).sum()} merge, {(y_train == 0).sum()} no-merge")

    X_val, y_val = extract_all_pairs(
        val_files, k_to_trainid,
        args.tau_low, args.tau_high, pca,
        max_images=args.max_val_images,
    )
    logger.info(f"Val: {len(X_val)} pairs, "
                f"{(y_val == 1).sum()} merge, {(y_val == 0).sum()} no-merge")

    if len(X_train) < 10:
        logger.error("Too few training pairs. Aborting.")
        sys.exit(1)

    # Save extracted data
    data_path = output_dir / "training_data.npz"
    np.savez_compressed(
        str(data_path),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    logger.info(f"Training data saved to {data_path}")

    # Phase 2: Train
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: Training MergePredictor")
    logger.info(f"{'='*60}")

    # Handle NaN/Inf in features
    nan_mask = np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1)
    if nan_mask.sum() > 0:
        logger.warning(f"Removing {nan_mask.sum()} NaN/Inf samples from train")
        X_train = X_train[~nan_mask]
        y_train = y_train[~nan_mask]

    nan_mask = np.isnan(X_val).any(axis=1) | np.isinf(X_val).any(axis=1)
    if nan_mask.sum() > 0:
        logger.warning(f"Removing {nan_mask.sum()} NaN/Inf samples from val")
        X_val = X_val[~nan_mask]
        y_val = y_val[~nan_mask]

    model, history, best_acc, best_epoch = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=str(device),
    )

    # Save checkpoint
    ckpt_path = output_dir / "best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": X_train.shape[1],
            "hidden_dim": args.hidden_dim,
            "tau_low": args.tau_low,
            "tau_high": args.tau_high,
            "pca_dim": args.pca_dim if not args.no_pca else None,
        },
        "epoch": best_epoch,
        "val_accuracy": best_acc,
        "history": history,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }, str(ckpt_path))
    logger.info(f"\nCheckpoint saved to {ckpt_path}")
    logger.info(f"Best val accuracy: {best_acc:.3f} at epoch {best_epoch}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Train pairs: {len(X_train)} "
                f"({(y_train==1).sum()} merge, {(y_train==0).sum()} no-merge)")
    logger.info(f"  Val pairs: {len(X_val)} "
                f"({(y_val==1).sum()} merge, {(y_val==0).sum()} no-merge)")
    logger.info(f"  Input dim: {X_train.shape[1]}")
    logger.info(f"  Best val accuracy: {best_acc:.3f} (epoch {best_epoch})")
    logger.info(f"  Checkpoint: {ckpt_path}")
    if pca is not None:
        logger.info(f"  PCA: {output_dir / 'pca.npz'}")


if __name__ == "__main__":
    main()
