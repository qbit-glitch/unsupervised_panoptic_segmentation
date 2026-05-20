#!/usr/bin/env python3
"""Train the lightweight superpixel affinity adapter ablation.

This is a separate ablation lane for learnable instances.  It keeps DCFA,
DINO, and optional CLIP features frozen, then trains a tiny edge MLP over a
superpixel graph.  The pseudo-instance prior is the existing depth-gradient
connected-component teacher used by the current DCFA/SIMCF pipeline:

    semantic k=80/DCFA -> map to trainID -> depth Sobel tau/A_min CC prior

The supervision follows the Superpixels paper's reliability idea: hard labels
are used only for clean superpixel pairs, while uncertain pairs receive a weak
soft affinity target from color/depth/DINO/CLIP agreement.

Example:
    python mbps_pytorch/train_superpixel_affinity_adapter.py \\
        --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes \\
        --semantic_subdir pseudo_semantic_raw_k80 \\
        --centroids_path /Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz \\
        --depth_subdir depth_depthpro \\
        --feature_subdir dinov2_features \\
        --output_dir checkpoints/superpixel_affinity_adapter \\
        --max_train_images 200 --max_val_images 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.adaptive_instance_semantics import (
    infer_semantic_spec,
    map_to_trainid,
)
from mbps_pytorch.instance_methods.superpixel_affinity import (
    SuperpixelExtractionConfig,
    build_depth_cc_prior,
    build_superpixel_graph,
    load_feature_grid,
    load_proposal_bank,
    resize_bilinear,
    resize_nearest,
)
from mbps_pytorch.models.instance.superpixel_affinity_adapter import (
    SuperpixelAffinityAdapter,
    weighted_bce_with_logits,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_existing(root: Path, candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_int_tuple(text: str | None) -> tuple[int, ...]:
    """Parse comma-separated class ids for class-aware ablations."""
    if text is None or str(text).strip() == "":
        return ()
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return tuple(out)


def discover_entries(
    cityscapes_root: Path,
    split: str,
    semantic_subdir: str,
    depth_subdir: str,
    feature_subdir: str | None,
    clip_feature_subdir: str | None,
    proposal_bank_dir: str | None,
) -> list[dict]:
    """Find matched image/semantic/depth/feature files."""
    img_root = cityscapes_root / "leftImg8bit" / split
    entries: list[dict] = []
    for image_path in sorted(img_root.rglob("*_leftImg8bit.png")):
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")
        base = f"{city}_{stem.split('_', 1)[1]}" if not stem.startswith(city) else stem

        sem_path = resolve_existing(cityscapes_root, [
            cityscapes_root / semantic_subdir / split / city / f"{stem}.png",
            cityscapes_root / semantic_subdir / split / city / f"{stem}_leftImg8bit.png",
            cityscapes_root / semantic_subdir / split / city / f"{base}.png",
        ])
        depth_path = resolve_existing(cityscapes_root, [
            cityscapes_root / depth_subdir / split / city / f"{stem}.npy",
            cityscapes_root / depth_subdir / split / city / f"{stem}_leftImg8bit.npy",
            cityscapes_root / depth_subdir / split / city / f"{base}.npy",
        ])
        feat_path = None
        if feature_subdir:
            feat_path = resolve_existing(cityscapes_root, [
                cityscapes_root / feature_subdir / split / city / f"{stem}_leftImg8bit.npy",
                cityscapes_root / feature_subdir / split / city / f"{stem}.npy",
                cityscapes_root / feature_subdir / split / city / f"{base}.npy",
            ])
        clip_path = None
        if clip_feature_subdir:
            clip_path = resolve_existing(cityscapes_root, [
                cityscapes_root / clip_feature_subdir / split / city / f"{stem}_leftImg8bit.npy",
                cityscapes_root / clip_feature_subdir / split / city / f"{stem}.npy",
                cityscapes_root / clip_feature_subdir / split / city / f"{base}.npy",
            ])
        proposal_path = None
        if proposal_bank_dir:
            proposal_root = Path(proposal_bank_dir)
            if not proposal_root.is_absolute():
                proposal_root = cityscapes_root / proposal_root
            proposal_path = resolve_existing(cityscapes_root, [
                proposal_root / split / city / f"{stem}.npz",
                proposal_root / split / city / f"{stem}_leftImg8bit.npz",
                proposal_root / split / city / f"{base}.npz",
                proposal_root / split / f"{stem}.npz",
                proposal_root / split / f"{base}.npz",
                proposal_root / city / f"{stem}.npz",
                proposal_root / city / f"{base}.npz",
                proposal_root / f"{stem}.npz",
                proposal_root / f"{base}.npz",
            ])

        if sem_path is None or depth_path is None:
            continue
        if feature_subdir and feat_path is None:
            continue
        entries.append({
            "city": city,
            "stem": stem,
            "image": image_path,
            "semantic": sem_path,
            "depth": depth_path,
            "feature": feat_path,
            "clip": clip_path,
            "proposal": proposal_path,
        })
    return entries


def load_entry_graph(
    entry: dict,
    semantic_spec,
    cfg: SuperpixelExtractionConfig,
    clip_prototypes: dict[int, np.ndarray] | None = None,
    dino_projection: np.ndarray | None = None,
    clip_projection: np.ndarray | None = None,
):
    image = np.array(Image.open(entry["image"]).convert("RGB"))
    h, w = image.shape[:2]
    sem_raw = np.array(Image.open(entry["semantic"]))
    if sem_raw.shape != (h, w):
        sem_raw = resize_nearest(sem_raw.astype(np.uint8), (h, w))
    sem_trainid = map_to_trainid(sem_raw, semantic_spec)

    depth = np.load(str(entry["depth"])).astype(np.float32)
    if depth.shape != (h, w):
        depth = resize_bilinear(depth, (h, w))

    dino = load_feature_grid(entry.get("feature"))
    clip = load_feature_grid(entry.get("clip"))
    proposal_masks, proposal_scores = load_proposal_bank(
        entry.get("proposal"), image_hw=(h, w))
    pseudo = build_depth_cc_prior(sem_trainid, depth, cfg)

    return build_superpixel_graph(
        image_rgb=image,
        semantic_trainid=sem_trainid,
        depth=depth,
        dino_features=dino,
        clip_features=clip,
        proposal_masks=proposal_masks,
        proposal_scores=proposal_scores,
        pseudo_instance_map=pseudo,
        clip_prototypes=clip_prototypes,
        dino_projection=dino_projection,
        clip_projection=clip_projection,
        cfg=cfg,
    )


def make_random_projection(
    entries: list[dict],
    key: str,
    out_dim: int,
    seed: int,
) -> np.ndarray | None:
    """Create a saved Gaussian projection for frozen DINO/CLIP features."""
    if out_dim <= 0:
        return None
    first = next((e.get(key) for e in entries if e.get(key) is not None), None)
    if first is None:
        return None
    feat = load_feature_grid(first)
    if feat is None or feat.shape[-1] <= out_dim:
        return None
    rng = np.random.default_rng(seed)
    proj = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(out_dim)),
        size=(feat.shape[-1], out_dim),
    ).astype(np.float32)
    log.info("Projection %s: %d -> %d", key, feat.shape[-1], out_dim)
    return proj


def fit_clip_prototypes(
    entries: list[dict],
    semantic_spec,
    cfg: SuperpixelExtractionConfig,
    max_images: int,
    prototypes_per_class: int,
    dino_projection: np.ndarray | None,
    clip_projection: np.ndarray | None,
) -> dict[int, np.ndarray] | None:
    """Fit UVIS-style class prototypes from optional CLIP feature caches."""
    if prototypes_per_class <= 0 or not any(e.get("clip") for e in entries):
        return None
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception:
        MiniBatchKMeans = None

    per_class: dict[int, list[np.ndarray]] = {c: [] for c in range(11, 19)}
    for entry in tqdm(entries[:max_images], desc="Fitting CLIP prototypes", ncols=100):
        if entry.get("clip") is None:
            continue
        graph = load_entry_graph(
            entry, semantic_spec, cfg, clip_prototypes=None,
            dino_projection=dino_projection, clip_projection=clip_projection)
        clip_feats = graph.clip_node_features
        if clip_feats.shape[1] == 0:
            continue
        reliable = (
            (graph.node_semantic_purity >= cfg.semantic_purity_min)
            & (graph.node_instance_purity >= cfg.instance_purity_min)
            & np.isin(graph.node_class, list(range(11, 19)))
        )
        for cls in range(11, 19):
            idx = np.where(reliable & (graph.node_class == cls))[0]
            if idx.size:
                per_class[cls].append(clip_feats[idx])

    prototypes: dict[int, np.ndarray] = {}
    for cls, chunks in per_class.items():
        if not chunks:
            continue
        feats = np.concatenate(chunks, axis=0).astype(np.float32)
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        k = min(prototypes_per_class, len(feats))
        if k <= 1 or MiniBatchKMeans is None:
            proto = feats.mean(axis=0, keepdims=True)
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048)
            km.fit(feats)
            proto = km.cluster_centers_.astype(np.float32)
        proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8)
        prototypes[cls] = proto.astype(np.float32)
        log.info("CLIP prototypes class %d: %d vectors from %d nodes", cls, len(proto), len(feats))
    return prototypes or None


def extract_pair_tensors(
    entries: list[dict],
    semantic_spec,
    cfg: SuperpixelExtractionConfig,
    max_images: int | None,
    clip_prototypes: dict[int, np.ndarray] | None,
    max_edges_per_image: int,
    dino_projection: np.ndarray | None,
    clip_projection: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract edge descriptors, targets, weights, and hard-label flags."""
    if max_images is not None:
        entries = entries[:max_images]
    all_x, all_y, all_w, all_hard = [], [], [], []
    stats = {
        "images": 0,
        "edges": 0,
        "hard_edges": 0,
        "positive_hard": 0,
        "negative_hard": 0,
        "skipped_empty": 0,
        "per_class": {
            str(c): {
                "edges": 0,
                "hard_edges": 0,
                "positive_hard": 0,
                "negative_hard": 0,
            }
            for c in range(11, 19)
        },
    }
    for entry in tqdm(entries, desc="Extracting superpixel edges", ncols=100):
        graph = load_entry_graph(
            entry, semantic_spec, cfg, clip_prototypes,
            dino_projection=dino_projection,
            clip_projection=clip_projection,
        )
        keep = graph.edge_weights > 0
        if keep.sum() == 0:
            stats["skipped_empty"] += 1
            continue
        idx = np.where(keep)[0]
        if max_edges_per_image > 0 and len(idx) > max_edges_per_image:
            rng = np.random.default_rng(abs(hash(entry["stem"])) % (2 ** 32))
            idx = rng.choice(idx, size=max_edges_per_image, replace=False)
        x = graph.edge_features[idx]
        y = graph.edge_targets[idx]
        w = graph.edge_weights[idx]
        hard = graph.edge_is_hard[idx]
        edge_cls = graph.node_class[graph.edge_index[idx, 0]]
        all_x.append(x)
        all_y.append(y)
        all_w.append(w)
        all_hard.append(hard)
        stats["images"] += 1
        stats["edges"] += int(len(idx))
        stats["hard_edges"] += int(hard.sum())
        stats["positive_hard"] += int(((y >= 0.5) & hard).sum())
        stats["negative_hard"] += int(((y < 0.5) & hard).sum())
        for cls in range(11, 19):
            cls_mask = edge_cls == cls
            if not cls_mask.any():
                continue
            cls_key = str(cls)
            cls_hard = hard & cls_mask
            stats["per_class"][cls_key]["edges"] += int(cls_mask.sum())
            stats["per_class"][cls_key]["hard_edges"] += int(cls_hard.sum())
            stats["per_class"][cls_key]["positive_hard"] += int(
                ((y >= 0.5) & cls_hard).sum())
            stats["per_class"][cls_key]["negative_hard"] += int(
                ((y < 0.5) & cls_hard).sum())

    if not all_x:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            stats,
        )
    return (
        np.concatenate(all_x, axis=0).astype(np.float32),
        np.concatenate(all_y, axis=0).astype(np.float32),
        np.concatenate(all_w, axis=0).astype(np.float32),
        np.concatenate(all_hard, axis=0).astype(bool),
        stats,
    )


def standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        mean.squeeze(0),
        std.squeeze(0),
    )


def evaluate_edge_metrics(logits: torch.Tensor, targets: torch.Tensor,
                          hard: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    out = {}
    hard_mask = hard.bool()
    if hard_mask.any():
        pred = probs[hard_mask] >= 0.5
        truth = targets[hard_mask] >= 0.5
        out["hard_acc"] = float((pred == truth).float().mean().item())
        tp = ((pred == 1) & (truth == 1)).sum().item()
        fp = ((pred == 1) & (truth == 0)).sum().item()
        fn = ((pred == 0) & (truth == 1)).sum().item()
        out["hard_precision"] = float(tp / max(tp + fp, 1))
        out["hard_recall"] = float(tp / max(tp + fn, 1))
    else:
        out["hard_acc"] = out["hard_precision"] = out["hard_recall"] = 0.0
    out["prob_mean"] = float(probs.mean().item())
    return out


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    log.info("Device: %s", device)
    root = Path(args.cityscapes_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_spec = infer_semantic_spec(
        root,
        args.semantic_subdir,
        semantic_mode=args.semantic_mode,
        centroids_path=args.centroids_path,
        num_semantic_classes=args.num_semantic_classes,
        split=args.split_train,
    )
    log.info("Semantic mode: %s | dim=%d | centroids=%s",
             semantic_spec.mode, semantic_spec.num_classes,
             semantic_spec.centroids_path or "none")

    cfg = SuperpixelExtractionConfig(
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.superpixel_sigma,
        min_superpixel_area=args.min_superpixel_area,
        pseudo_tau=args.pseudo_tau,
        pseudo_min_area=args.pseudo_min_area,
        pseudo_depth_sigma=args.pseudo_depth_sigma,
        pseudo_dilation=args.pseudo_dilation,
        semantic_purity_min=args.semantic_purity_min,
        instance_purity_min=args.instance_purity_min,
        positive_affinity_min=args.positive_affinity_min,
        negative_affinity_max=args.negative_affinity_max,
        negative_boundary_min=args.negative_boundary_min,
        intra_instance_negative_affinity_max=args.intra_instance_negative_affinity_max,
        intra_instance_negative_boundary_min=args.intra_instance_negative_boundary_min,
        intra_instance_negative_weight=args.intra_instance_negative_weight,
        balance_hard_negatives=not args.no_balance_hard_negatives,
        max_hard_pos_to_neg_ratio=args.max_hard_pos_to_neg_ratio,
        class_aware_negative_classes=parse_int_tuple(args.class_aware_negative_classes),
        class_aware_positive_affinity_min=args.class_aware_positive_affinity_min,
        class_aware_negative_weight=args.class_aware_negative_weight,
        class_aware_positive_weight=args.class_aware_positive_weight,
        class_aware_intra_instance_negative_affinity_max=(
            args.class_aware_intra_instance_negative_affinity_max),
        class_aware_intra_instance_negative_boundary_min=(
            args.class_aware_intra_instance_negative_boundary_min),
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        cross_class_negative_weight=args.cross_class_negative_weight,
        sigma_color=args.sigma_color,
        sigma_depth=args.sigma_depth,
        dino_temperature=args.dino_temperature,
        clip_temperature=args.clip_temperature,
        proposal_objectness_enabled=bool(args.proposal_objectness or args.proposal_bank_dir),
        proposal_objectness_top_k=args.proposal_objectness_top_k,
        proposal_objectness_min_score=args.proposal_objectness_min_score,
        proposal_objectness_support_thresh=args.proposal_objectness_support_thresh,
        proposal_soft_affinity_weight=args.proposal_soft_affinity_weight,
    )

    train_entries = discover_entries(
        root, args.split_train, args.semantic_subdir, args.depth_subdir,
        args.feature_subdir, args.clip_feature_subdir, args.proposal_bank_dir)
    val_entries = discover_entries(
        root, args.split_val, args.semantic_subdir, args.depth_subdir,
        args.feature_subdir, args.clip_feature_subdir, args.proposal_bank_dir)
    log.info("Files: train=%d val=%d", len(train_entries), len(val_entries))
    if cfg.proposal_objectness_enabled:
        log.info(
            "Proposal banks: train=%d/%d val=%d/%d root=%s",
            sum(1 for e in train_entries if e.get("proposal") is not None),
            len(train_entries),
            sum(1 for e in val_entries if e.get("proposal") is not None),
            len(val_entries),
            args.proposal_bank_dir or "none",
        )
    if len(train_entries) == 0:
        raise FileNotFoundError("No training entries found")
    if len(val_entries) == 0:
        log.warning("No val entries found; using a held-out slice of train")
        val_entries = train_entries[-max(1, min(64, len(train_entries) // 10)):]
        train_entries = train_entries[:-len(val_entries)] or train_entries
    if not args.no_shuffle_entries:
        rng = random.Random(args.seed)
        rng.shuffle(train_entries)
        rng = random.Random(args.seed + 1)
        rng.shuffle(val_entries)

    dino_projection = make_random_projection(
        train_entries, "feature", args.dino_project_dim, args.seed + 11)
    clip_projection = make_random_projection(
        train_entries, "clip", args.clip_project_dim, args.seed + 17)

    clip_prototypes = fit_clip_prototypes(
        train_entries,
        semantic_spec,
        cfg,
        max_images=min(args.clip_proto_images, len(train_entries)),
        prototypes_per_class=args.clip_prototypes_per_class,
        dino_projection=dino_projection,
        clip_projection=clip_projection,
    )

    train_x, train_y, train_w, train_hard, train_stats = extract_pair_tensors(
        train_entries, semantic_spec, cfg, args.max_train_images,
        clip_prototypes, args.max_edges_per_image,
        dino_projection=dino_projection,
        clip_projection=clip_projection)
    val_x, val_y, val_w, val_hard, val_stats = extract_pair_tensors(
        val_entries, semantic_spec, cfg, args.max_val_images,
        clip_prototypes, args.max_edges_per_image,
        dino_projection=dino_projection,
        clip_projection=clip_projection)
    log.info("Train stats: %s", train_stats)
    log.info("Val stats: %s", val_stats)
    if len(train_x) < 100:
        raise RuntimeError(f"Too few edge samples: {len(train_x)}")

    if len(val_x) == 0:
        val_x, val_y, val_w, val_hard = train_x[:1], train_y[:1], train_w[:1], train_hard[:1]
    train_x, val_x, feat_mean, feat_std = standardize(train_x, val_x)

    train_ds = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float(),
        torch.from_numpy(train_w).float(),
        torch.from_numpy(train_hard.astype(np.bool_)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_t = (
        torch.from_numpy(val_x).float().to(device),
        torch.from_numpy(val_y).float().to(device),
        torch.from_numpy(val_w).float().to(device),
        torch.from_numpy(val_hard.astype(np.bool_)).to(device),
    )

    model = SuperpixelAffinityAdapter(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
    ).to(device)
    log.info("Model params: %d | input_dim=%d",
             sum(p.numel() for p in model.parameters()), train_x.shape[1])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_metric = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for xb, yb, wb, _hard in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = weighted_bce_with_logits(logits, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.item()) * float(wb.sum().item())
            total_weight += float(wb.sum().item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vx, vy, vw, vh = val_t
            v_logits = model(vx)
            v_loss = weighted_bce_with_logits(v_logits, vy, vw)
            metrics = evaluate_edge_metrics(v_logits, vy, vh)
        train_loss = total_loss / max(total_weight, 1.0)
        metric = metrics["hard_acc"] if metrics["hard_acc"] > 0 else -float(v_loss.item())
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(v_loss.item()),
            **metrics,
        }
        history.append(row)
        log.info(
            "Epoch %02d/%02d train=%.4f val=%.4f hard_acc=%.3f P=%.3f R=%.3f prob=%.3f",
            epoch, args.epochs, train_loss, row["val_loss"],
            row["hard_acc"], row["hard_precision"], row["hard_recall"],
            row["prob_mean"],
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    proto_serialized = None
    if clip_prototypes:
        proto_serialized = {str(k): v.astype(np.float32) for k, v in clip_prototypes.items()}

    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_mean": feat_mean.astype(np.float32),
        "feature_std": feat_std.astype(np.float32),
        "config": {
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_layer_norm": not args.no_layer_norm,
            "semantic_subdir": args.semantic_subdir,
            "semantic_mode": semantic_spec.mode,
            "semantic_dim": semantic_spec.num_classes,
            "centroids_path": semantic_spec.centroids_path,
            "depth_subdir": args.depth_subdir,
            "feature_subdir": args.feature_subdir,
            "clip_feature_subdir": args.clip_feature_subdir,
            "proposal_bank_dir": args.proposal_bank_dir,
            "dino_project_dim": args.dino_project_dim,
            "clip_project_dim": args.clip_project_dim,
            "merge_threshold": args.default_merge_threshold,
            "min_area": args.output_min_area,
            "extraction": cfg.__dict__,
        },
        "dino_projection": dino_projection,
        "clip_projection": clip_projection,
        "clip_prototypes": proto_serialized,
        "history": history,
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    ckpt_path = out_dir / "best.pth"
    torch.save(ckpt, ckpt_path)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if args.save_edge_cache:
        try:
            np.savez_compressed(
                out_dir / "edge_training_cache.npz",
                train_x=train_x,
                train_y=train_y,
                train_w=train_w,
                train_hard=train_hard,
                val_x=val_x,
                val_y=val_y,
                val_w=val_w,
                val_hard=val_hard,
                feature_mean=feat_mean,
                feature_std=feat_std,
            )
        except OSError as exc:
            log.warning("Could not save edge cache: %s", exc)
    log.info("Saved checkpoint: %s", ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--semantic_subdir", default="pseudo_semantic_raw_k80")
    parser.add_argument("--semantic_mode", default="auto",
                        choices=["auto", "cluster", "trainid", "cause27"])
    parser.add_argument("--centroids_path", default=None)
    parser.add_argument("--num_semantic_classes", type=int, default=None)
    parser.add_argument("--depth_subdir", default="depth_depthpro")
    parser.add_argument("--feature_subdir", default="dinov2_features")
    parser.add_argument("--clip_feature_subdir", default=None)
    parser.add_argument("--proposal_bank_dir", default=None,
                        help="Optional proposal NPZ root; supports split/city, city, or flat layouts")
    parser.add_argument("--split_train", default="train")
    parser.add_argument("--split_val", default="val")
    parser.add_argument("--output_dir", default="checkpoints/superpixel_affinity_adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle_entries", action="store_true",
                        help="Keep filesystem order when applying max image limits")

    parser.add_argument("--n_segments", type=int, default=900)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--superpixel_sigma", type=float, default=0.8)
    parser.add_argument("--min_superpixel_area", type=int, default=12)
    parser.add_argument("--pseudo_tau", type=float, default=0.20)
    parser.add_argument("--pseudo_min_area", type=int, default=1000)
    parser.add_argument("--pseudo_depth_sigma", type=float, default=1.0)
    parser.add_argument("--pseudo_dilation", type=int, default=3)
    parser.add_argument("--semantic_purity_min", type=float, default=0.60)
    parser.add_argument("--instance_purity_min", type=float, default=0.55)
    parser.add_argument("--positive_affinity_min", type=float, default=0.45)
    parser.add_argument("--negative_affinity_max", type=float, default=0.25)
    parser.add_argument("--negative_boundary_min", type=float, default=0.08)
    parser.add_argument("--intra_instance_negative_affinity_max", type=float, default=0.18)
    parser.add_argument("--intra_instance_negative_boundary_min", type=float, default=0.10)
    parser.add_argument("--intra_instance_negative_weight", type=float, default=0.75)
    parser.add_argument("--no_balance_hard_negatives", action="store_true")
    parser.add_argument("--max_hard_pos_to_neg_ratio", type=float, default=4.0)
    parser.add_argument("--class_aware_negative_classes", default="",
                        help="Comma-separated thing trainIDs to emphasize, e.g. 11,12,18")
    parser.add_argument("--class_aware_positive_affinity_min", type=float, default=None)
    parser.add_argument("--class_aware_negative_weight", type=float, default=1.0)
    parser.add_argument("--class_aware_positive_weight", type=float, default=1.0)
    parser.add_argument("--class_aware_intra_instance_negative_affinity_max",
                        type=float, default=None)
    parser.add_argument("--class_aware_intra_instance_negative_boundary_min",
                        type=float, default=None)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--soft_weight", type=float, default=0.10)
    parser.add_argument("--cross_class_negative_weight", type=float, default=0.05)
    parser.add_argument("--sigma_color", type=float, default=0.08)
    parser.add_argument("--sigma_depth", type=float, default=0.04)
    parser.add_argument("--dino_temperature", type=float, default=0.20)
    parser.add_argument("--clip_temperature", type=float, default=0.20)
    parser.add_argument("--proposal_objectness", action="store_true",
                        help="Append proposal-bank objectness/support descriptors to the adapter")
    parser.add_argument("--proposal_objectness_top_k", type=int, default=100)
    parser.add_argument("--proposal_objectness_min_score", type=float, default=None)
    parser.add_argument("--proposal_objectness_support_thresh", type=float, default=0.25)
    parser.add_argument("--proposal_soft_affinity_weight", type=float, default=0.0,
                        help="Optionally blend proposal support into the soft edge target")
    parser.add_argument("--clip_prototypes_per_class", type=int, default=4)
    parser.add_argument("--clip_proto_images", type=int, default=256)
    parser.add_argument("--dino_project_dim", type=int, default=64,
                        help="Random-project frozen DINO node features to this dim; <=0 disables")
    parser.add_argument("--clip_project_dim", type=int, default=64,
                        help="Random-project optional CLIP node features to this dim; <=0 disables")

    parser.add_argument("--max_train_images", type=int, default=500)
    parser.add_argument("--max_val_images", type=int, default=100)
    parser.add_argument("--max_edges_per_image", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--no_layer_norm", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--default_merge_threshold", type=float, default=0.55)
    parser.add_argument("--output_min_area", type=int, default=1000)
    parser.add_argument("--save_edge_cache", action="store_true",
                        help="Optionally save full edge tensors; can be hundreds of MB")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
