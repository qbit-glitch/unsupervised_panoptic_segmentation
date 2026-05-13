"""DepthG drop-in baseline: extract features + cluster + assemble CUPS-format pseudo-labels.

Pipeline:
  1. Load DepthG ViT-B/8 checkpoint (cityscapes_vitb.ckpt)
  2. Run feature extraction on Cityscapes train images at 320x320 (their training res)
       -> 100-D feature maps at patch resolution
  3. Fit MiniBatchKMeans k=80 on a subsample of features
  4. Assign every pixel to nearest centroid -> 80-class semantic map
  5. Re-use our DepthPro depth-CC instance candidates (cups_pseudo_labels_adapter_V3_tau020)
  6. Save CUPS-format outputs: <stem>_semantic.png, <stem>_instance.png, <stem>.pt

Then evaluate with: scripts/evaluate_pseudolabel_quality.py
"""

from pathlib import Path
import sys
import argparse
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
sys.path.insert(0, str(PROJECT_ROOT / "refs/depthg/src"))
sys.path.insert(0, str(PROJECT_ROOT / "refs/depthg/src/dino"))

CKPT = PROJECT_ROOT / "checkpoints/depthg_official/cityscapes_vitb.ckpt"
RAW_INPUT_DIR = CR / "cups_pseudo_labels_adapter_V3_tau020"  # for instance candidates
OUTPUT_DIR = PROJECT_ROOT / "results/depthg_dropin/pseudo_labels"
FEATURE_CACHE = PROJECT_ROOT / "results/depthg_dropin/features"
KMEANS_PATH = PROJECT_ROOT / "results/depthg_dropin/kmeans_k80.npz"

DEPTHG_DIM = 100
K = 80
RES = 320  # DepthG training resolution
BATCH_SIZE = 1


def build_depthg_model(device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "dino_patch_size": 8,
        "dino_feat_type": "feat",
        "model_type": "vit_base",
        "pretrained_weights": None,
        "projection_type": "nonlinear",
        "dim": DEPTHG_DIM,
        "dropout": False,
        "continuous": True,
    })
    from modules import DinoFeaturizer
    model = DinoFeaturizer(dim=DEPTHG_DIM, cfg=cfg)
    # Load DepthG-trained weights
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Filter to net.* and strip prefix
    net_sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    msg = model.load_state_dict(net_sd, strict=False)
    print(f"  Loaded DepthG weights: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")
    model.eval().to(device)
    return model


def extract_features(model, image_path, device):
    """Extract 100-D DepthG features for one image. Returns (H/8, W/8, 100) numpy."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((RES, RES), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize per ImageNet stats (DINO convention)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        # forward returns (image_feat, code) tuple
        if isinstance(out, tuple):
            code = out[-1]  # (1, 100, H/8, W/8)
        else:
            code = out
    code_np = code.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return code_np  # (H/8, W/8, 100) at 40x40 = 1600 patches


def stage1_extract_features(stems, device, limit=None):
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    model = build_depthg_model(device)
    print(f"Extracting DepthG features for {len(stems)} images at {RES}x{RES}...")
    if limit:
        stems = stems[:limit]
    t0 = time.time()
    for i, stem in enumerate(tqdm(stems, desc="extract")):
        out = FEATURE_CACHE / f"{stem}.npy"
        if out.exists():
            continue
        city = stem.split("_")[0]
        rgb_path = CR / "leftImg8bit/train" / city / f"{stem}_leftImg8bit.png"
        feat = extract_features(model, rgb_path, device)
        np.save(out, feat)
    print(f"  feature extraction: {(time.time()-t0)/60:.1f} min")


def stage2_kmeans(stems, n_features_per_image=400):
    """Fit MiniBatchKMeans k=80 on a subsample of features."""
    print(f"\nFitting MiniBatchKMeans k={K} on {len(stems)} images x {n_features_per_image} samples each...")
    rng = np.random.default_rng(42)
    samples = []
    for stem in tqdm(stems, desc="sample"):
        feat = np.load(FEATURE_CACHE / f"{stem}.npy")
        feat = feat.reshape(-1, DEPTHG_DIM)
        idx = rng.choice(len(feat), size=min(n_features_per_image, len(feat)), replace=False)
        samples.append(feat[idx])
    X = np.vstack(samples)
    print(f"  total samples: {X.shape}")
    # L2-normalize codes (matching our pipeline)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, max_iter=300, n_init=3, verbose=0)
    km.fit(X)
    KMEANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(KMEANS_PATH, cluster_centers=km.cluster_centers_, n_clusters=K, dim=DEPTHG_DIM)
    print(f"  saved centroids: {KMEANS_PATH}")
    return km


def assign_centroid(feat_map, centroids):
    """feat_map: (H, W, D), centroids: (K, D). Returns label map (H, W) with values 0..K-1."""
    H, W, D = feat_map.shape
    X = feat_map.reshape(-1, D)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    # cosine == dot if both normalized; centroids assumed unit-norm
    centroids_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = X @ centroids_n.T  # (HW, K)
    labels = sim.argmax(axis=1).astype(np.uint8).reshape(H, W)
    return labels


def upsample_label(label, target_hw):
    """Upsample H/8 x W/8 label map to target H x W via nearest neighbor."""
    pil = Image.fromarray(label, mode="L")
    return np.array(pil.resize((target_hw[1], target_hw[0]), Image.NEAREST))


def stage3_assemble_pseudolabels(stems, centroids):
    """For each image: load DepthG features -> assign cluster -> upsample -> reuse instance from raw cache."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nAssembling CUPS-format pseudo-labels for {len(stems)} images -> {OUTPUT_DIR}")
    for stem in tqdm(stems, desc="assemble"):
        feat = np.load(FEATURE_CACHE / f"{stem}.npy")
        sem_lo = assign_centroid(feat, centroids)  # (40, 40)
        # Upsample to standard CUPS PNG resolution: 512x1024 (or whatever raw dir uses)
        raw_sem_path = RAW_INPUT_DIR / f"{stem}_leftImg8bit_semantic.png"
        if not raw_sem_path.exists():
            continue
        raw_sem = np.array(Image.open(raw_sem_path))
        H, W = raw_sem.shape[:2]
        sem_hi = upsample_label(sem_lo, (H, W))
        # Re-use the raw instance (DepthPro depth-CC, τ=0.20)
        raw_inst_path = RAW_INPUT_DIR / f"{stem}_leftImg8bit_instance.png"
        raw_inst = np.array(Image.open(raw_inst_path))
        # Save CUPS format
        out_sem = OUTPUT_DIR / f"{stem}_leftImg8bit_semantic.png"
        out_inst = OUTPUT_DIR / f"{stem}_leftImg8bit_instance.png"
        Image.fromarray(sem_hi.astype(np.uint8)).save(out_sem)
        Image.fromarray(raw_inst.astype(np.uint16)).save(out_inst)
        # Compute distributions .pt for compatibility (mirrors refine_simcf format)
        from collections import Counter
        # Per-instance class distribution
        inst_ids = np.unique(raw_inst)
        inst_ids = inst_ids[inst_ids > 0]
        dist = {}
        for k in inst_ids:
            mask = raw_inst == k
            classes, counts = np.unique(sem_hi[mask], return_counts=True)
            dist[int(k)] = dict(zip(classes.tolist(), counts.tolist()))
        torch.save(dist, OUTPUT_DIR / f"{stem}_leftImg8bit.pt")


def list_cityscapes_train_stems():
    stems = []
    for f in sorted((CR / "leftImg8bit/train").rglob("*_leftImg8bit.png")):
        stems.append(f.stem.replace("_leftImg8bit", ""))
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "kmeans", "assemble", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="Limit images (smoke test)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    stems = list_cityscapes_train_stems()
    print(f"Found {len(stems)} train images")

    if args.stage in ("features", "all"):
        device = torch.device(args.device)
        stage1_extract_features(stems, device, limit=args.limit)

    if args.stage in ("kmeans", "all"):
        sub = stems[:args.limit] if args.limit else stems
        sub = [s for s in sub if (FEATURE_CACHE / f"{s}.npy").exists()]
        stage2_kmeans(sub)

    if args.stage in ("assemble", "all"):
        data = np.load(KMEANS_PATH)
        centroids = data["cluster_centers"]
        sub = stems[:args.limit] if args.limit else stems
        sub = [s for s in sub if (FEATURE_CACHE / f"{s}.npy").exists()]
        stage3_assemble_pseudolabels(sub, centroids)

    print("\nDone. Output dir:", OUTPUT_DIR)
    print("Evaluate with:")
    print(f"  python3 scripts/evaluate_pseudolabel_quality.py \\")
    print(f"    --pseudo_dir {OUTPUT_DIR} \\")
    print(f"    --cityscapes_root {CR} \\")
    print(f"    --centroids_path {KMEANS_PATH} \\")
    print(f"    --split train --use_hungarian \\")
    print(f"    --output results/depthg_dropin/eval_train.json")


if __name__ == "__main__":
    main()
