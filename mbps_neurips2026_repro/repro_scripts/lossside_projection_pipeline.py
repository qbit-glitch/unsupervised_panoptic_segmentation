"""Option 1: Loss-side ablation.

Trains a learnable projection P: R^90 -> R^90 on cached CAUSE-TR codes with
ONLY DepthG's depth-correlation loss (no preservation, no residual). This
isolates the conditioning *mechanism* relative to DCFA at matched parameter
budget and identical downstream pipeline.

Architecture: P(z) = MLP(90 -> 384 -> 90), ~40K params (matches DCFA).
Loss: depth_guided_correlation_loss(P(z), depth) ONLY.
Forward: z' = P(z)  (no depth input to the architecture).

Then: project all train codes -> k-means k=80 -> CUPS-format pseudo-labels
re-using DepthPro depth-CC instances -> evaluate pseudo-label PQ.
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mbps_pytorch"))

from mbps_pytorch.train_depth_adapter import PreextractedCodesDataset, set_seed
from mbps_pytorch.models.semantic.stego_loss import depth_guided_correlation_loss

CODES_DIR = CR / "cause_codes_90d"
RAW_INPUT_DIR = CR / "cups_pseudo_labels_adapter_V3_tau020"
import os as _os
OUT_DIR = PROJECT_ROOT / _os.environ.get("LOSSSIDE_OUT", "results/lossside_projection")
PSEUDO_DIR = OUT_DIR / "pseudo_labels"
KMEANS_PATH = OUT_DIR / "kmeans_k80.npz"
CKPT_PATH = OUT_DIR / "lossside_projection.pt"

CODE_DIM = 90
HIDDEN = 384
K = 80
TRAIN_STEPS = 10_000
LR = 1e-3
BATCH_SIZE = 32
NUM_PAIRS = 1024
SIGMA_D = 0.5


class LossSideProjection(nn.Module):
    """Learnable projection P: R^90 -> R^90, 2-layer MLP with hidden 384.
    Same parameter count as DCFA (~40K) but with NO depth input."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE_DIM, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, CODE_DIM),
        )
        # Zero-init final layer so initial output ~= 0; then add identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        # Identity + small learnable shift (mirrors DCFA's residual structure
        # but without depth as an input — depth enters only through the loss).
        return z + self.net(z)


def train_projection(device, num_workers=0, num_steps=TRAIN_STEPS):
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training LossSideProjection on cached CAUSE codes (no depth input, no preserve loss)")
    print(f"  device={device}  steps={num_steps}  lr={LR}  batch={BATCH_SIZE}  pairs={NUM_PAIRS}")

    dataset = PreextractedCodesDataset(str(CODES_DIR), split="train")
    print(f"  dataset: {len(dataset)} images")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=False, drop_last=True)

    model = LossSideProjection().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params/1000:.1f}K params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()

    step = 0
    t0 = time.time()
    pbar = tqdm(total=num_steps, desc="train")
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break
            codes = batch["codes"].to(device)  # (B, ph, pw, 90)
            depth = batch["depth"].to(device)  # (B, ph, pw)

            adjusted = model(codes)
            l_corr = depth_guided_correlation_loss(
                adjusted, depth, sigma_d=SIGMA_D, num_pairs=NUM_PAIRS
            )
            lambda_p = float(_os.environ.get("LOSSSIDE_LAMBDA_P", "0.0"))
            if lambda_p > 0:
                l_preserve = F.mse_loss(adjusted, codes)
                loss = l_corr + lambda_p * l_preserve
            else:
                loss = l_corr  # NO preservation term

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            pbar.update(1)
            if step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    print(f"  training done in {(time.time()-t0)/60:.1f} min")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"  saved {CKPT_PATH}")
    return model


def project_all_codes(model, device):
    """Project every cached CAUSE code through P, save to OUT_DIR/projected/<split>/<city>/<stem>.npy"""
    proj_dir = OUT_DIR / "projected/train"
    proj_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    files = sorted((CODES_DIR / "train").rglob("*_codes.npy"))
    print(f"\nProjecting {len(files)} code maps -> {proj_dir}")
    for f in tqdm(files, desc="project"):
        stem = f.stem.replace("_codes", "")
        city = f.parent.name
        out = proj_dir / city / f"{stem}.npy"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        codes = np.load(f).astype(np.float32)  # (ph, pw, 90)
        with torch.no_grad():
            t = torch.from_numpy(codes).unsqueeze(0).to(device)
            adjusted = model(t).squeeze(0).cpu().numpy()
        np.save(out, adjusted.astype(np.float32))


def fit_kmeans(samples_per_image=400):
    print(f"\nFitting MiniBatchKMeans k={K}...")
    files = sorted((OUT_DIR / "projected/train").rglob("*.npy"))
    rng = np.random.default_rng(42)
    samples = []
    for f in tqdm(files, desc="sample"):
        z = np.load(f).reshape(-1, CODE_DIM)
        idx = rng.choice(len(z), size=min(samples_per_image, len(z)), replace=False)
        samples.append(z[idx])
    X = np.vstack(samples)
    print(f"  total samples: {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, max_iter=300, n_init=3)
    km.fit(X)
    np.savez(KMEANS_PATH, cluster_centers=km.cluster_centers_, n_clusters=K, dim=CODE_DIM)
    print(f"  saved {KMEANS_PATH}")
    return km


def assign_centroid(z, centroids):
    H, W, D = z.shape
    X = z.reshape(-1, D)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = X @ C.T
    return sim.argmax(axis=1).astype(np.uint8).reshape(H, W)


def assemble_pseudolabels(centroids):
    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((OUT_DIR / "projected/train").rglob("*.npy"))
    print(f"\nAssembling CUPS-format pseudo-labels for {len(files)} images")
    for f in tqdm(files, desc="assemble"):
        stem = f.stem
        z = np.load(f)
        sem_lo = assign_centroid(z, centroids)
        # raw inst from cached DepthPro depth-CC
        raw_inst_path = RAW_INPUT_DIR / f"{stem}_leftImg8bit_instance.png"
        raw_sem_path = RAW_INPUT_DIR / f"{stem}_leftImg8bit_semantic.png"
        if not raw_inst_path.exists():
            continue
        raw_inst = np.array(Image.open(raw_inst_path))
        raw_sem = np.array(Image.open(raw_sem_path))
        H, W = raw_sem.shape[:2]
        sem_hi = np.array(Image.fromarray(sem_lo).resize((W, H), Image.NEAREST))
        Image.fromarray(sem_hi.astype(np.uint8)).save(PSEUDO_DIR / f"{stem}_leftImg8bit_semantic.png")
        Image.fromarray(raw_inst.astype(np.uint16)).save(PSEUDO_DIR / f"{stem}_leftImg8bit_instance.png")
        # distributions .pt
        inst_ids = np.unique(raw_inst)
        inst_ids = inst_ids[inst_ids > 0]
        dist = {}
        for k in inst_ids:
            mask = raw_inst == k
            classes, counts = np.unique(sem_hi[mask], return_counts=True)
            dist[int(k)] = dict(zip(classes.tolist(), counts.tolist()))
        torch.save(dist, PSEUDO_DIR / f"{stem}_leftImg8bit.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "project", "kmeans", "assemble", "all"], default="all")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.stage in ("train", "all"):
        model = train_projection(device, num_workers=args.num_workers, num_steps=args.steps)
    else:
        model = LossSideProjection().to(device)
        if CKPT_PATH.exists():
            model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            print(f"Loaded {CKPT_PATH}")

    if args.stage in ("project", "all"):
        project_all_codes(model, device)

    if args.stage in ("kmeans", "all"):
        fit_kmeans()

    if args.stage in ("assemble", "all"):
        data = np.load(KMEANS_PATH)
        assemble_pseudolabels(data["cluster_centers"])

    print("\nDone. Evaluate with:")
    print(f"  python3 scripts/evaluate_pseudolabel_quality.py \\")
    print(f"    --pseudo_dir {PSEUDO_DIR} \\")
    print(f"    --cityscapes_root {CR} \\")
    print(f"    --split train --num_clusters {K} --use_hungarian \\")
    print(f"    --output {OUT_DIR}/eval_train.json")


if __name__ == "__main__":
    main()
