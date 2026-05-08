#!/usr/bin/env python3
"""DiffCut for Cityscapes: Diffusion features + recursive NCut + global clustering.

Extracts Stable Diffusion self-attention features, runs per-image recursive
normalized cuts, then clusters segment features globally via spherical k-means
to produce overclustered pseudo-labels.

Two modes:
  - standalone: DiffCut features only → k=100 global clusters
  - ensemble: DiffCut segments + DINOv2 features → weighted ensemble

Based on "DiffCut: Catalyzing Zero-Shot Semantic Segmentation" (NeurIPS 2024).

Usage:
    python mbps_pytorch/generate_diffcut_cityscapes_ablation.py \
        --cityscapes_root /data/cityscapes \
        --k 100 --seed 42 --device mps
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import eigh
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from mbps_pytorch.generate_clustering_ablation import (
    assign_clusters_cosine,
    compute_cluster_stats,
    find_feature_files,
    fit_spherical_kmeans,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUT_H, OUT_W = 512, 1024
SD_LATENT_H, SD_LATENT_W = 16, 32  # SD latent is 1/32 of 512×1024


class SDFeatureExtractor:
    """Extract self-attention features from Stable Diffusion UNet."""

    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        device: str = "mps",
    ) -> None:
        from diffusers import AutoPipelineForText2Image, DDIMScheduler

        self.device = torch.device(device)
        dtype = torch.float32 if device == "mps" else torch.float16

        logger.info("Loading SD model: %s on %s", model_name, device)
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.scheduler.set_timesteps(50)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self._features: Optional[torch.Tensor] = None
        logger.info("SD model loaded")

    def _register_hook(self) -> list:
        handles = []
        attn_block = (
            self.unet.down_blocks[-2]
            .attentions[-1]
            .transformer_blocks[-1]
            .attn1
        )

        def hook_fn(mod, inp, out):
            self._features = out.detach()

        handles.append(attn_block.register_forward_hook(hook_fn))
        return handles

    @torch.no_grad()
    def extract(
        self,
        image: torch.Tensor,
        step: int = 50,
    ) -> torch.Tensor:
        """Extract SD self-attention features.

        Args:
            image: (1, 3, H, W) tensor in [0, 1].
            step: diffusion timestep for noise injection.

        Returns:
            (1, N, C) features.
        """
        h, w = image.shape[2:]
        latent = self.vae.encode(2 * image - 1).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor

        timestep = self.pipe.scheduler.timesteps[
            min(step, len(self.pipe.scheduler.timesteps) - 1)
        ]
        noise = torch.randn_like(latent)
        noisy_latent = self.pipe.scheduler.add_noise(latent, noise, timestep)

        text_input = self.pipe.tokenizer(
            "", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        text_emb = self.text_encoder(text_input.input_ids)[0]

        handles = self._register_hook()
        self.unet(noisy_latent, timestep, encoder_hidden_states=text_emb)
        for h in handles:
            h.remove()

        feats = self._features
        self._features = None
        return feats


def recursive_ncut(
    affinity: np.ndarray,
    max_depth: int = 6,
    min_size: int = 20,
    tau: float = 0.02,
) -> np.ndarray:
    """Recursive normalized cut on an affinity matrix.

    Returns per-pixel segment IDs.
    """
    N = affinity.shape[0]
    segments = np.zeros(N, dtype=np.int32)
    next_id = [1]

    def _split(indices: np.ndarray, depth: int) -> None:
        if len(indices) < min_size * 2 or depth >= max_depth:
            segments[indices] = next_id[0]
            next_id[0] += 1
            return

        A = affinity[np.ix_(indices, indices)]
        D = np.diag(A.sum(axis=1) + 1e-10)
        L = D - A

        try:
            eigenvalues, eigenvectors = eigh(L, D, subset_by_index=[0, 1])
        except Exception:
            segments[indices] = next_id[0]
            next_id[0] += 1
            return

        fiedler = eigenvectors[:, 1]
        threshold = np.median(fiedler)
        mask_a = fiedler <= threshold
        mask_b = fiedler > threshold

        if mask_a.sum() < min_size or mask_b.sum() < min_size:
            segments[indices] = next_id[0]
            next_id[0] += 1
            return

        ncut_cost = eigenvalues[1]
        if ncut_cost > tau:
            segments[indices] = next_id[0]
            next_id[0] += 1
            return

        _split(indices[mask_a], depth + 1)
        _split(indices[mask_b], depth + 1)

    _split(np.arange(N), 0)
    return segments


def extract_and_segment_cityscapes(
    cityscapes_root: Path,
    split: str,
    device: str = "mps",
    step: int = 50,
    tau: float = 0.02,
    max_depth: int = 6,
    min_size: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Extract SD features and segment each Cityscapes image.

    Returns:
        seg_features: list of (n_segments, C) arrays per image
        seg_labels: list of (H_patch, W_patch) segment ID maps
        stems: list of image stem names
    """
    data_dir = cityscapes_root / "leftImg8bit" / split
    cache_dir = cityscapes_root / "sd_features_cityscapes" / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_dir.rglob("*_leftImg8bit.png"))
    logger.info(f"Found {len(image_paths)} {split} images")

    extractor = None
    existing_cache = {f.stem for f in cache_dir.rglob("*.npy")}

    need_extract = []
    for p in image_paths:
        stem = p.stem.replace("_leftImg8bit", "")
        if stem not in existing_cache:
            need_extract.append(p)

    if need_extract:
        logger.info(f"Extracting SD features for {len(need_extract)} images...")
        extractor = SDFeatureExtractor(device=device)

        for path in tqdm(need_extract, desc="SD features"):
            img = Image.open(path).convert("RGB")
            img_resized = img.resize((OUT_W, OUT_H), Image.BILINEAR)
            img_t = torch.tensor(
                np.array(img_resized).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            feats = extractor.extract(img_t, step=step)
            stem = path.stem.replace("_leftImg8bit", "")
            city = path.parent.name
            out_path = cache_dir / city
            out_path.mkdir(parents=True, exist_ok=True)
            np.save(
                str(out_path / f"{stem}.npy"),
                feats[0].cpu().float().numpy(),
            )

        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    logger.info(f"Running recursive NCut on {len(image_paths)} images...")
    all_seg_features = []
    all_seg_maps = []
    all_stems = []

    for path in tqdm(image_paths, desc="NCut segmentation"):
        stem = path.stem.replace("_leftImg8bit", "")
        city = path.parent.name
        feat_path = cache_dir / city / f"{stem}.npy"
        feats = np.load(str(feat_path))

        N, C = feats.shape
        h_patch = SD_LATENT_H
        w_patch = SD_LATENT_W

        if N != h_patch * w_patch:
            side = int(np.sqrt(N))
            h_patch = side
            w_patch = N // side

        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_normed = feats / norms

        affinity = feats_normed @ feats_normed.T
        affinity = np.clip(affinity, 0, None)

        seg_ids = recursive_ncut(
            affinity, max_depth=max_depth,
            min_size=min_size, tau=tau,
        )

        unique_segs = np.unique(seg_ids)
        seg_feats = []
        for sid in unique_segs:
            mask = seg_ids == sid
            seg_feats.append(feats_normed[mask].mean(axis=0))
        seg_feats = np.array(seg_feats)

        all_seg_features.append(seg_feats)
        all_seg_maps.append(seg_ids.reshape(h_patch, w_patch))
        all_stems.append(f"{city}/{stem}")

    return all_seg_features, all_seg_maps, all_stems


def global_cluster_and_assign(
    seg_features_list: List[np.ndarray],
    seg_maps_list: List[np.ndarray],
    stems: List[str],
    out_dir: Path,
    split: str,
    k: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Cluster segment features globally and write pseudo-label PNGs."""
    all_feats = np.concatenate(seg_features_list, axis=0)
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8
    all_feats = all_feats / norms
    logger.info(f"Global features: {all_feats.shape}")

    result = fit_spherical_kmeans(all_feats, k=k, seed=seed, refine_iters=20)
    centers = result["centers"]

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    for seg_feats, seg_map, stem in zip(seg_features_list, seg_maps_list, stems):
        n_segs = seg_feats.shape[0]
        feats_norm = seg_feats / (np.linalg.norm(seg_feats, axis=1, keepdims=True) + 1e-8)
        sim = feats_norm @ centers.T
        cluster_ids = sim.argmax(axis=1)

        unique_segs = np.unique(seg_map)
        label_map = np.zeros_like(seg_map, dtype=np.uint8)
        for i, sid in enumerate(unique_segs):
            label_map[seg_map == sid] = cluster_ids[i]

        label_upsampled = np.array(
            Image.fromarray(label_map).resize(
                (OUT_W, OUT_H), Image.NEAREST
            )
        )

        city = stem.split("/")[0] if "/" in stem else ""
        img_stem = stem.split("/")[-1]
        if city:
            (split_dir / city).mkdir(parents=True, exist_ok=True)
            out_path = split_dir / city / f"{img_stem}.png"
        else:
            out_path = split_dir / f"{img_stem}.png"
        Image.fromarray(label_upsampled).save(str(out_path))
        offset += n_segs

    return centers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffCut pseudo-label generation for Cityscapes"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_size", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    root = Path(args.cityscapes_root)
    out_dir = root / f"pseudo_semantic_raw_dinov3_k{args.k}_diffcut_vitl16"

    t0 = time.time()

    for split in args.splits:
        logger.info(f"Processing {split}...")
        seg_feats, seg_maps, stems = extract_and_segment_cityscapes(
            root, split,
            device=args.device,
            step=args.step,
            tau=args.tau,
            max_depth=args.max_depth,
            min_size=args.min_size,
        )

        centers = global_cluster_and_assign(
            seg_feats, seg_maps, stems,
            out_dir, split,
            k=args.k, seed=args.seed,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "centroids.npz"), centers=centers)

    fit_time = time.time() - t0
    stats = {
        "method": "diffcut",
        "k": args.k,
        "step": args.step,
        "tau": args.tau,
        "max_depth": args.max_depth,
        "min_size": args.min_size,
        "fit_time_seconds": fit_time,
    }
    with open(str(out_dir / "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"All done -> {out_dir} ({fit_time:.1f}s)")


if __name__ == "__main__":
    main()
