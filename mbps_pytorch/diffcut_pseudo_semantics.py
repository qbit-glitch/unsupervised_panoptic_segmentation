#!/usr/bin/env python3
"""DiffCut unsupervised semantic segmentation for COCO-Stuff-27.

Uses Stable Diffusion self-attention features + recursive Normalized Cut
for zero-shot unsupervised segmentation. No training required.

Pipeline:
  1. Extract SD self-attention features (cached to disk)
  2. Recursive NCut per image → variable K segments
  3. Optional PAMR post-processing
  4. Per-image Hungarian matching → mIoU evaluation
  5. Global DINOv3 clustering → pseudo-label generation

Usage:
    # Phase 1: Extract SD features (one-time)
    python diffcut_pseudo_semantics.py --phase extract \
        --coco_root /path/to/coco --device mps

    # Phase 2: Segment + evaluate
    python diffcut_pseudo_semantics.py --phase segment \
        --coco_root /path/to/coco --tau 0.5 --alpha 10 --pamr

    # Phase 3: Global clustering for pseudo-labels
    python diffcut_pseudo_semantics.py --phase cluster \
        --coco_root /path/to/coco --tau 0.5 --alpha 10 --pamr --K_global 27

    # All phases at once
    python diffcut_pseudo_semantics.py --phase all \
        --coco_root /path/to/coco --tau 0.5 --alpha 10 --pamr --K_global 27

References:
    DiffCut (NeurIPS 2024): arXiv 2406.02842
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── COCO-Stuff-27 constants ───

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


# ═══════════════════════════════════════════════════════════════════════
# SD Feature Extraction (adapted from DiffCut tools/ldm.py)
# ═══════════════════════════════════════════════════════════════════════

class SDFeatureExtractor:
    """Extract self-attention features from Stable Diffusion UNet.

    Adapted from DiffCut's LdmExtractor for MPS/CPU compatibility.
    """

    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        device: str = "mps",
    ) -> None:
        from diffusers import AutoPipelineForText2Image, DDIMScheduler

        self.device = torch.device(device)
        dtype = torch.float32 if device == "mps" else torch.float16

        logger.info("Loading SD model: %s on %s (dtype=%s)", model_name, device, dtype)
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

    def _register_hook(self) -> List:
        """Hook into the self-attention block to capture features."""
        handles = []
        # SD-1.4: down_blocks[-2].attentions[-1].transformer_blocks[-1].attn1
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
        img_size: int = 512,
    ) -> torch.Tensor:
        """Extract SD self-attention features for a single image.

        Args:
            image: (1, 3, H, W) tensor in [0, 1].
            step: Diffusion timestep for noise injection.
            img_size: Resize image to this size before encoding.

        Returns:
            Features tensor of shape (1, N, C) where N = (img_size/32)^2.
        """
        image = F.interpolate(image, size=(img_size, img_size), mode="bilinear")

        # Encode to latent
        latent = self.vae.encode(2 * image - 1).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor

        # Add noise at timestep
        rng = torch.Generator(device=self.device).manual_seed(42)
        noise = torch.randn(
            1, 4, img_size // 8, img_size // 8,
            generator=rng, device=self.device,
        )
        t = torch.tensor([step], device=self.device)
        noisy_latent = self.pipe.scheduler.add_noise(latent, noise, t)

        # Get text embeddings (empty prompt)
        prompt_embeds = self.pipe.encode_prompt(
            [""], device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        if isinstance(prompt_embeds, tuple):
            prompt_embeds = prompt_embeds[0]

        # Forward pass through UNet with hook
        handles = self._register_hook()
        noisy_latent = self.pipe.scheduler.scale_model_input(noisy_latent, t)

        if self.device.type == "mps":
            # MPS: use float32, no autocast
            self.unet(noisy_latent, t, encoder_hidden_states=prompt_embeds)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.unet(noisy_latent, t, encoder_hidden_states=prompt_embeds)

        for h in handles:
            h.remove()

        return self._features


# ═══════════════════════════════════════════════════════════════════════
# Recursive Normalized Cut (adapted from DiffCut)
# ═══════════════════════════════════════════════════════════════════════

class RecursiveNCut:
    """Recursive Normalized Cut for unsupervised segmentation.

    Adapted from DiffCut/diffcut/recursive_normalized_cut.py for
    device-agnostic operation (MPS/CUDA/CPU).
    """

    def __init__(self, device: str = "mps") -> None:
        self.device = torch.device(device)

    def _get_degree_matrix(self, A: torch.Tensor) -> torch.Tensor:
        return torch.diag(A.sum(dim=1))

    def _second_smallest_eigenvector(
        self, A: torch.Tensor, D: torch.Tensor
    ) -> Optional[torch.Tensor]:
        D_inv_sqrt = torch.diag(1.0 / torch.diag(D).sqrt())
        try:
            L_sym = D_inv_sqrt @ (D - A) @ D_inv_sqrt
            # eigh on MPS may fail — fall back to CPU
            if L_sym.device.type == "mps":
                _, eigvecs = torch.linalg.eigh(L_sym.cpu())
                eigvecs = eigvecs.to(self.device)
            else:
                _, eigvecs = torch.linalg.eigh(L_sym)
            return D_inv_sqrt @ eigvecs[:, 1]
        except Exception:
            return None

    def _get_bipartition(
        self, y: torch.Tensor, A: torch.Tensor, D: torch.Tensor, k: int = 100
    ) -> Tuple[torch.Tensor, float]:
        thresholds = torch.linspace(
            y.min().item() * 0.99, y.max().item() * 0.99, k
        )
        L = D - A
        sum_D = D.sum()
        diag_D = torch.diag(D)

        best_ncut = float("inf")
        best_partition = None

        for thresh in thresholds:
            x = (y > thresh).float()
            x_signed = 2 * x - 1
            k_val = (diag_D * (x_signed > 0)).sum() / sum_D
            b = k_val / (1 - k_val + 1e-10)
            y_val = (1 + x_signed) - b * (1 - x_signed)
            E = (y_val @ L @ y_val) / (y_val @ D @ y_val + 1e-10)
            ncut_val = E.item()
            if ncut_val < best_ncut:
                best_ncut = ncut_val
                best_partition = x

        return best_partition, best_ncut

    def _deterministic_sign_flip(self, u: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(torch.abs(u))
        u = u * torch.sign(u[idx])
        return u

    def recursive_ncut(
        self,
        feats: torch.Tensor,
        tau: float = 0.5,
        dims: Tuple[int, int] = (32, 32),
        painting: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        accumulator: Optional[List[torch.Tensor]] = None,
        level: int = 0,
        alpha: float = 10.0,
    ) -> List[torch.Tensor]:
        """Run recursive NCut on features.

        Args:
            feats: At level 0, (C, N) normalized features. At level > 0, affinity matrix.
            tau: NCut threshold — lower = more segments.
            dims: Spatial dimensions of the feature map.
            alpha: Exponent for affinity sharpening.

        Returns:
            List of binary segment masks, each (H, W).
        """
        dev = self.device

        if level == 0:
            accumulator = []
            feats = F.normalize(feats, p=2, dim=0)
            painting = torch.zeros(dims, device=dev)
            mask = torch.zeros(dims, device=dev)
            affinity = feats.T @ feats
            affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min() + 1e-10)
            affinity = affinity ** alpha
        else:
            affinity = feats

        # Mask explored areas
        painting = painting + mask.unsqueeze(0)
        painting = painting.clamp(0, 1)
        mask_affinity = painting.reshape(1, -1).T @ painting.reshape(1, -1)
        affinity = affinity * (1 - mask_affinity)

        # Build smaller affinity matrix (remove null rows)
        size = affinity.shape[0]
        null_idx = set(torch.unique(torch.where(affinity == 0)[0]).tolist())
        all_idx = set(range(size))

        if level == 0:
            idx2keep = list(all_idx)
            A = affinity
        else:
            idx2keep = sorted(all_idx - null_idx)
            if len(idx2keep) < 2:
                return accumulator
            A = affinity[:, idx2keep][idx2keep, :]

        D = self._get_degree_matrix(A)

        if A.shape[0] <= 1:
            return accumulator

        vec = self._second_smallest_eigenvector(A, D)
        if vec is None:
            return accumulator

        vec = self._deterministic_sign_flip(vec)
        bipartition, ncut = self._get_bipartition(vec, A, D)

        # Map bipartition back to full dimensions
        full_partition = torch.zeros(dims[0] * dims[1], device=dev)
        full_partition[idx2keep] = bipartition
        full_partition = full_partition.reshape(dims)

        if ncut < tau:
            accumulator.append(full_partition)
            self.recursive_ncut(
                affinity, tau, dims, painting.clone(),
                1 - full_partition, accumulator, level + 1
            )
            self.recursive_ncut(
                affinity, tau, dims, painting.clone(),
                full_partition, accumulator, level + 1
            )

        return accumulator

    def assemble_clusters(
        self, clusters: List[torch.Tensor], h: int, w: int
    ) -> torch.Tensor:
        """Merge binary masks into a single labeled map."""
        if not clusters:
            return torch.zeros((h, w), device=self.device)

        mask = torch.zeros((h, w), device=self.device)
        max_val = 1.0
        for cm in clusters:
            mask += max_val * cm
            max_val = mask.max().item() + 1

        final = torch.zeros((h, w), device=self.device)
        for i, uid in enumerate(torch.unique(mask)):
            final[mask == uid] = i
        return final

    def generate_masks(
        self,
        features: torch.Tensor,
        tau: float = 0.5,
        alpha: float = 10.0,
        mask_size: Tuple[int, int] = (128, 128),
        feat_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Full DiffCut pipeline: NCut → assemble → upsample.

        Args:
            features: (1, N, C) SD self-attention features.
            tau: NCut threshold.
            alpha: Affinity exponent.
            mask_size: Output resolution for segmentation.
            feat_hw: Feature map spatial dims. If None, inferred as sqrt(N).

        Returns:
            (segmentation_map, n_segments): map is (1, 1, H, W) int array.
        """
        _, n_tokens, c = features.shape
        if feat_hw is None:
            h = w = int(math.sqrt(n_tokens))
        else:
            h, w = feat_hw

        feats = features[0].T  # (C, N)
        feats_4d = features.permute(0, 2, 1).reshape(1, c, h, w)
        feats_norm = F.normalize(feats_4d, dim=1)

        # Recursive NCut
        x = F.normalize(feats, p=2, dim=0)  # (C, h*w)
        clusters = self.recursive_ncut(x, tau=tau, dims=(h, w), alpha=alpha)

        if not clusters:
            clusters.append(torch.zeros((h, w), device=self.device))

        masks = self.assemble_clusters(clusters, h, w)[None, None]  # (1,1,h,w)
        n_segments = int(masks.unique().numel())

        # Mean-pool features per cluster, upsample, cosine assignment
        n_clusters = n_segments
        cluster_embeds = torch.zeros(n_clusters, c, device=self.device)
        for k, uid in enumerate(torch.unique(masks)):
            m = (masks == uid).float()
            denom = m.sum() + 1e-8
            pooled = (feats_norm * m).sum(dim=(2, 3)) / denom  # (1, C)
            cluster_embeds[k] = pooled[0]

        up_feats = F.interpolate(feats_norm, size=mask_size, mode="bilinear")
        _, c2, hm, wm = up_feats.shape
        sim = up_feats[0].reshape(c2, -1).T @ cluster_embeds.T  # (H*W, K)
        seg = sim.argmax(dim=1).reshape(1, 1, hm, wm)

        return seg.cpu().numpy().astype(int), n_segments


# ═══════════════════════════════════════════════════════════════════════
# PAMR Post-Processing (adapted from DiffCut tools/pamr.py)
# ═══════════════════════════════════════════════════════════════════════

class LocalAffinity(torch.nn.Module):
    """Local shift kernel for PAMR."""

    def __init__(self, dilations: List[int], device: str = "mps") -> None:
        super().__init__()
        self.dilations = dilations
        weight = torch.zeros(8, 1, 3, 3, device=device)
        for i in range(8):
            weight[i, 0, 1, 1] = 1
        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1
        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1
        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1
        self.register_buffer("kernel", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, H, W = x.shape
        x = x.view(B * K, 1, H, W)
        affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode="replicate")
            affs.append(F.conv2d(x_pad, self.kernel, dilation=d))
        return torch.cat(affs, 1).view(B, K, -1, H, W)


class LocalAffinityCopy(torch.nn.Module):
    """Copy-shift kernel for PAMR."""

    def __init__(self, dilations: List[int], device: str = "mps") -> None:
        super().__init__()
        self.dilations = dilations
        weight = torch.zeros(8, 1, 3, 3, device=device)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1
        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        self.register_buffer("kernel", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, H, W = x.shape
        x = x.view(B * K, 1, H, W)
        affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode="replicate")
            affs.append(F.conv2d(x_pad, self.kernel, dilation=d))
        return torch.cat(affs, 1).view(B, K, -1, H, W)


class LocalStDev(LocalAffinity):
    """Local standard deviation for PAMR."""

    def __init__(self, dilations: List[int], device: str = "mps") -> None:
        super().__init__(dilations, device)
        weight = torch.zeros(9, 1, 3, 3, device=device)
        for i in range(9):
            weight[i, 0, i // 3, i % 3] = 1
        self.kernel = weight
        self.register_buffer("kernel_std", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, H, W = x.shape
        x = x.view(B * K, 1, H, W)
        affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode="replicate")
            affs.append(F.conv2d(x_pad, self.kernel_std, dilation=d))
        out = torch.cat(affs, 1).view(B, K, -1, H, W)
        return out.std(2, keepdim=True)


def pamr_refine(
    image: torch.Tensor,
    labels: torch.Tensor,
    num_iter: int = 10,
    dilations: List[int] = [1, 2, 4, 8],
    device: str = "mps",
) -> np.ndarray:
    """PAMR mask refinement.

    Args:
        image: (1, 3, H, W) in [0, 1].
        labels: (1, 1, H, W) integer segment labels.
        num_iter: Number of PAMR iterations.
        dilations: Dilation values for affinity kernels.
        device: Target device.

    Returns:
        Refined labels as (1, 1, H, W) int numpy array.
    """
    dev = torch.device(device)
    h, w = image.shape[2:]

    # Resize image to label resolution
    lh, lw = labels.shape[2:]
    img_resized = F.interpolate(image, size=(lh, lw), mode="bilinear").to(dev)

    # Create one-hot masks
    unique_labels = torch.unique(labels)
    n_cls = len(unique_labels)
    masks = torch.cat(
        [(labels == uid).float() for uid in unique_labels], dim=1
    ).to(dev)  # (1, K, H, W)

    # PAMR modules
    aff_x = LocalAffinity(dilations, device=str(dev))
    aff_m = LocalAffinityCopy(dilations, device=str(dev))
    aff_std = LocalStDev(dilations, device=str(dev))

    # Compute image affinity
    x_std = aff_std(img_resized)
    x_aff = -aff_x(img_resized) / (1e-8 + 0.1 * x_std)
    x_aff = x_aff.mean(1, keepdim=True)
    x_aff = F.softmax(x_aff, 2)

    # Iterative refinement
    for _ in range(num_iter):
        m = aff_m(masks)
        masks = (m * x_aff).sum(2)

    refined = masks.argmax(dim=1).cpu().numpy().astype(int)
    # Map back to original label IDs
    label_map = {i: uid.item() for i, uid in enumerate(unique_labels)}
    result = np.vectorize(label_map.get)(refined)

    # Median filter for smoothness
    result = median_filter(result, size=3).astype(int)
    return result[None]  # (1, H, W)


# ═══════════════════════════════════════════════════════════════════════
# GT Loading and Evaluation
# ═══════════════════════════════════════════════════════════════════════

def load_coco_panoptic_gt(coco_root: str, image_id: int) -> Optional[np.ndarray]:
    """Load COCO panoptic GT → 27-class semantic label map."""
    panoptic_json = Path(coco_root) / "annotations" / "panoptic_val2017.json"
    panoptic_dir = Path(coco_root) / "annotations" / "panoptic_val2017"

    if not hasattr(load_coco_panoptic_gt, "_cache"):
        with open(panoptic_json) as f:
            data = json.load(f)
        cat_map = {c["id"]: c["supercategory"] for c in data["categories"]}
        ann_map = {a["image_id"]: a for a in data["annotations"]}
        load_coco_panoptic_gt._cache = (cat_map, ann_map, str(panoptic_dir))

    cat_map, ann_map, pdir = load_coco_panoptic_gt._cache
    if image_id not in ann_map:
        return None

    ann = ann_map[image_id]
    pan_img = np.array(Image.open(Path(pdir) / ann["file_name"]))
    pan_id = (
        pan_img[:, :, 0].astype(np.int32)
        + pan_img[:, :, 1].astype(np.int32) * 256
        + pan_img[:, :, 2].astype(np.int32) * 256 * 256
    )

    sem_label = np.full(pan_id.shape, 255, dtype=np.uint8)
    for seg in ann["segments_info"]:
        supercat = cat_map.get(seg["category_id"])
        if supercat and supercat in SUPERCATEGORY_TO_COARSE:
            sem_label[pan_id == seg["id"]] = SUPERCATEGORY_TO_COARSE[supercat]
    return sem_label


def hungarian_miou(
    pred: np.ndarray, gt: np.ndarray, n_class: int = 27
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Per-image Hungarian matching and IoU computation.

    Args:
        pred: (H, W) predicted segment labels (arbitrary IDs).
        gt: (H, W) ground truth labels (0-26, 255=ignore).

    Returns:
        (miou, per_class_iou, mapping): mIoU percentage, per-class IoU, cluster→class map.
    """
    valid = gt != 255
    pred_v = pred[valid]
    gt_v = gt[valid]

    n_pred = int(pred_v.max()) + 1
    cost = np.zeros((n_pred, n_class))
    for p in range(n_pred):
        for c in range(n_class):
            cost[p, c] = ((pred_v == p) & (gt_v == c)).sum()

    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > 0:
            mapping[r] = c

    # Map predictions
    mapped = np.full_like(pred, 255)
    for p_id, c_id in mapping.items():
        mapped[pred == p_id] = c_id
    # Unmatched segments → best overlap class
    for p_id in range(n_pred):
        if p_id not in mapping:
            overlaps = cost[p_id]
            if overlaps.sum() > 0:
                mapped[pred == p_id] = overlaps.argmax()

    # Compute IoU
    per_class_iou = np.zeros(n_class)
    class_present = np.zeros(n_class, dtype=bool)
    for c in range(n_class):
        gt_mask = gt == c
        pred_mask = mapped == c
        inter = (gt_mask & pred_mask & valid).sum()
        union = ((gt_mask | pred_mask) & valid).sum()
        if union > 0:
            per_class_iou[c] = inter / union
            class_present[c] = True

    if class_present.any():
        miou = per_class_iou[class_present].mean() * 100
    else:
        miou = 0.0

    return miou, per_class_iou, mapping


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_sd_features(
    coco_root: str,
    device: str = "mps",
    img_size: int = 512,
    step: int = 50,
    n_images: Optional[int] = None,
) -> None:
    """Extract and cache SD self-attention features for val2017."""
    img_dir = Path(coco_root) / "val2017"
    out_dir = Path(coco_root) / f"sd_features_v14_s{step}" / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    if n_images:
        img_files = img_files[:n_images]

    # Skip already extracted
    existing = {f.stem for f in out_dir.glob("*.npy")}
    todo = [f for f in img_files if f.stem not in existing]
    logger.info(
        "SD feature extraction: %d images total, %d already done, %d to extract",
        len(img_files), len(existing), len(todo),
    )

    if not todo:
        logger.info("All features already extracted. Skipping.")
        return

    extractor = SDFeatureExtractor(device=device)

    for img_path in tqdm(todo, desc="Extracting SD features"):
        img = Image.open(img_path).convert("RGB")
        img_t = torch.tensor(
            np.array(img).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        feats = extractor.extract(img_t, step=step, img_size=img_size)
        # feats: (1, N, C) — save as (N, C)
        np.save(out_dir / f"{img_path.stem}.npy", feats[0].cpu().float().numpy())

    logger.info("SD features saved to %s", out_dir)


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Segmentation + Evaluation
# ═══════════════════════════════════════════════════════════════════════

def segment_and_evaluate(
    coco_root: str,
    device: str = "mps",
    tau: float = 0.5,
    alpha: float = 10.0,
    use_pamr: bool = False,
    img_size: int = 512,
    step: int = 50,
    mask_size: Tuple[int, int] = (128, 128),
    n_images: Optional[int] = None,
    dino_only: bool = False,
) -> Dict:
    """Run DiffCut segmentation and per-image Hungarian evaluation."""
    feat_dir = Path(coco_root) / f"sd_features_v14_s{step}" / "val2017"
    img_dir = Path(coco_root) / "val2017"

    feat_files = sorted(feat_dir.glob("*.npy"))

    if dino_only:
        dino_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        dino_ids = {f.stem for f in dino_dir.glob("*.npy")}
        feat_files = [f for f in feat_files if f.stem in dino_ids]
        logger.info("Filtered to %d images with DINOv3 features", len(feat_files))

    if n_images:
        feat_files = feat_files[:n_images]

    logger.info(
        "DiffCut segmentation: %d images, tau=%.2f, alpha=%.1f, pamr=%s",
        len(feat_files), tau, alpha, use_pamr,
    )

    ncut = RecursiveNCut(device=device)

    per_class_iou_acc = np.zeros(NUM_CLASSES)
    per_class_count = np.zeros(NUM_CLASSES)
    total_segments = []
    times = []

    for feat_path in tqdm(feat_files, desc="DiffCut segmentation"):
        image_id = int(feat_path.stem)
        gt = load_coco_panoptic_gt(coco_root, image_id)
        if gt is None:
            continue

        feats = np.load(feat_path)  # (N, C)
        feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

        t0 = time.time()
        seg_map, n_seg = ncut.generate_masks(
            feats_t, tau=tau, alpha=alpha, mask_size=mask_size,
        )
        total_segments.append(n_seg)

        # Optional PAMR refinement
        if use_pamr:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img_t = torch.tensor(
                    np.array(img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                seg_t = torch.tensor(seg_map)
                refined = pamr_refine(img_t, seg_t, device=device)
                seg_map = refined  # already (1,1,H,W) from pamr_refine

        dt = time.time() - t0
        times.append(dt)

        # Resize prediction to GT resolution — ensure 2D
        if seg_map.ndim == 4:
            pred = seg_map[0, 0]
        elif seg_map.ndim == 3:
            pred = seg_map[0]
        else:
            pred = seg_map
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        )

        miou_img, iou_per_class, _ = hungarian_miou(pred_resized, gt)

        for c in range(NUM_CLASSES):
            if (gt == c).any():
                per_class_iou_acc[c] += iou_per_class[c]
                per_class_count[c] += 1

    # Aggregate results
    valid = per_class_count > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid] = per_class_iou_acc[valid] / per_class_count[valid]
    miou = per_class_avg[valid].mean() * 100

    things_ious = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in THING_IDS and per_class_count[c] > 0]
    stuff_ious = [per_class_avg[c] * 100 for c in range(NUM_CLASSES) if c in STUFF_IDS and per_class_count[c] > 0]

    result = {
        "method": "DiffCut",
        "tau": tau,
        "alpha": alpha,
        "pamr": use_pamr,
        "step": step,
        "img_size": img_size,
        "n_images": len(feat_files),
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things_ious), 2) if things_ious else 0.0,
        "stuff_miou": round(np.mean(stuff_ious), 2) if stuff_ious else 0.0,
        "avg_segments": round(np.mean(total_segments), 1),
        "avg_time_s": round(np.mean(times), 3),
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if per_class_count[c] > 0
        },
    }

    logger.info(
        "DiffCut (tau=%.2f, alpha=%.1f, pamr=%s): mIoU=%.2f%%, "
        "Things=%.2f%%, Stuff=%.2f%%, avg_segments=%.1f",
        tau, alpha, use_pamr, miou,
        result["things_miou"], result["stuff_miou"],
        np.mean(total_segments),
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Global Clustering for Pseudo-Labels
# ═══════════════════════════════════════════════════════════════════════

def global_clustering(
    coco_root: str,
    device: str = "mps",
    tau: float = 0.5,
    alpha: float = 10.0,
    use_pamr: bool = False,
    K_global: int = 27,
    img_size: int = 512,
    step: int = 50,
    mask_size: Tuple[int, int] = (128, 128),
    feature_source: str = "dinov3",
    n_images: Optional[int] = None,
    dino_only: bool = False,
) -> Dict:
    """Pool features per DiffCut segment, global k-means, Hungarian to 27.

    This produces consistent pseudo-labels across all images for downstream
    training (unlike per-image Hungarian which is eval-only).
    """
    feat_dir = Path(coco_root) / f"sd_features_v14_s{step}" / "val2017"
    img_dir = Path(coco_root) / "val2017"

    # Load DINOv3 features for segment embedding
    if feature_source == "dinov3":
        dino_dir = Path(coco_root) / "dinov3_features" / "val2017"
        if not dino_dir.exists():
            dino_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        logger.info("Using DINOv3 features from %s for segment embeddings", dino_dir)
    else:
        dino_dir = None
        logger.info("Using SD features for segment embeddings")

    feat_files = sorted(feat_dir.glob("*.npy"))

    if dino_only:
        dino_filter_dir = Path(coco_root) / "dinov3_features_64x64" / "val2017"
        dino_ids = {f.stem for f in dino_filter_dir.glob("*.npy")}
        feat_files = [f for f in feat_files if f.stem in dino_ids]
        logger.info("Filtered to %d images with DINOv3 features", len(feat_files))

    if n_images:
        feat_files = feat_files[:n_images]

    ncut = RecursiveNCut(device=device)

    # Collect segment embeddings
    all_embeddings = []
    segment_info = []  # (image_id, segment_id, pixel_count)

    logger.info("Phase 3: Collecting segment embeddings from %d images...", len(feat_files))

    for feat_path in tqdm(feat_files, desc="Extracting segments"):
        image_id = int(feat_path.stem)

        sd_feats = np.load(feat_path)  # (N, C_sd)
        sd_feats_t = torch.tensor(sd_feats, dtype=torch.float32, device=device).unsqueeze(0)

        seg_map, n_seg = ncut.generate_masks(
            sd_feats_t, tau=tau, alpha=alpha, mask_size=mask_size,
        )

        # Optional PAMR
        if use_pamr:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img_t = torch.tensor(
                    np.array(img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                seg_t = torch.tensor(seg_map)
                refined = pamr_refine(img_t, seg_t, device=device)
                seg_map = refined  # already (1,1,H,W)

        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]

        # Load features for embedding — fallback to SD if DINOv3 missing
        use_dino = False
        if feature_source == "dinov3" and dino_dir:
            dino_path = dino_dir / f"{image_id:012d}.npy"
            if dino_path.exists():
                use_dino = True

        if use_dino:
            dino_feats = np.load(dino_path)  # (n_patches, C_dino=1024)
            n_patches = dino_feats.shape[0]
            grid = int(math.sqrt(n_patches))
            feat_2d = dino_feats.reshape(grid, grid, -1)
        else:
            # Fallback to SD features
            n_tokens = sd_feats.shape[0]
            grid = int(math.sqrt(n_tokens))
            feat_2d = sd_feats.reshape(grid, grid, -1)

        pred_grid = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (grid, grid), Image.NEAREST
            )
        )

        # Pool features per segment
        for seg_id in np.unique(pred_grid):
            mask = pred_grid == seg_id
            if mask.sum() < 1:
                continue
            seg_feats = feat_2d[mask]  # (n_pixels, C)
            embedding = seg_feats.mean(axis=0)  # (C,)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            all_embeddings.append(embedding)
            segment_info.append((image_id, int(seg_id), int(mask.sum())))

    if not all_embeddings:
        logger.error("No segment embeddings collected! Check feature paths.")
        return {"error": "No embeddings collected", "miou": 0.0}

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    # Clean NaN/Inf from MPS computation
    nan_mask = ~np.isfinite(all_embeddings).all(axis=1)
    if nan_mask.any():
        logger.warning("Removing %d NaN/Inf embeddings", nan_mask.sum())
        all_embeddings = all_embeddings[~nan_mask]
        segment_info = [s for s, m in zip(segment_info, ~nan_mask) if m]
    logger.info(
        "Collected %d segment embeddings (dim=%d) from %d images",
        len(all_embeddings), all_embeddings.shape[1], len(feat_files),
    )

    # Global k-means
    logger.info("Running MiniBatchKMeans with K=%d...", K_global)
    kmeans = MiniBatchKMeans(
        n_clusters=K_global, batch_size=4096, max_iter=300, random_state=42
    )
    cluster_labels = kmeans.fit_predict(all_embeddings)

    # Build segment→cluster mapping
    seg_to_cluster = {}
    for idx, (img_id, seg_id, _) in enumerate(segment_info):
        key = (img_id, seg_id)
        seg_to_cluster[key] = cluster_labels[idx]

    # Generate pseudo-labels and evaluate
    config_name = f"diffcut_tau{tau}_a{alpha}_{'pamr' if use_pamr else 'raw'}_{feature_source}_K{K_global}"
    out_dir = Path(coco_root) / f"pseudo_semantic_{config_name}" / "val2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build cost matrix for global Hungarian matching
    cost_matrix = np.zeros((K_global, NUM_CLASSES))

    for feat_path in tqdm(feat_files, desc="Generating pseudo-labels"):
        image_id = int(feat_path.stem)
        gt = load_coco_panoptic_gt(coco_root, image_id)
        if gt is None:
            continue

        # Re-segment (or load cached)
        sd_feats = np.load(feat_path)
        sd_feats_t = torch.tensor(sd_feats, dtype=torch.float32, device=device).unsqueeze(0)
        seg_map, _ = ncut.generate_masks(
            sd_feats_t, tau=tau, alpha=alpha, mask_size=mask_size,
        )
        if use_pamr:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img_t = torch.tensor(
                    np.array(img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                seg_t = torch.tensor(seg_map)
                refined = pamr_refine(img_t, seg_t, device=device)
                seg_map = refined  # already (1,1,H,W)

        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]

        # Map segments to global clusters
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        )
        clustered = np.full_like(pred_resized, 255, dtype=np.int32)
        for seg_id in np.unique(pred_resized):
            key = (image_id, int(seg_id))
            if key in seg_to_cluster:
                clustered[pred_resized == seg_id] = seg_to_cluster[key]

        # Accumulate cost matrix
        valid = (gt != 255) & (clustered != 255)
        for cluster_id in range(K_global):
            for gt_class in range(NUM_CLASSES):
                cost_matrix[cluster_id, gt_class] += (
                    (clustered[valid] == cluster_id) & (gt[valid] == gt_class)
                ).sum()

    # Global Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_class = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > 0:
            cluster_to_class[r] = c
    # Unmatched → best overlap
    for k in range(K_global):
        if k not in cluster_to_class:
            if cost_matrix[k].sum() > 0:
                cluster_to_class[k] = cost_matrix[k].argmax()

    # Save pseudo-labels with class IDs
    iou_per_class = np.zeros(NUM_CLASSES)
    count_per_class = np.zeros(NUM_CLASSES)

    for feat_path in tqdm(feat_files, desc="Saving pseudo-labels"):
        image_id = int(feat_path.stem)
        gt = load_coco_panoptic_gt(coco_root, image_id)

        # Re-segment
        sd_feats = np.load(feat_path)
        sd_feats_t = torch.tensor(sd_feats, dtype=torch.float32, device=device).unsqueeze(0)
        seg_map, _ = ncut.generate_masks(
            sd_feats_t, tau=tau, alpha=alpha, mask_size=mask_size,
        )
        if use_pamr:
            img_path = img_dir / f"{image_id:012d}.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img_t = torch.tensor(
                    np.array(img).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0)
                seg_t = torch.tensor(seg_map)
                refined = pamr_refine(img_t, seg_t, device=device)
                seg_map = refined  # already (1,1,H,W)

        pred = seg_map[0, 0] if seg_map.ndim == 4 else seg_map[0]
        h_orig, w_orig = (gt.shape if gt is not None else (pred.shape[0], pred.shape[1]))
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize(
                (w_orig, h_orig), Image.NEAREST
            )
        )

        # Map to 27-class
        semantic = np.full_like(pred_resized, 255, dtype=np.uint8)
        for seg_id in np.unique(pred_resized):
            key = (image_id, int(seg_id))
            if key in seg_to_cluster:
                cluster = seg_to_cluster[key]
                if cluster in cluster_to_class:
                    semantic[pred_resized == seg_id] = cluster_to_class[cluster]

        Image.fromarray(semantic).save(out_dir / f"{image_id:012d}.png")

        # Compute IoU for evaluation
        if gt is not None:
            for c in range(NUM_CLASSES):
                gt_mask = gt == c
                pred_mask = semantic == c
                inter = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                if union > 0:
                    iou_per_class[c] += inter / union
                    count_per_class[c] += 1

    valid = count_per_class > 0
    per_class_avg = np.zeros(NUM_CLASSES)
    per_class_avg[valid] = iou_per_class[valid] / count_per_class[valid]
    miou = per_class_avg[valid].mean() * 100

    things = [per_class_avg[c] * 100 for c in THING_IDS if count_per_class[c] > 0]
    stuff = [per_class_avg[c] * 100 for c in STUFF_IDS if count_per_class[c] > 0]

    result = {
        "method": f"DiffCut-global-{feature_source}",
        "tau": tau, "alpha": alpha, "pamr": use_pamr,
        "K_global": K_global, "feature_source": feature_source,
        "n_images": len(feat_files),
        "n_segments_total": len(all_embeddings),
        "miou": round(miou, 2),
        "things_miou": round(np.mean(things), 2) if things else 0.0,
        "stuff_miou": round(np.mean(stuff), 2) if stuff else 0.0,
        "output_dir": str(out_dir),
        "per_class_iou": {
            COCOSTUFF27_CLASSNAMES[c]: round(per_class_avg[c] * 100, 1)
            for c in range(NUM_CLASSES) if count_per_class[c] > 0
        },
    }

    logger.info(
        "Global clustering (K=%d, %s): mIoU=%.2f%%, Things=%.2f%%, Stuff=%.2f%%",
        K_global, feature_source, miou,
        result["things_miou"], result["stuff_miou"],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiffCut Unsupervised Semantic Segmentation")
    p.add_argument("--phase", choices=["extract", "segment", "cluster", "all"], default="all")
    p.add_argument("--coco_root", required=True, help="Path to COCO dataset root")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--img_size", type=int, default=512, help="SD input resolution")
    p.add_argument("--step", type=int, default=50, help="Diffusion timestep")
    p.add_argument("--tau", type=float, default=0.5, help="NCut threshold")
    p.add_argument("--alpha", type=float, default=10.0, help="Affinity exponent")
    p.add_argument("--pamr", action="store_true", help="Use PAMR post-processing")
    p.add_argument("--mask_size", type=int, default=128, help="Segmentation output resolution")
    p.add_argument("--K_global", type=int, default=27, help="Global clustering K")
    p.add_argument("--feature_source", default="dinov3", choices=["dinov3", "sd"])
    p.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    p.add_argument("--dino_only", action="store_true",
                    help="Filter to images that have DINOv3 features (501 val images)")
    p.add_argument("--output", default=None, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = {}

    if args.phase in ("extract", "all"):
        extract_sd_features(
            coco_root=args.coco_root,
            device=args.device,
            img_size=args.img_size,
            step=args.step,
            n_images=args.n_images,
        )

    if args.phase in ("segment", "all"):
        result = segment_and_evaluate(
            coco_root=args.coco_root,
            device=args.device,
            tau=args.tau,
            alpha=args.alpha,
            use_pamr=args.pamr,
            img_size=args.img_size,
            step=args.step,
            mask_size=(args.mask_size, args.mask_size),
            n_images=args.n_images,
            dino_only=args.dino_only,
        )
        results["per_image_hungarian"] = result

    if args.phase in ("cluster", "all"):
        result = global_clustering(
            coco_root=args.coco_root,
            device=args.device,
            tau=args.tau,
            alpha=args.alpha,
            use_pamr=args.pamr,
            K_global=args.K_global,
            img_size=args.img_size,
            step=args.step,
            mask_size=(args.mask_size, args.mask_size),
            feature_source=args.feature_source,
            n_images=args.n_images,
            dino_only=args.dino_only,
        )
        results["global_clustering"] = result

    # Save results
    if results:
        out_path = args.output or str(
            Path(args.coco_root) / "diffcut_results.json"
        )
        # Merge with existing results
        if Path(out_path).exists():
            with open(out_path) as f:
                existing = json.load(f)
        else:
            existing = {}

        config_key = f"tau{args.tau}_a{args.alpha}_{'pamr' if args.pamr else 'raw'}_s{args.step}"
        existing[config_key] = results

        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
