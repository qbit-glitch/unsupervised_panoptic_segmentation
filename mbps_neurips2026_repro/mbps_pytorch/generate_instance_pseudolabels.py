#!/usr/bin/env python3
"""Generate instance pseudo-labels via MaskCut on DINOv3 self-attention.

Implements iterative Normalized Cut (NCut) following CutLER/TokenCut methodology
(Wang et al., CVPR 2023) adapted for DINOv3. Uses self-attention affinity
(softmax(Q@K^T/sqrt(d))) from the last layer as the graph, with generalized
eigendecomposition, seed-based foreground selection, connected component
extraction, and iterative affinity subsetting.

Pipeline per image:
  1. Load image, preprocess (ImageNet normalization)
  2. Extract self-attention affinity from DINOv3 last layer (Q, K hooks)
  3. Apply generalized NCut: eigh(D-A, D) for Fiedler vector
  4. Threshold at mean, select CC containing seed patch
  5. Remove discovered patches from active set, repeat NCut on subset
  6. Filter masks by size and eigenvalue gap
  7. Optionally refine with Dense CRF
  8. Upsample masks to pixel resolution

Usage:
    python mbps_pytorch/generate_instance_pseudolabels.py \
        --image_dir /data/cityscapes/leftImg8bit/train \
        --output_dir /data/cityscapes/pseudo_instance/train \
        --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
        --max_instances 20
"""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.linalg import eigh
from tqdm import tqdm
from transformers import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_autocast_ctx(device: str):
    if "cuda" in device:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    elif "mps" in device:
        return torch.autocast("mps", dtype=torch.float16)
    else:
        return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# DINOv3 Feature Extraction (Fix 1)
# --------------------------------------------------------------------------- #
class DINOv3FeatureExtractor(torch.nn.Module):
    """Extract self-attention affinity matrix from DINOv3's last layer.

    Hooks the Q and K projections from the last self-attention layer,
    computes attention = softmax(Q @ K^T / sqrt(d)), and symmetrizes it
    to produce a patch-to-patch affinity matrix for NCut.

    Self-attention naturally encodes which patches "belong together" and
    works much better for object discovery than cosine similarity of
    Key or hidden-state features.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        logger.info(f"Loading DINOv3 model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

        # Detect architecture
        config = self.model.config
        self.embed_dim = getattr(config, "hidden_size", 768)
        self.patch_size = getattr(config, "patch_size", 16)
        self.num_registers = getattr(config, "num_register_tokens", 4)
        self.skip_tokens = 1 + self.num_registers  # CLS + registers

        # Register hooks on Q and K projections
        self._q_output = None
        self._k_output = None
        self._hooks = self._register_qk_hooks()

        logger.info(f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
                     f"registers={self.num_registers}")

    def _find_module(self, path: str):
        """Resolve a dot-separated path to a module."""
        module = self.model
        for part in path.split("."):
            module = getattr(module, part)
        return module

    def _register_qk_hooks(self):
        """Hook Q and K projections in the last attention layer."""
        num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        last = num_layers - 1

        # Module path patterns for Q/K projections
        patterns = [
            ("layer.{}.attention.q_proj", "layer.{}.attention.k_proj"),       # DINOv3ViT
            ("encoder.layer.{}.attention.attention.query", "encoder.layer.{}.attention.attention.key"),
        ]

        for q_pat, k_pat in patterns:
            try:
                q_mod = self._find_module(q_pat.format(last))
                k_mod = self._find_module(k_pat.format(last))
                logger.info(f"  Hooked Q at: model.{q_pat.format(last)}")
                logger.info(f"  Hooked K at: model.{k_pat.format(last)}")

                def hook_q(mod, inp, out):
                    self._q_output = out

                def hook_k(mod, inp, out):
                    self._k_output = out

                hq = q_mod.register_forward_hook(hook_q)
                hk = k_mod.register_forward_hook(hook_k)
                return [hq, hk]
            except AttributeError:
                continue

        logger.warning("Could not hook Q/K projections — will use cosine Key features")
        return []

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor):
        """Extract self-attention affinity matrix from last layer.

        Args:
            pixel_values: (1, 3, H, W) normalized image tensor.

        Returns:
            affinity: (N, N) numpy affinity matrix (symmetric, non-negative).
            h_patches, w_patches: spatial dimensions of the patch grid.
        """
        H, W = pixel_values.shape[-2:]
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        self._q_output = None
        self._k_output = None
        _ = self.model(pixel_values)

        if self._q_output is not None and self._k_output is not None:
            # Q, K shape: (1, num_tokens, D) — stay in torch for numerics
            Q = self._q_output[:, self.skip_tokens:, :].squeeze(0).float()
            K = self._k_output[:, self.skip_tokens:, :].squeeze(0).float()

            # Sanitize: clamp inf/NaN from float16 overflow during autocast
            Q = torch.nan_to_num(Q, nan=0.0, posinf=1e4, neginf=-1e4)
            K = torch.nan_to_num(K, nan=0.0, posinf=1e4, neginf=-1e4)

            # Compute attention in torch (numerically stable softmax)
            d_k = K.shape[-1]
            logits = torch.matmul(Q, K.T) / (d_k ** 0.5)
            attn = torch.softmax(logits, dim=-1)

            # Symmetrize and zero diagonal
            A = ((attn + attn.T) / 2.0).cpu().numpy()
            np.fill_diagonal(A, 0.0)
            return A, h_patches, w_patches

        # Fallback: use Key features with cosine similarity
        logger.warning("Q/K hooks failed — falling back to Key cosine similarity")
        outputs = self.model(pixel_values, output_hidden_states=True)
        hs = outputs.hidden_states[-1][:, self.skip_tokens:, :]
        feats = hs.squeeze(0).float().cpu().numpy()
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        feats_norm = feats / (norms + 1e-8)
        A = feats_norm @ feats_norm.T
        A = np.maximum(A, 0.0)
        np.fill_diagonal(A, 0.0)
        return A, h_patches, w_patches


def preprocess_image(image_path: str, device: str = "cpu", input_size: int = None):
    """Load and preprocess an image for DINOv3.

    Args:
        image_path: Path to image file.
        device: Torch device.
        input_size: If set, resize the shorter side to this value (preserving
            aspect ratio). Reduces patch count and speeds up eigendecomposition.

    Returns:
        pixel_values: (1, 3, H, W) tensor, ImageNet-normalized.
        orig_size: (H, W) original image size.
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = (img.height, img.width)

    # Resize if requested (preserves aspect ratio)
    if input_size is not None:
        h, w = img.height, img.width
        if h < w:
            new_h = input_size
            new_w = int(w * input_size / h)
        else:
            new_w = input_size
            new_h = int(h * input_size / w)
        img = img.resize((new_w, new_h), Image.BILINEAR)

    img_np = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalize
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = (img_np - mean) / std

    # To tensor (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Ensure dimensions are divisible by patch_size=16
    _, _, H, W = tensor.shape
    new_H = (H // 16) * 16
    new_W = (W // 16) * 16
    if new_H != H or new_W != W:
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)

    return tensor.to(device), orig_size


# --------------------------------------------------------------------------- #
# MaskCut Core Algorithm
# --------------------------------------------------------------------------- #
def compute_affinity(features: np.ndarray, tau: float = 0.2, eps: float = 1e-5) -> np.ndarray:
    """Compute binary cosine affinity matrix (Fix 2).

    Following TokenCut: uses binary graph where A[i,j] = 1 if cos(f_i, f_j) > tau,
    else eps. This is critical — soft thresholding causes NCut to produce
    scene-level decompositions instead of object-level masks.

    Args:
        features: (N, D) patch features.
        tau: Threshold for binary affinity.
        eps: Small value for below-threshold edges (avoid disconnected graph).

    Returns:
        (N, N) binary affinity matrix.
    """
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    features_norm = features / (norms + 1e-8)
    cosine_sim = features_norm @ features_norm.T

    # Binary graph: 1 if above tau, eps otherwise
    A = np.where(cosine_sim > tau, 1.0, eps)
    np.fill_diagonal(A, 0.0)
    return A


def ncut_bipartition(
    A: np.ndarray,
    active_mask: np.ndarray,
    h_patches: int,
    w_patches: int,
    eps: float = 1e-5,
    eigen_device: str = "cpu",
) -> tuple:
    """Apply Normalized Cut with CutLER/TokenCut algorithm (Fixes 3-6).

    Uses the normalized Laplacian L = I - D^{-1/2}AD^{-1/2} which is
    mathematically equivalent to the generalized form eigh(D-A, D).
    Supports GPU-accelerated eigendecomposition via torch on MPS/CUDA.

    Args:
        A: (N, N) affinity matrix for all patches.
        active_mask: (N,) boolean mask of active (undiscovered) patches.
        h_patches, w_patches: spatial dimensions of patch grid.
        eps: Small value for regularization.
        eigen_device: Device for eigendecomposition ("cpu", "mps", "cuda").

    Returns:
        (fg_mask, eigengap, seed_index):
            fg_mask: (N,) boolean in FULL patch space, True for foreground
            eigengap: eigenvalue gap (quality score)
            seed_index: index in full patch space of the seed patch
    """
    N_full = A.shape[0]
    active_indices = np.where(active_mask)[0]
    N_sub = len(active_indices)

    if N_sub < 3:
        return np.zeros(N_full, dtype=bool), 0.0, -1

    # Subset the affinity matrix to active patches only
    A_sub = A[np.ix_(active_indices, active_indices)]

    # Degree vector with regularization
    d = A_sub.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # Equivalent to generalized eigh(D-A, D) but works with standard eigh
    A_norm = A_sub * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    L = np.eye(N_sub) - A_norm

    # Eigendecomposition — use torch on GPU if available
    try:
        L_t = torch.from_numpy(L).float().to(eigen_device)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(L_t)
        # Take 2nd and 3rd smallest eigenvalues (skip trivial 0th)
        eigenvalues = eigenvalues_t[1:3].cpu().numpy()
        eigenvectors = eigenvectors_t[:, 1:3].cpu().numpy()
    except Exception:
        # Fallback to scipy on CPU
        try:
            eigenvalues, eigenvectors = eigh(L, subset_by_index=[1, 2])
        except Exception:
            return np.zeros(N_full, dtype=bool), 0.0, -1

    # Fiedler vector (second-smallest eigenvector of L)
    # Transform back: fiedler_original = D^{-1/2} @ fiedler_normalized
    fiedler = eigenvectors[:, 0] * d_inv_sqrt
    eigengap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else eigenvalues[0]

    # Threshold at mean (Fix 4)
    avg = np.mean(fiedler)
    bipartition_sub = fiedler > avg

    # Seed heuristic: most confident patch in subspace (Fix 5)
    seed_sub = np.argmax(np.abs(fiedler))
    if not bipartition_sub[seed_sub]:
        fiedler = -fiedler
        bipartition_sub = ~bipartition_sub

    # Map back to full patch grid for connected component extraction (Fix 6)
    bipartition_full = np.zeros(N_full, dtype=bool)
    bipartition_full[active_indices[bipartition_sub]] = True

    bipartition_2d = bipartition_full.reshape(h_patches, w_patches).astype(np.float64)
    objects, num_objects = ndimage.label(bipartition_2d)

    # Select the CC containing the seed patch (in full index space)
    seed_full = active_indices[seed_sub]
    seed_h, seed_w = divmod(seed_full, w_patches)
    seed_cc = objects[seed_h, seed_w]

    if seed_cc == 0:
        return np.zeros(N_full, dtype=bool), 0.0, -1

    # Only keep the connected component containing the seed
    cc_mask_2d = (objects == seed_cc)
    fg_mask = cc_mask_2d.reshape(N_full)

    return fg_mask, eigengap, seed_full


def maskcut_single_image(
    A: np.ndarray,
    h_patches: int,
    w_patches: int,
    max_instances: int = 20,
    min_patch_count: int = 4,
    min_eigengap: float = 0.01,
    eps: float = 1e-5,
    eigen_device: str = "cpu",
) -> tuple:
    """Run iterative MaskCut on a pre-computed affinity matrix.

    Iteratively:
    1. Subset the affinity to active (undiscovered) patches
    2. Run NCut with normalized Laplacian eigen, mean threshold, seed + CC
    3. Mark discovered patches as inactive
    4. Repeat until no more valid bipartitions

    Args:
        A: (N, N) affinity matrix (e.g. from self-attention).
        h_patches, w_patches: spatial dimensions.
        max_instances: Maximum instances to discover.
        min_patch_count: Minimum patches for a valid mask.
        min_eigengap: Minimum eigenvalue gap for a valid bipartition.
        eps: Small value for regularization.
        eigen_device: Device for eigendecomposition ("cpu", "mps", "cuda").

    Returns:
        (masks, scores, num_valid):
            masks: (M, N) binary masks at patch resolution
            scores: (M,) quality scores per mask
            num_valid: number of valid masks
    """
    N = A.shape[0]

    masks = np.zeros((max_instances, N), dtype=bool)
    scores = np.zeros(max_instances, dtype=np.float32)
    active_mask = np.ones(N, dtype=bool)
    num_valid = 0

    for i in range(max_instances):
        if active_mask.sum() < min_patch_count:
            break

        # NCut on active subset of the affinity matrix
        fg_mask, eiggap, seed = ncut_bipartition(
            A, active_mask, h_patches, w_patches, eps=eps,
            eigen_device=eigen_device,
        )

        if fg_mask.sum() < min_patch_count:
            break
        if eiggap < min_eigengap:
            break

        masks[num_valid] = fg_mask
        scores[num_valid] = eiggap
        num_valid += 1

        # Remove discovered patches from active set
        active_mask[fg_mask] = False

    return masks, scores, num_valid


# --------------------------------------------------------------------------- #
# CRF Post-Processing (Fix 8, optional)
# --------------------------------------------------------------------------- #
def crf_refine_mask(
    mask_2d: np.ndarray,
    image: np.ndarray,
    n_iters: int = 10,
    sxy_gauss: int = 3,
    compat_gauss: int = 3,
    sxy_bilateral: int = 80,
    srgb_bilateral: int = 13,
    compat_bilateral: int = 10,
) -> np.ndarray:
    """Refine a binary mask using Dense CRF.

    Args:
        mask_2d: (H, W) binary mask.
        image: (H, W, 3) uint8 RGB image.
        n_iters: CRF iterations.

    Returns:
        (H, W) refined binary mask.
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return mask_2d  # Return unrefined if pydensecrf not installed

    H, W = mask_2d.shape
    mask_prob = mask_2d.astype(np.float32)

    # Create unary potentials from mask probabilities
    probs = np.stack([1.0 - mask_prob, mask_prob], axis=0)  # (2, H, W)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    unary = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)

    # Gaussian pairwise (spatial smoothness)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)

    # Bilateral pairwise (appearance + spatial)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image, compat=compat_bilateral,
    )

    Q = d.inference(n_iters)
    result = np.argmax(np.array(Q).reshape(2, H, W), axis=0)
    return result.astype(bool)


def upsample_masks_to_pixels(
    masks: np.ndarray,
    num_valid: int,
    h_patches: int,
    w_patches: int,
    image_size: tuple,
    image: np.ndarray = None,
    use_crf: bool = False,
) -> np.ndarray:
    """Upsample binary masks from patch to pixel resolution.

    Args:
        masks: (M, N) binary masks at patch resolution.
        num_valid: number of valid masks.
        h_patches, w_patches: patch grid dimensions.
        image_size: (H, W) target pixel resolution.
        image: (H, W, 3) uint8 RGB image for CRF (optional).
        use_crf: Whether to apply CRF refinement.

    Returns:
        (H, W) int32 instance map (0=background, 1+=instances).
    """
    H, W = image_size
    instance_map = np.zeros((H, W), dtype=np.int32)

    if num_valid == 0:
        return instance_map

    for i in range(num_valid):
        mask_2d = masks[i].reshape(h_patches, w_patches)
        mask_img = Image.fromarray(mask_2d.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((W, H), Image.NEAREST)
        pixel_mask = np.array(mask_img) > 128

        # Optional CRF refinement
        if use_crf and image is not None:
            # Fill holes first
            pixel_mask = ndimage.binary_fill_holes(pixel_mask)
            pixel_mask = crf_refine_mask(pixel_mask, image)

        # Only assign where not yet assigned (earlier masks have priority)
        new_pixels = pixel_mask & (instance_map == 0)
        instance_map[new_pixels] = i + 1

    return instance_map


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def process_dataset(
    image_dir: str,
    output_dir: str,
    model_name: str,
    device: str,
    max_instances: int = 20,
    min_patch_count: int = 4,
    min_eigengap: float = 0.01,
    eps: float = 1e-5,
    use_crf: bool = False,
    input_size: int = None,
):
    """Process all images: extract features -> MaskCut -> save instance maps."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    image_files = sorted(
        list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))
    )
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return

    logger.info(f"Found {len(image_files)} images")
    if input_size:
        logger.info(f"Resizing shorter side to {input_size}px for speed")

    # Load model
    extractor = DINOv3FeatureExtractor(model_name, device=device)

    # Use MPS/CUDA for eigendecomposition if available
    eigen_device = device if device in ("mps", "cuda") else "cpu"

    stats = {"total_instances": 0, "total_images": 0, "instances_per_image": []}

    for img_path in tqdm(image_files, desc="MaskCut"):
        # Determine output paths (preserve city subdirectory structure)
        rel_path = img_path.relative_to(image_dir)
        out_instance_path = output_dir / rel_path.parent / (rel_path.stem + "_instance.png")
        out_masks_path = output_dir / rel_path.parent / (rel_path.stem + ".npz")

        # Skip if already processed
        if out_instance_path.exists() and out_masks_path.exists():
            # Load existing stats
            data = np.load(str(out_masks_path))
            nv = int(data.get("num_valid", 0))
            stats["total_instances"] += nv
            stats["total_images"] += 1
            stats["instances_per_image"].append(nv)
            continue

        # Load and preprocess image
        pixel_values, orig_size = preprocess_image(
            str(img_path), device=device, input_size=input_size,
        )

        # Extract self-attention affinity matrix
        with _get_autocast_ctx(device):
            affinity, h_patches, w_patches = extractor(pixel_values)

        N = h_patches * w_patches

        # Run MaskCut on the affinity matrix
        masks, scores, num_valid = maskcut_single_image(
            affinity,
            h_patches=h_patches,
            w_patches=w_patches,
            max_instances=max_instances,
            min_patch_count=min_patch_count,
            min_eigengap=min_eigengap,
            eps=eps,
            eigen_device=eigen_device,
        )

        # Save patch-resolution masks
        out_masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_masks_path),
            masks=masks[:num_valid] if num_valid > 0 else np.zeros((0, N), dtype=bool),
            scores=scores[:num_valid] if num_valid > 0 else np.zeros(0, dtype=np.float32),
            num_valid=num_valid,
            h_patches=h_patches,
            w_patches=w_patches,
        )

        # Create pixel-resolution instance map
        image_rgb = None
        if use_crf:
            image_rgb = np.array(Image.open(str(img_path)).convert("RGB"))
        instance_map = upsample_masks_to_pixels(
            masks, num_valid, h_patches, w_patches, orig_size,
            image=image_rgb, use_crf=use_crf,
        )
        out_instance_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(instance_map.astype(np.uint16)).save(str(out_instance_path))

        stats["total_instances"] += num_valid
        stats["total_images"] += 1
        stats["instances_per_image"].append(num_valid)

    # Print and save statistics
    if stats["total_images"] > 0:
        avg = stats["total_instances"] / stats["total_images"]
        logger.info(f"Statistics:")
        logger.info(f"  Images processed: {stats['total_images']}")
        logger.info(f"  Total instances: {stats['total_instances']}")
        logger.info(f"  Avg instances/image: {avg:.1f}")
        logger.info(f"  Max instances/image: {max(stats['instances_per_image'])}")
        logger.info(f"  Min instances/image: {min(stats['instances_per_image'])}")

        # Distribution
        counts = np.array(stats["instances_per_image"])
        logger.info(f"  Distribution: "
                     f"0-2: {(counts <= 2).sum()}, "
                     f"3-5: {((counts > 2) & (counts <= 5)).sum()}, "
                     f"6-10: {((counts > 5) & (counts <= 10)).sum()}, "
                     f"11-20: {((counts > 10) & (counts <= 20)).sum()}")

        summary = {
            "total_images": stats["total_images"],
            "total_instances": stats["total_instances"],
            "avg_instances_per_image": round(avg, 2),
            "max_instances_per_image": int(max(stats["instances_per_image"])),
            "min_instances_per_image": int(min(stats["instances_per_image"])),
            "config": {
                "model_name": model_name,
                "max_instances": max_instances,
                "min_patch_count": min_patch_count,
                "min_eigengap": min_eigengap,
                "eps": eps,
                "use_crf": use_crf,
            },
        }
        with open(output_dir / "stats.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate instance pseudo-labels via MaskCut on DINOv3 features"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory with images (supports city/ subdirectories)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save instance masks and maps"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace DINOv3 model name"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detect if not set)"
    )
    parser.add_argument("--max_instances", type=int, default=20)
    parser.add_argument("--min_patch_count", type=int, default=4)
    parser.add_argument("--min_eigengap", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-5,
                        help="Small value for non-edges in binary graph")
    parser.add_argument("--use_crf", action="store_true",
                        help="Apply Dense CRF post-processing (requires pydensecrf)")
    parser.add_argument("--input_size", type=int, default=512,
                        help="Resize shorter side to this (reduces patches, speeds up eigen). "
                             "Set 0 for full resolution. Default 512 → 32x64=2048 patches")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=device,
        max_instances=args.max_instances,
        min_patch_count=args.min_patch_count,
        min_eigengap=args.min_eigengap,
        eps=args.eps,
        use_crf=args.use_crf,
        input_size=args.input_size if args.input_size > 0 else None,
    )


if __name__ == "__main__":
    main()
