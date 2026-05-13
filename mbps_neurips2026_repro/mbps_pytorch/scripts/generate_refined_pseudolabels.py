#!/usr/bin/env python3
"""Generate refined pseudo-labels using a trained M2PR (Mamba2 Panoptic Refiner).

Pipeline:
  1. Load trained M2PR checkpoint
  2. Run inference on precomputed features (DINOv2, CAUSE, depth, instances)
  3. Post-process semantics: refined logits → upsample → optional CRF → argmax
  4. Post-process instances: NCut on instance embeddings → binary masks
  5. Panoptic merge: combine refined semantics + instances
  6. Convert to CUPS format (semantic PNG + instance PNG + distribution .pt)

Usage:
    python -m mbps_pytorch.scripts.generate_refined_pseudolabels \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/m2pr/best.pt \
        --split train \
        --output_subdir refined_pseudo_labels_m2pr

    # With CRF post-processing (requires RGB images):
    python -m mbps_pytorch.scripts.generate_refined_pseudolabels \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/m2pr/best.pt \
        --split train --use_crf

    # Quick test on a few images:
    python -m mbps_pytorch.scripts.generate_refined_pseudolabels \
        --cityscapes_root /path/to/cityscapes \
        --checkpoint checkpoints/m2pr/best.pt \
        --split val --limit 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mbps_pytorch.models.refiner.mamba2_panoptic_refiner import Mamba2PanopticRefiner
from mbps_pytorch.models.refiner.geometric_features import compute_geometric_features
from mbps_pytorch.models.refiner.instance_encoder import compute_instance_descriptor_fast
from mbps_pytorch.models.instance.cuts3d import compute_affinity_matrix, normalized_cut
from mbps_pytorch.convert_to_cups_format import (
    THING_IDS_27,
    compute_distributions,
)

# --- Instance extraction from embeddings ---


def extract_instances_from_embeddings(
    inst_embeddings: torch.Tensor,
    refined_sem: torch.Tensor,
    boundary: torch.Tensor,
    patch_h: int = 32,
    patch_w: int = 64,
    max_instances: int = 20,
    min_mask_patches: int = 4,
    tau_ncut: float = 0.0,
    thing_ids: set = THING_IDS_27,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract instance masks from M2PR instance embeddings via recursive NCut.

    Args:
        inst_embeddings: Instance embeddings (N, D_inst) from M2PR.
        refined_sem: Refined semantic logits (N, 27) from M2PR.
        boundary: Boundary predictions (N, 1) from M2PR.
        patch_h: Spatial grid height.
        patch_w: Spatial grid width.
        max_instances: Maximum instances to extract.
        min_mask_patches: Minimum patch count per instance.
        tau_ncut: NCut binarization threshold.
        thing_ids: Set of thing class IDs.

    Returns:
        (masks, scores, semantic_map):
            masks: (M, N) binary masks at patch resolution
            scores: (M,) confidence scores
            semantic_map: (N,) argmax semantic class per patch
    """
    N, D = inst_embeddings.shape
    device = inst_embeddings.device
    sem_pred = refined_sem.argmax(dim=-1)  # (N,)
    sem_conf = F.softmax(refined_sem, dim=-1).max(dim=-1).values  # (N,)

    # Identify thing patches
    is_thing = torch.zeros(N, dtype=torch.bool, device=device)
    for tid in thing_ids:
        is_thing |= (sem_pred == tid)

    # Build affinity matrix on instance embeddings
    W = compute_affinity_matrix(inst_embeddings)  # (N, N)

    # Weight affinity by boundary predictions (reduce cross-boundary affinity)
    boundary_flat = boundary.squeeze(-1)  # (N,)
    boundary_weight = 1.0 - boundary_flat
    boundary_mask = boundary_weight[:, None] * boundary_weight[None, :]
    W = W * boundary_mask

    masks_list = []
    scores_list = []
    active = torch.ones(N, dtype=torch.float32, device=device)

    for _ in range(max_instances):
        # Mask inactive patches
        active_thing = active * is_thing.float()
        if active_thing.sum() < min_mask_patches:
            break

        mask_2d = active_thing[:, None] * active_thing[None, :]
        W_masked = W * mask_2d

        # NCut
        try:
            bipartition, idx_src, idx_snk, eigvec = normalized_cut(
                W_masked, tau_ncut
            )
        except Exception:
            break

        bipartition = bipartition * active_thing

        # Pick the smaller partition (more likely to be a single instance)
        fg_count = bipartition.sum()
        bg_count = (active_thing - bipartition).sum()
        if bg_count < fg_count and bg_count >= min_mask_patches:
            bipartition = active_thing - bipartition

        # Size check
        if bipartition.sum() < min_mask_patches:
            break

        # Score: average semantic confidence within mask
        mask_bool = bipartition > 0.5
        score = sem_conf[mask_bool].mean().item()

        masks_list.append(bipartition.cpu().numpy())
        scores_list.append(score)

        # Remove segmented patches
        active = active * (1.0 - bipartition)

    if len(masks_list) == 0:
        return (
            np.zeros((0, N), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            sem_pred.cpu().numpy(),
        )

    masks = np.stack(masks_list, axis=0)  # (M, N)
    scores = np.array(scores_list, dtype=np.float32)
    return masks, scores, sem_pred.cpu().numpy()


def build_instance_map_from_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    semantic: np.ndarray,
    patch_h: int,
    patch_w: int,
    thing_ids: set = THING_IDS_27,
) -> np.ndarray:
    """Build a single-channel instance map from binary masks.

    Args:
        masks: (M, N) binary masks at patch resolution.
        scores: (M,) confidence scores.
        semantic: (N,) semantic class per patch.
        patch_h, patch_w: Spatial dimensions.
        thing_ids: Set of thing class IDs.

    Returns:
        instance_map: (patch_h, patch_w) uint16 instance IDs.
    """
    N = patch_h * patch_w
    instance_map = np.zeros((patch_h, patch_w), dtype=np.uint16)

    if masks.shape[0] == 0:
        return instance_map

    # Sort by score descending
    order = np.argsort(-scores)
    instance_id = 1

    for idx in order:
        mask = masks[idx].reshape(patch_h, patch_w)
        if mask.sum() == 0:
            continue

        # Check majority class is a thing
        mask_bool = mask > 0.5
        sem_vals = semantic.reshape(patch_h, patch_w)[mask_bool]
        if len(sem_vals) == 0:
            continue
        majority_class = int(np.bincount(sem_vals.astype(np.int64), minlength=27).argmax())

        if majority_class in thing_ids:
            new_pixels = mask_bool & (instance_map == 0)
            if new_pixels.sum() > 0:
                instance_map[new_pixels] = instance_id
                instance_id += 1

    return instance_map


# --- Main pipeline ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate refined pseudo-labels with M2PR"
    )
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained M2PR checkpoint")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])
    parser.add_argument("--output_subdir", type=str,
                        default="refined_pseudo_labels_m2pr")

    # Feature directories (relative to cityscapes_root)
    parser.add_argument("--dinov2_dir", type=str, default="dinov2_features")
    parser.add_argument("--semantic_dir", type=str, default="pseudo_semantic_cause")
    parser.add_argument("--depth_dir", type=str, default="depth_spidepth")
    parser.add_argument("--instance_dir", type=str, default="pseudo_instance_spidepth")

    # Model config
    parser.add_argument("--bridge_dim", type=int, default=256)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--inst_embed_dim", type=int, default=64)

    # Inference options
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--use_soft_logits", action="store_true", default=True)
    parser.add_argument("--no_soft_logits", dest="use_soft_logits",
                        action="store_false")
    parser.add_argument("--use_crf", action="store_true", default=False,
                        help="Apply CRF post-processing (requires RGB images)")
    parser.add_argument("--crf_iters", type=int, default=10)
    parser.add_argument("--max_instances", type=int, default=20)
    parser.add_argument("--target_h", type=int, default=1024)
    parser.add_argument("--target_w", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images (0 = all)")
    parser.add_argument("--cups_format", action="store_true", default=True,
                        help="Output in CUPS PseudoLabelDataset format")
    parser.add_argument("--no_cups_format", dest="cups_format",
                        action="store_false")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def find_samples(dinov2_root: str, semantic_root: str,
                 depth_root: str, instance_root: str) -> list:
    """Find all samples with complete feature sets."""
    samples = []
    if not os.path.isdir(dinov2_root):
        raise FileNotFoundError(f"DINOv2 dir not found: {dinov2_root}")

    for city in sorted(os.listdir(dinov2_root)):
        city_dir = os.path.join(dinov2_root, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if not fname.endswith(".npy"):
                continue
            stem = fname.replace(".npy", "")
            # Verify all required files
            paths = {
                "dinov2": os.path.join(dinov2_root, city, f"{stem}.npy"),
                "semantic": os.path.join(semantic_root, city, f"{stem}.png"),
                "logits": os.path.join(semantic_root, city, f"{stem}_logits.pt"),
                "depth": os.path.join(depth_root, city, f"{stem}.npy"),
                "instance": os.path.join(instance_root, city, f"{stem}.png"),
            }
            if all(os.path.exists(p) for p in [
                paths["dinov2"], paths["semantic"],
                paths["depth"], paths["instance"],
            ]):
                samples.append({"city": city, "stem": stem, "paths": paths})
    return samples


def load_sample(
    sample: dict,
    use_soft_logits: bool = True,
    num_classes: int = 27,
    target_h: int = 32,
    target_w: int = 64,
) -> dict:
    """Load precomputed features for a single sample."""
    paths = sample["paths"]

    # DINOv2 features
    dino_feat = torch.from_numpy(np.load(paths["dinov2"])).float()  # (N, 768)

    # CAUSE logits
    if use_soft_logits and os.path.exists(paths["logits"]):
        cause_logits = torch.load(
            paths["logits"], map_location="cpu", weights_only=True
        ).float()  # (27, H, W)
        cause_logits = cause_logits.permute(1, 2, 0).reshape(-1, num_classes)
    else:
        sem_img = np.array(Image.open(paths["semantic"]))
        sem_patch = np.array(
            Image.fromarray(sem_img).resize((target_w, target_h), Image.NEAREST)
        )
        sem_flat = sem_patch.ravel()
        cause_logits = torch.zeros(len(sem_flat), num_classes)
        for i, c in enumerate(sem_flat):
            if c < num_classes:
                cause_logits[i, c] = 1.0

    # Depth → geometric features
    depth_map = np.load(paths["depth"]).astype(np.float32)
    geo_features = compute_geometric_features(
        torch.from_numpy(depth_map), target_h, target_w
    )

    # Depth at patch level
    depth_patch = F.interpolate(
        torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze().reshape(-1)

    # Instance descriptor
    inst_img = np.array(Image.open(paths["instance"]))
    sem_img_np = np.array(Image.open(paths["semantic"]))
    inst_descriptor = compute_instance_descriptor_fast(
        inst_img, sem_img_np, depth_map, target_h, target_w
    )

    return {
        "dino_features": dino_feat,       # (N, 768)
        "cause_logits": cause_logits,     # (N, 27)
        "geo_features": geo_features,     # (N, 18)
        "inst_descriptor": inst_descriptor,  # (N, 8)
        "depth": depth_patch,             # (N,)
        "depth_map": depth_map,           # (H_d, W_d) original resolution
        "semantic_path": paths["semantic"],
        "instance_path": paths["instance"],
    }


@torch.no_grad()
def run_inference(
    model: Mamba2PanopticRefiner,
    sample_data: dict,
    device: torch.device,
) -> dict:
    """Run M2PR inference on a single sample."""
    # Add batch dimension and move to device
    dino_feat = sample_data["dino_features"].unsqueeze(0).to(device)
    cause_logits = sample_data["cause_logits"].unsqueeze(0).to(device)
    geo_features = sample_data["geo_features"].unsqueeze(0).to(device)
    inst_desc = sample_data["inst_descriptor"].unsqueeze(0).to(device)
    depth = sample_data["depth"].unsqueeze(0).to(device)

    outputs = model(
        dino_features=dino_feat,
        cause_logits=cause_logits,
        geo_features=geo_features,
        inst_descriptor=inst_desc,
        depth=depth,
        deterministic=True,
    )

    # Remove batch dimension
    return {
        "refined_sem": outputs["refined_sem"][0],        # (N, 27)
        "inst_embeddings": outputs["inst_embeddings"][0],  # (N, 64)
        "boundary": outputs["boundary"][0],              # (N, 1)
        "gate": outputs["gate"].item(),
    }


def postprocess_semantics(
    refined_sem: torch.Tensor,
    target_h: int = 1024,
    target_w: int = 2048,
    patch_h: int = 32,
    patch_w: int = 64,
    use_crf: bool = False,
    rgb_image: np.ndarray | None = None,
    crf_iters: int = 10,
) -> np.ndarray:
    """Post-process refined semantic logits to full-resolution label map.

    Args:
        refined_sem: Refined logits (N, 27) at patch resolution.
        target_h, target_w: Output resolution.
        patch_h, patch_w: Patch grid dimensions.
        use_crf: Apply CRF post-processing.
        rgb_image: RGB image (H, W, 3) uint8, needed for CRF.
        crf_iters: CRF iterations.

    Returns:
        semantic_map: (target_h, target_w) uint8 class IDs.
    """
    device = refined_sem.device
    N, C = refined_sem.shape

    if use_crf and rgb_image is not None:
        from mbps_pytorch.models.merger.crf_postprocess import crf_inference

        # Upsample logits to target resolution
        logits_2d = refined_sem.reshape(patch_h, patch_w, C).permute(2, 0, 1)
        logits_up = F.interpolate(
            logits_2d.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (C, H, W)

        # Flatten for CRF
        logits_flat = logits_up.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        log_probs = F.log_softmax(logits_flat, dim=-1)

        # Prepare RGB image
        img_tensor = torch.from_numpy(rgb_image).float().to(device)
        img_flat = img_tensor.reshape(-1, 3)  # (H*W, 3)

        # Run CRF
        refined_probs = crf_inference(
            log_probs, img_flat, target_h, target_w,
            num_iterations=crf_iters,
        )
        semantic_map = refined_probs.argmax(dim=-1).reshape(
            target_h, target_w
        ).cpu().numpy().astype(np.uint8)
    else:
        # Simple bilinear upsample + argmax
        logits_2d = refined_sem.reshape(patch_h, patch_w, C).permute(2, 0, 1)
        logits_up = F.interpolate(
            logits_2d.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (C, H, W)
        semantic_map = logits_up.argmax(dim=0).cpu().numpy().astype(np.uint8)

    return semantic_map


def postprocess_instances(
    inst_embeddings: torch.Tensor,
    refined_sem: torch.Tensor,
    boundary: torch.Tensor,
    patch_h: int = 32,
    patch_w: int = 64,
    target_h: int = 1024,
    target_w: int = 2048,
    max_instances: int = 20,
) -> np.ndarray:
    """Post-process instance embeddings to full-resolution instance map.

    Args:
        inst_embeddings: Instance embeddings (N, D) from M2PR.
        refined_sem: Refined semantic logits (N, 27).
        boundary: Boundary predictions (N, 1).
        patch_h, patch_w: Patch grid dimensions.
        target_h, target_w: Output resolution.
        max_instances: Maximum instances.

    Returns:
        instance_map: (target_h, target_w) uint16 instance IDs.
    """
    masks, scores, sem_pred = extract_instances_from_embeddings(
        inst_embeddings, refined_sem, boundary,
        patch_h=patch_h, patch_w=patch_w,
        max_instances=max_instances,
    )

    # Build instance map at patch resolution
    inst_map_patch = build_instance_map_from_masks(
        masks, scores, sem_pred, patch_h, patch_w,
    )

    # Upsample to target resolution
    instance_map = np.array(
        Image.fromarray(inst_map_patch).resize(
            (target_w, target_h), Image.NEAREST
        )
    )

    return instance_map


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    cs_root = Path(args.cityscapes_root)
    output_dir = cs_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Mamba2PanopticRefiner(
        bridge_dim=args.bridge_dim,
        mamba_layers=args.mamba_layers,
        inst_embed_dim=args.inst_embed_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = model.count_parameters()
    print(f"Model: {params['trainable']/1e6:.1f}M params")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch'] + 1}")
    if "val_losses" in ckpt:
        print(f"Checkpoint val loss: {ckpt['val_losses'].get('total', '?')}")

    # --- Find samples ---
    dinov2_root = str(cs_root / args.dinov2_dir / args.split)
    semantic_root = str(cs_root / args.semantic_dir / args.split)
    depth_root = str(cs_root / args.depth_dir / args.split)
    instance_root = str(cs_root / args.instance_dir / args.split)

    samples = find_samples(dinov2_root, semantic_root, depth_root, instance_root)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Processing {len(samples)} {args.split} images")

    # --- Process samples ---
    t0 = time.time()
    total_instances = 0
    gate_values = []

    for i, sample in enumerate(samples):
        stem = sample["stem"]
        city = sample["city"]

        # Load features
        sample_data = load_sample(
            sample,
            use_soft_logits=args.use_soft_logits,
        )

        # Run M2PR inference
        outputs = run_inference(model, sample_data, device)
        gate_values.append(outputs["gate"])

        # Load RGB image if CRF is requested
        rgb_image = None
        if args.use_crf:
            rgb_path = cs_root / "leftImg8bit" / args.split / city / f"{stem}_leftImg8bit.png"
            if rgb_path.exists():
                rgb_image = np.array(Image.open(rgb_path))

        # Post-process semantics
        semantic_map = postprocess_semantics(
            outputs["refined_sem"],
            target_h=args.target_h,
            target_w=args.target_w,
            use_crf=args.use_crf,
            rgb_image=rgb_image,
            crf_iters=args.crf_iters,
        )

        # Post-process instances
        instance_map = postprocess_instances(
            outputs["inst_embeddings"],
            outputs["refined_sem"],
            outputs["boundary"],
            target_h=args.target_h,
            target_w=args.target_w,
            max_instances=args.max_instances,
        )

        n_instances = int(instance_map.max())
        total_instances += n_instances

        if args.cups_format:
            # Save in CUPS flat directory format
            out_stem = f"{stem}_leftImg8bit"
            Image.fromarray(semantic_map.astype(np.uint8)).save(
                str(output_dir / f"{out_stem}_semantic.png")
            )
            Image.fromarray(instance_map.astype(np.uint16)).save(
                str(output_dir / f"{out_stem}_instance.png")
            )
            stats = compute_distributions(semantic_map, instance_map)
            torch.save(stats, str(output_dir / f"{out_stem}.pt"))
        else:
            # Save in city/stem directory structure
            city_dir = output_dir / city
            city_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(semantic_map.astype(np.uint8)).save(
                str(city_dir / f"{stem}_semantic.png")
            )
            Image.fromarray(instance_map.astype(np.uint16)).save(
                str(city_dir / f"{stem}_instance.png")
            )

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            avg_gate = np.mean(gate_values[-100:])
            print(
                f"  [{i+1}/{len(samples)}] {stem}: "
                f"{n_instances} instances | gate={avg_gate:.4f} | "
                f"{elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    avg_gate = np.mean(gate_values) if gate_values else 0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/max(len(samples),1):.2f}s/image)")
    print(f"Total: {len(samples)} images, {total_instances} thing instances")
    print(f"Average gate value: {avg_gate:.4f}")
    print(f"Output: {output_dir}")

    if args.cups_format:
        print(f"CUPS format: {len(samples) * 3} files")
        print(f"  Semantic: {{stem}}_leftImg8bit_semantic.png")
        print(f"  Instance: {{stem}}_leftImg8bit_instance.png")
        print(f"  Stats: {{stem}}_leftImg8bit.pt")


if __name__ == "__main__":
    main()
