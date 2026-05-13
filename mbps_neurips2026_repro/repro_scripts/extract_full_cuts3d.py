#!/usr/bin/env python3
"""Full CutS3D Pseudo-Mask Extraction — CPU Mode with GCS Batching.

Runs the complete CutS3D extraction pipeline (NCut + LocalCut + Spatial
Confidence + CRF) on CPU to avoid TPU XLA compilation issues with Dinic's
MaxFlow algorithm.

Downloads Cityscapes data from GCS one city at a time, extracts pseudo-masks,
uploads results to GCS, and deletes local data to save disk space.

Usage:
    # Force CPU mode — MUST be set before JAX imports
    JAX_PLATFORMS=cpu python scripts/extract_full_cuts3d.py \
        --config configs/cityscapes_full.yaml

    # Resume from a specific city
    JAX_PLATFORMS=cpu python scripts/extract_full_cuts3d.py \
        --config configs/cityscapes_full.yaml --resume-from hamburg
"""

# Force CPU before JAX import
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbps.models.backbone.dino_vits8 import DINOViTS8
from mbps.models.backbone.weights_converter import convert_dino_weights
from mbps.models.instance.cuts3d import extract_pseudo_masks


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    import yaml
    default_path = Path(config_path).parent / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)
    if config_path != str(default_path) and os.path.exists(config_path):
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        def deep_merge(base, over):
            for k, v in over.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
        deep_merge(config, override)
    return config


# ---------------------------------------------------------------------------
# DINO backbone loading
# ---------------------------------------------------------------------------

DINO_VITS8_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
DINO_VITS8_CACHE = os.path.expanduser("~/.cache/dino/dino_vits8_pretrain.pth")


def download_dino_weights(url=DINO_VITS8_URL, cache_path=DINO_VITS8_CACHE):
    if os.path.exists(cache_path):
        print(f"  Using cached DINO weights: {cache_path}")
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"  Downloading DINO ViT-S/8 weights...")
    import urllib.request
    urllib.request.urlretrieve(url, cache_path)
    print(f"  Saved to: {cache_path}")
    return cache_path


def interpolate_pos_embed(pos_embed, target_num_patches):
    cls_token = pos_embed[:, :1, :]
    patch_embed = pos_embed[:, 1:, :]
    N_orig = patch_embed.shape[1]
    if N_orig == target_num_patches:
        return pos_embed
    D = patch_embed.shape[2]
    orig_size = int(N_orig ** 0.5)
    patch_embed_2d = patch_embed.reshape(1, orig_size, orig_size, D)
    target_h = int(np.sqrt(target_num_patches))
    while target_num_patches % target_h != 0:
        target_h -= 1
    target_w = target_num_patches // target_h
    patch_embed_interp = jax.image.resize(
        patch_embed_2d, (1, target_h, target_w, D), method="bilinear"
    )
    patch_embed_interp = patch_embed_interp.reshape(1, target_num_patches, D)
    return jnp.concatenate([cls_token, patch_embed_interp], axis=1)


def load_pretrained_backbone(backbone, rng, dummy_img, image_size):
    params = backbone.init(rng, dummy_img)
    weights_path = download_dino_weights()
    pretrained_params = convert_dino_weights(weights_path)
    target_patches = (image_size[0] // 8) * (image_size[1] // 8)
    orig_pos = pretrained_params["params"]["pos_embed"]
    if orig_pos.shape[1] - 1 != target_patches:
        print(f"  Interpolating pos_embed: {orig_pos.shape[1]-1} -> {target_patches} patches")
        pretrained_params["params"]["pos_embed"] = interpolate_pos_embed(
            orig_pos, target_patches
        )
    print("  Pretrained DINO weights loaded")
    return pretrained_params


# ---------------------------------------------------------------------------
# GCS utilities
# ---------------------------------------------------------------------------

GCS_BUCKET = "gs://mbps-panoptic"
GCS_CITYSCAPES = f"{GCS_BUCKET}/datasets/cityscapes"
GCS_PSEUDO_MASKS = f"{GCS_BUCKET}/pseudo_masks/cityscapes"


def gsutil_run(cmd, check=True):
    """Run a gsutil command, return success bool."""
    print(f"  $ {cmd}")
    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"  gsutil error: {result.stderr.strip()}")
    return result.returncode == 0


def list_cities(split="train"):
    """List all city directories in Cityscapes on GCS."""
    cmd = f"gsutil ls {GCS_CITYSCAPES}/leftImg8bit/{split}/"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error listing cities: {result.stderr}")
        return []
    cities = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip().rstrip("/")
        if line:
            city = line.split("/")[-1]
            if city and city != ".keep":
                cities.append(city)
    return sorted(cities)


def download_city(city, split, local_base, img_h, img_w):
    """Download one city's images + depth from GCS.

    Returns list of (image_path, depth_path, image_id) tuples.
    """
    local_img_dir = Path(local_base) / "leftImg8bit" / split / city
    local_depth_dir = Path(local_base) / "depth_zoedepth" / split / city
    local_img_dir.mkdir(parents=True, exist_ok=True)
    local_depth_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    gsutil_run(
        f"gsutil -m cp {GCS_CITYSCAPES}/leftImg8bit/{split}/{city}/*_leftImg8bit.png "
        f"{local_img_dir}/",
        check=False,
    )

    # Download depth
    gsutil_run(
        f"gsutil -m cp {GCS_CITYSCAPES}/depth_zoedepth/{split}/{city}/*.npy "
        f"{local_depth_dir}/",
        check=False,
    )

    # Build sample list
    samples = []
    for img_path in sorted(local_img_dir.glob("*_leftImg8bit.png")):
        base = img_path.name.replace("_leftImg8bit.png", "")
        depth_path = local_depth_dir / f"{base}.npy"
        samples.append((str(img_path), str(depth_path), base))

    return samples


def delete_city(local_base, city, split):
    """Delete downloaded city data from local disk."""
    for subdir in ["leftImg8bit", "depth_zoedepth"]:
        city_dir = Path(local_base) / subdir / split / city
        if city_dir.exists():
            shutil.rmtree(city_dir)


def upload_pseudo_masks(local_mask_dir, gcs_dir):
    """Upload pseudo-masks to GCS."""
    gsutil_run(
        f"gsutil -m cp {local_mask_dir}/*.npz {gcs_dir}/",
        check=False,
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_path, img_h, img_w):
    """Load and resize image to (img_h, img_w, 3) float32 [0,1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_w, img_h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_depth(depth_path, img_h, img_w):
    """Load and resize depth to (img_h, img_w) float32."""
    if os.path.exists(depth_path):
        depth = np.load(depth_path).astype(np.float32)
        depth_pil = Image.fromarray(depth)
        depth = np.array(depth_pil.resize((img_w, img_h), Image.BILINEAR))
        return depth
    return np.zeros((img_h, img_w), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full CutS3D Extraction — CPU Mode with GCS Batching"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/pseudo_masks_full")
    parser.add_argument("--local-data-dir", type=str, default="/tmp/cityscapes_batch")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from this city (skip earlier cities)")
    parser.add_argument("--max-instances", type=int, default=3)
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip GCS upload (keep masks local only)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't delete downloaded data after processing")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    img_h, img_w = data_cfg["image_size"]
    patch_h, patch_w = img_h // 8, img_w // 8
    K = patch_h * patch_w
    max_instances = args.max_instances

    print("\n" + "=" * 70)
    print("  CutS3D Full Extraction — CPU Mode")
    print("=" * 70)
    print(f"  JAX backend:     {jax.default_backend()}")
    print(f"  JAX devices:     {jax.devices()}")
    print(f"  Image size:      {img_h}x{img_w}")
    print(f"  Patch grid:      {patch_h}x{patch_w} = {K} tokens")
    print(f"  Max instances:   {max_instances}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Local data dir:  {args.local_data_dir}")
    print("=" * 70)
    sys.stdout.flush()

    assert jax.default_backend() == "cpu", (
        "Must run with JAX_PLATFORMS=cpu! "
        "Usage: JAX_PLATFORMS=cpu python scripts/extract_full_cuts3d.py ..."
    )

    # Initialize DINO backbone
    print("\nInitializing DINO backbone...")
    backbone = DINOViTS8(freeze=True)
    rng = jax.random.PRNGKey(42)
    dummy_img = jnp.zeros((1, img_h, img_w, 3))
    backbone_params = load_pretrained_backbone(
        backbone, rng, dummy_img, (img_h, img_w)
    )

    # JIT compile backbone forward pass
    print("  Compiling backbone forward pass...")
    sys.stdout.flush()

    @jax.jit
    def extract_features(bp, image):
        return backbone.apply(bp, image)

    # Warmup
    _ = extract_features(backbone_params, dummy_img)
    print("  Backbone JIT done")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List cities from GCS
    print("\nListing cities from GCS...")
    cities = list_cities("train")
    print(f"  Found {len(cities)} cities: {', '.join(cities)}")

    # Handle resume
    if args.resume_from:
        if args.resume_from in cities:
            idx = cities.index(args.resume_from)
            cities = cities[idx:]
            print(f"  Resuming from city: {args.resume_from} ({len(cities)} remaining)")
        else:
            print(f"  WARNING: City '{args.resume_from}' not found, starting from beginning")

    # Process each city
    total_images = 0
    total_masks = 0
    total_time = 0.0
    global_img_idx = 0

    # Count existing masks for proper indexing
    existing = list(output_dir.glob("masks_*.npz"))
    if existing:
        max_existing = max(
            int(p.stem.split("_")[1]) for p in existing
        )
        global_img_idx = max_existing + 1
        print(f"  Found {len(existing)} existing masks, starting from index {global_img_idx}")

    for city_idx, city in enumerate(cities):
        print(f"\n{'='*70}")
        print(f"  City {city_idx+1}/{len(cities)}: {city}")
        print(f"{'='*70}")
        sys.stdout.flush()

        # Step 1: Download city data from GCS
        print(f"  Downloading {city} from GCS...")
        t_download = time.time()
        samples = download_city(
            city, "train", args.local_data_dir, img_h, img_w
        )
        t_download = time.time() - t_download
        print(f"  Downloaded {len(samples)} images in {t_download:.1f}s")
        sys.stdout.flush()

        if len(samples) == 0:
            print(f"  WARNING: No images found for {city}, skipping")
            continue

        # Step 2: Extract pseudo-masks for each image
        city_masks = 0
        city_start = time.time()

        for s_idx, (img_path, depth_path, image_id) in enumerate(samples):
            img_start = time.time()

            # Load image and depth
            image = load_image(img_path, img_h, img_w)
            depth = load_depth(depth_path, img_h, img_w)

            # Extract DINO features
            image_batch = jnp.array(image[None])  # (1, H, W, 3)
            features_batch = extract_features(backbone_params, image_batch)
            features = np.array(features_batch[0])  # (K, 384)

            # Run full CutS3D extraction (CPU — no JIT)
            result = extract_pseudo_masks(
                jnp.array(features),
                jnp.array(depth),
                jnp.array(image),
                patch_h=patch_h,
                patch_w=patch_w,
                max_instances=max_instances,
                tau_ncut=0.0,
                tau_knn=0.115,
                k=10,
                sigma_gauss=3.0,
                beta=0.45,
                min_mask_size=0.02,
                sc_samples=6,
                use_crf=True,
            )

            n_valid = int(result.num_valid)
            city_masks += n_valid
            total_masks += n_valid

            # Save pseudo-masks
            np.savez_compressed(
                output_dir / f"masks_{global_img_idx:08d}.npz",
                masks=np.array(result.masks[:max(n_valid, 1)]),
                spatial_confidence=np.array(result.spatial_confidence[:max(n_valid, 1)]),
                scores=np.array(result.scores[:max(n_valid, 1)]),
                num_valid=n_valid,
                image_id=image_id,
            )

            global_img_idx += 1
            total_images += 1
            img_time = time.time() - img_start

            # Progress
            if s_idx == 0 or (s_idx + 1) % 10 == 0 or s_idx == len(samples) - 1:
                print(
                    f"  [{s_idx+1:>4d}/{len(samples)}] {image_id}: "
                    f"{n_valid} masks, {img_time:.1f}s"
                )
                sys.stdout.flush()

        city_time = time.time() - city_start
        total_time += city_time
        print(
            f"  City {city}: {len(samples)} images, {city_masks} masks, "
            f"{city_time:.1f}s ({city_time/len(samples):.1f}s/img)"
        )

        # Step 3: Upload pseudo-masks to GCS
        if not args.no_upload:
            print(f"  Uploading masks to GCS...")
            upload_pseudo_masks(str(output_dir), GCS_PSEUDO_MASKS)

        # Step 4: Delete local city data
        if not args.no_cleanup:
            print(f"  Cleaning up {city} local data...")
            delete_city(args.local_data_dir, city, "train")

        # ETA
        avg_per_img = total_time / total_images if total_images > 0 else 0
        remaining_cities = len(cities) - city_idx - 1
        est_remaining = avg_per_img * remaining_cities * len(samples)
        print(
            f"  Progress: {total_images} images total, "
            f"avg {avg_per_img:.1f}s/img, "
            f"~{est_remaining/3600:.1f}h remaining"
        )
        sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 70)
    print("  Extraction Complete")
    print("=" * 70)
    print(f"  Total images:    {total_images}")
    print(f"  Total masks:     {total_masks}")
    print(f"  Total time:      {total_time:.1f}s ({total_time/3600:.2f}h)")
    if total_images > 0:
        print(f"  Avg per image:   {total_time/total_images:.1f}s")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  GCS upload:      {GCS_PSEUDO_MASKS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
