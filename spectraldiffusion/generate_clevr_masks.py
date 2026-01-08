#!/usr/bin/env python
"""
Generate approximate per-object masks for CLEVR from scene descriptions.

This creates simple circular/elliptical masks based on 3D object positions.
Not perfect but sufficient for computing ARI during training.

Usage:
    python generate_clevr_masks.py --clevr-dir ../datasets/CLEVR_v1.0 --split train --limit 1000
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def project_3d_to_2d(pos_3d, image_size=(320, 240)):
    """
    Approximate 3D to 2D projection for CLEVR.
    
    CLEVR uses specific camera parameters - this is an approximation
    based on typical rendering setup.
    """
    # CLEVR camera is at fixed position looking down at table
    # Approximate projection
    x, y, z = pos_3d
    
    # Camera parameters (approximated from CLEVR rendering)
    cam_distance = 10.0
    fov_scale = 35.0  # Approximate field of view scaling
    
    # Simple perspective projection
    img_w, img_h = image_size
    cx, cy = img_w / 2, img_h / 2
    
    # Project to image coordinates
    px = cx + fov_scale * x
    py = cy - fov_scale * y  # Flip y for image coordinates
    
    return px, py


def get_object_radius(obj, image_size=(320, 240)):
    """Estimate object radius in pixels based on size and distance."""
    size_map = {
        'small': 0.7,
        'large': 1.4,
    }
    base_size = size_map.get(obj.get('size', 'small'), 1.0)
    
    # Approximate radius in pixels
    radius = 15 * base_size
    return radius


def generate_masks_from_scene(scene, image_size=(320, 240)):
    """
    Generate approximate masks for all objects in a scene.
    
    Returns:
        masks: numpy array of shape [K, H, W] where K is number of objects
    """
    objects = scene.get('objects', [])
    if not objects:
        return None
    
    H, W = image_size[1], image_size[0]  # (W, H) -> (H, W)
    num_objects = len(objects)
    masks = np.zeros((num_objects + 1, H, W), dtype=np.float32)  # +1 for background
    
    # Create coordinate grids
    yy, xx = np.mgrid[:H, :W]
    
    # Generate mask for each object
    for i, obj in enumerate(objects):
        # Get 3D position
        pos_3d = obj.get('3d_coords', [0, 0, 0])
        if isinstance(pos_3d, list) and len(pos_3d) >= 3:
            px, py = project_3d_to_2d(pos_3d, (W, H))
        else:
            # Use 2D pixel coords if available
            pixel_coords = obj.get('pixel_coords', [W/2, H/2])
            px, py = pixel_coords[0], pixel_coords[1]
        
        # Get object radius
        radius = get_object_radius(obj, (W, H))
        
        # Create circular mask
        dist = np.sqrt((xx - px)**2 + (yy - py)**2)
        obj_mask = (dist < radius).astype(np.float32)
        
        # Soft edges
        soft_mask = np.clip(1.0 - (dist - radius) / 5.0, 0, 1)
        soft_mask = soft_mask * (dist < radius + 5)
        
        masks[i + 1] = np.maximum(soft_mask, obj_mask)
    
    # Background is everything not covered by objects
    masks[0] = np.clip(1.0 - masks[1:].sum(axis=0), 0, 1)
    
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevr-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 240])
    args = parser.parse_args()
    
    clevr_dir = Path(args.clevr_dir)
    scenes_file = clevr_dir / "scenes" / f"CLEVR_{args.split}_scenes.json"
    images_dir = clevr_dir / "images" / args.split
    masks_dir = clevr_dir / "masks" / args.split
    
    print(f"CLEVR dir: {clevr_dir}")
    print(f"Scenes file: {scenes_file}")
    
    # Load scenes
    if not scenes_file.exists():
        print(f"Error: Scenes file not found at {scenes_file}")
        return
    
    with open(scenes_file) as f:
        scenes_data = json.load(f)
    
    scenes = scenes_data['scenes']
    print(f"Loaded {len(scenes)} scenes")
    
    # Create masks directory
    masks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving masks to: {masks_dir}")
    
    # Generate masks
    limit = args.limit or len(scenes)
    for i, scene in enumerate(tqdm(scenes[:limit], desc="Generating masks")):
        image_filename = scene.get('image_filename', f"CLEVR_{args.split}_{i:06d}.png")
        mask_filename = image_filename.replace('.png', '_mask.npy')
        mask_path = masks_dir / mask_filename
        
        # Generate masks
        masks = generate_masks_from_scene(scene, tuple(args.image_size))
        
        if masks is not None:
            np.save(mask_path, masks)
    
    print(f"Generated {min(limit, len(scenes))} mask files in {masks_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
