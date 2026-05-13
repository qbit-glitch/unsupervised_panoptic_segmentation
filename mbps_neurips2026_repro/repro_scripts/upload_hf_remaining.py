#!/usr/bin/env python3
"""Upload remaining depthpro files + weights to HuggingFace."""
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "qbit-glitch/cityscapes-pseudo-labels"
api = HfApi()

def upload_with_retry(fn, desc, max_attempts=10):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            print(f"  {desc} attempt {attempt+1}/{max_attempts} failed: {e}")
            if attempt < max_attempts - 1:
                wait = min(120, 15 * (attempt + 1))
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

# Step 1: Upload remaining depthpro files
src = Path("/tmp/hf_remaining_depthpro")
if src.exists():
    files = sorted([f for f in src.iterdir() if f.is_symlink() or f.is_file()])
    print(f"[1] Uploading {len(files)} remaining depthpro files...")
    upload_with_retry(
        lambda: api.upload_folder(
            folder_path=str(src),
            path_in_repo="cups_pseudo_labels_depthpro_tau020",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add depthpro_tau020 remaining files (batches 15-18)",
        ),
        "depthpro_remaining"
    )
    print("[1] DONE")
else:
    print("[1] No remaining depthpro files found, skipping")

# Step 2: Upload weights
weights_dir = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights")
weight_files = [
    ("dinov3_vitb16_official.pth", "weights/dinov3_vitb16_official.pth"),
    ("cups.ckpt", "weights/cups.ckpt"),
]

for wf, repo_path in weight_files:
    wpath = weights_dir / wf
    if not wpath.exists():
        print(f"[W] {wf} not found, skipping")
        continue
    mb = wpath.stat().st_size / (1024 * 1024)
    print(f"[W] Uploading {wf} ({mb:.0f} MB)...")
    upload_with_retry(
        lambda p=str(wpath), rp=repo_path: api.upload_file(
            path_or_fileobj=p,
            path_in_repo=rp,
            repo_id=REPO_ID,
            repo_type="dataset",
        ),
        f"weight_{wf}"
    )
    print(f"[W] {wf} DONE")

print("\nAll uploads complete!")
print(f"https://huggingface.co/datasets/{REPO_ID}")
