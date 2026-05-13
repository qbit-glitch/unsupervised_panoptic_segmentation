#!/usr/bin/env python3
"""Upload pseudo-labels and weights to HuggingFace Hub.

Resumable: tracks progress in logs/hf_upload_checkpoint.txt.
Re-run safely — skips completed steps.

Usage:
    python -u scripts/upload_hf_dataset.py
"""
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

REPO_ID = "qbit-glitch/cityscapes-pseudo-labels"
DATASETS_DIR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
WEIGHTS_DIR = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights")
CHECKPOINT_FILE = Path("logs/hf_upload_checkpoint.txt")


def load_checkpoint() -> set:
    """Load completed steps from checkpoint file."""
    if CHECKPOINT_FILE.exists():
        return set(CHECKPOINT_FILE.read_text().strip().split("\n"))
    return set()


def save_checkpoint(step: str, completed: set) -> None:
    """Mark a step as done."""
    completed.add(step)
    CHECKPOINT_FILE.write_text("\n".join(sorted(completed)))


def retry(fn, description: str, max_attempts: int = 10):
    """Retry a function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            print(f"  {description} attempt {attempt+1}/{max_attempts} failed: {type(e).__name__}: {e}")
            if attempt < max_attempts - 1:
                wait = min(120, 15 * (attempt + 1))
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def main():
    api = HfApi()
    completed = load_checkpoint()

    # Step 1: Create repo
    if "repo_created" not in completed:
        print("[1/5] Creating repo...")
        retry(lambda: create_repo(REPO_ID, repo_type="dataset", exist_ok=True), "create_repo")
        save_checkpoint("repo_created", completed)
        print("  OK")
    else:
        print("[1/5] Repo — already done")

    # Step 2: README
    if "readme" not in completed:
        print("[2/5] Uploading README...")
        readme_path = Path("/tmp/hf_readme.md")
        readme_path.write_text(create_readme())
        retry(lambda: api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
        ), "readme")
        save_checkpoint("readme", completed)
        print("  OK")
    else:
        print("[2/5] README — already done")

    # Step 3: pseudo_semantic_raw_k80
    if "semantic_k80" not in completed:
        sem_path = DATASETS_DIR / "pseudo_semantic_raw_k80"
        file_count = sum(1 for _ in sem_path.rglob("*") if _.is_file())
        print(f"[3/5] Uploading pseudo_semantic_raw_k80 ({file_count} files)...")
        retry(lambda: api.upload_folder(
            folder_path=str(sem_path),
            path_in_repo="pseudo_semantic_raw_k80",
            repo_id=REPO_ID,
            repo_type="dataset",
            ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
        ), "semantic_k80")
        save_checkpoint("semantic_k80", completed)
        print("  OK")
    else:
        print("[3/5] semantic_k80 — already done")

    # Step 4: cups_pseudo_labels_depthpro_tau020 in batches
    depthpro_path = DATASETS_DIR / "cups_pseudo_labels_depthpro_tau020"
    all_files = sorted([f for f in depthpro_path.rglob("*") if f.is_file()
                       and not f.name.startswith(".") and f.suffix != ".pyc"])
    total = len(all_files)
    batch_size = 500
    num_batches = (total + batch_size - 1) // batch_size
    print(f"[4/5] Uploading depthpro_tau020 ({total} files, {num_batches} batches)...")

    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        step_name = f"depthpro_batch_{batch_num}"

        if step_name in completed:
            print(f"  Batch {batch_num}/{num_batches} — already done")
            continue

        batch = all_files[i:i + batch_size]
        operations = [
            CommitOperationAdd(
                path_in_repo=f"cups_pseudo_labels_depthpro_tau020/{f.relative_to(depthpro_path)}",
                path_or_fileobj=str(f),
            )
            for f in batch
        ]

        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} files)...")
        retry(lambda ops=operations, bn=batch_num: api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Add depthpro_tau020 batch {bn}/{num_batches}",
        ), f"batch_{batch_num}")
        save_checkpoint(step_name, completed)
        print(f"  Batch {batch_num}/{num_batches} OK")

    print("  All batches done")

    # Step 5: Weights
    weight_files = [
        ("dinov3_vitb16_official.pth", "dinov3"),
        ("cups.ckpt", "cups_ckpt"),
    ]
    print("[5/5] Uploading weights...")
    for wf, step_key in weight_files:
        step_name = f"weight_{step_key}"
        if step_name in completed:
            print(f"  weights/{wf} — already done")
            continue

        wpath = WEIGHTS_DIR / wf
        if not wpath.exists():
            print(f"  SKIP weights/{wf} — not found")
            continue

        actual_mb = wpath.stat().st_size / (1024 * 1024)
        print(f"  Uploading weights/{wf} ({actual_mb:.0f} MB)...")
        retry(lambda p=str(wpath), name=wf: api.upload_file(
            path_or_fileobj=p,
            path_in_repo=f"weights/{name}",
            repo_id=REPO_ID,
            repo_type="dataset",
        ), f"weight_{wf}")
        save_checkpoint(step_name, completed)
        print(f"  OK")

    print(f"\n{'='*60}")
    print(f"All uploads complete!")
    print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"\nDownload (no auth needed):")
    print(f"  huggingface-cli download {REPO_ID} --repo-type dataset --local-dir ./data/")
    return 0


def create_readme():
    return """---
license: apache-2.0
task_categories:
  - image-segmentation
tags:
  - panoptic-segmentation
  - unsupervised
  - cityscapes
  - pseudo-labels
  - depth-guided
pretty_name: Cityscapes Unsupervised Panoptic Pseudo-Labels
size_categories:
  - 1K<n<10K
---

# Cityscapes Unsupervised Panoptic Pseudo-Labels

Pseudo-labels for unsupervised panoptic segmentation on Cityscapes.

## Download

```bash
# Everything
huggingface-cli download qbit-glitch/cityscapes-pseudo-labels --repo-type dataset --local-dir ./data/

# Just pseudo-labels
huggingface-cli download qbit-glitch/cityscapes-pseudo-labels --repo-type dataset --local-dir ./data/ \\
  --include "pseudo_semantic_raw_k80/*" "cups_pseudo_labels_depthpro_tau020/*"

# Just weights
huggingface-cli download qbit-glitch/cityscapes-pseudo-labels --repo-type dataset --local-dir ./data/ \\
  --include "weights/*"
```

## Contents

| Directory | Description | Files |
|-----------|-------------|-------|
| `pseudo_semantic_raw_k80/` | Overclustered k=80 semantic labels | ~3.5K PNGs + centroids.npz |
| `cups_pseudo_labels_depthpro_tau020/` | CUPS-format (DepthPro tau=0.20) | ~9K files |
| `weights/dinov3_vitb16_official.pth` | DINOv3 ViT-B/16 backbone | 327MB |
| `weights/cups.ckpt` | CUPS Cascade Mask R-CNN | 916MB |
"""


if __name__ == "__main__":
    sys.exit(main())
