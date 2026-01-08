#!/usr/bin/env python
"""
End-to-end training on 1% of all datasets using FULL MODEL with DINOv3.

Runs training on Synthetic, CLEVR, and COCO datasets sequentially.
Uses FullSpectralDiffusion with DINOv3 backbone.
"""
import subprocess
import sys
import os
import time

# Set HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    print(f"HF_TOKEN found (length: {len(HF_TOKEN)})")
else:
    print("WARNING: HF_TOKEN not set! DINOv3 may fail to load.")
    print("Set it with: export HF_TOKEN=your_token")

DATASETS = [
    {"name": "synthetic", "epochs": 5, "num_slots": 8},
    {"name": "clevr", "epochs": 5, "num_slots": 11},
    {"name": "coco", "epochs": 5, "num_slots": 15},
]

SUBSET_PCT = 0.01  # 1%

def run_training(dataset: str, epochs: int, num_slots: int):
    """Run training for a single dataset using full model."""
    print("=" * 70)
    print(f"TRAINING: {dataset.upper()} - {int(SUBSET_PCT * 100)}% for {epochs} epochs")
    print("Using FULL MODEL with DINOv3 backbone")
    print("=" * 70)
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--num-slots", str(num_slots),
        "--batch-size", "4",  # Smaller batch for DINOv3
        "--lr", "5e-5",       # Lower LR for large backbone
        "--full-model",       # Use FullSpectralDiffusion
        "--use-dino",         # Enable DINOv3 backbone
        "--mixed-precision",  # Use bfloat16
        "--device", "mps",
        "--fast-dev-run",     # Quick test: 1% of data
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Set environment for HF token
    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, env=env)
        elapsed = time.time() - start_time
        print(f"\n✅ {dataset.upper()} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {dataset.upper()} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {dataset.upper()} interrupted by user")
        return False


def main():
    print("=" * 70)
    print("END-TO-END TRAINING ON 1% OF ALL DATASETS")
    print("FULL MODEL with DINOv3 BACKBONE")
    print("=" * 70)
    print(f"Datasets: {[d['name'] for d in DATASETS]}")
    print(f"Subset: {int(SUBSET_PCT * 100)}%")
    print("=" * 70)
    print()
    
    results = {}
    total_start = time.time()
    
    for dataset_config in DATASETS:
        success = run_training(
            dataset_config["name"],
            dataset_config["epochs"],
            dataset_config["num_slots"],
        )
        results[dataset_config["name"]] = success
        print()
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
