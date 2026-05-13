#!/usr/bin/env python3
"""Batch evaluate all existing local pseudo-label directories."""

import json
import subprocess
import sys
from pathlib import Path

DATASET_ROOT = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
RESULTS_DIR = PROJECT_ROOT / "results" / "auto_matrix"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Map existing pseudo-label dirs to their experiment configs
EXISTING_PSEUDOLABELS = {
    # (semantic_tag, instance_tag, dcfa, simcf) -> directory
    ("cause_k80", "depthpro_tau020", False, ""): DATASET_ROOT / "cups_pseudo_labels_depthpro_tau020",
    ("cause_k80", "depthpro_tau020", True, ""): DATASET_ROOT / "cups_pseudo_labels_adapter_V3_tau020",
    ("cause_k80", "depthpro_tau020", False, "ABC"): DATASET_ROOT / "cups_pseudo_labels_simcf_abc",
    ("cause_k80", "depthpro_tau020", True, "ABC"): DATASET_ROOT / "cups_pseudo_labels_dcfa_simcf_abc",
    ("cause_k80", "depthpro_tau020", False, "A"): DATASET_ROOT / "cups_pseudo_labels_simcf_a",
}

EVAL_SCRIPT = PROJECT_ROOT / "mbps_pytorch" / "evaluate_pseudolabels.py"


def evaluate_one(exp_id: str, pseudo_dir: Path, output_json: Path) -> bool:
    """Run evaluation on one pseudo-label directory."""
    if output_json.exists():
        print(f"SKIP {exp_id}: already evaluated")
        return True

    print(f"EVAL {exp_id}: {pseudo_dir}")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--semantic_dir", str(pseudo_dir),
        "--gt_dir", str(DATASET_ROOT / "gtFine" / "val"),
        "--num_classes", "19",
        "--image_size", "512", "1024",
        "--output", str(output_json),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"OK {exp_id}")
            return True
        else:
            print(f"FAIL {exp_id}: rc={result.returncode}")
            print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"ERROR {exp_id}: {e}")
        return False


def main():
    results = []
    for (sem, inst, dcfa, simcf), pseudo_dir in EXISTING_PSEUDOLABELS.items():
        parts = [sem, inst]
        if dcfa:
            parts.append("dcfa")
        if simcf:
            parts.append(f"simcf{simcf}")
        exp_id = "_".join(parts)

        output_json = RESULTS_DIR / f"{exp_id}.json"
        success = evaluate_one(exp_id, pseudo_dir, output_json)
        results.append({"exp_id": exp_id, "success": success, "path": str(output_json)})

    summary = RESULTS_DIR / "batch_summary.json"
    with open(summary, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["success"])
    print(f"\nDone: {ok}/{len(results)} evaluations completed")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
