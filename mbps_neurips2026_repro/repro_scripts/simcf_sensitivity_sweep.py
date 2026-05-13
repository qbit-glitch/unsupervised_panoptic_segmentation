"""SIMCF threshold sensitivity sweep on full Cityscapes train set.

Runs SIMCF with 3 (sim_threshold, sigma_threshold) configurations and reports
pseudo-label PQ/SQ/RQ/mIoU for each, evaluated against Cityscapes train GT.

Configurations:
  baseline:  sim=0.85, eta=2.5  (paper)
  tighter:   sim=0.75, eta=2.0  (more conservative merge + void)
  looser:    sim=0.95, eta=3.0  (less aggressive merge + void)

Usage:
    python scripts/simcf_sensitivity_sweep.py
"""

from pathlib import Path
import sys
import time
import json
import subprocess

PROJECT_ROOT = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation")
CR = Path("/Users/qbit-glitch/Desktop/datasets/cityscapes")
PYTHON = "/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python"

INPUT_DIR = CR / "cups_pseudo_labels_adapter_V3_tau020"
CENTROIDS = CR / "pseudo_semantic_adapter_V3_k80" / "kmeans_centroids.npz"
SWEEP_OUT = PROJECT_ROOT / "results" / "simcf_sensitivity_sweep"
SWEEP_OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("tighter",  0.75, 2.0),
    ("baseline", 0.85, 2.5),
    ("looser",   0.95, 3.0),
]


def run(cmd, log_path):
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    print(f"    done in {(time.time()-t0)/60:.1f} min  (exit {proc.returncode})")
    return proc.returncode


def main():
    results = {}
    for name, sim, eta in CONFIGS:
        out_dir = SWEEP_OUT / f"simcf_{name}_sim{sim}_eta{eta}"
        eval_json = SWEEP_OUT / f"eval_{name}_sim{sim}_eta{eta}.json"
        log_simcf = SWEEP_OUT / f"simcf_{name}.log"
        log_eval = SWEEP_OUT / f"eval_{name}.log"

        # Run SIMCF
        rc = run([
            PYTHON, str(PROJECT_ROOT / "scripts/refine_simcf.py"),
            "--input_dir", str(INPUT_DIR),
            "--output_dir", str(out_dir),
            "--centroids_path", str(CENTROIDS),
            "--cityscapes_root", str(CR),
            "--steps", "A,B,C",
            "--sim_threshold", str(sim),
            "--sigma_threshold", str(eta),
        ], log_simcf)
        if rc != 0:
            print(f"!!! SIMCF failed for {name}; skipping eval")
            continue

        # Evaluate pseudo-label quality on train split
        rc = run([
            PYTHON, str(PROJECT_ROOT / "scripts/evaluate_pseudolabel_quality.py"),
            "--pseudo_dir", str(out_dir),
            "--cityscapes_root", str(CR),
            "--centroids_path", str(CENTROIDS),
            "--split", "train",
            "--use_hungarian",
            "--output", str(eval_json),
        ], log_eval)
        if rc != 0:
            print(f"!!! Eval failed for {name}")
            continue

        # Read result
        try:
            d = json.load(open(eval_json))
            pq = d.get("PQ", d.get("metrics", {}).get("PQ", "?"))
            sq = d.get("SQ", d.get("metrics", {}).get("SQ", "?"))
            rq = d.get("RQ", d.get("metrics", {}).get("RQ", "?"))
            miou = d.get("mIoU", d.get("metrics", {}).get("mIoU", "?"))
            results[name] = {"sim": sim, "eta": eta, "PQ": pq, "SQ": sq, "RQ": rq, "mIoU": miou}
            print(f"    {name}: PQ={pq} SQ={sq} RQ={rq} mIoU={miou}")
        except Exception as e:
            print(f"    {name}: eval JSON parse failed: {e}")

    summary = SWEEP_OUT / "summary.json"
    with open(summary, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary: {summary}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
