import json
from pathlib import Path

results_dir = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/proxy_metrics_results")

# Train evaluations
variants = [
    ("k80", "train_k80_full.json", "CAUSE-TR k=80 (baseline)"),
    ("adapter_v3", "train_adapter_v3_full.json", "+ DCFA v3"),
    ("simcf_abc", "train_simcf_abc_full.json", "+ SIMCF-ABC"),
    ("dcfa_simcf_abc", "train_dcfa_simcf_abc_full.json", "+ DCFA + SIMCF-ABC"),
    ("dcfa_simcf_depthpro", "train_dcfa_simcf_depthpro.json", "+ DCFA + SIMCF-ABC + DepthPro"),
]

print("| Method | PQ | PQ_st | PQ_th | mIoU | Pixel Acc |")
print("|--------|-----|-------|-------|------|-----------|")
for key, filename, label in variants:
    d = json.load(open(results_dir / filename))
    p = d["panoptic"]
    s = d["semantic"]
    print(f"| {label:35s} | {p['all']['PQ']:.2f} | {p['stuff']['PQ']:.2f} | {p['things']['PQ']:.2f} | {s['mIoU']:.2f} | {s['pixel_accuracy']:.2f} |")

# Also save as JSON
combined = {}
for key, filename, label in variants:
    combined[key] = json.load(open(results_dir / filename))
    combined[key]["label"] = label

with open(results_dir / "combined_train_evals.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"\nSaved combined results to {results_dir / 'combined_train_evals.json'}")
