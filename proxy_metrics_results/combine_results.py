import json
from pathlib import Path

results_dir = Path("/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/proxy_metrics_results")

variants = [
    ("k80", "k80_proxies.json", "CAUSE-TR k=80 (baseline)"),
    ("adapter_v3", "adapter_v3_proxies.json", "+ DCFA v3"),
    ("simcf_a", "simcf_a_proxies.json", "+ SIMCF-A"),
    ("simcf_abc", "simcf_abc_proxies.json", "+ SIMCF-ABC"),
    ("dcfa_simcf_abc", "dcfa_simcf_abc_proxies.json", "+ DCFA + SIMCF-ABC"),
]

print("| Variant | SIC (%) | Fragments/img | Stuff Contam. (%) | LER |")
print("|---------|---------|---------------|-------------------|-----|")
for key, filename, label in variants:
    d = json.load(open(results_dir / filename))
    sic = d["SIC_mean"]
    frag = d["fragments_per_image_mean"]
    contam = d["stuff_contamination_mean"]
    ler = d["LER_mean"]
    print(f"| {label:25s} | {sic:6.2f} | {frag:6.2f} ± {d['fragments_per_image_std']:5.2f} | {contam:6.2f} ± {d['stuff_contamination_std']:5.2f} | {ler:.2f} |")

# Also save as JSON
combined = {}
for key, filename, label in variants:
    combined[key] = json.load(open(results_dir / filename))
    combined[key]["label"] = label

with open(results_dir / "combined_proxies.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"\nSaved combined results to {results_dir / 'combined_proxies.json'}")
