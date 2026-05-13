"""Generate paper figures for MBPS.

Produces publication-quality figures for the NeurIPS submission:
    - Figure 1: Architecture overview (placeholder)
    - Figure 2: Qualitative results grid
    - Figure 3: Ablation bar charts
    - Figure 4: Per-class PQ breakdown
    - Figure 5: Training curve (loss + PQ vs epoch)
    - Figure 6: Stuff-Things classifier accuracy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_results(results_dir: str) -> Dict:
    """Load evaluation results from JSON files.

    Args:
        results_dir: Path to results directory.

    Returns:
        Dict mapping experiment names to result dicts.
    """
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)

    return results


def generate_ablation_table(
    results: Dict,
    output_path: str,
) -> None:
    """Generate LaTeX ablation table.

    Args:
        results: Dict of experiment results.
        output_path: Path to save LaTeX table.
    """
    ablation_order = [
        "full_model",
        "no_mamba",
        "no_depth_cond",
        "no_bicms",
        "no_consistency",
        "oracle_stuff_things",
    ]

    metrics = ["PQ", "PQ_Th", "PQ_St", "SQ", "RQ", "mIoU", "AP"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on Cityscapes val set.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Experiment & " + " & ".join(metrics) + r" \\",
        r"\midrule",
    ]

    for exp_name in ablation_order:
        if exp_name not in results:
            continue

        r = results[exp_name]
        row = exp_name.replace("_", " ").title()
        for m in metrics:
            val = r.get(m, r.get(m.lower(), 0.0))
            if isinstance(val, (list, tuple)):
                row += f" & {np.mean(val):.1f}"
            else:
                row += f" & {val:.1f}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved ablation table to {output_path}")


def generate_training_curve(
    log_path: str,
    output_path: str,
) -> None:
    """Generate training loss and PQ vs epoch plot.

    Args:
        log_path: Path to training log JSON.
        output_path: Path to save figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curve figure")
        return

    if not os.path.exists(log_path):
        print(f"Training log not found: {log_path}")
        return

    with open(log_path) as f:
        logs = json.load(f)

    epochs = [entry["epoch"] for entry in logs]
    losses = [entry.get("loss", 0) for entry in logs]
    pqs = [entry.get("pq", 0) for entry in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, losses, "b-", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Training Loss")
    ax1.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Phase B")
    ax1.axvline(x=40, color="gray", linestyle=":", alpha=0.5, label="Phase C")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PQ curve
    ax2.plot(epochs, pqs, "r-", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PQ")
    ax2.set_title("Panoptic Quality")
    ax2.axvline(x=20, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=40, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(y=27.8, color="green", linestyle="--", alpha=0.5, label="CUPS baseline")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {output_path}")


def generate_ablation_bars(
    results: Dict,
    output_path: str,
) -> None:
    """Generate ablation bar chart comparing PQ across experiments.

    Args:
        results: Dict of experiment results.
        output_path: Path to save figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping ablation bars")
        return

    experiments = []
    pq_values = []

    for name, r in sorted(results.items()):
        experiments.append(name.replace("_", "\n"))
        pq_values.append(r.get("PQ", r.get("pq", 0.0)))

    if not experiments:
        print("No results found for ablation bars")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(experiments)), pq_values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, fontsize=8)
    ax.set_ylabel("PQ")
    ax.set_title("Ablation Study: Panoptic Quality")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, pq_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation bars to {output_path}")


def generate_qualitative_grid(
    images_dir: str,
    output_path: str,
    num_samples: int = 4,
) -> None:
    """Generate qualitative results grid.

    Args:
        images_dir: Directory with visualization images.
        output_path: Path to save figure.
        num_samples: Number of samples to include.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("matplotlib/PIL not available, skipping qualitative grid")
        return

    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Look for visualization images
    image_files = sorted(images_path.glob("*.png"))[:num_samples]

    if not image_files:
        print("No visualization images found")
        return

    fig, axes = plt.subplots(1, len(image_files), figsize=(4 * len(image_files), 4))
    if len(image_files) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, image_files):
        img = np.array(Image.open(img_path))
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--results", type=str, default="results/",
        help="Path to results directory"
    )
    parser.add_argument(
        "--output", type=str, default="figures/",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--training_log", type=str, default="results/training_log.json",
        help="Path to training log"
    )
    parser.add_argument(
        "--images_dir", type=str, default="results/visualizations/",
        help="Directory with visualization images"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load results
    results = load_results(args.results)
    print(f"Loaded results for {len(results)} experiments")

    # Generate figures
    generate_ablation_table(
        results,
        os.path.join(args.output, "ablation_table.tex"),
    )
    generate_training_curve(
        args.training_log,
        os.path.join(args.output, "training_curve.pdf"),
    )
    generate_ablation_bars(
        results,
        os.path.join(args.output, "ablation_bars.pdf"),
    )
    generate_qualitative_grid(
        args.images_dir,
        os.path.join(args.output, "qualitative_grid.pdf"),
    )

    print(f"\nAll figures saved to {args.output}")


if __name__ == "__main__":
    main()
