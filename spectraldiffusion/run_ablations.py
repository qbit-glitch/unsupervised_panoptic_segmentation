"""
Ablation Study Runner for SpectralDiffusion

Runs all ablation experiments defined in configs/ablations/

Usage:
    python run_ablations.py --category A        # Run core architecture ablations (A1-A5)
    python run_ablations.py --category B        # Run backbone ablations (B1-B4)
    python run_ablations.py --ablation A1       # Run specific ablation A1
    python run_ablations.py --all               # Run all ablations
    python run_ablations.py --list              # List available ablations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import yaml
import time

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# Ablation Registry
ABLATION_REGISTRY = {
    # Category A: Core Architecture
    "A1": {"file": "A1_init_mode.yaml", "name": "Spectral vs Random vs Learned Init"},
    "A2": {"file": "A2_attention_type.yaml", "name": "Mamba vs Attention Types"},
    "A3": {"file": "A3_decoder_type.yaml", "name": "Diffusion vs MLP vs CNN Decoder"},
    "A4": {"file": "A4_identifiability.yaml", "name": "Identifiability Prior"},
    "A5": {"file": "A5_pruning.yaml", "name": "Adaptive Pruning vs Fixed K"},
    
    # Category B: Backbone
    "B1": {"file": "B1_backbone_size.yaml", "name": "DINOv3 Model Size"},
    "B2": {"file": "B2_dino_version.yaml", "name": "DINOv3 vs DINOv2"},
    "B3": {"file": "B3_feature_dim.yaml", "name": "Feature Dimension"},
    "B4": {"file": "B4_backbone_finetune.yaml", "name": "Frozen vs Fine-tuned Backbone"},
    
    # Category C: Spectral Initialization
    "C1": {"file": "C1_multiscale.yaml", "name": "Multi-scale vs Single Scale"},
    "C2": {"file": "C2_slots_per_scale.yaml", "name": "Slots per Scale"},
    "C3": {"file": "C3_knn_neighbors.yaml", "name": "k-NN Neighbors"},
    "C4": {"file": "C4_power_iterations.yaml", "name": "Power Iteration Steps"},
    
    # Category D: Mamba-Slot Attention
    "D1": {"file": "D1_num_iterations.yaml", "name": "Slot Attention Iterations"},
    "D2": {"file": "D2_state_dim.yaml", "name": "Mamba State Dimension"},
    "D3": {"file": "D3_expand_factor.yaml", "name": "Mamba Expand Factor"},
    "D4": {"file": "D4_bidirectional.yaml", "name": "Bidirectional Mamba"},
    
    # Category E: Diffusion Decoder
    "E1": {"file": "E1_diffusion_steps.yaml", "name": "Diffusion Steps"},
    "E2": {"file": "E2_latent_dim.yaml", "name": "Latent Dimension"},
    "E3": {"file": "E3_noise_schedule.yaml", "name": "Noise Schedule"},
    "E4": {"file": "E4_unet_depth.yaml", "name": "U-Net Depth"},
    "E5": {"file": "E5_slot_conditioning.yaml", "name": "Slot Conditioning Method"},
    
    # Category F: Loss Weights
    "F1": {"file": "F1_lambda_diff.yaml", "name": "Diffusion Loss Weight"},
    "F2": {"file": "F2_lambda_spec.yaml", "name": "Spectral Loss Weight"},
    "F3": {"file": "F3_lambda_ident.yaml", "name": "Identifiability Loss Weight"},
    
    # Category G: Efficiency
    "G1": {"file": "G1_throughput.yaml", "name": "Inference Throughput"},
    "G2": {"file": "G2_memory.yaml", "name": "Memory Usage"},
    "G3": {"file": "G3_training_time.yaml", "name": "Training Time"},
    "G4": {"file": "G4_model_params.yaml", "name": "Model Parameters"},
}


def list_ablations():
    """Print available ablations."""
    print("\n" + "=" * 70)
    print("Available Ablation Studies")
    print("=" * 70)
    
    categories = {
        "A": "Core Architecture",
        "B": "Backbone",
        "C": "Spectral Initialization",
        "D": "Mamba-Slot Attention",
        "E": "Diffusion Decoder",
        "F": "Loss Weights",
        "G": "Efficiency",
    }
    
    current_category = None
    for ablation_id, info in sorted(ABLATION_REGISTRY.items()):
        cat = ablation_id[0]
        if cat != current_category:
            current_category = cat
            print(f"\n### Category {cat}: {categories.get(cat, 'Unknown')}")
            print("-" * 50)
        
        print(f"  {ablation_id}: {info['name']}")
    
    print("\n" + "=" * 70)
    print("Run with: python run_ablations.py --ablation <ID>")
    print("Or:       python run_ablations.py --category <A-G>")
    print("=" * 70 + "\n")


def load_ablation_config(ablation_id: str) -> Dict[str, Any]:
    """Load ablation configuration file."""
    if ablation_id not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation: {ablation_id}. Use --list to see available.")
    
    config_file = ABLATION_REGISTRY[ablation_id]["file"]
    config_path = Path(__file__).parent / "configs" / "ablations" / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Parse multi-document YAML
    with open(config_path) as f:
        content = f.read()
    
    # Split by --- and parse each document
    variants = {}
    for doc in content.split("---"):
        doc = doc.strip()
        if not doc:
            continue
        
        parsed = yaml.safe_load(doc)
        if parsed:
            # Find the variant name (first key)
            variant_name = list(parsed.keys())[0]
            variants[variant_name] = parsed[variant_name]
    
    return variants


def create_model_from_config(config: Dict, device: str):
    """Create model from configuration."""
    from train import create_model
    
    class Args:
        pass
    
    args = Args()
    
    # Model settings
    model_cfg = config.get('model', {})
    args.backbone = model_cfg.get('backbone', 'base')
    args.num_slots = model_cfg.get('num_slots', 12)
    args.use_mamba = model_cfg.get('use_mamba', True)
    args.use_diffusion = model_cfg.get('use_diffusion', True)
    args.init_mode = model_cfg.get('init_mode', 'spectral')
    
    # Data settings
    data_cfg = config.get('data', {})
    args.image_size = data_cfg.get('image_size', [128, 128])
    args.dataset = data_cfg.get('dataset', 'synthetic')
    args.data_dir = data_cfg.get('data_dir', './datasets')
    args.batch_size = data_cfg.get('batch_size', 16)
    args.num_workers = data_cfg.get('num_workers', 4)
    
    args.device = device
    
    return create_model(args), args


def run_single_variant(
    variant_name: str,
    config: Dict,
    output_dir: Path,
    args,
) -> Dict[str, Any]:
    """Run a single ablation variant."""
    print(f"\n  Running variant: {variant_name}")
    
    device = args.device
    
    try:
        model, model_args = create_model_from_config(config, device)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'variant': variant_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
        
        # Run training if not just benchmarking
        if not config.get('benchmark', {}).get('measure_fps', False):
            from train import create_dataloader, train_epoch, evaluate
            
            # Training
            train_loader = create_dataloader(model_args, split="train")
            val_loader = create_dataloader(model_args, split="val")
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get('training', {}).get('lr', 1e-4),
                weight_decay=config.get('training', {}).get('weight_decay', 1e-5),
            )
            
            epochs = min(args.epochs, config.get('training', {}).get('epochs', 30))
            
            best_ari = 0
            for epoch in range(1, epochs + 1):
                train_metrics = train_epoch(model, train_loader, optimizer, None, model_args, epoch)
                
                if epoch % 5 == 0 or epoch == epochs:
                    val_metrics = evaluate(model, val_loader, model_args)
                    if val_metrics.get('val_ari', 0) > best_ari:
                        best_ari = val_metrics['val_ari']
                
                if args.fast_dev_run:
                    break
            
            results['best_ari'] = best_ari
            results['final_loss'] = train_metrics.get('train_loss', 0)
        
        # Measure FPS
        if config.get('benchmark', {}).get('measure_fps', False):
            fps, latency = measure_fps(model, model_args, device)
            results['fps'] = fps
            results['latency_ms'] = latency
        
        # Measure memory
        if config.get('benchmark', {}).get('measure_memory', False):
            peak_memory = measure_memory(model, model_args, device)
            results['peak_memory_gb'] = peak_memory
        
        results['status'] = 'success'
        
    except Exception as e:
        results = {
            'variant': variant_name,
            'status': 'error',
            'error': str(e),
        }
        print(f"    Error: {e}")
    
    return results


def measure_fps(model, args, device: str) -> tuple:
    """Measure inference FPS."""
    model.eval()
    
    dummy_input = torch.randn(1, 3, *args.image_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, return_loss=False)
    
    # Measure
    num_runs = 50
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input, return_loss=False)
    
    elapsed = time.time() - start
    fps = num_runs / elapsed
    latency = (elapsed / num_runs) * 1000  # ms
    
    return fps, latency


def measure_memory(model, args, device: str) -> float:
    """Measure peak memory usage."""
    if device == "mps":
        # MPS doesn't have easy memory tracking
        return 0.0
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        dummy_input = torch.randn(args.batch_size, 3, *args.image_size, device=device)
        
        with torch.no_grad():
            _ = model(dummy_input, return_loss=False)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        return peak_memory
    
    return 0.0


def run_ablation(ablation_id: str, output_dir: Path, args) -> pd.DataFrame:
    """Run all variants of an ablation study."""
    print(f"\n{'='*60}")
    print(f"Ablation {ablation_id}: {ABLATION_REGISTRY[ablation_id]['name']}")
    print(f"{'='*60}")
    
    variants = load_ablation_config(ablation_id)
    
    results = []
    for variant_name, config in variants.items():
        result = run_single_variant(
            variant_name,
            config,
            output_dir / ablation_id,
            args,
        )
        result['ablation'] = ablation_id
        results.append(result)
    
    return pd.DataFrame(results)


def run_category(category: str, output_dir: Path, args) -> pd.DataFrame:
    """Run all ablations in a category."""
    category_ablations = [
        aid for aid in ABLATION_REGISTRY.keys()
        if aid.startswith(category)
    ]
    
    if not category_ablations:
        print(f"No ablations found for category {category}")
        return pd.DataFrame()
    
    all_results = []
    for ablation_id in category_ablations:
        results = run_ablation(ablation_id, output_dir, args)
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


def generate_report(results: pd.DataFrame, output_dir: Path, title: str):
    """Generate markdown report from results."""
    report_path = output_dir / "ablation_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# SpectralDiffusion Ablation Study: {title}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Summary table
        f.write("## Results Summary\n\n")
        
        for ablation_id in results['ablation'].unique():
            ablation_data = results[results['ablation'] == ablation_id]
            
            f.write(f"### {ablation_id}: {ABLATION_REGISTRY.get(ablation_id, {}).get('name', '')}\n\n")
            
            # Select columns to show
            cols = ['variant']
            if 'best_ari' in ablation_data.columns:
                cols.append('best_ari')
            if 'fps' in ablation_data.columns:
                cols.append('fps')
            if 'total_params' in ablation_data.columns:
                cols.append('total_params')
            if 'status' in ablation_data.columns:
                cols.append('status')
            
            # Table header
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            
            # Table rows
            for _, row in ablation_data.iterrows():
                values = []
                for col in cols:
                    val = row.get(col, 'N/A')
                    if isinstance(val, float):
                        if col == 'best_ari':
                            val = f"{val:.4f}"
                        elif col == 'fps':
                            val = f"{val:.1f}"
                        else:
                            val = f"{val:.2f}"
                    elif isinstance(val, int) and col == 'total_params':
                        val = f"{val:,}"
                    values.append(str(val))
                f.write("| " + " | ".join(values) + " |\n")
            
            f.write("\n")
        
        # Best performers
        f.write("## Key Findings\n\n")
        
        if 'best_ari' in results.columns:
            best = results.loc[results['best_ari'].idxmax()]
            f.write(f"- **Best ARI**: {best['variant']} ({best['best_ari']:.4f})\n")
        
        if 'fps' in results.columns:
            fastest = results.loc[results['fps'].idxmax()]
            f.write(f"- **Fastest**: {fastest['variant']} ({fastest['fps']:.1f} FPS)\n")
    
    print(f"\nReport saved to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SpectralDiffusion Ablation Runner")
    
    parser.add_argument("--ablation", type=str, default=None,
                       help="Specific ablation ID (e.g., A1, B2)")
    parser.add_argument("--category", type=str, default=None,
                       choices=["A", "B", "C", "D", "E", "F", "G"],
                       help="Run all ablations in category")
    parser.add_argument("--all", action="store_true",
                       help="Run all ablations")
    parser.add_argument("--list", action="store_true",
                       help="List available ablations")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Max epochs per variant")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to use")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="Quick test run")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list:
        list_ablations()
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SpectralDiffusion Ablation Runner")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    all_results = []
    title = ""
    
    if args.ablation:
        # Run specific ablation
        results = run_ablation(args.ablation, output_dir, args)
        all_results.append(results)
        title = f"Ablation {args.ablation}"
    
    elif args.category:
        # Run category
        results = run_category(args.category, output_dir, args)
        all_results.append(results)
        title = f"Category {args.category}"
    
    elif args.all:
        # Run all
        for category in ["A", "B", "C", "D", "E", "F", "G"]:
            results = run_category(category, output_dir, args)
            if not results.empty:
                all_results.append(results)
        title = "All Categories"
    
    else:
        print("Please specify --ablation, --category, --all, or --list")
        return
    
    # Combine and save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Save CSV
        csv_path = output_dir / "results.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Generate report
        generate_report(combined, output_dir, title)
    
    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
