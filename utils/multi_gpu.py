"""
Multi-GPU Utilities for SpectralDiffusion

Provides shared functionality for multi-GPU training across all training scripts.
Supports both CUDA (with DataParallel) and MPS (Apple Silicon).

Usage:
    from utils.multi_gpu import setup_device, wrap_model, get_model_state

    # Setup device and multi-GPU configuration
    use_multi_gpu, gpu_ids = setup_device(args)
    
    # Wrap model with DataParallel if needed
    model = wrap_model(model, use_multi_gpu, gpu_ids)
    
    # Get model state dict (handles DataParallel wrapper)
    state_dict = get_model_state(model)
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


def setup_device(args) -> Tuple[bool, Optional[List[int]]]:
    """
    Setup device and multi-GPU configuration.
    
    Args:
        args: Argparse namespace with 'device', 'multi_gpu', and 'gpu_ids' attributes
        
    Returns:
        use_multi_gpu: Whether multi-GPU training is enabled
        gpu_ids: List of GPU IDs to use (None for non-CUDA devices)
    """
    use_multi_gpu = False
    gpu_ids = None
    
    # Get multi_gpu and gpu_ids from args (with defaults for backward compatibility)
    multi_gpu = getattr(args, 'multi_gpu', False)
    gpu_ids_str = getattr(args, 'gpu_ids', None)
    
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            num_gpus = torch.cuda.device_count()
            print(f"Found {num_gpus} CUDA GPU(s)")
            
            # Parse GPU IDs if specified
            if gpu_ids_str:
                gpu_ids = [int(x) for x in gpu_ids_str.split(',')]
                print(f"Using specified GPUs: {gpu_ids}")
            else:
                gpu_ids = list(range(num_gpus))
            
            # Enable multi-GPU if requested and multiple GPUs available
            if multi_gpu and len(gpu_ids) > 1:
                use_multi_gpu = True
                print(f"Multi-GPU enabled with {len(gpu_ids)} GPUs: {gpu_ids}")
            elif multi_gpu and len(gpu_ids) == 1:
                print("Multi-GPU requested but only 1 GPU available, using single GPU")
            
            # Set primary device
            torch.cuda.set_device(gpu_ids[0])
            
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            args.device = "cpu"
        else:
            print("Using MPS (Apple Silicon)")
    else:
        print("Using CPU")
            
    return use_multi_gpu, gpu_ids


def wrap_model(model: nn.Module, use_multi_gpu: bool, gpu_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Wrap model with DataParallel if multi-GPU is enabled.
    
    Args:
        model: PyTorch model
        use_multi_gpu: Whether to use DataParallel
        gpu_ids: List of GPU IDs for DataParallel
        
    Returns:
        Wrapped model (or original if not multi-GPU)
    """
    if use_multi_gpu and gpu_ids is not None:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped with DataParallel on GPUs: {gpu_ids}")
    return model


def get_model_state(model: nn.Module) -> dict:
    """
    Get model state dict, handling DataParallel wrapper.
    
    Args:
        model: PyTorch model (possibly wrapped in DataParallel)
        
    Returns:
        Model state dict
    """
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()


def get_model_module(model: nn.Module) -> nn.Module:
    """
    Get the underlying model module (unwrap DataParallel if needed).
    
    Args:
        model: PyTorch model (possibly wrapped in DataParallel)
        
    Returns:
        Underlying model module
    """
    if hasattr(model, 'module'):
        return model.module
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters, handling DataParallel wrapper.
    
    Args:
        model: PyTorch model (possibly wrapped in DataParallel)
        
    Returns:
        Number of trainable parameters
    """
    model_for_params = get_model_module(model)
    return sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)


def add_multi_gpu_args(parser):
    """
    Add multi-GPU arguments to an argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument("--multi-gpu", action="store_true",
                       help="Enable multi-GPU training with DataParallel (CUDA only)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                       help="Comma-separated GPU IDs to use, e.g., '0,1' (default: all available)")
    return parser


def print_gpu_info():
    """Print GPU information for debugging."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"GPU Information ({num_gpus} device(s))")
        print('='*60)
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({total_mem:.1f} GB)")
        print('='*60 + '\n')
    elif torch.backends.mps.is_available():
        print("\nMPS (Apple Silicon) available")
    else:
        print("\nNo GPU available, using CPU")
