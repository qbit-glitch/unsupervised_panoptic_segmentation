"""
Mac-compatible Mamba2 + GatedDeltaNet (Pure PyTorch, no CUDA/Triton).

Based on the official state-spaces/mamba v2.3.0 and NVlabs/GatedDeltaNet
implementations, with all Triton/CUDA/FLA kernels replaced by pure PyTorch.
Works on CPU, MPS (Apple Silicon), and CUDA.
"""

from .mamba2 import Mamba2
from .gated_delta_net import GatedDeltaNet
from .norm import RMSNormGated
from .ssd import ssd_minimal_discrete, ssd_chunk_scan_combined
from .vision import VisionMamba2, CrossModalMamba2, SCAN_MODES, LAYER_TYPES

__all__ = [
    "Mamba2", "GatedDeltaNet", "RMSNormGated",
    "ssd_minimal_discrete", "ssd_chunk_scan_combined",
    "VisionMamba2", "CrossModalMamba2",
    "SCAN_MODES", "LAYER_TYPES",
]
