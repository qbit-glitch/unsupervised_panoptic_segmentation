"""Stage-4 dead-class recovery components (α + γ').

Exports:
    box_counting_dimension, per_class_fractal_dim, fracal_calibrate
    identify_rare_modes, hierarchical_merge
"""

from __future__ import annotations

from mbps_pytorch.stage4.fracal_calibration import (
    box_counting_dimension,
    fracal_calibrate,
    per_class_fractal_dim,
)
from mbps_pytorch.stage4.hierarchical_merge import (
    hierarchical_merge,
    identify_rare_modes,
)

__all__ = [
    "box_counting_dimension",
    "fracal_calibrate",
    "hierarchical_merge",
    "identify_rare_modes",
    "per_class_fractal_dim",
]
