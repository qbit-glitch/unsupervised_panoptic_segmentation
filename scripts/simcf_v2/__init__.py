"""SIMCF-v2: Enhanced Semantic-Instance Mutual Consistency Filtering.

Five enhancement passes (D-H) extending the base SIMCF-ABC pipeline:
  D -- SDAIR: Spectral Depth-Aware Instance Refinement (splitting)
  E -- WBIM:  Wasserstein Barycentric Instance Merging
  F -- ITCBS: Info-Theoretic Cluster Boundary Sharpening
  G -- DCCPR: Depth-Conditioned Class Prior Regularization
  H -- GSID:  Grassmannian Subspace Instance Discrimination
"""

from .sdair import step_d
from .wbim import step_e
from .itcbs import step_f
from .dccpr import step_g, fit_gmm_1d
from .gsid import step_h, grassmannian_distance

__all__ = [
    "step_d",
    "step_e",
    "step_f",
    "step_g",
    "step_h",
    "fit_gmm_1d",
    "grassmannian_distance",
]
