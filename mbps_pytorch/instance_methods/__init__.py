"""Instance decomposition methods for unsupervised panoptic segmentation.

All methods implement the same interface:

    instances = method_fn(semantic, depth, thing_ids, ..., features=None)
    -> List[(mask: np.ndarray(H,W) bool, class_id: int, score: float)]

Some experimental methods require optional packages.  The registry keeps those
imports lazy-tolerant so one missing optional dependency does not break unrelated
ablation scripts.
"""

from __future__ import annotations


def _missing_method(name: str, exc: Exception):
    def _fn(*_args, **_kwargs):
        raise ImportError(
            f"Instance method {name!r} is unavailable because an optional "
            f"dependency failed to import: {exc}"
        ) from exc

    return _fn


def _try_import(name: str, module: str, attr: str):
    try:
        mod = __import__(f"{__name__}.{module}", fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:  # optional dependency guard
        return _missing_method(name, exc)


sobel_cc_instances = _try_import("sobel_cc", "sobel_cc", "sobel_cc_instances")
morse_flow_instances = _try_import("morse", "morse_flow", "morse_flow_instances")
tda_instances = _try_import("tda", "tda_persistence", "tda_instances")
sinkhorn_instances = _try_import("ot", "optimal_transport", "sinkhorn_instances")
mumford_shah_instances = _try_import("mumford_shah", "mumford_shah", "mumford_shah_instances")
contrastive_instances = _try_import("contrastive", "contrastive_embed", "contrastive_instances")
learned_merge_instances = _try_import("learned_merge", "learned_merge", "learned_merge_instances")
feature_edge_cc_instances = _try_import("feature_edge_cc", "feature_edge_cc", "feature_edge_cc_instances")
joint_ncut_instances = _try_import("joint_ncut", "joint_ncut", "joint_ncut_instances")
learned_edge_cc_instances = _try_import("learned_edge_cc", "learned_edge_cc", "learned_edge_cc_instances")
plane_decomp_instances = _try_import("plane_decomp", "plane_decomp", "plane_decomp_instances")
adaptive_edge_instances = _try_import("adaptive_edge", "adaptive_edge", "adaptive_edge_instances")
depth_stratified_instances = _try_import("depth_stratified", "depth_stratified", "depth_stratified_instances")
picl_instances = _try_import("picl", "picl_embed", "picl_instances")
instances_from_edge_probs = _try_import(
    "superpixel_affinity",
    "superpixel_affinity",
    "instances_from_edge_probs",
)

METHODS = {
    "sobel_cc": sobel_cc_instances,
    "morse": morse_flow_instances,
    "tda": tda_instances,
    "ot": sinkhorn_instances,
    "mumford_shah": mumford_shah_instances,
    "contrastive": contrastive_instances,
    "learned_merge": learned_merge_instances,
    "feature_edge_cc": feature_edge_cc_instances,
    "joint_ncut": joint_ncut_instances,
    "learned_edge_cc": learned_edge_cc_instances,
    "plane_decomp": plane_decomp_instances,
    "adaptive_edge": adaptive_edge_instances,
    "depth_stratified": depth_stratified_instances,
    "picl": picl_instances,
}

__all__ = [
    "METHODS",
    "sobel_cc_instances",
    "learned_merge_instances",
    "instances_from_edge_probs",
]
