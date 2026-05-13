"""Instance decomposition methods for unsupervised panoptic segmentation.

All methods implement the same interface:
    instances = method_fn(semantic, depth, thing_ids, ..., features=None)
    -> List[(mask: np.ndarray(H,W) bool, class_id: int, score: float)]
"""

from .sobel_cc import sobel_cc_instances
from .morse_flow import morse_flow_instances
from .tda_persistence import tda_instances
from .optimal_transport import sinkhorn_instances
from .mumford_shah import mumford_shah_instances
from .contrastive_embed import contrastive_instances
from .learned_merge import learned_merge_instances
from .feature_edge_cc import feature_edge_cc_instances
from .joint_ncut import joint_ncut_instances
from .learned_edge_cc import learned_edge_cc_instances
from .plane_decomp import plane_decomp_instances
from .adaptive_edge import adaptive_edge_instances
from .depth_stratified import depth_stratified_instances
from .picl_embed import picl_instances

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
