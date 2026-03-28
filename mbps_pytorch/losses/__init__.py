"""Loss function modules for MBPS.

Combined loss:
    L_total = alpha·L_semantic + beta·L_instance + gamma·L_bridge + delta·L_consistency + epsilon·L_PQ
"""

from mbps_pytorch.losses.semantic_loss import SemanticLoss
from mbps_pytorch.losses.instance_loss import InstanceLoss
from mbps_pytorch.losses.bridge_loss import BridgeLoss
from mbps_pytorch.losses.consistency_loss import ConsistencyLoss
from mbps_pytorch.losses.pq_proxy_loss import PQProxyLoss
from mbps_pytorch.losses.gradient_balancing import GradientBalancer

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "BridgeLoss",
    "ConsistencyLoss",
    "PQProxyLoss",
    "GradientBalancer",
]
