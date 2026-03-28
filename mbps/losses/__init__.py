"""Loss function modules for MBPS.

Combined loss:
    L_total = α·L_semantic + β·L_instance + γ·L_bridge + δ·L_consistency + ε·L_PQ
"""

from mbps.losses.semantic_loss import SemanticLoss
from mbps.losses.instance_loss import InstanceLoss
from mbps.losses.bridge_loss import BridgeLoss
from mbps.losses.consistency_loss import ConsistencyLoss
from mbps.losses.pq_proxy_loss import PQProxyLoss
from mbps.losses.gradient_balancing import GradientBalancer

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "BridgeLoss",
    "ConsistencyLoss",
    "PQProxyLoss",
    "GradientBalancer",
]
