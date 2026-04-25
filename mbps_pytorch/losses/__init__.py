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

# New A* paper-inspired losses (Approach B)
from mbps_pytorch.losses.confidence_filtering import (
    select_confident_samples,
    avg_entropy,
    weighted_entropy,
    confidence_weighted_loss,
)
from mbps_pytorch.losses.feature_consistency import (
    FeatureConsistencyLoss,
    PredictionConsistencyLoss,
)
from mbps_pytorch.losses.mae_regularizer import (
    CLSTokenMAELoss,
    mae_regularizer_loss,
)

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "BridgeLoss",
    "ConsistencyLoss",
    "PQProxyLoss",
    "GradientBalancer",
    # Confidence filtering (LoRA-TTT)
    "select_confident_samples",
    "avg_entropy",
    "weighted_entropy",
    "confidence_weighted_loss",
    # Feature consistency (Uni-UVPT)
    "FeatureConsistencyLoss",
    "PredictionConsistencyLoss",
    # MAE regularizer (LoRA-TTT)
    "CLSTokenMAELoss",
    "mae_regularizer_loss",
]
