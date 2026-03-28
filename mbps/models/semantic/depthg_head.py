"""DepthG Semantic Segmentation Head.

Ported from visinf/depthg. Implements a 3-layer MLP projector on top
of frozen DINO features to produce a 90-dimensional semantic code space.

Architecture: Linear(384→384) → ReLU → Linear(384→384) → ReLU → Linear(384→90)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class DepthGHead(nn.Module):
    """DepthG semantic segmentation head.

    Maps DINO ViT-S/8 features (384-dim) to a 90-dimensional semantic
    code space via a 3-layer MLP.

    Attributes:
        input_dim: Input feature dimension (384 for DINO ViT-S/8).
        hidden_dim: Hidden layer dimension.
        code_dim: Output code space dimension (90 per DepthG paper).
        use_batch_norm: Whether to use BatchNorm (False = use LayerNorm).
    """

    input_dim: int = 384
    hidden_dim: int = 384
    code_dim: int = 90
    use_batch_norm: bool = False

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Compute semantic codes from DINO features.

        Args:
            features: DINO patch features of shape (B, N, 384).
            deterministic: If True, disable dropout.

        Returns:
            Semantic codes of shape (B, N, 90).
        """
        x = features

        # Layer 1: 384 → 384
        x = nn.Dense(self.hidden_dim, name="linear1")(x)
        x = nn.LayerNorm(name="norm1")(x)
        x = jax.nn.relu(x)

        # Layer 2: 384 → 384
        x = nn.Dense(self.hidden_dim, name="linear2")(x)
        x = nn.LayerNorm(name="norm2")(x)
        x = jax.nn.relu(x)

        # Layer 3: 384 → 90 (code space)
        x = nn.Dense(self.code_dim, name="linear3")(x)

        return x

    def get_cluster_assignments(
        self,
        codes: jnp.ndarray,
        num_clusters: int = 27,
    ) -> jnp.ndarray:
        """Get hard cluster assignments from soft codes via argmax.

        For actual clustering, use k-means on the code space.

        Args:
            codes: Semantic codes of shape (B, N, 90).
            num_clusters: Number of target clusters.

        Returns:
            Cluster assignments of shape (B, N).
        """
        # Simple approach: use first `num_clusters` dims as logits
        # In practice, k-means is used on the full 90-dim codes
        logits = codes[:, :, :num_clusters]
        return jnp.argmax(logits, axis=-1)


class DepthGHeadSpatial(nn.Module):
    """DepthG head that operates on spatial (2D) feature maps.

    Same MLP but accepts and returns spatial format features.

    Attributes:
        hidden_dim: Hidden layer dimension.
        code_dim: Output code space dimension.
    """

    hidden_dim: int = 384
    code_dim: int = 90

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Compute semantic codes from spatial features.

        Args:
            features: Spatial features of shape (B, H, W, 384).
            deterministic: If True, disable dropout.

        Returns:
            Semantic codes of shape (B, H, W, 90).
        """
        b, h, w, c = features.shape

        # Flatten spatial dims
        x = jnp.reshape(features, (b, h * w, c))

        # Apply MLP
        x = nn.Dense(self.hidden_dim, name="linear1")(x)
        x = nn.LayerNorm(name="norm1")(x)
        x = jax.nn.relu(x)

        x = nn.Dense(self.hidden_dim, name="linear2")(x)
        x = nn.LayerNorm(name="norm2")(x)
        x = jax.nn.relu(x)

        x = nn.Dense(self.code_dim, name="linear3")(x)

        # Reshape back to spatial
        return jnp.reshape(x, (b, h, w, self.code_dim))
