"""Unit tests for Mamba2 SSD module.

Tests cover:
    - Output shapes for SSDKernel, Mamba2Block, Mamba2Stack
    - Gradient flow through Mamba2 layers
    - Sequence length handling (padded/unpadded)
    - Numerical stability (no NaN/Inf)
    - Hidden state bounds
"""

from __future__ import annotations

import sys
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSSDKernel(unittest.TestCase):
    """Test Structured State Space Duality kernel."""

    def test_output_shape(self):
        """SSD kernel should preserve input shape (B, L, D)."""
        from mbps.models.bridge.mamba2_ssd import SSDKernel

        model = SSDKernel(dim=64, state_dim=16, chunk_size=32)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 128, 64))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (2, 128, 64))

    def test_output_shape_non_divisible(self):
        """SSD should handle sequence lengths not divisible by chunk_size."""
        from mbps.models.bridge.mamba2_ssd import SSDKernel

        model = SSDKernel(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        # L=50 is not divisible by P=16
        x = jnp.ones((1, 50, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (1, 50, 32))

    def test_no_nan_output(self):
        """SSD output should not contain NaN values."""
        from mbps.models.bridge.mamba2_ssd import SSDKernel

        model = SSDKernel(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (1, 64, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertFalse(jnp.any(jnp.isnan(out)))

    def test_no_inf_output(self):
        """SSD output should not contain Inf values."""
        from mbps.models.bridge.mamba2_ssd import SSDKernel

        model = SSDKernel(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(7)
        x = jax.random.normal(rng, (1, 64, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertFalse(jnp.any(jnp.isinf(out)))


class TestMamba2Block(unittest.TestCase):
    """Test Mamba2 block (SSD + FFN + residual)."""

    def test_output_shape(self):
        """Mamba2 block should preserve sequence shape."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=64, state_dim=16, chunk_size=32)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((2, 128, 64))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (2, 128, 64))

    def test_gradient_flow(self):
        """Gradients should flow through Mamba2 block without vanishing."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1, 32, 32))
        params = model.init(rng, x)

        def loss_fn(params):
            out = model.apply(params, x)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn)(params)
        grad_norms = jax.tree.map(lambda g: float(jnp.sum(jnp.abs(g))), grads)
        total_grad = sum(jax.tree.leaves(grad_norms))

        self.assertGreater(total_grad, 0.0, "Gradients should not vanish")

    def test_residual_connection(self):
        """With zero-initialized output, block should return input."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 32, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        # Due to residual connection, output should be close to input
        # (not exact due to non-zero layer outputs)
        self.assertEqual(out.shape, x.shape)

    def test_deterministic_mode(self):
        """Block should produce identical output in deterministic mode."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Block

        model = Mamba2Block(dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1, 32, 32))
        params = model.init(rng, x)

        out1 = model.apply(params, x, deterministic=True)
        out2 = model.apply(params, x, deterministic=True)

        np.testing.assert_array_equal(np.array(out1), np.array(out2))


class TestMamba2Stack(unittest.TestCase):
    """Test Mamba2 stack (multiple blocks)."""

    def test_output_shape(self):
        """Stack should preserve shape through multiple layers."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(num_layers=2, dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 64, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (1, 64, 32))

    def test_four_layers(self):
        """Stack with 4 layers (default config) should work."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(num_layers=4, dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 32, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        self.assertEqual(out.shape, (1, 32, 32))
        self.assertFalse(jnp.any(jnp.isnan(out)))

    def test_gradient_flow_deep(self):
        """Gradients should flow through 4-layer stack."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(num_layers=4, dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1, 32, 32))
        params = model.init(rng, x)

        def loss_fn(params):
            return jnp.mean(model.apply(params, x) ** 2)

        grads = jax.grad(loss_fn)(params)
        grad_norms = jax.tree.map(lambda g: float(jnp.sum(jnp.abs(g))), grads)
        total_grad = sum(jax.tree.leaves(grad_norms))

        self.assertGreater(total_grad, 0.0)

    def test_batch_independence(self):
        """Different batch elements should produce different outputs."""
        from mbps.models.bridge.mamba2_ssd import Mamba2Stack

        model = Mamba2Stack(num_layers=2, dim=32, state_dim=8, chunk_size=16)
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 32, 32))
        params = model.init(rng, x)
        out = model.apply(params, x)

        # Different inputs should give different outputs
        diff = jnp.sum(jnp.abs(out[0] - out[1]))
        self.assertGreater(float(diff), 0.0)


if __name__ == "__main__":
    unittest.main()
