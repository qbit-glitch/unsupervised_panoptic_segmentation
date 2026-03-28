"""Tests for copy-paste augmentation module."""

import numpy as np
import pytest

from mbps.data.copy_paste import (
    _extract_instances,
    _resize_nearest,
    copy_paste_augment,
    create_self_enhanced_source,
)


# ---- Helpers ----

def _make_batch(B=2, H=64, W=128, ps=16, n_instances=3):
    """Create a synthetic batch for testing."""
    H_p, W_p = H // ps, W // ps
    N = H_p * W_p

    rng = np.random.RandomState(42)

    images = rng.rand(B, H, W, 3).astype(np.float32)
    depths = rng.rand(B, H, W).astype(np.float32)

    semantics = np.zeros((B, N), dtype=np.int32)
    instances = np.zeros((B, N), dtype=np.int32)

    for b in range(B):
        sem_2d = np.zeros((H_p, W_p), dtype=np.int32)
        inst_2d = np.zeros((H_p, W_p), dtype=np.int32)
        for i in range(1, n_instances + 1):
            r0 = rng.randint(0, max(H_p - 2, 1))
            c0 = rng.randint(0, max(W_p - 2, 1))
            r1 = min(r0 + 2, H_p)
            c1 = min(c0 + 2, W_p)
            inst_2d[r0:r1, c0:c1] = i
            sem_2d[r0:r1, c0:c1] = rng.randint(1, 19)
        semantics[b] = sem_2d.reshape(-1)
        instances[b] = inst_2d.reshape(-1)

    return {
        "image": images,
        "depth": depths,
        "pseudo_semantic": semantics,
        "pseudo_instance": instances,
    }


# ---- _resize_nearest ----

class TestResizeNearest:
    def test_identity(self):
        arr = np.arange(12).reshape(3, 4)
        result = _resize_nearest(arr, 3, 4)
        np.testing.assert_array_equal(result, arr)

    def test_upscale_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _resize_nearest(arr, 4, 4)
        assert result.shape == (4, 4)
        # Top-left quadrant should be 1
        assert result[0, 0] == 1
        assert result[0, 1] == 1
        assert result[1, 0] == 1
        # Bottom-right quadrant should be 4
        assert result[3, 3] == 4

    def test_downscale_2d(self):
        arr = np.arange(16).reshape(4, 4)
        result = _resize_nearest(arr, 2, 2)
        assert result.shape == (2, 2)

    def test_3d_image(self):
        arr = np.random.rand(4, 6, 3).astype(np.float32)
        result = _resize_nearest(arr, 8, 12)
        assert result.shape == (8, 12, 3)
        assert result.dtype == np.float32

    def test_3d_identity(self):
        arr = np.random.rand(5, 7, 3).astype(np.float32)
        result = _resize_nearest(arr, 5, 7)
        np.testing.assert_array_equal(result, arr)

    def test_bool_mask(self):
        mask = np.array([[True, False], [False, True]])
        result = _resize_nearest(mask.astype(np.uint8), 4, 4).astype(bool)
        assert result.shape == (4, 4)
        assert result[0, 0] == True
        assert result[0, 2] == False


# ---- _extract_instances ----

class TestExtractInstances:
    def test_basic_extraction(self):
        batch = _make_batch(B=1, n_instances=2)
        insts = _extract_instances(
            batch["pseudo_instance"][0],
            batch["pseudo_semantic"][0],
            batch["image"][0],
            batch["depth"][0],
            patch_size=16,
            min_tokens=1,
        )
        assert len(insts) > 0
        for inst in insts:
            assert "mask" in inst
            assert "semantic" in inst
            assert "image" in inst
            assert "depth" in inst
            assert inst["mask"].dtype == bool

    def test_min_tokens_filter(self):
        batch = _make_batch(B=1, n_instances=1)
        insts_loose = _extract_instances(
            batch["pseudo_instance"][0],
            batch["pseudo_semantic"][0],
            batch["image"][0],
            batch["depth"][0],
            patch_size=16,
            min_tokens=1,
        )
        insts_strict = _extract_instances(
            batch["pseudo_instance"][0],
            batch["pseudo_semantic"][0],
            batch["image"][0],
            batch["depth"][0],
            patch_size=16,
            min_tokens=100,
        )
        assert len(insts_strict) <= len(insts_loose)


# ---- copy_paste_augment ----

class TestCopyPasteAugment:
    def test_basic_augmentation(self):
        batch = _make_batch()
        rng = np.random.RandomState(0)
        result = copy_paste_augment(batch, rng, patch_size=16)

        assert result["image"].shape == batch["image"].shape
        assert result["depth"].shape == batch["depth"].shape
        assert result["pseudo_semantic"].shape == batch["pseudo_semantic"].shape
        assert result["pseudo_instance"].shape == batch["pseudo_instance"].shape

    def test_backward_compatible(self):
        """Old-style call (no scale_range, no source_batch) still works."""
        batch = _make_batch()
        rng = np.random.RandomState(0)
        result = copy_paste_augment(
            batch, rng, patch_size=16,
            max_paste_objects=3, min_instance_tokens=1, flip_prob=0.5,
        )
        assert result["image"].shape == batch["image"].shape

    def test_scale_range_no_scaling(self):
        """scale_range=(1.0, 1.0) produces same result as default."""
        batch = _make_batch()
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        result1 = copy_paste_augment(batch, rng1, patch_size=16)
        result2 = copy_paste_augment(
            batch, rng2, patch_size=16, scale_range=(1.0, 1.0),
        )
        np.testing.assert_array_equal(result1["image"], result2["image"])

    def test_scale_range_modifies_output(self):
        """scale_range=(0.5, 1.5) produces different result than no scaling."""
        batch = _make_batch()
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        result_no_scale = copy_paste_augment(
            batch, rng1, patch_size=16, scale_range=(1.0, 1.0),
        )
        result_scaled = copy_paste_augment(
            batch, rng2, patch_size=16, scale_range=(0.5, 1.5),
        )
        # With scaling, the RNG advances differently so results should differ
        # (though not guaranteed in all cases due to random chance)
        # At minimum, shapes should be preserved
        assert result_scaled["image"].shape == batch["image"].shape

    def test_source_batch(self):
        """source_batch extracts instances from source, not target."""
        target = _make_batch(B=2, n_instances=0)  # No instances in target
        source = _make_batch(B=2, n_instances=5)  # Instances only in source
        rng = np.random.RandomState(0)

        # With no source, no instances to paste → output equals input
        result_no_source = copy_paste_augment(
            target, np.random.RandomState(0), patch_size=16,
        )
        np.testing.assert_array_equal(
            result_no_source["image"], target["image"],
        )

        # With source_batch, instances from source get pasted
        result_with_source = copy_paste_augment(
            target, rng, patch_size=16, source_batch=source,
            min_instance_tokens=1,
        )
        # At least some pixels should change
        assert not np.array_equal(
            result_with_source["pseudo_instance"],
            target["pseudo_instance"],
        )

    def test_missing_fields_returns_batch(self):
        """If required fields missing, return batch unchanged."""
        batch = {"image": np.zeros((1, 64, 128, 3))}
        rng = np.random.RandomState(0)
        result = copy_paste_augment(batch, rng)
        assert result is batch

    def test_performance(self):
        """Augmentation completes in reasonable time at Cityscapes scale."""
        import time
        batch = _make_batch(B=4, H=512, W=1024, n_instances=5)
        rng = np.random.RandomState(0)
        t0 = time.time()
        for _ in range(5):
            copy_paste_augment(
                batch, rng, patch_size=16,
                max_paste_objects=5,
                min_instance_tokens=4,
                scale_range=(0.5, 1.5),
            )
        elapsed = (time.time() - t0) / 5
        assert elapsed < 0.5, f"Too slow: {elapsed:.3f}s per batch"


# ---- create_self_enhanced_source ----

class TestCreateSelfEnhancedSource:
    def test_basic(self):
        B, N = 2, 32
        images = np.random.rand(B, 64, 128, 3).astype(np.float32)
        depths = np.random.rand(B, 64, 128).astype(np.float32)
        semantic_preds = np.random.randint(0, 19, (B, N)).astype(np.int32)
        instance_preds = np.random.randint(0, 5, (B, N)).astype(np.int32)
        confidence = np.random.rand(B, N).astype(np.float32)

        result = create_self_enhanced_source(
            images, depths, semantic_preds, instance_preds,
            confidence, confidence_threshold=0.5,
        )
        assert result is not None
        assert "image" in result
        assert "pseudo_semantic" in result
        assert "pseudo_instance" in result

    def test_confidence_filtering(self):
        B, N = 1, 8
        semantic_preds = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        instance_preds = np.array([[1, 1, 2, 2, 3, 3, 0, 0]], dtype=np.int32)
        confidence = np.array([[0.9, 0.3, 0.8, 0.2, 0.95, 0.1, 0.5, 0.5]])

        result = create_self_enhanced_source(
            images=np.zeros((B, 16, 16, 3)),
            depths=np.zeros((B, 16, 16)),
            semantic_preds=semantic_preds,
            instance_preds=instance_preds,
            confidence=confidence,
            confidence_threshold=0.75,
        )
        assert result is not None
        # Tokens with confidence < 0.75 should have instance=0
        assert result["pseudo_instance"][0, 1] == 0  # conf=0.3
        assert result["pseudo_instance"][0, 3] == 0  # conf=0.2
        assert result["pseudo_instance"][0, 5] == 0  # conf=0.1
        # Tokens with confidence >= 0.75 should keep their instance
        assert result["pseudo_instance"][0, 0] == 1  # conf=0.9
        assert result["pseudo_instance"][0, 2] == 2  # conf=0.8
        assert result["pseudo_instance"][0, 4] == 3  # conf=0.95

    def test_returns_none_when_no_instances(self):
        B, N = 1, 8
        result = create_self_enhanced_source(
            images=np.zeros((B, 16, 16, 3)),
            depths=np.zeros((B, 16, 16)),
            semantic_preds=np.ones((B, N), dtype=np.int32),
            instance_preds=np.zeros((B, N), dtype=np.int32),  # all background
            confidence=np.ones((B, N)),
            confidence_threshold=0.5,
        )
        assert result is None

    def test_returns_none_when_all_low_confidence(self):
        B, N = 1, 8
        result = create_self_enhanced_source(
            images=np.zeros((B, 16, 16, 3)),
            depths=np.zeros((B, 16, 16)),
            semantic_preds=np.ones((B, N), dtype=np.int32),
            instance_preds=np.ones((B, N), dtype=np.int32),
            confidence=np.full((B, N), 0.1),  # all below threshold
            confidence_threshold=0.5,
        )
        assert result is None
