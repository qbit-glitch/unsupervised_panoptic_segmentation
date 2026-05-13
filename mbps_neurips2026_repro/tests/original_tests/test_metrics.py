"""Unit tests for evaluation metrics.

Tests cover:
    - PQ computation matches expected values on synthetic data
    - mIoU computation with Hungarian matching
    - Instance AP computation
    - Hungarian matching correctness
    - Edge cases (empty predictions, perfect predictions)
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPanopticQuality(unittest.TestCase):
    """Test Panoptic Quality metric."""

    def test_perfect_prediction(self):
        """PQ should be 1.0 for perfect predictions."""
        from mbps.evaluation.panoptic_quality import compute_panoptic_quality

        # Simple case: 2 segments, perfect match
        h, w = 10, 10
        gt_panoptic = np.zeros((h, w), dtype=np.int64)
        gt_panoptic[:5, :] = 1001  # instance 1, class 1
        gt_panoptic[5:, :] = 2002  # instance 2, class 2

        pred_panoptic = gt_panoptic.copy()

        gt_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 2002, "category_id": 2},
        ]
        pred_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 2002, "category_id": 2},
        ]

        result = compute_panoptic_quality(
            pred_panoptic, gt_panoptic,
            pred_segments, gt_segments,
            thing_classes={1, 2}, stuff_classes=set(),
        )

        self.assertAlmostEqual(result.pq, 1.0, places=3)
        self.assertAlmostEqual(result.sq, 1.0, places=3)
        self.assertAlmostEqual(result.rq, 1.0, places=3)

    def test_no_match(self):
        """PQ should be 0.0 when no segments match."""
        from mbps.evaluation.panoptic_quality import compute_panoptic_quality

        h, w = 10, 10
        gt_panoptic = np.ones((h, w), dtype=np.int64) * 1001
        pred_panoptic = np.ones((h, w), dtype=np.int64) * 2002

        gt_segments = [{"id": 1001, "category_id": 1}]
        pred_segments = [{"id": 2002, "category_id": 2}]

        result = compute_panoptic_quality(
            pred_panoptic, gt_panoptic,
            pred_segments, gt_segments,
            thing_classes={1, 2}, stuff_classes=set(),
        )

        self.assertAlmostEqual(result.pq, 0.0, places=3)

    def test_pq_range(self):
        """PQ should be in [0, 1]."""
        from mbps.evaluation.panoptic_quality import compute_panoptic_quality

        h, w = 20, 20
        gt_panoptic = np.zeros((h, w), dtype=np.int64)
        gt_panoptic[:10, :] = 1001
        gt_panoptic[10:, :] = 2002

        pred_panoptic = np.zeros((h, w), dtype=np.int64)
        pred_panoptic[:8, :] = 1001  # Partial overlap
        pred_panoptic[8:, :] = 2002

        gt_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 2002, "category_id": 2},
        ]
        pred_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 2002, "category_id": 2},
        ]

        result = compute_panoptic_quality(
            pred_panoptic, gt_panoptic,
            pred_segments, gt_segments,
            thing_classes={1, 2}, stuff_classes=set(),
        )

        self.assertGreaterEqual(result.pq, 0.0)
        self.assertLessEqual(result.pq, 1.0)

    def test_things_vs_stuff(self):
        """PQ_things and PQ_stuff should be computed separately."""
        from mbps.evaluation.panoptic_quality import compute_panoptic_quality

        h, w = 10, 10
        gt_panoptic = np.zeros((h, w), dtype=np.int64)
        gt_panoptic[:5, :] = 1001  # thing
        gt_panoptic[5:, :] = 3     # stuff (instance_id=0, class=3)

        pred_panoptic = gt_panoptic.copy()

        gt_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 3, "category_id": 3},
        ]
        pred_segments = [
            {"id": 1001, "category_id": 1},
            {"id": 3, "category_id": 3},
        ]

        result = compute_panoptic_quality(
            pred_panoptic, gt_panoptic,
            pred_segments, gt_segments,
            thing_classes={1}, stuff_classes={3},
        )

        self.assertGreater(result.pq_things, 0.0)
        self.assertGreater(result.pq_stuff, 0.0)


class TestHungarianMatching(unittest.TestCase):
    """Test Hungarian matching algorithm."""

    def test_perfect_match(self):
        """Perfect clustering should give accuracy = 1.0."""
        from mbps.evaluation.hungarian_matching import hungarian_match

        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        gt = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

        mapping, acc = hungarian_match(pred, gt, 3, 3)
        self.assertAlmostEqual(acc, 1.0, places=5)

    def test_random_accuracy_bounded(self):
        """Random predictions should give accuracy < 1.0."""
        from mbps.evaluation.hungarian_matching import hungarian_match

        np.random.seed(0)
        pred = np.random.randint(0, 5, 100)
        gt = np.random.randint(0, 5, 100)

        _, acc = hungarian_match(pred, gt, 5, 5)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(acc, 0.0)

    def test_ignore_label(self):
        """Pixels with ignore_label should not affect accuracy."""
        from mbps.evaluation.hungarian_matching import hungarian_match

        pred = np.array([0, 0, 0, 1, 1])
        gt = np.array([1, 1, 255, 0, 0])  # 255 is ignore

        mapping, acc = hungarian_match(pred, gt, 2, 2, ignore_label=255)
        self.assertAlmostEqual(acc, 1.0, places=5)

    def test_miou_computation(self):
        """mIoU should be 1.0 for perfect clustering."""
        from mbps.evaluation.hungarian_matching import hungarian_match, compute_miou

        pred = np.array([0, 0, 1, 1, 2, 2])
        gt = np.array([1, 1, 2, 2, 0, 0])

        mapping, _ = hungarian_match(pred, gt, 3, 3)
        miou, per_class = compute_miou(pred, gt, mapping, 3)

        self.assertAlmostEqual(miou, 1.0, places=5)


class TestSemanticMetrics(unittest.TestCase):
    """Test semantic segmentation metrics."""

    def test_compute_miou_perfect(self):
        """Perfect prediction should give mIoU = 1.0."""
        from mbps.evaluation.semantic_metrics import compute_miou

        pred = np.array([0, 0, 1, 1, 2, 2])
        gt = np.array([0, 0, 1, 1, 2, 2])

        result = compute_miou(pred, gt, 3, 3)
        self.assertAlmostEqual(result.miou, 1.0, places=5)

    def test_compute_miou_with_matching(self):
        """mIoU should handle cluster-to-class mapping."""
        from mbps.evaluation.semantic_metrics import compute_miou

        # Clusters are permuted versions of GT
        pred = np.array([2, 2, 0, 0, 1, 1])
        gt = np.array([0, 0, 1, 1, 2, 2])

        result = compute_miou(pred, gt, 3, 3)
        self.assertAlmostEqual(result.miou, 1.0, places=5)

    def test_compute_miou_ignore_label(self):
        """Pixels with ignore_label should be excluded."""
        from mbps.evaluation.semantic_metrics import compute_miou

        pred = np.array([0, 0, 1, 1, 0])
        gt = np.array([0, 0, 1, 1, 255])  # Last pixel ignored

        result = compute_miou(pred, gt, 2, 2, ignore_label=255)
        self.assertAlmostEqual(result.miou, 1.0, places=5)

    def test_pixel_accuracy(self):
        """Pixel accuracy should match expected value."""
        from mbps.evaluation.semantic_metrics import compute_miou

        pred = np.array([0, 0, 0, 0])
        gt = np.array([0, 0, 1, 1])

        result = compute_miou(pred, gt, 2, 2)
        self.assertAlmostEqual(result.pixel_accuracy, 0.5, places=3)


class TestInstanceMetrics(unittest.TestCase):
    """Test instance segmentation metrics."""

    def test_mask_iou_perfect(self):
        """Perfect mask overlap should give IoU = 1.0."""
        from mbps.evaluation.instance_metrics import compute_mask_iou

        mask = np.array([True, True, False, False])
        iou = compute_mask_iou(mask, mask)

        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_mask_iou_no_overlap(self):
        """Non-overlapping masks should give IoU = 0.0."""
        from mbps.evaluation.instance_metrics import compute_mask_iou

        mask1 = np.array([True, True, False, False])
        mask2 = np.array([False, False, True, True])
        iou = compute_mask_iou(mask1, mask2)

        self.assertAlmostEqual(iou, 0.0, places=5)

    def test_ap_perfect(self):
        """Perfect predictions should give AP = 1.0."""
        from mbps.evaluation.instance_metrics import compute_ap

        n = 100
        gt_masks = np.zeros((3, n), dtype=bool)
        gt_masks[0, :30] = True
        gt_masks[1, 30:60] = True
        gt_masks[2, 60:90] = True

        pred_masks = gt_masks.copy()
        pred_scores = np.array([0.9, 0.8, 0.7])

        ap, _, _ = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=0.5)
        self.assertAlmostEqual(ap, 1.0, places=3)

    def test_ap_empty_predictions(self):
        """Empty predictions should give AP = 0.0."""
        from mbps.evaluation.instance_metrics import compute_ap

        gt_masks = np.ones((2, 50), dtype=bool)
        pred_masks = np.zeros((0, 50), dtype=bool)
        pred_scores = np.array([])

        ap, _, _ = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=0.5)
        self.assertAlmostEqual(ap, 0.0, places=5)

    def test_ap_range(self):
        """AP should be in [0, 1]."""
        from mbps.evaluation.instance_metrics import compute_ap_range

        np.random.seed(0)
        n = 200
        pred_masks = np.random.rand(10, n) > 0.5
        pred_scores = np.random.rand(10)
        gt_masks = np.random.rand(5, n) > 0.5

        result = compute_ap_range(pred_masks, pred_scores, gt_masks)

        self.assertGreaterEqual(result.ap, 0.0)
        self.assertLessEqual(result.ap, 1.0)
        self.assertGreaterEqual(result.ap_mean, 0.0)
        self.assertLessEqual(result.ap_mean, 1.0)


if __name__ == "__main__":
    unittest.main()
