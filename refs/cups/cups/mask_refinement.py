"""Classical mask refinement for CUPS self-training pseudo-labels.

Applies training-free refinement methods to instance masks after pseudo-label
generation and before augmentation. Methods:
  1. Morphological cleanup: removes small holes and protrusions
  2. Guided filter: edge-aware smoothing using the RGB image as guide
  3. Bilateral solver: edge-aligned mask refinement (optional, slower)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from detectron2.structures import BitMasks, Boxes, Instances
from torch import Tensor
from yacs.config import CfgNode

log = logging.getLogger(__name__)


class MaskRefiner:
    """Refines instance masks using classical image processing methods."""

    def __init__(
        self,
        enable_morphological: bool = True,
        enable_guided_filter: bool = True,
        enable_bilateral_solver: bool = False,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 0.01,
        min_area: int = 100,
    ) -> None:
        self.enable_morphological = enable_morphological
        self.enable_guided_filter = enable_guided_filter
        self.enable_bilateral_solver = enable_bilateral_solver
        self.gf_radius = guided_filter_radius
        self.gf_eps = guided_filter_eps
        self.min_area = min_area
        log.info(
            f"MaskRefiner: morph={enable_morphological}, "
            f"guided_filter={enable_guided_filter}(r={guided_filter_radius}, eps={guided_filter_eps}), "
            f"bilateral={enable_bilateral_solver}, min_area={min_area}"
        )

    @classmethod
    def from_config(cls, config: CfgNode) -> "MaskRefiner":
        cfg = config.MASK_REFINEMENT
        return cls(
            enable_morphological=cfg.MORPHOLOGICAL,
            enable_guided_filter=cfg.GUIDED_FILTER,
            enable_bilateral_solver=cfg.BILATERAL_SOLVER,
            guided_filter_radius=cfg.GUIDED_FILTER_RADIUS,
            guided_filter_eps=cfg.GUIDED_FILTER_EPS,
            min_area=cfg.MIN_AREA,
        )

    def refine_pseudo_labels(
        self,
        pseudo_labels: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Refine instance masks in pseudo-labels.

        Args:
            pseudo_labels: List of dicts with keys:
                "image": Tensor [C, H, W] (float, ImageNet-normalized)
                "sem_seg": Tensor [H, W] (long, class IDs)
                "instances": Instances with gt_masks (BitMasks), gt_boxes, gt_classes

        Returns:
            Same list with refined masks (modified in-place).
        """
        for sample in pseudo_labels:
            instances = sample["instances"]
            if len(instances) == 0:
                continue

            # Get RGB image as numpy uint8 for guided filter
            image_tensor = sample["image"]  # [C, H, W]
            if self.enable_guided_filter or self.enable_bilateral_solver:
                guide_img = self._tensor_to_uint8(image_tensor)
            else:
                guide_img = None

            # Process each mask
            masks_np = instances.gt_masks.tensor.cpu().numpy()  # [N, H, W] bool
            refined_masks = []
            valid_indices = []

            for i in range(masks_np.shape[0]):
                mask = masks_np[i].astype(np.uint8) * 255

                # 1. Morphological cleanup
                if self.enable_morphological:
                    mask = self._morphological_cleanup(mask)

                # 2. Guided filter
                if self.enable_guided_filter and guide_img is not None:
                    mask = self._guided_filter(mask, guide_img)

                # 3. Bilateral solver
                if self.enable_bilateral_solver and guide_img is not None:
                    mask = self._bilateral_solver(mask, guide_img)

                # Threshold back to binary
                binary = mask > 127
                if binary.sum() >= self.min_area:
                    refined_masks.append(binary)
                    valid_indices.append(i)

            # Rebuild instances with refined masks
            if len(refined_masks) > 0:
                device = instances.gt_masks.tensor.device
                refined_tensor = torch.from_numpy(
                    np.stack(refined_masks, axis=0)
                ).to(device=device, dtype=torch.bool)

                # Recompute bounding boxes from refined masks
                boxes = self._masks_to_boxes(refined_tensor)

                sample["instances"] = Instances(
                    image_size=instances.image_size,
                    gt_masks=BitMasks(refined_tensor),
                    gt_boxes=Boxes(boxes.to(device)),
                    gt_classes=instances.gt_classes[valid_indices],
                )
            else:
                # All masks were too small after refinement — keep empty
                H, W = instances.image_size
                sample["instances"] = Instances(
                    image_size=(H, W),
                    gt_masks=BitMasks(torch.zeros(0, H, W, dtype=torch.bool, device=instances.gt_masks.tensor.device)),
                    gt_boxes=Boxes(torch.zeros(0, 4, dtype=torch.long, device=instances.gt_masks.tensor.device)),
                    gt_classes=torch.zeros(0, dtype=torch.long, device=instances.gt_classes.device),
                )

        return pseudo_labels

    @staticmethod
    def _tensor_to_uint8(image_tensor: Tensor) -> np.ndarray:
        """Convert ImageNet-normalized [C,H,W] tensor to uint8 [H,W,3] BGR for OpenCV."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C] RGB
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _morphological_cleanup(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Morphological opening (remove small protrusions) then closing (fill holes)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _guided_filter(self, mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """Edge-aware guided filter using the RGB image as guide.

        Implements the O(1) guided filter from He et al. (ECCV 2010, TPAMI 2013).
        Uses box filter implementation for speed.
        """
        r = self.gf_radius
        eps = self.gf_eps

        # Convert to float
        I = guide.astype(np.float64) / 255.0  # [H, W, 3]
        p = mask.astype(np.float64) / 255.0   # [H, W]

        # Box filter helper
        ksize = 2 * r + 1

        mean_I = cv2.blur(I, (ksize, ksize))           # [H, W, 3]
        mean_p = cv2.blur(p, (ksize, ksize))            # [H, W]
        mean_Ip = cv2.blur(I * p[:, :, None], (ksize, ksize))  # [H, W, 3]
        cov_Ip = mean_Ip - mean_I * mean_p[:, :, None]  # [H, W, 3]

        # Variance of I (simplified: use grayscale variance for speed)
        I_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        mean_II = cv2.blur(I_gray * I_gray, (ksize, ksize))
        var_I = mean_II - cv2.blur(I_gray, (ksize, ksize)) ** 2

        # a = cov_Ip / (var_I + eps), b = mean_p - a * mean_I_gray
        a = cov_Ip.mean(axis=2) / (var_I + eps)  # [H, W]
        b = mean_p - a * cv2.blur(I_gray, (ksize, ksize))

        # Mean of a and b
        mean_a = cv2.blur(a, (ksize, ksize))
        mean_b = cv2.blur(b, (ksize, ksize))

        # Output
        q = mean_a * I_gray + mean_b
        result = np.clip(q * 255.0, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _bilateral_solver(mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """Fast bilateral solver for mask refinement.

        Simplified version using OpenCV bilateral filter as approximation.
        """
        # Use bilateral filter on the mask with guide as reference
        mask_float = mask.astype(np.float32) / 255.0
        # Apply bilateral filter to the mask using guide image edges
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        # Joint bilateral filter approximation
        filtered = cv2.bilateralFilter(mask.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    @staticmethod
    def _masks_to_boxes(masks: Tensor) -> Tensor:
        """Compute bounding boxes from binary masks. masks: [N, H, W] bool."""
        N = masks.shape[0]
        boxes = torch.zeros(N, 4, dtype=torch.float32)
        for i in range(N):
            ys, xs = torch.where(masks[i])
            if len(ys) == 0:
                continue
            boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
        return boxes
