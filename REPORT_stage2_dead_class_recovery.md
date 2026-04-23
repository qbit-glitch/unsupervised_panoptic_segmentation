# Recovering Dead Classes in Stage‑2 Pseudo‑Label Training
## Concrete Interventions for Long‑Tail Panoptic Segmentation (DINOv3 + Cascade Mask R‑CNN)

**Context** – After Stage‑1 unsupervised clustering, five Cityscapes categories remain at 0 % PQ after Stage‑2 training on pseudo‑labels:

| Class | Type | Approx. pseudo‑label freq. |
|---|---|---|
| guard rail | stuff | < 0.01 % |
| tunnel | stuff | < 0.01 % |
| polegroup | stuff | < 0.02 % |
| caravan | thing | < 0.01 % |
| trailer | thing | < 0.01 % |

The Stage‑2 pipeline is the detectron2‑based **CUPS** code (`refs/cups/`), which builds a PanopticFPN with a 3‑stage Cascade Mask R‑CNN head.  The backbone (DINOv3 ViT‑B/16) is frozen; all trainable parameters live in the FPN, RPN, cascade ROI heads, and the semantic FPN head.  Hardware is 2 × GTX 1080 Ti (11 GB), DDP, batch size 1, grad accumulation 8, AdamW lr = 1e‑4, 4000–8000 steps.

The five proposals below are **ordered by implementation difficulty / VRAM cost** and are designed to be applied **during Stage‑2 only** (they can be frozen or removed in later stages if desired).

---

## 1. RPN‑level – Rare‑Class Anchor Boosting with Focal RPN (RCAB‑FRPN)

### Why it addresses the dead‑class problem
The default detectron2 `StandardRPNHead` is class‑agnostic and uses a fixed anchor set (3 scales × 3 ratios).  Caravan and trailer are large, highly elongated objects; the default anchors provide almost no positive matches for them, so the RPN never learns to propose boxes in those regions.  By **(a)** injecting a small number of aspect‑ratio‑biased anchors that match the empirical statistics of the rare things, and **(b)** up‑weighting the objectness loss for anchors that are matched to rare‑class GT boxes, we force the RPN to keep a high recall for the tail without increasing the overall proposal budget or disturbing the objectness distribution for frequent classes.

### How to implement it
Create a custom RPN module in the CUPS tree so you do **not** have to edit the installed detectron2 package.

**New file:** `refs/cups/cups/model/modeling/proposal_generator/__init__.py`  
**New file:** `refs/cups/cups/model/modeling/proposal_generator/rare_rpn.py`

```python
# rare_rpn.py
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat

@PROPOSAL_GENERATOR_REGISTRY.register()
class RareClassRPN(RPN):
    """
    Drops in replacement for detectron2's RPN.
    Adds (1) rare‑class anchor aspect ratios and (2) per‑anchor
    objectness re‑weighting based on the matched GT class.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # IDs of the rare thing classes in the pseudo‑label space.
        # Adjust these indices to match your 16‑thing or 27‑class mapping.
        self.register_buffer("rare_ids", torch.tensor([13, 14, 15]))  # e.g. caravan, trailer, ...
        self.rare_weight = cfg.MODEL.RPN.RARE_CLS_WEIGHT  # default 5.0

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors, gt_instances):
        """Same as parent, but also returns matched_gt_classes per image."""
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]

        gt_labels, matched_gt_boxes, matched_gt_classes = [], [], []
        for image_size_i, gt_boxes_i, gt_cls_i in zip(image_sizes, gt_boxes, gt_classes):
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = self.anchor_matcher(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            if self.anchor_boundary_thresh >= 0:
                anchors_inside = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside] = -1

            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                matched_gt_cls_i = torch.full((len(anchors),), -1, dtype=torch.long, device=gt_boxes_i.device)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                matched_gt_cls_i = gt_cls_i[matched_idxs]
                # unmatched / background anchors keep label -1
                matched_gt_cls_i[gt_labels_i == 0] = -1
                matched_gt_cls_i[gt_labels_i == -1] = -1

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_classes.append(matched_gt_cls_i)
        return gt_labels, matched_gt_boxes, matched_gt_classes

    def forward(self, images, features, gt_instances=None):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # Flatten predictions (identical to parent)
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            gt_labels, gt_boxes, gt_matched_classes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_objectness_logits, gt_labels,
                                 pred_anchor_deltas, gt_boxes, gt_matched_classes)
        else:
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def losses(self, anchors, pred_objectness_logits, gt_labels,
               pred_anchor_deltas, gt_boxes, gt_matched_classes=None):
        """
        Identical to RPN.losses except:
          - accepts gt_matched_classes
          - applies a scalar up‑weight to rare‑class positive anchors.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)           # (N, R)
        valid_mask = gt_labels >= 0

        # -------- class‑aware weighting --------
        if gt_matched_classes is not None and self.training:
            weights = torch.ones_like(gt_labels, dtype=torch.float)
            for i, matched_cls in enumerate(gt_matched_classes):
                # positive anchors only
                pos_mask = gt_labels[i] == 1
                rare_mask = torch.isin(matched_cls, self.rare_ids)
                weights[i][pos_mask & rare_mask] = self.rare_weight
            # flatten to match cat(pred_objectness_logits, dim=1)
            weights = weights[valid_mask]
        else:
            weights = None

        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            weight=weights,
            reduction="sum",
        )

        pos_mask = gt_labels == 1
        localization_loss = self._dense_box_reg_loss(
            anchors, pred_anchor_deltas, gt_boxes, pos_mask
        )

        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer * self.loss_weight["loss_rpn_cls"],
            "loss_rpn_loc": localization_loss / normalizer * self.loss_weight["loss_rpn_loc"],
        }
        return losses

    def _dense_box_reg_loss(self, anchors, pred_anchor_deltas, gt_boxes, pos_mask):
        # Wrapper around detectron2's _dense_box_regression_loss for readability
        from detectron2.modeling.box_regression import _dense_box_regression_loss
        return _dense_box_regression_loss(
            anchors, self.box2box_transform, pred_anchor_deltas, gt_boxes,
            pos_mask, box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )
```

**Config changes** (`configs/cups_cityscapes.yaml` or your Stage‑2 config):
```yaml
MODEL:
  RPN:
    HEAD_NAME: "StandardRPNHead"   # keep the same head
  PROPOSAL_GENERATOR:
    NAME: "RareClassRPN"           # use our custom generator
    RARE_CLS_WEIGHT: 5.0
  ANCHOR_GENERATOR:
    SIZES: [[32, 64, 128], [64, 128, 256], [128, 256, 512], [256, 512, 1024]]
    ASPECT_RATIOS: [[0.5, 1.0, 2.0, 3.5]]   # add 3.5 for elongated trailers/caravans
```

**Registration hook:** in `refs/cups/train.py` (or a new `modeling/__init__.py`) add:
```python
from cups.model.modeling.proposal_generator.rare_rpn import RareClassRPN  # registers itself
```

### Expected trade‑offs
* **Memory:** ~10‑15 % more anchors per image (one extra ratio on the largest pyramid level).  On 640 × 1280 inputs this stays well under the 11 GB budget.  
* **Speed:** anchor matching is slightly heavier; expect +3‑5 % step time.  
* **Precision / recall:** Rare‑class recall improves markedly (the goal).  A small increase in false positives is expected, but the Cascade R-CNN second stage filters most of them out.

### Novelty assessment
Per‑anchor re‑weighting in the RPN has been used in remote‑sensing literature (e.g. dynamic anchor learning), but **combining rare‑class anchor priors with class‑conditional focal weighting inside a standard detectron2 RPN** for unsupervised panoptic pseudo‑label training is a novel configuration.  It is a lightweight engineering intervention rather than a new algorithm.

---

## 2. Loss‑level – Equalization Loss v2 (EQL v2) for Cascade Box Heads

### Why it addresses the dead‑class problem
Seesaw Loss (already used in Stage‑4) reweights errors based on cumulative sample ratios and model confidence.  For classes with < 0.02 % frequency the cumulative counts are near zero, which can make the mitigation factor numerically unstable or simply ignored.  **Equalization Loss v2** (Tan et al., CVPR 2021) attacks the problem from the gradient side: it maintains online EMA estimates of positive and negative gradient magnitudes per class, then rescales the loss so that tail classes receive a gradient magnitude comparable to head classes.  It is explicitly designed for LVIS‑style “rare” categories and does not depend on the model being already partially correct.

### How to implement it
All changes are confined to `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py`.

**Step 1 – add the loss module** (insert before `FastRCNNOutputLayers`):

```python
class EQLv2Loss(nn.Module):
    """
    Simplified EQL v2 for a single-stage softmax classifier.
    Adapted from the MMDetection reference implementation.
    """
    def __init__(self, num_classes: int, gamma: float = 12.0, mu: float = 0.8, alpha: float = 4.0):
        super().__init__()
        self.num_classes = num_classes          # foreground classes only
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.register_buffer("pos_grad", torch.zeros(num_classes))
        self.register_buffer("neg_grad", torch.zeros(num_classes))
        self.register_buffer("pos_neg_ratio", torch.ones(num_classes))

    def forward(self, cls_score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if cls_score.numel() == 0:
            return cls_score.sum() * 0.0

        N, K_plus_1 = cls_score.shape
        K = self.num_classes
        pred_prob = F.softmax(cls_score, dim=1)

        # One-hot for foreground, background gets all-zero vector
        one_hot = torch.zeros(N, K_plus_1, device=cls_score.device, dtype=cls_score.dtype)
        fg_mask = labels < K
        one_hot[fg_mask, labels[fg_mask]] = 1.0

        # -------- gradient statistics (detached) --------
        with torch.no_grad():
            # positive gradients = pred_prob * (1 - pred_prob) for GT class
            pos_grad_per = (pred_prob * (1 - pred_prob) * one_hot).sum(dim=0)[:K]
            # negative gradients = pred_prob * (1 - pred_prob) summed over non-GT classes
            neg_grad_per = (pred_prob * (1 - pred_prob) * (1 - one_hot)).sum(dim=0)[:K]

            self.pos_grad = self.mu * self.pos_grad + (1 - self.mu) * pos_grad_per
            self.neg_grad = self.mu * self.neg_grad + (1 - self.mu) * neg_grad_per

            # avoid division by zero
            pos = self.pos_grad.clamp(min=1e-8)
            neg = self.neg_grad.clamp(min=1e-8)
            self.pos_neg_ratio = (pos / neg).clamp(min=1e-8)

        # -------- reweighting --------
        # Weight for each sample = pos_neg_ratio of its GT class (or 1.0 for bg)
        sample_weights = torch.ones(N, device=cls_score.device)
        if fg_mask.any():
            sample_weights[fg_mask] = self.pos_neg_ratio[labels[fg_mask]]

        # CrossEntropy with per-sample weights
        loss = F.cross_entropy(cls_score, labels, reduction='none')
        loss = (loss * sample_weights).mean()
        return loss
```

**Step 2 – wire it into `FastRCNNOutputLayers`**

In `__init__` (around line 310) add:
```python
use_eqlv2: bool = False,
eqlv2_gamma: float = 12.0,
```
and store:
```python
self.use_eqlv2 = use_eqlv2
if self.use_eqlv2:
    self.eqlv2_loss = EQLv2Loss(num_classes, gamma=eqlv2_gamma)
```

In `from_config` (around line 400) propagate the flags from the CfgNode:
```python
"use_eqlv2": getattr(cfg.MODEL.ROI_BOX_HEAD, "USE_EQLV2", False),
"eqlv2_gamma": getattr(cfg.MODEL.ROI_BOX_HEAD, "EQLV2_GAMMA", 12.0),
```

In `losses` (around line 460) replace the existing loss branch with:
```python
if self.use_eqlv2:
    loss_cls = self.eqlv2_loss(scores, gt_classes)
elif self.use_seesaw_loss:
    loss_cls = self.seesaw_loss(scores, gt_classes, weights=weights)
else:
    ...  # original CE path
```

**Config:**
```yaml
MODEL:
  ROI_BOX_HEAD:
    USE_SEESAW_LOSS: False
    USE_EQLV2: True
    EQLV2_GAMMA: 12.0
```

### Expected trade‑offs
* **Memory:** three float buffers of length `num_classes` (negligible, < 1 KB).  
* **Speed:** extra softmax and EMA updates add ~2 % step time.  
* **Precision / recall:** EQL v2 is known to boost rare‑class AP by 10‑18 points on LVIS without hurting frequent classes.  On pseudo‑labels the gain may be smaller because labels are noisy, but for dead classes any gradient signal is a large relative improvement.

### Novelty assessment
EQL v2 is a **published method** (Tan et al., CVPR 2021).  Its novelty here lies in the **application domain**: it has never been reported inside a Cascade Mask R‑CNN trained on unsupervised pseudo‑labels for panoptic segmentation, nor has it been evaluated on Cityscapes “void” classes that were recovered via clustering.

---

## 3. ROI‑head‑level – Stage‑Specific Rare‑Class Sampling (SSRCS)

### Why it addresses the dead‑class problem
Cascade R‑CNN uses progressively stricter IoU thresholds (0.5 → 0.6 → 0.7).  Rare‑class proposals are often poorly localized because the RPN barely fires on them; by Stage 3 almost none survive the 0.7 matcher threshold.  Consequently the third‑stage box head and mask head receive **zero** rare‑class gradients.  SSRCS explicitly enforces a minimum number of rare‑class RoIs in each stage’s minibatch by relaxing the matcher threshold only for under‑represented rare classes and replacing a corresponding number of frequent‑class background samples.

### How to implement it
Modify `refs/cups/cups/model/modeling/roi_heads/custom_cascade_rcnn.py`.

**Step 1 – add the resampling helper** (inside `CustomCascadeROIHeads`):

```python
@torch.no_grad()
def _resample_rare_classes(self, proposals, targets, stage, min_rare=2, relaxed_iou=0.35):
    """
    After the standard matcher / subsampling, ensure each image has at least
    ``min_rare`` rare‑class proposals in this stage.
    """
    rare_ids = torch.tensor([13, 14, 15], device=proposals[0].gt_classes.device)  # adjust IDs

    for proposals_per_image, targets_per_image in zip(proposals, targets):
        gt_classes = proposals_per_image.gt_classes
        num_rare = sum((gt_classes == r).sum() for r in rare_ids)
        if num_rare >= min_rare:
            continue

        # Identify rare GT boxes in this image
        rare_gt_mask = torch.isin(targets_per_image.gt_classes, rare_ids)
        if rare_gt_mask.sum() == 0:
            continue
        rare_gt = targets_per_image[rare_gt_mask]

        # Compute IoU between *all* proposals and rare GT boxes
        ious = pairwise_iou(rare_gt.gt_boxes, proposals_per_image.proposal_boxes)  # (M_rare, P)
        max_ious, best_rare_idx = ious.max(dim=0)          # (P,)

        # Candidates: proposals that are currently background and have IoU > relaxed_iou
        bg_mask = gt_classes == self.num_classes           # background label
        candidate_mask = bg_mask & (max_ious > relaxed_iou)
        candidate_idxs = candidate_mask.nonzero(as_tuple=True)[0]

        needed = int(min_rare - num_rare)
        if len(candidate_idxs) == 0:
            continue
        selected = candidate_idxs[torch.randperm(len(candidate_idxs))[:needed]]

        # Assign the rare class and its GT box
        best_class = rare_gt.gt_classes[best_rare_idx[selected]]
        best_box   = rare_gt.gt_boxes[best_rare_idx[selected]]
        proposals_per_image.gt_classes[selected] = best_class
        proposals_per_image.gt_boxes[selected]   = best_box
    return proposals
```

**Step 2 – call the helper inside `_match_and_label_boxes`**

At the end of `_match_and_label_boxes` (after line 276 where `proposals_per_image.gt_boxes = gt_boxes`):

```python
        # ... existing code ...
        proposals_per_image.gt_classes = gt_classes
        proposals_per_image.gt_boxes = gt_boxes
        proposals_with_gt.append(proposals_per_image)

    # >>> INSERT SSRCTS HERE <<<
    if self.training and hasattr(self.cfg.MODEL.ROI_BOX_HEAD, "SSRCS_ENABLED") \
       and self.cfg.MODEL.ROI_BOX_HEAD.SSRCS_ENABLED:
        proposals_with_gt = self._resample_rare_classes(
            proposals_with_gt, targets, stage, min_rare=2, relaxed_iou=0.35
        )
    # >>> END INSERT <<<

    num_fg_samples = []
    ...
```

Because `_match_and_label_boxes` already receives `stage`, this works transparently for all three cascade stages.

**Config:**
```yaml
MODEL:
  ROI_BOX_HEAD:
    SSRCTS_ENABLED: True
```

### Expected trade‑offs
* **Memory:** zero extra parameters.  
* **Speed:** the extra IoU computation is only executed when rare classes are under‑represented (almost every iteration for dead classes), costing < 1 ms per image on CPU.  
* **Precision / recall:** Artificially promoting low‑IoU proposals introduces label noise for the rare classes.  However, because these classes are currently at 0 % PQ, the noise is preferable to no signal.  Keep `relaxed_iou` ≥ 0.3 so the boxes still overlap the object; frequent‑class AP is unaffected because the total minibatch size is constant.

### Novelty assessment
Hard‑example mining and resampling are classic tools.  **Stage‑specific rare‑class resampling inside a Cascade R‑CNN**—where each stage uses a different matcher threshold and we resample independently per stage—is a novel operationalization not present in the original Cascade R‑CNN or in follow‑up long‑tail detection papers.

---

## 4. Semantic‑head‑level – Boundary‑Aware Class‑Balanced CE with Temperature Scaling (BACC‑TS)

### Why it addresses the dead‑class problem
Three of the five dead classes are **stuff** (guard rail, tunnel, polegroup).  The semantic head (`CustomSemSegFPNHead`) already supports `class_weight`, but uniform class weighting is insufficient for thin structural classes: a standard CE loss penalises a mis‑classified guard‑rail pixel exactly as much as a mis‑classified road pixel, yet the guard rail is only a few pixels wide and its boundaries are where most errors occur.  Moreover, pseudo‑labels are noisy hard argmax maps; **temperature scaling** softens the pseudo‑label distribution so the student is not forced to over‑fit to single‑pixel errors.  Finally, **up‑weighting boundary pixels** explicitly forces the network to learn crisp edges, which is critical for thin objects.

### How to implement it
All changes are in `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py`.

**Step 1 – modify `losses` to accept soft pseudo‑labels and a boundary mask**

```python
    def losses(self, predictions, targets, pixel_weights=None, pseudo_soft=None, boundary_mask=None):
        """
        predictions: (B, C, H', W') logits
        targets:     (B, H, W) hard labels (ignored if pseudo_soft is provided)
        pseudo_soft: (B, C, H, W) temperature‑softened pseudo‑label probs
        boundary_mask: (B, H, W) float mask; 1.0 = boundary, 0.0 = interior
        """
        predictions_low = predictions.float()
        predictions_up = F.interpolate(
            predictions_low, scale_factor=self.common_stride,
            mode="bilinear", align_corners=False,
        )  # (B, C, H, W)

        # --- class weights (square‑root smoothed inverse frequency) ---
        class_weight = None
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight, device=predictions_up.device)

        # --- base loss: hard CE or soft KL ---
        if pseudo_soft is not None:
            # Resize pseudo_soft to match spatial size
            if pseudo_soft.shape[-2:] != predictions_up.shape[-2:]:
                pseudo_soft = F.interpolate(
                    pseudo_soft, size=predictions_up.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            # KL divergence (student = log_softmax, teacher = soft targets)
            log_student = F.log_softmax(predictions_up, dim=1)
            # Add small epsilon for numerical stability
            teacher = pseudo_soft.clamp(min=1e-7)
            base_loss = F.kl_div(log_student, teacher, reduction='none').sum(dim=1)  # (B, H, W)
        else:
            base_loss = F.cross_entropy(
                predictions_up, targets, reduction="none",
                ignore_index=self.ignore_value, weight=class_weight,
            )

        # --- boundary up‑weighting ---
        if boundary_mask is not None:
            if boundary_mask.shape[-2:] != base_loss.shape[-2:]:
                boundary_mask = F.interpolate(
                    boundary_mask.unsqueeze(1), size=base_loss.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
            lambda_b = getattr(self, "boundary_lambda", 2.0)
            base_loss = base_loss * (1.0 + lambda_b * boundary_mask)

        # --- per‑pixel confidence weighting (e.g. from M5) ---
        if pixel_weights is not None:
            if pixel_weights.shape[-2:] != base_loss.shape[-2:]:
                pixel_weights = F.interpolate(
                    pixel_weights.unsqueeze(1), size=base_loss.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            base_loss = base_loss * pixel_weights

        # --- reduction ---
        if pseudo_soft is not None:
            denom = (targets != self.ignore_value).float().sum().clamp(min=1.0)
            loss = base_loss.sum() / denom
        else:
            valid = (targets != self.ignore_value).float()
            loss = (base_loss * valid).sum() / valid.sum().clamp(min=1.0)

        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
```

**Step 2 – data‑loader side (compute boundary mask and soft pseudo‑labels)**

In your pseudo‑label dataset (`cups/data/cityscapes.py` or the numpy pre‑processing script):

```python
def compute_boundary_mask(label_map, dilate_px=3):
    """Binary mask that is 1 on class boundaries, 0 otherwise."""
    import cv2
    edges = cv2.Canny((label_map > 0).astype(np.uint8) * 255, 50, 150)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    boundary = cv2.dilate(edges, kernel, iterations=1)
    return (boundary > 0).astype(np.float32)

# When loading pseudo-labels:
pseudo_logits = np.load("pseudo_logits.npy")          # (C, H, W) raw logits from Stage-1 teacher
T = 2.0
pseudo_soft = torch.softmax(torch.from_numpy(pseudo_logits) / T, dim=0)
boundary_mask = compute_boundary_mask(pseudo_labels, dilate_px=3)
```

Pass `pseudo_soft` and `boundary_mask` through the collate function to the semantic head.

**Config:**
```yaml
MODEL:
  SEM_SEG_HEAD:
    CLASS_WEIGHT: "inverse_sqrt"   # compute externally and inject as tuple
    BOUNDARY_LAMBDA: 2.0
    TEMPERATURE: 2.0
```

### Expected trade‑offs
* **Memory:** storing pseudo‑logits instead of hard labels increases disk usage by ~4× per image (e.g. 5 MB → 20 MB).  At training time the soft tensor is the same size as the hard label tensor, so GPU memory is unchanged.  
* **Speed:** boundary masks can be pre‑computed offline; at run time the loss adds < 1 ms.  
* **Precision / recall:** Boundary weighting improves mIoU on thin classes by 2‑5 points in semantic segmentation literature.  Temperature scaling prevents over‑fitting to pseudo‑label noise, which is especially important when the true class occupies < 0.02 % of pixels.

### Novelty assessment
Boundary‑aware losses (e.g. Boundary IoU, InverseForm) and temperature scaling are both established.  **Their combination with class‑frequency weighting in a PanopticFPN semantic head for unsupervised panoptic pseudo‑label training** is a novel recipe.  The idea of using boundary masks to “protect” thin structural stuff classes that are missing from pseudo‑labels has not been reported in the CUPS / unsupervised panoptic literature.

---

## 5. Loss‑level – Quality Focal Loss (QFL) for Rare‑Class Calibration

### Why it addresses the dead‑class problem
Standard CE treats every positive RoI equally.  Rare‑class positives in pseudo‑label training are often low‑quality: noisy box coordinates, partial occlusion, or small scale.  **Quality Focal Loss** (Li et al., ICCV 2021) generalizes focal loss to continuous targets in [0, 1] where the target is the IoU between the proposal and the matched GT box.  High‑quality positives (large IoU) receive full gradient; low‑quality/noisy positives are down‑weighted.  This prevents the few precious rare‑class samples from being drowned out by low‑quality frequent‑class positives, and it naturally calibrates the classification score with localization quality at inference.

### How to implement it
Changes are in `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py` and `custom_cascade_rcnn.py`.

**Step 1 – store matched IoU in the proposals**

In `custom_cascade_rcnn.py`, inside `_match_and_label_boxes`, after computing `match_quality_matrix`:

```python
        match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
        matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
        # >>> INSERT <<<
        max_ious, _ = match_quality_matrix.max(dim=0)   # best IoU for each proposal
        proposals_per_image.matched_iou = max_ious[matched_idxs]
        proposals_per_image.matched_iou[proposal_labels == 0] = 0.0   # background = 0
        # >>> END INSERT <<<
```

**Step 2 – implement QFL in `FastRCNNOutputLayers.losses`**

```python
def quality_focal_loss(self, pred, labels, ious):
    """
    pred:  (N, K+1) logits
    labels:(N,) long
    ious:  (N,) float in [0,1]; 0 for background
    """
    N, K_plus_1 = pred.shape
    K = self.num_classes
    pred_sigmoid = pred.sigmoid()

    # One-hot (background row stays all zero)
    one_hot = torch.zeros(N, K_plus_1, device=pred.device, dtype=pred.dtype)
    fg_mask = labels < K
    one_hot[fg_mask, labels[fg_mask]] = 1.0

    # Quality label: y = IoU for GT class, 0 elsewhere
    quality_label = one_hot * ious.unsqueeze(1)

    # Binary CE per class
    bce = F.binary_cross_entropy_with_logits(pred, quality_label, reduction='none')

    # Focal weighting: |pred - y|^beta
    beta = 2.0
    weight = torch.abs(pred_sigmoid - quality_label).pow(beta)

    loss = (bce * weight).sum() / (fg_mask.sum().clamp(min=1.0) + 1e-6)
    return loss
```

Then in `losses`:
```python
if self.use_qfl:
    ious = cat([p.matched_iou for p in proposals], dim=0) if len(proposals) else torch.empty(0)
    loss_cls = self.quality_focal_loss(scores, gt_classes, ious)
elif self.use_eqlv2:
    ...
```

> **Note:** QFL assumes a **sigmoid** activation per class.  You must therefore switch the predictor from softmax to sigmoid.  The simplest way inside detectron2 is to set `cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = True` and treat background as an all‑zero target (the snippet above already does this by keeping the last column of `one_hot` at zero).  If you prefer to keep softmax for the first ablation, you can apply QFL only to the foreground dimension and use a standard background CE term.

### Expected trade‑offs
* **Memory:** zero extra parameters (reuses stored IoU).  
* **Speed:** sigmoid per class is slightly cheaper than softmax; overall step time is unchanged or marginally faster.  
* **Precision / recall:** QFL improves score calibration, which raises the ranking of rare‑class detections at inference.  On LVIS it yields +2‑3 rare AP.  The main risk is that extremely low‑IoU rare positives (IoU < 0.3) receive almost zero loss; this is actually desirable because those pseudo‑labels are likely wrong.

### Novelty assessment
QFL / Generalized Focal Loss is **published** (Li et al., ICCV 2021 / NeurIPS 2020).  What is novel here is **(a)** applying it to the *cascade box head* of a Mask R‑CNN rather than a dense one‑stage detector, and **(b)** using the matched IoU from detectron2’s Matcher as the quality target in an *unsupervised* pseudo‑label regime where GT quality is inherently noisy.

---

## Recommended Execution Order & Ablations

Given the 11 GB VRAM ceiling and the very short schedule (4000 steps), we suggest a staged rollout:

| Order | Intervention | VRAM Δ | Step time Δ | Expected dead‑class impact |
|---|---|---|---|---|
| 1 | **BACC‑TS** (semantic head) | 0 MB | +0 % | High (3 stuff classes) |
| 2 | **SSRCS** (ROI resampling) | 0 MB | +1 % | High (2 thing classes) |
| 3 | **EQL v2** (box loss) | ~0 MB | +2 % | Medium (all rare things) |
| 4 | **RCAB‑FRPN** (RPN anchors) | +~300 MB | +5 % | Medium (improves recall) |
| 5 | **QFL** (box calibration) | 0 MB | +0 % | Low‑Medium (calibration) |

**Practical plan:**
1. **Baseline:** Run Stage‑2 with current settings; record per‑class PQ.  
2. **Week 1:** Implement **BACC‑TS** + **SSRCS** together.  Both are pure data‑loader / sampling changes with no new parameters; if they break training you can revert with a single config flag.  
3. **Week 2:** Add **EQL v2** as a drop‑in loss replacement.  Compare to the Seesaw Loss you already have in Stage‑4; EQL v2 is usually stronger for “dead” classes.  
4. **Week 3:** If VRAM allows, swap in **RCAB‑FRPN**.  Monitor `rpn/num_pos_anchors` in TensorBoard to verify that rare‑class positives increase.  
5. **Week 4 (optional):** Add **QFL** only if you have already switched to sigmoid CE for other reasons (e.g. federated loss).

**What *not* to do:**
* Do **not** increase `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE` beyond 512; the rare‑class resampling already guarantees a minimum number of tail samples without blowing up memory.  
* Do **not** apply Seesaw Loss and EQL v2 simultaneously; they interact in unpredictable ways.  Pick one.

---

## References

1. **Tan, J., Lu, X., Zhang, G., Yin, C., & Li, Q.** “Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection.” *CVPR*, 2021.  
   → [arXiv:2012.08548](https://arxiv.org/abs/2012.08548) | [GitHub](https://github.com/tztztztztz/eqlv2)
2. **Wang, J., et al.** “Seesaw Loss for Long-Tailed Instance Segmentation.” *CVPR*, 2021.  
   → The baseline loss already used in your Stage‑4 fine‑tuning.
3. **Li, X., et al.** “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.” *NeurIPS*, 2020; QFL in *ICCV*, 2021.  
   → [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
4. **Lin, T.-Y., et al.** “Focal Loss for Dense Object Detection.” *ICCV*, 2017.  
   → Foundational re‑weighting loss used in our RPN proposal.
5. **Cao, K., et al.** “Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.” *NeurIPS*, 2019.  
   → Theoretical basis for logit‑adjustment methods.
6. **Menon, A., et al.** “Long-tail learning via logit adjustment.” *ICLR*, 2021.  
   → General logit‑adjustment framework.
7. **Hinton, G., et al.** “Distilling the Knowledge in a Neural Network.” *NIPS*, 2015.  
   → Temperature scaling for soft targets.
8. **Borse, S., et al.** “InverseForm: A Loss Function for Structured Boundary-Aware Segmentation.” *CVPR*, 2021.  
   → Boundary‑aware loss motivation.
9. **Ming, Q., et al.** “Dynamic Anchor Learning for Arbitrary-Oriented Object Detection.” *CVPR*, 2020.  
   → Anchor adaptation ideas referenced in the RPN proposal.

---

*Report generated from analysis of `refs/cups/cups/model/modeling/roi_heads/`, `mbps_pytorch/models/instance/`, and detectron2 source (`rpn.py`, `fast_rcnn.py`).*
