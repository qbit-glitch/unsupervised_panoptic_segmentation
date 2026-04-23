# Recovering Dead Classes in Unsupervised Panoptic Segmentation: 5 Novel Training Strategies

**Context:** DINOv3 ViT-B/16 + Cascade Mask R-CNN (PanopticFPN) on Cityscapes. Dead classes after 3 stages: `guard rail`, `tunnel`, `polegroup`, `caravan`, `trailer`. Hardware: 2× GTX 1080 Ti (11 GB), DDP, batch size 1, gradient accumulation 8.

**Goal:** Propose 5 concrete, *training-only* interventions (no architecture changes) with theoretical justification, PyTorch Lightning pseudocode, pipeline placement, and risk analysis.

---

## Proposal 1: Rare-First Curriculum Learning (RFCL)

**Pipeline placement:** Primarily **Stage-2** (pseudo-label bootstrap), with optional extension into early **Stage-3**.

### Theoretical Justification
Standard curriculum learning starts with "easy" frequent classes and progressively adds hard/rare ones. For dead classes in long-tail unsupervised segmentation, this is *exactly backwards*: by the time rare classes enter the curriculum, the feature space has already crystallized around frequent-class decision boundaries, and the optimizer lacks the gradient budget to carve out new regions for rare classes (especially with only 4 000 steps per stage). The **Rare-First Curriculum** inverts this: we begin training on a subset of images/ROIs that contain the rarest classes (identified by pseudo-label class histograms), then progressively dilute with the full dataset. This ensures rare-class centroids are established early in feature space before frequent classes dominate. The idea is inspired by dynamic rebalancing strategies in 2DRCL (NeurIPS 2024) and the observation that "simplicity bias" in standard pre-training hurts tail classes.

### Implementation

```python
# In mbps_pytorch/training/curriculum.py (or a new rare_first_curriculum.py)

class RareFirstCurriculum:
    """Stage-2 curriculum: start with rare-class-heavy batches, decay to uniform."""

    def __init__(
        self,
        rare_classes: List[int],          # e.g. [guard_rail_id, tunnel_id, ...]
        rare_fraction_start: float = 0.8,  # 80% rare-class images in batch at start
        rare_fraction_end: float = 0.0,    # 0% by end of stage → uniform
        total_steps: int = 4000,
        warmup_steps: int = 500,
    ):
        self.rare_classes = set(rare_classes)
        self.rare_frac_start = rare_fraction_start
        self.rare_frac_end = rare_fraction_end
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_sampling_weights(self, step: int, image_class_histograms: List[Dict[int, int]]) -> torch.Tensor:
        """Compute per-image sampling weight for the next batch.

        Args:
            step: Current training step.
            image_class_histograms: List of dicts mapping class_id → pixel count per image.
        Returns:
            weights: (N,) float tensor for WeightedRandomSampler.
        """
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        alpha = self.rare_frac_start * (1.0 - progress) + self.rare_frac_end * progress

        weights = []
        for hist in image_class_histograms:
            rare_pixels = sum(hist.get(c, 0) for c in self.rare_classes)
            total_pixels = sum(hist.values()) + 1e-6
            # Up-weight images containing rare classes; down-weight rare-only if too dominant
            w = (rare_pixels / total_pixels) ** 0.5 if rare_pixels > 0 else 0.1
            weights.append(alpha * w + (1 - alpha) * 1.0)

        return torch.tensor(weights, dtype=torch.float32)

    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Optionally up-weight rare-class CE loss early, decay to 1.0."""
        progress = min(1.0, step / self.total_steps)
        rare_boost = 3.0 * (1.0 - progress) + 1.0 * progress
        return {"rare_boost": rare_boost, "freq_boost": 1.0}

# Hook into Stage-2 data loader construction (train.py / pl_model_pseudo.py)
# 1. Pre-compute per-image class histograms from pseudo-labels (one-time O(N))
# 2. Wrap the Dataset in a WeightedRandomSampler using RareFirstCurriculum weights
# 3. Update every N steps (e.g. 50) since pseudo-labels are static in Stage-2

# In semantic loss (semantic_loss_v2.py), apply rare-class boost:
def semantic_cross_entropy_rare_first(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rare_classes: List[int],
    rare_boost: float = 2.0,
    label_smoothing: float = 0.1,
    ignore_index: int = 255,
) -> torch.Tensor:
    b, n, k = logits.shape
    logits_flat = logits.reshape(-1, k)
    labels_flat = labels.reshape(-1).long()
    valid = labels_flat != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    ce = F.cross_entropy(logits_flat[valid], labels_flat[valid], reduction='none')
    # Boost rare-class pixels
    weights = torch.ones_like(ce)
    for rc in rare_classes:
        weights[labels_flat[valid] == rc] = rare_boost
    return (ce * weights).sum() / (weights.sum() + 1e-8)
```

### Expected Impact and Risks
- **Impact:** Guard rail and tunnel (stuff) should benefit most because they appear in consistent contexts (road edges, overpasses) but are swamped by road/wall pixels. Polegroup, caravan, trailer (things) may see partial recovery if their pseudo-label masks are not completely missing.
- **Risk:** If pseudo-labels for dead classes are entirely absent (true 0% coverage), boosting non-existent labels does nothing. **Mitigation:** Run a pre-pass to verify rare-class pseudo-label coverage; if absent, skip to Proposal 4 (asymmetric self-training) first.
- **Risk:** Overfitting to rare-class artifacts in noisy pseudo-labels. **Mitigation:** Keep `rare_fraction_start ≤ 0.8` and use label smoothing (already present).

---

## Proposal 2: Class-Conditional OHEM with Asymmetric Hard-Negative Mining (CC-OHEM)

**Pipeline placement:** **Stage-2** (ROI-level) and **Stage-4** (mask-level with Seesaw).

### Theoretical Justification
Standard OHEM (Shrivastava et al., 2016) mines the highest-loss examples globally. In long-tail panoptic segmentation, the hardest examples almost always belong to frequent classes (road, car, building) because they dominate the loss landscape. Rare classes never get selected, so their gradients remain vanishingly small. **Class-Conditional OHEM** enforces a per-class hard-example quota: for each forward pass, we reserve a fixed fraction of the gradient budget for rare-class hard negatives. For stuff classes (guard rail, tunnel), we mine hard *pixels* (high-confidence misclassifications to fence/building). For thing classes (caravan, trailer, polegroup), we mine hard *ROIs* (proposals with high IoU to car/truck/pole but wrong class). This is a direct response to the observation in long-tail detection literature (e.g., FRACAL, CVPR 2025) that uniform-space class distribution matters as much as frequency.

### Implementation

```python
# In cups/model/modeling/roi_heads/roi_heads.py or a custom CC-OHEM wrapper

class ClassConditionalOHEM:
    """Wraps a Detectron2 ROI head to enforce per-class hard-example quotas."""

    def __init__(
        self,
        rare_classes: List[int],
        rare_quota: float = 0.25,      # 25% of sampled ROIs/pixels must be rare-class hard negatives
        hard_negative_iou_range: Tuple[float, float] = (0.1, 0.5),
    ):
        self.rare_classes = rare_classes
        self.rare_quota = rare_quota
        self.hard_negative_iou_range = hard_negative_iou_range

    def sample_rois(self, proposals, gt_instances, num_samples: int = 512):
        """Override standard ROI sampler.

        1. Sample normal positives/negatives as usual (Detectron2's default).
        2. Identify rare-class hard negatives: proposals with IoU ∈ [0.1,0.5] to a rare-class GT box.
        3. Replace the lowest-loss frequent-class samples with rare hard negatives until quota met.
        """
        # Step 1: default sampling
        sampled_pos, sampled_neg = self._default_sample(proposals, gt_instances, num_samples)

        # Step 2: mine rare hard negatives
        rare_hard = []
        for prop, gt in zip(proposals, gt_instances):
            ious = pairwise_iou(prop.proposal_boxes, gt.gt_boxes)  # (P, G)
            max_ious, max_gt_idx = ious.max(dim=1)
            is_hard = (max_ious > self.hard_negative_iou_range[0]) & \
                      (max_ious < self.hard_negative_iou_range[1])
            gt_classes = gt.gt_classes[max_gt_idx]
            is_rare = torch.tensor([c.item() in self.rare_classes for c in gt_classes],
                                   device=gt_classes.device)
            rare_hard_mask = is_hard & is_rare
            if rare_hard_mask.any():
                rare_hard.append(prop[rare_hard_mask])

        if not rare_hard:
            return sampled_pos, sampled_neg

        rare_hard = cat(rare_hard, dim=0)
        target_rare = int(num_samples * self.rare_quota)
        num_rare = min(len(rare_hard), target_rare)

        # Step 3: subsample frequent negatives to make room
        sampled_neg = sampled_neg[:num_samples - num_rare]
        sampled_neg = cat([sampled_neg, rare_hard[:num_rare]], dim=0) if num_rare > 0 else sampled_neg

        return sampled_pos, sampled_neg

# For semantic/stuff branch (pixel-level OHEM):
class PixelLevelCCOHEM(nn.Module):
    """Applied to semantic segmentation logits before loss computation."""

    def __init__(self, rare_classes: List[int], rare_pixel_ratio: float = 0.15, top_k: int = 65536):
        super().__init__()
        self.rare_classes = rare_classes
        self.rare_pixel_ratio = rare_pixel_ratio
        self.top_k = top_k

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return class-conditional OHEM masked loss."""
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)
        valid = labels_flat != 255
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]

        per_pixel_ce = F.cross_entropy(logits_flat, labels_flat, reduction='none')

        # Global top-K hard pixels (standard OHEM)
        global_hard = per_pixel_ce.topk(min(self.top_k, len(per_pixel_ce))).indices

        # Rare-class hard pixels: highest-loss pixels among rare-class labels
        rare_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
        for rc in self.rare_classes:
            rare_mask |= (labels_flat == rc)

        if rare_mask.any():
            rare_ce = per_pixel_ce[rare_mask]
            rare_hard = rare_ce.topk(min(int(self.top_k * self.rare_pixel_ratio), len(rare_ce))).indices
            # Map back to flat indices
            rare_indices = torch.nonzero(rare_mask, as_tuple=False).squeeze(1)[rare_hard]
            # Union with global hard
            selected = torch.cat([global_hard, rare_indices]).unique()
        else:
            selected = global_hard

        return per_pixel_ce[selected].mean()
```

### Expected Impact and Risks
- **Impact:** Directly forces the RPN and box head to learn discriminative features for caravan/trailer vs car/truck, and guard rail vs fence. Pixel-level variant helps tunnel (often confused with building wall) and polegroup (confused with pole).
- **Risk:** Hard negatives may be mislabeled pseudo-labels, amplifying noise. **Mitigation:** Only activate CC-OHEM in Stage-4 (Seesaw) where pseudo-labels are most reliable, or use a confidence threshold to filter pseudo-label hard negatives.
- **Risk:** 25% rare quota is expensive if rare GT boxes are extremely sparse. **Mitigation:** If batch contains < target rare negatives, simply use all available rare negatives and fill the rest with standard negatives (graceful degradation).

---

## Proposal 3: Rare-Class Spatial Mixup with Context-Preserving Blending (RC-SMix)

**Pipeline placement:** **Stage-2** and **Stage-3** (data augmentation layer).

### Theoretical Justification
CutMix and Mixup are standard for detection, but uniformly random mixing destroys rare-class context: pasting a `tunnel` patch onto `sky` or `vegetation` creates impossible training samples that confuse the model. **RC-SMix** conditionally blends rare-class patches *only* into semantically compatible regions (e.g., tunnel → building/road contexts; guard rail → road/fence contexts; caravan/trailer → road/vehicle contexts). We use the DINOv3 semantic pseudo-labels to define "compatible" classes for each rare class (pre-computed confusion matrix from Stage-1 or early Stage-2). For stuff classes, we perform pixel-level alpha blending with a spatial mask. For thing classes, we use the existing `CopyPasteAugmentation` but add: (1) class-biased sampling (rare classes sampled with probability ∝ frequency⁻⁰·⁵), (2) context-aware placement (paste only into images that already contain compatible classes), and (3) Poisson feathering at the boundary to reduce artifacts. This is a training-time variant of the "synthetic rare-class example" idea, grounded in the observation that augmentation validity matters more than augmentation diversity for tail classes.

### Implementation

```python
# In cups/augmentation.py or a new rare_class_mixup.py

class RareClassSpatialMixup(nn.Module):
    """Context-aware spatial mixup for rare stuff classes."""

    COMPATIBLE_MAP = {
        guard_rail_id: {road_id, sidewalk_id, fence_id, wall_id},
        tunnel_id:     {building_id, wall_id, road_id, sky_id},
        polegroup_id:  {pole_id, traffic_light_id, traffic_sign_id, sidewalk_id},
        caravan_id:    {car_id, truck_id, bus_id, road_id, sidewalk_id},
        trailer_id:    {car_id, truck_id, bus_id, road_id, sidewalk_id},
    }

    def __init__(
        self,
        rare_classes: List[int],
        mix_prob: float = 0.5,
        alpha_range: Tuple[float, float] = (0.3, 0.7),
    ):
        super().__init__()
        self.rare_classes = rare_classes
        self.mix_prob = mix_prob
        self.alpha_range = alpha_range

    @torch.no_grad()
    def forward(self, batch: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        if torch.rand(1).item() > self.mix_prob:
            return batch

        output = deepcopy(batch)
        # Build a bank of rare-class patches from the batch itself
        rare_patches: List[Tuple[Tensor, Tensor, int]] = []  # (image_crop, mask, class_id)
        for sample in batch:
            sem = sample["sem_seg"]
            img = sample["image"]
            for rc in self.rare_classes:
                mask = (sem == rc)
                if mask.sum() > 64:  # Minimum viable patch
                    y, x = torch.where(mask)
                    y0, y1 = y.min().item(), y.max().item() + 1
                    x0, x1 = x.min().item(), x.max().item() + 1
                    crop = img[:, y0:y1, x0:x1]
                    crop_mask = mask[y0:y1, x0:x1]
                    rare_patches.append((crop, crop_mask, rc))

        if not rare_patches:
            return output

        for sample in output:
            sem = sample["sem_seg"]
            img = sample["image"]
            # Pick a random rare patch
            patch, pmask, pcls = random.choice(rare_patches)
            compatible = self.COMPATIBLE_MAP.get(pcls, set())

            # Find compatible placement regions in target image
            compat_mask = torch.zeros_like(sem, dtype=torch.bool)
            for cc in compatible:
                compat_mask |= (sem == cc)
            if not compat_mask.any():
                continue  # Skip if no compatible context

            # Sample a random compatible anchor point
            cy, cx = torch.where(compat_mask)
            idx = torch.randint(0, len(cy), (1,)).item()
            anchor_y, anchor_x = cy[idx].item(), cx[idx].item()

            # Place patch centered at anchor (with boundary clipping)
            ph, pw = pmask.shape
            ih, iw = img.shape[1:]
            y1 = max(0, anchor_y - ph // 2)
            x1 = max(0, anchor_x - pw // 2)
            y2 = min(ih, y1 + ph)
            x2 = min(iw, x1 + pw)
            py2 = y2 - y1
            px2 = x2 - x1

            # Alpha blending with smooth boundary
            alpha = uniform(*self.alpha_range)
            region = img[:, y1:y2, x1:x2]
            patch_crop = patch[:, :py2, :px2]
            pmask_crop = pmask[:py2, :px2].float()

            # Feathered blend: strong inside mask, weak outside
            blend = alpha * pmask_crop * patch_crop + (1 - alpha * pmask_crop) * region
            img[:, y1:y2, x1:x2] = blend

            # Update semantic label
            sem[y1:y2, x1:x2][pmask_crop.bool()] = pcls

        return output

# Integration: insert into Stage-2/Stage-3 augmentation pipeline AFTER copy-paste,
# BEFORE photometric augmentations.
# In pl_model_pseudo.py / pl_model_self.py:
#   self.rare_mixup = RareClassSpatialMixup(rare_classes=DEAD_CLASSES)
#   pseudo_labels = self.rare_mixup(pseudo_labels)
```

### Expected Impact and Risks
- **Impact:** Directly increases effective sample count for dead classes by 2–5× without requiring extra data. Context-aware placement prevents distribution shift. Especially effective for tunnel (can be blended into building facades) and guard rail (road edges).
- **Risk:** Blending artifacts may create unrealistic texture boundaries that the model learns to exploit (e.g., detecting "blended edge" rather than tunnel). **Mitigation:** Use low `alpha_range` (0.3–0.5) and apply only in Stage-2 where the model is still learning texture invariance; disable in Stage-4.
- **Risk:** Pseudo-label masks for dead classes may be missing entirely, so the patch bank is empty. **Mitigation:** Use a pre-built patch bank extracted from the full Cityscapes training set using a *weak* open-vocabulary detector (e.g., RAM+SAM) to get seed rare-class masks, then blend those.

---

## Proposal 4: Asymmetric Multi-Round Self-Training with Rare-Class TTA Boosting (AMR-ST)

**Pipeline placement:** **Stage-3** (self-training with EMA teacher).

### Theoretical Justification
Current Stage-3 uses a single EMA teacher with a uniform confidence threshold and standard TTA. For dead classes, the teacher produces near-zero logits, so even a lenient threshold discards them. **AMR-ST** introduces three asymmetries: (1) *class-dependent confidence thresholds* — rare classes use `τ_rare = 0.3` while frequent classes use `τ_freq = 0.7`; (2) *TTA logit boosting* — during teacher inference, we ensemble multi-scale predictions and apply temperature scaling `T_rare = 0.7` to rare-class logits before softmax, sharpening their pseudo-labels; (3) *round-wise curriculum* — Round 1 uses standard thresholds; Round 2+ specifically targets classes with 0% PQ from Round 1 evaluation, lowering their thresholds further and increasing their loss weight. This is inspired by iterative pseudo-label refinement in semi-supervised domain adaptation (MC-PanDA, ECCV 2024) and the observation that asymmetric thresholds prevent confirmation bias for tail classes.

### Implementation

```python
# In cups/pl_model_self.py or mbps_pytorch/training/self_training.py

class AsymmetricPseudoLabelGenerator:
    """Generates pseudo-labels with class-aware thresholds and TTA boosting."""

    def __init__(
        self,
        rare_classes: List[int],
        tau_freq: float = 0.7,
        tau_rare: float = 0.35,
        tta_scales: Tuple[float, ...] = (0.8, 1.0, 1.2),
        rare_temperature: float = 0.7,   # T < 1 sharpens rare-class logits
    ):
        self.rare_classes = set(rare_classes)
        self.tau_freq = tau_freq
        self.tau_rare = tau_rare
        self.tta_scales = tta_scales
        self.rare_temp = rare_temperature

    @torch.no_grad()
    def generate(self, teacher_model, images: List[Dict[str, Tensor]]) -> List[Dict[str, Any]]:
        teacher_model.eval()
        pseudo_labels = []

        for img_dict in images:
            img = img_dict["image"]
            _, H, W = img.shape

            # Multi-scale TTA ensemble
            sem_probs_stack = []
            for scale in self.tta_scales:
                scaled = F.interpolate(img[None], scale_factor=scale, mode='bilinear', align_corners=False)
                pred = teacher_model([{"image": scaled[0]}])
                sem_logits = pred[0]["sem_seg"]  # (C, Hs, Ws)

                # Asymmetric temperature: rare classes get sharpened
                sem_logits_boosted = sem_logits.clone()
                for rc in self.rare_classes:
                    sem_logits_boosted[rc] = sem_logits_boosted[rc] / self.rare_temp

                prob = F.softmax(sem_logits_boosted, dim=0)
                prob = F.interpolate(prob[None], size=(H, W), mode='bilinear', align_corners=False)[0]
                sem_probs_stack.append(prob)

            avg_prob = torch.stack(sem_probs_stack).mean(dim=0)

            # Class-dependent thresholding
            max_prob, pred_class = avg_prob.max(dim=0)
            threshold_map = torch.full_like(max_prob, self.tau_freq)
            for rc in self.rare_classes:
                threshold_map[pred_class == rc] = self.tau_rare

            valid_mask = max_prob > threshold_map
            sem_pseudo = pred_class.clone()
            sem_pseudo[~valid_mask] = 255

            # Instance pseudo-labels (for thing classes) with rare-class box boosting
            # Lower score threshold for rare classes in RPN/ROI inference
            ...  # Use existing make_pseudo_labels but pass class-aware thresholds

            pseudo_labels.append({
                "image": img,
                "sem_seg": sem_pseudo,
                "instances": self._make_instances(...),
            })

        return pseudo_labels

# Multi-round curriculum in SelfTrainer (mbps_pytorch/training/self_training.py)
class AsymmetricSelfTrainer(SelfTrainer):
    def __init__(self, rare_classes, num_rounds=3, ...):
        super().__init__(num_rounds=num_rounds, ...)
        self.rare_classes = rare_classes
        self.round_class_weights = [1.0, 2.0, 3.0]  # Increase rare-class loss weight per round

    def advance_round(self, per_class_pq: Optional[Dict[int, float]] = None):
        super().advance_round()
        # If PQ data available, dynamically lower τ for classes still at 0%
        if per_class_pq is not None:
            for cls_id, pq in per_class_pq.items():
                if pq == 0.0 and cls_id in self.rare_classes:
                    self.label_generator.tau_rare = max(0.15, self.label_generator.tau_rare - 0.1)
                    print(f"Round {self.label_generator.current_round}: Lowered τ for class {cls_id} to {self.label_generator.tau_rare}")

# Hook in pl_model_self.py training_step:
#   pseudo_labels = self.asymmetric_generator.generate(self.teacher_model, batch)
#   # Apply rare-class loss weighting in student forward
#   loss_dict = self.model(pseudo_labels, class_weights=self.round_class_weights)
```

### Expected Impact and Risks
- **Impact:** This is arguably the highest-impact proposal for 0% PQ classes. If the teacher ever produces *any* signal for tunnel/guard rail (even at very low confidence), the asymmetric threshold retains it, and the student learns from it. Multi-round refinement compounds this signal. TTA boosting reduces variance.
- **Risk:** Low thresholds for rare classes may admit false positives, which then get reinforced (confirmation bias). **Mitigation:** Only lower τ_rare in Round 2+ after evaluating Round 1 PQ; cap τ_rare ≥ 0.2. Combine with Seesaw in Stage-4 to suppress false-positive gradients.
- **Risk:** TTA at 3 scales triples teacher inference cost. **Mitigation:** Cache teacher predictions across rounds (pseudo-labels are static for a round); only re-run teacher when EMA updates. On 2× 1080 Ti, this is feasible because teacher inference is `no_grad` and can run on the second GPU while student trains on the first.

---

## Proposal 5: Boundary-Aware Pixel Contrastive Loss (BAPC)

**Pipeline placement:** **Stage-2** and **Stage-4** (auxiliary loss, ~0.1–0.3 weight).

### Theoretical Justification
For stuff classes like `guard rail` and `tunnel`, the primary failure mode is not class recognition but *boundary localization*: the model confuses guard rail with fence, or tunnel interior with building wall. Standard cross-entropy and even Seesaw Loss operate at the pixel level and do not explicitly enforce that rare-class boundary pixels should be distinct from neighboring confusing classes. **BAPC** adds a pixel-level contrastive loss: we sample anchor pixels from rare-class boundaries (identified via Sobel filtering on pseudo-label maps), pull them toward the mean feature of rare-class *interior* pixels, and push them away from the mean features of the K nearest confusing classes (determined by a pre-computed confusion matrix). This is inspired by Boundary Contrastive Learning (BCLL, BMVC 2024) and NECO (ICLR 2025), but adapted for unsupervised pseudo-labels.

### Implementation

```python
# In mbps_pytorch/losses/boundary_contrastive_loss.py

class BoundaryAwarePixelContrastive(nn.Module):
    """Pixel-level contrastive loss for rare-class boundary refinement."""

    def __init__(
        self,
        rare_classes: List[int],
        confusing_neighbors: Dict[int, List[int]],  # e.g. tunnel: [building, wall]
        temperature: float = 0.1,
        boundary_width: int = 3,
        num_anchors: int = 256,
        lambda_boundary: float = 0.2,
    ):
        super().__init__()
        self.rare_classes = rare_classes
        self.confusing_neighbors = confusing_neighbors
        self.temperature = temperature
        self.boundary_width = boundary_width
        self.num_anchors = num_anchors
        self.lambda_boundary = lambda_boundary

    def _get_boundary_mask(self, label_map: torch.Tensor, target_class: int) -> torch.Tensor:
        """Erode label map; boundary = target_class pixels NOT in eroded mask."""
        binary = (label_map == target_class).float()[None, None]  # (1,1,H,W)
        kernel = torch.ones(1, 1, 3, 3, device=binary.device)
        eroded = (F.conv2d(binary, kernel, padding=1) >= 9.0).float()
        boundary = (binary.squeeze() > 0) & (eroded.squeeze() < 0.5)
        return boundary

    def forward(
        self,
        features: torch.Tensor,      # (B, D, H, W) from backbone/pixel decoder
        labels: torch.Tensor,        # (B, H, W) pseudo-labels
    ) -> torch.Tensor:
        B, D, H, W = features.shape
        loss = torch.tensor(0.0, device=features.device)
        total_anchors = 0

        for b in range(B):
            feat = features[b]    # (D, H, W)
            lab = labels[b]       # (H, W)

            for rc in self.rare_classes:
                if (lab == rc).sum() == 0:
                    continue

                boundary = self._get_boundary_mask(lab, rc)
                if boundary.sum() == 0:
                    continue

                interior = (lab == rc) & (~boundary)
                if interior.sum() == 0:
                    continue

                # Sample anchors from boundary
                b_idx = torch.nonzero(boundary, as_tuple=False)  # (N, 2)
                if len(b_idx) > self.num_anchors:
                    perm = torch.randperm(len(b_idx))[:self.num_anchors]
                    b_idx = b_idx[perm]

                anchor_feats = feat[:, b_idx[:, 0], b_idx[:, 1]]  # (D, N_a)

                # Positive: mean interior feature
                pos_feat = feat[:, interior].mean(dim=1, keepdim=True)  # (D, 1)

                # Negatives: mean features of confusing neighbor classes
                neg_feats = []
                for neg_cls in self.confusing_neighbors.get(rc, []):
                    mask = (lab == neg_cls)
                    if mask.sum() > 0:
                        neg_feats.append(feat[:, mask].mean(dim=1, keepdim=True))
                if not neg_feats:
                    continue
                neg_feats = torch.cat(neg_feats, dim=1)  # (D, N_neg)

                # InfoNCE over features
                pos_sim = F.cosine_similarity(anchor_feats, pos_feat, dim=0) / self.temperature
                neg_sim = F.cosine_similarity(anchor_feats.unsqueeze(2), neg_feats.unsqueeze(1), dim=0) / self.temperature
                # (N_a, 1) vs (N_a, N_neg)

                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N_a, 1+N_neg)
                labels_nce = torch.zeros(len(b_idx), device=feat.device, dtype=torch.long)
                loss = loss + F.cross_entropy(logits, labels_nce)
                total_anchors += len(b_idx)

        return self.lambda_boundary * (loss / max(total_anchors, 1))

# Integration into trainer.py / trainer_v2.py compute_loss:
#   if self.use_boundary_contrastive and epoch >= boundary_start_epoch:
#       b_loss = self.boundary_contrastive_loss(
#           features=model_output["pixel_decoder_features"],  # or FPN features
#           labels=batch["pseudo_semantic"],
#       )
#       total_loss += b_loss

# Confusing neighbors for Cityscapes dead classes (pre-computed from Stage-1 confusion matrix):
CONFUSING_NEIGHBORS = {
    guard_rail_id: [fence_id, wall_id, road_id],
    tunnel_id:     [building_id, wall_id, sky_id],
    polegroup_id:  [pole_id, traffic_light_id, vegetation_id],
    caravan_id:    [car_id, truck_id, bus_id],
    trailer_id:    [truck_id, bus_id, car_id],
}
```

### Expected Impact and Risks
- **Impact:** Guard rail and tunnel should see the biggest gains because their PQ is limited by boundary overlap with fence/building (PQ penalizes boundary errors quadratically). Polegroup may benefit from separation from pole.
- **Risk:** Features from the pixel decoder may not be semantically meaningful enough for contrastive learning in early Stage-2. **Mitigation:** Apply BAPC only in Stage-4 (fine-tuning) where backbone features are well-trained, or use DINOv3 frozen features as the contrastive feature space.
- **Risk:** Boundary definition on noisy pseudo-labels is unstable. **Mitigation:** Use a wider boundary width (5 pixels) and sample multiple scales; treat 255 (ignore) pixels as neutral (neither pos nor neg).

---

## Suggested Ablations & Execution Order

Given hardware constraints (2× 1080 Ti, 11 GB), we recommend the following ablation priority:

1. **Start with Proposal 4 (AMR-ST)** in Stage-3. It requires no extra model parameters and minimal extra compute (TTA can be cached). If any dead class has > 0% PQ after Stage-2, this will likely recover it.
2. **Add Proposal 1 (RFCL)** to Stage-2 if Stage-2 pseudo-labels contain rare-class pixels. Re-run Stage-2 with rare-first sampling.
3. **Combine Proposal 2 (CC-OHEM)** in Stage-4 with existing Seesaw Loss. This directly addresses the RPN/ROI head for thing classes (caravan, trailer, polegroup).
4. **Add Proposal 3 (RC-SMix)** if Stage-2 still lacks rare-class patches; consider using a small seed patch bank from an open-vocabulary model.
5. **Add Proposal 5 (BAPC)** in Stage-4 as a final polish for stuff-class boundaries.

### Expected Final Gains
Based on similar long-tail recovery interventions in LVIS/COCO literature (2DRCL, FRACAL, Seesaw+RS), we estimate:
- **Guard rail / Tunnel:** +15–30 PQ points (stuff boundaries are highly responsive to boundary-aware losses and context mixup).
- **Caravan / Trailer:** +10–25 PQ points (thing recovery is harder due to RPN suppression, but CC-OHEM + AMR-ST jointly address it).
- **Polegroup:** +5–15 PQ points (often merged with pole; may require explicit instance separation in the clustering stage).

These gains are conservative estimates assuming pseudo-labels contain *some* signal for dead classes. If pseudo-labels are truly 0% in Stage-2, Proposals 4 and 3 (with external seed patches) become critical.

---

## References

1. Duan et al., "Long-Tailed Object Detection Pre-training: Dynamic Rebalancing Contrastive Learning with Dual Reconstruction," NeurIPS 2024.
2. Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining," CVPR 2016.
3. Pariza et al., "NeCo: Improving DINOv2's spatial representations with Patch Neighbor Consistency," ICLR 2025.
4. Martinović et al., "MC-PanDA: Empowering Domain Adaptive Panoptics with Fine-Grained Uncertainty Quantification," ECCV 2024.
5. Wang et al., "Boundary Contrastive Learning for Label-Efficient Medical Image Segmentation," BMVC 2024.
6. Tan et al., "FRACAL: Fractal Calibration for Long-tailed Object Detection," CVPR 2025.
