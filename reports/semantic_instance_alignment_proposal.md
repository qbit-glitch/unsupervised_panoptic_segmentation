# Semantic-Instance Alignment: Proposed Ablation Study

## Bridging the PQ Gap Through Post-Hoc and Training-Time Alignment

---

## 1. Problem Statement

Our unsupervised panoptic pipeline produces two complementary but misaligned signals:

| Source | mIoU | PQ | PQ_stuff | PQ_things |
|--------|------|----|----------|-----------|
| Stage-1 pseudo-labels (k=80 + depth split) | ~45% | 26.74 | 32.08 | **19.41** |
| Trained BiFPN + CUPS Stage-2 (ep46) | 53.1% | 24.78 | **34.54** | 11.37 |
| Theoretical oracle combination | — | **~28.2** | 34.54 | 19.41 |

The trained model improves semantic quality (PQ_stuff +2.46) but destroys instance coherence (PQ_things -8.04). The root cause is **semantic-instance class misalignment**: the refined semantic predictions disagree with the class assignments encoded in pre-computed instance masks.

### Why PQ_things Drops

Consider a pre-computed "car" instance with 500 pixels:
- **Stage-1**: All 500 pixels labeled "car" → IoU with GT ≈ 0.7 → contributes to PQ
- **Trained model**: 420 pixels "car", 50 "truck", 30 "bus" → during panoptic merge, the instance gets class "car" but loses 80 boundary pixels → IoU drops below 0.5 → **zero PQ contribution**

The panoptic quality metric is brutally unforgiving: any segment with IoU < 0.5 scores zero. Even small class disagreements at instance boundaries can push IoU below the threshold.

## 2. Proposed Ablations

### Group A: Post-Hoc Methods (Zero Training Required)

These operate on the existing best checkpoint (epoch 46, PQ=24.78) at inference time.

#### A-1: Majority Voting Within Instances
For each pre-computed instance, assign all pixels the majority semantic class predicted by the trained model.

```python
for inst_id in unique(instance_map):
    if inst_id == 0: continue  # skip background
    mask = instance_map == inst_id
    votes = semantic_pred[mask]
    majority_class = mode(votes)
    aligned_pred[mask] = majority_class
```

**Hypothesis**: Recovers PQ_things to 15-18 while preserving PQ_stuff ≈ 34.
**Cost**: Zero training, ~10 seconds inference overhead.

#### A-2: Stuff-Things Selective Merge
Use trained model predictions for stuff-class pixels, stage-1 pseudo-label class assignments for thing-class pixels (within instances).

```python
aligned_pred = trained_semantic_pred.copy()  # stuff regions
for inst_id in unique(instance_map):
    if inst_id == 0: continue
    mask = instance_map == inst_id
    aligned_pred[mask] = stage1_class_for_instance[inst_id]
```

**Hypothesis**: PQ_stuff ≈ 34.54 (trained), PQ_things ≈ 19.41 (stage-1) → PQ ≈ 28.
**Cost**: Zero training, requires stage-1 class mapping at inference.

#### A-3: Confidence-Weighted Voting
Like A-1 but weight votes by the model's softmax confidence. High-confidence pixels dominate the class assignment.

```python
for inst_id in unique(instance_map):
    mask = instance_map == inst_id
    confidences = softmax_probs[mask]  # (N, 19)
    weighted_votes = confidences.sum(dim=0)  # (19,)
    majority_class = argmax(weighted_votes)
    aligned_pred[mask] = majority_class
```

**Hypothesis**: Better than A-1 for ambiguous instances (e.g., partially occluded vehicles).

#### A-4: Majority Voting + Stuff Preservation
Combine A-1 and A-2: majority voting for thing instances, trained model for stuff.

```python
aligned_pred = trained_semantic_pred.copy()
for inst_id in unique(instance_map):
    if inst_id == 0: continue
    mask = instance_map == inst_id
    votes = trained_semantic_pred[mask]
    majority_class = mode(votes)
    # Only override if majority is a thing class
    if majority_class in THING_IDS:
        aligned_pred[mask] = majority_class
```

**Hypothesis**: Best of both — consistent thing classes + refined stuff boundaries.

### Group B: Training-Time Losses (Require Retraining)

These add alignment losses during Stage-2 training. All use the existing BiFPN + CUPS Stage-2 setup as base.

#### B-1: Instance Uniformity Loss (L_uniform)
Minimize entropy of semantic predictions within each instance:

```python
L_uniform = 0
for inst_id in unique(instance_map):
    mask = instance_map == inst_id
    probs = softmax(logits[:, :, mask])  # (19, N)
    mean_probs = probs.mean(dim=-1)  # (19,)
    L_uniform += -sum(mean_probs * log(mean_probs + eps))
L_uniform /= num_instances
```

**Weight**: λ_uniform = 0.3 (from MBPS spec)
**Hypothesis**: Forces the model to make consistent predictions within instances → PQ_things +3-5.

#### B-2: Boundary Alignment Loss (L_boundary)
Penalize semantic boundary / instance boundary mismatch:

```python
sem_boundary = sobel(semantic_pred)
inst_boundary = sobel(instance_map > 0)
L_boundary = |sem_boundary - inst_boundary|.mean()
```

**Weight**: λ_boundary = 0.2
**Hypothesis**: Aligns semantic edges with instance edges → PQ_things +1-2.

#### B-3: Instance-Aware DropLoss
Modify DropLoss to never drop pixels near instance boundaries (within 3px dilation). Currently DropLoss drops the top 30% most confident thing pixels — some of these are at instance boundaries where confidence should NOT be suppressed.

```python
inst_boundary_mask = dilate(sobel(instance_map > 0) > 0, kernel=3)
drop_mask[inst_boundary_mask] = False  # protect boundary pixels
```

**Hypothesis**: Prevents DropLoss from degrading instance boundaries → PQ_things +1-2.

#### B-4: Full Alignment (B-1 + B-2 + B-3)
All three training-time losses combined.

**Hypothesis**: Cumulative gain of +4-7 PQ_things over baseline.

### Group C: Architecture Changes (Require Model Modification)

#### C-1: Instance-Conditioned Semantic Head
Feed pre-computed instance masks as an additional input channel to the semantic decoder:

```python
inst_embedding = instance_encoder(instance_map)  # learn instance features
fused = concat(fpn_features, inst_embedding)
logits = semantic_head(fused)
```

**Hypothesis**: Model learns to respect instance boundaries → PQ_things +3-5.

#### C-2: Thing Super-Class
Replace 8 thing classes with a single "thing" super-class during training. At inference, assign specific thing classes via majority voting from stage-1.

**Hypothesis**: Reduces thing-class confusion (car/truck/bus mix-ups) → PQ_things +2-4.

## 3. Ablation Matrix

| ID | Method | Type | Training | Expected PQ | Expected PQ_th | Priority |
|----|--------|------|----------|-------------|----------------|----------|
| A-1 | Majority voting | Post-hoc | None | 25.5-26.5 | 15-18 | **P0** |
| A-2 | Stuff-things selective | Post-hoc | None | 27-28.5 | 18-19.5 | **P0** |
| A-3 | Confidence-weighted voting | Post-hoc | None | 25.5-27 | 15-18 | P1 |
| A-4 | Majority + stuff preserve | Post-hoc | None | 27-28.5 | 16-19 | **P0** |
| B-1 | L_uniform | Training | 50 epochs | 25.5-27 | 14-17 | P1 |
| B-2 | L_boundary | Training | 50 epochs | 25-26 | 12-14 | P2 |
| B-3 | Instance-aware DropLoss | Training | 50 epochs | 25-26 | 12-14 | P2 |
| B-4 | B-1 + B-2 + B-3 | Training | 50 epochs | 26-28 | 15-18 | P1 |
| C-1 | Instance-conditioned head | Architecture | 50 epochs | 26-28 | 16-19 | P2 |
| C-2 | Thing super-class | Architecture | 50 epochs | 25.5-27 | 14-17 | P2 |

## 4. Execution Plan

### Phase 1: Post-Hoc Alignment (Day 1) — No Training
1. Implement A-1, A-2, A-3, A-4 as post-processing in the evaluation pipeline
2. Test all four on the existing best checkpoint (epoch 46)
3. Results determine if training-time alignment is necessary

### Phase 2: Training-Time Losses (Days 2-4) — If Phase 1 < PQ 28
4. Implement L_uniform (B-1) and L_boundary (B-2) losses
5. Add instance-aware DropLoss (B-3)
6. Run B-1, B-4 (combined) with BiFPN + CUPS Stage-2
7. Compare post-hoc alignment on B-4's checkpoint vs. Phase 1 results

### Phase 3: Architecture Changes (Days 5-7) — If Phase 2 < PQ 28
8. Implement instance-conditioned semantic head (C-1)
9. Implement thing super-class training (C-2)
10. Run with best alignment losses from Phase 2

## 5. Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| PQ | 24.78 | 27.0 | **28.0** (beat CUPS) |
| PQ_stuff | 34.54 | 34.0+ | 35.0 |
| PQ_things | 11.37 | 17.0 | 19.0 |
| mIoU | 53.11% | 52.0%+ | 54.0% |

**Primary goal**: PQ ≥ 27.0 via alignment (achievable with post-hoc methods alone).
**Stretch goal**: PQ ≥ 28.0 to surpass CUPS (may require training-time losses).

## 6. Risks and Mitigations

1. **Majority voting may pick wrong class for ambiguous instances** → Mitigate with confidence weighting (A-3)
2. **Selective merge assumes stage-1 thing classes are correct** → Stage-1 PQ_things=19.41 validates this
3. **Training-time losses may conflict with DropLoss** → Instance-aware DropLoss (B-3) addresses this
4. **Instance masks are at original resolution, model predicts at crop resolution** → Ensure consistent resizing in evaluation pipeline

---

*This ablation study targets the semantic-instance alignment gap as the primary path to surpassing CUPS (PQ=27.8) on Cityscapes.*
