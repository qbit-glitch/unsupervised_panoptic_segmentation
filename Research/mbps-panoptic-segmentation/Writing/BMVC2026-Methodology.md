---
type: writing
title: "BMVC 2026 — Methodology (§3)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, methodology]
updated: 2026-04-13
---

# Methodology (§3)

Two-phase pipeline: Phase 1 generates panoptic pseudo-labels, Phase 2 trains a panoptic network.

## §3.1 Overview

**Two-phase framing:**
- **Phase 1** — Pseudo-label generation (frozen foundation models, no training)
  - Semantic stream: DINOv2 + CAUSE-TR → overclustered labels (k=80)
  - Instance stream: DA3 monocular depth → Sobel + CC → instance masks
  - Merge: instance-first assembly with depth-split-ratio thing/stuff classifier
- **Phase 2** — Network training on pseudo-labels
  - Stage-2: Cascade Mask R-CNN + frozen DINOv3 ViT-B/16 backbone
  - Stage-3: EMA-guided self-training (teacher-student)

**Key design principle**: pseudo-label generator is absent at inference — deployed network sees only single RGB image.

---

## §3.2 Semantic Pseudo-Label Generation

### Problem: Centroid Collapse in CAUSE-TR

At standard k=27, CAUSE-TR features suffer severe centroid collapse:
- **14 of 27 centroids are dead** (never win argmax)
- **7 evaluation classes get zero IoU**: fence, pole, traffic light, traffic sign, rider, train, motorcycle

Absorption patterns:
| Collapsed Class | Primary Absorber | Secondary |
|----------------|-----------------|-----------|
| fence | wall (70%) | building (13%) |
| pole | building (65%) | vegetation (10%) |
| traffic light | building (89%) | vegetation (10%) |
| traffic sign | building (78%) | wall (10%) |
| rider | bicycle (43%) | person (28%) |
| train | bus (82%) | building (9%) |
| motorcycle | bicycle (71%) | car (16%) |

### Solution: Overclustering with k=80

- Fit k=80 centroids on L2-normalized 90-dim Segment_TR features
- Many-to-one mapping: each centroid → one of 19 Cityscapes classes via majority vote
- k=80 is the **smallest value** where all 7 collapsed classes reliably attract dedicated centroids

**Why 90-dim features matter**: Raw 768-dim DINOv2 tokens achieve only 46.0% mIoU vs 61.3% for Segment_TR at k=300 — a 15.3-point gap.

### Feature Extraction Details

- 322x322 sliding window, stride 161, horizontal flip
- L2-normalization → k-means equivalent to cosine similarity:
  $$\|\mathbf{f} - \mathbf{c}\|^2 = 2(1 - \mathbf{f}^\top\mathbf{c})$$
- Raw cluster assignments (0-79) saved as pseudo-labels
- Class correspondence resolved at eval via Hungarian matching (same as CUPS)

### Honest Framing

Overclustering is NOT a novel contribution — PiCIE, STEGO, HP all used it. Our contribution is the **diagnosis** of centroid collapse in frozen CAUSE-TR features and the empirical finding that k=80 is the minimum recovery threshold.

---

## §3.3 Depth-Guided Instance Pseudo-Label Generation

### Pipeline

1. Estimate dense depth: DA3 (Depth Anything v3)
2. Gaussian smoothing (sigma=1.0)
3. Sobel gradient magnitude:
   $$G(p) = \sqrt{G_x(p)^2 + G_y(p)^2}$$
4. Binary edge mask: $\mathcal{E}_\text{depth} = \{p : G(p) > \tau\}$
5. Per thing-class k: split mask $S_k = M_k \cap \neg\mathcal{E}_\text{depth}$
6. Connected component analysis on $S_k$
7. Filter: discard components < $A_\text{min}$ pixels
8. Dilation: 3-pixel reclaim boundary pixels

**Parameters**: tau=0.03, A_min=1000 (from depth model ablation in §4)

### Depth Model Hierarchy

| Model | tau | PQ_things |
|-------|-----|-----------|
| None (CC only) | — | 14.93 |
| SPIdepth | 0.20 | 19.41 |
| DAv2-L | 0.03 | 20.20 |
| **DAv3** | **0.03** | **20.90** |

**Key finding**: depth quality monotonically determines instance PQ. Alternative splitting algorithms (Canny, watershed, multiscale Sobel) provide zero improvement.

### Known Limitation: Co-Planarity Failure

When multiple instances share the same depth plane (e.g., pedestrians side by side), depth map shows no gradient at shared boundary. This is **unfixable** with depth-only methods — acknowledged in Limitations.

Impact: person PQ=4.2%, RQ=8.8% (only 170/3206 matched)

---

## §3.4 Panoptic Pseudo-Label Assembly

### Stuff/Things Classifier (No GT Labels)

**Depth-split ratio** for class k:
$$R_\text{split}(k) = \frac{\mathbb{E}_I[\text{CC}(M_k \cap \neg\mathcal{E}_\text{depth})]}{\mathbb{E}_I[\text{CC}(M_k)]}$$

- Thing classes: $R_\text{split} \gg 1$ (depth edges split adjacent objects)
- Stuff classes: $R_\text{split} \approx 1$ (no splitting)
- Top 8 classes by $R_\text{split}$ → things (excluding large-coverage classes)

### Instance-First Merging

1. Place instance masks in **descending order of area**
2. Each instance gets semantic class by **majority vote** from semantic pseudo-labels
3. Stuff segments fill remaining pixels

**Result**: PQ=27.37% with DAv3 instances — exceeds CUPS pseudo-labels (PQ=26.5%) that need stereo+flow.

---

## §3.5 Network Training

### Stage-2: Supervised Training on Pseudo-Labels

- **Architecture**: Cascade Mask R-CNN
- **Backbone**: Frozen DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`)
- **Input**: k=80 pseudo-labels from disk

**CUPS training recipe** (credited to CUPS, not our contribution):
- DropLoss: $w_i = 1 - \mathbb{1}[\max_j \text{IoU}(b_i, g_j) \leq 0.4]$ — prevents penalizing correct detections absent from noisy PLs
- CopyPaste augmentation
- Resolution jitter (384-512px)
- Gradient clipping (norm 0.1)
- Label smoothing (epsilon=0.1)

**Result**: PQ=27.87% (matches CUPS 27.8% — suggesting pseudo-label quality, not backbone, matters at Stage-2)

### Stage-3: EMA Self-Training

- Teacher initialized from Stage-2, updated via EMA:
  $$\theta_T^{(t+1)} = 0.999 \cdot \theta_T^{(t)} + 0.001 \cdot \theta_S^{(t)}$$
- Teacher generates predictions via TTA (3 scales: 0.5x, 0.75x, 1.0x + horizontal flip)
- Student retrains on teacher's high-confidence outputs for 8,000 steps
- Frozen BatchNorm, DropLoss disabled, gradient clipping norm 1.0

**Result**: PQ=32.76% (+4.89 over Stage-2)

### Self-Training Scaling Insight

| Teacher Backbone | Stage-2 PQ | Stage-3 PQ | Delta |
|-----------------|-----------|-----------|-------|
| ResNet-50 | 24.68 | 25.93 | +1.25 |
| DINOv3 ViT-B/16 | 27.87 | 32.76 | **+4.89** |

PQ_things is the primary beneficiary: +10.9 points for DINOv3. PQ_stuff nearly flat (+1.33). Self-training bootstraps instance quality beyond what depth pseudo-labels provide.

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[BMVC2026-Experiments]]
- [[BMVC2026-Supplementary]]
- [[Papers/CUPS-2025]] — Training recipe source
- [[Papers/CAUSE-2024]] — Semantic feature source
- [[Papers/DepthAnythingV3]] — Instance depth source
- [[Papers/DINOv3-2025]] — Backbone
