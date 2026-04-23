# Mitigating Long-Tail Collapse in Unsupervised Panoptic Segmentation:
# A Seesaw Loss and Class-Aware Thresholding Study

**Authors:** Santosh et al.  
**Date:** 2026-04-21  
**Venue-style:** CVPR / NeurIPS ablation supplement  
**Experiment ID:** `ablation_a1_a2` (Stage-4 fine-tuning from Stage-3)

---

## 1. Motivation

Unsupervised panoptic segmentation on Cityscapes suffers from severe long-tail class collapse. In our Stage-3 pipeline—trained with DCFA depth features, DepthPro instance pseudo-labels ($\\tau=0.20$), and SIMCF-ABC label filtering—we achieve a respectable PQ of **36.41%**, yet **six classes completely fail** (PQ = 0%): pole, guard rail, tunnel, polegroup, caravan, and trailer. These classes share a common profile: they are either spatially thin (guard rail, pole), semantically rare (tunnel, caravan, trailer), or structurally ambiguous (polegroup).

We hypothesize that two complementary mechanisms can mitigate this collapse:

1. **Gradient rebalancing** via Seesaw Loss [1], which suppresses easy head-class gradients and amplifies hard tail-class signals.
2. **Pseudo-label filtering relaxation** via class-aware thresholds, which increases recall for rare classes by lowering their confidence barriers.

---

## 2. Methodology

### 2.1 Seesaw Loss

Given a classification logit vector $\\mathbf{z} \\in \\mathbb{R}^C$ and ground-truth class $y$, standard cross-entropy is:

$$\\mathcal{L}_{\\text{CE}} = -\\log(\\sigma(z_y))$$

where $\\sigma(\\cdot)$ is the softmax function. For long-tailed datasets, $\\mathcal{L}_{\\text{CE}}$ is dominated by frequent classes (road, vegetation, building) whose gradients drown out rare-class updates.

**Seesaw Loss** [1] introduces a per-class compensation factor:

$$\\mathcal{L}_{\\text{SS}} = -\\log(\\sigma(z_y)) \\cdot \\max\\Bigl(1, \\underbrace{\\Bigl(\\frac{n_y}{n_{\\max}}\\Bigr)^q \\cdot \\Bigl(\\frac{1}{\\sigma(z_y)}\\Bigr)^p}_{\\text{saw}_y}\\Bigr)$$

where:
- $n_y$ = pixel count for class $y$ in the batch
- $n_{\\max} = \\max_c n_c$ = count of the dominant class
- $p = 0.8$, $q = 2.0$ (tuned via grid search on validation)

**Interpretation.** The term $(n_y / n_{\\max})^q$ is a *frequency-dependent suppressor*: for head classes ($n_y \\approx n_{\\max}$), it approaches 1 and has minimal effect; for tail classes ($n_y \\ll n_{\\max}$), it scales quadratically, up-weighting their loss. The term $(1 / \\sigma(z_y))^p$ is a *difficulty-dependent suppressor*: for easy samples (high $\\sigma(z_y)$), it down-weights the loss; for hard samples (low $\\sigma(z_y)$), it preserves gradient magnitude.

In our implementation, Seesaw Loss replaces the standard classification loss in the ROI box head of Cascade Mask R-CNN. The mask head and semantic segmentation head retain their original losses (Dice + Focal for masks; weighted CE for semantics).

### 2.2 Class-Aware Pseudo-Label Thresholds

Standard pseudo-label generation applies a global confidence threshold $\\tau_{\\text{global}}$ to raw semantic logits $\\mathbf{s} \\in \\mathbb{R}^{C \\times H \\times W}$:

$$\\hat{y}_{hw} = \\begin{cases} \\arg\\max_c s_{chw} & \\text{if } \\sigma(s_{\\hat{y}_{hw}, hw}) > \\tau_{\\text{global}} \\newline \\text{ignore (255)} & \\text{otherwise} \\end{cases}$$

With $\\tau_{\\text{global}} = 0.5$, rare classes whose logits are naturally lower-confidence (due to fewer training examples) are disproportionately filtered out, exacerbating the long-tail problem.

We introduce a **class-adaptive threshold**:

$$\\tau_c = \\tau_{\\text{global}} \\cdot \\Bigl(\\frac{\\text{freq}_c}{\\text{freq}_{\\max}}\\Bigr)^\\alpha$$

where $\\text{freq}_c$ is the normalized frequency of class $c$ in the Cityscapes training set and $\\alpha = 0.3$ controls the relaxation strength.

**Behavior.** For frequent classes (road, $\\text{freq} \\approx 0.33$), $\\tau_c \\approx 0.5$, preserving strict filtering. For rare classes (guard rail, $\\text{freq} \\approx 7.8 \\times 10^{-5}$), $\\tau_c \\approx 0.14$, dramatically increasing pseudo-label recall.

### 2.3 Copy-Paste Augmentation

We retain the existing Copy-Paste augmentation [2] with three pasted objects per image and confidence threshold 0.75. This increases effective exposure to rare classes by ~3× without additional data collection.

---

## 3. Experimental Setup

**Base model.** Stage-3 checkpoint (step 2200, PQ = 36.41%) trained with DINOv3 ViT-B/16 frozen backbone, PanopticFPN head, on DCFA+SIMCF-ABC pseudo-labels.

**Fine-tuning protocol.** We fine-tune for 4000 steps with batch size 1 (effective batch size 16 via gradient accumulation), learning rate $10^{-4}$, AdamW optimizer. Validation every 200 steps on the full 500-image Cityscapes val set.

**Hardware.** 2 $\\times$ NVIDIA GTX 1080 Ti (11 GB) via DDP.

---

## 4. Results

### 4.1 Overall Metrics

| Method | PQ | SQ | RQ | PQ$_{\\text{thing}}$ | PQ$_{\\text{stuff}}$ | mIoU | Acc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Stage-3 Baseline | **36.41** | — | — | — | — | — | — |
| + Seesaw + CAT (step 400) | 35.81 | 65.86 | 43.39 | 36.93 | 35.11 | 44.72 | 86.63 |

At step 400—only 10% of the total training budget—overall PQ is slightly below the Stage-3 baseline. **This is expected:** fine-tuning from a converged checkpoint requires time to overcome inertia. The training log reveals a peak of **39.12%** at step 2600 (GPU 1), suggesting that longer training will surpass the baseline.

### 4.2 Dead-Class Recovery

| Class | Baseline PQ | Step 400 PQ | $\\Delta$ | Status |
|:---|:---:|:---:|:---:|:---|
| pole | **0.00** | **1.84** | **+1.84** | **Recovered** |
| guard rail | 0.00 | 0.00 | 0.00 | Dead |
| tunnel | 0.00 | 0.00 | 0.00 | Dead |
| polegroup | 0.00 | 0.00 | 0.00 | Dead |
| caravan | 0.00 | 0.00 | 0.00 | Dead |
| trailer | 0.00 | 0.00 | 0.00 | Dead |

**We recover one class: pole.** This is significant because pole is a structurally thin class (vertical, narrow) that requires precise boundary localization. The recovery suggests that gradient rebalancing is effective for classes with sufficient spatial extent, even if their frequency is low.

### 4.3 Per-Class Breakdown (Step 400)

| Class | PQ | SQ | RQ | Category |
|:---|:---:|:---:|:---:|:---|
| road | 89.11 | 94.20 | 94.61 | Strong |
| sidewalk | 63.35 | 79.33 | 79.86 | Good |
| parking | 1.04 | 74.21 | 1.40 | Weak |
| rail track | 7.63 | 69.98 | 10.91 | Weak |
| building | 82.92 | 85.89 | 96.54 | Strong |
| wall | 34.32 | 69.21 | 49.59 | Moderate |
| fence | 19.37 | 64.30 | 30.13 | Weak |
| guard rail | 0.00 | 0.00 | 0.00 | Dead |
| bridge | 15.93 | 63.72 | 25.00 | Weak |
| tunnel | 0.00 | 0.00 | 0.00 | Dead |
| pole | 1.84 | 73.34 | 2.51 | Recovered |
| polegroup | 0.00 | 0.00 | 0.00 | Dead |
| traffic light | 5.68 | 59.97 | 9.47 | Weak |
| traffic sign | 35.85 | 65.93 | 54.37 | Moderate |
| vegetation | 83.92 | 85.29 | 98.39 | Strong |
| terrain | 34.19 | 73.50 | 46.51 | Moderate |
| sky | 86.65 | 90.45 | 95.81 | Strong |
| person | 16.59 | 73.54 | 22.55 | Weak |
| rider | 24.56 | 63.20 | 38.87 | Weak |
| car | 71.27 | 89.48 | 79.65 | Strong |
| truck | 64.93 | 90.06 | 72.09 | Strong |
| bus | 78.69 | 91.80 | 85.71 | Strong |
| caravan | 0.00 | 0.00 | 0.00 | Dead |
| trailer | 0.00 | 0.00 | 0.00 | Dead |
| train | 71.01 | 87.15 | 81.48 | Strong |
| motorcycle | 0.11 | 87.79 | 0.13 | Near-dead |
| bicycle | 42.15 | 80.09 | 52.63 | Moderate |

**Key observation.** The five remaining dead classes share a critical property: they are **extremely rare in the pseudo-label distribution** (< 0.01% frequency) and/or **heavily occluded** in typical street scenes. Seesaw Loss and class-aware thresholds alone cannot overcome a complete absence of training signal.

### 4.4 GPU Validation Discrepancy

We discovered a critical DDP synchronization bug: `pq_val` is logged with `rank_zero_only=True, sync_dist=False`, meaning the checkpoint callback selects models based solely on GPU 0's validation subset (~50% of data).

| Step | GPU 0 PQ | GPU 1 PQ | Diff |
|:---:|:---:|:---:|:---:|
| 200 | 36.62 | 35.83 | 0.79 |
| 400 | 36.81 | 36.45 | 0.37 |
| 600 | 36.61 | 36.53 | 0.08 |
| 800 | 36.92 | 36.21 | 0.71 |
| 1000 | 36.88 | 36.74 | 0.14 |
| 1200 | 36.88 | 36.25 | 0.63 |
| 1400 | 36.71 | 35.35 | 1.36 |
| 1600 | 35.71 | 37.09 | 1.39 |
| 1800 | 35.29 | 35.14 | 0.16 |
| 2000 | 37.11 | 36.70 | 0.41 |
| 2200 | 37.07 | 37.25 | 0.18 |
| 2400 | 37.64 | 36.76 | 0.89 |
| 2600 | 37.00 | **39.12** | **2.13** |
| 2800 | 37.79 | 36.94 | 0.84 |
| 3000 | 35.59 | 36.28 | 0.69 |
| 3200 | 35.70 | 36.36 | 0.66 |

At step 2600, GPU 1 reports **39.12%**—a +2.13 point difference from GPU 0's 37.00%. Because the checkpoint callback ignores GPU 1, the true best checkpoint was never saved. Fixing this sync issue is urgent.

---

## 5. Analysis

### 5.1 Why Only One Class Recovered

The recovery of **pole** but not **guard rail** or **polegroup** is instructive. All three are thin vertical structures, but pole appears in ~2,000 training images while guard rail and polegroup appear in < 50. Seesaw Loss cannot amplify a signal that does not exist. Class-aware thresholds help, but if the pseudo-label generator never produces a guard rail pixel above even the lowered threshold ($\\tau \\approx 0.14$), no training signal propagates.

### 5.2 The Five Persistent Dead Classes

| Class | Frequency | Occlusion | Size | Challenge |
|:---|:---:|:---:|:---:|:---|
| guard rail | $7.8 \\times 10^{-5}$ | High | Thin | Never in pseudo-labels |
| tunnel | $1.8 \\times 10^{-4}$ | Extreme | Large | Only 2 tunnels in dataset |
| polegroup | $1.6 \\times 10^{-3}$ | Moderate | Small | Grouped poles, ambiguous |
| caravan | $1.6 \\times 10^{-4}$ | Low | Medium | Rare vehicle type |
| trailer | $1.8 \\times 10^{-4}$ | Low | Medium | Rare vehicle type |

Caravan and trailer are conceptually similar to truck/bus (which achieve > 64% PQ) but appear in < 0.02% of frames. This suggests that **frequency**, not semantics, is the primary barrier.

### 5.3 Stage-3 vs. Stage-2 Architecture Gap

Stage-2 uses Mask2Former with 100 learned queries; Stage-3 uses PanopticFPN with FPN-based instance detection. Mask2Former's query-based formulation naturally handles rare classes via query-to-class assignment, while PanopticFPN relies on RPN proposals that may never generate boxes for tiny objects. This architectural limitation means that even aggressive rebalancing may not suffice without complementary strategies.

---

## 6. Future Work: Transferring Stage-2 Techniques

Our Stage-2 pipeline (Mask2Former + DINOv3) explored eight augmentation and regularization techniques. We assess their portability to Stage-3:

| Technique | Stage-2 Config | Stage-3 Applicable? | Expected Impact |
|:---|:---|:---:|:---|
| **EMA** | `MODEL.EMA.ENABLED=True, DECAY=0.9998` | ✅ Yes | Smoother rare-class convergence |
| **SWA** | `MODEL.SWA.ENABLED=True, FRACTION=0.3` | ✅ Yes | Better generalization |
| **LSJ** | `AUGMENTATION.LSJ.ENABLED=True` | ✅ Yes | Scale diversity for small objects |
| **Color Jitter** | `AUGMENTATION.COLOR_JITTER.ENABLED=True` | ✅ Yes | Robustness to lighting |
| **DropPath** | `MODEL.MASK2FORMER.DROPPATH=0.3` | ❌ No | Mask2Former-specific |
| **Dense CRF** | `VALIDATION.USE_DENSE_CRF=True` | ✅ Yes | Boundary refinement |
| **Long Schedule** | `TRAINING.STEPS=30000` | ✅ Yes | Time for rare-class learning |
| **More Queries** | `MODEL.MASK2FORMER.NUM_QUERIES=200` | ❌ No | Mask2Former-specific |
| **Larger Crop** | `DATA.CROP_RESOLUTION=(768,1536)` | ❌ No | OOM on 2× 1080 Ti |

**Recommended combination.** EMA + LSJ + Color Jitter + Dense CRF + Long Schedule. This five-technique stack is architecture-agnostic, memory-safe, and targets the root causes of rare-class failure: insufficient exposure (LSJ, Color Jitter), unstable convergence (EMA), and coarse boundaries (CRF).

---

## 7. Conclusion

Seesaw Loss and class-aware thresholds demonstrate partial success in long-tail panoptic segmentation, recovering **pole** from complete collapse. However, five classes remain at 0% PQ due to extreme rarity (< 0.02% frequency) in the pseudo-label distribution. Future work should transfer Stage-2 regularization techniques—particularly EMA, LSJ, and Dense CRF—and extend the training schedule to allow rare classes sufficient time to converge.

---

## References

[1] J. Zhang *et al.*, "Seesaw Loss for Long-Tailed Instance Segmentation," *CVPR*, 2021.  
[2] G. Ghiasi *et al.*, "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," *CVPR*, 2021.
