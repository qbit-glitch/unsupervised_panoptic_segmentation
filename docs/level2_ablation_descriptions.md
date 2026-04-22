# Level-2 Dead-Class Recovery Ablations — Short Descriptions

## Seesaw Loss

**Seesaw Loss** (Wang et al., CVPR 2021) is a long-tail classification loss that reweights per-sample gradients based on two factors: (1) a **mitigation factor** that reduces penalty when a rare class is predicted but the GT is frequent, and (2) a **compensation factor** that upweights hard negatives when the model is overconfident on frequent classes. It maintains online cumulative sample counts per class and rescales logits so tail classes aren't drowned out by head-class gradients. In the box head, it replaces standard CE to give dead/rare thing classes (caravan, trailer, etc.) comparable gradient magnitude to frequent classes (road, building).

**"Seesaw-enabled"** means `USE_SEESAW_LOSS: True` in `MODEL.ROI_BOX_HEAD` — the Cascade R-CNN box classifier uses Seesaw instead of plain cross-entropy.

---

## Ablations

### L2A — BACC-TS (Boundary-Aware Class-Balanced CE + Temperature Scaling)

Combines three semantic-head interventions: (1) **class-frequency weighting** (inverse-sqrt normalized weights) on the CE loss so rare stuff classes contribute equally to frequent ones; (2) **boundary-aware CE aux loss** that detects semantic boundaries via dilation and up-weights boundary pixels 3×, forcing sharper class edges where dead classes often bleed into neighbors; (3) **temperature-scaled KD** (T=2.0) that preserves soft pseudo-label distributions instead of hard argmax, preserving fine-grained class relationships for rare stuff. *Papers: Class-balanced loss (Cui et al., CVPR 2019); boundary CE (generalized from segmentation literature); temperature scaling (Hinton et al., NeurIPS 2015 distillation).*

### L2B — SSRCTS (Stage-Specific Rare-Class Sampling)

After standard Cascade R-CNN matching assigns proposals to GT boxes, this hook checks each stage's output. If fewer than 2 rare-class proposals survived, it computes IoU between **background proposals** and **rare GT boxes** at a relaxed threshold (0.35 instead of 0.5), then promotes the top candidates to foreground with the rare class label. This guarantees every training image feeds at least 2 rare-class RoIs per cascade stage. *Papers: Inspired by OHEM and Libra R-CNN rare-class sampling; relaxed IoU from soft-label detection literature.*

### L2C — EQL v2 (Equalization Loss v2)

Replaces Seesaw with Equalization Loss v2, which tracks online EMA of positive and negative gradient magnitudes per class. For each forward pass, it computes the ratio `pos_grad / neg_grad` per class and rescales the loss so classes with weak positive gradients (rare/dead classes) get boosted. Unlike Seesaw, which uses sample counts, EQLv2 uses actual gradient statistics, making it more responsive to training dynamics. *Paper: Tan et al., "Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection," CVPR 2021.*

### L2D — RFCL (Rare-First Curriculum Learning)

Wraps the training dataloader in a curriculum sampler. At step 0, images containing rare-class pixels are sampled with probability 0.8; by step 8000 this decays to 0 (uniform). The sampling weight per image is proportional to its rare-class pixel fraction, raised to a temperature. This forces the model to see dead-class examples early when gradients are largest, then gradually mixes in the full dataset as the rare-class representations stabilize. *Papers: Curriculum learning (Bengio et al., ICML 2009); rare-first sampling adapted from decoupled training literature (Kang et al., ICLR 2020).*

### L2E — Combined (BACC-TS + SSRCTS + EQL v2 + RFCL)

Stacks all four singles simultaneously: BACC-TS on the semantic head, SSRCTS in all three cascade stages, EQLv2 in the box head, and RFCL in the dataloader. Tests whether the interventions are additive or interfere. *No new paper — compositional test.*

### L3A — AMR-ST (Asymmetric Multi-Round Self-Training)

Extends Stage-3 EMA self-training with asymmetric confidence thresholds: frequent stuff classes use τ=0.7, dead/rare classes use τ=0.35 (decaying 0.1 per round, min 0.15). Rare-class logits are sharpened with T=0.7 before thresholding, and per-round adaptive lowering lets the teacher retain more rare-class pseudo-labels as training progresses. *Papers: Self-training thresholds from CUPS/FixMatch; asymmetric thresholds adapted from class-aware pseudo-labeling in semi-supervised segmentation.*

---

## Baseline

**AnyDesk Baseline:** Seesaw-enabled Stage-2 training with DINOv3 ViT-B/16 frozen backbone, PanopticFPN, 3-stage Cascade Mask R-CNN. Effective batch 16 (bs=8, accum=2, 1× RTX Pro 6000 48GB). Matches the Santosh Stage-2 config that produced the checkpoint later refined to PQ=35.83% in Stage-3.
