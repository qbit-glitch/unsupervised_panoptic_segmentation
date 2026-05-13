---
type: writing
title: "BMVC 2026 — Related Work (§2)"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, related-work, literature]
updated: 2026-04-13
---

# Related Work (§2)

Five subsections, organized **methodologically** (not paper-by-paper).

---

## §2.1 Unsupervised Semantic Segmentation

Dominant paradigm: clustering dense features from self-supervised ViTs.

| Paper | Key Idea | Our Connection |
|-------|----------|----------------|
| PiCIE (Cho et al., CVPR 2021) | Pixel-level invariance + equivariance constraints | Early baseline, foundational |
| STEGO (Hamilton et al., ICLR 2022) | Distill DINO feature correspondences into contrastive head | Improved over PiCIE |
| HP (Seong et al., 2023) | Hidden positives for clustering | Refined quality |
| EAGLE (Kim et al., 2024) | Entropy-guided learning | Further refinement |
| **CAUSE (Kim et al., 2025)** | Causal framework + modularity codebook + Segment_TR head | **We build on CAUSE-TR features** |

**Our position**: We use CAUSE-TR 90-dim features but diagnose centroid collapse at k=27 and resolve via k=80 overclustering.

---

## §2.2 Unsupervised Instance Segmentation

Methods rely on appearance or geometric cues without per-object annotations.

| Paper | Key Idea | Our Connection |
|-------|----------|----------------|
| FreeSOLO (Wang et al., 2022) | Coarse masks from self-supervised features → SOLO detector | Appearance-only |
| CutLER (Wang et al., 2023) | MaskCut — iterative normalized cuts on DINO attention | Struggles with same-class adjacent |
| CuVLER (Arica et al., 2024) | CutLER + DINOv2 features for better masks | Same limitation |
| DINOSAUR (Seitzer et al., 2023) | Slot attention on DINO features | Object-centric decomposition |
| U2Seg (Niu et al., 2024) | CutLER instances + STEGO semantics → unified panoptic | Closest to our goal |

**Our position**: All above operate on appearance features alone. Our depth-guided splitting exploits geometric prior that physically distinct objects at different depths produce gradient discontinuities — complementary signal.

---

## §2.3 Monocular Depth Estimation

| Paper | Key Idea | Our Connection |
|-------|----------|----------------|
| Monodepth2 (Godard et al., 2019) | Photometric self-supervision framework | Foundational |
| SPIdepth (Lavreniuk et al., 2025) | SOTA self-supervised depth | Our baseline depth model |
| Depth Anything v1 (Yang et al., 2024) | Foundation-scale supervised depth | — |
| Depth Anything v2 (Yang et al., 2024) | Improved with synthetic data | Ablation entry |
| **Depth Anything v3 (Lin et al., 2025)** | Massive synthetic training, unprecedented quality | **Our primary depth model** |

**Our position**: We leverage depth maps as an instance boundary signal (Sobel gradient thresholding → binary edge mask → per-class CC), not for 3D reconstruction.

---

## §2.4 Unsupervised Panoptic Segmentation

| Paper | Key Idea | Our Connection |
|-------|----------|----------------|
| **CUPS (Hahn et al., CVPR 2025)** | SF2SE3 motion segmentation from stereo flow + DINO semantic network | **Only prior UPS method on scene-centric data; our training recipe** |

CUPS achieves PQ=27.8% on Cityscapes using stereo pairs, video sequences, and optical flow.

**Our position**: We adopt the CUPS training protocol (Stage-2 + Stage-3) but **replace the pseudo-label generation entirely** — monocular images with frozen FMs instead of stereo+flow. Our PLs achieve comparable PQ_th (20.90 vs 17.7) and trained network surpasses CUPS by +5.0 PQ.

---

## §2.5 Vision Foundation Models

| Paper | Key Idea | Our Connection |
|-------|----------|----------------|
| DINO (Caron et al., 2021) | Self-supervised ViT, attention-based features | Basis for CutLER |
| **DINOv2 (Oquab et al., 2024)** | Scaled self-supervised ViT | **Used by CAUSE-TR for features** |
| **DINOv3 (Fang et al., 2025)** | 1.7B images, ViT-S to ViT-g | **Our downstream backbone (ViT-B/16)** |
| Mask2Former (Cheng et al., 2022) | Universal panoptic/instance/semantic architecture | Supervised upper bound |
| Cascade Mask R-CNN (Cai et al., 2021) | Multi-stage refinement detection | **Our detection head** |

---

## Citation Notes

All citations in `paper_bmvc2026/references.bib`. Key refs to verify:
- [ ] CUPS — hahn2025cups
- [ ] CAUSE — kim2025cause
- [ ] DINOv3 — fang2025dinov3
- [ ] DA3 — lin2025da3
- [ ] SPIdepth — lavreniuk2025spidepth
- [ ] CutLER — wang2023cutler
- [ ] U2Seg — niu2024u2seg
- [ ] DINOv2 — oquab2024dinov2
- [ ] Cascade Mask R-CNN — cai2021cascade
- [ ] Mask2Former — cheng2022mask2former

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[Papers/CUPS-2025]]
- [[Papers/CAUSE-2024]]
- [[Papers/DINOv3-2025]]
- [[Papers/DepthAnythingV3]]
- [[Papers/SPIdepth]]
- [[Papers/DINOv2-2024]]
