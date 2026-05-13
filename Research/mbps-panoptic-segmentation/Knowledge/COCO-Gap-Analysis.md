---
type: knowledge
title: COCO Gap Analysis vs Falcon — Improvement Directions
project: mbps-panoptic-segmentation
status: active
tags: [coco, falcon, mmgd-cut, gap-analysis, improvement-directions]
updated: 2026-04-08
---

# COCO Gap Analysis vs Falcon

## The Real Gap

- **Our result**: 46.39% mIoU (MMGD-Cut, DINOv3, no post-processing)
- **Falcon reported**: 52.6% — but this **includes NAMR post-processing on SSD-1B**
- **Falcon raw NCut baseline**: 50.37% mIoU
- **Real gap to beat**: ~4 points (not 6.7)

Part of the gap is recoverable without algorithmic changes.

| Factor | Estimated contribution |
|--------|----------------------|
| PAMR/NAMR post-processing (Falcon uses NAMR) | +1–3 mIoU |
| Feature quality (DINOv3 vs their backbone) | +1–2 mIoU |
| Evaluation protocol (full 5K vs our 500) | ~0 (negligible) |

## 5 Improvement Directions (Ranked by Expected Impact)

### 1. NeCo Post-Training (~+5% mIoU) ★★★★★
- Fine-tunes DINOv2/v3 with patch neighbor consistency loss in 19 GPU hours
- Literature: +5.5% ADE20k, +5.7% COCO-Stuff for linear segmentation (ICLR 2025)
- Feature quality is our primary bottleneck → biggest single lever
- Expected: DINOv3 alone 45.79% → ~50%+; fusion 46.39% → potentially 52%+
- Cost: 19 GPU hours on GTX 1080 Ti (feasible)

### 2. NAMR Post-Processing (+1–3 mIoU) ★★★★
- Nonlinear Adaptive Mask Refinement — φ(x) = x + 1.5·ELU(x) with temperature averaging T={0.06...0.18}
- **NOT the same as standard PAMR** (which hurt us by −12.4 mIoU)
- NAMR is specifically designed for NCut outputs
- Combine with multi-modal fusion (not tested in Falcon paper) → novelty angle
- Cost: ~1 hour implementation

### 3. Adaptive Per-Image K via Eigengap Heuristic (+0.5–2 mIoU) ★★★
- Fixed K=54 is suboptimal — optimal K varies per image (CLASP, ICCV 2025)
- Eigengap-silhouette heuristic on affinity Laplacian selects K adaptively
- Novelty: Adaptive K + multi-modal affinity = principled automatic segmentation
- Cost: ~half day implementation

### 4. Cross-Attention Fusion (Learnable Per-Token Weights) (+0.5–1.5 mIoU) ★★
- Current: uniform weighted concatenation of DINOv3 + SSD-1B tokens
- Proposed: lightweight cross-attention where DINOv3 tokens attend to SSD-1B tokens
- Learns where each modality is most informative (semantics vs. boundaries)
- Novelty: spatial-adaptive multi-modal fusion for NCut (genuinely novel)
- Cost: small training loop, few GPU hours

### 5. Multi-Scale Hierarchical NCut ★
- Run NCut at 32×32 (1024 tokens) and 64×64 (4096 tokens)
- Coarse: global context; fine: boundary detail; merge via consensus
- Risk: 64×64 DINOv3 was worse in isolation (COCO-Semantic-Ablation)
- Higher risk, explore last

## Recommended Priority Order

1. NeCo post-training → biggest lever, well-validated
2. NAMR implementation → quick win, closes post-processing gap
3. Adaptive K → principled improvement, moderate effort
4. Cross-attention fusion → novel contribution for the paper
5. Multi-scale NCut → higher risk, explore last

**NeCo + NAMR alone** could push 46.39% → 52%+, matching or exceeding published Falcon while adding genuine multi-modal novelty.

## Sources
- Falcon: Fractional Alternating Cut (ICLR 2026)
- NeCo: Improving DINOv2's spatial representations (ICLR 2025)
- CLASP: Adaptive Spectral Clustering (ICCV 2025)
- DiffCut: Zero-Shot Semantic Segmentation (NeurIPS 2024)

## Links
- [[00-Hub]]
- [[Knowledge/Key-Lessons]]
