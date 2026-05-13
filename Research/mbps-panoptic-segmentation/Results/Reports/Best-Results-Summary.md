---
type: result
title: Best Results Summary
project: mbps-panoptic-segmentation
language: en
updated: 2026-03-30T10:00:00Z
---

# Best Results Summary

## Cityscapes Val

### Overall Best: DINOv3 Stage-3 (step 1800)
| Metric | Value |
|--------|-------|
| PQ (27-class CAUSE+Hungarian) | **30.255%** |
| PQ_things | 28.495% |
| PQ_stuff | 31.291% |
| Comparison | CUPS 27.8% (+2.5 PQ) |

### Best Semantics: UNet P2-B Attention (ep8)
| Metric | Value |
|--------|-------|
| PQ (19-class standard) | 28.00 |
| PQ_stuff | 35.04 |
| PQ_things | 18.32 |
| mIoU | 57% |

### Best Instances: Mumford-Shah (Phase A, 100 imgs)
| Metric | Value |
|--------|-------|
| PQ_things | 23.27 |
| Delta vs Sobel+CC | +3.86 (+19.9%) |
| Best config | alpha=1.0, beta=1.0, k=10, A_min=1000 |
| Awaiting | Phase B (500 imgs) validation |

### Stage-1 Pseudo-Labels
| Metric | Value |
|--------|-------|
| PQ | 26.74 |
| PQ_stuff | 32.08 |
| PQ_things | 19.41 (Sobel+CC) |

## COCO-Stuff-27 Semantic Segmentation (500 val images)

### Best: MMGD-Cut (DINOv3+SSD-1B Fusion, 2026-04-02)
| Metric | Value |
|--------|-------|
| mIoU | **46.39%** |
| Things mIoU | 45.03% |
| Stuff mIoU | 47.48% |
| Method | Multi-modal Falcon NCut (DINOv3+SSD-1B, K=54, alpha=5.5, reg_lambda=0.7) |
| Gap to published Falcon | -6.21 (52.6%) |

### Falcon NCut (SD-1.4 only, 2026-03-31)
| Metric | Value |
|--------|-------|
| mIoU | **42.02%** |
| Things mIoU | 38.41% |
| Stuff mIoU | 44.91% |
| Config | K=54, alpha=5.5, reg_lambda=0.7, no PAMR |

### Pseudo-Label Methods (k-means / SAM)
| Method | K | mIoU | Things | Stuff |
|--------|---|------|--------|-------|
| k-means | 3000 | **38.4%** | 44.1% | 33.7% |
| SAM consensus | 1000 | **34.4%** | 41.1% | 29.1% |
| Spectral enrichment | 1000 | 33.5% | 41.6% | 26.9% |
| k-means | 1000 | 31.9% | 39.6% | 25.9% |

MMGD-Cut surpasses all k-means baselines. DINOv3 features are the primary driver (+3.77 over SD-only). SSD-1B fusion pending.

## Cross-Dataset (DINOv3 Stage-3 step 8000)
| Dataset | PQ | PQ_things | PQ_stuff |
|---------|-----|-----------|----------|
| MOTS (2,862 imgs) | 63.4% | 30.4% | 96.4% |
| KITTI (200 imgs) | 29.3% | 24.9% | 32.0% |
| COCO-Stuff-27 (5,000 imgs) | 8.0% | 7.3% | 8.6% |

## Baselines
| Method | PQ | Notes |
|--------|-----|-------|
| CUPS (CVPR 2025) | 27.8 | 27-class CAUSE+Hungarian |
| Semi-Supervised | 30.4 | Uses GT labels |
| CUPS Stage-2 (our) | 24.68 | step 6500, 8000 total |
| RepViT+BiFPN (our) | 24.78 | Mobile, 5.05M params |
