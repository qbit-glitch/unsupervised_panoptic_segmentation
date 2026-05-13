---
type: codebase-module
title: Pseudo-Label Pipeline (Stage 1)
project: mbps-panoptic-segmentation
language: en
tags: [codebase, pseudo-labels, stage-1]
paths:
  - mbps_pytorch/generate_*.py
  - scripts/refine_*.py
related:
  - "[[Generators-Semantic]]"
  - "[[Generators-Instance]]"
  - "[[Refinement-SIMCF]]"
  - "[[Scripts-Extract]]"
  - "[[Refs-CUPS]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Depth]]"
---

# Pseudo-Label Pipeline (Stage 1)

End-to-end dataflow for unsupervised pseudo-labels on Cityscapes / COCO-Stuff-27.

```
[image]
   │
   ├─→ DINOv2 / DINOv3 features ──┐
   │                              ▼
   │                       k-means / spectral / CAUSE ──→ semantic pseudo-labels (k=80, k=300, …)
   │                              │
   │                              └─→ refinement → see [[Refinement-SIMCF]]
   │
   ├─→ Depth (DepthPro / DA3 / SPIdepth / ZoeDepth) ──┐
   │                                                  ▼
   │                                         depth-guided splitting / spectral / Morse
   │                                                  │
   │                                                  ▼
   │                                         instance pseudo-labels
   │
   └─→ CutLER / CuVLER / DINOSAUR / MaskCut → alternative instance labels
                                                  ▲
                                                  └── inferior to depth-guided (see [[Instance-Methods]])
```

## Best Stage-1 (Cityscapes, 27-class CAUSE + Hungarian)

| Component | Method | Score |
|-----------|--------|-------|
| Semantic | k=80 raw + depth-guided splitting | PQ_stuff=32.08 |
| Instance | DepthPro depth-guided (τ=0.01, A_min=1000) | PQ_things=23.35 |
| Combined | k=80 + DepthPro instances | **PQ=28.40** |

## On-disk artefacts

| Path | Contents |
|------|----------|
| `pseudo_semantic_raw_k80/{train,val}/` | Raw k=80 cluster IDs (0–79). |
| `pseudo_semantic_raw_k80/kmeans_centroids.npz` | k-means centroids (single source of truth — never re-run k-means). |
| `pseudo_instance_spidepth/` | SPIdepth depth-guided instances. |
| `cups_pseudo_labels_depthpro/` | DepthPro instances in CUPS format. |
| `cups_pseudo_labels_dcfa_simcf_abc/` | DCFA + SIMCF-ABC pseudo-labels (see [[Refinement-SIMCF]]). |

## Linked notes

- [[Generators-Semantic]] — list of all `generate_*_semantics*.py` scripts.
- [[Generators-Instance]] — list of all `generate_*_instances*.py` scripts.
- [[Refinement-SIMCF]] — `refine_simcf.py`, `refine_simcf_v2.py`, `refine_cn_simcf.py`.
- [[Scripts-Extract]] — pre-extraction of features / codes / surface normals.
- [[Refs-CUPS]] — Stage-2 trainer that consumes these labels.

## Reports

- `reports/depthpro_instance_ablation.md`
- `reports/depth_semantic_ablation_complete.md`
- `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md`
- `reports/pseudolabel_quality_ablation.md`
- See [[Reports-Index]].
