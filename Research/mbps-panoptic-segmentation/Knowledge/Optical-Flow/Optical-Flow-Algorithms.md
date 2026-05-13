---
type: knowledge
title: "Optical Flow Algorithm Patterns"
project: mbps-panoptic-segmentation
status: active
updated: 2026-05-07
tags:
  - optical-flow
  - algorithms
  - literature-map
---

# Optical Flow Algorithm Patterns

[[Papers/Optical-Flow/Optical-Flow-SMURF-Lineage|SMURF lineage]] | [[Optical-Flow-Losses]] | [[Optical-Flow-MBPS-Relevance]]

## Patterns

| Pattern | Papers | Mechanism | When To Use |
| --- | --- | --- | --- |
| Unsupervised RAFT modernization | [[SMURF-2021]], [[Reviving-Unsupervised-Optical-Flow-2026]], [[U2Flow-2026]] | recurrent all-pairs matching with photometric, smoothness, self-supervision losses | Fair comparison to CUPS/SMURF-style pseudo-label generation |
| Uncertainty-aware flow | [[U2Flow-2026]] | predict log variance, use reliability to modulate refinement/loss/fusion | Filtering unreliable flow for scene-flow or instance proposals |
| Synthetic data search | [[Self-Supervised-AutoFlow-2023]] | learn renderer hyperparameters from unlabeled target-domain proxy loss | Adapt flow pretraining to a target video domain without labels |
| Transformer cost-volume pretraining | [[FlowFormerPlusPlus-2023]] | masked cost-volume autoencoding before supervised fine-tuning | Strong supervised teacher; representation pretraining ideas |
| Multi-frame feature propagation | [[VideoFlow-2023]], [[MEMFOF-2025]], [[ARFlow-2026]] | propagate temporal/motion features beyond two frames | Occlusions and out-of-frame motion in videos |
| Memory-efficient high resolution | [[MEMFOF-2025]], [[DPFlow-2025]] | lower-resolution correlation, dual/adaptive pyramids, native high-resolution training | Avoid crop/downsample/tiling artifacts |
| Foundation-prior global matching | [[MegaFlow-2026]] | DINO/VGGT-style features, global correspondence, local recurrent refinement | Large displacement, point tracking, cross-domain teacher labels |

## Lineage

- [[SMURF-2021]] is the anchor for CUPS-style unsupervised flow.
- [[Self-Supervised-AutoFlow-2023]] treats SMURF losses as a proxy metric for data generation.
- [[Reviving-Unsupervised-Optical-Flow-2026]] simplifies and reproduces the unsupervised RAFT path.
- [[U2Flow-2026]] adds uncertainty to the same path and is the closest direct successor.
- [[VideoFlow-2023]] establishes the modern supervised multi-frame flow family.
- [[MEMFOF-2025]] and [[ARFlow-2026]] make multi-frame flow more memory-efficient and temporally scalable.
- [[DPFlow-2025]] attacks resolution generalization.
- [[MegaFlow-2026]] moves from RAFT local matching toward foundation-feature global matching.

## Implementation Implications

- For a CUPS-compatible drop-in, the output must remain dense pairwise flow between adjacent frames and must be usable by SF2SE3 or related scene-flow code.
- Uncertainty maps from [[U2Flow-2026]] are especially useful because MBPS instance failures often come from unreliable geometric boundaries; they could become masks for rejecting bad flow-derived proposals.
- Supervised or foundation teachers can be used to generate pseudo-labels, but the method section must not describe them as unsupervised optical flow unless the training setup is actually unlabeled.

