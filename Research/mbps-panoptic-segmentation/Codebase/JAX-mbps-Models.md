---
type: codebase-module
title: JAX/Flax — mbps/models/
project: mbps-panoptic-segmentation
module: mbps.models
framework: jax
paths:
  - mbps/models/
  - mbps/models/mbps_model.py
  - mbps/models/mbps_v2_model.py
tags: [codebase, jax, models]
related:
  - "[[JAX-mbps]]"
  - "[[JAX-mbps-Losses]]"
  - "[[JAX-mbps-Training]]"
  - "[[Refs-Backbones]]"
  - "[[Refs-Semantic]]"
  - "[[Refs-Depth]]"
---

# JAX/Flax — `mbps/models/`

End-to-end model definitions used by [[JAX-mbps-Training]].

## Top-level

| File | Role |
|------|------|
| `mbps_model.py` | v1 model: frozen DINO ViT-S/8 backbone, [[Refs-Semantic|DepthG]] semantic head, CutS3D + Cascade Mask R-CNN instance heads, projection bridge with depth conditioning, stuff/things classifier, panoptic merger. |
| `mbps_v2_model.py` | v2 model: DINOv3 ViT-B/16 (768d), simple MLP semantic head, per-pixel instance embeddings, same bridge, trained on pseudo-labels rather than fully unsupervised. |

## Subdirectories

### `backbone/` — frozen feature extractors → [[Refs-Backbones]]
- `dino_vits8.py` — DINO ViT-S/8 (384-dim).
- `dinov3_vitb.py` — DINOv3 ViT-B/16 (768-dim).
- `weights_converter.py`, `dinov3_weights_converter.py` — `.safetensors` → JAX PyTree.

### `semantic/` — semantic heads → [[Refs-Semantic]]
- `depthg_head.py` — 3-layer MLP, 384 → 90-dim semantic codes.
- `stego_loss.py` — STEGO correspondence + depth-guided correlation; consumed by [[JAX-mbps-Losses]].

### `instance/` — instance heads → [[Refs-Instance]]
- `cascade_mask_rcnn.py` — class-agnostic cascade mask head over CutS3D pseudo-masks; spatial-confidence + drop-loss.
- `cuts3d.py` — CutS3D pseudo-instance extraction (3D-aware NCut + LocalCut).
- `embedding_clustering.py` — per-pixel instance embedding clustering.
- `instance_loss.py` — Dice + BCE; see [[JAX-mbps-Losses]].

### `bridge/` — cross-modal fusion (non-mamba parts only)
- `projection.py` — adaptive projection bridge: 90-dim semantic codes and 384-dim DINO features → shared 192-dim space (forward + inverse).
- `depth_conditioning.py` — FiLM-style unified depth conditioning consumed by both semantic and instance branches.
- `bicms.py` — bidirectional cross-modal scan (token interleave + gated merge); the per-direction Mamba2 SSD it wraps is **excluded** from this graph.

### `classifier/` — stuff/things head
- `cues.py` — DBD / FCC / IDF cue extraction.
- `stuff_things_mlp.py` — MLP `[3 → 16 → 8 → 1]` over the cues.

### `merger/` — panoptic assembly
- `panoptic_merge.py` — semantic × instance × stuff/things → panoptic map (Algorithm 9 in [[Algorithms]]).
- `crf_postprocess.py` — optional dense CRF refinement.

## Data flow

```
image ─→ backbone ─┬─→ depthg_head ──→ semantic 90-dim codes
                   │
                   └─→ projection (with depth_conditioning)
                            │
                            ├─→ bridge.bicms (excluded inner SSD)
                            │
                   ┌────────┘
                   ▼
        cuts3d + cascade_mask_rcnn ──→ instance masks
                   │
                   ▼
        cues + stuff_things_mlp ──→ stuff/thing labels
                   │
                   ▼
        panoptic_merge (+ crf_postprocess) ──→ panoptic
```
