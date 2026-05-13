---
type: codebase-module
title: JAX/Flax — mbps/losses/
project: mbps-panoptic-segmentation
module: mbps.losses
framework: jax
paths:
  - mbps/losses/
tags: [codebase, jax, losses]
related:
  - "[[JAX-mbps]]"
  - "[[JAX-mbps-Models]]"
  - "[[JAX-mbps-Training]]"
---

# JAX/Flax — `mbps/losses/`

Composite training objective. Top-level exports: `SemanticLoss`, `InstanceLoss`, `BridgeLoss`, `ConsistencyLoss`, `PQProxyLoss`, `GradientBalancer`.

| File | Formula / role |
|------|----------------|
| `semantic_loss.py` | `L_semantic = L_stego + λ · L_depthg` — STEGO correspondence + depth-guided correlation. |
| `instance_loss.py` | `L_instance = L_dice + λ_drop · L_bce + λ_box · L_box`. |
| `bridge_loss.py` | `L_bridge = L_recon + λ_cka · L_cka + λ_h · L_state` — projection reconstruction, CKA alignment, state-space regularization. |
| `consistency_loss.py` | `L_consistency = λ_u · L_uniform + λ_b · L_boundary + λ_dbc · L_DBC`. |
| `pq_proxy_loss.py` | Differentiable PQ surrogate via soft-IoU matching to EMA-teacher segments. |
| `gradient_balancing.py` | Gradient-norm balancing across loss heads. |
| `instance_embedding_loss.py` | Discriminative push/pull for per-pixel instance embeddings. |
| `semantic_loss_v2.py` | Simpler cross-entropy used by `mbps_v2_model.py`. |

## Phase activation

The schedule that decides which of these to apply each epoch lives in [[JAX-mbps-Training]] (`curriculum.py`).
