# SKILL: Unsupervised Mamba-Bridge Panoptic Segmentation (MBPS)

## Skill Metadata
- **Name**: mbps-panoptic-segmentation
- **Version**: 1.0.0
- **Purpose**: Generate complete codebase for Unsupervised Panoptic Segmentation by fusing DepthG (semantic) and CutS3D (instance) via Mamba2 State Space Duality
- **Target**: NeurIPS 2026 submission
- **Framework**: JAX/Flax (TPU-optimized), with PyTorch fallback for GPU

---

## Project Overview

### Research Goal
Create a novel unsupervised panoptic segmentation model that fuses:
- **DepthG**: Depth-guided unsupervised semantic segmentation (CVPR 2024)
- **CutS3D**: 3D-aware unsupervised instance segmentation (ICCV 2025)
- **Mamba2 Bridge**: State Space Duality model for efficient feature fusion

### Key Novelty Claims
1. First Mamba-based unsupervised panoptic segmentation
2. Monocular depth-only stuff-things disambiguation (no stereo video required)
3. Cross-branch consistency losses via bidirectional SSM fusion

### Baseline to Surpass
- **CUPS** (CVPR 2025): PQ=27.8 on Cityscapes (requires stereo video)
- **U2Seg**: PQ=18.4 on Cityscapes

### Target Benchmarks
- Cityscapes (primary): Target PQ > 28
- COCO-Stuff-27: Target PQ > 22 (CUPS cannot run here)
- ADE20K, PASCAL-Context (secondary)

---

## Architecture Specification

### Component Overview
```
Input (RGB + Depth) 
    → DINO ViT-S/8 (frozen, 384-dim features)
    → [Semantic Branch (DepthG)] → S ∈ ℝ^(N×90)
    → [Instance Branch (CutS3D)] → F ∈ ℝ^(N×384)
    → [Adaptive Projection Bridge] → S', F' ∈ ℝ^(N×D_b), D_b=192
    → [Unified Depth Conditioning] → S_d, F_d
    → [Mamba2 BiCMS Bridge] → S_fused, F_fused
    → [Inverse Projections] → S_out, F_out
    → [Prediction Heads] → Semantic labels, Instance masks
    → [Stuff-Things Classifier] → Thing/Stuff classification
    → [Panoptic Merger] → Final panoptic output P = (id, class)^(H×W)
```

### Hyperparameters (Defaults)
```python
CONFIG = {
    # Architecture
    "backbone": "dino_vits8",  # Frozen
    "backbone_dim": 384,
    "semantic_code_dim": 90,
    "bridge_dim": 192,  # D_b
    "mamba_layers": 4,
    "mamba_state_dim": 64,  # N
    "mamba_chunk_size": 128,  # P, TPU-aligned
    
    # Stuff-Things Classifier
    "stc_mlp_dims": [3, 16, 8, 1],
    "stc_threshold": 0.5,
    
    # Training
    "total_epochs": 60,
    "phase_a_end": 20,
    "phase_b_end": 40,
    "self_training_rounds": 3,
    "batch_size": 8,  # per TPU core
    "learning_rate": 1e-4,
    "lr_schedule": "cosine",
    "ema_momentum": 0.999,
    "gradient_clip_norm": 1.0,
    
    # Loss weights (Phase C defaults)
    "alpha_semantic": 0.8,
    "beta_instance": 1.0,
    "gamma_bridge": 0.1,
    "delta_consistency": 0.3,
    "epsilon_pq": 0.2,
    
    # Sub-loss weights
    "lambda_depthg": 0.3,
    "lambda_drop": 0.5,
    "lambda_box": 1.0,
    "lambda_cka": 0.1,
    "lambda_recon": 0.5,
    "lambda_state": 0.01,
    "lambda_uniform": 0.3,
    "lambda_boundary": 0.2,
    "lambda_dbc": 0.2,
    
    # Self-training
    "conf_threshold_init": 0.7,
    "conf_threshold_increment": 0.05,
}
```

---

## Mathematical Specifications

### Loss Functions

#### 1. Semantic Loss
```
L_semantic = L_STEGO + λ_d · L_DepthG

L_STEGO = -Σ_{(i,j)∈P+} log[exp(s_i·s_j/τ) / Σ_k exp(s_i·s_k/τ)]
    where P+ = KNN positive pairs in DINO space, τ=0.1

L_DepthG = Σ_{i,j} w_ij^d · (1 - cos(s_i, s_j))²
    where w_ij^d = exp(-|D_i - D_j|² / 2σ_d²), σ_d=0.5
```

#### 2. Instance Loss
```
L_instance = L_BCE_CW + λ_drop · L_Drop + λ_box · L_box

L_BCE_CW = -Σ_i C_spatial(i) · [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)]
    where C_spatial = min confidence over nearby anchors

L_Drop = Σ_{unmatched m} ||f_m||²
```

#### 3. Bridge Loss
```
L_bridge = λ_r · L_recon + λ_cka · L_CKA + λ_h · L_state

L_recon = ||s - W_s†(W_s(s))||² + ||f - W_f†(W_f(f))||²

L_CKA = 1 - ||S'^T F'||_F² / (||S'^T S'||_F · ||F'^T F'||_F)

L_state = ||h_T||²  (final hidden state regularization)
```

#### 4. Consistency Loss
```
L_consistency = λ_u · L_uniform + λ_b · L_boundary + λ_dbc · L_DBC

L_uniform = (1/K) · Σ_k H(semantic | instance_k)
    (entropy of semantic labels within each instance)

L_boundary = Σ_{adj(i,j)} |B_sem(i,j) - B_inst(i,j)|

L_DBC = Σ_{adj} (B_depth - B_sem)² + (B_depth - B_inst)²
```

#### 5. Differentiable PQ Proxy
```
L_PQ = 1 - PQ_proxy

PQ_proxy = 2·TP_soft / (TP_soft + |pred| + |gt|)
    where TP_soft = Σ σ((IoU_soft - 0.5) / τ_pq), τ_pq=0.1
```

#### 6. Total Loss
```
L_total = α·L_semantic + β·L_instance + γ·L_bridge + δ·L_consistency + ε·L_PQ
```

### Stuff-Things Classifier Cues
```
DBD(c) = count(G_d[R_c] > τ_d) / |R_c|
    where G_d = sqrt((∂D/∂x)² + (∂D/∂y)²)

FCC(c) = 1 - trace(Σ_c) / trace(Σ_total)
    where Σ_c = Cov(DINO features in cluster c)

IDF(c) = num_overlapping_instances / normalized_area(c)

ST_score(c) = MLP([DBD(c); FCC(c); IDF(c)])
class(c) = "thing" if σ(ST_score) > 0.5 else "stuff"
```

### Mamba2 SSD Formulation (TPU-Optimized)
```
# Input-dependent parameters
Δ = Softplus(Linear(x))
B = Linear(x)
C = Linear(x)
Ā = exp(Δ ⊙ A)  # A is learnable diagonal

# Chunked computation (P=128)
For each chunk k:
    M_k = BuildSemiseparableMatrix(Ā_k, B_k, C_k)  # Lower triangular
    y_from_h = h_{k-1} @ C_k.T
    y_intra = M_k @ (x_k ⊙ B_k)
    y_k = y_from_h + y_intra
    h_k = UpdateState(h_{k-1}, Ā_k, B_k, x_k)

y = Concat(y_1, ..., y_{L/P}) + D · x  # Skip connection

# Bidirectional Cross-Modal Scan (BiCMS)
Z = Interleave(S_d, F_d)  # [S_1, F_1, S_2, F_2, ...]
Y_fwd = Mamba2(Z)
Y_bwd = Reverse(Mamba2(Reverse(Z)))
Y = Linear([Y_fwd; Y_bwd]) + Z  # Residual
S_fused, F_fused = Deinterleave(Y)
```

---

## Code Organization

Generate the following directory structure:

```
mbps/
├── configs/
│   ├── default.yaml
│   ├── cityscapes.yaml
│   ├── coco_stuff27.yaml
│   └── ablations/
│       ├── no_mamba.yaml
│       ├── no_depth_cond.yaml
│       ├── no_bicms.yaml
│       ├── no_consistency.yaml
│       └── oracle_stuff_things.yaml
├── data/
│   ├── __init__.py
│   ├── datasets.py          # Dataset loaders
│   ├── transforms.py        # Augmentations
│   ├── tfrecord_utils.py    # TFRecord I/O for TPU
│   └── depth_cache.py       # ZoeDepth caching utilities
├── models/
│   ├── __init__.py
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── dino_vits8.py    # DINO ViT-S/8 in Flax
│   │   └── weights_converter.py  # PyTorch → JAX conversion
│   ├── semantic/
│   │   ├── __init__.py
│   │   ├── depthg_head.py   # DepthG semantic head
│   │   └── stego_loss.py    # STEGO correspondence loss
│   ├── instance/
│   │   ├── __init__.py
│   │   ├── cuts3d.py        # NCut + LocalCut 3D
│   │   ├── cascade_mask_rcnn.py
│   │   └── instance_loss.py
│   ├── bridge/
│   │   ├── __init__.py
│   │   ├── projection.py    # Adaptive Projection Bridge
│   │   ├── depth_conditioning.py  # Unified Depth Conditioning
│   │   ├── mamba2_ssd.py    # Mamba2 SSD for TPU
│   │   └── bicms.py         # Bidirectional Cross-Modal Scan
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── cues.py          # DBD, FCC, IDF computation
│   │   └── stuff_things_mlp.py
│   ├── merger/
│   │   ├── __init__.py
│   │   ├── panoptic_merge.py
│   │   └── crf_postprocess.py
│   └── mbps.py              # Full MBPS model
├── losses/
│   ├── __init__.py
│   ├── semantic_loss.py
│   ├── instance_loss.py
│   ├── bridge_loss.py
│   ├── consistency_loss.py
│   ├── pq_proxy_loss.py
│   └── gradient_balancing.py  # Gradient projection, curriculum
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Main training loop
│   ├── curriculum.py        # Phase A/B/C/D scheduling
│   ├── ema.py               # EMA teacher
│   ├── self_training.py     # Pseudo-label generation
│   └── checkpointing.py     # Orbax checkpointing
├── evaluation/
│   ├── __init__.py
│   ├── panoptic_quality.py  # PQ, SQ, RQ metrics
│   ├── semantic_metrics.py  # mIoU
│   ├── instance_metrics.py  # AP
│   ├── hungarian_matching.py
│   └── visualizer.py
├── scripts/
│   ├── precompute_depth.py  # Cache ZoeDepth predictions
│   ├── convert_weights.py   # PyTorch → JAX
│   ├── create_tfrecords.py  # Dataset conversion
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   ├── run_ablations.py     # Run all ablation experiments
│   └── generate_figures.py  # Paper figures
├── tests/
│   ├── test_mamba2.py
│   ├── test_losses.py
│   ├── test_projection.py
│   ├── test_panoptic_merge.py
│   └── test_metrics.py
├── notebooks/
│   ├── 01_prototype_nyu_depth.ipynb
│   ├── 02_debug_mamba_bridge.ipynb
│   ├── 03_visualize_stuff_things.ipynb
│   └── 04_analyze_results.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## Implementation Phases

### Phase 0: Environment Setup
1. Create TPU v4 environment with JAX/Flax
2. Install dependencies: jax[tpu], flax, optax, orbax, einops, ml_collections
3. Prepare datasets: Cityscapes, COCO-Stuff-27, NYU Depth V2 (prototype)
4. Pre-compute ZoeDepth for all images
5. Create TFRecord pipelines

### Phase 1: Baseline Components
1. Port DINO ViT-S/8 to JAX (verify outputs match PyTorch)
2. Implement DepthG head + losses (verify mIoU within ±1.5% of paper)
3. Implement CutS3D NCut + LocalCut (verify AP within ±0.5% of paper)
4. Build naive panoptic baseline (majority voting merge)
5. Implement evaluation metrics (PQ, mIoU, AP)

### Phase 2: Projection Bridge
1. Implement W_s, W_f projections with LayerNorm
2. Implement inverse projections W_s†, W_f†
3. Implement CKA alignment loss
4. Implement reconstruction loss
5. Train bridge in isolation, verify CKA > 0.3

### Phase 3: Depth Conditioning
1. Implement sinusoidal depth positional encoding
2. Implement gated depth conditioning
3. Implement depth consistency loss
4. Integrate with projection bridge

### Phase 4: Mamba2 Bridge
1. Implement Mamba2 SSD core in JAX (chunked matmuls)
2. Implement input-dependent Δ, B, C projections
3. Implement BiCMS (interleave, forward/backward, deinterleave)
4. Stack 4 layers with residuals and LayerNorm
5. Add state regularization loss

### Phase 5: Stuff-Things Classifier
1. Implement DBD (depth boundary density)
2. Implement FCC (feature cluster compactness)
3. Implement IDF (instance decomposition frequency)
4. Train MLP classifier on cues
5. Verify accuracy > 80% on validation set

### Phase 6: Loss Functions
1. Implement full semantic loss (STEGO + DepthG)
2. Implement full instance loss (BCE_CW + Drop + Box)
3. Implement bridge loss (Recon + CKA + State)
4. Implement consistency loss (Uniform + Boundary + DBC)
5. Implement differentiable PQ proxy
6. Implement gradient-balanced aggregation

### Phase 7: Training Pipeline
1. Implement Phase A training (semantic only)
2. Implement Phase B training (+ instance with gradient projection)
3. Implement Phase C training (+ bridge + consistency + PQ)
4. Implement Phase D self-training with confidence filtering
5. Implement EMA teacher updates
6. Add checkpointing and logging

### Phase 8: Panoptic Merging
1. Implement instance-semantic class assignment
2. Implement overlap resolution (greedy by confidence)
3. Implement stuff region filling
4. Implement CRF post-processing
5. Verify output format matches panoptic standard

### Phase 9: Ablations
Create configs and run experiments for:
1. Full model vs no Mamba bridge (simple concat)
2. Full model vs no depth conditioning
3. Full model vs no BiCMS (forward-only scan)
4. Full model vs no consistency losses
5. Full model vs oracle stuff-things labels
6. Sweep: D_b ∈ {128, 192, 256, 384}
7. Sweep: Mamba layers ∈ {2, 4, 6, 8}
8. Sweep: loss weights ±50%

---

## Testing Requirements

### Unit Tests
- `test_mamba2.py`: Verify Mamba2 output shapes, gradient flow, TPU execution
- `test_losses.py`: Verify each loss computes correctly on synthetic data
- `test_projection.py`: Verify reconstruction error < 0.05
- `test_panoptic_merge.py`: Verify no pixel belongs to multiple instances
- `test_metrics.py`: Verify PQ computation matches reference implementation

### Integration Tests
- End-to-end forward pass on single batch
- End-to-end training step (verify loss decreases)
- Checkpoint save/load consistency
- Multi-TPU pmap execution

### Validation Criteria
- Phase 1 complete: Naive baseline PQ ≥ 18 on Cityscapes
- Phase 4 complete: Mamba bridge adds ≥ 2 PQ points over naive
- Phase 7 complete: Full model PQ ≥ 25 before self-training
- Phase 8 complete: Full model PQ ≥ 28 after self-training (surpass CUPS)

---

## Ablation Study Protocol

For each ablation experiment:
1. Use identical training hyperparameters (only change the ablated component)
2. Train for full 60 epochs + 3 self-training rounds
3. Evaluate on Cityscapes val set
4. Report: PQ, PQ_St, PQ_Th, SQ, RQ, mIoU, AP
5. Compute relative change vs full model
6. Run 3 seeds, report mean ± std

### Required Ablations Table
| Experiment | Description | Expected Impact |
|------------|-------------|-----------------|
| Full Model | All components | Baseline |
| - Mamba Bridge | Replace with concat + MLP | -2 to -4 PQ |
| - Depth Conditioning | Remove depth gates | -1 to -2 PQ |
| - BiCMS | Forward scan only | -1 to -2 PQ |
| - L_consistency | Remove all consistency losses | -2 to -3 PQ |
| - L_uniform | Remove semantic uniformity | -0.5 to -1 PQ |
| - L_boundary | Remove boundary alignment | -0.5 to -1 PQ |
| - Self-training | No Phase D | -1 to -2 PQ |
| + Oracle ST | Use GT stuff/things labels | +1 to +2 PQ |

---

## Code Style Guidelines

1. **JAX/Flax conventions**: Use `nn.Module` for all layers, `@nn.compact` for inline submodules
2. **Type hints**: All functions must have full type annotations
3. **Docstrings**: Google-style docstrings for all public functions
4. **Config management**: Use `ml_collections.ConfigDict` for all hyperparameters
5. **Logging**: Use `absl.logging` for all log messages
6. **Reproducibility**: Set JAX random seeds, log all hyperparameters
7. **Memory efficiency**: Use gradient checkpointing for Mamba layers
8. **TPU optimization**: Pad sequences to multiples of 128, use bfloat16

---

## Quick Start Commands

```bash
# Prototype on NYU Depth V2
python scripts/train.py --config configs/prototype_nyu.yaml --phase all

# Full training on Cityscapes
python scripts/train.py --config configs/cityscapes.yaml --phase all --tpu v4-8

# Run specific phase
python scripts/train.py --config configs/cityscapes.yaml --phase semantic_only

# Evaluate checkpoint
python scripts/evaluate.py --checkpoint path/to/ckpt --dataset cityscapes

# Run all ablations
python scripts/run_ablations.py --config configs/cityscapes.yaml --output results/ablations/

# Generate paper figures
python scripts/generate_figures.py --results results/ --output figures/
```

---

## Key Implementation Notes

### Mamba2 TPU Optimization
- Use chunk size P=128 (matches TPU systolic array)
- Implement semiseparable matrix as explicit matmul, not sequential scan
- Use `jnp.einsum` with optimal contraction paths
- Apply `jax.lax.scan` for sequential state updates within chunks
- Use bfloat16 for all matrix multiplications

### Gradient Projection (Phase B)
```python
def project_gradients(g_sem, g_inst):
    """Remove conflicting component from instance gradient."""
    dot = jnp.sum(g_sem * g_inst)
    conflict = jnp.minimum(0.0, dot)
    g_inst_proj = g_inst - (conflict / (jnp.sum(g_sem**2) + 1e-8)) * g_sem
    return g_inst_proj
```

### Confidence-Weighted Self-Training
```python
def compute_joint_confidence(sem_logits, inst_scores, alpha=0.5):
    """Combine semantic and instance confidence."""
    c_sem = jnp.max(jax.nn.softmax(sem_logits), axis=-1)
    c_inst = jax.nn.sigmoid(inst_scores)
    c_joint = (c_sem ** alpha) * (c_inst ** (1 - alpha))
    return c_joint
```

### Panoptic Merging Priority
1. Sort instances by confidence (descending)
2. Assign pixels to highest-confidence overlapping instance
3. Fill remaining pixels with stuff predictions
4. Merge contiguous same-class stuff regions

---

## References

- Panoptic Segmentation: https://arxiv.org/abs/1801.00868
- CUPS (baseline): https://arxiv.org/abs/2504.01955
- Mamba as a Bridge: https://arxiv.org/abs/2504.03193
- DepthG: https://arxiv.org/abs/2309.12378
- CutS3D: https://arxiv.org/abs/2411.16319
- Mamba2 SSD: https://arxiv.org/abs/2405.21060
