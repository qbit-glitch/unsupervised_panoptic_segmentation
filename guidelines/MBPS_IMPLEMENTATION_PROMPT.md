# ONE-SHOT PROMPT: Implement Unsupervised Mamba-Bridge Panoptic Segmentation (MBPS)

## 🎯 MISSION

You are an expert ML research engineer tasked with implementing a complete, production-ready codebase for **Unsupervised Mamba-Bridge Panoptic Segmentation (MBPS)**—a novel research project targeting NeurIPS 2026. This model fuses DepthG (semantic segmentation) and CutS3D (instance segmentation) via Mamba2 State Space Duality to achieve unsupervised panoptic segmentation.

**Your implementation must be TPU-native using JAX/Flax.** All Mamba2 operations must use chunked matrix multiplications compatible with TPU systolic arrays.

---

## 📁 REFERENCE FILES (ALREADY IN YOUR WORKING DIRECTORY)

Before writing ANY code, you MUST read and internalize these reference files:

1. **`SKILL.md`** — Contains:
   - Complete architecture specification
   - All hyperparameter defaults
   - Mathematical formulas for ALL loss functions
   - Exact directory structure to generate
   - Implementation phases with validation criteria
   - Ablation study protocol
   - Code style guidelines

2. **`mamba_panoptic_technical_report.md`** — Contains:
   - CLRS-style algorithms for all 7 critical challenges
   - Detailed mathematical derivations with gradients
   - Training curriculum (Phase A/B/C/D)
   - Gradient-balanced curriculum learning algorithm
   - Stuff-things classifier cues (DBD, FCC, IDF)

3. **`mbps_incremental_implementation_guide.md`** — Contains:
   - 46 step-by-step implementation instructions
   - Verification criteria for each step
   - Risk mitigation strategies
   - Timeline and checkpoints

4. **`mbps_architecture.svg`** — Visual architecture diagram for reference

---

## 🔧 CRITICAL IMPLEMENTATION RULES

### Rule 1: Use Battle-Tested Code from Existing Repositories

**DO NOT write core algorithms from scratch.** Port and adapt from these official repositories:

| Component | Source Repository | What to Extract |
|-----------|------------------|-----------------|
| DINO ViT-S/8 | `facebookresearch/dino` | Backbone architecture, pretrained weights |
| DepthG | `visinf/depthg` (or `leonsick/depthg`) | Semantic head, STEGO loss, depth correlation loss |
| CutS3D | `visinf/cuts3d` | NCut implementation, LocalCut 3D, instance head |
| CUPS | `visinf/cups` | Panoptic merging logic, evaluation metrics, CRF post-processing |
| Mamba2 | `state-spaces/mamba` | SSD formulation, selective scan math (adapt to JAX) |
| MFuser | `devinxzhang/MFuser` | MVFuser bridge architecture pattern |
| ZoeDepth | `isl-org/ZoeDepth` | Depth estimation (for pre-computing depth maps) |

**Porting Protocol:**
```
1. Clone the source repo
2. Identify the exact file/function needed
3. Understand the PyTorch implementation
4. Rewrite in JAX/Flax maintaining identical logic
5. Write a unit test verifying outputs match (tolerance < 1e-5)
6. Document the source with commit hash
```

### Rule 2: TPU-Native Implementation (NON-NEGOTIABLE)

All code must run efficiently on TPU v4. Follow these constraints:

```python
# ✅ CORRECT: TPU-friendly patterns
import jax
import jax.numpy as jnp
from flax import linen as nn

# Use static shapes (no dynamic sizing)
# Use powers of 2 for dimensions when possible
# Use chunk size P=128 for Mamba2 (TPU systolic array alignment)
# Use bfloat16 for matmuls
# Use jax.pmap for multi-core TPU execution
# Use jax.lax.scan for sequential operations
# Pad sequences to multiples of 128

# ❌ WRONG: GPU-only patterns (DO NOT USE)
# torch.cuda.* 
# Custom CUDA kernels
# Dynamic shapes
# Python loops over sequence length
# Non-contiguous memory access
```

**Mamba2 TPU Implementation:**
```python
# The key insight: Mamba2 SSD reformulates selective scan as matrix multiplication
# This is TPU-friendly because TPUs excel at large matmuls

# Instead of sequential: h_t = A * h_{t-1} + B * x_t (slow on TPU)
# Use chunked matmul: Y_chunk = M_chunk @ X_chunk (fast on TPU)
# Where M_chunk is the semiseparable matrix built from A, B, C

# Chunk size P=128 aligns with TPU's 128x128 systolic array
```

### Rule 3: Incremental Implementation with Verification

**DO NOT implement everything at once.** Follow the phases in `SKILL.md` exactly:

```
Phase 0 → Verify: TPU environment works, datasets load correctly
Phase 1 → Verify: DepthG mIoU within ±1.5% of paper, CutS3D AP within ±0.5%
Phase 2 → Verify: Projection bridge CKA > 0.3, reconstruction error < 0.05
Phase 3 → Verify: Depth conditioning doesn't degrade downstream metrics
Phase 4 → Verify: Mamba2 gradients flow correctly, hidden states bounded
Phase 5 → Verify: Stuff-things classifier accuracy > 80%
Phase 6 → Verify: All losses compute without NaN, gradient magnitudes balanced
Phase 7 → Verify: Training loss decreases, validation metrics improve
Phase 8 → Verify: Panoptic output format correct, no overlapping instances
```

**After each phase, STOP and verify before proceeding.**

---

## 📋 IMPLEMENTATION CHECKLIST

Execute these steps IN ORDER. Check off each item before proceeding.

### Phase 0: Environment & Data (Week 1-2)

```
[ ] 0.1 Create project structure matching SKILL.md directory layout
[ ] 0.2 Create requirements.txt:
        - jax[tpu]>=0.4.20
        - flax>=0.8.0
        - optax>=0.1.7
        - orbax-checkpoint>=0.4.0
        - einops>=0.7.0
        - ml_collections>=0.1.1
        - tensorflow>=2.15.0 (for TFRecord)
        - absl-py>=2.0.0
[ ] 0.3 Create configs/default.yaml with ALL hyperparameters from SKILL.md
[ ] 0.4 Implement data/datasets.py:
        - Port Cityscapes loader from CUPS repo
        - Port COCO-Stuff-27 loader from DepthG repo
        - Add NYU Depth V2 loader for prototyping
[ ] 0.5 Implement data/tfrecord_utils.py for TPU-efficient data loading
[ ] 0.6 Implement scripts/precompute_depth.py using ZoeDepth
[ ] 0.7 Create data/depth_cache.py for loading pre-computed depth
[ ] 0.8 Verify: Load one batch on TPU, print shapes
```

### Phase 1: Baseline Components (Week 3-6)

```
[ ] 1.1 Implement models/backbone/dino_vits8.py:
        - Port architecture from facebookresearch/dino
        - Use Flax nn.Module pattern
        - Load pretrained weights (convert from PyTorch)
        - Freeze all parameters with jax.lax.stop_gradient
        - TEST: Compare outputs with PyTorch on same input (diff < 1e-5)

[ ] 1.2 Implement models/backbone/weights_converter.py:
        - Convert DINO PyTorch state_dict to Flax params
        - Handle attention weight reshaping (PyTorch vs Flax conventions)

[ ] 1.3 Implement models/semantic/depthg_head.py:
        - Port from visinf/depthg or leonsick/depthg
        - MLP: 384 → 384 → 384 → 90
        - Output semantic codes S ∈ ℝ^(N×90)

[ ] 1.4 Implement models/semantic/stego_loss.py:
        - Port STEGO correspondence loss from mhamilton723/STEGO
        - KNN positive pair mining in DINO feature space
        - InfoNCE loss with temperature τ=0.1

[ ] 1.5 Implement losses/semantic_loss.py:
        - Combine L_STEGO + λ_d * L_DepthG
        - Use formulas from mamba_panoptic_technical_report.md
        - Depth correlation weight: w_ij = exp(-|D_i - D_j|² / 2σ²)

[ ] 1.6 Train DepthG on COCO-Stuff-27 for 30 epochs
        - VERIFY: mIoU within ±1.5% of reported 44.8%

[ ] 1.7 Implement models/instance/cuts3d.py:
        - Port Normalized Cut from visinf/cuts3d
        - Compute affinity W_ij = cos(f_i, f_j)
        - Graph Laplacian L = D - W
        - Second-smallest eigenvector via jnp.linalg.eigh
        - LocalCut: 3D point cloud from depth + DBSCAN

[ ] 1.8 Implement models/instance/cascade_mask_rcnn.py:
        - Port Cascade Mask R-CNN head from detectron2/mmdetection
        - Adapt to Flax (this is complex, consider using scenic library)

[ ] 1.9 Implement losses/instance_loss.py:
        - Spatial confidence-weighted BCE
        - DropLoss for unmatched instances
        - Box regression loss

[ ] 1.10 Train CutS3D pipeline on ImageNet-1K
         - VERIFY: Instance AP within ±0.5% of reported 10.7% on COCO val

[ ] 1.11 Implement models/merger/panoptic_merge.py:
         - Port merging logic from visinf/cups
         - Majority voting for instance class assignment
         - Greedy overlap resolution by confidence
         - Stuff region filling

[ ] 1.12 Implement evaluation/panoptic_quality.py:
         - Port PQ computation from panopticapi
         - Implement Hungarian matching with IoU > 0.5 threshold
         - Compute PQ, PQ_St, PQ_Th, SQ, RQ

[ ] 1.13 Build naive panoptic baseline:
         - Run DepthG → semantic labels
         - Run CutS3D → instance masks
         - Merge with simple heuristics
         - VERIFY: PQ ≥ 18 on Cityscapes val
```

### Phase 2: Projection Bridge (Week 7-8)

```
[ ] 2.1 Implement models/bridge/projection.py:
        - W_s: Linear(90, 192) + LayerNorm
        - W_f: Linear(384, 192) + LayerNorm
        - W_s_inv: Linear(192, 90)
        - W_f_inv: Linear(192, 384)
        - Xavier init scaled by 0.1

[ ] 2.2 Implement losses/bridge_loss.py:
        - L_recon = ||s - W_s_inv(W_s(s))||² + ||f - W_f_inv(W_f(f))||²
        - L_CKA = 1 - CKA(S', F') using formula from technical report
        - L_state = ||h_T||² (add later with Mamba)

[ ] 2.3 Train projection bridge in isolation (10 epochs):
        - Freeze DINO, DepthG, CutS3D
        - Optimize only projection weights
        - Loss = L_recon + 0.1 * L_CKA
        - VERIFY: Reconstruction error < 0.05, CKA > 0.3
```

### Phase 3: Depth Conditioning (Week 9-10)

```
[ ] 3.1 Implement models/bridge/depth_conditioning.py:
        - Sinusoidal depth encoding (frequencies [1,2,4,8,16,32])
        - MLP: 12 → 64 → 192
        - Gated conditioning: S_d = S' ⊙ σ(gate_s) + S'
        - Same for F_d

[ ] 3.2 Implement depth consistency loss in losses/consistency_loss.py:
        - Sample 1024 pixel pairs per image
        - Weight by depth similarity
        - Add to L_consistency

[ ] 3.3 Integrate depth conditioning with projection bridge
        - VERIFY: Downstream metrics don't degrade
```

### Phase 4: Mamba2 Bridge (Week 11-14) — CRITICAL FOR TPU

```
[ ] 4.1 Study Mamba2 SSD paper (arxiv:2405.21060) thoroughly
        - Understand state space duality
        - Understand semiseparable matrix formulation
        - Understand chunked computation

[ ] 4.2 Implement models/bridge/mamba2_ssd.py:
        ⚠️ THIS IS THE MOST CRITICAL FILE FOR TPU PERFORMANCE
        
        Port from state-spaces/mamba BUT rewrite for JAX/TPU:
        
        class Mamba2SSD(nn.Module):
            dim: int = 192
            state_dim: int = 64  # N
            chunk_size: int = 128  # P, TPU-aligned
            
            @nn.compact
            def __call__(self, x):
                B, L, D = x.shape
                
                # Input-dependent parameters
                delta = nn.Dense(D)(x)
                delta = jax.nn.softplus(delta)
                B_param = nn.Dense(self.state_dim)(x)
                C_param = nn.Dense(self.state_dim)(x)
                
                # Learnable A (diagonal, initialized for stability)
                A = self.param('A', nn.initializers.normal(0.01), (D, self.state_dim))
                A_bar = jnp.exp(delta[..., None] * A)  # Discretization
                
                # Chunk the sequence
                num_chunks = L // self.chunk_size
                x_chunked = x.reshape(B, num_chunks, self.chunk_size, D)
                
                # Process chunks with scan
                def process_chunk(carry, chunk_data):
                    h_prev = carry
                    x_chunk, A_chunk, B_chunk, C_chunk = chunk_data
                    
                    # Build semiseparable matrix M for this chunk
                    # M_ij = C_i^T @ A^{i-j} @ B_j for i >= j, else 0
                    M = build_semiseparable_matrix(A_chunk, B_chunk, C_chunk, self.chunk_size)
                    
                    # Contribution from previous hidden state
                    y_from_h = jnp.einsum('bdn,bpn->bpd', h_prev, C_chunk)
                    
                    # Intra-chunk computation via matmul
                    y_intra = jnp.einsum('bpq,bqd->bpd', M, x_chunk * B_chunk[..., None])
                    
                    y_chunk = y_from_h + y_intra
                    
                    # Update hidden state
                    h_new = update_hidden_state(h_prev, A_chunk, B_chunk, x_chunk)
                    
                    return h_new, y_chunk
                
                # Initial hidden state
                h_init = jnp.zeros((B, D, self.state_dim))
                
                # Scan over chunks
                _, y_chunks = jax.lax.scan(
                    process_chunk,
                    h_init,
                    (x_chunked, A_bar_chunked, B_chunked, C_chunked)
                )
                
                # Reshape back
                y = y_chunks.reshape(B, L, D)
                
                # Skip connection
                D_param = self.param('D', nn.initializers.ones, (D,))
                y = y + D_param * x
                
                return y
        
        def build_semiseparable_matrix(A, B, C, P):
            '''Build P×P lower-triangular semiseparable matrix.'''
            # M_ij = C_i^T @ (A^{i-j}) @ B_j for i >= j
            # Use cumulative product for efficiency
            indices = jnp.arange(P)
            i_minus_j = indices[:, None] - indices[None, :]  # P×P
            mask = i_minus_j >= 0
            
            # Compute A^k for k=0,1,...,P-1 using scan
            def power_scan(carry, _):
                A_power = carry
                return A_power * A, A_power
            A_init = jnp.ones_like(A)
            _, A_powers = jax.lax.scan(power_scan, A_init, None, length=P)
            
            # Gather appropriate powers
            A_diff = A_powers[i_minus_j.clip(0)]  # P×P×D×N
            
            # Compute M_ij = C_i @ A_diff @ B_j
            M = jnp.einsum('pin,pqdn,qjn->pq', C, A_diff, B) * mask
            
            return M

[ ] 4.3 Implement models/bridge/bicms.py:
        - Interleave: Z = [S_d[0], F_d[0], S_d[1], F_d[1], ...]
        - Forward scan: Y_fwd = Mamba2SSD(Z)
        - Backward scan: Y_bwd = reverse(Mamba2SSD(reverse(Z)))
        - Fusion: Y = Linear([Y_fwd; Y_bwd], D_b) + Z
        - Deinterleave: S_fused = Y[::2], F_fused = Y[1::2]

[ ] 4.4 Stack 4 Mamba2 layers with residual connections:
        - Pre-LayerNorm architecture
        - FFN between layers: Linear → GELU → Linear (expansion 2x)
        - Dropout(0.1) during training

[ ] 4.5 Add state regularization:
        - Track final hidden state h_T
        - Add L_state = 0.01 * ||h_T||² to bridge loss

[ ] 4.6 TEST Mamba2 extensively:
        - Verify output shapes
        - Verify gradients flow (no NaN/Inf)
        - Verify hidden states stay bounded (< 100)
        - Benchmark: Should process 1024 tokens in < 50ms on TPU v4
```

### Phase 5: Stuff-Things Classifier (Week 15-16)

```
[ ] 5.1 Implement models/classifier/cues.py:
        - DBD: Sobel filter on depth, threshold, count per cluster
        - FCC: Cluster covariance ratio using DINO features
        - IDF: Count instances overlapping each semantic cluster

[ ] 5.2 Implement models/classifier/stuff_things_mlp.py:
        - MLP: Linear(3,16) → ReLU → Linear(16,8) → ReLU → Linear(8,1)
        - Sigmoid output

[ ] 5.3 Create training data for classifier:
        - Compute [DBD, FCC, IDF] for all clusters in 1000 images
        - Use GT stuff/thing labels from dataset metadata
        - Split 800/200 train/val

[ ] 5.4 Train classifier:
        - Binary cross-entropy loss
        - Adam lr=1e-3, 100 epochs
        - VERIFY: Accuracy > 80% on validation
```

### Phase 6: Complete Loss Functions (Week 17-18)

```
[ ] 6.1 Verify losses/semantic_loss.py matches formulas in technical report
[ ] 6.2 Verify losses/instance_loss.py matches formulas in technical report
[ ] 6.3 Verify losses/bridge_loss.py matches formulas in technical report

[ ] 6.4 Implement losses/consistency_loss.py:
        - L_uniform: Entropy of semantic labels within each instance
        - L_boundary: |B_sem - B_inst| for adjacent pixels
        - L_DBC: Depth-boundary coherence
        - Combine with weights from SKILL.md

[ ] 6.5 Implement losses/pq_proxy_loss.py:
        - Soft IoU using sigmoid masks
        - Soft matching via Hungarian algorithm
        - Differentiable PQ approximation

[ ] 6.6 Implement losses/gradient_balancing.py:
        - Compute gradient magnitudes for each loss
        - Implement gradient projection (Algorithm 2 from technical report)
        - Implement curriculum weight scheduling

[ ] 6.7 Create losses/__init__.py with combined loss function:
        def total_loss(params, batch, phase, epoch):
            # Get loss weights based on phase and epoch
            weights = get_curriculum_weights(phase, epoch)
            
            # Compute individual losses
            l_sem = semantic_loss(...)
            l_inst = instance_loss(...)
            l_bridge = bridge_loss(...)
            l_consist = consistency_loss(...)
            l_pq = pq_proxy_loss(...)
            
            # Apply gradient balancing if Phase B
            if phase == 'B':
                # Project instance gradient
                ...
            
            # Combine
            total = (weights.alpha * l_sem + 
                     weights.beta * l_inst +
                     weights.gamma * l_bridge +
                     weights.delta * l_consist +
                     weights.epsilon * l_pq)
            
            return total, {
                'semantic': l_sem,
                'instance': l_inst,
                'bridge': l_bridge,
                'consistency': l_consist,
                'pq_proxy': l_pq,
            }

[ ] 6.8 TEST: Run one training step, verify no NaN, gradients reasonable
```

### Phase 7: Training Pipeline (Week 19-22)

```
[ ] 7.1 Implement training/curriculum.py:
        - Phase A config (epochs 1-20): semantic only
        - Phase B config (epochs 21-40): + instance with gradient projection
        - Phase C config (epochs 41-60): + bridge + consistency + PQ
        - Phase D config (self-training rounds)

[ ] 7.2 Implement training/ema.py:
        - EMA parameter tracking
        - Momentum update: θ_ema = μ * θ_ema + (1-μ) * θ
        - Integration with Flax TrainState

[ ] 7.3 Implement training/self_training.py:
        - Generate pseudo-labels with EMA teacher
        - Compute joint confidence
        - Filter by confidence threshold
        - Increase threshold each round

[ ] 7.4 Implement training/checkpointing.py:
        - Use orbax for TPU-compatible checkpointing
        - Save every 5 epochs
        - Save best model by validation PQ

[ ] 7.5 Implement training/trainer.py:
        - Main training loop with jax.pmap
        - Phase-aware loss computation
        - Logging to W&B or TensorBoard
        - Gradient clipping

[ ] 7.6 Implement scripts/train.py:
        - Argument parsing
        - Config loading
        - Training orchestration
        - Multi-phase execution

[ ] 7.7 Run full training on Cityscapes:
        - Phase A: 20 epochs
        - Phase B: 20 epochs  
        - Phase C: 20 epochs
        - Phase D: 3 rounds × 5 epochs
        - VERIFY: PQ improves each phase, final PQ ≥ 28
```

### Phase 8: Panoptic Merging & Evaluation (Week 23-24)

```
[ ] 8.1 Complete models/merger/panoptic_merge.py:
        - Instance-semantic assignment (majority voting)
        - Confidence-based overlap resolution
        - Stuff region filling
        - Output format: (instance_id, semantic_class) per pixel

[ ] 8.2 Implement models/merger/crf_postprocess.py:
        - Port dense CRF from pydensecrf or implement in JAX
        - Bilateral kernel parameters
        - 10 mean-field iterations

[ ] 8.3 Implement scripts/evaluate.py:
        - Load checkpoint
        - Run inference on val set
        - Compute all metrics
        - Save predictions for visualization

[ ] 8.4 Implement evaluation/visualizer.py:
        - Overlay panoptic predictions on images
        - Distinct colors per instance
        - Save as PNG/PDF for paper figures

[ ] 8.5 VERIFY final output:
        - No pixel belongs to multiple instances
        - All pixels have valid labels
        - PQ computation matches panopticapi
```

### Phase 9: Ablations (Week 25-28)

```
[ ] 9.1 Create configs/ablations/*.yaml for each experiment:
        - no_mamba.yaml (replace Mamba with concat+MLP)
        - no_depth_cond.yaml
        - no_bicms.yaml (forward scan only)
        - no_consistency.yaml
        - oracle_stuff_things.yaml
        - sweep_bridge_dim.yaml (D_b ∈ {128, 192, 256, 384})
        - sweep_mamba_layers.yaml (L ∈ {2, 4, 6, 8})

[ ] 9.2 Implement scripts/run_ablations.py:
        - Run all ablation configs
        - 3 seeds per experiment
        - Save results to JSON/CSV

[ ] 9.3 Run all ablations on Cityscapes

[ ] 9.4 Compile results table matching format in SKILL.md
```

### Phase 10: Testing & Documentation (Week 29-30)

```
[ ] 10.1 Implement all unit tests in tests/:
         - test_mamba2.py
         - test_losses.py
         - test_projection.py
         - test_panoptic_merge.py
         - test_metrics.py

[ ] 10.2 Run full test suite: pytest tests/ -v

[ ] 10.3 Write comprehensive README.md:
         - Installation instructions
         - Dataset preparation
         - Training commands
         - Evaluation commands
         - Pretrained model links

[ ] 10.4 Create notebooks for analysis:
         - 01_prototype_nyu_depth.ipynb
         - 02_debug_mamba_bridge.ipynb
         - 03_visualize_stuff_things.ipynb
         - 04_analyze_results.ipynb

[ ] 10.5 Generate paper figures using scripts/generate_figures.py
```

---

## ⚠️ COMMON PITFALLS TO AVOID

1. **Dynamic shapes**: TPU requires static shapes. Use padding and masking instead.

2. **Python loops over sequence**: Use `jax.lax.scan` or `jax.vmap` instead.

3. **Small matmuls**: TPU is inefficient for small matrices. Batch operations.

4. **Forgetting stop_gradient**: DINO backbone MUST be frozen with `jax.lax.stop_gradient`.

5. **NaN in losses**: Add epsilon (1e-8) to all divisions and log operations.

6. **Memory overflow**: Use gradient checkpointing for Mamba layers.

7. **Mismatched features**: Verify DINO outputs match between PyTorch and JAX before proceeding.

8. **Wrong loss weights**: Use EXACTLY the values from SKILL.md CONFIG dict.

9. **Skipping verification**: ALWAYS verify each phase before proceeding to next.

10. **Ignoring reference repos**: The battle-tested code in official repos handles edge cases you'll miss.

---

## 🏁 SUCCESS CRITERIA

Your implementation is complete when:

- [ ] All unit tests pass
- [ ] Naive baseline achieves PQ ≥ 18 on Cityscapes
- [ ] Full model achieves PQ ≥ 28 on Cityscapes (surpasses CUPS)
- [ ] Full model achieves PQ ≥ 22 on COCO-Stuff-27 (first unsupervised result)
- [ ] All ablations complete with results table
- [ ] Code runs on TPU v4-8 with batch size 64 (8 per core)
- [ ] Training completes in < 48 hours on TPU v4-8
- [ ] Documentation is complete with README and notebooks

---

## 📚 REFERENCE REPOSITORIES (Clone These First)

```bash
# Clone all reference repos
git clone https://github.com/facebookresearch/dino.git refs/dino
git clone https://github.com/visinf/cups.git refs/cups
git clone https://github.com/visinf/depthg.git refs/depthg  # or leonsick/depthg
git clone https://github.com/visinf/cuts3d.git refs/cuts3d
git clone https://github.com/state-spaces/mamba.git refs/mamba
git clone https://github.com/devinxzhang/MFuser.git refs/mfuser
git clone https://github.com/isl-org/ZoeDepth.git refs/zoedepth
git clone https://github.com/mhamilton723/STEGO.git refs/stego
```

---

## 🚀 START HERE

1. Read all reference files in working directory
2. Clone all reference repositories  
3. Set up TPU environment
4. Begin Phase 0, Step 0.1
5. Proceed incrementally, verifying each step
6. Refer to technical report for algorithm details
7. Refer to SKILL.md for exact specifications
8. Refer to incremental guide for step-by-step instructions

**Remember: Quality over speed. Verify each phase before proceeding. Use battle-tested code from reference repos. All code must be TPU-native.**

Good luck! 🎯
