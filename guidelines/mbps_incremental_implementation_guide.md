# MBPS: Step-by-Step Incremental Implementation Guide

## Complete Roadmap from Zero to NeurIPS 2026 Submission

---

## Phase 0: Environment & Infrastructure Setup

### Step 0.1: Set Up TPU Development Environment
Create a Google Cloud TPU v4 pod slice (v4-8 or v4-32) with JAX/Flax pre-installed, configure your project with appropriate quotas, install core dependencies (jax, flax, optax, orbax for checkpointing), and set up a persistent GCS bucket for storing datasets, checkpoints, and logs—ensure you test a simple JAX pmap operation to verify multi-core TPU communication works correctly before proceeding.

### Step 0.2: Prepare Datasets
Download and preprocess COCO-Stuff-27 (train/val splits with 27 semantic classes), Cityscapes (with fine annotations for 19 classes), and optionally ADE20K and PASCAL-Context; convert all datasets to TFRecord format for efficient TPU data loading, pre-compute and cache ZoeDepth predictions for all images (this takes ~2-3 days for COCO), and organize everything into a unified data pipeline that yields (image, depth_map, image_id) tuples with proper augmentations (random crop, horizontal flip, color jitter).

### Step 0.3: Implement Evaluation Infrastructure
Build a comprehensive evaluation module that computes Panoptic Quality (PQ, PQ_St, PQ_Th), Segmentation Quality (SQ), Recognition Quality (RQ), semantic mIoU, and instance AP metrics; implement the Hungarian matching algorithm for segment-to-ground-truth assignment with IoU > 0.5 threshold, create visualization utilities that overlay panoptic predictions on images with distinct colors per instance, and set up Weights & Biases or TensorBoard logging for tracking all metrics across experiments.

---

## Phase 1: Reproduce Baseline Components

### Step 1.1: Port DINO ViT-S/8 to JAX
Convert the pretrained DINO ViT-S/8 weights from PyTorch to JAX/Flax format using the `torch2jax` pattern (extract state dict, reshape weights for Flax conventions), implement the Vision Transformer architecture in Flax with patch embedding (8×8 patches), 12 transformer blocks (384-dim, 6 heads), and ensure your JAX implementation produces identical outputs to the PyTorch version on the same input (max absolute difference < 1e-5)—freeze all parameters and wrap in a `jax.lax.stop_gradient` call.

### Step 1.2: Implement ZoeDepth Inference Pipeline
Either port ZoeDepth to JAX (complex, ~1 week) or use a hybrid approach where depth maps are pre-computed using PyTorch and cached to disk; if porting, focus on the MiDaS-based encoder and metric depth decoder, verify depth predictions match the original implementation on sample images, and create a depth normalization function that maps raw depth to [0, 1] range with dataset-specific statistics (Cityscapes has different depth distributions than COCO indoor scenes).

### Step 1.3: Reproduce DepthG Semantic Segmentation
Implement the DepthG architecture: a 3-layer MLP projector (384→384→384→90) on top of frozen DINO features producing the 90-dimensional semantic code space, implement the STEGO correspondence loss using KNN (k=7) in DINO feature space to find positive pairs, implement the depth-guided feature correlation loss that weights feature similarity by depth proximity, train on COCO-Stuff-27 for 30 epochs and verify you achieve mIoU within ±1.5 of the reported 44.8%—if not, debug by comparing intermediate feature statistics.

### Step 1.4: Reproduce CutS3D Instance Segmentation
Implement Normalized Cut on DINO feature affinity matrices: compute pairwise cosine similarities W_ij = cos(f_i, f_j), construct the graph Laplacian L = D - W, solve for the second-smallest eigenvector using JAX's linalg.eigh, and threshold to obtain binary masks; implement LocalCut that projects pixels to 3D using depth and camera intrinsics, then separates instances via 3D spatial clustering (DBSCAN or spectral clustering on 3D coordinates); train a Cascade Mask R-CNN head on the pseudo-masks and verify instance AP on COCO val2017 is within ±0.5 of reported 10.7%.

### Step 1.5: Build Naive Panoptic Baseline
Create a simple panoptic assembly pipeline: run DepthG to get semantic labels, run CutS3D to get instance masks, assign each instance mask the majority semantic class from pixels within it, classify semantic clusters as "things" or "stuff" using a simple heuristic (e.g., classes with instances are things), merge by giving priority to high-confidence instance masks over stuff regions, and evaluate PQ on Cityscapes—this naive baseline should achieve PQ ~18-22 and serves as your lower bound.

---

## Phase 2: Feature Alignment & Projection Bridge

### Step 2.1: Implement Adaptive Projection Bridge
Create two learnable linear projections: W_s ∈ ℝ^(192×90) mapping semantic codes to bridge dimension D_b=192, and W_f ∈ ℝ^(192×384) mapping instance features to the same dimension; add LayerNorm after each projection for training stability, initialize with Xavier/Glorot initialization scaled by 0.1 for gentle initial impact, and implement the corresponding inverse projections W_s† and W_f† as separate learned matrices (not transposes) that map back to original dimensions.

### Step 2.2: Implement CKA Alignment Loss
Compute Centered Kernel Alignment between projected semantic and instance features: center both feature matrices by subtracting mean, compute Gram matrices K_s = S'S'^T and K_f = F'F'^T, calculate CKA = ||K_s^T K_f||_F² / (||K_s^T K_s||_F × ||K_f^T K_f||_F), and use L_CKA = 1 - CKA as an alignment loss that encourages the two branches to capture complementary but related information—this loss should be weighted at λ_cka = 0.1 initially.

### Step 2.3: Implement Feature Reconstruction Loss
Add reconstruction losses to ensure the projection-inverse-projection cycle preserves information: L_recon = ||s - W_s†(W_s(s))||² + ||f - W_f†(W_f(f))||², use stop_gradient on the target (original features) to prevent trivial solutions, weight this loss at λ_recon = 0.5, and monitor the reconstruction error during training—if it plateaus above 0.1, increase the bridge dimension D_b or add skip connections.

### Step 2.4: Train Projection Bridge in Isolation
Before adding Mamba, train just the projection bridge for 10 epochs: freeze DINO, DepthG head, and CutS3D head, optimize only W_s, W_f, W_s†, W_f† with Adam (lr=1e-4), minimize L_recon + λ_cka × L_CKA, and verify that (a) reconstruction error < 0.05, (b) CKA > 0.3, and (c) downstream semantic mIoU and instance AP don't degrade when using reconstructed features—this ensures the bridge doesn't destroy information before Mamba processing.

---

## Phase 3: Unified Depth Conditioning

### Step 3.1: Implement Depth Positional Encoding
Create a depth encoder that converts scalar depth values to rich feature vectors: apply sinusoidal positional encoding with frequencies [1, 2, 4, 8, 16, 32] to normalized depth values, concatenate sin and cos components to get 12-dimensional depth encoding per pixel, pass through a small MLP (12→64→D_b) to match bridge dimension, and reshape to spatial feature maps—this gives the model explicit depth awareness beyond what's implicit in DINO features.

### Step 3.2: Implement Gated Depth Conditioning
Apply depth conditioning to both branches via learned gating: compute gate_s = σ(W_gate_s × depth_features) and gate_f = σ(W_gate_f × depth_features), apply as S_d = S' ⊙ gate_s + S' (residual gating) and F_d = F' ⊙ gate_f + F', where ⊙ is element-wise multiplication—this allows the model to selectively modulate features based on depth, emphasizing nearby objects differently from distant background.

### Step 3.3: Implement Depth Consistency Loss
Add a loss encouraging depth-similar pixels to have similar features: for each pixel pair (i,j) with depth difference |D_i - D_j| < τ_depth, add a term pulling their features together, and for pairs with large depth differences, add a margin-based repulsion term; formally L_depth_consist = Σ_{i,j} w_ij × ||S_d[i] - S_d[j]||² where w_ij = exp(-|D_i - D_j|²/2σ²), and use efficient sampling (random 1024 pairs per image) rather than all O(N²) pairs.

---

## Phase 4: Mamba2 Bridge Implementation

### Step 4.1: Implement Mamba2 SSD Core in JAX
Port the Mamba2 State Space Duality formulation to JAX: implement the selective scan as chunked matrix multiplications with chunk size P=128 (TPU-aligned), create the input-dependent Δ, B, C projections as Linear layers, implement the discretization Ā = exp(Δ ⊙ A) where A is a learnable diagonal matrix, build the semiseparable matrix M_k for each chunk, and compute y = C × M × (B ⊙ x) using jnp.einsum with optimized contraction paths—test on random inputs to verify output shapes and gradient flow.

### Step 4.2: Implement Bidirectional Cross-Modal Scan
Create the BiCMS module: interleave depth-conditioned features as Z = [S_d[1], F_d[1], S_d[2], F_d[2], ...] creating a sequence of length 2N, run forward Mamba2 scan to get Y_fwd, reverse Z and run another Mamba2 scan to get Y_bwd_rev then reverse it back, concatenate [Y_fwd; reverse(Y_bwd)] along feature dimension, project back to D_b with a linear layer, add residual connection Z, and deinterleave to recover S_fused (even indices) and F_fused (odd indices)—this ensures each token sees context from both directions and both modalities.

### Step 4.3: Implement Mamba State Regularization
Add regularization to prevent SSM hidden state explosion: track the final hidden state h_T after processing the full sequence, add L_state = λ_state × ||h_T||² with λ_state = 0.01, also add spectral normalization to the A matrix to ensure stable dynamics (spectral radius < 1), and monitor hidden state norms during training—if they grow unboundedly, reduce learning rate or increase regularization.

### Step 4.4: Stack Multiple Mamba Layers
Create a 4-layer Mamba2 bridge with residual connections: each layer applies BiCMS then adds a feedforward block (Linear→GELU→Linear with expansion ratio 2), add LayerNorm before each sublayer (pre-norm architecture), use dropout (p=0.1) between layers during training, and implement gradient checkpointing for memory efficiency—the full 4-layer bridge should add ~1.5M parameters.

---

## Phase 5: Stuff-Things Classifier

### Step 5.1: Implement Depth Boundary Density (DBD)
For each semantic cluster c, compute the proportion of pixels with high depth gradients: apply Sobel filters to get depth gradient magnitude G_d = sqrt((∂D/∂x)² + (∂D/∂y)²), threshold at τ_d (tune on validation set, typically 0.1-0.3), count boundary pixels within cluster region R_c, compute DBD(c) = count(G_d[R_c] > τ_d) / |R_c|, and normalize across all clusters to [0,1] range—high DBD indicates "thing" classes with sharp depth boundaries.

### Step 5.2: Implement Feature Cluster Compactness (FCC)
Measure how tightly clustered each semantic class is in DINO feature space: for cluster c, extract all DINO features {f_i : semantic(i) = c}, compute cluster covariance Σ_c = Cov(features), compute total covariance Σ_total over all features, calculate FCC(c) = 1 - trace(Σ_c)/trace(Σ_total), normalize to [0,1]—high FCC indicates "thing" classes that form tight, distinctive clusters compared to diffuse "stuff" classes.

### Step 5.3: Implement Instance Decomposition Frequency (IDF)
Count how often each semantic region gets split into multiple instances: for each semantic cluster c, find all instance masks that overlap >50% with c's pixels, compute IDF(c) = num_overlapping_instances / normalized_area(c), where normalized_area divides by image area to make comparable across images, normalize to [0,1]—high IDF indicates "thing" classes that CutS3D naturally separates into multiple instances.

### Step 5.4: Train the Stuff-Things MLP Classifier
Build a small MLP: Linear(3, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, 1) → Sigmoid, create a training set by computing [DBD, FCC, IDF] for each semantic cluster across 1000 images and using ground-truth stuff/thing labels from dataset metadata, train with binary cross-entropy for 100 epochs with lr=1e-3, evaluate accuracy on held-out 200 images—target >80% accuracy, and if lower, add more features or increase MLP capacity.

---

## Phase 6: Loss Function Implementation

### Step 6.1: Implement Full Semantic Loss
Combine STEGO and DepthG losses: L_STEGO uses InfoNCE over KNN-derived positive pairs with temperature τ=0.1, L_DepthG weights feature correlation by depth similarity with σ_d=0.5, implement efficient batched computation using jax.vmap, combine as L_semantic = L_STEGO + λ_d × L_DepthG with λ_d=0.3, and verify gradient magnitudes are balanced (ratio within 0.1-10x).

### Step 6.2: Implement Full Instance Loss
Combine CutS3D losses: L_BCE uses spatial confidence-weighted binary cross-entropy where confidence comes from mask prediction consistency across augmentations, L_Drop penalizes unmatched instance predictions to handle missing pseudo-labels, L_box adds smooth-L1 regression for bounding boxes, combine as L_instance = L_BCE + λ_drop × L_Drop + λ_box × L_box with λ_drop=0.5, λ_box=1.0.

### Step 6.3: Implement Cross-Branch Consistency Loss
Create the novel consistency losses: L_uniform = (1/K) × Σ_k H(semantic | instance_k) measures entropy of semantic labels within each instance mask (should be low—each instance should have one dominant class), L_boundary = Σ_{adj(i,j)} |B_sem(i,j) - B_inst(i,j)| penalizes misaligned semantic and instance boundaries, L_DBC = Σ_{adj} (B_depth - B_sem)² + (B_depth - B_inst)² enforces both boundaries align with depth edges, combine as L_consistency = λ_u × L_uniform + λ_b × L_boundary + λ_dbc × L_DBC.

### Step 6.4: Implement Differentiable PQ Proxy Loss
Create a soft, differentiable approximation to Panoptic Quality: use soft IoU = Σ min(m_pred, m_gt) / Σ max(m_pred, m_gt) where masks are soft (sigmoid outputs), compute soft matching via Hungarian algorithm with IoU matrix, compute soft TP = Σ σ((IoU - 0.5)/τ_pq) where σ is sigmoid with temperature τ_pq=0.1, approximate PQ_proxy = 2×TP / (TP + |pred| + |gt|), and use L_PQ = 1 - PQ_proxy as loss—this gives direct gradient signal toward better panoptic quality.

### Step 6.5: Implement Gradient-Balanced Loss Aggregation
Combine all losses with adaptive weighting: compute gradient magnitudes ||∇L_sem||, ||∇L_inst||, ||∇L_bridge||, ||∇L_consist||, ||∇L_PQ|| using jax.grad, normalize weights inversely proportional to gradient magnitudes to prevent any single loss from dominating, apply curriculum schedule (Phase A: only L_sem; Phase B: add L_inst; Phase C: add all), and log individual loss values and weights to monitor training dynamics.

---

## Phase 7: Training Pipeline

### Step 7.1: Implement Gradient Projection for Phase B
During Phase B when instance loss is introduced, detect and resolve gradient conflicts: compute g_sem = ∇_θ L_semantic and g_inst = ∇_θ L_instance, check if ⟨g_sem, g_inst⟩ < 0 (conflicting), if so project g_inst to remove the conflicting component: g_inst_proj = g_inst - (min(0, ⟨g_inst, g_sem⟩) / ||g_sem||²) × g_sem, apply combined gradient g_sem + β_t × g_inst_proj where β_t ramps from 0 to 1 over Phase B epochs—this ensures instance gradients never fight against semantic gradients.

### Step 7.2: Implement EMA Teacher for Self-Training
Create an Exponential Moving Average copy of the model: initialize θ_ema = θ at start of Phase C, after each training step update θ_ema = μ × θ_ema + (1-μ) × θ with momentum μ=0.999, use θ_ema (not θ) to generate pseudo-labels for consistency losses and self-training, implement using Flax's TrainState with a separate ema_params field, and ensure EMA weights are saved in checkpoints for resumption.

### Step 7.3: Implement Confidence-Based Pseudo-Label Filtering
During self-training, filter low-confidence predictions: compute semantic confidence c_sem = max(softmax(logits)) per pixel, compute instance confidence c_inst from mask prediction scores, combine as c_joint = c_sem^α × c_inst^(1-α) with α=0.5, create valid_mask = c_joint > τ_conf where τ_conf starts at 0.7 and increases by 0.05 each round, apply valid_mask to all losses so low-confidence regions don't contribute gradients.

### Step 7.4: Implement Full Training Loop
Create the complete training pipeline: Phase A (epochs 1-20) trains semantic branch only, Phase B (epochs 21-40) adds instance branch with gradient projection and curriculum weight ramp-up, Phase C (epochs 41-60) adds Mamba bridge and all consistency losses with full joint training, Phase D (rounds 1-3, ~5 epochs each) runs self-training with increasing confidence thresholds; use cosine learning rate decay from 1e-4 to 1e-6, batch size 8 per TPU core (64 total on v4-8), gradient clipping at norm 1.0, and checkpoint every 5 epochs.

---

## Phase 8: Panoptic Merging & Post-Processing

### Step 8.1: Implement Instance-Semantic Assignment
For each predicted instance mask, assign a semantic class: compute histogram of semantic predictions within mask, take argmax as the instance's class, handle edge cases where mask spans multiple classes by using confidence-weighted voting, and filter instances whose dominant class confidence is below 0.6—these ambiguous instances often span semantic boundaries and hurt PQ.

### Step 8.2: Implement Overlap Resolution
Resolve conflicts when instance masks overlap: sort instances by confidence score (descending), iterate through sorted list, for each instance mark its pixels as "used", subsequent instances only keep pixels not already used, remove instances that lose >70% of their pixels due to overlap, and ensure no pixel belongs to multiple instances—this greedy approach matches the standard panoptic format requirement.

### Step 8.3: Implement Stuff Region Filling
Fill remaining pixels with stuff predictions: after assigning all thing instances, identify pixels not covered by any instance, for these pixels use the semantic segmentation prediction directly, assign instance_id = 0 for stuff regions (standard convention), merge contiguous stuff regions of the same class into single segments, and ensure the final output covers 100% of pixels with no gaps.

### Step 8.4: Implement CRF Post-Processing
Apply dense CRF for boundary refinement: use the permutohedral lattice implementation for efficiency, set bilateral kernel parameters (θ_α=80, θ_β=13 for appearance, θ_γ=3 for smoothness), run 10 mean-field iterations, apply to both semantic logits before clustering AND to final panoptic segments, and tune CRF parameters on validation set—good CRF tuning typically improves PQ by 1-2 points.

---

## Phase 9: Experiments & Ablations

### Step 9.1: Run Component Ablation Study
Systematically measure each component's contribution: train full model, then variants removing (a) Mamba bridge (use simple concatenation instead), (b) depth conditioning, (c) BiCMS (use forward-only scan), (d) consistency losses, (e) stuff-things classifier (use oracle labels), (f) self-training rounds; report PQ, PQ_St, PQ_Th, mIoU, AP for each variant, and compute relative improvement to identify which components matter most.

### Step 9.2: Run Hyperparameter Sensitivity Analysis
Sweep key hyperparameters: bridge dimension D_b ∈ {128, 192, 256, 384}, number of Mamba layers ∈ {2, 4, 6, 8}, chunk size P ∈ {64, 128, 256}, loss weights (vary each by ±50%), learning rate ∈ {5e-5, 1e-4, 2e-4, 5e-4}, and EMA momentum μ ∈ {0.99, 0.999, 0.9999}; use a subset of training data (10%) for fast sweeps, identify stable operating regions, and report sensitivity of final PQ to each hyperparameter.

### Step 9.3: Run Cross-Dataset Generalization
Test domain generalization: train on Cityscapes, evaluate zero-shot on KITTI and BDD100K; train on COCO-Stuff-27, evaluate on ADE20K and PASCAL-Context; report PQ drops compared to in-domain evaluation, analyze which components degrade most (often stuff-things classifier due to different class distributions), and consider domain-adaptive fine-tuning strategies if drops exceed 10 PQ points.

### Step 9.4: Run Computational Efficiency Analysis
Profile training and inference: measure GPU/TPU memory usage, training throughput (images/second), inference latency (ms/image), parameter count for each component, FLOPs for forward pass, and compare against CUPS baseline; create efficiency-accuracy tradeoff plots (PQ vs FLOPs, PQ vs latency), and highlight the Mamba bridge's efficiency advantage over attention-based fusion alternatives.

---

## Phase 10: Paper Writing & Submission

### Step 10.1: Prepare Main Results Tables
Create comprehensive results tables: Table 1 shows PQ, PQ_St, PQ_Th, SQ, RQ on Cityscapes comparing against CUPS, U2Seg, and supervised baselines; Table 2 shows same metrics on COCO-Stuff-27 where CUPS cannot run (your method enables this); Table 3 shows semantic mIoU and instance AP for component-level comparison against DepthG and CutS3D individually; use bold for best unsupervised results, underline for second-best.

### Step 10.2: Create Visualizations
Generate compelling figures: Figure 1 shows architecture diagram (the SVG I created), Figure 2 shows qualitative results with input image, depth, semantic, instance, and panoptic outputs side-by-side, Figure 3 shows failure cases with analysis (e.g., thin structures, unusual viewpoints), Figure 4 shows stuff-things classification accuracy vs the three cues, Figure 5 shows loss curves and training dynamics; ensure all figures are high-resolution (300 DPI) and colorblind-friendly.

### Step 10.3: Write Technical Sections
Draft the paper following NeurIPS format: Introduction (2 pages) motivates unsupervised panoptic + Mamba fusion, Related Work (1.5 pages) covers unsupervised segmentation + state space models + panoptic segmentation, Method (3 pages) details architecture with equations (use the mathematical derivations I provided), Experiments (2.5 pages) presents results and ablations, Conclusion (0.5 pages) summarizes contributions and limitations; target 9 pages main + unlimited appendix.

### Step 10.4: Prepare Supplementary Material
Create comprehensive appendix: Appendix A provides full implementation details (all hyperparameters, training schedules, hardware specs), Appendix B shows extended ablations, Appendix C provides per-class breakdown of results, Appendix D shows more qualitative examples (10+ images per dataset), Appendix E provides failure case analysis, Appendix F discusses societal impact and limitations; also prepare code repository with clean, documented implementation for camera-ready submission.

---

## Summary Timeline

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Phase 0 | Week 1-2 | TPU environment ready, datasets preprocessed |
| Phase 1 | Week 3-6 | DepthG and CutS3D reproduced within ±1.5% of paper |
| Phase 2 | Week 7-8 | Projection bridge trained, CKA > 0.3 |
| Phase 3 | Week 9-10 | Depth conditioning integrated, L_depth_consist stable |
| Phase 4 | Week 11-14 | Mamba2 bridge working, BiCMS validated |
| Phase 5 | Week 15-16 | Stuff-things classifier > 80% accuracy |
| Phase 6 | Week 17-18 | All losses implemented and balanced |
| Phase 7 | Week 19-22 | Full training pipeline, first complete results |
| Phase 8 | Week 23-24 | Panoptic merging optimized, CRF tuned |
| Phase 9 | Week 25-28 | Ablations complete, final numbers locked |
| Phase 10 | Week 29-32 | Paper written, submitted to NeurIPS 2026 |

---

## Critical Checkpoints

✅ **Checkpoint 1 (Week 6):** Naive panoptic baseline achieves PQ ≥ 18 on Cityscapes  
✅ **Checkpoint 2 (Week 14):** Mamba bridge improves over naive baseline by ≥ 2 PQ points  
✅ **Checkpoint 3 (Week 18):** Full model (without self-training) achieves PQ ≥ 25 on Cityscapes  
✅ **Checkpoint 4 (Week 24):** Full model with self-training achieves PQ ≥ 28 on Cityscapes (surpasses CUPS)  
✅ **Checkpoint 5 (Week 28):** Ablations confirm each component contributes, no major bugs found  
✅ **Checkpoint 6 (Week 32):** Paper submitted with clean code repository  

---

## Risk Mitigation

**Risk 1: Mamba2 JAX port fails** → Fallback to PyTorch implementation with GPU training instead of TPU; slower but still feasible.

**Risk 2: Stuff-things classifier accuracy too low** → Fallback to oracle stuff-things labels for main results, report classifier-based results as ablation; still novel contribution.

**Risk 3: Cannot surpass CUPS on Cityscapes** → Pivot focus to COCO-Stuff-27 where CUPS cannot run; being first to report unsupervised panoptic on COCO is itself a contribution.

**Risk 4: Training instability with joint losses** → Increase curriculum warmup duration, reduce loss weights, add gradient clipping; stability is solvable with patience.

**Risk 5: Running out of time** → Prioritize core experiments over exhaustive ablations; a solid paper with fewer ablations beats no paper.
