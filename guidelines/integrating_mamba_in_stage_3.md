Research Synthesis: State-of-the-Art Ideas for Mamba2 in Self-Training

After surveying the latest papers, textbooks, and classical literature, here are the key insights and 4 novel research directions that go far beyond a simple refinement filter:

Key Papers Found

┌──────────────────────────────┬──────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
│            Paper             │    Venue     │                                           Key Insight                                            │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TTT-E2E (Sun et al., 2025)   │ -            │ Mamba2/GatedDeltaNet are special cases of Test-Time Training with specific update rules          │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MFuser (CVPR 2025 Highlight) │ CVPR 2025    │ "Mamba as a Bridge" for domain-generalized segmentation — uses SSM to fuse multi-domain features │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PanopMamba (Jan 2026)        │ -            │ First Mamba-based panoptic segmentation model (medical imaging)                                  │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ EMPL (MedIA 2024)            │ MICCAI/MedIA │ Formal proof that pseudo-labeling = EM algorithm (E-step: generate labels, M-step: retrain)      │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Spatial-Mamba (ICLR 2025)    │ ICLR 2025    │ Structure-aware spatial scanning for Mamba in dense prediction                                   │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ VSSD (ICCV 2025)             │ ICCV 2025    │ Vision SSM with non-causal dual scan for segmentation                                            │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TV3S (CVPR 2025)             │ CVPR 2025    │ Test-time adaptation with visual SSMs                                                            │
├──────────────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ CRF-RNN (Zheng et al., 2015) │ ICCV 2015    │ Unrolled mean-field CRF as recurrent neural network — classical predecessor                      │
└──────────────────────────────┴──────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

4 Novel Directions (Ranked by Impact)

1. Mamba2-EM: Variational EM Self-Training (Most Novel)

EMPL (2024) formally proved that iterative pseudo-labeling is an EM algorithm:
- E-step: Generate pseudo-labels from current model → posterior over latent labels
- M-step: Retrain model on pseudo-labels → maximize expected log-likelihood

Our innovation: Replace the "hard E-step" (argmax pseudo-label) with a learned E-step using Mamba2. The Mamba2 hidden state maintains a running belief over the latent segmentation field,
performing continuous soft inference instead of discrete argmax. This is mathematically grounded — the SSM state transition becomes the message-passing operator in variational inference.

Standard self-training:  model → argmax → pseudo-label → retrain
Mamba2-EM:              model → Mamba2(logits, features, depth) → soft posterior → retrain

The Mamba2 state acts as a learned variational distribution q(z|x) that approximates the true posterior p(z|x, theta). The sequential scan naturally handles spatial dependencies that argmax
ignores.

2. CRF-as-SSM: Unrolled CRF via Mamba2 State Transitions (Most Principled)

CRF-RNN (ICCV 2015) showed you can unroll mean-field CRF iterations as an RNN. We can take this further: Mamba2's state transition IS a learned CRF pairwise potential.

In mean-field CRF:
Q_i(l) ∝ exp(-ψ_u(l)) × Π_j≠i Σ_l' k(f_i, f_j) μ(l, l') Q_j(l')

In Mamba2 SSD:
h_t = A_t h_{t-1} + B_t x_t       (state transition = pairwise message)
y_t = C_t h_t                       (output = marginal belief)

The A_t matrix IS the learned pairwise potential — it determines how neighboring pixel beliefs propagate. Unlike hand-crafted Gaussian CRF kernels, Mamba2 learns these from data. The 2D
scanning pattern (our cross-scan from Mamba2 module) handles the non-causal nature of 2D CRFs.

3. TTT-Mamba2: Test-Time Training for Pseudo-Label Adaptation (Most Practical)

TTT-E2E (2025) showed that Mamba2 and GatedDeltaNet are special cases of TTT where the hidden state IS a model being trained at test time. This means:

- Standard Mamba2: state h_t encodes compressed history → fixed function
- TTT interpretation: state h_t IS a linear model being SGD-updated at each token → adaptive

For self-training: At each self-training round, the Mamba2 bridge doesn't just refine labels — it adapts its own parameters based on the incoming pseudo-label stream. High-confidence
pseudo-labels update the state more (larger learning rate); low-confidence ones update less. This is TTT applied to pseudo-label refinement.

This is the most practical because we already have the Mamba2 module built. We just need to:
1. Initialize TTT-Mamba2 with CUPS model features
2. Run spatial scan over each image's logits + features
3. Output refined soft labels
4. Use confidence-weighted self-supervised loss

4. Structure-Aware Spatial Scan (Most Complementary)

Spatial-Mamba (ICLR 2025) showed that scan order matters enormously for dense prediction. Instead of fixed raster/hilbert scan, use depth-guided adaptive scanning:

- Scan along depth contours (same-depth regions first → coherent object surfaces)
- Use GatedDeltaNet's gate to reset state at depth discontinuities (object boundaries)
- This naturally groups stuff regions (continuous depth) and separates thing instances (depth gaps)

---
My Recommendation: Start with Direction 3 (TTT-Mamba2)

Why:
1. We already built the Mamba2 module (101/101 tests passing)
2. TTT interpretation requires minimal new code — just change how we interpret the hidden state update
3. It's the most practical to implement within the CUPS pipeline
4. Direction 2 (CRF-as-SSM) is the backup if TTT doesn't work — it's more principled but harder to implement
5. Direction 1 (Mamba2-EM) is the most novel for a paper but requires the most engineering

Concrete plan for TTT-Mamba2 in CUPS Stage-3:
1. After Stage-2 training finishes (current run, ~7 more hours)
2. At each self-training round, run TTT-Mamba2 refinement on model predictions
3. The Mamba2 bridge takes (DINOv2 features, model logits, depth map) as input
4. Outputs refined soft pseudo-labels with spatial coherence
5. Use confidence-weighted retraining (high-confidence → standard CE, low-confidence → down-weighted)
6. ~2M extra parameters, fits in 11GB GPU memory alongside CUPS

This would be a genuinely novel contribution — no one has combined TTT with SSM-based pseudo-label refinement for panoptic segmentation.