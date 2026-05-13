# NeurIPS 2026 Review — Reviewer 2 (Methods / Theory)

**Paper**: Depth-Conditioned Pseudo-Labels for Scene-Centric Unsupervised Panoptic Segmentation
**Reviewer focus**: Formal methodology, mathematical rigor, principled design choices.

---

# Summary

The paper presents a monocular-only unsupervised panoptic pipeline. The core technical claims are two compact mechanisms placed inside the standard CUPS (Hahn et al., 2025) two-stage recipe: (i) **DCFA**, a ~225K-parameter residual MLP that conditions a frozen unsupervised semantic code (CAUSE-TR's 90-D clusterbook on a frozen DINOv2 ViT-B/14 backbone) on monocular depth via a sinusoidal embedding and an L2 preserve term with weight $\lambda_p=20$; and (ii) **SIMCF**, a learning-free, three-step rule-based filter (dominant-class reassignment, feature-similarity adjacent merge, depth-implausibility void). The paper claims a contribution-isolated +3.07 PQ over a matched monocular baseline (32.76 -> 35.83 PQ on Cityscapes val), with the headline 35.83 PQ exceeding published CUPS (27.80) under the 27-class CAUSE Hungarian protocol.

# Strengths

1. **Honest controlled comparison.** The authors define a matched monocular baseline that shares the Stage-2/3 trainer and the DINOv3 ViT-B/16 backbone, so the +3.07 PQ delta is genuinely attributable to DCFA+SIMCF rather than to backbone substitution. This addresses the most obvious confound a methods reviewer would raise.

2. **Conditioning-mechanism decomposition is well-targeted (Table 4).** Comparing DepthG (loss-side, end-to-end), loss-side P(z) without preserve, loss-side P(z) with preserve, and DCFA at matched preservation strength is a clean factorial design that isolates *architectural* depth conditioning (input vs. loss) at fixed regularization. This is the kind of ablation a theoretically-minded reviewer asks for.

3. **The preservation term is mathematically motivated and verifiably reproduces a known regularization pattern.** Zero-init output projection plus L2 to the frozen code at high $\lambda_p$ is a defensible inductive bias: it makes the adapter a perturbation around identity, much like LoRA's zero-init in NLP, and the ablation shows that without it both DepthG and loss-side P(z) collapse below the raw baseline (-11.21, -11.92 PQ). That collapse is genuine empirical evidence that unconstrained re-shaping of the frozen manifold breaks k-means structure.

4. **Loss definitions are formally complete in the appendix (sec. F).** $\mathcal{L}_\text{corr}$ and $\mathcal{L}_\text{preserve}$ are defined with explicit sampling sets, kernel bandwidth ($\sigma_d=0.5$), and aggregation order. This is more rigorous than most CVPR-style applied papers.

5. **The over-cluster-to-concept map $\phi$ and where ground truth enters are explicitly disentangled in Appendix B.6.** This pre-empts the standard "you tuned on test" criticism for $\phi$; only the 27-to-27 Hungarian at metric time uses GT.

6. **SIMCF sensitivity sweep (Table 8)** quantifies the threshold-tuning concern: the 0.71 PQ band across $\pm$12% perturbation is genuinely smaller than the +1.31 PQ Stage-1 contribution, supporting the robustness claim.

# Weaknesses

## W1. DCFA is engineering-justified but theoretically thin.

The DCFA design space is enormous (depth encoding scheme, conditioning site, fusion operator, hidden width, depth, regularization), yet the paper presents one fixed instantiation with no theoretical argument for the specific architectural choices.

**Concretely missing justifications**:

- **Why MLP, not cross-attention?** Pixel-wise depth is a *local* scalar conditioning signal, but cross-attention over depth tokens (or even over spatial neighbours of depth) would be a more expressive alternative with comparable parameter budget. The paper does not test this.
- **Why concatenation, not FiLM?** FiLM (Perez et al., 2018) is the canonical conditioning operator: $\tilde{z} = \gamma(d) \odot z + \beta(d)$. It has a clean interpretation (per-channel affine modulation by depth) and is parameter-efficient. Concatenation followed by an MLP entangles depth and feature dimensions in an unstructured way and burns parameters on cross-terms that may be unnecessary.
- **Why a 16-D sinusoidal encoding with $\omega_k = 2^k\pi$?** This is identical to NeRF/Transformer positional encoding for $d \in [0,1]$. The choice is sensible but completely unjustified for *depth*. Why does an exponentially-spaced frequency basis match the structure of monocular depth distributions? Depth statistics in driving scenes are heavily skewed (most pixels at mid-range, sparse far-field), and a log-depth or learned encoding may be more natural. No comparison given.
- **Why hidden width $h=384$?** The paper states it without an ablation. A scaling sweep over $h$ would be the minimum theoretical due-diligence here.

**The bottom line**: DCFA looks like a well-tuned engineering artefact, not a principled theoretical construction. The +0.68 PQ over raw at the Stage-1 level is real, but the design space is hand-picked without comparative ablation.

## W2. SIMCF is three engineering rules stitched together with no unifying theoretical principle.

The paper explicitly calls SIMCF "Semantic Instance Mutual Consistency Filtering", but the three steps have orthogonal motivations and the paper does not provide a unifying objective:

- **Step (A)** dominant-class reassignment: a heuristic for cleaning class-noisy instance regions. No formal objective.
- **Step (B)** adjacent-pair merge by DINOv3 similarity above $\tau_\text{sim}=0.85$: a heuristic for un-fragmenting depth oversegmentation. The choice of DINOv3 features here (rather than the DINOv2 features used everywhere else in Stage-1) is justified only by "the slightly stronger off-the-shelf descriptor", which is not a principled argument and introduces a backbone inconsistency inside Stage-1 itself.
- **Step (C)** depth-implausibility void at $\eta\sigma_c$: per-class depth statistics outlier rejection. This is the closest to a principled rule (a $z$-score test on per-class depth), but $\eta=2.5$ is set without justification and the implicit Gaussianity assumption on per-class depth distributions is not stated, let alone tested.

**A theoretically cleaner formulation** would express the joint filter as the MAP estimate under a generative model with explicit semantic, geometric, and depth-likelihood factors. The paper instead presents three pipelined rules whose ordering matters but is not justified ("Step B then Step C are applied in this order" — why not the reverse? what changes if Step A is iterated until fixed point?). This is the textbook "engineering trick stitched as a method" pattern.

## W3. Hungarian assignment protocol: K=80 -> 27 mapping under-discussed.

The over-clustering choice $k=80$ is justified by "CUPS reports that this regime monotonically improves panoptic quality as k exceeds the evaluation cardinality" with a citation to CUPS. The paper does not:

- Provide its own k-sweep (k=27, 40, 54, 80, 120) at the Stage-1 level. CUPS Table 7b reports $\{27, 40, 54\}$; the paper takes this as transferring to the monocular setting without verifying. Section 3.2 ("collapsing to 27 classes at this stage would erase fine visual modes that the depth-based instance branch later relies on") is a plausibility argument, not an ablation.
- Discuss the Hungarian-assignment trade-off when source partition cardinality (80) differs from target (27). Is this rectangular Hungarian (unbalanced bipartite matching), and how is the many-to-one constraint handled? If multiple K-clusters map to the same Cityscapes class, do they compete? The failure-mode table (Table 6) says "1-to-1 matching can lock onto a spurious benchmark class" — this is a real problem and the proposed mitigation ("many-to-1 assignment") is not implemented in the reported numbers.
- Report sensitivity of the final PQ to the Hungarian assignment computed on the val set. Standard practice in unsupervised semantic segmentation (PASS, STEGO, HP, CAUSE) discusses this in detail; this paper relegates it to a single sentence in 3.5.

## W4. Theoretical novelty over LoRA/DoRA/IA3 is overstated.

The DCFA construction $\tilde z = z + r([z; e(d)])$ with zero-init output projection and L2 preserve is structurally a **conditional residual adapter with identity initialization and explicit anchoring**. This is the exact pattern of:

- **LoRA** (Hu et al., 2021): low-rank perturbation around frozen weights, identity at init via zero-init of $B$.
- **DoRA** (Liu et al., 2024): magnitude-direction decomposition of LoRA with explicit anchoring.
- **IA3** (Liu et al., 2022): scalar-vector rescaling with identity init.
- **Adapter** (Houlsby et al., 2019): bottleneck MLP with residual connection and zero-init.

What is genuinely new is the **conditioning signal** (monocular depth) and the **conditioning site** (a frozen *unsupervised* clusterbook code, not a transformer block). The paper claims novelty as the "first parameter-efficient adapter that conditions a frozen unsupervised semantic code on monocular depth via residual feature-level injection". This is technically defensible but narrow — the underlying architectural primitive is well-known. The paper does not engage with the LoRA/DoRA/Adapter literature at all (zero citations to any of them in the bibliography). For a methods-focused reviewer this is a serious gap: the formal contribution is essentially "LoRA-style adapter applied at the clusterbook output, conditioned on sinusoidally-encoded depth, with L2 preserve". Stating this honestly would not weaken the contribution — the empirical validation is what matters — but pretending the architectural construction is novel weakens the methodological credibility.

## W5. Mathematical inconsistencies and notation issues.

- **Eq. 2** ($e(d_u)$ definition): the encoding is described as 16-dimensional with eight frequencies, but each frequency contributes both sin and cos, giving $8 \times 2 = 16$ entries. This is correct as stated but the "$\omega_k = 2^k \pi$ for $k=0,\dots,7$" range puts the highest frequency at $128\pi$ on $d \in [0,1]$ — which causes ~64 cycles across the depth range. For depth differences in the natural range (say, 1m at near-field), this is **aliased** at any reasonable image resolution. No analysis of whether this frequency range is matched to the depth statistics.

- **Eq. 4 ($\mathcal{L}_\text{DCFA}$)**: $\lambda_p = 20$ is set but $\mathcal{L}_\text{corr}$ and $\mathcal{L}_\text{preserve}$ have different units. $\mathcal{L}_\text{corr}$ is a similarity-weighted cosine distance term in $[0, 2]$; $\mathcal{L}_\text{preserve}$ is a squared-L2 distance on 90-D codes whose magnitude depends on the normalization scheme of $z$. The paper says codes are L2-normalized for k-means, but does not state whether the same normalization applies inside the loss. With normalized codes, $\|\tilde z - z\|_2^2 \leq 4$, but without normalization the magnitude is unbounded. Stating the normalization explicitly (and showing the empirical magnitudes of the two terms at convergence) is necessary for $\lambda_p = 20$ to be meaningful.

- **Eq. 5** (depth boundary): The Sobel gradient magnitude $g_D$ is computed on raw depth values $D \in \mathbb{R}^{H \times W}$, but the threshold $\tau_d = 0.20$ has no stated unit. Is this normalized by max-depth, by 99th percentile, or absolute? The choice critically affects scale invariance and cross-dataset transfer.

- **Sec. 3.4 SIMCF**: Step (A) says "the dominant pseudo-class is identified and within-region pixels of conflicting classes are reassigned", but the criterion for "dominant" (majority? plurality? mode-of-modes?) is not specified. Step (C) uses "per-class depth mean" $\mu_c$ and "per-class depth std" $\sigma_c$, but where are these computed — per-image, per-dataset, per-batch? The paper does not say.

- **Notation drift**: $\hat S_x$ is introduced in Eq. 1 as the "semantic pseudo-label", but in §3.2 it is "the over-cluster semantic map" (i.e., $K=80$ assignments), and in §3.4 "$S_x^0 = \hat S_x$" appears, after which $\hat S_x$ becomes the *filtered* output. So $\hat S_x$ has at least two distinct meanings in the same paper.

## W6. The "monocular baseline 32.76 PQ" number is provenance-thin.

Table 1 reports the controlled monocular baseline at 32.76 PQ vs. 27.80 for published CUPS (a +5.0 PQ gap). The paper attributes this gap to "the pseudo-label-source substitution alone" (abstract) and uses it as the matched comparison anchor. But this baseline is also trained on the same DINOv3 ViT-B/16 backbone as MBPS — the limitations section acknowledges this confound and calls a same-backbone CUPS rerun "future work". The author's clarification (advisor said this is OK) is noted, but a methods-focused reviewer would still want at least an estimate of how much of the +5.0 PQ comes from backbone vs. pseudo-label substitution. The DINOv3 ViT-B/16 backbone is meaningfully stronger than CUPS' DINO ResNet-50, and the published evidence in unsupervised semantic segmentation (CAUSE, EAGLE) shows that backbone substitution alone can shift PQ by several points. Without that decomposition, the +3.07 PQ contribution-isolated number is the only number this paper can defend, and the headline 35.83 vs. 27.80 framing is borderline misleading.

## W7. Single-seed evaluation.

The 35.83 PQ is from a single Stage-3 run (acknowledged in §6, point v). Variance across seeds in Stage-3 self-training has historically been ~$\pm 0.5$-1.0 PQ in CUPS-family methods. The +3.07 PQ delta over the matched baseline is large enough to survive seed noise, but the headline 35.83 vs. 27.80 (+8.03) is at the boundary of what would survive a 3-seed average. Without seed reporting, the methods reviewer cannot calibrate uncertainty.

# Soundness (1-4): **3**

The mathematics is largely correct but with notational drift (W5) and missing normalization details. The empirical design is tight on the controlled comparison axis (matched baseline, DCFA conditioning ablation), but the SIMCF design is theoretically thin (W2) and the Hungarian protocol is under-discussed (W3). The +3.07 PQ contribution-isolated improvement is well-supported; the headline +8.03 PQ vs. CUPS is partly a backbone-substitution artifact (W6). Score: 3 (acceptable but not outstanding for a methods-track paper).

# Presentation (1-4): **3**

The paper is well-organized with clear pipeline diagrams and a thoughtful limitations section. The notation drift on $\hat S_x$ (W5) and the missing units on $\tau_d$ are presentation issues. The conditioning-mechanism table (Table 4) is exceptionally clear. The appendix-relegated SIMCF sweep should be in the main body given its importance to the threshold-tuning concern. Score: 3.

# Contribution (1-4): **2**

The empirical contribution is real and reproducible (+3.07 PQ controlled, +5 PQ over matched-trainer monocular baseline, generalization to KITTI/Waymo/Mapillary). The theoretical contribution is modest: DCFA is a LoRA-pattern adapter applied at a new site with a new conditioning signal, and SIMCF is a three-rule heuristic filter. Neither is a fundamental architectural innovation. For a methods-track NeurIPS reviewer, this is solid empirical engineering with a thin formal core. Score: 2 (some contribution).

# Overall (1-10): **6** (borderline accept, leaning weak accept)

The paper makes a real, controlled empirical contribution (+3.07 PQ) on a meaningful problem (monocular unsupervised panoptic), with honest limitations and a clean ablation suite. The methods are not theoretically deep — DCFA is a known adapter pattern with new conditioning, SIMCF is rule-based — and the paper does not engage with the parameter-efficient adapter literature it shadow-borrows from (LoRA, DoRA, IA3, Houlsby adapters). I would not vote against accepting this paper on methods grounds alone, but I would advocate strongly for the authors to (a) cite and discuss the LoRA/adapter family, (b) ablate DCFA design choices (FiLM vs. concat, attention vs. MLP, h sweep, depth encoding), (c) provide a SIMCF unified objective, (d) report multi-seed results, and (e) provide at least one same-backbone CUPS rerun even at reduced budget.

# Confidence (1-5): **4**

I am familiar with the unsupervised panoptic segmentation literature (CUPS, U2Seg, S2-UniSeg), the parameter-efficient adapter literature (LoRA, DoRA, IA3, FiLM), and the unsupervised semantic segmentation literature (STEGO, HP, CAUSE, DINOv2/v3). I am moderately confident in this assessment but not 100% on the exact numerical comparability of CUPS' 27-class metric across re-implementations.

# Questions for authors

**Architectural ablations (W1):**
Q1. Why concatenation over FiLM? Have you compared $\tilde z = \gamma(e(d)) \odot z + \beta(e(d))$ at matched parameter budget?
Q2. Why MLP over cross-attention? Even a single attention head over a small set of depth-bin tokens would let the conditioning be context-aware.
Q3. Why the specific 16-D sinusoidal encoding with $\omega_k = 2^k\pi$? Have you compared to log-depth, learned depth tokens, or a coarser frequency basis matched to typical depth statistics?
Q4. Hidden width $h=384$ is fixed; can you provide a sweep over $h \in \{64, 128, 256, 384, 512\}$ at fixed $\lambda_p$?

**SIMCF formalism (W2):**
Q5. Can you state SIMCF as the MAP estimate of a generative model with explicit semantic, geometric, and depth likelihoods? If not, why is the rule-based formulation preferred?
Q6. Why is the order Step A -> Step B -> Step C? Have you tested permuted orderings or iterated-to-fixed-point variants?
Q7. Step B uses DINOv3 features while the rest of Stage-1 uses DINOv2-trained CAUSE-TR. Is this strictly necessary? What is the PQ if Step B uses the same DINOv2-trained code?
Q8. Step C assumes per-class depth follows an approximately-Gaussian distribution at $\eta\sigma_c$ rejection. Is this assumption tested? What is the empirical depth distribution per class?

**Hungarian protocol (W3):**
Q9. Can you provide your own k-sweep at the Stage-1 level for $k \in \{27, 40, 54, 80, 120\}$? CUPS' transferring to your monocular setting is plausible but not verified.
Q10. Is the Hungarian assignment 1-to-1 (square) or many-to-1 (rectangular)? If the former, how do K-clusters that have no benchmark match get handled?
Q11. How sensitive is the final PQ to recomputing the Hungarian on different val splits?

**Mathematical clarity (W5):**
Q12. What unit is $\tau_d = 0.20$ in (Eq. 5)? Normalized depth? Inverse depth? Per-image max?
Q13. In Eq. 4, are $z$ and $\tilde z$ L2-normalized when $\mathcal{L}_\text{preserve}$ is computed? With normalization, $\|\tilde z - z\|_2^2 \leq 4$, so $\lambda_p = 20$ is a strong but bounded prior; without normalization the scale is unconstrained.
Q14. What is "dominant pseudo-class" in SIMCF Step A — strict majority, plurality, or mode?

**Theoretical positioning (W4):**
Q15. How does DCFA differ formally from a LoRA-style adapter (zero-init output, residual, anchored to frozen weights)? If the answer is "conditioning input + adapter site", state this explicitly and cite the LoRA/DoRA/IA3/Houlsby literature.

**Confound isolation (W6, W7):**
Q16. Can you provide even a single-seed estimate of how much of the 32.76 vs. 27.80 PQ gap comes from the DINOv3 backbone vs. the monocular pseudo-label substitution? A frozen-DINOv3 + CUPS-stereo+flow run would partition this exactly.
Q17. Multi-seed Stage-3 numbers (mean $\pm$ std over $\geq 3$ seeds) for the headline 35.83 PQ row?
