# Stage-4 Dead-Class Recovery — α + γ' Ablation Plan

**Date**: 2026-04-30
**Branch**: `dino-cause-dora-adapter`
**Constraint**: class-name-free (no CLIP / FC-CLIP / Grounded-SAM-with-text / open-vocab models)
**Baseline (verified locally)**: Stage-3 DCFA+SIMCF-ABC at step 3000 = **35.83 PQ** (per `logs/eval_stage3_dcfa_simcf_abc_step3000.log`). Seesaw fine-tune saved at step 400 = 35.81 PQ (regression at saved step; training peaked 39.12 mid-run but was not snapshotted). Therefore the Stage-4 starting line is the Stage-3 ckpt, not the Seesaw ckpt.
**Six dead classes (PQ ≈ 0%)**: parking, guard rail, tunnel, polegroup, caravan, trailer (per paper §4 / `eval_da3_causetr_stage3.log`)

---

## 1 Diagnosis

The 6 dead classes are dead because the k=80 unsupervised pseudo-label generator produces **zero or near-zero pixels** mapped to these classes after Hungarian alignment to the 27-class evaluation vocabulary. Confirmed root cause:

- **Stuff** (parking, guard rail, tunnel, polegroup): cluster centroids for these classes are *absorbed* by neighboring high-density classes during Hungarian alignment. Hungarian collapse, not recognition failure.
- **Things** (caravan, trailer): RPN suppression compounds with vanishing pseudo-label coverage.

Existing Stage-4 (Seesaw + CAT) **rebalances signal that already exists**. It cannot create signal where there is none. This explains why dead classes remain dead despite Seesaw raising overall PQ.

**Implication**: Stage-4 must (a) recover lost cluster geometry *upstream* of Hungarian, AND (b) maximize gradient utilization of any rare-class pseudo-label that does survive.

---

## 2 Approach: α + γ' Hybrid

### α — Cluster Geometry First (NEW Stage-1.5)

| Component | Purpose | Mechanism |
|---|---|---|
| **α-1 NeCo post-training** | Sharpen DCFA-adapted DINOv3 patch features so small-volume modes stop being absorbed | Patch-Neighbor-Consistency loss (Pariza ICLR 2025), 19 GPU-hours. Self-supervised. No labels. |
| **α-2 k=300 reclustering** | Higher cluster cardinality preserves rare modes that k=80 collapses | Re-fit k-means at k=300 on NeCo-sharpened features |
| **α-3 t-NEB hierarchical merge** | Merge k=300 → ~74 clusters via maximum-density-path, **freezing the 6 lowest-density centroids** to prevent Hungarian collapse | t-NEB density-path merging (arXiv 2503.15582) with rare-mode freeze constraint. CPU, hours. |

### γ' — Post-Seesaw Long-Tail Polishing (NEW, drops Seesaw which is already done)

| Component | Purpose | Mechanism |
|---|---|---|
| **γ'-1 FreeMatch SAT** | Per-cluster adaptive confidence threshold in Stage-3 EMA | τ_c = (p̄_c / max_c p̄_c) · τ_global; tail clusters automatically get lower τ. ICLR 2023. |
| **γ'-2 BLV logit perturbation** | Inject Gaussian noise into DCFA semantic logits proportional to class-frequency-inverse | σ_c = a·log(N_max/N_c). CVPR 2023. |
| **γ'-3 AUCSeg T-Memory Bank** | Tail-class feature buffer + AUC loss bypasses batch-size limits | Per-cluster bank (1024 features each), SquareAUCLoss, weight α=0.1, activate after ep5. NeurIPS 2024. |
| **γ'-4 FRACAL post-hoc calibration** | Free inference-time fix using fractal dimension of per-class spatial distribution | No retraining. λ_fractal=1.0. CVPR 2025. |

**Why both α AND γ'?**
- α produces *more diverse pseudo-labels* with rare modes preserved
- γ' ensures the few new rare-class pixels get *full gradient credit* during training
- Stack-orthogonal — different bottlenecks (vocabulary, threshold, classifier, feature space, inference)

---

## 3 Ablation Matrix (5 experiments)

All experiments fine-tune from existing Stage-3 DCFA+SIMCF-ABC checkpoint (35.83 PQ).

| ID | Description | Components added | Compute | Success gate |
|---|---|---|---|---|
| **T0** | FRACAL only (free) | γ'-4 (FRACAL) post-hoc | 0 (eval only) | ΔPQ ≥ +0.3 OR ≥1 dead class non-zero |
| **T1** | γ' full | γ'-1 + γ'-2 + γ'-3 + γ'-4 | ~3 days A6000 (Stage-3 fine-tune 5K steps) | ΔPQ ≥ +0.5; ≥2 dead classes non-zero |
| **T2** | α full | α-1 + α-2 + α-3 + Stage-2 retrain + Stage-3 | NeCo 19h + Stage-2 retrain 2 days + Stage-3 1 day = ~4-5 days A6000 | ΔPQ ≥ +1.0; ≥3 dead classes non-zero |
| **T3** | α + γ' (headline) | All α + γ' | Same as T2 | ΔPQ ≥ +1.5; ≥4 dead classes non-zero |
| **T4** | NeCo-only diagnostic *(conditional)* | α-1 only (no merge) | NeCo 19h + Stage-2 1 day | Run only if T2 fails — isolates NeCo contribution |

**Decision flow**: T0 → if positive, T1 → if positive, T2 → T3. Branch to T4 only if T2 underperforms.

---

## 4 Component Implementation Plan

### α-1 NeCo post-training
- **New file**: `mbps_pytorch/stage4/neco_postrain.py`
- **Inputs**: existing DCFA-adapted DINOv3 ViT-B/16 checkpoint
- **Outputs**: same backbone + NeCo loss applied; saves `dcfa_neco.pt`
- **Reference**: NeCo (Pariza et al., ICLR 2025); reuse author's official 1-conv pooling head
- **Hyperparameters**: 19 GPU-hours, 100 epochs over Cityscapes train (no labels), patch ordering temperature τ=0.07
- **Tests**: (a) loss converges, (b) k-means on NeCo features yields ≥k=300 distinct centroids, (c) cluster purity by visual inspection

### α-2 k=300 reclustering
- **Modify**: `mbps_pytorch/generate_overclustered_semantics.py` to support `--k 300`
- **Use**: NeCo-sharpened features from α-1
- **Output**: pseudo-semantic at k=300 + new centroids npz

### α-3 t-NEB hierarchical merge with rare-mode freeze
- **New file**: `mbps_pytorch/stage4/hierarchical_merge.py`
- **Algorithm**:
  1. Compute pairwise centroid cosine similarity in DINOv3 space
  2. Compute per-centroid depth-statistics fingerprint (mean depth, depth variance, vertical position)
  3. Identify 6 "rare candidates" = centroids with smallest pixel coverage AND distinctive depth fingerprints
  4. Build merge graph excluding edges that would absorb rare candidates
  5. Apply t-NEB maximum-density-path merging until cardinality drops to ~74
  6. Run Hungarian alignment to 27-class
- **Output**: pseudo-semantic at k≈74 with rare modes preserved
- **Tests**: (a) frozen rare centroids survive, (b) merged centroids preserve majority semantic identity, (c) per-cluster pixel histograms match expected dead-class pixel counts (>50px/img for at least 4 of 6)

### γ'-1 FreeMatch SAT
- **Modify**: `refs/cups/cups/pl_model_self.py` — replace fixed τ in EMA self-training
- **Add**: per-cluster running-mean confidence buffer (EMA decay 0.999)
- **Replace** fixed `tau` with `tau_c = (p̄_c / max_c p̄_c) * τ_global`, clamped τ_c ≥ 0.5
- **Hyperparameters**: τ_global=0.95 (reuse existing CUPS default)
- **Tests**: (a) τ_c per cluster converges within 500 steps, (b) low-population clusters get τ_c < τ_global, (c) PQ on val ≥ baseline within 1K steps

### γ'-2 BLV logit perturbation
- **Modify**: `mbps_pytorch/models/semantic/depth_adapter_v2.py` — final classifier
- **Add**: training-time-only Gaussian noise: `logits += randn_like(logits) * σ_c`
  where `σ_c = a·log(N_max/N_c) + b` with a=0.5, b=0.1
- **Class counts** estimated from the merged k≈74 pseudo-label histogram (no GT)
- **Gating**: `if self.training:` only
- **Tests**: (a) noise σ_c larger for low-frequency clusters, (b) inference unchanged when `model.eval()`, (c) gradient flow not broken

### γ'-3 AUCSeg T-Memory Bank
- **New file**: `mbps_pytorch/stage4/aucseg_memory_bank.py`
- **Maintain**: per-rare-cluster feature buffer of size 1024
- **Loss**: SquareAUCLoss between buffer features and head-cluster features in current batch
- **Weight**: α=0.1, activate at ep ≥ 5 to avoid early collapse
- **Tests**: (a) buffer fills within 1K steps, (b) AUC loss decreases monotonically, (c) loss removed cleanly when α=0

### γ'-4 FRACAL post-hoc calibration
- **New file**: `mbps_pytorch/stage4/fracal_calibration.py`
- **Algorithm** (training-free):
  1. Run inference on val set, collect per-class binary masks
  2. Compute fractal dimension D_c per class using box-counting
  3. Calibration: `logit_c ← logit_c + λ · (D̄ − D_c)` at inference
- **λ_fractal=1.0** (CVPR 2025 default)
- **Tests**: (a) fractal-dim computation deterministic, (b) calibration zero-mean (no PQ shift on average), (c) measurable gain on rare classes

---

## 5 Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| NeCo doesn't separate rare classes (they're feature-collinear with neighbors) | Medium | T4 diagnostic isolates NeCo; if dead, fall back to γ'-only |
| t-NEB rare-mode freeze identifies wrong centroids | Medium | Use depth-statistics fingerprints to bias selection toward known-distinctive centroids; visual validation before training |
| FreeMatch SAT lowers τ too aggressively → confirmation bias | Low | Hard floor τ_c ≥ 0.5; only activate after Stage-3 round 1 |
| BLV noise destabilizes early DCFA training | Low | Gate on `training=True`, schedule σ_c warm-up over first 1K steps |
| AUCSeg T-bank starves clusters with no pixels | Medium | Skip clusters with <128 pixels accumulated; AUCSeg degrades gracefully |
| FRACAL hurts head classes | Low | Per-class λ_c ∝ tail-ness; cap shift magnitude |

---

## 6 Compute & Hardware

- **Primary**: A6000 48GB on anydesk
- **Secondary**: 2× GTX 1080 Ti 11GB on santosh (for parallel γ' fine-tunes if A6000 is busy)
- **Total budget**: ~5-6 days for full ablation (T0→T1→T2→T3); +3 days if T4 needed

---

## 7 Success Criteria (Stage-4 Win Conditions)

- **Headline**: T3 (α+γ' full) achieves PQ ≥ 37.5 (+1.7 over Stage-3 35.83 baseline)
- **Dead-class recovery**: ≥4 of 6 dead classes have PQ > 1% (currently all ≈0%)
- **Per-component evidence**: Each of α, γ' individually shows ΔPQ > 0.5 (additivity argument for paper)
- **Novelty for NeurIPS**: Density-frozen hierarchical merge + NeCo-on-DCFA = clean new contribution; no concurrent unsupervised panoptic uses this combo

---

## 8 Tests & Verification

- Unit tests for each new module (`mbps_pytorch/stage4/tests/`)
- Smoke test: 100-step micro-training on 10 images (catches integration errors)
- Per-experiment validation: PQ + per-class breakdown logged to W&B; comparison table auto-generated
- Pre-commit hooks: type-check, ruff, pytest

---

## 9 References (all accessible without GT labels)

- **NeCo** — Pariza et al., ICLR 2025. arXiv:2408.11054. https://vpariza.github.io/NeCo/
- **t-NEB** — arXiv:2503.15582
- **FreeMatch** — Wang et al., ICLR 2023. arXiv:2205.07246
- **BLV** — Wang et al., CVPR 2023. arXiv:2306.02061
- **AUCSeg** — Han et al., NeurIPS 2024. arXiv:2409.20398
- **FRACAL** — Alexandridis et al., CVPR 2025. arXiv:2410.11774
