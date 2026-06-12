# DepthG × CAUSE-TR Fusion Adapters — Scale-Separated Cross-Model Distillation

- **Date**: 2026-06-12
- **Status**: Approved design (user-approved in session; Phase 1 = A + B decoupled, Phase 3 = coupled bridge)
- **Decision log**: CCR discussion D002
- **Owner branch**: main (mono/semantics track)

## 1. Summary

Two frozen unsupervised semantic models coexist in this project with complementary
strengths: CAUSE-TR (DINOv2 ViT-B/14, 90-dim codes — strong semantic purity, coarse
14-px grid) and DepthG (DINO ViT-S/8, STEGO-style code — weak semantics, fine 8-px
grid with depth-guided relational structure). This spec defines a pair of small,
preservation-anchored residual adapters that let each model teach the other **only at
the scale where it is strong**: DepthG teaches short-range (local structure,
boundaries, thin objects); CAUSE teaches long-range (semantic identity, global
consistency). The adapters are trained decoupled against frozen teachers, so neither
model can degrade the other.

Primary deliverable: better Stage-1 mono semantic pseudo-labels (Adapter A output
replaces DCFA codes in the existing spherical k=80 → SIMCF-ABC pipeline). Secondary
deliverable: an improved monocular DepthG (Adapter B), targeting recovery of the
documented cluster-probe regression of the DepthPro retrain.

## 2. Goals, gates, and non-goals

### Goals and acceptance gates

| Phase | Goal | Gate (kill rule if failed) |
|---|---|---|
| 0 | Quantify complementarity | GT-oracle headroom ≥ ~1.5 mIoU over the DCFA baseline AND disagreement pixels not dominated by both-wrong; otherwise stop and write a one-page negative note |
| 1A | Adapter A beats DCFA codes | ≥ +0.3 PQ or +1.0 mIoU over **PQ 26.41 / mIoU 56.57** on the locked same-script spherical-k80 eval (train split, identical script/mapping/split as the 2026-06-09 result) |
| 1B | Adapter B improves mono DepthG | Cluster-probe mIoU above the `depthg_depthpro_monocular` retrain baseline (numbers in `reports/2026-06-03_1208_depthg_depthpro_retrain.md`); stretch goal: parity with the CUPS-release DepthG cluster probe (closing the −7.5 gap) |
| 2 | Integration | Adapter-A labels through SIMCF-ABC beat the full DCFA+SIMCF-ABC baseline on train-locked eval, confirmed on val |
| 3 | Coupled bridge (deferred) | Only entered if BOTH 1A and 1B gates pass |

### Non-goals

- PQ_things improvement (owned by the motion/A6000 track per the 2026-06-12 campaign decision).
- Modifying SIMCF-ABC, the k=80 clustering protocol, the φ/LUT machinery, or the Stage-2 detector recipe.
- Joint/bidirectional training in Phase 1 (adapters are strictly decoupled; coupling is Phase 3).
- Improving the faithful-CUPS baseline control. The CUPS-baseline reconstruction must keep vanilla (un-adapted) DepthG — an adapter-improved baseline is no longer the baseline.

## 3. Verified background facts

- **CAUSE-TR codes**: DINOv2 ViT-B/14 + TR decoder, 90-dim, frozen. Native 14-px patch grid at the generation resolution used by the existing k=80 pipeline.
- **Current Stage-1 best (the baseline to beat)**: DCFA codes → spherical k-means k=80 → SIMCF-ABC: **PQ 26.41, PQ_things 15.19, mIoU 56.57** (locked same-script eval, 2026-06-09, train split). Labels: `/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc_spherical_k80/{train,val}`.
- **DCFA (V3, canonical)**: depth-only residual adapter, `z' = z + r(e(d))`, 16-D sinusoidal depth encoding, `r: 16→384→90`, ~40K params, loss = correlation term + λ_preserve·L_preserve with λ_preserve = 20, σ_d = 0.5, P = 1024 sampled pairs per step.
- **DepthG model**: DINO ViT-S/8 + STEGO-style segmentation head producing a dense code `g` of dimension d_g (read from the checkpoint head config at implementation time; the adapter projection layer is shape-agnostic, `Linear(d_g, w)`). Inference: sliding window, 320×320 crops, stride 160, at 640×1280, via `refs/cups/cups/semantics/model.py::DepthG` (`model.net(...)` returns the code).
- **DepthG checkpoints**:
  - Shippable (mono-pure): `checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt`.
  - Audit reference only: the CUPS-release DepthG checkpoint used by `refs/cups` `gen_pseudo_labels.py`. Its training-depth provenance must be verified before it could ever ship in the mono pipeline; until verified, treat as non-mono.
- **DepthG labels on disk**: `/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_depthg_depthpro_monocular/train` (semantic-only generation, CRF-refined).
- **Existing comparison scaffold**: `notebooks/compare_retrained_vs_cups_baseline.ipynb`, `mbps_pytorch/probe_depthg_depthpro_monocular.py`.
- **Known dead classes**: the k=80 cluster→class LUT maps zero clusters to motorcycle (trainID 17) and traffic light (trainID 6) — thin/rare structures that an 8-px depth-guided model may separate. This is the concrete complementarity hypothesis Phase 0 tests.

## 4. Method

### 4.1 Notation and alignment

For image x: CAUSE code `z(p) ∈ R^90` on the 14-px grid; DepthG code `g(q) ∈ R^{d_g}`
on the 8-px grid. Cross-resolution alignment is bilinear in both directions:
`ĝ = bilinear(g → CAUSE grid)` for Adapter A; `ẑ = bilinear(z → DepthG grid)` for
Adapter B. Both source models are frozen everywhere; only adapter weights train.

### 4.2 Stratified pair sampling

Each training step samples P = 1024 pixel pairs per image, split by construction into
two pools on the CAUSE grid (equivalent metric distances in pixels on the DepthG grid):

- **Short-range pool P_s (512 pairs)**: spatial offset ≤ 4 patches (≈ 56 px). Rationale: covers thin-structure neighborhoods and boundary bands where the 8-px model is reliable.
- **Long-range pool P_l (512 pairs)**: spatial offset ≥ 8 patches (≈ 112 px). Rationale: beyond local texture; similarity at this range is a semantic statement, where CAUSE is reliable.

Pairs with intermediate offsets (4–8 patches) are not sampled. The offsets are
committed defaults; one ablation may vary them, but Phase 1 runs use these values.

### 4.3 Teachers

Teacher and student similarities use cosine similarity clamped at zero,
`S(i,j) = max(cos(·_i, ·_j), 0)`, so the cross-model teachers occupy the same [0,1]
range as DCFA's depth kernel and the existing correlation-loss implementation is
reused **verbatim** from the DCFA/V3 trainer, parameterized by a teacher similarity
matrix (no new loss math).

### 4.4 Adapter A (CAUSE side) — pipeline deliverable

```
z'(p) = z(p) + r_A( W_A · ĝ(p) )      W_A: d_g → 16,  r_A: 16 → 384 → 90  (~45K params)
```

Exact architectural mirror of DCFA v4 with the 16-D sinusoidal depth encoding slot
replaced by a 16-D projection of the DepthG code. Loss:

```
L_A =   Σ_{P_s} corr( S_{z'}, S_g )      # cross-teacher: DepthG teaches local structure
      + Σ_{P_l} corr( S_{z'}, S_z )      # self-teacher: frozen z anchors global semantics
      + λ_p · mean_p ‖z'(p) − z(p)‖²     # pointwise preservation, λ_p = 20
```

The long-range self-teacher is the guard against the weak teacher corrupting CAUSE's
global semantics; the pointwise term is the identity anchor (the CSCMRefineNet
identity-shortcut lesson, inverted: here the shortcut is the safe default and the
adapter must earn its deviations).

**Phase 1A run matrix** (4 runs):
- A1: all-pairs DepthG teacher, no stratification (the plain X-DCFA control).
- A2: stratified as above (**primary**).
- A3: dual short-range teacher — mean of DepthG similarity and the DCFA depth kernel.
- A4: A2 with W_A width 32 (capacity check — 16-D may bottleneck a learned d_g-dim feature in a way it does not bottleneck scalar depth).

### 4.5 Adapter B (DepthG side) — model-improvement deliverable

```
g'(q) = g(q) + r_B( W_B · ẑ(q) )      W_B: 90 → 16,  r_B: 16 → 384 → d_g
```

Symmetric loss with the teacher roles swapped:

```
L_B =   Σ_{P_l} corr( S_{g'}, S_z )      # cross-teacher: CAUSE teaches semantics at range
      + Σ_{P_s} corr( S_{g'}, S_g )      # self-teacher: frozen g anchors local 8-px structure
      + λ_p · mean_q ‖g'(q) − g(q)‖²     # λ_p = 20
```

The short-range self-teacher prevents CAUSE's 14-px blockiness from smearing DepthG's
fine structure (at long range, blockiness is irrelevant). After training, fit
spherical k-means (27 clusters, CUPS Cityscapes class count) on `g'` over the train
split **once** and freeze the centroids (centroids are single-source-of-truth — the
A6000 reproducibility lesson). Linear probe may be reported as an eval-only
diagnostic; it never ships.

**Phase 1B run matrix** (2 runs): B1 stratified primary; B2 with W_B width 64.

### 4.6 Training configuration (both adapters)

- Data: full Cityscapes train split (2975 images), features pre-cached (§6.2).
- Batch ≥ 32 images, ~50 epochs (recurring pattern P093: self-supervised adapters need bs ≥ 32, epochs ≥ 50 for stable convergence).
- Optimizer/schedule: copy the DCFA/V3 trainer defaults unchanged.
- Every checkpoint stores `adapter_config` (architecture, widths, teacher mode, pair offsets) — recurring pattern P086.
- Compute: local M4 Pro (MPS/CPU), same as DCFA training. Fallback: santosh 2×1080 Ti. Long runs use `nohup` + log file under `logs/` in the project root.

## 5. Phases

### Phase 0 — Complementarity audit (~half day, kill-gated)

Extend `notebooks/compare_retrained_vs_cups_baseline.ipynb`:

1. Per-class IoU of DepthG semantic-only labels vs DCFA spherical-k80 labels under the same protocol; inspect traffic light, motorcycle, pole, person specifically.
2. Pixel agreement maps (agree-right / agree-wrong / A-only-right / B-only-right).
3. GT-oracle per-pixel-best upper bound (analysis only; GT never enters any training path) → the headroom number for the gate.
4. Zero-training concat sanity check: per-branch whiten + L2-norm, concat `[z ; ĝ]`, spherical k=80, locked eval.
5. Run 1–3 for both DepthG checkpoints (mono-retrained and CUPS-release). If only the CUPS-release checkpoint shows complementarity, that is a kill signal for the mono pipeline and gets recorded as a finding.

### Phase 1 — A + B decoupled (~3–4 days including training cycles)

Build the shared feature cache, train the A run matrix and B run matrix
independently (they share the cache; neither blocks the other), evaluate each against
its gate in §2.

### Phase 2 — Integration (~1 day)

Winning A variant: regenerate codes → spherical k=80 → SIMCF-ABC → locked train eval
→ val confirmation → write report `reports/depthg_cause_fusion_adapter_report.md`
with the full run matrix, negative results included. Winning B variant: probe
evaluation + the gap-closure comparison against the CUPS-release checkpoint.

### Phase 3 — Coupled bridge (deferred; only if 1A AND 1B gates pass)

Joint training of both adapters with a cross-model agreement term between `S_{z'}`
and `S_{g'}` (cycle-consistent mutual distillation; candidate mixer: the in-repo
Mamba2 module). Out of scope for the current implementation plan; this spec only
reserves the phase and its entry gate. A separate spec is written if Phase 3 is
entered.

## 6. Implementation surface

### 6.1 New files (mbps_pytorch/, following existing module patterns)

| File | Purpose | Size guard |
|---|---|---|
| `mbps_pytorch/cache_fusion_features.py` | One-pass cache builder: per image, dump fp16 `z` grid (CAUSE) and `g` grid (DepthG sliding-window) to the data drive | ≤ 300 lines |
| `mbps_pytorch/models/adapters/cross_model_adapter.py` | `CrossModelAdapter` module (covers A and B via config: in_dim, proj_width, out_dim) + stratified pair sampler | ≤ 300 lines |
| `mbps_pytorch/train_fusion_adapter.py` | Trainer for both adapters, cloned from the DCFA/V3 trainer (locate via repo index: config label `V3_dd16_h384_l2`); teacher mode and pair stratification as CLI flags | ≤ 400 lines |
| `mbps_pytorch/eval_fusion_adapter.py` | Glue: adapted codes → spherical k=80 → locked eval (A); cluster-probe refit + CUPS-protocol eval (B) | ≤ 300 lines |

Audit work goes into the existing comparison notebook, not new scripts.

### 6.2 Feature cache

- Location: `/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/{cause_z,depthg_g}/train/`.
- Format: fp16 `.npy` per image. Estimated ~5 MB/image → ~15 GB for the full train split.
- The DepthG pass reuses the sliding-window path from `gen_semantic_only_depthg_depthpro.py` (same geometry: 640×1280, 320×320 crops, stride 160), dumping codes instead of (or alongside) argmax labels.

### 6.3 Evaluation protocol (locked)

- **Adapter A / Phase 2**: the exact same evaluation script, cluster→class mapping, and split that produced the 2026-06-09 spherical-k80 result (PQ 26.41 / mIoU 56.57). No protocol variations; numbers are only ever compared same-script. All matching is global Hungarian per the CUPS-standard project rule where the script does matching.
- **Adapter B**: cluster-probe mIoU under the existing `probe_depthg_depthpro_monocular.py` protocol, against the retrain-report baseline.
- The φ/LUT GT-derivation disclosure status is unchanged by this work — this spec adds no new GT touchpoints to any training path and inherits the existing pending remedy decision.

## 7. Risks and guardrails

1. **Redundancy with DCFA**: depth is already injected. Run A1/A2 vs the existing DCFA baseline answers whether *learned* depth-guided features beat raw depth as a conditioning signal. If A loses to DCFA across the matrix, record the negative result and stop — do not stack A on top of DCFA in Phase 1 (a stacked variant is allowed only as a Phase 2 follow-up if A alone passes its gate).
2. **Weak-teacher corruption (A)**: guarded by long-range self-teaching + λ_p = 20 pointwise preservation.
3. **Blockiness smearing (B)**: guarded by short-range self-teaching + preservation.
4. **Identity shortcut**: the preservation anchor makes identity the safe default; gates are defined as *improvements over* the un-adapted baselines, so a do-nothing adapter fails its gate and is reported as such.
5. **Checkpoint purity**: only the mono-retrained DepthG checkpoint ships. CUPS-release checkpoint is audit-reference only pending provenance verification.
6. **Centroid drift**: every k-means in this spec (k=80 for A, 27-way for B) is fit once, saved, and reused — never refit per-machine or per-eval.
7. **Novelty (paper angle)**: before any paper claim, run the standard "has this been done before" check against multi-backbone feature-fusion literature (e.g., SD+DINO-style fusion, CLIP+DINO open-vocab couplings) and cross-model distillation work. The candidate claim is the scale-separated bidirectional coupling of two *trained unsupervised segmenters* with preservation anchors — not feature fusion per se. Until checked, this is an engineering experiment, not a contribution.
8. **Campaign scope**: this is a semantics/PQ_stuff/mIoU play. It does not reopen the closed PQ_things local program; any PQ_things movement is incidental and not chased.

## 8. Reporting

Phase 2 ends with `reports/depthg_cause_fusion_adapter_report.md` (full run matrix,
all gates pass/fail, negative results, reproducibility commands) plus CCR memory and
`MEMORY.md` index updates. If Phase 0 kills the project, the one-page negative note
takes the report's place and memory records the headroom number so the idea is not
re-explored from scratch later.
