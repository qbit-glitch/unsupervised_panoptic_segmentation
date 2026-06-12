# DepthG × CAUSE-TR Fusion Adapters — Scale-Separated Cross-Model Distillation

- **Date**: 2026-06-12
- **Status**: Approved design, revision 2 (user-approved: Phase 1 = A + B decoupled, Phase 3 = coupled bridge; rev 2: comparison target is the baseline papers' published results, NOT the internal k=80 overclustering eval)
- **Decision log**: CCR discussion D002
- **Owner branch**: main (mono/semantics track)

## 1. Summary

Two frozen unsupervised semantic models coexist in this project with complementary
strengths: CAUSE-TR (DINOv2 ViT-B/14, 90-dim codes — strong semantic purity, coarse
14-px grid) and DepthG (DINO ViT-B/8, STEGO-style code — weak semantics, fine 8-px
grid with depth-guided relational structure). This spec defines a pair of small,
preservation-anchored residual adapters that let each model teach the other **only at
the scale where it is strong**: DepthG teaches short-range (local structure,
boundaries, thin objects); CAUSE teaches long-range (semantic identity, global
consistency). The adapters are trained decoupled against frozen teachers, so neither
model can degrade the other.

Deliverable: a standalone unsupervised-semantic-segmentation result — both adapted
models evaluated under their own papers' standard protocols and compared against the
**published** Cityscapes numbers of the CAUSE and DepthG papers. The claim being
tested: cross-model coupling improves *both* models over their published baselines.
The internal k=80 overclustering pipeline is explicitly **not** the comparison target
and is untouched by this work.

## 2. Goals, gates, and non-goals

Baseline papers: CAUSE (Kim et al. — CAUSE-TR variant) and DepthG (Sick et al.).
Their exact published Cityscapes numbers are **pinned during Phase 0** from the
papers/official repos into a verified-numbers ledger (project citation-verification
rule: no paper number is quoted from memory anywhere in specs, reports, or prose).

### Goals and acceptance gates

| Phase | Goal | Gate (kill rule if failed) |
|---|---|---|
| 0a | Protocol fidelity | Our re-evaluation of each *vanilla* model under its paper's official protocol matches the published number within ~1.0 mIoU; any larger gap is reconciled and documented before proceeding |
| 0b | Quantify complementarity | GT-oracle headroom ≥ ~1.5 mIoU over the stronger vanilla model (27-class protocol) AND disagreement pixels not dominated by both-wrong; otherwise stop and write a one-page negative note |
| 1A | Adapter A improves CAUSE-TR | Adapted CAUSE-TR beats the **published CAUSE-TR Cityscapes cluster-probe mIoU** (and our own 0a reproduction) by ≥ +1.0 mIoU under the CAUSE paper's protocol |
| 1B | Adapter B improves DepthG | Adapted mono DepthG beats the **published DepthG Cityscapes cluster-probe mIoU** (and our 0a reproduction) under the DepthG/STEGO protocol; secondary report: delta over the mono-retrain baseline (`reports/2026-06-03_1208_depthg_depthpro_retrain.md`) |
| 2 | Benchmark table + report | Full comparison table with verified published numbers (STEGO, DepthG, CAUSE at minimum) + internal attribution rows; all run-matrix results reported including negatives |
| 3 | Coupled bridge (deferred) | Only entered if BOTH 1A and 1B gates pass |

### Non-goals

- Stage-1 k=80 pipeline integration. The fusion adapters are NOT evaluated through, gated on, or merged into the spherical-k80 → SIMCF-ABC pseudo-label pipeline. Revisiting that is a separate future decision with its own spec.
- PQ / panoptic metrics of any kind (this is a semantic-segmentation result; PQ_things remains owned by the motion/A6000 track).
- Modifying SIMCF-ABC, the k=80 machinery, the φ/LUT, or the Stage-2 detector recipe.
- Joint/bidirectional training in Phase 1 (adapters are strictly decoupled; coupling is Phase 3).
- Improving the faithful-CUPS baseline control. The CUPS-baseline reconstruction must keep vanilla (un-adapted) DepthG — an adapter-improved baseline is no longer the baseline.

## 3. Verified background facts (project context, not comparison targets)

- **CAUSE-TR codes**: DINOv2 ViT-B/14 + TR decoder, 90-dim, frozen. Native 14-px patch grid. The official CAUSE eval path is the one already exercised by the Mode B (DINOv3 codebook) work.
- **DepthG model**: DINO ViT-B/8 + STEGO-style segmentation head producing a dense code `g` of dimension d_g (read from the checkpoint head config at implementation time; the adapter projection layer is shape-agnostic, `Linear(d_g, w)`). Inference: sliding window, 320×320 crops, stride 160, at 640×1280, via `refs/cups/cups/semantics/model.py::DepthG`. Official eval/training code vendored at `refs/cups/external/depthg/`.
- **DepthG checkpoints**:
  - Shippable (mono-pure): `checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt`. Known to sit below the official checkpoint on cluster probe (−7.5; see retrain report).
  - The CUPS-release DepthG checkpoint used by `refs/cups` `gen_pseudo_labels.py`: used for Phase 0a protocol reproduction (it is the artifact closest to the published numbers) and as an audit reference. Its training-depth provenance must be verified before it could ship in any mono-pure claim.
- **DepthG labels on disk** (for the audit): `/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_depthg_depthpro_monocular/train`.
- **Existing comparison scaffold**: `notebooks/compare_retrained_vs_cups_baseline.ipynb`, `mbps_pytorch/probe_depthg_depthpro_monocular.py`.
- **Known CAUSE-TR weakness** (complementarity hypothesis for the audit): thin/rare structures — e.g., in the k=80 study, zero clusters mapped to motorcycle (trainID 17) and traffic light (trainID 6). The k=80 machinery itself is out of scope, but the *diagnosis* (which classes CAUSE misses and whether DepthG sees them) transfers to the 27-class protocol.
- **DCFA (V3, canonical)** — architectural template only: depth-only residual adapter, `z' = z + r(e(d))`, 16-D sinusoidal depth encoding, `r: 16→384→90`, ~40K params, loss = correlation term + λ_preserve·L_preserve with λ_preserve = 20, σ_d = 0.5, P = 1024 sampled pairs per step.

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

### 4.4 Adapter A (CAUSE side)

```
z'(p) = z(p) + r_A( [z(p) ; W_A · ĝ(p)] )      W_A: d_g → 16,
r_A: (90+16) → 384 → 384 → 90, zero-init output     (~225K params incl. W_A)
```

Exact mirror of the *deployed* DCFA-V3 code path
(`mbps_pytorch/models/semantic/depth_adapter.py::DepthAdapter`, concat
conditioning, two hidden layers at h=384, zero-initialized residual head) with the
16-D sinusoidal depth encoding slot replaced by a learned 16-D projection of the
DepthG code. Loss:

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

### 4.5 Adapter B (DepthG side)

```
g'(q) = g(q) + r_B( [g(q) ; W_B · ẑ(q)] )      W_B: 90 → 16,
r_B: (d_g+16) → 384 → 384 → d_g, zero-init output (same deployed-DCFA mirror)
```

Symmetric loss with the teacher roles swapped:

```
L_B =   Σ_{P_l} corr( S_{g'}, S_z )      # cross-teacher: CAUSE teaches semantics at range
      + Σ_{P_s} corr( S_{g'}, S_g )      # self-teacher: frozen g anchors local 8-px structure
      + λ_p · mean_q ‖g'(q) − g(q)‖²     # λ_p = 20
```

The short-range self-teacher prevents CAUSE's 14-px blockiness from smearing DepthG's
fine structure (at long range, blockiness is irrelevant).

**Phase 1B run matrix** (2 runs): B1 stratified primary; B2 with W_B width 64.

### 4.6 Probes on adapted codes

Both baseline papers' published numbers come from probes *trained jointly with* the
model (CAUSE's `cluster_tr` probe; DepthG's `cluster_probe`/`linear_probe` inside
the Lightning checkpoint). The preservation anchor keeps adapted codes close to the
original code distribution, so the **primary readout applies the frozen original
probes unchanged to the adapted codes** — the most conservative and most
protocol-faithful comparison (identical readout weights as the published row; only
the code geometry changes). Secondary readout: 27-way spherical k-means refit
**once** on adapted train-split codes, centroids saved and frozen (centroids are
single-source-of-truth — the A6000 reproducibility lesson), reported alongside.
Linear probe is the standard eval-only diagnostic (GT-trained by definition; never
ships into any training path).

### 4.7 Training configuration (both adapters)

- Data: full Cityscapes train split (2975 images), features pre-cached (§6.2).
- Batch ≥ 32 images, ~50 epochs (recurring pattern P093: self-supervised adapters need bs ≥ 32, epochs ≥ 50 for stable convergence).
- Optimizer/schedule: copy the DCFA/V3 trainer defaults unchanged.
- Every checkpoint stores `adapter_config` (architecture, widths, teacher mode, pair offsets) — recurring pattern P086.
- Compute: local M4 Pro (MPS/CPU), same as DCFA training. Fallback: santosh 2×1080 Ti. Long runs use `nohup` + log file under `logs/` in the project root.

## 5. Phases

### Phase 0 — Protocol fidelity + complementarity audit (~1 day, kill-gated)

**0a. Reproduce the baselines under their own protocols.**

1. Pin the published Cityscapes numbers (cluster + linear probe) for CAUSE-TR and DepthG from the papers/official repos into a verified-numbers ledger (in the audit notebook + final report). No number enters the spec, report, or prose without this verification.
2. Re-evaluate vanilla CAUSE-TR through the official CAUSE eval path, and vanilla DepthG (CUPS-release checkpoint) through the official DepthG/STEGO eval path (`refs/cups/external/depthg/`). Match published numbers within ~1.0 mIoU; reconcile and document any larger gap (resolution, CRF, multi-scale settings are the usual suspects — the protocols, whatever they are, are then **locked** for all subsequent rows).
3. Also evaluate the mono-retrained DepthG checkpoint under the same locked protocol (it is Adapter B's substrate; its baseline number anchors the secondary delta report).

**0b. Complementarity audit** (extend `notebooks/compare_retrained_vs_cups_baseline.ipynb`):

1. Per-class IoU of vanilla CAUSE-TR vs vanilla DepthG under the locked 27-class protocol; inspect traffic light, motorcycle, pole, person specifically.
2. Pixel agreement maps (agree-right / agree-wrong / A-only-right / B-only-right).
3. GT-oracle per-pixel-best upper bound (analysis only; GT never enters any training path) → the headroom number for the gate.
4. Zero-training concat sanity check: per-branch whiten + L2-norm, concat `[z ; ĝ]`, 27-way k-means + Hungarian under the locked protocol.
5. Run 1–3 for both DepthG checkpoints (mono-retrained and CUPS-release). If only the CUPS-release checkpoint shows complementarity, that is a kill signal for any mono-pure claim and gets recorded as a finding.

### Phase 1 — A + B decoupled (~3–4 days including training cycles)

Build the shared feature cache, train the A run matrix and B run matrix
independently (they share the cache; neither blocks the other), refit probes per
§4.6, evaluate each variant against its gate in §2 under the Phase-0-locked
protocols.

### Phase 2 — Benchmark table + report (~1 day)

Assemble the comparison table under the locked protocols: published rows (STEGO,
DepthG, CAUSE-TR at minimum — every published number citation-verified), our
reproduction rows, attribution rows (vanilla CAUSE-TR → +DCFA → +Adapter A;
vanilla/mono DepthG → +Adapter B), and the winning adapter variants. Write
`reports/depthg_cause_fusion_adapter_report.md` with the full run matrix, all gates
pass/fail, negative results, and reproducibility commands. Update CCR memory and
`MEMORY.md`.

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
| `mbps_pytorch/models/semantic/cross_model_adapter.py` | `CrossModelAdapter` module (covers A and B via config: code_dim, cond_dim, proj_width) + stratified pair sampler + teacher loss — lives beside `depth_adapter.py`, the DCFA pattern it wraps | ≤ 300 lines |
| `mbps_pytorch/train_fusion_adapter.py` | Trainer for both adapters, cloned from the DCFA/V3 trainer (locate via repo index: config label `V3_dd16_h384_l2`); teacher mode and pair stratification as CLI flags | ≤ 400 lines |
| `mbps_pytorch/eval_fusion_adapter.py` | Glue: adapted codes → probe refit (§4.6) → official-protocol evaluation for the A side and the B side | ≤ 300 lines |

Audit and protocol-reproduction work goes into the existing comparison notebook, not
new scripts.

### 6.2 Feature cache

- Location: `/Volumes/code_files/datasets/cityscapes/fusion_feature_cache/{cause_z,depthg_g_mono[,depthg_g_official]}/train/`.
- Format: fp16 `.npy` per image — `z` at (32, 64, 90) from a 448×896 forward, `g` at (40, 80, 100) from DepthG's own half-res flip-averaged path (320×640), depth pooled to the z grid. ~1 MB/image → ~3 GB for the full train split per DepthG checkpoint.
- Training caches only; evaluation computes cross-features live on the official protocols' own tensors (no spatial bookkeeping between protocols).

### 6.3 Evaluation protocol (locked in Phase 0a)

- **Comparison target**: the baseline papers' published Cityscapes results. All our rows are produced by the official eval code paths (CAUSE repo path for the A side; `refs/cups/external/depthg/` STEGO-style eval for the B side), with settings matched to what the published numbers used, then frozen for every subsequent row.
- **Primary metric**: 27-class Cityscapes val cluster-probe mIoU (unsupervised assignment + Hungarian matching at evaluation only). Secondary: linear-probe mIoU (eval-only diagnostic, reported for protocol completeness).
- **No internal-protocol numbers** (k=80, φ/LUT, SIMCF, PQ) appear in any comparison row of this work.
- The φ/LUT GT-derivation disclosure issue is untouched and irrelevant here — nothing in this spec consumes the LUT.

## 7. Risks and guardrails

1. **Protocol mismatch — the new top risk.** USS numbers swing wildly with eval settings (this project once measured CAUSE-TR far above its published number using multi-scale + CRF). Comparing against published results is only honest after Phase 0a reproduces them with the official code; every later row uses the identical locked protocol. No row skips 0a.
2. **Redundancy with DCFA**: depth is already injectable. The attribution rows (vanilla → +DCFA → +Adapter A, all under the locked protocol) answer whether *learned* depth-guided features beat raw depth as a conditioning signal. If Adapter A loses to DCFA across the matrix, record the negative result.
3. **Weak-teacher corruption (A)**: guarded by long-range self-teaching + λ_p = 20 pointwise preservation.
4. **Blockiness smearing (B)**: guarded by short-range self-teaching + preservation.
5. **Identity shortcut**: the preservation anchor makes identity the safe default; gates are defined as *improvements over* the un-adapted baselines, so a do-nothing adapter fails its gate and is reported as such.
6. **Checkpoint purity**: any mono-pure claim ships only the mono-retrained DepthG checkpoint. The CUPS-release checkpoint is used for protocol reproduction and audit; its training-depth provenance must be verified before it appears in any claim about monocular inputs.
7. **Centroid/probe drift**: every probe fit in this spec is fit once, saved, and reused — never refit per-machine or per-eval.
8. **Novelty (paper angle)**: before any paper claim, run the standard "has this been done before" check against multi-backbone feature-fusion literature (e.g., SD+DINO-style fusion, CLIP+DINO open-vocab couplings) and cross-model distillation work. The candidate claim is the scale-separated bidirectional coupling of two *trained unsupervised segmenters* with preservation anchors — not feature fusion per se. Until checked, this is an engineering experiment, not a contribution.
9. **Campaign scope**: standalone semantics result. It does not touch the closed PQ_things local program, the pseudo-label pipeline, or the detector stages.

## 8. Reporting

Phase 2 ends with `reports/depthg_cause_fusion_adapter_report.md` (benchmark table
with citation-verified published numbers, verified-numbers ledger, full run matrix,
all gates pass/fail, negative results, reproducibility commands) plus CCR memory and
`MEMORY.md` index updates. If Phase 0 kills the project, the one-page negative note
takes the report's place and memory records the headroom number so the idea is not
re-explored from scratch later.
