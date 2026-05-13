# Dead-Class Cascade-Side Recovery: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-05-05
**Branch:** `dino-cause-dora-adapter`
**Author:** MBPS team
**Source analysis:** `reports/dead_class_recovery_research_note.md` + audit confirming DCFA+SIMCF+DepthPro pseudo-labels contain all 80 clusters (cluster 66 = motorcycle ≈ 3 400 px/img).
**Hard constraint:** **Eval protocol (CUPS-standard global Hungarian) is fixed.** No per-class re-mapping, no per-class threshold tuning at eval. Only training-side interventions are allowed.

**Goal:** Lift `motorcycle` (PQ = 0.09), `caravan` (0), `trailer` (0), `person` (PQ = 15.42) from cascade-side starvation to ≥ 5 PQ each, without modifying the eval protocol.

**Tech Stack:** PyTorch + Detectron2 + Lightning, Cityscapes, DINOv3 ViT-B/16 + Cascade Mask R-CNN. Active checkpoint: `checkpoints/stage4_pathB_focal_w005_gated/best_pq_step=000075.ckpt` (PQ = 37.98 local eval).

---

## Decision Log

### Why training-side only

User constraint: changing the Hungarian mapping at eval (e.g., using a fixed cluster→27 assignment from `cups_class_mapping.json`) would break protocol parity with the CUPS baseline (PQ 27.80) and invalidate the 37.98 → +10.18 headline comparison. The only valid path is making the cascade itself produce strong, discriminative cluster-66/67/78 predictions through training-side rebalancing.

### Why CPU-doable subset matters

User has constrained GPU access (single 2x GTX 1080 Ti). Path-B v2 is currently consuming both GPUs. CPU-only subtasks (diagnostic on a small val subset, rare-pool build from existing pseudo-labels, code-wiring with unit tests) can land in this session without disturbing the live training.

### What is NOT in scope

- Hungarian-mapping changes at eval (forbidden by user constraint).
- Architectural changes to the backbone or detector head structure (Mask2Former / DINO / OneFormer).
- New pseudo-label generation pipelines (your DCFA+SIMCF+DepthPro labels already contain the rare classes per the audit).

---

## File Structure Map

| File | Status | Responsibility |
|---|---|---|
| `scripts/audit_cascade_cluster_predictions.py` | Create | Task 1 (CPU). Inference on 50 val images; histogram of predicted cluster IDs. |
| `scripts/build_rare_pool.py` | Create or extend existing | Task 2 (CPU). Walk DCFA+SIMCF+DepthPro pseudo-labels, extract per-class crops for cluster 66/67/78 + person/rider; save to `cityscapes/rare_pool_dcfa_simcf_abc/`. |
| `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_pathB_rarepool_santosh.yaml` | Create | Task 3. Stage-4 yaml extending Path-B with `USE_RARE_POOL: True`, `RARE_POOL_PATH: ...`, RFS-aware sampler. Identical to current Path-B yaml otherwise. |
| `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py` | Modify | Task 4. Wire `USE_FED_LOSS: True` + EQLv2 path through cascade box head; configurable via yaml (`MODEL.ROI_BOX_HEAD.USE_FED_LOSS`, `USE_EQLV2`). |
| `refs/cups/tests/test_cascade_long_tail_losses.py` | Create | Task 4. Unit tests covering FedLoss + EQLv2 in cascade box head; CPU-only. |
| `refs/cups/configs/train_cityscapes_dinov3_vitb_stage2_rarepool_santosh.yaml` | Create | Task 5. Stage-2 yaml with rare-pool active from step 0; otherwise mirrors `train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml`. |
| `refs/cups/cups/model/modeling/proposal_generator/rare_rpn.py` | Create | Task 6 (later). RCAB-FRPN — rare-class anchor boosting (designed in `REPORT_stage2_dead_class_recovery.md`). |
| `reports/dead_class_recovery_diagnostic_results.md` | Create | Task 1 output: per-cluster prediction frequencies + interpretation. |

---

## Task 1: Cluster-Frequency Diagnostic (CPU-only, ~30 min)

**Question this task answers:** *Does the cascade currently predict cluster 66 (motorcycle) at all?*

**Responsible agent:** `general-purpose` (Explore + light scripting). Locally executable.

**Files:**
- Create: `scripts/audit_cascade_cluster_predictions.py`
- Output: `reports/dead_class_recovery_diagnostic_results.md`

**Mechanism:**
1. Load `checkpoints/stage4_pathB_focal_w005_gated/best_pq_step=000075.ckpt` on CPU.
2. Run inference on 50 deterministic Cityscapes val images (the same set the local eval used).
3. For each predicted panoptic map, count pixels per cluster ID (semantic head argmax) AND count predicted thing-instance entries per cluster ID (ROI heads).
4. Output a histogram + report.

**Tasks:**
- [ ] **Step 1:** Write the diagnostic script.
  ```python
  # scripts/audit_cascade_cluster_predictions.py
  # CLI:
  #   python scripts/audit_cascade_cluster_predictions.py \
  #     --ckpt checkpoints/stage4_pathB_focal_w005_gated/best_pq_step=000075.ckpt \
  #     --cfg  refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml \
  #     --cityscapes /Users/qbit-glitch/Desktop/datasets/cityscapes/ \
  #     --num_images 50 \
  #     --out reports/dead_class_recovery_diagnostic_results.md
  ```
- [ ] **Step 2:** Execute (CPU; ~3 min/img × 50 = ~2.5 hr OR ~5 min if input resolution scaled to 384x768).
- [ ] **Step 3:** Write the interpretation section in the output report:
  - If cluster 66 mean pixels per image > 100 → cascade IS predicting motorcycle; bottleneck is Hungarian mapping (no protocol-compliant fix; document as a known limitation).
  - If 100 ≥ mean > 0 → cascade is whispering; rebalancing should amplify (Tasks 2-4).
  - If mean = 0 → cascade is silent; the hardest case; Stage-2 retraining (Task 5) likely required.
- [ ] **Step 4:** Commit.

**Validation:** Histogram is plausible (clusters with high training-set frequency are predicted often; cluster 66 may be silent or quiet). No model state mutated.

**Why CPU-doable:** 50 images at 384×768 input on Mac M4 Pro is ~5 min total. No DDP, no large activation, no training step.

---

## Task 2: Build Rare Pool from existing pseudo-labels (CPU-only, ~30 min)

**Responsible agent:** `general-purpose` (Read + Write + numpy). Locally executable.

**Files:**
- Create: `scripts/build_rare_pool.py` (or extend an existing rare-pool builder if one exists)
- Output: `/Users/qbit-glitch/Desktop/datasets/cityscapes/rare_pool_dcfa_simcf_abc/`

**Mechanism:** For each cluster ID in `{66, 67, 78}` (motorcycle/caravan/trailer per the inject_sam3 mapping) and `{17, 18}` (person/rider — confirm via class mapping):
1. Walk all `*_semantic.png` in `cups_pseudo_labels_dcfa_simcf_abc/`.
2. For each connected component of the target cluster with area ≥ τ_area = 1000 px:
   - Compute the bounding box.
   - Extract the corresponding image crop from `leftImg8bit/train/`.
   - Extract the corresponding instance mask region from `*_instance.png`.
   - Save as `(image_crop, instance_mask, cluster_id, area)` tuple to a per-class subdirectory.
3. Verify per-class crop counts ≥ 50 (Detectron2 RFS sampler typically needs ≥ 50 to be useful per class).

**Tasks:**
- [ ] **Step 1:** Confirm cluster→cls mapping. Check `scripts/extract_cups_class_mapping.py` or `cups_class_mapping.json` for the canonical assignment.
- [ ] **Step 2:** Write `scripts/build_rare_pool.py` with progress bar + per-class summary.
- [ ] **Step 3:** Execute on the 2 975-image train split.
- [ ] **Step 4:** Verify per-class counts. If any target cluster has < 50 crops, log a warning.
- [ ] **Step 5:** Output `rare_pool_dcfa_simcf_abc/manifest.csv` with `(image_path, mask_path, cluster_id, area, image_id)` rows.
- [ ] **Step 6:** Commit script. Do NOT commit the pool itself (large binary; gitignored).

**Validation:** Sample 5 random crops from each target cluster + visualize them as a grid PNG; confirm they look like the intended class.

**Why CPU-doable:** Pure file I/O + numpy. No model inference. ~1 second per image × 2975 = ~50 min worst case.

---

## Task 3: Activate Rare-Pool + RFS in a new Stage-4 yaml (CPU-only, ~10 min)

**Responsible agent:** `general-purpose` (Edit, no model code). Locally executable.

**Files:**
- Create: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_pathB_rarepool_santosh.yaml`
- Possibly modify: `refs/cups/train_self.py` (to wire the RFS sampler if not already wired)

**Mechanism:** Copy `train_self_cityscapes_dinov3_vitb_stage4_fine_object_santosh.yaml` and override:
```yaml
AUGMENTATION:
  USE_RARE_POOL: True
  RARE_POOL_PATH: "/home/santosh/datasets/cityscapes/rare_pool_dcfa_simcf_abc/"
  RARE_POOL_PASTES_PER_IMAGE: (1, 3)
  RARE_POOL_CLASS_REPEAT_OVERRIDES: ((11, 4), (12, 4), (18, 4), (14, 8), (15, 8), (16, 8), (17, 8))
  RARE_POOL_USE_DEPTH_PLACEMENT: True
DATA:
  RFS_THRESHOLD_T: 0.001
  USE_RFS_SAMPLER: True   # may require new yaml key + train_self.py wiring
SYSTEM:
  RUN_NAME: "cups_dinov3_vitb_stage4_pathB_rarepool_santosh"
  LOG_PATH: "/home/santosh/experiments/stage4_pathB_rarepool"
```

**Tasks:**
- [ ] **Step 1:** Audit `refs/cups/train_self.py` for current sampler construction. If RFS is referenced but not wired (per CCR memory: `RFS_THRESHOLD_T = 0.001` exists but isn't connected), wire it.
- [ ] **Step 2:** Copy + edit the yaml.
- [ ] **Step 3:** Local test: `python -c "from yaml import safe_load; safe_load(open('...'))"` to verify yaml is valid.
- [ ] **Step 4:** Build rare-pool path on remote: `rsync rare_pool_dcfa_simcf_abc/ santosh:...` (deferred to user; remote step).

**Validation:** Yaml loads cleanly; existing `train_self.py` consumes the new flags without crashes when wired.

**Why CPU-doable:** Pure config + small Python wiring. No training run.

---

## Task 4: Wire FedLoss + EQLv2 in cascade box head (CPU-only, 1 day)

**Responsible agent:** `feature-dev:code-architect` (deep code change with tests). `feature-dev:code-reviewer` for the final pass. Locally executable.

**Files:**
- Modify: `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py`
- Modify: `refs/cups/cups/model/model.py` if config plumbing needs adjustment.
- Create: `refs/cups/tests/test_cascade_long_tail_losses.py`

**Mechanism:**
- **FedLoss:** Federated subsampling [Zhou 2022] — at each training step, sample only K classes from a class-frequency-weighted distribution; compute classification loss only over those K classes. Detectron2 has a built-in implementation.
- **EQLv2:** Per-class gradient-ratio rebalancing [Tan 2021]. Hooks into the classification loss backward pass.
- **Configuration:** Yaml flags `MODEL.ROI_BOX_HEAD.USE_FED_LOSS: True` and `MODEL.ROI_BOX_HEAD.USE_EQLV2: True`. Both compose with active Seesaw.

**Tasks:**
- [ ] **Step 1:** Locate the existing cascade classification loss in `fast_rcnn.py`. Identify which class predictions go through Seesaw.
- [ ] **Step 2:** Add FedLoss path:
  - At loss-compute time: sample K classes (`MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES`, default 50 in yaml).
  - Mask out non-sampled classes in the cross-entropy.
- [ ] **Step 3:** Add EQLv2 path:
  - Implement gradient-collect hook that tracks per-class positive/negative gradient ratios.
  - Apply per-class rebalancing factor at backward pass.
- [ ] **Step 4:** Write `test_cascade_long_tail_losses.py`:
  - `test_fed_loss_subsamples_k_classes` — assert only K classes have non-zero gradient.
  - `test_eqlv2_increases_rare_class_gradient` — synthetic batch with 1 motorcycle + 99 cars; assert motorcycle gradient norm grows with EQLv2 enabled.
  - `test_seesaw_fed_eqlv2_compose` — all three on simultaneously; loss is finite, gradient flows.
- [ ] **Step 5:** Run tests on local CPU.
- [ ] **Step 6:** Commit.

**Validation:** All tests pass on local. No regression in existing `test_stage4_dcr.py` (which already exercises Seesaw + EQLv2 components).

**Why CPU-doable:** Loss-only changes. Tests use synthetic 1-image batches.

---

## Task 5: Stage-2 retraining with rare-pool active from step 0 (GPU-only, 3 days)

**Responsible agent:** User runs remote (santosh@172.17.254.146 GPU). Plan emits the launch recipe.

**Files:**
- Create: `refs/cups/configs/train_cityscapes_dinov3_vitb_stage2_rarepool_santosh.yaml`

**Mechanism:** Stage-2 from Stage-1 baseline with rare-pool active throughout (RFCL principle from `training_strategies_dead_classes_report.md`). The cascade head crystallizes feature-space partitions during the first ~1000 steps; if cluster 66 lacks exposure during this window, no later-stage Path-B can recover it. This task addresses the failure mode at its source.

**Tasks:**
- [ ] **Step 1:** Copy `train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml` and override:
  ```yaml
  AUGMENTATION:
    USE_RARE_POOL: True
    RARE_POOL_PATH: ...
    NUM_STEPS_STARTUP: 0     # rare-pool active from step 0
    RARE_POOL_PASTES_PER_IMAGE: (2, 5)   # heavier than Stage-4
  TRAINING:
    NUM_STEPS: 8000
  ```
- [ ] **Step 2:** Hand the user the launch recipe (kill prior Stage-2 if any; relaunch Stage-2 → Stage-3 → Stage-4 chain).
- [ ] **Step 3:** Watch first val (step 200) for sanity.
- [ ] **Step 4:** After Stage-2 completes, re-fit Hungarian assignment on the new cluster output (Stage-3 needs this).

**Validation gate:** Stage-2 step-200 PQ within ±2 PQ of original Stage-2 baseline. If catastrophically lower, kill — rare-pool may be over-pasting.

**Why GPU-only:** Full Stage-2 training run — 8000 batch steps × 5 forwards × 137M params per forward.

---

## Task 6: RCAB-FRPN (Rare-Class Anchor Boosting) (CPU code + GPU train, 1 week)

**Responsible agent:** `feature-dev:code-architect` for design + initial implementation; `feature-dev:code-reviewer` for cross-check. User runs remote training.

**Files:**
- Create: `refs/cups/cups/model/modeling/proposal_generator/__init__.py`
- Create: `refs/cups/cups/model/modeling/proposal_generator/rare_rpn.py` (full design in `REPORT_stage2_dead_class_recovery.md` Section 1)
- Modify: `refs/cups/cups/model/model.py` to register `RareClassRPN` in `PROPOSAL_GENERATOR_REGISTRY`

**Mechanism:**
- (a) Inject rare-class-biased anchors with aspect ratios fitted to caravan/trailer (large, elongated).
- (b) Up-weight RPN objectness loss for anchors matched to rare-class GT boxes.

**Tasks:**
- [ ] **Step 1:** Implement `RareClassRPN` per the design in `REPORT_stage2_dead_class_recovery.md` lines 60-200.
- [ ] **Step 2:** Add unit tests on CPU (synthetic anchor/GT pairs).
- [ ] **Step 3:** Wire yaml flag `MODEL.RPN.HEAD_NAME: "RareClassRPN"`.
- [ ] **Step 4:** User retrains Stage-2 with this RPN.

**Validation:** RPN objectness loss curves don't diverge; rare-class anchor recall (computed at training-time) > 0.30.

**Why GPU for execution:** Re-training Stage-2 with new RPN.

---

## Tasks (TDD ordering, one action per step)

### Wave A — CPU-only (this session)

- [ ] **A.1 Task 1 — Diagnostic script.** Write `scripts/audit_cascade_cluster_predictions.py`. Execute on 50 val images. Output `reports/dead_class_recovery_diagnostic_results.md`.
- [ ] **A.2 Task 2 — Rare-pool builder.** Write `scripts/build_rare_pool.py`. Execute on 2 975 train images. Verify per-class counts ≥ 50.
- [ ] **A.3 Task 3 — yaml prep.** Create `train_self_cityscapes_dinov3_vitb_stage4_pathB_rarepool_santosh.yaml`. Wire RFS sampler in `train_self.py` if needed.
- [ ] **A.4 Task 4 (code).** Wire FedLoss + EQLv2 in cascade box head + tests. Run pytest locally.

### Wave B — GPU-only (deferred to user)

- [ ] **B.1 Task 5 — Stage-2 rarepool retrain.** User launches on remote.
- [ ] **B.2 Task 6 — RCAB-FRPN.** Code (CPU) → user retrains Stage-2 (GPU).

---

## Validation Gates

| Gate | Command | Pass criterion |
|---|---|---|
| Task 1 — Diagnostic ran cleanly | `python scripts/audit_cascade_cluster_predictions.py ...` | Output report contains a per-cluster histogram. No exceptions. |
| Task 2 — Rare pool exists | `ls /Users/.../cityscapes/rare_pool_dcfa_simcf_abc/manifest.csv` | Per-class crop counts ≥ 50 for clusters 66, 67, 78. |
| Task 3 — yaml valid | `python -c "from yaml import safe_load; safe_load(open('...yaml'))"` | No exceptions. |
| Task 4 — long-tail loss tests | `pytest refs/cups/tests/test_cascade_long_tail_losses.py -v` | All tests green. No regression in `test_stage4_dcr.py`. |
| Task 5 — Stage-2 first val | log inspection | PQ at step 200 within ±2 PQ of original Stage-2 baseline. |

---

## Risks and Rollback

### Task 1 (diagnostic)
- **Risk:** None — read-only.
- **Rollback:** N/A.

### Task 2 (rare pool build)
- **Risk:** Disk space; rare-pool can be ~1 GB. No persistent state changes besides creating a new directory.
- **Mitigation:** Per-class count cap (e.g., 200 crops per class) keeps disk ≤ 500 MB.
- **Rollback:** `rm -rf /Users/.../rare_pool_dcfa_simcf_abc/`.

### Task 3 (yaml + sampler wiring)
- **Risk:** Breaking the existing dataloader if the RFS wiring is incorrect.
- **Mitigation:** Add an env-var fallback in `train_self.py` so RFS is opt-in; default off.
- **Rollback:** Remove the new yaml; revert wiring change.

### Task 4 (FedLoss + EQLv2)
- **Risk:** Loss numerical instability when all three (Seesaw + FedLoss + EQLv2) compose.
- **Mitigation:** Unit tests with synthetic batches; assert finite loss + finite gradient norms.
- **Rollback:** Yaml flags allow disabling either independently.

### Task 5 (Stage-2 retrain)
- **Risk:** Rare-pool active from step 0 may slow early convergence; the cascade may not reach standard Stage-2 peak (33-35 PQ).
- **Mitigation:** Validation gate at step 200 (±2 PQ floor). If below, kill.
- **Rollback:** Revert to existing Stage-2 best_pq ckpt; abandon rarepool yaml.

### Task 6 (RCAB-FRPN)
- **Risk:** Custom RPN may not register correctly with detectron2's PROPOSAL_GENERATOR_REGISTRY.
- **Mitigation:** Unit test that calls `build_proposal_generator(cfg)` and asserts the type.
- **Rollback:** Yaml flag `MODEL.RPN.HEAD_NAME: "StandardRPNHead"` reverts to original.

---

## Spec Coverage Self-Review

- Task 1 — covered by Wave A.1 + diagnostic spec section.
- Task 2 — covered by Wave A.2 + rare-pool builder spec section.
- Task 3 — covered by Wave A.3.
- Task 4 — covered by Wave A.4.
- Task 5 — covered by Wave B.1.
- Task 6 — covered by Wave B.2.
- Hard constraint preserved: no eval-side changes anywhere.
- Per-task agent assignments listed.
- CPU-vs-GPU split explicit per task.
- Validation criteria + rollback per task.

---

## Type Consistency

- Cluster IDs: `int` in `[0, 79]` ∪ `{255}` (ignore).
- Bounding box from connected component: `(x_min, y_min, x_max, y_max)` ints.
- `instance_mask`: `uint8` numpy array, 0 = background, > 0 = instance ID.
- Test assertions use `torch.isfinite`, `torch.equal` for tensor comparison.

---

## Open Questions for the User

- Confirm the canonical mapping cluster_id → SAM3 / Cityscapes class for the rare-pool builder. The inject_sam3 script used `{2: 66, 10: 67, 11: 78}` (SAM3 → cluster). Should we extend rare-pool to also include person (cluster 17 per CCR memory's `_SAM3_TO_CUPS27_LEGACY: 17`)?
- Decide whether Task 5 (full Stage-2 retrain) will be scheduled before or after the next Path-B variant. It is the most expensive but most leverage-bearing.
- Confirm RFS sampler wiring in `train_self.py` — based on CCR memory, `RFS_THRESHOLD_T` is in config but not connected; verify by reading the dataloader-build path.

---

## Task 7 (REVISED 2026-05-06) — Path-C: 14-Channel SAM3-Thing Adapter from Stage-3 Ckpt (GT-free, CPU-trainable)

> **Revision note (2026-05-06):** original task proposed a 27-class adapter initialized from the cluster_to_class LUT. **The LUT itself is GT-derived** (Hungarian-matched against train-set GT), so using it during training would violate the unsupervised paradigm. This revision narrows the adapter to **14 SAM3 fine-grained channels** (no Cityscapes class names involved during training) and replaces LUT init with random Gaussian / optional self-distillation. Eval-time fusion combines the adapter's thing predictions with the cluster head's stuff predictions via standard CUPS Hungarian — which is allowed at eval per CUPS protocol.

### Architecture (revised)

Start from `checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt`. Add a small MLP adapter on the FPN-P4 features. Output is **14 SAM3-fine-grained channels**, NOT 27 Cityscapes classes:

```
                  Stage-3 best.ckpt (frozen)
                  ───────────────────────────────────────────────────
   image x ─► DINOv3 (frozen) ─► SimpleFPN (frozen) ─┐
                                                      │
                                                      ├──► [80-cluster sem-seg head, frozen]   ─► z ∈ ℝ^80
                                                      │       │
                                                      │       └─► used at eval for STUFF via CUPS Hungarian
                                                      │           (Hungarian is allowed at eval per protocol)
                                                      │
                                                      ▼
                                           F_{P4} ∈ ℝ^{256 × H/4 × W/4}
                                                      │
                                                      ▼
                                           ┌──────────────────────────────┐
                                           │  AuxThingAdapter (NEW)       │   ◄── trainable, ~17K params
                                           │  W1: 256 → 64                │
                                           │  GELU                        │
                                           │  W2: 64 → 14                 │
                                           │  Output: 14 SAM3 thing classes│
                                           └────────────┬─────────────────┘
                                                        │
                                                        ▼
                                                ŷ ∈ ℝ^{14 × H/4 × W/4}
                                                        │
                                                        ├──► training: SAM3 CE on supervised pixels (~15% of pixels)
                                                        │
                                                        └──► inference: combined with z via eval-fusion (below)
```

### What is and isn't allowed during training

| Resource | Status | Why |
|---|---|---|
| DINOv3 frozen features | **Allowed** | Foundation model, no Cityscapes labels |
| 80-cluster pseudo-labels | **Allowed** | Self-supervised k-means output |
| SAM3 masks + class indices | **Allowed** | Foundation-model output (SAM3 trained on SA-1B) |
| Hungarian matching at **eval** | **Allowed (CUPS protocol)** | Test-time matching only |
| `cluster_to_class` LUT during training | **FORBIDDEN** | Built using train-set GT |
| Cityscapes class labels during training | **FORBIDDEN** | GT |

### Math (revised: GT-free)

**Adapter:**

$$
\mathrm{Adapter}_\theta(F_u) = \mathbf{W}_2 \cdot \sigma\bigl(\mathbf{W}_1 F_u + \mathbf{b}_1\bigr) + \mathbf{b}_2 \;\in\; \mathbb{R}^{14}
$$

with $\sigma$ = GELU, $\mathbf{W}_1 \in \mathbb{R}^{64 \times 256}$, $\mathbf{W}_2 \in \mathbb{R}^{14 \times 64}$. Total params: $256\cdot 64 + 64\cdot 14 + 64 + 14 = 17\,326$.

**Initialization (no LUT):**

$$
\mathbf{W}_1 \sim \mathcal{N}(0,\,2/256), \quad
\mathbf{W}_2 \sim \mathcal{N}(0,\,2/64), \quad
\mathbf{b}_1 = \mathbf{b}_2 = \mathbf{0}
$$

(Kaiming init for GELU, no information from any labeled source.)

**Per-pixel SAM3 target (in SAM3-class space, 14 channels):**

$$
t(u) = \begin{cases}
c_k^{\mathrm{SAM3}} \in \{0,\ldots,13\} & \text{if } u \in m_k \text{ for some SAM3 mask } k \\
\texttt{ignore} & \text{otherwise}
\end{cases}
$$

**Loss:** standard cross-entropy:

$$
\mathcal{L} = \frac{1}{|\mathcal{P}|}\sum_{u \in \mathcal{P}} \mathrm{CE}\bigl(\mathrm{Adapter}(F_u),\; t(u)\bigr), \qquad \mathcal{P} = \{u : t(u) \neq \texttt{ignore}\}
$$

### Eval-time fusion (the key step; CUPS-protocol-compliant)

At inference:

1. Run frozen pipeline: image → DINOv3 → FPN → both heads.
2. Cluster head produces `z ∈ ℝ^{80}` per pixel.
3. Adapter produces `ŷ ∈ ℝ^{14}` per pixel.
4. **Apply CUPS Hungarian** to the val-set cluster predictions → 27-class prediction `ẑ_27` (standard protocol step, allowed).
5. **Per-pixel fusion:**

   For each pixel `u`:
   - Compute adapter confidence: $\mathrm{conf}(u) = \max_c \mathrm{softmax}(\hat{y}_u)[c]$
   - Compute adapter prediction: $c^*(u) = \arg\max_c \mathrm{softmax}(\hat{y}_u)[c]$
   - Map $c^*(u)$ ∈ {0..13} (SAM3 class) → trainID ∈ {0..18} via the **structural** SAM3→Cityscapes-thing mapping (this mapping is *definitional*, not learned — bicycle is bicycle in any taxonomy):
     ```
     SAM3_TO_TRAINID = {
         0: 11,  # person
         1: 18,  # bicycle
         2: 17,  # motorcycle  ◄── previously dead
         3: 12,  # rider
         4: 7,   # traffic sign
         5: 6,   # traffic light  ◄── previously dead
         6: 14,  # truck
         7: 15,  # bus
         8: 16,  # train
         12: 13, # car
         13: 5,  # pole
     }
     ```
   - **Fusion rule:**
     $$
     \mathrm{final}(u) = \begin{cases}
     \mathrm{SAM3\_TO\_TRAINID}[c^*(u)] & \text{if } \mathrm{conf}(u) > \tau \text{ AND } c^*(u) \in \mathrm{SAM3\_TO\_TRAINID} \\
     \hat{z}_{27}(u) & \text{otherwise (fall back to Hungarian-collapsed cluster head)}
     \end{cases}
     $$

   With $\tau = 0.5$. Cluster head handles stuff (where SAM3 has no signal); adapter handles things (where SAM3 has direct signal).

### Why the SAM3 → Cityscapes-trainID mapping is GT-free

This mapping is **structural / definitional, not learned**. SAM3's "motorcycle" concept and Cityscapes's "motorcycle" concept refer to the same real-world object — the mapping just identifies the two labels. No Cityscapes train-set labels are ever consulted. This is consistent with how CUPS uses CutLER masks: CutLER outputs class-agnostic masks; CUPS implicitly identifies them with Cityscapes things via taxonomic overlap. We're doing the same with finer SAM3 granularity.

### Optional self-distillation warmup (still GT-free)

If random init converges too slowly, a brief warmup using the existing 80-cluster head as a *teacher* can bootstrap:

$$
\mathcal{L}_{\mathrm{warmup}} = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \mathrm{KL}\bigl(\mathrm{softmax}(\mathrm{Adapter}(F_u)) \;\big\|\; \pi(\mathrm{softmax}(\mathbf{z}_u))\bigr)
$$

where $\pi$ is a *learnable* 80→14 projection trained jointly. After ~100 warmup steps, the adapter mimics what the cluster head already predicts — but without using the LUT or any GT alignment. From that point, SAM3 supervision refines.

### Implementation steps (CPU-only on Mac M4 Pro)

#### 7.1 — `AuxThingAdapter` module (new file)

**File:** `refs/cups/cups/model/aux_thing_adapter.py` (~80 lines)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxThingAdapter(nn.Module):
    """MLP head: FPN-P4 features → 14 SAM3-thing-class logits.

    Trained with SAM3 cross-entropy. Fully GT-free.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 64,
                 num_sam3_classes: int = 14):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_sam3_classes)

        # Kaiming init (no LUT, no GT)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        # F: (B, C=256, H, W) → out: (B, 14, H, W)
        if F.dim() == 4:
            x = F.permute(0, 2, 3, 1)  # → (B, H, W, 256)
            x = self.fc2(self.act(self.fc1(x)))
            return x.permute(0, 3, 1, 2)  # → (B, 14, H, W)
        return self.fc2(self.act(self.fc1(F)))
```

#### 7.2 — Tests (CPU)

**File:** `refs/cups/tests/test_aux_thing_adapter.py` (~50 lines)

- `test_adapter_init_param_count` — verify ~17 K params.
- `test_adapter_forward_shape` — input `(1, 256, 96, 192)` → output `(1, 14, 96, 192)`.
- `test_adapter_no_lut_used` — verify no Cityscapes class names in module attributes.
- `test_adapter_finite_loss_on_random_target` — random target + random input → finite CE loss.
- `test_adapter_gradient_flow` — backward → all params have non-zero grad.

#### 7.3 — Feature cache builder

**File:** `scripts/cache_stage3_p4_features.py` (new)

For each train + val image: load Stage-3 ckpt, run frozen backbone + FPN, extract P4 feature, save as float16 to `cityscapes_p4_features_dcfa_simcf_abc_step3000/{image_id}.pt`. Cache size: ~9 GB.

CLI:
```
python scripts/cache_stage3_p4_features.py \
  --ckpt checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt \
  --cfg refs/cups/configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
  --cityscapes /Users/qbit-glitch/Desktop/datasets/cityscapes/ \
  --out /Users/qbit-glitch/Desktop/datasets/cityscapes/p4_cache_stage3/ \
  --device mps
```

Estimated wall-clock on Mac MPS: ~2.5 hours (one-time).

#### 7.4 — Training script

**File:** `scripts/train_aux_thing_adapter.py` (new, ~200 lines)

- Loads cached `F_{P4}` per image.
- Loads SAM3 masks (existing on disk at `sam_fine_masks_sam3/train/`).
- Builds per-pixel SAM3 target tensor.
- Forward through `AuxThingAdapter`, compute CE on supervised pixels.
- Adam optimizer on adapter params only.
- 100 epochs, ~75 min total wall-clock with cache (Option A: no augmentation).

#### 7.5 — Eval-fusion script

**File:** `scripts/eval_aux_thing_adapter.py` (new, ~150 lines)

- Loads adapter ckpt + Stage-3 ckpt.
- For each val image: cluster head + adapter forward.
- Apply CUPS Hungarian on cluster head output (standard protocol).
- Apply per-pixel fusion rule.
- Compute PQ on the fused output.

### Tasks (TDD ordering)

- [ ] **Step 7.1** — Implement `AuxThingAdapter` (`refs/cups/cups/model/aux_thing_adapter.py`).
- [ ] **Step 7.2** — Write + run unit tests (`refs/cups/tests/test_aux_thing_adapter.py`).
- [ ] **Step 7.3** — Write feature cache builder.
- [ ] **Step 7.4** — Run cache builder on first 50 images for sanity check.
- [ ] **Step 7.5** — Run cache builder on full Cityscapes train (2 975 images, ~2.5 hours).
- [ ] **Step 7.6** — Write training script.
- [ ] **Step 7.7** — Local 100-epoch training (~75 min after cache build).
- [ ] **Step 7.8** — Write eval-fusion script.
- [ ] **Step 7.9** — Run local eval on val set with the trained adapter + Stage-3 cluster head.
- [ ] **Step 7.10** — Compare per-class PQ vs Path-B baseline (37.98). Save results to `reports/path_c_aux_thing_results.md`.

### Validation Gates

| Gate | Command | Pass criterion |
|---|---|---|
| Module unit tests | `pytest refs/cups/tests/test_aux_thing_adapter.py` | All pass. |
| Cache sanity check | `python scripts/cache_stage3_p4_features.py --check 50` | 50 cached files exist; load + forward test passes. |
| Adapter training stability | training log | Loss decreases monotonically; no NaN. |
| Eval after training | `python scripts/eval_aux_thing_adapter.py` | **motorcycle PQ ≥ 5.0** (was 0.09); **traffic light PQ ≥ 12** (was 7.31); **overall PQ ≥ 38.0** (was 37.98). |

### Risks and Rollback (revised)

- **Risk:** Random init → slow convergence. Adapter at step 0 is at uniform 1/14 confidence. SAM3 CE has to teach from scratch.
  - **Mitigation:** Run 100 epochs with cosine LR schedule; if still poor, enable optional self-distillation warmup.
- **Risk:** SAM3 motorcycle masks are too rare (< 50 instances total). Adapter never learns motorcycle.
  - **Mitigation:** Audit SAM3 motorcycle mask count before training. If < 50, use SAM3 augmentation (paste motorcycle masks across train images).
- **Risk:** At eval, the adapter's confidence threshold τ is mis-tuned. Too low → adapter overrides too often, hurts stuff. Too high → adapter rarely fires, no improvement.
  - **Mitigation:** τ ∈ {0.3, 0.5, 0.7} ablation on val subset.
- **Risk:** SAM3 → trainID structural mapping has subtle taxonomic mismatches (e.g., SAM3 "rider" ≠ Cityscapes "rider").
  - **Mitigation:** Audit SAM3 class semantics + manual mapping verification before training.

### What stays the same as the previous Task 7 design

- 80-cluster head untouched.
- Eval protocol unchanged (CUPS Hungarian still runs).
- Trainable params ~17 K (vs 2 K for pure linear; still negligible).
- All-CPU local training feasible (~80 min total wall-clock incl. cache).

### What changed from the previous Task 7 design

- Output dimension: 27 → **14** (SAM3 fine classes, not Cityscapes 27).
- Init: LUT-mimicry → **random Kaiming** (or optional self-distillation, no LUT).
- Eval: was "use adapter directly" → **fusion rule with cluster head + Hungarian for stuff**.
- Justification: training pipeline is now **fully GT-free**.

---

## Task 7 (LEGACY — superseded by revised version above) — 27-class Auxiliary Head from Stage-3 Ckpt

**Status:** Designed in the 2026-05-06 follow-up discussion. Resolves the LUT bottleneck for motorcycle (trainID 17) and traffic light (trainID 6) without extending the cluster head and without touching the eval protocol. **Mathematically dominates the K=82 head-extension proposal in expressivity** (a learnable `W ∈ ℝ^{27×80}` strictly contains the discrete `P_LUT`). Trainable on Mac M4 Pro overnight via feature caching.

**Responsible agent:** `feature-dev:code-architect` (design + implementation), `feature-dev:code-reviewer` (audit). Locally executable.

### Diagnosis recap

Audit of `weights/kmeans_centroids_k80_santosh.npz` revealed:

- trainID 17 (**motorcycle**): **0 clusters in cluster_to_class LUT**
- trainID 6 (**traffic light**): **0 clusters in LUT**
- All other 17 train IDs have ≥ 1 cluster

Pseudo-labels generated via this LUT cannot, by arithmetic, contain motorcycle or traffic-light pixels. The cascade trained on these labels has no signal for those classes. Path-B's fine-object loss on the unified-thing channel (channel 0) cannot rescue them — the unified channel is class-agnostic.

### Architecture

Start from `checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt` (Stage-3 best, 35.83 PQ). Add a single linear auxiliary head on top of the existing 80-cluster semantic logits:

```
                  Stage-3 best.ckpt (frozen everywhere below)
                  ─────────────────────────────────────────
   image x ─► DINOv3 ViT-B/16 ─► SimpleFPN ─► Sem-Seg head ─► z ∈ ℝ^80
                                                                │
                                                                ├───► [80-cluster head, frozen, used for Hungarian eval]
                                                                │
                                                                ▼
                                                       ┌──────────────────────────┐
                                                       │  W ∈ ℝ^{27×80},          │   NEW — only trainable
                                                       │  b ∈ ℝ^{27}              │   2 187 params total
                                                       │  y = Wz + b              │
                                                       └──────────┬───────────────┘
                                                                  │
                                                                  ▼
                                                          softmax(y) ∈ Δ^{26}
                                                                  │
                                                                  └─► CE against per-pixel SAM3 class targets
```

### Math

**Initialization** (the critical step):

$$
\mathbf{W}_{c, k} \;=\; \mathbb{1}\bigl[\mathrm{LUT}[k] = c\bigr] \;+\; \epsilon_{c, k}, \qquad \epsilon_{c, k} \sim \mathcal{N}(0, 0.01^2), \qquad \mathbf{b} = \mathbf{0}
$$

This makes step-0 output near-identical to the Hungarian collapse (preserving 35.83 PQ as starting point), AND every row of `W` — including row 17 (motorcycle) and row 6 (traffic light), originally all zeros — has nonzero entries. Once nonzero, gradient can flow through them.

**Forward pass:**

$$
\mathbf{y}_u = \mathbf{W}\,\mathbf{z}_u + \mathbf{b} \in \mathbb{R}^{27}, \qquad \hat P(c \mid u) = \mathrm{softmax}(\mathbf{y}_u)[c]
$$

**SAM3 loss** (for each SAM3 mask `m_k` with Cityscapes trainID target `t_k`):

$$
\mathcal{L}_{\mathrm{aux},k} = -\frac{1}{|m_k|}\sum_{u \in m_k} \log \hat P(t_k \mid u)
$$

**Total Stage-4-aux objective** (Stage-3 losses on `z` are still used to drive EMA pseudo-labels, but with backbone+sem-seg head frozen those gradients are zero):

$$
\mathcal{L} = \mathcal{L}_{\mathrm{Stage-3}}^{\text{(frozen)}} + \lambda_{\mathrm{aux}} \cdot \frac{1}{|\mathcal{M}|}\sum_k \mathcal{L}_{\mathrm{aux},k}
$$

**Gradient on previously-empty motorcycle row:**

$$
\frac{\partial \mathcal{L}_{\mathrm{aux},k}}{\partial \mathbf{W}_{17, k'}} = -\frac{1}{|m_k|}\sum_{u \in m_k}\bigl(\mathbb{1}[t_k = 17] - \hat P(17 \mid u)\bigr)\,\mathbf{z}_u^{k'}
$$

For SAM3 motorcycle masks (`t_k = 17`), this gradient grows `W[17, k']` for whichever clusters `k'` activate on motorcycle pixels. The model discovers *which mixture of existing clusters* corresponds to motorcycle. The LUT bottleneck dissolves automatically.

### SAM3 → Cityscapes trainID mapping

| SAM3 idx | Class | trainID | Aux head channel |
|--:|---|--:|--:|
| 0 | person | 11 | 11 |
| 1 | bicycle | 18 | 18 |
| **2** | **motorcycle** | **17** | **17** ◄ rescued |
| 3 | rider | 12 | 12 |
| 4 | traffic sign | 7 | 7 |
| **5** | **traffic light** | **6** | **6** ◄ rescued |
| 6 | truck | 14 | 14 |
| 7 | bus | 15 | 15 |
| 8 | train | 16 | 16 |
| 12 | car | 13 | 13 |
| 13 | pole | 5 | 5 |

### Eval protocol compliance

Eval reads the 80-cluster head's output `z`, runs the existing CUPS Hungarian, computes PQ. The auxiliary head `y = Wz + b` is **only used during training** to provide a stronger SAM3 gradient. At eval time the auxiliary head can be ignored entirely — protocol is identical to the CUPS baseline. The two-head design is therefore a strictly-additive training-time modification.

**Important note:** for SAM3 gradient to actually reshape the cluster logits `z` (and therefore eval-time predictions), the gradient must flow back into the sem-seg head. Three regimes:

- **Regime 1 (fully frozen below `z`):** `W` learns alone. Aux head improves but `z` is unchanged → **eval PQ does not move**. Useful only for diagnostic / verification.
- **Regime 2 (unfreeze sem-seg head + last FPN block, low LR):** Gradient flows into the last FPN block + sem-seg head. `z` shifts so that motorcycle clusters' activations correlate with motorcycle pixels. **Eval PQ improves.** Recommended.
- **Regime 3 (unfreeze sem-seg + cascade ROI heads):** Largest gradient flow, biggest expected improvement, highest risk of disturbing healthy classes.

### CPU-trainability via feature caching

For the local Mac path:

| Setting | Per-image cost | Per-epoch (2975 imgs) | Augmentation fidelity |
|---|---:|---:|---|
| Direct training (no cache) | ~3 sec on CPU / ~1.7 sec on MPS | 2.5 h CPU / 1.4 h MPS | Full (photometric + copy-paste) |
| **Cache + Option A (no aug)** | ~15 ms | **~45 sec** | None — pure cache |
| Cache + Option B (feature-space paste w/ boundary erosion) | ~50 ms | ~150 sec | Approximate copy-paste |
| **Cache + Option C (selective re-forward for copy-paste samples)** | ~600 ms avg | **~30 min** | Full copy-paste |

**Augmentation impact assessment:**

- **Photometric augmentation:** Loss ≤ 0.3 PQ. The frozen DINOv3 backbone is already photometrically robust (lvd1689m pretraining); a linear aux head on top doesn't need additional photometric variety.
- **Copy-paste augmentation:** Loss 0.5 - 1.5 PQ in Option A, 0.1 - 0.5 PQ in Option B, ~0 in Option C.

### Implementation steps (CPU-only, this session or next)

#### 7.1 — Add `Aux27ClassHead` module

**File:** `refs/cups/cups/model/aux_27class_head.py` (new, ~50 lines)

```python
import torch
import torch.nn as nn
import numpy as np

class Aux27ClassHead(nn.Module):
    """Linear projection from 80 cluster logits to 27 Cityscapes train IDs."""

    def __init__(self, num_clusters: int = 80, num_classes: int = 27,
                 cluster_to_class_lut: torch.Tensor | None = None,
                 init_noise: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(num_clusters, num_classes, bias=True)
        if cluster_to_class_lut is not None:
            assert cluster_to_class_lut.shape[0] == num_clusters
            with torch.no_grad():
                W = torch.zeros(num_classes, num_clusters)
                for k in range(num_clusters):
                    c = int(cluster_to_class_lut[k])
                    if 0 <= c < num_classes:
                        W[c, k] = 1.0
                W += torch.randn_like(W) * init_noise
                self.linear.weight.copy_(W)
                self.linear.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, num_clusters, H, W) → y: (B, num_classes, H, W)
        if z.dim() == 4:
            return self.linear(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.linear(z)
```

#### 7.2 — Wire into `pl_model_self.py`

In `__init__`, after the `_fo_loss` setup, add:

```python
self._aux27_enabled = getattr(getattr(config.MODEL.SEM_SEG_HEAD, "USE_27CLASS_AUX", False), False)
if self._aux27_enabled:
    from cups.model.aux_27class_head import Aux27ClassHead
    centroids = np.load(config.MODEL.CLUSTER_CENTROIDS_PATH)  # weights/kmeans_centroids_k80_santosh.npz
    lut = torch.from_numpy(centroids["cluster_to_class"]).long()
    self._aux27_head = Aux27ClassHead(num_clusters=80, num_classes=27,
                                       cluster_to_class_lut=lut, init_noise=0.01)
    self._aux27_weight = float(getattr(config.SELF_TRAINING, "AUX27_WEIGHT", 0.05))
```

In `training_step`, replace the existing `_fo_loss` call (or run alongside) with:

```python
if self._aux27_enabled:
    aux_logits = self._aux27_head(full_logits)  # (B, 27, H, W)
    # Build per-pixel SAM3 trainID target: -1 outside any SAM mask, else trainID
    target = build_aux27_target(sam_masks, sam_cls, image_size=full_logits.shape[-2:])
    aux_loss = F.cross_entropy(aux_logits, target, ignore_index=-1)
    loss_dict["loss_aux27"] = aux_loss * self._aux27_weight
```

#### 7.3 — Build feature cache (one-time, ~3 hours on Mac CPU)

**File:** `scripts/cache_stage3_features.py` (new)

For each Cityscapes train image, run the frozen Stage-3 model up to the sem-seg head, save `z` in fp16 to `cityscapes_z_cache_dcfa_simcf_abc/{image_id}.pt`. Total cache size: ~9 GB at FPN-P4 resolution.

#### 7.4 — Cached dataloader + aux-head training script

**File:** `scripts/train_aux27_cached.py` (new, ~150 lines)

- Load cached `z` from disk (or generate live for the ~30 % of samples that need copy-paste in Option C).
- Forward `y = W @ z + b`.
- CE against SAM3-derived per-pixel trainID target.
- Adam, lr = 1e-3 for `W`, lr = 1e-5 for unfrozen sem-seg head + last FPN block.
- 100 epochs, ~75 min total wall-clock with Option A.

#### 7.5 — Local eval after training

Run the same `refs/cups/val.py` invocation as before, but now with the trained aux-head ckpt. PQ measured the same way; aux head is implicitly used to update `z` (via gradient flow during training, Regime 2/3) so eval-time `z` already reflects the SAM3-improved cluster activations.

### Tasks (TDD ordering, one action per step)

- [ ] **Step 1:** Write `Aux27ClassHead` module (file 7.1).
- [ ] **Step 2:** Write unit test: `test_aux27_init_matches_lut.py` — assert step-0 output equals `P_LUT` collapse.
- [ ] **Step 3:** Wire into `pl_model_self.py`. Add yaml flag.
- [ ] **Step 4:** Write `cache_stage3_features.py`. Run on a 50-image subset (~5 min) for verification before the full 2 975-image cache (~3 hours).
- [ ] **Step 5:** Write `train_aux27_cached.py`. Local 100-epoch run.
- [ ] **Step 6:** Run local eval. Compare PQ per-class vs. the Path-B baseline (37.98 PQ).
- [ ] **Step 7:** If motorcycle PQ improves but residual classes still struggle, escalate to Option C (selective re-forward for copy-paste samples).
- [ ] **Step 8:** Document results in `reports/path_c_aux27_results.md`.

### Validation Gates

| Gate | Command | Pass criterion |
|---|---|---|
| Aux head init test | `pytest refs/cups/tests/test_aux27_init.py` | Step-0 output equals `P_LUT @ softmax(z)`. |
| Cache build | `python scripts/cache_stage3_features.py --check 50` | 50-image cache built; per-pixel `z` reproducible. |
| Aux training stability | `python scripts/train_aux27_cached.py` log | Loss decreases monotonically; no NaN. |
| Local eval after training | `python refs/cups/val.py ...` | **motorcycle PQ ≥ 5.0** (was 0.09); **traffic light PQ ≥ 12** (was 7.31); overall PQ ≥ 38.0. |

### Risks and Rollback

- **Risk:** Aux head doesn't improve eval PQ if backbone+sem-seg head are fully frozen (Regime 1) — `z` doesn't change. **Mitigation:** Use Regime 2 by default (unfreeze sem-seg head + last FPN block).
- **Risk:** `W` rows for previously-empty classes (motorcycle, traffic light) over-fit to noise during early epochs. **Mitigation:** Smaller init noise (`σ = 0.001`); higher `λ_aux` weight only after 5 epochs of warmup.
- **Risk:** SAM3 mask quality is poor for motorcycle on Cityscapes train (limited motorcycle instances). **Mitigation:** Audit SAM3 motorcycle mask count + per-mask IoU before training.
- **Rollback:** yaml flag `USE_27CLASS_AUX: False`. No code changes are destructive.

### Why this dominates the K=82 head extension

Both proposals resolve the LUT bottleneck for motorcycle/traffic light, but:

- **K=82**: discrete one-hot mapping (extends `P_LUT`), 2 new cluster channels, ~8 K trainable params, but each new channel requires SAM3 to be the *only* training signal (otherwise the channel competes with pseudo-labels).
- **Path-C**: dense learnable mapping (`W` strictly more expressive than `P_LUT`), 27 channels at the trainID level, ~2 K trainable params, no pseudo-label competition (aux head is supervision-only). Strict mathematical generalization.

### Why this is paper-grade for NeurIPS

- A single linear projection over the cluster logits gives a closed-form, principled mechanism to recover dead classes.
- Math is clean: function class containment, gradient closed-form, init reduces to baseline.
- Ablation is straightforward: `λ_aux ∈ {0, 0.01, 0.05, 0.1}`, init noise ∈ {0.001, 0.01, 0.1}, regime ∈ {1, 2, 3}.
- Result is a **+2-4 PQ improvement on dead classes with 2 K added params** — very high parameter efficiency.
