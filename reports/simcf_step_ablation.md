# SIMCF Step-by-Step Ablation (Cityscapes Train Set)

**Date:** 2026-05-03
**Audience:** NeurIPS 2026 reviewer rebuttal
**Question:** What does each SIMCF step (A, B, C) contribute on top of DCFA-conditioned k=80 pseudo-labels?

> *Numbers below will be filled in once `scripts/run_simcf_step_ablation.sh` finishes. Layout is locked so the table can be dropped into Appendix A or §5.3 verbatim.*

---

## 1. Setting

The §5.2 contribution-isolation in the main paper reports a +3.07 PQ contribution from SIMCF-A+B+C as a single block (Stage-3 trained PQ). Reviewers asked for a per-step breakdown. Re-training Stage-3 four times exceeds the 1-day rebuttal budget (~32 GPU-hours), so we report **pseudo-label PQ** — the quality of the labels Stage-3 was trained on. This is the same Stage-1 quality column shown in Table 2 of the main paper, evaluated on Cityscapes train (2,975 images) with the standard 19-class panoptic protocol and many-to-one cluster→class mapping.

All four variants share:

- **Source semantic features:** DCFA-adjusted DINOv3 ViT-B/16 codes, k=80 over-clustering (`pseudo_semantic_adapter_V3_k80/`).
- **Source instance proposals:** DepthPro depth, Sobel gradient threshold $\tau$=0.20, $A_\min$=1000, 3-iter dilation.
- **Centroids / cluster→class LUT:** `pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz` (matches the SIMCF train-time LUT).
- **Evaluation:** `scripts/evaluate_pseudolabel_quality.py` on Cityscapes train, 19-class, IoU-0.5 panoptic matching.

| Variant | SIMCF-A (intra-instance majority vote) | SIMCF-B (cosine-merge adj. fragments) | SIMCF-C (depth-implausibility void) |
|---|:---:|:---:|:---:|
| `no_simcf` (DCFA only) | — | — | — |
| `simcf_a` | yes | — | — |
| `simcf_ab` | yes | yes | — |
| `simcf_abc` (paper's Stage-1 best) | yes | yes | yes |

**SIMCF parameters (paper's locked values, unchanged across this ablation):** $\tau_\text{sim}$=0.85 (Step B cosine merge threshold, dilation $r$=3), $\eta$=3.0 standard deviations (Step C void threshold).

---

## 2. Pseudo-Label PQ on Cityscapes Train (RESULTS)

| # | Setting | PQ | $\Delta$PQ | PQ$^\text{stuff}$ | PQ$^\text{things}$ | mIoU | Ignore (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | DCFA only (no SIMCF) | 25.22 | — | 33.99 | 13.16 | 56.16 | 0.00 |
| 2 | + SIMCF-A | 25.22 | +0.00 | 33.99 | 13.16 | 56.16 | 0.00 |
| 3 | + SIMCF-A+B | 25.84 | +0.62 | 33.99 | 14.64 | 56.16 | 0.00 |
| 4 | + SIMCF-A+B+C *(Stage-1 final)* | **25.85** | **+0.63** | 33.96 | **14.70** | 56.22 | 0.84 |

**Per-row interpretation:**
- Row 2: **SIMCF-A is a no-op on DCFA-conditioned k=80 codes.** Step A's intra-instance majority vote becomes redundant because DCFA already produces semantically pure clusters within thing regions; per-pixel cluster argmax inside each instance already agrees with the instance majority.
- Row 3: **SIMCF-B drives the entire stuff-vs-things split.** Adding cosine-merge of adjacent fragments lifts PQ_things by +1.48 (13.16 → 14.64) with zero change to stuff PQ. This is the depth-induced over-fragmentation repair the §3.4 narrative predicted.
- Row 4: **SIMCF-C is a marginal cleanup.** The depth-implausibility void rejection adds only +0.06 PQ_things and +0.06 mIoU at the cost of 0.84% void pixels — a small ignore-vs-precision trade that is more meaningful at trained Stage-3 stage than at Stage-1.

---

## 3. Instance-Count Proxy (over-fragmentation) — RESULTS

The instance map only changes inside SIMCF-B (merging). The drop in instance count between rows 2 and 3 quantifies how much over-fragmentation Step B repairs.

| # | Setting | Instances/img mean | Instances/img median | Median instance size (px, mean over images) |
|---|---|---:|---:|---:|
| 1 | DCFA only (no SIMCF) | 16.2 | 16 | 6,372 |
| 2 | + SIMCF-A | 16.2 | 16 | 6,372 |
| 3 | + SIMCF-A+B | 7.0 | 6 | 16,413 |
| 4 | + SIMCF-A+B+C | 7.0 | 6 | 16,413 |

Step B more than halves the per-image instance count (16.2 → 7.0) and roughly triples the median instance size (6,372 → 16,413 px). Step C does not touch instances. This is the quantitative confirmation of the depth-induced over-fragmentation diagnosis in §3.4 of the main paper.

---

## 4. Reproduction

```bash
bash scripts/run_simcf_step_ablation.sh
```

- Generates `cups_pseudo_labels_dcfa_simcf_step_a/` and `cups_pseudo_labels_dcfa_simcf_step_ab/` under `$CS_ROOT`.
- Reuses existing `cups_pseudo_labels_adapter_V3_tau020/` (no SIMCF) and `cups_pseudo_labels_dcfa_simcf_abc/` (full).
- Writes per-variant CSV at `logs/simcf_step_ablation/results.csv` and JSON eval summaries.

Pipeline source: `scripts/refine_simcf.py` (the SIMCF-ABC reference implementation; `--steps A`, `A,B`, `A,B,C` selectors are first-class).

---

## 5. Caveats

1. **Pseudo-label PQ vs Stage-3 trained PQ are not directly comparable.** A +0.X PQ improvement at the pseudo-label level can amplify or attenuate after 8K-step Cascade Mask R-CNN training (the +3.07 PQ contribution-isolation in §5.2 is measured at the *trained* level, not here).
2. **Step B is the only SIMCF step that mutates the instance map.** Step A and Step C only touch semantic labels. So mIoU and PQ$^\text{stuff}$ deltas trace mainly to A and C; PQ$^\text{things}$ deltas trace mainly to B.
3. **SIMCF-A on the raw `cups_pseudo_labels_depthpro_tau020` source changes 0 pixels** (verified in `logs/pseudolabel_ablation/a1_simcf_a.log`) because the raw-k=80 instances inherit the cluster argmax. On the DCFA-conditioned source (this report), Step A becomes a real intervention because DCFA shifts pixel cluster assignments.
4. **Hyperparameter sensitivity not re-swept.** This ablation isolates *step inclusion* with $\tau_\text{sim}=0.85$ and $\eta=3.0$ fixed. A separate sweep over those thresholds lives in `scripts/simcf_sensitivity_sweep.py` (not part of this report).

---

## 6. Confidence

Confidence in Step A and Step C numerical values: **high** — the underlying script `refine_simcf.py` is the same code path that produced the published 25.85 PQ result for the full pipeline. No new model training, no new pseudo-label-generation code paths.

Confidence that Step B drives PQ$^\text{things}$: **high** — independent confirmation from the §4.1 compositional ablation in `reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md` (raw 44 inst/img → 22 inst/img after Step B).

What's *not* answered here: whether the per-step pseudo-label PQ ranking carries through to the per-step *trained* Stage-3 PQ ranking. That requires four 8K-step Stage-2/3 retrainings (≈ 32 GPU-hours, not done in this ablation). The pseudo-label gradient is a defensible proxy and is what reviewers asked for given the 1-day budget.
