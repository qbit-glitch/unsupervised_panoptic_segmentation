---
type: writing
title: "BMVC 2026 — Reviewer Response & Open Issues"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, reviewer-response, control-experiment]
updated: 2026-04-13
---

# Reviewer Response & Open Issues

## Critical Blocker: E1 Control Experiment

### Reviewer Concern

Both BMVC reviewers flagged the same issue: the +5.0 PQ improvement could be entirely from the **backbone upgrade** (CUPS uses ResNet-50 → we use DINOv3 ViT-B/16), not from better pseudo-labels.

### The Experiment

**E1**: Train DINOv3 backbone on CUPS pipeline pseudo-labels. This isolates backbone contribution from pseudo-label contribution.

### Decision Matrix

| CUPS+DINOv3 PQ | Interpretation | Paper Action |
|----------------|---------------|-------------|
| > 33 PQ | Backbone explains everything | Pivot to analysis paper |
| 30-33 PQ | Mixed contribution | Reframe: "better PLs + better backbone" |
| < 30 PQ | Our pseudo-labels genuinely matter | Strengthen method claims |

Our method's PQ = 30.255% (step 1800) / 32.76% (step 8000).

### Pipeline Status (as of 2026-04-12)

- **Script**: `mbps_pytorch/gen_cups_pseudo_labels_remote.py` (two-pass architecture)
- **Remote**: santosh@172.17.254.146, GTX 1080 Ti 11GB, conda env `cups`

| Phase | Status | Details |
|-------|--------|---------|
| Pass 1 (RAFT-SMURF + SF2SE3 → instances) | COMPLETE | 2975 .pt files, 351 SF2SE3 failures |
| Pass 2 (DepthG + CRF → semantics, merge) | RUNNING | ~12.5s/img, ETA ~10h (started 2026-04-12) |
| Copy PLs to local / training machine | PENDING | — |
| Stage-2 training (DINOv3 on CUPS PLs) | PENDING | Config ready: `train_cityscapes_dinov3_vitb_cups_official_mps.yaml` |
| Stage-3 self-training | PENDING | Config ready: `train_self_cityscapes_dinov3_vitb_cups_official_mps.yaml` |
| Evaluate and compare | PENDING | — |

### Key Fixes Applied During Pipeline

1. `valid_pixels` double-scaling → create at original 1024x2048, scale once
2. DepthG OOM (32 crops at once) → batched 4 crops + autocast
3. `multiprocessing Pool + CUDA` spawn hang → sequential dense_crf (no Pool)
4. SF2SE3 Bool tensor type mismatch → bitwise AND fix

### Training Configs

- **Stage-2**: `refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_mps.yaml`
  - bs=1, accum=8, 8000 steps, MPS
- **Stage-3**: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_cups_official_mps.yaml`
  - 3 rounds x 500 steps
- ROOT_PSEUDO must point to `cups_pseudo_labels_pipeline/` after copy

### Log Location

- Pass 2: `/home/santosh/cups_pipeline_pass2v3.log`, PID 5024

---

## Other Reviewer Concerns

### Concern: "No algorithmic novelty"

**Response strategy**: Honest framing. This is an empirical pipeline paper, not an algorithm paper. Each component (CAUSE-TR, DA3, CUPS protocol) is prior work. The contribution is the composition + analysis.

**Mitigation**: Strong ablation coverage (15 instance methods, 4 depth models, 2 backbones, k-sweep) demonstrates that the specific composition choices matter and were carefully validated.

### Concern: "Single seed"

**Acknowledged in Limitations**: "All results from single seed; Stage-2 margin over CUPS (+0.07 PQ) is within noise."

**If E1 goes well**: could run additional seeds (42, 123, 456) to demonstrate significance. Requires ~3x compute budget.

### Concern: "PQ_stuff below CUPS"

**Root cause**: 7 non-standard CAUSE classes where model achieves PQ < 1%. Our frozen semantic pipeline cannot recover classes that CAUSE-TR's own features don't distinguish.

**Data**: On the 16 classes with PQ > 10%, our average PQ is 49.0%.

### Concern: "Monocular" claim when DA3 uses multi-view pretraining

**Addressed in paper**: "We note that while our method requires only monocular input, DAv3 was itself pretrained with multi-view geometry — the 'monocular' characterization refers to deployment requirements, not to foundation model pretraining."

---

## Paper Weaknesses (Self-Assessment)

### Acknowledged

1. Co-planarity failure mode (person PQ=4.2%)
2. 7 zero-PQ CAUSE classes → PQ_stuff ceiling
3. Single seed
4. Self-training analysis with only 2 backbone configs

### Not Acknowledged (Potential Reviewer Pushback)

1. **Fairness of backbone comparison**: CUPS uses RN50, we use DINOv3 ViT-B/16 (86M vs 25M params). Stage-2 only +0.07 PQ suggests backbone isn't the full story, but Stage-3 amplifies the gap.
2. **No COCO results**: Single-dataset evaluation limits generalizability claims.
3. **Reliance on frozen FMs**: If foundation models improve further, the entire pipeline numbers change. This is a feature (composability) but also a weakness (no learning).
4. **Cross-dataset "generalization" is really domain transfer**: Mapillary PQ=39.85% exceeding Cityscapes is an artifact of class mapping, not genuine generalization.

---

## Timeline to Resolution

1. Check Pass 2 completion on remote → ~April 13
2. Copy pseudo-labels to training machine → same day
3. Run Stage-2 training (8000 steps) → ~2-3 days on 1080 Ti
4. Run Stage-3 self-training → ~1-2 days
5. Evaluate → same day
6. Update paper numbers and conclusion → 1 day

**Estimated E1 completion**: ~April 18-19, 2026

---

## Links

- [[BMVC2026-Paper-Overview]]
- [[BMVC2026-Experiments]] — Table 2c (backbone ablation)
- [[paper-discussion-2026-04-06]] — Discussion of contribution framing
- [[Knowledge/Key-Lessons]] — Metric pitfalls to avoid in rebuttal
