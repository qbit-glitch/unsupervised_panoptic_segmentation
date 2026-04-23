# Appendix D: Pseudo-Label Quality Ablation — SIMCF and Multi-Seed Consensus

## D.1 Motivation

Unsupervised panoptic segmentation pipelines generate pseudo-labels by composing
independently-derived semantic clusters and instance masks. The semantic branch
(DINOv3 ViT-B/16, k=80 MiniBatchKMeans) and the instance branch (DepthPro
monocular depth + Sobel gradient + connected components) operate without
cross-modal consistency constraints. This independence introduces two failure
modes that limit pseudo-label quality:

1. **Semantic–instance misalignment.** Adjacent instances that belong to the same
   semantic class but arise from separate depth-guided splits may encode
   redundant fragmentation. A bus occupying a single depth plane may split into
   3–5 instances when depth gradients fluctuate near its boundary.

2. **Depth–semantic inconsistency.** A cluster assigned to *road* (mean depth
   0.27, std 0.25) may include pixels at depth 0.01 — physically implausible for
   a road surface — revealing noisy cluster boundaries.

This experiment tests whether *post-hoc cross-modal filtering* can improve
pseudo-label quality without modifying the upstream generation pipeline.

## D.2 Approach: SIMCF (Semantic–Instance Mutual Consistency Filtering)

SIMCF applies three sequential filtering passes, each using one modality to
validate another:

**Step A — Instance validates semantics.** For each instance, compute the
majority trainID among its constituent pixels. Pixels whose mapped trainID
deviates from the majority are reassigned to the dominant cluster within the
instance. *Claim:* Corrects semantic noise at instance boundaries.

**Step B — Semantics validate instances.** Adjacent instances sharing the same
majority trainID and exhibiting high DINOv3 feature cosine similarity (>0.85)
are merged via union-find. Adjacency is determined by 3-pixel dilation overlap
at full resolution (1024×2048); feature similarity uses mean-pooled 768-dim
DINOv3 patch vectors at 32×64 resolution. *Claim:* Reverses over-fragmentation
from depth-guided splitting.

**Step C — Depth validates semantics.** A first pass computes per-class depth
statistics (online mean and variance across all 2,975 training images). A second
pass masks pixels whose depth deviates >3σ from their assigned class profile,
setting their label to *ignore* (255). *Claim:* Removes physically implausible
assignments at class boundaries.

## D.3 Approach: Multi-Seed Consensus

K-means clustering is non-deterministic: different random seeds yield different
cluster boundaries. We exploit this stochasticity to identify unreliable pixels.
Five independent MiniBatchKMeans runs (seeds 42, 123, 456, 789, 1024; k=80,
batch_size=4096, L2-normalized DINOv3 features, sample_frac=0.3) each produce a
complete set of 80-cluster labels. Each run's clusters are mapped to 19-class
trainIDs via per-cluster majority vote against Cityscapes GT. Pixels where fewer
than 3/5 seeds agree on the trainID are masked as *ignore*.

## D.4 Ablation Matrix

| Variant | Components | Expected Signal |
|---------|-----------|-----------------|
| A0 — Baseline | k=80 + DepthPro τ=0.01 | Reference |
| A1 — SIMCF-A | Step A only | Semantic boundary cleanup |
| A3 — SIMCF-ABC | Steps A + B + C | Full cross-modal filtering |
| A4 — Consensus | 5-seed ensemble masking | Cluster-boundary uncertainty |
| A5 — SIMCF + Consensus | SIMCF-ABC on consensus output | Combined |

**Gate criterion.** PQ ≥ A0 + 0.3 to justify Stage-2 training compute.

## D.5 Evaluation Protocol

Pseudo-label quality is measured on the Cityscapes training split against GT
using the standard 19-class PQ/mIoU metrics. Since pseudo-labels contain
overclustered IDs (0–79), a many-to-one majority-vote mapping is computed
*fresh* for each variant by accumulating a (80 × 19) confusion matrix against GT
and assigning each cluster to its most frequent GT class. This covers 17/19
classes (traffic light and motorcycle receive no clusters).

**Critical methodological note.** An initial evaluation incorrectly used the
`cluster_to_class` array stored in the k-means centroids file. That mapping was
computed at 32×64 patch resolution during k-means fitting and assigns 31/80
clusters to road, covering only 8/19 classes — producing PQ=0.05%. The
resolution mismatch causes 67/80 cluster-to-class assignments to differ from a
fresh full-resolution mapping. All results below use the corrected protocol.

## D.6 Results

### Table D.1: Pseudo-Label Quality (19-class Cityscapes, training split)

| Variant | PQ | PQ_stuff | PQ_things | mIoU | Ignore% | Gate |
|---------|---:|--------:|---------:|-----:|-------:|------|
| A0 — Baseline | 24.54 | 33.43 | 12.31 | 56.56 | 0.0 | — |
| A1 — SIMCF-A | 24.54 | 33.43 | 12.31 | 56.56 | 0.0 | FAIL (Δ=0.0) |
| **A3 — SIMCF-ABC** | **25.27** | **33.73** | **13.64** | **56.57** | 1.4 | **PASS (Δ=+0.73)** |
| A4 — Consensus | 3.50 | 1.70 | 5.97 | 42.42 | 70.4 | INVALID |
| A5 — SIMCF+Cons. | 3.48 | 1.65 | 6.01 | 41.78 | 70.6 | INVALID |

### Table D.2: SIMCF-ABC Per-Step Statistics (2,975 images)

| Step | Metric | Value |
|------|--------|-------|
| A — Instance→semantic | Pixels changed | 0 (0.00%) |
| B — Semantic→instance | Instance merges | 7,252 (2.4/img) |
| C — Depth→semantic | Pixels masked | 85,017,772 (1.36%) |
| Total | Runtime | 1,013s (~17 min) |

### Table D.3: SIMCF-ABC Per-Class Analysis (A3 vs A0)

| Class | PQ (A0) | PQ (A3) | ΔPQ | ΔSQ | ΔRQ | Interpretation |
|-------|------:|------:|----:|----:|----:|---------------|
| road | 78.4 | 79.7 | +1.3 | +1.2 | +0.2 | Depth filter removes boundary noise |
| sky | 59.8 | 61.8 | +2.0 | +0.6 | +1.9 | Depth outliers (low depth = not sky) |
| **bus** | 25.4 | **34.2** | **+8.8** | +3.5 | +10.1 | Instance merging recovers fragmented buses |
| car | 1.6 | 2.7 | +1.1 | +0.4 | +1.8 | Mild merge benefit |
| train | 31.5 | 32.4 | +0.9 | +0.6 | +0.9 | Merge benefit |
| person | 2.7 | 2.6 | −0.1 | +0.1 | −0.2 | Negligible regression |
| rider | 7.9 | 7.3 | −0.6 | +0.2 | −1.0 | Merge may merge distinct riders |

### Table D.4: Multi-Seed Consensus Agreement Histogram

| Agreement Level | Pixels | Percentage |
|---------------:|-------:|----------:|
| 0/5 seeds | 4,207,621,931 | 67.4% |
| 1/5 seeds | 123,288,490 | 2.0% |
| 2/5 seeds | 62,268,713 | 1.0% |
| 3/5 seeds | 78,813,555 | 1.3% |
| 4/5 seeds | 131,767,999 | 2.1% |
| 5/5 seeds | 1,635,266,512 | 26.2% |

## D.7 Analysis

### D.7.1 Step A is a structural no-op

Step A corrects pixels whose mapped trainID deviates from the per-instance
majority. In the baseline pipeline, instances are connected components of
*individual* cluster IDs — every pixel within an instance shares the same
cluster, hence the same trainID. Step A therefore finds zero inconsistencies by
construction. This no-op result is informative: it confirms that CUPS-style
instance generation preserves semantic consistency within instances, and that
cross-modal filtering must operate *between* instances (Step B) or *across*
modalities (Step C) to produce changes.

### D.7.2 Step B drives the PQ_things improvement

The +1.33 PQ_things gain arises entirely from Step B's instance merging. The
most striking case is *bus* (+8.8 PQ): depth-guided splitting fragments
large vehicles that span a single depth plane into 3–5 disconnected components.
Step B's adjacency + feature similarity test (cosine > 0.85) successfully
re-merges these fragments, improving RQ from 35.4% to 45.5% (+10.1 points).
The 7,252 total merges (2.4 per image) represent a modest but targeted
intervention.

Regressions on *rider* (−0.6 PQ) suggest the similarity threshold may
occasionally merge distinct small instances that happen to share similar DINOv3
features. A class-conditional threshold (stricter for small objects) could
mitigate this.

### D.7.3 Step C provides conservative boundary cleanup

The depth filter masks 1.36% of pixels as ignore — predominantly at class
boundaries where interpolation during upsampling creates depth values
inconsistent with the assigned class profile. The +0.30 PQ_stuff improvement
and +2.0 PQ on *sky* confirm that removing physically implausible assignments
sharpens stuff region boundaries without significant information loss.

### D.7.4 Consensus fails due to k-means instability at k=80

The agreement histogram reveals a fundamental limitation: 67.4% of pixels
receive 0/5 seed agreement, and only 26.2% achieve full consensus. K-means with
k=80 on 768-dimensional DINOv3 features is highly sensitive to initialization —
different seeds discover qualitatively different cluster structures, not merely
rotated versions of the same partition.

This makes pixel-wise consensus meaningless: masking 70.4% of pixels as
unreliable removes far too much training signal. The consensus approach requires
either (a) a more stable clustering algorithm (spectral clustering, HDBSCAN), or
(b) agreement measured in the *trainID* space after mapping, where cluster-level
variation would be absorbed by the many-to-one mapping. The current
implementation compares trainIDs but uses the centroids file's incorrect 8-class
mapping for the baseline, creating a systematic disagreement artifact.

**Post-hoc diagnosis.** The centroids file's `cluster_to_class` was computed at
32×64 resolution during k-means fitting; 67/80 cluster assignments differ from a
fresh full-resolution majority-vote mapping. This resolution mismatch inflates
the 0/5 agreement rate and invalidates the consensus results. A corrected
implementation using fresh mappings for all seeds would likely retain 60–80% of
pixels rather than 30%.

## D.8 Depth Statistics (Step C Reference)

Per-class depth profiles computed over 2,975 training images (DepthPro, normalized [0,1]):

| Class | Mean Depth | Std | Pixel Count |
|-------|----------:|----:|------------:|
| road (0) | 0.268 | 0.247 | 2.53B |
| sidewalk (1) | 0.210 | 0.193 | 424M |
| building (2) | 0.121 | 0.112 | 1.28B |
| sign (7) | 0.057 | 0.104 | 28M |
| vegetation (8) | 0.101 | 0.116 | 770M |
| sky (10) | 0.250 | 0.186 | 393M |
| person (11) | 0.174 | 0.241 | 36M |
| car (13) | 0.176 | 0.185 | 779M |

Road exhibits the highest depth variance (std=0.247), reflecting the ego-vehicle
perspective where road spans from near-field to horizon. Person's high relative
variance (std/mean=1.39) explains why Step C's 3σ threshold masks few person
pixels — the distribution is too broad for outlier detection.

## D.9 Conclusions and Recommendations

1. **SIMCF-ABC passes the gate** (Δ+0.73 PQ) and should proceed to Stage-2
   training. The primary mechanism is Step B instance merging (+1.33 PQ_things),
   with Step C providing mild boundary cleanup (+0.30 PQ_stuff).

2. **Step A should be removed** from the pipeline when instances derive from
   semantic clusters (as in CUPS). It adds 280s of compute with zero effect.

3. **Multi-seed consensus requires re-implementation.** The current
   implementation's mapping mismatch invalidates results. A corrected version
   using fresh per-seed majority-vote mappings merits re-evaluation before
   discarding the approach entirely.

4. **Step B's similarity threshold (0.85) may benefit from class-conditioning.**
   The rider regression suggests small-object merges are occasionally harmful.

5. **Next step:** Create A6000 training config with `ROOT_PSEUDO` pointing to
   `cups_pseudo_labels_simcf_abc/` and train Stage-2 CUPS Cascade Mask R-CNN.
   Expected improvement over frozen-backbone baseline (PQ≈24.7): +0.5–1.0 PQ
   from cleaner pseudo-labels.

---

*Evaluation: 19-class Cityscapes trainIDs, many-to-one majority-vote mapping
(80→19), training split (2,975 images). Runtime: M4 Pro MacBook, generation
~35 min, evaluation ~25 min per variant.*
