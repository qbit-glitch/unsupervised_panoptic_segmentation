# GA-UniAP: Geometric Agglomerative Pooling for Single-Stage Unsupervised Panoptic Segmentation

**Date:** 2026-06-26
**Status:** Design — pending user review
**Branch target:** new branch off current work (e.g. `feat/ga-uniap-geometric-pooling`)

---

## 1. Motivation & Contribution

### What we're adopting
S2-UniSeg (Xu et al., arXiv 2508.06995, Aug 2025) eliminates the two pain points of the
CUPS / U2Seg / UnSAM paradigm — slow **offline pseudo-mask generation** and **discontinuous
multi-round self-training** — by folding both into one continuous online teacher–student loop.
The enabler is **UniAP**: a fast (~45 ms/img) nonparametric agglomerative pooling that the EMA
teacher runs *per image, every iteration* to produce pseudo-masks. MBPS's current pipeline is
exactly the two-stage paradigm S2-UniSeg criticizes (offline DepthG+CAUSE+depth-guided pseudo-labels
→ EoMT/Cascade training with self-training rounds).

### What is NOT the contribution
"Single-stage" is **already owned by S2-UniSeg.** We are not claiming it. Neither stage is
"eliminated" in the literal sense the idea was first phrased: pseudo-label generation moves *inside*
the loop (UniAP is still a pseudo-labeler), and the student segmentation network is still trained
(it is the deployable model). Single-stage = **fused**, not removed.

### What IS the contribution
UniAP merges nodes by **2D DINO appearance adjacency** — one cosine dot product
(`FastUniAP.py:30, 148-152`). MBPS's identity is **monocular depth / 3D geometry**. The contribution
is to make the online merge criterion operate in **3D scene-layout space** (depth-derived surface
normals + height-above-ground) instead of image-appearance space:

```
S_ij = w_f · cos(f_i, f_j) + w_n · (n_i · n_j) + w_h · (ĥ_i · ĥ_j)
```

This preserves the depth-guided contribution that the MBPS paper is built on, while inheriting the
clean continuous-optimization framing for free. **The paper rests entirely on GA-UniAP beating
vanilla UniAP** — an empirics-carry-it situation consistent with the prior geometric-affinity
novelty verdict (partially-novel / incremental; novelty only on role + setting, empirics decide).

### Honest risk (stated, not buried)
Prior MBPS probes established that geometry separates vertical-vs-ground and different-height objects
but **does not separate same-depth crowds** (only motion does; equal-depth slice AUROC: geom 0.80,
depth 0.54, but crowd pedestrians co-planar). So GA-UniAP's expected win lands on **stuff / semantic
boundaries and non-crowd instances** — where MBPS is *already* strong (PQ_stuff ≈ 35). It likely will
**not** move the person/crowd PQ_things ceiling. Phase 0 measures exactly where it helps. A negative
result ("geometry does not help inside agglomerative pooling on Cityscapes") is still a publishable
finding and costs ~1 day.

---

## 2. Substrate & Reuse Inventory (all verified present)

| Piece | Path | Role |
|-------|------|------|
| S2-UniSeg + UniAP (cloned) | `test-instance-labels/S2-UniSeg/{S2UniSeg,FastUniAP}.py` | Loop + pooling to inject into |
| Surface-normal / height code | `mbps_pytorch/premise_check_geometry_affinity.py` | Per-node geometry (validated, AUROC 0.80) |
| Global-Hungarian eval harness | `scripts/reeval19/` | Pseudo-mask PQ (CUPS protocol) |
| Depth (precomputed, DepthPro) | DepthPro maps used by current k27 pipeline | Input channel, NOT a stage |
| Backbones / refs | `refs/{dino,eomt,cups,depthg,spidepth,zoedepth}` | Features, student decoder |

**The injection point is one function.** Each UniAP node currently carries `normalized_feature`. We
add `normal` (3-vec) and `height` (scalar) per node, and replace the single dot product with the
weighted sum above, in `aggo_merge` and `aggo_merge_graph`. Estimated diff: ~30–50 lines.

---

## 3. Phase 0 — Kill-Gate (the thing we build first)

**Goal:** decide whether the geometric signal helps *inside an agglomerative segmenter* (not just as
a pairwise correlation, which the premise-check already confirmed). Cheapest way to be wrong: ~1 day,
no training, runs **locally on Mac CPU** (per project eval-locally rule; UniAP is CPU-fast).

### 3.1 Inputs (per image, N = 50–150 Cityscapes val)
- DINO **ViT-B/8** patch features (S2-UniSeg's backbone — keep apples-to-apples; only the affinity
  changes between variants).
- Precomputed **DepthPro** depth map.
- **Geometry** (reuse `premise_check_geometry_affinity.py`): back-project depth with Cityscapes
  intrinsics → XYZ point cloud → per-pixel **surface normal** (cross-product of spatial gradients) and
  **height-above-ground** (camera-frame Y, camera height 1.22 m). Average-pool normals/height down to
  the DINO feature grid so every UniAP node has `(feature, normal, height)`.

### 3.2 Affinity variants to ablate (user decision: ablate ALL)
| ID | Name | Affinity | Note |
|----|------|----------|------|
| V0 | Vanilla UniAP | `cos(f_i,f_j)` | Baseline = the number to beat |
| V1 | Augment | `w_f·cos + w_n·(n·n) + w_h·(ĥ·ĥ)` | Geometry added to appearance |
| V2 | Split by task | feat-only for **semantic** pooling; geometry-heavy for **instance** pooling | Matches UniAP's two pooling modes |
| V3 | Hard-replace | geometry only (`w_f=0`) | Pure scene-space pooling |

Weight sweep for V1/V2: initialize from the premise-check finding (**normals load-bearing, height
noisy under mono depth** → `w_n > w_h`). Coarse grid `w_n, w_h ∈ {0.3, 0.5, 0.7}`, `w_f` fixed.
Keep the grid small; this is a kill-gate, not a tuning marathon.

### 3.3 Mask → panoptic → metrics
- UniAP **instance pooling** → thing masks; **semantic pooling** → stuff/semantic masks (unchanged
  from S2-UniSeg).
- Score with the existing **global-Hungarian** harness (`scripts/reeval19/`): pseudo-segments matched
  to Cityscapes GT, report **PQ, PQ_stuff, PQ_things, SQ, RQ, instance recall@IoU0.5** (per-class,
  with person called out explicitly).
- **Reference points** on the same images: vanilla UniAP (V0) and the current k27/depth pipeline
  (~26–30 PQ).

### 3.4 Gate criterion
A geometric variant (V1/V2/V3) must beat V0 (vanilla UniAP) by a margin clearly above subset noise —
target **≥ +1 PQ_things OR clearly higher instance recall@0.5** on the val subset.
- **Pass** → proceed to Phase 1 with the winning variant + weights.
- **Fail** → stop. Write the negative report; the single-stage pivot is not worth three weeks.

### 3.5 Deliverables
- One script (e.g. `mbps_pytorch/ga_uniap_phase0.py`) producing a variant × metric table.
- Qualitative viz on ~5 scenes: crowd (where geometry should fail) vs non-crowd / strong-layout
  (where it should win), to show *where* the signal acts.
- Report `reports/ga_uniap_phase0.md`.
- One runnable self-check (assert-based) that the geometric affinity reduces to V0 when `w_n=w_h=0`.

---

## 4. Phase 1 — Online Single-Stage Trainer (sketch; only if Phase 0 passes)

Deliberately under-specified — YAGNI until the gate clears.

- Inject the winning GA-UniAP variant as the teacher's online pseudo-mask op in `S2UniSeg.py`.
- Depth = precomputed input channel loaded alongside the image (keeps the loop single-stage w.r.t.
  *labels*; depth is sensor-like input, not a pseudo-label).
- Student = S2-UniSeg's mask decoder (or swap MBPS's EoMT student — decide at Phase 1 time).
- Train continuously on **Cityscapes** (training stays **remote**: santosh / fics-lab, per project
  rule). Eval trained-student PQ **locally** vs the current two-stage pipeline.
- **Contribution claim:** first single-stage, online, self-supervised panoptic method that bootstraps
  from monocular **geometric** agglomerative pooling — matching (or beating) a two-stage depth
  pipeline with no offline generation and no discontinuous self-training.

---

## 5. Defaults Chosen (stated, overridable)
- **Depth source:** DepthPro (matches current k27 pipeline). Knob if we want ZoeDepth/SPIdepth A/B.
- **Backbone:** DINO ViT-B/8 (S2-UniSeg's), *not* DINOv3 — keep the A/B clean (only affinity varies).
- **Dataset:** Cityscapes first; COCO-Stuff-27 deferred.
- **Eval protocol:** CUPS global-Hungarian, 19-class panoptic (project standard).
- **Run location:** Phase 0 local CPU; Phase 1 training remote, eval local.

## 6. Out of Scope (YAGNI)
Mamba bridge, motion / common-fate cues, SA-1B-scale pretraining, COCO, multi-granular geometric
hierarchy (height-band thresholding) — note the last as a *possible* Phase-2 extension, do not build.

## 7. Success / Kill Summary
- **Phase 0 pass:** a geometric variant beats vanilla UniAP (≥ +1 PQ_things or clear recall gain).
- **Phase 0 kill:** none beat V0 → stop, ship negative report.
- **Phase 1 success:** single-stage trained student ≥ current two-stage pipeline PQ, with the cleaner
  one-stage story.
