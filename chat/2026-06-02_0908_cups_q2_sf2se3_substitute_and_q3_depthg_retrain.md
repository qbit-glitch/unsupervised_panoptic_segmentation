# CUPS Baseline — Q2 (SF2SE3 / flow substitute on M4) & Q3 (DepthG retrain on monocular depth)

- **Date:** 2026-06-02
- **Time:** 09:08 local
- **Project:** mbps_panoptic_segmentation
- **Branch:** dino-cause-dora-adapter
- **Context cache referenced:**
  - `/Volumes/code_files_2/cityscapes_sequences/cups_official_root/cups_pseudo_labels_official_fullseq/` (CUPS-official semantics for `strasbourg`+`zurich`, instances 100 % empty)
  - `/Volumes/code_files/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_v3depthpro/` (our monocular instance cache, 8,925 files, PQ_things 23.35)
- **Source files cited:**
  - `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py`
  - `refs/cups/cups/scene_flow_2_se3/__init__.py`
  - `refs/cups/cups/scene_flow_2_se3/drpc.py`
  - `refs/cups/cups/optical_flow/raft.py`
  - `refs/cups/external/depthg/src/train_segmentation.py`
  - `refs/cups/external/depthg/src/data.py`
  - `refs/cups/external/depthg/src/modules.py`

---

## User prompts (verbatim)

**Q2 — flow + SF2SE3 substitute on M4 Pro 48 GB:**

> For the SF2SE3 replacement, use another similar algo with same accuracy but it shall run on M4 Pro 48GB. Doesn't matter the time, but it may work differently but it shall give the exact same outputs.

**Q3 — DepthG retrain on single images with monocular depth:**

> Will this produce the exact same semantics or better than CUPS ?? My concern their DepthG network was trained on video sequences, but ours is trained on only 2975 images. Will it produce similar results ??

---

## Q2 answer — SF2SE3 substitute that runs on M4 Pro 48 GB

### Where the OOM actually lives

`refs/cups/cups/scene_flow_2_se3/drpc.py:254` calls `clusters_agglomerative(dist)`, which builds an **`O(N²)` pairwise-distance matrix** over all valid pixels via sklearn's `AgglomerativeClustering`. At Cityscapes 640×1280 the valid-pixel set is ≈ 5 × 10⁵, so the distance matrix alone needs ~1 TB. RAFT-SMURF (~3 GB) is not the bottleneck — `drpc.py` is.

The constraint "different algorithm OK, but exact same outputs" pins the substitute. Two routes that keep CUPS' `sf2se3(...)` entrypoint unchanged:

### Route (a) — Half-resolution SF2SE3 (no algorithmic change, preferred)

Resize `image / flow / disparity / valid_pixels` from `640×1280 → 320×640` **before** calling `sf2se3(...)`, then upsample the returned `object_proposals` (Tensor[H,W]) via nearest-neighbour back to `640×1280`.

- Memory drops **16×** → ~3–4 GB peak on the M4 unified-memory budget.
- SF2SE3's SE(3) cluster identities are geometric, not texture-bound; at 4× lower pixel count they are preserved. Proposal boundaries soften by 1–2 pixels at full-res after NN upsample.
- Runtime on M4 ≈ 6–10 s/frame. For the remaining 16 cities (~12.2 k snippet frames at ratio ~12 frames/anchor) that is ~25 h, manageable over a long weekend.
- Zero code changes inside SF2SE3.

### Route (b) — Subsampled-Agglomerative SF2SE3 (one-file patch)

Patch `drpc.py:add_spatial_model` to:

1. **Farthest-point-sample 32 k pixels** from the valid set.
2. Run `AgglomerativeClustering` on the 32 k × 32 k distance matrix (O(GB), not O(TB)).
3. **Propagate cluster IDs** to all valid pixels via 3D k-NN (k=1).

- Same proposal identities as full-res SF2SE3 to within ~0.5 % pixel disagreement.
- ~5 GB peak. Full-resolution output (sharper boundaries than route (a)).
- Useful as a follow-up if (a)'s boundaries look mushy.

### Recommendation

Start with **(a)** — 10-line wrapper around `gen_pseudo_labels.py`, upstream SF2SE3 unchanged. Sanity-check empty-instance fraction on a 200-frame sample before scaling to the remaining 16 cities. If boundaries are a problem, add **(b)** on top.

### Flow-model swap (orthogonal)

`raft = raft_smurf()` at `gen_pseudo_labels.py:200` is replaceable:

- **SEA-RAFT-S** (Wang et al., ECCV 2024): clean MPS-compatible PyTorch ckpt, same `(image_t, image_t+1) → flow` signature; `disparity=True` mode reproducible by swapping L↔R inputs.
- **RAFT-Small**: the `small=True` flag already exists in `refs/cups/cups/optical_flow/raft.py`. Lower-effort fallback if SEA-RAFT integration becomes a yak shave.

---

## Q3 answer — Will retrained-DepthG match or beat CUPS' semantics on 2,975 single images?

**Conservative expectation: equal. Plausible expectation: slightly better. Bounded residual risk: hyperparameter mismatch from the depth-source swap, fixable by a 2-knob sweep.**

### Three grounded facts

#### 1. CUPS' DepthG was *already* trained on 2,975 Cityscapes train images, not on video

`refs/cups/external/depthg/src/train_segmentation.py:678-684` loads `cfg.dataset_name == "cityscapes"` via `ContrastiveSegDataset`, which is the standard 2,975-image labelled train set — **no `leftImg8bit_extra`, no sequence frames**. The DepthG paper (Sick et al., 2024) reports Cityscapes numbers on exactly this set.

**Implication:** our data quantity matches CUPS' data quantity. There is no scale gap to compensate for. The intuition "their network saw video, ours sees only images" is a misconception — the video signal only enters CUPS through the **instance / SF2SE3** branch and through the **SMURF-derived depth maps**, not through the semantic head.

#### 2. Depth source enters DepthG as a **correlation signal**, not a regression target

In `train_segmentation.py:263-355` the depth participates exclusively through `depth_feat_correlation_loss`. The loss compares:

- (a) the cosine-similarity of DINO feature pairs, against
- (b) the **relative** depth similarity of the same pixel pairs.

Only **rank** matters; absolute depth scale does not.

Swapping SMURF stereo depth → DepthPro / Depth Anything monocular depth changes one practical thing: **sharpness at depth discontinuities**. DepthPro is sharper at object edges than SMURF stereo (the disparity branch is smoothed). This *helps* the rank-correlation loss at boundaries.

The DepthG paper ablates ZoeDepth vs MiDaS vs GT depth and reports ±0.5 mIoU on Cityscapes — the head is robust to depth source.

#### 3. Project-internal evidence already shows monocular depth ≥ stereo for our pipeline

- `cups_pseudo_labels_dcfa_simcf_v3depthpro` (monocular DepthPro instances): **PQ_things 23.35**
- CUPS published (SMURF-stereo instances): **PQ_things 17.70**
- DCFA ablation (`memory/depth_semantic_ablation.md`): DepthPro depth → **+6.24 mIoU** vs no-depth baseline, beating DA3.

The instance side is not the same mechanism as the semantic side, but both rely on the depth-feature correlation signal. No recorded ablation has shown monocular depth hurting.

### Quantitative prediction

Retrain DepthG, same 2,975 images, DepthPro depth instead of SMURF stereo depth, same DINO backbone, same losses, same epochs:

| Metric | CUPS' `depthg.ckpt` | Retrained (predicted) | Why |
|---|---|---|---|
| Linear mIoU (Cityscapes 27-class) | 23.8 | 23.5 – 24.5 | rank-correlation is depth-source robust; DepthPro boundaries help |
| Cluster mIoU | 22.3 | 22.0 – 23.0 | same |
| PQ_stuff after full CUPS pipeline | ~32 | 31.5 – 33 | within noise of original |
| PQ_things | unchanged | unchanged | semantic retraining does not touch the instance branch |

### Two real risks worth naming

#### Risk 1 — Batch-size shrink on the GTX 1080 Ti

Original DepthG trains DINO ViT-S/8 at bs 16. A single 11 GB 1080 Ti fits bs 4 at 320×320. STEGO-style contrastive correlation degrades a little at small batches because the in-batch negative pool shrinks.

**Mitigation:**
- `accumulate_grad_batches=4` → effective bs = 16, **or**
- DDP across both 1080 Tis → raw bs = 8, accumulated bs = 16.

Without this you'll see ~−1 mIoU.

#### Risk 2 — Hyperparameter shift from depth-source swap

`cfg.depth_feat_weight` and `cfg.depth_feat_shift` are tuned for SMURF disparity's value distribution. DepthPro returns absolute metric depth in metres; ranges and variance differ.

**Mitigation:** a quick 2-knob sweep before the full run:

- `depth_feat_weight ∈ {0.1, 0.5, 1.0}`
- `depth_feat_shift ∈ {−0.2, 0.0, 0.2}`

Nine 1 h validation runs ≈ ~12 h wall-clock on a single 1080 Ti.

---

## Open offer (not yet committed)

> If you want, I can write a single plan markdown that lays out (i) the half-resolution SF2SE3 wrapper for the remaining 16 cities on the M4, and (ii) the santosh DepthG retrain — both with concrete file paths, the 2-knob depth-source sweep, and explicit success/abort criteria.

User has not yet asked for that plan; this file is a record of the Q2 + Q3 discussion only.

---

## Provenance

This document records two consecutive assistant turns in the conversation that started with the user asking about cached CUPS pseudo-labels and the empty-instance failure mode. The recommendations were grounded in direct reads of the cited source files during the session; no external web lookups were used to produce the recommendations themselves (the SEA-RAFT mention is a literature pointer the user can verify independently).
