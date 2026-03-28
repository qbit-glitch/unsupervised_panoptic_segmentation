Plan: Instance Pseudo-Label Improvement — Path to PQ > 29

Context

CenterOffsetHead v2 failed (PQ_things=9.79 vs depth-guided baseline 19.41). But this was just ONE approach, and it was the wrong one — learned
center/offset regression degrades noisy pseudo-labels by ~50%.

The critical finding that changes everything: CUPS Stage-2 (Cascade Mask R-CNN + DINOv2 ViT-B/14) already achieved PQ_things = 20.6-22.6, which BREAKS
THROUGH the depth-guided ceiling of 19.41. The detection framework learned to produce better instances than its training signal. This proves
improvement IS possible.

Additionally, there are 6 fully-implemented instance generation methods in the codebase that have NEVER been evaluated on the val set. We have
unexplored territory.

Goal: PQ > 29 by combining the best instances with UNet semantics (PQ_stuff=35.04).
Math: PQ = (35.04 * 11 + PQ_things * 8) / 19. We need PQ_things >= 20.7 for PQ >= 29.

---
Phase 1: Evaluate Untested Methods (LOCAL, ~2-3h total)

Run locally on M4 Pro while Stage-3 trains on remote. All scripts exist and are ready.

Step 1.1: CuVLER Pre-Trained Detector (HIGHEST PRIORITY)

CuVLER is a self-trained class-agnostic Cascade Mask R-CNN. It discovers objects using visual features, not depth — exactly what we need for co-planar
pedestrians.

- Script: mbps_pytorch/generate_cutler_detector_instances.py
- Weights: refs/cuvler/weights/cuvler_self_trained.pth (575MB, local)
- Quick test (5 images), then full val set (500 images)
- Evaluate: mbps_pytorch/evaluate_cascade_pseudolabels.py with --instance_subdir pointing to CuVLER output
- Success: PQ_things > 15 = useful for ensemble; > 20 = major win

Step 1.2: CutLER Pre-Trained Detector

Same pipeline, different weights (refs/cutler/weights/cutler_cascade_final.pth). Comparison shows value of self-training.

Step 1.3: DINOv2 HDBSCAN Clustering (Feature-Based Splitting)

Uses pre-computed DINOv2 features (768-dim, already at dinov2_features/val/) + depth to cluster large connected components into separate instances.

- Script: mbps_pytorch/generate_dino_cluster_instances.py
- Needs: --semantic_subdir pseudo_semantic_mapped_k80 --feature_subdir dinov2_features --depth_subdir depth_spidepth
- Note: Default feature_subdir is dinov3_features — check if DINOv2 features use same format
- Success: PQ_things > 16 = useful complement to depth-guided

Step 1.4: Depth Layer Instances (Quantile Binning)

Replaces Sobel threshold with adaptive depth quantile bins. Could handle gradual depth transitions.

- Script: mbps_pytorch/generate_depth_layer_instances.py
- Quick to run (CPU-only, numpy)
- Success: PQ_things > 19.41 = direct improvement over current baseline

Step 1.5: Post-Processing Pipeline on Depth-Guided Instances

4-step refinement (morphological cleanup, guided filter, superpixel snapping, fragment merging). Free improvement.

- Script: mbps_pytorch/postprocess_instances.py
- Apply to: existing pseudo_instance_spidepth/val/
- Success: Any PQ_things improvement over 19.41

---
Phase 2: CUPS Instances + UNet Semantics (REMOTE, after Stage-3)

This is the highest-confidence path to PQ > 29.

Step 2.1: Extract CUPS Instance Masks from Stage-2/3 Checkpoint

On remote, run CUPS inference on all 500 val images. Save as NPZ files compatible with our evaluation pipeline.

- Need: Script to load CUPS Detectron2 checkpoint, run inference, save instance masks
- Existing: scripts/cups_unet_panoptic_eval_v2.py has partial pipeline — adapt for full instance extraction
- Output: cups_vitb_instances/val/{city}/*.npz

Step 2.2: Transfer + Evaluate Combination

Combine UNet P2-B semantics (checkpoints/unet_p2b_attention/best.pth) with CUPS instances using the A-2 selective merge method.

- Script: mbps_pytorch/alignment_ablation.py (method A-2: stuff from UNet, things from CUPS)
- Expected: PQ ~ 29.0-30.0 (PQ_stuff=35.04 + PQ_things=20.6-22.6+)

Step 2.3: If Stage-3 Is Still Running — Use Stage-2 Checkpoint NOW

Don't wait. The Stage-2 checkpoint already exists on remote. Extract instances from Stage-2 first, evaluate the combination, then upgrade to Stage-3
when available.

---
Phase 3: Ensemble Best Methods (~1-2h after Phase 1+2)

Step 3.1: Gap-Fill Merge

Take best method (likely CUPS) as primary, fill uncovered thing regions with second-best (depth-guided or CuVLER).

- Script: mbps_pytorch/merge_instance_sources.py (gap_fill_merge mode)
- Expected: +0.5-1.0 PQ_things over best single source

Step 3.2: Per-Class Best-Source Selection

From Phase 1 evaluation, identify which method wins per class (e.g., depth-guided for truck/bus, CuVLER for person). Build a class-aware selector.

Step 3.3: Apply Post-Processing to Ensemble Output

Stack the 4-step post-processing on the ensemble result for final polish.

---
Phase 4: Feature Gradient Boundaries (ONLY if PQ < 29 after Phase 1-3)

Augment depth edges with DINOv2 feature gradients to split co-planar objects. New code needed (~3h).

- Compute spatial gradients of DINOv2 features at 32x64 grid
- L2 norm of gradient vector = boundary strength
- Combine with depth Sobel edges (weighted sum)
- Use combined boundary map in depth-guided splitting pipeline

---
Verification

After each step, evaluate with:
python mbps_pytorch/evaluate_cascade_pseudolabels.py \
--cityscapes_root /path/to/cityscapes --split val \
--semantic_subdir pseudo_semantic_mapped_k80 \
--instance_subdir <METHOD_OUTPUT_DIR> \
--eval_size 512 1024

Track per-class PQ_things (especially person, car, rider) to understand where each method helps.

Final success criterion: PQ > 29.0 on full Cityscapes val (500 images).

---
Key Files

┌────────────────────────────────────────────────────┬─────────────────────────────┐
│                        File                        │           Purpose           │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/generate_cutler_detector_instances.py │ CuVLER/CutLER inference     │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/generate_dino_cluster_instances.py    │ HDBSCAN feature clustering  │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/generate_depth_layer_instances.py     │ Quantile depth binning      │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/postprocess_instances.py              │ 4-step post-processing      │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/merge_instance_sources.py             │ Multi-source ensemble       │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/evaluate_cascade_pseudolabels.py      │ Panoptic evaluation         │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ mbps_pytorch/alignment_ablation.py                 │ UNet + instance merge (A-2) │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ refs/cuvler/weights/cuvler_self_trained.pth        │ CuVLER weights (local)      │
├────────────────────────────────────────────────────┼─────────────────────────────┤
│ refs/cutler/weights/cutler_cascade_final.pth       │ CutLER weights (local)      │
└────────────────────────────────────────────────────┴─────────────────────────────┘