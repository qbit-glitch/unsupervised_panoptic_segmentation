Plan: Instance Quality Improvement — Breaking the Depth-Only Ceiling                                                                                   

Context                                                                                                                                                
                                                    
Semantics are solved (PQ_stuff=35.04, mIoU=57% via DepthGuidedUNet). The bottleneck is instance quality: depth-guided splitting gives PQ_things=19.41
but fails for co-planar objects (person PQ=4.2, RQ=8.8% — only 170/3206 matched). The trained mobile model (PQ=24.78) actually makes things worse
(PQ_things drops to 11.37) due to semantic-instance class misalignment.

Goal: Push PQ_things > 22 and overall PQ > 29 by (1) fixing alignment, (2) training a dedicated instance model, and (3) iterative self-training.

---
Phase 0: Post-Hoc Semantic-Instance Alignment (~1 day, local only)

Why: The UNet gives PQ_stuff=35.04 and depth-guided gives PQ_things=19.41. Simply combining them via post-hoc alignment should yield PQ~28 with zero
training. This was designed in reports/semantic_instance_alignment_proposal.md but never executed.

What to implement (in mbps_pytorch/evaluate_cascade_pseudolabels.py or new script):

1. A-2 (Selective Merge): UNet semantics for stuff pixels, stage-1 k=80 class assignments for thing pixels within depth-guided instance masks.
Expected: PQ~28.
2. A-4 (Majority Vote + Stuff Preserve): For each thing instance, take majority class from UNet predictions (only if it's a thing class). Preserves
UNet stuff while potentially correcting some k=80 thing errors.
3. A-3 (Confidence-Weighted): Weight votes by UNet softmax confidence for ambiguous instances.

Inputs needed:
- UNet semantic predictions: run DepthGuidedUNet inference on 500 val images → 19-class predictions
- Depth-guided instance masks: from pseudo_instance_spidepth/ (already generated)
- k=80 mapped semantics: from pseudo_semantic_mapped_k80/ (already generated)

Evaluation: PQ, PQ_stuff, PQ_things against Cityscapes val GT.

Expected outcome: PQ ≈ 27.5-28.5 (validates the approach, establishes strong baseline).

---
Phase 1: CUPS Stage-2 Cascade Mask R-CNN on Better Pseudo-Labels (~3-5 days, remote GPU)

Why: A Cascade Mask R-CNN learns object shapes from pseudo-labels and generalizes beyond depth boundaries. Previous Stage-2 run reached PQ=24.68 on
CAUSE-CRF labels; now we have much better pseudo-labels (k=80 mapped, mIoU~55% vs 42%).

Architecture: DINO ResNet-50 backbone + Cascade Mask R-CNN (3 stages), via Detectron2 on remote.

Config (clone from refs/cups/configs/train_cityscapes_resnet50_v4.yaml):
- 8000 steps, batch_size=1 per GPU (2x GTX 1080 Ti)
- AdamW, LR=1e-4, WD=1e-5
- Multi-scale: [384×768] to [512×1024]
- DropLoss (IoU threshold 0.4), copy-paste (max 8 objects)
- IGNORE_UNKNOWN_THING_REGIONS: True (critical)
- Val every 500 steps

Data pipeline:
1. Run mbps_pytorch/convert_to_cups_format.py with --semantic_subdir pseudo_semantic_mapped_k80 and --instance_subdir pseudo_instance_spidepth
2. Upload to remote machine
3. Update config ROOT_PSEUDO path

Key change from prior attempt: Use k=80 overclustered semantics (mIoU~55%) instead of CAUSE-CRF (42%).

Expected outcome: PQ_things ≈ 20-23. The model should learn to detect objects that depth alone misses.

---
Phase 2: Iterative Self-Training (CutLER-Style, ~2-3 days, remote GPU)

Why: Self-training with a strong teacher iteratively improves pseudo-labels beyond the initial ceiling. Prior self-training failed because the teacher
(mIoU=53% semantic model) was too noisy. A Cascade Mask R-CNN teacher should be much stronger.

Config (from refs/cups/configs/train_self_cityscapes.yaml):
- 3 rounds × 500 steps
- LR=1e-5 (10x lower than Stage-2)
- Crop (1024×2048), full resolution teacher inference
- EMA decay 0.999, confidence threshold 0.7
- No DropLoss during self-training
- Mask refinement: morphological cleanup + guided filter (refs/cups/cups/mask_refinement.py)

Implementation: Use CUPS's pl_model_self.py (SelfSupervisedModel class) which handles EMA teacher, pseudo-label generation, and mask refinement.

Expected outcome: PQ_things ≈ 22-25 (CUPS reports +2-3 PQ from self-training).

---
Phase 3: Final Panoptic Assembly

Combine best components:
- Stuff: DepthGuidedUNet semantics (PQ_stuff=35.04)
- Things: Phase 2 self-trained Cascade Mask R-CNN instances
- Merge: Post-hoc alignment (Phase 0's A-4 method) — UNet for stuff, majority-voted classes for things within Cascade R-CNN masks

Expected outcome: PQ ≈ 29-31.

---
Key Files

┌─────────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
│                        File                         │                       Role                        │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ reports/semantic_instance_alignment_proposal.md     │ Phase 0 design (A-1 through C-2 methods)          │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ mbps_pytorch/evaluate_cascade_pseudolabels.py       │ Evaluation script to extend with alignment        │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ mbps_pytorch/convert_to_cups_format.py              │ Pseudo-label → CUPS format converter              │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ refs/cups/configs/train_cityscapes_resnet50_v4.yaml │ Stage-2 training config                           │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ refs/cups/configs/train_self_cityscapes.yaml        │ Self-training config                              │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ refs/cups/cups/pl_model_self.py                     │ Self-training implementation (EMA, pseudo-labels) │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ refs/cups/cups/mask_refinement.py                   │ Classical mask post-processing                    │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ mbps_pytorch/refine_net.py                          │ DepthGuidedUNet model (for Phase 0 inference)     │
├─────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ mbps_pytorch/train_refine_net.py                    │ UNet training script (for loading checkpoints)    │
└─────────────────────────────────────────────────────┴───────────────────────────────────────────────────┘

Compute Plan

┌─────────┬────────────────────────────────────┬────────────────────────────────────────────┐
│  Phase  │              Machine               │                    Time                    │
├─────────┼────────────────────────────────────┼────────────────────────────────────────────┤
│ Phase 0 │ Local M4 Pro (MPS)                 │ ~2 hours (UNet inference + alignment eval) │
├─────────┼────────────────────────────────────┼────────────────────────────────────────────┤
│ Phase 1 │ Remote 2x 1080 Ti or Anydesk A6000 │ ~4-6 hours training                        │
├─────────┼────────────────────────────────────┼────────────────────────────────────────────┤
│ Phase 2 │ Remote 2x 1080 Ti or Anydesk A6000 │ ~2-3 hours (3 rounds)                      │
├─────────┼────────────────────────────────────┼────────────────────────────────────────────┤
│ Phase 3 │ Local M4 Pro                       │ ~1 hour (merge + eval)                     │
└─────────┴────────────────────────────────────┴────────────────────────────────────────────┘

Verification

1. Phase 0: PQ ≥ 27.5 confirms alignment recovers the gap
2. Phase 1: PQ_things > 20 confirms Cascade R-CNN learns beyond depth
3. Phase 2: PQ_things > 22 confirms self-training helps
4. Phase 3: PQ > 29 confirms final assembly beats CUPS significantly
5. Per-class: person RQ > 12% (from 8.8%), car PQ > 18 (from 16.6)