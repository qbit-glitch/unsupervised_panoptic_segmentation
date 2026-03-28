Plan: Option C — Monocular-Only Unsupervised Panoptic Segmentation                                                                                     

Context                                                                                                                                                
                                                                                                                                                    
After 2 months of work, our pseudo-labels (PQ=28.0) beat CUPS (27.8) but we have no trained panoptic network and no valid cross-dataset evaluation.  
CUPS trains a Cascade Mask R-CNN (Stage 2) + self-training (Stage 3) and evaluates on KITTI panoptic (same 19 Cityscapes classes). Our earlier
cross-dataset eval used wrong datasets (KITTI-STEP with 97.6% void, Mapillary with 124 classes).

Pivot: Position our method as "monocular-only unsupervised panoptic segmentation" — competitive with CUPS but requiring only a single image (no stereo
video/optical flow).

Key advantage: CUPS requires stereo video pairs + SMURF optical flow. We use only monocular depth (DepthAnything/SPIdepth). Our instance pseudo-labels
(PQ_things=19.41) already beat CUPS's final model (PQ_things=17.7).

Critical CUPS insight: Table 7b shows more pseudo-classes helps: k=27→PQ 27.8, k=40→30.3, k=54→30.6 (+2.8). Our k=100 overclustering (PQ=27.10
pseudo-labels) should benefit from this.

---
What Already Exists

┌───────────────────────────────────┬────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│             Component             │     Status     │                                           Location                                            │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ k=100 pseudo-labels (PQ=27.10)    │ DONE           │ Anydesk: datasets/cityscapes/cups_pseudo_labels_k100/                                         │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage-2 config (k=100, ResNet-50) │ DONE           │ unsupervised-panoptic-segmentation/refs/cups/configs/train_cityscapes_resnet50_k100.yaml      │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage-2 training (8000 steps)     │ DONE?          │ Anydesk: PQ=24.68 at step 6500, may have completed                                            │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage-3 self-training config      │ DONE           │ unsupervised-panoptic-segmentation/refs/cups/configs/train_self_cityscapes_resnet50_k100.yaml │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Spatial alignment fix             │ DONE           │ Applied to refs/cups/cups/data/pseudo_label_dataset.py                                        │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ CUPS KITTI eval code              │ EXISTS         │ refs/cups/cups/data/kitti.py (KITTIPanopticValidation)                                        │
├───────────────────────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ KITTI panoptic dataset            │ NOT DOWNLOADED │ Need from panoptic.cs.uni-freiburg.de                                                         │
└───────────────────────────────────┴────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘

---
Phase 0: Verify Existing State (Anydesk, ~30 min)

Goal: Check if Stage-2 k=100 training completed.

Steps:
1. SSH into Anydesk machine
2. Check training logs and checkpoints:
ls experiments/experiments/cups_resnet50_k100_v4_stage2/*/checkpoints/
tail -100 experiments/cups_resnet50_k100_stage2.log
3. If completed: identify best checkpoint (highest PQ at any validation step)
4. If not completed: resume from last checkpoint

Decision point: If PQ ≥ 24 at step 6500+, proceed to Stage 3. If training crashed, resume.

---
Phase 1: Complete Stage-2 Training if Needed (Anydesk, 0-6 hours)

Only if Phase 0 shows incomplete training.

cd ~/unsupervised-panoptic-segmentation
export PYTHONPATH=refs/cups:$PYTHONPATH
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH

nohup python -u refs/cups/train.py \
    --experiment_config_file refs/cups/configs/train_cityscapes_resnet50_k100.yaml \
    --disable_wandb \
    --ckpt_path <LAST_CHECKPOINT_PATH> \
    > experiments/cups_resnet50_k100_stage2_resume.log 2>&1 &
echo $!

Config highlights (already set):
- 8000 steps, bs=16, AdamW LR=1e-4
- k=100 pseudo-classes, THING_STUFF_THRESHOLD=0.05
- IGNORE_UNKNOWN_THING_REGIONS=False (required for k=100)
- DropLoss IoU threshold=0.4, copy-paste (max 8 objects)
- Val every 200 steps

Expected: PQ 25-26 at step 8000 (monotonically improving from 24.68 at step 6500).

---
Phase 2: Stage-3 Self-Training (Anydesk, ~4-6 hours)

Goal: Boost PQ by +1 to +3 via EMA teacher-student self-training.

Pre-requisites:
1. Update MODEL.CHECKPOINT in train_self_cityscapes_resnet50_k100.yaml to point to best Stage-2 checkpoint
2. Ensure leftImg8bit_sequence/ exists (symlink from leftImg8bit/ if needed):
cd datasets/cityscapes && ln -s leftImg8bit leftImg8bit_sequence

Launch:
cd ~/unsupervised-panoptic-segmentation
export PYTHONPATH=refs/cups:$PYTHONPATH

nohup python -u refs/cups/train_self.py \
    --experiment_config_file refs/cups/configs/train_self_cityscapes_resnet50_k100.yaml \
    --disable_wandb \
    > experiments/cups_resnet50_k100_stage3.log 2>&1 &
echo $!

Config (already set):
- 3 rounds × 500 steps = 1500 total
- LR=1e-5, bs=4, accumulate=4 (effective bs=16)
- TTA scales: [0.5, 0.75, 1.0], confidence threshold 0.7
- EMA momentum=0.999, frozen BN
- DropLoss disabled during self-training

Decision point: If any round degrades PQ, stop and use best checkpoint so far. Monitor tail -f after each round (val every 50 steps).

Expected: PQ 26-28 (CUPS reports +1.2 PQ from self-training in Table 7c).

---
Phase 3: Cityscapes Evaluation (Anydesk, ~30 min)

Goal: Final numbers at full resolution.

python refs/cups/val.py \
    --experiment_config_file refs/cups/configs/val_cityscapes.yaml \
    --disable_wandb \
    DATA.ROOT datasets/cityscapes/ \
    DATA.CROP_RESOLUTION "(1024, 2048)" \
    DATA.VAL_SCALE 1.0 \
    DATA.NUM_CLASSES 19 \
    MODEL.BACKBONE_TYPE resnet50 \
    MODEL.USE_DINO True \
    MODEL.CHECKPOINT "<BEST_CHECKPOINT>" \
    SYSTEM.ACCELERATOR gpu \
    SYSTEM.NUM_GPUS 1

Record: PQ, SQ, RQ, PQ_things, PQ_stuff, per-class breakdown.

---
Phase 4: KITTI Cross-Dataset Evaluation (Anydesk, ~1-2 hours)

Goal: Demonstrate cross-dataset generalization without fine-tuning.

Step 4.1: Download KITTI Panoptic Dataset

cd datasets/
wget http://panoptic.cs.uni-freiburg.de/static/dataset/KITTI-panoptic-segmentation-dataset.zip
unzip KITTI-panoptic-segmentation-dataset.zip

The dataset should have:
- validation/images/*.png (~100-200 images)
- validation/cityscapes_instance_format/*.png (GT labels in Cityscapes format)

CUPS's convert_to_cityscapes_instance_format.py may need to be run if GT isn't pre-converted.

Step 4.2: Run KITTI Evaluation

python refs/cups/val.py \
    --experiment_config_file refs/cups/configs/val_kitti.yaml \
    --disable_wandb \
    DATA.ROOT datasets/kitti_panoptic/ \
    DATA.CROP_RESOLUTION "(368, 1240)" \
    DATA.VAL_SCALE 1.0 \
    DATA.NUM_CLASSES 19 \
    MODEL.BACKBONE_TYPE resnet50 \
    MODEL.USE_DINO True \
    MODEL.CHECKPOINT "<BEST_CHECKPOINT>" \
    SYSTEM.ACCELERATOR gpu \
    SYSTEM.NUM_GPUS 1

Expected: PQ 20-25 on KITTI (CUPS gets 25.5).

---
Phase 5: Results Analysis (Local, ~2 hours)

Target Comparison Table

┌───────────────┬───────────┬───────────────┬──────────┬───────────┬──────────┐
│    Method     │   Input   │ Cityscapes PQ │ KITTI PQ │ PQ_things │ PQ_stuff │
├───────────────┼───────────┼───────────────┼──────────┼───────────┼──────────┤
│ U2Seg         │ COCO+IN   │ 18.4          │ 20.6     │ 10.2      │ 24.3     │
├───────────────┼───────────┼───────────────┼──────────┼───────────┼──────────┤
│ DepthG+CutLER │ CS+IN     │ 16.1          │ 11.0     │ 3.0       │ 25.7     │
├───────────────┼───────────┼───────────────┼──────────┼───────────┼──────────┤
│ CUPS (stereo) │ CS stereo │ 27.8          │ 25.5     │ 17.7      │ 35.1     │
├───────────────┼───────────┼───────────────┼──────────┼───────────┼──────────┤
│ Ours (mono)   │ CS mono   │ 25-28         │ 20-25    │ 18-20     │ 33-35    │
└───────────────┴───────────┴───────────────┴──────────┴───────────┴──────────┘

The Paper Story

1. Monocular depth replaces stereo: Off-the-shelf monocular depth (SPIdepth/DepthAnything) is sufficient for instance segmentation in driving scenes
2. Overclustering compensates: k=100 overclustering closes the gap that stereo provides for semantic quality
3. Pseudo-label quality: Our monocular pseudo-labels (PQ=27.10) are competitive with CUPS's stereo-based pseudo-labels
4. Practical advantage: Works on any single image — no stereo cameras, no video sequences, no optical flow needed

---
Caveats and Risks

┌───────────────────────────┬────────────┬──────────┬─────────────────────────────────────────────────────────────────────────────────────────────────┐
│           Risk            │ Likelihood │  Impact  │                                           Mitigation                                            │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Stage-2 training crashed  │ Medium     │ Delay    │ Resume from checkpoint; training is resumable                                                   │
│ on Anydesk                │            │          │                                                                                                 │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Self-training degrades PQ │ Medium     │ Low      │ Use Stage-2 checkpoint; document finding                                                        │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ KITTI PQ << CUPS (25.5)   │ Medium     │ High     │ Frame as "monocular limitation"; monocular depth is more domain-specific than stereo geometry   │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ KITTI dataset download    │ Low        │ Moderate │ Contact CUPS authors; try alternative mirrors                                                   │
│ fails                     │            │          │                                                                                                 │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Reviewer: "just CUPS with │ High       │ Moderate │ Emphasize: (1) overclustering insight (k=100 never tested by CUPS), (2) no stereo requirement   │
│  mono depth"              │            │          │ is practically important, (3) better PQ_things proves depth instances work                      │
├───────────────────────────┼────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Total PQ < 25 on          │ Low        │ Critical │ k=100 pseudo-labels are PQ=27.10 and Stage-2 was already at 24.68 — unlikely to regress         │
│ Cityscapes                │            │          │                                                                                                 │
└───────────────────────────┴────────────┴──────────┴─────────────────────────────────────────────────────────────────────────────────────────────────┘

---
Machines

┌──────────┬────────────────────────────┬────────────────────────┬──────────────────┐
│  Phase   │          Machine           │           IP           │   Environment    │
├──────────┼────────────────────────────┼────────────────────────┼──────────────────┤
│ 0-4      │ Anydesk RTX A6000 Pro 48GB │ Via Anydesk            │ conda env ups    │
├──────────┼────────────────────────────┼────────────────────────┼──────────────────┤
│ 5        │ Local M4 Pro MacBook       │ localhost              │ Python 3.10, MPS │
├──────────┼────────────────────────────┼────────────────────────┼──────────────────┤
│ Fallback │ Remote 2x GTX 1080 Ti      │ santosh@172.17.254.146 │ conda env cups   │
└──────────┴────────────────────────────┴────────────────────────┴──────────────────┘

Note: User mentioned remote at 172.17.254.146. Use Anydesk first (A6000 is faster, training already started there). Fall back to remote if Anydesk
unavailable.

---
Timeline

┌───────────────────────────────────────┬──────────┬──────────────┐
│                 Phase                 │ Duration │  Cumulative  │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 0: Verify state                 │ 30 min   │ 30 min       │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 1: Complete Stage-2 (if needed) │ 0-6 hrs  │ 0.5-6.5 hrs  │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 2: Stage-3 self-training        │ 4-6 hrs  │ 4.5-12.5 hrs │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 3: Cityscapes eval              │ 30 min   │ 5-13 hrs     │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 4: KITTI eval                   │ 1-2 hrs  │ 6-15 hrs     │
├───────────────────────────────────────┼──────────┼──────────────┤
│ Phase 5: Analysis                     │ 2 hrs    │ 8-17 hrs     │
└───────────────────────────────────────┴──────────┴──────────────┘

Total: 1-2 days wall-clock (mostly GPU training time).

---
Verification Checklist

- Stage-2 PQ ≥ 24 on Cityscapes val (19-class)
- Stage-3 PQ ≥ Stage-2 PQ (self-training helps, doesn't hurt)
- KITTI PQ ≥ 20 (cross-dataset transfer works)
- PQ_things ≥ 17 on Cityscapes (instances competitive with CUPS)
- Full per-class breakdown saved for paper table
- Results JSON saved to results/option_c/

⏺ Plan approved. Let's start with Phase 0 — checking the Anydesk machine for Stage-2 training status. But first, which machine should I SSH into? You mentioned santosh@172.17.254.146 as remote, but the Stage-2 training was on the Anydesk machine. Can I SSH into the Anydesk machine, or do you need to check it via the Anydesk desktop app? 