# CUPS Backbone Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 7 CUPS Stage-2 (16K steps) + Stage-3 (12K steps) backbone ablations on the remote 2×GTX 1080 Ti machine to determine which backbone yields the best panoptic quality.

**Architecture:** CUPS Cascade Mask R-CNN with 7 backbone variants trained on `cups_pseudo_labels_k80` (CAUSE-TR k=80 semantics + SF2SE3 instances). Stage-2 trains the full panoptic model on pseudo-labels. Stage-3 performs 3-round EMA self-training on Stage-2 predictions. Results compared via PQ on Cityscapes val (500 images, 27-class CAUSE+Hungarian metric).

**Tech Stack:** Python 3.8, PyTorch 2.1.2+cu118, Detectron2 0.6, PyTorch Lightning, DINOv2 (facebookresearch/dinov2), DINOv3 (facebookresearch/dinov3), conda env `cups`, remote `santosh@172.17.254.146`

---

## Execution Order (Priority)

| # | Ablation | Status | Stage-2 Config | Stage-3 Config |
|---|----------|--------|----------------|----------------|
| A1 | DINOv2 ResNet-50 | Config exists | `train_cityscapes_resnet50_k80_16k_2gpu.yaml` | `train_self_resnet50_k80_12k_2gpu.yaml` |
| A2 | DINOv2 ViT-B/14 | Config exists | `train_cityscapes_vitb_k80_16k_2gpu.yaml` | `train_self_vitb_k80_12k_2gpu.yaml` |
| A3 | DINOv3 ViT-B/16 | Config exists | `train_cityscapes_dinov3_vitb_k80_16k_2gpu.yaml` | `train_self_dinov3_vitb_k80_12k_2gpu.yaml` |
| A4 | DINOv2 ViT-S/14 | **Backbone not implemented** | `train_cityscapes_dinov2_vits_k80_16k_2gpu.yaml` | `train_self_dinov2_vits_k80_12k_2gpu.yaml` |
| A5 | DINOv2 ViT-L/14 | **Backbone not implemented** | `train_cityscapes_dinov2_vitl_k80_16k_2gpu.yaml` | `train_self_dinov2_vitl_k80_12k_2gpu.yaml` |
| A6 | DINOv3 ViT-S/14 | **Backbone not implemented** | `train_cityscapes_dinov3_vits_k80_16k_2gpu.yaml` | `train_self_dinov3_vits_k80_12k_2gpu.yaml` |
| A7 | DINOv3 ViT-L/14 | **Backbone not implemented** | `train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml` | `train_self_dinov3_vitl_k80_12k_2gpu.yaml` |

---

## Infrastructure

**Remote machine:** `santosh@172.17.254.146`  
**GPUs:** 2× GTX 1080 Ti (11GB VRAM each)  
**Conda env:** `cups`  
**Experiment root:** `/media/santosh/Kuldeep/panoptic_segmentation/experiments`  
**Pseudo-labels root:** `/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/`  
**Data root:** `/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/`  
**⚠️ Disk:** 99% full (54GB free as of 2026-04-04) — delete old experiment logs before starting  
**LD_LIBRARY_PATH fix (always required):**
```bash
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
```

**Batch size / precision by backbone:**

| Backbone | bs/GPU | grad_accum | eff_bs | precision |
|----------|--------|------------|--------|-----------|
| ResNet-50 | 2 | 4 | 16 | 16-mixed |
| DINOv2 ViT-S | 2 | 4 | 16 | 16-mixed |
| DINOv2 ViT-B | 2 | 4 | 16 | 16-mixed |
| DINOv2 ViT-L | 1 | 8 | 16 | 16-mixed |
| DINOv3 ViT-S | 2 | 4 | 16 | 32-true |
| DINOv3 ViT-B | 1 | 8 | 16 | 32-true |
| DINOv3 ViT-L | 1 | 8 | 16 | 32-true |

---

## File Structure

### Create (configs — all in `refs/cups/configs/`)
```
train_cityscapes_resnet50_k80_16k_2gpu.yaml       (A1 Stage-2)
train_self_resnet50_k80_12k_2gpu.yaml              (A1 Stage-3, already exists — update STEPS)
train_cityscapes_vitb_k80_16k_2gpu.yaml            (A2 Stage-2)
train_self_vitb_k80_12k_2gpu.yaml                  (A2 Stage-3)
train_cityscapes_dinov3_vitb_k80_16k_2gpu.yaml     (A3 Stage-2)
train_self_dinov3_vitb_k80_12k_2gpu.yaml           (A3 Stage-3, already exists — update STEPS)
train_cityscapes_dinov2_vits_k80_16k_2gpu.yaml     (A4 Stage-2)
train_self_dinov2_vits_k80_12k_2gpu.yaml           (A4 Stage-3)
train_cityscapes_dinov2_vitl_k80_16k_2gpu.yaml     (A5 Stage-2)
train_self_dinov2_vitl_k80_12k_2gpu.yaml           (A5 Stage-3)
train_cityscapes_dinov3_vits_k80_16k_2gpu.yaml     (A6 Stage-2)
train_self_dinov3_vits_k80_12k_2gpu.yaml           (A6 Stage-3)
train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml     (A7 Stage-2)
train_self_dinov3_vitl_k80_12k_2gpu.yaml           (A7 Stage-3)
```

### Create (backbone code — `refs/cups/cups/model/`)
```
backbone_dinov2_vits.py     DINOv2 ViT-S/14 backbone wrapper + Cascade Mask R-CNN factory
backbone_dinov2_vitl.py     DINOv2 ViT-L/14 backbone wrapper + Cascade Mask R-CNN factory
backbone_dinov3_vits.py     DINOv3 ViT-S/14 backbone wrapper + Cascade Mask R-CNN factory
backbone_dinov3_vitl.py     DINOv3 ViT-L/14 backbone wrapper + Cascade Mask R-CNN factory
```

### Modify
```
refs/cups/cups/pl_model_pseudo.py    Add elif branches for 4 new backbone types (lines ~510-540)
refs/cups/cups/pl_model_self.py      Add elif branches for 4 new backbone types (lines ~347-370)
scripts/run_cups_ablation_a1.sh      Launch script for A1 (ResNet-50)
scripts/run_cups_ablation_a2.sh      Launch script for A2 (DINOv2 ViT-B)
scripts/run_cups_ablation_a3.sh      Launch script for A3 (DINOv3 ViT-B)
scripts/run_cups_ablation_a4.sh      Launch script for A4 (DINOv2 ViT-S)
scripts/run_cups_ablation_a5.sh      Launch script for A5 (DINOv2 ViT-L)
scripts/run_cups_ablation_a6.sh      Launch script for A6 (DINOv3 ViT-S)
scripts/run_cups_ablation_a7.sh      Launch script for A7 (DINOv3 ViT-L)
```

---

## Task 0: Pre-flight — Remote Disk Cleanup

**Files:** SSH to remote only

- [ ] **Step 0.1: Check disk usage and free space**
```bash
ssh santosh@172.17.254.146 "df -h /media/santosh/Kuldeep/ && du -sh /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/*/"
```
Expected: List of experiment dirs with sizes.

- [ ] **Step 0.2: Delete old experiment wandb logs (keep only checkpoints)**
```bash
ssh santosh@172.17.254.146 "
find /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/ \
  -name 'wandb' -type d -exec du -sh {} \;
"
```
Delete wandb dirs from finished runs (they can be 5-20GB each):
```bash
ssh santosh@172.17.254.146 "
find /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/ \
  -name 'wandb' -type d -exec rm -rf {} + 2>/dev/null; echo done
"
```

- [ ] **Step 0.3: Verify ≥80GB free after cleanup**
```bash
ssh santosh@172.17.254.146 "df -h /media/santosh/Kuldeep/"
```
Expected: Avail column ≥80G. If not, also delete old non-best checkpoints.

- [ ] **Step 0.4: Verify conda env and GPU**
```bash
ssh santosh@172.17.254.146 "
  export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH
  ~/anaconda3/envs/cups/bin/python -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))'
"
```
Expected: `2 NVIDIA GeForce GTX 1080 Ti`

- [ ] **Step 0.5: Sync latest refs/cups code to remote**
```bash
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups/ \
  santosh@172.17.254.146:~/mbps_panoptic_segmentation/refs/cups/
```

---

## Task 1: Create Stage-2 + Stage-3 Configs for A1 (ResNet-50)

**Files:**
- Create: `refs/cups/configs/train_cityscapes_resnet50_k80_16k_2gpu.yaml`
- Modify: `refs/cups/configs/train_self_cityscapes_resnet50_k80_2gpu.yaml` (update ROUND_STEPS to 4000, 3 rounds = 12K)

- [ ] **Step 1.1: Create Stage-2 config (16K steps)**
```yaml
# refs/cups/configs/train_cityscapes_resnet50_k80_16k_2gpu.yaml
# A1: DINOv2 ResNet-50 Cascade Mask R-CNN — Stage-2, 16K steps
# Remote: 2x GTX 1080 Ti | eff_bs=16 (2gpu × bs2 × accum4) | fp16
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "resnet50"
  USE_DINO: True
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 2
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_resnet50_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 2
  ACCUMULATE_GRAD_BATCHES: 4
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 1.0
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 1.2: Create Stage-3 config (12K steps = 3 rounds × 4K)**

The Stage-3 config needs `MODEL.CHECKPOINT` filled in AFTER Stage-2 finishes. Create the template now with a placeholder path:
```yaml
# refs/cups/configs/train_self_resnet50_k80_12k_2gpu.yaml
# A1: DINOv2 ResNet-50 — Stage-3, 3 rounds × 4000 steps = 12K total
# Fill in MODEL.CHECKPOINT after Stage-2 completes.
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "resnet50"
  USE_DINO: True
  CHECKPOINT: "FILL_IN_AFTER_STAGE2"
  TTA_SCALES:
    - 0.5
    - 0.75
    - 1.0
AUGMENTATION:
  NUM_STEPS_STARTUP: 0
  COPY_PASTE: True
  CONFIDENCE: 0.75
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 2
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_resnet50_k80_12k_stage3"
TRAINING:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 200
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 500
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 1.0
SELF_TRAINING:
  ROUND_STEPS: 4000
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 1.3: Create launch script**
```bash
# scripts/run_cups_ablation_a1.sh
#!/bin/bash
# A1: DINOv2 ResNet-50 — Stage-2 (16K) then Stage-3 (12K)
set -e
REMOTE="santosh@172.17.254.146"
CUPS="~/mbps_panoptic_segmentation/refs/cups"
PYTHON="~/anaconda3/envs/cups/bin/python"
LD="export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH"

echo "=== A1: ResNet-50 Stage-2 (16K steps) ==="
ssh $REMOTE "cd $CUPS && $LD && nohup $PYTHON train.py \
  --config configs/train_cityscapes_resnet50_k80_16k_2gpu.yaml \
  > ~/logs/cups_a1_stage2.log 2>&1 &
  echo PID:\$!"

echo "Monitor: ssh $REMOTE 'tail -f ~/logs/cups_a1_stage2.log'"
echo "After Stage-2 completes, fill in CHECKPOINT in train_self_resnet50_k80_12k_2gpu.yaml"
echo "Then run Stage-3:"
echo "  ssh $REMOTE \"cd $CUPS && $LD && nohup $PYTHON train_self.py \\"
echo "    --config configs/train_self_resnet50_k80_12k_2gpu.yaml \\"
echo "    > ~/logs/cups_a1_stage3.log 2>&1 &\""
```

- [ ] **Step 1.4: Sync configs and launch**
```bash
rsync -av refs/cups/configs/train_cityscapes_resnet50_k80_16k_2gpu.yaml \
          refs/cups/configs/train_self_resnet50_k80_12k_2gpu.yaml \
  santosh@172.17.254.146:~/mbps_panoptic_segmentation/refs/cups/configs/

ssh santosh@172.17.254.146 "mkdir -p ~/logs"
bash scripts/run_cups_ablation_a1.sh
```

---

## Task 2: Configs for A2 (DINOv2 ViT-B)

**Files:**
- Create: `refs/cups/configs/train_cityscapes_vitb_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_vitb_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a2.sh`

- [ ] **Step 2.1: Create Stage-2 config**
```yaml
# refs/cups/configs/train_cityscapes_vitb_k80_16k_2gpu.yaml
# A2: DINOv2 ViT-B/14 — Stage-2, 16K steps | fp16 | eff_bs=16
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov2_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov2_vitb_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 2
  ACCUMULATE_GRAD_BATCHES: 4
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 2.2: Create Stage-3 config template**
```yaml
# refs/cups/configs/train_self_vitb_k80_12k_2gpu.yaml
# A2: DINOv2 ViT-B/14 — Stage-3, 3 rounds × 4000 = 12K steps
# Fill in MODEL.CHECKPOINT after Stage-2 completes.
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov2_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
  CHECKPOINT: "FILL_IN_AFTER_STAGE2"
  TTA_SCALES:
    - 0.5
    - 0.75
    - 1.0
AUGMENTATION:
  NUM_STEPS_STARTUP: 0
  COPY_PASTE: True
  CONFIDENCE: 0.75
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov2_vitb_k80_12k_stage3"
TRAINING:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 200
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 500
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
SELF_TRAINING:
  ROUND_STEPS: 4000
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 2.3: Create launch script `scripts/run_cups_ablation_a2.sh`**
```bash
#!/bin/bash
# A2: DINOv2 ViT-B/14 — Stage-2 (16K) then Stage-3 (12K)
set -e
REMOTE="santosh@172.17.254.146"
CUPS="~/mbps_panoptic_segmentation/refs/cups"
PYTHON="~/anaconda3/envs/cups/bin/python"
LD="export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH"

echo "=== A2: DINOv2 ViT-B Stage-2 (16K steps) ==="
ssh $REMOTE "cd $CUPS && $LD && nohup $PYTHON train.py \
  --config configs/train_cityscapes_vitb_k80_16k_2gpu.yaml \
  > ~/logs/cups_a2_stage2.log 2>&1 & echo PID:\$!"
echo "Monitor: ssh $REMOTE 'tail -f ~/logs/cups_a2_stage2.log'"
```

---

## Task 3: Configs for A3 (DINOv3 ViT-B) — Update Step Count Only

**Files:**
- Create: `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_dinov3_vitb_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a3.sh`

- [ ] **Step 3.1: Create Stage-2 config (copy dinov3_vitb_k80_2gpu.yaml, change STEPS to 16000)**
```yaml
# refs/cups/configs/train_cityscapes_dinov3_vitb_k80_16k_2gpu.yaml
# A3: DINOv3 ViT-B/16 — Stage-2, 16K steps | fp32 | eff_bs=16
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov3_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
  TTA_SCALES:
    - 0.75
    - 1.0
    - 1.25
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov3_vitb_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "32-true"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 3.2: Create Stage-3 config template**
```yaml
# refs/cups/configs/train_self_dinov3_vitb_k80_12k_2gpu.yaml
# A3: DINOv3 ViT-B/16 — Stage-3, 3 rounds × 4000 = 12K steps
# Fill in MODEL.CHECKPOINT after Stage-2 completes.
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov3_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
  CHECKPOINT: "FILL_IN_AFTER_STAGE2"
  TTA_SCALES:
    - 0.5
    - 0.75
    - 1.0
AUGMENTATION:
  NUM_STEPS_STARTUP: 0
  COPY_PASTE: True
  CONFIDENCE: 0.75
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov3_vitb_k80_12k_stage3"
TRAINING:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "32-true"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 200
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 500
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
SELF_TRAINING:
  ROUND_STEPS: 4000
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 3.3: Create launch script `scripts/run_cups_ablation_a3.sh`**
```bash
#!/bin/bash
# A3: DINOv3 ViT-B/16 — Stage-2 (16K) then Stage-3 (12K)
set -e
REMOTE="santosh@172.17.254.146"
CUPS="~/mbps_panoptic_segmentation/refs/cups"
PYTHON="~/anaconda3/envs/cups/bin/python"
LD="export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH"

echo "=== A3: DINOv3 ViT-B Stage-2 (16K steps) ==="
ssh $REMOTE "cd $CUPS && $LD && nohup $PYTHON train.py \
  --config configs/train_cityscapes_dinov3_vitb_k80_16k_2gpu.yaml \
  > ~/logs/cups_a3_stage2.log 2>&1 & echo PID:\$!"
echo "Monitor: ssh $REMOTE 'tail -f ~/logs/cups_a3_stage2.log'"
```

---

## Task 4: Implement DINOv2 ViT-S Backbone (A4)

**Context:** DINOv2 ViT-S/14 has patch_size=14, embed_dim=384. Its FPN adapter follows the same pattern as ViT-B (embed_dim=768) but with smaller dims. Look at `refs/cups/cups/model/backbone_dinov2_vitb.py` for the pattern to replicate.

**Files:**
- Read: `refs/cups/cups/model/backbone_dinov2_vitb.py` (reference)
- Create: `refs/cups/cups/model/backbone_dinov2_vits.py`
- Modify: `refs/cups/cups/pl_model_pseudo.py` (add `elif backbone_type == "dinov2_vits":`)
- Modify: `refs/cups/cups/pl_model_self.py` (add same elif)
- Create: `refs/cups/configs/train_cityscapes_dinov2_vits_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_dinov2_vits_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a4.sh`

- [ ] **Step 4.1: Read existing DINOv2 ViT-B backbone to understand the pattern**
```bash
cat refs/cups/cups/model/backbone_dinov2_vitb.py
```

- [ ] **Step 4.2: Create `refs/cups/cups/model/backbone_dinov2_vits.py`**

Copy `backbone_dinov2_vitb.py`, change:
- `embed_dim=768` → `embed_dim=384`
- model name: `dinov2_vitb14` → `dinov2_vits14`
- FPN channel sizes: scale by 0.5 (384→192 for intermediate projections)
- Function names: `panoptic_cascade_mask_r_cnn_dinov2` → `panoptic_cascade_mask_r_cnn_dinov2_vits`

Verify the exact channel dims by reading the ViT-B file first (Step 4.1), then mirror the structure.

- [ ] **Step 4.3: Add `dinov2_vits` branch in `pl_model_pseudo.py`**

After line ~525 (after the `elif backbone_type == "dinov3_vitb":` block):
```python
elif backbone_type == "dinov2_vits":
    from cups.model.backbone_dinov2_vits import panoptic_cascade_mask_r_cnn_dinov2_vits
    log.info("Using DINOv2 ViT-S/14 backbone")
    model: nn.Module = panoptic_cascade_mask_r_cnn_dinov2_vits(
        num_classes=config.DATA.NUM_CLUSTERS,
        freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
    )
```

- [ ] **Step 4.4: Add same branch in `pl_model_self.py`** (same pattern, same location ~line 364)

- [ ] **Step 4.5: Create Stage-2 config**
```yaml
# refs/cups/configs/train_cityscapes_dinov2_vits_k80_16k_2gpu.yaml
# A4: DINOv2 ViT-S/14 — Stage-2, 16K steps | fp16 | eff_bs=16
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov2_vits"
  DINOV2_FREEZE: True
  USE_DINO: False
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov2_vits_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 2
  ACCUMULATE_GRAD_BATCHES: 4
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 4.6: Create Stage-3 config template**
```yaml
# refs/cups/configs/train_self_dinov2_vits_k80_12k_2gpu.yaml
# A4: DINOv2 ViT-S/14 — Stage-3, 3 rounds × 4000 = 12K
# Fill in MODEL.CHECKPOINT after Stage-2 completes.
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov2_vits"
  DINOV2_FREEZE: True
  USE_DINO: False
  CHECKPOINT: "FILL_IN_AFTER_STAGE2"
  TTA_SCALES:
    - 0.5
    - 0.75
    - 1.0
AUGMENTATION:
  NUM_STEPS_STARTUP: 0
  COPY_PASTE: True
  CONFIDENCE: 0.75
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov2_vits_k80_12k_stage3"
TRAINING:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 200
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 500
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
SELF_TRAINING:
  ROUND_STEPS: 4000
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 4.7: Create launch script `scripts/run_cups_ablation_a4.sh`**
```bash
#!/bin/bash
# A4: DINOv2 ViT-S/14 — Stage-2 (16K) then Stage-3 (12K)
set -e
REMOTE="santosh@172.17.254.146"
CUPS="~/mbps_panoptic_segmentation/refs/cups"
PYTHON="~/anaconda3/envs/cups/bin/python"
LD="export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH"

echo "=== A4: DINOv2 ViT-S Stage-2 (16K steps) ==="
ssh $REMOTE "cd $CUPS && $LD && nohup $PYTHON train.py \
  --config configs/train_cityscapes_dinov2_vits_k80_16k_2gpu.yaml \
  > ~/logs/cups_a4_stage2.log 2>&1 & echo PID:\$!"
echo "Monitor: ssh $REMOTE 'tail -f ~/logs/cups_a4_stage2.log'"
```

---

## Task 5: Implement DINOv2 ViT-L Backbone (A5)

**Context:** DINOv2 ViT-L/14 has embed_dim=1024, patch_size=14. Larger than ViT-B — fits on 11GB at bs=1.

**Files:**
- Create: `refs/cups/cups/model/backbone_dinov2_vitl.py`
- Modify: `refs/cups/cups/pl_model_pseudo.py` (add `elif backbone_type == "dinov2_vitl":`)
- Modify: `refs/cups/cups/pl_model_self.py`
- Create: `refs/cups/configs/train_cityscapes_dinov2_vitl_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_dinov2_vitl_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a5.sh`

- [ ] **Step 5.1: Create `refs/cups/cups/model/backbone_dinov2_vitl.py`**

Copy `backbone_dinov2_vitb.py`, change:
- `embed_dim=768` → `embed_dim=1024`
- model name: `dinov2_vitb14` → `dinov2_vitl14`
- Function names: `panoptic_cascade_mask_r_cnn_dinov2` → `panoptic_cascade_mask_r_cnn_dinov2_vitl`

- [ ] **Step 5.2: Add `dinov2_vitl` elif branch in `pl_model_pseudo.py` and `pl_model_self.py`**
```python
elif backbone_type == "dinov2_vitl":
    from cups.model.backbone_dinov2_vitl import panoptic_cascade_mask_r_cnn_dinov2_vitl
    log.info("Using DINOv2 ViT-L/14 backbone")
    model: nn.Module = panoptic_cascade_mask_r_cnn_dinov2_vitl(
        num_classes=config.DATA.NUM_CLUSTERS,
        freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
    )
```

- [ ] **Step 5.3: Create Stage-2 config**
```yaml
# refs/cups/configs/train_cityscapes_dinov2_vitl_k80_16k_2gpu.yaml
# A5: DINOv2 ViT-L/14 — Stage-2, 16K steps | fp16 | eff_bs=16
# bs=1 (larger model) with accum=8
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov2_vitl"
  DINOV2_FREEZE: True
  USE_DINO: False
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov2_vitl_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 5.4: Create Stage-3 config template** (same pattern as A4 but `dinov2_vitl`, `bs=1`, `accum=8`)

- [ ] **Step 5.5: Create launch script `scripts/run_cups_ablation_a5.sh`** (same pattern as A4)

---

## Task 6: Implement DINOv3 ViT-S Backbone (A6)

**Context:** DINOv3 ViT-S is not in the official dinov3 repo's standard models. Check `refs/dinov3/` for available model names. Likely uses `dinov3_vits14` variant. Uses fp32 (same as DINOv3 ViT-B) due to MPS/CUDA stability.

**Files:**
- Read: `refs/cups/cups/model/backbone_dinov3_vitb.py` or `model_vitb.py` (wherever DINOv3 ViT-B is defined)
- Create: `refs/cups/cups/model/backbone_dinov3_vits.py`
- Modify: `refs/cups/cups/pl_model_pseudo.py`
- Modify: `refs/cups/cups/pl_model_self.py`
- Create: `refs/cups/configs/train_cityscapes_dinov3_vits_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_dinov3_vits_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a6.sh`

- [ ] **Step 6.1: Find DINOv3 ViT-B implementation and check available model names**
```bash
grep -rn "dinov3_vitb\|vit_small\|ViT-S\|vits14\|vit_base" refs/cups/cups/model/ --include="*.py" | head -20
grep -rn "def .*vit" refs/dinov3/dinov3/ --include="*.py" | head -20
```

- [ ] **Step 6.2: Create `backbone_dinov3_vits.py`**

Load `dinov3_vits14` (or equivalent) from `refs/dinov3/`, build FPN adapter with embed_dim=384, same Cascade Mask R-CNN head as ViT-B. Function name: `panoptic_cascade_mask_r_cnn_dinov3_vits`.

- [ ] **Step 6.3: Add `dinov3_vits` elif branch in `pl_model_pseudo.py` and `pl_model_self.py`**
```python
elif backbone_type == "dinov3_vits":
    from cups.model.backbone_dinov3_vits import panoptic_cascade_mask_r_cnn_dinov3_vits
    log.info("Using DINOv3 ViT-S/14 backbone")
    model: nn.Module = panoptic_cascade_mask_r_cnn_dinov3_vits(
        num_classes=config.DATA.NUM_CLUSTERS,
        freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
    )
```

- [ ] **Step 6.4: Create Stage-2 config**
```yaml
# refs/cups/configs/train_cityscapes_dinov3_vits_k80_16k_2gpu.yaml
# A6: DINOv3 ViT-S/14 — Stage-2, 16K steps | fp32 | eff_bs=16
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_VAL: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/"
  ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80/"
MODEL:
  BACKBONE_TYPE: "dinov3_vits"
  DINOV2_FREEZE: True
  USE_DINO: False
AUGMENTATION:
  NUM_STEPS_STARTUP: 500
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments"
  RUN_NAME: "cups_dinov3_vits_k80_16k_stage2"
TRAINING:
  STEPS: 16000
  BATCH_SIZE: 2
  ACCUMULATE_GRAD_BATCHES: 4
  PRECISION: "32-true"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 1000
  LOG_EVERT_N_STEPS: 100
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 2000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 6.5: Create Stage-3 config template** (same pattern, `dinov3_vits`, `bs=1`, `accum=8`, `32-true`)

- [ ] **Step 6.6: Create launch script `scripts/run_cups_ablation_a6.sh`**

---

## Task 7: Implement DINOv3 ViT-L Backbone (A7)

**Context:** DINOv3 ViT-L/14, embed_dim=1024. At bs=1 fp32 on 11GB 1080 Ti, VRAM may be tight (~10GB). If OOM, try PRECISION="16-mixed" as fallback.

**Files:**
- Create: `refs/cups/cups/model/backbone_dinov3_vitl.py`
- Modify: `refs/cups/cups/pl_model_pseudo.py`
- Modify: `refs/cups/cups/pl_model_self.py`
- Create: `refs/cups/configs/train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml`
- Create: `refs/cups/configs/train_self_dinov3_vitl_k80_12k_2gpu.yaml`
- Create: `scripts/run_cups_ablation_a7.sh`

- [ ] **Step 7.1: Create `backbone_dinov3_vitl.py`**

Load `dinov3_vitl14` from `refs/dinov3/`, FPN embed_dim=1024. Function: `panoptic_cascade_mask_r_cnn_dinov3_vitl`.

- [ ] **Step 7.2: Add `dinov3_vitl` elif branch in `pl_model_pseudo.py` and `pl_model_self.py`**
```python
elif backbone_type == "dinov3_vitl":
    from cups.model.backbone_dinov3_vitl import panoptic_cascade_mask_r_cnn_dinov3_vitl
    log.info("Using DINOv3 ViT-L/14 backbone")
    model: nn.Module = panoptic_cascade_mask_r_cnn_dinov3_vitl(
        num_classes=config.DATA.NUM_CLUSTERS,
        freeze_backbone=getattr(config.MODEL, "DINOV2_FREEZE", True),
    )
```

- [ ] **Step 7.3: Create Stage-2 config** (bs=1, accum=8, 32-true, `dinov3_vitl`)

- [ ] **Step 7.4: Create Stage-3 config template**

- [ ] **Step 7.5: Create launch script `scripts/run_cups_ablation_a7.sh`**

- [ ] **Step 7.6: Test VRAM on 1 batch before full run**
```bash
ssh santosh@172.17.254.146 "
  export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH
  cd ~/mbps_panoptic_segmentation/refs/cups
  ~/anaconda3/envs/cups/bin/python train.py \
    --config configs/train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml \
    --limit_train_batches 2 --max_steps 2
"
```
If OOM: change `PRECISION: "16-mixed"` in the config.

---

## Task 8: Evaluation — After Each Ablation Completes

For each completed ablation (A1-A7), run evaluation immediately after Stage-3 finishes.

- [ ] **Step 8.1: Find best Stage-3 checkpoint**
```bash
# Replace RUN_NAME with actual run name, e.g. cups_resnet50_k80_12k_stage3
ssh santosh@172.17.254.146 "
  ls -lt /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/{RUN_NAME}/*/checkpoints/best_pq*.ckpt
"
```

- [ ] **Step 8.2: Run val evaluation (500 images)**
```bash
ssh santosh@172.17.254.146 "
  export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH
  cd ~/mbps_panoptic_segmentation/refs/cups
  ~/anaconda3/envs/cups/bin/python val.py \
    --config configs/val_cityscapes.yaml \
    --checkpoint /path/to/best_pq_step=XXXXXX.ckpt \
    > ~/logs/eval_{A_ID}_stage3.log 2>&1
  cat ~/logs/eval_{A_ID}_stage3.log | grep -E 'PQ|pq|panoptic'
"
```

- [ ] **Step 8.3: Record results in `reports/cups_backbone_ablation_results.md`**

Template to fill in:

| Ablation | Stage-2 PQ | Stage-3 PQ | Stage-2 PQ_th | Stage-3 PQ_th | Best ckpt step |
|----------|-----------|-----------|--------------|--------------|----------------|
| A1 ResNet-50 | | | | | |
| A2 DINOv2 ViT-B | | | | | |
| A3 DINOv3 ViT-B | | | | | |
| A4 DINOv2 ViT-S | | | | | |
| A5 DINOv2 ViT-L | | | | | |
| A6 DINOv3 ViT-S | | | | | |
| A7 DINOv3 ViT-L | | | | | |

---

## Task 9: Fill in Stage-3 Checkpoint Paths

After each Stage-2 completes, find and fill in the checkpoint path before launching Stage-3.

- [ ] **Step 9.1: Find best Stage-2 checkpoint**
```bash
ssh santosh@172.17.254.146 "
  ls -lt /media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/{RUN_NAME_STAGE2}/*/checkpoints/ | head -5
"
```

- [ ] **Step 9.2: Edit Stage-3 config on remote**
```bash
# Example for A1:
ssh santosh@172.17.254.146 "
  sed -i 's|FILL_IN_AFTER_STAGE2|/media/santosh/Kuldeep/panoptic_segmentation/experiments/experiments/cups_resnet50_k80_16k_stage2/.../checkpoints/best_pq_step=XXXXXX.ckpt|g' \
    ~/mbps_panoptic_segmentation/refs/cups/configs/train_self_resnet50_k80_12k_2gpu.yaml
"
```

- [ ] **Step 9.3: Launch Stage-3**
```bash
ssh santosh@172.17.254.146 "
  export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:\$LD_LIBRARY_PATH
  cd ~/mbps_panoptic_segmentation/refs/cups
  nohup ~/anaconda3/envs/cups/bin/python train_self.py \
    --config configs/train_self_{BACKBONE}_k80_12k_2gpu.yaml \
    > ~/logs/cups_{ABLATION_ID}_stage3.log 2>&1 &
  echo PID:\$!
"
```

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Remote disk OOM (54GB free) | Task 0: delete wandb dirs first; keep only best checkpoints |
| ViT-L OOM on 11GB 1080 Ti (fp32) | Step 7.6: smoke test; fallback to fp16 if needed |
| DINOv3 ViT-S/L model names unknown | Step 6.1: grep refs/dinov3/ before implementing |
| Stage-3 checkpoint path wrong | Step 9.2: verify path exists before sed |
| A1-A3 running serially takes ~7 days | Run A1+A4 in parallel, A2+A5 in parallel (one per GPU pair) |

## Time Estimates

Each ablation (Stage-2 + Stage-3) on 2× GTX 1080 Ti:
- **ResNet-50**: ~8h (Stage-2) + ~6h (Stage-3) = ~14h
- **DINOv2 ViT-S**: ~10h + ~7h = ~17h
- **DINOv2 ViT-B**: ~12h + ~8h = ~20h
- **DINOv2 ViT-L**: ~16h + ~10h = ~26h
- **DINOv3 ViT-S**: ~12h + ~8h = ~20h (fp32 slower)
- **DINOv3 ViT-B**: ~16h + ~10h = ~26h
- **DINOv3 ViT-L**: ~20h + ~12h = ~32h

**Total if serial**: ~155h (~6.5 days). Parallelism is possible since runs are independent — only one run per 2-GPU machine at a time.
