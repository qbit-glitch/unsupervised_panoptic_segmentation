# NeurIPS Experiment Launch Plan
# Deadline: May 6, 2026 (14 days remaining)
# Author: Qbit

## Remote Setup

Primary training server:
- Host: santosh@172.17.254.146
- Code: /home/santosh/cups/
- Datasets: /home/santosh/datasets/
- GPUs: 2x GTX 1080 Ti
- Conda env: cups
- Disk: 231GB free

## Already-Available Configs (on remote)

### Stage 2 (Initial Training)
| Backbone | Config Path |
|---|---|
| DINOv3 ViT-B/16 | `configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml` |
| DINOv3 ViT-L/16 | `configs/train_cityscapes_dinov3_vitl_k80_16k_2gpu.yaml` |
| DINOv2 ViT-B/16 | `configs/train_cityscapes_vitb_v3.yaml` |
| DINO ResNet-50 | `configs/train_cityscapes_resnet50_k80.yaml` |

### Stage 3 (Self-Training)
| Backbone | Config Path |
|---|---|
| DINOv3 ViT-B/16 | `configs/train_self_cityscapes_dinov3_vitb_k80_2gpu.yaml` |
| DINOv3 ViT-L/16 | `configs/train_self_cityscapes_dinov3_vitl_k80_2gpu.yaml` |
| DINO ResNet-50 | `configs/train_self_cityscapes_resnet50_k80_2gpu.yaml` |

## Priority 1: Seed Robustness (Day 1-2)

Run 2 additional seeds for your main DINOv3 ViT-B/16 result.

### Stage 2 - Seed 43
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed43_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments"
EOF
```

### Stage 2 - Seed 44
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  SYSTEM.SEED 44 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed44_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments"
EOF
```

### Stage 3 (Self-Training) - Seed 43
After Stage 2 seed 43 completes:
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train_self.py \
  --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  SYSTEM.SEED 43 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed43_stage3" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  MODEL.CHECKPOINT "experiments/dinov3_vitb_seed43_stage2/checkpoints/last.ckpt"
EOF
```

### Stage 3 (Self-Training) - Seed 44
After Stage 2 seed 44 completes:
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train_self.py \
  --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  SYSTEM.SEED 44 \
  SYSTEM.RUN_NAME "dinov3_vitb_seed44_stage3" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  MODEL.CHECKPOINT "experiments/dinov3_vitb_seed44_stage2/checkpoints/last.ckpt"
EOF
```

---

## Priority 2: Self-Training Scaling (Day 1-5)

Add DINOv2 ViT-B/16 to establish the trend. DINOv2 isolates pretraining data (same era as DINOv2 ResNet-50) vs. architecture.

### Stage 2 - DINOv2 ViT-B/16
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train.py \
  --experiment_config_file configs/train_cityscapes_vitb_v3.yaml \
  SYSTEM.SEED 1996 \
  SYSTEM.RUN_NAME "dinov2_vitb_seed1996_stage2" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments"
EOF
```

### Stage 3 - DINOv2 ViT-B/16
After Stage 2 completes:
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /home/santosh/cups
conda activate cups
python train_self.py \
  --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_k80_2gpu.yaml \
  SYSTEM.SEED 1996 \
  MODEL.BACKBONE_TYPE "dinov2_vitb" \
  MODEL.DINOV2_FREEZE True \
  SYSTEM.RUN_NAME "dinov2_vitb_seed1996_stage3" \
  SYSTEM.LOG_PATH "/home/santosh/cups/experiments" \
  MODEL.CHECKPOINT "experiments/dinov2_vitb_seed1996_stage2/checkpoints/last.ckpt"
EOF
```

**Note:** The self-training config may need adjustment for DINOv2. Check if `train_self_cityscapes_vitb_local.yaml` exists, or modify the DINOv3 self-training config to use `dinov2_vitb` backbone.

---

## Priority 3: Oracle Upper Bound (Day 3-5)

Train with ground-truth labels to determine headroom.

### Config to Create

Create `configs/oracle_gt_cityscapes_dinov3_vitb.yaml`:

```yaml
DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  ROOT: "/home/santosh/cups/datasets/cityscapes/"
  ROOT_VAL: "/home/santosh/cups/datasets/cityscapes/"
  # Use GT labels instead of pseudo-labels
  ROOT_PSEUDO: "/home/santosh/cups/datasets/cityscapes/gtFine/train/"
  NUM_CLASSES: 27
MODEL:
  BACKBONE_TYPE: "dinov3_vitb"
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
  NUM_WORKERS: 4
  LOG_PATH: "/home/santosh/cups/experiments"
  RUN_NAME: "oracle_gt_dinov3_vitb"
  SEED: 1996
TRAINING:
  STEPS: 8000
  BATCH_SIZE: 1
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 500
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
VALIDATION:
  CACHE_DEVICE: "cpu"
```

**Important:** The pseudo-label dataset loader may not support GT labels directly. You may need to:
1. Convert GT panoptic labels to the same format as pseudo-labels
2. Or modify `PseudoLabelDataset` to load GT when available

### Launch Command
```bash
ssh santosh@172.17.254.146 << 'EOF'
cd /media/santosh/Kuldeep/panoptic_segmentation
conda activate cups
python train.py \
  --experiment_config_file configs/oracle_gt_cityscapes_dinov3_vitb.yaml
EOF
```

**Alternative if GT format conversion is hard:** Use your existing DINOv3 Stage 2 config but train for longer (16k steps instead of 8k) to see if performance saturates. This gives a proxy for capacity ceiling.

---

## Parallel Execution Strategy

With 2 GPUs, you can run 2 experiments in parallel. Suggested allocation:

| Time | GPU 0 | GPU 1 |
|---|---|---|
| Days 1-2 | Stage 2 Seed 43 | Stage 2 Seed 44 |
| Days 3-4 | Stage 2 DINOv2 ViT-B | Stage 3 Seed 43 (after S2 finishes) |
| Days 5-6 | Stage 3 Seed 44 | Stage 3 DINOv2 (after S2 finishes) |
| Days 7-8 | Oracle GT (if ready) | - |

---

## Monitoring Commands

Tail logs remotely:
```bash
ssh santosh@172.17.254.146 "tail -f /home/santosh/cups/experiments/*/logs/*.log"
```

Check GPU usage:
```bash
ssh santosh@172.17.254.146 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 5"
```

List running experiments:
```bash
ssh santosh@172.17.254.146 "ls -lt /home/santosh/cups/experiments/ | head -10"
```

---

## Expected Timeline

| Day | Deliverable |
|---|---|
| 1 | Launch Stage 2 seeds 43, 44 |
| 2 | Launch Stage 2 DINOv2 ViT-B |
| 3 | Launch Stage 3 Seed 43 |
| 4 | Launch Stage 3 Seed 44 |
| 5 | Launch Stage 3 DINOv2 ViT-B |
| 6 | Oracle GT experiment (if feasible) |
| 7-8 | Collect results, compute mean±std |
| 9-10 | Update paper tables, write theory section |
| 11-12 | Revise introduction, abstract |
| 13-14 | Final editing, figure polish, submit |

---

## Next Steps

1. **Confirm GPU availability** on the remote server
2. **Create the oracle GT config** (or decide on the proxy approach)
3. **Launch the first two seed experiments today**
4. While those run, I'll draft the self-training scaling theory section for the paper
