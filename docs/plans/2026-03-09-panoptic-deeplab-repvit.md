# Panoptic-DeepLab + RepViT: Implementation Plan

> **Goal:** Implement 4 lightweight panoptic architectures with RepViT-M0.9 backbone, trained with adapted CUPS stage-2/3 recipe. Full ablation study.

## Architecture Variants

| ID | Architecture | FPN | Instance Method | Est. Params | Key Class |
|----|-------------|-----|-----------------|-------------|-----------|
| A | Panoptic-DeepLab | BiFPN | Center + Offset | ~8-10M | `PanopticDeepLab` |
| B | kMaX-DeepLab | BiFPN | k-means cross-attn | ~10-12M | `KMaXDeepLab` |
| C | MaskConver | BiFPN | Conv centers | ~10-13M | `MaskConver` |
| D | Mask2Former-Lite | BiFPN | Query + mask (2-layer) | ~8M | `Mask2FormerLite` |

## CUPS Training Adaptations (12 components)

### Stage-2 (Pseudo-Label Training)
1. Pixel-level DropLoss for thing classes
2. IGNORE_UNKNOWN_THING_REGIONS
3. Copy-paste augmentation (instance-level)
4. Self-enhanced copy-paste (after 500 steps)
5. Discrete resolution jitter (7 levels)
6. Gradient clipping (L2 norm=1.0)
7. Norm weight decay separation
8. Optimizer: AdamW LR=1e-4, WD=1e-5
9. Cascade gradient scaling (1/K per head)
10. BCE with logits (autocast-safe)

### Stage-3 (Self-Training)
11. EMA teacher-student (decay=0.999)
12. TTA teacher inference (2 scales + flip)
13. Per-class confidence thresholding
14. Frozen backbone, heads-only training
15. 3 rounds x 500 steps, escalating confidence

## Ablation Matrix

### Architecture Ablation (4 runs)
| Run | Architecture | Stage-2 | Stage-3 | GPU |
|-----|-------------|---------|---------|-----|
| A-1 | Panoptic-DeepLab + BiFPN | Full CUPS | Yes | GPU-0 |
| B-1 | kMaX-DeepLab + BiFPN | Full CUPS | Yes | GPU-1 |
| C-1 | MaskConver + BiFPN | Full CUPS | Yes | Sequential |
| D-1 | Mask2Former-Lite + BiFPN | Full CUPS | Yes | Sequential |

### Training Recipe Ablation (on best architecture)
| Run | What Changes | Est. Impact |
|-----|-------------|-------------|
| T-0 | Baseline (CE only, no CUPS tricks) | Reference |
| T-1 | + DropLoss | +0.8-1.2 PQ |
| T-2 | + Copy-paste | +0.8-1.0 PQ |
| T-3 | + Resolution jitter | +0.2-0.3 PQ |
| T-4 | + All stage-2 tricks | Cumulative |
| T-5 | + Stage-3 self-training | +1.0-1.5 PQ |

### FPN Ablation (on best architecture)
| Run | FPN Type | Params |
|-----|----------|--------|
| F-1 | SimpleFPN (current) | 0.2M |
| F-2 | BiFPN | 0.5-1M |
| F-3 | PANet | 1-2M |

Total: ~13 runs, ~65 GPU-hours

## File Structure

```
mbps_pytorch/
  panoptic_deeplab.py          # All 4 architectures + BiFPN
  train_panoptic_deeplab.py    # Training with CUPS adaptations
scripts/
  run_panoptic_deeplab_ablations.sh  # Launch script
```

## Remote Execution

```bash
# On santosh@100.93.203.100:
export LD_LIBRARY_PATH=/home/santosh/anaconda3/envs/cups/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
cd /media/santosh/Kuldeep/panoptic_segmentation

# Primary run:
nohup python mbps_pytorch/train_panoptic_deeplab.py \
    --cityscapes_root datasets/cityscapes \
    --arch panoptic_deeplab \
    --fpn_type bifpn \
    --cups_stage2 --cups_stage3 \
    --num_epochs 50 --batch_size 4 \
    > logs/panoptic_deeplab_bifpn_full.log 2>&1 &
```
