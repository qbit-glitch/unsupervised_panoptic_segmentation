# Unsupervised Mamba-Bridge Panoptic Segmentation (MBPS)

> **NeurIPS 2026 Submission** — First unsupervised panoptic segmentation system 
> using state space models for cross-modal fusion.

## Overview

MBPS tackles the challenge of **unsupervised panoptic segmentation** by bridging
semantic and instance segmentation branches through a novel Mamba2-based
Structured State Space Duality (SSD) fusion mechanism. The entire system operates
without any human annotations.

### Key Innovations

1. **Adaptive Projection Bridge (APB)** — Projects heterogeneous feature streams
   (90-dim semantic codes, 384-dim DINO features) into a shared 192-dim bridge space
2. **Unified Depth Conditioning Module (UDCM)** — FiLM-style feature modulation
   using sinusoidal depth encoding from monocular ZoeDepth estimates
3. **Bidirectional Cross-Modal Scan (BiCMS)** — Mamba2 SSD processing on
   interleaved semantic+instance tokens with learned directional gating
4. **Stuff-Things Classifier (STC)** — Automatic discrimination using
   DBD (Depth Boundary Density), FCC (Feature Cluster Compactness),
   and IDF (Instance Decomposition Frequency) cues

### Architecture

```
Image → DINO ViT-S/8 (frozen) → Features (B, N, 384)
    ├── Semantic Branch: DepthG Head → Codes (B, N, 90)
    │   └── STEGO correspondence loss
    ├── Instance Branch: CutS3D + Cascade → Masks (B, M, N)
    │   └── NCut + LocalCut 3D
    └── Bridge Fusion:
        APB → UDCM → BiCMS (Mamba2) → Inverse Projection
            └── Cross-branch consistency losses
                    └── Stuff-Things → Panoptic Merge
```

## Installation

```bash
# Clone and install
git clone <repository-url>
cd mbps_panoptic_segmentation

# Install dependencies (requires JAX with TPU/GPU support)
pip install -e .

# For TPU:
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Requirements

- Python ≥ 3.9
- JAX/Flax (TPU-optimized)
- TensorFlow (data pipeline only)
- einops, ml_collections, PyYAML
- wandb (optional, for logging)

## Project Structure

```
mbps_panoptic_segmentation/
├── configs/
│   ├── default.yaml              # All hyperparameters
│   ├── cityscapes.yaml           # Cityscapes overrides
│   ├── coco_stuff27.yaml         # COCO-Stuff-27 overrides
│   └── ablations/                # Ablation configs
├── mbps/
│   ├── data/                     # Dataset loading & transforms
│   │   ├── datasets.py           # Cityscapes, COCO, NYU loaders
│   │   ├── transforms.py         # JAX-native augmentations
│   │   ├── tfrecord_utils.py     # TFRecord I/O for TPU
│   │   └── depth_cache.py        # ZoeDepth cache
│   ├── models/
│   │   ├── backbone/             # Frozen DINO ViT-S/8
│   │   ├── semantic/             # DepthG head + STEGO loss
│   │   ├── instance/             # CutS3D + Cascade mask
│   │   ├── bridge/               # APB + UDCM + Mamba2 + BiCMS
│   │   ├── classifier/           # Stuff-Things MLP
│   │   ├── merger/               # Panoptic merge + CRF
│   │   └── mbps_model.py         # Unified model
│   ├── losses/                   # All loss functions
│   ├── training/                 # Curriculum, EMA, self-training
│   └── evaluation/               # PQ, mIoU, visualization
├── scripts/
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Evaluation script
│   ├── precompute_depth.py       # ZoeDepth pre-computation
│   ├── create_tfrecords.py       # TFRecord creation
│   └── run_ablations.py          # Ablation experiments
└── tests/
    └── test_mbps.py              # Unit tests
```

## Quick Start

### 1. Pre-compute Depth Maps

```bash
python scripts/precompute_depth.py \
    --data_dir /path/to/cityscapes/leftImg8bit \
    --output_dir /path/to/depth_cache \
    --dataset cityscapes
```

### 2. Create TFRecords (for TPU)

```bash
python scripts/create_tfrecords.py \
    --config configs/cityscapes.yaml \
    --output_dir /path/to/tfrecords \
    --split train
```

### 3. Convert DINO Weights

```python
from mbps.models.backbone.weights_converter import convert_dino_weights
convert_dino_weights("dino_vits8_pretrain.pth", "dino_vits8_flax.npz")
```

### 4. Train

```bash
# Full training on Cityscapes
python scripts/train.py --config configs/cityscapes.yaml

# Resume from checkpoint
python scripts/train.py --config configs/cityscapes.yaml --resume checkpoints/latest
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/cityscapes.yaml \
    --checkpoint checkpoints/best \
    --use_crf
```

### 6. Run Ablations

```bash
python scripts/run_ablations.py --config configs/cityscapes.yaml
```

## Training Curriculum

Training proceeds through 4 phases with smooth loss weight transitions:

| Phase | Epochs | Components | Key Loss Weights |
|-------|--------|-----------|-----------------|
| A | 1-20 | Semantic only | α=1.0 |
| B | 21-40 | + Instance (gradient projection) | α=1.0, β=0→1 |
| C | 41-60 | Full model (bridge + consistency + PQ) | α=0.8, β=1.0, γ=0.1, δ=0.3, ε=0.2 |
| D | Post | Self-training refinement (3 rounds) | Confidence threshold: 0.7→0.85 |

## Target Performance

| Dataset | PQ | PQ^Th | PQ^St | mIoU |
|---------|-----|-------|-------|------|
| COCO-Stuff-27 | ≥22.5 | ≥15.0 | ≥28.0 | ≥28.0 |
| Cityscapes | ≥25.0 | ≥14.5 | ≥32.0 | ≥20.0 |

## Ablation Studies

| Experiment | Description |
|-----------|-------------|
| `no_mamba` | Replace Mamba2 with concatenation+MLP |
| `no_depth_cond` | Remove depth conditioning module |
| `no_bicms` | Unidirectional scan only |
| `no_consistency` | Remove cross-branch consistency losses |
| `oracle_stuff_things` | Use ground-truth stuff/things labels |

## Testing

```bash
python -m pytest tests/test_mbps.py -v
```

## Key Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Backbone | DINO ViT-S/8 (frozen) | guidelines |
| Semantic dim | 90 | SKILL.md |
| Bridge dim | 192 | SKILL.md |
| Mamba2 layers | 4 per direction | SKILL.md |
| SSM state dim | 64 | SKILL.md |
| Chunk size | 128 (TPU-aligned) | SKILL.md |
| Learning rate | 4e-4 | SKILL.md |
| EMA momentum | 0.999 | SKILL.md |
| Gradient clip | 1.0 | SKILL.md |

## Citation

```bibtex
@article{mbps2026,
    title={Unsupervised Mamba-Bridge Panoptic Segmentation},
    year={2026},
    journal={NeurIPS},
}
```

## License

Research use only.
