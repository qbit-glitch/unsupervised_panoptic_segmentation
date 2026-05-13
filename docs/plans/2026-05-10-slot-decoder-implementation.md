# Depth-Conditioned Slot Attention Decoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a slot attention decoder on DINOv3 features + DepthPro depth to produce instance masks that beat current Sobel+CC pipeline (PQ_things=23.35).

**Architecture:** Depth-FiLM fuses depth geometry into DINOv3 patch features → Slot Attention discovers objects via competitive binding → Spatial Broadcast Decoder reconstructs features → Attention maps = instance masks. 3-phase training: reconstruction → pseudo-label bootstrap → self-training.

**Tech Stack:** PyTorch, pre-extracted DINOv3 ViT-L/16 features (1024-dim), DepthPro depth maps, MPS device (M4 Pro 48GB)

---

## Data Available

| Resource | Path | Shape |
|----------|------|-------|
| DINOv3 features (train) | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features_vitl16/train/{city}/*.npy` | (2048, 1024) |
| DINOv3 features (val) | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features_vitl16/val/{city}/*.npy` | (2048, 1024) |
| DepthPro depth (train) | `/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_depthpro/train/{city}/*.npy` | (512, 1024) |
| DepthPro depth (val) | `/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_depthpro/val/{city}/*.npy` | (512, 1024) |
| Instance pseudo-labels | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_instance_depthpro/train/{city}/*.npz` | masks: (N, 524288) |
| Semantic pseudo-labels (k=80) | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_raw_k80/train/{city}/*.png` | (512, 1024) |

## File Structure

```
mbps_pytorch/
├── models/
│   └── slot_decoder/
│       ├── __init__.py          # Public API + factory
│       ├── depth_film.py        # DepthFiLM conditioning module (~80 lines)
│       ├── slot_attention.py    # Depth-conditioned SlotAttention (~120 lines)
│       ├── decoder.py           # SpatialBroadcastDecoder (~100 lines)
│       └── model.py             # DepthSlotDecoder full model (~150 lines)
├── train_slot_decoder.py        # Training script with 3-phase curriculum (~350 lines)
└── tests/
    └── test_slot_decoder.py     # Unit tests (~150 lines)
```

---

## Task 1: DepthFiLM Module

**Files:** Create `mbps_pytorch/models/slot_decoder/depth_film.py`

The module encodes depth geometry (depth map + Sobel gradients) into FiLM parameters (γ, β) that modulate DINOv3 features per-patch.

**Input:** depth map (B, 512, 1024) → downsample to (B, 32, 64) → flatten to (B, 2048)  
**Process:** sinusoidal encoding (16 freqs) + Sobel grads (2-D) → MLP → γ, β (per-patch, 1024-D each)  
**Output:** modulated features = γ * features + β

---

## Task 2: Depth-Conditioned Slot Attention

**Files:** Create `mbps_pytorch/models/slot_decoder/slot_attention.py`

Extends existing SlotAttention (from train_dinosaur.py) with depth-conditioned key/value projections. Depth information enters via FiLM-modulated features as input.

---

## Task 3: Spatial Broadcast Decoder

**Files:** Create `mbps_pytorch/models/slot_decoder/decoder.py`

Same as existing SBD but adapted for 1024-dim target features. Each slot broadcasts to all positions, adds positional encoding, passes through MLP → (feat_dim + 1).

---

## Task 4: Full Model Assembly

**Files:** Create `mbps_pytorch/models/slot_decoder/model.py`

Combines DepthFiLM + SlotAttention + Decoder into one `nn.Module`. Handles:
- Feature loading and preprocessing
- Depth encoding and FiLM modulation
- Slot attention (returns slots + attention maps)
- Decoder reconstruction
- Instance mask extraction from attention maps

---

## Task 5: Training Script

**Files:** Create `mbps_pytorch/train_slot_decoder.py`

3-phase curriculum:
- **Phase 1 (epochs 1-30):** Pure reconstruction loss (MSE on DINOv3 features)
- **Phase 2 (epochs 31-60):** + pseudo-label matching loss (CE on slot-to-instance assignment)
- **Phase 3 (epochs 61-90):** Self-training (generate masks → filter by confidence → retrain)

---

## Task 6: Tests + Verification

**Files:** Create `mbps_pytorch/tests/test_slot_decoder.py`

Test forward pass, shapes, gradient flow, mask extraction.
