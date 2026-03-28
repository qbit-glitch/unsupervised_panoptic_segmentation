# MBPS Execution Guidelines: Ablations, Proxy Testing & Final Run

> Step-by-step playbook for validating on small proxy datasets, running all ablation studies,
> and executing the final full training runs on Cityscapes and COCO-Stuff-27.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Proxy Dataset Setup](#2-proxy-dataset-setup)
3. [Smoke Tests on Proxy Data](#3-smoke-tests-on-proxy-data)
4. [Running Ablation Studies](#4-running-ablation-studies)
5. [Hyperparameter Sweeps](#5-hyperparameter-sweeps)
6. [Final Full Training Runs](#6-final-full-training-runs)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Results Collection & Reporting](#8-results-collection--reporting)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Proxy smoke tests | 1× GPU (16GB+) | 1× TPU v4-8 |
| Ablation studies | 4× A100 (40GB) | 1× TPU v4-32 |
| Final full run | 4× A100 (80GB) | 1× TPU v4-64 |

### 1.2 Software Setup

```bash
# Clone repo and install
cd /path/to/mbps_panoptic_segmentation
pip install -e .

# Verify JAX sees accelerators
python -c "import jax; print(f'Devices: {jax.devices()}')"

# Verify all imports work
python -c "from mbps.models.mbps_model import MBPSModel; print('OK')"
```

### 1.3 Pre-computed Assets Required

Before any training, ensure these exist:

| Asset | Command | Location |
|-------|---------|----------|
| DINO weights (Flax) | `python -c "from mbps.models.backbone.weights_converter import convert_dino_weights; convert_dino_weights('refs/dino/dino_vits8_pretrain.pth', 'weights/dino_vits8_flax.npz')"` | `weights/dino_vits8_flax.npz` |
| Depth maps | `python scripts/precompute_depth.py --data_dir <IMAGES> --output_dir <DEPTH_DIR>` | Dataset-specific |
| TFRecords (TPU only) | `python scripts/create_tfrecords.py --config <CONFIG> --output_dir <TFRECORD_DIR>` | Dataset-specific |

### 1.4 Unit Tests — Run First

```bash
# Must pass before any training
python -m pytest tests/test_mbps.py -v

# Expected: 20+ tests pass, covering all modules
```

> [!CAUTION]
> Do NOT proceed to training if unit tests fail. Fix issues first.

---

## 2. Proxy Dataset Setup

The proxy dataset strategy uses **small subsets** to validate the full pipeline in minutes rather than days.

### 2.1 Creating Proxy Subsets

#### Cityscapes Proxy (50 images)

```bash
# Create a small proxy split from Cityscapes
mkdir -p data/cityscapes_proxy/{leftImg8bit,gtFine}/{train,val}

# Sample 40 train + 10 val images (from different cities)
cd /data/cityscapes/leftImg8bit/train
for city in aachen bochum bremen; do
    cp -r ${city} /path/to/data/cityscapes_proxy/leftImg8bit/train/
done
# Do the same for gtFine labels and depth maps
```

#### COCO-Stuff-27 Proxy (200 images)

```bash
mkdir -p data/coco_proxy/{images,annotations}/{train2017,val2017}

# Sample first 150 train + 50 val images
ls /data/coco/images/train2017 | head -150 | while read f; do
    cp /data/coco/images/train2017/$f data/coco_proxy/images/train2017/
done
ls /data/coco/images/val2017 | head -50 | while read f; do
    cp /data/coco/images/val2017/$f data/coco_proxy/images/val2017/
done
```

### 2.2 Proxy Config Files

Create `configs/proxy_cityscapes.yaml`:

```yaml
_base_: "default.yaml"

data:
  dataset: "cityscapes"
  data_dir: "data/cityscapes_proxy"
  depth_dir: "data/cityscapes_proxy/depth_zoedepth"
  num_classes: 19
  num_stuff_classes: 11
  num_thing_classes: 8
  image_size: [256, 512]      # Half resolution for speed
  crop_size: [256, 256]

training:
  total_epochs: 6             # 10% of full (60)
  phase_a_end: 2              # Phase A: epochs 1-2
  phase_b_end: 4              # Phase B: epochs 3-4
  # Phase C: epochs 5-6
  self_training_rounds: 1     # 1 round instead of 3
  self_training_epochs_per_round: 1
  batch_size: 2               # Small batch for proxy
  learning_rate: 4.0e-4       # Higher LR for fewer epochs

checkpointing:
  save_every_n_epochs: 2
  checkpoint_dir: "checkpoints/proxy_cityscapes/"

logging:
  log_every_n_steps: 5
  eval_every_n_epochs: 2
  use_wandb: false            # Disable W&B for proxy
```

Create `configs/proxy_coco.yaml` similarly with `total_epochs: 6`, smaller image size, and pointing to the COCO proxy data.

### 2.3 Pre-compute Depth for Proxy

```bash
python scripts/precompute_depth.py \
    --data_dir data/cityscapes_proxy/leftImg8bit \
    --output_dir data/cityscapes_proxy/depth_zoedepth \
    --image_size 256 512
```

---

## 3. Smoke Tests on Proxy Data

### 3.1 Phase A Smoke Test (Semantic Only)

**Goal:** Verify semantic loss decreases over 2 epochs.

```bash
python scripts/train.py --config configs/proxy_cityscapes.yaml --seed 42
```

**What to check:**
- [ ] Training starts without errors
- [ ] `L_semantic` decreases over steps
- [ ] `L_total` ≈ `L_semantic` in Phase A (β=0)
- [ ] Checkpoint saved at epoch 2
- [ ] Memory usage is reasonable (< 16GB on single GPU)

**Expected output (approximate):**
```
Epoch 1: L_total=2.45, L_semantic=2.45, phase=A
Epoch 2: L_total=1.80, L_semantic=1.80, phase=A
```

### 3.2 Phase B Smoke Test (+ Instance)

**Goal:** Verify instance loss kicks in and gradient projection works.

**What to check (epochs 3-4):**
- [ ] `L_instance` appears in logs
- [ ] `L_instance` decreases
- [ ] β ramps up from 0→1 over Phase B epochs
- [ ] No NaN/Inf in any loss

### 3.3 Phase C Smoke Test (Full Model)

**Goal:** Verify bridge, consistency, and PQ losses activate.

**What to check (epochs 5-6):**
- [ ] `L_bridge`, `L_consistency`, `L_pq` all appear
- [ ] No single loss dominates (checks gradient balancing)
- [ ] All loss values are finite and decreasing

### 3.4 Full Pipeline Smoke Test

**Goal:** Verify evaluation, CRF, and checkpoint loading work.

```bash
# Evaluate with CRF post-processing
python scripts/evaluate.py \
    --config configs/proxy_cityscapes.yaml \
    --checkpoint checkpoints/proxy_cityscapes/latest \
    --use_crf \
    --output results/proxy_eval.json

# Verify eval produces valid metrics
cat results/proxy_eval.json
```

**What to check:**
- [ ] Evaluation completes without crash
- [ ] `miou` is a number > 0
- [ ] `per_class_iou` has correct number of classes

### 3.5 Proxy Success Criteria

| Check | Pass Condition |
|-------|---------------|
| No crashes | Full 6-epoch run completes |
| Loss decreases | `L_total` epoch 6 < epoch 1 |
| All losses activate | Bridge/consistency/PQ appear in Phase C |
| No NaN | All logged values finite |
| Checkpoint works | Save + load succeeds |
| Eval runs | `evaluate.py` produces valid JSON |

> [!IMPORTANT]
> Only proceed to ablation studies after ALL proxy checks pass.

---

## 4. Running Ablation Studies

### 4.1 Ablation Study Design

Each ablation removes or replaces ONE component while keeping everything else identical.

| # | Ablation | Config | What Changes |
|---|----------|--------|-------------|
| A1 | **Full Model** | `cityscapes.yaml` | Baseline (nothing removed) |
| A2 | **No Mamba Bridge** | `ablations/no_mamba.yaml` | Mamba2 → concat+MLP fusion |
| A3 | **No Depth Conditioning** | `ablations/no_depth_cond.yaml` | Remove UDCM module |
| A4 | **No Bidirectional Scan** | `ablations/no_bicms.yaml` | Forward-only Mamba scan |
| A5 | **No Consistency Losses** | `ablations/no_consistency.yaml` | Set δ=0 (consistency weight) |
| A6 | **Oracle Stuff-Things** | `ablations/oracle_stuff_things.yaml` | GT stuff/things labels |

### 4.2 Ablation Execution Order

> [!TIP]
> Run ablations on the **proxy dataset first** to catch config issues, then on the full dataset.

#### Step 1: Proxy Ablations (~30 min each)

```bash
# Run each ablation on proxy, sequentially or in parallel on separate devices

# A1: Full model (proxy baseline)
python scripts/train.py --config configs/proxy_cityscapes.yaml \
    --seed 42 2>&1 | tee logs/proxy_full_s42.log

# A2: No Mamba (merge the ablation config manually or via script)
python scripts/train.py --config configs/proxy_cityscapes.yaml \
    --seed 42 2>&1 | tee logs/proxy_no_mamba_s42.log
# NOTE: You must override use_mamba_bridge=false. See §4.3 for how.

# A3-A6: same pattern...
```

#### Step 2: Full Ablations (3 seeds × 6 configs = 18 runs)

```bash
# For each ablation config and each seed:
for SEED in 42 123 456; do
    for ABL in full no_mamba no_depth_cond no_bicms no_consistency oracle_stuff_things; do
        echo "Running ablation=${ABL}, seed=${SEED}"
        python scripts/train.py \
            --config configs/cityscapes.yaml \
            --seed ${SEED} \
            2>&1 | tee logs/ablation_${ABL}_s${SEED}.log
    done
done
```

### 4.3 Config Override Mechanism

To merge ablation overrides with the base dataset config, create a helper script or use the override pattern:

```python
# In scripts/train.py, add CLI flag:
#   --ablation configs/ablations/no_mamba.yaml
# Then in load_config(), merge ablation overrides on top of dataset config.
```

Alternatively, create combined configs:

```bash
# configs/ablation_runs/cityscapes_no_mamba.yaml
_base_: "../cityscapes.yaml"

architecture:
  use_mamba_bridge: false

ablation:
  name: "no_mamba"
```

Create one such file for each `(dataset × ablation)` combination.

### 4.4 Ablation-Specific Instructions

#### A2: No Mamba Bridge
- Set `architecture.use_mamba_bridge: false` in config
- The model falls back to `concat_mlp` fusion (see `MBPSModel.setup()`)
- **Expected impact:** −2 to −4 PQ (bridge is core contribution)

#### A3: No Depth Conditioning
- Set `architecture.use_depth_conditioning: false`
- Bridge still uses Mamba2 but without FiLM modulation from depth
- **Expected impact:** −1 to −2 PQ

#### A4: No Bidirectional Scan
- Set `architecture.mamba.use_bidirectional: false`
- Only forward scan, no reverse + gating
- **Expected impact:** −1 to −2 PQ

#### A5: No Consistency Losses
- Set `loss_weights.delta_consistency: 0.0`
- Uniformity, boundary alignment, and DBC are all disabled
- **Expected impact:** −2 to −3 PQ

#### A6: Oracle Stuff-Things
- Use ground truth stuff/things class assignments instead of learned STC
- **Expected impact:** +1 to +2 PQ (quantifies STC quality)
- This requires GT labels, so only works on supervised validation

### 4.5 Timing Estimates

| Setting | Per-Run Time | Total (18 runs) |
|---------|-------------|-----------------|
| Proxy (6 epochs, 50 images) | ~15-30 min | ~6-9 hours |
| Full Cityscapes (60 epochs + self-training) | ~48-72 hours | ~36-54 GPU-days |
| Full on TPU v4-32 | ~12-18 hours | ~9-13.5 TPU-days |

---

## 5. Hyperparameter Sweeps

### 5.1 Sweep Protocol

Run sweeps on **10% of training data** for fast iteration:

```bash
# Create 10% subset
python scripts/create_subset.py --config configs/cityscapes.yaml --fraction 0.1 \
    --output_dir data/cityscapes_10pct
```

### 5.2 Sweep Configurations

#### Bridge Dimension D_b

| D_b | Config Override | Expected |
|-----|----------------|----------|
| 128 | `architecture.bridge_dim: 128` | Slight PQ drop, faster |
| 192 | default | Baseline |
| 256 | `architecture.bridge_dim: 256` | Marginal gain, slower |
| 384 | `architecture.bridge_dim: 384` | Overfitting risk |

#### Mamba Layers

| Layers | Config Override | Expected |
|--------|----------------|----------|
| 2 | `architecture.mamba.num_layers: 2` | Underfitting |
| 4 | default | Baseline |
| 6 | `architecture.mamba.num_layers: 6` | Marginal gain |
| 8 | `architecture.mamba.num_layers: 8` | Diminishing returns |

#### Loss Weight Sensitivity (±50%)

```bash
# Example: sweep lambda_depthg
for WEIGHT in 0.15 0.3 0.45; do
    python scripts/train.py --config configs/proxy_cityscapes.yaml \
        2>&1 | tee logs/sweep_depthg_${WEIGHT}.log
done
```

Key weights to sweep: `alpha_semantic`, `beta_instance`, `gamma_bridge`, `delta_consistency`, `epsilon_pq`.

### 5.3 Sweep Results Format

Record results in `results/sweeps/`:

```
results/sweeps/
├── bridge_dim_sweep.json
├── mamba_layers_sweep.json
├── loss_weight_sweep.json
└── summary.md
```

---

## 6. Final Full Training Runs

### 6.1 Pre-Flight Checklist

Before launching the full run, verify:

- [ ] All proxy smoke tests pass (§3)
- [ ] At least one ablation completes on proxy without errors (§4)
- [ ] Full dataset is available and pre-processed (depth maps, TFRecords)
- [ ] DINO Flax weights are converted and loadable
- [ ] Sufficient disk space for checkpoints (~2GB per checkpoint × top-3)
- [ ] W&B is configured (or disabled if not using it)
- [ ] Nohup/tmux/screen session ready for long runs

### 6.2 Cityscapes Full Run

```bash
# Launch in background with logging
nohup python scripts/train.py \
    --config configs/cityscapes.yaml \
    --seed 42 \
    2>&1 | tee logs/cityscapes_full_s42.log &

# Monitor
tail -f logs/cityscapes_full_s42.log
```

**Training schedule (60 epochs + self-training):**

| Epochs | Phase | Active Losses | Key Events |
|--------|-------|--------------|------------|
| 1-5 | A (warmup) | L_semantic | LR warmup 0→1e-4 |
| 6-20 | A | L_semantic | Semantic clusters forming |
| 21-25 | B (ramp) | L_semantic + L_instance (β: 0→0.5) | Instance head activates gradually |
| 26-40 | B | L_semantic + L_instance | Gradient projection active |
| 41-45 | C (ramp) | All losses (γ,δ,ε ramping) | Bridge + consistency activate |
| 46-60 | C | All losses at full weight | Full model convergence |
| 61-75 | D | All + pseudo-labels | 3 self-training rounds (5 epochs each) |

**Monitoring checkpoints during training:**

```bash
# Check loss at epoch 20 (end of Phase A):
grep "Epoch 20" logs/cityscapes_full_s42.log
# L_semantic should be < 1.0

# Check loss at epoch 40 (end of Phase B):
grep "Epoch 40" logs/cityscapes_full_s42.log
# L_instance should be < 0.5

# Check loss at epoch 60 (end of Phase C):
grep "Epoch 60" logs/cityscapes_full_s42.log
# All losses should be decreasing
```

### 6.3 COCO-Stuff-27 Full Run

```bash
nohup python scripts/train.py \
    --config configs/coco_stuff27.yaml \
    --seed 42 \
    2>&1 | tee logs/coco_stuff27_full_s42.log &
```

### 6.4 Multi-Seed Runs

For final reported numbers, run 3 seeds and report mean ± std:

```bash
for SEED in 42 123 456; do
    for DATASET in cityscapes coco_stuff27; do
        nohup python scripts/train.py \
            --config configs/${DATASET}.yaml \
            --seed ${SEED} \
            2>&1 | tee logs/${DATASET}_full_s${SEED}.log &
    done
done
```

---

## 7. Evaluation Protocol

### 7.1 Standard Evaluation

```bash
# Cityscapes
python scripts/evaluate.py \
    --config configs/cityscapes.yaml \
    --checkpoint checkpoints/cityscapes/best \
    --use_crf \
    --output results/cityscapes_eval.json

# COCO-Stuff-27
python scripts/evaluate.py \
    --config configs/coco_stuff27.yaml \
    --checkpoint checkpoints/coco_stuff27/best \
    --use_crf \
    --output results/coco_stuff27_eval.json
```

### 7.2 Reported Metrics

Each evaluation must report:

| Metric | Description | Target (Cityscapes) | Target (COCO-Stuff-27) |
|--------|-------------|---------------------|------------------------|
| **PQ** | Panoptic Quality | ≥ 25.0 (28.0 with self-training) | ≥ 22.5 |
| **PQ^Th** | PQ for thing classes | ≥ 14.5 | ≥ 15.0 |
| **PQ^St** | PQ for stuff classes | ≥ 32.0 | ≥ 28.0 |
| **SQ** | Segmentation Quality | Report | Report |
| **RQ** | Recognition Quality | Report | Report |
| **mIoU** | Mean IoU (Hungarian) | ≥ 20.0 | ≥ 28.0 |
| **Accuracy** | Pixel accuracy | Report | Report |

### 7.3 Evaluation with and without CRF

Run evaluation **both ways** to quantify CRF benefit:

```bash
# Without CRF
python scripts/evaluate.py --config configs/cityscapes.yaml \
    --checkpoint checkpoints/cityscapes/best \
    --output results/cityscapes_no_crf.json

# With CRF
python scripts/evaluate.py --config configs/cityscapes.yaml \
    --checkpoint checkpoints/cityscapes/best \
    --use_crf \
    --output results/cityscapes_with_crf.json
```

---

## 8. Results Collection & Reporting

### 8.1 Directory Structure

```
results/
├── proxy/                    # Proxy smoke test results
│   ├── proxy_eval.json
│   └── proxy_train.log
├── ablations/                # All ablation results
│   ├── cityscapes/
│   │   ├── full_s42.json
│   │   ├── full_s123.json
│   │   ├── full_s456.json
│   │   ├── no_mamba_s42.json
│   │   ├── ...
│   │   └── summary.md
│   └── coco_stuff27/
│       └── ...
├── sweeps/                   # Hyperparameter sweep results
│   ├── bridge_dim_sweep.json
│   └── ...
├── final/                    # Final full training results
│   ├── cityscapes_eval.json
│   ├── coco_stuff27_eval.json
│   └── summary_table.md
└── visualizations/           # Qualitative examples
    ├── cityscapes/
    └── coco_stuff27/
```

### 8.2 Ablation Results Table Template

After all ablation runs complete, compile into this table:

```markdown
| Experiment | PQ↑ | PQ^Th↑ | PQ^St↑ | mIoU↑ | ΔPQ |
|------------|-----|--------|--------|-------|-----|
| Full Model | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | XX.X ± X.X | — |
| − Mamba Bridge | XX.X ± X.X | | | | −X.X |
| − Depth Cond. | XX.X ± X.X | | | | −X.X |
| − BiCMS | XX.X ± X.X | | | | −X.X |
| − Consistency | XX.X ± X.X | | | | −X.X |
| + Oracle ST | XX.X ± X.X | | | | +X.X |
```

### 8.3 Expected Ablation Outcomes

If results deviate significantly from these expectations, investigate:

| Ablation | Expected ΔPQ | If ΔPQ ≈ 0 | If ΔPQ too large |
|----------|-------------|------------|------------------|
| − Mamba Bridge | −2 to −4 | The bridge isn't helping → debug BiCMS | The bridge is carrying too much → check base heads |
| − Depth Cond. | −1 to −2 | Depth features weak → check ZoeDepth quality | Depth overfitting → reduce depth loss weight |
| − BiCMS | −1 to −2 | Reverse scan adds little → OK, still report | Asymmetric scan issues → check reverse padding |
| − Consistency | −2 to −3 | Losses are too weak → increase δ | Regularization too strong → decrease δ |
| + Oracle ST | +1 to +2 | STC already good → great result | STC very poor → improve cue features |

---

## 9. Troubleshooting

### 9.1 Common Training Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **OOM** | CUDA/TPU out-of-memory | Reduce `batch_size` or `image_size` |
| **NaN loss** | Loss becomes NaN | Reduce `learning_rate`, increase `gradient_clip_norm` |
| **Loss plateau** | Loss stops decreasing | Check phase transitions, verify β/γ/δ ramp-up |
| **Phase B instability** | Loss spikes when instance kicks in | Slow down β ramp-up (extend Phase B), reduce initial instance LR |
| **Mamba memory** | SSD kernel OOM | Reduce `mamba_chunk_size` to 64, or reduce `mamba_state_dim` to 32 |
| **Slow training** | < 1 img/sec | Check TFRecord pipeline, increase `prefetch_buffer`, verify TPU utilization |

### 9.2 Debugging Gradient Issues

```python
# Add to training loop for diagnosis:
grad_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
for name, norm in jax.tree.leaves_with_path(grad_norms):
    if float(norm) > 100 or float(norm) < 1e-8:
        print(f"WARNING: {name} grad norm = {norm}")
```

### 9.3 Phase Transition Verification

Check that phase transitions happen at correct epochs:

```python
from mbps.training.curriculum import TrainingCurriculum
c = TrainingCurriculum(phase_a_end=20, phase_b_end=40, total_epochs=60)
for e in [1, 20, 21, 40, 41, 60]:
    config = c.get_config(e)
    print(f"Epoch {e}: Phase {c.get_phase(e)}, β={config.beta:.2f}, "
          f"bridge={config.use_bridge}")
```

**Expected:**
```
Epoch 1:  Phase A, β=0.00, bridge=False
Epoch 20: Phase A, β=0.00, bridge=False
Epoch 21: Phase B, β=0.05, bridge=False
Epoch 40: Phase B, β=1.00, bridge=False
Epoch 41: Phase C, β=1.00, bridge=True
Epoch 60: Phase C, β=1.00, bridge=True
```

---

## Appendix: Quick Reference Commands

```bash
# ─── PROXY ───────────────────────────────────────
# Proxy smoke test
python scripts/train.py --config configs/proxy_cityscapes.yaml

# Proxy evaluation
python scripts/evaluate.py --config configs/proxy_cityscapes.yaml \
    --checkpoint checkpoints/proxy_cityscapes/latest

# ─── ABLATIONS ───────────────────────────────────
# Run all ablations (sequential)
python scripts/run_ablations.py --config configs/cityscapes.yaml

# Run specific ablation
python scripts/run_ablations.py --config configs/cityscapes.yaml \
    --ablations no_mamba no_depth_cond

# ─── FULL TRAINING ───────────────────────────────
# Cityscapes (background)
nohup python scripts/train.py --config configs/cityscapes.yaml --seed 42 &

# COCO-Stuff-27 (background)
nohup python scripts/train.py --config configs/coco_stuff27.yaml --seed 42 &

# ─── EVALUATION ──────────────────────────────────
# Evaluate best checkpoint with CRF
python scripts/evaluate.py --config configs/cityscapes.yaml \
    --checkpoint checkpoints/cityscapes/best --use_crf

# ─── UNIT TESTS ──────────────────────────────────
python -m pytest tests/test_mbps.py -v
```
