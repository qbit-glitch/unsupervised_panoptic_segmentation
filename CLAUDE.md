# MBPS: Mamba-Bridge Panoptic Segmentation

## Project Overview

MBPS is an unsupervised panoptic segmentation model targeting NeurIPS 2026. It fuses DepthG (depth-guided semantic segmentation) with CutS3D (3D-aware instance segmentation) through a Mamba2 state-space bridge. All training runs on Google Cloud TPU v4/v5e VMs using JAX/Flax.

**Key metric targets:**
- Cityscapes: PQ >= 25.0 (28.0 with self-training)
- COCO-Stuff-27: PQ >= 22.5

## Architecture

```
DINO ViT-S/8 backbone (frozen, 384-dim)
    |
    +-> DepthG semantic head -> 90-dim code space -> semantic labels
    |
    +-> Adaptive Projection Bridge (384 -> 192-dim)
    |       |
    |       +-> Depth Conditioning (FiLM)
    |       +-> Mamba2 SSD (bidirectional cross-modal scan)
    |       |
    +-> CutS3D instance head -> NCut + LocalCut 3D -> instance masks
    |
    +-> Stuff-Things Classifier (DBD + FCC + IDF cues)
    |
    +-> Panoptic Merge + CRF post-processing -> final panoptic map
```

## Quick Reference: Key Paths

| Resource | Path |
|----------|------|
| Project root | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/` |
| Model code | `mbps/models/mbps_model.py` |
| Training loop | `mbps/training/trainer.py` |
| Orchestrator | `scripts/orchestrate.py` |
| GCS bucket | `gs://mbps-panoptic` |
| Configs | `configs/` |
| Guidelines | `guidelines/` |

## GCS Bucket Structure

```
gs://mbps-panoptic/
  datasets/
    cityscapes/
      tfrecords/{train,val}/     # Sharded TFRecords
      depth_zoedepth/            # Precomputed depth maps (.npy)
      leftImg8bit/               # Raw images
      gtFine/                    # Ground truth
    coco/
      tfrecords/{train,val}/
      depth_zoedepth/
      images/
  checkpoints/
    {experiment_name}/{vm_name}/checkpoint_epoch_XXXX/
  results/
    {experiment_name}/{vm_name}/
  weights/
    dino_vits8_flax.npz          # Converted DINO weights
  logs/
```

## TPU Quota (TRC Grant - 30 days)

| Accelerator | Type | Zone | Chips | VMs (8-chip) |
|-------------|------|------|-------|--------------|
| v4-8 | On-demand | us-central2-b | 32 | 4 |
| v4-8 | Spot | us-central2-b | 32 | 4 |
| v5e-8 | Spot | us-central1-a | 64 | 8 |
| **Total** | | | **128** | **16** |

No on-demand v5e quota. All v5e must be spot.

## Experiment Matrix: 36 Runs in 3 Waves

- **2 datasets**: Cityscapes, COCO-Stuff-27
- **6 configurations**: full + 5 ablations (no_mamba, no_depth_cond, no_bicms, no_consistency, oracle_stuff_things)
- **3 seeds**: 42, 123, 456
- **Training**: 60 epochs (phases A/B/C) + 15 self-training epochs = 75 total
- **Wave 1**: 16 jobs (full runs + seed-42 ablations)
- **Wave 2**: 16 jobs (seed 123/456 ablations)
- **Wave 3**: 4 jobs (remaining seed 456 ablations)

---

## For Claude Code Agents: How to Run

### Step 0: One-Time Data Setup (run on mbps-v4-0)

Data must be on GCS before any training. The bucket structure is already created.

```bash
# On a VM with internet access:
bash scripts/setup_data_pipeline.sh all

# Or individually:
bash scripts/setup_data_pipeline.sh weights      # DINO weights
bash scripts/setup_data_pipeline.sh cityscapes    # ~20 GB
bash scripts/setup_data_pipeline.sh coco          # ~25 GB
bash scripts/setup_data_pipeline.sh verify        # Check GCS contents
```

**Cityscapes requires credentials:** Set `CITYSCAPES_USERNAME` and `CITYSCAPES_PASSWORD` env vars (register at https://www.cityscapes-dataset.com/register/).

### Step 1: The Orchestrator (Single Command)

The orchestrator manages the full lifecycle across 16 VMs:

```bash
# Full pipeline: smoke test -> create VMs -> setup -> train -> monitor
python scripts/orchestrate.py --phase all

# Dry run first (always recommended):
python scripts/orchestrate.py --phase all --dry_run

# Smoke test only (validates pipeline on 2 VMs with proxy data):
python scripts/orchestrate.py --phase smoke

# Skip smoke test if it already passed:
python scripts/orchestrate.py --phase all --skip_smoke
```

### Step 2: Individual Phases (if needed)

```bash
python scripts/orchestrate.py --phase create     # Create 16 TPU VMs
python scripts/orchestrate.py --phase setup      # Rsync code + install deps
python scripts/orchestrate.py --phase launch     # Launch all 3 waves
python scripts/orchestrate.py --phase launch --waves 1   # Wave 1 only
python scripts/orchestrate.py --phase monitor    # Live status dashboard
python scripts/orchestrate.py --phase status     # One-shot status check
python scripts/orchestrate.py --phase cleanup --force    # Delete all VMs
```

### Step 3: Training on a Single VM (manual)

If you need to run training directly on a VM:

```bash
# Full Cityscapes training
python3 scripts/train.py \
    --config configs/cityscapes_gcs.yaml \
    --seed 42 \
    --vm_name mbps-v4-0 \
    --experiment cityscapes_full

# Ablation
python3 scripts/train.py \
    --config configs/cityscapes_gcs.yaml \
    --ablation configs/ablations/no_mamba.yaml \
    --seed 42 \
    --vm_name mbps-v4-3 \
    --experiment cityscapes_ablation_no_mamba

# Resume from checkpoint
python3 scripts/train.py \
    --config configs/cityscapes_gcs.yaml \
    --seed 42 \
    --vm_name mbps-v4-0 \
    --experiment cityscapes_full \
    --resume gs://mbps-panoptic/checkpoints/cityscapes_full/mbps-v4-0/checkpoint_epoch_0035
```

### Step 4: Evaluation

```bash
python3 scripts/evaluate.py \
    --config configs/cityscapes_gcs.yaml \
    --checkpoint gs://mbps-panoptic/checkpoints/cityscapes_full/mbps-v4-0/checkpoint_epoch_0075
```

### Step 5: Coordinator (aggregates results from all workers)

The orchestrator auto-launches this on v4-0 after wave 1 starts. To run manually:

```bash
python3 scripts/coordinate.py \
    --config configs/cityscapes_gcs.yaml \
    --experiment cityscapes_full
```

---

## Training Curriculum (4 Phases)

| Phase | Epochs | Active Losses | Key Events |
|-------|--------|--------------|------------|
| A (Semantic) | 1-20 | L_semantic (STEGO + DepthG) | LR warmup (1-5), semantic clusters form |
| B (Instance) | 21-40 | + L_instance (mask coherence) | Instance loss ramps via beta, gradient projection |
| C (Bridge) | 41-60 | + L_bridge + L_consistency + L_pq | Full model convergence |
| D (Self-train) | 61-75 | Pseudo-label training (3 rounds x 5 epochs) | EMA teacher, confidence thresholding |

## Loss Components

- `L_semantic` = alpha * (L_stego + lambda_depthg * L_depthg)
- `L_instance` = beta * (L_drop + lambda_box * L_box)
- `L_bridge` = gamma * (lambda_recon * L_recon + lambda_cka * L_cka + lambda_state * L_state)
- `L_consistency` = delta * (lambda_uniform * L_uniform + lambda_boundary * L_boundary + lambda_dbc * L_dbc)
- `L_pq` = epsilon * L_pq_proxy

---

## Ablation Configs

| Ablation | Config Override | What Changes |
|----------|----------------|-------------|
| no_mamba | `configs/ablations/no_mamba.yaml` | Replace Mamba2 bridge with MLP fusion |
| no_depth_cond | `configs/ablations/no_depth_cond.yaml` | Disable depth conditioning (FiLM) |
| no_bicms | `configs/ablations/no_bicms.yaml` | Forward-only Mamba (no bidirectional scan) |
| no_consistency | `configs/ablations/no_consistency.yaml` | Set delta=0 (no cross-branch losses) |
| oracle_stuff_things | `configs/ablations/oracle_stuff_things.yaml` | Use GT stuff/things labels |

---

## Troubleshooting

### VM won't create
- Check quota: `gcloud compute tpus tpu-vm list --zone=us-central2-b`
- v5e are spot-only; ensure `--spot` flag is set
- Quota is 4 on-demand v4-8 + 4 spot v4-8 + 8 spot v5e-8

### Training crashes on start
- Verify JAX: `python3 -c "import jax; print(jax.device_count())"` (should show 4 for v4-8)
- Verify GCS: `gsutil ls gs://mbps-panoptic/datasets/`
- Check logs: `tail -100 ~/mbps_panoptic_segmentation/logs/<job_id>.log`

### Spot VM preempted
- The orchestrator auto-recovers: delete -> recreate -> setup -> resume from GCS checkpoint
- Manual recovery: create VM, run `--phase setup`, then `train.py --resume <latest_checkpoint>`

### NaN loss
- Check gradient clipping is enabled (default: 1.0)
- Phase B/C transitions can be unstable; check curriculum scheduling
- See `guidelines/guidelines_for_execution_of_ablations_and_final_run.md` Section 9

### OOM on TPU
- Reduce batch size in config (default: 8 per device)
- v4-8 has 32GB HBM per chip; v5e-8 has 16GB per chip

---

## Code Style and Conventions

- **Framework**: JAX/Flax with `nn.Module` using `setup()` (not `@nn.compact`)
- **Data parallelism**: `jax.pmap` across TPU cores, `lax.pmean` for gradient averaging
- **File I/O**: Always use `tf.io.gfile` for GCS-compatible reads/writes
- **Checkpointing**: `_save_npy()` / `_load_npy()` via BytesIO + tf.io.gfile
- **Config**: YAML with deep merge (`default.yaml` + dataset config + optional ablation config)
- **Logging**: W&B for training metrics, absl.logging for console/file output
- **Tests**: pytest in `tests/`, run with `python -m pytest tests/ -v`

## Key Files for Modification

If you need to change training behavior, these are the critical files:

1. `mbps/models/mbps_model.py` - Model architecture (MBPSModel class)
2. `mbps/training/trainer.py` - Training loop + W&B logging
3. `mbps/training/curriculum.py` - Phase A/B/C/D loss weight scheduling
4. `mbps/losses/` - All loss functions
5. `configs/default.yaml` - Base hyperparameters
6. `scripts/train.py` - CLI entry point for training
7. `scripts/orchestrate.py` - Multi-VM orchestration

## Current Pipeline Status (March 2026)

### Stage 1: Pseudo-Label Generation (COMPLETE)
- **Best pseudo-labels**: k=80 overclustering + depth-guided splitting (τ=0.20, A_min=1000)
  - PQ=26.74, PQ_stuff=32.08, PQ_things=19.41 (beats CUPS 17.70)
  - Gap to CUPS (27.8) is only 1.06 PQ, entirely from PQ_stuff
- **Pseudo-label locations**:
  - Raw k=80 clusters: `pseudo_semantic_raw_k80/{train,val}/` (values 0-79)
  - Centroids: `pseudo_semantic_raw_k80/kmeans_centroids.npz`
  - Depth-guided instances: `pseudo_instance_spidepth/` (with τ=0.20, A_min=1000)
- **CSCMRefineNet** (semantic refinement, Conv2d blocks): PQ=21.87 on CAUSE-CRF labels
  - Best checkpoint: `checkpoints/refine_net/best.pth` (1.83M params)
  - Conv2d blocks >> Mamba blocks (11.5x faster, +0.53 PQ better)
- **JointRefineNet** (semantic + instance): PQ=21.96 on CAUSE-CRF labels
  - Instance boundary head failed (PQ_things_bnd=2.74 vs CC-based 10.24)

### CSCMRefineNet on k=80 — Hyperparameter Sweep (COMPLETE)
- **80-class training FAILED**: changed_pct=32%, PQ degraded 26.74→25.39
- **19-class mapped training**: changed_pct=6.2%, PQ_stuff improved 32.08→33.38 (+1.3), mIoU +5%
- **Best baseline (Run D, ep16)**: PQ=26.52, PQ_stuff=33.38, PQ_things=17.10, mIoU=55.31
- **Problem**: PQ_things regresses 19.41→17.10 across all configs (structural, not hyperparameter)
- **Root cause**: 32×64 resolution too coarse for things; uniform refinement; boundary-agnostic loss
- Checkpoints: `checkpoints/refine_net_k80_19cls_conservative/best.pth`

### CSCMRefineNet Ablation Study (Approach A+) — COMPLETE, NO IMPROVEMENT
- TAD, BPL, ASPP-lite, TAD+BPL all failed to beat Run D baseline (PQ=26.52)
- 32×64 resolution confirmed as fundamental bottleneck
- Report: `reports/cscmrefinenet_k80_ablation.md`

### DepthGuidedUNet Ablation Study (Semantics — COMPLETE)
- **Architecture**: `DepthGuidedUNet` in `refine_net.py` — progressive multi-stage decoder (32×64 → 128×256+) with depth Sobel skip connections
- **Training**: `mbps_pytorch/train_refine_net.py` with `--model_type unet`
- **Best result: P2-B 2-stage attention: PQ=28.00 (ep8) — BEATS CUPS 27.8!** PQ_stuff=35.04, PQ_things=18.32, mIoU=57%, 5.45M params.
- Phase 1 (Loss): Focal γ=1.0 best (+0.12 PQ). Phase 2 (Arch): Attention >> conv. Phase 3 (Grad Accum): Implemented, not run.
- **Reports**: `reports/unet_ablation_study.md`, `reports/unet_phase2_architecture_ablation.md`, `reports/unet_unified_ablation_study.md`

### Instance-Conditioned UNet (IC-C) — PIVOTED AWAY
- Added InstanceSkipBlock (boundary+distance features injected via skip connections) + instance uniformity loss
- Implementation complete in `refine_net.py` and `train_refine_net.py` (--use_instance, --inst_skip_dim, etc.)
- **Pivoted**: Semantics already strong (PQ_stuff=35.04). Bottleneck is instance quality, not semantic-instance alignment.
- Instance conditioning doesn't fix bad instances (e.g., merged pedestrians at same depth).

### Current Task: Instance Quality Improvement
- **Problem**: Depth-guided instances fail for co-planar objects (person PQ=4.2, RQ=8.8%, only 170/3206 matched)
- **Current instances**: SPIdepth depth → Sobel gradients → threshold (τ=0.20, A_min=1000) → PQ_things=19.41
- **Direction**: Train dedicated instance segmentation model on pseudo-labels, potentially with iterative self-training
- **Evaluation**: PQ_things, per-class RQ (especially person), TP/FP/FN counts against Cityscapes GT
- **Goal**: Break through depth-only ceiling, then combine improved instances with existing UNet semantics (PQ_stuff=35.04)
- Remote: `santosh@100.93.203.100`, 2x GTX 1080 Ti

### Earlier: HiRes RefineNet Upsampling Ablation (COMPLETE)
- Transposed conv WINNER (PQ=27.50), bilinear (27.29), pixelshuffle (27.26)
- Evolved into DepthGuidedUNet with progressive decoder and depth skip connections
- Reports: `reports/hires_refinenet_128x256.md`, `reports/ub_transposed_conv_detailed.md`

### Stage 2: CUPS Cascade Mask R-CNN (IN PROGRESS)
- Training on k=80 pseudo-labels via CUPS pipeline
- Best so far: PQ=22.5 at step 4000 (v4 fixed, still improving)

## Reference Documentation

- `guidelines/SKILL.md` - Master architecture specification
- `guidelines/MBPS_IMPLEMENTATION_PROMPT.md` - Implementation checklist
- `guidelines/mbps_incremental_implementation_guide.md` - 46-step build guide
- `guidelines/mamba_panoptic_technical_report.md` - Math and algorithms
- `guidelines/guidelines_for_execution_of_ablations_and_final_run.md` - Ablation playbook
