# M5 Confidence-Weighted LoRA on DepthPro k=80 (santosh 2×GPU) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Conv-DoRA + Progressive LoRA + M5 (confidence-weighted semantic loss) on DINOv3 ViT-B/16, using DepthPro τ=0.20 instance pseudo-labels + k=80 DINOv3 semantic pseudo-labels, on santosh (2× GTX 1080 Ti), to push Stage-2 PQ past the frozen-backbone baseline of 28.40.

**Architecture:** Two-stage pipeline. **Stage-2** runs vanilla Conv-DoRA (rank 4, late_block_start=6) on fixed DepthPro k=80 pseudo-labels for 8000 steps to establish a student trained with Conv-DoRA on this label set. **Stage-3** resumes from the best Stage-2 checkpoint and runs 3 rounds × 500 steps of self-training with EMA teacher, Progressive LoRA (rank 2→4→8, coverage late→all blocks), and M5 confidence-weighted semantic loss — which requires a teacher (confirmed in code at `refs/cups/cups/pl_model_self.py:217-230`), so it is a Stage-3-only mitigation.

**Tech Stack:** PyTorch 2.1.2 + CUDA 11.8, Detectron2 0.6, Lightning 2.x, DDP across 2 GPUs, `conda env: cups` on santosh, existing CUPS pipeline (`refs/cups/`).

**Targets / Gates:**
- Stage-2 gate: PQ ≥ 28.28 (match prior Conv-DoRA best; if below, abort pipeline)
- Stage-3 final gate: PQ ≥ 28.9 (baseline 28.40 + 0.5 Hungarian gate)
- Fallback: if Stage-3 fails gate, park LoRA line, pivot to Stage-2 P0–P4 aux-loss plan

---

## Context the engineer needs

### Remote layout (santosh@100.93.203.100, ssh key-auth already set up)

| Resource | Path |
|---|---|
| CUPS repo | `/home/santosh/cups` (= `refs/cups/` rsynced) |
| Cityscapes images + GT | `/home/santosh/datasets/cityscapes/{leftImg8bit,gtFine}` |
| Pseudo-labels (DepthPro τ=0.20) | `/home/santosh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020/` (8925 files = 2975 × 3: `_semantic.png`, `_instance.png`, `.pt`) |
| DINOv3 weights | `/home/santosh/cups/weights/dinov3_vitb16_official.pth` |
| Python env | `/home/santosh/anaconda3/envs/cups/bin/python` (PyTorch 2.1.2+cu118, Detectron2 0.6) |
| Logs | `/home/santosh/cups/logs/` |
| Experiments | `/home/santosh/cups/experiments/` |
| GPUs | 2× GTX 1080 Ti, 11 GB each, both free |

### Why Stage-2 + Stage-3, not just Stage-3

M5 computes `confidence_weights = softmax(teacher_logits / T).max(dim=0).values.clamp(min=min_weight)` (`pl_model_self.py:229`). There is no teacher in Stage-2 — pseudo-labels are fixed PNG files from disk, not teacher output. So **M5 is a no-op in Stage-2**. We still need Stage-2 to build a student that can seed an EMA teacher for Stage-3.

### Prior best numbers (for sanity-check gates)

- Stage-2 frozen-backbone baseline on DepthPro k=80: **PQ = 28.40**
- Previous Conv-DoRA Stage-2 best: **PQ = 28.28** (gate: +0.5 over frozen = 28.90 → missed by −0.62)
- Now adding: Stage-3 with EMA teacher + M5 + Progressive LoRA (r=2→4→8) to push past 28.40

### Files to create (all under repo root)

- `refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml` — Stage-2 Conv-DoRA + DepthPro k=80 + 2 GPUs
- `refs/cups/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml` — Stage-3 Conv-DoRA + Progressive + M5 + DepthPro k=80 + 2 GPUs
- `scripts/run_m5_depthpro_santosh.sh` — verify / smoke / stage2 / stage3 / status subcommands

### Files to read (no modifications required)

- `refs/cups/cups/model/lora.py` — `DoRAConfig`, `MitigationConfig`, `inject_dora_into_model()`
- `refs/cups/cups/pl_model_self.py:217-230` — M5 confidence-weights computation
- `refs/cups/cups/model/modeling/meta_arch/panoptic_fpn.py:130-184` — `pixel_weights` plumbing into semantic head
- `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py:174-279` — `pixel_weights` applied to per-pixel CE
- `refs/cups/tests/test_mitigations.py` — 8 mitigation unit tests (`test_confidence_map_range`, `test_m_init_norm_buffer`, etc.)
- `refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_1gpu.yaml` — the 1-GPU Conv-DoRA + DepthPro baseline we adapt to 2 GPUs
- `refs/cups/configs/train_self_dinov3_vitb_k80_m5_conf_loss.yaml` — the existing Stage-3 M5 YAML (k=80 semantic-only; we need the DepthPro-instance variant)

---

## Task 1: Create Stage-2 Conv-DoRA 2-GPU config for DepthPro k=80

**Files:**
- Create: `refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml`

Derived from `train_cityscapes_dinov3_vitb_depthpro_e2_dora_1gpu.yaml`; swap `NUM_GPUS: 1` → `NUM_GPUS: 2` and `ACCUMULATE_GRAD_BATCHES: 16` → `ACCUMULATE_GRAD_BATCHES: 8` so effective batch stays 16 (2 × 1 × 8). Change `RUN_NAME` so outputs land in a distinct dir.

- [ ] **Step 1: Write the file**

```yaml
# Stage-2 Conv-DoRA on DINOv3 ViT-B/16 + DepthPro tau=0.20 instance PLs + k=80 semantic PLs
# 2x GTX 1080 Ti, bs=1, accum=8 → effective batch 16, fp16-mixed
# This is the Stage-2 feeder for the Stage-3 M5 run
# (configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml)
# Baseline (frozen DINOv3 on this label set): PQ = 28.40
# Previous Conv-DoRA best on k=80: PQ = 28.28 — we expect similar here.

DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/home/santosh/datasets/cityscapes/"
  ROOT_VAL: "/home/santosh/datasets/cityscapes/"
  ROOT_PSEUDO: "/home/santosh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020/"
MODEL:
  BACKBONE_TYPE: "dinov3_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
  TTA_SCALES:
    - 0.75
    - 1.0
    - 1.25
  LORA:
    ENABLED: True
    VARIANT: "conv_dora"
    RANK: 4
    ALPHA: 4.0
    DROPOUT: 0.05
    LATE_BLOCK_START: 6
    LR_A: 0.00001
    LR_B: 0.00005
    MAGNITUDE_WD: 0.001
    DELAYED_START_STEPS: 500
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
  LOG_PATH: "/home/santosh/cups/experiments"
  RUN_NAME: "e2_depthpro_conv_dora_r4_2gpu"
TRAINING:
  STEPS: 8000
  BATCH_SIZE: 1
  ACCUMULATE_GRAD_BATCHES: 8
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 500
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 1000
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
SELF_TRAINING:
  ROUND_STEPS: 500
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 2: Verify YAML loads under Detectron2 CfgNode**

Run locally (no CUDA required):

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups
python - <<'PY'
from cups.config import get_cfg
cfg = get_cfg()
cfg.merge_from_file("configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml")
assert cfg.SYSTEM.NUM_GPUS == 2
assert cfg.TRAINING.ACCUMULATE_GRAD_BATCHES == 8
assert cfg.MODEL.LORA.ENABLED is True
assert cfg.MODEL.LORA.VARIANT == "conv_dora"
assert "depthpro_tau020" in cfg.DATA.ROOT_PSEUDO
print("OK — Stage-2 2-GPU DepthPro Conv-DoRA config valid")
PY
```

Expected: `OK — Stage-2 2-GPU DepthPro Conv-DoRA config valid`

- [ ] **Step 3: Commit**

```bash
git add refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml
git commit -m "feat(e2): add 2-GPU Conv-DoRA config for DepthPro k=80 pseudo-labels"
```

---

## Task 2: Create Stage-3 Conv-DoRA + M5 config for DepthPro k=80

**Files:**
- Create: `refs/cups/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml`

Derived from `train_self_dinov3_vitb_k80_m5_conf_loss.yaml`. Two substantive changes:
1. `ROOT_PSEUDO` → `cups_pseudo_labels_depthpro_tau020/` (instead of `cups_pseudo_labels_dinov3_k80/`)
2. `NUM_GPUS: 2`, `ACCUMULATE_GRAD_BATCHES: 2` so effective batch = 2 × 2 × 2 = 8 (the Stage-3 reference bs used in the k=80 M5 config)
3. Update `ROOT` to the santosh path (`/home/santosh/datasets/cityscapes/`)
4. Distinct `RUN_NAME` and `LOG_PATH`

- [ ] **Step 1: Write the file**

```yaml
# Stage-3 Conv-DoRA + Progressive LoRA + M5 (Confidence-Weighted Semantic Loss)
# on DINOv3 ViT-B/16 + DepthPro tau=0.20 instance PLs + k=80 semantic PLs.
# 2x GTX 1080 Ti, bs=2 x accum=2 → effective batch 8 across both GPUs combined (4 per GPU).
# Progressive ranks (2, 4, 8), coverages late→all blocks.
# Resumes from Stage-2 best checkpoint (set via --ckpt_path CLI flag).
# Baseline to beat (Stage-2 frozen): PQ = 28.40. Gate: PQ >= 28.9 (+0.5).

DATA:
  CROP_RESOLUTION: (640, 1280)
  SCALE: 0.625
  DATASET: "cityscapes"
  IGNORE_UNKNOWN_THING_REGIONS: False
  THING_STUFF_THRESHOLD: 0.05
  ROOT: "/home/santosh/datasets/cityscapes/"
  ROOT_VAL: "/home/santosh/datasets/cityscapes/"
  ROOT_PSEUDO: "/home/santosh/datasets/cityscapes/cups_pseudo_labels_depthpro_tau020/"
MODEL:
  BACKBONE_TYPE: "dinov3_vitb"
  DINOV2_FREEZE: True
  USE_DINO: False
  CHECKPOINT: null
  TTA_SCALES: [0.75, 1.0, 1.25]
  LORA:
    ENABLED: True
    VARIANT: "conv_dora"
    RANK: 4
    ALPHA: 4.0
    DROPOUT: 0.05
    LATE_BLOCK_START: 6
    LR_A: 0.00001
    LR_B: 0.00005
    MAGNITUDE_WD: 0.001
    DELAYED_START_STEPS: 0
    PROGRESSIVE:
      ENABLED: True
      RANKS: [2, 4, 8]
      ALPHAS: [2.0, 4.0, 8.0]
      COVERAGES: [6, 6, 0]
    MITIGATIONS:
      CONFIDENCE_WEIGHTED_LOSS:
        ENABLED: True
        TEMPERATURE: 1.0
        MIN_WEIGHT: 0.1
AUGMENTATION:
  NUM_STEPS_STARTUP: 0
  COPY_PASTE: True
  MAX_NUM_PASTED_OBJECTS: 3
  RESOLUTIONS:
    - [384, 768]
    - [416, 832]
    - [448, 896]
    - [480, 960]
    - [512, 1024]
    - [544, 1088]
    - [576, 1152]
    - [608, 1216]
    - [640, 1280]
SYSTEM:
  ACCELERATOR: "gpu"
  NUM_GPUS: 2
  NUM_NODES: 1
  NUM_WORKERS: 4
  LOG_PATH: "/home/santosh/cups/experiments"
  RUN_NAME: "stage3_depthpro_m5_conv_dora_progressive_2gpu"
TRAINING:
  STEPS: 1500
  BATCH_SIZE: 2
  ACCUMULATE_GRAD_BATCHES: 2
  PRECISION: "16-mixed"
  OPTIMIZER: "adamw"
  ADAMW:
    LEARNING_RATE: 0.0001
    WEIGHT_DECAY: 0.00001
  VAL_EVERY_N_STEPS: 500
  LOG_EVERT_N_STEPS: 50
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.4
  LOG_MEDIA_N_STEPS: 500
  GRADIENT_CLIP_ALGORITHM: "norm"
  GRADIENT_CLIP_VAL: 0.1
SELF_TRAINING:
  ROUND_STEPS: 500
  ROUNDS: 3
  CONFIDENCE_STEP: 0.05
  USE_DROP_LOSS: False
  SEMANTIC_SEGMENTATION_THRESHOLD: 0.5
  DISABLE_EMA: False
VALIDATION:
  CACHE_DEVICE: "cpu"
  USE_TTA: False
```

- [ ] **Step 2: Verify YAML loads**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups
python - <<'PY'
from cups.config import get_cfg
cfg = get_cfg()
cfg.merge_from_file("configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml")
assert cfg.MODEL.LORA.PROGRESSIVE.ENABLED is True
assert tuple(cfg.MODEL.LORA.PROGRESSIVE.RANKS) == (2, 4, 8)
assert cfg.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.ENABLED is True
assert cfg.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.TEMPERATURE == 1.0
assert cfg.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.MIN_WEIGHT == 0.1
assert cfg.SYSTEM.NUM_GPUS == 2
assert "depthpro_tau020" in cfg.DATA.ROOT_PSEUDO
assert cfg.SELF_TRAINING.ROUNDS == 3
assert cfg.SELF_TRAINING.ROUND_STEPS == 500
print("OK — Stage-3 DepthPro M5 + Progressive Conv-DoRA config valid")
PY
```

Expected: `OK — Stage-3 DepthPro M5 + Progressive Conv-DoRA config valid`

- [ ] **Step 3: Commit**

```bash
git add refs/cups/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml
git commit -m "feat(m5): add Stage-3 Conv-DoRA + Progressive LoRA + M5 config for DepthPro k=80 (2 GPU)"
```

---

## Task 3: Create santosh launch script `run_m5_depthpro_santosh.sh`

**Files:**
- Create: `scripts/run_m5_depthpro_santosh.sh` (mode 755)

Subcommands: `verify`, `smoke`, `stage2`, `stage3`, `status`. Logs go to `/home/santosh/cups/logs/`. PIDs written to `<log_dir>/<stage>.pid` for kill hygiene.

- [ ] **Step 1: Write the file**

```bash
#!/bin/bash
# M5 Conv-DoRA + DepthPro k=80 pipeline on santosh (2x GTX 1080 Ti)
# Stage-2: Conv-DoRA fine-tune (8000 steps, ~90 min on 2 GPUs)
# Stage-3: Progressive Conv-DoRA + M5 self-training (3 rounds x 500 steps, ~60 min)
#
# Usage:
#   bash scripts/run_m5_depthpro_santosh.sh verify   # pre-flight checks
#   bash scripts/run_m5_depthpro_santosh.sh smoke    # 1-step wiring smoke test
#   bash scripts/run_m5_depthpro_santosh.sh stage2   # launch Stage-2 Conv-DoRA
#   bash scripts/run_m5_depthpro_santosh.sh stage3   # launch Stage-3 M5 (auto-finds Stage-2 ckpt)
#   bash scripts/run_m5_depthpro_santosh.sh status   # show latest log tail + running PIDs

set -euo pipefail

CUPS_ROOT="/home/santosh/cups"
DATASET_ROOT="/home/santosh/datasets/cityscapes"
PSEUDO_LABELS="${DATASET_ROOT}/cups_pseudo_labels_depthpro_tau020"
WEIGHTS="${CUPS_ROOT}/weights/dinov3_vitb16_official.pth"
LOG_DIR="${CUPS_ROOT}/logs"
EXP_DIR="${CUPS_ROOT}/experiments"
STAGE2_CONFIG="${CUPS_ROOT}/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml"
STAGE3_CONFIG="${CUPS_ROOT}/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml"
STAGE2_RUN_NAME="e2_depthpro_conv_dora_r4_2gpu"
STAGE3_RUN_NAME="stage3_depthpro_m5_conv_dora_progressive_2gpu"

export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/santosh/anaconda3/envs/cups/bin/python"

mkdir -p "${LOG_DIR}"

verify() {
    echo "=== M5 DepthPro Pre-flight ==="
    local ok=0 fail=0

    local sem inst pts
    sem=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*_semantic.png" | wc -l)
    inst=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*_instance.png" | wc -l)
    pts=$(find "${PSEUDO_LABELS}" -maxdepth 1 -name "*.pt" | wc -l)
    echo "[labels] semantic=${sem} instance=${inst} pt=${pts}"
    if [ "${sem}" -ge 2900 ] && [ "${inst}" -ge 2900 ] && [ "${pts}" -ge 2900 ]; then
        ((ok++))
    else
        echo "  FAIL: expected ~2975 of each"; ((fail++))
    fi

    if [ -f "${WEIGHTS}" ]; then
        echo "[weights] ${WEIGHTS}"; ((ok++))
    else
        echo "  FAIL: weights missing"; ((fail++))
    fi

    for cfg in "${STAGE2_CONFIG}" "${STAGE3_CONFIG}"; do
        if [ -f "${cfg}" ]; then
            echo "[cfg] ${cfg}"; ((ok++))
        else
            echo "  FAIL: ${cfg} missing"; ((fail++))
        fi
    done

    local gpus
    gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "[gpu] ${gpus} GPUs"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    if [ "${gpus}" -ge 2 ]; then ((ok++)); else echo "  FAIL: need 2 GPUs"; ((fail++)); fi

    echo "=== ${ok} ok, ${fail} fail ==="
    [ "${fail}" -eq 0 ] || return 1
}

smoke() {
    echo "=== M5 smoke test: confidence_weights wiring ==="
    cd "${CUPS_ROOT}"
    # Run the mitigation unit tests only (fast, CPU OK)
    ${PYTHON} -m pytest tests/test_mitigations.py -v --no-header 2>&1 | tail -30
}

stage2() {
    verify || exit 1
    local logfile="${LOG_DIR}/stage2_$(date +%Y%m%d_%H%M%S).log"
    echo "=== Stage-2 Conv-DoRA launch ==="
    echo "Config:  ${STAGE2_CONFIG}"
    echo "Log:     ${logfile}"
    echo "Monitor: tail -f ${logfile}"
    cd "${CUPS_ROOT}"
    nohup ${PYTHON} -u train.py \
        --experiment_config_file "${STAGE2_CONFIG}" \
        --disable_wandb \
        > "${logfile}" 2>&1 &
    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/stage2.pid"
}

stage3() {
    # Auto-find best Stage-2 checkpoint
    local s2_dir="${EXP_DIR}/${STAGE2_RUN_NAME}"
    if [ ! -d "${s2_dir}" ]; then
        echo "ERROR: Stage-2 experiment dir not found: ${s2_dir}"
        echo "Run stage2 first."
        exit 1
    fi
    local ckpt
    ckpt=$(find "${s2_dir}" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -z "${ckpt}" ]; then
        echo "ERROR: No .ckpt found under ${s2_dir}"
        exit 1
    fi
    echo "Using Stage-2 checkpoint: ${ckpt}"
    local logfile="${LOG_DIR}/stage3_m5_$(date +%Y%m%d_%H%M%S).log"
    echo "Config:  ${STAGE3_CONFIG}"
    echo "Log:     ${logfile}"
    echo "Monitor: tail -f ${logfile}"
    cd "${CUPS_ROOT}"
    nohup ${PYTHON} -u train_self.py \
        --experiment_config_file "${STAGE3_CONFIG}" \
        --ckpt_path "${ckpt}" \
        --disable_wandb \
        > "${logfile}" 2>&1 &
    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/stage3.pid"
}

status() {
    echo "=== PIDs ==="
    for stage in stage2 stage3; do
        local pidfile="${LOG_DIR}/${stage}.pid"
        if [ -f "${pidfile}" ]; then
            local pid
            pid=$(cat "${pidfile}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "  ${stage}: RUNNING pid=${pid}"
            else
                echo "  ${stage}: DEAD pid=${pid} (stale pidfile)"
            fi
        fi
    done

    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

    echo "=== Latest logs (last 20 lines each) ==="
    for f in $(ls -t "${LOG_DIR}"/stage2_*.log "${LOG_DIR}"/stage3_m5_*.log 2>/dev/null | head -2); do
        echo "--- ${f} ---"
        tail -20 "${f}"
    done
}

case "${1:-help}" in
    verify) verify ;;
    smoke)  smoke ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    status) status ;;
    *)
        echo "Usage: $0 {verify|smoke|stage2|stage3|status}"
        exit 1
        ;;
esac
```

- [ ] **Step 2: Mark executable**

```bash
chmod +x scripts/run_m5_depthpro_santosh.sh
```

- [ ] **Step 3: Lint (bash -n)**

```bash
bash -n scripts/run_m5_depthpro_santosh.sh
```

Expected: no output (syntax OK)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_m5_depthpro_santosh.sh
git commit -m "feat(m5): add santosh launch script for DepthPro M5 pipeline"
```

---

## Task 4: Smoke-test M5 wiring locally (no GPU)

**Files:**
- Test: `refs/cups/tests/test_mitigations.py` (existing, no modification)

Run the existing mitigation unit tests to confirm M5 code paths are intact after any recent edits.

- [ ] **Step 1: Run mitigations test suite**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups
python -m pytest tests/test_mitigations.py -v --no-header 2>&1 | tail -20
```

Expected: `8 passed` including `test_confidence_map_range`, `test_m_init_norm_buffer`, `test_mitigation_config_defaults`.

- [ ] **Step 2: Verify pixel_weights reach the CE loss on a synthetic tensor**

Create a throwaway script to confirm `semantic_seg.losses()` actually applies `pixel_weights` to the per-pixel cross-entropy:

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups
python - <<'PY'
import torch
import torch.nn.functional as F

# Replicate the weighted-CE path from semantic_seg.py:261-279
B, C, H, W = 2, 19, 64, 128
predictions = torch.randn(B, C, H, W)
targets = torch.randint(0, C, (B, H, W))
pixel_weights = torch.rand(B, H, W).clamp(min=0.1)  # M5 style
valid = (targets != 255).float()
loss_unreduced = F.cross_entropy(predictions, targets, reduction="none", ignore_index=255)
weighted = loss_unreduced * pixel_weights * valid
loss_m5 = weighted.sum() / (pixel_weights * valid).sum().clamp(min=1.0)
loss_plain = (loss_unreduced * valid).sum() / valid.sum().clamp(min=1.0)
print(f"plain CE   = {loss_plain.item():.4f}")
print(f"weighted CE= {loss_m5.item():.4f}")
assert torch.isfinite(loss_m5)
assert abs(loss_m5.item() - loss_plain.item()) > 1e-3, "weighted != plain (sanity)"
print("OK — M5 weighted CE differs from plain CE as expected")
PY
```

Expected: both losses printed, and the assertion passes (they should differ).

- [ ] **Step 3: No commit needed (tests/smoke are read-only)**

---

## Task 5: Rsync to santosh + remote verify

**Files:** (no new files)

Push the new configs and script to santosh, then run `verify`.

- [ ] **Step 1: Rsync refs/cups config dir + scripts**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
rsync -avz --progress \
    refs/cups/configs/train_cityscapes_dinov3_vitb_depthpro_e2_dora_2gpu.yaml \
    refs/cups/configs/train_self_dinov3_vitb_depthpro_m5_2gpu.yaml \
    santosh@100.93.203.100:/home/santosh/cups/configs/

rsync -avz --progress \
    scripts/run_m5_depthpro_santosh.sh \
    santosh@100.93.203.100:/home/santosh/cups/scripts/
```

Expected: two config files + one script file transferred.

- [ ] **Step 2: Run remote verify**

```bash
ssh santosh@100.93.203.100 'chmod +x /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh && bash /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh verify'
```

Expected: `=== 6 ok, 0 fail ===` (or similar — all checks green, 2 GPUs free).

- [ ] **Step 3: Remote smoke (unit tests)**

```bash
ssh santosh@100.93.203.100 'bash /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh smoke'
```

Expected: `8 passed` from test_mitigations.py.

---

## Task 6: Launch Stage-2 Conv-DoRA on santosh

**Files:** (no file changes; uses `run_m5_depthpro_santosh.sh stage2`)

- [ ] **Step 1: Launch**

```bash
ssh santosh@100.93.203.100 'bash /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh stage2'
```

Expected: prints PID and logfile path, writes `/home/santosh/cups/logs/stage2.pid`.

- [ ] **Step 2: Monitor first 100 steps (check loss is finite, no OOM)**

```bash
ssh santosh@100.93.203.100 'tail -n 100 -f $(ls -t /home/santosh/cups/logs/stage2_*.log | head -1)' &
# Let it run for ~5 min, check no loss=inf spikes, then detach with Ctrl-C
```

**Red flags** that require killing the run:
- `loss=inf` or `loss=nan` within first 200 steps (DoRA column-norm instability)
- GPU OOM on either card
- `NaN in grad` warnings repeating
- Training throughput < 1 step/sec (DDP is broken)

If any red flag:

```bash
ssh santosh@100.93.203.100 'kill $(cat /home/santosh/cups/logs/stage2.pid)'
```

then investigate before re-launching.

- [ ] **Step 3: Wait for completion (~90 min) + collect best PQ**

```bash
ssh santosh@100.93.203.100 'bash /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh status'
```

When training finishes, inspect `experiments/e2_depthpro_conv_dora_r4_2gpu/` for validation PQ at each checkpoint. The best `.ckpt` file is the one we feed to Stage-3.

- [ ] **Step 4: Gate check**

Stage-2 gate: PQ ≥ 28.28 (match prior Conv-DoRA k=80 best).
- If Stage-2 best PQ < 28.0: likely training instability or bad pseudo-labels — debug before Stage-3.
- If 28.0 ≤ Stage-2 best PQ < 28.28: marginal; still proceed to Stage-3 to see if M5 + Progressive recovers.
- If Stage-2 best PQ ≥ 28.28: proceed normally.

---

## Task 7: Launch Stage-3 M5 + Progressive on best Stage-2 ckpt

- [ ] **Step 1: Launch Stage-3**

```bash
ssh santosh@100.93.203.100 'bash /home/santosh/cups/scripts/run_m5_depthpro_santosh.sh stage3'
```

The script auto-finds the most recent `.ckpt` under `experiments/e2_depthpro_conv_dora_r4_2gpu/`. If the "most recent" is not the "best", manually pass the path by editing `stage3()` in the script or invoking `train_self.py` directly with `--ckpt_path`.

- [ ] **Step 2: Verify M5 actually fires — confirm `loss_sem` shifts when M5 on vs off**

In the first 50 log lines, you should see the student model being initialized with a teacher, and the per-step log should list loss keys. The Stage-3 log line should show `loss_sem` slightly different from a Stage-3 run without `CONFIDENCE_WEIGHTED_LOSS.ENABLED=True`.

Tail the log:

```bash
ssh santosh@100.93.203.100 'tail -f $(ls -t /home/santosh/cups/logs/stage3_m5_*.log | head -1)'
```

Look for:
- 3 rounds × 500 steps structure (total 1500 steps)
- Progressive rank expansion at round boundaries: rank 2 → 4 → 8
- `EMA teacher updated` messages every batch
- `val_pq_panoptic` logged at each validation checkpoint

- [ ] **Step 3: Gate check at end of training (~60 min)**

Final gate: **PQ ≥ 28.9** (frozen baseline 28.40 + 0.5).

Possible outcomes:
- PQ ≥ 28.9: gate passed. Commit results, update memory, move to E2 ablation write-up.
- 28.4 ≤ PQ < 28.9: partial win — Conv-DoRA + M5 matches frozen but doesn't clearly beat it. Consider also enabling M1 (cosine warmup) + M2 (magnitude warmup) via the combined config.
- PQ < 28.4: M5 doesn't move the needle over frozen. Park the LoRA line, pivot to Stage-2 P0–P4 aux-loss.

---

## Task 8: Eval + write-up

- [ ] **Step 1: Pull the Stage-3 best checkpoint back (optional, for offline eval)**

```bash
mkdir -p checkpoints/stage3_m5_depthpro
rsync -avz --progress santosh@100.93.203.100:/home/santosh/cups/experiments/stage3_depthpro_m5_conv_dora_progressive_2gpu/ checkpoints/stage3_m5_depthpro/
```

- [ ] **Step 2: Write report**

Create `reports/stage3_m5_depthpro_results.md` with:
- Stage-2 best PQ (per checkpoint curve)
- Stage-3 final PQ + per-class breakdown (things vs stuff)
- Training-loss curves screenshot (if feasible)
- Comparison vs frozen baseline (28.40) and vs prior k=80 Conv-DoRA (28.28)
- Decision: gate passed/failed, next step

- [ ] **Step 3: Update memory**

Edit `memory/e2_conv_dora_results.md`:
- Correct the wrong PQ=26.4 → PQ=28.28 (prior) / PQ=<new> (this run)
- Update "Decision Gate" section to reflect the DepthPro M5 outcome
- Save under a new timestamp

Also add a short entry to `memory/MEMORY.md` linking the new report.

- [ ] **Step 4: Commit the report + memory updates**

```bash
git add reports/stage3_m5_depthpro_results.md memory/e2_conv_dora_results.md memory/MEMORY.md
git commit -m "docs(m5): DepthPro M5 pipeline results + memory correction"
```

---

## Self-Review

**Spec coverage:**
- [x] Stage-2 config on DepthPro k=80 + 2 GPUs — Task 1
- [x] Stage-3 M5 + Progressive config on DepthPro k=80 + 2 GPUs — Task 2
- [x] Launch script with verify/smoke/stage2/stage3/status — Task 3
- [x] Pre-run smoke tests (unit + synthetic CE) — Task 4
- [x] Remote deployment + verify — Task 5
- [x] Stage-2 launch + monitor + gate — Task 6
- [x] Stage-3 launch + M5 verification + final gate — Task 7
- [x] Results eval + memory correction — Task 8

**Key risks identified:**
- M5 no-op in Stage-2 → handled by splitting pipeline
- `loss=inf` from DoRA column-norm → monitored in Task 6 Step 2; if recurs, M3 spectral-norm-ball is the fallback
- Auto-finding "latest" ckpt ≠ "best" ckpt → Task 7 Step 1 flags this; override path if needed

**No placeholders.** Each task lists exact file paths, exact bash commands, and expected outputs.
