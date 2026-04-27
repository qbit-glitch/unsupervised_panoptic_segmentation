# E1 CUPS Pseudo-Label Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the full CUPS pseudo-label set (semantic + instance) for Cityscapes on the RTX A6000 48 GB machine, using the upstream CUPS pipeline at `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py`. New downloads land on the **/Data1** drive (1.8 TB free); existing left/seq/gtFine on `/home` are symlinked into the unified root.

**Architecture:** Two-pass per-image pipeline:

1. **Pass 1 (instance proposals):** RAFT-SMURF optical flow on adjacent video frames + stereo image pairs → SF2SE3 (scene-flow → SE(3) decomposition) → object proposals → instance maps.
2. **Pass 2 (semantics + merge):** Frozen DepthG semantic checkpoint → dense CRF refinement → ThingStuffSplitter merge with instance maps → final panoptic pseudo-labels.

Existing `scripts/e1_*.sh` toolchain handles most of this; this plan wires in the A6000-specific paths and the `/Data1`-vs-`/home` data layout.

**Tech Stack:** PyTorch 2.5.1 + CUDA 12.1, `cups` package under `refs/cups/`, `pydensecrf`, `RAFT-SMURF`, `SF2SE3`. venv (`~/umesh/ups_env`) on A6000 (NOT a conda env named `ups`).

**Out of scope for this plan:** Stage-2 + Stage-3 training, evaluation. Those become follow-on plans once pseudo-labels verify clean.

**Critical history (from project memory + 2026-04-27 discovery):**
- Prior santosh attempt on 11 GB GPU silently produced 84.5 % all-zero instance maps — SF2SE3 OOM was caught and swallowed. **Always run `scripts/e1_verify_pseudolabels.py` after Pass 1**, not just trust file counts.
- Existing on-disk Cityscapes is incomplete: missing `rightImg8bit`, `rightImg8bit_sequence`, `camera` (~335 GB to download).
- A6000 GPU at discovery time was 33.7 GB used / 14.9 GB free — **see open question (1) in the environment snapshot.**

---

## Pre-resolved variables (from `2026-04-27-e1-environment-snapshot.md`)

```bash
export REPO_ROOT="/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation"
export CITYSCAPES_ROOT="/Data1/cityscapes"
export WORK_DIR="/Data1"
export OUTPUT_DIR="/Data1/cups_pseudo_labels_e1"
export VENV_ACTIVATE="$HOME/umesh/ups_env/bin/activate"
export PYTHON_BIN="$HOME/umesh/ups_env/bin/python"
export NUM_SUBSPLITS=2     # 1 if GPU contention persists
export NUM_WORKERS=6
```

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `scripts/e1_master_stage_assets.sh` | Existing | Master-node stager: clone repo + download `depthg.ckpt` + `raft_smurf.pt` from public HF with SHA verify. |
| `scripts/e1_download_cityscapes.sh` | Existing | Credentialed Cityscapes downloader (pkgs 1/3/4/8/14/15). |
| `scripts/e1_check_cityscapes.sh` | Existing | Cityscapes layout inventory. |
| `scripts/e1_stage1_pseudolabel_gen_a6000.sh` | Existing | Pre-flight + 2-way subsplit launcher for Pass 1+2 + sanity gates. |
| `scripts/e1_verify_pseudolabels.py` | Existing | Post-run quality verifier (target: empty-frac < 1 %). |
| `scripts/e1/discover_a6000_environment.sh` | Existing (this branch) | Read-only A6000 discovery. |
| `scripts/e1/setup_data1_cityscapes_layout.sh` | **New (this branch)** | Build `/Data1/cityscapes/` with symlinks for left/seq/gtFine + dirs for missing right/right_seq/camera. |
| `docs/plans/2026-04-27-e1-cups-pseudo-label-generation.md` | This plan | — |
| `docs/plans/2026-04-27-e1-environment-snapshot.md` | New (this branch) | Discovery snapshot + open user decisions. |

---

### Task 1: User decisions (block before starting)

**Files:** none.

- [ ] **Step 1: Confirm GPU contention status**

```bash
ssh cvpr_ug_5@gpunode2 'nvidia-smi --query-compute-apps=pid,user,used_memory,process_name --format=csv'
```

Identify whose process is holding the 33.7 GB. Three outcomes:
- It's our own stale process → kill it and proceed with `NUM_SUBSPLITS=2`.
- It's another user's job and short-lived → wait, then `NUM_SUBSPLITS=2`.
- It's another user's job and long-running → set `NUM_SUBSPLITS=1` (~10 h instead of 5 h, but won't OOM).

- [ ] **Step 2: Confirm `leftImg8bit_sequence/` is the full 30-frame package**

The `setup_data1_cityscapes_layout.sh` script (Task 3) reports the frames-per-labeled-image ratio. If it reports `~1`, the user must `rm -rf $EXISTING_CITYSCAPES/leftImg8bit_sequence/` and re-download via `scripts/e1_download_cityscapes.sh 14` before Pass 1 can run.

---

### Task 2: Stage repo + checkpoints on the A6000 (master-side)

**Files:** none new — uses existing `scripts/e1_master_stage_assets.sh`.

- [ ] **Step 1: Pull this branch on the A6000**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT &&
  git fetch origin implement-dora-adapters &&
  git checkout implement-dora-adapters &&
  git pull --ff-only origin implement-dora-adapters
"
```

- [ ] **Step 2: Run the master stager (downloads `depthg.ckpt` + `raft_smurf.pt`)**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT &&
  REPO_ROOT=$REPO_ROOT bash scripts/e1_master_stage_assets.sh
"
```

Expected: `[ok] depthg.ckpt SHA256 verified`, `[ok] raft_smurf.pt SHA256 verified`, `STAGER COMPLETE`.

If SHA mismatch: delete the offending file in `$REPO_ROOT/weights/` or `$REPO_ROOT/refs/cups/cups/optical_flow/checkpoints/` and re-run.

---

### Task 3: Build `/Data1/cityscapes/` layout

**Files:** uses new `scripts/e1/setup_data1_cityscapes_layout.sh`.

- [ ] **Step 1: Run the layout helper**

```bash
ssh cvpr_ug_5@gpunode2 "bash $REPO_ROOT/scripts/e1/setup_data1_cityscapes_layout.sh"
```

Expected output:
- ✅ symlinks created for `leftImg8bit`, `leftImg8bit_sequence`, `gtFine`.
- ⚠ empty dirs for `rightImg8bit`, `rightImg8bit_sequence`, `camera` (needs Task 4).
- Sequence frame ratio report (must be > 25 for the full package; see Task 1 Step 2).

- [ ] **Step 2: Confirm via inventory**

```bash
ssh cvpr_ug_5@gpunode2 "bash $REPO_ROOT/scripts/e1_check_cityscapes.sh /Data1/cityscapes"
```

Expected: `STATUS: incomplete — packages marked MISSING above must be downloaded` listing exactly `rightImg8bit`, `rightImg8bit_sequence`, `camera`.

---

### Task 4: Download missing Cityscapes packages onto /Data1

**Files:** uses existing `scripts/e1_download_cityscapes.sh`. **Requires user-supplied Cityscapes credentials.**

- [ ] **Step 1: Export credentials on A6000 (do NOT commit)**

User runs (one-shot, in their A6000 shell):

```bash
export CITYSCAPES_USER='<your_login>'
export CITYSCAPES_PASS='<your_password>'
```

(Register at https://www.cityscapes-dataset.com/register/ if needed.)

- [ ] **Step 2: Launch the download (background, ~3–6 h depending on link speed)**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT &&
  CITYSCAPES_ROOT=/Data1/cityscapes \
  DELETE_ZIPS=1 \
  nohup bash scripts/e1_download_cityscapes.sh 4 15 8 \
    > /Data1/e1_cityscapes_download.log 2>&1 &
  echo \"download PID=\$!\"
"
```

`DELETE_ZIPS=1` removes the .zip after each successful extract — saves ~120 GB on the right_sequence package.

- [ ] **Step 3: Monitor**

```bash
ssh cvpr_ug_5@gpunode2 "tail -n 5 /Data1/e1_cityscapes_download.log"
```

Use `Monitor` (deferred tool) or a recurring `/loop` to poll every ~15 min.

- [ ] **Step 4: Verify completion**

```bash
ssh cvpr_ug_5@gpunode2 "bash $REPO_ROOT/scripts/e1_check_cityscapes.sh /Data1/cityscapes"
```

Expected: `STATUS: complete — all six packages present.` — proceeds to Task 5.

---

### Task 5: Run CUPS Pass 1+2 pseudo-label generation

**Files:** uses existing `scripts/e1_stage1_pseudolabel_gen_a6000.sh`.

- [ ] **Step 1: Smoke-import sanity (1 minute)**

```bash
ssh cvpr_ug_5@gpunode2 "
  source ~/umesh/ups_env/bin/activate &&
  cd $REPO_ROOT/refs/cups &&
  python -c '
import torch
from cups.semantics.model import DepthG
from cups.optical_flow import raft_smurf
from cups.scene_flow_2_se3 import sf2se3
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"device count\", torch.cuda.device_count())
'
"
```

Expected: `torch 2.5.1 cuda True device count 1`. If imports fail, the existing launcher's `SKIP_DEPS=0` path will pip-install — but if our venv is the right one, all should already be there.

- [ ] **Step 2: Launch Stage-1 (with `SKIP_GIT=1 SKIP_DOWNLOAD=1 SKIP_DEPS=1` since master-stager already did all of that)**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT &&
  module load gcc-9.3.0 &&
  source ~/umesh/ups_env/bin/activate &&
  export PATH=\$HOME/cuda-12.1/bin:\$PATH \
         LD_LIBRARY_PATH=\$HOME/cuda-12.1/lib64:\$LD_LIBRARY_PATH \
         CUDA_HOME=\$HOME/cuda-12.1 \
         REPO_ROOT=$REPO_ROOT \
         CITYSCAPES_ROOT=/Data1/cityscapes \
         WORK_DIR=/Data1 \
         OUTPUT_DIR=/Data1/cups_pseudo_labels_e1 \
         GPU_ID=0 \
         NUM_SUBSPLITS=$NUM_SUBSPLITS \
         NUM_WORKERS=$NUM_WORKERS \
         SKIP_GIT=1 SKIP_DOWNLOAD=1 SKIP_DEPS=1 \
         PYTHON_BIN=\$(which python) &&
  nohup bash scripts/e1_stage1_pseudolabel_gen_a6000.sh > /Data1/e1_stage1.out 2>&1 &
  echo \"PID=\$!\"
"
```

Expected: prints PID. Per-subsplit logs at `/Data1/logs/e1_split_{1,2}_<TS>.log`. Master log at `/Data1/e1_stage1.out`.

ETA on A6000 with no contention + `NUM_SUBSPLITS=2`: ~5 h.

- [ ] **Step 3: Monitor**

```bash
ssh cvpr_ug_5@gpunode2 "
  tail -n 5 /Data1/logs/e1_split_*.log /Data1/e1_stage1.out
  echo --- &&
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv
"
```

Watch for `CUDA out of memory` lines — if any subsplit hits this, kill all processes and rerun with `NUM_SUBSPLITS=1`.

- [ ] **Step 4: Wait for completion**

The launcher prints `E1 Stage-1 generation COMPLETE.` followed by the verify-report path when done.

---

### Task 6: Verify pseudo-label quality

The launcher already runs `scripts/e1_verify_pseudolabels.py` automatically with `--max_empty_frac 0.01`. If it fails, re-run manually for a richer report.

- [ ] **Step 1: Inspect the verify report**

```bash
ssh cvpr_ug_5@gpunode2 "
  cat /Data1/logs/e1_verify_*.json | tail -1 | python -m json.tool
"
```

Pass criteria:
- `expected_count: 2975` matches `actual_count`.
- `empty_instance_frac < 0.01` (vs 0.845 on the broken santosh run).
- `avg_unique_instance_ids_per_image >= 5`.

- [ ] **Step 2: Persist the result into project memory**

Append to `~/.claude/projects/.../memory/cups_dinov3_control_experiment.md`:

```markdown
## 2026-04-27 A6000 regeneration
- Pipeline: refs/cups/cups/pseudo_labels/gen_pseudo_labels.py (commit <COMMIT>)
- Hardware: RTX A6000 48 GB, NUM_SUBSPLITS=<N>
- Output: /Data1/cups_pseudo_labels_e1/
- Files: <count> instance + <count> semantic
- Empty instance maps: <X>%  (vs santosh broken run 84.5%)
- Avg unique instance ids per image: <Y>
- Wall-clock: <hours>
- Verdict: READY for Stage-2 / NEEDS rerun
```

- [ ] **Step 3: Commit logs**

```bash
git add docs/plans/2026-04-27-e1-environment-snapshot.md
git commit -m "chore(e1): record A6000 pseudo-label quality stats"
```

---

## Out of scope (follow-on plans)

1. **e1 Stage-2 training plan** — train DINOv3 ViT-B/16 + Cascade Mask R-CNN for 8 000 steps on the new labels. Config: `refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml` (update `DATA.ROOT_PSEUDO=/Data1/cups_pseudo_labels_e1`, `DATA.ROOT=/Data1/cityscapes`).
2. **e1 Stage-3 self-training plan** — 3 rounds × 500 steps EMA self-training from Stage-2 ckpt.
3. **e1 evaluation + decision plan** — populate the decision matrix in `cups_dinov3_control_experiment.md`.

## Self-review

- **Spec coverage:** all five user-stated needs addressed: (1) target machine A6000 — Tasks 5; (2) datasets on /Data1 — Tasks 3-4; (3) regenerate pseudo-labels — Tasks 2/5/6; (4) clear OOM blocker — Tasks 1+5; (5) Stage-2/3 deliberately out of scope.
- **Placeholder scan:** only `<your_login>`/`<your_password>` (user-supplied secrets) and report-fill-ins (`<COMMIT>`, `<count>`, `<X>`, `<Y>`, `<hours>`) — by design.
- **Type consistency:** env-var contract (`CITYSCAPES_ROOT`, `OUTPUT_DIR`, `WORK_DIR`, `NUM_SUBSPLITS`, etc.) is identical across Tasks 3/4/5/6 and matches the existing `scripts/e1_*.sh` env-var contract.
