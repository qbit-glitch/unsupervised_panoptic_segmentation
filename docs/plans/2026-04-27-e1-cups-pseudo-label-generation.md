# E1 CUPS Pseudo-Label Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the full CUPS pseudo-label set (semantic + instance) for Cityscapes on the RTX A6000 48GB machine, using the official upstream CUPS pipeline at `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py`. Datasets and pseudo-label outputs live on the **Data1 drive** of the A6000 host.

**Architecture:** Two-pass per-image pipeline driven by `gen_pseudo_labels.py`:

1. **Pass 1 (instance proposals):** RAFT-SMURF optical flow on adjacent video frames + stereo disparity → SF2SE3 (scene-flow → SE(3) decomposition) → object proposals → instance maps.
2. **Pass 2 (semantics + merge):** Frozen DepthG semantic segmentation checkpoint → dense CRF refinement → ThingStuffSplitter merge with instance maps → final panoptic pseudo-labels.

Upstream `pseudolabel_gen.sh` parallelises Cityscapes by splitting the train set into `NUM_PREPROCESSING_SUBSPLITS` chunks; we will run **2 chunks in parallel** on a single A6000 (≥16 GB free per process).

**Tech Stack:** PyTorch + Detectron2 (CUDA), `cups` package under `refs/cups/`, `dense-crf`, `RAFT-SMURF`, `SF2SE3`, conda env (`ups` per CLAUDE.md). Cityscapes Stereo+Video subset (`leftImg8bit_sequence`, `disparity`, `gtFine`, `leftImg8bit`).

**Out of scope for this plan:** Stage-2 + Stage-3 training, evaluation against our DCFA/SIMCF-ABC numbers. Those become a follow-on plan once the pseudo-labels are verified clean.

**Critical blocker history (from project memory, 2026-04-15):** Prior santosh attempt on GTX 1080 Ti (11 GB) silently produced 84.5% all-zero instance maps — SF2SE3 OOM was caught and swallowed. **Must verify non-empty instance content after Pass 1**, not just trust file counts.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py` | Existing (read-only) | Upstream CUPS entrypoint. Do NOT modify. |
| `refs/cups/cups/pseudo_labels/config_pseudo_labels.yaml` | Existing (read-only) | Default config; we override via CLI flags. |
| `refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh` | **Create** | A6000-specific launcher with our paths and 2-process parallelism. |
| `scripts/e1/check_pseudo_label_quality.py` | **Create** | Verifier: counts non-empty instance maps, checks `np.unique` per file, reports % broken. |
| `scripts/e1/setup_a6000_env.sh` | **Create** | One-shot env setup: clone branch, create conda env if missing, install deps, validate `import cups`. |
| `docs/plans/2026-04-27-e1-cups-pseudo-label-generation.md` | This plan | — |
| `$DATA1_ROOT/cityscapes/` | Data | Cityscapes raw (left/right/seq/disp/gtFine). |
| `$DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2/` | Data (output) | Pass-1+Pass-2 outputs. `_v2` suffix to avoid clobbering any santosh leftovers. |

---

## Variables to Confirm Before Execution

These must be set in the executing shell (`export VAR=value`). Task 1 establishes them.

```bash
export A6000_HOST="cvpr_ug_5@gpunode2"     # per CLAUDE.md anydesk machine
export DATA1_ROOT="/mnt/data1"             # PLACEHOLDER — confirm in Task 1
export REPO_ROOT="$HOME/umesh/unsupervised_panoptic_segmentation"  # per CLAUDE.md anydesk paths
export CONDA_ENV="ups"                      # per CLAUDE.md anydesk conda env
export CUDA_HOME="$HOME/cuda-12.1"          # per CLAUDE.md
export DEPTHG_CKPT="$REPO_ROOT/checkpoints/depthg.ckpt"  # CUPS DepthG semantic checkpoint
```

---

### Task 1: Confirm A6000 environment and Data1 mount path

**Files:** none (discovery only).

- [ ] **Step 1: SSH into the A6000 host and capture environment baseline**

```bash
ssh cvpr_ug_5@gpunode2 'bash -lc "
  echo === HOSTNAME ===
  hostname
  echo === GPU ===
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo === MOUNTS ===
  mount | grep -i -E \"data|nvme|ssd\" || true
  echo === DF ===
  df -h | grep -v -E \"snap|tmpfs|devtmpfs\"
  echo === CONDA ===
  source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
  conda env list
"'
```

Expected: One GPU named `RTX A6000` with `49140 MiB` total, conda env `ups` listed, a mount or directory called something like `/mnt/data1`, `/media/data1`, `/data1`, or under `/media/<user>/...`.

- [ ] **Step 2: Locate the Data1 mount and free space**

If the previous step did not surface the Data1 path explicitly, ask the user. Once known, set `DATA1_ROOT` and confirm:

```bash
ssh cvpr_ug_5@gpunode2 "ls -la $DATA1_ROOT && df -h $DATA1_ROOT"
```

Expected: directory exists, ≥150 GB free (Cityscapes Stereo+Video ≈ 90 GB, pseudo-labels ≈ 5 GB, working buffer ≈ 50 GB).

- [ ] **Step 3: Confirm Cityscapes presence on Data1 (or note that it must be downloaded)**

```bash
ssh cvpr_ug_5@gpunode2 "
  for d in leftImg8bit leftImg8bit_sequence disparity gtFine; do
    if [ -d \"$DATA1_ROOT/cityscapes/\$d/train\" ]; then
      n=\$(find \"$DATA1_ROOT/cityscapes/\$d/train\" -mindepth 2 -maxdepth 2 -type f 2>/dev/null | wc -l)
      echo \"\$d/train: \$n files\"
    else
      echo \"\$d/train: MISSING\"
    fi
  done
"
```

Expected: each directory present with the right counts (Cityscapes train has 2975 images per split for `leftImg8bit`/`gtFine`, ~89,250 for `leftImg8bit_sequence` since each train sample has 30 frames).

- [ ] **Step 4: Record findings**

Write the resolved values into a session note `docs/plans/2026-04-27-e1-environment-snapshot.md`:

```markdown
# E1 A6000 Environment Snapshot (YYYY-MM-DD)

- Hostname: ...
- GPU: RTX A6000, ... MiB
- Driver: ...
- DATA1_ROOT: <resolved path>
- DATA1 free space: ... GB
- Cityscapes status:
  - leftImg8bit/train: <count> files
  - leftImg8bit_sequence/train: <count> files
  - disparity/train: <count> files
  - gtFine/train: <count> files
- Conda env `ups` present: yes / no
- DepthG checkpoint present at $DEPTHG_CKPT: yes / no
```

- [ ] **Step 5: Commit**

```bash
git add docs/plans/2026-04-27-e1-cups-pseudo-label-generation.md docs/plans/2026-04-27-e1-environment-snapshot.md
git commit -m "docs(e1): plan + A6000 environment snapshot for CUPS pseudo-label generation"
```

---

### Task 2: Stage Cityscapes on Data1 (download or symlink)

**Files:**
- Create: `scripts/e1/setup_cityscapes_on_data1.sh`
- Modify: none.

**Decision rule:** if Task 1 Step 3 showed all four directories populated with correct counts, **skip to Task 3**. Otherwise download what is missing.

- [ ] **Step 1: Write `scripts/e1/setup_cityscapes_on_data1.sh`**

```bash
#!/bin/bash
# scripts/e1/setup_cityscapes_on_data1.sh
# Downloads Cityscapes Stereo+Video subset to $DATA1_ROOT/cityscapes/.
# Requires CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars.

set -euo pipefail

: "${DATA1_ROOT:?DATA1_ROOT must be set}"
: "${CITYSCAPES_USERNAME:?register at https://www.cityscapes-dataset.com/register/}"
: "${CITYSCAPES_PASSWORD:?}"

DEST="$DATA1_ROOT/cityscapes"
mkdir -p "$DEST"
cd "$DEST"

# Login (writes a cookie to cookies.txt)
wget --keep-session-cookies --save-cookies=cookies.txt \
     --post-data "username=$CITYSCAPES_USERNAME&password=$CITYSCAPES_PASSWORD&submit=Login" \
     https://www.cityscapes-dataset.com/login/ -O /dev/null

# Package IDs (from cityscapes-dataset.com):
# 1  = gtFine_trainvaltest.zip       (241 MB)
# 3  = leftImg8bit_trainvaltest.zip  (11 GB)
# 14 = leftImg8bit_sequence.zip      (324 GB — TRAIN+VAL+TEST)
# 7  = disparity_trainvaltest.zip    (3.5 GB)
for id in 1 3 14 7; do
  wget --load-cookies cookies.txt --content-disposition \
       "https://www.cityscapes-dataset.com/file-handling/?packageID=$id"
done

# Unpack
for f in gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip \
         leftImg8bit_sequence.zip disparity_trainvaltest.zip; do
  unzip -q -o "$f" -d "$DEST"
  rm -f "$f"
done

rm -f cookies.txt
echo "DONE: Cityscapes Stereo+Video staged at $DEST"
```

- [ ] **Step 2: Push the script to the A6000 (rsync from local repo)**

```bash
rsync -avz scripts/e1/ cvpr_ug_5@gpunode2:$REPO_ROOT/scripts/e1/
```

Expected: 1 file transferred, no errors.

- [ ] **Step 3: Run the download on the A6000 (only if Task 1 reported missing dirs)**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT &&
  export DATA1_ROOT=$DATA1_ROOT CITYSCAPES_USERNAME=<ASK_USER> CITYSCAPES_PASSWORD=<ASK_USER> &&
  nohup bash scripts/e1/setup_cityscapes_on_data1.sh > $DATA1_ROOT/cityscapes_download.log 2>&1 &
  echo \$!
"
```

**Note:** `leftImg8bit_sequence.zip` is 324 GB. Confirm Data1 has the headroom **before** kicking this off. If under 400 GB free, ask the user whether to download `leftImg8bit_sequence_trainvaltest.zip` (already 119 GB unpacked) or use a smaller subset.

- [ ] **Step 4: Wait for completion + verify**

Poll `$DATA1_ROOT/cityscapes_download.log` until it prints `DONE: Cityscapes Stereo+Video staged at ...`. Then re-run Task 1 Step 3 verification — all four directories should now report the right file counts.

- [ ] **Step 5: Commit**

```bash
git add scripts/e1/setup_cityscapes_on_data1.sh
git commit -m "feat(e1): cityscapes stereo+video downloader for Data1 drive"
```

---

### Task 3: Set up the conda env and CUPS dependencies on the A6000

**Files:**
- Create: `scripts/e1/setup_a6000_env.sh`
- Modify: none.

- [ ] **Step 1: Write `scripts/e1/setup_a6000_env.sh`**

```bash
#!/bin/bash
# scripts/e1/setup_a6000_env.sh
# One-shot env setup on A6000:
#   - ensures conda env `ups` exists with PyTorch 2.5.1 + CUDA 12.1
#   - installs CUPS deps (raft-smurf, sf2se3, dense-crf, detectron2)
#   - validates `import cups`
set -euo pipefail

: "${REPO_ROOT:?}"
: "${CONDA_ENV:=ups}"

cd "$REPO_ROOT"

# Load gcc-9 module (per CLAUDE.md, required for some CUDA extension builds)
module load gcc-9.3.0 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-12.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/anaconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^$CONDA_ENV "; then
  echo "Creating conda env $CONDA_ENV..."
  conda create -n "$CONDA_ENV" python=3.10 -y
fi

conda activate "$CONDA_ENV"

# PyTorch + CUDA 12.1
python -c "import torch; print(torch.__version__)" || \
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# CUPS deps (already in refs/cups; install in editable mode)
cd refs/cups
pip install -r requirements.txt 2>/dev/null || pip install \
  yacs lightning timm einops opencv-python-headless tqdm \
  matplotlib scikit-image scipy numpy h5py
pip install -e . || true  # cups package

# Dense CRF (system gcc must be ≤9 for compilation)
pip install pydensecrf 2>/dev/null || pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Detectron2 0.6 (binary wheel for torch 2.5 + cu121 may not exist; build from source)
python -c "import detectron2" 2>/dev/null || \
  pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

cd "$REPO_ROOT"
python -c "import cups; from cups.optical_flow import raft_smurf; from cups.scene_flow_2_se3 import sf2se3; print('cups OK')"
echo "DONE: env $CONDA_ENV ready"
```

- [ ] **Step 2: rsync + run on A6000**

```bash
rsync -avz scripts/e1/setup_a6000_env.sh cvpr_ug_5@gpunode2:$REPO_ROOT/scripts/e1/
ssh cvpr_ug_5@gpunode2 "cd $REPO_ROOT && bash scripts/e1/setup_a6000_env.sh 2>&1 | tee logs/e1_env_setup.log"
```

Expected final lines: `cups OK` and `DONE: env ups ready`.

- [ ] **Step 3: Stage the DepthG checkpoint**

DepthG is the CUPS semantic checkpoint used in Pass 2. The CUPS authors publish it on their HF Hub or model zoo; verify it exists locally first:

```bash
ssh cvpr_ug_5@gpunode2 "ls -la $DEPTHG_CKPT 2>/dev/null || echo MISSING"
```

If `MISSING`, download from CUPS official release:

```bash
ssh cvpr_ug_5@gpunode2 "
  mkdir -p \$(dirname $DEPTHG_CKPT) &&
  wget -O $DEPTHG_CKPT https://huggingface.co/lukashoyer/cups/resolve/main/depthg.ckpt
"
```

(If the URL has changed, search the upstream CUPS README on `refs/cups/README.md` for the canonical link.)

- [ ] **Step 4: Smoke-import sanity check**

```bash
ssh cvpr_ug_5@gpunode2 "
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate $CONDA_ENV &&
  cd $REPO_ROOT/refs/cups &&
  python -c '
import torch
from cups.semantics.model import DepthG
from cups.optical_flow import raft_smurf
from cups.scene_flow_2_se3 import sf2se3
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"gpus\", torch.cuda.device_count())
'
"
```

Expected: `torch 2.5.1 cuda True gpus 1` (or 2 if A6000 host has multiple GPUs visible).

- [ ] **Step 5: Commit**

```bash
git add scripts/e1/setup_a6000_env.sh
git commit -m "feat(e1): A6000 conda env + cups deps setup script"
```

---

### Task 4: Author the A6000 launcher `pseudolabel_gen_a6000.sh`

**Files:**
- Create: `refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
# refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh
# Launches CUPS pseudo-label generation on a single A6000 (48 GB),
# splitting Cityscapes train into 2 parallel processes.
#
# Required env vars (set by caller):
#   DATA1_ROOT     — Data1 mount root (parent of cityscapes/)
#   REPO_ROOT      — repo root on A6000
#   DEPTHG_CKPT    — absolute path to depthg.ckpt
#   PSEUDO_OUT     — absolute path for pseudo-label outputs (defaults to $DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2)
#   NUM_WORKERS    — DataLoader workers per process (default 4)

set -euo pipefail

: "${DATA1_ROOT:?}"
: "${REPO_ROOT:?}"
: "${DEPTHG_CKPT:?}"
PSEUDO_OUT="${PSEUDO_OUT:-$DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SUBSPLITS=2

mkdir -p "$PSEUDO_OUT"
mkdir -p "$REPO_ROOT/logs"

cd "$REPO_ROOT/refs/cups"

# Subsplit 1
CUDA_VISIBLE_DEVICES=0 nohup python -u cups/pseudo_labels/gen_pseudo_labels.py \
  --DATA.DATASET cityscapes \
  --DATA.ROOT "$DATA1_ROOT/cityscapes/" \
  --DATA.PSEUDO_ROOT "$PSEUDO_OUT/" \
  --DATA.NUM_PREPROCESSING_SUBSPLITS $NUM_SUBSPLITS \
  --DATA.PREPROCESSING_SUBSPLIT 1 \
  --MODEL.CHECKPOINT "$DEPTHG_CKPT" \
  --SYSTEM.NUM_WORKERS $NUM_WORKERS \
  > "$REPO_ROOT/logs/e1_pseudo_subsplit1.log" 2>&1 &
PID1=$!
echo "subsplit 1 PID: $PID1"

# Subsplit 2
CUDA_VISIBLE_DEVICES=0 nohup python -u cups/pseudo_labels/gen_pseudo_labels.py \
  --DATA.DATASET cityscapes \
  --DATA.ROOT "$DATA1_ROOT/cityscapes/" \
  --DATA.PSEUDO_ROOT "$PSEUDO_OUT/" \
  --DATA.NUM_PREPROCESSING_SUBSPLITS $NUM_SUBSPLITS \
  --DATA.PREPROCESSING_SUBSPLIT 2 \
  --MODEL.CHECKPOINT "$DEPTHG_CKPT" \
  --SYSTEM.NUM_WORKERS $NUM_WORKERS \
  > "$REPO_ROOT/logs/e1_pseudo_subsplit2.log" 2>&1 &
PID2=$!
echo "subsplit 2 PID: $PID2"

echo "Both subsplits launched. Tail logs:"
echo "  tail -f $REPO_ROOT/logs/e1_pseudo_subsplit1.log"
echo "  tail -f $REPO_ROOT/logs/e1_pseudo_subsplit2.log"

wait $PID1
RC1=$?
wait $PID2
RC2=$?
echo "subsplit 1 exit: $RC1"
echo "subsplit 2 exit: $RC2"
test $RC1 -eq 0 -a $RC2 -eq 0 && echo "ALL SCRIPTS FINISHED!"
```

- [ ] **Step 2: Verify the launcher is syntactically valid**

```bash
bash -n refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 3: rsync to A6000**

```bash
rsync -avz refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh \
  cvpr_ug_5@gpunode2:$REPO_ROOT/refs/cups/cups/pseudo_labels/
ssh cvpr_ug_5@gpunode2 "chmod +x $REPO_ROOT/refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh"
```

- [ ] **Step 4: Commit**

```bash
git add refs/cups/cups/pseudo_labels/pseudolabel_gen_a6000.sh
git commit -m "feat(e1): A6000 launcher for CUPS pseudo-label generation (2-way subsplit)"
```

---

### Task 5: Smoke test on a 10-image subset

Goal: catch SF2SE3 OOM, missing-checkpoint, and wrong-data-path failures **before** committing to the full 2975-image run (~10 h ETA per memory).

**Files:**
- Create: `scripts/e1/smoke_test_pseudo_labels.sh`
- Create: `scripts/e1/check_pseudo_label_quality.py`

- [ ] **Step 1: Write `scripts/e1/check_pseudo_label_quality.py`**

```python
#!/usr/bin/env python3
"""scripts/e1/check_pseudo_label_quality.py.

Verify CUPS pseudo-labels are non-trivial. Reports:
  - total .pt / .png files
  - per-file: instance map non-empty (>=1 unique non-zero id)
  - per-file: semantic map has >=3 unique class ids
  - aggregate: % non-empty instance, % non-trivial semantic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _load_instance(p: Path) -> np.ndarray:
    if p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "instance" in obj:
            return np.asarray(obj["instance"])
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
    if p.suffix == ".png":
        from PIL import Image
        return np.asarray(Image.open(p))
    raise ValueError(f"unknown format: {p}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_root", required=True, type=Path)
    parser.add_argument("--instance_subdir", default="instance")
    parser.add_argument("--semantic_subdir", default="semantic")
    parser.add_argument("--min_instance_ids", type=int, default=1,
                        help="instance map is OK if it has >= this many non-zero ids")
    parser.add_argument("--min_semantic_classes", type=int, default=3)
    args = parser.parse_args()

    inst_dir = args.pseudo_root / args.instance_subdir
    sem_dir = args.pseudo_root / args.semantic_subdir

    inst_files = sorted(p for p in inst_dir.rglob("*") if p.suffix in {".pt", ".png"})
    sem_files = sorted(p for p in sem_dir.rglob("*") if p.suffix in {".pt", ".png"})
    print(f"instance files: {len(inst_files)}")
    print(f"semantic files: {len(sem_files)}")

    n_empty = 0
    for p in inst_files:
        try:
            arr = _load_instance(p)
            ids = set(np.unique(arr).tolist()) - {0}
            if len(ids) < args.min_instance_ids:
                n_empty += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  load fail {p.name}: {exc}")
            n_empty += 1

    pct_empty = 100.0 * n_empty / max(len(inst_files), 1)
    print(f"empty/broken instance maps: {n_empty}/{len(inst_files)} ({pct_empty:.1f}%)")

    n_trivial_sem = 0
    for p in sem_files[: min(len(sem_files), 200)]:  # sample to keep this fast
        arr = _load_instance(p)
        if len(np.unique(arr)) < args.min_semantic_classes:
            n_trivial_sem += 1
    print(f"trivial semantic maps (sample of {min(len(sem_files), 200)}): {n_trivial_sem}")

    return 0 if pct_empty < 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `scripts/e1/smoke_test_pseudo_labels.sh`**

```bash
#!/bin/bash
# scripts/e1/smoke_test_pseudo_labels.sh
# Runs CUPS pseudo-label gen on Cityscapes subsplit 1 with NUM_PREPROCESSING_SUBSPLITS=300
# (i.e. ~10 images), then runs the quality verifier. Goal: catch OOM / missing checkpoints
# / bad paths in <10 minutes before the full 10h run.
set -euo pipefail

: "${DATA1_ROOT:?}"
: "${REPO_ROOT:?}"
: "${DEPTHG_CKPT:?}"

PSEUDO_OUT="$DATA1_ROOT/cityscapes/cups_pseudo_labels_smoke"
mkdir -p "$PSEUDO_OUT"
mkdir -p "$REPO_ROOT/logs"

cd "$REPO_ROOT/refs/cups"

CUDA_VISIBLE_DEVICES=0 python -u cups/pseudo_labels/gen_pseudo_labels.py \
  --DATA.DATASET cityscapes \
  --DATA.ROOT "$DATA1_ROOT/cityscapes/" \
  --DATA.PSEUDO_ROOT "$PSEUDO_OUT/" \
  --DATA.NUM_PREPROCESSING_SUBSPLITS 300 \
  --DATA.PREPROCESSING_SUBSPLIT 1 \
  --MODEL.CHECKPOINT "$DEPTHG_CKPT" \
  --SYSTEM.NUM_WORKERS 2 \
  2>&1 | tee "$REPO_ROOT/logs/e1_smoke.log"

cd "$REPO_ROOT"
python scripts/e1/check_pseudo_label_quality.py --pseudo_root "$PSEUDO_OUT"
```

- [ ] **Step 3: rsync + run on A6000**

```bash
rsync -avz scripts/e1/check_pseudo_label_quality.py scripts/e1/smoke_test_pseudo_labels.sh \
  cvpr_ug_5@gpunode2:$REPO_ROOT/scripts/e1/
ssh cvpr_ug_5@gpunode2 "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV &&
  export DATA1_ROOT=$DATA1_ROOT REPO_ROOT=$REPO_ROOT DEPTHG_CKPT=$DEPTHG_CKPT &&
  bash $REPO_ROOT/scripts/e1/smoke_test_pseudo_labels.sh
"
```

Expected: `gen_pseudo_labels.py` finishes without OOM, then `check_pseudo_label_quality.py` reports `empty/broken instance maps: 0/N (0.0%)` (or at most 1/10).

- [ ] **Step 4: If smoke test fails — debug before scaling up**

Likely failure modes and fixes:

| Symptom | Root cause | Action |
|---|---|---|
| `CUDA out of memory. Tried to allocate ...` in log | SF2SE3 OOM | Reduce `--SYSTEM.NUM_WORKERS` to 1, or shrink `CROP_RESOLUTION` to (768, 1536). |
| `FileNotFoundError: .../depthg.ckpt` | DepthG ckpt missing | Re-run Task 3 Step 3. |
| `KeyError: 'leftImg8bit_sequence'` | Cityscapes Stereo+Video missing | Re-run Task 2. |
| Quality verifier reports >50% empty | Silent SF2SE3 failure | grep `cuda` and `error` in `e1_smoke.log` — likely OOM. Reduce workers and retry. |

Do **not** proceed to Task 6 until smoke test reports `<5%` empty.

- [ ] **Step 5: Commit (regardless of pass/fail) — fixes go in follow-up commits**

```bash
git add scripts/e1/check_pseudo_label_quality.py scripts/e1/smoke_test_pseudo_labels.sh
git commit -m "feat(e1): smoke test + quality verifier for CUPS pseudo-labels"
```

---

### Task 6: Run full Cityscapes pseudo-label generation

**Files:** none new; uses Task 4 launcher.

- [ ] **Step 1: Confirm prerequisites**

Before launching the long run, re-check:

```bash
ssh cvpr_ug_5@gpunode2 "
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv &&
  df -h $DATA1_ROOT &&
  ls -la $DEPTHG_CKPT
"
```

Expected: GPU `memory.used` < 1000 MiB (no zombie processes), Data1 free space > 50 GB, DepthG ckpt present.

- [ ] **Step 2: Launch the 2-way parallel run**

```bash
ssh cvpr_ug_5@gpunode2 "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV &&
  export DATA1_ROOT=$DATA1_ROOT REPO_ROOT=$REPO_ROOT DEPTHG_CKPT=$DEPTHG_CKPT &&
  cd $REPO_ROOT/refs/cups &&
  nohup bash cups/pseudo_labels/pseudolabel_gen_a6000.sh \
    > $REPO_ROOT/logs/e1_pseudolabel_master.log 2>&1 &
  echo \"master PID: \$!\"
"
```

Expected: prints master PID. The launcher script tees its own per-subsplit logs.

- [ ] **Step 3: Monitor**

```bash
ssh cvpr_ug_5@gpunode2 "tail -n 5 $REPO_ROOT/logs/e1_pseudo_subsplit1.log $REPO_ROOT/logs/e1_pseudo_subsplit2.log"
```

Use `Monitor` (deferred tool) or a recurring `/loop` to poll every ~30 min. The earlier santosh run reported ~12.5 s/img on Pass 2; with 2 parallel processes on A6000 (≈2× faster per process, since no DepthG↔SF2SE3 contention) ETA is **~3-5 hours**.

- [ ] **Step 4: Wait for completion**

Either wait synchronously (`ssh cvpr_ug_5@gpunode2 "wait <master_pid>"`) or poll the log for `ALL SCRIPTS FINISHED!`.

Expected: both subsplits exit 0, master log ends with `ALL SCRIPTS FINISHED!`.

- [ ] **Step 5: No commit — outputs live on Data1, not in git**

---

### Task 7: Verify pseudo-label quality (full run)

**Files:** none new.

- [ ] **Step 1: Run the verifier on the full output**

```bash
ssh cvpr_ug_5@gpunode2 "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV &&
  cd $REPO_ROOT &&
  python scripts/e1/check_pseudo_label_quality.py \
    --pseudo_root $DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2 \
    2>&1 | tee logs/e1_pseudolabel_quality.log
"
```

Expected:
- `instance files: 2975`
- `semantic files: 2975`
- `empty/broken instance maps: 0/2975 (0.0%)` — or at most ~3% for a healthy run.
- `trivial semantic maps (sample of 200): 0`

If `empty/broken` exceeds 10%, **stop and investigate** (compare against `e1_pseudo_subsplit1.log` for OOM lines) before declaring the run done.

- [ ] **Step 2: Compare aggregate stats vs santosh broken run**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $REPO_ROOT && python -c '
import torch, numpy as np
from pathlib import Path
root = Path(\"$DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2/instance\")
files = sorted(root.rglob(\"*.pt\"))[:500]
n_ids = []
for p in files:
    obj = torch.load(p, weights_only=False, map_location=\"cpu\")
    arr = obj[\"instance\"] if isinstance(obj, dict) else obj
    n_ids.append(len(set(np.unique(np.asarray(arr)).tolist()) - {0}))
print(f\"avg unique instance ids per image (sample of 500): {np.mean(n_ids):.2f}\")
print(f\"min, median, max: {np.min(n_ids)}, {int(np.median(n_ids))}, {np.max(n_ids)}\")
'
"
```

Expected: avg ≥ 5 unique instance ids per image. (santosh's broken run averaged ~0.15.)

- [ ] **Step 3: Save quality report into project memory**

Append a note to `~/.claude/projects/.../memory/cups_dinov3_control_experiment.md`:

```markdown
## 2026-04-XX: A6000 regeneration

- Pipeline: refs/cups/cups/pseudo_labels/gen_pseudo_labels.py (upstream, unmodified)
- Hardware: RTX A6000 48 GB
- Output: $DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2/
- Files: 2975 instance + 2975 semantic
- Empty instance maps: <X>%  (vs santosh broken run 84.5%)
- Avg unique instance ids per image: <Y>  (vs santosh ~0.15)
- DepthG ckpt: <hash>
- Wall-clock: <hours>
- Verdict: READY for Stage-2 / NEEDS rerun
```

- [ ] **Step 4: Commit logs and quality report**

```bash
git add logs/e1_pseudolabel_quality.log docs/plans/2026-04-27-e1-environment-snapshot.md
git commit -m "chore(e1): record A6000 pseudo-label quality stats"
```

---

### Task 8: Stage labels for downstream Stage-2 training

**Files:**
- Modify: `refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml` (path-only edit, optional — branch off if author cannot edit upstream copy)

This task does **not** start training; it just makes the new pseudo-label set discoverable from the existing Stage-2 config.

- [ ] **Step 1: Decide the canonical path**

Option A (preferred): leave outputs at `$DATA1_ROOT/cityscapes/cups_pseudo_labels_pipeline_v2/` and update Stage-2 configs to point there.

Option B: symlink `$DATA1_ROOT/cityscapes/cups_pseudo_labels` → `cups_pseudo_labels_pipeline_v2/` so existing configs don't need editing.

Use Option B unless the user has another active CUPS pseudo-label set on the same drive.

- [ ] **Step 2: Create the symlink (Option B)**

```bash
ssh cvpr_ug_5@gpunode2 "
  cd $DATA1_ROOT/cityscapes &&
  test -e cups_pseudo_labels && mv cups_pseudo_labels cups_pseudo_labels.santosh_broken_$(date +%s) || true ;
  ln -s cups_pseudo_labels_pipeline_v2 cups_pseudo_labels &&
  ls -la cups_pseudo_labels
"
```

Expected: symlink resolves to `cups_pseudo_labels_pipeline_v2`.

- [ ] **Step 3: Mark plan task complete**

Append to `docs/plans/2026-04-27-e1-environment-snapshot.md`:

```markdown
## Pseudo-label staging

- Canonical path: $DATA1_ROOT/cityscapes/cups_pseudo_labels (-> cups_pseudo_labels_pipeline_v2)
- Stage-2 config to use: refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml
  - update DATA.ROOT, DATA.ROOT_VAL, DATA.ROOT_PSEUDO to A6000-on-Data1 paths
- Stage-3 config to use: refs/cups/configs/train_self_cityscapes_dinov3_vitb_cups_official_1gpu.yaml
```

- [ ] **Step 4: Final commit**

```bash
git add docs/plans/2026-04-27-e1-environment-snapshot.md
git commit -m "chore(e1): pseudo-labels staged on Data1; ready for Stage-2 plan"
```

---

## Self-Review Checklist

- **Spec coverage:** all five user-stated needs are addressed: (1) target machine = A6000 — Tasks 1, 3-7; (2) data on Data1 — Tasks 1-2, 7-8; (3) regenerate pseudo-labels = full pipeline — Tasks 4-7; (4) clear blocker memory of OOM — explicit smoke test (Task 5) + verifier (Task 5+7); (5) Stage-2/3 deliberately scoped out — only path staging in Task 8.
- **Placeholder scan:** the only deliberate placeholders are environment values (`DATA1_ROOT`, `CITYSCAPES_USERNAME`/`_PASSWORD`) which Task 1/2 explicitly resolve before they're consumed.
- **Type consistency:** the verifier's CLI flags (`--pseudo_root`, `--instance_subdir`, `--semantic_subdir`) are used identically in Tasks 5 and 7. The launcher's env var contract (`DATA1_ROOT`, `REPO_ROOT`, `DEPTHG_CKPT`) is consistent across smoke test, full run, and verifier.
- **Risk note:** if Task 1 surfaces that `leftImg8bit_sequence` is not on Data1 and Data1 has <400 GB free, Task 2 will block. The plan includes a fallback (use the already-unpacked subset) that the executing agent should escalate to the user before downloading.

---

## Open follow-on plans (not in scope here)

1. **e1 Stage-2 training plan** — once pseudo-labels verified clean, train DINOv3 ViT-B/16 + Cascade Mask R-CNN for 8000 steps on the new labels. Expected wall-clock on A6000: ~10 h.
2. **e1 Stage-3 self-training plan** — 3 rounds × 500 steps EMA self-training from the Stage-2 checkpoint.
3. **e1 evaluation + decision plan** — run `evaluate_with_hungarian.py`, populate the decision matrix from `cups_dinov3_control_experiment.md`, and update the paper accordingly.
