# E1 — Hugging Face Upload Layout & A6000 Run Recipe

This is the operational checklist for E1 (same-backbone CUPS rerun). It
covers (a) the populated public HF repo the A6000 box pulls from with
`wget` (no auth), (b) how to launch Stage-1 pseudo-label regeneration, and
(c) the Stage-2 hand-off.

## 1. Public Hugging Face repo (already populated)

Repo: <https://huggingface.co/qbit-glitch/mbps-cups-e1>

| File | Size | SHA256 |
|---|---|---|
| `depthg.ckpt`   | 352 639 220 B | `09556e83b791a8177a369c218800d54bc8bfa77a61759840d9758f6211ecab39` |
| `raft_smurf.pt` |  21 049 890 B | `692f1deece783ae69d058608b18ddeb97096c2505bd4529a1a08b8f5cab20d47` |
| `README.md`     | provenance + license | — |

Anonymous wget URL pattern:

```
https://huggingface.co/qbit-glitch/mbps-cups-e1/resolve/main/<filename>
```

**Not mirrored** (intentionally):

- `cups.ckpt` — the AFTER-Stage-3 final panoptic checkpoint. Using it as a
  Stage-1 input would be circular.
- `dino_RN50_pretrain_d2_format.pkl` — only needed for Stage-2 backbone init
  in CUPS-original. E1 swaps in DINOv3 ViT-B/16 instead.

If a future control experiment needs them, fetch directly from TUdatalib.

## 2. A6000 Stage-1 launch

### 2a. Generic A6000 box

```bash
export HF_REPO_BASE="https://huggingface.co/qbit-glitch/mbps-cups-e1/resolve/main"
export CITYSCAPES_ROOT=/data/cityscapes      # must contain leftImg8bit_sequence/, rightImg8bit_sequence/, camera/, gtFine/
export WORK_DIR=$HOME/mbps_e1
export GPU_ID=0
export NUM_SUBSPLITS=2                       # 2 parallel processes share the 48 GB

bash scripts/e1_stage1_pseudolabel_gen_a6000.sh
```

### 2b. Air-gapped `gpunode2` (RTX A6000 48 GB, NO internet) + master login node (internet, no GPU)

This is the cluster topology in use: gpunode2 has the A6000 but no
external network; the master/login node has internet but no GPU. We
two-step it: master *stages* (downloads HF assets + git pulls), then
gpunode runs the offline launcher. This works as long as `/home` is
mounted on both nodes (typical NFS layout). For non-shared FS, see § 3.

#### Step 1 — master node (internet, no GPU)

```bash
# Anything with /home/cvpr_ug_5 mounted will do.
cd /home/cvpr_ug_5/umesh
# Get the stager onto the box. If the repo is already cloned anywhere on
# shared FS, just `git pull` there; otherwise the stager will clone for you.
REPO_ROOT=/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation \
bash /home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation/scripts/e1_master_stage_assets.sh
```

If the repo isn't on the box yet, prime it once (one-shot, master node only):

```bash
git clone -b implement-dora-adapters \
    https://github.com/qbit-glitch/unsupervised_panoptic_segmentation.git \
    /home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation
```

Stager output ends with the exact paste-block for the GPU node, with the
right `REPO_ROOT`, `SKIP_GIT=1`, `SKIP_DOWNLOAD=1` already filled in.

#### Step 2 — Anydesk into `gpunode2` (offline)

The stager prints these commands; copy them verbatim:

```bash
cd /home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation
module load gcc-9.3.0
source ~/umesh/ups_env/bin/activate
export PATH=$HOME/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$HOME/cuda-12.1

export REPO_ROOT=/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation
export CITYSCAPES_ROOT=/home/cvpr_ug_5/umesh/datasets/cityscapes
export WORK_DIR=/home/cvpr_ug_5/umesh
export OUTPUT_DIR=/home/cvpr_ug_5/umesh/cups_pseudo_labels_e1
export GPU_ID=0
export NUM_SUBSPLITS=2
export NUM_WORKERS=6
export SKIP_DEPS=1
export SKIP_GIT=1
export SKIP_DOWNLOAD=1
export PYTHON_BIN="$(which python)"

nohup bash scripts/e1_stage1_pseudolabel_gen_a6000.sh \
    > /home/cvpr_ug_5/umesh/e1_stage1.out 2>&1 &
echo "PID=$!"
```

Monitor:

```bash
tail -f /home/cvpr_ug_5/umesh/e1_stage1.out
nvidia-smi   # both processes on GPU 0
```

> If the venv lacks CUPS deps (kornia, pykeops, opencv, open3d, etc.),
> drop `SKIP_DEPS=1` on the first attempt — the launcher will
> `pip install -r refs/cups/requirements.txt` against the venv, and that
> step does need internet, so do it on a node that has it (or use
> `pip download` from master and `pip install --no-index`).

## 3. Plan B — `/home` is NOT shared between master and gpunode

Stage on master to a local path, then rsync over the cluster's internal
network to the GPU node:

```bash
# On master:
LOCAL_STAGE=/scratch/$USER/e1_stage REPO_ROOT=$LOCAL_STAGE/unsupervised_panoptic_segmentation \
    bash scripts/e1_master_stage_assets.sh
rsync -avh --progress $LOCAL_STAGE/unsupervised_panoptic_segmentation/ \
    gpunode2:/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation/
```

The rsync transfers the freshly-staged repo (≈600 MB with the two HF
checkpoints) over the cluster LAN — usually a few seconds. After that,
gpunode2 commands are unchanged.

### 2c. Wall-clock + resume notes

Wall-clock estimate on a single A6000: **~6–8 h** for 2975 train images
(was ~10 h ETA on 1080 Ti before the OOMs).

The launcher is resume-friendly. Re-running skips already-downloaded
checkpoints and any image whose `*_semantic.png` + `*_instance.png` pair
already exists in `OUTPUT_DIR`.

## 3. What this script enforces (and why)

These were the four failure modes from our prior 1080 Ti run; the script
either fixes them upfront or gates on them post-run.

| Issue | How it manifested | Fix in this run |
|---|---|---|
| 11 GB VRAM ceiling | `CUDA out of memory. Tried to allocate 2.41 GiB` in SF2SE3 | Pre-flight checks reject GPUs <16 GB; A6000 has 48 GB. |
| 84.5% empty instance maps | OOM caught silently, instance map = all-zeros, downstream PQ_things=0% | `e1_verify_pseudolabels.py` requires `<=1%` empty instances. |
| 461 non-empty training samples (vs 2975) | Same root cause | Verifier requires `>=2975` matched pairs. |
| 3 config drifts vs CUPS-original | `MAX_NUM_PASTED_OBJECTS=3`, `NUM_STEPS_STARTUP=500`, `IGNORE_UNKNOWN_THING_REGIONS=True` | Launcher writes `config_pseudo_labels_e1.yaml` with CUPS defaults `8`, `1000`, `False`. |

## 4. Stage-2 hand-off (separate, after Stage-1 PASSES verify)

Stage-2 has a known bug in the training-time dataloader that is unrelated
to Stage-1 generation but blocks training:

**Bug**: `refs/cups/cups/data/pseudo_label_dataset.py` lines 355–382 apply
`scale_factor=self.ground_truth_scale` (=0.625) to labels that are already
generated at `640×1280`. The result is `400×800`, then `CenterCrop(640,1280)`
fails because the input is smaller than the crop.

**Fix** (apply once before launching Stage-2):

```python
# In refs/cups/cups/data/pseudo_label_dataset.py, replace each F.interpolate
# call that uses scale_factor=self.ground_truth_scale with an explicit size
# matching the labels' native resolution, e.g.:
#   F.interpolate(label[None, None].float(), size=(target_h, target_w), mode="nearest")
```

After patching, launch Stage-2:

```bash
python refs/cups/train.py \
    --config refs/cups/configs/train_cityscapes_dinov3_vitb_cups_official_1gpu.yaml \
    DATA.ROOT_PSEUDO $WORK_DIR/cups_pseudo_labels_e1 \
    DATA.ROOT $CITYSCAPES_ROOT
```

then Stage-3 self-training using the matching `train_self_*` config.

## 5. Decision matrix (NeurIPS Limitation #1 in the paper)

After Stage-3 reports PQ on Cityscapes val:

| CUPS+DINOv3 (E1) PQ | Interpretation | Paper action |
|---|---|---|
| **>33 PQ** | Backbone alone explains most of the 35.83 vs 27.80 gap | Pivot to analysis paper, reframe contributions around backbone scaling |
| **30–33 PQ** | Backbone explains ~half; pseudo-label source genuinely contributes | Reframe contributions; keep DCFA + SIMCF-ABC as the "what we add on top" story |
| **<30 PQ** | Our pseudo-labels are the dominant lever | Strengthen method claims; this is the strongest outcome for the current title |

Our final number (35.83 PQ on Cityscapes val) becomes meaningful only
relative to this E1 baseline.
