# E1 A6000 Environment Snapshot — 2026-04-27

Captured from `scripts/e1/discover_a6000_environment.sh` run.

## Host

- **Hostname:** `master` (login node — see contention note below)
- **User:** `cvpr_ug_5`
- **OS:** RHEL 8 / kernel 4.18.0-348.el8.0.2.x86_64
- **Active shell env:** `(ups_env) (base)` — venv-style, *not* conda env named `ups`. Located at `~/umesh/ups_env`.

## GPU

| Idx | Name | Total | Used | Free | Driver |
|-----|------|------:|-----:|-----:|-------:|
| 0 | RTX A6000 | 49 140 MiB | **33 766 MiB** | **14 904 MiB** | 550.78 |

**⚠ Contention note (open question for user):** GPU is currently 70% loaded by some other process. CUPS Stage-1 needs ~13 GB for SF2SE3 + DepthG combined. With only ~14.5 GB free, parallel `NUM_SUBSPLITS=2` (≥ 14 GB total) will OOM. Either: (a) wait for the other job, (b) run with `NUM_SUBSPLITS=1` and accept the ~2× wall-clock penalty (~10 h instead of 5 h), or (c) confirm the 33 GB is our own stale process and clear it.

## Data1 drive

- **Mount:** `/Data1`
- **Free:** 1.8 TB
- **Owner / perms:** `root:root`, `1777` (sticky+world-writable, like `/tmp`) — writable by `cvpr_ug_5`.

## Cityscapes — currently at `/home/cvpr_ug_5/umesh/datasets/cityscapes/`

CUPS Stage-1 (`CityscapesStereoVideo`) needs **6** packages:

| Package | pkgID | Status (on /home) | Action (default) |
|---|---|---|---|
| `leftImg8bit/{train,val,test}` | 3 | ✅ present (2975/500/1525) | download fresh to `/Data1/cityscapes/` (~11 GB) |
| `leftImg8bit_sequence/{train,val,test}` | 14 | ✅ present (2975/500/1525) | download fresh to `/Data1/cityscapes/` (~119 GB) |
| `gtFine/{train,val,test}` | 1 | ✅ present (11 900/2000/6100) | download fresh to `/Data1/cityscapes/` (~0.25 GB) |
| `rightImg8bit/{train,val,test}` | 4 | ❌ MISSING | download to `/Data1/cityscapes/` (~11 GB) |
| `rightImg8bit_sequence/{train,val,test}` | 15 | ❌ MISSING | download to `/Data1/cityscapes/` (~119 GB) |
| `camera/{train,val,test}` | 8 | ❌ MISSING | download to `/Data1/cityscapes/` (~2 MB) |

Total fresh download: **~260 GB unpacked**. Comfortable on `/Data1` (1.8 TB free).

**Default = full fresh download** (no symlinks): `/home` is presumed to be NFS on this HPC node, so symlinking the 119 GB `leftImg8bit_sequence` would force every CUPS pseudo-label read to traverse NFS during the multi-hour Stage-1 run. Local `/Data1` reads avoid that bottleneck. The symlink optimisation is opt-in via `USE_HOME_SYMLINKS=1`.

## Existing CUPS asset state

- ❌ `depthg.ckpt` (CUPS Pass-2 semantic ckpt) — **not found anywhere**.
- ❌ `raft_smurf.pt` (CUPS Pass-1 optical-flow + disparity) — **not found anywhere**.
- ✅ Both are downloadable from the public HF repo `huggingface.co/qbit-glitch/mbps-cups-e1` via `scripts/e1_master_stage_assets.sh` (with SHA verification baked in).

## Repo on A6000

- `/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation` — branch `implement-dora-adapters`, HEAD `5d3e0cf` (matches GitHub).

## Resolved variables for downstream tasks

```bash
export A6000_HOST="cvpr_ug_5@gpunode2"   # or login node `master` if SSH alias resolves there
export REPO_ROOT="/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation"
export CITYSCAPES_ROOT="/Data1/cityscapes"   # full fresh download by default
export WORK_DIR="/Data1"                      # pseudo-labels + temp on Data1
export OUTPUT_DIR="/Data1/cups_pseudo_labels_e1"
export VENV_ACTIVATE="$HOME/umesh/ups_env/bin/activate"   # NOT a conda env
export PYTHON_BIN="$HOME/umesh/ups_env/bin/python"
export DEPTHG_CKPT="$REPO_ROOT/weights/depthg.ckpt"
export SMURF_CKPT="$REPO_ROOT/refs/cups/cups/optical_flow/checkpoints/raft_smurf.pt"
export NUM_SUBSPLITS=2   # reduce to 1 if GPU contention persists
export NUM_WORKERS=6
```

## Decisions taken (auto mode, can be overridden)

1. **Cityscapes layout = full fresh download onto /Data1.** All 6 packages (~260 GB unpacked) downloaded into `/Data1/cityscapes/`. No symlinks to `/home` (NFS read bottleneck would slow CUPS Stage-1 substantially). Single canonical root: `/Data1/cityscapes/`. Symlink optimisation is opt-in via `USE_HOME_SYMLINKS=1` if your `/home` and `/Data1` share the same physical volume.
2. **Reuse the existing `scripts/e1_*.sh` toolchain** (master_stage_assets, stage1_pseudolabel_gen, check_cityscapes, download_cityscapes, verify_pseudolabels). The new `scripts/e1/` directory holds *only* the discovery script + a small Data1-layout helper.
3. **Plan revision** (`docs/plans/2026-04-27-e1-cups-pseudo-label-generation.md`) trims duplicate tasks and points at the existing scripts.

## Open user decisions

1. **GPU contention** — is the 33 GB ours or another user's? If another user's, OK to set `NUM_SUBSPLITS=1` and proceed, or wait?
2. **`leftImg8bit_sequence` completeness** — the `e1_download_cityscapes.sh` warning says: if each `<city>/` only has 1 file per labeled image (frame `000019`), the full 30-frame package was never extracted. Need to verify before relying on it for SMURF flow.
