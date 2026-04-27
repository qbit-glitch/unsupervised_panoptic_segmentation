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

| Package | pkgID | Status | Action |
|---|---|---|---|
| `leftImg8bit/{train,val,test}` | 3 | ✅ present (2975/500/1525) | symlink into `/Data1/cityscapes/` |
| `leftImg8bit_sequence/{train,val,test}` | 14 | ✅ present (2975/500/1525) | symlink — but **verify per-image frame count** (warning in `e1_download_cityscapes.sh` re: 30-frame vs 1-frame stubs) |
| `gtFine/{train,val,test}` | 1 | ✅ present (11 900/2000/6100) | symlink into `/Data1/cityscapes/` |
| `rightImg8bit/{train,val,test}` | 4 | ❌ **MISSING** | download to `/Data1/cityscapes/rightImg8bit/` (~11 GB) |
| `rightImg8bit_sequence/{train,val,test}` | 15 | ❌ **MISSING** | download to `/Data1/cityscapes/rightImg8bit_sequence/` (~324 GB) |
| `camera/{train,val,test}` | 8 | ❌ **MISSING** | download to `/Data1/cityscapes/camera/` (~2 MB) |

Total missing download: **~335 GB**. Comfortable on `/Data1` (1.8 TB free).

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
export CITYSCAPES_ROOT="/Data1/cityscapes"   # after layout setup; existing left/seq/gtFine symlinked in
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

1. **Cityscapes layout = symlink-existing + download-missing-on-Data1.** Saves ~120 GB I/O versus a full move. New canonical root: `/Data1/cityscapes/`.
2. **Reuse the existing `scripts/e1_*.sh` toolchain** (master_stage_assets, stage1_pseudolabel_gen, check_cityscapes, download_cityscapes, verify_pseudolabels). The new `scripts/e1/` directory holds *only* the discovery script + a small Data1-layout helper.
3. **Plan revision** (`docs/plans/2026-04-27-e1-cups-pseudo-label-generation.md`) trims duplicate tasks and points at the existing scripts.

## Open user decisions

1. **GPU contention** — is the 33 GB ours or another user's? If another user's, OK to set `NUM_SUBSPLITS=1` and proceed, or wait?
2. **`leftImg8bit_sequence` completeness** — the `e1_download_cityscapes.sh` warning says: if each `<city>/` only has 1 file per labeled image (frame `000019`), the full 30-frame package was never extracted. Need to verify before relying on it for SMURF flow.
