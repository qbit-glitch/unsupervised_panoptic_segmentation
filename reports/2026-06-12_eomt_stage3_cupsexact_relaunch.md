# EoMT Stage-3 Relaunch — Exact Official CUPS Protocol

> **SUPERSEDED 2026-06-13.** This EoMT-side port collapsed (pseudo-val PQ
> 16.4 → 4.0 in 1000 optimizer steps): the query-level port of the CUPS
> relative stuff threshold kept every class's best query regardless of
> absolute score, flooding the target set with junk masks (no per-pixel
> argmax containment as in CUPS's semantic maps). Run killed at epoch 6.
> Replaced by the ORIGINAL CUPS pipeline with EoMT behind a Detectron2
> adapter — see `reports/2026-06-13_cups_orig_stage3_eomt_adapter.md`.

**Date:** 2026-06-12
**Run:** `cityscapes_panoptic_eomt_base_dinov2_dcfa_simcf_abc_spherical_k80_stage3_selftrain_cupsexact`
**Host:** `santosh@172.17.254.146`, PID 36298 (launched 19:09)
**Replaces:** PID 29428 (killed; declining run diagnosed in
`reports/2026-06-12_eomt_stage3_self_training_decline_diagnosis.md`)

## What changed vs the killed run

All 8 diagnosed config gaps closed, plus two protocol-level findings from a
line-by-line read of the official CUPS Stage-3 code
(`refs/cups/cups/pl_model_self.py`, `refs/cups/train_self.py`,
`refs/cups/cups/augmentation.py`):

| Knob | Old run | New run (CUPS-exact) |
|---|---|---|
| weight_decay | 0.05 | **1e-5** |
| LR | 5e-5 + warmup + poly | **1e-4 constant, no schedule** |
| Optimizer scope | all params, LLRD | **backbone frozen; last-3 blocks + norm + queries + heads (25.5M train / 65.3M frozen)** |
| Teacher TTA | none | **scales [0.5, 0.75, 1.0] × h-flip = 6 views, per-query logit averaging** |
| Copy-paste | off | **on, max 3 thing objects, scale 0.25–1.5, flip** |
| Photometric aug | dataloader color jitter only | **CUPS triple: blur p=1.0, jitter 0.4/0.4/0.4/0.1 p=0.5, grayscale p=0.2** |
| Multi-resolution | single 640×640 | **RandomCrop r∈[512,1024] (r,2r) → ResolutionJitter [(384,768),(416,832),(448,896)]** |
| Crop aspect | 640×640 square | **clean 640×1280 (0.625 aspect-preserving)** |
| Round steps | 3×700 = 2100 | **3×4000 = 12000 optimizer steps** |
| Teacher warmup | 100 steps | **0 (CUPS NUM_STEPS_STARTUP=0)** |
| Gradient clip | 0.01 | **1.0 norm (CUPS)** |
| Attn-mask annealing | re-enabled with new ramp | **disabled (keep Stage-2 endpoint; CUPS has no analog)** |
| Batch geometry | bs=1 × acc 8 × 2-GPU DDP = 16 | unchanged (CUPS-identical) |
| EMA | 0.999 per batch | unchanged (CUPS-identical) |

## Two findings about the "official" config

1. **`CLASS_THRESHOLD_ALPHA`, `CLASS_FREQUENCIES`, and `CONFIDENCE_STEP` are
   dead config in CUPS** — defined in `cups/config.py` and the yaml, but
   never consumed anywhere in the executed code. There is no round-based
   confidence ramp and no frequency-weighted threshold in the official
   Stage-3. The executed recipe is a CONSTANT threshold for all 12000 steps:
   - semantic (stuff): relative per-class threshold
     `0.5 × per-class spatial max` (this IS the class-balancing mechanism);
   - instances (things): absolute 0.5 detection floor.
   Ported to EoMT queries: stuff-labeled queries keep `score ≥ 0.5 × max
   score of that class in the image`; thing-labeled queries keep
   `score ≥ 0.5` absolute.

2. **CUPS applies all augmentation INSIDE training_step, AFTER teacher
   inference**: the teacher labels the clean full image; the student trains
   on the copy-pasted/photometric/cropped/jittered view. The killed run
   augmented in the dataloader, so teacher and student saw the same image —
   removing the input-asymmetry consistency regularisation that drives
   self-training. This is now reproduced exactly.

## Implementation

| artefact | path |
|---|---|
| New augmentation port | `refs/eomt/training/cups_stage3_augmentation.py` |
| Rewritten self-train module | `refs/eomt/training/mask_classification_panoptic_self_train.py` |
| Clean dataloader mode | `refs/eomt/datasets/cityscapes_panoptic_pseudo.py` (`self_train_clean: True`) |
| New config | `refs/eomt/configs/dinov3/cityscapes/panoptic/eomt_base_640_santosh_stage3_cupsexact.yaml` |
| Launcher | `scripts/run_eomt_stage3_cupsexact_santosh.sh` |
| Remote log | `santosh:/home/santosh/experiments/stage3_eomt_dinov2_vitb_dcfa_simcf_abc_spherical_k80_cupsexact/logs/eomt_stage3_cupsexact_20260612_190936.log` |
| Checkpoints (top-6 by val PQ + last) | `santosh:.../stage3_..._cupsexact/checkpoints/` |

EoMT's fixed `patch_embed.grid_size` is now set dynamically per forward
(`_set_grid_size`) to support the 6 TTA scales and the 3 training
resolutions.

Verified before launch: CPU smoke test ran one full training_step with the
real DINOv2-base EoMT (TTA 6 views → 62 pseudo-targets → copy-paste →
photometric → crop → jitter → loss 56.7 → backward → EMA update), and
parameter split confirmed 25.5M trainable / 65.3M frozen.

## Status at launch

- Both GPUs 100% util, 5–6 GB / 11 GB used, ~0.40 it/s (batch steps).
- Loss in expected initial range (7–12).
- ~180 optimizer steps/epoch → 12000 steps ≈ 67 epochs ≈ **3–4 days** on
  2× 1080 Ti (the 6-view teacher TTA dominates per-batch cost). Best-PQ
  checkpoints are harvested continuously (val every 200 batch steps), so
  intermediate checkpoints are usable before completion.
- Note: pseudo-val now runs at 640×1280 (was 320×640 letterboxed into
  640×640), so the first val tick re-baselines the seed PQ — do not compare
  raw values against the old run's 28.8 seed directly.
