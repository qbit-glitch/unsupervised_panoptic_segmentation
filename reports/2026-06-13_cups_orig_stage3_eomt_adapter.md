# Stage-3: ORIGINAL CUPS Pipeline with DINOv2 + EoMT (D2 Adapter)

**Date:** 2026-06-13
**Host:** `santosh@172.17.254.146`, 2× GTX 1080 Ti
**Supersedes:** both prior EoMT-side Stage-3 attempts
(`eomt_base_640_santosh_stage3.yaml` — slow decline 28.8→25.8; and
`eomt_base_640_santosh_stage3_cupsexact.yaml` — collapse 16.4→4.0).

## Approach

Per user direction: instead of porting the CUPS protocol into the EoMT
trainer, run the **unmodified official CUPS Stage-3 pipeline** and swap only
the network. Everything below executes CUPS's original code:

- `cups/pl_model_self.py` `SelfSupervisedModel` — teacher TTA inference,
  `make_pseudo_labels` (relative per-class semantic threshold with per-pixel
  argmax containment + absolute 0.5 instance floor), EMA 0.999/batch,
  head-only AdamW lr=1e-4 wd=1e-5, grad clip 1.0.
- `cups/augmentation.py` — CopyPasteAugmentation (max 3), Photometric,
  RandomCrop(512–1024, ×2), ResolutionJitter [(384,768),(416,832),(448,896)].
- `cups/data/cityscapes.py` — CityscapesSelfTraining (clean 640×1280 train
  images) + CityscapesPanopticValidation (**real 27-class GT, Hungarian
  matching — pq_val is now directly comparable to CUPS's published 27.8**).
- `train_self.py` trainer wiring — bs=1 × acc 8 × 2-GPU DDP = effective 16,
  3×4000 = 12000 optimizer steps, val every 200 batches, top-6 ckpts by
  pq_val.

## New code (adapter only)

| artefact | path |
|---|---|
| D2 adapter | `refs/cups/cups/model/eomt_d2_adapter.py` |
| Entry script (builder-only diff vs train_self.py) | `refs/cups/train_self_eomt.py` |
| Config (copy of canonical stage-3 yaml) | `refs/cups/configs/train_self_cityscapes_eomt_dinov2_dcfa_simcf_abc_santosh.yaml` |
| Launcher (SMOKE=1 for foreground check) | `scripts/run_cups_stage3_eomt_santosh.sh` |

`EoMTPanopticShim` translates between EoMT's unified k=80 space and CUPS's
split spaces (thing index 0–15 / stuff semantic channel 1–64, channel 0 =
things): train forward consumes D2 `{image, sem_seg, instances}` pseudo-
labels and returns the weighted Mask2Former loss dict; eval forward returns
D2 `{panoptic_seg: (map, segments_info), sem_seg: (65,H,W)}` via EoMT's
panoptic merge (0.5 score floor, argmax competition, 0.8 overlap pruning).
`EoMTWithTTA` mirrors `PanopticFPNWithTTA` (scales 0.5/0.75/1.0 + flip,
per-query logit averaging) and exposes `.model` for CUPS's EMA update.
Backbone frozen except the last 3 blocks (the EoMT decoder); CUPS's
head-only optimizer name filter (`"head" in name and "norm" not in name`)
applies via the `eomt_head` attribute name.

## Why the previous attempt collapsed

The EoMT-side port applied CUPS's relative stuff threshold per QUERY: any
query that was its class's best scorer was kept as a full target, even at
score 0.03 — ~60 junk masks/image. In CUPS the relative threshold operates
on per-pixel semantic maps where a weak class only claims pixels it wins in
the argmax. The original `make_pseudo_labels` (now running unmodified)
contains this by construction.

## Verification before launch

- Local CPU smoke: D2→EoMT target conversion exact (thing idx / stuff
  channel ↔ unified ids), weighted per-layer loss + backward, eval output
  consumed by CUPS's own `prediction_to_standard_format`, TTA path OK.
- Remote 1-GPU foreground smoke (SMOKE=1) before full DDP launch.
- Remote `refs/cups/cups/` package re-synced from local (remote was stale,
  missing `stage4_utils`).
