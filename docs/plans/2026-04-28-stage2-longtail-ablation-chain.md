# Stage-2 Long-Tail Preservation — Ablation Chain on santosh

## Context

The MBPS Stage-2 trained model on T0 pseudo-labels (DCFA + DepthPro + SIMCF-ABC) currently leaves several Cityscapes thing classes severely under-represented in trained PQ — most notably **traffic light = 0 TP** and **motorcycle = 0 TP** at the pseudo-label level (intractable without external seeds), plus **wall/fence/pole/terrain/rider/truck/bus/train/bicycle** in the "Suppressed/Starved" regime where stronger learning signal CAN help.

T0 baseline (locally re-evaluated, hungarian remap): **PQ=25.09**, person TP=547, car TP=942, rider TP=176, bicycle TP=238 — meaningful gradient available for class-balanced loss + cross-batch copy-paste to amplify.

The CUPS codebase already implements EQLv2 (`refs/cups/cups/losses/long_tail.py:10-93`) and SeesawSoftmaxLoss (`:96-157`) and an in-batch CopyPasteAugmentation (`refs/cups/cups/augmentation.py:35-192`), but **all long-tail flags default OFF** and **no persistent rare-instance pool, no Repeat Factor Sampling, and no LDAM exist**. This plan adds the missing pieces and runs an ablation chain on the remote `santosh@172.17.254.146` (2× GTX 1080 Ti 11GB, conda env `cups`) to isolate per-feature impact, mirroring the user's prior T0–T4c ablation discipline.

The intended outcome is to reach the prior recommendation's projected **+5 PQ aggregate** (PQ ≈ 33–35 at Stage-2 end vs current trained baseline PQ=28.40), with non-zero TP gains for at least 4 of {wall, fence, terrain, rider, truck, bus, train, bicycle}.

---

## Decisions (confirmed with user)

- **Launch strategy**: Ablation chain — 3 sequential Stage-2 runs, each ~40h on santosh.
- **Pseudo-label root**: T0 (`cups_pseudo_labels_dcfa_simcf_depthpro`) — must verify or upload to santosh.
- **Rare-Pool boost**: Moderate — `4×` for person/rider/bicycle, `8×` for truck/bus/train; pole skipped (only 1 TP available); traffic-light/motorcycle skipped (0 TPs, deferred to external-seed work).
- **EQLv2 chosen over Seesaw** (mutual exclusivity at `fast_rcnn.py:287`): EQLv2's gradient-tracking buffers adapt during training; Seesaw's pre-computed prior is fragile under drifting pseudo-label distributions.

---

## Ablation chain (3 runs)

| Run | Features | New code? | Config flags | Wall-clock (santosh) | Decision gate |
|---|---|---|---|---|---|
| **A** | EQLv2 + RFS | NO (EQLv2 in code) + Sampler (~150 LOC) | `USE_EQLV2=True`, `USE_REPEAT_FACTOR_SAMPLER=True` | ~40h | proceed if PQ ≥ 28.4 OR ≥3 Suppressed classes gain TP |
| **B** | A + Rare-Instance Pool | + Pool builder (~180 LOC), + Augmentation class (~260 LOC) | + `USE_RARE_POOL=True` | ~40h | proceed if PQ ≥ Run A AND person/rider/bicycle TP grow |
| **C** | B + LDAM (semantic head) | + LDAM class (~70 LOC), + semantic_seg integration | + `LDAM_ENABLED=True` | ~40h | promote winner to Stage-3 |

---

## File-by-file changes

### Run A (EQLv2 + RFS)

1. **NEW** `refs/cups/cups/data/repeat_factor_sampler.py` (~150 LOC)
   - `class RepeatFactorTrainingSampler(torch.utils.data.Sampler)` — DDP-aware; mirrors Detectron2's pattern but operates on `PseudoLabelDataset`.
   - For each image, compute trainID set from semantic PNG (cache to JSON).
   - Class freq `freq[c] = images_containing(c) / N`. Per-class `r_c = max(1, sqrt(t/freq[c]))`. Per-image `r_i = max(r_c for c in image_classes_i)`. Threshold `t = 0.001`.
   - On each epoch, fractional rounding (Detectron2 trick) → integer indices. DDP shard via `torch.distributed.get_rank()` lazily.

2. **MODIFY** `refs/cups/cups/config.py` — append under `_C.DATA` (after line ~170):
   ```python
   _C.DATA.USE_REPEAT_FACTOR_SAMPLER = False
   _C.DATA.RFS_THRESHOLD_T = 0.001
   _C.DATA.RFS_PRECOMPUTE_CACHE = ""   # JSON cache path; empty = recompute
   ```

3. **MODIFY** `refs/cups/train.py` lines `170-182` — branch the dataloader:
   ```python
   if config.DATA.USE_REPEAT_FACTOR_SAMPLER:
       sampler = RepeatFactorTrainingSampler(
           training_dataset,
           threshold_t=config.DATA.RFS_THRESHOLD_T,
           num_samples=config.TRAINING.STEPS * config.SYSTEM.NUM_GPUS *
                       config.TRAINING.BATCH_SIZE * config.TRAINING.ACCUMULATE_GRAD_BATCHES,
           cache_path=config.DATA.RFS_PRECOMPUTE_CACHE or None,
       )
       train_dataloader = DataLoader(training_dataset, sampler=sampler,
                                     batch_size=config.TRAINING.BATCH_SIZE, ...)
   else:
       # existing StepDataset + shuffle=True path unchanged
   ```

4. **NEW** YAML `refs/cups/configs/train_cityscapes_t0_longtail_runA_santosh.yaml` — fork of `train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml`, with these overrides:
   ```yaml
   DATA:
     ROOT_PSEUDO: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro/"
     USE_REPEAT_FACTOR_SAMPLER: True
     RFS_THRESHOLD_T: 0.001
     RFS_PRECOMPUTE_CACHE: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/rfs_cache_t0.json"
   MODEL:
     ROI_BOX_HEAD:
       USE_EQLV2: True
       EQLV2_GAMMA: 12.0
       EQLV2_MU: 0.8
       EQLV2_ALPHA: 4.0
   SYSTEM:
     LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments/t0_longtail_runA"
     RUN_NAME: "cups_t0_longtail_runA"
   ```
   (All other keys inherited unchanged: BATCH_SIZE=1, ACCUMULATE_GRAD_BATCHES=8, STEPS=8000, PRECISION="32-true", LR=1e-4, COPY_PASTE=True with stock in-batch augmentation.)

5. **NEW** `refs/cups/tests/test_repeat_factor_sampler.py` (~80 LOC)
   - Toy 4-image dataset with class sets `[{0},{0},{0,1},{1}]` → assert per-class freq, per-image repeat factors, sampled-index histogram over-represents class-1 images.
   - `test_ddp_sharding`: simulate 4 replicas, assert no index repeats across replicas in one "epoch."

### Run B (adds Rare-Instance Pool)

6. **NEW** `scripts/build_rare_instance_pool.py` (~180 LOC) — runs ONCE locally, M4 Pro CPU.
   - Inputs: `--pseudo-dir`, `--image-dir`, `--depth-dir`, `--centroids` (npz with `cluster_to_class`), `--out`, `--max-per-class 3000`, `--workers 8`.
   - For each pseudo-label: per instance with `area ≥ 256` and bbox min-side ≥ 32, extract crop + mask + DepthPro depth median (normalised). Map cluster→trainID via centroids. Bucket into `pool[trainID]`.
   - `RARE_TRAINIDS = {3,4,5,6,8,9,11,12,14,15,16,18}` (wall, fence, pole, traffic_light, vegetation→drop, terrain, person, rider, truck, bus, train, bicycle — verify final list at runtime; `traffic_light=6` and `motorcycle=17` will be empty buckets, builder logs warning).
   - Encode crops as JPEG quality-85 + masks as PNG; pickle protocol-4 dict `{trainID: List[InstanceCrop]}`.
   - Output: `~/Desktop/datasets/cityscapes/rare_instance_pool/pool_t0.pkl` (~70 MB on disk).
   - Estimated build time: 12–20 min on M4 Pro.

7. **NEW** `refs/cups/cups/augmentation_rare_pool.py` (~260 LOC)
   - `class RareInstancePoolCopyPaste(nn.Module)` — duck-types `CopyPasteAugmentation.forward(batch_source, batch_target) -> List[Dict]`.
   - Constructor: `(thing_class, pool_path, scale_range=(0.25, 1.5), pastes_per_image=(1, 3), use_depth_placement=True, class_repeat_overrides=(), thing_id_to_trainid=None)`.
   - On init: `pool = pickle.load(pool_path)`; precompute `n_class[c]` and `weights[c] = (1/sqrt(n_class[c])) * override_factor[c]`.
   - `forward`:
     1. Ignore `batch_source`; for each target sample sample `K ~ randint(1,3)` rare classes by weights; uniformly sample one crop from `pool[c]`.
     2. Decode JPEG/PNG on the fly.
     3. **Depth-aware placement**: if target has `depth` (CUPS already loads it via `pseudo_label_dataset._add_depth_and_onehot:196-220`), pick paste position whose target depth median is within ±0.15 of crop's `src_depth_quantile`. 3 retries → fall back to random.
     4. Reuse the existing pad/blend/mask-update math from `CopyPasteAugmentation.forward:141-170` via a shared private helper `_paste_one(...)`.

8. **MODIFY** `refs/cups/cups/augmentation.py` — extract the existing pad/blend/mask-update code at lines `141-170` into a module-level helper `_paste_one(...)` so both `CopyPasteAugmentation` and `RareInstancePoolCopyPaste` share it. Keep `CopyPasteAugmentation.forward` semantics unchanged.

9. **MODIFY** `refs/cups/cups/config.py` — append under `_C.AUGMENTATION` (after line ~292):
   ```python
   _C.AUGMENTATION.USE_RARE_POOL = False
   _C.AUGMENTATION.RARE_POOL_PATH = ""
   _C.AUGMENTATION.RARE_POOL_PASTES_PER_IMAGE = (1, 3)
   _C.AUGMENTATION.RARE_POOL_USE_DEPTH_PLACEMENT = True
   _C.AUGMENTATION.RARE_POOL_CLASS_REPEAT_OVERRIDES = ((11, 4), (12, 4), (18, 4), (14, 8), (15, 8), (16, 8))
   ```

10. **MODIFY** `refs/cups/train.py` ~lines `208-215` (copy-paste constructor block) — branch on `config.AUGMENTATION.USE_RARE_POOL`:
    ```python
    if config.AUGMENTATION.USE_RARE_POOL:
        copy_paste_augmentation = RareInstancePoolCopyPaste(
            thing_class=config.MODEL.NUM_THING_CLASSES,
            pool_path=config.AUGMENTATION.RARE_POOL_PATH,
            pastes_per_image=config.AUGMENTATION.RARE_POOL_PASTES_PER_IMAGE,
            use_depth_placement=config.AUGMENTATION.RARE_POOL_USE_DEPTH_PLACEMENT,
            class_repeat_overrides=config.AUGMENTATION.RARE_POOL_CLASS_REPEAT_OVERRIDES,
        )
    elif config.AUGMENTATION.COPY_PASTE:
        # existing CopyPasteAugmentation construction unchanged
    ```

11. **NEW** YAML `refs/cups/configs/train_cityscapes_t0_longtail_runB_santosh.yaml` — fork of Run A YAML, additionally:
    ```yaml
    AUGMENTATION:
      USE_RARE_POOL: True
      RARE_POOL_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/rare_instance_pool/pool_t0.pkl"
      RARE_POOL_PASTES_PER_IMAGE: (1, 3)
      RARE_POOL_USE_DEPTH_PLACEMENT: True
      RARE_POOL_CLASS_REPEAT_OVERRIDES: ((11, 4), (12, 4), (18, 4), (14, 8), (15, 8), (16, 8))
    SYSTEM:
      LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments/t0_longtail_runB"
      RUN_NAME: "cups_t0_longtail_runB"
    ```

12. **NEW** tests:
    - `refs/cups/tests/test_rare_pool_builder.py` (~80 LOC) — synthetic 4×4 image with 2 fake instances; assert pool entries.
    - `refs/cups/tests/test_rare_pool_copy_paste.py` (~120 LOC) — mock pool dict with 3 fake jpeg/png crops; run forward; assert pasted classes/masks present.

### Run C (adds LDAM semantic loss)

13. **MODIFY** `refs/cups/cups/losses/long_tail.py` — append (~70 LOC):
    ```python
    class LDAMSemanticLoss(nn.Module):
        def __init__(self, num_classes, class_freq, max_margin=0.5, s=30.0,
                     class_weight=None, ignore_index=-1):
            super().__init__()
            m = 1.0 / (np.array(class_freq) + 1e-6) ** 0.25
            m = m * (max_margin / m.max())
            self.register_buffer("margins", torch.from_numpy(m).float())
            self.s = s
            self.ignore_index = ignore_index
            self.class_weight = class_weight

        def forward(self, logits, targets, **kw):
            valid = targets != self.ignore_index
            B, C, H, W = logits.shape
            margin_map = torch.zeros_like(logits)
            t_safe = targets.clamp(0, C - 1)
            margin_one_hot = self.margins[t_safe]
            margin_map.scatter_(1, t_safe.unsqueeze(1), margin_one_hot.unsqueeze(1))
            adj_logits = self.s * (logits - margin_map)
            cw = (torch.tensor(self.class_weight, device=logits.device)
                  if self.class_weight else None)
            return F.cross_entropy(adj_logits, targets, weight=cw,
                                   ignore_index=self.ignore_index, reduction="mean")
    ```

14. **MODIFY** `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py`:
    - Constructor: add `self.ldam_enabled = ldam_enabled; self.ldam = LDAMSemanticLoss(...)` lazily constructed if enabled.
    - `from_config`: read `cfg.MODEL.SEM_SEG_HEAD.LDAM_ENABLED / LDAM_MAX_MARGIN / LDAM_S / LDAM_CLASS_FREQ`.
    - `losses()` lines `298-304` (and weighted branch `286-296`): replace `F.cross_entropy(...)` with:
      ```python
      if self.ldam_enabled:
          loss = self.ldam(predictions, targets, ignore_value=self.ignore_value)
      else:
          loss = F.cross_entropy(predictions, targets, weight=class_weight, ignore_index=self.ignore_value)
      ```
      Apply same to weighted branch (M5 confidence weighting still works on top of LDAM-adjusted logits).

15. **MODIFY** `refs/cups/cups/config.py` — append under `_C.MODEL.SEM_SEG_HEAD` (after line ~137):
    ```python
    _C.MODEL.SEM_SEG_HEAD.LDAM_ENABLED = False
    _C.MODEL.SEM_SEG_HEAD.LDAM_MAX_MARGIN = 0.5
    _C.MODEL.SEM_SEG_HEAD.LDAM_S = 30.0
    _C.MODEL.SEM_SEG_HEAD.LDAM_CLASS_FREQ = ()  # empty → derived from class_distribution in train.py
    ```

16. **MODIFY** `refs/cups/train.py` — when constructing the model, if `LDAM_ENABLED`, derive `LDAM_CLASS_FREQ` from `pseudo_label_dataset.class_distribution` (already computed at lines `199-206`) and pass into the SEM_SEG_HEAD constructor via cfg.

17. **NEW** YAML `refs/cups/configs/train_cityscapes_t0_longtail_runC_santosh.yaml` — fork of Run B YAML, additionally:
    ```yaml
    MODEL:
      SEM_SEG_HEAD:
        LDAM_ENABLED: True
        LDAM_MAX_MARGIN: 0.5
        LDAM_S: 30.0
    SYSTEM:
      LOG_PATH: "/media/santosh/Kuldeep/panoptic_segmentation/experiments/t0_longtail_runC"
      RUN_NAME: "cups_t0_longtail_runC"
    ```

18. **NEW** `refs/cups/tests/test_ldam_loss.py` (~70 LOC) — uniform freq → equivalent to scaled CE; skewed freq → larger margin for rarer class; numerical match to a hand-computed 3-class example.

### Remote deployment infrastructure

19. **NEW** `scripts/run_t0_longtail_stage2_santosh.sh` (~80 LOC) — accepts `--run A|B|C` argument, single launcher:
    ```bash
    REMOTE=santosh@172.17.254.146
    REMOTE_REPO=/media/santosh/Kuldeep/panoptic_segmentation/mbps_panoptic_segmentation
    REMOTE_DATA=/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes
    RSYNC="rsync -avz --exclude=__pycache__ --exclude='*.pyc'"

    # 1. (Run B/C only) Build pool locally if missing
    if [ "$RUN" != "A" ] && [ ! -f ~/Desktop/datasets/cityscapes/rare_instance_pool/pool_t0.pkl ]; then
        python scripts/build_rare_instance_pool.py \
          --pseudo-dir ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro/train \
          --image-dir ~/Desktop/datasets/cityscapes/leftImg8bit/train \
          --depth-dir ~/Desktop/datasets/cityscapes/depth_pro/train \
          --centroids ~/Desktop/datasets/cityscapes/pseudo_semantic_raw_dinov3_k80/kmeans_centroids.npz \
          --out ~/Desktop/datasets/cityscapes/rare_instance_pool/pool_t0.pkl \
          --max-per-class 3000 --workers 8
    fi

    # 2. Verify T0 pseudo-labels on remote (upload if missing)
    ssh $REMOTE "test -d $REMOTE_DATA/cups_pseudo_labels_dcfa_simcf_depthpro" || \
      $RSYNC ~/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_depthpro/ \
             $REMOTE:$REMOTE_DATA/cups_pseudo_labels_dcfa_simcf_depthpro/

    # 3. Sync code + config + (B/C) pool
    $RSYNC refs/cups/ $REMOTE:$REMOTE_REPO/refs/cups/
    [ "$RUN" != "A" ] && $RSYNC ~/Desktop/datasets/cityscapes/rare_instance_pool/pool_t0.pkl \
      $REMOTE:$REMOTE_DATA/rare_instance_pool/

    # 4. Launch
    CONFIG="train_cityscapes_t0_longtail_run${RUN}_santosh.yaml"
    LOGDIR="$REMOTE_DATA/../experiments/t0_longtail_run${RUN}"
    ssh $REMOTE "bash -lc 'cd $REMOTE_REPO/refs/cups && \
      source ~/miniconda3/etc/profile.d/conda.sh && conda activate cups && \
      mkdir -p $LOGDIR && \
      CUDA_VISIBLE_DEVICES=0,1 nohup python train.py \
        --config-file configs/$CONFIG \
        > $LOGDIR/run.log 2>&1 & \
      echo PID=\$!'"
    echo "Monitor: ssh $REMOTE 'tail -f $LOGDIR/run.log'"
    ```

---

## Critical files for implementation (modify)

- `refs/cups/train.py` (lines 170-182 dataloader, 208-215 augmentation construction)
- `refs/cups/cups/config.py` (append flags under `_C.DATA`, `_C.AUGMENTATION`, `_C.MODEL.SEM_SEG_HEAD`)
- `refs/cups/cups/augmentation.py` (extract `_paste_one` helper from lines 141-170)
- `refs/cups/cups/losses/long_tail.py` (append `LDAMSemanticLoss`)
- `refs/cups/cups/model/modeling/roi_heads/semantic_seg.py` (replace `F.cross_entropy` at 286-304)

## Critical files for implementation (create)

- `refs/cups/cups/data/repeat_factor_sampler.py`
- `refs/cups/cups/augmentation_rare_pool.py`
- `scripts/build_rare_instance_pool.py`
- `scripts/run_t0_longtail_stage2_santosh.sh`
- `refs/cups/configs/train_cityscapes_t0_longtail_run{A,B,C}_santosh.yaml`
- `refs/cups/tests/test_repeat_factor_sampler.py`
- `refs/cups/tests/test_rare_pool_builder.py`
- `refs/cups/tests/test_rare_pool_copy_paste.py`
- `refs/cups/tests/test_ldam_loss.py`

## Reuse from existing code

- `refs/cups/cups/losses/long_tail.py:10-93` — `EQLv2Loss` (no changes)
- `refs/cups/cups/augmentation.py:141-170` — pad/blend/mask-update code (extract into shared helper)
- `refs/cups/cups/data/pseudo_label_dataset.py:130-156` — `class_distribution` (use as RFS / LDAM input)
- `refs/cups/cups/data/pseudo_label_dataset.py:196-220` — depth loading (used by depth-aware paste)
- `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py:404-407` — loss selection if/elif (untouched; just flip flag)
- `scripts/run_v3_adapter_stage2_santosh.sh` — launcher template (fork into the new launcher)

---

## Verification

### Local smoke (M4 Pro CPU, ~5 min, before each remote launch)

```bash
PY=/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python
$PY -m pytest refs/cups/tests/test_repeat_factor_sampler.py \
              refs/cups/tests/test_rare_pool_builder.py \
              refs/cups/tests/test_rare_pool_copy_paste.py \
              refs/cups/tests/test_ldam_loss.py -v
$PY -c "from cups.config import get_config; c = get_config(); \
        c.merge_from_file('refs/cups/configs/train_cityscapes_t0_longtail_runA_santosh.yaml'); \
        print('USE_EQLV2', c.MODEL.ROI_BOX_HEAD.USE_EQLV2, 'RFS', c.DATA.USE_REPEAT_FACTOR_SAMPLER)"
# Tiny 4-step CPU dry-run using train_cityscapes_dinov3_vitb_cups_official_cpu.yaml
# overlaid with the new flags + STEPS=4, BATCH_SIZE=2 — confirms loss is finite,
# augmentation runs, RFS sampler yields, EQLv2 buffers update.
```

### Remote first-1000-step checks

```bash
# After launch
ssh santosh@172.17.254.146 'tail -f /media/santosh/Kuldeep/panoptic_segmentation/experiments/t0_longtail_runA/run.log' &
ssh santosh@172.17.254.146 'watch -n 5 nvidia-smi'

# Verify within 1000 steps:
# - loss_cls, loss_sem_seg, loss_box_reg all finite (no NaN/inf)
# - per-GPU memory < 10.5 GB
# - (B/C) wandb logs rare_pool/pasted_per_batch ≈ 1-3
# - (A) wandb logs rfs/rare_class_share ≈ 0.20 (vs baseline ~0.09)
```

### Final eval (after each run completes at step 8000)

```bash
# Pull best checkpoint
scp santosh@172.17.254.146:/media/santosh/Kuldeep/panoptic_segmentation/experiments/t0_longtail_runA/best.ckpt \
    ./checkpoints/t0_longtail_runA_best.ckpt

# Eval on Cityscapes val
$PY refs/cups/evaluate_cityscapes.py \
    --config-file refs/cups/configs/train_cityscapes_t0_longtail_runA_santosh.yaml \
    --ckpt ./checkpoints/t0_longtail_runA_best.ckpt \
    --output reports/longtail/runA.json

# Compare per-class TP delta vs prior best (PQ=28.40 ckpt)
$PY scripts/compare_per_class.py reports/longtail/runA.json reports/longtail/baseline_pq28.40.json
```

### Decision gates

| After Run | Promote if | Else |
|---|---|---|
| A | PQ ≥ 28.4 OR ≥3 of {wall, fence, terrain, rider, truck, bus, train, bicycle} TP grow ≥30% | Investigate which (EQLv2 vs RFS) hurt — toggle one off, retry single-feature |
| B | PQ ≥ Run A AND person/rider/bicycle TP each grow ≥10% over Run A | Inspect copy-paste telemetry (paste fraction, depth fallback rate); if pool sampling degenerate, lower `MAX_NUM_PASTED_OBJECTS` |
| C | PQ ≥ Run B (or stuff PQ + 0.5 even if aggregate flat) | Lower `LDAM_MAX_MARGIN` from 0.5 → 0.3 and re-launch |

Final winner of A/B/C is the Stage-3 self-training input.

---

## Risks & mitigations (top 3)

| Risk | Mitigation |
|---|---|
| **EQLv2 destabilises early training** (cold buffers) | Buffers init to 0 → `neg_w ≈ 0` initially → all-positive weighting (safe). If divergence, warm-start 200 steps with vanilla CE. |
| **LDAM × inverse-freq class_weight stacking over-penalises common classes** | Use LDAM-DRW recipe: cap `LDAM_MAX_MARGIN=0.5`. If road/building IoU drops >2%, lower to 0.3. |
| **DDP + custom RFS sampler races** (shard mismatch) | Detectron2-style fractional rounding with per-rank seed shift; unit test in `test_repeat_factor_sampler.py::test_ddp_sharding`. |

---

## Total effort estimate

- Implementation + tests: **1.5 days**
- Pool build (one-time, M4 Pro): **15 min**
- Remote sync (T0 pseudo-labels first time only, 8925 files ~12 GB): **~30 min** over LAN
- Remote training: **3 × 40h = 120h serial** (~5 days wall-clock for the chain)
- Local eval after each run: **~30 min**
