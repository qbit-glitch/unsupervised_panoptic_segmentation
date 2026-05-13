# Dead-Class Recovery Plan: SAM3 Pseudo-Label Injection + Seesaw Loss

## Context
Stage-3 CUPS (DINOv3 ViT-B/16, PQ=35.83%) has 6 permanently dead classes: **motorcycle, guard_rail, caravan, trailer, tunnel, polegroup** (PQ=0%). A prior Stage-4 attempt using `thing_focal_only` SAM supervision hurt bicycle (-7.9%) and rider (-6.9%) because forcing the semantic head toward channel-0 doesn't give the ROI classifier any instance-level signal for motorcycle — there are no motorcycle pseudo-label instances to produce positive RoIs.

**Root cause**: pseudo-labels have 0 motorcycle thing instances → DropLoss IoU filter silences all weak predictions → `cum_samples[motorcycle]=0` → Seesaw loss has nothing to amplify. Fix: inject real instances first, then amplify with Seesaw + relaxed DropLoss.

**Why prior Stage-4 failed**: `thing_focal_only` pushes semantic channel-0 high but ROI head still sees 0 motorcycle proposals. No instance PNGs = no positive RoIs = no classification gradient for motorcycle. The existing channel-0 pressure just adds noise to bicycle/rider's gradients.

---

## Three-Track Architecture

```
Track A (offline, Day 1 morning)         Track B (training, Day 1 evening)
──────────────────────────────           ──────────────────────────────────
A0: Extract CUPS pseudo-class mapping    B: Stage-3 fine-tune with:
    (Hungarian: which cluster=motorcycle)    • enriched pseudo-labels
          │                                  • Seesaw P=0.5, Q=2.0
          ▼                                  • DropLoss 0.4 → 0.25
A1: inject_sam3_dead_classes.py              • CLASS_THRESHOLD_ALPHA=0.5
    motorcycle (SAM3 class 2) →          FINE_OBJECT: DISABLED for things
    semantic PNG + instance PNG          
    + .pt distributions update          Track C (optional, Day 2 only if guard_rail still dead)
          │                             ────────────────────────────────────────────────────
          ▼                             • Add `stuff_only_entropy` mode to fine_object.py
A2: Verify: instanceness > 0.05?        • FINE_OBJECT.ENABLED=True, MODE=stuff_only_entropy
    visual audit of 5 images            • STUFF_CHANNEL_MAP from cups_class_mapping.json
```

---

## Track A — Offline Pseudo-Label Enrichment

### A0. Extract pseudo-class mapping (30 min, no new code)

Run existing script on Stage-3 checkpoint:
```bash
python scripts/extract_cups_class_mapping.py \
    --checkpoint /path/to/best_pq_step=003000.ckpt \
    --pseudo_root /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/ \
    --output cups_class_mapping.json
```

**Needed output** (from the JSON):
- `sam3_thing_class_map[2]` → CUPS pseudo-class ID (0–79) for motorcycle
- `sam3_thing_class_map[10]` → caravan pseudo-class ID
- `sam3_thing_class_map[11]` → trailer pseudo-class ID
- `sam3_stuff_channel_map[9]` → semantic head channel for guard_rail (for Track C)

If Hungarian gives no motorcycle assignment (model has zero pixels for that cluster), fall back to max-IoU cluster on a sample of GT motorcycle pixels from `gtFine/val/`.

### A1. SAM3 injection script — `scripts/inject_sam3_dead_classes.py` (NEW, ~250 lines)

**SAM3 data format** (verified from actual files):
- NPZ keys: `masks` (bool, N×H×W), `iou_scores` (float32, N), `areas` (int32, N), `class_labels` (int32, N)
- SAM3 class 2 = motorcycle (~34 masks / 2975 train images), class 9 = guard_rail (~1 mask total)
- IoU range: 0.35–0.98, typical Q1–Q3: 0.44–0.79

**CUPS pseudo-label format** (verified from `pseudo_label_dataset.py` lines 135–156, 373–375):
- Semantic PNG: uint8, (1024, 2048), values 0–79 = pseudo-class index
- Instance PNG: uint16, (1024, 2048), 0=background, 1–N=thing instance IDs
- .PT file: `{'distribution all pixels': Tensor[80], 'distribution inside object proposals': Tensor[80]}`
- Thing/stuff split: `instanceness = inst[c] / (all[c] + 1e-6) > 0.05` → must flip this for motorcycle

**CLI:**
```bash
python scripts/inject_sam3_dead_classes.py \
    --sam3_dir /Users/qbit-glitch/Desktop/datasets/cityscapes/sam_fine_masks_sam3/train/ \
    --pseudo_in  /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/ \
    --pseudo_out /Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_sam3_enriched/ \
    --class_mapping cups_class_mapping.json \
    --target_classes 2,10,11 \
    --min_iou 0.40 \
    --min_area 300 \
    --max_per_image 6
```

**Core algorithm per image:**
```python
# Load SAM3 masks, filter quality
keep = isin(labels, target_classes) & (ious >= 0.40) & (areas >= 300) & (areas < 0.30*H*W)
order = argsort(-ious[keep])[:max_per_image]  # highest IoU first

next_id = ins.max() + 1
claimed = zeros_like(ins, dtype=bool)

for mask, sam_class in zip(final_masks, final_labels):
    write = mask & (~claimed)                   # no double-writing
    if write.sum() < min_area: continue
    
    pseudo_cls = class_mapping[sam_class]       # from A0
    sem[write] = pseudo_cls                     # label semantic PNG
    ins[write] = next_id                        # new instance ID
    claimed |= write
    
    # Update .pt distributions (critical: must flip instanceness > 0.05)
    n = write.sum()
    dist['distribution all pixels'][pseudo_cls] += n
    dist['distribution inside object proposals'][pseudo_cls] += n
    next_id += 1

# Write enriched files to pseudo_out
```

**Edge cases:**
- Image with no SAM3 motorcycle masks → identity copy (still write to pseudo_out, dataloader expects every image)
- Two SAM3 masks overlap → higher IoU claimed first, lower-IoU shrinks; skip if remaining area < min_area
- SAM3 mask covers existing non-dead pseudo-class pixels → SAM3 overrides (SAM3 quality > k=80 clusters for fine-grained objects)

**Expected injection budget:**
| Class | SAM3 idx | Masks/2975 imgs (est.) | New instances added |
|-------|----------|----------------------|---------------------|
| motorcycle | 2 | ~25–35 (after IoU≥0.40) | ~25–35 total |
| caravan | 10 | ~5–10 | ~5–10 total |
| trailer | 11 | ~5–10 | ~5–10 total |

25–35 total motorcycle instances across all images is enough to break `cum_samples=0` and let Seesaw amplify.

### A2. Verification (--verify mode in same script, 30 min)

**Programmatic checks (fail loud):**
1. File-count parity: `len(pseudo_out) == len(pseudo_in)`
2. `instanceness[motorcycle_pseudo_idx] > 0.05` across all `.pt` files → if false, injection failed; drop `--min_iou` to 0.35 and re-run
3. Pixel count delta matches running total computed during injection
4. Save 5 side-by-side visualization PNGs to `reports/sam3_enrichment_audit/` for manual inspection

**If instanceness never exceeds 0.05** (too few SAM3 masks for motorcycle): the motorcycle pseudo-class will still be classified as stuff → no instance proposals → injection is wasted. In that case, lower threshold to 0.35, re-run, re-verify.

---

## Track B — Training Configuration

### Config: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_motorcycle_recovery.yaml`

Base: inherit from Stage-3 DCFA+SIMCF-ABC config (ROUNDS=3, ROUND_STEPS=4000 in original).

```yaml
DATA:
  ROOT_PSEUDO: "/path/to/cups_pseudo_labels_sam3_enriched/"  # enriched labels

MODEL:
  CHECKPOINT: "/path/to/stage3_best_pq_step=003000.ckpt"     # resume Stage-3 best
  ROI_BOX_HEAD:
    USE_SEESAW_LOSS: True
    SEESAW_P: 0.5     # more aggressive than default 0.8 — amplifies sparse motorcycle signal
    SEESAW_Q: 2.0     # unchanged

TRAINING:
  ROUNDS: 1           # fine-tune, not retrain (was 3)
  ROUND_STEPS: 2500   # fine-tune (was 4000)
  ADAMW:
    LEARNING_RATE: 0.000025   # half Stage-3 LR (0.00005 / 2)
  DROP_LOSS: True
  DROP_LOSS_IOU_THRESHOLD: 0.25   # was 0.4 — allows weaker motorcycle preds to contribute

SELF_TRAINING:
  CLASS_THRESHOLD_ALPHA: 0.5   # was 0.0 — lowers threshold for rare classes in self-train rounds
  FINE_OBJECT:
    ENABLED: False   # CRITICAL: do NOT use thing_focal_only — that's what broke Stage-4
```

**Why these values:**
- `SEESAW_P=0.5`: Paper ablations show P∈[0.5, 0.8] is sweet spot. P=0.5 is more aggressive — right for borderline-dead classes. Going lower (P=0.3) over-amplifies noisy motorcycle labels.
- `DROP_LOSS_IOU_THRESHOLD=0.25`: Stage-3 default 0.4 silences any motorcycle prediction with IoU < 0.4 (likely in early steps). 0.25 allows weak-but-real predictions to contribute gradients. Going to 0.0 introduces too much noise from completely wrong predictions.
- `ROUNDS=1, ROUND_STEPS=2500`: fine-tuning, not retraining. Abort if val PQ drops below 35.0 at any 500-step checkpoint.
- `FINE_OBJECT.ENABLED=False`: the prior Stage-4 failure proved thing_focal_only is counterproductive — it adds gradients to the semantic channel-0 but doesn't produce ROI proposals. Off entirely for things.

### Monitor training
Eval every 500 steps. If any of bicycle/rider/person regress by >1.0 PQ from Stage-3 baseline → abort and raise `SEESAW_P` back to 0.7.

---

## Track C — Stuff Dead-Class Recovery (optional, only if guard_rail still PQ=0 after Track B)

### C1. Add `stuff_only_entropy` mode — `refs/cups/cups/losses/fine_object.py`

Surgical edit (~20 lines):
1. Add `"stuff_only_entropy"` to `_VALID_MODES` list
2. In `_split_thing_stuff_loss()` (line 296): add branch for new mode that **skips thing branch entirely** (symmetric to `thing_focal_only`)
3. This prevents thing classes from being touched — unlike the original Stage-4 approach

### C2. Config: `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_stuff_recovery.yaml`

```yaml
TRAINING:
  CHECKPOINT: "<Track-B best checkpoint>"
  ROUNDS: 1
  ROUND_STEPS: 1500
  ADAMW:
    LEARNING_RATE: 0.000008  # very conservative

SELF_TRAINING:
  FINE_OBJECT:
    ENABLED: True
    MODE: "stuff_only_entropy"   # new mode from C1
    WEIGHT: 0.3                  # conservative; halve if instability
    SAM_MASKS_DIR: "/path/to/sam_fine_masks_sam3/train/"
    MIN_IOU_SCORE: 0.40
    STUFF_CHANNEL_MAP:
      - [9, <guard_rail_channel_from_A0>]  # guard_rail → stuff head channel
```

**Guard_rail SAM3 coverage**: only ~1 mask / 2975 train images — the entropy loss will be extremely sparse. May not help. Run C only if time budget allows.

**Tunnel and polegroup**: SAM3 has no tunnel class; polegroup needs pole→polegroup split. Both are out of scope for this plan — accept as permanently dead for now.

---

## Files to Create / Modify

| Action | File | Track |
|--------|------|-------|
| **CREATE** | `scripts/inject_sam3_dead_classes.py` (~250 lines) | A1 + A2 |
| **REUSE (no edit)** | `scripts/extract_cups_class_mapping.py` | A0 |
| **CREATE** | `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_motorcycle_recovery.yaml` | B |
| **EDIT (~20 lines)** | `refs/cups/cups/losses/fine_object.py` — add `stuff_only_entropy` mode | C1 |
| **CREATE** | `refs/cups/configs/train_self_cityscapes_dinov3_vitb_stage4_stuff_recovery.yaml` | C2 |
| **NO CHANGE** | `refs/cups/cups/data/pseudo_label_dataset.py` | — (handles enriched labels transparently) |
| **NO CHANGE** | `refs/cups/cups/losses/long_tail.py` | — (Seesaw already implemented, just enable via config) |

---

## Key File References

| File | Lines | Why |
|------|-------|-----|
| `refs/cups/cups/data/pseudo_label_dataset.py` | 135–149 | instanceness formula → must flip > 0.05 for motorcycle |
| `refs/cups/cups/data/pseudo_label_dataset.py` | 432–448 | per-instance semantic class read from PNG |
| `refs/cups/cups/losses/long_tail.py` | 97–158 | `SeesawSoftmaxLoss` — `cum_samples` buffer, P/Q formula |
| `refs/cups/cups/losses/fine_object.py` | 122, 296 | `_VALID_MODES`, `_split_thing_stuff_loss` — edit for Track C |
| `refs/cups/cups/model/modeling/roi_heads/roi_heads.py` | 380–430 | DropLoss weight computation; kwarg `droploss_iou_thresh` |
| `refs/cups/cups/model/modeling/roi_heads/fast_rcnn.py` | 305–308, 340–342 | Seesaw enabled via `USE_SEESAW_LOSS`, `num_classes` set here |
| `scripts/extract_cups_class_mapping.py` | — | Hungarian mapping; reuse unchanged |

---

## Sequencing

```
Day 1 morning (4–5h engineering)
  [A0] Run extract_cups_class_mapping.py → cups_class_mapping.json   (30 min)
  [A1] Write inject_sam3_dead_classes.py                             (3–4h)
       └── test on 10 images first, then full 2975-image run (~30min)
  [A2] Verify: instanceness check + visual audit                     (30 min)
       └── If instanceness < 0.05: lower --min_iou to 0.35, re-run

Day 1 evening → Day 2 morning (training)
  [B]  Launch motorcycle_recovery.yaml on remote GPU                 (4–6h)
       └── Eval every 500 steps; abort if PQ < 35.0

Day 2 morning (eval + optional Track C)
  [Eval] Check per-class PQ: motorcycle/caravan/trailer alive?
  [C]   If guard_rail still dead: implement stuff_only_entropy + run (2–3h)

Day 2 afternoon
  [Report] reports/dead_class_recovery_results.md
```

---

## Acceptance Criteria

**Track B success:**
- motorcycle PQ > 0 (even PQ=2.0 is "alive")
- No alive class regresses by > 1.0 PQ (bicycle/rider/person are the watch classes)
- Overall PQ ≥ 35.33% (Stage-3 − 0.5 PQ acceptable for 3 new alive classes)

**Stretch goal:** Overall PQ > 35.83% (Track A's instance injection helps adjacent thing classes too)

**Track C success:** guard_rail PQ > 0, no stuff regression > 0.5 PQ

---

## Loss Function Summary

| Dead Class | Primary Fix | Loss | NOT Used |
|------------|------------|------|----------|
| motorcycle (thing) | Pseudo-label injection (A1) | Seesaw P=0.5 on ROI head | thing_focal_only |
| caravan, trailer (things) | Pseudo-label injection (A1) | Seesaw P=0.5 | thing_focal_only |
| guard_rail (stuff) | stuff_only_entropy (C) | Entropy min on SAM3 masks | any thing loss |
| tunnel (stuff) | No SAM3 coverage | — | — (accept dead) |
| polegroup (stuff) | No SAM3 class | — | — (accept dead) |
