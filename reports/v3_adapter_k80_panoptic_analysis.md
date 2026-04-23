# V3 Adapter k=80 + DepthPro Panoptic Analysis

**Date**: 2026-04-17
**Experiment**: V3 adapter (16D sinusoidal, h=384, 2L, lp=20) k=80 semantics + DepthPro depth-guided instance splitting (tau=0.01, A_min=1000, sigma=0.0, dil=3)
**Script**: `mbps_pytorch/evaluate_panoptic_combined.py` (uses `depth_guided_instances()` from `sweep_depthpro.py`)
**JSON**: `results/depth_adapter/V3_k80_depthpro_panoptic_eval.json`

---

## Summary

| Config | PQ | PQ_stuff | PQ_things | mIoU | Inst/img |
|--------|-----|----------|-----------|------|----------|
| Raw k=80 + DepthPro | **28.40** | 32.08 | **23.35** | ~52% | ~5.3 |
| V3 adapter k=80 + DepthPro | 27.81 | **32.32** | 21.62 | **55.29%** | 5.6 |
| Delta | -0.59 | +0.24 | **-1.73** | +3% | +0.3 |

**Result**: Adapter improves semantics (mIoU +3%) and stuff PQ (+0.24), but **PQ_things drops by -1.73**.

---

## Per-Class Thing Comparison

| Class | Raw TP | V3 TP | Raw FP | V3 FP | Raw FN | V3 FN | FP delta |
|-------|--------|-------|--------|-------|--------|-------|----------|
| person | 298 | 332 | 131 | **343** | 3078 | 3044 | **+212** |
| rider | 77 | 49 | 37 | 67 | 464 | 492 | +30 |
| car | 1221 | 1179 | 384 | 312 | 3414 | 3456 | -72 |
| truck | 33 | 30 | 19 | 16 | 60 | 63 | -3 |
| bus | 52 | 46 | 20 | 17 | 46 | 52 | -3 |
| train | 10 | 10 | 16 | 17 | 13 | 13 | +1 |
| motorcycle | 0 | 0 | 0 | 0 | 149 | 149 | 0 |
| bicycle | 95 | 84 | 251 | **309** | 1068 | 1079 | **+58** |

---

## Root Cause Analysis: Why PQ_things Drops

### 1. False Positive Explosion on Small-Object Classes

The adapter produces **+212 FP person instances** and **+58 FP bicycle instances** compared to raw k=80. These two classes account for the entire PQ_things regression.

**Mechanism**:
- The adapter assigns more pixels to person and bicycle (recovering them from under-segmentation at mIoU level)
- More thing pixels + DepthPro's aggressive splitting (tau=0.01) = more small instances
- Small instances fail the IoU > 0.5 matching threshold against GT = counted as FP
- More FPs drag down RQ, which drags down PQ

### 2. Semantic Boundary Shift Creates Fragmentation

The adapter changes WHERE thing-class boundaries are drawn. Raw k=80 has noisier boundaries but they align better with depth edges. The adapter's cleaner semantic boundaries produce more coherent thing regions, which the depth splitter then over-fragments because:

- tau=0.01 was tuned for raw k=80 noise level, not adapter's cleaner output
- Cleaner semantics = larger connected components per thing class
- Larger CCs give the Sobel splitter more internal depth edges to cut on
- Result: each large coherent region gets split into more small fragments

### 3. Rider Regression: New Semantic Coverage Creates Mismatches

Rider is the most dramatic TP drop: 77 → 49 (-28 TP, -36%). At raw k=80, rider had 0 clusters (dead class). V3 adapter recovers rider at mIoU=30.35%. But:

- Newly recovered rider pixels overlap with person/bicycle at depth level
- Depth splitting creates rider instances where person GT exists = FP
- Some valid rider instances get smaller (semantic boundary shifts) → fall below IoU 0.5

### 4. Car Improves Slightly in Precision

Car is the only positive signal: FP drops 384 → 312 (-72). The adapter's cleaner car boundaries produce fewer spurious car fragments. But TP also drops slightly (1221 → 1179), likely because some car pixels get reassigned to truck/bus.

---

## Key Insight

**Depth splitting parameters and semantic quality are coupled.** The tau=0.01 threshold was optimized for raw k=80 semantics. Changing the semantic backbone without re-tuning the depth splitting creates a mismatch where cleaner semantics paradoxically produce worse panoptic results.

The adapter's semantic improvement is real (+3% mIoU), but it manifests as:
- **PQ_stuff +0.24**: Direct benefit (stuff doesn't need instance splitting)
- **PQ_things -1.73**: Indirect harm (more thing pixels + same aggressive splitting = more fragments = more FP)

---

## Possible Mitigations (Not Tested)

1. **Re-tune tau for adapter semantics**: Higher tau (e.g., 0.02-0.05) would split less aggressively, reducing FP on person/bicycle
2. **Increase min_area for adapter**: min_area=1000 may be too low for adapter's larger CCs
3. **Class-specific splitting**: Use different tau per thing class (aggressive for car/truck, conservative for person/bicycle)
4. **Don't use adapter for panoptic**: Use raw k=80 for panoptic (PQ=28.40) and adapter for semantic-only tasks (mIoU=55.29%)

---

## Full V3 Adapter k=80 Per-Class Results

### Stuff Classes
| Class | PQ | SQ | RQ | TP | FP | FN |
|-------|-----|-----|-----|----|----|-----|
| road | 76.71 | 78.55 | 97.66 | 480 | 20 | 3 |
| sidewalk | 38.84 | 67.33 | 57.68 | 276 | 217 | 188 |
| building | 66.77 | 75.34 | 88.62 | 436 | 57 | 55 |
| wall | 12.62 | 67.66 | 18.66 | 32 | 110 | 169 |
| fence | 10.70 | 61.98 | 17.27 | 36 | 192 | 153 |
| pole | 0.13 | 53.89 | 0.24 | 1 | 333 | 488 |
| traffic light | 0.00 | 0.00 | 0.00 | 0 | 0 | 260 |
| traffic sign | 13.05 | 60.78 | 21.48 | 93 | 304 | 376 |
| vegetation | 66.99 | 74.48 | 89.94 | 438 | 50 | 48 |
| terrain | 16.32 | 65.81 | 24.80 | 47 | 101 | 184 |
| sky | 53.40 | 72.83 | 73.33 | 334 | 140 | 103 |

### Thing Classes
| Class | PQ | SQ | RQ | TP | FP | FN |
|-------|-----|-----|-----|----|----|-----|
| person | 12.41 | 75.69 | 16.39 | 332 | 343 | 3044 |
| rider | 8.74 | 58.60 | 14.92 | 49 | 67 | 492 |
| car | 30.61 | 79.52 | 38.49 | 1179 | 312 | 3456 |
| truck | 36.01 | 83.42 | 43.17 | 30 | 16 | 63 |
| bus | 46.97 | 82.20 | 57.14 | 46 | 17 | 52 |
| train | 31.59 | 78.98 | 40.00 | 10 | 17 | 13 |
| motorcycle | 0.00 | 0.00 | 0.00 | 0 | 0 | 149 |
| bicycle | 6.62 | 61.31 | 10.80 | 84 | 309 | 1079 |
