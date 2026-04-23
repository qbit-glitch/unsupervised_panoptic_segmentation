# E1 Control Experiment: Why CUPS Instance Proposals Fail on Consumer GPUs

**Date:** 2026-04-15
**Machine:** santosh@172.17.254.146, 2x GTX 1080 Ti (11GB each)
**Experiment:** E1 Stage-2 — CUPS official pseudo-labels + DINOv3 ViT-B/16

---

## 1. The Observation

E1 Stage-2 training converged to **PQ = 8.2%, PQ_things = 0.0%, PQ_stuff = 11.3%** after 8000 steps. The semantic head learned (mIoU = 21%, Acc = 55%), but the instance head produced zero thing detections across all validation checkpoints. This is not a training failure — the model never received meaningful instance supervision.

## 2. Root Cause: Silent SF2SE3 OOM

The CUPS pseudo-label pipeline (`gen_pseudo_labels.py`) generates instance proposals via a three-stage cascade:

```
Stereo video pairs → RAFT-SMURF optical flow → Stereo disparity → SF2SE3 scene flow decomposition → Instance proposals
```

SF2SE3 requires computing dense scene flow from stereo video, which demands ~13GB peak VRAM. On the GTX 1080 Ti (11GB), the DepthG semantic model alone occupies ~7GB, leaving only ~4GB free. Every SF2SE3 call fails with:

```
CUDA out of memory. Tried to allocate 2.41 GiB.
GPU 0 has a total capacity of 10.91 GiB of which 656 MiB is free.
```

Critically, `gen_pseudo_labels.py` **catches this exception silently** — it writes an all-zeros instance map and continues to the next image. The pipeline completes without error, producing 2975 semantic labels (correct) and 2975 instance labels (84.5% all-zeros).

## 3. Evidence

### 3.1 Instance Map Statistics

| Metric | Value |
|--------|-------|
| Total instance pseudo-labels | 2975 |
| Non-empty (has >= 1 instance) | 461 (15.5%) |
| All-zeros (no instances) | 2514 (84.5%) |
| Non-empty samples after filtering | 461 / 2975 |

The 461 non-empty samples likely contain noise or partial proposals from the rare cases where SF2SE3 partially succeeded before OOM. The training log confirms: `"461 training samples and 500 validation samples detected"` — 84% of the training set was silently discarded.

### 3.2 Pipeline Log Confirmation

From `cups_pipeline_v3.log`, every image fails identically:

```
SF2SE3 failed for aachen_000000_000019
Failed for aachen_000000_000019: CUDA out of memory. Tried to allocate 2.41 GiB.
SF2SE3 failed for aachen_000001_000019
Failed for aachen_000001_000019: CUDA out of memory. Tried to allocate 2.41 GiB.
SF2SE3 failed for aachen_000002_000019
Failed for aachen_000002_000019: CUDA out of memory. Tried to allocate 2.41 GiB.
...
```

### 3.3 Training Metrics Confirm Dead Instance Head

| Step | PQ | PQ_things | PQ_stuff | mIoU |
|------|-----|-----------|----------|------|
| 500 | 9.0% | 0.06% | 14.2% | 17.1% |
| 1000 | 9.6% | 0.01% | 15.2% | 18.3% |
| 1500 | 10.0% | 0.00% | 15.9% | 19.4% |
| 3000 | 8.4% | 0.88% | 12.7% | 17.6% |
| 5000 | 7.9% | 2.63% | 10.9% | 19.3% |
| 7000 | 7.9% | 3.17% | 10.7% | 19.7% |

PQ_things remains near zero throughout training. The marginal increase from 0.0% to 3.2% late in training comes from the 461 surviving samples — too few for the instance head to learn meaningful object detection. Meanwhile, PQ_stuff peaks early at ~15% then degrades, suggesting the model overfits to the small effective training set.

**Note:** These metrics are from DDP evaluation (2 GPUs, no sync_dist), so each GPU evaluates ~250 images. The absolute numbers may be slightly unreliable, but the PQ_things = 0.0% pattern is unambiguous — the instance head is dead.

## 4. Why the Failure Was Silent

The CUPS pipeline's error handling follows this pattern (from `gen_pseudo_labels.py`, lines 220-254):

```python
try:
    object_proposals = get_object_proposals(
        image_1_l=image_1_l,
        optical_flow_l_forward=...,
        disparity_1_forward=...,
        ...
    )
except Exception:
    print(Exception)
    failed_images.append(img_name)
    tqdm.write("Failed for: " + str(img_name))
    object_proposals = torch.zeros(image_0_l.shape[-2], image_0_l.shape[-1]).long()
```

The bare `except Exception` catches the CUDA OOM, prints a warning, writes zeros, and continues. The semantic prediction (DepthG + CRF) runs before SF2SE3 and succeeds because it fits in 7GB. The instance map is then saved as all-zeros — a valid PNG file indistinguishable from "no instances detected" unless you check the content.

## 5. Additional Issues Found

### 5.1 Config Mismatches vs CUPS Original

Three settings diverged from the CUPS paper defaults:

| Setting | CUPS default | Our E1 config | Impact |
|---------|-------------|---------------|--------|
| MAX_NUM_PASTED_OBJECTS | 8 | 3 | 62% fewer augmented instances |
| NUM_STEPS_STARTUP | 1000 | 500 | Augmentations enabled too early |
| IGNORE_UNKNOWN_THING_REGIONS | False | True | Changes thing/stuff overlap handling |

### 5.2 Pseudo-Label Resolution Mismatch

CUPS `gen_pseudo_labels.py` generates labels at 640x1280 (inference resolution). The training code (`PseudoLabelDataset`) applies `scale_factor=0.625` assuming full 1024x2048 input, double-scaling labels to 400x800. This caused a `CenterCrop(640, 1280)` assertion failure on the 400x800 labels.

**Fix applied:** Changed `F.interpolate(pseudo_label, scale_factor=0.625)` to `F.interpolate(pseudo_label, size=(target_h, target_w))`, which resizes labels to match the scaled image dimensions regardless of their original resolution.

## 6. VRAM Budget Analysis

| Component | VRAM (approx) |
|-----------|---------------|
| DepthG (DINO ViT-B/8 + cluster probe) | ~6.5 GB |
| CRF (batched, CPU-offloaded) | ~0.5 GB |
| RAFT-SMURF optical flow (2 directions) | ~2.0 GB |
| Stereo disparity (2 directions) | ~1.5 GB |
| SF2SE3 scene flow decomposition | ~2.5 GB |
| **Total** | **~13.0 GB** |
| **GTX 1080 Ti capacity** | **11.0 GB** |
| **Deficit** | **~2.0 GB** |

The CUPS pipeline was designed for GPUs with >=16GB VRAM (likely V100 32GB or A100 40GB based on the CUPS authors' infrastructure at ETH Zurich).

## 7. Recommendations

### Option A: Regenerate on A6000 (Preferred)

Run the full CUPS pipeline on the Anydesk A6000 (48GB VRAM). This guarantees SF2SE3 succeeds for all 2975 images, producing the exact pseudo-labels the CUPS paper uses.

**Pros:** True apples-to-apples comparison with CUPS paper.
**Cons:** Requires ~8-12 hours pipeline time + data transfer.

### Option B: Use k=80 Depth-Guided Instance Labels

Use our existing depth-guided instance labels (PQ_things = 19.41%) which are already on the santosh machine. These use Sobel edge detection on depth maps instead of SF2SE3 scene flow.

**Pros:** Immediately available, known quality.
**Cons:** Not CUPS-official pseudo-labels — changes the E1 control variable. But E1's purpose is to isolate the *backbone* contribution, not the instance pipeline. Using the same instance labels across experiments still achieves this.

### Option C: Two-Pass Generation on 1080 Ti

Run DepthG semantics in pass 1, free GPU, then run SF2SE3 in pass 2. This avoids the co-resident memory pressure.

**Pros:** Uses existing hardware.
**Cons:** Requires rewriting the pipeline to save intermediate optical flow / disparity. SF2SE3 alone may still need >11GB.

## 8. Lessons Learned

1. **Always verify pseudo-label quality before training.** Check non-empty instance count, visualize a few samples, and read the pipeline logs.

2. **Silent failures are the most dangerous.** The CUPS pipeline's `except Exception: write_zeros()` pattern means OOM produces valid-looking output. Future pipeline runs should assert `instance_map.max() > 0` for at least 80% of images.

3. **GPU memory requirements must be documented explicitly.** The CUPS paper and codebase do not mention the ~13GB VRAM requirement for pseudo-label generation. This is a significant reproducibility barrier for researchers without V100/A100 access.

4. **Config defaults matter.** Three seemingly minor config changes (copy-paste objects, startup delay, thing region handling) compound to significantly degrade training quality.
