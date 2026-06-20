# Mobile-EoMT Panoptic — Phase 1 & 2 GPU Handoff

Phase 0 (vertical slice) is complete + verified on Mac. Phases 1–2 are GPU-bound
(A6000 / fics-lab). All scripts below already exist and are CPU-verified at small
scale; here are the exact full-scale launch commands + the few glue steps.

**Honest target:** PQ band **30–42** (vs MaskConver's ~30 on-phone). In-session
overfit-8 losses are NOT accuracy. Real PQ comes from these runs.

---

## Prerequisites on the GPU box

1. COCO `train2017/` + `val2017/` images and `annotations/panoptic_val2017.{json,/}` present.
2. Cached weights: `facebook/sam3` (3.2 GB), `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov2-small`. (`HF_HUB_OFFLINE=1 SAM3_OFFLINE=1` after first fetch.)
3. Two envs as on Mac: one with `sam3`+`INSID3`+transformers (auto-label), one with `lightning`+`timm`+`onnxruntime` (train/export). On a CUDA box a single env usually serves both.
4. Stuff exemplars: `python auto_annotation/scripts/build_coco_stuff_exemplars.py` (regenerates the 53-concept bank from val GT; ~30 s).

---

## Phase 1 — full label-free auto-label of COCO train2017 (GPU-days)

```bash
# Shard across N GPU workers (resumable; skips existing *_panoptic.npy).
# Example: 8 workers, worker k:
HF_HUB_OFFLINE=1 SAM3_OFFLINE=1 PYTHONPATH=. python \
  auto_annotation/scripts/autolabel_coco_full.py \
  --img_dir /path/to/coco/train2017 \
  --device cuda --n_things 80 --n_stuff 53 --limit 0 \
  --shard k/8 --out /path/to/coco/autolabels_train
```
Output per image: `<stem>_panoptic.npy` + `<stem>.png` (rgb2id) + `<stem>.json` (segments_info).

**Disclosed upper-bound arm (optional, NOT label-free):** add a Mask2Former-COCO
semantic pass for stuff and report it separately — it is distillation from a
COCO-supervised teacher, not label-free.

---

## Phase 1.5 — aggregate per-image jsons into one COCO panoptic json

EoMT's `coco_panoptic` loader wants a single `panoptic_train.json` + a png dir.

```python
import json, glob, shutil
from pathlib import Path
from auto_annotation import taxonomy_coco as T

src = Path("/path/to/coco/autolabels_train"); dst = src / "panoptic"; dst.mkdir(exist_ok=True)
images, annotations = [], []
for jf in sorted(src.glob("*.json")):
    a = json.loads(jf.read_text())
    images.append({"id": a["image_id"], "file_name": f"{a['image_id']}.jpg",
                   "height": 0, "width": 0})          # h/w optional for training
    a["image_id"] = a["image_id"]
    annotations.append(a)
    shutil.copy(src / f"{jf.stem}.png", dst / f"{jf.stem}.png")
cats = [{"id": i, "name": c.name, "isthing": int(c.is_thing)} for i, c in T.COCO_CLASSES.items()]
(src / "panoptic_train.json").write_text(json.dumps(
    {"images": images, "annotations": annotations, "categories": cats}))
```

---

## Phase 2 — train + self-train + eval + export, per student/encoder (GPU-days each)

### EoMT student — encoder ablation
Reuse `refs/eomt` Lightning training; one config per encoder (set `data` to the
auto-labels, `network.encoder.backbone_name` to the variant):

| variant | backbone_name |
|---|---|
| DINOv2-S | `facebook/dinov2-small` |
| DINOv3-S | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| DeiT-Ti  | `deit_tiny_patch16_224` (timm) |
| DeiT-S   | `deit_small_patch16_224` (timm) |
| distilled-tiny | (Phase-2b: distill DINOv3 → tiny plain ViT, then point here) |

```bash
cd refs/eomt
python3 main.py fit -c configs/dinov2/coco/panoptic/eomt_small_640.yaml \
  --model.network.encoder.backbone_name facebook/dinov2-small \
  --trainer.precision 32-true --compile_disabled \
  --data.<path args> /path/to/coco/autolabels_train/panoptic_train.json
# Self-training: swap the head to training/mask_classification_panoptic_self_train.py
# (EMA teacher + DropLoss already wired) and resume from the warm-up ckpt.
```

### Conv student
Extend `mbps_pytorch/mobile_panoptic_sup/smoke_conv.py` `overfit()` into a full
loop over the aggregated auto-labels (semantic CE at 1/4 res, `pretrained=True`),
then the same EMA self-training recipe.

### Eval (both, every variant) — on COCO val GT
```python
from mbps_pytorch.mobile_panoptic_sup.coco_eval import eval_pq, load_coco_gt
# run model on each val image -> (pred_pan, pred_seg2cat) via the student's
# postprocess; gts via load_coco_gt; then eval_pq(preds, gts) -> {"pq","pq_things","pq_stuff"}
```

### Export + on-device latency
```bash
python -m mbps_pytorch.mobile_panoptic_sup.export_onnx   # ONNX + CPU latency, both students
# then: ONNX -> Core ML (coremltools) / TFLite (ai-edge-torch) -> measure on NPU.
```

---

## Results table to fill (paper-facing)

| student | encoder | label source | COCO val PQ | PQ_th | PQ_st | params | ONNX latency (NPU) |
|---|---|---|---|---|---|---|---|
| EoMT | DINOv2-S | INSID3+SAM3 (label-free) | — | — | — | — | — |
| EoMT | DeiT-Ti | label-free | — | — | — | — | — |
| conv | RepViT-M1.5 | label-free | — | — | — | — | — |
| EoMT | DINOv2-S | +Mask2Former (disclosed UB) | — | — | — | — | — |

Position against MaskConver (~30 PQ @ 33 FPS Pixel 6) — the on-phone bar.
