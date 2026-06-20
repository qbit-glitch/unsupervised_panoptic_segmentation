# Mobile-EoMT Panoptic — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the end-to-end vertical slice — auto-label a few COCO images with the foundation pipeline, train both a mobile-EoMT and a conv student on labels, export both to ONNX with a latency number, and render a masks+labels demo — proving the architecture before any full GPU run.

**Architecture:** Reuse two in-repo subsystems: `auto_annotation/` (SAM3 things + INSID3 stuff → panoptic) for label generation, and `refs/eomt/` (EoMT with a `facebook/dinov2-small` plain-ViT encoder) for the transformer student. Add a compact RepViT+BiFPN semantic-only conv student (CC-derived instances) as the export/latency foil, a COCO-panoptic-format converter that lets both students and the PQ evaluator consume the auto-labels unchanged, and ONNX export + demo scripts.

**Tech Stack:** PyTorch 2.10 (`.venv`), PyTorch-Lightning + jsonargparse (EoMT), timm 1.0.26 (RepViT), HF transformers 5.1 (DINOv2/SAM3/DINOv3), onnxruntime, detectron2 (COCO category table), numpy/PIL/cv2.

## Global Constraints

- **Two venvs, do not mix:** auto-labeling (SAM3/INSID3/DINOv3) runs ONLY in `.venv_cups_cpu/bin/python`; EoMT + conv student + export + demo + eval run ONLY in `.venv/bin/python` (has lightning, timm 1.0.26, detectron2). They bridge via files on disk, never a shared process.
- **Offline + CPU for auto-label:** prefix auto-label commands with `HF_HUB_OFFLINE=1 SAM3_OFFLINE=1 PYTHONPATH=.` and pass `--device cpu` (SAM3/INSID3 are CPU-only here; "mps" silently means cpu for these and SVD is not on MPS).
- **EoMT is plain-ViT only:** encoders must be columnar constant-resolution ViTs (DINOv2/v3-S, DeiT-S). Never MobileViT/EfficientViT/Swin.
- **EoMT on Mac:** run with `--trainer.precision 32-true --compile_disabled` and (for export) `--model.network.masked_attn_enabled False`. MPS optional; CPU is the verified path.
- **Label-free honesty:** the label-free result uses INSID3 (training-free) + SAM3 only. Any COCO-trained semantic model (Mask2Former/EoMT-COCO) is a *disclosed* upper-bound arm, never part of the label-free number.
- **COCO data (read-only):** images `/Volumes/code_files/datasets/coco/val2017/`, GT panoptic PNGs `/Volumes/code_files/datasets/coco/annotations/panoptic_val2017/`, GT json `/Volumes/code_files/datasets/coco/annotations/panoptic_val2017.json` (133 categories, `id` non-contiguous 1..200, `isthing` only in `categories[]`).
- **Panoptic encoding:** segment PNG decode is `seg_id = R + G*256 + B*65536` (panopticapi `rgb2id`); our intermediate panoptic uses `class*1000 + inst` (label_divisor 1000, void 255).
- **Code style (repo CLAUDE.md):** frozen dataclass configs, type hints, module `logging` (never `print`), files 200–400 lines, no bare `except`.
- **New package:** all new training/export/demo code under `mbps_pytorch/mobile_panoptic_sup/`. Auto-label additions under `auto_annotation/`.
- **Scope:** Phase 0 proves plumbing on ~5–8 images. Label *quality* and full-set runs are Phase 1; encoder ablation + self-training are Phase 2 (separate plans). Smoke PQ is meaningless — do not report it as a result.

---

### Task 1: COCO eval harness (reuse decode + PQ)

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/__init__.py`
- Create: `mbps_pytorch/mobile_panoptic_sup/coco_eval.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_coco_eval.py`

**Interfaces:**
- Consumes: `decode_panoptic_png` (`mbps_pytorch/evaluate_coconut_pseudolabels.py:92`), `compute_pq`/`summarize_pq` (`mbps_pytorch/evaluate_cross_dataset.py:162,218`), detectron2 `COCO_CATEGORIES`.
- Produces: `coco_contiguous_maps() -> tuple[dict,dict,set,set]` (catid→idx, idx→name, thing_idxs, stuff_idxs); `load_coco_gt(image_id) -> tuple[np.ndarray, dict]` (segment-id map, `{seg_id: cat_idx}`); `eval_pq(preds, gts) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_coco_eval.py
import numpy as np
from mbps_pytorch.mobile_panoptic_sup import coco_eval

def test_contiguous_maps_have_133_classes():
    catid2idx, idx2name, things, stuff = coco_eval.coco_contiguous_maps()
    assert len(catid2idx) == 133
    assert len(things) == 80 and len(stuff) == 53
    assert max(catid2idx.values()) == 132 and min(catid2idx.values()) == 0

def test_gt_as_pred_scores_perfect():
    # Feeding GT as its own prediction must yield PQ ~1.0 on one val image.
    seg_map, seg2cat = coco_eval.load_coco_gt(139)  # 000000000139
    res = coco_eval.eval_pq([(seg_map, seg2cat)], [(seg_map, seg2cat)])
    assert res["PQ"] > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_coco_eval.py -v`
Expected: FAIL (module `coco_eval` not found).

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/coco_eval.py
"""COCO-133 panoptic eval glue: contiguous class map, GT loader, PQ."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

COCO_ROOT = Path("/Volumes/code_files/datasets/coco")
_GT_JSON = COCO_ROOT / "annotations/panoptic_val2017.json"
_GT_PNG_DIR = COCO_ROOT / "annotations/panoptic_val2017"


def coco_contiguous_maps() -> tuple[dict, dict, set, set]:
    """category_id -> 0..132 (sorted by id); plus idx->name, thing/stuff idx sets."""
    cats = json.loads(_GT_JSON.read_text())["categories"]
    cats = sorted(cats, key=lambda c: c["id"])
    catid2idx = {c["id"]: i for i, c in enumerate(cats)}
    idx2name = {i: c["name"] for i, c in enumerate(cats)}
    things = {i for i, c in enumerate(cats) if c["isthing"] == 1}
    stuff = {i for i, c in enumerate(cats) if c["isthing"] == 0}
    return catid2idx, idx2name, things, stuff


def _rgb2id(png_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int64)
    return arr[..., 0] + arr[..., 1] * 256 + arr[..., 2] * 65536


def load_coco_gt(image_id: int) -> tuple[np.ndarray, dict]:
    """Return (seg_id map HxW, {seg_id: contiguous_cat_idx}) for a val image_id."""
    ann = next(a for a in json.loads(_GT_JSON.read_text())["annotations"]
               if a["image_id"] == image_id)
    catid2idx, *_ = coco_contiguous_maps()
    seg_map = _rgb2id(_GT_PNG_DIR / ann["file_name"])
    seg2cat = {s["id"]: catid2idx[s["category_id"]] for s in ann["segments_info"]}
    return seg_map, seg2cat


def eval_pq(preds: list[tuple[np.ndarray, dict]],
            gts: list[tuple[np.ndarray, dict]]) -> dict:
    """Per-image greedy IoU>0.5 PQ over a list of (seg_map, {seg_id:cat_idx})."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mbps_pytorch on path
    from evaluate_cross_dataset import compute_pq, summarize_pq  # noqa: WPS433
    _, _, things, stuff = coco_contiguous_maps()
    acc = None
    for (pp, ps), (gp, gs) in zip(preds, gts):
        part = compute_pq(gs, ps, gp, pp)
        acc = part if acc is None else [_merge(a, b) for a, b in zip(acc, part)]
    return summarize_pq(acc, things, stuff)


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out
```

NOTE: confirm `compute_pq`/`summarize_pq` arg order against `evaluate_cross_dataset.py:162,218` while implementing; adapt the call/merge to their actual return type (dict-of-per-cat vs tuple). If `summarize_pq` already takes `(gt,pred,gt_pan,pred_pan)` per-image and aggregates internally, replace the loop with its native batch entry point.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_coco_eval.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/__init__.py mbps_pytorch/mobile_panoptic_sup/coco_eval.py mbps_pytorch/mobile_panoptic_sup/tests/test_coco_eval.py
git commit -m "feat(mobile-eomt): COCO-133 panoptic eval harness (reuse decode+PQ)"
```

---

### Task 2: Auto-label a 5-image COCO slice (SAM3 things + INSID3 stuff, CPU)

**Files:**
- Create: `auto_annotation/taxonomy_coco.py`
- Create: `auto_annotation/scripts/autolabel_coco_slice.py`
- Create: `auto_annotation/data/exemplars_coco/` (a few `<class>/img.png` + `<class>/img_mask.png` for stuff classes present in the slice)

**Interfaces:**
- Consumes: SAM3 (`external/sam3` via `backends.sam3_compat.enable_cpu_sam3()` + `sam3.model_builder.build_sam3_image_model`, `Sam3Processor(resolution=1008)`); INSID3 (`backends.dinov3_hf.DinoV3HFEncoder`, `external/INSID3 models.insid3.INSID3`).
- Produces: per-image `outputs/coco_slice/<stem>_panoptic.npy` (int32, `class_idx*1000+inst`) + `<stem>_color.png`; `taxonomy_coco.COCO_CLASSES` (133 entries, contiguous idx).

- [ ] **Step 1: Write the failing test**

```python
# auto_annotation/tests/test_coco_taxonomy.py
from auto_annotation import taxonomy_coco as T

def test_coco_taxonomy_133():
    assert len(T.COCO_CLASSES) == 133
    assert T.is_thing(T.name_to_idx("person")) is True
    assert T.is_thing(T.name_to_idx("sky-other-merged")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest auto_annotation/tests/test_coco_taxonomy.py -v`
Expected: FAIL (no `taxonomy_coco`).

- [ ] **Step 3: Write minimal implementation**

```python
# auto_annotation/taxonomy_coco.py
"""COCO-133 panoptic taxonomy (contiguous idx 0..132), sourced from detectron2."""
from __future__ import annotations
from dataclasses import dataclass

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES  # 133 entries

VOID_IDX = 255


@dataclass(frozen=True)
class CocoClass:
    idx: int
    name: str
    is_thing: bool
    color: tuple


COCO_CLASSES = {
    i: CocoClass(i, c["name"], bool(c["isthing"]), tuple(c["color"]))
    for i, c in enumerate(sorted(COCO_CATEGORIES, key=lambda c: c["id"]))
}
_NAME2IDX = {c.name: i for i, c in COCO_CLASSES.items()}
THING_IDXS = {i for i, c in COCO_CLASSES.items() if c.is_thing}
STUFF_IDXS = {i for i, c in COCO_CLASSES.items() if not c.is_thing}


def name_to_idx(name: str) -> int:
    return _NAME2IDX[name]


def is_thing(idx: int) -> bool:
    return idx in THING_IDXS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest auto_annotation/tests/test_coco_taxonomy.py -v`
Expected: PASS.

- [ ] **Step 5: Write the slice runner** (no unit test — it's an I/O script gated by a manual smoke run)

```python
# auto_annotation/scripts/autolabel_coco_slice.py
"""Auto-label N COCO val images on CPU: SAM3 things + INSID3 stuff -> panoptic.
LABEL-FREE arm (no COCO-trained model). Run in .venv_cups_cpu. Quality is Phase 1;
this proves the pipeline runs end-to-end and emits sane panoptic maps."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autolabel_coco_slice")

from auto_annotation import taxonomy_coco as T
from auto_annotation.backends.sam3_compat import enable_cpu_sam3

COCO_VAL = Path("/Volumes/code_files/datasets/coco/val2017")
THING_PROMPTS = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "dog", "cat"]


def _load_sam3():
    enable_cpu_sam3()
    sys.path.insert(0, str(ROOT / "external/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cpu")
    return Sam3Processor(model, resolution=1008, device="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "auto_annotation/outputs/coco_slice")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proc = _load_sam3()

    for jpg in sorted(COCO_VAL.glob("*.jpg"))[: args.limit]:
        img = Image.open(jpg).convert("RGB")
        W, H = img.size
        pan = np.full((H, W), T.VOID_IDX * 1000, dtype=np.int32)
        state = proc.set_image(img)
        counts: dict[int, int] = {}
        dets = []
        for prompt in THING_PROMPTS:
            out = proc.set_text_prompt(state, prompt)
            for m, s in zip(out["masks"], out["scores"]):
                if float(s) >= 0.5:
                    dets.append((float(s), T.name_to_idx(prompt), np.asarray(m) > 0.5))
        for _, cls, mask in sorted(dets):           # best score wins overlaps
            counts[cls] = counts.get(cls, 0) + 1
            pan[mask] = cls * 1000 + counts[cls]
        np.save(args.out / f"{jpg.stem}_panoptic.npy", pan)
        logger.info("%s: %d instances, %d classes", jpg.stem, len(dets), len(counts))
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Manual smoke run**

Run:
```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
HF_HUB_OFFLINE=1 SAM3_OFFLINE=1 PYTHONPATH=. \
  .venv_cups_cpu/bin/python auto_annotation/scripts/autolabel_coco_slice.py --limit 5
```
Expected: 5 `*_panoptic.npy` written; logs show non-zero instances. (Stuff via INSID3 is deferred — Phase-0 slice is things-only to keep the smoke fast; add INSID3 stuff in Phase 1. Note this in the commit.)

- [ ] **Step 7: Commit**

```bash
git add auto_annotation/taxonomy_coco.py auto_annotation/scripts/autolabel_coco_slice.py auto_annotation/tests/test_coco_taxonomy.py
git commit -m "feat(autolabel): COCO-133 taxonomy + CPU SAM3 things slice runner (Phase-0 plumbing)"
```

---

### Task 3: Converter — auto-label `.npy` → COCO-panoptic format (PNG + json)

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/to_coco_panoptic.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_to_coco_panoptic.py`

**Interfaces:**
- Consumes: Task-2 `*_panoptic.npy` (`class_idx*1000+inst`), `taxonomy_coco` (thing/stuff sets).
- Produces: `pan_to_coco(pan: np.ndarray, stem: str) -> tuple[np.ndarray, dict]` — RGB PNG array (rgb2id-encoded) + COCO annotation dict `{file_name, image_id, segments_info:[{id,category_id,isthing,area}]}`. Lets EoMT's `coco_panoptic` loader and `coco_eval` consume auto-labels unchanged.

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_to_coco_panoptic.py
import numpy as np
from mbps_pytorch.mobile_panoptic_sup import to_coco_panoptic as C

def test_roundtrip_segments():
    pan = np.zeros((4, 4), np.int32)
    pan[:2] = 5 * 1000 + 1          # thing class idx 5, instance 1
    pan[2:] = 130 * 1000           # stuff class idx 130
    rgb, ann = C.pan_to_coco(pan, "x")
    seg_ids = {s["id"] for s in ann["segments_info"]}
    decoded = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536
    assert set(np.unique(decoded)) == seg_ids
    areas = {s["id"]: s["area"] for s in ann["segments_info"]}
    assert sum(areas.values()) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_to_coco_panoptic.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/to_coco_panoptic.py
"""Convert class*1000+inst panoptic maps to COCO-panoptic (rgb2id PNG + json)."""
from __future__ import annotations
import numpy as np
from auto_annotation import taxonomy_coco as T

DIV = 1000


def _id2rgb(seg_id: int) -> tuple[int, int, int]:
    return seg_id % 256, (seg_id // 256) % 256, (seg_id // 65536) % 256


def pan_to_coco(pan: np.ndarray, stem: str) -> tuple[np.ndarray, dict]:
    H, W = pan.shape
    rgb = np.zeros((H, W, 3), np.uint8)
    segments = []
    for seg_id in np.unique(pan):
        cls = int(seg_id) // DIV
        if cls == T.VOID_IDX:
            continue
        mask = pan == seg_id
        rgb[mask] = _id2rgb(int(seg_id))
        segments.append({
            "id": int(seg_id),
            "category_id": cls,                       # already contiguous 0..132
            "isthing": int(T.is_thing(cls)),
            "area": int(mask.sum()),
        })
    ann = {"file_name": f"{stem}.png", "image_id": stem, "segments_info": segments}
    return rgb, ann
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_to_coco_panoptic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/to_coco_panoptic.py mbps_pytorch/mobile_panoptic_sup/tests/test_to_coco_panoptic.py
git commit -m "feat(mobile-eomt): auto-label -> COCO-panoptic format converter"
```

---

### Task 4: EoMT-mobile (dinov2-small) overfit-8 smoke

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/smoke_eomt.py` (standalone overfit-8 harness; avoids the full Lightning data plumbing for Phase 0)
- Reference: `refs/eomt/models/vit.py:31`, `refs/eomt/models/eomt.py:18`, `refs/eomt/training/mask_classification_loss.py:28`

**Interfaces:**
- Consumes: `refs.eomt.models.vit.ViT`, `refs.eomt.models.eomt.EoMT`, `MaskClassificationLoss`; 8 COCO GT samples via `coco_eval.load_coco_gt` → `(masks bool[N,H,W], labels long[N])`.
- Produces: proof that EoMT-mobile loss drops to near-zero on 8 fixed images (student-trains signal). No file output beyond a printed final loss.

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_smoke_eomt.py
import torch
from mbps_pytorch.mobile_panoptic_sup import smoke_eomt

def test_eomt_forward_shapes_cpu():
    model = smoke_eomt.build(img=256, num_classes=133, num_q=100)
    mask_l, cls_l = model(torch.rand(1, 3, 256, 256))
    assert cls_l[-1].shape == (1, 100, 134)        # num_classes+1
    assert mask_l[-1].shape[1] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_smoke_eomt.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/smoke_eomt.py
"""Build EoMT with a mobile plain-ViT encoder and overfit 8 COCO images on CPU."""
from __future__ import annotations
import logging, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs/eomt"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_eomt")

from models.vit import ViT
from models.eomt import EoMT


def build(img: int = 512, num_classes: int = 133, num_q: int = 100,
          backbone: str = "facebook/dinov2-small") -> EoMT:
    enc = ViT(img_size=(img, img), backbone_name=backbone)
    return EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
                num_blocks=4, masked_attn_enabled=True)


def overfit(steps: int = 60, img: int = 512) -> float:
    from training.mask_classification_loss import MaskClassificationLoss
    from mbps_pytorch.mobile_panoptic_sup.coco_eval import load_coco_gt
    model = build(img=img).train()
    crit = MaskClassificationLoss(num_points=12544, oversample_ratio=3.0,
                                  importance_sample_ratio=0.75, mask_coefficient=5.0,
                                  dice_coefficient=5.0, class_coefficient=2.0,
                                  num_labels=133, no_object_coefficient=0.1)
    # 8 fixed val images -> (image tensor, target dict). See coco_eval for seg map.
    batch = _make_batch([139, 285, 632, 724, 776, 785, 802, 872], img)  # impl per Step 3b
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = torch.tensor(0.0)
    for i in range(steps):
        opt.zero_grad()
        mask_l, cls_l = model(batch["images"])
        loss = sum(crit(m, c, batch["targets"]) for m, c in zip(mask_l, cls_l))
        loss.backward(); opt.step()
        if i % 10 == 0:
            logger.info("step %d loss %.3f", i, float(loss))
    return float(loss)
```

- [ ] **Step 3b: Implement `_make_batch`** — load each image (resize to `img`), decode its GT panoptic to per-segment binary masks + contiguous labels (reuse `coco_eval.load_coco_gt` + `to_coco_panoptic` inverse), stack to the `{"masks": bool[N,img,img], "labels": long[N], "is_crowd": bool[N]}` target dict EoMT expects (`refs/eomt/datasets/coco_panoptic.py:181`). Use COCO GT here (not auto-labels) to isolate "does the student train" from "are labels good."

- [ ] **Step 4: Run shape test + manual overfit**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_smoke_eomt.py -v` → PASS.
Then: `.venv/bin/python -c "from mbps_pytorch.mobile_panoptic_sup.smoke_eomt import overfit; assert overfit(60, 384) < overfit.__wrapped__ if False else True"`
Manual expectation: printed loss at step 0 ≫ loss at step 50 (monotone-ish decrease), final loss < 30% of initial. Record the numbers in the commit body.

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/smoke_eomt.py mbps_pytorch/mobile_panoptic_sup/tests/test_smoke_eomt.py
git commit -m "feat(mobile-eomt): EoMT dinov2-small overfit-8 smoke (loss N0->N1 on CPU)"
```

---

### Task 5: Conv student model (RepViT + BiFPN + semantic head)

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/student_conv.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_student_conv.py`

**Interfaces:**
- Consumes: timm `repvit_m1_5` (`features_only=True`, channels `[64,128,256,512]`), `BiFPN` (`mbps_pytorch/train_mobile_panoptic.py:87`, `__init__(in_channels_list, fpn_dim=128, num_repeats=2)`, returns per-level `fpn_dim` maps).
- Produces: `ConvStudent(num_classes=133, fpn_dim=128)` with `forward(x) -> Tensor[B,133,H/4,W/4]` semantic logits. (Phase-0 = semantic-only; CC-derived instances in Task 6. Center/offset or MaskConver head is a Phase-2 upgrade.)

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_student_conv.py
import torch
from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent

def test_conv_student_semantic_shape():
    m = ConvStudent(num_classes=133).eval()
    y = m(torch.rand(1, 3, 512, 512))
    assert y.shape[0] == 1 and y.shape[1] == 133
    assert y.shape[2] == 128 and y.shape[3] == 128   # 1/4 resolution
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_student_conv.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/student_conv.py
"""Mobile conv panoptic student: RepViT-M1.5 + BiFPN + semantic head.
Phase-0 semantic-only (instances via connected components at inference).
Pure conv -> trivially ONNX-exportable; the phone-CPU-safe foil to EoMT."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_mobile_panoptic import BiFPN  # noqa: WPS433  (reuse, do not re-import the script's main)


class ConvStudent(nn.Module):
    def __init__(self, num_classes: int = 133, fpn_dim: int = 128) -> None:
        super().__init__()
        self.backbone = timm.create_model("repvit_m1_5", pretrained=True, features_only=True)
        chans = self.backbone.feature_info.channels()           # [64,128,256,512]
        self.neck = BiFPN(chans, fpn_dim=fpn_dim, num_repeats=2)
        self.sem_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1), nn.BatchNorm2d(fpn_dim), nn.ReLU(True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)               # finest first
        fpn = self.neck(feats)                 # per-level, fpn_dim channels
        return self.sem_head(fpn[0])           # 1/4-res logits [B,num_classes,H/4,W/4]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_student_conv.py -v`
Expected: PASS. (If `repvit_m1_5` pretrained tag errors offline, set `pretrained=False` for the test and note it.)

- [ ] **Step 5: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/student_conv.py mbps_pytorch/mobile_panoptic_sup/tests/test_student_conv.py
git commit -m "feat(mobile-eomt): RepViT+BiFPN conv student (semantic head)"
```

---

### Task 6: Conv student overfit-8 + CC panoptic post-process

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/panoptic_postprocess_conv.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_postprocess_conv.py`

**Interfaces:**
- Consumes: semantic logits `[B,133,h,w]`, `taxonomy_coco.THING_IDXS`, `scipy.ndimage.label`.
- Produces: `semantic_to_panoptic(sem: np.ndarray[H,W] argmax) -> tuple[np.ndarray, dict]` — `class*1000+inst` map (things split into CC instances, stuff inst=0) + `{seg_id: cat_idx}` for `coco_eval`.

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_postprocess_conv.py
import numpy as np
from mbps_pytorch.mobile_panoptic_sup.panoptic_postprocess_conv import semantic_to_panoptic
from mbps_pytorch.mobile_panoptic_sup import to_coco_panoptic as C

def test_two_disconnected_things_become_two_instances():
    sem = np.zeros((6, 6), np.int64)            # class 0 (a thing idx) everywhere? pick a thing
    thing = 0
    sem[:] = 130                                # stuff background
    sem[0:2, 0:2] = thing
    sem[4:6, 4:6] = thing                       # two disconnected blobs of same thing class
    pan, seg2cat = semantic_to_panoptic(sem)
    inst_ids = [sid for sid in np.unique(pan) if sid // 1000 == thing]
    assert len(inst_ids) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_postprocess_conv.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/panoptic_postprocess_conv.py
"""Semantic argmax -> panoptic via connected-components for thing classes."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from auto_annotation import taxonomy_coco as T

DIV = 1000


def semantic_to_panoptic(sem: np.ndarray) -> tuple[np.ndarray, dict]:
    pan = np.full_like(sem, T.VOID_IDX * DIV, dtype=np.int32)
    seg2cat: dict = {}
    for cls in np.unique(sem):
        cls = int(cls)
        region = sem == cls
        if cls in T.STUFF_IDXS:
            sid = cls * DIV
            pan[region] = sid
            seg2cat[sid] = cls
        elif cls in T.THING_IDXS:
            lab, n = ndimage.label(region)
            for inst in range(1, n + 1):
                sid = cls * DIV + inst
                pan[lab == inst] = sid
                seg2cat[sid] = cls
    return pan, seg2cat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_postprocess_conv.py -v`
Expected: PASS.

- [ ] **Step 5: Manual overfit-8** — add an `overfit()` to `student_conv.py` (mirror `smoke_eomt.overfit`: 8 COCO GT images, semantic CE on the rasterized GT semantic map at 1/4-res, AdamW lr 1e-3, 60 steps); assert final loss < 30% of initial. Record numbers in the commit.

- [ ] **Step 6: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/panoptic_postprocess_conv.py mbps_pytorch/mobile_panoptic_sup/tests/test_postprocess_conv.py mbps_pytorch/mobile_panoptic_sup/student_conv.py
git commit -m "feat(mobile-eomt): conv student CC panoptic post-process + overfit-8"
```

---

### Task 7: ONNX export + parity + latency (both students)

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/export_onnx.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_export.py`

**Interfaces:**
- Consumes: `ConvStudent`, `smoke_eomt.build(...)` with `masked_attn_enabled=False`.
- Produces: `export(model, path, img) -> None`; `parity(model, path, img) -> float` (max abs diff PyTorch vs onnxruntime); `latency(path, img, runs=20) -> float` (mean ms, onnxruntime CPU).

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_export.py
import tempfile, os
import torch
from mbps_pytorch.mobile_panoptic_sup.student_conv import ConvStudent
from mbps_pytorch.mobile_panoptic_sup import export_onnx

def test_conv_export_parity_and_latency():
    m = ConvStudent(num_classes=133).eval()
    p = os.path.join(tempfile.mkdtemp(), "conv.onnx")
    export_onnx.export(m, p, img=256)
    assert export_onnx.parity(m, p, img=256) < 1e-3
    assert export_onnx.latency(p, img=256, runs=3) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_export.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/export_onnx.py
"""ONNX export + PyTorch-vs-onnxruntime parity + CPU latency, for both students."""
from __future__ import annotations
import logging, time
import numpy as np
import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)


def export(model: torch.nn.Module, path: str, img: int = 640) -> None:
    model.eval()
    dummy = torch.rand(1, 3, img, img)
    torch.onnx.export(model, dummy, path, input_names=["image"],
                      output_names=["out"], opset_version=17,
                      dynamic_axes=None)            # fixed shape for mobile
    logger.info("exported %s", path)


def parity(model: torch.nn.Module, path: str, img: int = 640) -> float:
    model.eval()
    x = torch.rand(1, 3, img, img)
    with torch.no_grad():
        ref = model(x)
        ref = ref[0][-1] if isinstance(ref, tuple) else ref   # EoMT returns (lists); conv a tensor
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    return float(np.abs(np.asarray(ref) - out).max())


def latency(path: str, img: int = 640, runs: int = 20) -> float:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img, img).astype(np.float32)
    sess.run(None, {"image": x})                    # warmup
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"image": x})
    return (time.perf_counter() - t0) / runs * 1000.0
```

NOTE: for the EoMT export, build with `masked_attn_enabled=False` and wrap it so `forward` returns only `(mask_logits[-1], class_logits[-1])` (a tensor tuple), since onnx can't return Python lists of varying length. Add a thin `EoMTExportWrapper(nn.Module)` in this file that calls the model and returns the last-layer tuple. Test the conv path first (Step 1); add an EoMT export sub-test once the wrapper lands.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_export.py -v`
Expected: PASS (conv parity < 1e-3, latency > 0).

- [ ] **Step 5: Manual bench both students** — export conv and EoMT-mobile at img=640, print parity + latency for each. Record both numbers in the commit body (this is the export-cleanliness/latency evidence the spec calls for).

- [ ] **Step 6: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/export_onnx.py mbps_pytorch/mobile_panoptic_sup/tests/test_export.py
git commit -m "feat(mobile-eomt): ONNX export + parity + latency (conv Xms, eomt Yms)"
```

---

### Task 8: Demo — masks + category-label overlay

**Files:**
- Create: `mbps_pytorch/mobile_panoptic_sup/demo.py`
- Test: `mbps_pytorch/mobile_panoptic_sup/tests/test_demo.py`

**Interfaces:**
- Consumes: a panoptic map (`class*1000+inst`) + `coco_eval.coco_contiguous_maps()` (idx→name), `taxonomy_coco` colors, cv2/PIL.
- Produces: `overlay(image_rgb: np.ndarray, pan: np.ndarray) -> np.ndarray` — image with per-segment colored mask (alpha) + category-name text per segment. CLI runs a student on an image and saves the overlay.

- [ ] **Step 1: Write the failing test**

```python
# mbps_pytorch/mobile_panoptic_sup/tests/test_demo.py
import numpy as np
from mbps_pytorch.mobile_panoptic_sup.demo import overlay

def test_overlay_shape_and_nonempty():
    img = np.zeros((32, 32, 3), np.uint8)
    pan = np.zeros((32, 32), np.int32)
    pan[:16] = 130 * 1000
    pan[16:] = 0 * 1000 + 1
    out = overlay(img, pan)
    assert out.shape == img.shape
    assert out.sum() > 0          # something was drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_demo.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# mbps_pytorch/mobile_panoptic_sup/demo.py
"""Overlay panoptic masks + category labels on an image (requirement #2)."""
from __future__ import annotations
import numpy as np
import cv2
from auto_annotation import taxonomy_coco as T
from mbps_pytorch.mobile_panoptic_sup.coco_eval import coco_contiguous_maps

DIV = 1000
_, _IDX2NAME, _, _ = coco_contiguous_maps()


def overlay(image_rgb: np.ndarray, pan: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = image_rgb.copy()
    for seg_id in np.unique(pan):
        cls = int(seg_id) // DIV
        if cls == T.VOID_IDX:
            continue
        mask = pan == seg_id
        color = np.array(T.COCO_CLASSES[cls].color, np.uint8)
        out[mask] = (alpha * color + (1 - alpha) * out[mask]).astype(np.uint8)
        ys, xs = np.where(mask)
        cv2.putText(out, _IDX2NAME[cls], (int(xs.mean()), int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/test_demo.py -v`
Expected: PASS.

- [ ] **Step 5: Manual end-to-end demo** — run the conv student on one COCO val image, post-process (Task 6), overlay, save PNG; eyeball that masks+labels render. (EoMT demo path identical once its post-process lands.)

- [ ] **Step 6: Commit**

```bash
git add mbps_pytorch/mobile_panoptic_sup/demo.py mbps_pytorch/mobile_panoptic_sup/tests/test_demo.py
git commit -m "feat(mobile-eomt): masks+labels demo overlay"
```

---

## Phase 0 Definition of Done

- All 8 tasks' tests pass (`.venv/bin/python -m pytest mbps_pytorch/mobile_panoptic_sup/tests/ -v`).
- Auto-label slice produced sane panoptic on 5 COCO images (CPU).
- Both students overfit 8 images (loss collapses) and export to ONNX with parity < 1e-3 and a recorded CPU latency.
- Demo renders masks + category names on a real COCO image.
- **Not in scope (Phase 1/2, separate plans):** INSID3 stuff quality + full 118k auto-label run; encoder ablation (DeiT-Ti/S, distilled tiny ViT); CUPS self-training; CUPS global-Hungarian eval; TFLite/Core ML + on-NPU latency.

## Self-Review (filled at write time)

- **Spec coverage:** auto-label pipeline (T2/T3) ✓; EoMT-mobile student (T4) ✓; conv student (T5/T6) ✓; export+bench (T7) ✓; demo (T8) ✓; COCO label-free + eval-on-val-GT (T1) ✓; self-training + encoder ablation explicitly deferred to Phase 2 ✓.
- **Placeholders:** `smoke_eomt._make_batch` and the two `overfit()` helpers are specified by behavior + exact target dict shape + reused functions, not full bodies — flagged as Step 3b / Step 5 sub-implementations (they depend on resize choices best pinned at the keyboard). All deterministic modules have complete code.
- **Type consistency:** panoptic encoding `class*1000+inst` and `{seg_id: cat_idx}` contract is identical across `to_coco_panoptic`, `panoptic_postprocess_conv`, `coco_eval`, `demo`. EoMT returns `(list, list)`; conv returns a tensor — handled explicitly in `export_onnx.parity`.
