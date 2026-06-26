# auto_annotation — monocular-video panoptic auto-labeling

Foundation-model pipeline to auto-annotate your own dashcam video for panoptic
segmentation on **unstructured Indian/South-Asian roads** (IDD-style taxonomy,
auto-rickshaws), then route low-confidence frames to human verification.

Design comes from the verified deep-research report (run `wf_b7e9efce-09a`, 2026-06-17)
and the paper **INSID3** (the in-context semantic engine).

```
raw video
  │  extract_keyframes (cv2, stride)
  ▼
keyframes ──► SEMANTIC: INSID3 in-context (1 DINOv3 exemplar / class)   ┐
  │          (rare/local classes; common classes via FC-CLIP/Mask2Former)│
  │      ──► INSTANCES: SAM 3 concept prompts (text + image exemplars)   ┘
  ▼
SAM-snap boundary refine ─► panoptic merge ─► QA score (cleanlab soft-min + entropy)
  ▼
route:  split=train & high-quality ─► AUTO train labels   (outputs/auto/)
        split=val/test OR low-quality ─► HUMAN verify     (outputs/review/)
```

## Why these models (verified citations)

| Stage | Model | Citation |
|-------|-------|----------|
| Semantic (rare/local, in-context) | **INSID3** training-free, frozen DINOv3 | Cuttano, Trivigno, Reich, Cremers, Masone, Roth. *INSID3: Training-Free In-Context Segmentation with DINOv3.* CVPR 2026 Oral. [arXiv:2603.28480](https://arxiv.org/abs/2603.28480) · [code](https://github.com/visinf/INSID3) |
| Instances (promptable, trackable) | **SAM 3** detect+segment+track from concepts | Meta/FAIR. *SAM 3: Segment Anything with Concepts.* [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) |
| Instances (alt) | **Grounded-SAM-2** (Grounding DINO → SAM 2) | [repo](https://github.com/IDEA-Research/Grounded-SAM-2) · Grounding DINO [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) · SAM 2 [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) |
| Semantic (common classes, alt) | FC-CLIP / CAT-Seg open-vocab | FC-CLIP [arXiv:2308.02487](https://arxiv.org/abs/2308.02487) · CAT-Seg [arXiv:2303.11797](https://arxiv.org/abs/2303.11797) |
| QA / routing | cleanlab soft-min + U2PL entropy | [arXiv:2307.05080](https://arxiv.org/abs/2307.05080) · U2PL CVPR 2022 |
| Taxonomy | India Driving Dataset (IDD) 4-level | [arXiv:1811.10200](https://arxiv.org/abs/1811.10200) |

## Layout

```
auto_annotation/
├── config.py            # frozen PipelineConfig (all hyperparameters)
├── taxonomy.py          # IDD-style classes + thing/stuff split  (EDIT for your set)
├── schemas.py           # SemanticResult / InstanceResult / PanopticResult
├── io_utils.py          # video decode, keyframe sampling, label I/O, colorize
├── pipeline.py          # orchestrator
├── run.py               # CLI  (python -m auto_annotation.run ...)
├── stages/
│   ├── __init__.py        # backend registry (factory)
│   ├── semantic_insid3.py # INSID3 seam + dummy
│   ├── instances_sam3.py  # SAM3 seam + dummy
│   ├── boundary_refine.py # SAM-snap (implemented)
│   ├── panoptic_merge.py  # semantic+instances -> panoptic (implemented)
│   └── quality.py         # label-quality + routing (implemented)
└── tests/test_smoke.py    # end-to-end dummy test (5 passing)
```

## Quickstart (no weights — verifies merge/QA/IO)

```bash
.venv_cups_cpu/bin/python -m pytest auto_annotation/tests/test_smoke.py -v
.venv_cups_cpu/bin/python -m auto_annotation.run \
    --video <your.mp4> --semantic dummy --instance dummy --split train
```

## Status (verified 2026-06-17, Mac CPU)

- ✅ **INSID3 backend WORKS** (`stages/semantic_insid3.py`, real). Loads frozen DINOv3
  via HF `transformers` (`backends/dinov3_hf.py`) — no gated `.pth` needed. Validated:
  1-shot road IoU **0.896** vs gtFine; 7-concept demo on 5 Cityscapes imgs = mean
  semantic mIoU **0.564**. See `scripts/validate_insid3.py`, `scripts/run_cityscapes_demo.py`.
- ✅ **SAM 3 backend WORKS on CPU** (`stages/instances_sam3.py`, real). SAM 3 is
  CUDA-coupled as shipped (hard `import triton`, hardcoded `device="cuda"`, bf16, 
  `.pin_memory()`), but `backends/sam3_compat.py::enable_cpu_sam3()` shims all of it so
  the image path runs on CPU. **Verified**: text 'car' on Cityscapes → 5 masks (scores
  ~0.95) in **~3 s/img**; scaffold registry → `{car:5, person:5, bicycle:1}`. Probe:
  `scripts/try_sam3_cpu.py`. CUDA still preferred for dataset-scale throughput; use
  `build_sam3_video_predictor` (CUDA) for temporally-consistent video instance ids.

### Run with both real models
```bash
# weights auto-download from HF (facebook/dinov3-*, facebook/sam3) with an
# accepted-license HF token — no manual Meta form needed.
# CPU (Mac), works today:
python -m auto_annotation.run --semantic insid3 --instance sam3 --device cpu ...
# CUDA box (A6000/santosh), faster: pip install -e external/sam3 first, then --device cuda
```
SAM 3 deps for CPU (already installed in `.venv_cups_cpu`): `ftfy iopath regex timm`
(sam3 itself installed with `--no-deps` to keep torch 2.12 / numpy 2.x).

### INSID3 exemplars (in-context support)

One folder per concept; folder name must match a `taxonomy.py` class name:

```
auto_annotation/data/exemplars/
├── auto rickshaw/0001.png  0001_mask.png
├── road/0001.png           0001_mask.png
└── ...
```

3–5 exemplars per class is plenty (it's one-shot/few-shot).

## Annotation protocol (verified-val / auto-train)

- `--split train` → frames auto-accepted unless QA flags them (`outputs/auto/`).
- `--split val` / `--split test` → **always** routed to human verification
  (`outputs/review/`) so they become trusted ground truth.
- Verify flagged frames in CVAT / Label Studio / X-AnyLabeling with a SAM backend.

## Do this FIRST (one calibration pass)

Hand-label ~50–200 of *your own* frames and measure each backend's recall/IoU —
especially auto-rickshaws and dense clusters. All published metrics are on
COCO/LVIS/Cityscapes; **none on Indian unstructured traffic**. Calibrate before
scaling.

## Honest caveats

- INSID3 is a *semantic / rare-class* annotator, **not** an instance separator and
  **not** a one-shot full-scene panoptic model — instances come from SAM 3.
- DINOv3/open-vocab boundaries are patch-grid-soft → keep `use_boundary_refine` on and
  feed SAM "everything"-mode masks for stuff edges (the thing masks are auto-fed).
- SAM 2/3 tracking degrades in crowds / long occlusions — those frames should fail QA
  and route to review; consider DVIS++ / TarVIS for hard sequences.
