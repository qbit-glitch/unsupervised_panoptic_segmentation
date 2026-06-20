# Auto-Label → Mobile-EoMT Panoptic Segmentation (COCO-133, label-free) — Design Spec

**Date:** 2026-06-21
**Status:** Design (v2, supersedes v1 distillation draft) — pending user review → writing-plans
**Context:** A *label-free* mobile panoptic model. A foundation-model pipeline (Mask2Former + INSID3 for semantics, SAM3 for instances) auto-annotates the training set; we then train + self-train a compact mobile student (EoMT with a mobile ViT encoder, plus a pure-conv baseline) into an export-clean, real-time, on-device panoptic model that displays masks + categories.

This reuses two subsystems already in the repo and proven on the Jaipur dataset: `auto_annotation/` (the exact INSID3+SAM3+merge+QA pipeline) and `refs/eomt/` (EoMT with a pluggable ViT encoder). The only genuinely new build is the **mobile ViT encoder sweep + a conv student + export/bench/demo + a COCO adapter for the auto-labeler**.

---

## 1. Goal & honest bars

Produce a real-time, on-device panoptic model trained **without human labels on the training images**, benchmarked on COCO-Panoptic (133 classes), displaying masks + object categories.

**Accuracy bar (verified 2026-06-21, hardware-honest — memory `edge-segmentation-benchmarks`):** the only published *on-phone* panoptic result is MaskConver-256 (29.7 PQ COCO @ 33 FPS Pixel 6, pure-conv, 3.4M). Everything else ("real-time" YOSO/PEM/k-MaX) is desktop-GPU. Realistic phone band ≈ **PQ 30–42**. We additionally lose accuracy by training on *auto-labels instead of GT*, so the honest expectation is **below GT-supervised mobile models**; the contribution is **label-free deployability**, not beating GT.

**Two honesty caveats baked into the design:**
1. **"Label-free" precision.** Mask2Former is COCO-GT-trained, so using it to label COCO is distillation from a supervised teacher, *not* label-free. The defensible label-free result on COCO uses **INSID3 (training-free, frozen DINOv3, few-shot exemplars) + SAM3 (concept prompts)** only. Mask2Former is an explicitly-disclosed **upper-bound arm**. INSID3's per-class exemplars make this *few-shot*, not zero-label — stated, not hidden. The genuinely label-free story is strongest on a no-GT domain (Jaipur) — noted as the transfer application.
2. **EoMT ↔ mobile tension.** EoMT needs a **plain/columnar ViT** (it appends query tokens to a constant-resolution token stream). The most phone-efficient backbones (MobileViT, EfficientViT, iFormer) are hierarchical/hybrid and **cannot** host EoMT. So mobile-EoMT efficiency comes from a **small plain ViT** (ViT-S/Ti) at modest resolution → realistically **edge-GPU / high-end-phone-NPU** real-time, not phone-CPU. The pure-conv student is the phone-CPU-safe anchor; the EoMT encoder sweep tests how far a plain ViT can be pushed onto an NPU.

---

## 2. Data

- **Dataset:** COCO-Panoptic, 133 contiguous classes. On disk `/Volumes/code_files/datasets/coco/`: train2017 (118,287 imgs), val2017 (5,000 imgs + 5,000 panoptic GT PNGs), `panoptic_{train,val}2017.json`.
- **Training labels = auto-generated** (Section 3). **COCO train GT is NOT used** → the missing `panoptic_train2017` PNGs are no longer a blocker for the label-free arm. (Only the disclosed Mask2Former-upper-bound and any GT-oracle ablation would need them.)
- **Evaluation = COCO val GT** (present): panoptic PQ / SQ / RQ, split into things/stuff, on 5,000 val images.
- **Augmentation:** CUPS-style — large-scale jitter + crop + flip + photometric + copy-paste (reuse `refs/eomt` / repo aug).

---

## 3. Auto-label pipeline (the "teacher")

Reuse `auto_annotation/` (stages: `semantic_insid3.py`, `instances_sam3.py`, `boundary_refine.py`, `panoptic_merge.py`, `quality.py`; backends `dinov3_hf.py`, `sam3_compat.py`). New work = a **COCO taxonomy + config adapter** and a runner over train2017.

- **Semantics (label-free arm):** INSID3 in-context, 1 frozen-DINOv3 exemplar / COCO class (133 exemplars curated once).
- **Instances:** SAM3 concept prompts (text + exemplar) per thing class.
- **Refine + merge:** SAM-snap boundary refine → panoptic merge → QA (cleanlab soft-min + entropy) → CUPS-format panoptic auto-labels (reuses the repo's pseudo-label format and tooling).
- **Disclosed upper-bound arm:** add Mask2Former-COCO to the common-class semantic path → higher-quality labels, reported separately as supervised-teacher distillation (not label-free).
- **Compute:** running INSID3(DINOv3) + SAM3 over 118k images is heavy (GPU-days on the A6000) — a one-time offline job, cached. The pipeline already routes low-QA frames aside; for COCO we keep all auto-labels (no human verify loop).

---

## 4. Students — build both, shared auto-labels + self-training

**Student E — EoMT-mobile (`refs/eomt`, encoder swept).** EoMT framework, encoder = a **mobile plain ViT**, ablated across: **DINOv2/v3 ViT-S**, **DeiT-Tiny**, **DeiT-Small**, and a **DINOv3→tiny-ViT distilled** encoder. EoMT loads any backbone by name via its existing `ViT` wrapper; we add mobile-encoder configs under `refs/eomt/configs` and reuse its training loop (incl. DropLoss). EoMT inference anneals out masked-attention → plain forward → export-clean.

**Student C — pure-conv baseline (MaskConver-style).** Mobile conv backbone (RepViT-M1.5, timm) + lightweight panoptic head (class-wise centers + conv mask generator; Panoptic-DeepLab-style fallback). The proven on-phone paradigm and the phone-CPU-safe anchor. New code.

Both students train on the **same auto-labels**, then **self-train** (Section 5). Same eval, same export harness, same demo.

---

## 5. Training & self-training

1. **Warm-up on auto-labels:** supervised-style training of each student on the cached auto-labels (EoMT set-prediction loss for E; center/mask/semantic loss for C). ImageNet/DINO-pretrained encoders speed convergence.
2. **Self-training (CUPS recipe, reuse refs/eomt):** EMA teacher-student, per-class confidence thresholding, DropLoss, copy-paste, multi-res — the project's proven self-training that lifted EoMT on Jaipur. Refines past the auto-label noise ceiling.

Optional levers (add only if a baseline plateaus): soft-logit distillation from the foundation pipeline; feature distillation; teacher-TTA pseudo-labels.

---

## 6. Panoptic post-processing, export, demo

- **Post-process (host-side):** E → EoMT's query→panoptic merge; C → center-NMS + conv-mask assembly + stuff-from-semantic merge. Output panoptic id-map + per-segment category + thing instance ids.
- **Export + bench:** each variant → `torch.onnx.export` at fixed resolution → ONNX; parity vs PyTorch; latency via onnxruntime (CPU + CoreML EP on the Mac). Report ms/FPS per variant. Hypothesis under test: conv (C) exports cleaner/faster; which EoMT encoder, if any, is NPU-viable. TFLite/Core ML conversion is the device-specific follow-on.
- **Demo (`demo.py`):** image/folder/webcam → post-process → per-instance colored masks + category-name labels → save/display (requirement #2).

---

## 7. File layout

- **Reuse, lightly extend `auto_annotation/`:** add `taxonomy_coco.py` + `configs/coco.yaml` + a train2017 runner script. (~250 new LOC; rest reused.)
- **Reuse `refs/eomt/`:** add mobile-encoder configs + an encoder registry for the ablation; reuse its training loop. (~150 new LOC.)
- **New package `mbps_pytorch/mobile_panoptic_sup/`** (200–400 LOC/file):

| file | purpose | ~LOC |
|---|---|---|
| `data.py` | load CUPS-format auto-labels + COCO imgs + aug; COCO val-GT loader | 280 |
| `student_conv.py` | MaskConver-style conv student + loss | 320 |
| `selftrain.py` | EMA teacher-student loop (shared by E and C) | 240 |
| `train.py` | entry, frozen-dataclass config, `--student {eomt,conv}`, `--encoder ...` | 320 |
| `eval_coco.py` | panoptic PQ on COCO val GT (things/stuff) | 200 |
| `export_onnx.py` | ONNX export + parity + latency, all variants | 220 |
| `demo.py` | mask + label overlay viz | 160 |
| `tests/` | auto-label loader, conv forward/loss, EoMT-mobile smoke, overfit-8 | 240 |

---

## 8. Honest scope & phasing (this is a multi-week research program, not one session)

The full matrix — auto-label 118k imgs + 4 EoMT encoders + conv student + self-training each + export/bench all + COCO eval — is GPU-weeks. Phasing:

- **Phase 0 (this session):** COCO adapter for `auto_annotation` + auto-label a **small slice (~50–200 imgs)** end-to-end; wire **one** mobile encoder into EoMT + the conv student; overfit-8 smoke (proves both students learn from auto-labels); export+bench that pair; demo on a smoke checkpoint. *In-session PQ is meaningless (smoke).*
- **Phase 1 (handoff, A6000):** full COCO auto-labeling run (GPU-days, cached).
- **Phase 2 (handoff):** train + self-train each student/encoder variant; eval on COCO val GT; export+bench all; fill the results table.

Each phase is independently verifiable; I hand off the commands + frozen configs.

---

## 9. Explicitly skipped (add when)

- Human-verify loop from `auto_annotation` (COCO keeps all auto-labels) — add for the Jaipur transfer.
- Hierarchical mobile backbones for EoMT — incompatible (plain-ViT only); don't attempt.
- SSM/Mamba encoder — doesn't export to NPU; paper-experiment only.
- Soft-logit/feature distillation, teacher ensemble — accuracy levers after baselines hold.
- TFLite/Core ML conversion + on-NPU latency — when a device is chosen.
- Full native app — separate project.

---

## 10. Success criteria

- **Phase 0:** COCO auto-labels generated for a slice and visually sane; both students overfit 8 auto-labeled images to near-zero loss (committed smoke test); both export to ONNX with PyTorch parity; latency reported for both; demo renders correct masks+labels on a COCO image.
- **Full run readiness:** auto-label + train + self-train + eval + export commands and frozen configs handed off, with the honest PQ-band and label-free caveats documented.
- **Paper-facing result (post-run):** a COCO val PQ for the label-free (INSID3+SAM3) arm and the disclosed Mask2Former upper-bound arm, per student and per EoMT encoder, with on-device latency — positioned against MaskConver's ~30 PQ on-phone bar.
