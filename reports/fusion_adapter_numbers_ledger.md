# Fusion Adapter — Verified Numbers Ledger

Spec: `docs/superpowers/specs/2026-06-12-depthg-cause-fusion-adapter-design.md` (rev 2).
Rule: no number appears in any spec/report/prose unless it has a row here.

## Published baselines (verified 2026-06-12)

| # | Value | Metric | System | Protocol notes | Source |
|---|-------|--------|--------|----------------|--------|
| P1 | 29.9 | cluster mIoU | CAUSE-TR, DINOv2 ViT-B/14, Cityscapes | 27-class, Hungarian, CRF (official repo eval) | CAUSE official repo README results table (github.com/byungkwanlee/causal-unsupervised-segmentation); paper: Kim, Lee, Ro, "Causal Unsupervised Semantic Segmentation", arXiv 2310.07379 / Pattern Recognition 171 (2026). DINOv2 row is README-only (paper tables carry DINO rows). |
| P2 | 89.8 | pAcc | CAUSE-TR, DINOv2 ViT-B/14, Cityscapes | as P1 | as P1 |
| P3 | 28.0 / 90.8 | cluster mIoU / pAcc | CAUSE-TR, DINO ViT-B/8, Cityscapes | context row | as P1 |
| P4 | 23.1 | unsupervised (cluster) mIoU | DepthG, DINO ViT-B/8, Cityscapes | 27-class, Hungarian, CRF; Table 2 | Sick, Engel, Hermosilla, Ropinski, "Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling", CVPR 2024, arXiv 2309.12378v2, Table 2 |
| P5 | 81.6 | unsupervised Accuracy | DepthG, ViT-B/8, Cityscapes | as P4 | as P4 |
| P6 | 21.0 / 73.2 | U-mIoU / U-Acc | STEGO, ViT-B/8, Cityscapes | baseline row, same table | arXiv 2309.12378v2, Table 2 |
| P7 | 18.4 / 79.5 | U-mIoU / U-Acc | Hidden Positives (HP), ViT-B/8, Cityscapes | baseline row, same table | arXiv 2309.12378v2, Table 2 |

DepthG paper reports no Cityscapes linear-probe row in Table 2 — linear numbers below are
local diagnostics only.

## Local reproductions / anchors

| # | Value | Metric | System | Source |
|---|-------|--------|--------|--------|
| L1 | 29.9 / 89.8 | mIoU / pAcc | CAUSE-TR DINOv2 ViT-B/14 — prior local reproduction claim | `refs/cause/eval_cause_tr_dinov2.py` docstring ("Reproduces published results"). Re-run pending → R1. |
| L2 | 20.94 (cluster), 29.13 (linear) | mIoU | DepthG eval of ckpt `saved_models/cityscapes_vit_base_1.ckpt`, CRF on, 267 images, CPU (remote Linux box) | `refs/depthg/metrics.json` + `eval_cityscapes.log`. ⚠ Provenance: filename pattern suggests a *local retrain artifact* (`vit_base` run "1"), NOT the official release (`cityscapes_vitb.ckpt`). Treated as a retrain anchor, not an official reproduction. |
| L3 | 14.8 | cluster mIoU | DepthG mono-retrain ckpt `checkpoints/depthg_depthpro_monocular/epoch6_step1680.ckpt` | `mbps_pytorch/probe_depthg_depthpro_monocular.py` docstring (retrain report `reports/2026-06-03_1208_depthg_depthpro_retrain.md`) |
| L4 | — | official DepthG ckpt | `refs/depthg/saved_models/cityscapes_vitb.ckpt` downloaded 2026-06-12 from the paper's release Drive folder (id `1F_B1NLM0tfhWtuuHgWNq3Vu1x4-UfIjV`) | DepthG README "pretrained models" link. Local eval pending → R2. |

## Gates (from spec §2, instantiated)

| Gate | Condition |
|------|-----------|
| 0a | R1 (CAUSE re-run) within ~1.0 mIoU of P1=29.9; R2 (official DepthG ckpt local eval) within ~1.0 mIoU of P4=23.1. Larger gaps reconciled + documented here before proceeding. |
| 0b | GT-oracle headroom ≥ ~1.5 mIoU over the stronger vanilla model; disagreements not dominated by both-wrong. |
| 1A | best adapted CAUSE-TR cluster mIoU ≥ P1 + 1.0 = **30.9** (CAUSE protocol, CRF, frozen `cluster_tr` probe) |
| 1B | best adapted mono DepthG cluster mIoU > P4 = **23.1** (DepthG protocol, CRF, frozen probes). Secondary report: delta vs L3 = 14.8. Note: substrate starts 8.3 below the gate — this is intentionally a hard gate; the secondary delta documents partial progress. |

## Open reconciliation items

1. L2 (20.94) vs P4 (23.1): resolved if R2 ≈ 23.1 confirms L2 was a retrain artifact, not the official ckpt. Also note L2 evaluated 267 images — count what the val loader yields locally and document (torchvision Cityscapes val = 500 frames).
2. The mono-retrain report's "−7.5 cluster" delta was computed against whichever baseline that report used; this ledger's canonical deltas are vs P4 (published) and L3 (mono).

## Reproduction results (filled by Task 3 / Task 7)

| # | Value | System | Command/log |
|---|-------|--------|-------------|
| R1 | **29.8 mIoU / 89.8 Acc (CRF, 500 imgs)** — PASS vs P1=29.9 (Δ 0.1) | CAUSE-TR vanilla re-run, official script, MPS, 2026-06-12 | `logs/phase0a_cause_eval_20260612_180747.log` |
| R2 | pending | DepthG official ckpt (`cityscapes_vitb.ckpt`), local eval via fusion glue `--vanilla` | `logs/glue_check_B_official_*.log` |
| R3 | pending | DepthG mono ckpt via fusion glue `--vanilla` (expect ≈ L3) | `logs/glue_check_B_mono_*.log` |
| R4 | pending | CAUSE-TR vanilla via fusion glue `--vanilla` (expect ≈ R1) | `logs/glue_check_A_*.log` |
