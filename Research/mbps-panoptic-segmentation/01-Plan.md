---
type: project
title: Plan - MBPS Panoptic Segmentation
project: mbps-panoptic-segmentation
language: en
status: active
updated: 2026-04-24T00:00:00Z
---

# Plan

## Active Goals
1. **COCO panoptic evaluation — remaining configs** — R4 (NAMR) and R6+R4 still pending.
2. **Cityscapes panoptic evaluation** — only baseline done (PQ=11.68%). Need R6, R4, R6+R4.
3. **Re-run MMGD-Cut semantic rounds with CUPS-standard protocol** — get corrected mIoU table (global Hungarian, comparable to published numbers).
4. **Investigate Falcon eval protocol gap** — understand why published Falcon reports 52.6% and whether they use the same protocol.
5. **Adapter strategy pivot (NeurIPS professor advice)** — Remove adapters from Stage 2/3; use DoRA only in Stage 1 pseudo-label generation.

## Active Tasks
- [x] MMGD-Cut Rounds 1–3: multi-modal fusion validated (DINOv3+SSD-1B best)
- [x] CUPS eval protocol identified and implemented
- [x] COCO panoptic: baseline + R6 complete (27.30% / 27.75% mIoU, CUPS-standard)
- [x] Cityscapes panoptic: baseline complete (PQ=11.68%)
- [x] Git worktrees merged to main
- [ ] COCO panoptic: R4 (NAMR) + R6+R4
- [ ] Cityscapes panoptic: R6, R4, R6+R4
- [ ] Re-run MMGD-Cut semantic (all rounds) with CUPS-standard mIoU
- [ ] Investigate Falcon published eval protocol (52.6% — is it per-image Hungarian?)
- [ ] Triple fusion: DINOv3+SD+SSD-1B
- [ ] NeCo fine-tuning (R7) on remote GPU
- [ ] **P0: Audit Stage 2/3 code for lingering adapter injection** — Remove `inject_lora_into_*` from any Stage 2/3 training configs/scripts
- [ ] **P0: Run semantic adapter training (Stage 1)** — DoRA on frozen DINOv2+CAUSE-TR, generate adapted k=54/k=80 pseudo-labels
- [ ] **P1: Run depth adapter training (Stage 1)** — DoRA on frozen DA2-Large / DA3 / DepthPro, generate adapted instance pseudo-labels
- [ ] **P1: Evaluate adapted pseudo-labels** — mIoU for semantic, PQ_things for instance, vs frozen baselines
- [ ] **P2: Feed adapted pseudo-labels into standard Stage 2/3** — Full backprop, no adapters; compare to frozen-PL baseline
- [x] **DONE: Deep architectural analysis with ASCII diagrams** — DINOv2+CAUSE-TR, DepthPro, DA2-Large, DA3 adapter architectures documented

## Milestones Achieved
| Date | Milestone | Key Result |
|------|-----------|------------|
| 2026-04-24 | NeurIPS advisor meeting | Adapter strategy pivot: DoRA only in Stage 1, never in Stage 2/3 |
| 2026-04-23 | Cross-dataset zero-shot eval | Mapillary PQ=39.19% (+3.36 over Cityscapes), KITTI 34.85%, COCO 7.83% |
| 2026-04-23 | Silent LoRA drop bug fix | `generate_semantic_pseudolabels_adapted.py` now correctly loads adapter weights |
| 2026-04-13 | CUPS eval protocol fixed | Global k-means+Hungarian+argmax; 27.75% mIoU (R6) |
| 2026-04-04 | COCO panoptic R6 | PQ=7.57%, mIoU=27.75% (CUPS-standard) |
| 2026-04-02 | MMGD-Cut Round 3 | DINOv3+SSD-1B fusion validated; graph diffusion harmful |
| 2026-03-29 | Instance ablation Phase A | Mumford-Shah PQ_th=23.27 (+19.9% over Sobel+CC) |
| 2026-03-28 | COCO semantic ablation | K-means k=1000 mIoU=33.2% dominates |
| 2026-03-28 | Depth model ablation | DA3 PQ_th=20.90 on Cityscapes |
| 2026-03-16 | DINOv3 Stage-3 | PQ=30.255% — beats CUPS 27.8% |

## Open Questions
- Does published Falcon (52.6%) use per-image Hungarian? If so, our corrected ~27% could be competitive on global Hungarian.
- What is CUPS Stage-1 mIoU on COCO-Stuff-27 with global Hungarian? (Needed for direct comparison.)
- Can NeCo fine-tuning improve DINOv3 features enough to close the gap?
- **NEW:** Will DoRA-adapted pseudo-labels in Stage 1 outperform frozen baselines when fed into standard Stage 2/3?

## Compute Resources
- **Local**: M4 Pro MacBook 48GB (MPS, pseudo-label generation)
- **Remote GPU**: `santosh@172.17.254.146`, 2x GTX 1080 Ti 11GB, conda `cups`
- **Anydesk**: RTX A6000 Pro 48GB, conda `ups`
- **GCP**: TPU v4/v5e VMs (TRC grant, 30 days)




To Prove: Logically Monocular is better than Stereo