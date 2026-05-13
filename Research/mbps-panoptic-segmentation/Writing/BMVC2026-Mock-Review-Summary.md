---
type: writing
title: "BMVC 2026 — Mock Review Summary"
project: mbps-panoptic-segmentation
status: active
venue: BMVC 2026
tags: [paper-writing, bmvc-2026, mock-review, self-review]
updated: 2026-04-13
---

# Mock Review Summary

Three simulated brutal BMVC reviewers evaluated `main_backup_2026-04-13.tex` (post-narrative rewrite).

## Scores

| Reviewer | Expertise | Score | Verdict |
|----------|-----------|-------|---------|
| [[Writing/BMVC2026-Reviewer-1\|R1]] — Unsupervised Segmentation | High | **5/10** | Borderline Reject |
| [[Writing/BMVC2026-Reviewer-2\|R2]] — Foundation Models / SSL | High | **6/10** | Weak Accept |
| [[Writing/BMVC2026-Reviewer-3\|R3]] — Applied Panoptic | Medium-High | **5/10** | Borderline |
| **Mean** | | **5.3/10** | **Borderline** |

## Consensus Blocking Issues

| # | Issue | Severity | Agreed |
|---|-------|----------|--------|
| 1 | **DINOv3 confound unresolved** — no CUPS PLs on DINOv3 control | Critical | 3/3 |
| 2 | **Single seed, no variance** — Stage-2 margin (+0.07) is noise | Critical | 3/3 |
| 3 | **Page limit violation** — 11pp content vs BMVC 9pp limit | Desk-reject | 3/3 |
| 4 | **Broken "Section ??" reference** on page 6 | Must fix | 3/3 |
| 5 | **Thin novelty** — composing existing tools, no new algorithm | Major | 2/3 |
| 6 | **Self-training scaling unexplained** — best finding is under-analyzed | Major | 1/3 |

## What Flips Scores to Accept

1. **E1 control experiment** — CUPS PLs on DINOv3 backbone (already in progress, ETA April 18-19)
2. **3 seeds minimum** with mean ± std reported
3. **Cut to 9 pages** — move Fig 5 to supplementary, compress §3.5
4. **Fix broken "Section ??" reference** on page 6
5. **Add paragraph explaining self-training amplification mechanism**

## Strengths All Reviewers Agreed On

- Strong empirical result (+5.0 PQ, +16.4 PQ_th)
- Well-designed progressive ablation structure (Table 2)
- Honest limitations section
- Overclustering diagnosis (k=27 → 7 dead classes) is a real insight
- Depth quality as binding constraint (Table 2b) is useful for the community
- Writing quality is above average for BMVC

## Per-Reviewer Detail Notes

- [[Writing/BMVC2026-Reviewer-1]] — Novelty, DINOv3 confound, single seed
- [[Writing/BMVC2026-Reviewer-2]] — Self-training scaling, cross-dataset weakness, stuff regression
- [[Writing/BMVC2026-Reviewer-3]] — Page limit, practical concerns, narrow evaluation

## Action Priority

```
P0 (before submission):
  □ Fix "Section ??" broken reference (page 6)
  □ Cut paper to 9 pages (move Fig 5 to supplementary)
  □ E1 control experiment results

P1 (strongly recommended):
  □ Run seeds 123, 456 — report mean ± std
  □ Add 1 paragraph: WHY self-training scales super-linearly
  □ Add computational cost comparison vs CUPS

P2 (nice to have):
  □ Depth-split-ratio sensitivity analysis
  □ Inference latency breakdown
  □ Discuss path to recover 7 dead CAUSE classes
```

## Links

- [[Writing/BMVC2026-Paper-Overview]] — Paper structure and contributions
- [[Writing/BMVC2026-Reviewer-Response]] — E1 control experiment pipeline status
- [[Writing/BMVC2026-Experiments]] — Current ablation coverage
- [[Knowledge/Key-Lessons]] — Metric pitfalls to avoid
