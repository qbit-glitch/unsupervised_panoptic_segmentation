---
project_id: mbps-panoptic-segmentation
repo_root: /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
vault_root: /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/Research/mbps-panoptic-segmentation
hub_note: Research/mbps-panoptic-segmentation/00-Hub.md
language: en
last_sync_at: 2026-04-21T20:51:00Z
last_synced_head: 31c44751b20a76e9b6b140141d88a4ebaaac3e5d
status: active
auto_sync: true
---

# Project Memory: mbps-panoptic-segmentation

## Current Question
Can Seesaw Loss + Class-Aware Thresholds recover dead classes in Stage-3 panoptic segmentation, and which Stage-2 techniques should be transferred next?

## Hypotheses
- Seesaw Loss rebalancing (p=0.8, q=2.0) will improve rare-class PQ by up-weighting tail-class gradients
- Class-aware thresholds (α=0.3) will increase rare-class pseudo-label recall
- Stage-2 techniques (EMA, SWA, LSJ, Color Jitter, Dense CRF) can be ported to Stage-3 PanopticFPN

## Active Tasks
- Seesaw Loss ablation training running on santosh@172.17.254.146 (2× GTX 1080 Ti)
- Full-dataset CPU evaluation completed for step-400 checkpoint
- DDP validation sync issue identified (rank_zero_only=True ignores GPU1 scores)

## Open Experiments
- [[results/ablation_seesaw_step400.md]] — Step 400 evaluation results and dead-class analysis
- [[docs/seesaw_loss_finetuning_guide.md]] — Architectural changes and math formulations

## Recent Results
### Stage-3 Baseline (DCFA+SIMCF-ABC, step 2200)
- PQ: **36.41%**
- 6 dead classes (0% PQ): pole, guard rail, tunnel, polegroup, caravan, trailer

### Stage-4 Ablation (Seesaw Loss + Class-Aware Thresholds, step 400)
- PQ: **35.81%** (full 500-image validation, CPU)
- Dead-class recovery: **6 → 5** (pole recovered: 0% → 1.84%)
- Still dead: guard rail, tunnel, polegroup, caravan, trailer
- Training log peak: **39.12%** at step 2600 (GPU1, not checkpointed due to sync issue)

### Per-Class Breakdown (Step 400)
- Strong (PQ > 70%): road, vegetation, sky, car, truck, bus, train
- Moderate (PQ 20–70%): sidewalk, building, wall, traffic sign, terrain, rider, bicycle
- Weak (PQ < 20%): parking, rail track, fence, bridge, traffic light, person, pole, motorcycle
- Dead (PQ = 0%): guard rail, tunnel, polegroup, caravan, trailer

### GPU Discrepancy Discovery
- Validation PQ logged with `rank_zero_only=True, sync_dist=False`
- Checkpoint callback monitors only GPU 0's PQ (50% of val data)
- At step 2600: GPU0=37.00%, GPU1=39.12% → best checkpoint missed by 2.12 points

## Proposed Next Steps
1. Fix DDP PQ aggregation (`sync_dist=True` or custom gather)
2. Port Stage-2 techniques to Stage-3: EMA + LSJ + Color Jitter + Dense CRF + Longer Schedule
3. Continue training to convergence (>4k steps) to validate peak PQ

## Recent Sync Status
- 2026-04-21: Seesaw Loss step-400 evaluation completed, results note created, dead-class recovery documented.
