⏺ Searched for 1 pattern, read 3 files (ctrl+o to expand)

⏺ Now I have the full per-class breakdown. Let me analyze the key bottleneck:

  Per-class PQ^Th with depth instances (best pipeline):

  ┌────────────┬──────┬─────┬──────┬──────┬───────────┬────────┐
  │   Class    │  PQ  │ TP  │  FP  │  FN  │ Precision │ Recall │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ car        │ 16.3 │ 900 │ 2365 │ 3735 │ 27.6%     │ 19.4%  │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ truck      │ 24.1 │ 25  │ 46   │ 68   │ 35.2%     │ 26.9%  │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ bus        │ 19.8 │ 30  │ 113  │ 68   │ 21.0%     │ 30.6%  │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ person     │ 6.7  │ 221 │ 713  │ 3155 │ 23.7%     │ 6.5%   │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ bicycle    │ 5.0  │ 72  │ 472  │ 1091 │ 13.2%     │ 6.2%   │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ rider      │ 0.0  │ 0   │ 0    │ 541  │ -         │ 0%     │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ train      │ 0.0  │ 0   │ 0    │ 23   │ -         │ 0%     │
  ├────────────┼──────┼─────┼──────┼──────┼───────────┼────────┤
  │ motorcycle │ 0.0  │ 0   │ 0    │ 149  │ -         │ 0%     │
  └────────────┴──────┴─────┴──────┴──────┴───────────┴────────┘

  Two clear problems:

  Problem A — Massive FP (over-fragmentation): Car has 2365 FP segments. Depth gradients split cars into many small fragments that individually fail the IoU>0.5 threshold. Person
  and bicycle are even worse.

  Problem B — 3 zero classes: rider/train/motorcycle have 0% semantic IoU in CAUSE-TR, so no instances can ever be generated.

  Here are 4 ideas ranked by impact/effort ratio:

  Idea 1: Connected components baseline (free — 1 minute)

  Run eval with --thing_mode connected_components instead of depth-splitting. If plain connected components on the semantic map scores better, then depth is actually hurting by
  over-fragmenting. This tells us whether to tune depth or abandon it.

  Idea 2: Tune depth instance parameters (quick — 10 min sweep)

  Current settings are arbitrary: grad_threshold=0.05, min_area=100. The FP analysis suggests over-segmentation. Increasing grad_threshold to 0.10-0.15 and min_area to 500-1000
  would reduce fragmentation. Each run takes ~30s.

  Idea 3: Instance merging post-processing (moderate — 1 hour)

  After depth splitting, merge small adjacent same-class fragments back together. This directly targets the FP problem: 2365 car FP are mostly tiny fragments from one car split
  into pieces.

  Idea 4: Improve semantic pseudo-labels (biggest lever — needs GPU)

  Train the DINOv3 ViT-L/16 linear probe (generate_semantic_pseudolabels_dinov3.py already exists). Would push mIoU from 42.9% to ~70%, fix zero classes, and fundamentally improve
  both PQ^St and PQ^Th. This is the only way to reach CUPS-level PQ.

  I'd recommend starting with Idea 1 (takes 30 seconds, purely diagnostic) to understand if depth splitting helps or hurts.

✻ Crunched for 2m 29s

❯ yes start with idea 1, run connected components baseline. note: dont proceed with idea 4 by doing linear probe with DINOv3 as I want this to be unsupervised. 

⏺ Let me quickly check how the connected_components mode works in the eval script, then run it.

⏺ Searched for 1 pattern (ctrl+o to expand)

⏺ Connected components mode only needs semantic labels — no instance dir needed. Let me run it.

⏺ Bash(source .venv/bin/activate && python mbps_pytorch/evaluate_cascade_pseudolabels.py --cityscapes_root /Users/qbit-glitch/Desktop/datasets/cityscapes --split val -…)
  ⎿  Cascade Pseudo-Label Evaluation
       Dataset: Cityscapes val
       Eval resolution: 512x1024
     … +127 lines (ctrl+o to expand)
  ⎿  (timeout 5m)

⏺ Very revealing results. Here's the full comparison:

  Per-class PQ^Th: Depth vs Connected Components

  ┌─────────┬───────┬──────┬──────────────┐
  │  Class  │ Depth │  CC  │    Winner    │
  ├─────────┼───────┼──────┼──────────────┤
  │ car     │ 16.3  │ 8.5  │ Depth (+7.8) │
  ├─────────┼───────┼──────┼──────────────┤
  │ truck   │ 24.1  │ 29.0 │ CC (+4.9)    │
  ├─────────┼───────┼──────┼──────────────┤
  │ bus     │ 19.8  │ 24.3 │ CC (+4.5)    │
  ├─────────┼───────┼──────┼──────────────┤
  │ person  │ 6.7   │ 4.4  │ Depth (+2.3) │
  ├─────────┼───────┼──────┼──────────────┤
  │ bicycle │ 5.0   │ 3.4  │ Depth (+1.6) │
  └─────────┴───────┴──────┴──────────────┘

  Summary table

  ┌───────────────────┬──────┬───────┬───────┐
  │     Pipeline      │  PQ  │ PQ^St │ PQ^Th │
  ├───────────────────┼──────┼───────┼───────┤
  │ Attn ST + depth   │ 21.8 │ 31.2  │ 9.0   │
  ├───────────────────┼──────┼───────┼───────┤
  │ Attn ST + CC      │ 21.7 │ 31.1  │ 8.7   │
  ├───────────────────┼──────┼───────┼───────┤
  │ Attn ST + MaskCut │ 18.4 │ 30.4  │ 1.9   │
  └───────────────────┴──────┴───────┴───────┘

  Key insight

  The optimal instance method depends on object size:
  - CC wins for large isolated things (truck +4.9, bus +4.5) — depth over-fragments them
  - Depth wins for small/clustered things (car +7.8, person +2.3, bicycle +1.6) — CC merges adjacent instances into one blob

  A hybrid approach (CC for truck/bus, depth for car/person/bicycle) would give PQ^Th ~10.2 — a free +1.2 improvement.

  But the bigger opportunity is depth parameter tuning. Car has 2365 FP with depth — most are tiny fragments from over-segmentation. Increasing grad_threshold (currently 0.05) and
  min_area (currently 100) could reduce this FP flood while keeping the splitting ability that makes depth better than CC for cars.

  Want me to proceed with Idea 2: depth parameter sweep (trying grad_threshold in [0.08, 0.10, 0.15] and min_area in [200, 500])?