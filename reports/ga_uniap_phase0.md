# GA-UniAP Phase 0 — Kill-Gate Results

Grid 32x64, thresholds (0.96, 0.94, 0.92, 0.9), min_size 4, n_images 120.
Backbone DINOv3 ViT-B/16. Oracle GT-majority labeling (diagnostic upper bound; isolates grouping, not labeling).

| Variant | PQ | PQ_things | PQ_stuff | SQ | RQ |
|---|---|---|---|---|---|
| V0_vanilla | 27.49 | 7.37 | 42.13 | 70.74 | 27.84 |
| V1_augment | 16.02 | 0.35 | 27.42 | 68.09 | 17.17 |
| V2_split | 24.72 | 6.43 | 38.03 | 68.67 | 25.22 |
| V3_geom_only | 8.96 | 0.10 | 15.41 | 66.39 | 10.03 |

**Best geometric Δ PQ_things vs V0 = -0.94 (V2_split 6.43 vs V0 7.37). FAIL — geometry does not beat appearance inside the pooling; stop.**

## V1 weight sweep (w_f=1)

| w_n | w_h | PQ | PQ_things |
|---|---|---|---|
| 0.3 | 0.3 | 16.18 | 0.56 |
| 0.3 | 0.5 | 14.99 | 0.51 |
| 0.3 | 0.7 | 14.59 | 0.19 |
| 0.5 | 0.3 | 16.02 | 0.35 |
| 0.5 | 0.5 | 14.69 | 0.28 |
| 0.5 | 0.7 | 13.97 | 0.21 |
| 0.7 | 0.3 | 16.14 | 0.38 |
| 0.7 | 0.5 | 14.61 | 0.26 |
| 0.7 | 0.7 | 13.84 | 0.22 |

Best sweep PQ_things = 0.56 (vs V0 7.37).

## Interpretation

The result is decisive and monotonic: any geometry weight degrades grouping, and
PQ_things collapses to ~0 for the geometry-heavy variants. The cause is structural —
**surface normals and height-above-ground are smooth *within* an object but also
*across* adjacent co-planar same-class neighbours** (two pedestrians on the same
pavement, parked cars in a row). So geometry acts as a *merge* cue, not a *split* cue:
it pulls together exactly the instances appearance would keep apart, destroying
PQ_things. DINOv3 appearance alone is the stronger grouping signal at this setting.

This is consistent with prior MBPS probes (depth/3D-normals cannot separate same-depth
adjacent objects; only motion does). The kill-gate did its job: ~1 day, not three weeks.

**Scope / what this does NOT claim.** This refutes geometric affinity as a *node-merge
criterion inside UniAP-style agglomerative pooling* (32×64 grid, oracle-labeled, DINOv3
ViT-B/16). It does not refute depth's value elsewhere in MBPS (the DepthG semantic
pipeline, or depth as a post-hoc instance *splitter*). The single-stage *framing* could
still be adopted with appearance-only UniAP — but that is S2-UniSeg with no added novelty.

**Decision: do NOT proceed to Phase 1.** The geometric-pooling direction is closed.