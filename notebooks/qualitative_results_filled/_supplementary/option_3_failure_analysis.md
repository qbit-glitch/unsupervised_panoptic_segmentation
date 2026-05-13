# Option 3 — Failure mode analysis (Supplementary Figure)

This document accompanies `option_3_failure_modes.png`. We surveyed all 80
rendered MBPS predictions across Cityscapes, KITTI, Mapillary Vistas and
Waymo, computed per-image failure signals (number of thing instances,
mean instance size as % of image, void %, total connected components,
boundary-pixel %), then selected one image per failure mode for honest
diagnosis. The 2x2 figure pairs the input RGB (top) with the Ours panoptic
overlay (bottom) per cell.

Pseudo-class IDs: thing classes are over-cluster IDs in `[64, 79]`;
stuff classes are over-cluster IDs in `[0, 63]`; `255` denotes void
pixels (argmax abstention or unmapped clusters).

---

## Over-fragmentation

- **Dataset / Image**: `kitti` / `000057_10`
- **Signals**: things=88, thing classes=2, stuff classes=11, void=6.70%, biggest thing=0.46%, mean thing=0.01%, tiny things=86, boundary px=0.95%, total CCs=67
- **WHAT failed (visual)**: The overlay reports 88 thing instances spanning only 2 thing classes, with the median instance size at 3 pixels and 86 of 88 instances smaller than 0.05% of the image. One or two actual physical objects (a parked car and a distant vehicle) are shattered into dozens of single-pixel micro-instances scattered around their true silhouette.
- **WHY (likely mechanism)**: After per-class semantic argmax, the panoptic merger groups contiguous thing-class pixels into separate instances by connected-component analysis — every isolated pixel that happens to match a thing class becomes its own instance. The upstream Stage-1 pseudo-label generator (DepthPro Sobel threshold τ_d ≈ 0.20 + connected components + A_min=1000) was never asked to learn an instance ID assignment that the detector would inherit; instead, Cascade Mask R-CNN's mask head produces low-confidence soft masks for the dominant thing class and the merger crystallises every single-pixel hit into a separate instance because A_min is not enforced at merge time.
- **Mitigation**: Enforce A_min at the merger stage (drop instances < 200 px before assigning IDs), or run a one-shot instance-merge pass using DINOv3 cosine similarity on adjacent fragments — the SIMCF-B mechanism from §3.4, currently only invoked during pseudo-label generation.

## Under-detection

- **Dataset / Image**: `waymo` / `12831741023324393102_2673_230_2693_230_1508975625799296_cam5_image`
- **Signals**: things=0, thing classes=0, stuff classes=20, void=4.38%, biggest thing=0.00%, mean thing=0.00%, tiny things=0, boundary px=0.92%, total CCs=192
- **WHAT failed (visual)**: The input shows a residential street with several parked cars and visible traffic infrastructure, yet the overlay reports 0 thing instance(s) — the entire scene is collapsed into stuff classes (road, vegetation, building, sky) and every vehicle is silently absorbed.
- **WHY (likely mechanism)**: Two compounding factors: (1) Cascade Mask R-CNN's ROI head applies a confidence threshold to filter proposals; on OOD domains (Waymo's lower-camera / wider-FoV captures) the score distribution shifts down and most thing proposals fall below threshold. (2) The panoptic merger's thing/stuff arbitration then routes those pixels to whichever stuff class wins the semantic head's argmax — typically building or vegetation — producing a thing-free output. The semantic head was never trained to *not* claim those pixels as stuff.
- **Mitigation**: Either domain-aware proposal-score calibration on a held-out OOD shard, or replace the absolute confidence threshold with a top-K-per-image rule so at least K proposals always survive the merger.

## Class confusion (void / wrong class)

- **Dataset / Image**: `waymo` / `14081240615915270380_4399_000_4419_000_1518657497113004_cam5_image`
- **Signals**: things=1, thing classes=1, stuff classes=21, void=37.81%, biggest thing=8.68%, mean thing=8.68%, tiny things=0, boundary px=1.35%, total CCs=356
- **WHAT failed (visual)**: 37.8% of pixels are tagged void (black holes in the overlay), concentrated on the ego-vehicle bonnet at the bottom of the frame, and several visible scene regions carry semantically implausible class colours — building wall painted with the 'rider' palette and traffic-light heads mapped to a sidewalk-coloured pseudo-class.
- **WHY (likely mechanism)**: Two compounding faults: (1) The ego-vehicle bonnet is a texture/colour pattern absent from Cityscapes (Cityscapes crops the ego car out), so its DINOv3 features land in pseudo-clusters whose Hungarian assignment to any of the 27 benchmark classes is unstable; the evaluation script then maps those clusters to void. (2) Hungarian 1-to-1 remapping is fit globally on training-set per-pixel co-occurrence and can lock onto a spurious benchmark class when a pseudo-cluster is dataset-biased, producing the implausible colour swaps in the rest of the image.
- **Mitigation**: Mask the ego-vehicle region per dataset before evaluation; replace 1-to-1 Hungarian matching with a many-to-1 assignment that allows several pseudo-clusters per benchmark class; and reject low-confidence cluster mappings at inference instead of routing them to void.

## Boundary noise

- **Dataset / Image**: `waymo` / `8956556778987472864_3404_790_3424_790_1513450825606540_cam5_image`
- **Signals**: things=0, thing classes=0, stuff classes=14, void=5.05%, biggest thing=0.00%, mean thing=0.00%, tiny things=0, boundary px=1.78%, total CCs=977
- **WHAT failed (visual)**: Stuff-region edges are jagged and pixelated, with 977 total connected components in the semantic map and boundary pixels making up 1.8% of adjacent pixel pairs. Large stuff classes (vegetation, building, road) appear peppered with isolated micro-regions of a different class — evidence that the upsampled per-pixel argmax flips frequently near boundaries.
- **WHY (likely mechanism)**: The semantic head predicts at strided feature resolution (1/14 for DINOv3 ViT-B/16) then upsamples bilinearly to image resolution. Per-pixel argmax over a soft probability map amplifies tiny logit margins into hard label flips. The panoptic merger's STUFF_AREA_LIMIT (default 4096 px) keeps small stuff fragments whenever a single pixel exceeds the probability threshold, so border noise survives as separate connected components. There is no CRF or bilateral smoothing stage in the current inference pipeline to absorb these flips.
- **Mitigation**: Either add a lightweight DenseCRF post-process conditioned on the input RGB, or raise STUFF_AREA_LIMIT to drop sub-200-pixel regions and reassign them to the dominant 4-connected neighbour.


---

## Survey methodology (reproducibility)

`build_option_3.py` (this directory) loads every `ours_raw_panoptic.npz`
and computes:

- `void_pct`: percentage of pixels with `semantic_id == 255`
- `num_thing_instances`: count of unique `instance_id > 0`
- `num_thing_classes` / `num_stuff_classes`: distinct thing / stuff cluster
  IDs present (using `thing_classes` / `stuff_classes` from the npz)
- `biggest_thing_pct`, `mean_thing_pct`, `median_thing_pct`: thing instance
  sizes as percentage of image area
- `tiny_things_count`: thing instances smaller than 0.05% of image area
- `boundary_pixel_pct`: percentage of 4-connected pixel pairs whose
  semantic IDs disagree (proxy for contour density)
- `cc_total`: total 4-connected components across every non-void semantic
  class (proxy for over-segmentation)

Picking heuristics:

- **Over-fragmentation** = argmax of `num_things / (1 + mean_thing_pct)`
- **Under-detection** = argmax of `num_stuff_classes - 5 * num_things`
- **Class confusion** = argmax of `void_pct` among scenes with `>= 3` stuff
  classes (high void in a content-rich frame indicates clusters that the
  Hungarian assignment failed to attach to a benchmark class)
- **Boundary noise** = argmax of `cc_total * boundary_pixel_pct`

Excluded-IDs guarding ensures the four picked cases are distinct.
