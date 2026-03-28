# On the Granularity of Pseudo-Class Taxonomies in Unsupervised Panoptic Segmentation: Why Fewer Active Classes Bottleneck Stage-2 Learning

## 1. Introduction

Unsupervised panoptic segmentation pipelines typically operate in two stages: (1) generating pseudo-labels from self-supervised signals, and (2) training a panoptic segmentation network on those pseudo-labels to produce refined predictions. A critical but underexplored design choice in this pipeline is the *number of pseudo-classes* used to represent the semantic taxonomy. CUPS (Hahn et al., CVPR 2025) demonstrates that increasing the pseudo-class count from 27 to 54 improves panoptic quality from PQ=27.8 to PQ=30.6---a +2.8 point gain with no architectural or training changes (Table 7b in [1]). This observation suggests that pseudo-class granularity is a first-order factor in unsupervised panoptic segmentation quality.

In this work, we conduct a systematic analysis of pseudo-class granularity in our monocular unsupervised panoptic pipeline, which achieves PQ=25.6 at the pseudo-label level using overclustered CAUSE-TR semantics (k=300 k-means on 90-dimensional Segment_TR features, mIoU=60.7%) with connected-component instance segmentation. We discover that our current pipeline collapses 300 discriminative overclusters into only **19 active pseudo-classes** out of 27 Cityscapes label IDs---8 classes receive zero training pixels across all 2,975 training images. This collapse discards precisely the fine-grained semantic granularity that CUPS's ablation study identifies as beneficial, and simultaneously prevents connected-component instance segmentation from separating same-class objects. We propose maintaining 50--100 raw overclusters as pseudo-classes for Stage-2 training, which simultaneously (i) eliminates dead classes, (ii) provides free instance separation for touching same-class objects, and (iii) aligns with CUPS's empirically optimal pseudo-class count.

## 2. Background: How CUPS Uses Pseudo-Classes

CUPS generates 27 *unsupervised* pseudo-classes via stochastic cosine-distance k-means applied to DINO features, enhanced by depth-guided inference at two resolutions. Crucially, these 27 classes are not mapped to predefined Cityscapes category names---they are arbitrary cluster identities whose correspondence to ground-truth categories is established only at evaluation time via Hungarian matching [1, Sec. 4]. The thing-stuff split is determined automatically by computing each pseudo-class's frequency inside motion-segmented instance masks; classes exceeding a threshold $\psi^{ts}=0.08$ are designated "thing" classes.

Table 7b of [1] ablates the number of pseudo-classes:

| Pseudo-classes | PQ | SQ | RQ |
|---|---|---|---|
| 27 (default) | 27.8 | 57.4 | 35.2 |
| 40 | 30.3 | 64.3 | 37.5 |
| 54 | 30.6 | 65.1 | 37.8 |

PQ improves monotonically with pseudo-class count. The SQ improvement (+7.7 points from 27 to 54) indicates that finer pseudo-class boundaries produce tighter segment matches. The RQ improvement (+2.6 points) indicates that more proposals achieve non-zero overlap with ground-truth segments---consistent with the hypothesis that finer semantic granularity helps the detector discover distinct object instances.

## 3. Analysis: Our Pipeline Has Only 19 Active Pseudo-Classes

Our pseudo-label generation pipeline proceeds as follows: (1) extract 90-dimensional CAUSE-TR features from DINOv2 ViT-B/14 at full resolution via sliding-window inference; (2) assign each pixel to one of k=300 k-means centroids via cosine similarity; (3) map each of the 300 clusters to one of 27 Cityscapes label IDs via majority-vote against ground-truth categories; (4) save the mapped 27-class label as a uint8 PNG. This pipeline achieves 60.7% mIoU and 90.5% pixel accuracy at 1024x2048 resolution.

However, Step 3 introduces a critical information bottleneck. We conducted a complete census across all 2,975 training pseudo-labels (`cups_pseudo_labels_v3/`) and found:

| Class ID | Name | Images Present | Pixel Share |
|---|---|---|---|
| 0 | road | 2975/2975 (100%) | 39.56% |
| 1 | sidewalk | 2961/2975 (99.5%) | 6.12% |
| 4 | building | 2956/2975 (99.4%) | 22.77% |
| 14 | vegetation | 2935/2975 (98.7%) | 14.86% |
| 19 | car | 2849/2975 (95.8%) | 6.64% |
| 16 | sky | 2806/2975 (94.3%) | 3.76% |
| 10 | pole | 2777/2975 (93.3%) | 0.71% |
| 17 | person | 2189/2975 (73.6%) | 1.20% |
| 13 | traffic sign | 2040/2975 (68.6%) | 0.33% |
| 6 | fence | 1919/2975 (64.5%) | 0.86% |
| 15 | terrain | 1767/2975 (59.4%) | 1.04% |
| 5 | wall | 1494/2975 (50.2%) | 0.65% |
| 26 | bicycle | 1394/2975 (46.9%) | 0.41% |
| 12 | traffic light | 1337/2975 (44.9%) | 0.23% |
| 18 | rider | 628/2975 (21.1%) | 0.11% |
| 25 | motorcycle | 483/2975 (16.2%) | 0.11% |
| 20 | truck | 386/2975 (13.0%) | 0.25% |
| 21 | bus | 226/2975 (7.6%) | 0.20% |
| 24 | train | 114/2975 (3.8%) | 0.20% |

**8 classes are entirely absent** (zero pixels in all 2,975 images):

| Class ID | Name | Reason for Absence |
|---|---|---|
| 2 | parking | Rare Cityscapes category, absorbed by road/sidewalk in overclustering |
| 3 | rail track | Extremely rare (4 images in GT), no dedicated cluster at k=300 |
| 7 | guard rail | Absorbed by fence/wall |
| 8 | bridge | Absorbed by building |
| 9 | tunnel | Absorbed by building/road |
| 11 | polegroup | Absorbed by pole |
| 22 | caravan | Absorbed by car/truck (< 10 GT instances) |
| 23 | trailer | Absorbed by truck (< 5 GT instances) |

The Cascade Mask R-CNN's semantic segmentation head allocates 27 output channels, but 8 receive zero gradient throughout training. Effectively, we train a 27-class model with 19-class supervision. For comparison, CUPS's 27 unsupervised clusters are all active by construction---k-means guarantees every centroid has assigned pixels.

## 4. The Instance Separation Problem

The pseudo-class collapse has a second, more consequential effect on instance segmentation quality. Our instance pipeline uses connected-component labeling on semantic pseudo-label regions: each contiguous region of the same class ID becomes one instance. With only 19 classes, adjacent or touching objects of the same semantic category (e.g., two parked cars, a cluster of pedestrians) are merged into a single connected component.

Consider a concrete example: a Cityscapes image with 5 parked cars in a row. Under the 19-class label map, all 5 receive class ID 19 ("car"). Since their pixels form one connected blob, connected-component labeling produces a single instance encompassing all 5 vehicles. The Stage-2 detector trains on this merged instance, learning to predict a single large bounding box covering the entire car group---directly degrading PQ_things.

With 50+ overclusters, the same 5 cars would likely receive *different* cluster IDs (reflecting differences in color, lighting, depth, or position in the image). Each car becomes a distinct connected component, yielding 5 separate instance masks---without any explicit instance segmentation algorithm. This is **free instance separation** from finer semantic granularity.

The overclustering analysis from our prior work (see `cause_tr_refinement.md`) confirms that k-means overclustering on CAUSE-TR features produces clusters with strong intra-cluster visual coherence. At k=300, individual object instances frequently occupy their own clusters: a red car, a white car, and a silver car in the same scene receive three distinct cluster IDs. This property arises because DINO-family features encode both semantic category and instance-level appearance, and k-means with large k partitions this joint space finely enough to distinguish individual objects.

## 5. Proposed Approach: Raw Overclusters as Pseudo-Classes

We propose modifying the pseudo-label generation pipeline to preserve raw overcluster IDs rather than collapsing to named Cityscapes categories:

**Current pipeline (collapsed):**
```
CAUSE-TR features → k-means (k=300) → cluster ID 0-299 → majority-vote map to 27 classes → save class ID 0-26
```

**Proposed pipeline (raw):**
```
CAUSE-TR features → k-means (k=50) → cluster ID 0-49 → save cluster ID 0-49 directly
```

The key differences are: (i) no majority-vote mapping, preserving the full discriminative granularity of the overclustering; (ii) all k pseudo-classes are active by construction; (iii) the thing-stuff split is determined automatically by CUPS's `PseudoLabelDataset`, which computes each pseudo-class's frequency inside instance masks and applies a threshold.

### 5.1 Choice of k

Our prior overclustering analysis measured mIoU at varying k (after majority-vote mapping to 19 evaluation classes):

| k | Patch-level mIoU | Notes |
|---|---|---|
| 27 (baseline) | 40.4% | Original CAUSE-TR; 7 classes at 0% IoU |
| 50 | 47.4% | fence and traffic sign recovered |
| 100 | 56.9% | traffic light and rider recovered |
| 200 | 59.4% | pole and motorcycle recovered |
| 300 | 61.3% | All 19 eval classes represented |

For Stage-2 pseudo-class count, however, the relevant criterion is not mIoU but whether the detector can learn effectively from each class. CUPS's ablation (Table 7b) shows diminishing returns beyond 54 classes. We recommend:

- **k=50 (primary):** Closest to CUPS's empirically optimal count of 54. Each cluster averages ~2% of total pixels, providing dense supervision. Expected to recover the +2.8 PQ that CUPS observes from 27→54.
- **k=100 (secondary):** Finer instance separation, but sparser per-class supervision. May benefit PQ_things at the cost of PQ_stuff stability.
- **k=300 (aggressive):** Maximum granularity. Risk of many very small clusters (~0.33% pixels each) that provide insufficient gradient signal for the Cascade Mask R-CNN's class-specific heads.

### 5.2 Impact on Instance Quality

To quantify the expected instance separation improvement, we analyze the connected-component statistics under different pseudo-class granularities. With 19 active classes, our current pseudo-labels produce an average of 4.3 thing instances per image via connected-component labeling (as reported in `overclustered_spidepth_sweep.md` at the CC-only operating point). Under k=50 overclusters, the same connected-component procedure would produce more instances because:

1. Single-class regions containing multiple objects split into per-cluster regions.
2. Cluster boundaries within object interiors are less harmful than class-level under-segmentation, because the Cascade Mask R-CNN's mask head can merge fragments that the box proposal covers.
3. DropLoss (Eq. 4 in [1]) tolerates "thing" region proposals that don't overlap any pseudo-instance mask, allowing the network to discover objects beyond what connected components provide.

### 5.3 Implementation Requirements

The modifications are confined to two scripts:

1. **`mbps_pytorch/generate_overclustered_semantics.py`**: Add a `--raw_clusters` flag that saves cluster IDs (0 to k-1) directly instead of majority-vote mapped class IDs. The k-means centroids and mapping table are saved alongside for evaluation.

2. **`mbps_pytorch/convert_to_cups_format.py`**: Accept `--num_classes` parameter (default 27 → now 50 or 100). The `.pt` distribution files must have dimensionality matching the new class count. The thing-stuff split threshold remains at 0.08 (CUPS default).

The CUPS training code requires no modification: `PseudoLabelDataset` infers the class count from the `.pt` distribution files (line 131--142 in `pseudo_label_dataset.py`), and `build_model_pseudo` (line 505--506) sets the Cascade Mask R-CNN head sizes accordingly. The `PanopticQualitySemanticMatching` metric performs Hungarian matching between predicted pseudo-class IDs and ground-truth classes, which is agnostic to the number of pseudo-classes.

## 6. Stage-2 Training: Why Our Current Pipeline Degrades PQ

Our Stage-2 training with the spatial alignment fix (v4) achieves PQ=22.5 at step 4000/8000, below the input pseudo-label PQ of 25.6. In contrast, CUPS's Stage-2 improves PQ from 18.1 (pseudo-labels) to 27.8 (final model)---a +9.7 gain. We attribute our degradation to three interacting factors:

1. **Dead pseudo-classes (this work):** 8/27 output channels receive zero gradient, wasting model capacity and potentially introducing noise through random initialization of unused heads.

2. **Poor instance supervision:** Connected components on 19-class maps merge touching same-class objects, teaching the detector that multi-object groups are single instances. The Cascade Mask R-CNN learns to predict large, imprecise bounding boxes.

3. **Plateauing at step 4000:** Our training metrics show oscillation between PQ=19.5 and PQ=22.5 after step 4000, suggesting the model has extracted all learnable signal from the current pseudo-labels. Finer pseudo-class granularity would provide additional structure for the detector to learn from.

With k=50 raw overclusters, all three problems are addressed simultaneously: no dead classes, better instance separation, and richer per-pixel supervision signal.

## 7. Projected Impact

| Configuration | Pseudo-classes (active) | Instance Method | Expected PQ | Basis |
|---|---|---|---|---|
| Current (v4, 19-class) | 19 / 27 | CC on 19-class map | 22.5 (step 4000) | Measured |
| Raw k=50 overclusters | 50 / 50 | CC on 50-class map | 27--29 | CUPS 27→54 scaling (+2.8 PQ) + better instances |
| Raw k=50 + self-training | 50 / 50 | CC + EMA self-labels | 29--32 | CUPS Stage 3 adds +1.5--2.0 PQ |
| Raw k=100 overclusters | 100 / 100 | CC on 100-class map | 28--30 | Finer instances, diminishing semantic returns |

These projections assume that Stage-2 training on properly granular pseudo-labels achieves gains comparable to CUPS's observed +9.7 PQ improvement. Our pseudo-labels start at a higher baseline (PQ=25.6 vs. CUPS's 18.1), so the absolute improvement may be smaller, but the final PQ should be competitive with or exceed CUPS's 27.8.

## 8. Comparison: Our Approach vs. CUPS

| Dimension | Ours (Current) | Ours (Proposed k=50) | CUPS |
|---|---|---|---|
| Input data | Monocular images | Monocular images | Stereo video |
| Semantic method | CAUSE-TR (ViT-B/14) | CAUSE-TR (ViT-B/14) | DepthG (DINO-distilled) |
| Semantic mIoU | 60.7% | ~47--50% (raw clusters, mapped at eval) | 26.8% |
| Pseudo-classes | 19 active / 27 | 50 active / 50 | 27 active / 27 |
| Instance method | CC on 19-class map | CC on 50-class map | SF2SE3 motion segmentation |
| Motion cues | None | None | Stereo optical flow |
| Pseudo-label PQ | 25.6 | ~27--29 (projected) | 18.1 |
| Stage-2 final PQ | 22.5 (degraded) | 29--32 (projected) | 27.8 |

The comparison reveals that our semantic representation is substantially stronger than CUPS's (60.7% vs. 26.8% mIoU), but this advantage is negated by the class collapse and weak instance segmentation. The proposed k=50 overclustering approach addresses both weaknesses without requiring stereo video, potentially enabling a monocular-only pipeline that matches or exceeds CUPS's stereo-dependent results.

## 9. Conclusion

We have identified that our current pseudo-label pipeline discards critical information by collapsing 300 overclusters into 19 active Cityscapes classes, directly contradicting CUPS's finding that more pseudo-classes improve panoptic quality. The proposed fix---preserving 50--100 raw overclusters as pseudo-classes for Stage-2 training---simultaneously eliminates dead classes, enables free instance separation via connected components on finer-grained semantic maps, and aligns with the empirically optimal pseudo-class granularity established by CUPS's ablation study. This represents the most promising path to closing the remaining gap between our monocular pipeline (PQ=22.5) and CUPS's stereo-based state of the art (PQ=27.8), with the potential to surpass it by leveraging our 2.3x stronger semantic representation.

---

## References

[1] O. Hahn, C. Reich, N. Araslanov, D. Cremers, C. Rupprecht, and S. Roth. "Scene-Centric Unsupervised Panoptic Segmentation." CVPR, 2025.

## Scripts

- `mbps_pytorch/generate_overclustered_semantics.py` --- Overclustered pseudo-label generation (to be modified for raw cluster output)
- `mbps_pytorch/convert_to_cups_format.py` --- CUPS format conversion (to be modified for variable class count)
- `mbps_pytorch/overclustering_cause.py` --- K-means overclustering evaluation at patch level
- `refs/cups/cups/data/pseudo_label_dataset.py` --- CUPS dataset (auto-detects class count from `.pt` files)
- `refs/cups/cups/pl_model_pseudo.py` --- CUPS model builder (auto-sizes heads from pseudo-class count)
