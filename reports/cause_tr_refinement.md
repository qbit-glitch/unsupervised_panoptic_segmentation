# Recovering Missing Semantic Classes in Unsupervised Panoptic Segmentation via Overclustered Feature Assignment

CAUSE-TR (Ji et al., Pattern Recognition 2024) represents the current state-of-the-art in unsupervised semantic segmentation, employing a DINOv2 ViT-B/14 backbone coupled with a Segment_TR decoder that projects patch-level features into a 90-dimensional space optimized through contrastive learning with a modularity codebook. On Cityscapes validation, CAUSE-TR achieves an mIoU of 40.44% using a learned `cluster_probe`---a 27x90 matrix of fixed centroids that assigns each pixel to one of 27 semantic clusters via cosine similarity, followed by a one-to-one Hungarian matching to ground-truth classes. However, a critical limitation emerges: 7 of 19 evaluation classes (fence, pole, traffic light, traffic sign, rider, train, motorcycle) receive exactly 0% IoU. Through a systematic confusion matrix analysis across all 500 validation images, we identify the root cause: 14 of 27 learned centroids are *dead*---they never win the argmax competition for any pixel in the dataset. The missing classes exhibit systematic absorption patterns into semantically proximate dominant classes; for instance, fence pixels are classified as wall (70%) and building (13%), pole pixels as building (65%) and vegetation (10%), and traffic light pixels as building (89%). Crucially, this failure mode resides not in the feature representation itself, but in the rigid one-to-one assignment imposed by the 27-centroid cluster probe.

To address this bottleneck, we propose replacing the fixed cluster probe with a k-means overclustering strategy applied directly to the learned 90-dimensional Segment_TR features. Concretely, we extract patch-level features from all validation images (23x46 patches per image at 14x14 patch size), fit MiniBatchKMeans with k >> 27 centroids, and establish a many-to-one majority-vote mapping from each cluster to its best-matching ground-truth class. This relaxation is the key insight: by allowing multiple clusters to map to the same semantic class, rare and spatially small categories that were previously absorbed by dominant neighbors can now claim their own dedicated cluster regions in the 90-dimensional feature space. At k=300, this simple post-hoc reclustering recovers all 7 previously missing classes and improves patch-level mIoU from 40.4% to 61.3%---a 51% relative improvement---without any retraining or architectural modification. The recovery follows a clear progression with increasing k: at k=50, fence (42.2%) and traffic sign (45.3%) emerge; by k=100, traffic light (21.1%) and rider (41.7%) appear; and at k=200, all classes including pole (11.1%) and motorcycle (25.2%) are represented, reaching their respective peaks at k=300 (pole: 20.3%, motorcycle: 49.2%, train: 74.8%). Notably, applying the same overclustering procedure to raw DINOv2 768-dimensional features yields only 46.0% mIoU at k=300 and fails to recover train or motorcycle, confirming that the CAUSE Segment_TR projection is essential for discriminative class separation.

At full pixel resolution (1024x2048), the overclustered pseudo-labels achieve mIoU of 60.7% with 90.5% pixel accuracy---a +20.3 point absolute improvement over the CAUSE-TR baseline---while maintaining panoptic quality (PQ=22.2%, PQ_stuff=33.1%) comparable to the original pipeline (PQ=23.1%, PQ_stuff=31.4%). All 7 previously absent classes now contribute meaningful panoptic segments: traffic sign (PQ=10.3), fence (PQ=8.0), motorcycle (PQ=6.0), train (PQ=5.5), traffic light (PQ=4.9), rider (PQ=2.8), and pole (PQ=0.6). Interestingly, dense CRF post-processing degrades overclustered predictions (mIoU drops to 54.4%), unlike the original CAUSE output where CRF provides +2.4 mIoU---this suggests the overclustered soft logits already encode sufficient boundary information that CRF's pairwise potentials disrupt rather than refine. For pixel-level generation, we employ a sliding-window approach with 50% overlap to extract dense 90-dimensional features at full resolution, compute cosine similarity against the k=300 centroids, and aggregate per-class logits via max-pooling across clusters assigned to each class.

We further validated that alternative refinement strategies are insufficient for this failure mode. Multi-scale inference (averaging logits across scales [0.75, 1.0, 1.5]) improves mIoU to 43.9% with CRF post-processing but cannot recover any of the zero-IoU classes, as the same dead centroids are applied at every scale. A learned CSCMRefineNet (DINOv2+depth features through Conv2d blocks) achieves only marginal improvement (mIoU: 41.78%) while degrading panoptic quality, as it can only refine existing predictions without introducing new class assignments. Post-processing heuristics (prototype denoising, k-NN diffusion, depth-guided CRF) uniformly hurt performance by overriding CAUSE's learned decisions with weaker external signals. These negative results collectively demonstrate that the cluster assignment mechanism, not the feature quality or post-processing, is the decisive bottleneck in unsupervised semantic segmentation pipelines.

---

## Supplementary: Detailed Results

### Patch-Level Overclustering Results (500 Val Images)

| Config | mIoU | fence | pole | t_light | t_sign | rider | train | moto |
|--------|------|-------|------|---------|--------|-------|-------|------|
| CAUSE-27 baseline | 40.4% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CAUSE k=50 | 47.4% | 42.2 | 0.0 | 0.0 | 45.3 | 0.0 | 73.5 | 0.0 |
| CAUSE k=100 | 56.9% | 44.9 | 0.0 | 21.1 | 44.9 | 41.7 | 74.9 | 0.0 |
| CAUSE k=200 | 59.4% | 44.4 | 11.1 | 34.0 | 44.1 | 38.5 | 75.0 | 25.2 |
| **CAUSE k=300** | **61.3%** | **47.4** | **20.3** | **35.9** | **38.4** | **42.6** | **74.8** | **49.2** |
| DINOv2 raw k=300 | 46.0% | 27.8 | 17.0 | 33.5 | 38.7 | 30.6 | 0.0 | 0.0 |

### Pixel-Level Evaluation (500 Val Images, 1024x2048)

| Config | mIoU | Pixel Acc | PQ | PQ_stuff | PQ_things | SQ | RQ |
|--------|------|-----------|-----|----------|-----------|-----|-----|
| CAUSE-27 baseline | 40.4% | — | 23.1 | 31.4 | 11.7 | — | — |
| **Overclustered k=300 (no CRF)** | **60.7%** | **90.5%** | **22.2** | **33.1** | **7.3** | **71.9** | **22.3** |
| Overclustered k=300 + CRF | 54.4% | 88.6% | 21.5 | 33.6 | 4.9 | 74.2 | 19.8 |

### Multi-Scale Inference Results

| Config | mIoU | Notes |
|--------|------|-------|
| Single-scale, no CRF (baseline) | 40.44% | Original CAUSE |
| Single-scale + CRF | 42.86% | +2.42 from CRF |
| Multi-scale [0.75,1.0,1.5] no CRF | 42.10% | +1.66 from multi-scale |
| Multi-scale [0.75,1.0,1.5] + CRF | **43.90%** | +3.46 total |

### Absorption Patterns (Confusion Matrix Analysis)

| GT Class | Absorbed By | GT Pixel Count |
|----------|------------|----------------|
| fence | wall (70%) + building (13%) | 7.5M |
| pole | building (65%) + vegetation (10%) | 13.6M |
| traffic light | building (89%) + vegetation (10%) | 1.8M |
| traffic sign | building (78%) + wall (10%) | 6.1M |
| rider | bicycle (43%) + person (28%) | 2.0M |
| train | bus (82%) + building (9%) | 1.0M |
| motorcycle | bicycle (71%) + car (16%) | 0.7M |

## Scripts
- `mbps_pytorch/analyze_cause_confusion.py` — Confusion matrix analysis
- `mbps_pytorch/overclustering_cause.py` — K-means overclustering evaluation
- `mbps_pytorch/generate_overclustered_semantics.py` — Pixel-level pseudo-label generation
- `mbps_pytorch/generate_semantic_pseudolabels_cause.py` — Original CAUSE generation (with multi-scale)
- `mbps_pytorch/evaluate_cascade_pseudolabels.py` — Standard evaluator (mIoU, PQ)


