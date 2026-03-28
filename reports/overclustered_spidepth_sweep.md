# Depth-Guided Instance Segmentation with Overclustered Semantic Pseudo-Labels: A Parameter Sensitivity Study

Unsupervised panoptic segmentation requires both semantic class assignment and instance-level object delineation, yet these two components interact in non-trivial ways. In this work, we investigate the interaction between overclustered CAUSE-TR semantic pseudo-labels---which recover all 19 Cityscapes classes at 60.7% mIoU by replacing the original 27-centroid cluster probe with k=300 k-means overclustering on the learned 90-dimensional Segment_TR features---and SPIdepth-based instance segmentation, which splits same-class connected regions at depth discontinuities detected via Sobel filtering on monocular depth estimates. Our central finding is that depth-guided splitting provides no benefit when applied to overclustered semantic boundaries: the best panoptic quality (PQ=25.6) is achieved by simple connected-component labeling without any depth-based splitting, and all depth-guided configurations strictly underperform this baseline.

We conduct a systematic parameter sweep over two key hyperparameters of the depth-guided splitting pipeline: the depth gradient threshold (controlling the sensitivity of edge detection, ranging from 0.05 to 1.0) and the minimum instance area (filtering spurious fragments, ranging from 100 to 2000 pixels at 512x1024 resolution). Across 14 configurations evaluated on the full Cityscapes validation set (500 images), we observe a monotonic relationship: reducing the number of depth-induced splits consistently improves PQ. At the default settings (gradient threshold 0.05, minimum area 100), the pipeline produces 11.2 instances per image but achieves only PQ=22.7 with PQ_things=8.5, as aggressive depth splitting fragments object regions into segments too small to match ground-truth instances. Progressively increasing the gradient threshold to 0.50 (min_area=1000) reduces fragmentation to 4.3 instances per image and raises PQ to 25.0 (PQ_things=14.0). At threshold 0.80 and above, the depth gradient detector fires on so few pixels that the output converges to pure connected-component instances, reaching PQ=25.5--25.6 with PQ_things=15.0--15.2 regardless of the minimum area setting. The connected-component baseline without any instance segmentation module achieves PQ=25.6 (PQ_things=15.2), confirming that depth-guided splitting is entirely redundant for this semantic representation.

The root cause of this negative result lies in the spatial resolution mismatch between the overclustered semantic boundaries and the depth discontinuity map. The overclustered pseudo-labels are derived from patch-level features at 14x14 pixel resolution (ViT-B/14), producing semantic boundaries that are inherently quantized to a coarse grid. Depth edges, computed at the full 512x1024 working resolution, do not align with these patch-level boundaries and instead introduce cuts within semantically coherent regions, creating false instance splits. This contrasts with the original CAUSE-TR semantic labels, where depth-guided splitting provided marginal benefit (PQ_things=11.7 vs. 15.2 for CC-only) because the 27-centroid assignment produced sharper but less complete boundaries. These results indicate that the primary bottleneck for improving panoptic quality beyond PQ=25.6 now resides in the instance segmentation component: a method capable of producing instance-level object boundaries at higher spatial fidelity than connected components---such as learned mask prediction or slot attention---is required to close the remaining gap to the CUPS (CVPR 2025) baseline of PQ=27.8.

---

## Detailed Results

### Parameter Sweep: Depth-Guided Splitting with Overclustered Semantics (k=300, no CRF)

| grad_thresh | min_area | inst/img | PQ | PQ_stuff | PQ_things | AR@100 |
|------------|----------|----------|-----|----------|-----------|--------|
| 0.05 | 100 | 11.2 | 22.7 | 33.1 | 8.5 | 18.9% |
| 0.08 | 200 | 8.9 | 23.1 | 33.1 | 9.5 | 18.2% |
| 0.08 | 500 | 6.0 | 23.5 | 33.1 | 10.3 | 17.0% |
| 0.12 | 200 | 8.8 | 23.4 | 33.1 | 10.0 | 17.1% |
| 0.12 | 500 | 6.1 | 23.6 | 33.1 | 10.6 | 16.0% |
| 0.20 | 200 | 8.4 | 23.7 | 33.1 | 10.9 | 15.1% |
| 0.20 | 500 | 6.0 | 23.9 | 33.1 | 11.3 | 14.6% |
| 0.30 | 500 | 5.7 | 24.3 | 33.1 | 12.3 | 13.6% |
| 0.30 | 1000 | 4.4 | 24.5 | 33.1 | 12.6 | 12.8% |
| 0.50 | 500 | 5.6 | 25.0 | 33.1 | 13.9 | 12.7% |
| 0.50 | 1000 | 4.3 | 25.0 | 33.1 | 14.0 | 12.1% |
| **0.60** | **500** | | **25.3** | | | |
| 0.80 | 500 | 5.5 | 25.5 | 33.1 | 15.0 | 12.5% |
| 0.80 | 1000 | 4.3 | 25.5 | 33.1 | 15.0 | 11.9% |
| 0.80 | 2000 | 3.2 | 25.5 | 33.1 | 15.0 | 10.7% |
| 0.90 | 1000 | 4.3 | 25.5 | 33.1 | 15.1 | 11.9% |
| 1.00 | 500 | 5.5 | 25.6 | 33.1 | 15.2 | 12.5% |
| 1.00 | 1000 | 4.3 | 25.6 | 33.1 | 15.2 | 11.9% |
| 1.00 | 2000 | 3.2 | 25.6 | 33.1 | 15.2 | 10.7% |
| **CC-only** | **—** | **—** | **25.6** | **33.1** | **15.2** | **0.4%** |

### Comparison with Prior Configurations

| Pipeline | mIoU | PQ | PQ_stuff | PQ_things |
|----------|------|----|----------|-----------|
| CAUSE-27 + SPIdepth (old baseline) | 40.4% | 23.1 | 31.4 | 11.7 |
| Overclustered k=300 + SPIdepth (default) | 60.7% | 22.7 | 33.1 | 8.5 |
| **Overclustered k=300 (CC-only)** | **60.7%** | **25.6** | **33.1** | **15.2** |
| CUPS (CVPR 2025 SOTA) | — | 27.8 | 35.1 | 17.7 |
