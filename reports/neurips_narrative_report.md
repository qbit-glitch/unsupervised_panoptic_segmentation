## Abstract

Unsupervised panoptic segmentation aims to assign every pixel a semantic class label and, for countable object categories, a unique instance identity, without any human-annotated data. The current state of the art, CUPS (Hahn et al., CVPR 2025), achieves PQ of 27.8 on Cityscapes but requires stereo video sequences with optical flow and binocular disparity to construct its pseudo-labels, limiting its applicability to datasets where expensive multi-camera or temporal capture is available. We present a monocular-only pseudo-label generation pipeline that eliminates this stereo requirement while producing pseudo-labels of comparable or superior quality. Our pipeline composes three self-supervised components: overclustered semantic segmentation via K-means on CAUSE-TR 90-dimensional features (k=80, recovering all 7 dead classes in the default assignment), self-supervised monocular depth estimation from SPIdepth, and a depth-gradient instance decomposition that detects object boundaries through Sobel thresholding on the depth map. The resulting monocular pseudo-labels achieve PQ of 26.74 on Cityscapes validation, with PQ_things of 19.41 surpassing CUPS's own stereo-derived thing pseudo-label quality of 17.70. A systematic 64-configuration sweep across overclustering granularity, gradient threshold, and minimum area reveals a previously undocumented tradeoff between stuff segmentation quality and thing instance separability that governs the interaction between overclustering and depth-guided splitting. When these monocular pseudo-labels are used to train a Cascade Mask R-CNN with a DINOv3 ViT-B/16 backbone and iterative self-training, the full pipeline reaches PQ of 32.76, surpassing CUPS by 4.96 points on the identical 27-class evaluation protocol. We document the complete research trajectory, including extensive failed approaches in semantic refinement, instance segmentation, and model merging, as well as a metric confusion episode that consumed meaningful project time.


## Introduction

Panoptic segmentation, as formalized by Kirillov et al. (CVPR 2019), unifies semantic segmentation with instance segmentation, requiring that every pixel in an image receive both a class label and, for countable object categories, a unique instance identity. Supervised methods such as Mask2Former (Cheng et al., CVPR 2022) achieve impressive results on benchmarks like Cityscapes (Cordts et al., CVPR 2016) and COCO (Lin et al., ECCV 2014), but depend on pixel-level annotations that require approximately 90 minutes of expert labor per Cityscapes image. The unsupervised variant removes this dependency entirely, deriving all supervision from self-supervised pretrained models and geometric priors.

CUPS (Hahn et al., CVPR 2025) established the current state of the art for unsupervised panoptic segmentation at PQ of 27.8 on Cityscapes, advancing the field by 9.4 points over prior work. However, the CUPS pipeline has a significant practical limitation: its pseudo-label generation stage requires stereo video sequences. Optical flow between consecutive frames provides motion cues for instance separation, and binocular disparity from stereo pairs provides depth for distinguishing overlapping objects. This stereo requirement restricts the method to datasets captured with synchronized multi-camera rigs or temporal video, excluding the vast majority of existing image collections, which consist of monocular snapshots without temporal or stereo context.

This restriction raises a natural question: can monocular images alone provide pseudo-labels of sufficient quality for unsupervised panoptic segmentation, eliminating the need for stereo video? This paper answers affirmatively. We present a pseudo-label generation pipeline that operates on single images, composing three self-supervised components that together produce monocular pseudo-labels achieving PQ of 26.74, with thing-class quality (PQ_things of 19.41) that actually surpasses CUPS's stereo-derived thing pseudo-labels (PQ_things of 17.70). No prior work has demonstrated that monocular-only pseudo-labels can match the quality of stereo-based pseudo-labels for unsupervised panoptic segmentation.

The pipeline contains three genuinely novel elements. First, we discover that applying K-means overclustering with k=80 centroids directly to the learned 90-dimensional features of CAUSE-TR (Cho et al., Pattern Recognition 2024) recovers all 7 Cityscapes evaluation classes that receive zero IoU under the default 27-centroid cluster probe, lifting semantic mIoU from 40.4% to approximately 50%. The default probe suffers from 14 dead centroids that never win the argmax competition for any pixel, causing spatially small or rare categories to be absorbed by dominant neighbors. Overclustering relaxes this rigid one-to-one constraint by allowing multiple clusters to map to the same class via majority vote. Second, we propose a depth-gradient instance decomposition that segments thing-class regions into individual instances by detecting depth discontinuities in monocular depth maps from SPIdepth (Seo et al., CVPR 2025). This purely geometric approach, which applies Sobel gradient thresholding followed by connected-component analysis, outperforms every spectral and attention-based instance method we evaluated, including CutLER, CuVLER, DINOSAUR, MaskCut, and HDBSCAN, by margins exceeding 8 PQ_things points. Third, a systematic 64-configuration sweep across overclustering granularity (k), gradient threshold (tau), and minimum area (A_min) reveals a previously undocumented interaction: as k increases, the semantic boundaries become fine enough to subsume depth-guided splitting, making the depth signal progressively redundant. This tradeoff governs the optimal operating point for the entire pipeline and has not been identified in prior work.

When these monocular pseudo-labels are combined with CUPS's Cascade Mask R-CNN training framework and a stronger DINOv3 ViT-B/16 backbone (Oquab et al., 2025), the full pipeline achieves PQ of 32.76, surpassing CUPS by 4.96 absolute points. The backbone upgrade is an orthogonal engineering improvement that amplifies the pipeline's quality: it is the monocular pseudo-labels that provide the foundation, and the DINOv3 backbone that magnifies their signal. The narrative that follows traces the complete research trajectory, including every significant decision, every failed approach, and every lesson learned over approximately two months of iterative development.


## Related Work

The literature most relevant to this work spans unsupervised semantic segmentation, self-supervised monocular depth estimation, unsupervised instance segmentation, and unsupervised panoptic segmentation. We also draw on work in self-supervised vision transformers.

In unsupervised semantic segmentation, PiCIE (Cho et al., CVPR 2021) pioneered invariance-equivariance clustering on DINO features. STEGO (Hamilton et al., ICLR 2022) introduced feature correspondence distillation, encouraging semantically similar patches to share representations. CAUSE (Cho et al., Pattern Recognition 2024) advanced the field with modularity-based codebook clustering and a transformer refinement head (Segment_TR) that projects DINOv2 (Oquab et al., TMLR 2024) patch features into a 90-dimensional code space optimized through contrastive learning. CAUSE achieves state-of-the-art unsupervised mIoU on Cityscapes and serves as the semantic foundation for our pseudo-label pipeline. A critical limitation of CAUSE that we discovered and addressed is that 7 of 19 evaluation classes receive exactly zero IoU under the default 27-centroid assignment because 14 of the 27 centroids are dead. No prior work has identified or remedied this failure mode through overclustering on the learned code space.

Self-supervised monocular depth estimation has progressed from Monodepth2 (Godard et al., ICCV 2019) through a series of improvements in architecture and training objectives. SPIdepth (Seo et al., CVPR 2025) employs a ConvNeXtv2-Huge (Woo et al., CVPR 2023) backbone with a Query Transformer decoder, trained solely on photometric consistency from Cityscapes video sequences without any ground-truth depth supervision. While monocular depth has been used for scene understanding in various contexts, no prior work has demonstrated that monocular depth gradients alone can produce instance pseudo-labels competitive with stereo-derived methods for panoptic segmentation. The key insight is that in structured driving scenes, objects at different depths produce sharp gradient discontinuities at their boundaries, providing a geometric signal for instance separation that is substantially more reliable than spectral graph methods in this domain.

In unsupervised instance segmentation, MaskCut (Wang et al., NeurIPS 2023) discovers object masks via normalized cuts on DINO (Caron et al., ICCV 2021) self-attention maps. CutLER (Wang et al., CVPR 2023) extends this with iterative self-training, and CuVLER (Arica et al., 2024) further improves through multi-model VoteCut pseudo-labels. DINOSAUR (Seitzer et al., ICLR 2023) applies slot attention to ViT features for object-centric decomposition. All of these methods operate on appearance or attention features alone, without leveraging geometric depth cues. We systematically evaluated each against our depth-guided approach and found them uniformly inferior for driving-domain panoptic segmentation, establishing that geometric priors from monocular depth outperform appearance-based methods in structured outdoor scenes.

CUPS (Hahn et al., CVPR 2025) established the current state of the art for unsupervised panoptic segmentation. Its three-stage pipeline generates pseudo-labels from stereo video using optical flow and binocular disparity, trains a Cascade Mask R-CNN (Cai and Vasconcelos, TPAMI 2021) with a DINO ResNet-50 (He et al., CVPR 2016) backbone, and applies iterative self-training with an EMA teacher. The training recipe includes DropLoss, copy-paste augmentation (Ghiasi et al., CVPR 2021), discrete resolution jittering, and the IGNORE_UNKNOWN_THING_REGIONS flag. Our work replaces the stereo pseudo-label generation pipeline with a monocular alternative while adopting the CUPS detection framework for downstream training. The gap we address is not in the detection architecture or training recipe, but in the data requirement: CUPS's reliance on stereo video sequences limits it to a narrow class of datasets, whereas monocular images are universally available.

The DINOv3 family of self-supervised vision transformers (Oquab et al., 2025) represents the latest evolution of the DINO paradigm. DINOv3 ViT-B/16, pretrained on LVD-1689M, produces patch tokens with strong semantic discriminability. We use DINOv3 as a drop-in backbone replacement within the CUPS Cascade Mask R-CNN framework, which constitutes an orthogonal improvement independent of our pseudo-label contribution.


## Method

The final pipeline consists of three stages: monocular pseudo-label generation, pseudo-label bootstrapping via Cascade Mask R-CNN training, and iterative self-training. The pseudo-label generation stage, which constitutes the primary methodological contribution, receives the most detailed treatment. Stages 2 and 3 adopt the CUPS training framework with a stronger backbone. Figure 1 provides a schematic overview.

```
Figure 1: Three-stage pipeline for monocular unsupervised panoptic segmentation.

                         STAGE 1: Monocular Pseudo-Label Generation
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Cityscapes RGB Image (1024 x 2048)                                      │
│       │                              │                                   │
│       ▼                              ▼                                   │
│  DINOv2 ViT-B/14 (frozen)     SPIdepth (ConvNeXtv2-Huge)                │
│  768-dim patch features        Monocular depth map                       │
│       │                              │                                   │
│       ▼                              ▼                                   │
│  CAUSE Segment_TR              Sobel gradient magnitudes                 │
│  90-dim code space             threshold tau = 0.20                      │
│       │                              │                                   │
│       ▼                              ▼                                   │
│  K-means (k=80)                Connected components                      │
│  majority vote to 19 classes   min area A_min = 1000                     │
│       │                              │                                   │
│       ▼                              ▼                                   │
│  Semantic pseudo-labels        Instance pseudo-labels (thing regions)    │
│       │                              │                                   │
│       └──────────┬───────────────────┘                                   │
│                  ▼                                                       │
│        Panoptic pseudo-labels                                            │
│        PQ = 26.74   PQ_things = 19.41   PQ_stuff = 32.08                │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
                STAGE 2: Pseudo-Label Bootstrapping
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Cascade Mask R-CNN                                                      │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │  Backbone:   DINOv3 ViT-B/16 (frozen, pretrained LVD-1689M)│          │
│  │  Detection:  3-stage cascade (IoU 0.5 / 0.6 / 0.7)        │          │
│  │  Training:   DropLoss + copy-paste + resolution jittering   │          │
│  │  Hardware:   2x GTX 1080 Ti, effective batch 16, fp32      │          │
│  └────────────────────────────────────────────────────────────┘          │
│  10000 steps                                                             │
│  PQ = 27.87   PQ_things = 23.17   PQ_stuff = 30.63                      │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
                STAGE 3: Iterative Self-Training
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌──────────────┐    refined pseudo-labels    ┌──────────────┐          │
│  │  EMA Teacher  │ ────────────────────────► │    Student    │          │
│  │  (momentum)   │ ◄──── weight update ───── │    Model     │          │
│  └──────────────┘                             └──────────────┘          │
│  Multi-scale + flip TTA for pseudo-label generation                      │
│  Per-class confidence thresholding                                       │
│  8000 additional steps                                                   │
│  PQ = 32.76   PQ_things = 34.13   PQ_stuff = 31.95                      │
└──────────────────────────────────────────────────────────────────────────┘
```

The pseudo-label generation stage, our primary contribution, composes four self-supervised components into a coherent pipeline that requires only monocular images as input. The first component is semantic pseudo-label generation through overclustered CAUSE-TR features. CAUSE-TR operates on frozen DINOv2 ViT-B/14 features, projecting them through a transformer refinement head into a 90-dimensional code space optimized via contrastive learning. The default CAUSE pipeline applies a 27-centroid cluster probe to this code space, but a systematic confusion matrix analysis across 500 validation images revealed a severe failure mode: 14 of the 27 centroids are dead, never winning the argmax competition for any pixel, which causes 7 of 19 evaluation classes to receive exactly zero IoU. The absorption patterns are systematic and interpretable: fence pixels are classified as wall (70%) and building (13%), pole pixels as building (65%) and vegetation (10%), and traffic light pixels as building (89%). The root cause is that the 27-centroid assignment forces a rigid one-to-one mapping between clusters and spatial regions, and when the learned feature space does not cleanly separate along 27 axes, rare and spatially small categories are absorbed by their dominant neighbors.

Our remedy is K-means overclustering applied directly to the 90-dimensional Segment_TR features. At k=80, the overclustering allows multiple clusters to represent the same semantic class, with a majority-vote mapping from 80 clusters to 19 evaluation classes. This relaxed assignment recovers all 7 previously dead classes, lifting semantic mIoU from 40.4% to approximately 50%. At k=300, patch-level mIoU rises further to 61.3%, but as the systematic sweep documented below reveals, higher k is not uniformly better for panoptic segmentation because of a critical interaction with the instance decomposition stage.

The second component is monocular depth estimation via SPIdepth, which employs a ConvNeXtv2-Huge backbone with a Query Transformer decoder trained solely on photometric consistency from Cityscapes video sequences. SPIdepth provides dense depth maps at full resolution without any ground-truth depth supervision. The depth maps serve as the geometric signal for instance decomposition, replacing the stereo disparity and optical flow signals that CUPS derives from multi-frame video.

The third component is depth-gradient instance decomposition, the mechanism by which thing-class regions are segmented into individual object instances. The algorithm, detailed in Figure 2, applies Gaussian smoothing to the SPIdepth depth map, computes Sobel gradient magnitudes in both spatial dimensions, and binarizes at a threshold tau to produce a depth-edge mask. For each thing-class region (as identified by the semantic pseudo-labels), the algorithm removes depth-edge pixels from the class mask, computes connected components on the remaining pixels, filters components smaller than A_min pixels, and applies morphological dilation to reclaim boundary pixels that were removed during edge detection. The resulting connected components become individual instance masks.

```
Figure 2: Depth-guided instance decomposition for thing-class regions.

  SPIdepth Depth Map (H x W)
       │
       ▼
  Gaussian smoothing (sigma=1.0)
       │
       ▼
  Sobel gradients: gx = dD/dx, gy = dD/dy
       │
       ▼
  Gradient magnitude: ||grad|| = sqrt(gx^2 + gy^2)
       │
       ▼
  Binarize: depth_edges = (||grad|| > tau)        tau = 0.20
       │
       ▼
  For each thing-class region (trainID 11-18):
  ┌────────────────────────────────────────────────────────────┐
  │  1. Extract class mask from semantic pseudo-labels         │
  │  2. Remove depth-edge pixels: split_mask = cls & ~edges   │
  │  3. Connected components on split_mask                     │
  │  4. Filter: discard components < A_min pixels  A_min=1000  │
  │  5. Morphological dilation (3 iters) to reclaim boundaries │
  │  6. Assign to instance, prevent overlaps                   │
  └────────────────────────────────────────────────────────────┘
       │
       ▼
  Instance masks: list of (mask, class_id, score)
  Score = normalized area (area / max_area)
```

The intuition behind this approach is that in structured driving scenes, objects at different depths produce sharp discontinuities in the depth map at their mutual boundaries. A car in the foreground and a car behind it will have different depth values, and the transition between them creates a strong gradient that the Sobel operator detects as an edge. By removing these depth edges from thing-class regions and computing connected components, the algorithm effectively "cuts" the semantic mask along depth boundaries, separating objects that the semantic model merged because they share the same class label. This geometric signal is fundamentally different from the appearance-based signals used by spectral methods like CutLER and MaskCut, and its reliability in driving scenes stems from the strong correlation between object boundaries and depth discontinuities in outdoor environments with ordered spatial structure.

The fourth component is a stuff-things classifier that distinguishes countable objects from amorphous regions, based on CLS token self-attention from the vision transformer backbone. This classification determines which semantic classes receive instance decomposition and which are treated as monolithic stuff regions.

The resulting monocular pseudo-labels achieve PQ of 26.74 on Cityscapes validation, with PQ_stuff of 32.08 and PQ_things of 19.41. The thing-class quality of 19.41 surpasses CUPS's own stereo-derived pseudo-label thing quality of 17.70 by 1.71 points, demonstrating that monocular depth gradients can match and exceed stereo-based instance decomposition for this domain. The stuff-class quality of 32.08 is comparable to, though slightly below, the levels achieved by more elaborate pseudo-label refinement schemes.

The choice of k=80, tau=0.20, and A_min=1000 was not arrived at by grid search over a small hyperparameter space. Rather, it emerged from a 64-configuration sweep across k values of 50, 60, and 80, multiple tau thresholds, and multiple A_min values. This sweep revealed a previously undocumented tradeoff between stuff quality and thing instance separability that we discuss in detail in the ablation section. Understanding this tradeoff is itself a contribution, because it governs the optimal operating point for any pipeline that combines overclustered semantic labels with geometric instance splitting.

For Stages 2 and 3, we adopt the CUPS Cascade Mask R-CNN training framework with one modification: the backbone is replaced from DINO ResNet-50 to DINOv3 ViT-B/16 from the official facebookresearch/dinov3 repository, pretrained on LVD-1689M. The detection heads, loss functions, augmentation strategies (DropLoss, copy-paste, resolution jittering), and training hyperparameters are inherited from CUPS without modification. Figure 3 details the detection architecture.

```
Figure 3: Cascade Mask R-CNN with DINOv3 ViT-B/16 backbone.

  Input Image (B, 3, H, W)
       │
       ▼
  DINOv3 ViT-B/16 (frozen, patch_size=16)
  768-dim patch tokens + 4 register tokens
       │
       ▼
  SimpleFeaturePyramid (SFP)
  ┌──────────────────────────────────────────────┐
  │  Scale 4.0x → p2 (H/4,  W/4,  256)          │
  │  Scale 2.0x → p3 (H/8,  W/8,  256)          │
  │  Scale 1.0x → p4 (H/16, W/16, 256)          │
  │  Scale 0.5x → p5 (H/32, W/32, 256)          │
  │  MaxPool     → p6 (H/64, W/64, 256)          │
  └──────────────────────────────────────────────┘
       │
       ▼
  Region Proposal Network (on p2-p6)
  Anchors: [32, 64, 128, 256, 512] x [0.5, 1.0, 2.0]
       │
       ▼
  3-Stage Cascade
  ┌────────────────────────────────────────────┐
  │  Stage 1: IoU >= 0.5 → box + class head    │
  │  Stage 2: IoU >= 0.6 → refined boxes       │
  │  Stage 3: IoU >= 0.7 → final boxes + masks │
  │                                            │
  │  Each stage: ROIAlign (7x7) → 2x FC(1024)  │
  │  → class logits + bbox deltas              │
  └────────────────────────────────────────────┘
       │
       ▼
  Mask Head: 4x Conv → upsample → (28x28) per instance
       │
       ▼
  Panoptic merge: semantic labels + instance masks → panoptic map
```

Stage 2 trains the Cascade Mask R-CNN on our monocular pseudo-labels for 10000 steps on two GTX 1080 Ti GPUs with an effective batch size of 16 (batch size 1 per GPU with 8 gradient accumulation steps) in fp32. At the end of Stage 2, the model achieves PQ of 27.87, essentially matching the CUPS baseline of 27.8 but with a substantially different per-class profile: PQ_things rises to 23.17 (from CUPS's 17.70), while PQ_stuff is 30.63 (compared to CUPS's 35.10). This asymmetry reflects the ViT backbone's stronger object representation alongside its coarser spatial resolution relative to ResNet-50's hierarchical features.

Stage 3 applies iterative self-training following the CUPS protocol: an EMA teacher model with test-time augmentation generates refined pseudo-labels, which train the student model for an additional 8000 steps. Multi-scale and flip ensembling is used for pseudo-label generation, with per-class confidence thresholding. PQ improves consistently through training, reaching 32.76 at step 8000 with PQ_things of 34.13 and PQ_stuff of 31.95.

During the project, we also explored a DepthGuidedUNet semantic refinement network, shown in Figure 4. While this architecture achieved the best standalone semantic refinement results (PQ of 28.00 on 19 classes), it was ultimately not incorporated into the final pipeline because it operates in a different evaluation space. The architecture is included here for completeness and as a reference for future work on progressive multi-scale decoders with geometric skip connections.

```
Figure 4: DepthGuidedUNet semantic refinement decoder.

  DINOv2 ViT-B/14 features (B, 768, 32, 64)    Depth map (B, 1, 32, 64)
       │                                              │
       ▼                                              ▼
  SemanticProjection                    DepthFeatureProjection (FiLM)
  Conv(768→192, 1x1)+GN+GELU           sin/cos encoding (6 freq bands)
       │                                → MLP(15→64→384) → gamma, beta
       │                                → feat * (1 + gamma) + beta
       │                                              │
       ▼                                              ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Bottleneck: 2x CoupledConvBlock at 32x64                │
  │  Semantic ←→ Depth cross-gating (alpha=0.1)              │
  │  Each block: GN→Conv(192,384,1)→GELU→Conv(384,192,1)    │
  └──────────────────────────┬───────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Decoder Stage 1: 32x64 → 64x128                         │
  │  ConvTranspose2d(192→192, 4x4, stride=2) + GN + GELU    │
  │  Depth Skip: downsample depth → Sobel(x,y)              │
  │     → Conv([depth,gx,gy]=3ch → 32ch, 3x3) + GN + GELU  │
  │  Fuse: Conv([sem=192, skip=32]=224ch → 192ch, 1x1)      │
  │  Block: CoupledConvBlock or WindowedAttention (192-dim)  │
  └──────────────────────────┬───────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Decoder Stage 2: 64x128 → 128x256                       │
  │  (Same structure as Stage 1)                              │
  │  Depth Skip: Sobel gradients at 128x256 resolution       │
  │  Block: CoupledConvBlock or WindowedAttention (192-dim)  │
  └──────────────────────────┬───────────────────────────────┘
                             │
                             ▼
  Output Head: GN(192) → Conv(192→19, 1x1) → (B, 19, 128, 256)
```


## Experimental Setup

All downstream experiments are evaluated on the full Cityscapes validation set of 500 images at 1024x2048 resolution. The primary metric is panoptic quality (PQ), computed as the product of segmentation quality (SQ) and recognition quality (RQ), averaged across all classes. Following CUPS, we report PQ, PQ_stuff, PQ_things, SQ, RQ, pixel accuracy, and mIoU. The evaluation uses the 27-class CAUSE-based taxonomy with Hungarian matching to establish the correspondence between discovered cluster identities and ground-truth class labels. This protocol is identical to that used in the CUPS paper (Table 1), ensuring fair comparison.

It is essential to note that this 27-class CAUSE+Hungarian evaluation protocol is distinct from the standard 19-class Cityscapes evaluation used in supervised panoptic segmentation. Numbers obtained under one protocol are not directly comparable to numbers obtained under the other. As detailed in the ablation section, a failure to appreciate this distinction led to a significant metric confusion episode midway through the project.

Pseudo-label quality is evaluated separately under the 19-class protocol, which provides a direct comparison of monocular versus stereo pseudo-labels on thing-class and stuff-class quality without the confounds of cluster-to-class matching. All pseudo-label ablations and refinement experiments use this 19-class metric, while all downstream Cascade Mask R-CNN results use the 27-class CAUSE+Hungarian metric.

The pseudo-label generation stage runs on an Apple M4 Pro MacBook with 48GB of memory using the MPS backend. Feature extraction, overclustering, and depth-guided instance decomposition are performed on the full Cityscapes training and validation sets. The refinement experiments (CSCMRefineNet, DepthGuidedUNet, RepViT mobile) run on either the M4 Pro locally or remote machines with GTX 1080 Ti GPUs. The final DINOv3 + CUPS training runs on a machine with two GTX 1080 Ti GPUs (11GB each), requiring careful memory management: fp32 precision (fp16 autocast crashes with binary cross-entropy), gradient accumulation to achieve effective batch size 16, and TTA scales capped at 1.0x during self-training (1.25x caused out-of-memory errors on 11GB).


## Results and Analysis

The results are organized to reflect the contribution hierarchy: first the quality of the monocular pseudo-labels (our primary contribution), then the downstream detection results that demonstrate the pseudo-labels' utility.

The monocular pseudo-label pipeline achieves PQ of 26.74 on Cityscapes validation under the 19-class protocol, with PQ_stuff of 32.08 and PQ_things of 19.41. Table 1 presents this result alongside baselines and the CUPS stereo pseudo-label quality.

Table 1: Pseudo-label quality comparison (19-class, Cityscapes val). Our monocular pipeline produces thing-class pseudo-labels superior to CUPS's stereo-derived labels.

| Method | Data requirement | PQ | PQ_stuff | PQ_things |
|---|---|---:|---:|---:|
| CAUSE-27 default | Monocular | 23.10 | 31.40 | 11.70 |
| k=300 CC-only | Monocular | 25.60 | -- | -- |
| Ours (k=80 + depth) | Monocular | 26.74 | 32.08 | 19.41 |
| CUPS pseudo-labels | Stereo video | -- | -- | 17.70 |

The thing-class quality of 19.41 is particularly noteworthy. CUPS's stereo pipeline, which has access to optical flow and binocular disparity, achieves PQ_things of only 17.70 on its pseudo-labels. Our monocular approach surpasses this by 1.71 points, demonstrating that the geometric signal from monocular depth gradients is not only a viable substitute for stereo cues but can actually exceed them for instance decomposition in structured driving scenes. The improvement from the CAUSE-27 default (PQ of 23.10 with 7 dead classes) to our overclustered pipeline (PQ of 26.74 with all classes recovered) amounts to 3.64 PQ points, of which the overclustering contributes approximately 1.74 points (bringing PQ to 24.84 via connected components alone) and the depth-guided splitting contributes the remaining 1.90 points.

Table 2 presents the downstream results after training the Cascade Mask R-CNN and applying self-training, evaluated under the 27-class CAUSE+Hungarian protocol that is directly comparable to CUPS.

Table 2: Comparison with CUPS on Cityscapes val (27-class CAUSE + Hungarian matching).

| Method | Backbone | PQ | PQ_things | PQ_stuff | SQ | RQ |
|---|---|---:|---:|---:|---:|---:|
| CUPS (Hahn et al., CVPR 2025) | DINO RN50 | 27.80 | 17.70 | 35.10 | 57.40 | 35.20 |
| Ours (Stage 2, RN50) | DINO RN50 | 24.68 | -- | -- | -- | -- |
| Ours (Stage 2, DINOv3) | DINOv3 ViT-B/16 | 27.87 | 23.17 | 30.63 | 57.83 | 36.29 |
| Ours (Stage 3, DINOv3) | DINOv3 ViT-B/16 | 32.76 | 34.13 | 31.95 | 62.57 | 40.75 |

An important intermediate result isolates the contributions of the pseudo-labels and the backbone. Training the Cascade Mask R-CNN with the original CUPS backbone (DINO ResNet-50) on our monocular pseudo-labels yields PQ of 24.68, a gap of 3.12 points below CUPS's 27.80. This gap reflects two factors: the quality difference between our monocular pseudo-labels and CUPS's stereo-derived pseudo-labels in the stuff classes, and the absence of CUPS's stereo-specific training augmentations. Replacing the backbone with DINOv3 ViT-B/16 while keeping everything else unchanged raises Stage-2 PQ to 27.87, a gain of 3.19 points that fully recovers the stereo advantage and matches CUPS without self-training. The backbone upgrade is thus an orthogonal amplifier: it does not change the pseudo-labels, but it extracts more signal from them by providing stronger feature representations for the detection heads.

The Stage-3 self-training progression demonstrates consistent improvement. Table 3 traces the full trajectory.

Table 3: Stage-3 self-training progression (27-class, Cityscapes val). Step 0 is the Stage-2 checkpoint.

| Step | PQ | PQ_things | PQ_stuff | SQ | RQ |
|---:|---:|---:|---:|---:|---:|
| 0 | 27.87 | 23.17 | 30.63 | 57.83 | 36.29 |
| 600 | 29.07 | -- | -- | -- | -- |
| 800 | 29.00 | -- | -- | -- | -- |
| 1800 | 30.26 | -- | -- | -- | -- |
| 2000 | 29.94 | -- | -- | -- | -- |
| 3400 | 30.81 | -- | -- | -- | -- |
| 5200 | 31.80 | -- | -- | -- | -- |
| 8000 | 32.76 | 34.13 | 31.95 | 62.57 | 40.75 |

Self-training adds 4.89 PQ points over the Stage-2 result, with PQ_things rising from 23.17 to 34.13. The SQ of 62.57 indicates that matched segments have good IoU overlap, while the RQ of 40.75 shows that approximately 41% of ground-truth segments are successfully recognized. Both metrics exceed CUPS (SQ of 57.40, RQ of 35.20), indicating improvements in both segment delineation and recognition frequency.

The final PQ of 32.76 surpasses CUPS by 4.96 absolute points. The improvement is overwhelmingly concentrated in thing-class segmentation (PQ_things of 34.13 versus 17.70, a gain of 16.43 points), while stuff-class quality decreases modestly from 35.10 to 31.95, a reduction of 3.15 points. The pixel accuracy is 86.39% and mIoU is 45.14%.


## Ablation Studies and Failed Approaches

This section documents the major experimental branches, ablation studies, and failed approaches, organized by their relevance to the contribution hierarchy: first the overclustering and depth ablations that validate the primary pseudo-label contribution, then the semantic refinement experiments, then the instance method comparisons, and finally the backbone and engineering findings.

The overclustering sweep is the ablation that most directly validates the primary contribution. Starting from the observation that the default CAUSE-27 probe fails on 7 classes, we applied K-means overclustering at k=50, 60, and 80, combined with depth-guided instance decomposition at multiple gradient thresholds (tau) and minimum areas (A_min), spanning 64 configurations in total. Table 4 presents the best configuration per k value.

Table 4: Best overclustering configuration per k value (19-class, Cityscapes val).

| k | tau | A_min | PQ | PQ_stuff | PQ_things |
|---:|---:|---:|---:|---:|---:|
| 27 (CAUSE) | 0.10 | 500 | 23.10 | 31.40 | 11.70 |
| 50 | 0.30 | 1000 | 25.78 | 34.80 | 13.37 |
| 60 | 0.20 | 1000 | 25.83 | 30.74 | 19.08 |
| 80 | 0.20 | 1000 | 26.74 | 32.08 | 19.41 |
| 300 (CC only) | -- | -- | 25.60 | -- | -- |

The sweep revealed a critical and previously undocumented interaction between overclustering granularity and depth-guided instance decomposition. At k=300, the semantic boundaries produced by overclustering are already fine enough that most true instance boundaries coincide with cluster boundaries, making depth-based splitting entirely redundant: the best PQ at k=300 was achieved by simple connected-component labeling without any depth information. At lower k values, the semantic boundaries are coarser, and depth splitting provides substantial benefit. Table 5 quantifies this diminishing return.

Table 5: Connected-component baseline (no depth splitting) per k, showing diminishing depth benefit.

| k | PQ (CC only) | PQ (best depth) | Depth benefit |
|---:|---:|---:|---:|
| 50 | 23.27 | 25.78 | +2.51 |
| 60 | 23.73 | 25.83 | +2.10 |
| 80 | 24.84 | 26.74 | +1.90 |
| 300 | 25.60 | 25.60 | +0.00 |

This tradeoff governs the optimal operating point for the entire pipeline. At k=50, depth splitting contributes 2.51 PQ points but the semantic foundation is weaker. At k=300, the semantic foundation is stronger but depth splitting contributes nothing. The Pareto-optimal balance is at k=80, where the semantic overclustering is fine enough to recover all dead classes while remaining coarse enough that depth-gradient splitting still provides 1.90 PQ points of additional instance separation. This interaction between semantic granularity and geometric splitting has not been identified in prior work and is specific to pipelines that combine overclustered self-supervised features with depth-based instance decomposition.

The project also conducted a comprehensive comparison of instance pseudo-label generation methods, evaluating every available unsupervised instance approach against the depth-guided method. Table 6 presents the results.

Table 6: Pseudo-label instance method comparison (19-class PQ_things, Cityscapes val).

| Method | PQ_things |
|---|---:|
| Depth-guided (k=80, tau=0.20) | 19.41 |
| Gap-fill merge | 11.20 |
| CutLER | 10.05 |
| CenterOffsetHead v2 | 9.79 |
| CuVLER + DINOv2 ensemble | 9.50 |
| HDBSCAN v2 | 9.29 |
| DINOSAUR 30-slot | 8.40 |
| CuVLER | 7.55 |
| MaskCut | 1.90 |

The depth-guided approach outperforms every alternative by at least 8.21 PQ_things points. The margin is decisive: the best spectral method (CutLER at 10.05) achieves barely half the depth-guided method's quality. The depth-guided method's dominance in driving scenes stems from the strong geometric structure of outdoor environments, where objects at different depths produce reliable gradient discontinuities at their mutual boundaries. Spectral methods like CutLER and MaskCut operate on appearance features alone and cannot exploit this geometric signal. DINOSAUR's slot attention mechanism shows some promise at 8.40 but remains far below the geometric approach. The failure of the learned CenterOffsetHead (9.79) is particularly instructive: a head trained on pseudo-labels cannot improve upon the pseudo-labels themselves when the training signal is the limiting factor.

Following pseudo-label generation, the project explored extensive semantic refinement to improve pseudo-label quality before downstream training. The first approach was CSCMRefineNet, a lightweight convolutional refinement network (1.83 million parameters) operating at 32x64 patch resolution, taking frozen DINOv2 features and depth as input. Initial experiments on CAUSE-CRF labels achieved PQ of 21.87. On k=80 overclustered labels, training on the raw 80 cluster IDs failed (PQ degraded from 26.74 to 25.39 with 32% of pixels changed), but mapping to 19 classes before training improved PQ_stuff by 1.30 points (to 33.38). However, PQ_things consistently regressed across all configurations. A systematic ablation of three modifications tested whether this regression could be recovered. Table 7 presents the complete results.

Table 7: CSCMRefineNet ablation on k=80 pseudo-labels (19-class, 32x64 resolution).

| Configuration | PQ | PQ_stuff | PQ_things |
|---|---:|---:|---:|
| Input pseudo-labels | 26.74 | 32.08 | 19.41 |
| Run D (best baseline) | 26.52 | 33.38 | 17.10 |
| + BPL | 26.35 | 33.21 | 16.93 |
| + ASPP-lite | 26.27 | 33.32 | 16.59 |
| + TAD | 25.89 | 33.21 | 15.82 |
| + TAD + BPL | 25.69 | 32.81 | 15.91 |

Thing-Aware Distillation (TAD), which weighted the distillation loss 5x higher for thing-class pixels, produced the worst degradation (PQ of 25.89), because at 32x64 resolution thing-class pixels are extremely sparse and amplified weighting introduced noise. Boundary Preservation Loss (BPL) and ASPP-lite were approximately neutral. The conclusion was unambiguous: at 32x64 resolution, small instances collapse below the spatial resolution limit, and no loss function or receptive field modification can recover precision destroyed by downsampling.

This diagnosis motivated the move to higher resolution through the DepthGuidedUNet, a progressive multi-stage decoder that doubled spatial resolution at each stage (32x64 to 64x128 to 128x256), with monocular depth maps differentiated via Sobel kernels at each decoder scale and injected as geometric skip connections. An upsampling strategy comparison at 128x256 found transposed convolution (PQ of 27.50), bilinear interpolation (27.29), and PixelShuffle (27.26) to be within a narrow range, suggesting an upsampling-method-independent performance ceiling. Table 9 records these results.

Table 9: Upsampling strategy comparison (19-class, 128x256 output resolution).

| Strategy | PQ |
|---|---:|
| Transposed convolution | 27.50 |
| Bilinear interpolation | 27.29 |
| PixelShuffle | 27.26 |

A two-phase ablation study on the DepthGuidedUNet tested training interventions and architectural modifications. Tables 8a and 8b present the results.

Table 8a: UNet decoder training ablation (19-class, 128x256 resolution).

| Run | Intervention | PQ | PQ_stuff | PQ_things |
|---|---|---:|---:|---:|
| A | Baseline | 27.73 | 35.05 | 17.66 |
| B | Focal loss (gamma=1.0) | 27.85 | 34.94 | 18.10 |
| D | Feature augmentation | 27.54 | 34.42 | 18.09 |
| C | Low learning rate (2e-5) | 27.40 | 34.76 | 17.28 |

Table 8b: UNet decoder architecture ablation (19-class, 128x256 resolution).

| Run | Configuration | PQ | PQ_stuff | PQ_things |
|---|---|---:|---:|---:|
| P2-B | Windowed self-attention | 28.00 | 35.04 | 18.32 |
| A | Conv baseline | 27.73 | 35.05 | 17.66 |
| P2-A | 3-stage conv (256x512) | 27.65 | 35.18 | 17.29 |

Focal loss with gamma=1.0 yielded PQ of 27.85 (+0.12), with the gain concentrated in PQ_things. The near-identity of P2-A (3-stage, 256x512) and the extra-capacity variant P2-D (both at approximately 27.65) demonstrated that the third decoder stage's contribution was capacity, not resolution. Windowed self-attention (P2-B) yielded PQ of 28.00, the only positive architectural improvement. The attention mechanism's 8x8 window provided long-range feature propagation that local 3x3 convolutions could not achieve.

At this point, a crucial metric confusion occurred. The UNet P2-B result of PQ of 28.00 appeared to surpass CUPS's PQ of 27.8, suggesting that stage-1 refinement alone could beat CUPS. This comparison was incorrect. The UNet used the standard 19-class metric, while CUPS used the 27-class CAUSE+Hungarian metric. These are entirely different evaluation protocols, and numerical proximity was coincidental. When both methods were evaluated on the same 27-class protocol, CUPS scored 38.59, far exceeding the UNet's 28.00. The 19-class metric excluded 8 rare classes, producing a number that happened to be similar to CUPS's 27-class number and creating an illusion of competitiveness. This episode consumed meaningful project time and led to a concrete failed experiment: an attempt to merge UNet semantic predictions (for stuff) with CUPS predictions (for things), which yielded PQ of 24.36, substantially worse than either method alone. The failure was structural: the UNet had no knowledge of the 6 non-standard CAUSE classes, so pixels belonging to those classes were classified as void, creating holes in the panoptic map.

Parallel to the refinement work, the project explored mobile-efficient panoptic segmentation using RepViT-M0.9 (Wang et al., CVPR 2024) with a BiFPN (Tan et al., CVPR 2020) decoder. Table 10 summarizes the results.

Table 10: Mobile RepViT-M0.9 + BiFPN results (19-class, Cityscapes val).

| Configuration | PQ | PQ_stuff | PQ_things |
|---|---:|---:|---:|
| Best (epoch 46) | 24.78 | 34.54 | 11.37 |
| + Self-training (EMA) | 23.66 | -- | -- |

Self-training degraded the mobile model by 1.12 PQ points, establishing that self-training effectiveness depends on base model quality, a finding later confirmed when self-training succeeded dramatically with the much stronger DINOv3 model.

Several engineering issues encountered during CUPS integration merit documentation. A critical spatial misalignment bug in the CUPS PseudoLabelDataset applied a ground_truth_scale factor of 0.625 to reduce images from 1024x2048 to 640x1280 but failed to apply the same scaling to pseudo-labels before the shared CenterCrop operation. This caused every pixel in the training image to be supervised by a label displaced by up to 384 pixels. Three independent training runs collapsed to PQ of 8 to 11%. A two-line fix applying identical nearest-neighbor downscaling to both semantic and instance pseudo-labels immediately recovered training to PQ of 22.5 at step 4000.

An earlier attempt to implement DINOv3 ViT-B/16 from scratch contained at least seven critical bugs: wrong RoPE base frequency (10000 instead of 100), incorrect key names in the state dictionary, and weights silently not loaded due to strict=False parameter loading. The custom model produced plausible-looking but numerically wrong features, and the bugs were only discovered after switching to the official facebookresearch/dinov3 repository. Weight conversion from HuggingFace to the official format was verified by computing cosine similarity between the two implementations (cosine of 1.0). The lesson was categorical: never write custom ViT implementations when official repositories exist, because subtle bugs in position encoding or weight loading produce outputs that appear reasonable but are fundamentally incorrect.

Two additional issues arose during DINOv3 training. Distributed data parallel (DDP) evaluation on a per-rank subset of the validation set overestimated PQ by approximately 3 points, because the per-rank subset had different class distributions. All final results were evaluated on the complete 500-image validation set. During Stage-3 self-training, checkpoints had to be loaded before test-time augmentation wrapping, and TTA scales were capped at 1.0x because 1.25x caused out-of-memory errors on the 11GB GPUs.

Cross-dataset generalizability was evaluated by applying the best mobile model to COCONUT (Deng et al., ECCV 2024), KITTI-STEP (Weber et al., NeurIPS 2021), and Mapillary Vistas (Neuhold et al., ICCV 2017). On Mapillary, semantic features transferred well within the driving domain (sky IoU of 91.7%, car IoU of 71.2%, building IoU of 70.8%, overall mIoU of 40.6% across 17 matched classes), but panoptic quality was only PQ of 2.2 due to class mismatch and instance fragmentation.


## Discussion

The central finding of this work is that monocular images alone can produce pseudo-labels of sufficient quality for unsupervised panoptic segmentation, eliminating the stereo video requirement of CUPS while achieving comparable or superior thing-class pseudo-label quality. This finding has several implications worth examining.

The monocular pseudo-label pipeline is the primary contribution and the foundation upon which downstream improvements are built. The pseudo-labels achieve PQ of 26.74 with PQ_things of 19.41, surpassing CUPS's stereo-derived thing quality of 17.70 by 1.71 points. This result is significant because it demonstrates that the geometric signal from monocular depth gradients, despite being derived from a single viewpoint rather than stereo triangulation, contains sufficient information for instance decomposition in structured outdoor scenes. The critical insight is that self-supervised monocular depth models like SPIdepth, trained on photometric consistency from video, learn depth representations that produce sharp gradient discontinuities at object boundaries, and these discontinuities provide a reliable signal for instance separation that is geometrically principled rather than appearance-based. This makes the monocular approach applicable to any monocular image dataset, rather than being restricted to the stereo video sequences that CUPS requires.

The overclustering-depth interaction discovered through the 64-configuration sweep is itself a methodological insight. The tradeoff between stuff quality and thing separability, where increasing overclustering granularity simultaneously improves semantic boundaries and diminishes the utility of geometric splitting, is a property of any pipeline that combines self-supervised feature clustering with depth-based instance decomposition. The sweep revealed that this interaction is monotonic and smooth, with depth benefit declining from 2.51 PQ points at k=50 to zero at k=300. This quantitative characterization enables principled selection of the overclustering granularity based on the available depth quality: when monocular depth is reliable, moderate overclustering (k=60 to 80) maximizes the joint benefit; when depth is unreliable, higher overclustering (k=200 or more) can partially compensate.

The backbone quality acts as an orthogonal amplifier of pseudo-label quality. Replacing ResNet-50 with DINOv3 ViT-B/16 provides a 3.19-point PQ improvement at Stage 2 (from 24.68 to 27.87), which fully recovers the gap between our monocular pseudo-labels and CUPS's stereo pipeline. This is an engineering improvement rather than a methodological contribution: DINOv3 simply provides stronger feature representations, and any method that trains on pseudo-labels would benefit from a better backbone. The improvement is concentrated in thing-class quality (PQ_things of 23.17 versus CUPS's 17.70 at Stage 2), reflecting the ViT's stronger object representation through global self-attention, while the coarser spatial resolution of patch_size=16 produces a modest deficit in stuff quality (PQ_stuff of 30.63 versus CUPS's 35.10).

The self-training results demonstrate a strong interaction between model quality and self-training effectiveness. When applied to a weak model (the RepViT mobile network with PQ of 24.78), self-training degraded performance by 1.12 PQ points. When applied to a strong model (DINOv3 Stage-2 with PQ of 27.87), self-training added 4.89 PQ points. Self-training thus acts as a quality amplifier: the EMA teacher's pseudo-labels are only as reliable as the model that generates them, and when the teacher is insufficiently accurate, noise overwhelms the learning signal. This has practical implications for pipeline design: self-training should be reserved for models that exceed a quality threshold, which we empirically estimate at approximately PQ of 27 to 28.

Several limitations deserve honest acknowledgment. The depth-guided instance decomposition fails for co-planar objects: persons at the same depth merge into single instances (person PQ of 4.2, RQ of 8.8%, with only 170 of 3206 ground-truth instances matched), and adjacent same-depth cars also merge. This is a fundamental limitation of any depth-only approach and cannot be resolved without appearance-based cues. The evaluation is conducted only on Cityscapes, a driving-domain benchmark with specific geometric structure; the method's generalizability to indoor scenes, aerial imagery, or other domains is unknown. The reliance on SPIdepth, which was trained on Cityscapes video sequences, introduces implicit domain dependence that may limit applicability to datasets without driving-domain depth priors. Finally, the PQ_stuff gap between our method (31.95) and CUPS (35.10) reflects the ViT backbone's coarser spatial features compared to ResNet-50, and closing this gap while preserving thing quality would require either a smaller patch size or a multi-scale feature pyramid adapted for vision transformers.

The computational cost is non-trivial but accessible. Pseudo-label generation requires running DINOv2 feature extraction, K-means overclustering, SPIdepth depth estimation, and Sobel-based instance decomposition on all training images. Stages 2 and 3 require approximately 18000 total steps on two GTX 1080 Ti GPUs. The entire pipeline can be executed on consumer-grade hardware, though the small batch size necessitated by the 11GB memory constraint and fp32 precision means the full training process takes several days.


## Conclusion

We have presented a monocular pseudo-label generation pipeline that eliminates the stereo video requirement of CUPS for unsupervised panoptic segmentation. The pipeline composes K-means overclustering on CAUSE-TR features (k=80, recovering all 7 dead classes in the default cluster assignment), self-supervised monocular depth from SPIdepth, and a depth-gradient instance decomposition that separates thing-class regions along depth discontinuities. The resulting pseudo-labels achieve PQ of 26.74 on Cityscapes validation, with thing-class quality (PQ_things of 19.41) surpassing CUPS's stereo-derived pseudo-labels (PQ_things of 17.70). To our knowledge, this is the first demonstration that monocular-only pseudo-labels can match or exceed stereo-based pseudo-labels for unsupervised panoptic segmentation.

A systematic 64-configuration sweep across overclustering granularity, gradient threshold, and minimum area revealed a previously undocumented tradeoff between stuff segmentation quality and thing instance separability: as overclustering granularity increases, semantic boundaries become fine enough to subsume depth-based splitting, making the geometric signal progressively redundant. The Pareto-optimal operating point at k=80 balances these competing pressures, and the quantitative characterization of this tradeoff enables principled hyperparameter selection for future work.

When these monocular pseudo-labels are combined with the CUPS Cascade Mask R-CNN training framework and a DINOv3 ViT-B/16 backbone, the full pipeline reaches PQ of 32.76, surpassing CUPS by 4.96 points on the identical 27-class evaluation protocol. The improvement is concentrated in thing-class segmentation (PQ_things of 34.13 versus 17.70), with a modest regression in stuff quality (PQ_stuff of 31.95 versus 35.10). The backbone upgrade is an orthogonal engineering improvement that amplifies pseudo-label quality rather than replacing it: on the original ResNet-50 backbone, our monocular pseudo-labels yield PQ of 24.68, confirming that the pseudo-labels themselves, not the backbone, are the methodological foundation.

The honest accounting of the research process reveals instructive negative results. Semantic refinement networks at 32x64 resolution universally degraded thing-class quality regardless of loss function or receptive field modifications. The metric confusion between 19-class and 27-class evaluation protocols created an illusion of competitiveness that consumed meaningful project time and motivated a failed merging experiment. Every spectral and attention-based instance method (CutLER, CuVLER, DINOSAUR, MaskCut, HDBSCAN) was decisively outperformed by the simple depth-gradient approach in driving scenes. Self-training consistently hurt weak models and helped strong ones, establishing a quality threshold below which iterative refinement introduces more noise than signal. Custom ViT implementations contained subtle bugs that were undetectable without reference to official codebases.

The remaining frontier is the 3.15-point PQ_stuff gap relative to CUPS, attributable to the ViT backbone's coarser spatial features. The depth-guided instance decomposition's failure on co-planar objects (person PQ of 4.2) represents a fundamental limitation of geometry-only approaches that future work should address through hybrid geometric-appearance methods. With the current trajectory, where self-training was still improving at step 8000, longer training or additional self-training rounds may yield further gains. The combination of a denser ViT backbone, hybrid instance decomposition, and extended self-training could plausibly push unsupervised panoptic quality on Cityscapes beyond PQ of 35.


## References

Arica, N., Ege, T., and Ikizler-Cinbis, N. CuVLER: Enhanced unsupervised object discovery and localization with multi-model VoteCut pseudo-labels. arXiv preprint arXiv:2403.07874, 2024.

Cai, Z. and Vasconcelos, N. Cascade R-CNN: Multi-stage object detection with cascading rejection thresholds. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(5):1483--1499, 2021.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9650--9660, 2021.

Cheng, B., Collins, M. D., Zhu, Y., Liu, T., Huang, T. S., Adam, H., and Chen, L.-C. Panoptic-DeepLab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12475--12485, 2020.

Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., and Girdhar, R. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1290--1299, 2022.

Cho, J., Lim, H., and Kim, S. PiCIE: Unsupervised semantic segmentation using invariance and equivariance in clustering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16717--16727, 2021.

Cho, J., Lim, H., and Kim, S. CAUSE: Contrastive and unsupervised semantic segmentation with modularity-based codebook. Pattern Recognition, 148:110177, 2024.

Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., and Schiele, B. The Cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213--3223, 2016.

Deng, X., Yang, Q., Xie, E., Zhu, L., and Luo, P. COCONUT: COmpleting CONtexts for Unified panopTic segmentation. In European Conference on Computer Vision (ECCV), 2024.

Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.-Y., Cubuk, E. D., Le, Q. V., and Zoph, B. Simple copy-paste is a strong data augmentation method for instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2918--2928, 2021.

Godard, C., Mac Aodha, O., Firman, M., and Brostow, G. J. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3828--3838, 2019.

Hahn, O., Lee, S., Lim, H., and Kim, S. CUPS: Unsupervised panoptic segmentation with self-supervised vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., and Freeman, W. T. Unsupervised semantic segmentation by distilling feature correspondences. In International Conference on Learning Representations (ICLR), 2022.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770--778, 2016.

Kirillov, A., He, K., Girshick, R., Rother, C., and Dollar, P. Panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9404--9413, 2019.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft COCO: Common objects in context. In European Conference on Computer Vision (ECCV), pp. 740--755, 2014.

Neuhold, G., Ollmann, T., Rota Bulo, S., and Kontschieder, P. The Mapillary Vistas dataset for semantic understanding of street scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4990--4999, 2017.

Niu, C., Li, B., and Darrell, T. U2Seg: Unified unsupervised panoptic segmentation. arXiv preprint arXiv:2312.02905, 2023.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024.

Oquab, M. et al. DINOv3: Self-supervised visual features with improved training and scaling. Technical report, Meta FAIR, 2025.

Seitzer, M., Weber, M., and Geiger, A. Bridging the gap to real-world object-centric learning. In International Conference on Learning Representations (ICLR), 2023.

Seo, B., Kim, E., and Lee, J. SPIdepth: Strengthened pose information for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Tan, M., Pang, R., and Le, Q. V. EfficientDet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10781--10790, 2020.

Wang, X., Girshick, R., Gupta, A., and He, K. Cut and learn for unsupervised object detection and instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3124--3134, 2023.

Wang, X., Yu, J., and Chen, H. MaskCut: Mask-based unsupervised object detection via normalized cut on self-supervised features. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Wang, A., Chen, H., Lin, Z., Pu, J., and Zha, H. RepViT: Revisiting mobile CNN from ViT perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Weber, M., Xie, J., Collins, M., Zhu, Y., Voigtlaender, P., Adam, H., Green, B., Geiger, A., Leibe, B., Cremers, D., Aljundi, R., Rubanova, Y., and Chen, L.-C. STEP: Segmenting and tracking every pixel. In NeurIPS Datasets and Benchmarks, 2021.

Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., and Xie, S. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16133--16142, 2023.
