---
section: "2 Related Work"
status: locked v3 (2026-04-29) — six-paragraph polished shape
paragraphs: 6
words: ~735 (hard cap 800)
audit_history:
  - v1 first survey-dump draft
  - v2 trimmed and retitled
  - v3 user's six-paragraph prescriptive shape; survey-dump removed; PQ ladder pushed to §4
inviolable_wording:
  - "Unlike these prior unsupervised depth-guided methods that use depth as a loss-side signal or sampling prior, DCFA conditions a frozen unsupervised semantic code on monocular depth through a small residual adapter."
  - "depth gradients provide a 2-D proxy for the separation cue CutS3D obtains after unprojecting depth to 3-D, namely local discontinuities between physically distinct surfaces"
  - "Our setting differs from these concurrent monocular alternatives by generating pseudo-labels from target-dataset monocular frames alone, without stereo/video capture, explicit 3-D reconstruction, or SA-1B-scale segmentation pretraining."
  - "We inherit CUPS-style panoptic bootstrapping and panoptic network training, and replace only the pseudo-label generator."
  - "pseudo-label-time stereo and motion cues" (NOT "stereo supervision")
forbidden_in_section_2:
  - 90-D code, 16-D sinusoidal, concat 106-D input, ~225K params (225,114), λ_preserve (→ §3 Method)
  - Sobel kernel, τ_d, A_min, dilation iterations (→ §3 Method)
  - τ_sim = 0.85, SIMCF-A/B/C definitions (→ §3 Method)
  - DINOv3 specifics (→ §3 / §4)
  - PQ ladder numbers (→ §4)
  - "First / to our knowledge first" claims (→ §1 contributions)
---

# 2 Related Work

Unsupervised Semantic Segmentation. Early unsupervised semantic segmentation methods optimized clustering objectives over learned image representations: IIC [Ji et al., 2019] maximized mutual information between augmented views, and PiCIE [Cho et al., 2021] enforced photometric and geometric invariance to recover pixel-level groupings. The arrival of self-supervised vision transformers [Caron et al., 2021; Oquab et al., 2024] reshaped the field, with dense ViT features clustering cleanly enough that distillation methods became dominant. STEGO [Hamilton et al., 2022] distilled DINO correspondences into compact dense embeddings; HP [Seong et al., 2023] improved this contrastive pool by discovering hidden positives; CAUSE [Kim et al., 2024] structured the latent space through a concept clusterbook, producing a concept-coherent semantic code. Recent variants refine the clustering objective on top of these SSL features but do not change what is being clustered: spectral aggregation in EAGLE [Kim et al., 2024b], proxy-anchor mining in PPAP [Seong et al., 2024], diffusion-feature partitioning in DiffCut [Couairon et al., 2024], and recursive hierarchy clustering [Bonnet et al., 2024].

Monocular depth has been considered as an auxiliary signal throughout this lineage. DepthG [Sick et al., 2024] adds a depth-feature correlation loss together with depth-guided positive/negative sampling on top of a STEGO-style backbone, while CUPS [Hahn et al., 2025] uses depth as a contrastive auxiliary in its DINO-distillation semantic head. Earlier work used depth as a multi-task pre-training signal [Hoyer et al., 2021]. In a separate, fully *supervised* line, RGB-D fusion methods inject depth into attention but require dense labels (DFormerv2 [Yin et al., 2025]). Unlike these prior unsupervised depth-guided methods that use depth as a loss-side signal or sampling prior, DCFA conditions a frozen unsupervised semantic code on monocular depth through a small residual adapter, leaving the backbone untouched.

Unsupervised Instance Segmentation. The dominant line of unsupervised instance discovery applies graph-cut-style partitioning to self-supervised features. LOST [Siméoni et al., 2021] localized a single salient object per image; TokenCut [Wang et al., 2023a] generalized this through normalized cuts on DINO patch tokens; MaskCut [Wang et al., 2023b] iterated TokenCut to extract multiple masks per image; CutLER [Wang et al., 2023b] turned MaskCut pseudo-masks into supervision for a class-agnostic Cascade Mask R-CNN, establishing the cut-and-learn template. MaskDistill [Van Gansbeke et al., 2022] and FreeSOLO [Wang et al., 2022] explored alternative pseudo-mask formulations within the same paradigm.

Recent extensions diversify the cut-and-learn template. CuVLER [Arica et al., 2024] strengthens MaskCut with multi-SSL feature voting and soft-target distillation; COLER [Feng et al., 2025] cuts in a single pass without k-means; ProMerge [Li et al., 2024] introduces prompt-and-merge grouping with background-based pruning; unMORE [Yang et al., 2025] learns existence/center/boundary fields; UnSAM and UnSAMv2 [Wang et al., 2024; Yu et al., 2025] add granularity-controlled segmentation on SA-1B-scale data. Slot-attention models such as DINOSAUR [Seitzer et al., 2023] and its motion-refined extension MR-DINOSAUR [Gong et al., 2025] are adjacent alternatives that largely target object-centric or class-agnostic instance discovery rather than the instance branch of scene-centric panoptic pseudo-labeling.

A separate subline injects geometric information explicitly, and CutS3D [Sick et al., 2025] is the closest foil to our instance pipeline. CutS3D lifts each image to a 3-D point cloud through monocular depth, constructs a 3-D k-NN graph, and applies LocalCut to disambiguate adjacent instances that 2-D NCut merges. We share CutS3D's monocular premise but stay entirely in 2-D: depth gradients provide a 2-D proxy for the separation cue CutS3D obtains after unprojecting depth to 3-D, namely local discontinuities between physically distinct surfaces. After Sobel filtering and connected-component analysis within thing-eligible semantic regions, this 2-D treatment supplies the instance side of our pseudo-labels without lifting to a point cloud or constructing a 3-D graph.

Unsupervised Panoptic Segmentation. Unsupervised panoptic segmentation is a young task. The most directly comparable Cityscapes results to date come from three works. U2Seg [Niu et al., 2024] is the first unified framework, composing STEGO and MaskCut on object-centric data and transferring to scene-centric Cityscapes; it inherits the dominant-foreground bias of MaskCut and reaches roughly 18 PQ. CUPS [Hahn et al., 2025] is the current scene-centric state of the art and constructs pseudo-labels from pseudo-label-time stereo and motion cues: stereo video, unsupervised optical flow [Stone et al., 2021], and SF2SE3 scene-flow segmentation [Sommer et al., 2022], then trains a Cascade Mask R-CNN with DropLoss bootstrapping and three rounds of EMA self-training. S2-UniSeg [Xu et al., 2025] is a concurrent monocular alternative that performs well but relies on large-scale segmentation-oriented pretraining on SA-1B before adaptation. Our setting differs from these concurrent monocular alternatives by generating pseudo-labels from target-dataset monocular frames alone, without stereo/video capture, explicit 3-D reconstruction, or SA-1B-scale segmentation pretraining. We inherit CUPS-style panoptic bootstrapping and panoptic network training, and replace only the pseudo-label generator.
