# Unsupervised panoptic segmentation via Mamba-bridged depth-semantic-instance fusion

**Fusing DepthG's depth-guided semantic segmentation with CutS3D's 3D-aware instance segmentation through a Mamba state-space bridge represents a viable and novel path to NeurIPS 2026.** The approach exploits a critical gap: no existing work applies Mamba-based feature bridging to unsupervised segmentation, and only two prior methods (U2Seg, CUPS) tackle unsupervised panoptic segmentation at all. Both DepthG and CutS3D share DINO ViT-S/8 backbones and ZoeDepth estimators, creating strong architectural compatibility. The primary technical challenge lies in adapting MFuser's supervised VFM-VLM bridging mechanism to an unsupervised semantic-instance fusion setting with fundamentally different output representations. This report provides a complete architectural analysis, feasibility assessment, technical challenges map, incremental roadmap, and novelty positioning for the proposed research.

---

## Architectural foundations across all five papers

**Panoptic segmentation** as defined by Kirillov et al. maps each pixel *i* to a pair (l_i, z_i) ∈ L × ℕ, where l_i is the semantic class and z_i the instance ID. The label set partitions into stuff classes L_St (amorphous regions: sky, road) and thing classes L_Th (countable objects: cars, people). The Panoptic Quality metric decomposes as **PQ = SQ × RQ**, where SQ (Segmentation Quality) measures average IoU of matched segments and RQ (Recognition Quality) is the F1 score over segment matching at IoU > 0.5. This decomposition into PQ_St and PQ_Th per category is the evaluation framework the proposed method must target.

**CUPS** (CVPR 2025 Highlight) is the current state-of-the-art unsupervised panoptic method, achieving **PQ = 27.8 on Cityscapes**—a +9.4 point improvement over U2Seg's 18.4. Its three-stage pipeline generates pseudo-labels from stereo video (motion segmentation via SF2SE3 for instances, DepthG for semantics), trains a Panoptic Cascade Mask R-CNN, then self-trains with an EMA momentum network. CUPS classifies stuff versus things using motion frequency ratios: semantic pseudo-classes frequently appearing within motion-derived instance masks are labeled "things." A critical limitation is its **stereo video dependency** for pseudo-label generation, restricting it to driving datasets and excluding COCO-Stuff-27, ADE20K, and PASCAL-Context.

**MFuser** (CVPR 2025 Highlight) bridges frozen VFM (DINOv2-L) and VLM (EVA02-CLIP-L) encoders through MVFuser modules inserted at every transformer block layer. The mechanism concatenates patch tokens from both encoders [x_VFM; x_VLM], applies a bottleneck projection, processes through dual branches—a sequential SSM path and a spatial convolution path—then gates their outputs via element-wise multiplication before residual addition. With only **1.67M parameters and 17.21 GFLOPs** per adapter (versus 4.20M/98.64G for self-attention), MVFuser achieves **68.20 average mIoU** on synthetic-to-real DGSS benchmarks, outperforming all alternatives including cross-attention adapters.

**DepthG** (CVPR 2024) extends STEGO by incorporating a depth-feature correlation loss and farthest-point sampling on ZoeDepth-derived point clouds. Depth serves purely as a training-time loss signal—the model runs on RGB alone at inference. The architecture outputs a **90-dimensional code space** (s ∈ ℝ^{90×H×W}) from a DINO ViT-S/8 backbone (C=384), which is then clustered via k-means and refined with a CRF. **CutS3D** (ICCV 2025) performs unsupervised instance segmentation by applying Normalized Cut on DINO feature affinity graphs, then separating instances via LocalCut on 3D point clouds constructed from ZoeDepth predictions. Its output is class-agnostic instance masks with confidence scores, trained into a Cascade Mask R-CNN through self-training rounds.

---

## Fusion feasibility is high due to shared infrastructure

The compatibility between DepthG and CutS3D for Mamba-based fusion is remarkably strong, driven by three shared components. Both use **DINO ViT-S/8** with identical C=384 feature dimensions and 8×8 patch sizes. Both use **ZoeDepth** for monocular depth estimation. Both convert depth to point clouds for 3D operations. Both are **depth-free at inference time**. Both apply CRF post-processing. Notably, both papers share the same first author (Leon Sick) and research group at Ulm University.

The representation gap is structural but complementary. DepthG produces dense per-pixel semantic label maps through its 90-dimensional code space, while CutS3D produces sparse per-instance binary masks with bounding boxes and confidence scores. This is precisely the semantic-plus-instance decomposition that panoptic segmentation requires. For naive panoptic assembly, one would assign DepthG semantic labels to CutS3D instances via majority voting within each mask, then use DepthG's semantic map directly for stuff regions.

The deeper question is whether **Mamba-based feature fusion** can improve upon this naive assembly. MFuser's MVFuser was designed to bridge VFM and VLM token sequences—features with the same spatial structure but different semantic content. Adapting this for semantic-instance fusion requires rethinking what the two "branches" represent. The semantic branch produces dense per-patch class-probability features (90-dim code space), while the instance branch produces per-patch affinity features used for graph cutting (384-dim DINO features). A projection layer mapping the 90-dim semantic code to 384-dim (or a shared intermediate dimension) would enable MVFuser-style concatenation and joint processing. The key insight is that Mamba's selective scan can learn to propagate semantic class information into instance boundary decisions and vice versa, potentially resolving ambiguities that neither branch handles alone.

---

## Seven critical technical challenges for the fusion

**Feature dimension mismatch** is the most immediate engineering challenge. DepthG's 90-dimensional code space versus CutS3D's 384-dimensional backbone features requires learned projection layers. The MVFuser bottleneck design naturally handles this: project both to a shared dimension d_bridge before concatenation, process through the SSM, then project back to each branch's native dimension. Ablation should test d_bridge ∈ {64, 128, 256}.

**Training dynamics conflict** presents a deeper problem. DepthG optimizes a contrastive feature distillation loss (L_STEGO + λ_DepthG · L_DepthG) that encourages compact semantic clusters, while CutS3D optimizes instance detection losses (spatial-confidence-weighted BCE + DropLoss) that encourage precise boundary delineation. Joint optimization risks one loss dominating the other. A curriculum strategy—first training semantic features to convergence, then introducing instance losses with the Mamba bridge—mirrors CUPS's staged approach and should stabilize training.

**Stuff-things disambiguation without motion** is perhaps the hardest conceptual challenge. CUPS uses motion (optical flow + scene flow) to distinguish stuff from things—objects that move are "things." Without stereo video, this signal vanishes. Alternative cues include: (a) depth discontinuity frequency—things tend to have sharp depth boundaries while stuff regions are spatially contiguous at similar depths; (b) DINO feature cluster compactness—thing classes form tighter clusters in feature space; (c) CutS3D's instance mask statistics—regions frequently decomposed into multiple instances are likely things. A learned stuff-things classifier operating on these signals could replace CUPS's motion-based approach.

**Depth consistency** across semantic and instance predictions requires that depth information coherently guides both branches. Since both already use ZoeDepth, a shared depth-conditioning module that injects depth features into the Mamba bridge would enforce consistency. The depth-feature correlation from DepthG and the spatial importance sharpening from CutS3D could be unified into a single depth-aware attention mechanism within the MVFuser.

**Memory and compute constraints** differ significantly between GPU and TPU. MFuser's `mamba_ssm` library (v2.2.2) uses custom CUDA kernels optimized for GPU execution. For **GPU deployment**, a single A5000 (24GB) suffices for MFuser training; the combined DepthG+CutS3D+Mamba system should fit on an A100 (80GB) with batch size 2-4. For **TPU deployment**, the hardware-aware parallel scan in Mamba relies on CUDA-specific memory hierarchy optimizations (SRAM tiling) that don't directly translate to TPU's systolic array architecture. The Mamba-2/SSD formulation, which converts SSM operations to matrix multiplications, is more TPU-friendly since it leverages tensor cores. Implementing the MVFuser with Mamba-2's chunk-based computation on TPU via JAX/Flax should be prioritized over porting Mamba-1's CUDA kernels.

**Scan direction for semantic-instance fusion** requires careful design. MFuser places VFM tokens before VLM tokens, exploiting Mamba's causal scan to let fine-grained visual features influence text-aligned features. For semantic-instance fusion, two orderings are worth testing: (a) semantic-first [s_semantic; f_instance], allowing class identity to inform instance boundary refinement; (b) instance-first [f_instance; s_semantic], allowing boundary information to sharpen semantic predictions. Bidirectional scanning (processing both orderings and summing) may capture both directions of information flow.

**Pseudo-label quality ceiling** bounds overall performance. DepthG's semantic pseudo-labels and CutS3D's instance pseudo-masks both have noise and errors. The Mamba bridge must be robust to this noise, potentially through confidence-weighted loss terms (analogous to CutS3D's spatial confidence maps) and consistency regularization between the two branches.

---

## A five-phase roadmap targeting NeurIPS 2026

**Phase 1 (Weeks 1-4): Baseline reproduction and component analysis.** Reproduce DepthG on COCO-Stuff-27 and Cityscapes using the official codebase. Reproduce CutS3D's pseudo-mask generation on ImageNet-1K and evaluate instance AP on COCO. Build a naive panoptic assembly baseline: run DepthG for semantic labels, CutS3D for instance masks, merge via majority voting within instance regions. Evaluate with PQ, PQ_St, PQ_Th on COCO-Stuff-27 and Cityscapes. This establishes the performance floor that the Mamba bridge must surpass. Expected timeline: 4 weeks on 4×A100 cluster.

**Phase 2 (Weeks 5-8): Feature alignment experiments.** Design and test projection layers mapping DepthG's 90-dim code space to CutS3D's 384-dim feature space and vice versa. Implement a shared DINO backbone serving both branches simultaneously. Test whether DepthG's depth-feature correlation loss improves CutS3D's semantic affinity matrix (replacing or augmenting the cosine similarity W_{i,j} with depth-correlated features). Measure instance AP and semantic mIoU independently to verify that shared features don't degrade either branch.

**Phase 3 (Weeks 9-14): Mamba bridge adaptation.** Port MFuser's MVFuser architecture from the DGSS setting to the semantic-instance fusion setting. Replace the VFM/VLM encoder pair with the semantic/instance feature pair. Key design decisions to implement and ablate:

- **Concatenation dimension**: semantic tokens before instance tokens versus interleaved
- **Bridge depth**: MVFuser at every DINO block versus only at final layers
- **Stuff-things classifier**: implement depth-discontinuity + cluster-compactness + instance-frequency features; train a lightweight MLP classifier
- **Mamba variant**: test Mamba-1 (selective scan) versus Mamba-2 (SSD with matrix multiplications) for TPU compatibility
- **Bidirectional scanning**: forward + reverse scan fusion versus single direction

Concrete ablation experiments for this phase should include: (i) MVFuser versus cross-attention versus simple concatenation for feature fusion, measuring PQ improvement per additional compute; (ii) number of MVFuser insertion points (every block versus every 2/4/6 blocks); (iii) bottleneck dimension sweep (64, 128, 256); (iv) scan direction ablation (semantic-first, instance-first, bidirectional); (v) with/without depth conditioning in the bridge.

**Phase 4 (Weeks 15-20): Joint training and loss design.** Design the unified loss function combining:

- L_semantic: depth-feature correlation loss from DepthG (contrastive, with guidance scheduling)
- L_instance: spatial-confidence-weighted BCE from CutS3D
- L_consistency: a novel cross-branch consistency loss enforcing that semantic predictions within an instance mask are uniform, and instance boundaries align with semantic boundaries
- L_panoptic: direct PQ-proxy loss operating on the merged panoptic output

The consistency loss L_consistency is a key novelty opportunity. Define it as: L_consistency = Σ_k H(semantic_labels | instance_mask_k), measuring the entropy of semantic label distributions within each predicted instance mask. Low entropy means each instance has a single dominant semantic class—the desired behavior. This loss directly bridges the semantic and instance branches through the Mamba fusion without requiring panoptic ground truth.

**Phase 5 (Weeks 21-26): Ablation studies and benchmark evaluation.** Full evaluation on all four target benchmarks (COCO-Stuff-27, Cityscapes, ADE20K, PASCAL-Context) using PQ, PQ_St, PQ_Th, SQ, RQ, plus semantic mIoU and instance AP for component analysis. Key ablation matrix:

- Component contribution: full method versus remove Mamba bridge versus remove depth guidance versus remove instance branch
- Stuff-things classifier accuracy versus oracle stuff-things labels
- Scaling: ViT-S/8 versus ViT-B/8 backbone
- Self-training rounds: 0, 1, 2, 3 rounds
- Compute/accuracy tradeoff: Mamba versus cross-attention versus concatenation at matched FLOPs

---

## Three distinct novelty contributions position this work

**First: Mamba-based unsupervised panoptic segmentation is entirely unexplored.** The literature survey confirms that all existing Mamba segmentation papers (Vision Mamba, VMamba, U-Mamba, SegMamba, Sigma, MambaVision) operate in supervised settings. No prior work applies Mamba to unsupervised segmentation of any kind. Similarly, MFuser bridges VFMs and VLMs for supervised DGSS—adapting this mechanism to bridge semantic and instance features in an unsupervised setting is novel. The combination of "Mamba + unsupervised + panoptic" occupies a completely empty cell in the method taxonomy.

**Second: a unified depth-guided framework for both stuff and things without stereo video.** CUPS requires stereo video for motion-based stuff-things separation, limiting it to driving datasets. The proposed method uses monocular depth (ZoeDepth) as the unifying geometric signal for both semantic grouping (via DepthG's depth-feature correlation) and instance separation (via CutS3D's LocalCut on 3D point clouds). A novel depth-based stuff-things classifier—using depth boundary statistics, feature cluster compactness, and instance decomposition frequency—eliminates the stereo requirement. This enables evaluation on COCO-Stuff-27, ADE20K, and PASCAL-Context, where CUPS cannot operate.

**Third: cross-branch consistency losses for unsupervised panoptic quality.** The semantic-instance consistency loss L_consistency (entropy of semantic labels within instance masks) provides direct gradient signal connecting the two branches through the Mamba bridge, without requiring panoptic annotations. Additional novel constraints include: (a) **depth-boundary alignment loss** enforcing that both semantic boundaries and instance boundaries coincide with depth discontinuities; (b) **mutual refinement loss** where high-confidence semantic predictions supervise uncertain instance boundaries and vice versa; (c) **feature cycle-consistency** requiring that semantic features reconstructed from instance features via the Mamba bridge match the original semantic features.

---

## The competitive landscape and target performance

The proposed method competes against three categories of baselines. Against **unsupervised panoptic methods**, CUPS (PQ=27.8 on Cityscapes) and U2Seg (PQ=18.4) set the bar. Against **unsupervised semantic methods**, DynaSeg (mIoU≈30.5 on COCO-Stuff-27), EAGLE, and DepthG define the semantic component ceiling. Against **unsupervised instance methods**, CutS3D (AP=10.7 on COCO val2017) and CutLER (AP=9.7) define the instance component ceiling.

The target performance should exceed CUPS on Cityscapes (PQ > 28) while also providing competitive results on COCO-Stuff-27 and PASCAL-Context where CUPS cannot operate due to its stereo requirement. A realistic target is **PQ = 20-25 on COCO-Stuff-27** (establishing the first unsupervised panoptic benchmark on this dataset) and **PQ > 30 on Cityscapes** (surpassing CUPS without stereo video). The Mamba bridge's efficiency advantage—**2.5× fewer parameters and 5.7× fewer FLOPs** than attention-based fusion—provides a strong computational story regardless of absolute PQ improvements.

For implementation, the full pipeline should train on **4×A100 GPUs in under 24 hours** (CUPS trains in ~10 hours on 4×A100). The Mamba bridge adds minimal overhead: MVFuser requires only 1.67M additional parameters per insertion point. MFuser's full training takes just 15 hours on a single A5000 (24GB). TPU deployment should use the Mamba-2/SSD formulation implemented in JAX, converting selective scans to chunk-based matrix multiplications that map efficiently to TPU systolic arrays.

---

## Conclusion

This research direction is well-positioned for NeurIPS 2026 because it targets three simultaneous gaps in the literature: Mamba applied to unsupervised segmentation, monocular-depth-only unsupervised panoptic segmentation, and learned feature bridging between semantic and instance branches. The shared DINO+ZoeDepth infrastructure between DepthG and CutS3D makes the engineering path tractable, while the MVFuser architecture provides a proven, efficient fusion mechanism requiring minimal adaptation. The strongest risk factor is the stuff-things disambiguation without motion cues—this represents the core research question the paper must convincingly answer. The strongest novelty factor is the cross-branch consistency loss through the Mamba bridge, which provides a principled unsupervised signal for panoptic coherence. If the depth-based stuff-things classifier achieves even 70-80% accuracy relative to oracle labels, the overall pipeline should substantially advance unsupervised panoptic segmentation beyond the current state of the art.