# Graph Report - .  (2026-04-13)

## Corpus Check
- 1998 files · ~15,308,451 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 17391 nodes · 31707 edges · 441 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 7034 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `PQProxyLoss` - 156 edges
2. `Segment_TR` - 145 edges
3. `Mamba2Stack` - 141 edges
4. `Attack` - 140 edges
5. `DINOViTS8` - 139 edges
6. `AdaptiveProjectionBridge` - 138 edges
7. `InstanceHead` - 134 edges
8. `EMAState` - 130 edges
9. `TrainingCurriculum` - 126 edges
10. `Mamba2Block` - 124 edges

## Surprising Connections (you probably didn't know these)
- `Dataset loaders for MBPS.  Supports Cityscapes, COCO-Stuff-27, and NYU Depth V2.` --uses--> `Transform`  [INFERRED]
  mbps_pytorch/data/datasets.py → refs/refs/ocl/transforms.py
- `NOTE: this interface is experimental.         Args:             in_channels: cha` --uses--> `PositionEmbeddingSine`  [INFERRED]
  refs/cutler/videocutler/mask2former/modeling/transformer_decoder/maskformer_transformer_decoder.py → mbps_pytorch/models/mask2former/position_encoding.py
- `Compute Sobel dx, dy on a 2D depth map.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py
- `Load a single sample from val set.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py
- `Run inference and instrument every stage.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (480): attn_vis(), cls_padding(), grid_show(), highlight_grid(), grid_size=14是因为patch_size=16, 所以有224/16=14个patch     grid_index代表的是第几个grid,这个gri, visualize_grid_to_grid(), visualize_grid_to_grid_with_cls(), visualize_heads() (+472 more)

### Community 1 - "Community 1"
Cohesion: 0.01
Nodes (568): BidirectionalCrossModalScan, __call__(), deinterleave_tokens(), interleave_tokens(), Bidirectional Cross-Modal Scan (BiCMS).  Implements bidirectional Mamba2 scannin, Apply BiCMS fusion.          Args:             semantic: Semantic tokens (B, N,, Get L2 norms of internal SSM states for regularization.          This is called, Get L2 norms of internal SSM states for regularization.          This is called (+560 more)

### Community 2 - "Community 2"
Cohesion: 0.01
Nodes (369): APGD, r"""     APGD in the paper 'Reliable evaluation of adversarial robustness with a, r"""         Overridden., APGDT, r"""     APGD-Targeted in the paper 'Reliable evaluation of adversarial robustne, r"""         Overridden., Attack, Attack (+361 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (335): ABC, accuracy(), Accuracy calculation module., Module to calculate the accuracy.          Args:             topk (tuple, option, Calculate accuracy according to the prediction and target.      Args:         pr, Forward function to calculate accuracy.          Args:             pred (torch.T, ADE20KDataset, ADE20K dataset.      In segmentation map annotation for ADE20K, 0 stands for bac (+327 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (347): ADE20K, _file_to_segmentation_path(), _load_file_paths(), _load_segmentation(), ADE20KPanoptic, ADE20KSemantic, _Split, BaseImageProcessor (+339 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (398): Accumulator, _cat_and_gather_tensor_list(), NoOpAccumulator, Accumulate predictions and targets across processes, ResultsAccumulator, adaptive_edge_instances(), Adaptive Depth-Feature Edge Fusion for instance decomposition.  Uses depth Sobel, Instance decomposition via adaptive depth-feature edge fusion.      Feature edge (+390 more)

### Community 6 - "Community 6"
Cohesion: 0.01
Nodes (227): MVFuser, hidden_states: (B, L, D)         Returns: same shape as hidden_states, BaseTransformerLayer, Config, BlockChunk, DinoVisionTransformer, Args:             img_size (int, tuple): input image size             patch_size, Attention (+219 more)

### Community 7 - "Community 7"
Cohesion: 0.01
Nodes (354): ac_compile_parallelize_and_init(), Order of the wrappers:     1/ Activation checkpointing on blocks     2/ Compile, compute_distributions(), main(), process_city(), Compute per-class pixel distributions required by CUPS PseudoLabelDataset., Process all images in a single city., build_model_and_tokenizer() (+346 more)

### Community 8 - "Community 8"
Cohesion: 0.01
Nodes (316): CityscapesSeg, Coco, Coco171, Coco81, ContrastiveSegDataset, CroppedDataset, dataloader(), Dataset (+308 more)

### Community 9 - "Community 9"
Cohesion: 0.01
Nodes (175): DataAugmentationDINO, GaussianBlur, Apply Solarization to the PIL image., Apply Gaussian Blur to the PIL image., Solarization, DataAugmentationDINO, bilateral_solver_output(), BilateralGrid (+167 more)

### Community 10 - "Community 10"
Cohesion: 0.01
Nodes (282): DatasetWithEnumeratedTargets, If pad_dataset is set, pads based on torch's DistributedSampler implementation,, infer_type(), parse_unknown(), convert_path_or_url_to_url(), dinov2_vitb14(), dinov2_vitb14_reg(), dinov2_vitg14() (+274 more)

### Community 11 - "Community 11"
Cohesion: 0.01
Nodes (183): ConvFFN, deform_inputs(), DINOv3_Adapter, drop_path(), DropPath, DWConv, Extractor, get_reference_points() (+175 more)

### Community 12 - "Community 12"
Cohesion: 0.01
Nodes (180): my_app(), _random_crops(), RandomCropComputer, Crop the given image into four corners and the central crop.     If the image is, Crop the given image into four corners and the central crop.     If the image is, bit_get(), build_dataloader(), CityscapesSeg (+172 more)

### Community 13 - "Community 13"
Cohesion: 0.01
Nodes (204): RandomCrop, This class implements random crop augmentation given a Detectron2 input batch., Args:             batch (List[Dict[str, Tensor | Instances]]): Detectron2 panopt, build_batch_data_loader(), build_detection_test_loader(), build_detection_train_loader(), build_lr_scheduler(), build_model() (+196 more)

### Community 14 - "Community 14"
Cohesion: 0.01
Nodes (166): ChainedGenerator, DictMapper, patched_pathsplit(), Patches `torchdata` for behavior to be consistent with webdatasets., Split a path into a WebDataset prefix and suffix.      The version of pathsplit, Simple interface to allow chaining via a generator function.      This mirrors f, _collect_fields(), DummyDataModule (+158 more)

### Community 15 - "Community 15"
Cohesion: 0.01
Nodes (218): evaluate_results(), load_cups_model(), load_unet_model(), main(), merge_panoptic(), Run UNet semantic inference on validation images., Merge semantic and instance predictions into panoptic maps., Evaluate panoptic predictions against ground truth. (+210 more)

### Community 16 - "Community 16"
Cohesion: 0.01
Nodes (203): CustomCascadeROIHeads, _match_and_label_boxes(), Args:             features, targets: the same as in                 Same as in :, Args:             features, targets: the same as in                 Same as in :, Args:             features, targets: the same as in                 Same as in :, # NOTE: confidence score, # NOTE: maximum overlapping with GT (IoU), # NOTE: confidence score (+195 more)

### Community 17 - "Community 17"
Cohesion: 0.02
Nodes (150): chunk_gated_delta_rule(), GatedDeltaNet, Sequential gated delta rule (pure PyTorch, for testing).      Step-by-step recur, GatedDeltaNet layer with pure PyTorch backend.      Drop-in replacement for Mamb, Chunked gated delta rule (pure PyTorch).      Uses WY representation for efficie, Forward pass.          Arguments:             u: (B, L, D) — input sequence, recurrent_gated_delta_rule(), backward() (+142 more)

### Community 18 - "Community 18"
Cohesion: 0.02
Nodes (176): BboxCorLocMetric, BboxRecallMetric, Metrics related to the evaluation of bounding boxes., Computes IoU metric for bounding boxes when correspondences to ground truth are, Compute IoU between two sets of bounding boxes.      Args:         pred_bboxes:, Update this metric.          Args:             prediction: Predicted mask of sha, unsupervised_bbox_iou(), UnsupervisedBboxIoUMetric (+168 more)

### Community 19 - "Community 19"
Cohesion: 0.02
Nodes (135): DepthDecoder, LiteDepthDecoder, DepthDecoderUncert, AvgPool, BNGELU, CDilated, Conv, DilatedConv (+127 more)

### Community 20 - "Community 20"
Cohesion: 0.01
Nodes (135): CopydaysDataset, extract_features(), ImgListDataset, Compute the average precision of one search.     ranks = ordered list of ranks o, score_ap_from_ranks_1(), OxfordParisDataset, extract_feature_pipeline(), extract_features() (+127 more)

### Community 21 - "Community 21"
Cohesion: 0.01
Nodes (127): BackboneWithPositionEncoding, BasicBlock, Bottleneck, build_backbone(), conv1x1(), conv3x3(), DINOBackbone, build_dinov2_vitb_fpn_backbone() (+119 more)

### Community 22 - "Community 22"
Cohesion: 0.02
Nodes (108): Attention, CausalSelfAttention, LinearKMaskedBias, MemEffAttention, rope_apply(), rope_rotate_half(), SelfAttention, add_residual() (+100 more)

### Community 23 - "Community 23"
Cohesion: 0.02
Nodes (139): CopyPasteAugmentation, PhotometricAugmentations, This class implements resolution jitter augmentation giving a Detectron2 input b, Constructor method.          Args:             scales (Tuple[float, ...]): Scale, This class implements resolution jitter augmentation giving a Detectron2 input b, Constructor method.          Args:             scales (Tuple[float, ...]): Scale, Args:             batch (List[Dict[str, Tensor | Instances]]): Detectron2 panopt, This class implements photometric augmentations. (+131 more)

### Community 24 - "Community 24"
Cohesion: 0.03
Nodes (121): build_augmentation(), get_label_efficient_augmentations(), get_pseudo_label_augmentations(), CustomAugmentation, FixedSizeCrop, get_output_shape(), get_transform(), __init__() (+113 more)

### Community 25 - "Community 25"
Cohesion: 0.02
Nodes (136): CKA_impl(), CKA_main(), CKA_vis(), cls_padding(), grid_show(), highlight_grid(), hsic(), grid_size=14是因为patch_size=16, 所以有224/16=14个patch     grid_index代表的是第几个grid,这个gri (+128 more)

### Community 26 - "Community 26"
Cohesion: 0.02
Nodes (99): AttractorLayer, AttractorLayerUnnormed, exp_attractor(), inv_attractor(), Attractor layer for bin centers. Bin centers are unbounded, Args:             x (torch.Tensor) : feature block; shape - n, c, h, w, Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a =, Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attract (+91 more)

### Community 27 - "Community 27"
Cohesion: 0.02
Nodes (107): cleanup_tmp_files(), CocoAnnotationsWorker, collect_to_single_ann_dict(), create_ann_for_single_image(), get_info(), get_license_info(), A worker class for handling temp coco annotations files, Flushes the current annotations to a temp file (+99 more)

### Community 28 - "Community 28"
Cohesion: 0.02
Nodes (95): benchmark_backbone(), main(), Lightweight FPN decoder for panoptic segmentation., Benchmark a single backbone + FPN decoder., SimpleFPNDecoder, sync_device(), ClipImageModel, ClipTextModel (+87 more)

### Community 29 - "Community 29"
Cohesion: 0.02
Nodes (79): BaseModule, Attention, Block, DWConv, mit_b0, mit_b1, mit_b2, mit_b3 (+71 more)

### Community 30 - "Community 30"
Cohesion: 0.02
Nodes (83): BaseTrainer, is_rank_zero(), Base Trainer class for training a model., BaseTrainer, angle_rots(), binary_dice_loss(), BinsChamferLoss, calc_census_loss() (+75 more)

### Community 31 - "Community 31"
Cohesion: 0.02
Nodes (73): Callback, FreezeParameters, Callback to update hyperparameter schedulers found `ocl.scheduling`., Freeze parameters of model prior to training., Sets environment variable `EPOCH` which is used by [ocl.transforms.SampleSlices], Initialize FreezeParameters callback.          Args:             parameter_group, Build parameter groups from specification., Restore a subset of parameters using a checkpoint form a different model. (+65 more)

### Community 32 - "Community 32"
Cohesion: 0.02
Nodes (80): AddBBoxFromInstanceMasks, AddEmptyBboxes, AddEmptyMasks, AddImageSize, AddSegmentationMaskFromInstanceMask, CanonicalizeBboxes, _cast_squeeze_in(), _cast_squeeze_out() (+72 more)

### Community 33 - "Community 33"
Cohesion: 0.03
Nodes (111): extract(), extract_sd_features(), global_clustering(), hungarian_miou(), load_coco_panoptic_gt(), LocalAffinity, LocalAffinityCopy, LocalStDev (+103 more)

### Community 34 - "Community 34"
Cohesion: 0.02
Nodes (55): BaseDataset, Returns a single training item from the dataset as a dictionary.          Values, Superclass for dataloaders, Resize colour images to the required scales and augment if required          We, BaseDataset, CityscapesEvalDataset, Cityscapes evaluation dataset - here we are loading the raw, original images rat, Convert index in the dataset to a folder name, frame_idx and any other bits (+47 more)

### Community 35 - "Community 35"
Cohesion: 0.02
Nodes (72): CombinedModel, Implementation of combined model., Core pytorch lightning model used for training, loss compuation and visualizatio, Initialize combined model.          Args:             models: The model to run t, evaluate(), EvaluationProbingConfig, extend_model(), ExtractDataFromPredictions (+64 more)

### Community 36 - "Community 36"
Cohesion: 0.03
Nodes (52): BASELINE, Forward function for training.         Args:             img (Tensor): Input ima, The iteration step during training.          This method defines an iteration st, calc_grad_magnitude(), DACS, The iteration step during training.          This method defines an iteration st, Forward function for training.          Args:             img (Tensor): Input im, make_data_loader() (+44 more)

### Community 37 - "Community 37"
Cohesion: 0.03
Nodes (101): align_a1_majority(), align_a2_selective(), align_a3_confidence(), align_a4_majority_stuff(), build_panoptic_with_precomputed_instances(), load_precomputed_instances(), main(), Build panoptic map using pre-computed instances for things.      - Stuff: each c (+93 more)

### Community 38 - "Community 38"
Cohesion: 0.03
Nodes (100): _add_ids(), colorize_mask(), depth2rgb(), disp2rgb(), draw_arrows_in_rgb(), draw_grid_arrows_in_rgb(), _draw_instance_contours(), draw_pixels() (+92 more)

### Community 39 - "Community 39"
Cohesion: 0.02
Nodes (70): COCOevalMaxDets, COCOEvaluator, _evaluate_box_proposals(), _evaluate_predictions_on_coco(), instances_to_coco_json(), Args:             inputs: the inputs to a COCO model (e.g., GeneralizedRCNN)., Args:             inputs: the inputs to a COCO model (e.g., GeneralizedRCNN)., Args:             img_ids: a list of image IDs to evaluate on. Default to None f (+62 more)

### Community 40 - "Community 40"
Cohesion: 0.03
Nodes (89): build_method_kwargs(), compute_pq_from_accumulators(), discover_files(), evaluate_panoptic_single(), expand_grid(), main(), Expand a parameter grid dict into a list of config dicts., Evaluate one image. Returns per-class (tp, fp, fn, iou) arrays. (+81 more)

### Community 41 - "Community 41"
Cohesion: 0.03
Nodes (46): aug_test(), BaseSegmentor, forward(), forward_train(), _parse_losses(), The iteration step during training.          This method defines an iteration st, The iteration step during validation.          This method shares the same signa, Base class for segmentors. (+38 more)

### Community 42 - "Community 42"
Cohesion: 0.03
Nodes (57): accuracy(), AnyMatchAccuracy, AveragingMethod, build_classification_metric(), build_topk_accuracy_metric(), build_topk_any_match_accuracy_metric(), build_topk_recall_metric(), ClassificationMetricType (+49 more)

### Community 43 - "Community 43"
Cohesion: 0.03
Nodes (49): chunk_scan(), backward(), chunk_scan(), _chunk_scan_bwd_dC(), _chunk_scan_bwd_dcb(), _chunk_scan_bwd_ddAcs_unstable(), _chunk_scan_bwd_dstates(), _chunk_scan_bwd_dx() (+41 more)

### Community 44 - "Community 44"
Cohesion: 0.05
Nodes (47): AutoregressivePatchDecoder, BBoxOutput, build_grid_of_positions(), DensityPredictingSlotAttentionDecoder, DepthReconstructionOutput, DVAEDecoder, get_dvae_decoder(), get_dvae_encoder() (+39 more)

### Community 45 - "Community 45"
Cohesion: 0.03
Nodes (70): transforms(), check_image_size(), CocoClipDatasetMapper, filter_empty_instances(), _get_dummy_anno(), When loaded image has difference width/height compared with annotation., Raise an error if the image does not match the size specified in the dict., Args:             dataset_dict (dict): Metadata of one image, in Detectron2 Data (+62 more)

### Community 46 - "Community 46"
Cohesion: 0.04
Nodes (55): DirectionalBiCMS, FourDirectionalCrossModalScan, 4-Directional Cross-Modal Mamba2 Scanning (VMamba-style).  Processes a 2D featur, 4-directional cross-modal Mamba2 scanning.      Processes 2D spatial grids with, Apply 4-directional cross-modal scanning.          Args:             stream_sem:, Bidirectional cross-modal scan for a single direction.      Interleaves two stre, Run bidirectional cross-modal scan.          Args:             stream_a: First s, build_instance_map_from_masks() (+47 more)

### Community 47 - "Community 47"
Cohesion: 0.04
Nodes (36): calculate_uncertainty(), dice_loss(), Classification loss (NLL)         targets dicts must contain the key "labels" co, Compute the losses related to the masks: the focal loss and the dice loss., This performs the loss computation.         Parameters:              outputs: di, This performs the loss computation.         Parameters:              outputs: di, Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre (+28 more)

### Community 48 - "Community 48"
Cohesion: 0.04
Nodes (42): cleanup_checkpoint(), find_all_checkpoints(), find_latest_checkpoint(), init_fsdp_model_from_checkpoint(), _is_int(), keep_checkpoint_copy(), keep_last_n_checkpoints(), load_checkpoint() (+34 more)

### Community 49 - "Community 49"
Cohesion: 0.06
Nodes (64): build_job_matrix(), build_smoke_jobs(), check_job_progress(), check_vm_state(), create_vm(), delete_vm(), gcloud_ssh(), generate_smoke_vm_specs() (+56 more)

### Community 50 - "Community 50"
Cohesion: 0.04
Nodes (37): _RAFT, raft_smurf(), This class wraps Torchvision's RAFT models and always ensures eval mode., Forward pass.          Args:             images_1 (Tensor): Batch of first image, Build a pre-trained (unsupervised, SMURF) RAFT small model.      Returns:, BottleneckBlock, ConvGRU, CorrBlock (+29 more)

### Community 51 - "Community 51"
Cohesion: 0.04
Nodes (35): _get_builtin_metadata(), _get_coco_instances_meta(), _get_coco_panoptic_separated_meta(), _get_imagenet_instances_meta(), _get_UVO_instances_meta(), Returns metadata for "separated" version of the panoptic segmentation dataset., Returns metadata for "separated" version of the panoptic segmentation dataset., # NOTE: I randomly picked a color for things (+27 more)

### Community 52 - "Community 52"
Cohesion: 0.04
Nodes (34): all_reduce_dict(), all_reduce_scalar(), all_reduce_tensor(), _allreduce_coalesced(), allreduce_grads(), DistOptimizerHook, _get_global_gloo_group(), obj2tensor() (+26 more)

### Community 53 - "Community 53"
Cohesion: 0.06
Nodes (28): add_residual(), Attention, Block, BlockChunk, dinov2_vit_base_14(), dinov2_vit_large_14(), dinov2_vit_small_14(), DinoVisionTransformer (+20 more)

### Community 54 - "Community 54"
Cohesion: 0.04
Nodes (34): _make_one_hot(), Unit tests for evaluation metrics (PyTorch/numpy).  Tests cover:     - PQ comput, PQ_things and PQ_stuff should be computed separately., Test Hungarian matching algorithm., Perfect clustering should give accuracy = 1.0., Random predictions should give accuracy < 1.0., Pixels with ignore_label should not affect accuracy., mIoU should be 1.0 for perfect clustering. (+26 more)

### Community 55 - "Community 55"
Cohesion: 0.06
Nodes (32): get_model(), NormalizeInverse, Undoes the normalization and returns the reconstructed images in the input domai, ResNet50Bottom, EigenDecompositionFcnFast, forward(), PyTorch autograd function for eigen decomposition real symmetric matrices. Retur, uniform_solution_direction() (+24 more)

### Community 56 - "Community 56"
Cohesion: 0.06
Nodes (40): _cleanup_ddp(), evaluate_changed_pct(), evaluate_miou(), evaluate_only(), evaluate_panoptic(), generate_labels(), _get_device(), _is_main() (+32 more)

### Community 57 - "Community 57"
Cohesion: 0.06
Nodes (34): calc_intersect_cross(), calc_ios1(), calc_ios2(), calc_iou(), calc_iou_cross(), calc_prob_same_mask(), calc_union_cross(), dev2mask_valid() (+26 more)

### Community 58 - "Community 58"
Cohesion: 0.07
Nodes (41): calc_accuracy_multilabel_segmentation(), calc_depth_inlier_perc(), calc_epe(), calc_f_measure(), calc_f_measure_avg(), calc_outlier_percentage(), calc_outlier_percentage_from_pixelwise(), calc_outlier_pixelwise() (+33 more)

### Community 59 - "Community 59"
Cohesion: 0.07
Nodes (35): CAUSETRModel, COCOImageDataset, contrastive_loss(), DINOv3Backbone, entropy_regularization(), evaluate_model(), load_coco_panoptic_gt(), main() (+27 more)

### Community 60 - "Community 60"
Cohesion: 0.06
Nodes (31): CLIPLoss, _compute_detr_cost_matrix(), _compute_detr_seg_const_matrix(), DETRSegLoss, EM_rec_loss, _focal_loss(), LatentDupplicateSuppressionLoss, Compute latent dupplicate suppression loss.          This also takes into accoun (+23 more)

### Community 61 - "Community 61"
Cohesion: 0.08
Nodes (45): classify_stuff_things(), decode_coconut_panoptic(), depth_guided_instances(), evaluate_coconut(), extract_dinov2_features(), generate_depth_maps(), generate_instances(), generate_panoptic() (+37 more)

### Community 62 - "Community 62"
Cohesion: 0.11
Nodes (45): a(), Ac(), Al(), bc(), bl(), c(), cl(), dc() (+37 more)

### Community 63 - "Community 63"
Cohesion: 0.06
Nodes (46): _build_knn_graph(), compute_affinity_matrix(), compute_spatial_confidence(), crf_refine(), extract_pseudo_masks(), extract_pseudo_masks_batch(), _gaussian_kernel_2d(), local_cut_3d() (+38 more)

### Community 64 - "Community 64"
Cohesion: 0.07
Nodes (33): AdaptiveInstanceNet, ConvBlock, AdaptiveInstanceNet: Conv2d-based adaptive instance decomposition.  Replaces the, Args:             dinov2_features: (B, 768, H, W) DINOv2 patch features, Depthwise-separable Conv2d block with residual connection.      GroupNorm → DW-C, Conv2d-based adaptive instance decomposition network.      Architecture:, _find(), generate() (+25 more)

### Community 65 - "Community 65"
Cohesion: 0.06
Nodes (31): apply_ignore_unknown_thing_regions(), build_optimizer(), CascadeGradientScaler, _cc_inference(), compute_center_offset_loss(), compute_mask_cls_loss(), CopyPasteAugmentation, CUPSCityscapesDataset (+23 more)

### Community 66 - "Community 66"
Cohesion: 0.07
Nodes (31): build_model(), build_scheduler(), CityscapesSegDataset, DINOv3FeatureExtractor, evaluate_model(), generate_pseudolabels(), _get_autocast_ctx(), get_layer_indices() (+23 more)

### Community 67 - "Community 67"
Cohesion: 0.08
Nodes (27): clusters_agglomerative(), ConsensusSFlowDRPCs, DRPCs, Create new Dynamic Rigid Point Clouds (DRPCs) from parameters.          Paramete, Create new Dynamic Rigid Point Clouds (DRPCs) from parameters.          Paramete, Specify pt3d and pt3d_assign for inliers and make sure that they are connected., Specify pt3d and pt3d_assign for inliers and make sure that they are connected., Select subset with ids to reduce number of drpcs.          Parameters         -- (+19 more)

### Community 68 - "Community 68"
Cohesion: 0.07
Nodes (35): gather_all_tensors(), Reduce the values in the dictionary from all processes so that all processes, Copied from https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchme, reduce_dict(), _simple_gather_all_tensors(), enable_distributed(), _get_available_port(), _get_master_port() (+27 more)

### Community 69 - "Community 69"
Cohesion: 0.07
Nodes (30): crf_refine_slot_masks(), load_model(), main(), process_single_image(), Convert DINOSAUR slot masks to instance masks with semantic classification., Convert pixel-resolution slot masks (from CRF) to instances.      Args:, Process one image through DINOSAUR and extract instances.      Args:         use, Load trained DINOSAUR model from checkpoint. (+22 more)

### Community 70 - "Community 70"
Cohesion: 0.08
Nodes (29): generate_pseudolabels(), get_image_paths(), image_name_from_path(), Generate pseudo-labels from the official CUPS model checkpoint.  Runs the traine, Extract image name: e.g. 'aachen_000000_000019_leftImg8bit'., Generate pseudo-labels from CUPS model inference., Get all image paths from Cityscapes leftImg8bit split., _depth_guided_sliding_window_lowmem() (+21 more)

### Community 71 - "Community 71"
Cohesion: 0.06
Nodes (31): backward(), dropout_add_layer_norm(), _dropout_add_layer_norm_backward(), _dropout_add_layer_norm_forward(), dropout_add_layer_norm_parallel_residual(), _dropout_add_layer_norm_parallel_residual_backward(), _dropout_add_layer_norm_parallel_residual_forward(), dropout_add_layer_norm_subset() (+23 more)

### Community 72 - "Community 72"
Cohesion: 0.07
Nodes (20): differential_evolution(), DifferentialEvolutionSolver, Copied from "https://github.com/DebangLi/one-pixel-attack-pytorch/"  A slight mo, This class implements the differential evolution solver     Parameters     -----, Finds the global minimum of a multivariate function.     Differential Evolution, Initializes the population with Latin Hypercube Sampling.         Latin Hypercub, Initialises the population at random.  This type of initialization         can p, Initialises the population with a user specified population.         Parameters (+12 more)

### Community 73 - "Community 73"
Cohesion: 0.07
Nodes (21): BaseSampler, Base class of samplers., Sample positive samples., Sample negative samples., Sample positive and negative bboxes.          This is a simple implementation of, _sample_neg(), _sample_pos(), BaseSampler (+13 more)

### Community 74 - "Community 74"
Cohesion: 0.07
Nodes (23): Calibration, generate_depth_map(), inverse_rigid_trans(), load_velodyne_points(), point_cloud_adjustment(), Helper methods for loading and parsing KITTI data.  Author: Charles R. Qi Date:, Input and Output are nx3 points, velodyne coord:     front x, left y, up z (+15 more)

### Community 75 - "Community 75"
Cohesion: 0.07
Nodes (25): flatten_dict(), main(), nested_convert_to_numpy(), nested_convert_to_python(), Convert a multi-object records dataset into a webdataset., Simple but not entirely flexible routine to convert tfds outputs to webdataset o, serialize_dict(), serialize_instance() (+17 more)

### Community 76 - "Community 76"
Cohesion: 0.1
Nodes (28): depth2rgb(), disp2rgb(), draw_arrows_in_rgb(), draw_circles_in_rgb(), draw_grid_arrows_in_rgb(), draw_pixels(), draw_text_in_rgb(), flow2rgb() (+20 more)

### Community 77 - "Community 77"
Cohesion: 0.06
Nodes (39): compute_pq_from_accumulated(), depth_guided_instances(), evaluate_panoptic_single(), _get_extra_args(), main(), Get method-specific extra keyword arguments., SegFix-style: replace boundary pixel labels with nearest interior labels.      F, Dense CRF refinement: snap instance boundaries to color edges.      Uses pydense (+31 more)

### Community 78 - "Community 78"
Cohesion: 0.09
Nodes (35): add_model_config(), create_model(), create_model_and_transforms(), create_model_from_pretrained(), get_backbone(), get_model_config(), get_pretrained_tag(), get_tokenizer() (+27 more)

### Community 79 - "Community 79"
Cohesion: 0.11
Nodes (34): build_job_matrix(), check_job_progress(), check_vm_state(), create_vm(), delete_vm(), gcloud_ssh(), generate_vm_specs(), get_latest_checkpoint() (+26 more)

### Community 80 - "Community 80"
Cohesion: 0.06
Nodes (33): draw_annotation_badge(), draw_arrow(), draw_data_panel(), draw_feature_grid(), draw_flat_module_box(), draw_legend(), draw_panel_labels(), draw_patch_grid_overlay() (+25 more)

### Community 81 - "Community 81"
Cohesion: 0.09
Nodes (34): depth_2_disp(), depth_2_pt3d(), disp_2_depth(), disp_2_mask_valid(), disp_2_pt3d(), intr2x3_to_3x3(), intr2x3_to_4x4(), oflow_2_mask_inside() (+26 more)

### Community 82 - "Community 82"
Cohesion: 0.07
Nodes (25): BackprojectDepth, compute_depth_errors(), Conv3x3, ConvBlock, disp_to_depth(), get_smooth_loss(), get_translation_matrix(), Project3D (+17 more)

### Community 83 - "Community 83"
Cohesion: 0.09
Nodes (19): compute_loss_masks(), compute_proxy_supervised_loss(), Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences. (+11 more)

### Community 84 - "Community 84"
Cohesion: 0.07
Nodes (24): BackprojectDepth, compute_depth_errors(), Conv3x3, ConvBlock, disp_to_depth(), get_smooth_loss(), get_translation_matrix(), Project3D (+16 more)

### Community 85 - "Community 85"
Cohesion: 0.08
Nodes (17): ac_compile_parallelize(), activation_checkpoint_convnext(), activation_checkpoint_transformer(), compile_convnext(), compile_transformer(), get_activation_checkpoint_wrapper(), Order of the wrappers:     1/ Activation checkpointing on blocks     2/ Compile, wrap_compile_block() (+9 more)

### Community 86 - "Community 86"
Cohesion: 0.07
Nodes (26): DefaultTrainer, build_evaluator(), main(), Create configs and perform basic setups., Create configs and perform basic setups., # FIXME: brute force changes to test datasets and evaluation tasks, # FIXME: brute force changes to test datasets and evaluation tasks, Create configs and perform basic setups. (+18 more)

### Community 87 - "Community 87"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 88 - "Community 88"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 89 - "Community 89"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 90 - "Community 90"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 91 - "Community 91"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 92 - "Community 92"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 93 - "Community 93"
Cohesion: 0.09
Nodes (31): check_num_fg_corners(), densecrf(), detect_box(), DINOv3Feat, get_affinity_matrix(), get_cityscapes_images(), get_masked_affinity_matrix(), get_salient_areas() (+23 more)

### Community 94 - "Community 94"
Cohesion: 0.08
Nodes (13): _make_batch(), Tests for copy-paste augmentation module., Old-style call (no scale_range, no source_batch) still works., scale_range=(1.0, 1.0) produces same result as default., Create a synthetic batch for testing., scale_range=(0.5, 1.5) produces different result than no scaling., source_batch extracts instances from source, not target., If required fields missing, return batch unchanged. (+5 more)

### Community 95 - "Community 95"
Cohesion: 0.1
Nodes (13): Attention, Block, drop_path(), DropPath, ibot_vit_base_16(), ibot_vit_small_16(), Mlp, _no_grad_trunc_normal_() (+5 more)

### Community 96 - "Community 96"
Cohesion: 0.09
Nodes (24): COCOFeatureDataset, entropy_regularization(), evaluate_matryoshka(), load_coco_panoptic_gt(), main(), matryoshka_consistency_loss(), MatryoshkaClusterer, MatryoshkaHead (+16 more)

### Community 97 - "Community 97"
Cohesion: 0.09
Nodes (24): augment_feature_grid(), cluster_entropy_loss(), ClusterLookup, COCOFeatureDataset, _correlation_helper(), get_gt_semantic(), hungarian_match_and_miou(), load_coco_panoptic_gt() (+16 more)

### Community 98 - "Community 98"
Cohesion: 0.08
Nodes (13): backward(), chunk_bwd_dhu_fn(), chunk_bwd_dqkw_fn(), chunk_fwd_h_fn(), chunk_fwd_o_fn(), ChunkGatedDeltaRuleFunction, forward(), fwd_prepare_du() (+5 more)

### Community 99 - "Community 99"
Cohesion: 0.1
Nodes (15): Attention, Block, ConvEmbed, drop_path(), DropPath, MLP, msn_vit_base_16(), msn_vit_small_16() (+7 more)

### Community 100 - "Community 100"
Cohesion: 0.1
Nodes (31): classify_mask(), discover_images(), find_npz(), _generate_superpixels(), _guided_filter_color(), guided_filter_refine(), load_instances(), load_rgb() (+23 more)

### Community 101 - "Community 101"
Cohesion: 0.09
Nodes (15): build_mlp(), build_two_layer_mlp(), get_activation_fn(), Convenience functions for the construction neural networks using config., Build a two layer MLP, with optional initial layer norm.      Separate class as, ReLUSquared, Extensions of existing layers to implement additional functionality., Modified nn.TransformerDecoder class that returns attention weights over memory. (+7 more)

### Community 102 - "Community 102"
Cohesion: 0.1
Nodes (30): classify_person_failures(), compute_coplanar_separation_rate(), compute_depth_edges(), compute_edge_alignment(), compute_gt_instance_boundaries(), compute_gt_thing_boundaries(), compute_pq(), depth_guided_instances() (+22 more)

### Community 103 - "Community 103"
Cohesion: 0.09
Nodes (18): __call__(), forward(), MLP, MultiHeadSelfAttention, PatchEmbedding, DINOv3 ViT-B/16 backbone in PyTorch.  Loads from HuggingFace: facebook/dinov3-vi, Feed-forward network in Transformer block.      Attributes:         dim: Input/o, Single Transformer encoder block (pre-norm).      Attributes:         dim: Embed (+10 more)

### Community 104 - "Community 104"
Cohesion: 0.09
Nodes (12): AnnotationAggregator, convert_to_bytes(), GzipHandler, Handler, JsonHandler, main(), NumpyHandler, Convert a multi-object records dataset into a webdataset. (+4 more)

### Community 105 - "Community 105"
Cohesion: 0.09
Nodes (19): __call__(), forward(), MLP, MultiHeadSelfAttention, PatchEmbedding, DINO ViT-S/8 backbone in PyTorch.  Ported from facebookresearch/dino. This modul, Feed-forward network in Transformer block.      Attributes:         dim: Input/o, Feed-forward network in Transformer block.      Attributes:         dim: Input/o (+11 more)

### Community 106 - "Community 106"
Cohesion: 0.08
Nodes (17): ClassificationCost, CrossEntropyLossCost, DiceCost, FocalLossCost, MaskFocalLossCost, FocalLossCost.       Args:          weight (int | float, optional): loss_weight, Args:             cls_pred (Tensor): Predicted classification logits, shape, Cost of mask assignments based on dice losses.      Args:         weight (int | (+9 more)

### Community 107 - "Community 107"
Cohesion: 0.1
Nodes (15): backward(), chunk_bwd_dqkwg(), chunk_bwd_dv_local(), chunk_fwd_o(), chunk_gated_delta_rule(), chunk_gated_delta_rule_bwd(), chunk_gated_delta_rule_bwd_dhu(), chunk_gated_delta_rule_fwd() (+7 more)

### Community 108 - "Community 108"
Cohesion: 0.12
Nodes (14): constant_init(), ConvModule, FeatureFusionBlock, Interpolate, kaiming_init(), norm(), PreActResidualConvUnit, A 3 layer Convolutional head with intermediate upsampling      Args:     - featu (+6 more)

### Community 109 - "Community 109"
Cohesion: 0.09
Nodes (25): backward(), _chunk_scan_chunk_state_bwd_dx(), ensure_stride(), forward(), mamba_chunk_scan(), mamba_chunk_scan_combined(), _mamba_chunk_scan_combined_bwd(), _mamba_chunk_scan_combined_fwd() (+17 more)

### Community 110 - "Community 110"
Cohesion: 0.11
Nodes (27): crf_refine(), discover_images(), estimate_depth_batch(), extract_dino_features_batch(), extract_pseudo_masks_gpu(), gpu_build_knn_graph(), load_depth_model(), load_dino_backbone() (+19 more)

### Community 111 - "Community 111"
Cohesion: 0.1
Nodes (17): __call__(), CascadeStage, MaskHead, Cascade Mask R-CNN -- Class-Agnostic Detector (CAD) for CutS3D.  Faithful implem, Predict class scores and box deltas.          Args:             pooled_features:, Mask prediction head.      Predicts per-pixel binary masks from RoI features., Mask prediction head.      Predicts per-pixel binary masks from RoI features., One stage of the Cascade Mask R-CNN.      Each stage:     1. Pools features usin (+9 more)

### Community 112 - "Community 112"
Cohesion: 0.1
Nodes (26): build_instance_map(), build_instance_map_cc(), build_instance_map_depth_cc(), build_instance_map_depth_guided(), compute_distributions(), determine_thing_cluster_ids(), find_semantic_files(), load_instance_npz() (+18 more)

### Community 113 - "Community 113"
Cohesion: 0.1
Nodes (17): _bdist_wheel, BaseModelContext, build_model_for_eval(), CachedWheelsCommand, get_autocast_dtype(), get_package_version(), get_platform(), get_torch_hip_version() (+9 more)

### Community 114 - "Community 114"
Cohesion: 0.1
Nodes (13): convert_frozenbatchnorm2d_to_batchnorm2d(), CycleBatchNormList, FrozenBatchNorm2d, get_norm(), LayerNorm, NaiveSyncBatchNorm, BatchNorm2d where the batch statistics and the affine parameters are fixed., Args:         norm (str or callable): either one of BN, SyncBN, FrozenBN, GN; (+5 more)

### Community 115 - "Community 115"
Cohesion: 0.11
Nodes (25): cc_only_instances(), compute_pq_from_accumulators(), depth_guided_instances(), discover_files(), evaluate_panoptic_single(), get_phase2_configs(), main(), print_comparison_table() (+17 more)

### Community 116 - "Community 116"
Cohesion: 0.08
Nodes (17): Unit tests for panoptic merging module (PyTorch).  Tests cover:     - No pixel b, Stuff-class pixels should have instance_id=0., Stuff-class pixels should have instance_id=0., Instances below score threshold should be excluded., Should handle case with zero instances gracefully., Test batch panoptic merge., Batch merge should return (B, N) outputs., Test panoptic merge algorithm. (+9 more)

### Community 117 - "Community 117"
Cohesion: 0.1
Nodes (15): LearntConditioning, RandomConditioning, RandomConditioningWithQMCSampling, Implementation of conditioning approaches for slots., Random conditioning with potentially learnt mean and stddev., Generate conditioning vectors for `batch_size` instances.          Args:, Random conditioning with learnt mean and stddev for each slot.      Removes perm, Initialize SlotwiseLearntConditioning.          Args:             object_dim: Di (+7 more)

### Community 118 - "Community 118"
Cohesion: 0.1
Nodes (13): KMeansGrouping, Implementations of perceptual grouping algorithms.  We denote methods that group, Implementation of SlotAttention for perceptual grouping., Initialize Slot Attention Grouping.          Args:             feature_dim: Dime, Apply slot attention based perceptual grouping.          Args:             featu, Implementation of SlotAttention.      Based on the slot attention implementation, Perceptual grouping based on a stick-breaking process.      The idea is to pick, Initialize stick-breaking-based perceptual grouping.          Args: (+5 more)

### Community 119 - "Community 119"
Cohesion: 0.12
Nodes (10): MonoDatasetSingleCam, Returns a single training item from the dataset as a dictionary.          Values, Superclass for monocular dataloaders      Args:         data_path         filena, Resize colour images to the required scales and augment if required          We, MonoDatasetSingleCam, NYUDataset, NYUrawDataset, Superclass for different types of KITTI dataset loaders (+2 more)

### Community 120 - "Community 120"
Cohesion: 0.12
Nodes (24): crf_refine(), discover_samples(), extract_dino_features_batch(), extract_pseudo_masks_gpu(), gpu_build_knn_graph(), load_config(), load_dino_backbone(), main() (+16 more)

### Community 121 - "Community 121"
Cohesion: 0.11
Nodes (19): build_dinov3_model(), _build_val_records(), COCOPatchDataset, get_patch_features(), main(), neco_loss(), parse_args(), ProjectionHead (+11 more)

### Community 122 - "Community 122"
Cohesion: 0.11
Nodes (20): compute_affinity(), crf_refine_mask(), DINOv3FeatureExtractor, _get_autocast_ctx(), main(), maskcut_single_image(), ncut_bipartition(), preprocess_image() (+12 more)

### Community 123 - "Community 123"
Cohesion: 0.11
Nodes (9): MCDataset, Superclass for different types of MC dataset loaders, # NOTE: Make sure your intrinsics matrix is *normalized* by the original image s, read_file(), MonoDatasetMultiCam, Returns a single training item from the dataset as a dictionary.          Values, Superclass for monocular dataloaders      Args:         data_path         filena, Resize colour images to the required scales and augment if required          We (+1 more)

### Community 124 - "Community 124"
Cohesion: 0.12
Nodes (20): anomaly_dir_attn_cpu(), anomaly_dir_mlp_ls_regular(), anomaly_dir_regular(), CityscapesImageDataset, fc1_act(), get_neighbor_loss_device(), main(), make_transform() (+12 more)

### Community 125 - "Community 125"
Cohesion: 0.13
Nodes (23): assign_pixels_to_clusters(), dino_cluster_instances(), feature_cc_instances(), find_triples(), get_patch_features_for_cc(), global_cluster_instances(), infer_grid(), main() (+15 more)

### Community 126 - "Community 126"
Cohesion: 0.11
Nodes (14): annealing_cos(), annealing_linear(), Compute the learning rate of each parameter group., Compute the learning rate of each parameter group., A variant of MultiStepLR with a warmup on top which potentially         replaces, Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0., Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0., Compute the learning rate of each parameter group. (+6 more)

### Community 127 - "Community 127"
Cohesion: 0.13
Nodes (17): benchmark_config(), BenchmarkAttentionBlock, main(), manual_windowed_attention(), measure_memory(), print_table(), Synchronize device for accurate timing., Return peak memory in MB. (+9 more)

### Community 128 - "Community 128"
Cohesion: 0.1
Nodes (13): ClassificationCost, DiceCost, FocalLossCost, MaskFocalLossCost, FocalLossCost.       Args:          weight (int | float, optional): loss_weight, Args:             cls_pred (Tensor): Predicted classification logits, shape, Cost of mask assignments based on dice losses.      Args:         weight (int |, Args:             mask_preds (Tensor): Mask prediction in shape (N1, H, W). (+5 more)

### Community 129 - "Community 129"
Cohesion: 0.16
Nodes (16): Autoplay(), Breakpoints(), bulmaCarousel(), _classCallCheck(), Coordinate(), _defineProperty(), EventEmitter(), Fade() (+8 more)

### Community 130 - "Community 130"
Cohesion: 0.1
Nodes (5): LARS, Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a     :class, Optimizer, MockOptimizer, test_cosine_annealing_with_warmup()

### Community 131 - "Community 131"
Cohesion: 0.16
Nodes (20): _batch_iou(), _connected_components_things(), discover_pairs(), evaluate_instances(), evaluate_panoptic(), evaluate_semantic(), _load_gt_instances(), _load_pred_instances() (+12 more)

### Community 132 - "Community 132"
Cohesion: 0.12
Nodes (8): MlvlPointGenerator, PointGenerator, Generate grid Points of a single level.          Note:             This function, Generate valid flags of points of multiple feature levels.          Args:, Generate the valid flags of points of a single feature map.          Args:, Generate sparse points according to the ``prior_idxs``.          Args:, Standard points generator for multi-level (Mlvl) feature maps in 2D     points-b, Generate grid points of multiple feature levels.          Args:             feat

### Community 133 - "Community 133"
Cohesion: 0.15
Nodes (16): binarize_saliency(), classify_stuff_things(), CLSAttentionExtractor, compute_overlap_statistics(), discover_pairs(), extract(), main(), otsu_threshold() (+8 more)

### Community 134 - "Community 134"
Cohesion: 0.15
Nodes (17): cc_only_instances(), COCOPanopticGT, compute_pq(), compute_pq_from_accumulators(), depth_guided_instances(), discover_files(), main(), Standard Sobel splitting — same as Cityscapes version. (+9 more)

### Community 135 - "Community 135"
Cohesion: 0.11
Nodes (13): confidence_weighted_loss(), PseudoLabelGenerator, Self-Training with Confidence-Weighted Pseudo-Labels.  Phase D of training: uses, Weight loss by pseudo-label confidence.      L_weighted = sum(w_i * L_i) / sum(w, Weight loss by pseudo-label confidence.      L_weighted = Σ w_i · L_i / Σ w_i, Generate pseudo-labels for current round.          Args:             teacher_out, Generate pseudo-labels for current round.          Args:             teacher_out, Generate confidence-weighted pseudo-labels from EMA teacher.      Args: (+5 more)

### Community 136 - "Community 136"
Cohesion: 0.16
Nodes (9): DecoderBN, LiteResnetEncoderDecoder, Constructs a resnet model with varying number of input images.     Adapted from, Constructs a ResNet model.     Args:         num_layers (int): Number of resnet, Pytorch module for a resnet encoder, resnet_multiimage_input(), ResnetEncoder, ResNetMultiImageInput (+1 more)

### Community 137 - "Community 137"
Cohesion: 0.17
Nodes (13): calc_sigma_optimum(), calc_sigma_rel_implicit(), calc_sigma_rel_implicit_proposals(), calc_sigma_rel_optimum(), em(), estimate_std(), fit(), likelihood() (+5 more)

### Community 138 - "Community 138"
Cohesion: 0.14
Nodes (11): DistributedDataParallelWrapper, Forward function.          Args:             inputs (tuple): Input data., Train step function.          Args:             inputs (Tensor): Input Tensor., A DistributedDataParallel wrapper for models in MMGeneration.      In MMedting,, Validation step function.          Args:             inputs (tuple): Input data., Wrap models with separate MMDistributedDataParallel.          It only wraps the, Scatter function.          Args:             inputs (Tensor): Input Tensor., Set random seed.      Args:         seed (int): Seed to be used.         determi (+3 more)

### Community 139 - "Community 139"
Cohesion: 0.14
Nodes (11): KoLeoLoss, KoLeoLossDistributed, Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 -, Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 -, Pairwise nearest neighbors for L2-normalized vectors.         Uses Torch rather, Pairwise nearest neighbors for L2-normalized vectors.         Uses Torch rather, Args:             student_output (BxD): backbone output of student, Args:             student_output (BxD): backbone output of student (+3 more)

### Community 140 - "Community 140"
Cohesion: 0.14
Nodes (10): depth_edges_mask(), get_mesh(), predict_depth(), Returns a mask of edges in the depth map.     Args:     depth: 2D numpy array of, depth_edges_mask(), get_mesh(), pano_depth_to_world_points(), predict_depth() (+2 more)

### Community 141 - "Community 141"
Cohesion: 0.15
Nodes (9): PseudoLabelDataset, Loads each instance label and omits the samples that do not include at least a s, Loads the KITTI path for both the pseudo labels and images.          Args:, Loads the Cityscapes path for both the pseudo labels and images.          Args:, Returns the length of the dataset.          Returns:             length (int): L, Function loads ground truth Cityscapes label if available.          Args:, Returns on instance of the dataset.          Args:             index (int): Inde, This class implements the panoptic pseudo label dataset. (+1 more)

### Community 142 - "Community 142"
Cohesion: 0.12
Nodes (16): ade_classes(), ade_palette(), cityscapes_classes(), cityscapes_palette(), get_classes(), get_palette(), Pascal VOC palette for external use., Get class names of a dataset. (+8 more)

### Community 143 - "Community 143"
Cohesion: 0.17
Nodes (13): calc_essential_matrix(), cam_and_oflow_2_optical_centers_and_rays(), midpoint_triangulate(), described in: Multiple view geometry in computer vision     also helpful: https:, For a given optical flow and.      Parameters     ----------     intr_inv_cam1 t, method str: "dlt", "midpoint", described in: Multiple view geometry in computer vision     also helpful: https:, For a given optical flow and.      Parameters     ----------     extr_cam1 torch (+5 more)

### Community 144 - "Community 144"
Cohesion: 0.18
Nodes (11): angle_rots(), calc_optical_flow_registration(), calc_pointsets_registration(), calc_pointsets_registration_from_corresp3d(), dist_angle_transfs(), dist_transls(), filter_sim_se3(), mask_points() (+3 more)

### Community 145 - "Community 145"
Cohesion: 0.17
Nodes (9): batch_filter_size(), filter_erode(), filter_masks_ids(), filter_masks_valid(), filter_max_area_recall(), filter_multiple_assignment(), filter_overlap(), filter_size() (+1 more)

### Community 146 - "Community 146"
Cohesion: 0.13
Nodes (11): get_instance_masks(), get_intersect_fraction(), This class implements the KITTI panoptic validation dataset., Constructor method.          Args:             root (str): Path to the dataset., Returns the length of the dataset.          Returns:             length (int): L, Method returns an instances of the dataset given its index.          Args:, find the percentage of the input points that are within the 3D box defined by th, return individual masks based on the input semantic and instance label maps (+3 more)

### Community 147 - "Community 147"
Cohesion: 0.19
Nodes (15): cc_only_instances(), compute_pq_from_accumulators(), depth_guided_instances(), discover_files(), evaluate_panoptic_single(), main(), Connected component instances without depth splitting., Evaluate one image. Returns per-class (tp, fp, fn, iou) dicts. (+7 more)

### Community 148 - "Community 148"
Cohesion: 0.13
Nodes (11): binary_cross_entropy(), cross_entropy(), CrossEntropyLoss, _expand_onehot_labels(), mask_cross_entropy(), Calculate the CrossEntropy loss for masks.      Args:         pred (torch.Tensor, # TODO: handle these two reserved arguments, cross_entropy. The wrapper function for :func:`F.cross_entropy`      Args: (+3 more)

### Community 149 - "Community 149"
Cohesion: 0.2
Nodes (8): dino_vit_base_16(), dino_vit_base_8(), dino_vit_small_16(), dino_vit_small_8(), DINOMAEVisionTransformer, mae_vit_base_16(), PatchEmbed_DimensionFree, 2D Image to Patch Embedding

### Community 150 - "Community 150"
Cohesion: 0.18
Nodes (15): _build_walk_path(), get_tree_element(), is_namedtuple(), is_tensor_or_module(), map_tree(), Utilities for working with our own version of PyTrees which focus on torch tenso, Apply reduction function to a list of nested dicts.      This only considers ten, Apply a function to each element of a tree.      This only considers tensors at (+7 more)

### Community 151 - "Community 151"
Cohesion: 0.21
Nodes (5): build_global_rpe_decomp_decoder(), GlobalCrossAttention, GlobalDecoder, GlobalDecoderLayer, with_pos_embed()

### Community 152 - "Community 152"
Cohesion: 0.19
Nodes (15): create_coco_subset(), download_file(), download_nyu_depth_v2(), download_pascal_voc(), download_zoedepth(), _extract_nyu_mat(), main(), _progress_hook() (+7 more)

### Community 153 - "Community 153"
Cohesion: 0.18
Nodes (15): canny_instances(), compute_pq(), discover_files(), evaluate_panoptic_single(), main(), multiscale_sobel_instances(), Split thing regions using multi-scale Sobel gradients on depth.      Phase 2.2:, Split thing regions using Canny edge detection on depth.      Phase 2.3: Canny h (+7 more)

### Community 154 - "Community 154"
Cohesion: 0.19
Nodes (15): cc_only_instances(), compute_pq(), depth_guided_instances(), discover_files(), evaluate_panoptic_single(), main(), Connected component instances without depth splitting (baseline)., Evaluate one image. Returns per-class (tp, fp, fn, iou) arrays. (+7 more)

### Community 155 - "Community 155"
Cohesion: 0.2
Nodes (15): _compute_mask_iou(), evaluate_instances(), evaluate_panoptic(), evaluate_semantic(), _find_gt_file(), _load_gt_instances(), main(), IoU between two binary masks. (+7 more)

### Community 156 - "Community 156"
Cohesion: 0.28
Nodes (13): _kernel_close_to_default(), _kernel_is_reproducible(), _make_inputs(), _max_abs_diff(), _run_case_outputs(), _set_deterministic(), _set_seeds(), test_combined_kernel_close_to_default() (+5 more)

### Community 157 - "Community 157"
Cohesion: 0.22
Nodes (5): build_global_ape_decoder(), GlobalCrossAttention, GlobalDecoder, GlobalDecoderLayer, with_pos_embed()

### Community 158 - "Community 158"
Cohesion: 0.14
Nodes (9): OracleStuffThings, Stuff-Things MLP Classifier.  Takes DBD, FCC, IDF cue features and classifies ea, Classify using ground truth.          Args:             cluster_labels: Predicte, Oracle stuff-things classifier using ground truth labels.      Used for ablation, Initialize oracle classifier.          Args:             thing_class_ids: List o, Classify using ground truth.          Args:             cluster_labels: Predicte, Initialize StuffThingsClassifier.          Args:             hidden_dims: Hidden, Oracle stuff-things classifier using ground truth labels.      Used for ablation (+1 more)

### Community 159 - "Community 159"
Cohesion: 0.19
Nodes (13): _bytes_feature(), create_tfrecord_dataset(), _int64_feature(), parse_example(), TFRecord utilities for TPU-efficient data loading.  Provides serialization/deser, Parse a serialized TFRecord example.      Args:         serialized: Serialized t, Create a bytes feature for TFRecord., Create an int64 feature for TFRecord. (+5 more)

### Community 160 - "Community 160"
Cohesion: 0.22
Nodes (4): BaseEncoder, DecoderBN, Encoder, UpSampleBN

### Community 161 - "Community 161"
Cohesion: 0.19
Nodes (4): ExtractDataFromPredictions, Save outputs to disk in numpy or pickle format., Callback used for extracting model outputs during validation and prediction., save_outputs()

### Community 162 - "Community 162"
Cohesion: 0.21
Nodes (13): _apply_crf(), assign_labels(), evaluate_pseudolabels(), fit_kmeans(), hungarian_match(), load_features_for_kmeans(), main(), Fit MiniBatchKMeans on subsampled features.      Args:         features: (N, 768 (+5 more)

### Community 163 - "Community 163"
Cohesion: 0.24
Nodes (13): discover_pairs(), evaluate_panoptic(), evaluate_semantic(), load_instance_npz(), main(), Compute semantic mIoU by remapping k=50 cluster IDs to trainIDs., Compute PQ/SQ/RQ by remapping k=50 cluster IDs to trainIDs.      Args:         t, Remap Cityscapes labelIds to trainIds (0-18, 255=ignore). (+5 more)

### Community 164 - "Community 164"
Cohesion: 0.23
Nodes (8): DataModelRelation, DynamicRigidObjects, Create relation between data/dataframe and K models likelihood torch.Tensor: KxH, Create collection of Dynamic Rigid Objects.          Parameters         --------, Visualize 3D points in color.          Parameters         ----------         vis, Create visualization settings object.          Parameters         ----------, SceneFlow, VisualizationSettings

### Community 165 - "Community 165"
Cohesion: 0.21
Nodes (4): MonoDataset, Resize colour images to the required scales and augment if required          We, Returns a single training item from the dataset as a dictionary.          Values, Superclass for monocular dataloaders      Args:         data_path         filena

### Community 166 - "Community 166"
Cohesion: 0.18
Nodes (3): cexp2f(), cexpf(), complex_t()

### Community 167 - "Community 167"
Cohesion: 0.21
Nodes (5): angle_rots(), calc_transform_between_pointclouds(), dist_angle_transfs(), dist_transls(), ransac()

### Community 168 - "Community 168"
Cohesion: 0.2
Nodes (9): alloc_tile_workspace(), autotune_configs(), _estimate_config_cost(), _filter_configs_by_block_sizes(), Estimate shared memory cost of a config. Lower is cheaper., Filter configs by TRITON_AUTOTUNE_BLOCK_SIZE_* env vars., Select autotune configs for deterministic mode.          Uses cached autotuning, Allocate buffer for deterministic per-program reductions. (+1 more)

### Community 169 - "Community 169"
Cohesion: 0.21
Nodes (10): backward(), forward(), mamba_inner_ref(), MambaInnerFn, if return_last_state is True, returns (out, last_state)     last_state has shape, u: r(B D L)     delta: r(B D L)     A: c(D N) or r(D N)     B: c(D N) or r(B N L, rms_norm_forward(), selective_scan_fn() (+2 more)

### Community 170 - "Community 170"
Cohesion: 0.26
Nodes (11): _bytes_feature(), find_matching_file(), generate_tfrecords(), get_image_list(), _int64_feature(), main(), Get sorted list of image paths., Find file in target_dir matching the image path structure. (+3 more)

### Community 171 - "Community 171"
Cohesion: 0.22
Nodes (5): label2onehot(), label2unique(), label2unique2onehot(), mask_select_broadcasted(), Select partial x depending on mask.      Parameters     ----------     x_in torc

### Community 172 - "Community 172"
Cohesion: 0.29
Nodes (3): forward(), LEASTEREOWrapperCPU, LEASTStereoWrapper

### Community 173 - "Community 173"
Cohesion: 0.22
Nodes (3): LinearHead, ProjectionHead, ProjectionHead2d

### Community 174 - "Community 174"
Cohesion: 0.24
Nodes (6): fit_se3_to_corresp_3d_2d(), fit_se3_to_corresp_3d_2d_and_masks(), fit_se3_to_corresp_3d_2d_opencv(), Calculates se3 fit.      Parameters     ---------     masks_in torch.Tensor: KxH, Calculates se3 fit.      Parameters     ---------     pts1 torch.Tensor: KxNxC1,, Calculates se3 fit.      Parameters     ---------     pts1 torch.Tensor: KxNxC1,

### Community 175 - "Community 175"
Cohesion: 0.31
Nodes (7): color_jitter(), gaussian_blur(), generate_class_mask(), get_class_masks(), one_mix(), renorm_(), strong_transform()

### Community 176 - "Community 176"
Cohesion: 0.29
Nodes (9): compute_depth_gradients(), compute_geometric_features(), compute_surface_normals(), Geometric feature computation from monocular depth maps.  Computes surface norma, Compute all geometric features from a depth map at patch level.      Computes su, Compute Sobel depth gradients.      Args:         depth_2d: Depth map of shape (, Compute surface normals from depth map.      Normal = (-dD/dx, -dD/dy, 1) / ||(-, Sinusoidal positional encoding of depth values.      Args:         depth: Depth (+1 more)

### Community 177 - "Community 177"
Cohesion: 0.31
Nodes (8): calculate_zy_rotation_for_arrow(), create_arrow(), get_arrow(), Calculates the rotations required to go from the vector vec to the z axis vector, Create an arrow in for Open3D., Creates an arrow from an origin point to an end point, or create an arrow from a, Calculates a vector's magnitude.      Args:         - vec ():, vector_magnitude()

### Community 178 - "Community 178"
Cohesion: 0.28
Nodes (8): arrow(), main(), draw_semantic_pipeline_dataflow.py  Data flow diagram for the semantic pseudo-la, Thin coloured bar as section divider., Draw a rounded rectangle with text., Draw a downward arrow between two boxes., rounded_box(), section_bar()

### Community 179 - "Community 179"
Cohesion: 0.36
Nodes (7): convert_siglip_checkpoint(), create_rename_keys(), flatten_nested_dict(), get_siglip_config(), Copy/paste/tweak model's weights to our SigLIP structure., read_in_q_k_v_head(), rename_key()

### Community 180 - "Community 180"
Cohesion: 0.31
Nodes (8): generate_split(), get_five_crops(), main(), Generate pre-cropped training data for CAUSE-TR.  The official CAUSE/STEGO pipel, Generate 5 crops: 4 corners + center at given ratio., Remap Cityscapes label IDs to 1-based (1-27). Void=0., Generate crops for one split (train/val)., remap_cityscapes_label()

### Community 181 - "Community 181"
Cohesion: 0.39
Nodes (7): calc_transf_from_opticalflow(), calc_transform_between_pointclouds(), meanvar2normtransf(), quat2mat(), quat2mat_3x3transp(), # TODO: check if reprojection_matrix must be used for these normalized coordinat, translrot2transf()

### Community 182 - "Community 182"
Cohesion: 0.46
Nodes (7): calc_batch_gradients(), calc_div2d(), calc_img_gradients(), dilate(), erode(), minimize_tv(), open()

### Community 183 - "Community 183"
Cohesion: 0.32
Nodes (3): checkpoint_module(), load_model(), network_loader()

### Community 184 - "Community 184"
Cohesion: 0.36
Nodes (4): bulmaSlider(), _classCallCheck(), EventEmitter(), _possibleConstructorReturn()

### Community 185 - "Community 185"
Cohesion: 0.36
Nodes (7): get_metadata(), load_mapillary_vistas_panoptic_json(), Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", # TODO: currently we assume image and label has the same filename but, Register a "standard" version of ADE20k panoptic segmentation dataset named `nam, register_all_mapillary_vistas_panoptic(), register_mapillary_vistas_panoptic()

### Community 186 - "Community 186"
Cohesion: 0.36
Nodes (7): get_metadata(), load_ade20k_panoptic_json(), Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", # TODO: currently we assume image and label has the same filename but, Register a "standard" version of ADE20k panoptic segmentation dataset named `nam, register_ade20k_panoptic(), register_all_ade20k_panoptic()

### Community 187 - "Community 187"
Cohesion: 0.36
Nodes (7): colorize_instances(), colorize_semantic(), depth_guided_instances(), main(), Map trainID semantic map to RGB., Color each instance with a unique random color, stuff in gray., Split thing regions using depth gradient edges.

### Community 188 - "Community 188"
Cohesion: 0.39
Nodes (7): gap_fill_merge(), main(), merge_npz(), Merge primary + supplement instances with coverage-based gap filling.      Keep, Resize boolean masks to target resolution using nearest neighbor., Merge masks from multiple NPZ files, sorted by score.      Handles resolution di, _resize_masks()

### Community 189 - "Community 189"
Cohesion: 0.39
Nodes (7): generate_depth_da2(), generate_depth_da3(), get_image_paths(), main(), Fallback: Generate depth maps using Depth Anything V2 via HF Transformers., Get all Cityscapes image paths., Generate depth maps using Depth Anything V3.      Args:         data_dir: Path t

### Community 190 - "Community 190"
Cohesion: 0.48
Nodes (4): selective_scan_bwd(), selective_scan_fwd(), set_ssm_params_bwd(), set_ssm_params_fwd()

### Community 191 - "Community 191"
Cohesion: 0.52
Nodes (6): get_benchmarks(), main(), make_tensors(), _peak_memory_mb(), _reset_peak_memory(), _run_one()

### Community 192 - "Community 192"
Cohesion: 0.43
Nodes (6): get_metadata(), load_coco_panoptic_json(), # TODO: currently we assume image and label has the same filename but, Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", register_all_coco_panoptic_annos_sem_seg(), register_coco_panoptic_annos_sem_seg()

### Community 193 - "Community 193"
Cohesion: 0.43
Nodes (4): add_config_as_page(), augment_defaults_list_links(), convert_dl_entry_to_link(), get_doc_page_source()

### Community 194 - "Community 194"
Cohesion: 0.43
Nodes (6): analyze_confusion(), build_confusion_matrix(), main(), Analyze the confusion matrix, focusing on zero-IoU classes., Build full 19×19 confusion matrix: conf[pred_class, gt_class] = pixel count., _remap_to_trainids()

### Community 195 - "Community 195"
Cohesion: 0.47
Nodes (5): classify_stuff_things(), compute_cluster_statistics(), main(), Two-stage stuff/things classification using depth-split ratio.      Stage 1: For, Compute per-cluster statistics including depth-split ratio.      For each class

### Community 196 - "Community 196"
Cohesion: 0.47
Nodes (3): main(), test(), test_without_crf()

### Community 197 - "Community 197"
Cohesion: 0.47
Nodes (3): main(), test(), test_without_crf()

### Community 198 - "Community 198"
Cohesion: 0.4
Nodes (2): n(), s()

### Community 199 - "Community 199"
Cohesion: 0.4
Nodes (5): create_triangles(), depth_to_points(), get_intrinsics(), Intrinsics for a pinhole camera model.     Assume fov of 55 degrees and central, Reference: https://github.com/google-research/google-research/blob/e96197de06613

### Community 200 - "Community 200"
Cohesion: 0.33
Nodes (2): get_intersect_fraction(), find the percentage of the input points that are within the 3D box defined by th

### Community 201 - "Community 201"
Cohesion: 0.47
Nodes (5): download_file(), login(), main(), Login to Cityscapes and return authenticated session., Download a single Cityscapes package.

### Community 202 - "Community 202"
Cohesion: 0.8
Nodes (4): em(), fit(), fit_and_likelihood(), likelihood()

### Community 203 - "Community 203"
Cohesion: 0.5
Nodes (2): calc_inlier_hard(), calc_inlier_soft()

### Community 204 - "Community 204"
Cohesion: 0.5
Nodes (2): i(), o()

### Community 205 - "Community 205"
Cohesion: 0.4
Nodes (4): find_checkpoint(), get_commandline_config_path(), Find checkpoint in output path of previous run., Get the path of a config path specified on the command line.

### Community 206 - "Community 206"
Cohesion: 0.67
Nodes (2): remove_suffix(), remove_suffixes()

### Community 207 - "Community 207"
Cohesion: 0.5
Nodes (1): MonodepthOptions

### Community 208 - "Community 208"
Cohesion: 0.5
Nodes (2): Create semantic segmentation annotations from panoptic segmentation     annotati, separate_coco_semantic_from_panoptic()

### Community 209 - "Community 209"
Cohesion: 0.5
Nodes (2): Convert a in the experiment folder to a valid setting for experiment., _remove_filename_components()

### Community 210 - "Community 210"
Cohesion: 0.83
Nodes (3): _fmt_overrides(), _is_metric_conf(), main()

### Community 211 - "Community 211"
Cohesion: 0.67
Nodes (2): load_custom_callable(), _load_modules_from_dir()

### Community 212 - "Community 212"
Cohesion: 0.67
Nodes (3): generate_targets_for_image(), main(), Generate center heatmap, offset map, boundary map for one image.      Args:

### Community 213 - "Community 213"
Cohesion: 0.67
Nodes (3): load_coco_panoptic_gt(), main(), Load COCO panoptic GT and convert to 27-class semantic label map.

### Community 214 - "Community 214"
Cohesion: 1.0
Nodes (2): calc_batch_gradients(), calc_batch_k_gradients()

### Community 215 - "Community 215"
Cohesion: 1.0
Nodes (2): j_linkage(), j_linkage_single_step()

### Community 216 - "Community 216"
Cohesion: 0.67
Nodes (2): complete(), Given invalid points complete points by completing depth using recursive neighbo

### Community 217 - "Community 217"
Cohesion: 0.67
Nodes (1): Download a HuggingFace Mask2Former checkpoint to ./checkpoints/.

### Community 218 - "Community 218"
Cohesion: 0.67
Nodes (1): @article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b

### Community 219 - "Community 219"
Cohesion: 0.67
Nodes (0): 

### Community 220 - "Community 220"
Cohesion: 0.67
Nodes (0): 

### Community 221 - "Community 221"
Cohesion: 0.67
Nodes (0): 

### Community 222 - "Community 222"
Cohesion: 1.0
Nodes (2): _get_ade20k_full_meta(), register_all_ade20k_full()

### Community 223 - "Community 223"
Cohesion: 1.0
Nodes (2): _get_mapillary_vistas_meta(), register_all_mapillary_vistas()

### Community 224 - "Community 224"
Cohesion: 1.0
Nodes (2): _get_coco_stuff_meta(), register_all_coco_stuff_10k()

### Community 225 - "Community 225"
Cohesion: 1.0
Nodes (2): compare_directories(), get_all_files()

### Community 226 - "Community 226"
Cohesion: 0.67
Nodes (1): Create TFRecords from Dataset.  Usage:     python scripts/create_tfrecords.py \

### Community 227 - "Community 227"
Cohesion: 1.0
Nodes (0): 

### Community 228 - "Community 228"
Cohesion: 1.0
Nodes (0): 

### Community 229 - "Community 229"
Cohesion: 1.0
Nodes (0): 

### Community 230 - "Community 230"
Cohesion: 1.0
Nodes (0): 

### Community 231 - "Community 231"
Cohesion: 1.0
Nodes (0): 

### Community 232 - "Community 232"
Cohesion: 1.0
Nodes (0): 

### Community 233 - "Community 233"
Cohesion: 1.0
Nodes (0): 

### Community 234 - "Community 234"
Cohesion: 1.0
Nodes (1): @article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b

### Community 235 - "Community 235"
Cohesion: 1.0
Nodes (0): 

### Community 236 - "Community 236"
Cohesion: 1.0
Nodes (0): 

### Community 237 - "Community 237"
Cohesion: 1.0
Nodes (0): 

### Community 238 - "Community 238"
Cohesion: 1.0
Nodes (0): 

### Community 239 - "Community 239"
Cohesion: 1.0
Nodes (1): # NOTE: there is some overlap between semantic and instance annotation

### Community 240 - "Community 240"
Cohesion: 1.0
Nodes (0): 

### Community 241 - "Community 241"
Cohesion: 1.0
Nodes (1): Generate the code reference pages and navigation.

### Community 242 - "Community 242"
Cohesion: 1.0
Nodes (0): 

### Community 243 - "Community 243"
Cohesion: 1.0
Nodes (0): 

### Community 244 - "Community 244"
Cohesion: 1.0
Nodes (0): 

### Community 245 - "Community 245"
Cohesion: 1.0
Nodes (0): 

### Community 246 - "Community 246"
Cohesion: 1.0
Nodes (0): 

### Community 247 - "Community 247"
Cohesion: 1.0
Nodes (1): Apply SSD selective scan.          Args:             x: Input sequence of shape

### Community 248 - "Community 248"
Cohesion: 1.0
Nodes (1): Apply Mamba2 block.          Args:             x: Input of shape (B, L, D).

### Community 249 - "Community 249"
Cohesion: 1.0
Nodes (1): Apply stack of Mamba2 blocks.          Args:             x: Input of shape (B, L

### Community 250 - "Community 250"
Cohesion: 1.0
Nodes (1): Project features to bridge dimension.          Args:             semantic_codes:

### Community 251 - "Community 251"
Cohesion: 1.0
Nodes (1): Inverse project from bridge dimension.          Args:             x: Fused featu

### Community 252 - "Community 252"
Cohesion: 1.0
Nodes (1): Condition features on depth.          Args:             depth: Depth values of s

### Community 253 - "Community 253"
Cohesion: 1.0
Nodes (1): Classify clusters as stuff or things.          Args:             cues: Concatena

### Community 254 - "Community 254"
Cohesion: 1.0
Nodes (1): Generate instance masks from features.          In inference mode with depth ava

### Community 255 - "Community 255"
Cohesion: 1.0
Nodes (1): Refine masks through cascade stages.          Args:             features: Featur

### Community 256 - "Community 256"
Cohesion: 1.0
Nodes (1): Generate proposals from features.          Args:             features: Backbone

### Community 257 - "Community 257"
Cohesion: 1.0
Nodes (1): Predict class scores and box deltas.          Args:             pooled_features:

### Community 258 - "Community 258"
Cohesion: 1.0
Nodes (1): Predict mask features.          Args:             features: Pooled RoI features,

### Community 259 - "Community 259"
Cohesion: 1.0
Nodes (1): Run one cascade stage.          Args:             features: Feature map, shape (

### Community 260 - "Community 260"
Cohesion: 1.0
Nodes (1): Generate instance masks and scores.          Args:             features: Input f

### Community 261 - "Community 261"
Cohesion: 1.0
Nodes (1): Compute semantic codes from DINO features.          Args:             features:

### Community 262 - "Community 262"
Cohesion: 1.0
Nodes (1): Compute semantic codes from spatial features.          Args:             feature

### Community 263 - "Community 263"
Cohesion: 1.0
Nodes (1): Extract patch embeddings.          Args:             x: Input image of shape (B,

### Community 264 - "Community 264"
Cohesion: 1.0
Nodes (1): Apply multi-head self-attention.          Args:             x: Input of shape (B

### Community 265 - "Community 265"
Cohesion: 1.0
Nodes (1): Apply MLP.          Args:             x: Input of shape (B, N, D).             d

### Community 266 - "Community 266"
Cohesion: 1.0
Nodes (1): Apply Transformer block (pre-norm).          Args:             x: Input of shape

### Community 267 - "Community 267"
Cohesion: 1.0
Nodes (1): Extract DINO features from input image.          Args:             x: Input imag

### Community 268 - "Community 268"
Cohesion: 1.0
Nodes (1): Extract patch embeddings.          Args:             x: Input image of shape (B,

### Community 269 - "Community 269"
Cohesion: 1.0
Nodes (1): Apply multi-head self-attention.          Args:             x: Input of shape (B

### Community 270 - "Community 270"
Cohesion: 1.0
Nodes (1): Extract DINOv3 features from input image.          Args:             x: Input im

### Community 271 - "Community 271"
Cohesion: 1.0
Nodes (0): 

### Community 272 - "Community 272"
Cohesion: 1.0
Nodes (1): Inference method. Switch model to `eval` mode,          call `.forward(x)` with

### Community 273 - "Community 273"
Cohesion: 1.0
Nodes (1): Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B

### Community 274 - "Community 274"
Cohesion: 1.0
Nodes (1): Convert ImageNet-normalized [C,H,W] tensor to uint8 [H,W,3] BGR for OpenCV.

### Community 275 - "Community 275"
Cohesion: 1.0
Nodes (1): Morphological opening (remove small protrusions) then closing (fill holes).

### Community 276 - "Community 276"
Cohesion: 1.0
Nodes (1): Fast bilateral solver for mask refinement.          Simplified version using Ope

### Community 277 - "Community 277"
Cohesion: 1.0
Nodes (1): Compute bounding boxes from binary masks. masks: [N, H, W] bool.

### Community 278 - "Community 278"
Cohesion: 1.0
Nodes (1): Updates the internal state of the metric. In particular, we track update the cos

### Community 279 - "Community 279"
Cohesion: 1.0
Nodes (1): Remaps the semantic classes to the target class using the latest assignments.

### Community 280 - "Community 280"
Cohesion: 1.0
Nodes (1): Getter method to access things prototypes.          Returns:             things_

### Community 281 - "Community 281"
Cohesion: 1.0
Nodes (1): Getter method to access stuffs prototypes.          Returns:             things_

### Community 282 - "Community 282"
Cohesion: 1.0
Nodes (1): Setter method to access stuffs prototypes.          Args:             value (Set

### Community 283 - "Community 283"
Cohesion: 1.0
Nodes (1): Setter method to access stuffs prototypes.          Args:             value (Set

### Community 284 - "Community 284"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 285 - "Community 285"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 286 - "Community 286"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 287 - "Community 287"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 288 - "Community 288"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape: sh

### Community 289 - "Community 289"
Cohesion: 1.0
Nodes (0): 

### Community 290 - "Community 290"
Cohesion: 1.0
Nodes (0): 

### Community 291 - "Community 291"
Cohesion: 1.0
Nodes (0): 

### Community 292 - "Community 292"
Cohesion: 1.0
Nodes (0): 

### Community 293 - "Community 293"
Cohesion: 1.0
Nodes (1): Get parameters for ``crop`` for a random sized crop.         Args:             i

### Community 294 - "Community 294"
Cohesion: 1.0
Nodes (1): torch.Tensor: concatenated positive and negative boxes

### Community 295 - "Community 295"
Cohesion: 1.0
Nodes (1): Returns a dictionary of info about the object.

### Community 296 - "Community 296"
Cohesion: 1.0
Nodes (1): Args:             rng (None | int | numpy.random.RandomState): seed or state.

### Community 297 - "Community 297"
Cohesion: 1.0
Nodes (1): int: number of feature levels that the generator will be applied

### Community 298 - "Community 298"
Cohesion: 1.0
Nodes (1): list[int]: The number of priors (points) at a point         on the feature grid

### Community 299 - "Community 299"
Cohesion: 1.0
Nodes (1): Placeholder for sample function.

### Community 300 - "Community 300"
Cohesion: 1.0
Nodes (1): Randomly select an img_scale from given candidates.          Args:             i

### Community 301 - "Community 301"
Cohesion: 1.0
Nodes (1): Randomly sample an img_scale when ``multiscale_mode=='range'``.          Args:

### Community 302 - "Community 302"
Cohesion: 1.0
Nodes (1): Randomly sample an img_scale when ``ratio_range`` is specified.          A ratio

### Community 303 - "Community 303"
Cohesion: 1.0
Nodes (1): Loss Name.          This function must be implemented and will return the name o

### Community 304 - "Community 304"
Cohesion: 1.0
Nodes (1): Forward function for `MultiheadAttention`.          **kwargs allow passing a mor

### Community 305 - "Community 305"
Cohesion: 1.0
Nodes (1): Forward function for `FFN`.          The function would add x to the output tens

### Community 306 - "Community 306"
Cohesion: 1.0
Nodes (1): Forward function for `FFN`.         The function would add x to the output tenso

### Community 307 - "Community 307"
Cohesion: 1.0
Nodes (1): Get the reference points used in decoder.          Args:             spatial_sha

### Community 308 - "Community 308"
Cohesion: 1.0
Nodes (1): Assign boxes to either a ground truth boxes or a negative boxes.

### Community 309 - "Community 309"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the first convolution layer

### Community 310 - "Community 310"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the second convolution layer

### Community 311 - "Community 311"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the first convolution layer

### Community 312 - "Community 312"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the second convolution layer

### Community 313 - "Community 313"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the third convolution layer

### Community 314 - "Community 314"
Cohesion: 1.0
Nodes (1): nn.Module: the normalization layer named "norm1"

### Community 315 - "Community 315"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has neck

### Community 316 - "Community 316"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has auxiliary head

### Community 317 - "Community 317"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has decode head

### Community 318 - "Community 318"
Cohesion: 1.0
Nodes (1): Placeholder for extract features from images.

### Community 319 - "Community 319"
Cohesion: 1.0
Nodes (1): Placeholder for encode images with backbone and decode into a         semantic s

### Community 320 - "Community 320"
Cohesion: 1.0
Nodes (1): Placeholder for Forward function for training.

### Community 321 - "Community 321"
Cohesion: 1.0
Nodes (1): Placeholder for single image test.

### Community 322 - "Community 322"
Cohesion: 1.0
Nodes (1): Placeholder for augmentation test.

### Community 323 - "Community 323"
Cohesion: 1.0
Nodes (1): Calls either :func:`forward_train` or :func:`forward_test` depending         on

### Community 324 - "Community 324"
Cohesion: 1.0
Nodes (1): Parse the raw outputs (losses) of the network.          Args:             losses

### Community 325 - "Community 325"
Cohesion: 1.0
Nodes (1): Placeholder of forward function.

### Community 326 - "Community 326"
Cohesion: 1.0
Nodes (1): Compute segmentation loss.

### Community 327 - "Community 327"
Cohesion: 1.0
Nodes (1): r"""         Instantiate a [`SiglipConfig`] (or a derived class) from siglip tex

### Community 328 - "Community 328"
Cohesion: 1.0
Nodes (1): Preprocess an image or batch of images.          Args:             images (`Imag

### Community 329 - "Community 329"
Cohesion: 1.0
Nodes (0): 

### Community 330 - "Community 330"
Cohesion: 1.0
Nodes (0): 

### Community 331 - "Community 331"
Cohesion: 1.0
Nodes (0): 

### Community 332 - "Community 332"
Cohesion: 1.0
Nodes (0): 

### Community 333 - "Community 333"
Cohesion: 1.0
Nodes (0): 

### Community 334 - "Community 334"
Cohesion: 1.0
Nodes (0): 

### Community 335 - "Community 335"
Cohesion: 1.0
Nodes (0): 

### Community 336 - "Community 336"
Cohesion: 1.0
Nodes (0): 

### Community 337 - "Community 337"
Cohesion: 1.0
Nodes (0): 

### Community 338 - "Community 338"
Cohesion: 1.0
Nodes (0): 

### Community 339 - "Community 339"
Cohesion: 1.0
Nodes (0): 

### Community 340 - "Community 340"
Cohesion: 1.0
Nodes (0): 

### Community 341 - "Community 341"
Cohesion: 1.0
Nodes (0): 

### Community 342 - "Community 342"
Cohesion: 1.0
Nodes (0): 

### Community 343 - "Community 343"
Cohesion: 1.0
Nodes (0): 

### Community 344 - "Community 344"
Cohesion: 1.0
Nodes (0): 

### Community 345 - "Community 345"
Cohesion: 1.0
Nodes (0): 

### Community 346 - "Community 346"
Cohesion: 1.0
Nodes (0): 

### Community 347 - "Community 347"
Cohesion: 1.0
Nodes (0): 

### Community 348 - "Community 348"
Cohesion: 1.0
Nodes (0): 

### Community 349 - "Community 349"
Cohesion: 1.0
Nodes (0): 

### Community 350 - "Community 350"
Cohesion: 1.0
Nodes (0): 

### Community 351 - "Community 351"
Cohesion: 1.0
Nodes (0): 

### Community 352 - "Community 352"
Cohesion: 1.0
Nodes (0): 

### Community 353 - "Community 353"
Cohesion: 1.0
Nodes (0): 

### Community 354 - "Community 354"
Cohesion: 1.0
Nodes (0): 

### Community 355 - "Community 355"
Cohesion: 1.0
Nodes (0): 

### Community 356 - "Community 356"
Cohesion: 1.0
Nodes (0): 

### Community 357 - "Community 357"
Cohesion: 1.0
Nodes (0): 

### Community 358 - "Community 358"
Cohesion: 1.0
Nodes (0): 

### Community 359 - "Community 359"
Cohesion: 1.0
Nodes (0): 

### Community 360 - "Community 360"
Cohesion: 1.0
Nodes (0): 

### Community 361 - "Community 361"
Cohesion: 1.0
Nodes (0): 

### Community 362 - "Community 362"
Cohesion: 1.0
Nodes (0): 

### Community 363 - "Community 363"
Cohesion: 1.0
Nodes (1): Momentum update of evaluation model (exponential moving average)

### Community 364 - "Community 364"
Cohesion: 1.0
Nodes (0): 

### Community 365 - "Community 365"
Cohesion: 1.0
Nodes (1): x: (batch_size, seqlen, nheads, headdim)             cos, sin: (seqlen, rotary_d

### Community 366 - "Community 366"
Cohesion: 1.0
Nodes (1): logits: (batch, vocab_size)         labels: (batch,)         If process_group is

### Community 367 - "Community 367"
Cohesion: 1.0
Nodes (1): Compute proxy supervised loss (depth hint loss) for prediction.              - v

### Community 368 - "Community 368"
Cohesion: 1.0
Nodes (1): Compute loss masks for each of standard reprojection and depth hint         repr

### Community 369 - "Community 369"
Cohesion: 1.0
Nodes (1): The best solution from the solver         Returns         -------         x : nd

### Community 370 - "Community 370"
Cohesion: 1.0
Nodes (1): The standard deviation of the population energies divided by their         mean.

### Community 371 - "Community 371"
Cohesion: 1.0
Nodes (0): 

### Community 372 - "Community 372"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 373 - "Community 373"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 374 - "Community 374"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 375 - "Community 375"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 376 - "Community 376"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 377 - "Community 377"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             is_train: wheth

### Community 378 - "Community 378"
Cohesion: 1.0
Nodes (0): 

### Community 379 - "Community 379"
Cohesion: 1.0
Nodes (1): If process_group is not None and sequence_parallel=True, we're doing Tensor Para

### Community 380 - "Community 380"
Cohesion: 1.0
Nodes (1): xz: (batch, dim, seqlen)

### Community 381 - "Community 381"
Cohesion: 1.0
Nodes (1): If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil

### Community 382 - "Community 382"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 383 - "Community 383"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 384 - "Community 384"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 385 - "Community 385"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 386 - "Community 386"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 387 - "Community 387"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 388 - "Community 388"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 389 - "Community 389"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 390 - "Community 390"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 391 - "Community 391"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 392 - "Community 392"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 393 - "Community 393"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 394 - "Community 394"
Cohesion: 1.0
Nodes (0): 

### Community 395 - "Community 395"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 396 - "Community 396"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 397 - "Community 397"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 398 - "Community 398"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 399 - "Community 399"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 400 - "Community 400"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 401 - "Community 401"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 402 - "Community 402"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 403 - "Community 403"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             is_train: wheth

### Community 404 - "Community 404"
Cohesion: 1.0
Nodes (0): 

### Community 405 - "Community 405"
Cohesion: 1.0
Nodes (1): Inference interface for the model for PIL image         Args:             pil_im

### Community 406 - "Community 406"
Cohesion: 1.0
Nodes (1): Load a '.png' segmentation mask, ignoring any colour map.

### Community 407 - "Community 407"
Cohesion: 1.0
Nodes (1): Load a '.mat' segmentation mask of the kind used in the SBD dataset.

### Community 408 - "Community 408"
Cohesion: 1.0
Nodes (1): Fields that will be transformed with this transform.

### Community 409 - "Community 409"
Cohesion: 1.0
Nodes (1): Application of transform to input pipe.          Args:             input_pipe: I

### Community 410 - "Community 410"
Cohesion: 1.0
Nodes (1): Comput visualization output.          A visualization method takes some inputs a

### Community 411 - "Community 411"
Cohesion: 1.0
Nodes (1): Convert instance to segmentation mask.          Args:             instance_mask:

### Community 412 - "Community 412"
Cohesion: 1.0
Nodes (1): Return current value of hyperparameter based on global step.          Returns:

### Community 413 - "Community 413"
Cohesion: 1.0
Nodes (1): Updates the internal state of the metric. In particular, we track update the cos

### Community 414 - "Community 414"
Cohesion: 1.0
Nodes (1): Remaps the semantic classes to the target class using the latest assignments.

### Community 415 - "Community 415"
Cohesion: 1.0
Nodes (1): Getter method to access things prototypes.          Returns:             things_

### Community 416 - "Community 416"
Cohesion: 1.0
Nodes (1): Getter method to access stuffs prototypes.          Returns:             things_

### Community 417 - "Community 417"
Cohesion: 1.0
Nodes (1): Setter method to access things prototypes.          Args:             value (Set

### Community 418 - "Community 418"
Cohesion: 1.0
Nodes (1): Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg

### Community 419 - "Community 419"
Cohesion: 1.0
Nodes (1): Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (

### Community 420 - "Community 420"
Cohesion: 1.0
Nodes (1): Files that match these patterns are not deleted by cleanup

### Community 421 - "Community 421"
Cohesion: 1.0
Nodes (1): maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and n

### Community 422 - "Community 422"
Cohesion: 1.0
Nodes (0): 

### Community 423 - "Community 423"
Cohesion: 1.0
Nodes (1): Build resume command using latest GCS checkpoint.

### Community 424 - "Community 424"
Cohesion: 1.0
Nodes (1): Explicit Test-Time Training adaptation.          For each image, perform K gradi

### Community 425 - "Community 425"
Cohesion: 1.0
Nodes (1): CRF-inspired pairwise consistency loss (differentiable CRF energy).          For

### Community 426 - "Community 426"
Cohesion: 1.0
Nodes (1): Fuse DINOv3 and SSD-1B features via learned cross-attention.          Args:

### Community 427 - "Community 427"
Cohesion: 1.0
Nodes (1): Get instance masks from the model.          Args:             features: (B, N, 7

### Community 428 - "Community 428"
Cohesion: 1.0
Nodes (1): Generate pseudo-labels from EMA teacher predictions.          Returns semantic l

### Community 429 - "Community 429"
Cohesion: 1.0
Nodes (1): Generate pseudo-labels with optional TTA.          Returns:             labels:

### Community 430 - "Community 430"
Cohesion: 1.0
Nodes (1): Extract CLS attention map as (H_patches, W_patches) numpy array.          Args:

### Community 431 - "Community 431"
Cohesion: 1.0
Nodes (1): Args:             img: (1, 3, H, W) tensor, normalized         Returns:

### Community 432 - "Community 432"
Cohesion: 1.0
Nodes (1): Extract self-attention affinity matrix from last layer.          Args:

### Community 433 - "Community 433"
Cohesion: 1.0
Nodes (1): Extract SD self-attention features for a single image.          Args:

### Community 434 - "Community 434"
Cohesion: 1.0
Nodes (1): Extract SSD-1B self-attention features for a single image.          Args:

### Community 435 - "Community 435"
Cohesion: 1.0
Nodes (1): Extract patch tokens from images.          Args:             images: (B, 3, H, W

### Community 436 - "Community 436"
Cohesion: 1.0
Nodes (1): Bipartite matching between predictions and targets.          Returns:

### Community 437 - "Community 437"
Cohesion: 1.0
Nodes (1): Post-process a batch of predictions.          Args:             pred_logits: (B,

### Community 438 - "Community 438"
Cohesion: 1.0
Nodes (1): Extract DINO features from input image.          Args:             x: Input imag

### Community 439 - "Community 439"
Cohesion: 1.0
Nodes (1): Extract DINOv3 features.          Args:             x: Input image (B, 3, H, W),

### Community 440 - "Community 440"
Cohesion: 1.0
Nodes (1): Load pretrained DINOv3 weights from HuggingFace.          Args:             mode

## Knowledge Gaps
- **3514 isolated node(s):** `setup_notebooklm.py — Bootstrap a NotebookLM notebook for MBPS BMVC 2026.  Creat`, `Run notebooklm CLI with given args.`, `Run CLI with --json flag and parse output.`, `Return notebook ID if a notebook named NOTEBOOK_NAME already exists.`, `Create the notebook and return its ID.` (+3509 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 227`** (2 nodes): `boxes3d.py`, `fit()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 228`** (2 nodes): `pt3d_oflow.py`, `fit_se3_to_pt3d_oflow_and_masks()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 229`** (2 nodes): `_3d.py`, `visualize_se3s()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 230`** (2 nodes): `preprocess_waymo.py`, `preprocess()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 231`** (2 nodes): `remap_cause27_to_trainid.py`, `remap_split()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 232`** (2 nodes): `restore_centroids_k80.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 233`** (2 nodes): `remap_raw_clusters_to_trainid.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 234`** (2 nodes): `pretrained_download.py`, `@article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 235`** (2 nodes): `gdrive_downloader.py`, `download()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 236`** (2 nodes): `softplus.py`, `softplus()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 237`** (2 nodes): `evaluate_coco_boundary_ap.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 238`** (2 nodes): `prepare_ade20k_sem_seg.py`, `convert()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 239`** (2 nodes): `prepare_ade20k_pan_seg.py`, `# NOTE: there is some overlap between semantic and instance annotation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 240`** (2 nodes): `test_conditioning.py`, `test_slot_conditionings()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 241`** (2 nodes): `generate_api_docs.py`, `Generate the code reference pages and navigation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 242`** (2 nodes): `validate_cups_stage3.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 243`** (2 nodes): `diagnose_masks.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 244`** (2 nodes): `mumford_shah_phase_b.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 245`** (2 nodes): `make_qualitative.py`, `load()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 246`** (1 nodes): `sample.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 247`** (1 nodes): `Apply SSD selective scan.          Args:             x: Input sequence of shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 248`** (1 nodes): `Apply Mamba2 block.          Args:             x: Input of shape (B, L, D).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 249`** (1 nodes): `Apply stack of Mamba2 blocks.          Args:             x: Input of shape (B, L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 250`** (1 nodes): `Project features to bridge dimension.          Args:             semantic_codes:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 251`** (1 nodes): `Inverse project from bridge dimension.          Args:             x: Fused featu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 252`** (1 nodes): `Condition features on depth.          Args:             depth: Depth values of s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 253`** (1 nodes): `Classify clusters as stuff or things.          Args:             cues: Concatena`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 254`** (1 nodes): `Generate instance masks from features.          In inference mode with depth ava`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 255`** (1 nodes): `Refine masks through cascade stages.          Args:             features: Featur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 256`** (1 nodes): `Generate proposals from features.          Args:             features: Backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 257`** (1 nodes): `Predict class scores and box deltas.          Args:             pooled_features:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 258`** (1 nodes): `Predict mask features.          Args:             features: Pooled RoI features,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 259`** (1 nodes): `Run one cascade stage.          Args:             features: Feature map, shape (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 260`** (1 nodes): `Generate instance masks and scores.          Args:             features: Input f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 261`** (1 nodes): `Compute semantic codes from DINO features.          Args:             features:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 262`** (1 nodes): `Compute semantic codes from spatial features.          Args:             feature`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 263`** (1 nodes): `Extract patch embeddings.          Args:             x: Input image of shape (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 264`** (1 nodes): `Apply multi-head self-attention.          Args:             x: Input of shape (B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 265`** (1 nodes): `Apply MLP.          Args:             x: Input of shape (B, N, D).             d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 266`** (1 nodes): `Apply Transformer block (pre-norm).          Args:             x: Input of shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 267`** (1 nodes): `Extract DINO features from input image.          Args:             x: Input imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 268`** (1 nodes): `Extract patch embeddings.          Args:             x: Input image of shape (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 269`** (1 nodes): `Apply multi-head self-attention.          Args:             x: Input of shape (B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 270`** (1 nodes): `Extract DINOv3 features from input image.          Args:             x: Input im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 271`** (1 nodes): `download_spidepth.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 272`** (1 nodes): `Inference method. Switch model to `eval` mode,          call `.forward(x)` with`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 273`** (1 nodes): `Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 274`** (1 nodes): `Convert ImageNet-normalized [C,H,W] tensor to uint8 [H,W,3] BGR for OpenCV.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 275`** (1 nodes): `Morphological opening (remove small protrusions) then closing (fill holes).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 276`** (1 nodes): `Fast bilateral solver for mask refinement.          Simplified version using Ope`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 277`** (1 nodes): `Compute bounding boxes from binary masks. masks: [N, H, W] bool.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 278`** (1 nodes): `Updates the internal state of the metric. In particular, we track update the cos`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 279`** (1 nodes): `Remaps the semantic classes to the target class using the latest assignments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 280`** (1 nodes): `Getter method to access things prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 281`** (1 nodes): `Getter method to access stuffs prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 282`** (1 nodes): `Setter method to access stuffs prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 283`** (1 nodes): `Setter method to access stuffs prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 284`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 285`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 286`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 287`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 288`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape: sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 289`** (1 nodes): `gmm.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 290`** (1 nodes): `spectral_clustering.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 291`** (1 nodes): `eval_seg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 292`** (1 nodes): `link_coco20k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 293`** (1 nodes): `Get parameters for ``crop`` for a random sized crop.         Args:             i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 294`** (1 nodes): `torch.Tensor: concatenated positive and negative boxes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 295`** (1 nodes): `Returns a dictionary of info about the object.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 296`** (1 nodes): `Args:             rng (None | int | numpy.random.RandomState): seed or state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 297`** (1 nodes): `int: number of feature levels that the generator will be applied`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 298`** (1 nodes): `list[int]: The number of priors (points) at a point         on the feature grid`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 299`** (1 nodes): `Placeholder for sample function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 300`** (1 nodes): `Randomly select an img_scale from given candidates.          Args:             i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 301`** (1 nodes): `Randomly sample an img_scale when ``multiscale_mode=='range'``.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 302`** (1 nodes): `Randomly sample an img_scale when ``ratio_range`` is specified.          A ratio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 303`** (1 nodes): `Loss Name.          This function must be implemented and will return the name o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 304`** (1 nodes): `Forward function for `MultiheadAttention`.          **kwargs allow passing a mor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 305`** (1 nodes): `Forward function for `FFN`.          The function would add x to the output tens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 306`** (1 nodes): `Forward function for `FFN`.         The function would add x to the output tenso`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 307`** (1 nodes): `Get the reference points used in decoder.          Args:             spatial_sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 308`** (1 nodes): `Assign boxes to either a ground truth boxes or a negative boxes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 309`** (1 nodes): `nn.Module: normalization layer after the first convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 310`** (1 nodes): `nn.Module: normalization layer after the second convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 311`** (1 nodes): `nn.Module: normalization layer after the first convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 312`** (1 nodes): `nn.Module: normalization layer after the second convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 313`** (1 nodes): `nn.Module: normalization layer after the third convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 314`** (1 nodes): `nn.Module: the normalization layer named "norm1"`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 315`** (1 nodes): `bool: whether the segmentor has neck`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 316`** (1 nodes): `bool: whether the segmentor has auxiliary head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 317`** (1 nodes): `bool: whether the segmentor has decode head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 318`** (1 nodes): `Placeholder for extract features from images.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 319`** (1 nodes): `Placeholder for encode images with backbone and decode into a         semantic s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 320`** (1 nodes): `Placeholder for Forward function for training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 321`** (1 nodes): `Placeholder for single image test.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 322`** (1 nodes): `Placeholder for augmentation test.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 323`** (1 nodes): `Calls either :func:`forward_train` or :func:`forward_test` depending         on`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 324`** (1 nodes): `Parse the raw outputs (losses) of the network.          Args:             losses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 325`** (1 nodes): `Placeholder of forward function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 326`** (1 nodes): `Compute segmentation loss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 327`** (1 nodes): `r"""         Instantiate a [`SiglipConfig`] (or a derived class) from siglip tex`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 328`** (1 nodes): `Preprocess an image or batch of images.          Args:             images (`Imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 329`** (1 nodes): `default_runtime.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 330`** (1 nodes): `cityscapes_1024x1024.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 331`** (1 nodes): `dg_gta_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 332`** (1 nodes): `gta2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 333`** (1 nodes): `syn2city.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 334`** (1 nodes): `gta2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 335`** (1 nodes): `city2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 336`** (1 nodes): `city2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 337`** (1 nodes): `cityscapes_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 338`** (1 nodes): `syn2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 339`** (1 nodes): `bdd100k_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 340`** (1 nodes): `mapillary_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 341`** (1 nodes): `city2bdd-1024.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 342`** (1 nodes): `gta_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 343`** (1 nodes): `syn2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 344`** (1 nodes): `gta2city-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 345`** (1 nodes): `schedule_80k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 346`** (1 nodes): `schedule_40k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 347`** (1 nodes): `schedule_20k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 348`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 349`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 350`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 351`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 352`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 353`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 354`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 355`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 356`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 357`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 358`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 359`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 360`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 361`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 362`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 363`** (1 nodes): `Momentum update of evaluation model (exponential moving average)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 364`** (1 nodes): `download_models.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 365`** (1 nodes): `x: (batch_size, seqlen, nheads, headdim)             cos, sin: (seqlen, rotary_d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 366`** (1 nodes): `logits: (batch, vocab_size)         labels: (batch,)         If process_group is`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 367`** (1 nodes): `Compute proxy supervised loss (depth hint loss) for prediction.              - v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 368`** (1 nodes): `Compute loss masks for each of standard reprojection and depth hint         repr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 369`** (1 nodes): `The best solution from the solver         Returns         -------         x : nd`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 370`** (1 nodes): `The standard deviation of the population energies divided by their         mean.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 371`** (1 nodes): `self_training_ann.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 372`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 373`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 374`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 375`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 376`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 377`** (1 nodes): `NOTE: this interface is experimental.          Args:             is_train: wheth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 378`** (1 nodes): `static_switch.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 379`** (1 nodes): `If process_group is not None and sequence_parallel=True, we're doing Tensor Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 380`** (1 nodes): `xz: (batch, dim, seqlen)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 381`** (1 nodes): `If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 382`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 383`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 384`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 385`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 386`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 387`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 388`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 389`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 390`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 391`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 392`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 393`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 394`** (1 nodes): `prepare_ade20k_ins_seg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 395`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 396`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 397`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 398`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 399`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 400`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 401`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 402`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 403`** (1 nodes): `NOTE: this interface is experimental.          Args:             is_train: wheth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 404`** (1 nodes): `merge_jsons.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 405`** (1 nodes): `Inference interface for the model for PIL image         Args:             pil_im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 406`** (1 nodes): `Load a '.png' segmentation mask, ignoring any colour map.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 407`** (1 nodes): `Load a '.mat' segmentation mask of the kind used in the SBD dataset.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 408`** (1 nodes): `Fields that will be transformed with this transform.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 409`** (1 nodes): `Application of transform to input pipe.          Args:             input_pipe: I`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 410`** (1 nodes): `Comput visualization output.          A visualization method takes some inputs a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 411`** (1 nodes): `Convert instance to segmentation mask.          Args:             instance_mask:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 412`** (1 nodes): `Return current value of hyperparameter based on global step.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 413`** (1 nodes): `Updates the internal state of the metric. In particular, we track update the cos`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 414`** (1 nodes): `Remaps the semantic classes to the target class using the latest assignments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 415`** (1 nodes): `Getter method to access things prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 416`** (1 nodes): `Getter method to access stuffs prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 417`** (1 nodes): `Setter method to access things prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 418`** (1 nodes): `Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 419`** (1 nodes): `Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 420`** (1 nodes): `Files that match these patterns are not deleted by cleanup`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 421`** (1 nodes): `maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 422`** (1 nodes): `check_k80.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 423`** (1 nodes): `Build resume command using latest GCS checkpoint.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 424`** (1 nodes): `Explicit Test-Time Training adaptation.          For each image, perform K gradi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 425`** (1 nodes): `CRF-inspired pairwise consistency loss (differentiable CRF energy).          For`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 426`** (1 nodes): `Fuse DINOv3 and SSD-1B features via learned cross-attention.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 427`** (1 nodes): `Get instance masks from the model.          Args:             features: (B, N, 7`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 428`** (1 nodes): `Generate pseudo-labels from EMA teacher predictions.          Returns semantic l`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 429`** (1 nodes): `Generate pseudo-labels with optional TTA.          Returns:             labels:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 430`** (1 nodes): `Extract CLS attention map as (H_patches, W_patches) numpy array.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 431`** (1 nodes): `Args:             img: (1, 3, H, W) tensor, normalized         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 432`** (1 nodes): `Extract self-attention affinity matrix from last layer.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 433`** (1 nodes): `Extract SD self-attention features for a single image.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 434`** (1 nodes): `Extract SSD-1B self-attention features for a single image.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 435`** (1 nodes): `Extract patch tokens from images.          Args:             images: (B, 3, H, W`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 436`** (1 nodes): `Bipartite matching between predictions and targets.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 437`** (1 nodes): `Post-process a batch of predictions.          Args:             pred_logits: (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 438`** (1 nodes): `Extract DINO features from input image.          Args:             x: Input imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 439`** (1 nodes): `Extract DINOv3 features.          Args:             x: Input image (B, 3, H, W),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 440`** (1 nodes): `Load pretrained DINOv3 weights from HuggingFace.          Args:             mode`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Attack` connect `Community 2` to `Community 9`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Why does `Trainer` connect `Community 2` to `Community 0`, `Community 19`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Are the 151 inferred relationships involving `PQProxyLoss` (e.g. with `Instance decomposition methods for unsupervised panoptic segmentation.  All meth` and `TestSemanticLoss`) actually correct?**
  _`PQProxyLoss` has 151 INFERRED edges - model-reasoned connections that need verification._
- **Are the 143 inferred relationships involving `Segment_TR` (e.g. with `Resize so short side = crop_size, both dims divisible by patch_size.` and `Extract 90-dim CAUSE features for a single 322x322 crop.     Returns: (1, 90, 23`) actually correct?**
  _`Segment_TR` has 143 INFERRED edges - model-reasoned connections that need verification._
- **Are the 136 inferred relationships involving `Mamba2Stack` (e.g. with `BidirectionalCrossModalScan` and `Bidirectional Cross-Modal Scan (BiCMS).  Implements bidirectional Mamba2 scannin`) actually correct?**
  _`Mamba2Stack` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 119 inferred relationships involving `Attack` (e.g. with `Phy_obj_atk_Square` and `r"""     Square Attack in the paper 'Square Attack: a query-efficient black-box`) actually correct?**
  _`Attack` has 119 INFERRED edges - model-reasoned connections that need verification._
- **Are the 133 inferred relationships involving `DINOViTS8` (e.g. with `MBPSModel` and `Unified MBPS Model.  End-to-end Unsupervised Mamba-Bridge Panoptic Segmentation`) actually correct?**
  _`DINOViTS8` has 133 INFERRED edges - model-reasoned connections that need verification._