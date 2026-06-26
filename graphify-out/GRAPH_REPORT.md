# Graph Report - .  (2026-06-16)

## Corpus Check
- 11990 files · ~243,345,246 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 79334 nodes · 116601 edges · 3019 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Trainer` - 121 edges
2. `__Pyx_AddTraceback()` - 88 edges
3. `_create_resnet()` - 68 edges
4. `utils()` - 58 edges
5. `KITTIDataset` - 53 edges
6. `Dataset` - 50 edges
7. `NuScenes` - 46 edges
8. `_gen_efficientnet()` - 46 edges
9. `LidarDataset` - 43 edges
10. `VisionTransformer` - 40 edges

## Surprising Connections (you probably didn't know these)
- `MBPS Evaluation Script (PyTorch).  Uses torch.no_grad() for efficient inference` --uses--> `ExperimentConfig`  [INFERRED]
  mbps_pytorch/scripts/evaluate.py → unsupervised_instance_generator/source/test-instance-labels/Superpixels/superpixels_uis/config.py
- `Dataset loaders for MBPS.  Supports Cityscapes, COCO-Stuff-27, and NYU Depth V2.` --uses--> `Dataset`  [INFERRED]
  mbps_pytorch/data/datasets.py → unsupervised-panoptic-segmentation/refs/noah-research/CURL/data.py
- `Compute Sobel dx, dy on a 2D depth map.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py
- `Load a single sample from val set.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py
- `Run inference and instrument every stage.` --uses--> `DepthGuidedUNet`  [INFERRED]
  debug_instance_inference.py → mbps_pytorch/refine_net.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (1144): find_calib(), main(), base = <city>_<seq>_000019 ; prefer the exact 019 camera json, else closest in t, build_backbone(), main(), Return (token_extractor(x)->[B,N,C]) for a frozen backbone., build_backbone(), main() (+1136 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (1204): accuracy(), Accuracy calculation module., Module to calculate the accuracy.          Args:             topk (tuple, option, Calculate accuracy according to the prediction and target.      Args:         pr, Forward function to calculate accuracy.          Args:             pred (torch.T, BaseActor, Adahessian, AdaHessian Optimizer  Lifted from https://github.com/davda54/ada-hessian/blob/ma (+1196 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (838): AntiAliasDownsampleLayer, Downsample, DownsampleJIT, ATSSTargetAssigner, Args:             anchors: [(N, 7), ...]             gt_boxes: (B, M, 8), Args:             anchors: (N, 7) [x, y, z, dx, dy, dz, heading]             gt_, Reference: https://arxiv.org/abs/1912.02424, Adain (+830 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (1240): adaptive_edge_instances(), Adaptive Depth-Feature Edge Fusion for instance decomposition.  Uses depth Sobel, Instance decomposition via adaptive depth-feature edge fusion.      Feature edge, BaseImageProcessor, BEVFeaturesInterpolation, bilinear_interpolate_torch(), Args:         im: (H, W, C) [y, x]         x: (N)         y: (N)      Returns:, contrastive_instances() (+1232 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (1293): encode_semantic_onehot(), infer_semantic_spec(), map_to_trainid(), Semantic-label handling for AdaptiveInstanceNet.  The adaptive instance adapter, Encode semantic labels as smoothed one-hot channels., Map semantic labels to Cityscapes trainIDs, using 255 for unknown., Validate semantic/depth wiring and return summary stats., Resolved semantic-label contract for adapter training/generation. (+1285 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (1231): assign_and_save(), balanced_kmeans(), collect_features(), evaluate_hungarian(), find_feature_files(), main(), Run balanced k-means: standard k-means + iterative capacity balancing.      Args, Assign each pixel to nearest centroid, save as PNG. (+1223 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (1244): assign_hierarchical(), collect_per_cluster_features(), compute_adaptive_sub_k(), depth_to_segments(), downsample_depth(), evaluate_hungarian(), find_files(), fit_sub_clusters() (+1236 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (738): DatasetWithEnumeratedTargets, If pad_dataset is set, pads based on torch's DistributedSampler implementation,, BDD10kPanopticValidation, get_bdd2cs_class_mapping(), This class implements the KITTI panoptic validation dataset., Constructor method.          Args:             root (str): Path to the dataset., Returns the length of the dataset.          Returns:             length (int): L, Method returns an instances of the dataset given its index.          Args: (+730 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (736): activation_count_operators(), find_unused_parameters(), flop_count_operators(), FlopCountAnalysis, Implement operator-level activations counting using jit.     This is a wrapper o, Given a model, find parameters that do not contribute     to the loss.      Args, Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models., Args:             model (nn.Module):             inputs (Any): inputs of the giv (+728 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (833): AdaptiveLoss, AdaptiveLossConfig, This is an implementation of the loss function accompanying the adaptive softmax, Compute the loss for the given sample.          Returns a tuple with three eleme, align_a1_majority(), align_a2_selective(), align_a3_confidence(), align_a4_majority_stuff() (+825 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (587): GlobalSimilar, GSA, ImgShift(), LocalSimilar, LSA, PatchExtra(), PositionEmbeddingSine, This is a more standard version of the position embedding, very similar to the o (+579 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (722): ADE20KDataset, ADE20K dataset.      In segmentation map annotation for ADE20K, 0 stands for bac, add_sin_difference(), Anchor3DHead, loss(), Initialize the target assigner and sampler of the head., Initialize neural network layers of the head., Forward function on a single-scale feature map.          Args:             x (to (+714 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (486): C_IC_SS(), computeAffinities2(), Algorithm, AngleDiff(), AngleDiffPrecomputed(), ComputeAlpha(), DiffAt(), GetBeta() (+478 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (699): ADE20K, _file_to_segmentation_path(), _load_file_paths(), _load_segmentation(), _Split, convert_path_or_url_to_url(), dinov2_vitb14(), dinov2_vitb14_reg() (+691 more)

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (542): CompDecoder, CompDecoderStrong, CompEncoder, Flatten, ImageEncoderBg, MaskDecoder, PredictComp, PredictMask (+534 more)

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (465): AsyncNodeWithCacheAndConcurrencyLimit, AdapterConfig, AsyncNodeWithCacheAndConcurrencyLimit, aug_test(), BASE, Base3DDetector, Base3DSegmentor, BaseAdapter (+457 more)

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (472): ADE20KPanoptic, ADE20KSemantic, calc_dynamics(), load_seq_class(), main(), plot_cartography(), Plot a cartography table     Based on: https://www.aclweb.org/anthology/2020.emn, load_seq_class() (+464 more)

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (543): attn_vis(), cls_padding(), grid_show(), highlight_grid(), grid_size=14是因为patch_size=16, 所以有224/16=14个patch     grid_index代表的是第几个grid,这个gri, visualize_grid_to_grid(), visualize_grid_to_grid_with_cls(), visualize_heads() (+535 more)

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (422): AnchorHead, get_bboxes(), loss(), Get anchors according to feature map sizes.          Args:             featmap_s, Anchor-based head (RPN, RetinaNet, SSD, etc.).      Args:         num_classes (i, Transform outputs for a single batch item into labeled boxes., AnchorHead, ATSSHead (+414 more)

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (531): AnchorGenerator, _broadcast_params(), BufferList, build_anchor_generator(), _create_grid_offsets(), DefaultAnchorGenerator, __init__(), Returns:             list[Tensor]: #featuremap tensors, each is (#locations x #c (+523 more)

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (417): build_group_list(), build_group_manifest(), _cleanup_old_tasks(), create_app(), _gallery_url_join(), InferenceRequest, InferenceResponse, _is_plain_name() (+409 more)

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (375): BaseConvBboxHead, Forward.          Args:             feats (Tensor): Input features          Retu, r"""More general bbox head, with shared conv layers and two optional     separat, Add shared or separable branch., BaseMono3DDenseHead, get_bboxes(), loss(), Args:             x (list[Tensor]): Features from FPN.             img_metas (li (+367 more)

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (356): BaseTrainer, is_rank_zero(), Adds model and dataloaders to the trainer.          Overriding methods should ca, Completes the setup of the trainer.          Overriding methods should call this, Base Trainer class for training a model., Basic code for training 1 epoch, Basic code for running inference, Checking which metric to use for selecting a checkpoint (+348 more)

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (312): ABC, IAssets, AssignResult, BaseAssigner, IdentityAssigner, MaskHungarianAssigner, Collection of assign results., Base assigner that assigns boxes to ground truth boxes. (+304 more)

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (317): discover_worker_checkpoints(), evaluate_checkpoint(), load_config(), load_eval_result(), main(), Coordinator script for aggregating results from worker VMs.  Runs on a single co, Evaluate a single checkpoint.      Imports evaluate_model lazily to avoid loadin, Coordinator main loop. (+309 more)

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (354): child_runnable::run(), run(), runnable_group::run(), start(), cluster_sizes(), number_of_clusters(), compare(), FilterParsingError (+346 more)

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (322): Crop, my_app(), _random_crops(), RandomCropComputer, Crop the given image into four corners and the central crop.     If the image is, Crop the given image into four corners and the central crop.     If the image is, Crops image and masks., Adobe5kDataLoader (+314 more)

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (329): infer_type(), parse_unknown(), BaseTracker, build_tracker_head(), A parent class for all trackers, Args:             predictions: D2 Instances for predictions of the current frame, Build a tracker head from `cfg.TRACKER_HEADS.TRACKER_NAME`.      Args:         c, BaseHungarianTracker (+321 more)

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (324): Backbone, BackboneWithPositionEncoding, BasicBlock, BasicConv2d, Block17, Block35, Block8, Bottleneck (+316 more)

### Community 29 - "Community 29"
Cohesion: 0.01
Nodes (315): _cfg_to_stage_args(), _create_cspnet(), create_stem(), CrossStage, cspdarknet53(), cspdarknet53_iabn(), CspNet, cspresnet50() (+307 more)

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (314): AdjustGamma, Albu, apply_min_size(), Aug, bbox2result(), bbox2roi(), bbox3d2result(), bbox3d2roi() (+306 more)

### Community 31 - "Community 31"
Cohesion: 0.0
Nodes (268): data_parallel(), DataParallel, ListDataParallel, r"""Implements data parallelism at the module level.      This container paralle, r"""     Slices variables into approximately equal chunks and     distributes th, # TODO: update notes/cuda.rst when this class handles 8+ GPUs well, r"""Scatter with support for kwargs dictionary, r"""Evaluates module(input) in parallel across the GPUs given in device_ids. (+260 more)

### Community 32 - "Community 32"
Cohesion: 0.0
Nodes (210): Adam, FairseqAdam, FairseqAdamConfig, Performs a single optimization step.          Args:             closure (callabl, Adam optimizer for fairseq.      Important note: this optimizer corresponds to t, Reduce Params is only used during BMUF distributed training., r"""Implements Adam algorithm.      This implementation is modified from torch.o, AdaptiveSpanCriterion (+202 more)

### Community 33 - "Community 33"
Cohesion: 0.01
Nodes (324): formatResearchStatus(), statusLabel(), build_messages_for_session(), convert_session_to_tag_input(), create_conversation_id(), extract_sessions(), get_index_path(), load_config() (+316 more)

### Community 34 - "Community 34"
Cohesion: 0.0
Nodes (337): CityscapesEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, Evaluate semantic segmentation results on cityscapes dataset using cityscapes AP, Base class for evaluation using cityscapes API., Args:             dataset_name (str): the name of the dataset.                 I, Evaluate instance segmentation results on cityscapes dataset using cityscapes AP, Returns:             dict: has a key "segm", whose value is a dict of "AP" and " (+329 more)

### Community 35 - "Community 35"
Cohesion: 0.0
Nodes (342): all_reduce(), all_reduce_norm(), get_async_norm_states(), _get_reduce_op(), pyobj2tensor(), serialize picklable python object to tensor, deserialize tensor to picklable python object, Apply all reduce function for python dict object.     NOTE: make sure that every (+334 more)

### Community 36 - "Community 36"
Cohesion: 0.0
Nodes (315): benchmark_backbone(), main(), Lightweight FPN decoder for panoptic segmentation., Benchmark a single backbone + FPN decoder., SimpleFPNDecoder, sync_device(), ClipImageModel, ClipTextModel (+307 more)

### Community 37 - "Community 37"
Cohesion: 0.0
Nodes (298): ActGLU, CGUBidirDualEncoder, CGU, CGUStage, CrossActGLU, DWConv, LayerTransition, Image to Patch Embedding (+290 more)

### Community 38 - "Community 38"
Cohesion: 0.0
Nodes (277): generate(), postprocess_small_regions(), prompt_switch(), Using a SAM model, generates masks for the entire image.         Generates a gri, Using a SAM model, generates masks for the entire image.         Generates a gri, SamAutomaticMaskGenerator, SemanticSamAutomaticMaskGenerator, BaseDataset (+269 more)

### Community 39 - "Community 39"
Cohesion: 0.0
Nodes (181): AsrDataset, Return an example's size as a float or tuple. This value is used when         fi, Return an ordered list of indices. Batches will be constructed based         on, A dataset representing speech and corresponding transcription.      Args:, Merge a list of samples to form a mini-batch.          Args:             samples, backtranslate_samples(), BacktranslationDataset, Merge and backtranslate a list of samples to form a mini-batch.          Using t (+173 more)

### Community 40 - "Community 40"
Cohesion: 0.0
Nodes (330): batched_mask_to_box(), build_all_layer_point_grids(), build_point_grid(), calculate_stability_score(), generate_crop_boxes(), get_amg_kwargs(), is_box_near_crop_edge(), main() (+322 more)

### Community 41 - "Community 41"
Cohesion: 0.0
Nodes (326): load_coco_unlabel_json(), _get_builtin_metadata(), _get_coco_instances_meta(), _get_coco_panoptic_separated_meta(), _get_imagenet_instances_meta(), _get_sa1b_instances_meta(), _get_UVO_instances_meta(), Returns metadata for "separated" version of the panoptic segmentation dataset. (+318 more)

### Community 42 - "Community 42"
Cohesion: 0.01
Nodes (269): apply_metric_scaling(), compute_alignment_mask(), compute_sky_mask(), least_squares_scale_scalar(), Sample tensor elements for quantile computation to reduce memory usage.      Arg, Apply metric scaling to depth based on camera intrinsics.      Args:         dep, Set sky regions to maximum depth and high confidence.      Args:         depth:, Compute least squares scale factor s such that a ≈ s * b.      Args:         a: (+261 more)

### Community 43 - "Community 43"
Cohesion: 0.01
Nodes (362): CNNBlockBase, DummyResNet, Implements a dummy ResNet wrapper for demonstration purpose.     Args:         *, NoStemRegNet, Override the original function that do not initialize a stem layer         since, Forward function of backbone.          Args:             x (torch.Tensor): Featu, RegNet backbone without Stem for 3D detection.      More details can be found in, adjust_block_compatibility() (+354 more)

### Community 44 - "Community 44"
Cohesion: 0.01
Nodes (325): adaptive_avgmax_pool2d(), adaptive_catavgmax_pool2d(), adaptive_pool_feat_mult(), AdaptiveAvgMaxPool2d, AdaptiveCatAvgMaxPool2d, FastAdaptiveAvgPool2d, PyTorch selectable adaptive pooling Adaptive pooling with the ability to select, Selectable global pooling function with dynamic input kernel size (+317 more)

### Community 45 - "Community 45"
Cohesion: 0.01
Nodes (229): BaseTransformerLayer, create_embedder(), Embedder, EmbedderType, GloveEmbedder, load_glove_vectors(), preprocess_glove_vectors(), Produce vertex embeddings for the specific mesh; vertex embeddings are         a (+221 more)

### Community 46 - "Community 46"
Cohesion: 0.01
Nodes (293): BaseObserver, BaseQuantizer, _add_category_id_to_contiguous_id_maps_to_metadata(), _add_category_info_to_bootstrapping_metadata(), _add_category_maps_to_metadata(), _add_category_whitelists_to_metadata(), _BootstrapDatasetFactoryCatalog, build_backbone() (+285 more)

### Community 47 - "Community 47"
Cohesion: 0.01
Nodes (318): BasePredictor, area(), iou(), postprocess(), create_annotation_info(), create_image_info(), Return image_info in COCO style     Args:         image_id: the image ID, Return annotation info in COCO style     Args:         annotation_id: the annota (+310 more)

### Community 48 - "Community 48"
Cohesion: 0.01
Nodes (201): EncoderDecoderAug, Run forward function and calculate loss for decode head in         training., Forward function for training.          Args:             img (Tensor): Input im, Encoder Decoder segmentors.      EncoderDecoder typically consists of backbone,, EncoderDecoderPrompt, load_state_dict(), MultiscaleEncoderDecoderPrompt, Encode images with backbone and decode into a semantic segmentation         map (+193 more)

### Community 49 - "Community 49"
Cohesion: 0.01
Nodes (301): CopydaysDataset, extract_features(), ImgListDataset, Compute the average precision of one search.     ranks = ordered list of ranks o, score_ap_from_ranks_1(), OxfordParisDataset, extract_feature_pipeline(), extract_features() (+293 more)

### Community 50 - "Community 50"
Cohesion: 0.01
Nodes (148): _bdist_wheel, build_ext, get_compiling_cuda_version(), get_cudart_version(), ApexScaler, NativeScaler, CUDA / AMP utils  Hacked together by / Copyright 2020 Ross Wightman, InterpolationFunction (+140 more)

### Community 51 - "Community 51"
Cohesion: 0.01
Nodes (220): AdaptiveMask, AdaptiveSpan, mask attention with the right span, how much of memory can be trimmed to reduce computation, trim out unnecessary memory beforehand to reduce computation, Soft masking function for adaptive size.     It masks out the last K values of a, determine how long the cache should be, a loss term for regularizing the span length (+212 more)

### Community 52 - "Community 52"
Cohesion: 0.01
Nodes (145): build_model(), check_decoder_output(), check_encoder_output(), CrossEntropyCriterionTestBase, _current_postion_info(), DummyEncoder, DummyEncoderModel, DummyTask (+137 more)

### Community 53 - "Community 53"
Cohesion: 0.01
Nodes (239): generate_and_evaluate_baseline(), get_prediction_json_path(), panop_baselines_from_lidarseg_detect_track(), prepare_files(), Script to generate baselines for Panoptic nuScenes tasks. Code written by Motion, Generate panoptic predictions by merging a lidarseg method and a tracking (or de, Prepare the files containing the predictions of the various method names.     :p, Get the name of the json file in a directory (abort if there is more than one). (+231 more)

### Community 54 - "Community 54"
Cohesion: 0.01
Nodes (294): ChannelShuffle, CondConvResidual, ConvBnAct, DepthwiseSeparableConv, EdgeResidual, get_bn_args_tf(), InvertedResidual, make_divisible() (+286 more)

### Community 55 - "Community 55"
Cohesion: 0.01
Nodes (297): abs_py_ssize_t(), assert_direct_dimensions(), __Pyx_AddTraceback(), __pyx_align_pointer(), __pyx_array___cinit__(), __pyx_array___dealloc__(), __pyx_array_get_memview(), __pyx_array___getattr__() (+289 more)

### Community 56 - "Community 56"
Cohesion: 0.01
Nodes (187): fetch_dataloader(), FlowDataset, FlyingThings3D, HD1K, KITTI, MpiSintel, MpiSintel_submission, fetch_dataloader() (+179 more)

### Community 57 - "Community 57"
Cohesion: 0.01
Nodes (191): Script to verify that the nuScenes installation is complete., verify_setup(), main(), Performs inference for all of the baseline models defined in the physics model m, get_lidarseg_num_points_per_class(), get_panoptic_num_instances_per_class(), Truncate a given class name according to a pre-defined map.     :param class_nam, Get the number of points belonging to each class for the given nuScenes split. (+183 more)

### Community 58 - "Community 58"
Cohesion: 0.01
Nodes (236): AgentRepresentation, add_present_time_to_history(), AgentBoxesWithFadedHistory, default_colors(), draw_agent_boxes(), fade_color(), get_track_box(), pixels_to_box_corners() (+228 more)

### Community 59 - "Community 59"
Cohesion: 0.01
Nodes (195): create_attn_masks_and_pos(), DataCollatorWithPaddingForCLM, DataCollatorWithPaddingForCorruptCLM, Data collator used for causal language modeling.     - collates batches of tenso, Data collator used for causal language modeling.     - collates batches of tenso, Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10%, CustomTrainer, Setup the scheduler. The optimizer of the trainer must have been set up either b (+187 more)

### Community 60 - "Community 60"
Cohesion: 0.01
Nodes (173): build_lr_scheduler(), Build a LR scheduler from config., CompositeParamScheduler, FeaturizerWithUpsampling, PART 1: First run the below to generate outputs python eval_davis_video_seg.py -, run_video_segmentation(), my_app(), SemSegEvaluator (+165 more)

### Community 61 - "Community 61"
Cohesion: 0.01
Nodes (256): acc_all(), acc_all_stderr(), accuracy(), AccuracyAveraging, adjusted_rand_index(), AdjustedRandIndex, aggregate(), _aggregate_from_pairwise_values() (+248 more)

### Community 62 - "Community 62"
Cohesion: 0.01
Nodes (187): AspectRatioGroupedDataset, AspectRatioGroupedDataset, AspectRatioGroupedDatasetTwoCrop, AverageMeter, avg_pool(), avg_unpool(), _collect_config_files(), conv() (+179 more)

### Community 63 - "Community 63"
Cohesion: 0.01
Nodes (152): Aggregate, Attention, Attention1D, BertLayerNorm, BroadMultiHeadAttention, __call__(), CausalSelfAttention, convert_qkv_to_q_and_kv_proj() (+144 more)

### Community 64 - "Community 64"
Cohesion: 0.01
Nodes (176): all_reduce_dict(), all_reduce_scalar(), all_reduce_tensor(), _allreduce_coalesced(), allreduce_grads(), DistOptimizerHook, _get_global_gloo_group(), obj2tensor() (+168 more)

### Community 65 - "Community 65"
Cohesion: 0.01
Nodes (167): ApproxMaxIoUAssigner, Assign gt to approxs.          This method assign a gt bbox to each group of app, Assign gt to approxs.          This method assign a gt bbox to each group of app, Assign a corresponding gt bbox or background to each bbox.      Each proposals w, CenterHead, Args:             gt_boxes: (N, 8)             feature_map_size: (2), [x, y], Args:             gt_boxes: (B, M, 8)             range_image_polar: (B, 3, H, W, reorder_rois_for_refining() (+159 more)

### Community 66 - "Community 66"
Cohesion: 0.01
Nodes (135): Simple Feature Pyramid for ViT → multi-scale features.  Converts single-scale Vi, Convert multi-layer ViT features to a multi-scale FPN.      Args:         in_dim, Convert 4 ViT layer features to multi-scale pyramid.          Args:, SimpleFeaturePyramid, DINOv3Backbone, Mask2FormerPanoptic, Mask2Former for Panoptic Segmentation with DINOv3 ViT-L/16 Backbone.  Full model, Full forward pass.          Args:             images: (B, 3, H, W) normalized in (+127 more)

### Community 67 - "Community 67"
Cohesion: 0.01
Nodes (133): CNN, Conv2dPixelShuffle, DecoderTAESD, DecoderVAESD, DINO2ViT, DINO2ViT_P16, EncoderTAESD, EncoderVAESD (+125 more)

### Community 68 - "Community 68"
Cohesion: 0.01
Nodes (198): ConverterV1, ConverterV2, downgrade(), downgrade_config(), guess_version(), A converter that handles simple rename., A large bulk of rename, before public release., Upgrade a config from its current version to a newer version.      Args: (+190 more)

### Community 69 - "Community 69"
Cohesion: 0.01
Nodes (96): BroadcastMLPDecoder, build(), DINOSAUR, DINOSAURT, - input: video, shape=(b,t,c,h,w)         - condit: condition, shape=(b,t,n,c), - input: destructed target, shape=(b,m,c)         - slotz: slots, shape=(b,n,c), Args:             encoder: Module that creates features from the input image., - input: image, shape=(b,c,h,w)         - condit: condition, shape=(b,n,c) (+88 more)

### Community 70 - "Community 70"
Cohesion: 0.01
Nodes (161): AddBBoxFromInstanceMasks, AddEmptyBboxes, AddEmptyMasks, AddImageSize, AddSegmentationMaskFromInstanceMask, AddTemporalAxis, adjust_small_size(), CanonicalizeBboxes (+153 more)

### Community 71 - "Community 71"
Cohesion: 0.01
Nodes (145): accuracy(), benchmark_data(), benchmark_data_advanced(), benchmark_eval(), benchmark_train(), create_data_benchmark(), DataLoaderBenchmark, _EmptyMapDataset (+137 more)

### Community 72 - "Community 72"
Cohesion: 0.01
Nodes (109): GPT2Config, GPT2OnnxConfig, # TODO: how to do that better?, This is the configuration class to store the configuration of a [`GPT2Model`] or, custom_get_block_length_and_num_blocks(), custom_unfold(), expand_attention_types_params(), GPTNeoConfig (+101 more)

### Community 73 - "Community 73"
Cohesion: 0.01
Nodes (180): Augmentation, apply_augmentations(), AugInput, Augmentation, AugmentationList, build_augmentation(), _check_img_dtype(), CopyPasteAugmentation (+172 more)

### Community 74 - "Community 74"
Cohesion: 0.01
Nodes (52): Att_pooling, Att_pooling2, Diffusion_cond, Diffusion_net, Diffusion_war, DISN, EFEMSDF, EFEMSDF_UNet (+44 more)

### Community 75 - "Community 75"
Cohesion: 0.01
Nodes (181): CLEVRTEX, CLEVRTEX_Evaluator, CLEVRTEXPair, DatasetReadError, get_clevrtex(), get_clevrtex_pair(), Drop unimportanat, unsued or incorrect data from metadata.         Data may beco, Reindexes tensor along <dim> using reindex_tensor.         Effectivelly permutes (+173 more)

### Community 76 - "Community 76"
Cohesion: 0.01
Nodes (111): anchor_inside_flags(), anchor_target(), anchor_target_single(), images_to_levels(), Unmap a subset of item (data) back to the original set of items (of     size cou, Compute regression and classification targets for anchors.      Args:         an, Convert targets by image to targets by feature level.      [target_img0, target_, unmap() (+103 more)

### Community 77 - "Community 77"
Cohesion: 0.01
Nodes (84): accuracy(), AnyMatchAccuracy, AveragingMethod, build_classification_metric(), build_topk_accuracy_metric(), build_topk_any_match_accuracy_metric(), build_topk_recall_metric(), ClassificationMetricType (+76 more)

### Community 78 - "Community 78"
Cohesion: 0.01
Nodes (147): get_args(), set_remaining_args(), _overfit_model(), test_overfit(), test_train(), _train_one_pass(), _train_overfit(), _validate() (+139 more)

### Community 79 - "Community 79"
Cohesion: 0.01
Nodes (104): build_decoder(), build_encoder(), FairseqNATDecoder, FairseqNATEncoder, FairseqNATModel, Abstract class for all nonautoregressive-based models, build_encoder(), gru_transformer_base_architecture() (+96 more)

### Community 80 - "Community 80"
Cohesion: 0.01
Nodes (129): ColorAugSSDTransform, A color related data augmentation used in Single Shot Multibox Detector (SSD)., calc_padding(), calc_params(), calc_slicing(), cam_quat_xyzw_to_world_quat_wxyz(), CenterCrop, Clip (+121 more)

### Community 81 - "Community 81"
Cohesion: 0.01
Nodes (95): AutoCausalLM, AutoSeq2SeqLM, _get_accelerate_args(), _get_dtype(), HuggingFaceAutoLM, MultiTokenEOSCriteria, Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation., # TODO: Support evaluating causal models with special tokens. Currently, (+87 more)

### Community 82 - "Community 82"
Cohesion: 0.01
Nodes (136): ASPP, Atrous Spatial Pyramid Pooling (ASPP)., Args:             in_channels (int): number of input channels for ASPP., convert_frozenbatchnorm2d_to_batchnorm2d(), CycleBatchNormList, FrozenBatchNorm2d, get_norm(), LayerNorm (+128 more)

### Community 83 - "Community 83"
Cohesion: 0.02
Nodes (51): get_graph_feature(), get_graph_feature_cross(), get_graph_mean(), get_shell_mean_cross(), knn(), x: point features of shape [B, N_feat, 3, N_samples, ...], x: point features of shape [B, N_feat, 3, N_samples, ...], x: point features of shape [B, N_feat, 3, N_samples, ...] (+43 more)

### Community 84 - "Community 84"
Cohesion: 0.02
Nodes (95): PatchEmbedUnSafe, Vision Transformer with support for patch or hybrid CNN input stage, Vision Transformer with support for global average pooling, Vision Transformer with support for global average pooling, Image to Patch Embedding, Vision Transformer with support for global average pooling, SharedAdapter, VisionTransformer (+87 more)

### Community 85 - "Community 85"
Cohesion: 0.02
Nodes (143): append_history(), DINOv3DenseDecoder, export(), feature_batch(), load_feature(), main(), ManifestDataset, mask_box() (+135 more)

### Community 86 - "Community 86"
Cohesion: 0.02
Nodes (142): _add_ids(), color_map(), colorize_mask(), depth2rgb(), disp2rgb(), draw_arrows_in_rgb(), draw_grid_arrows_in_rgb(), _draw_instance_contours() (+134 more)

### Community 87 - "Community 87"
Cohesion: 0.02
Nodes (53): _lang_token(), _lang_token_index(), MultiddsMultilingualTranslationTask, prepare(), Load a dataset split., Return language token index., Do forward and backward, and return the loss as computed by *criterion*, A task for training multiple translation models simultaneously.      We iterate (+45 more)

### Community 88 - "Community 88"
Cohesion: 0.02
Nodes (76): apply_augment(), Augment, augment_list(), Cutout, get_augment(), CustomFormatter, init(), init_cfg() (+68 more)

### Community 89 - "Community 89"
Cohesion: 0.02
Nodes (69): MVFuser, hidden_states: (B, L, D)         Returns: same shape as hidden_states, BlockChunk, DinoVisionTransformer, Args:             img_size (int, tuple): input image size             patch_size, Attention, Block, DropPath (+61 more)

### Community 90 - "Community 90"
Cohesion: 0.02
Nodes (97): get_filepath(), get_pointcloud(), get_transforms(), KittiDB, KITTIInstanceSegmentation, KITTIPanopticValidation, KITTIPanopticValidationStereo, KITTIRaw (+89 more)

### Community 91 - "Community 91"
Cohesion: 0.02
Nodes (103): __Pyx_AddTraceback(), __pyx_bisect_code_objects(), __Pyx_BufFmt_CheckString(), __Pyx_BufFmt_DescribeTypeChar(), __Pyx_BufFmt_ExpectNumber(), __Pyx_BufFmt_Init(), __pyx_buffmt_parse_array(), __Pyx_BufFmt_ParseNumber() (+95 more)

### Community 92 - "Community 92"
Cohesion: 0.02
Nodes (83): add_file_to_dictionary(), _add_file_to_dictionary_single_worker(), Dictionary, index_data(), load(), Return unknown string, optionally escaped as: <<unk>>, Threshold on the word frequency counts., Adds a word to the dictionary (+75 more)

### Community 93 - "Community 93"
Cohesion: 0.02
Nodes (107): extract(), extract_sd_features(), global_clustering(), hungarian_miou(), load_coco_panoptic_gt(), LocalAffinity, LocalAffinityCopy, LocalStDev (+99 more)

### Community 94 - "Community 94"
Cohesion: 0.02
Nodes (57): butterfly4D, CenterPivotConv4d, conv4d(), fullConv4d, projfeat4d, r""" Implementation of center-pivot 4D convolution, This is done by stacking results of multiple 3D convolutions, and is very slow., Turn 3d projection into 2d projection (+49 more)

### Community 95 - "Community 95"
Cohesion: 0.02
Nodes (78): cached_path(), filename_to_url(), get_from_cache(), http_get(), load_archive_file(), Utilities for working with the local dataset cache. This file is adapted from th, Split a full s3 path into the bucket name and path., Return the url and etag (which may be ``None``) stored for `filename`.     Raise (+70 more)

### Community 96 - "Community 96"
Cohesion: 0.02
Nodes (56): AverageMeter, _DerivedMeter, Meter, MetersDict, Computes the average occurrence of some event per second, Computes the sum/avg duration of some event in seconds, A sorted dictionary of :class:`Meters`.      Meters are sorted according to a pr, Get a single smoothed value. (+48 more)

### Community 97 - "Community 97"
Cohesion: 0.02
Nodes (102): Cluster, ClusterType, get_checkpoint_path(), get_cluster_type(), get_most_confident_predictions(), get_slurm_account(), get_slurm_executor_parameters(), get_slurm_partition() (+94 more)

### Community 98 - "Community 98"
Cohesion: 0.02
Nodes (76): DepthModel, compute_errors(), eval(), predict_tta(), add_asr_eval_argument(), check_args(), cli_main(), ExistingEmissionsDecoder (+68 more)

### Community 99 - "Community 99"
Cohesion: 0.03
Nodes (78): NoopSessionStore, AgentSession, _auto_approval_summary(), _can_access_session(), _cleanup_sandbox(), EventBroadcaster, _has_active_sandbox_preload(), Operation (+70 more)

### Community 100 - "Community 100"
Cohesion: 0.02
Nodes (58): Callback, Callback, DINOEMA, update(), AverageLog, CollectLog, recursively convert data into numpy, SaveModel (+50 more)

### Community 101 - "Community 101"
Cohesion: 0.03
Nodes (44): BartAttention, BartClassificationHead, BartDecoder, BartDecoderLayer, BartDecoderWrapper, BartEncoder, BartEncoderLayer, BartForCausalLM (+36 more)

### Community 102 - "Community 102"
Cohesion: 0.02
Nodes (51): apply_mv_norm(), batch_by_size(), calc_mean_invstddev(), collate_tokens(), collect_filtered(), compute_mask_indices(), encoder_padding_mask_to_lengths(), filter_by_size() (+43 more)

### Community 103 - "Community 103"
Cohesion: 0.03
Nodes (50): CrossAttentionLayer, HDADecoder, initialize_flow(), MemoryDecoder, MemoryDecoderLayer, PreActBlock, Spatial Broadcast Decoder for feature reconstruction from slots.  Based on DINOS, x:      [B*H1*W1, 1, C]             memory: [B*H1*W1, H2'*W2', C]             co (+42 more)

### Community 104 - "Community 104"
Cohesion: 0.02
Nodes (41): FreeMaskPreprocessing, BasePreprocessing, _dict_to_yaml(), _load_yaml(), make_instance_database(), preprocess(), process_file.          Args:             filepath: path to the main file, process_file.          Args:             filepath: path to the main file (+33 more)

### Community 105 - "Community 105"
Cohesion: 0.02
Nodes (103): affine_matrix_from_points(), angle_between_vectors(), Arcball, arcball_constrain_to_axis(), arcball_map_to_sphere(), arcball_nearest_axis(), clip_matrix(), compose_matrix() (+95 more)

### Community 106 - "Community 106"
Cohesion: 0.02
Nodes (85): Unit tests for MBPS core modules (PyTorch).  Tests cover:     - Model components, Instance scores should be in [0, 1]., Test Adaptive Projection Bridge., Projection should output bridge-dim features., Test Adaptive Projection Bridge., Projection should output bridge-dim features., Alignment loss should be non-negative., Test Mamba2 SSD module. (+77 more)

### Community 107 - "Community 107"
Cohesion: 0.02
Nodes (63): Block, ConvBlock, CrossBlock, CrossGlobalSubSampleAttn, CrossGlobalSubSampleAttnRPE, FeatureFusionBlock, GlobalSubSampleAttn, GlobalSubSampleAttnRPE (+55 more)

### Community 108 - "Community 108"
Cohesion: 0.04
Nodes (60): getFloat3x3(), getInverse(), getTranslation(), inverse(), setFloat3x3(), setIdentity(), setTranslation(), setZero() (+52 more)

### Community 109 - "Community 109"
Cohesion: 0.03
Nodes (65): FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder(), _make_pretrained_resnext101_wsl(), _make_resnet_backbone(), _make_scratch(), Interpolation module. (+57 more)

### Community 110 - "Community 110"
Cohesion: 0.03
Nodes (44): DinoV3Feature, ConvBlock, CSyncBatchNorm, CustomSequential, DINOHead, DPTHead, FeatureFusionBlock, iBOTHead (+36 more)

### Community 111 - "Community 111"
Cohesion: 0.03
Nodes (36): fake_sparse_idx(), PartA2FCHead, Args:             batch_dict:                 batch_size:                 rois:, Args:             batch_dict:          Returns:, PointRCNNHead, Args:             batch_dict:          Returns:, Args:             batch_dict:                 batch_size:                 rois:, PointRCNNScoreHead (+28 more)

### Community 112 - "Community 112"
Cohesion: 0.02
Nodes (67): Unit tests for all loss functions (PyTorch).  Tests cover:     - Semantic loss (, InstanceLoss should return dict with total., InstanceLoss should return dict with total., loss = 1 - iou + d/c         where,         d = (distance between centers of the, Instance loss should not produce NaN., Instance loss should not produce NaN., Test bridge loss functions., Test bridge loss functions. (+59 more)

### Community 113 - "Community 113"
Cohesion: 0.03
Nodes (65): bbox_overlaps(), Calculate the ious between each bbox of bboxes1 and bboxes2.      Args:, ade_classes(), ade_palette(), cityscapes_classes(), cityscapes_palette(), get_classes(), get_palette() (+57 more)

### Community 114 - "Community 114"
Cohesion: 0.03
Nodes (68): build_model_mask2former(), Mask2FormerUnsupervisedModel, PyTorch Lightning module for Mask2Former training within the CUPS pipeline.  Sub, Optimizer with differential learning rates for Mask2Former.          - Backbone, Build a Mask2Former model wrapped in the CUPS Lightning module.      Args:, Lightning module for Mask2Former trained on pseudo-labels.      Inherits validat, Training step adapted for Mask2Former.          Mask2Former returns losses with, build_model_pseudo_loss_only() (+60 more)

### Community 115 - "Community 115"
Cohesion: 0.03
Nodes (54): build_local_aggregation_module(), get_dense_voxels_by_center(), PointnetFPModule, PointnetLFPModuleMSG, PointnetSAModule, _PointnetSAModuleBase, PointnetSAModuleMSG, PointnetSAModuleMSGVotes (+46 more)

### Community 116 - "Community 116"
Cohesion: 0.03
Nodes (62): NoisyChannelBeamSearch, generate(), make_dict2dict(), NoisyChannelSequenceGenerator, normalized_scores_with_batch_vocab(), Generates translations of a given source sentence,            using beam search, Get normalized probabilities (or log probs) from a net's output         w.r.t. v, reorder_all_tokens() (+54 more)

### Community 117 - "Community 117"
Cohesion: 0.03
Nodes (45): BaseBBoxCoder, CenterPointBBoxCoder, Bbox coder for CenterPoint.      Args:         pc_range (list[float]): Range of, Decode bboxes.          Args:             heat (torch.Tensor): Heatmap with the, Given feats and indexes, returns the gathered feats.          Args:, Get indexes based on scores.          Args:             scores (torch.Tensor): s, Given feats and indexes, returns the transposed and gathered feats.          Arg, DeltaXYZWLHRBBoxCoder (+37 more)

### Community 118 - "Community 118"
Cohesion: 0.03
Nodes (22): AV2Dataset, get_av2_train_dataset(), get_av2_val_dataset(), KittiObjectDataset, main(), get_kitti_train_dataset(), KittiRawDataset, main() (+14 more)

### Community 119 - "Community 119"
Cohesion: 0.04
Nodes (33): FeatureExtractor, LiteFlowNet2, liteflownet2_pseudoreg, LiteFlowNet2PseudoReg, Matching, PseudoRegularization, PseudoSubpixel, Regularization (+25 more)

### Community 120 - "Community 120"
Cohesion: 0.03
Nodes (62): calculate_uncertainty(), dice_loss(), dice_loss_droploss(), dice_loss_weight(), pairwise_iou(), Create the criterion.         Parameters:             num_classes: number of obj, Args:         inputs: A float tensor of arbitrary shape.                 The pre, Args:         inputs: A float tensor of arbitrary shape.                 The pre (+54 more)

### Community 121 - "Community 121"
Cohesion: 0.04
Nodes (48): crop_from_xywh(), extract_openclip_crop_features(), OpenCLIPTextEncoder, PrecomputedTextClassifier, PromptBank, Equation 1 classifier using precomputed CLIP text and crop features., Optional helper for producing the text side of Eq. 1., cosine_similarity() (+40 more)

### Community 122 - "Community 122"
Cohesion: 0.03
Nodes (43): BaseFairseqModel, decoder(), encoder(), FairseqEncoderDecoderModel, FairseqEncoderModel, FairseqLanguageModel, FairseqModel, FairseqMultiModel (+35 more)

### Community 123 - "Community 123"
Cohesion: 0.04
Nodes (29): ConvFFN, DWConv, Extractor, get_reference_points(), Injector, InteractionBlockPrompt, SpatialPriorModule, Attention (+21 more)

### Community 124 - "Community 124"
Cohesion: 0.04
Nodes (81): build_instance_map(), build_instance_map_cc(), build_instance_map_depth_cc(), build_instance_map_depth_guided(), compute_distributions(), determine_thing_cluster_ids(), find_semantic_files(), load_instance_npz() (+73 more)

### Community 125 - "Community 125"
Cohesion: 0.04
Nodes (47): FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder(), _make_scratch(), Interpolation module., Init.          Args:             scale_factor (float): scaling             mode, Forward pass.          Args:             x (tensor): input          Returns: (+39 more)

### Community 126 - "Community 126"
Cohesion: 0.04
Nodes (39): BlockChunk, DinoV2, DINOv2Featurizer, DinoVisionTransformer, init_weights_vit_timm(), named_apply(), ViT weight initialization, original timm impl (for reproducibility), ViT weight initialization, original timm impl (for reproducibility) (+31 more)

### Community 127 - "Community 127"
Cohesion: 0.04
Nodes (17): create_dummy_roberta_head_data(), eval_lm_main(), # TODO: langs should be in and out right?, Neither ----checkpoint-activations nor --offload-activations should change loss, --checkpoint-activations should not change loss, read_last_log_entry(), test_transformer_xl_bptt_lm(), TestActivationCheckpointing (+9 more)

### Community 128 - "Community 128"
Cohesion: 0.04
Nodes (54): build_model(), main(), Group-B (mobile/BiFPN) semantic-prediction dumper for 19-class re-eval.  Loads a, apply_ignore_unknown_thing_regions(), BiFPN, boundary_loss(), BoundaryHead, build_cups_optimizer() (+46 more)

### Community 129 - "Community 129"
Cohesion: 0.04
Nodes (40): Base3DSegmentor, BroadcastDecoderNet, EncoderDecoder, EncoderDecoder3D, EncoderNet, _input_generation(), Run forward function and calculate loss for decode head in         inference., Run forward function and calculate loss for decode head in         inference. (+32 more)

### Community 130 - "Community 130"
Cohesion: 0.05
Nodes (62): Custom30M, MyRes16UNet14, MyRes16UNet18, Res16UNet101, Res16UNet14, Res16UNet14A, Res16UNet14A2, Res16UNet14B (+54 more)

### Community 131 - "Community 131"
Cohesion: 0.05
Nodes (75): _abandon_pending_approval(), _approval_decision(), ApprovalDecision, _assistant_message_from_result(), _base_needs_approval(), _budget_block_reason(), _call_llm_non_streaming(), _call_llm_streaming() (+67 more)

### Community 132 - "Community 132"
Cohesion: 0.06
Nodes (75): add_resolution_checks(), append_published_issue_section(), apply_resolution_checks(), async_main(), _call_json_llm(), _classification_messages(), classify_records(), collect_github_sources() (+67 more)

### Community 133 - "Community 133"
Cohesion: 0.03
Nodes (73): bbox_corner_dist_measure(), point_cloud_to_bbox(), point_cloud_to_image(), point_cloud_to_image_batch(), point_cloud_to_volume(), point_cloud_to_volume_batch(), point_cloud_to_volume_v2(), point_cloud_to_volume_v2_batch() (+65 more)

### Community 134 - "Community 134"
Cohesion: 0.04
Nodes (31): _catalog_shared_params(), _get_module_by_path(), lr_scheduler(), MultiUncertainTrainer, optimizer(), Sync logging outputs across workers. all_gather_list_sync is         suitable wh, Sync logging outputs across workers. fast_stat_sync_sum is         faster than a, Check that grad norms are consistent across workers. (+23 more)

### Community 135 - "Community 135"
Cohesion: 0.04
Nodes (31): _catalog_shared_params(), _get_module_by_path(), lr_scheduler(), MultiddsTrainer, optimizer(), Sync logging outputs across workers. all_gather_list_sync is         suitable wh, Sync logging outputs across workers. fast_stat_sync_sum is         faster than a, Check that grad norms are consistent across workers. (+23 more)

### Community 136 - "Community 136"
Cohesion: 0.07
Nodes (64): artifact_paths(), batch_to_vis_state(), build_cups_stage_model(), center_crop_arr(), center_crop_vis_state(), channel_contact_sheet(), city_from_stem(), cups_center_crop_tensor() (+56 more)

### Community 137 - "Community 137"
Cohesion: 0.03
Nodes (24): Adadelta, Adafactor, FairseqAdafactor, _get_lr(), _get_options(), Adafactor Optimizer  Lifted from https://github.com/pytorch/fairseq/blob/master/, Performs a single optimization step.          Args:             closure (callabl, Implements Adafactor algorithm.     This implementation is based on: `Adafactor: (+16 more)

### Community 138 - "Community 138"
Cohesion: 0.04
Nodes (68): box2d_to_corner_jit(), box3d_to_bbox(), box_camera_to_lidar(), boxes3d_to_corners3d_lidar(), camera_to_lidar(), center_to_corner_box2d(), center_to_corner_box3d(), center_to_minmax_2d() (+60 more)

### Community 139 - "Community 139"
Cohesion: 0.04
Nodes (26): McpToolManager, Update the tool mapping for a specific server by fetching available tools from M, Split a fully qualified tool name into server name and tool name components., Add new MCP servers to the manager.          Args:             mcp_server_links:, Remove MCP servers from the manager and clean up connections.          Args:, List available tools, optionally filtered by server name and tool controller., MCP (Model Context Protocol) Tool Manager for managing multiple MCP servers and, Execute tool calls for the provided tasks.          Args:             tasks: Lis (+18 more)

### Community 140 - "Community 140"
Cohesion: 0.04
Nodes (38): RectifyNet, RotDecoder, DecoderBN, Compute the depths bins used to build the cost volume. Bins will depend upon, Compute a cost volume based on L1 difference between current_feats and lookup_fe, Compute the depths bins used to build the cost volume. Bins will depend upon, Constructs a resnet model with varying number of input images.     Adapted from, Compute a cost volume based on L1 difference between current_feats and lookup_fe (+30 more)

### Community 141 - "Community 141"
Cohesion: 0.06
Nodes (64): build_job_matrix(), build_smoke_jobs(), check_job_progress(), check_vm_state(), create_vm(), delete_vm(), gcloud_ssh(), generate_smoke_vm_specs() (+56 more)

### Community 142 - "Community 142"
Cohesion: 0.05
Nodes (18): change_dict_keys(), change_k_v(), collate_list_data(), downsample_dict(), draw_heat_regression_maps(), get_augmentation_transform(), get_centermaps_downsampling_factor(), get_centermaps_output_grid_size() (+10 more)

### Community 143 - "Community 143"
Cohesion: 0.04
Nodes (26): BackgroundConsumer, BufferedIterator, _chunk_iterator(), CountingIterator, EpochBatchIterating, EpochBatchIterator, GroupedIterator, next_epoch_idx() (+18 more)

### Community 144 - "Community 144"
Cohesion: 0.03
Nodes (32): BasePoints, cat(), color(), device(), height(), Base class for Points.      Args:         tensor (torch.Tensor | np.ndarray | li, Shuffle the points.          Returns:             torch.Tensor: The shuffled ind, Rotate points with the given rotation matrix or angle.          Args: (+24 more)

### Community 145 - "Community 145"
Cohesion: 0.05
Nodes (40): augment_and_mix_transform(), AugmentOp, augmix_ops(), AugMixAugment, auto_augment_policy(), auto_augment_policy_original(), auto_augment_policy_originalr(), auto_augment_policy_v0() (+32 more)

### Community 146 - "Community 146"
Cohesion: 0.04
Nodes (27): LinformerSentenceEncoderLayer, Implements a Linformer Encoder Layer used in BERT/XLM style pre-trained     mode, LinformerSentenceEncoder, Implementation for a Bi-directional Linformer based Sentence Encoder used     in, _append_prev_key_padding_mask(), MultiheadLinearAttention, Input shape: Time x Batch x Channel          Args:             key_padding_mask, Multi-headed linformer attention.      Projects the key and values down to the c (+19 more)

### Community 147 - "Community 147"
Cohesion: 0.04
Nodes (32): CommonMetricPrinter, EventStorage, EventWriter, get_event_storage(), has_event_storage(), JSONWriter, Args:             json_file (str): path to the json file. New data will be appen, Write all scalars to a tensorboard file. (+24 more)

### Community 148 - "Community 148"
Cohesion: 0.06
Nodes (63): _anchor_scores_from_indices(), anchored_proposal_mask_loss(), anchored_query_class_loss(), anchored_query_score_loss(), balanced_mask_bce_with_logits(), code_preservation_loss(), compute_jpc_up_losses(), dice_loss_from_probs() (+55 more)

### Community 149 - "Community 149"
Cohesion: 0.05
Nodes (27): BasicEncoder, BasicMotionEncoder_v2, BasicUpdateBlock, bilinear_sampler(), BottleneckBlock, ConvBNReLU, ConvGRU, coords_grid() (+19 more)

### Community 150 - "Community 150"
Cohesion: 0.05
Nodes (19): DeformDataset, load_deformable_dset_slice_train_val(), load_point_maze_slice_train_val(), PointMazeDataset, load_pusht_slice_train_val(), PushTDataset, Subset, _accumulate() (+11 more)

### Community 151 - "Community 151"
Cohesion: 0.05
Nodes (45): backward(), forward(), hard_mish_bwd(), hard_mish_fwd(), hard_mish_jit_bwd(), hard_mish_jit_fwd(), hard_sigmoid_bwd(), hard_sigmoid_fwd() (+37 more)

### Community 152 - "Community 152"
Cohesion: 0.05
Nodes (27): actualDisplay(), addCombinator(), ajaxConvert(), ajaxHandleResponses(), Animation(), augmentWidthOrHeight(), condense(), createFxNow() (+19 more)

### Community 153 - "Community 153"
Cohesion: 0.04
Nodes (23): DynamicLossScaler, build_fp32_params(), build_optimizer(), FP16Optimizer, _FP16OptimizerMixin, MemoryEfficientFP16Optimizer, _MemoryEfficientFP16OptimizerMixin, Multiplies grads by a constant ``c``. (+15 more)

### Community 154 - "Community 154"
Cohesion: 0.04
Nodes (37): RandomResizedCrop, center_crop(), CenterCropVideo, CenterFullCropVideo, crop(), DenormalizeVideo, from_tensor(), FromTensorVideo (+29 more)

### Community 155 - "Community 155"
Cohesion: 0.05
Nodes (39): AutoregressivePatchDecoder, BBoxOutput, build_grid_of_positions(), DensityPredictingSlotAttentionDecoder, DepthReconstructionOutput, DVAEDecoder, get_dvae_decoder(), get_dvae_encoder() (+31 more)

### Community 156 - "Community 156"
Cohesion: 0.04
Nodes (54): _build_knn_graph(), CascadeMaskHead, compute_affinity_matrix(), compute_spatial_confidence(), crf_refine(), CutS3DModule, extract_pseudo_masks(), extract_pseudo_masks_batch() (+46 more)

### Community 157 - "Community 157"
Cohesion: 0.05
Nodes (24): Block, ConvNeXt, ConvNeXt_Extractor, ConvNextBlock, ConvNextLayer, drop_path(), DropPath, LayerNorm (+16 more)

### Community 158 - "Community 158"
Cohesion: 0.04
Nodes (48): compute_panoptic_quality(), _distributed_sum(), map_to_target(), _miou_compute(), _panoptic_quality_compute(), PanopticQualitySemanticMatching, PQResult, print_metrics() (+40 more)

### Community 159 - "Community 159"
Cohesion: 0.05
Nodes (32): _clip_coords(), _compute_intersection(), CustomDataAugmentation, CustomRandomHorizontalFlip, CustomTwoCrop, GaussianBlur, _get_coord(), _get_image_size() (+24 more)

### Community 160 - "Community 160"
Cohesion: 0.06
Nodes (36): beta(), create(), FastAIMixedOptim, get_master(), is_tuple(), listify(), lr(), master2model() (+28 more)

### Community 161 - "Community 161"
Cohesion: 0.06
Nodes (28): add_residual(), Attention, Block, BlockChunk, dinov2_vit_base_14(), dinov2_vit_large_14(), dinov2_vit_small_14(), DinoVisionTransformer (+20 more)

### Community 162 - "Community 162"
Cohesion: 0.05
Nodes (51): write optical flow in .flo format to file as used in the Sintel dataset (Butler, write optical flow in .flo format to file as used in the Sintel dataset (Butler, read optical flow from file stored in png file format as used in the KITTI 12 (G, read flow files in several formats. The resulting flow has shape height x width, read optical flow from file stored in png file format as used in the KITTI 12 (G, write optical flow to file png file format as used in the KITTI 12 (Geiger et al, write optical flow to file png file format as used in the KITTI 12 (Geiger et al, read numpy array from file.     filepath: file to read from     returns: numpy a (+43 more)

### Community 163 - "Community 163"
Cohesion: 0.05
Nodes (32): FairseqNATDecoder, FairseqNATModel, build_decoder(), forward_mask_ins(), forward_word_del(), forward_word_ins(), levenshtein_base_architecture(), levenshtein_transformer_vaswani_wmt_en_de_big() (+24 more)

### Community 164 - "Community 164"
Cohesion: 0.06
Nodes (40): amp_ctx(), ArmModule, build_index(), compute_pq_for_image(), Config, copy_paste_batch(), correspondence_loss(), decide_mode() (+32 more)

### Community 165 - "Community 165"
Cohesion: 0.05
Nodes (43): _is_dora(), _is_plain_linear(), MockCustomBlock, MockDA2Model, MockDA3Model, MockDepthPro, MockDinov2Attention, MockDinov2Layer (+35 more)

### Community 166 - "Community 166"
Cohesion: 0.06
Nodes (27): AvgPool, BNGELU, CDilated, Conv, DilatedConv, LayerNorm, LGFI, LiteCVEncoder (+19 more)

### Community 167 - "Community 167"
Cohesion: 0.04
Nodes (43): compute_fg_mask(), FocalLossCenterNet, _gather_feat(), get_corner_loss_lidar(), neg_loss_cornernet(), Sigmoid focal cross entropy loss., Sigmoid focal cross entropy loss., Args:             input: (B, #anchors, #codes) float tensor.                 Eco (+35 more)

### Community 168 - "Community 168"
Cohesion: 0.06
Nodes (28): add_residual(), Attention, Block, BlockChunk, dinov2_vit_base_14(), dinov2_vit_large_14(), dinov2_vit_small_14(), DinoVisionTransformer (+20 more)

### Community 169 - "Community 169"
Cohesion: 0.07
Nodes (31): _add_default_env(), _add_environment_variables(), _async_call(), _build_uv_command(), _ensure_hf_transfer_dependency(), _filter_uv_install_output(), hf_jobs_handler(), HfJobsTool (+23 more)

### Community 170 - "Community 170"
Cohesion: 0.04
Nodes (50): box2d_iou(), box3d_iou(), box3d_vol(), convex_hull_intersection(), get_3d_box(), get_3d_box_batch(), get_iou(), poly_area() (+42 more)

### Community 171 - "Community 171"
Cohesion: 0.07
Nodes (54): _clean_description(), _error(), _find_section(), _format_citation_entry(), _format_citation_graph(), _format_collections(), _format_collections_compact(), _format_datasets() (+46 more)

### Community 172 - "Community 172"
Cohesion: 0.06
Nodes (31): bilateral_solver_output(), BilateralGrid, BilateralSolver, bistochastize(), get_valid_idx(), Compute diagonal matrices to bistochastize a bilateral grid, Find which values are present in a list and where they are located, Hacky function to turn a coordinate into a unique value (+23 more)

### Community 173 - "Community 173"
Cohesion: 0.06
Nodes (22): BasicConv, Conv2x, DAP, DICL, DICL_MODULE, FeatureGA, FlowEntropy, FlowRegression (+14 more)

### Community 174 - "Community 174"
Cohesion: 0.07
Nodes (14): Conv2d, Embedding, Linear, LinearMoELora, mark_only_lora_as_trainable(), MoeLoraConfig, MoeLoraLayer, MoeLoraModel (+6 more)

### Community 175 - "Community 175"
Cohesion: 0.06
Nodes (48): _demodata_refine_boxes(), _get_config_directory(), _get_config_module(), _get_head_cfg(), _get_parta2_bbox_head_cfg(), _get_pointrcnn_bbox_head_cfg(), _get_pointrcnn_rpn_head_cfg(), _get_pts_bbox_head_cfg() (+40 more)

### Community 176 - "Community 176"
Cohesion: 0.06
Nodes (30): gelu(), gelu_tanh(), GELUTanh, hard_mish(), hard_sigmoid(), hard_swish(), HardMish, HardSigmoid (+22 more)

### Community 177 - "Community 177"
Cohesion: 0.05
Nodes (45): auth_status(), _cleanup_expired_states(), get_me(), get_redirect_uri(), logout(), _missing_required_scopes(), oauth_callback(), oauth_login() (+37 more)

### Community 178 - "Community 178"
Cohesion: 0.06
Nodes (26): DetailedRetrievalResult, MultiStageRetriever, Multi-Stage Retriever  Three-stage retrieval pipeline: 1. Temporal filtering usi, 快速评估是否需要 Stage 3                  基于方案 B：标签覆盖度判断                  Args:, 从 storage 获取记忆数据                  尝试从 episode storage 和 semantic storage 中查找, 从内存索引批量加载（O(n) 但常数极小）                  优先从 UnifiedIndex._episode_metadata 获取, 格式化 Episode 内容                  如果 Episode.content 为空，从 messages 生成摘要, 统计总记忆数                  包含 episodes 和 semantic memories (+18 more)

### Community 179 - "Community 179"
Cohesion: 0.04
Nodes (19): FairseqTask, LegacyFairseqTask, Load a given dataset split.          Args:             split (str): name of the, Return a loaded dataset split.          Args:             split (str): name of t, Filter examples that are too large          Args:             indices (np.array), Get an iterator that yields batches of data from the given dataset.          Arg, Tasks store dictionaries and provide helpers for loading/iterating over     Data, Build the :class:`~fairseq.models.BaseFairseqModel` instance for this         ta (+11 more)

### Community 180 - "Community 180"
Cohesion: 0.04
Nodes (35): ConvDoRALinear, cosine_warmup_lambda(), DoRAConfig, DoRALinear, _expand_adapter_rank(), expand_lora_coverage(), expand_lora_rank(), get_dora_param_groups() (+27 more)

### Community 181 - "Community 181"
Cohesion: 0.06
Nodes (49): accumulate_sim3_transforms(), align_point_maps(), apply_transformation_numba(), compute_alignment_error(), compute_chunk_scale_advanced(), compute_huber_weights_numba(), compute_residuals_numba(), compute_scale_ransac() (+41 more)

### Community 182 - "Community 182"
Cohesion: 0.05
Nodes (31): ConvDoRALinear, count_adapter_params(), count_total_params(), DoRALinear, freeze_non_adapter_params(), LoRAConv2d, LoRALinear, merge_all_adapters() (+23 more)

### Community 183 - "Community 183"
Cohesion: 0.05
Nodes (38): _concat(), DictSchema, flatten(), flatten_to_tuple(), IdentitySchema, InstancesSchema, ListSchema, For classes that are simple wrapper of tensors, e.g.     Boxes, RotatedBoxes, Bi (+30 more)

### Community 184 - "Community 184"
Cohesion: 0.06
Nodes (34): calc_intersect_cross(), calc_ios1(), calc_ios2(), calc_iou(), calc_iou_cross(), calc_prob_same_mask(), calc_union_cross(), dev2mask_valid() (+26 more)

### Community 185 - "Community 185"
Cohesion: 0.05
Nodes (19): MemoryManager, 将 Episode 转换为用于 embedding 的文本                  Args:             episode: Episod, 删除 Episode（同时删除索引和存储）                  Args:             user_id: 用户ID, 添加 SemanticMemory（同时更新索引和存储）                  Args:             memory: Semantic, 删除 SemanticMemory（同时删除索引和存储）                  Args:             user_id: 用户ID, 列出用户的 SemanticMemories, 统计用户的 SemanticMemory 数量, 检索相关的 Episodes（语义搜索 + 过滤）                  Args:             user_id: 用户ID (+11 more)

### Community 186 - "Community 186"
Cohesion: 0.07
Nodes (35): CAUSETRModel, COCOImageDataset, contrastive_loss(), DINOv3Backbone, entropy_regularization(), evaluate_model(), load_coco_panoptic_gt(), main() (+27 more)

### Community 187 - "Community 187"
Cohesion: 0.06
Nodes (23): differential_evolution(), DifferentialEvolutionSolver, Copied from "https://github.com/DebangLi/one-pixel-attack-pytorch/"  A slight mo, This class implements the differential evolution solver     Parameters     -----, Finds the global minimum of a multivariate function.     Differential Evolution, Initializes the population with Latin Hypercube Sampling.         Latin Hypercub, Initialises the population at random.  This type of initialization         can p, Initialises the population with a user specified population.         Parameters (+15 more)

### Community 188 - "Community 188"
Cohesion: 0.06
Nodes (50): build_depth_cc_prior(), build_superpixel_graph(), compute_clip_confidence(), cosine_rows(), depth_gradient(), find_superpixel_edges(), generate_superpixels(), instances_from_edge_probs() (+42 more)

### Community 189 - "Community 189"
Cohesion: 0.05
Nodes (27): get_metric_data(), _mock_results(), Drop one prediction from the GT submission., Drop the first three predictions from the GT submission., Calculate and check the AP value.         :param gts: Ground truth data., Change the tracking_id of one frame from the GT submission., Tests the correctness of AP calculation for simple cases., Drop one box from the GT. (+19 more)

### Community 190 - "Community 190"
Cohesion: 0.07
Nodes (19): attention(), Attention_Layer, build_transformer(), CrossAttention, _get_activation_fn(), _get_clones(), MLP, MLP_v2() (+11 more)

### Community 191 - "Community 191"
Cohesion: 0.08
Nodes (20): create_lang_dictionary(), get_dataset_key(), get_langtok_index(), get_shard_id(), _get_shard_num_dict(), _lang_id(), load_all_dictionaries(), load_data() (+12 more)

### Community 192 - "Community 192"
Cohesion: 0.07
Nodes (48): _batch_box_iou(), _batch_iou(), _box_iou_one(), _compute_hungarian_mapping(), _compute_majority_mapping(), _connected_components_things(), discover_pairs(), evaluate_instances() (+40 more)

### Community 193 - "Community 193"
Cohesion: 0.06
Nodes (28): _configure_libraries(), fixup_module_metadata(), _import_file(), Load custom environment setup by importing a Python source file or a     module,, Fix the __qualname__ of module members to be their exported api name, so     whe, Set the random seed for the RNG in torch, numpy and python.      Args:         s, Configurations for some libraries., Perform environment setup work. The default setup is a no-op, but this     funct (+20 more)

### Community 194 - "Community 194"
Cohesion: 0.07
Nodes (27): Cluster, cluster_assignment_matrix(), codebook_index(), compute_modularity_based_codebook(), compute_self_distance_batch(), cos_distance_matrix(), Decoder, flatten() (+19 more)

### Community 195 - "Community 195"
Cohesion: 0.05
Nodes (42): box_regression_loss(), confidence_alpha_blending(), confident_copy_paste_selection(), copy_paste_augment(), CutS3DInstanceLoss, dice_loss(), drop_loss(), greedy_iou_matching() (+34 more)

### Community 196 - "Community 196"
Cohesion: 0.07
Nodes (27): ConvGNAct, _coord_grid(), DepthFiLM2d, DepthwiseResidualBlock, DynamicKernelLifter, EfficientPanopticCouplingBlock, JointPanopticCouplerUp, JPCUpConfig (+19 more)

### Community 197 - "Community 197"
Cohesion: 0.08
Nodes (41): affinity_to_rama_cost(), CoarseMaskResult, corner_background_objectness(), _corner_count(), _edge_ring(), generate_coarse_masks(), local_affinity_map(), _mask_area_fraction() (+33 more)

### Community 198 - "Community 198"
Cohesion: 0.07
Nodes (16): BasicTransformerBlock, CrossAttention, DiffusionModel, GEGLU, ResBlock, SpatialTransformer, td_dot(), Upsample (+8 more)

### Community 199 - "Community 199"
Cohesion: 0.09
Nodes (12): compute_dice_loss(), compute_sigmoid_ce_loss(), ddim_sample(), extract(), get_evenly_distributed_colors(), linear_beta_schedule(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre (+4 more)

### Community 200 - "Community 200"
Cohesion: 0.09
Nodes (12): compute_dice_loss(), compute_sigmoid_ce_loss(), ddim_sample(), extract(), get_evenly_distributed_colors(), linear_beta_schedule(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre (+4 more)

### Community 201 - "Community 201"
Cohesion: 0.06
Nodes (24): Heading, _clip_to_width(), _format_help_row(), format_help_text(), format_plan_display(), _format_stats(), _help_column_widths(), _LeftHeading (+16 more)

### Community 202 - "Community 202"
Cohesion: 0.06
Nodes (21): ChainedGenerator, DictMapper, patched_pathsplit(), Patches `torchdata` for behavior to be consistent with webdatasets., Split a path into a WebDataset prefix and suffix.      The version of pathsplit, Simple interface to allow chaining via a generator function.      This mirrors f, IterDataPipe, FakeDataset (+13 more)

### Community 203 - "Community 203"
Cohesion: 0.06
Nodes (16): AnchorHeadMulti, SingleHead, AnchorHeadSingle, AnchorHeadSingleV2, get_layer(), add_sin_difference(), AnchorHeadTemplate, generate_anchors() (+8 more)

### Community 204 - "Community 204"
Cohesion: 0.06
Nodes (22): bb3d_2_bb2d(), corners3d_to_img_boxes(), get_registration_angle(), :param corners3d: (N, 8, 3) corners in rect coordinate     :return: boxes: (None, register_bbs(), initialize the the 3D tracker         Args:             tracking_features: bool,, compute the cost map between detections and predictions         Returns:, greedy assign the IDs for detected state based on the cost map         Returns: (+14 more)

### Community 205 - "Community 205"
Cohesion: 0.08
Nodes (11): _doc_id(), get_session_store(), MongoSessionStore, NoopSessionStore, _now(), Optional durable session persistence for the hosted backend.  The public CLI mus, MongoDB-backed session store., Return a Mongo-safe message document payload.      Mongo's hard document limit i (+3 more)

### Community 206 - "Community 206"
Cohesion: 0.06
Nodes (31): apply_ignore_unknown_thing_regions(), build_optimizer(), CascadeGradientScaler, _cc_inference(), compute_center_offset_loss(), compute_mask_cls_loss(), CopyPasteAugmentation, CUPSCityscapesDataset (+23 more)

### Community 207 - "Community 207"
Cohesion: 0.06
Nodes (18): get_ind(), R3MBuffer, IterableDataset, code(), CombinedDataset, CombinedDatasetIterator, dtype(), PackedDataset (+10 more)

### Community 208 - "Community 208"
Cohesion: 0.05
Nodes (24): build_camera_matrix(), Camera, extrinsics2RT(), load_labels(), permute_pointcloud(), PlyWriter, point_indices_from_group(), Save an RGB point cloud as a PLY file.      Args:       points_3d: Nx6 matrix wh (+16 more)

### Community 209 - "Community 209"
Cohesion: 0.05
Nodes (44): boxes3d_kitti_camera_to_imageboxes(), boxes3d_kitti_camera_to_lidar(), boxes3d_kitti_fakelidar_to_lidar(), boxes3d_kitti_lidar_to_fakelidar(), boxes3d_lidar_to_aligned_bev_boxes(), boxes3d_lidar_to_kitti_camera(), boxes3d_nearest_bev_iou(), boxes3d_to_corners3d_kitti_camera() (+36 more)

### Community 210 - "Community 210"
Cohesion: 0.07
Nodes (31): build_model(), build_scheduler(), CityscapesSegDataset, DINOv3FeatureExtractor, evaluate_model(), generate_pseudolabels(), _get_autocast_ctx(), get_layer_indices() (+23 more)

### Community 211 - "Community 211"
Cohesion: 0.08
Nodes (28): CIFAR100SSL, CIFAR10SSL, get_cifar10(), get_cifar100(), get_cifar10_cld_aug(), get_labeled_inds(), TransformFixMatch, x_u_split() (+20 more)

### Community 212 - "Community 212"
Cohesion: 0.07
Nodes (38): box3d_transform_(), box_collision_test(), get_pyramids(), global_rotation(), global_scaling(), local_pyramid_dropout(), local_pyramid_sparsify(), local_pyramid_swap() (+30 more)

### Community 213 - "Community 213"
Cohesion: 0.08
Nodes (20): build(), build_mlp(), build_two_layer_mlp(), get_activation_fn(), Presence_NN, Presence_NN_sim_cluster, Presence_NN_sim_former, Presence_NN_sim_merger (+12 more)

### Community 214 - "Community 214"
Cohesion: 0.06
Nodes (23): Block, get_1d_sincos_pos_embed_from_grid(), get_2d_sincos_pos_embed(), get_2d_sincos_pos_embed_from_grid(), GlobalSubSampleAttn, LocallyGroupedAttn, PatchEmbed, PosConv (+15 more)

### Community 215 - "Community 215"
Cohesion: 0.07
Nodes (35): AdaptiveInstanceLoss, _build_fusion_inputs(), contrastive_embedding_loss(), _count_cityscapes_images(), _count_sam3_files(), embedding_regularization_loss(), evaluate_pq_things(), feature_boundary_loss() (+27 more)

### Community 216 - "Community 216"
Cohesion: 0.11
Nodes (9): compute_dice_loss(), compute_sigmoid_ce_loss(), get_evenly_distributed_colors(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre, remove_duplications(), ReplayMemory, sample_points_from_mesh() (+1 more)

### Community 217 - "Community 217"
Cohesion: 0.05
Nodes (33): build_model_pseudo(), log_visualizations(), Just wraps the forward pass of the Cascade Panoptic Mask R-CNN.          Args:, Just wraps the forward pass of the Cascade Panoptic Mask R-CNN.          Args:, Training step.          Args:             batch (List[Dict[str, Any]])): Batch o, Training step.          Args:             batch (List[Dict[str, Any]])): Batch o, Just wraps the forward pass of the Cascade Panoptic Mask R-CNN.          Args:, Training step.          Args:             batch (List[Dict[str, Any]])): Batch o (+25 more)

### Community 218 - "Community 218"
Cohesion: 0.07
Nodes (30): crf_refine_slot_masks(), load_model(), main(), process_single_image(), Convert DINOSAUR slot masks to instance masks with semantic classification., Convert pixel-resolution slot masks (from CRF) to instances.      Args:, Process one image through DINOSAUR and extract instances.      Args:         use, Load trained DINOSAUR model from checkpoint. (+22 more)

### Community 219 - "Community 219"
Cohesion: 0.07
Nodes (24): absolute2relative(), Edge, GraphCut, GraphCutMaster, a class for a pixel, which is represented as a vertex in the graph, return the total weights of edges connected to the node, implementation of graph cut, :param square_distance_matrix: squared distance matrix for each feature point (+16 more)

### Community 220 - "Community 220"
Cohesion: 0.11
Nodes (10): compute_dice_loss(), compute_sigmoid_ce_loss(), generation(), get_evenly_distributed_colors(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre, remove_duplications(), ReplayMemory (+2 more)

### Community 221 - "Community 221"
Cohesion: 0.1
Nodes (30): depth2rgb(), disp2rgb(), draw_arrows_in_rgb(), draw_circles_in_rgb(), draw_grid_arrows_in_rgb(), draw_pixels(), draw_text_in_rgb(), flow2rgb() (+22 more)

### Community 222 - "Community 222"
Cohesion: 0.06
Nodes (31): backward(), dropout_add_layer_norm(), _dropout_add_layer_norm_backward(), _dropout_add_layer_norm_forward(), dropout_add_layer_norm_parallel_residual(), _dropout_add_layer_norm_parallel_residual_backward(), _dropout_add_layer_norm_parallel_residual_forward(), dropout_add_layer_norm_subset() (+23 more)

### Community 223 - "Community 223"
Cohesion: 0.07
Nodes (15): id_next(), lock(), unlock(), abstract_lock(), abstract_read_lock(), abstract_read_unlock(), abstract_unlock(), abstract_write_lock() (+7 more)

### Community 224 - "Community 224"
Cohesion: 0.08
Nodes (16): CrossAttentionLayer, FFNLayer, _get_activation_fn(), Mask3D, PositionalEncoding3D, :param channels: The last dimension of the tensor you want to apply pos emb to., :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)         :return: Po, :param channels: The last dimension of the tensor you want to apply pos emb to. (+8 more)

### Community 225 - "Community 225"
Cohesion: 0.11
Nodes (9): compute_dice_loss(), compute_sigmoid_ce_loss(), get_evenly_distributed_colors(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre, remove_duplications(), ReplayMemory, sample_points_from_mesh() (+1 more)

### Community 226 - "Community 226"
Cohesion: 0.11
Nodes (9): compute_dice_loss(), compute_sigmoid_ce_loss(), get_evenly_distributed_colors(), Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre, remove_duplications(), ReplayMemory, sample_points_from_mesh() (+1 more)

### Community 227 - "Community 227"
Cohesion: 0.06
Nodes (21): BENGE, BENGEDataModule, BENGESampleCollateFn, KorniaPrepAugmentation, Method to setup dataset and corresponding splits., Return sample `idx` as dictionary from the dataset., Return training dataset loader., Return validation dataset loader. (+13 more)

### Community 228 - "Community 228"
Cohesion: 0.07
Nodes (16): HumanRankEval, HumanRankEvalAppleAndroid, HumanRankEvalCPP, HumanRankEvalCSDB, HumanRankEvalEnglish, HumanRankEvalHTML, HumanRankEvalJava, HumanRankEvalLanguagesSciences (+8 more)

### Community 229 - "Community 229"
Cohesion: 0.1
Nodes (40): apply_semantic_mapping(), colorize_id_map(), compute_cityscapes27_assignment(), fill_panoptic_void(), _is_waymo_rgb(), load_image_tensor(), load_models(), load_semantic_mappings() (+32 more)

### Community 230 - "Community 230"
Cohesion: 0.07
Nodes (21): DTU, Get path to ground truth point cloud for a scene., Evaluate fused point cloud against DTU GT with ObsMask/Plane.          Args:, Load DTU depth validity masks.          Args:             mask_files: List of pa, Fuse per-view depths into a point cloud and save to PLY.          Args:, Back-project depth map into 3D world coordinates.          Args:             dep, Homography warping for multi-view consistency checking.          Args:, Compute geometric consistency between reference and source depths.          Args (+13 more)

### Community 231 - "Community 231"
Cohesion: 0.06
Nodes (9): Smoke tests for the JPC-Up panoptic pseudo-label module., test_anchor_supervision_uses_proposal_conditioned_queries_directly(), test_direct_positive_mask_losses_bypass_empty_consensus_masks(), test_jpc_up_forward_shapes_cpu(), test_jpc_up_losses_allow_missing_evidence(), test_jpc_up_losses_run_and_backprop(), test_jpc_up_proposal_conditioning_anchors_masks(), test_latent_consensus_targets_and_graph_loss() (+1 more)

### Community 232 - "Community 232"
Cohesion: 0.06
Nodes (31): PseudoLabelDataset, Loads each instance label and omits the samples that do not include at least a s, Loads each instance label and omits the samples that do not include at least a s, Loads each instance label and omits the samples that do not include at least a s, Loads the KITTI path for both the pseudo labels and images.          Args:, Add depth map and pseudo-onehot to output dict (in-place).          Args:, Add depth map and pseudo-onehot to output dict (in-place).          Args:, Loads the KITTI path for both the pseudo labels and images.          Args: (+23 more)

### Community 233 - "Community 233"
Cohesion: 0.08
Nodes (12): ConvInRelu, Ghiasi, ResidualBlock, UpsampleConvInRelu, StyleAugmentor, BasicConv2d, InceptionA, InceptionB (+4 more)

### Community 234 - "Community 234"
Cohesion: 0.07
Nodes (39): _build_openapi_index(), _build_search_index(), explore_hf_docs_handler(), _extract_all_endpoints(), _extract_all_tags(), _fetch_endpoint_docs(), _fetch_gradio_docs(), _fetch_openapi_spec() (+31 more)

### Community 235 - "Community 235"
Cohesion: 0.1
Nodes (21): _async_call(), _build_repo_url(), hf_repo_git_handler(), HfRepoGitTool, HF Repo Git Tool - Git-like operations on Hugging Face repositories  Operations:, List branches and tags., Create a pull request., List PRs and discussions. (+13 more)

### Community 236 - "Community 236"
Cohesion: 0.08
Nodes (30): all_gather(), all_reduce(), _check_env_variable(), _collect_env_vars(), enable(), _get_available_port(), get_global_rank(), get_global_size() (+22 more)

### Community 237 - "Community 237"
Cohesion: 0.08
Nodes (20): auto_detect(), BILOU, entities(), Entity, IOB1, IOB2, IOBES, IOE1 (+12 more)

### Community 238 - "Community 238"
Cohesion: 0.06
Nodes (39): compute_pq_from_accumulated(), depth_guided_instances(), evaluate_panoptic_single(), _get_extra_args(), main(), Get method-specific extra keyword arguments., SegFix-style: replace boundary pixel labels with nearest interior labels.      F, Dense CRF refinement: snap instance boundaries to color edges.      Uses pydense (+31 more)

### Community 239 - "Community 239"
Cohesion: 0.07
Nodes (16): aggo_merge_graph(), aggo_whole_batch(), get_edge_score(), hc_graph(), UnpoolInfo, agg_sample(), AggSample, box_to_grid() (+8 more)

### Community 240 - "Community 240"
Cohesion: 0.07
Nodes (18): FFN, loss(), multi_head_attention_forward(), MultiheadAttention, PositionEmbeddingLearned, Forward pass.          Args:             feats (list[torch.Tensor]): Multi-level, Generate training targets.          Args:             gt_bboxes_3d (:obj:`LiDARI, Generate training targets for a single sample.          Args:             gt_bbo (+10 more)

### Community 241 - "Community 241"
Cohesion: 0.11
Nodes (34): build_job_matrix(), check_job_progress(), check_vm_state(), create_vm(), delete_vm(), gcloud_ssh(), generate_vm_specs(), get_latest_checkpoint() (+26 more)

### Community 242 - "Community 242"
Cohesion: 0.07
Nodes (27): PseudoLabelDatasetLayered, Per-pixel panoptic layering of k=80 semantics + SF2SE3 instances (GT-free)., Detectron2 sample with no instances (mirrors parent's empty case)., Load one sample and build per-pixel layered Detectron2 targets., PseudoLabelDataset, apply_joint_augmentation(), boundary_distillation_loss(), boundary_embedding_instances() (+19 more)

### Community 243 - "Community 243"
Cohesion: 0.06
Nodes (21): create_adapter(), DepthAdapterAdaptFormer, DepthAdapterCrossAttn, DepthAdapterDeep, DepthAdapterFiLM, DepthAdapterWindowAttn, DepthAdapterX, DepthAdapterX2 (+13 more)

### Community 244 - "Community 244"
Cohesion: 0.09
Nodes (13): box_rigid_transform(), compute_confidence(), corner_align(), correct_heading(), correct_orientation(), density_guided_drift(), get_registration_angle(), hierarchical_occupancy_score() (+5 more)

### Community 245 - "Community 245"
Cohesion: 0.08
Nodes (26): Arguments, capture_graph(), decode(), DecodingCGCache, example_generation_pangu(), example_generation_pycodegpt(), GenerationMixin, InferenceParams (+18 more)

### Community 246 - "Community 246"
Cohesion: 0.07
Nodes (29): apply_augmentation(), boundary_preservation_loss(), center_offset_loss(), cross_view_consistency_loss(), depth_boundary_alignment_loss(), entropy_loss(), evaluate_panoptic(), feature_prototype_loss() (+21 more)

### Community 247 - "Community 247"
Cohesion: 0.06
Nodes (33): draw_annotation_badge(), draw_arrow(), draw_data_panel(), draw_feature_grid(), draw_flat_module_box(), draw_legend(), draw_panel_labels(), draw_patch_grid_overlay() (+25 more)

### Community 248 - "Community 248"
Cohesion: 0.09
Nodes (35): BaseImage, detect_model_format(), Image, main(), qvec2rotmat(), see: src/colmap/scene/reconstruction.cc         void Reconstruction::WriteCamera, see: src/colmap/scene/reconstruction.cc         void Reconstruction::WriteCamera, see: src/colmap/scene/reconstruction.cc         void Reconstruction::WriteCamera (+27 more)

### Community 249 - "Community 249"
Cohesion: 0.07
Nodes (24): _isArrayLike(), Print information about the annotation file.         :return:, Print information about the annotation file.         :return:, Get ann ids that satisfy given filter conditions. default skips that filter, Get ann ids that satisfy given filter conditions. default skips that filter, filtering parameters. default skips that filter.         :param catNms (str arra, filtering parameters. default skips that filter.         :param catNms (str arra, Get vid ids that satisfy given filter conditions.         :param vidIds (int arr (+16 more)

### Community 250 - "Community 250"
Cohesion: 0.08
Nodes (18): CBatchNorm1d, CBatchNorm1d_legacy, CResnetBlockConv1d, Decoder, DecoderCat, DecoderCatBN, DecoderCBatchNorm, Conditional batch normalization layer class.      Args:         c_dim (int): dim (+10 more)

### Community 251 - "Community 251"
Cohesion: 0.09
Nodes (34): depth_2_disp(), depth_2_pt3d(), disp_2_depth(), disp_2_mask_valid(), disp_2_pt3d(), intr2x3_to_3x3(), intr2x3_to_4x4(), oflow_2_mask_inside() (+26 more)

### Community 252 - "Community 252"
Cohesion: 0.11
Nodes (14): Asserts last token of every sentence in x is EOS, This verifies that with a given x, x_len, max_shuffle_distance, and         voca, Args:             append_eos: if True, each input sentence in the source tokens, The purpose of this is to test shuffling logic with word vocabs, Same result as word shuffle with eos except no EOS at end, Same result as word shuffle without eos except using BPE end token, Asserts that the last token of each sentence in x is not EOS, Same result as word dropout with eos except no EOS at end (+6 more)

### Community 253 - "Community 253"
Cohesion: 0.06
Nodes (12): FairseqOptimizer, LegacyFairseqOptimizer, Multiplies grads by a constant *c*., Performs a single optimization step., Clears the gradients of all optimized parameters., Broadcasts a global state dict to all ranks.         Useful for optimizers that, Return the current learning rate., Set the learning rate. (+4 more)

### Community 254 - "Community 254"
Cohesion: 0.08
Nodes (21): _dora_params(), ema_update(), MockAdapter, MockDINOv2, MockDINOv2Block, MockMultiheadAttention, MockSegmentTR, MockSelfAttention (+13 more)

### Community 255 - "Community 255"
Cohesion: 0.09
Nodes (22): CityscapesCandidateObjectnessDataset, ConvBlock, expand_box(), image_gradient(), InstanceObjectnessNet, load_cityscapes_image(), load_npz_candidates(), mask_iou_one() (+14 more)

### Community 256 - "Community 256"
Cohesion: 0.08
Nodes (23): add_batch_counter_hook_function(), add_batch_counter_variables_or_reset(), add_flops_counter_hook_function(), add_flops_counter_variable_or_reset(), add_flops_counting_methods(), add_flops_mask_variable_or_reset(), compute_average_flops_cost(), flops_to_string() (+15 more)

### Community 257 - "Community 257"
Cohesion: 0.08
Nodes (23): DeepLabV3PlusHead, build_ins_embed_branch(), __init__(), PanopticDeepLab, PanopticDeepLabInsEmbedHead, PanopticDeepLabSemSegHead, A semantic segmentation head described in :paper:`Panoptic-DeepLab`., Returns:             In training, returns (None, dict of losses)             In (+15 more)

### Community 258 - "Community 258"
Cohesion: 0.09
Nodes (27): create_mask(), decode_json(), get_specific_frame(), main(), save_decoded_images(), save_a_image(), save_a_pkl(), self_check_path_create() (+19 more)

### Community 259 - "Community 259"
Cohesion: 0.08
Nodes (24): __call__(), DINOViTS8, forward(), MLP, MultiHeadSelfAttention, PatchEmbedding, DINO ViT-S/8 backbone in PyTorch.  Ported from facebookresearch/dino. This modul, Feed-forward network in Transformer block.      Attributes:         dim: Input/o (+16 more)

### Community 260 - "Community 260"
Cohesion: 0.1
Nodes (18): EnsembleModel, EnsembleModelWithAlignment, forward(), forward_decoder(), forward_encoder(), generate(), Iterate over a batched dataset and yield individual translations.         Args:, Generate translations. Match the api of other fairseq generators.          Args: (+10 more)

### Community 261 - "Community 261"
Cohesion: 0.1
Nodes (34): _base_stem(), compute_depth_stats(), compute_distributions(), _extract_city(), list_cups_images(), main(), Merge adjacent same-class instances with high DINOv3 feature similarity.      Ar, First pass: compute per-class depth mean and std across all images.      Returns (+26 more)

### Community 262 - "Community 262"
Cohesion: 0.07
Nodes (25): BackprojectDepth, compute_depth_errors(), Conv3x3, ConvBlock, disp_to_depth(), get_smooth_loss(), get_translation_matrix(), Project3D (+17 more)

### Community 263 - "Community 263"
Cohesion: 0.09
Nodes (22): assign_instances_for_scan(), assign_instances_for_scan_with_gt(), compute_averages(), compute_metric_averages(), evaluate(), evaluate_matches(), Evaluator, get_options() (+14 more)

### Community 264 - "Community 264"
Cohesion: 0.1
Nodes (24): FakeGitHubClient, FakeIssueClient, FakeResponse, _load(), RateLimitResponse, test_append_published_issue_section_adds_local_link(), test_async_main_fails_early_when_issue_publish_token_missing(), test_call_json_llm_retries_after_invalid_json() (+16 more)

### Community 265 - "Community 265"
Cohesion: 0.08
Nodes (22): __call__(), DINOv3ViTB, forward(), MLP, MultiHeadSelfAttention, PatchEmbedding, DINOv3 ViT-B/16 backbone in PyTorch.  Loads from HuggingFace: facebook/dinov3-vi, Feed-forward network in Transformer block.      Attributes:         dim: Input/o (+14 more)

### Community 266 - "Community 266"
Cohesion: 0.13
Nodes (30): all_extracted_files(), call(), check_need_manual_downalod(), check_wmt_test_bleu(), concat_files(), concat_into_splits(), convert2czeng17(), convert_file_if_needed() (+22 more)

### Community 267 - "Community 267"
Cohesion: 0.07
Nodes (24): BackprojectDepth, compute_depth_errors(), Conv3x3, ConvBlock, disp_to_depth(), get_smooth_loss(), get_translation_matrix(), Project3D (+16 more)

### Community 268 - "Community 268"
Cohesion: 0.08
Nodes (22): _assignment_rule(), get_ground_truth(), _paste_mask_lists_in_image(), permute_all_cls_and_box_to_N_HWA_K_and_concat(), _postprocess(), Paste a list of masks that are of various resolutions (e.g., 28 x 28) into an im, Post-process the output boxes for TensorMask.     The input images are often res, For a set of image sizes and feature maps, computes a set of anchors for TensorM (+14 more)

### Community 269 - "Community 269"
Cohesion: 0.12
Nodes (12): abs_yaw_diff(), angle_diff(), BaseODMetrics, calc_ap(), get_conf_prec_rec(), main(), map_scores_from_neg_infs_to_actual_min_score(), ObjectDetectionMetrics (+4 more)

### Community 270 - "Community 270"
Cohesion: 0.09
Nodes (15): FlowLoss, :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]         :p, Loss function defined over sequence of flow predictions, Loss function defined over sequence of flow predictions, unFlowLoss, unFlowLoss_Raft, abs_robust_loss(), get_edge_weights() (+7 more)

### Community 271 - "Community 271"
Cohesion: 0.09
Nodes (24): _assistant_call(), _Fn, _Msg, Tests for the doom-loop detector — repeated/cycling tool call patterns., Three calls with reordered keys should produce identical signatures., Regression for the bug: same call with shuffled keys must collapse., Cycle [research, read, research, read, ...] survives key reordering., Same tool, three different arg values — not a loop. (+16 more)

### Community 272 - "Community 272"
Cohesion: 0.09
Nodes (33): cancel_sandbox_preload(), _cleanup_user_orphan_sandboxes(), _clear_persisted_sandbox(), _create_sandbox_locked(), _ensure_sandbox(), get_active_or_preloaded_sandbox(), _get_sandbox_create_lock(), get_sandbox_tools() (+25 more)

### Community 273 - "Community 273"
Cohesion: 0.08
Nodes (21): DownSamplingShuffle(), Flatten, LocalNet, MidNet2, MidNet4, Defines a double convolution          :param x_in: input convolutional feature, Initialisation function          :param in_channels:  number of input channels, Network with dilation rate 2          :param x_in: input convolutional feature (+13 more)

### Community 274 - "Community 274"
Cohesion: 0.07
Nodes (16): EpochListening, FairseqDataset, FairseqIterableDataset, Given an ordered set of indices, return batches according to         *max_tokens, Mixin for receiving updates whenever the epoch increments., Filter a list of sample indices. Remove those that are longer than         speci, For datasets that need to be read sequentially, usually because the data is, Will receive the updated epoch number at the beginning of the epoch. (+8 more)

### Community 275 - "Community 275"
Cohesion: 0.1
Nodes (16): constant_init(), ConvModule, DPTHead, FeatureFusionBlock, Interpolate, kaiming_init(), norm(), PreActResidualConvUnit (+8 more)

### Community 276 - "Community 276"
Cohesion: 0.09
Nodes (31): check_num_fg_corners(), densecrf(), detect_box(), DINOv3Feat, get_affinity_matrix(), get_cityscapes_images(), get_masked_affinity_matrix(), get_salient_areas() (+23 more)

### Community 277 - "Community 277"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 278 - "Community 278"
Cohesion: 0.09
Nodes (17): Convert all models to training mode, Convert all models to testing/evaluation mode, Run the entire training pipeline, Run a single epoch of training and validation, Pass a minibatch through the network and generate images and losses, Predict poses between input frames for monocular sequences., Validate the model on a single minibatch, Generate the warped (reprojected) color images for a minibatch.         Generate (+9 more)

### Community 279 - "Community 279"
Cohesion: 0.12
Nodes (32): apply_prior_gates(), boundary_mean_in_masks(), class_aware_nms(), first_npz_array(), fragment_iou(), fragment_touch_fraction(), gate_one(), generate_tiny_fragments() (+24 more)

### Community 280 - "Community 280"
Cohesion: 0.1
Nodes (12): build(), DummyDataModule, _get_padding(), _get_webdataset(), _make_squares_dataset(), _pad(), Check that we have appropriate settings for the distributed setting.          We, Estimate upper bound on the number of samples per data worker.          It is on (+4 more)

### Community 281 - "Community 281"
Cohesion: 0.1
Nodes (24): benchmark_launcher(), eval_log_regression(), eval_log_regression_with_model(), EvalConfig, evaluate_logreg_model(), evaluate_model(), Evaluator, FewShotConfig (+16 more)

### Community 282 - "Community 282"
Cohesion: 0.1
Nodes (13): Attention, Block, drop_path(), DropPath, ibot_vit_base_16(), ibot_vit_small_16(), Mlp, _no_grad_trunc_normal_() (+5 more)

### Community 283 - "Community 283"
Cohesion: 0.13
Nodes (32): _add_to_collection(), _append_section(), artifact_collection_title(), _artifact_key(), augment_repo_card_content(), _body_without_metadata(), build_hub_artifact_sitecustomize(), _collection_session_id_fragment() (+24 more)

### Community 284 - "Community 284"
Cohesion: 0.08
Nodes (13): _make_batch(), Tests for copy-paste augmentation module., Old-style call (no scale_range, no source_batch) still works., scale_range=(1.0, 1.0) produces same result as default., Create a synthetic batch for testing., scale_range=(0.5, 1.5) produces different result than no scaling., source_batch extracts instances from source, not target., If required fields missing, return batch unchanged. (+5 more)

### Community 285 - "Community 285"
Cohesion: 0.08
Nodes (22): annealing_cos(), annealing_linear(), constant(), cosine_decay_with_warmup(), exp_decay_with_warmup(), linear_warmup(), Compute the learning rate of each parameter group., Compute the learning rate of each parameter group. (+14 more)

### Community 286 - "Community 286"
Cohesion: 0.09
Nodes (24): COCOFeatureDataset, entropy_regularization(), evaluate_matryoshka(), load_coco_panoptic_gt(), main(), matryoshka_consistency_loss(), MatryoshkaClusterer, MatryoshkaHead (+16 more)

### Community 287 - "Community 287"
Cohesion: 0.09
Nodes (24): augment_feature_grid(), cluster_entropy_loss(), ClusterLookup, COCOFeatureDataset, _correlation_helper(), get_gt_semantic(), hungarian_match_and_miou(), load_coco_panoptic_gt() (+16 more)

### Community 288 - "Community 288"
Cohesion: 0.07
Nodes (15): Attack, r"""         Set the return type of adversarial images: `int` or `float`., r"""         Set training mode during attack process.          Arguments:, r"""         Initializes internal attack state.          Arguments:, r"""         Save adversarial images as torch.tensor from given torch.utils.data, r"""         Function for changing the attack mode.         Return input labels., r"""         Function for changing the attack mode.         Return least likely, r"""         Function for changing the return type.         Return images as int (+7 more)

### Community 289 - "Community 289"
Cohesion: 0.09
Nodes (10): Diffusion_cond, Diffusion_net, Diffusion_net_sum, Diffusion_war, EFEMSDF, PointNet2_wpos, PoseNet, Create sinusoidal timestep embeddings.     :param timesteps: a 1-D Tensor of N i (+2 more)

### Community 290 - "Community 290"
Cohesion: 0.1
Nodes (27): binary_xloss(), flatten_binary_scores(), flatten_probas(), hinge_jaccard_loss(), iou(), iou_binary(), jaccard_loss(), lovasz_grad() (+19 more)

### Community 291 - "Community 291"
Cohesion: 0.11
Nodes (20): batch_box_data_for_batched_smoothing(), batched_displacement_from_pos(), BatchedBikeModel, BatchedSmoothTrack, car_dynamics(), cauchy(), forward_compiled(), get_nth_element_from_each_dict_entry() (+12 more)

### Community 292 - "Community 292"
Cohesion: 0.1
Nodes (15): Attention, Block, ConvEmbed, drop_path(), DropPath, MLP, msn_vit_base_16(), msn_vit_small_16() (+7 more)

### Community 293 - "Community 293"
Cohesion: 0.08
Nodes (9): get_teacher_output(), MultiDistillationMetaArch, Multidistillation version of SSLMetaArchCompilableGram:     - baked-in scales fo, get_teacher_output(), interpolate_pos_embed(), Modified version of SSLMetaArchCompilable including gram loss:     - Gram loss i, This is an operation that takes a tensor from the default process group, gathers, SSLMetaArch (+1 more)

### Community 294 - "Community 294"
Cohesion: 0.08
Nodes (13): backward(), chunk_bwd_dhu_fn(), chunk_bwd_dqkw_fn(), chunk_fwd_h_fn(), chunk_fwd_o_fn(), ChunkGatedDeltaRuleFunction, forward(), fwd_prepare_du() (+5 more)

### Community 295 - "Community 295"
Cohesion: 0.08
Nodes (18): DepthAnything3Net, NestedDepthAnything3Net, Forward pass through the network.          Args:             x: Input images (B,, Process mono sky estimation., Process ray pose estimation if ray pose decoder is available., Process features through the depth prediction head., Process camera pose estimation if camera decoder is available., Process 3DGS parameters estimation if 3DGS head is available. (+10 more)

### Community 296 - "Community 296"
Cohesion: 0.1
Nodes (29): _align_preset_class_ids(), _apply_direct_class_filters(), _class_keep_probs_from_counts(), _direct_class_counts_from_cache(), _direct_class_counts_from_target_bank(), _direct_positive_targets(), _first_npz_array(), _hard_semantic_target() (+21 more)

### Community 297 - "Community 297"
Cohesion: 0.1
Nodes (19): BasicBlock, Bottleneck, SEBasicBlock, SEBasicBlockIN, SEBasicBlockLN, SEBasicBlockSN, SEBottleneck, SEBottleneckIN (+11 more)

### Community 298 - "Community 298"
Cohesion: 0.12
Nodes (8): createEmpty(), extract_box_motion_transform_without_sensor_odometry(), extract_motion_in_pred_box_coordinates(), from_list_of_npy_shapes(), from_list_of_shapes(), get_points_in_box_bool_mask(), Shape, soft_align_box_flip_orientation_with_motion_trafo()

### Community 299 - "Community 299"
Cohesion: 0.08
Nodes (19): Calibration, get_calib_from_file(), :param corners3d: (N, 8, 3) corners in rect coordinate         :return: boxes: (, :param pts_lidar: (N, 3)         :return pts_rect: (N, 3), :param pts_lidar: (N, 3)         :return pts_rect: (N, 3), :param pts_rect: (N, 3)         :return pts_img: (N, 2), :param pts_rect: (N, 3)         :return pts_img: (N, 2), :param pts_lidar: (N, 3)         :return pts_img: (N, 2) (+11 more)

### Community 300 - "Community 300"
Cohesion: 0.09
Nodes (13): build_fmow_dataset(), build_transform(), CustomDatasetFromImages, EuroSat, Creates Dataset for regular RGB image classification (usually used for fMoW-RGB, Normalization for Sentinel-2 imagery, inspired from     https://github.com/Servi, Creates dataset for multi-spectral single image classification.         Usually, Gets image (x,y) pair given index in dataset.         :param idx: Index of (imag (+5 more)

### Community 301 - "Community 301"
Cohesion: 0.09
Nodes (20): __call__(), CascadeStage, InstanceHead, MaskHead, Cascade Mask R-CNN -- Class-Agnostic Detector (CAD) for CutS3D.  Faithful implem, Predict class scores and box deltas.          Args:             pooled_features:, Mask prediction head.      Predicts per-pixel binary masks from RoI features., Mask prediction head.      Predicts per-pixel binary masks from RoI features. (+12 more)

### Community 302 - "Community 302"
Cohesion: 0.12
Nodes (11): Decoder, DecoderLayer, Encoder, EncoderLayer, get_attn_decoder_mask(), get_attn_pad_mask(), IQARegression, MultiHeadAttention (+3 more)

### Community 303 - "Community 303"
Cohesion: 0.07
Nodes (18): ChatMem, Basically merge the extracted, Based on mem summary info to update basic messages, Multi round conversation, prompt-driven implementation, format the memory system prompt, Long term mem db operation, Extract memory feats from message lists         :param messages: contexts, based on given features to merge and update memory         :param messages: cont (+10 more)

### Community 304 - "Community 304"
Cohesion: 0.08
Nodes (15): NoisingDataset, Generate a noisy version of a sentence, without changing words themselves., # TODO: speed up the following loop, Shuffle words by no more than k positions., Implements the default configuration for noising in UnsupervisedMT     (github.c, Wrap a :class:`~torch.utils.data.Dataset` and apply noise to the         samples, Returns a single noisy sample. Multiple samples are fed to the collater, The length of the noising dataset is the length of src. (+7 more)

### Community 305 - "Community 305"
Cohesion: 0.07
Nodes (22): Unit tests for Adaptive Projection Bridge (PyTorch).  Tests cover:     - Forward, Inverse projection should output correct dimension., Inverse projection should output correct dimension., Inverse projection for semantic should restore dim 90., Inverse projection for semantic should restore dim 90., Test round-trip reconstruction quality., Reconstruction error should be bounded after training init., Reconstruction error should be bounded after training init. (+14 more)

### Community 306 - "Community 306"
Cohesion: 0.1
Nodes (25): _cleanup_ddp(), evaluate_changed_pct(), evaluate_miou(), evaluate_only(), evaluate_panoptic(), generate_labels(), _get_device(), _is_main() (+17 more)

### Community 307 - "Community 307"
Cohesion: 0.1
Nodes (30): classify_stuff_things(), decode_coconut_panoptic(), depth_guided_instances(), evaluate_coconut(), extract_dinov2_features(), generate_depth_maps(), generate_instances(), generate_panoptic() (+22 more)

### Community 308 - "Community 308"
Cohesion: 0.09
Nodes (24): build_cause_args(), CauseDepthDataset, depth_correlation_loss(), ema_update(), grid_sample(), main(), norm(), patch_cluster_for_device() (+16 more)

### Community 309 - "Community 309"
Cohesion: 0.1
Nodes (30): classify_person_failures(), compute_coplanar_separation_rate(), compute_depth_edges(), compute_edge_alignment(), compute_gt_instance_boundaries(), compute_gt_thing_boundaries(), compute_pq(), depth_guided_instances() (+22 more)

### Community 310 - "Community 310"
Cohesion: 0.06
Nodes (16): EventHandlers, Perform reconstruction using the already-created target_dir/images.          Arg, Reload saved predictions from npz, create (or reuse) the GLB for new parameters,, Handle file uploads and update gallery.          Args:             input_video:, Handles all event callbacks and user interactions for the Gradio app., Load a scene from examples directory.          Args:             scene_name: Nam, Initialize the event handlers., Navigate depth view.          Args:             processed_data: Processed data d (+8 more)

### Community 311 - "Community 311"
Cohesion: 0.14
Nodes (29): add_candidate(), adjacency(), bincount_mean(), build_views(), cache_city_stem(), Candidate, connected_patch_masks(), feature_hwc() (+21 more)

### Community 312 - "Community 312"
Cohesion: 0.09
Nodes (30): adaptive_self_training_loss(), _bfs_path_max(), compute_sgm_losses(), compute_sp_means(), depth_aware_edge_weights(), hard_loss(), _mst_adjacency(), mst_soft_labels() (+22 more)

### Community 313 - "Community 313"
Cohesion: 0.12
Nodes (29): git_commit(), make_mcg_batch(), _matlab_bin(), matlab_quote(), _mcg_octave_mex_count(), _octave_bin(), print_status(), rama_binary() (+21 more)

### Community 314 - "Community 314"
Cohesion: 0.09
Nodes (13): BasicBlock, Args:             batch_dict:                 batch_size: int                 vf, Args:             batch_dict:                 batch_size: int                 vf, Args:             batch_dict:                 batch_size: int                 vf, Args:             batch_dict:                 batch_size: int                 vf, SparseBasicBlock, VoxelBackBone8x, VoxelResBackBone8x (+5 more)

### Community 315 - "Community 315"
Cohesion: 0.08
Nodes (23): build_self_training_coco(), confidence_weighted_loss(), _load_required_masks(), _prediction_index(), PseudoLabelGenerator, Self-Training with Confidence-Weighted Pseudo-Labels.  Phase D of training: uses, Weight loss by pseudo-label confidence.      L_weighted = sum(w_i * L_i) / sum(w, Weight loss by pseudo-label confidence.      L_weighted = Σ w_i · L_i / Σ w_i (+15 more)

### Community 316 - "Community 316"
Cohesion: 0.1
Nodes (13): EmbeddingDatasetWriter, EmbeddingWriterConfig, H5Writer, input_fnames(), input_path(), output_path(), Prediction, PretrainedWav2VecModel (+5 more)

### Community 317 - "Community 317"
Cohesion: 0.09
Nodes (13): COCOeval, Params, Run per image evaluation on given images and store results (a list of dict) in s, Run per image evaluation on given images and store results (a list of dict) in s, perform evaluation for single category and image         :return: dict (single i, perform evaluation for single category and image         :return: dict (single i, Accumulate per image evaluation results and store the result in self.eval, Accumulate per image evaluation results and store the result in self.eval (+5 more)

### Community 318 - "Community 318"
Cohesion: 0.11
Nodes (11): ConvFFN, deform_inputs(), DINOv3_Adapter, drop_path(), DropPath, DWConv, Extractor, get_reference_points() (+3 more)

### Community 319 - "Community 319"
Cohesion: 0.11
Nodes (11): fmt_k_line(), InputProcessor, make_K(), results: List[Tuple[torch.Tensor, Tuple[H, W], Optional[np.ndarray], Optional[np, Center-crop all tensors to the smallest H, W; adjust intrinsics' cx, cy accordin, Floor each dimension to the nearest multiple of PATCH_SIZE via center crop., Prepares a batch of images for model inference.     This processor converts a li, Round each dimension to nearest multiple of PATCH_SIZE via small resize. (+3 more)

### Community 320 - "Community 320"
Cohesion: 0.11
Nodes (11): BatchNorm1d, BatchNorm2d, _BNBase, Conv1d, Conv2d, _ConvBase, FC, get_norm_layer() (+3 more)

### Community 321 - "Community 321"
Cohesion: 0.13
Nodes (25): compute_ephe_score(), count_neighbors(), display_args(), eprint(), get_relative_pose(), in_hull(), main(), :param p: (N, K) test points     :param hull: (M, K) M corners of a box     :ret (+17 more)

### Community 322 - "Community 322"
Cohesion: 0.12
Nodes (19): _async_call(), _build_repo_url(), _content_to_bytes(), private_hf_repo_handler(), PrivateHfRepoTool, Private HF Repos Tool - Manage private Hugging Face repositories  PRIMARY USE: S, Show usage instructions when tool is called with no arguments., Show help for a specific operation. (+11 more)

### Community 323 - "Community 323"
Cohesion: 0.09
Nodes (14): LocalityAwareVectorIndex, Block of vectors with locality          Stores vectors for a specific tag/contex, Consolidate temporary embeddings into tag blocks                  Groups by tags, Search for similar embeddings                  Args:             query_embedding, Search within this block                  Args:             query: Query embeddi, Remove embedding from index, 导出所有 tag blocks（用于 consolidation）                  Returns:             tag -> V, 导入新的 tag blocks（consolidation 后）                  Args:             new_tag_bloc (+6 more)

### Community 324 - "Community 324"
Cohesion: 0.11
Nodes (28): build_mask_id_mapping(), build_tfrecord_pipeline(), download_dino_weights(), evaluate(), extract_ncut_masks_single(), extract_phase(), generate_pseudo_labels_with_cad(), interpolate_pos_embed() (+20 more)

### Community 325 - "Community 325"
Cohesion: 0.1
Nodes (15): backward(), chunk_bwd_dqkwg(), chunk_bwd_dv_local(), chunk_fwd_o(), chunk_gated_delta_rule(), chunk_gated_delta_rule_bwd(), chunk_gated_delta_rule_bwd_dhu(), chunk_gated_delta_rule_fwd() (+7 more)

### Community 326 - "Community 326"
Cohesion: 0.12
Nodes (23): CandidateBankTargetDataset, choose_target_bank(), compute_instance_losses(), ddp_barrier(), dice_loss(), greedy_match(), init_distributed(), load_cluster_lut() (+15 more)

### Community 327 - "Community 327"
Cohesion: 0.11
Nodes (28): case_to_md(), _compute_boundary_pct(), _count_total_cc(), diagnose(), FailureCase, ImageStats, main(), pick_boundary_noise() (+20 more)

### Community 328 - "Community 328"
Cohesion: 0.1
Nodes (23): calc_sigma_optimum(), calc_sigma_rel_implicit(), calc_sigma_rel_implicit_proposals(), calc_sigma_rel_optimum(), draw_heatmap_gaussian(), ellip_gaussian2D(), em(), estimate_std() (+15 more)

### Community 329 - "Community 329"
Cohesion: 0.08
Nodes (17): fill_invalid(), fill_invalid_slow(), flow_read(), flow_to_rgb(), flow_write(), merge_flows(), Basic functions to handle optical flow., Write optical flow to file.      This is just a wrapper for flowpy (for .flo and (+9 more)

### Community 330 - "Community 330"
Cohesion: 0.11
Nodes (27): crf_refine(), discover_images(), estimate_depth_batch(), extract_dino_features_batch(), extract_pseudo_masks_gpu(), gpu_build_knn_graph(), load_depth_model(), load_dino_backbone() (+19 more)

### Community 331 - "Community 331"
Cohesion: 0.09
Nodes (25): backward(), _chunk_scan_chunk_state_bwd_dx(), ensure_stride(), forward(), mamba_chunk_scan(), mamba_chunk_scan_combined(), _mamba_chunk_scan_combined_bwd(), _mamba_chunk_scan_combined_fwd() (+17 more)

### Community 332 - "Community 332"
Cohesion: 0.11
Nodes (19): batch_dense_targets(), _connected_components(), ConvBlock, dense_instance_loss(), DenseInstanceUNet, DenseLossWeights, dice_loss_from_logits(), DownBlock (+11 more)

### Community 333 - "Community 333"
Cohesion: 0.07
Nodes (10): Unit tests for Depth-Conditioned Slot Attention Decoder., Decoder masks should sum to 1 over slots., Verify model can overfit on a single sample., FiLM should initialize to approximate identity (gamma=1, beta=0)., Attention should sum to 1 over slots for each patch., TestDepthFiLM, TestDepthSlotDecoder, TestLossFunctions (+2 more)

### Community 334 - "Community 334"
Cohesion: 0.09
Nodes (16): FusionConcat, FusionXAttn, Depth-aware SGM Adapter.  Cross-attention fusion of frozen DINOv2 features with, Decoder: fused patch features -> per-thing-class foreground probability., Args:             fbar: ``(B, N, d_fusion)`` with ``N == patch_h * patch_w``., Fusion + SGM head wrapper., Forward pass.          Args:             f_dino: ``(B, N, d_dino)``., Hyper-parameters for the SGM adapter. (+8 more)

### Community 335 - "Community 335"
Cohesion: 0.16
Nodes (11): __init__(), _LegacySubClass, _LegacySubClassNotCfg, _NewSubClassNewInit, _test_func(), _TestClassA, _TestClassB, _TestClassC (+3 more)

### Community 336 - "Community 336"
Cohesion: 0.17
Nodes (19): MyResNet14, ResNet101, ResNet14, ResNet18, ResNet34, ResNet50, ResNetBase, STResNet101 (+11 more)

### Community 337 - "Community 337"
Cohesion: 0.1
Nodes (17): filter_labels(), is_valid_cluster(), above_plane(), cart2hom(), closeness_rectangle(), distance_to_plane(), estimate_plane(), get_lowest_point_rect() (+9 more)

### Community 338 - "Community 338"
Cohesion: 0.08
Nodes (26): boxes_bev_iou_cpu(), boxes_dis(), boxes_iou3d_gpu(), boxes_iou_bev(), check_numpy_to_torch(), nms_gpu(), nms_normal_gpu(), 3D IoU Calculation and Rotated NMS Written by Shaoshuai Shi All Rights Reserved (+18 more)

### Community 339 - "Community 339"
Cohesion: 0.12
Nodes (18): _async_call(), _build_repo_url(), _format_size(), hf_repo_files_handler(), HfRepoFilesTool, HF Repo Files Tool - File operations on Hugging Face repositories  Operations: l, Read file content from a repository., Upload content to a repository. (+10 more)

### Community 340 - "Community 340"
Cohesion: 0.09
Nodes (13): Represents a time range, Query most recent episodes                  Args:             user_id: User ID, Check if timestamp is within this interval, Remove episode from temporal index, Merge overlapping time intervals, Check if this interval overlaps with another, Node in temporal index tree, L1: Temporal Index          Fast time-based filtering using sorted timeline per (+5 more)

### Community 341 - "Community 341"
Cohesion: 0.13
Nodes (11): LM, FairseqLM, Normalize tokens by handling CTC blank, ASG replabels, etc., Evaluate language model based on the current lm state and new word         Param, Evaluate eos for language model based on the current lm state          Returns:, Generate a batch of inferences., Run encoder and normalize emissions, W2lDecoder (+3 more)

### Community 342 - "Community 342"
Cohesion: 0.12
Nodes (23): _build_parser(), _dav3_inference_batch(), DepthAdapterDataset, _extract_depth(), load_da2_model(), load_dav3_model(), load_depthpro_model(), main() (+15 more)

### Community 343 - "Community 343"
Cohesion: 0.09
Nodes (16): MonoDataset, Returns a single training item from the dataset as a dictionary.          Values, Returns a single training item from the dataset as a dictionary.          Values, Resize colour images to the required scales and augment if required          We, Resize colour images to the required scales and augment if required          We, This function should be called in each training iteration., This function should be called in each training iteration., Returns a single training item from the dataset as a dictionary.          Values (+8 more)

### Community 344 - "Community 344"
Cohesion: 0.12
Nodes (18): colorize_value(), Colors, load_metrics_from_dir(), main(), MetricsPrinter, Initialize the printer.          Args:             use_color: Whether to use ANS, Print evaluation metrics in a beautiful tabular format.          Args:, Print comparison table for multiple evaluation runs.          Args: (+10 more)

### Community 345 - "Community 345"
Cohesion: 0.1
Nodes (10): Args:             batched_inputs: a list, batched outputs of :class:`DatasetMapp, Args:             batched_inputs: a list, batched outputs of :class:`DatasetMapp, Main class for mask classification semantic segmentation architectures., # TODO: make it configurable, Main class for mask classification semantic segmentation architectures., Args:             batched_inputs: a list, batched outputs of :class:`DatasetMapp, # TODO: make it configurable, Main class for mask classification semantic segmentation architectures. (+2 more)

### Community 346 - "Community 346"
Cohesion: 0.12
Nodes (21): calculate_zy_rotation_for_arrow(), create_arrow(), create_mesh(), faces(), geodists(), get_arrow(), load_mesh_auxiliary_data(), load_mesh_data() (+13 more)

### Community 347 - "Community 347"
Cohesion: 0.12
Nodes (9): # TODO: add test to dump scripting, # TODO: this test requires manifold access, see: T88318502, testCascadeRCNN(), testMaskRCNNC4(), testMaskRCNNFPN_batched(), testRetinaNet(), TestScripting, TestTorchscriptUtils (+1 more)

### Community 348 - "Community 348"
Cohesion: 0.11
Nodes (4): test_box_convert_cuda_tensor(), TestBoxes, TestBoxIOU, TestBoxMode

### Community 349 - "Community 349"
Cohesion: 0.1
Nodes (11): cat(), Instances, Returns:             dict: a dict which maps names (str) to data of the fields, Returns:             Instances: all fields are called with a `to(device)`, if th, Args:             item: an index-like object and will be used to index all the f, Args:             image_size (height, width): the spatial size of the image., Set the field named `name` to `value`.         The length of `value` must be the, Returns:             bool: whether the field called `name` exists. (+3 more)

### Community 350 - "Community 350"
Cohesion: 0.1
Nodes (13): AlignedAnchor3DRangeGenerator, AlignedAnchor3DRangeGeneratorPerCls, Anchor3DRangeGenerator, 3D Anchor Generator by range.      This anchor generator generates anchors by th, Generate grid anchors of a single level feature map.          This function is u, Generate anchors in a single range.          Args:             feature_size (lis, Aligned 3D Anchor Generator by range.      This anchor generator uses a differen, Generate anchors in a single range.          Args:             feature_size (lis (+5 more)

### Community 351 - "Community 351"
Cohesion: 0.14
Nodes (13): RLAlgorithm, compute_ari(), compute_mIoU(), get_segmentation(), History, _overlay_segmentation_on_image(), Take the argmax+1 of mask prob as the seg id. If the max prob is below     the t, Converted from the JAX version:     https://github.com/google-research/slot-atte (+5 more)

### Community 352 - "Community 352"
Cohesion: 0.12
Nodes (10): FlowAugmentor, ImageAugmentor, Photometric augmentation, Photometric augmentation, Photometric augmentation, Photometric augmentation, Occlusion augmentation, Occlusion augmentation (+2 more)

### Community 353 - "Community 353"
Cohesion: 0.12
Nodes (2): FlowDataModule, Parse the input string into the selected dataset and their multipliers and param

### Community 354 - "Community 354"
Cohesion: 0.13
Nodes (7): BasicBlock, Bottleneck, conv3x3(), get_cls_net(), HighResolutionModule, HighResolutionNet, 3x3 convolution with padding

### Community 355 - "Community 355"
Cohesion: 0.09
Nodes (12): 统一搜索接口                  Args:             user_id: 用户ID             query_embedd, 统一索引 - 三层过滤架构          查询流程:     Query -> L1 (Temporal) -> L2 (Tag DAG) -> L3 (V, 在候选集内进行向量搜索                  策略：         1. 从候选集中提取所有相关的 tags         2. 在这些 tag, 清空用户的所有索引数据                  Args:             user_id: 用户ID, O(1) 获取 memory content                  Args:             memory_id: Memory ID, 批量获取 memory 数据（O(n) 但常数极小）                  性能目标: < 1ms for 100 items, 获取索引统计信息                  Args:             user_id: 可选的用户ID，指定则返回该用户的统计, 离线 consolidation         在系统空闲时执行，优化物理布局 (+4 more)

### Community 356 - "Community 356"
Cohesion: 0.1
Nodes (15): InstanceEmbeddingHead, MBPSv2Model, MBPS v2 Model: DINOv3 + Mamba Bridge Panoptic Segmentation (PyTorch).  Simplifie, MBPS v2 Panoptic Segmentation Model.      Attributes:         num_classes: Numbe, Initialize all sub-modules., Forward pass through MBPS v2.          Args:             image: Input images (B,, Forward pass through MBPS v2.          Args:             image: Input images (B,, Flatten spatial depth map to token-level depth.          Args:             depth (+7 more)

### Community 357 - "Community 357"
Cohesion: 0.13
Nodes (12): LoopDetector, main(), Create image transformation function, Get paths of all image files in directory, Extract image feature descriptors, Apply Non-Maximum Suppression (NMS) filtering to loop pairs, Save loop detection results to file, Run complete loop detection pipeline (+4 more)

### Community 358 - "Community 358"
Cohesion: 0.11
Nodes (13): ETH3D, Parse COLMAP-style cameras.txt file.          Returns:             Dict mapping, Parse COLMAP-style images.txt file.          Returns:             Dict mapping i, Check if image should be filtered out based on known problematic views., Collect per-view image paths, intrinsics/extrinsics for a scene.          Args:, Evaluate fused point cloud against ETH3D ground truth mesh.          Args:, Load saved GT meta (extrinsics, intrinsics, image_files) for fusion.          Th, Fuse per-view depths into a point cloud using TSDF fusion.          Pipeline: (+5 more)

### Community 359 - "Community 359"
Cohesion: 0.14
Nodes (22): boundary_smoothness_loss(), cosine_code_loss(), crop_teacher_cosine_loss(), DCFACodePairDataset, downsample_consistency_loss(), iter_epoch(), main(), make_model() (+14 more)

### Community 360 - "Community 360"
Cohesion: 0.08
Nodes (18): CoordinateEncoderStateInit, FixedLearnedInit, GaussianStateInit, ParamStateInit, RandomInit, # NOTE: 0th entry inputs_oh[..., 0] will typically correspond to background., # NOTE: 0th entry inputs_oh[..., 0] will typically correspond to background., State init that encodes bounding box coordinates as conditional input.    Attrib (+10 more)

### Community 361 - "Community 361"
Cohesion: 0.1
Nodes (7): _a_slow_func(), _MyData, test_using_lazy_path(), TestAspectRatioGrouping, TestDataLoader, TestDatasetFromList, TestMapDataset

### Community 362 - "Community 362"
Cohesion: 0.1
Nodes (12): DecoderBatchNorm, DecoderCBatchNorm, DecoderCBatchNorm2, DecoderCBatchNormNoResnet, DecoderInner, DecoderInnerUNet, Decoder class.      It does not perform any form of normalization.      Args:, Decoder with conditional batch normalization (CBN) class.      Args:         dim (+4 more)

### Community 363 - "Community 363"
Cohesion: 0.12
Nodes (24): apply_affine_transformation(), compute_segment_sign(), discretize(), discretize_lane(), _find_index(), get_curvature_at_distance_along_lane(), _get_lie_algebra(), get_transformation_at_step() (+16 more)

### Community 364 - "Community 364"
Cohesion: 0.1
Nodes (15): LearntConditioning, RandomConditioning, RandomConditioningWithQMCSampling, Implementation of conditioning approaches for slots., Random conditioning with potentially learnt mean and stddev., Generate conditioning vectors for `batch_size` instances.          Args:, Random conditioning with learnt mean and stddev for each slot.      Removes perm, Initialize SlotwiseLearntConditioning.          Args:             object_dim: Di (+7 more)

### Community 365 - "Community 365"
Cohesion: 0.1
Nodes (13): KMeansGrouping, Implementations of perceptual grouping algorithms.  We denote methods that group, Implementation of SlotAttention for perceptual grouping., Initialize Slot Attention Grouping.          Args:             feature_dim: Dime, Apply slot attention based perceptual grouping.          Args:             featu, Implementation of SlotAttention.      Based on the slot attention implementation, Perceptual grouping based on a stick-breaking process.      The idea is to pick, Initialize stick-breaking-based perceptual grouping.          Args: (+5 more)

### Community 366 - "Community 366"
Cohesion: 0.11
Nodes (14): hard_mish_jit(), hard_sigmoid_jit(), hard_swish_jit(), HardMishJit, HardSigmoidJit, HardSwishJit, mish_jit(), MishJit (+6 more)

### Community 367 - "Community 367"
Cohesion: 0.1
Nodes (13): clusters_agglomerative(), ConsensusSFlowDRPCs, DRPCs, Create new Dynamic Rigid Point Clouds (DRPCs) from parameters.          Paramete, Create new Dynamic Rigid Point Clouds (DRPCs) from parameters.          Paramete, Specify pt3d and pt3d_assign for inliers and make sure that they are connected., Specify pt3d and pt3d_assign for inliers and make sure that they are connected., Select subset with ids to reduce number of drpcs.          Parameters         -- (+5 more)

### Community 368 - "Community 368"
Cohesion: 0.25
Nodes (24): _ev(), _load(), Unit tests for the KPI rollup math.  We exercise the pure functions (``_session_, Load ``scripts/build_kpis.py`` without treating ``scripts`` as a package., Sessions that never called a tool would otherwise crush the median., The aggregate row keeps pro_cta_clicks + hf_jobs_blocked columns     even if the, _session(), test_aggregate_day_cache_hit_and_users() (+16 more)

### Community 369 - "Community 369"
Cohesion: 0.08
Nodes (17): Unit tests for panoptic merging module (PyTorch).  Tests cover:     - No pixel b, Stuff-class pixels should have instance_id=0., Stuff-class pixels should have instance_id=0., Instances below score threshold should be excluded., Should handle case with zero instances gracefully., Test batch panoptic merge., Batch merge should return (B, N) outputs., Test panoptic merge algorithm. (+9 more)

### Community 370 - "Community 370"
Cohesion: 0.17
Nodes (24): a1_frequency_aware_overclustering(), a2_depth_edge_semantic_split(), a3_knn_propagation(), a4_geometric_copypaste(), combine_a1_a2(), combine_all(), get_cityscapes_images(), load_cause_codes() (+16 more)

### Community 371 - "Community 371"
Cohesion: 0.12
Nodes (24): crf_refine(), discover_samples(), extract_dino_features_batch(), extract_pseudo_masks_gpu(), gpu_build_knn_graph(), load_config(), load_dino_backbone(), main() (+16 more)

### Community 372 - "Community 372"
Cohesion: 0.11
Nodes (20): compute_affinity(), crf_refine_mask(), DINOv3FeatureExtractor, _get_autocast_ctx(), main(), maskcut_single_image(), ncut_bipartition(), preprocess_image() (+12 more)

### Community 373 - "Community 373"
Cohesion: 0.12
Nodes (9): MockHFAttention, MockHFBlock, MockHFDepthModel, MockHFEncoder, MockHFMLP, MockProcessor, Tiny HF-style depth model: 12 blocks, dim=192, patch_grid=8x8., smoke_test() (+1 more)

### Community 374 - "Community 374"
Cohesion: 0.1
Nodes (10): MonoDatasetSingleCam, Returns a single training item from the dataset as a dictionary.          Values, Superclass for monocular dataloaders      Args:         data_path         filena, Resize colour images to the required scales and augment if required          We, MonoDatasetSingleCam, NYUDataset, NYUrawDataset, Superclass for different types of KITTI dataset loaders (+2 more)

### Community 375 - "Community 375"
Cohesion: 0.14
Nodes (10): DA3_Streaming, depth_to_point_cloud_vectorized(), data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...], depth: [N, H, W] numpy array or torch tensor     intrinsics: [N, 3, 3] numpy arr, Save camera poses from all chunks to txt and ply files         - txt file: Each, Clean up temporary files and calculate reclaimed disk space.          This metho, remove_duplicates(), create_point_cloud() (+2 more)

### Community 376 - "Community 376"
Cohesion: 0.13
Nodes (22): euler_to_matrix(), extrinsics_to_pivot_parameters(), generate_coordinate_frame(), generate_rotation_coordinate_frame(), generate_wobble_transformation(), interpolate_circular(), interpolate_extrinsics(), interpolate_intrinsics() (+14 more)

### Community 377 - "Community 377"
Cohesion: 0.17
Nodes (24): ascii_text(), curated_inventory(), entry(), family_for_path(), get_ccr_coverage(), get_graphify_state(), get_repo_state(), invalid_results() (+16 more)

### Community 378 - "Community 378"
Cohesion: 0.13
Nodes (23): _concat_mask_banks(), _config_from_checkpoint(), export(), _majority_label_in_mask(), _max_in_masks(), _mean_in_masks(), _nms_masks(), output_to_pseudolabels() (+15 more)

### Community 379 - "Community 379"
Cohesion: 0.18
Nodes (23): add_gt_stats(), add_matched_stats(), batch_iou_safe(), cache_city_stem(), class_table(), compact_table(), diagnose_candidate_setting(), diagnose_setting() (+15 more)

### Community 380 - "Community 380"
Cohesion: 0.13
Nodes (16): build_feats(), CentroidHead, DualHead, fit_finch(), fit_kmeans(), fit_spherical(), list_cache(), main() (+8 more)

### Community 381 - "Community 381"
Cohesion: 0.08
Nodes (19): Tests the Assigner objects.  CommandLine:     pytest tests/test_assigner.py, Test corner case where an network might predict no boxes and no gt, Test corner case where an image might have no true detections, Test corner case where an image might predict no points and no gt, Test corner case where an image might have no true detections, Test corner case where an network might predict no boxes, Test corner case where an network might predict no boxes and no gt, Test random instantiation of assign result to catch corner cases (+11 more)

### Community 382 - "Community 382"
Cohesion: 0.11
Nodes (23): build(), _build_dinosaur(), build_dinosaur_base_patch14_224_topk3(), build_dinosaur_base_patch14_518_topk3(), build_dinosaur_small_patch14_224_topk3(), build_dinosaur_small_patch14_518_topk3(), build_preprocessing(), _cfg() (+15 more)

### Community 383 - "Community 383"
Cohesion: 0.12
Nodes (10): ends_with(), segment_graph(), segment_mesh(), segment_mesh_wrapper(), size(), getNormalized(), length(), math() (+2 more)

### Community 384 - "Community 384"
Cohesion: 0.13
Nodes (1): LSegmentationModule

### Community 385 - "Community 385"
Cohesion: 0.13
Nodes (11): AllEntrySelector, EntrySelector, _FieldEntryRangePredicate, FieldEntrySelector, _FieldEntryValuePredicate, from_string(), Selector that accepts all entries, Selector that accepts only entries that match provided field     specifier(s). O (+3 more)

### Community 386 - "Community 386"
Cohesion: 0.13
Nodes (23): all_gather(), create_local_process_group(), gather(), _get_global_gloo_group(), get_local_process_group(), get_local_rank(), get_local_size(), get_rank() (+15 more)

### Community 387 - "Community 387"
Cohesion: 0.09
Nodes (18): check_intersection_interval(), draw_sample(), fps_downsample(), get_class_models(), load_meshes(), Samples n_object from model_dict, Samples n_object scales in intervl scale_interval, Loads the meshes in the list and scales according to provided list.     The load (+10 more)

### Community 388 - "Community 388"
Cohesion: 0.13
Nodes (9): build_2dconv_block(), build_3dconv_block(), Conv3DBlock_layernorm, conv_devide_H, conv_dv_WH, conv_dv_WH2d, conv_keep_all, conv_keep_all2d (+1 more)

### Community 389 - "Community 389"
Cohesion: 0.1
Nodes (13): DataModule, freeze_bn_affine(), MegaFlowLit, Perform a single training step.          Parameters         ----------         d, Validation step for chairs, sintel, and spring datasets.          Parameters, PyTorch Lightning module for MegaFlow optical flow estimation.      This class i, Validation step for KITTI dataset.          Parameters         ----------, Main validation step that routes to specific validation methods.          Parame (+5 more)

### Community 390 - "Community 390"
Cohesion: 0.13
Nodes (16): HTMLParser, _AnchorParser, build_search_url(), collapse_whitespace(), decode_duckduckgo_redirect(), dedupe_hits(), execute_web_search(), _extract_links() (+8 more)

### Community 391 - "Community 391"
Cohesion: 0.13
Nodes (9): input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)         ou, input : z: (b, num_hist, num_patches, emb_dim)         output: z: (b, num_hist,, input :   z: (b, num_frames, num_patches, emb_dim)         output: obs: (b, num_, input :   z: (b, num_frames, num_patches, emb_dim)         output: obs: (b, num_, input: z (tensor)         output: z_obs (dict), z_act (tensor), input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size), input:  obs_0 (dict): (b, n, 3, img_size, img_size)                   act: (b, t, input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) (+1 more)

### Community 392 - "Community 392"
Cohesion: 0.12
Nodes (23): build_panels(), colorize_19cls(), colorize_k80(), draw_arrow_with_label(), draw_panel(), draw_panel_annotation(), extract_cause_features_crop(), load_cause_models() (+15 more)

### Community 393 - "Community 393"
Cohesion: 0.12
Nodes (23): build_thumbnails(), colorize_19cls(), colorize_k80(), draw_box(), draw_img(), draw_merge_lines(), draw_varrow(), load_cause_models() (+15 more)

### Community 394 - "Community 394"
Cohesion: 0.11
Nodes (11): backward(), forward(), Fp32LayerNorm, FusedLayerNorm, _layer_norm_bwd(), _layer_norm_fwd(), LayerNorm(), LayerNormFn (+3 more)

### Community 395 - "Community 395"
Cohesion: 0.11
Nodes (21): draw_camera_bbox3d_on_img(), draw_depth_bbox3d_on_img(), draw_lidar_bbox3d_on_img(), plot_rect3d_on_img(), project_pts_on_img(), # TODO: remove third parameter in all functions here in favour of img_metas, Project the 3D bbox on 2D plane and draw on input image.      Args:         bbox, Project the 3D points cloud on 2D image.      Args:         points (numpy.array) (+13 more)

### Community 396 - "Community 396"
Cohesion: 0.11
Nodes (20): change_color_of_binary_mask(), color_by_yaw(), correct_yaw(), draw_lanes_in_agent_frame(), draw_lanes_on_image(), get_lanes_in_radius(), get_patchbox(), load_all_maps() (+12 more)

### Community 397 - "Community 397"
Cohesion: 0.1
Nodes (13): Combined, DataRouter, Utility function related to routing of information.  These utility functions all, Data router for modules that don't support the RoutableMixin.      This allows t, Module to combine multiple modules and store their outputs.      A combined modu, Module to apply another module in a recurrent fashion over a axis.      This mod, Mixin class that allows to connect any element of a (nested) dict with a module, Initialize recurrent module.          Args:             module: The module that (+5 more)

### Community 398 - "Community 398"
Cohesion: 0.12
Nodes (16): Write optical flow to file.      If v is None, uv is assumed to contain both u a, Read .flo file in Middlebury format, Read .flo file in Middlebury format, Write optical flow to file.          If v is None, uv is assumed to contain both, Write optical flow to file.          If v is None, uv is assumed to contain both, Write optical flow to file.      If v is None, uv is assumed to contain both u a, read_flow_generic(), read_gen() (+8 more)

### Community 399 - "Community 399"
Cohesion: 0.16
Nodes (8): apply_transform_to_params(), denormalize_coords(), _IdentityParams, normalize_coords(), RandomAffineFlow, RandomMirror, scale indices from [-1, 1] to [0, width/height], scale indices from [0, width/height] to [-1, 1]

### Community 400 - "Community 400"
Cohesion: 0.23
Nodes (22): _ev(), Tests for agent.sft.tagger — one test per tag namespace., test_cost_buckets(), test_empty_trajectory_has_required_tags(), test_feedback_tags(), test_hf_job_oom(), test_hf_job_tags(), test_model_family() (+14 more)

### Community 401 - "Community 401"
Cohesion: 0.12
Nodes (7): generate_mask_matrix(), QAttention, QFeedForward, QTransformer, QTransformerBlock, QViTPredictor, 替代 nn.ModuleList([QAttention, QFeedForward])

### Community 402 - "Community 402"
Cohesion: 0.12
Nodes (12): EpisodeSegmentor, Episode Segmentor - 智能对话分段和Episode生成 将粗粒度的Session拆分为细粒度的Episodes, 判断Session是否需要拆分                  Args:             message_count: 消息数量, 将一个Session的消息拆分为多个Episodes                  Args:             user_id: 用户ID, 使用LLM检测话题边界                  Args:             messages: 消息列表, 从消息组构建Episode                  Args:             user_id: 用户ID             messa, 使用LLM生成Episode摘要                  Args:             messages: 消息列表, 提取Episode时间戳（直接使用消息的session时间戳） (+4 more)

### Community 403 - "Community 403"
Cohesion: 0.12
Nodes (10): _apply_ins_words(), build_decoder(), forward_word_ins(), _get_ins_targets(), InsertionTransformerDecoder, InsertionTransformerModel, NegativeDistanceScore, # TODO: decoding for InsertionTransformer (+2 more)

### Community 404 - "Community 404"
Cohesion: 0.16
Nodes (16): allocate(), char2int(), insertAfter(), isdelim(), isdigit(), isspace(), isxdigit(), jsonParse() (+8 more)

### Community 405 - "Community 405"
Cohesion: 0.15
Nodes (16): AdapterTrainingDataset, build_cause_args(), cross_view_consistency_loss(), denormalize(), depth_correlation_loss(), dino_distillation_loss(), ema_update(), grid_sample() (+8 more)

### Community 406 - "Community 406"
Cohesion: 0.13
Nodes (13): create_ring_transforms(), example_usage(), Convert absolute pose sequence back to sequential relative transforms         T_, Convert SE3 to Sim3 (add unit scale), Build loop closure constraints, Compute residuals (modified from original code), Main optimization function          Args:             sequential_transforms: Inp, Generate a ring of Sim3 transforms with rotation, adding slight rotational noise (+5 more)

### Community 407 - "Community 407"
Cohesion: 0.17
Nodes (22): aggregate_thing_metrics(), average_precision_at_threshold(), boundary_score(), center_score(), compute_mask_ap(), depth_edge_map(), discover_sam_slice(), dominant_thing_class() (+14 more)

### Community 408 - "Community 408"
Cohesion: 0.17
Nodes (14): _as_str(), average_logs(), build_samples(), consensus_quality(), crop_box_from_mask(), load_npz(), main(), parse_args() (+6 more)

### Community 409 - "Community 409"
Cohesion: 0.18
Nodes (22): build_cache(), load_appearance(), load_bank_prior_maps(), load_mask_bank(), load_semantic_codes(), main(), masks_from_instance_png(), masks_from_npz() (+14 more)

### Community 410 - "Community 410"
Cohesion: 0.19
Nodes (21): _get_config_directory(), _get_config_module(), _get_detector_cfg(), _get_model_cfg(), Find the predefined detector config directory., Load a configuration as a python module., Grab configs necessary to create a model.      These are deep copied to allow fo, Grab configs necessary to create a detector.      These are deep copied to allow (+13 more)

### Community 411 - "Community 411"
Cohesion: 0.19
Nodes (20): computeContentHash(), dbAddResult(), dbGetProblemId(), dbGetProgramId(), dbInitConnection(), detectCPUModel(), detectCPUs(), getArg() (+12 more)

### Community 412 - "Community 412"
Cohesion: 0.17
Nodes (21): _demo_mm_inputs(), _get_config_directory(), _get_config_module(), _get_detector_cfg(), pytest tests/test_forward.py, Find the predefined detector config directory, Create a superset of inputs needed to run test or train batches.      Args:, Find the predefined detector config directory. (+13 more)

### Community 413 - "Community 413"
Cohesion: 0.12
Nodes (9): Params, Run per image evaluation on given images and store results (a list of dict) in s, perform evaluation for single category and image         :return: dict (single i, Accumulate per image evaluation results and store the result in self.eval, Compute and display summary metrics for evaluation results.         Note this fu, Params for coco evaluation api, Initialize CocoEval using coco APIs for gt and dt         :param cocoGt: coco ob, Prepare ._gts and ._dts for evaluation based on params         :return: None (+1 more)

### Community 414 - "Community 414"
Cohesion: 0.1
Nodes (13): PanopticTrackingEval, Multi-object Panoptic Tracking evaluation. Code written by Motional and the Robo, Panoptic tracking evaluator, Add panoptic tracking metrics for one frame/batch.         :param scene: str, na, :param n_classes: Number of classes.         :param min_stuff_cls_id: Minimum st, Calculate PTQ metrics.         :return: (mean_PTQ, all_class_PTQ, mean_sPTQ, all, Calculate MOTSA metrics.         :return: (mean_MOTSA, mean_sMOTSA, mean_MOTSP)., Calculate Lidar Segmentation and Tracking Quality (LSTQ) metric. https://arxiv.o (+5 more)

### Community 415 - "Community 415"
Cohesion: 0.16
Nodes (10): dataclass_to_dict(), Embedding, Figure, Image, Images, Classes for handling different types of visualizations., Placeholder class for SummaryWriter.      Emulates interface of `torch.utils.ten, SummaryWriter (+2 more)

### Community 416 - "Community 416"
Cohesion: 0.13
Nodes (18): _at_least_x_are_equal(), _decode_and_center_crop(), _decode_and_random_crop(), distorted_bounding_box_crop(), _flip(), preprocess_for_eval(), preprocess_for_train(), preprocess_image() (+10 more)

### Community 417 - "Community 417"
Cohesion: 0.12
Nodes (10): AKOrN, Create a strided convolution layer., Create a readout block., Create all network layers., Artficial Kuramoto Oscillator Neurons (AKOrN) for classification tasks., Extract features from input through the network layers., Forward pass through the network.                  Args:             inp: Input, Expand parameter to match the number of layers. (+2 more)

### Community 418 - "Community 418"
Cohesion: 0.16
Nodes (16): Autoplay(), Breakpoints(), bulmaCarousel(), _classCallCheck(), Coordinate(), _defineProperty(), EventEmitter(), Fade() (+8 more)

### Community 419 - "Community 419"
Cohesion: 0.13
Nodes (6): FlowAugmentor, ImageAugmentor, Photometric augmentation, Photometric augmentation, Occlusion augmentation, SparseFlowAugmentor

### Community 420 - "Community 420"
Cohesion: 0.14
Nodes (21): _make_cm(), _msg(), Regression tests for the 2026-05-03 infinite-compaction-loop bug.  Pod logs from, The system prompt is the agent's instructions — must never be truncated.      Ca, token_counter occasionally raises on edge-case content. A blip there     must NO, The whole point of the new behavior: don't loop on a useless     compaction call, Regression for the second P0 caught by bot review on PR #213.      When ``len(it, Happy path: when compaction does its job, no exception raised. (+13 more)

### Community 421 - "Community 421"
Cohesion: 0.1
Nodes (14): extract_usage(), HeartbeatSaver, _infer_push_to_hub(), All agent observability in one module.  Every telemetry signal the agent emits —, Emit ``hf_job_submit``. Returns the monotonic start timestamp so the     caller, Flat usage dict from a litellm response or final-chunk usage object.      Normal, Emit a ``pro_conversion`` event for a user we've previously observed     as non-, Emit a ``credits_topped_up`` event when an hf_job submits successfully     in a (+6 more)

### Community 422 - "Community 422"
Cohesion: 0.11
Nodes (11): EmbeddingManager, Generate embeddings for multiple texts (with caching)                  Args:, Embedding manager with caching and batch processing     Manages embedding genera, Compute cosine similarity between two embeddings                  Args:, Compute similarities between query and multiple embeddings                  Args, Generate cache key from text (MD5 hash), Clear all cached embeddings, Get number of cached embeddings (+3 more)

### Community 423 - "Community 423"
Cohesion: 0.13
Nodes (6): 更新标签的 embeddings（用于后续的对齐模型）, 标签 DAG 索引          核心思想：     1. Tags 形成 DAG 结构（如：工作 -> 编程 -> Python）     2. 查询时沿, 添加 episode 的标签                  Args:             episode_id: Episode ID, 基于标签查询                  Args:             query_tags: 查询标签             expand_de, TagDAGIndex, TagNode

### Community 424 - "Community 424"
Cohesion: 0.13
Nodes (3): ArgTypes, DatasetWriter, FilesDataset

### Community 425 - "Community 425"
Cohesion: 0.12
Nodes (9): COCOeval, Params, Run per image evaluation on given images and store results (a list of dict) in s, perform evaluation for single category and image         :return: dict (single i, Accumulate per image evaluation results and store the result in self.eval, Compute and display summary metrics for evaluation results.         Note this fu, Params for coco evaluation api, Initialize CocoEval using coco APIs for gt and dt         :param cocoGt: coco ob (+1 more)

### Community 426 - "Community 426"
Cohesion: 0.13
Nodes (21): compute_mbps_loss(), create_model_from_config(), create_synthetic_batch(), dry_run(), evaluate_model(), load_config(), main(), print_device_info() (+13 more)

### Community 427 - "Community 427"
Cohesion: 0.13
Nodes (21): compute_dataset_ap_range(), compute_dataset_ap_single_threshold(), download_dino_weights(), extract_gt_instance_masks(), extract_predictions(), interpolate_pos_embed(), load_cad_checkpoint(), load_config() (+13 more)

### Community 428 - "Community 428"
Cohesion: 0.13
Nodes (19): checkpoint_paths(), load_checkpoint(), load_checkpoint_to_cpu(), load_model_ensemble(), load_model_ensemble_and_task(), load_pretrained_component_from_model(), prune_state_dict(), Load a checkpoint and restore the training iterator.      *passthrough_args* wil (+11 more)

### Community 429 - "Community 429"
Cohesion: 0.14
Nodes (21): auto_device(), generate_da2(), generate_depthpro(), generate_dinov3_vitl(), generate_marigold(), generate_spidepth(), generate_zoedepth(), get_image_paths() (+13 more)

### Community 430 - "Community 430"
Cohesion: 0.12
Nodes (16): dice_loss(), gather_from_masks(), _hungarian_match(), Mask2FormerCriterion, Mask2Former Loss with Hungarian Matching.  Implements bipartite matching between, Compute total loss with deep supervision.          Args:             outputs: {p, Compute loss for one decoder output layer., Dice loss on point-sampled masks.      Args:         inputs: (N_matched, num_poi (+8 more)

### Community 431 - "Community 431"
Cohesion: 0.12
Nodes (19): _adapt_depth_decoder(), DepthAdapter, _find_encoder_blocks(), _fingerprint_attention_style(), _get_block_ancestor(), _inject_generic_vit(), inject_lora_into_depth_model(), Frozen-Feature Depth Adapter.  Learns a small residual correction to frozen CAUS (+11 more)

### Community 432 - "Community 432"
Cohesion: 0.09
Nodes (12): Tests for hierarchical density-frozen cluster merging.  Algorithm: greedy agglom, When target_k == input k, no merging happens; identity mapping., First merge happens between the two most-similar non-frozen centroids., Identify low-population centroids to freeze as rare modes., The n_freeze indices with lowest pixel counts are returned., End-to-end agglomerative merge with frozen-centroid constraint., Final cluster count equals target_k., Centroids in frozen_indices are unchanged in the output. (+4 more)

### Community 433 - "Community 433"
Cohesion: 0.13
Nodes (10): ColumnParallelEmbedding, ColumnParallelLinear, parallel_linear_func(), ParallelEmbeddings, ParallelLinearFunc, We're doing Tensor Parallel with sequence parallelism: we do the matmul and then, If max_position_embeddings <= 0, there's no position embeddings, input_ids: (batch, seqlen)         position_ids: (batch, seqlen) (+2 more)

### Community 434 - "Community 434"
Cohesion: 0.14
Nodes (12): EoMTPanopticShim, EoMTWithTTA, forward(), _forward_eval(), logits_to_d2_prediction(), _Predictor, Detectron2-interface adapter that lets the EoMT (DINOv2 ViT-B + masked attention, Raw EoMT forward on a [B,3,H,W] float image batch in [0,1]. (+4 more)

### Community 435 - "Community 435"
Cohesion: 0.12
Nodes (17): compress_quantized_densepose_chart_result(), decompress_compressed_densepose_chart_result(), DensePoseChartResult, DensePoseChartResultCompressed, DensePoseChartResultQuantized, DensePoseChartResultWithConfidences, quantize_densepose_chart_result(), DensePose results for chart-based methods represented by labels and inner     co (+9 more)

### Community 436 - "Community 436"
Cohesion: 0.12
Nodes (12): calculate_uncertainty(), dice_loss(), Create the criterion.         Parameters:             num_classes: number of obj, Classification loss (NLL)         targets dicts must contain the key "labels" co, Compute the losses related to the masks: the focal loss and the dice loss., This performs the loss computation.         Parameters:              outputs: di, Compute the DICE loss, similar to generalized IOU for masks     Args:         in, Args:         inputs: A float tensor of arbitrary shape.                 The pre (+4 more)

### Community 437 - "Community 437"
Cohesion: 0.11
Nodes (13): cal_all_metrics(), cal_ap_frame(), cal_ap_video(), cal_dice_safe(), cal_J_safe(), calculate_model_flops_slot_difussion(), get_component_flops(), get_model_infer_flops() (+5 more)

### Community 438 - "Community 438"
Cohesion: 0.15
Nodes (8): build_2dconv_block(), build_3dconv_block(), conv_devide_H, conv_dv_WH, conv_dv_WH2d, conv_keep_all, conv_keep_all2d, conv_keep_all_true3D

### Community 439 - "Community 439"
Cohesion: 0.13
Nodes (19): affinity_matrix_regularization(), align_permutations_across_frames(), align_permutations_across_n_frames(), apply_permutation(), compute_cosine_similarity_matrix(), compute_soft_permutation(), cosine_similarity_loss_neighboring(), cosine_similarity_matrix() (+11 more)

### Community 440 - "Community 440"
Cohesion: 0.1
Nodes (7): AllSampleAdapter, MEMLoRATPTTTARunner, this dataset classs is based on https://github.com/yossigandelsman/test_time_tra, SingleSampleAdapter, TestTimeTrainer, TPTMAETTARunner, TTTTrainDataset

### Community 441 - "Community 441"
Cohesion: 0.16
Nodes (20): flow_warp(), get_corresponding_map(), get_guassian_consistency_mask(), get_occu_mask_backward(), get_occu_mask_bidirection(), mask_out_of_image(), mesh_grid(), norm_grid() (+12 more)

### Community 442 - "Community 442"
Cohesion: 0.16
Nodes (20): cal_unsup_loss(), compute_fb_consistency(), compute_occlusion(), compute_range_map(), coords_grid(), flow_to_warp(), mask_invalid(), photo_loss_fn() (+12 more)

### Community 443 - "Community 443"
Cohesion: 0.13
Nodes (12): build_mask2former(), Mask2FormerWrapper, Mask2Former wrapper for CUPS unsupervised panoptic segmentation.  Loads a Huggin, Freeze the Swin backbone (pixel_level_module.encoder)., Normalize a CUPS image tensor to Mask2Former input format.          CUPS format:, Convert CUPS Detectron2-format batch to Mask2Former training inputs.          CU, Convert Mask2Former outputs to CUPS panoptic format.          Remaps M2F unified, Forward pass.          Training: returns Dict[str, Tensor] of losses.         In (+4 more)

### Community 444 - "Community 444"
Cohesion: 0.12
Nodes (14): __call__(), Mamba2Block, Mamba2Stack, Mamba2 Structured State Space Duality (SSD) for GPU.  GPU-optimized implementati, Single Mamba2 block with SSD kernel + residual + norm.      Args:         dim: M, Single Mamba2 block with SSD kernel + residual + norm.      Attributes:, Apply Mamba2 block.          Args:             x: Input of shape (B, L, D)., Structured State Space Duality (SSD) Kernel.      GPU-optimized implementation u (+6 more)

### Community 445 - "Community 445"
Cohesion: 0.1
Nodes (14): OracleStuffThings, Stuff-Things MLP Classifier.  Takes DBD, FCC, IDF cue features and classifies ea, Classify using ground truth.          Args:             cluster_labels: Predicte, Oracle stuff-things classifier using ground truth labels.      Used for ablation, Initialize oracle classifier.          Args:             thing_class_ids: List o, Classify using ground truth.          Args:             cluster_labels: Predicte, MLP classifier for stuff vs. things discrimination.      Takes three cue feature, Initialize StuffThingsClassifier.          Args:             hidden_dims: Hidden (+6 more)

### Community 446 - "Community 446"
Cohesion: 0.13
Nodes (12): DepthAnything3App, main(), Create and configure the Gradio application.          Returns:             Confi, Set up all event handlers for the application.          Args:             demo:, Main application class for Depth Anything 3 Gradio app., Initialize the application.          Args:             model_dir: Path to the mo, Set up visualization update handlers., Set up navigation handlers for measure tab. (+4 more)

### Community 447 - "Community 447"
Cohesion: 0.2
Nodes (19): _candidate_stems(), _cross_source_consensus(), _dilate4(), _erode4(), _image_gradient(), _image_index(), _load_full_npz(), main() (+11 more)

### Community 448 - "Community 448"
Cohesion: 0.14
Nodes (19): __call__(), compute_reference_frame_45_deg(), compute_reference_frame_90_deg(), compute_rotation_matrix_45_deg(), compute_rotation_matrix_90_deg(), compute_weighted_covariance(), InvertedDotProductAttentionKeyPerQuery, Slot Attention module with positional encodings in keys and values.    Feature p (+11 more)

### Community 449 - "Community 449"
Cohesion: 0.16
Nodes (19): Action, add_arguments(), add_parser(), create_argument_parser(), create_context(), DumpAction, execute(), execute_on_outputs() (+11 more)

### Community 450 - "Community 450"
Cohesion: 0.15
Nodes (8): BasicBlock, Bottleneck, Normalize, NormedLinear, ResNet in PyTorch.  For Pre-activation ResNet, see 'preact_resnet.py'.  Referenc, ResNet, ResNet18(), test()

### Community 451 - "Community 451"
Cohesion: 0.13
Nodes (15): _cast_to_config(), LazyCall, LazyConfig, load(), load_rel(), _patch_import(), _random_package_name(), Enhance relative import statements in config files, so that they:     1. locate (+7 more)

### Community 452 - "Community 452"
Cohesion: 0.1
Nodes (5): Test the iou calculation of boxes in different modes.      CommandLine:, Test the conversion of boxes between different modes.      CommandLine:, test_boxes3d_overlaps(), test_boxes_conversion(), test_get_box_type

### Community 453 - "Community 453"
Cohesion: 0.15
Nodes (16): FileManager, flow_read(), flow_read_flo(), flow_read_png(), flow_to_rgb(), flow_write(), flow_write_flo(), flow_write_png() (+8 more)

### Community 454 - "Community 454"
Cohesion: 0.1
Nodes (0): 

### Community 455 - "Community 455"
Cohesion: 0.16
Nodes (6): Decoder, Encoder, Quantize, input: (b, t, num_patches, emb_dim), ResBlock, VQVAE

### Community 456 - "Community 456"
Cohesion: 0.14
Nodes (11): 为对话生成标签和标签关系                  Args:             conversation: 对话数据（LoCoMo格式）, 对话标签生成器          使用LLM分析对话内容，提取关键主题作为标签，并识别标签之间的层级关系, 清理和验证标签关系                  Args:             relations_data: LLM 返回的关系数据 [{"pare, 构建对话摘要（避免token超限）                  Args:             conversation: 对话数据, 初始化标签生成器                  Args:             llm_client: LLM客户端（如果为None，会自动创建）, 清理和规范化标签                  Args:             tags: 原始标签列表             max_tags: 最, 关键词提取回退方案（当LLM失败时）                  使用简单的关键词匹配, 批量生成标签（向后兼容）                  Args:             conversations: 对话列表 (+3 more)

### Community 457 - "Community 457"
Cohesion: 0.16
Nodes (19): delete_city(), download_city(), download_dino_weights(), gsutil_run(), interpolate_pos_embed(), list_cities(), load_config(), load_depth() (+11 more)

### Community 458 - "Community 458"
Cohesion: 0.15
Nodes (17): cc_only_instances(), COCOPanopticGT, compute_pq(), compute_pq_from_accumulators(), depth_guided_instances(), discover_files(), main(), Standard Sobel splitting — same as Cityscapes version. (+9 more)

### Community 459 - "Community 459"
Cohesion: 0.12
Nodes (11): PseudoLabelCorrector, Adaptive pseudo-label correction inspired by Uni-UVPT (NeurIPS 2023).  Ported fr, Compute per-class IoU and append to queues., Evaluate whether to correct pseudo-labels for each class., Correct pseudo-labels for noisy classes., Reset all tracking state., Simpler alternative: entropy-based filtering without curve monitoring.      Filt, Filter high-entropy pseudo-labels.          Args:             logits: (B, C, H, (+3 more)

### Community 460 - "Community 460"
Cohesion: 0.13
Nodes (14): apply_rotary(), apply_rotary_emb(), apply_rotary_emb_torch(), ApplyRotaryEmb, backward(), forward(), Arguments:         x: (batch, seqlen, nheads, headdim) if cu_seqlens is None, x: (batch_size, seqlen, nheads, headdim)     cos, sin: (seqlen, rotary_dim / 2) (+6 more)

### Community 461 - "Community 461"
Cohesion: 0.16
Nodes (18): _boundary_from_masks(), __call__(), _center_from_masks(), ConsensusTargets, _edge_targets_from_superpixels(), LatentConsensusBuilder, LatentConsensusConfig, _mask_iou() (+10 more)

### Community 462 - "Community 462"
Cohesion: 0.11
Nodes (1): TestUVISEquations

### Community 463 - "Community 463"
Cohesion: 0.13
Nodes (12): KoLeoLoss, KoLeoLossDistributed, Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 -, Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 -, Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 -, Pairwise nearest neighbors for L2-normalized vectors.         Uses Torch rather, Pairwise nearest neighbors for L2-normalized vectors.         Uses Torch rather, Args:             student_output (BxD): backbone output of student (+4 more)

### Community 464 - "Community 464"
Cohesion: 0.13
Nodes (9): KITTI2Waymo, Convert action for single file.          Args:             file_idx (int): Index, Length of the filename list., Transform the coordinates with matrix T.          Args:             T (np.ndarra, KITTI predictions to Waymo converter.      This class serves as the converter to, Combine predictions in waymo format for each sample together.          Args:, Get file names of waymo raw data., Create folder for data conversion. (+1 more)

### Community 465 - "Community 465"
Cohesion: 0.16
Nodes (11): batched_multivariate_gaussian(), batched_render_gaussian_kabsch_mask(), get_box_pixel_weights(), get_mask_softness_fun(), is_point_in_box_array(), KabschDecoder, map_nan_padding_to_zeros(), # TODO: this seems like a bug: we normalize if normalize_gaussian == False (+3 more)

### Community 466 - "Community 466"
Cohesion: 0.16
Nodes (7): Processor, Args:             point3D: shapes (n_row, 3), while 3 represent x,y,z axis in or, Args:             point5D: shapes (n_row, 3+2), while 5 represent x,y,z,seg,bin, Args:         max_slope(float): Local maximum slope of the ground.         max_e, Args:             point5D: shapes (n_row, 5), while 5 represent x,y,z,seg,bin ax, Module: Processor      Args:         n_segments(int): The number of fan-shaped r, Segmentation

### Community 467 - "Community 467"
Cohesion: 0.12
Nodes (7): color_mask_to_instance_mask(), color_mask_to_instance_mask_coco(), label_from_coco(), label_from_movi(), label_from_pascal(), Optimized version using vectorized operations and integer hashing     - Excludes, Optimized version for COCO-style masks     - Only excludes black [0,0,0]

### Community 468 - "Community 468"
Cohesion: 0.12
Nodes (10): LinearSplitter, Projector, Projector MLP          Args:             in_features (int): input channels, x : feature block; shape - n, c, h, w         b_prev : previous bin widths norme, Bin center regressor network. Bin centers are bounded on (min_depth, max_depth), Returns tensor of bin_width vectors (centers). One vector b for every pixel, Bin center regressor network. Bin centers are unbounded          Args:, Returns tensor of bin_width vectors (centers). One vector b for every pixel (+2 more)

### Community 469 - "Community 469"
Cohesion: 0.2
Nodes (18): add_stats(), depth_gradient(), depth_path(), feature_path(), format_duration(), infer_feature_grid(), load_cluster_to_class(), main() (+10 more)

### Community 470 - "Community 470"
Cohesion: 0.13
Nodes (4): FlatFolderDataset, PhotoWCT, Copyright (C) 2018 NVIDIA Corporation.  All rights reserved. Licensed under the, Timer

### Community 471 - "Community 471"
Cohesion: 0.14
Nodes (10): OutputProcessor, Extract extrinsics tensor from model output and convert to numpy.          Args:, Extract intrinsics tensor from model output and convert to numpy.          Args:, Extract sky tensor from model output and convert to numpy.          Args:, Extract auxiliary data from model output and convert to numpy.          Args:, Output processor for converting model outputs to Prediction objects.      Handle, Initialize the output processor., Convert model output to Prediction object.          Args:             model_outp (+2 more)

### Community 472 - "Community 472"
Cohesion: 0.25
Nodes (18): city(), colorize_instances(), colorize_labels(), depth_to_rgb(), load_dcfa_sem(), load_depth(), load_depthcc(), load_label() (+10 more)

### Community 473 - "Community 473"
Cohesion: 0.18
Nodes (16): get_coco_style_results(), get_distortions_from_file(), get_distortions_from_results(), get_results(), get_voc_style_results(), main(), print_coco_results(), coco_eval_with_return() (+8 more)

### Community 474 - "Community 474"
Cohesion: 0.12
Nodes (17): __call__(), get_normal_initializer(), get_uniform_initializer(), ParamStateInitLearnablePositions, ParamStateInitLearnablePositionsRotationsScales, ParamStateInitLearnablePositionsScales, ParamStateInitRandomPositions, ParamStateInitRandomPositionsRotationsScales (+9 more)

### Community 475 - "Community 475"
Cohesion: 0.17
Nodes (7): cleanup_annotation(), DensePoseDataRelative, extract_segmentation_mask(), Dense pose relative annotations that can be applied to any bounding box:, # NOTE: This assumes that HorizFlipTransform is the only one that does flip, # NOTE: This assumes that HorizFlipTransform is the only one that does flip, # TODO: annotation instance is accepted if it contains either

### Community 476 - "Community 476"
Cohesion: 0.16
Nodes (1): TestVisualizer

### Community 477 - "Community 477"
Cohesion: 0.16
Nodes (8): BasicBlock, build_wideresnet(), mish(), NetworkBlock, PSBatchNorm2d, Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv, How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/20, WideResNet

### Community 478 - "Community 478"
Cohesion: 0.18
Nodes (5): Pointnet2Backbone, Pointnet2Backbone_tiny, Pointnet2Backbone_tiny_noatten, r"""        Backbone network for point cloud feature learning.        Based on P, SelfAttentionLayer

### Community 479 - "Community 479"
Cohesion: 0.18
Nodes (2): get_evenly_distributed_colors(), Trainer

### Community 480 - "Community 480"
Cohesion: 0.18
Nodes (6): ASPP, ASPPConv, ASPPPooling, DeepLabHead, DeepLabV3, ResnetDilated

### Community 481 - "Community 481"
Cohesion: 0.16
Nodes (6): dist_collect(), This class is based on https://github.com/azshue/TPT/blob/main/clip/custom_clip., collect all tensor from all GPUs     args:         x: shape (mini_batch, ...), This function comes from https://github.com/azshue/TPT/blob/main/clip/custom_cli, TPTHFOpenCLIP, TPTTextEncoder

### Community 482 - "Community 482"
Cohesion: 0.14
Nodes (10): _FakeDB, _FakeProUsers, Unit tests for the optional durable session store abstraction., Re-checking a converted user must not re-emit the event., In-memory stand-in for the ``pro_users`` collection.      Supports just enough o, Joining as Pro shouldn't count as a conversion., _store_with_fake_db(), test_mark_pro_seen_emits_conversion_after_seeing_user_as_free() (+2 more)

### Community 483 - "Community 483"
Cohesion: 0.15
Nodes (14): boundary_alignment_loss(), ConsistencyLoss, depth_boundary_coherence_loss(), Cross-Branch Consistency Losses.  L_consistency = lambda_u * L_uniform + lambda_, Depth-Boundary Coherence (DBC) Loss.      Ensures that prediction boundaries ali, Depth-Boundary Coherence (DBC) Loss.      Ensures that prediction boundaries ali, Combined cross-branch consistency loss.      Args:         lambda_uniform: Weigh, Combined cross-branch consistency loss.      Args:         lambda_uniform: Weigh (+6 more)

### Community 484 - "Community 484"
Cohesion: 0.14
Nodes (17): _convert_attention_hf(), convert_dinov3_weights(), _convert_layernorm(), _convert_mlp_hf(), load_flax_params(), _load_hf_state_dict(), PyTorch to JAX/Flax weight converter for DINOv3 ViT-B/16.  Converts pretrained D, Convert HF-format attention weights to Flax format.      HuggingFace DINOv3 stor (+9 more)

### Community 485 - "Community 485"
Cohesion: 0.16
Nodes (7): AdaptiveSoftmax, In order to be efficient, the AdaptiveSoftMax does not compute the         score, Args:             input: (b x t x d)             target: (b x t)         Returns, Computes the log probabilities for all the words of the vocabulary,         give, This is an implementation of the efficient softmax approximation for     graphic, TiedHeadModule, TiedLinear

### Community 486 - "Community 486"
Cohesion: 0.17
Nodes (2): _get_word_log_probs(), LikelihoodratioModule

### Community 487 - "Community 487"
Cohesion: 0.18
Nodes (15): align_poses_umeyama(), _apply_sim3_to_poses(), apply_umeyama_alignment_to_ext(), batch_align_poses_umeyama(), _median_nn_thresh(), _poses_from_ext(), _rand_pose(), _rand_rot() (+7 more)

### Community 488 - "Community 488"
Cohesion: 0.16
Nodes (13): _ConvBlock, corrupt_seed_masks(), dice_loss_from_logits(), mask_edges(), _min_pool2d(), Trainable UIS proposal refiner for RGB-only Cityscapes instance masks.  This is, Return a normalized RGB edge map for crop-local boundary alignment., Apply differentiable-free mask corruption for denoising training. (+5 more)

### Community 489 - "Community 489"
Cohesion: 0.14
Nodes (5): ConvDoRALinear, DoRALinear, LoRAConv2d, LoRALinear, wrap_conv2d_if_match()

### Community 490 - "Community 490"
Cohesion: 0.22
Nodes (16): boundary_scores(), class_fraction_scores(), depth_boundary_map(), load_bank_with_classes(), load_depth_map(), main(), parse_class_float_map(), parse_class_int_map() (+8 more)

### Community 491 - "Community 491"
Cohesion: 0.2
Nodes (17): candidate_features(), class_is_plausible(), generate_one(), interval_score(), load_base_anchors(), load_semantic_trainid(), main(), mask_semantic_fractions() (+9 more)

### Community 492 - "Community 492"
Cohesion: 0.21
Nodes (7): nms_edit_distance(), Compare the "keep" result of two nms call.     They are allowed to be different, Args:             box_scores (N, 5): boxes in corner-form and probabilities., test_batched_nms_rotated_0_degree_cuda(), test_nms_rotated_0_degree_cuda(), TestNMSRotated, TestScriptable

### Community 493 - "Community 493"
Cohesion: 0.21
Nodes (13): fg_hard_score(), fg_score_prev(), get_scale(), get_scale2(), get_scale3(), get_transform(), reward_kl(), reward_mix() (+5 more)

### Community 494 - "Community 494"
Cohesion: 0.14
Nodes (9): DDNDeepLabV3, Initializes DDNDeepLabV3 model         Args:             backbone_name: string,, DDNTemplate, Forward pass         Args:             images: (N, 3, H_in, W_in), Input images, Preprocess images         Args:             images: (N, 3, H, W), Input images, Initializes depth distribution network.         Args:             constructor: f, Get model         Args:             constructor: function, Model constructor, Removes layers from pretrained state dict that are not used or changed in model (+1 more)

### Community 495 - "Community 495"
Cohesion: 0.21
Nodes (10): clamp_preserve_gradients(), euclidian_distance(), euclidian_norm(), flatten(), ICSBP, pixel_coords(), ScalarGate, SemiConv (+2 more)

### Community 496 - "Community 496"
Cohesion: 0.18
Nodes (6): Block, DummyModel, LinearModel, make_simple_model(), MLP, ResidualMLP

### Community 497 - "Community 497"
Cohesion: 0.16
Nodes (15): box3d_iou(), box3d_vol(), convex_hull_intersection(), get_3d_box(), poly_area(), polygon_clip(), Clip a polygon with another polygon.     Ref: https://rosettacode.org/wiki/Suthe, Calculate 3D bounding box corners from its parameterization.      Input: (+7 more)

### Community 498 - "Community 498"
Cohesion: 0.26
Nodes (16): addStyleSheet(), contains(), createDocumentFragment(), createElement(), getElements(), getExpandoData(), is(), isEventSupported() (+8 more)

### Community 499 - "Community 499"
Cohesion: 0.15
Nodes (11): cal_all_metrics(), cal_ap_frame(), cal_ap_video(), cal_dice(), cal_J(), calculate_model_flops_slot_difussion(), get_component_flops(), get_model_infer_flops() (+3 more)

### Community 500 - "Community 500"
Cohesion: 0.13
Nodes (12): GradientBalancer, project_conflicting_gradients(), Gradient Balancing for Multi-Task Learning.  Implements PCGrad-style gradient pr, Balance gradients in-place on model parameters.          Convenience method that, Project secondary gradient to remove conflict with primary.      If the secondar, Project secondary gradient to remove conflict with primary.      If the secondar, Multi-task gradient balancing with PCGrad-style projection.      Resolves gradie, Multi-task gradient balancing with PCGrad-style projection.      Resolves gradie (+4 more)

### Community 501 - "Community 501"
Cohesion: 0.16
Nodes (13): BridgeLoss, cka_loss(), Bridge Loss Functions.  L_bridge = L_recon + lambda_cka * L_cka + lambda_h * L_s, Combined bridge loss.      L_bridge = L_recon + lambda_cka * L_cka + lambda_stat, Combined bridge loss.      L_bridge = L_recon + λ_cka · L_cka + λ_state · L_stat, Compute combined bridge loss.          Args:             original_semantic: Orig, Compute combined bridge loss.          Args:             original_semantic: Orig, Compute reconstruction loss after bridge round-trip.      L_recon = ||X - X_reco (+5 more)

### Community 502 - "Community 502"
Cohesion: 0.12
Nodes (10): DepthGHead, DepthGHeadSpatial, DepthG Semantic Segmentation Head.  Ported from visinf/depthg. Implements a 3-la, DepthG head that operates on spatial (2D) feature maps.      Same MLP but accept, Compute semantic codes from spatial features.          Args:             feature, DepthG semantic segmentation head.      Maps DINO ViT-S/8 features (384-dim) to, Compute semantic codes from DINO features.          Args:             features:, Get hard cluster assignments from soft codes via argmax.          For actual clu (+2 more)

### Community 503 - "Community 503"
Cohesion: 0.16
Nodes (2): MultiWozConvGraph, SelfPlayConvGraph

### Community 504 - "Community 504"
Cohesion: 0.13
Nodes (8): FairseqDecoder, Base class for decoders., Args:             prev_output_tokens (LongTensor): shifted output tokens of shap, Returns:             tuple:                 - the decoder's features of shape `(, Project features to the default output size, e.g., vocabulary size.          Arg, Get normalized probabilities (or log probs) from a net's output., Maximum input length supported by the decoder., Upgrade old state dicts to work with newer code.

### Community 505 - "Community 505"
Cohesion: 0.16
Nodes (8): _collect_grad(), EQLv2Loss, LDAMSemanticLoss, Softmax-head-compatible EQLv2-style foreground BCE loss.      The CUPS ROI head, LDAM for dense semantic logits.      Margins are larger for rarer classes and ar, Small Seesaw-loss implementation for CUPS ablations., SeesawSoftmaxLoss, _update_counts()

### Community 506 - "Community 506"
Cohesion: 0.17
Nodes (5): PanopticEval, Binaries and/or source for the following packages or projects are presented unde, Panoptic evaluation using numpy      authors: Andres Milioto and Jens Behley, Calculate Panoptic Quality (PQ) metrics, IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]

### Community 507 - "Community 507"
Cohesion: 0.18
Nodes (15): _build_walk_path(), get_tree_element(), is_namedtuple(), is_tensor_or_module(), map_tree(), Utilities for working with our own version of PyTrees which focus on torch tenso, Apply reduction function to a list of nested dicts.      This only considers ten, Apply a function to each element of a tree.      This only considers tensors at (+7 more)

### Community 508 - "Community 508"
Cohesion: 0.18
Nodes (13): apply_closing(), apply_opening(), Cam_mask_post_process(), CAM_to_slice_hardlabel(), clear_boundary(), decode_mask_with_multi_coord(), post_process_softmask(), post_process_softmask2() (+5 more)

### Community 509 - "Community 509"
Cohesion: 0.17
Nodes (13): calc_essential_matrix(), cam_and_oflow_2_optical_centers_and_rays(), midpoint_triangulate(), described in: Multiple view geometry in computer vision     also helpful: https:, For a given optical flow and.      Parameters     ----------     intr_inv_cam1 t, method str: "dlt", "midpoint", described in: Multiple view geometry in computer vision     also helpful: https:, For a given optical flow and.      Parameters     ----------     extr_cam1 torch (+5 more)

### Community 510 - "Community 510"
Cohesion: 0.18
Nodes (11): angle_rots(), calc_optical_flow_registration(), calc_pointsets_registration(), calc_pointsets_registration_from_corresp3d(), dist_angle_transfs(), dist_transls(), filter_sim_se3(), mask_points() (+3 more)

### Community 511 - "Community 511"
Cohesion: 0.18
Nodes (7): DoubleConv, LiFT, load_lift_checkpoints(), LiFT Module for ViT feature upsampling.  Code by: Saksham Suri and Matthew Walme, (convolution => [BN] => ReLU) * 2, Upscaling then double conv, Up

### Community 512 - "Community 512"
Cohesion: 0.17
Nodes (15): check_for_doom_loop(), detect_identical_consecutive(), detect_repeating_sequence(), extract_recent_tool_signatures(), _hash_args(), _normalize_args(), Doom-loop detection for repeated tool call patterns.  Detects when the agent is, Return the tool name if threshold+ identical consecutive calls are found. (+7 more)

### Community 513 - "Community 513"
Cohesion: 0.14
Nodes (7): Test whether unordered graph systems are created correctly., Tests that the set of next tokens is correct., Ensures the list of lists of tensors gets packed correctly., tensorize(), TestHelperRoutines, TestOrderedConstraintState, TestUnorderedConstraintState

### Community 514 - "Community 514"
Cohesion: 0.19
Nodes (15): create_coco_subset(), download_file(), download_nyu_depth_v2(), download_pascal_voc(), download_zoedepth(), _extract_nyu_mat(), main(), _progress_hook() (+7 more)

### Community 515 - "Community 515"
Cohesion: 0.17
Nodes (10): CauseTRInstanceConfig, CauseTRInstanceDecoderLayer, CauseTRInstanceModel, CauseTRInstanceOutput, _depth_encoding(), CAUSE-TR-style trainable decoder for target-unsupervised instances.  This module, Configuration for :class:`CauseTRInstanceModel`., Outputs from the trainable instance decoder. (+2 more)

### Community 516 - "Community 516"
Cohesion: 0.13
Nodes (14): kitti_data_prep(), lyft_data_prep(), nuscenes_data_prep(), Prepare the info file for scannet dataset.      Args:         root_path (str): P, Prepare the info file for s3dis dataset.      Args:         root_path (str): Pat, Prepare the info file for sunrgbd dataset.      Args:         root_path (str): P, Prepare the info file for waymo dataset.      Args:         root_path (str): Pat, Prepare data related to Kitti dataset.      Related data consists of '.pkl' file (+6 more)

### Community 517 - "Community 517"
Cohesion: 0.19
Nodes (5): dist_collect(), HFOpenCLIP, HFOpenCLIPImageEncoder, HFOpenCLIPImageProjector, collect all tensor from all GPUs     args:         x: shape (mini_batch, ...)

### Community 518 - "Community 518"
Cohesion: 0.14
Nodes (10): AttractorLayer, AttractorLayerUnnormed, exp_attractor(), inv_attractor(), Attractor layer for bin centers. Bin centers are unbounded, Args:             x (torch.Tensor) : feature block; shape - n, c, h, w, Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a =, Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attract (+2 more)

### Community 519 - "Community 519"
Cohesion: 0.21
Nodes (4): MockDDPWrapper, Model, A simple wrapper with an interface similar to DistributedDataParallel., TestModuleProxyWrapper

### Community 520 - "Community 520"
Cohesion: 0.28
Nodes (13): _kernel_close_to_default(), _kernel_is_reproducible(), _make_inputs(), _max_abs_diff(), _run_case_outputs(), _set_deterministic(), _set_seeds(), test_combined_kernel_close_to_default() (+5 more)

### Community 521 - "Community 521"
Cohesion: 0.26
Nodes (9): apply_transformation_residual_triton(), compute_huber_weights_triton(), compute_weighted_covariance_triton(), compute_weighted_mean_triton(), robust_weighted_estimate_sim3_triton(), warmup_triton(), weighted_estimate_se3_triton(), weighted_estimate_sim3_numba_triton() (+1 more)

### Community 522 - "Community 522"
Cohesion: 0.22
Nodes (5): Att_pooling, PointnetFPModule, PointnetSAModule, _PointnetSAModuleBase, PointnetSAModuleMSG

### Community 523 - "Community 523"
Cohesion: 0.15
Nodes (5): flip, M(), rota_coords, scale_coords, trans_coords

### Community 524 - "Community 524"
Cohesion: 0.21
Nodes (7): get_max_iou_with_same_class(), ProposalTargetLayer, Args:             batch_dict:                 batch_size:                 rois:, Args:             batch_dict:                 batch_size:                 rois:, Args:             batch_dict:                 batch_size:                 rois:, Args:             batch_dict:                 batch_size:                 rois:, sample_bg_inds()

### Community 525 - "Community 525"
Cohesion: 0.25
Nodes (8): artificial_flow_network_output(), artificial_logit_network_output(), artificial_network_output(), castf(), get_voxel_center_coords_m(), HeadDecoder, homogenize_coors(), scale_gradient()

### Community 526 - "Community 526"
Cohesion: 0.18
Nodes (12): _commit_switch(), is_valid_model_id(), _print_hf_routing_info(), print_model_listing(), probe_and_switch_model(), _probe_local_model(), Model-switching logic for the interactive CLI's ``/model`` command.  Split out o, Render the default ``/model`` (no-arg) view: current + suggested. (+4 more)

### Community 527 - "Community 527"
Cohesion: 0.21
Nodes (10): check_train_all(), check_train_pairs(), check_train_sentences(), count_train_in_other_set(), get_all_test_data(), get_messed_up_test_pairs(), load_pairs(), load_sentences() (+2 more)

### Community 528 - "Community 528"
Cohesion: 0.19
Nodes (7): DistributedDataParallelWrapper, Forward function.          Args:             inputs (tuple): Input data., Train step function.          Args:             inputs (Tensor): Input Tensor., A DistributedDataParallel wrapper for models in MMGeneration.      In MMedting,, Validation step function.          Args:             inputs (tuple): Input data., Wrap models with separate MMDistributedDataParallel.          It only wraps the, Scatter function.          Args:             inputs (Tensor): Input Tensor.

### Community 529 - "Community 529"
Cohesion: 0.29
Nodes (11): apply_sim3_direct_torch(), apply_transformation_torch(), compute_huber_weights_torch(), compute_residuals_torch(), huber_loss_torch(), PyTorch SIM3     point_maps: (b, h, w, 3) numpy array     s: scalar or (b,) arra, robust_weighted_estimate_sim3_torch(), warmup_torch() (+3 more)

### Community 530 - "Community 530"
Cohesion: 0.26
Nodes (12): box_iou_one(), checkpoint_config(), class_aware_nms(), export_one(), mask_iou_one(), masks_to_boxes(), resize_mask_bank(), resize_semantic() (+4 more)

### Community 531 - "Community 531"
Cohesion: 0.24
Nodes (8): BasicBlock, BasicBlockBase, BasicBlockIN, BasicBlockINBN, Bottleneck, BottleneckBase, BottleneckIN, BottleneckINBN

### Community 532 - "Community 532"
Cohesion: 0.23
Nodes (11): apply_mask(), get_mask_plot_colors(), merge_image_pair(), r""" Visualize model predictions, Get nr_colors uniformly spaced hues to plot mask values., to_numpy(), unnormalize(), vis_GT_gray() (+3 more)

### Community 533 - "Community 533"
Cohesion: 0.17
Nodes (2): TestClass, TestConstruction

### Community 534 - "Community 534"
Cohesion: 0.15
Nodes (1): TestTransformAnnotations

### Community 535 - "Community 535"
Cohesion: 0.15
Nodes (0): 

### Community 536 - "Community 536"
Cohesion: 0.23
Nodes (12): build_data_cfg(), main(), parse_args(), Visualize 3D point cloud and 3D bboxes., Visualize 3D point cloud and segmentation mask., Visualize 3D bboxes on 2D image by projection., Build data config for loading visualization data., Convert points and bboxes to Depth Coord and Depth Box mode. (+4 more)

### Community 537 - "Community 537"
Cohesion: 0.15
Nodes (5): Test_add_present_time_to_history, Test_fade_color, Test_get_track_box, Test_reverse_history, TestAgentBoxesWithFadedHistory

### Community 538 - "Community 538"
Cohesion: 0.24
Nodes (2): C_PROTO, CSS

### Community 539 - "Community 539"
Cohesion: 0.24
Nodes (3): apply(), Augment, Augment object for RAFT

### Community 540 - "Community 540"
Cohesion: 0.15
Nodes (3): Tests for LLM error classification helpers in agent.core.agent_loop.  Covers two, The whole point of the rate-limit schedule: total wait time should     exceed th, test_rate_limit_total_budget_covers_bedrock_bucket_recovery()

### Community 541 - "Community 541"
Cohesion: 0.21
Nodes (12): apply_edit(), fuzzy_find(), fuzzy_find_original_match(), _map_back(), _normalize_unicode(), Shared utilities for file editing tools — fuzzy matching, syntax validation, and, Find the *original* text in content that matches pattern fuzzily.      Returns (, Apply an edit operation to content.      Modes:       - replace: replace first o (+4 more)

### Community 542 - "Community 542"
Cohesion: 0.15
Nodes (1): Event

### Community 543 - "Community 543"
Cohesion: 0.24
Nodes (7): batch_by_size_baseline(), DataUtilsTest, _get_error_message(), Compare reference batch_by_size implementation with batch_by_size_baseline, TestBatchBySize, TestBatchBySizeFn, TestBatchBySizeVec

### Community 544 - "Community 544"
Cohesion: 0.18
Nodes (3): cexp2f(), cexpf(), complex_t()

### Community 545 - "Community 545"
Cohesion: 0.22
Nodes (9): build_group_list(), build_group_manifest(), gallery(), GalleryHandler, _is_plain_name(), main(), Main entry point for gallery server., _url_join() (+1 more)

### Community 546 - "Community 546"
Cohesion: 0.36
Nodes (11): _component_from_vec(), _copy_rama_subset(), main(), _mask_iou(), maskcut_masks(), _norm_features(), parse_args(), _resize_mask() (+3 more)

### Community 547 - "Community 547"
Cohesion: 0.27
Nodes (1): DiffSeg

### Community 548 - "Community 548"
Cohesion: 0.17
Nodes (8): apply_augmentation(), create_path(), get_fine_to_coarse(), process_image(), Map fine label indexing to coarse label indexing., This function creates data loading paths., This function reads and resizes images and labels., This function applies image augmentation to batches.

### Community 549 - "Community 549"
Cohesion: 0.26
Nodes (3): grid_sample_roi_align(), RoiAlign with scale 1.0., ROIAlignTest

### Community 550 - "Community 550"
Cohesion: 0.17
Nodes (1): TestLazyPythonConfig

### Community 551 - "Community 551"
Cohesion: 0.26
Nodes (8): generate_data(), make_dataset_dicts(), make_mask(), Makes a donut shaped binary mask., Returns a list of dicts that represents a single COCO data point for     object, TestConvertCOCO, TestRLEToJson, uncompressed_rle()

### Community 552 - "Community 552"
Cohesion: 0.3
Nodes (11): compute_aum(), crop_fov(), display_args(), eprint(), filter_by_ppscore(), load_velo_scan(), main(), process_batch_scene() (+3 more)

### Community 553 - "Community 553"
Cohesion: 0.26
Nodes (7): create_groundtruth_database(), crop_image_patch(), GTDatabaseCreater, _parse_coco_ann_info(), _poly2mask(), Given the raw data, generate the ground truth database.      Args:         datas, Given the raw data, generate the ground truth database. This is the     parallel

### Community 554 - "Community 554"
Cohesion: 0.21
Nodes (11): flow_tensor_to_image(), flow_to_image(), flow_uv_to_colors(), make_colorwheel(), Expects a two dimensional flow image of shape.      Args:         flow_uv (np.nd, Expects a two dimensional flow image of shape.      Args:         flow_uv (np.nd, Used for tensorboard visualization, Generates a color wheel for optical flow visualization as presented in: (+3 more)

### Community 555 - "Community 555"
Cohesion: 0.21
Nodes (8): load_summarized_table(), Create a reduced version of the validation metrics table.  The reduced version i, Generate and save the plot to disk.      Parameters     ----------     args : ar, Summarize the results and save them to the disk.      Parameters     ----------, Load the DataFrame and keep only columns according to the selected metrics., save_plots(), _shorten_columns_names(), summarize()

### Community 556 - "Community 556"
Cohesion: 0.17
Nodes (7): BrailleCanvas, _define_font(), Braille-character canvas for high-resolution terminal graphics.  Each terminal c, Convert text string to a list of (x, y) pixel positions using bitmap font., A pixel canvas that renders to braille characters., Define a simple 5×7 bitmap font for uppercase ASCII., text_to_pixels()

### Community 557 - "Community 557"
Cohesion: 0.32
Nodes (11): _bbox(), _boundary(), _city_from_stem(), _depth_grad(), _depth_path(), main(), BoxTeacher-style pseudo-instance quality scoring.  BoxTeacher scores pseudo mask, _resize_like() (+3 more)

### Community 558 - "Community 558"
Cohesion: 0.24
Nodes (11): evaluate_depthg(), load_config(), main(), Evaluate DepthG on validation set., Save checkpoint to disk., Load config with default fallback., Shard batch across devices., Train DepthG semantic segmentation. (+3 more)

### Community 559 - "Community 559"
Cohesion: 0.31
Nodes (10): area_prior(), border_fraction(), compactness_score(), normalize_component(), RankWeights, Unsupervised proposal ranking for coarse instance-mask banks., Score proposal masks using only image-local unsupervised signals., rerank_and_select() (+2 more)

### Community 560 - "Community 560"
Cohesion: 0.35
Nodes (10): _area_prior(), _border_fraction(), _existing_record(), fuse_one(), main(), _normalize(), parse_args(), _parse_bank() (+2 more)

### Community 561 - "Community 561"
Cohesion: 0.38
Nodes (10): components_from_labels(), coverage(), load_gt_patch_masks(), load_or_extract_features(), main(), parse_args(), percentiles(), resize_bool() (+2 more)

### Community 562 - "Community 562"
Cohesion: 0.18
Nodes (8): apply_augmentation(), create_path(), get_fine_to_coarse(), process_image(), This function applies image augmentation to batches., Map fine label indexing to coarse label indexing., This function creates data loading paths., This function reads and resizes images and labels.

### Community 563 - "Community 563"
Cohesion: 0.35
Nodes (3): limit_period(), main(), OpenPCDetWaymoDetectionMetricsEstimator

### Community 564 - "Community 564"
Cohesion: 0.35
Nodes (3): limit_period(), main(), OpenPCDetWaymoDetectionMetricsEstimator

### Community 565 - "Community 565"
Cohesion: 0.29
Nodes (3): forward(), LEASTEREOWrapperCPU, LEASTStereoWrapper

### Community 566 - "Community 566"
Cohesion: 0.31
Nodes (9): extract_flow(), filter_pc(), main(), read_root(), transform_global(), extract_flow(), extract_flow_on_camera(), main() (+1 more)

### Community 567 - "Community 567"
Cohesion: 0.18
Nodes (1): Tests for the secret scrubber used before session upload.

### Community 568 - "Community 568"
Cohesion: 0.31
Nodes (8): _sandbox_app(), test_file_and_command_routes_accept_sandbox_header_with_hf_bearer(), test_file_and_command_routes_reject_authorization_bearer_token(), test_file_and_command_routes_require_bearer_token(), test_health_is_public(), test_hf_bearer_alone_is_rejected_when_sandbox_token_is_configured(), test_legacy_hf_token_fallback_is_rejected(), test_protected_routes_fail_closed_without_configured_token()

### Community 569 - "Community 569"
Cohesion: 0.22
Nodes (1): Action

### Community 570 - "Community 570"
Cohesion: 0.25
Nodes (10): do_radial_blur(), is_lower(), layered_bluring(), layered_bluring_from_cost_volume(), lowres_layered_bluring_from_lowres_cost_volume(), Return a tensor with 1 if x<=y and 0 otherwise.          differenciable: will us, Apply a depth-variying radial blur an images using upsampled silces of a cost vo, Apply a depth-variying radial blur to an image, given its disparity, using a dis (+2 more)

### Community 571 - "Community 571"
Cohesion: 0.24
Nodes (6): FairseqMultiModel, base_multilingual_architecture(), build_model(), multilingual_transformer_iwslt_de_en(), MultilingualTransformerModel, Train Transformer models for multiple language pairs simultaneously.      Requir

### Community 572 - "Community 572"
Cohesion: 0.24
Nodes (4): BatchNormDim1Swap, GenericMLP, x: HW x N x C         permute to N x C x HW         Apply BN on C         permut, Used for nn.Transformer that uses a HW x N x C rep

### Community 573 - "Community 573"
Cohesion: 0.24
Nodes (4): BatchNormDim1Swap, GenericMLP, x: HW x N x C         permute to N x C x HW         Apply BN on C         permut, Used for nn.Transformer that uses a HW x N x C rep

### Community 574 - "Community 574"
Cohesion: 0.36
Nodes (4): M(), Args:           voxel_size: side length of a voxel           clip_bound: boundar, test(), Voxelizer

### Community 575 - "Community 575"
Cohesion: 0.47
Nodes (1): TestRotationTransform

### Community 576 - "Community 576"
Cohesion: 0.33
Nodes (8): assign_instances_for_scan(), compute_averages(), evaluate(), evaluate_matches(), log_results(), make_pred_info(), print_results(), # NOTE: The prediction files must live in the root of the given prediction path.

### Community 577 - "Community 577"
Cohesion: 0.33
Nodes (8): assign_instances_for_scan(), compute_averages(), evaluate(), evaluate_matches(), log_results(), make_pred_info(), print_results(), # NOTE: The prediction files must live in the root of the given prediction path.

### Community 578 - "Community 578"
Cohesion: 0.27
Nodes (9): describe_element(), header_properties(), parse_header(), parse_mesh_header(), Read ".ply" files      Parameters     ----------     filename : string         t, Write ".ply" files      Parameters     ----------     filename : string, Takes the columns of the dataframe and builds a ply-like description      Parame, read_ply() (+1 more)

### Community 579 - "Community 579"
Cohesion: 0.36
Nodes (9): _assistant(), Regression test for doom-loop false-positive on legitimate polling.  Reproduces, If the same poll returns the same number, the job is genuinely     stuck and the, If three identical calls have no tool results (e.g. all cancelled     or errored, test_different_args_does_not_fire(), test_identical_calls_with_no_results_yet_still_fires(), test_polling_with_progressing_results_does_not_fire(), test_truly_stuck_polling_with_identical_results_still_fires() (+1 more)

### Community 580 - "Community 580"
Cohesion: 0.22
Nodes (9): is_local_model_id(), is_reserved_local_model_id(), local_model_name(), local_model_provider(), Helpers for CLI local OpenAI-compatible model ids., Return provider config for a local model id, if it uses a local prefix., Return the backend model name with the local provider prefix removed., Return True for non-empty, whitespace-free local model ids. (+1 more)

### Community 581 - "Community 581"
Cohesion: 0.27
Nodes (3): EntropyScore, LogitsScoreFunction, MSPScore

### Community 582 - "Community 582"
Cohesion: 0.29
Nodes (5): find_span(), find_token(), get_detokenizer(), get_spacy_nlp(), jsonl_iterator()

### Community 583 - "Community 583"
Cohesion: 0.29
Nodes (9): check_checkpoints(), check_dataset(), check_disk_space(), check_python_deps(), main(), Check available disk space., Verify Cityscapes dataset structure., Verify required model checkpoints exist. (+1 more)

### Community 584 - "Community 584"
Cohesion: 0.42
Nodes (8): greedy_oracle_recall(), load_proposals(), main(), parse_args(), percentiles(), proposal_stems(), resize_masks(), score_recall()

### Community 585 - "Community 585"
Cohesion: 0.42
Nodes (8): _best_iou_to_gt(), _candidate_stems(), _gt_for_stem(), _load_candidate_stack(), main(), parse_args(), _parse_candidate(), _quantiles()

### Community 586 - "Community 586"
Cohesion: 0.25
Nodes (4): ClevrTex, DatasetBuilder for CLEVRTex dataset., Returns the dataset metadata., Returns SplitGenerators.

### Community 587 - "Community 587"
Cohesion: 0.22
Nodes (0): 

### Community 588 - "Community 588"
Cohesion: 0.39
Nodes (8): adjust_video_length(), concatenate_videos(), load_all_lables(), main(), merge_sequential_videos_with_same_labels(), Adjust the length of the video array to match the target length.          Parame, read_a_pkl(), save_as_pkl()

### Community 589 - "Community 589"
Cohesion: 0.42
Nodes (8): _get_datamodule(), test_chairs(), test_chairs2(), test_hd1k(), test_kitti(), test_sintel(), test_things(), test_things_subset()

### Community 590 - "Community 590"
Cohesion: 0.39
Nodes (8): _load(), Smoke tests for backend/kpis_scheduler.py.  Exercise the pure / fast paths only:, test_backfill_calls_run_hour_for_each_hour(), test_load_build_kpis_exposes_run_for_hour(), test_shutdown_is_no_op_when_not_started(), test_start_is_no_op_when_disabled(), test_start_skips_cleanly_without_apscheduler(), test_token_resolution_order()

### Community 591 - "Community 591"
Cohesion: 0.28
Nodes (4): _content_block(), _FakeResponse, test_web_search_extracts_duckduckgo_results_and_filters_domains(), test_web_search_generic_fallback_dedupes_and_rejects_bad_base_url()

### Community 592 - "Community 592"
Cohesion: 0.44
Nodes (8): _make_cm(), Regression tests for `_patch_dangling_tool_calls`.  Reproduces the failure mode, Two-turn history where the FIRST turn was interrupted.      Old patcher stopped, test_multiple_dangling_tool_calls_in_one_assistant_message_are_all_patched(), test_no_orphan_means_no_stub(), test_orphan_in_earlier_turn_still_gets_patched(), test_orphan_tool_use_followed_by_user_message_is_patched(), _tool_call()

### Community 593 - "Community 593"
Cohesion: 0.36
Nodes (7): make_dir(), make_directories(), save_tmp_images(), train(), truncateAndSave(), writeLossDataToFile(), writeMetaDataToJSON()

### Community 594 - "Community 594"
Cohesion: 0.31
Nodes (8): evaluate_naive_panoptic(), load_config(), main(), naive_panoptic_merge(), Evaluate naive panoptic baseline., # TODO: Implement checkpoint loading, Load config with default fallback., Naive panoptic assembly from semantic + instance outputs.          Args:

### Community 595 - "Community 595"
Cohesion: 0.28
Nodes (8): arrow(), main(), draw_semantic_pipeline_dataflow.py  Data flow diagram for the semantic pseudo-la, Thin coloured bar as section divider., Draw a rounded rectangle with text., Draw a downward arrow between two boxes., rounded_box(), section_bar()

### Community 596 - "Community 596"
Cohesion: 0.22
Nodes (8): get_acknowledgements_html(), get_description_html(), get_gradio_theme(), get_header_html(), Generate the main header HTML with logo and title.      Args:         logo_base6, Generate the main description and getting started HTML.      Returns:         st, Generate the acknowledgements section HTML.      Returns:         str: HTML stri, Get the configured Gradio theme with adaptive tech colors.      Returns:

### Community 597 - "Community 597"
Cohesion: 0.44
Nodes (8): copy_flat(), copy_tree(), main(), _progress(), Run the notebook's own build_index on the staged dir (the exact Kaggle-layout pa, repack_features(), verify_layout(), write_metadata()

### Community 598 - "Community 598"
Cohesion: 0.43
Nodes (6): build_one(), command_for(), main(), octave_source(), patch_header(), prepare_header_patches()

### Community 599 - "Community 599"
Cohesion: 0.39
Nodes (5): add_plot_parser(), add_time_parser(), load_json_logs(), main(), parse_args()

### Community 600 - "Community 600"
Cohesion: 0.29
Nodes (0): 

### Community 601 - "Community 601"
Cohesion: 0.32
Nodes (2): gen_uniform_unsigned_long(), gen_uniform_unsigned_long_long()

### Community 602 - "Community 602"
Cohesion: 0.36
Nodes (7): get_metadata(), load_mapillary_vistas_panoptic_json(), Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", # TODO: currently we assume image and label has the same filename but, Register a "standard" version of ADE20k panoptic segmentation dataset named `nam, register_all_mapillary_vistas_panoptic(), register_mapillary_vistas_panoptic()

### Community 603 - "Community 603"
Cohesion: 0.36
Nodes (7): get_metadata(), load_ade20k_panoptic_json(), Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", # TODO: currently we assume image and label has the same filename but, Register a "standard" version of ADE20k panoptic segmentation dataset named `nam, register_ade20k_panoptic(), register_all_ade20k_panoptic()

### Community 604 - "Community 604"
Cohesion: 0.29
Nodes (0): 

### Community 605 - "Community 605"
Cohesion: 0.25
Nodes (1): TestEventWriter

### Community 606 - "Community 606"
Cohesion: 0.43
Nodes (6): cart2hom(), get_relative_pose(), load_velo_scan(), main(), remove_center(), transform_points()

### Community 607 - "Community 607"
Cohesion: 0.29
Nodes (3): MinkResNet, r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets     <https://arx, Forward pass of ResNet.          Args:             x (ME.SparseTensor): Input sp

### Community 608 - "Community 608"
Cohesion: 0.25
Nodes (1): TestLoad

### Community 609 - "Community 609"
Cohesion: 0.32
Nodes (6): events(), merge_event_dataframes(), MOTAccumulatorCustom, new_event_dataframe(), new_event_dataframe_with_data(), nuScenes dev-kit. Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbo

### Community 610 - "Community 610"
Cohesion: 0.36
Nodes (4): bulmaSlider(), _classCallCheck(), EventEmitter(), _possibleConstructorReturn()

### Community 611 - "Community 611"
Cohesion: 0.25
Nodes (6): extract_frames_from_task(), get_video_length(), Get the length of a video in seconds.          :param video_path: Path to the, Extract frames from a task between start_time and stop_time at a specified frequ, Save task data (frames, tool presence, clip length) as a .pkl file.          :, save_task_as_pkl()

### Community 612 - "Community 612"
Cohesion: 0.32
Nodes (3): Attention, Block, # NOTE: drop path for stochastic depth, we shall see if this is better than drop

### Community 613 - "Community 613"
Cohesion: 0.29
Nodes (2): DefaultSetter, load_json_config()

### Community 614 - "Community 614"
Cohesion: 0.25
Nodes (0): 

### Community 615 - "Community 615"
Cohesion: 0.25
Nodes (0): 

### Community 616 - "Community 616"
Cohesion: 0.29
Nodes (5): get_1d_sincos_pos_embed(), get_1d_sincos_pos_embed_from_grid(), ProprioceptiveEmbedding, emb_dim: output dimension for each position     pos: a list of positions to be e, emb_dim: output dimension for each position     grid_size: int of the grid lengt

### Community 617 - "Community 617"
Cohesion: 0.36
Nodes (6): get_clean_sentences(), ignore_sentence(), main(), ignore sentences with the following patterns     '! scope="col" and '! scope="r, remove lines with <doc>, </doc> and titles (without [[]])     there are several, replace_symbols()

### Community 618 - "Community 618"
Cohesion: 0.43
Nodes (6): check_train_all(), check_train_sentences(), get_all_test_data(), load_sentences(), main(), swap_direction()

### Community 619 - "Community 619"
Cohesion: 0.32
Nodes (4): knnGPU_sharded(), load_batch(), score(), score_candidates()

### Community 620 - "Community 620"
Cohesion: 0.39
Nodes (7): create_readme(), load_checkpoint(), main(), Load completed steps from checkpoint file., Retry a function with exponential backoff., retry(), save_checkpoint()

### Community 621 - "Community 621"
Cohesion: 0.36
Nodes (7): colorize_instances(), colorize_semantic(), depth_guided_instances(), main(), Map trainID semantic map to RGB., Color each instance with a unique random color, stuff in gray., Split thing regions using depth gradient edges.

### Community 622 - "Community 622"
Cohesion: 0.46
Nodes (7): atomic_write(), convert_hf_path(), export_shard(), iter_train_shards(), load_status(), main(), save_status()

### Community 623 - "Community 623"
Cohesion: 0.52
Nodes (6): extract_objectness(), load_or_extract_features(), load_proposals(), main(), parse_args(), resize_masks()

### Community 624 - "Community 624"
Cohesion: 0.57
Nodes (6): _area_frac(), main(), _paired_iou(), parse_args(), _summarize(), _write_fallback_bank()

### Community 625 - "Community 625"
Cohesion: 0.52
Nodes (6): _build_split(), _image_index(), main(), _mask_shapes(), _materialize_images(), parse_args()

### Community 626 - "Community 626"
Cohesion: 0.52
Nodes (6): main(), _mask_area(), _mean_numeric(), _pair_rows(), parse_args(), _score_threshold_sweep()

### Community 627 - "Community 627"
Cohesion: 0.43
Nodes (6): main(), parse_rec(), Parse a PASCAL VOC xml file, ap = voc_ap(rec, prec, [use_07_metric])     Compute VOC AP given precision and r, voc_ap(), voc_eval()

### Community 628 - "Community 628"
Cohesion: 0.43
Nodes (6): get_metadata(), load_coco_panoptic_json(), # TODO: currently we assume image and label has the same filename but, Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", register_all_coco_panoptic_annos_sem_seg(), register_coco_panoptic_annos_sem_seg()

### Community 629 - "Community 629"
Cohesion: 0.29
Nodes (0): 

### Community 630 - "Community 630"
Cohesion: 0.33
Nodes (4): DensePoseCheckpointer, Same as :class:`DetectionCheckpointer`, but is able to handle HRNet weights, _rename_HRNet_weights(), DetectionCheckpointer

### Community 631 - "Community 631"
Cohesion: 0.48
Nodes (3): coco_test_fun(), lvis_test_fun(), TestDatasetLoadedAnnotations

### Community 632 - "Community 632"
Cohesion: 0.29
Nodes (2): PicklableWrapper, Wrap an object to make it more picklable, note that it uses     heavy weight ser

### Community 633 - "Community 633"
Cohesion: 0.43
Nodes (5): gen_gt_labels(), in_hull(), main(), range_cutoff(), :param p: (N, K) test points     :param hull: (M, K) M corners of a box     :ret

### Community 634 - "Community 634"
Cohesion: 0.48
Nodes (6): closeness_rectangle(), fit_2d_box_modest(), minimum_bounding_rectangle(), PCA_rectangle(), Find the smallest bounding rectangle for a set of points.     Returns a set of p, variance_rectangle()

### Community 635 - "Community 635"
Cohesion: 0.38
Nodes (6): calc_mean_best_overlap(), compute_iou_matrix(), mean_best_overlap_single_sample(), Compute the Mean Best Overlap (MBO) for a single sample between ground truth and, Compute the Intersection over Union (IoU) matrix between ground truth and predic, Calculate the Mean Best Overlap (MBO) for a batch of ground truth and predicted

### Community 636 - "Community 636"
Cohesion: 0.48
Nodes (6): box_cut(), la_sampling(), random_drop_out(), input:         box: array, shape=(7,)  (x, y, z, l, w, h, yaw)         cloud: ar, remove_past(), to_sphere_coords()

### Community 637 - "Community 637"
Cohesion: 0.43
Nodes (6): get_metadata(), load_coco_panoptic_json(), # TODO: currently we assume image and label has the same filename but, Args:         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017", register_all_coco_panoptic_annos_sem_seg(), register_coco_panoptic_annos_sem_seg()

### Community 638 - "Community 638"
Cohesion: 0.48
Nodes (6): _malformed_tool_msg(), Regression test for the malformed-JSON loop in observatory session 7750e82f (202, test_one_malformed_does_not_trigger(), test_streak_broken_by_successful_tool_call_does_not_trigger(), test_two_consecutive_malformed_same_tool_triggers(), test_two_malformed_different_tools_does_not_trigger()

### Community 639 - "Community 639"
Cohesion: 0.33
Nodes (2): FakeToolRouter, test_no_tool_response_retries_when_plan_is_incomplete()

### Community 640 - "Community 640"
Cohesion: 0.38
Nodes (5): NotificationProvider, _format_slack_mrkdwn(), _format_text(), Convert common Markdown constructs to Slack's mrkdwn syntax., SlackProvider

### Community 641 - "Community 641"
Cohesion: 0.33
Nodes (6): datacli(), get_dataclass_params(), make_parser(), Create an argument parser from a dataclass., Parse command line arguments into a 'cls' object., Extract fields that interest us from dataclass

### Community 642 - "Community 642"
Cohesion: 0.29
Nodes (0): 

### Community 643 - "Community 643"
Cohesion: 0.33
Nodes (6): _encode_gif(), gif_summary(), _py_gif_summary(), Outputs a `Summary` protocol buffer with gif animations.   Args:     name: Name, Encodes numpy images into gif string.   Args:     images: A 5-D `uint8` `np.arra, Outputs a `Summary` protocol buffer with gif animations.   Args:     tag: Name o

### Community 644 - "Community 644"
Cohesion: 0.29
Nodes (1): CycleGAN

### Community 645 - "Community 645"
Cohesion: 0.38
Nodes (1): TestIterators

### Community 646 - "Community 646"
Cohesion: 0.38
Nodes (5): IndexError, main(), OOVIndexError, Replaces <unk-N> tokens in the target text with the corresponding word in     th, replace_oovs()

### Community 647 - "Community 647"
Cohesion: 0.38
Nodes (6): pack_replabels(), Replabel symbols used in flashlight, currently just "1", "2", ...     This preve, Pack a token sequence so that repeated symbols are replaced by replabels, Unpack a token sequence so that replabels are replaced by repeated symbols, replabel_symbol(), unpack_replabels()

### Community 648 - "Community 648"
Cohesion: 0.43
Nodes (5): binarize_(), call(), call_output(), encode_spm(), get_data_size()

### Community 649 - "Community 649"
Cohesion: 0.38
Nodes (5): get_examples(), InputExample, main(), Extract paragraph and question-answer list from each json file, Helper script to extract paragraphs questions and answers from RACE datasets.

### Community 650 - "Community 650"
Cohesion: 0.48
Nodes (4): test_repeat_factor_sampler_computes_freq_and_oversamples_rare_images(), test_repeat_factor_sampler_ddp_sharding_has_no_overlap_for_single_epoch(), ToyDataset, _write_semantic()

### Community 651 - "Community 651"
Cohesion: 0.48
Nodes (4): selective_scan_bwd(), selective_scan_fwd(), set_ssm_params_bwd(), set_ssm_params_fwd()

### Community 652 - "Community 652"
Cohesion: 0.6
Nodes (5): gt_paths_from_args(), main(), parse_args(), percentiles(), resize_bool()

### Community 653 - "Community 653"
Cohesion: 0.4
Nodes (0): 

### Community 654 - "Community 654"
Cohesion: 0.4
Nodes (5): create_annotation_info(), create_image_info(), Return annotation info in COCO style     Args:         annotation_id: the annota, Return image_info in COCO style     Args:         image_id: the image ID, resize_binary_mask()

### Community 655 - "Community 655"
Cohesion: 0.47
Nodes (3): clean_mesh(), do_felzenszwalb(), main()

### Community 656 - "Community 656"
Cohesion: 0.33
Nodes (0): 

### Community 657 - "Community 657"
Cohesion: 0.53
Nodes (4): get_bbox(), get_lidar(), get_linemarks(), showvelo()

### Community 658 - "Community 658"
Cohesion: 0.33
Nodes (3): BasicBlock2D, Applies convolutional block         Args:             features: (B, C_in, H, W),, Initializes convolutional block         Args:             in_channels: int, Numb

### Community 659 - "Community 659"
Cohesion: 0.33
Nodes (0): 

### Community 660 - "Community 660"
Cohesion: 0.33
Nodes (1): TestPhysicsBaselines

### Community 661 - "Community 661"
Cohesion: 0.33
Nodes (3): Test valid and invalid inputs for quaternion_yaw()., Test the box.in_box method., TestGeometryUtils

### Community 662 - "Community 662"
Cohesion: 0.4
Nodes (4): dataset(), feature_descriptions(), Create a dictionary desc   ribing the dataset features.      Args:       max_num, Read, decompress, and parse the TFRecords file.      Args:       tfrecords_path:

### Community 663 - "Community 663"
Cohesion: 0.6
Nodes (5): compute_ephe_score(), compute_ppscore(), count_neighbors(), points_rigid_transform(), save_pp_score()

### Community 664 - "Community 664"
Cohesion: 0.33
Nodes (0): 

### Community 665 - "Community 665"
Cohesion: 0.6
Nodes (5): create_mask(), decode_json(), get_specific_frame(), main(), save_decoded_images()

### Community 666 - "Community 666"
Cohesion: 0.33
Nodes (0): 

### Community 667 - "Community 667"
Cohesion: 0.33
Nodes (0): 

### Community 668 - "Community 668"
Cohesion: 0.33
Nodes (0): 

### Community 669 - "Community 669"
Cohesion: 0.4
Nodes (4): json_to_args(), parse_args(), Parse command line arguments and merge them with JSON configuration.      This f, Convert JSON configuration file to argparse.Namespace object.      Parameters

### Community 670 - "Community 670"
Cohesion: 0.33
Nodes (0): 

### Community 671 - "Community 671"
Cohesion: 0.47
Nodes (4): _FakeConfig, _mk_session(), Heartbeat + stable-local-path tests for Session.  We don't spin up the real agen, test_stable_local_path_overwrites()

### Community 672 - "Community 672"
Cohesion: 0.6
Nodes (5): _load(), Smoke test for the SFT reshape — raw passthrough with tags attached., _session_row(), test_reshape_handles_missing_tools_field(), test_reshape_preserves_messages_and_tools_and_adds_tags()

### Community 673 - "Community 673"
Cohesion: 0.47
Nodes (5): is_sandbox_fork(), log(), main(), JSON Lines log so downstream tooling can grep / parse., Filter: matches the ml-intern sandbox naming pattern.      NOTE: We initially tr

### Community 674 - "Community 674"
Cohesion: 0.33
Nodes (0): 

### Community 675 - "Community 675"
Cohesion: 0.47
Nodes (5): _do_download(), _get_urls(), prepare_dataset(), Get the ulrs of the imaegs to downlaod per category.     This has a lot of hard, Download the images from the urls.     Uses one thread per category. By default

### Community 676 - "Community 676"
Cohesion: 0.47
Nodes (1): TestResamplingDataset

### Community 677 - "Community 677"
Cohesion: 0.47
Nodes (5): download_file(), login(), main(), Login to Cityscapes and return authenticated session., Download a single Cityscapes package.

### Community 678 - "Community 678"
Cohesion: 0.47
Nodes (5): compute_frequencies(), load_panoptic_labels(), main(), Load all panoptic PNG labels from the given directory., Compute normalized per-class pixel frequencies from panoptic labels.      Args:

### Community 679 - "Community 679"
Cohesion: 0.47
Nodes (5): link(), main(), parse_args(), Assemble a Stage-2-ready CUPS-format pseudo-label cache from two pieces:    - Se, idempotent symlink: skip if already correct, replace if wrong, create otherwise.

### Community 680 - "Community 680"
Cohesion: 0.7
Nodes (4): load_mcg_masks(), main(), parse_args(), stem_from_mcg()

### Community 681 - "Community 681"
Cohesion: 0.6
Nodes (3): analyze_results(), main(), makeplot()

### Community 682 - "Community 682"
Cohesion: 0.4
Nodes (3): plot_grad_flow_v2(), https://github.com/alwynmathew/gradflow-check, Plots the gradients flowing through different layers in the net during training.

### Community 683 - "Community 683"
Cohesion: 0.5
Nodes (1): HPNLearner

### Community 684 - "Community 684"
Cohesion: 0.4
Nodes (2): TestCollectEnv, TestProjects

### Community 685 - "Community 685"
Cohesion: 0.4
Nodes (4): default_coco_scheduler(), default_X_scheduler(), Returns the config for a default multi-step LR scheduler such as "50epochs",, Returns the config for a default multi-step LR scheduler such as "1x", "3x",

### Community 686 - "Community 686"
Cohesion: 0.4
Nodes (4): create_dummy_class(), create_dummy_func(), When a dependency of a function is not available, create a dummy function which, When a dependency of a class is not available, create a dummy class which throws

### Community 687 - "Community 687"
Cohesion: 0.7
Nodes (4): convert_obj(), display_args(), eprint(), main()

### Community 688 - "Community 688"
Cohesion: 0.5
Nodes (3): get_layer_mocks(), test_make_rasterization(), TestStaticLayerRasterizer

### Community 689 - "Community 689"
Cohesion: 0.4
Nodes (4): load_bin_file(), panoptic_to_lidarseg(), Convert panoptic label array to lidarseg label array     :param panoptic_labels:, Loads a .bin file containing the lidarseg or lidar panoptic labels.     :param b

### Community 690 - "Community 690"
Cohesion: 0.4
Nodes (4): find_checkpoint(), get_commandline_config_path(), Find checkpoint in output path of previous run., Get the path of a config path specified on the command line.

### Community 691 - "Community 691"
Cohesion: 0.4
Nodes (1): Model

### Community 692 - "Community 692"
Cohesion: 0.5
Nodes (1): OYSTER

### Community 693 - "Community 693"
Cohesion: 0.5
Nodes (1): MFCF

### Community 694 - "Community 694"
Cohesion: 0.4
Nodes (4): extend_video_and_masks(), load_pkl(), Load PKL file and return video frames and masks., Extend video stack to 30 FPS and align masks correctly.

### Community 695 - "Community 695"
Cohesion: 0.8
Nodes (4): em(), fit(), fit_and_likelihood(), likelihood()

### Community 696 - "Community 696"
Cohesion: 0.5
Nodes (2): calc_inlier_hard(), calc_inlier_soft()

### Community 697 - "Community 697"
Cohesion: 0.5
Nodes (4): get_layer_id_for_vit(), param_groups_lrd(), Parameter groups for layer-wise lr decay     Following BEiT: https://github.com/, Assign a parameter with its layer id     Following BEiT: https://github.com/micr

### Community 698 - "Community 698"
Cohesion: 0.5
Nodes (2): get_mean_score_and_loss(), get_score()

### Community 699 - "Community 699"
Cohesion: 0.4
Nodes (0): 

### Community 700 - "Community 700"
Cohesion: 0.4
Nodes (0): 

### Community 701 - "Community 701"
Cohesion: 0.6
Nodes (3): check_diff(), get_directions(), main()

### Community 702 - "Community 702"
Cohesion: 0.8
Nodes (4): every_n_checkpoints(), last_n_checkpoints(), main(), parse_checkpoints()

### Community 703 - "Community 703"
Cohesion: 0.7
Nodes (4): decoder_state(), encoder_state(), find_weight_norm(), push_state()

### Community 704 - "Community 704"
Cohesion: 0.4
Nodes (4): convert_to_unicode(), Converts `text` to Unicode (if it's not already), assuming UTF-8 input., Strips accents from a piece of text., run_strip_accents()

### Community 705 - "Community 705"
Cohesion: 0.4
Nodes (2): CameraEnc, CameraHead predicts camera parameters from token representations using iterative

### Community 706 - "Community 706"
Cohesion: 0.5
Nodes (4): main(), Merge DBLP-fetched raw .bib files into the paper bibliography.  Reads raw files, Return (our_key, cleaned_entry) for one raw block., transform()

### Community 707 - "Community 707"
Cohesion: 0.7
Nodes (4): latest_progress(), main(), read_tail(), render_bar()

### Community 708 - "Community 708"
Cohesion: 0.6
Nodes (4): checkpoint_cell(), code(), md(), Build a step-by-step supplementary visualization notebook.  The generated notebo

### Community 709 - "Community 709"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Launch official SOLO adaptive self-training with the paper's L_ad head.

### Community 710 - "Community 710"
Cohesion: 1.0
Nodes (3): _cmake_build(), main(), _manual_build()

### Community 711 - "Community 711"
Cohesion: 0.83
Nodes (3): randomMT(), reloadMT(), seedMT()

### Community 712 - "Community 712"
Cohesion: 0.5
Nodes (0): 

### Community 713 - "Community 713"
Cohesion: 0.5
Nodes (3): get_config(), Get the default hyperparameter configuration., # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),

### Community 714 - "Community 714"
Cohesion: 0.5
Nodes (3): get_config(), Get the default hyperparameter configuration., # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),

### Community 715 - "Community 715"
Cohesion: 0.5
Nodes (3): get_config(), Get the default hyperparameter configuration., # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),

### Community 716 - "Community 716"
Cohesion: 0.5
Nodes (3): get_config(), Get the default hyperparameter configuration., # NOTE: MOVi-A, MOVi-B, and MOVi-C only contain up to 10 instances (objects),

### Community 717 - "Community 717"
Cohesion: 0.5
Nodes (2): Create semantic segmentation annotations from panoptic segmentation     annotati, separate_coco_semantic_from_panoptic()

### Community 718 - "Community 718"
Cohesion: 0.83
Nodes (3): main(), parse_args(), pseudo_for_split()

### Community 719 - "Community 719"
Cohesion: 0.5
Nodes (2): Create semantic segmentation annotations from panoptic segmentation     annotati, separate_coco_semantic_from_panoptic()

### Community 720 - "Community 720"
Cohesion: 0.5
Nodes (2): Create semantic segmentation annotations from panoptic segmentation     annotati, separate_coco_semantic_from_panoptic()

### Community 721 - "Community 721"
Cohesion: 0.5
Nodes (1): TestOptimizer

### Community 722 - "Community 722"
Cohesion: 0.5
Nodes (1): TestTensorboardXWriter

### Community 723 - "Community 723"
Cohesion: 0.5
Nodes (1): TestMMDetWrapper

### Community 724 - "Community 724"
Cohesion: 0.83
Nodes (3): display_args(), eprint(), main()

### Community 725 - "Community 725"
Cohesion: 0.83
Nodes (3): display_args(), eprint(), main()

### Community 726 - "Community 726"
Cohesion: 0.5
Nodes (0): 

### Community 727 - "Community 727"
Cohesion: 0.67
Nodes (2): main(), parse_result()

### Community 728 - "Community 728"
Cohesion: 0.5
Nodes (2): mmdet3d2torchserve(), Converts MMDetection3D model (config + checkpoint) to TorchServe `.mar`.      Ar

### Community 729 - "Community 729"
Cohesion: 0.5
Nodes (3): create_indoor_info_file(), Create indoor information file.      Get information of the raw data and save it, # TODO: do we need to generate on val set?

### Community 730 - "Community 730"
Cohesion: 0.5
Nodes (0): 

### Community 731 - "Community 731"
Cohesion: 0.83
Nodes (3): find_cols(), find_jumps(), kitti_pcl_projection_get_rows_cols()

### Community 732 - "Community 732"
Cohesion: 0.5
Nodes (2): Convert a in the experiment folder to a valid setting for experiment., _remove_filename_components()

### Community 733 - "Community 733"
Cohesion: 0.83
Nodes (3): _fmt_overrides(), _is_metric_conf(), main()

### Community 734 - "Community 734"
Cohesion: 0.5
Nodes (2): dataset(), Read, decompress, and parse the TFRecords file.    Args:     tfrecords_path: str

### Community 735 - "Community 735"
Cohesion: 0.83
Nodes (3): convert_dataset(), encode_mask(), read_sequence_folder()

### Community 736 - "Community 736"
Cohesion: 0.83
Nodes (3): load_all_lables(), main(), merge_sequential_videos_with_same_labels()

### Community 737 - "Community 737"
Cohesion: 0.83
Nodes (3): filter_object_point_gtflow(), generate_bbox(), main()

### Community 738 - "Community 738"
Cohesion: 0.83
Nodes (3): extract_segment_frontcamera(), get_camera_labels(), main()

### Community 739 - "Community 739"
Cohesion: 0.5
Nodes (0): 

### Community 740 - "Community 740"
Cohesion: 0.67
Nodes (3): main(), map_label(), Cityscapes labelId -> stored 27-class label (class+1, 0=ignore).

### Community 741 - "Community 741"
Cohesion: 0.67
Nodes (3): Opt-in live sandbox communication test.  This test creates a real private Huggin, _skip_without_live_sandbox(), test_live_sandbox_authenticated_agent_communication()

### Community 742 - "Community 742"
Cohesion: 0.5
Nodes (3): check_training_script_save_pattern(), Reliability checks for job submissions and other operations, Check if a training script properly saves models.

### Community 743 - "Community 743"
Cohesion: 0.83
Nodes (3): get_classifier(), get_kwargs_for_mahalanobis_score(), main()

### Community 744 - "Community 744"
Cohesion: 0.5
Nodes (1): MoviesSpider

### Community 745 - "Community 745"
Cohesion: 0.5
Nodes (0): 

### Community 746 - "Community 746"
Cohesion: 0.67
Nodes (3): heuristics(), main(), update continuous identical labels to obey BIO rules,      1) the first should b

### Community 747 - "Community 747"
Cohesion: 0.67
Nodes (3): is_normal_hyperlink(), three cases: (1) remove [[...]] e.g. [[Category:...]] or [[:Category:...]], replaceInternalLinksCustomised()

### Community 748 - "Community 748"
Cohesion: 0.83
Nodes (3): get_args(), load_sop(), main()

### Community 749 - "Community 749"
Cohesion: 0.5
Nodes (1): TestIOPath

### Community 750 - "Community 750"
Cohesion: 0.67
Nodes (3): convert_yaml_to_tuple(), parse_config_yaml(), Converts a yaml dictionary with two keys: `key` and `value` into a two     argum

### Community 751 - "Community 751"
Cohesion: 0.67
Nodes (2): byte_decode(), smart_byte_decode()

### Community 752 - "Community 752"
Cohesion: 0.83
Nodes (3): dedup(), existing_data(), main()

### Community 753 - "Community 753"
Cohesion: 0.83
Nodes (3): compute_accuracy(), compute_dist(), load_embeddings()

### Community 754 - "Community 754"
Cohesion: 0.67
Nodes (3): main(), Run full train-set label-free tau/A_min sweeps.  This is a thin orchestrator ove, run_step()

### Community 755 - "Community 755"
Cohesion: 0.67
Nodes (3): get_gcs_stems(), main(), Get set of depth map stems already on GCS.

### Community 756 - "Community 756"
Cohesion: 0.67
Nodes (3): check(), main(), Run a check and print pass/fail.

### Community 757 - "Community 757"
Cohesion: 0.67
Nodes (3): main(), SIMCF threshold sensitivity sweep on full Cityscapes train set.  Runs SIMCF with, run()

### Community 758 - "Community 758"
Cohesion: 0.67
Nodes (3): main(), Run one fragmentation configuration., run_config()

### Community 759 - "Community 759"
Cohesion: 0.67
Nodes (3): evaluate_one(), main(), Run evaluation on one pseudo-label directory.

### Community 760 - "Community 760"
Cohesion: 0.67
Nodes (3): main(), organize(), Create city subdirectories and symlinks for evaluation.

### Community 761 - "Community 761"
Cohesion: 0.67
Nodes (2): load_custom_callable(), _load_modules_from_dir()

### Community 762 - "Community 762"
Cohesion: 0.5
Nodes (1): MonodepthOptions

### Community 763 - "Community 763"
Cohesion: 0.67
Nodes (3): fn_kv_csv(), parse_scalar(), Parse a string of comma-separated triplets: fn:key:value      Returns:         d

### Community 764 - "Community 764"
Cohesion: 0.67
Nodes (3): main(), parse(), Assemble the unified 19-class Hungarian PQ comparison table (Group A + Group B).

### Community 765 - "Community 765"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 766 - "Community 766"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 767 - "Community 767"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 768 - "Community 768"
Cohesion: 1.0
Nodes (2): _get_ade20k_full_meta(), register_all_ade20k_full()

### Community 769 - "Community 769"
Cohesion: 1.0
Nodes (2): _get_mapillary_vistas_meta(), register_all_mapillary_vistas()

### Community 770 - "Community 770"
Cohesion: 1.0
Nodes (2): _get_coco_stuff_meta(), register_all_coco_stuff_10k()

### Community 771 - "Community 771"
Cohesion: 0.67
Nodes (0): 

### Community 772 - "Community 772"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 773 - "Community 773"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 774 - "Community 774"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 775 - "Community 775"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 776 - "Community 776"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 777 - "Community 777"
Cohesion: 1.0
Nodes (2): main(), register_all_benchmark_splits()

### Community 778 - "Community 778"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 779 - "Community 779"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 780 - "Community 780"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 781 - "Community 781"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 782 - "Community 782"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 783 - "Community 783"
Cohesion: 1.0
Nodes (2): gen_seeds(), subsample_idx()

### Community 784 - "Community 784"
Cohesion: 0.67
Nodes (2): get_config(), Get the default hyperparameter configuration.

### Community 785 - "Community 785"
Cohesion: 0.67
Nodes (2): get_config(), Get the default hyperparameter configuration.

### Community 786 - "Community 786"
Cohesion: 0.67
Nodes (2): get_config(), Get the default hyperparameter configuration.

### Community 787 - "Community 787"
Cohesion: 0.67
Nodes (0): 

### Community 788 - "Community 788"
Cohesion: 0.67
Nodes (1): TestStructures

### Community 789 - "Community 789"
Cohesion: 0.67
Nodes (2): cocofy_lvis(), Filter LVIS instance segmentation annotations to remove all categories that are

### Community 790 - "Community 790"
Cohesion: 0.67
Nodes (1): TestCaffe2RPN

### Community 791 - "Community 791"
Cohesion: 0.67
Nodes (0): 

### Community 792 - "Community 792"
Cohesion: 0.67
Nodes (0): 

### Community 793 - "Community 793"
Cohesion: 0.67
Nodes (0): 

### Community 794 - "Community 794"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 795 - "Community 795"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 796 - "Community 796"
Cohesion: 0.67
Nodes (0): 

### Community 797 - "Community 797"
Cohesion: 0.67
Nodes (0): 

### Community 798 - "Community 798"
Cohesion: 0.67
Nodes (2): get_polynomial_decay_schedule_with_warmup(), https://github.com/huggingface/transformers/blob/225de5ccbbd392735013fb1b1b5604f

### Community 799 - "Community 799"
Cohesion: 1.0
Nodes (2): scatter_add_nd_numpy(), scatter_mean_nd_numpy()

### Community 800 - "Community 800"
Cohesion: 0.67
Nodes (1): TestRasterizer

### Community 801 - "Community 801"
Cohesion: 0.67
Nodes (1): TestPrediction

### Community 802 - "Community 802"
Cohesion: 0.67
Nodes (1): TestCase

### Community 803 - "Community 803"
Cohesion: 0.67
Nodes (2): calc_fgari_score(), Calculate Adjusted Rand Index (ARI) score for object discovery evaluation.

### Community 804 - "Community 804"
Cohesion: 0.67
Nodes (0): 

### Community 805 - "Community 805"
Cohesion: 0.67
Nodes (2): load(), Loads the specified simple game and wraps it.     Args:         game (str): name

### Community 806 - "Community 806"
Cohesion: 1.0
Nodes (2): get_prompt_templates(), prompt_engineering()

### Community 807 - "Community 807"
Cohesion: 1.0
Nodes (2): calc_batch_gradients(), calc_batch_k_gradients()

### Community 808 - "Community 808"
Cohesion: 1.0
Nodes (2): j_linkage(), j_linkage_single_step()

### Community 809 - "Community 809"
Cohesion: 0.67
Nodes (2): complete(), Given invalid points complete points by completing depth using recursive neighbo

### Community 810 - "Community 810"
Cohesion: 1.0
Nodes (2): extract_segment_frontcamera(), get_camera_labels()

### Community 811 - "Community 811"
Cohesion: 1.0
Nodes (2): extract_segment_frontcamera(), get_camera_labels()

### Community 812 - "Community 812"
Cohesion: 1.0
Nodes (2): get_coco_captions_df(), get_coco_captions_test_df()

### Community 813 - "Community 813"
Cohesion: 0.67
Nodes (1): Function taken and adapted from https://github.com/ppliuboy/SelFlow/blob/master/

### Community 814 - "Community 814"
Cohesion: 0.67
Nodes (1): @article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b

### Community 815 - "Community 815"
Cohesion: 1.0
Nodes (2): main(), map_label()

### Community 816 - "Community 816"
Cohesion: 1.0
Nodes (2): main(), map_label()

### Community 817 - "Community 817"
Cohesion: 0.67
Nodes (0): 

### Community 818 - "Community 818"
Cohesion: 1.0
Nodes (2): cli_main(), gen_and_reprocess_nbest()

### Community 819 - "Community 819"
Cohesion: 1.0
Nodes (2): cli_main(), score_lm()

### Community 820 - "Community 820"
Cohesion: 1.0
Nodes (2): cli_main(), score_bw()

### Community 821 - "Community 821"
Cohesion: 0.67
Nodes (0): 

### Community 822 - "Community 822"
Cohesion: 1.0
Nodes (2): main(), _normalize_spaces()

### Community 823 - "Community 823"
Cohesion: 0.67
Nodes (0): 

### Community 824 - "Community 824"
Cohesion: 0.67
Nodes (2): main(), Tokenizes, preserving tabs

### Community 825 - "Community 825"
Cohesion: 1.0
Nodes (2): check_data_test_bleu(), run_eval_bleu()

### Community 826 - "Community 826"
Cohesion: 1.0
Nodes (2): deup(), main()

### Community 827 - "Community 827"
Cohesion: 1.0
Nodes (2): get_parser(), main()

### Community 828 - "Community 828"
Cohesion: 0.67
Nodes (1): Download a HuggingFace Mask2Former checkpoint to ./checkpoints/.

### Community 829 - "Community 829"
Cohesion: 0.67
Nodes (1): Verify the aux-loss registry exposes all six callables.

### Community 830 - "Community 830"
Cohesion: 0.67
Nodes (1): Verify CUPS config exposes SEM_SEG_HEAD aux-loss weight keys.

### Community 831 - "Community 831"
Cohesion: 0.67
Nodes (1): Compute exact Mapillary v2 → Cityscapes-19 coverage statistics.  Outputs: - Tota

### Community 832 - "Community 832"
Cohesion: 0.67
Nodes (1): Convert k=80 pseudo-labels to CUPS flat format for training.  Reads:   - pseudo_

### Community 833 - "Community 833"
Cohesion: 0.67
Nodes (1): Create TFRecords from Dataset.  Usage:     python scripts/create_tfrecords.py \

### Community 834 - "Community 834"
Cohesion: 0.67
Nodes (0): 

### Community 835 - "Community 835"
Cohesion: 1.0
Nodes (2): get_style(), Log()

### Community 836 - "Community 836"
Cohesion: 0.67
Nodes (0): 

### Community 837 - "Community 837"
Cohesion: 0.67
Nodes (1): Sequential 19-class CPU re-eval for Group-B (mobile/BiFPN) checkpoints.  For eac

### Community 838 - "Community 838"
Cohesion: 0.67
Nodes (1): Sequential 19-class CPU re-evaluation driver for Group-A checkpoints.  Reads man

### Community 839 - "Community 839"
Cohesion: 0.67
Nodes (1): Tests for instance candidate objectness/ranking utilities.

### Community 840 - "Community 840"
Cohesion: 1.0
Nodes (0): 

### Community 841 - "Community 841"
Cohesion: 1.0
Nodes (0): 

### Community 842 - "Community 842"
Cohesion: 1.0
Nodes (0): 

### Community 843 - "Community 843"
Cohesion: 1.0
Nodes (0): 

### Community 844 - "Community 844"
Cohesion: 1.0
Nodes (0): 

### Community 845 - "Community 845"
Cohesion: 1.0
Nodes (0): 

### Community 846 - "Community 846"
Cohesion: 1.0
Nodes (0): 

### Community 847 - "Community 847"
Cohesion: 1.0
Nodes (0): 

### Community 848 - "Community 848"
Cohesion: 1.0
Nodes (0): 

### Community 849 - "Community 849"
Cohesion: 1.0
Nodes (0): 

### Community 850 - "Community 850"
Cohesion: 1.0
Nodes (0): 

### Community 851 - "Community 851"
Cohesion: 1.0
Nodes (0): 

### Community 852 - "Community 852"
Cohesion: 1.0
Nodes (0): 

### Community 853 - "Community 853"
Cohesion: 1.0
Nodes (0): 

### Community 854 - "Community 854"
Cohesion: 1.0
Nodes (0): 

### Community 855 - "Community 855"
Cohesion: 1.0
Nodes (0): 

### Community 856 - "Community 856"
Cohesion: 1.0
Nodes (0): 

### Community 857 - "Community 857"
Cohesion: 1.0
Nodes (1): Utility script to check dataset integrity.

### Community 858 - "Community 858"
Cohesion: 1.0
Nodes (0): 

### Community 859 - "Community 859"
Cohesion: 1.0
Nodes (0): 

### Community 860 - "Community 860"
Cohesion: 1.0
Nodes (0): 

### Community 861 - "Community 861"
Cohesion: 1.0
Nodes (0): 

### Community 862 - "Community 862"
Cohesion: 1.0
Nodes (0): 

### Community 863 - "Community 863"
Cohesion: 1.0
Nodes (0): 

### Community 864 - "Community 864"
Cohesion: 1.0
Nodes (0): 

### Community 865 - "Community 865"
Cohesion: 1.0
Nodes (0): 

### Community 866 - "Community 866"
Cohesion: 1.0
Nodes (0): 

### Community 867 - "Community 867"
Cohesion: 1.0
Nodes (0): 

### Community 868 - "Community 868"
Cohesion: 1.0
Nodes (0): 

### Community 869 - "Community 869"
Cohesion: 1.0
Nodes (0): 

### Community 870 - "Community 870"
Cohesion: 1.0
Nodes (0): 

### Community 871 - "Community 871"
Cohesion: 1.0
Nodes (0): 

### Community 872 - "Community 872"
Cohesion: 1.0
Nodes (0): 

### Community 873 - "Community 873"
Cohesion: 1.0
Nodes (1): Generate the code reference pages and navigation.

### Community 874 - "Community 874"
Cohesion: 1.0
Nodes (0): 

### Community 875 - "Community 875"
Cohesion: 1.0
Nodes (0): 

### Community 876 - "Community 876"
Cohesion: 1.0
Nodes (0): 

### Community 877 - "Community 877"
Cohesion: 1.0
Nodes (0): 

### Community 878 - "Community 878"
Cohesion: 1.0
Nodes (0): 

### Community 879 - "Community 879"
Cohesion: 1.0
Nodes (0): 

### Community 880 - "Community 880"
Cohesion: 1.0
Nodes (0): 

### Community 881 - "Community 881"
Cohesion: 1.0
Nodes (0): 

### Community 882 - "Community 882"
Cohesion: 1.0
Nodes (0): 

### Community 883 - "Community 883"
Cohesion: 1.0
Nodes (0): 

### Community 884 - "Community 884"
Cohesion: 1.0
Nodes (0): 

### Community 885 - "Community 885"
Cohesion: 1.0
Nodes (0): 

### Community 886 - "Community 886"
Cohesion: 1.0
Nodes (0): 

### Community 887 - "Community 887"
Cohesion: 1.0
Nodes (0): 

### Community 888 - "Community 888"
Cohesion: 1.0
Nodes (0): 

### Community 889 - "Community 889"
Cohesion: 1.0
Nodes (0): 

### Community 890 - "Community 890"
Cohesion: 1.0
Nodes (0): 

### Community 891 - "Community 891"
Cohesion: 1.0
Nodes (0): 

### Community 892 - "Community 892"
Cohesion: 1.0
Nodes (0): 

### Community 893 - "Community 893"
Cohesion: 1.0
Nodes (0): 

### Community 894 - "Community 894"
Cohesion: 1.0
Nodes (0): 

### Community 895 - "Community 895"
Cohesion: 1.0
Nodes (0): 

### Community 896 - "Community 896"
Cohesion: 1.0
Nodes (0): 

### Community 897 - "Community 897"
Cohesion: 1.0
Nodes (1): @article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b

### Community 898 - "Community 898"
Cohesion: 1.0
Nodes (0): 

### Community 899 - "Community 899"
Cohesion: 1.0
Nodes (0): 

### Community 900 - "Community 900"
Cohesion: 1.0
Nodes (0): 

### Community 901 - "Community 901"
Cohesion: 1.0
Nodes (0): 

### Community 902 - "Community 902"
Cohesion: 1.0
Nodes (0): 

### Community 903 - "Community 903"
Cohesion: 1.0
Nodes (0): 

### Community 904 - "Community 904"
Cohesion: 1.0
Nodes (0): 

### Community 905 - "Community 905"
Cohesion: 1.0
Nodes (0): 

### Community 906 - "Community 906"
Cohesion: 1.0
Nodes (0): 

### Community 907 - "Community 907"
Cohesion: 1.0
Nodes (0): 

### Community 908 - "Community 908"
Cohesion: 1.0
Nodes (0): 

### Community 909 - "Community 909"
Cohesion: 1.0
Nodes (0): 

### Community 910 - "Community 910"
Cohesion: 1.0
Nodes (0): 

### Community 911 - "Community 911"
Cohesion: 1.0
Nodes (0): 

### Community 912 - "Community 912"
Cohesion: 1.0
Nodes (0): 

### Community 913 - "Community 913"
Cohesion: 1.0
Nodes (0): 

### Community 914 - "Community 914"
Cohesion: 1.0
Nodes (0): 

### Community 915 - "Community 915"
Cohesion: 1.0
Nodes (0): 

### Community 916 - "Community 916"
Cohesion: 1.0
Nodes (0): 

### Community 917 - "Community 917"
Cohesion: 1.0
Nodes (0): 

### Community 918 - "Community 918"
Cohesion: 1.0
Nodes (0): 

### Community 919 - "Community 919"
Cohesion: 1.0
Nodes (0): 

### Community 920 - "Community 920"
Cohesion: 1.0
Nodes (0): 

### Community 921 - "Community 921"
Cohesion: 1.0
Nodes (0): 

### Community 922 - "Community 922"
Cohesion: 1.0
Nodes (0): 

### Community 923 - "Community 923"
Cohesion: 1.0
Nodes (0): 

### Community 924 - "Community 924"
Cohesion: 1.0
Nodes (0): 

### Community 925 - "Community 925"
Cohesion: 1.0
Nodes (0): 

### Community 926 - "Community 926"
Cohesion: 1.0
Nodes (0): 

### Community 927 - "Community 927"
Cohesion: 1.0
Nodes (0): 

### Community 928 - "Community 928"
Cohesion: 1.0
Nodes (1): MambaConfig

### Community 929 - "Community 929"
Cohesion: 1.0
Nodes (0): 

### Community 930 - "Community 930"
Cohesion: 1.0
Nodes (0): 

### Community 931 - "Community 931"
Cohesion: 1.0
Nodes (0): 

### Community 932 - "Community 932"
Cohesion: 1.0
Nodes (0): 

### Community 933 - "Community 933"
Cohesion: 1.0
Nodes (0): 

### Community 934 - "Community 934"
Cohesion: 1.0
Nodes (0): 

### Community 935 - "Community 935"
Cohesion: 1.0
Nodes (0): 

### Community 936 - "Community 936"
Cohesion: 1.0
Nodes (0): 

### Community 937 - "Community 937"
Cohesion: 1.0
Nodes (0): 

### Community 938 - "Community 938"
Cohesion: 1.0
Nodes (0): 

### Community 939 - "Community 939"
Cohesion: 1.0
Nodes (0): 

### Community 940 - "Community 940"
Cohesion: 1.0
Nodes (0): 

### Community 941 - "Community 941"
Cohesion: 1.0
Nodes (0): 

### Community 942 - "Community 942"
Cohesion: 1.0
Nodes (0): 

### Community 943 - "Community 943"
Cohesion: 1.0
Nodes (0): 

### Community 944 - "Community 944"
Cohesion: 1.0
Nodes (0): 

### Community 945 - "Community 945"
Cohesion: 1.0
Nodes (0): 

### Community 946 - "Community 946"
Cohesion: 1.0
Nodes (0): 

### Community 947 - "Community 947"
Cohesion: 1.0
Nodes (0): 

### Community 948 - "Community 948"
Cohesion: 1.0
Nodes (0): 

### Community 949 - "Community 949"
Cohesion: 1.0
Nodes (0): 

### Community 950 - "Community 950"
Cohesion: 1.0
Nodes (0): 

### Community 951 - "Community 951"
Cohesion: 1.0
Nodes (0): 

### Community 952 - "Community 952"
Cohesion: 1.0
Nodes (0): 

### Community 953 - "Community 953"
Cohesion: 1.0
Nodes (0): 

### Community 954 - "Community 954"
Cohesion: 1.0
Nodes (0): 

### Community 955 - "Community 955"
Cohesion: 1.0
Nodes (0): 

### Community 956 - "Community 956"
Cohesion: 1.0
Nodes (0): 

### Community 957 - "Community 957"
Cohesion: 1.0
Nodes (0): 

### Community 958 - "Community 958"
Cohesion: 1.0
Nodes (0): 

### Community 959 - "Community 959"
Cohesion: 1.0
Nodes (0): 

### Community 960 - "Community 960"
Cohesion: 1.0
Nodes (0): 

### Community 961 - "Community 961"
Cohesion: 1.0
Nodes (0): 

### Community 962 - "Community 962"
Cohesion: 1.0
Nodes (0): 

### Community 963 - "Community 963"
Cohesion: 1.0
Nodes (0): 

### Community 964 - "Community 964"
Cohesion: 1.0
Nodes (0): 

### Community 965 - "Community 965"
Cohesion: 1.0
Nodes (0): 

### Community 966 - "Community 966"
Cohesion: 1.0
Nodes (0): 

### Community 967 - "Community 967"
Cohesion: 1.0
Nodes (0): 

### Community 968 - "Community 968"
Cohesion: 1.0
Nodes (0): 

### Community 969 - "Community 969"
Cohesion: 1.0
Nodes (0): 

### Community 970 - "Community 970"
Cohesion: 1.0
Nodes (0): 

### Community 971 - "Community 971"
Cohesion: 1.0
Nodes (0): 

### Community 972 - "Community 972"
Cohesion: 1.0
Nodes (0): 

### Community 973 - "Community 973"
Cohesion: 1.0
Nodes (0): 

### Community 974 - "Community 974"
Cohesion: 1.0
Nodes (0): 

### Community 975 - "Community 975"
Cohesion: 1.0
Nodes (0): 

### Community 976 - "Community 976"
Cohesion: 1.0
Nodes (0): 

### Community 977 - "Community 977"
Cohesion: 1.0
Nodes (0): 

### Community 978 - "Community 978"
Cohesion: 1.0
Nodes (0): 

### Community 979 - "Community 979"
Cohesion: 1.0
Nodes (0): 

### Community 980 - "Community 980"
Cohesion: 1.0
Nodes (0): 

### Community 981 - "Community 981"
Cohesion: 1.0
Nodes (0): 

### Community 982 - "Community 982"
Cohesion: 1.0
Nodes (0): 

### Community 983 - "Community 983"
Cohesion: 1.0
Nodes (0): 

### Community 984 - "Community 984"
Cohesion: 1.0
Nodes (0): 

### Community 985 - "Community 985"
Cohesion: 1.0
Nodes (0): 

### Community 986 - "Community 986"
Cohesion: 1.0
Nodes (0): 

### Community 987 - "Community 987"
Cohesion: 1.0
Nodes (0): 

### Community 988 - "Community 988"
Cohesion: 1.0
Nodes (0): 

### Community 989 - "Community 989"
Cohesion: 1.0
Nodes (0): 

### Community 990 - "Community 990"
Cohesion: 1.0
Nodes (0): 

### Community 991 - "Community 991"
Cohesion: 1.0
Nodes (0): 

### Community 992 - "Community 992"
Cohesion: 1.0
Nodes (0): 

### Community 993 - "Community 993"
Cohesion: 1.0
Nodes (0): 

### Community 994 - "Community 994"
Cohesion: 1.0
Nodes (0): 

### Community 995 - "Community 995"
Cohesion: 1.0
Nodes (0): 

### Community 996 - "Community 996"
Cohesion: 1.0
Nodes (0): 

### Community 997 - "Community 997"
Cohesion: 1.0
Nodes (0): 

### Community 998 - "Community 998"
Cohesion: 1.0
Nodes (0): 

### Community 999 - "Community 999"
Cohesion: 1.0
Nodes (0): 

### Community 1000 - "Community 1000"
Cohesion: 1.0
Nodes (0): 

### Community 1001 - "Community 1001"
Cohesion: 1.0
Nodes (0): 

### Community 1002 - "Community 1002"
Cohesion: 1.0
Nodes (0): 

### Community 1003 - "Community 1003"
Cohesion: 1.0
Nodes (0): 

### Community 1004 - "Community 1004"
Cohesion: 1.0
Nodes (0): 

### Community 1005 - "Community 1005"
Cohesion: 1.0
Nodes (0): 

### Community 1006 - "Community 1006"
Cohesion: 1.0
Nodes (0): 

### Community 1007 - "Community 1007"
Cohesion: 1.0
Nodes (0): 

### Community 1008 - "Community 1008"
Cohesion: 1.0
Nodes (0): 

### Community 1009 - "Community 1009"
Cohesion: 1.0
Nodes (0): 

### Community 1010 - "Community 1010"
Cohesion: 1.0
Nodes (0): 

### Community 1011 - "Community 1011"
Cohesion: 1.0
Nodes (0): 

### Community 1012 - "Community 1012"
Cohesion: 1.0
Nodes (0): 

### Community 1013 - "Community 1013"
Cohesion: 1.0
Nodes (0): 

### Community 1014 - "Community 1014"
Cohesion: 1.0
Nodes (0): 

### Community 1015 - "Community 1015"
Cohesion: 1.0
Nodes (0): 

### Community 1016 - "Community 1016"
Cohesion: 1.0
Nodes (0): 

### Community 1017 - "Community 1017"
Cohesion: 1.0
Nodes (0): 

### Community 1018 - "Community 1018"
Cohesion: 1.0
Nodes (0): 

### Community 1019 - "Community 1019"
Cohesion: 1.0
Nodes (0): 

### Community 1020 - "Community 1020"
Cohesion: 1.0
Nodes (0): 

### Community 1021 - "Community 1021"
Cohesion: 1.0
Nodes (0): 

### Community 1022 - "Community 1022"
Cohesion: 1.0
Nodes (0): 

### Community 1023 - "Community 1023"
Cohesion: 1.0
Nodes (0): 

### Community 1024 - "Community 1024"
Cohesion: 1.0
Nodes (0): 

### Community 1025 - "Community 1025"
Cohesion: 1.0
Nodes (0): 

### Community 1026 - "Community 1026"
Cohesion: 1.0
Nodes (0): 

### Community 1027 - "Community 1027"
Cohesion: 1.0
Nodes (0): 

### Community 1028 - "Community 1028"
Cohesion: 1.0
Nodes (0): 

### Community 1029 - "Community 1029"
Cohesion: 1.0
Nodes (0): 

### Community 1030 - "Community 1030"
Cohesion: 1.0
Nodes (0): 

### Community 1031 - "Community 1031"
Cohesion: 1.0
Nodes (0): 

### Community 1032 - "Community 1032"
Cohesion: 1.0
Nodes (0): 

### Community 1033 - "Community 1033"
Cohesion: 1.0
Nodes (0): 

### Community 1034 - "Community 1034"
Cohesion: 1.0
Nodes (0): 

### Community 1035 - "Community 1035"
Cohesion: 1.0
Nodes (0): 

### Community 1036 - "Community 1036"
Cohesion: 1.0
Nodes (0): 

### Community 1037 - "Community 1037"
Cohesion: 1.0
Nodes (0): 

### Community 1038 - "Community 1038"
Cohesion: 1.0
Nodes (0): 

### Community 1039 - "Community 1039"
Cohesion: 1.0
Nodes (0): 

### Community 1040 - "Community 1040"
Cohesion: 1.0
Nodes (0): 

### Community 1041 - "Community 1041"
Cohesion: 1.0
Nodes (0): 

### Community 1042 - "Community 1042"
Cohesion: 1.0
Nodes (0): 

### Community 1043 - "Community 1043"
Cohesion: 1.0
Nodes (0): 

### Community 1044 - "Community 1044"
Cohesion: 1.0
Nodes (0): 

### Community 1045 - "Community 1045"
Cohesion: 1.0
Nodes (0): 

### Community 1046 - "Community 1046"
Cohesion: 1.0
Nodes (0): 

### Community 1047 - "Community 1047"
Cohesion: 1.0
Nodes (0): 

### Community 1048 - "Community 1048"
Cohesion: 1.0
Nodes (0): 

### Community 1049 - "Community 1049"
Cohesion: 1.0
Nodes (0): 

### Community 1050 - "Community 1050"
Cohesion: 1.0
Nodes (0): 

### Community 1051 - "Community 1051"
Cohesion: 1.0
Nodes (0): 

### Community 1052 - "Community 1052"
Cohesion: 1.0
Nodes (0): 

### Community 1053 - "Community 1053"
Cohesion: 1.0
Nodes (0): 

### Community 1054 - "Community 1054"
Cohesion: 1.0
Nodes (0): 

### Community 1055 - "Community 1055"
Cohesion: 1.0
Nodes (0): 

### Community 1056 - "Community 1056"
Cohesion: 1.0
Nodes (0): 

### Community 1057 - "Community 1057"
Cohesion: 1.0
Nodes (0): 

### Community 1058 - "Community 1058"
Cohesion: 1.0
Nodes (0): 

### Community 1059 - "Community 1059"
Cohesion: 1.0
Nodes (0): 

### Community 1060 - "Community 1060"
Cohesion: 1.0
Nodes (0): 

### Community 1061 - "Community 1061"
Cohesion: 1.0
Nodes (0): 

### Community 1062 - "Community 1062"
Cohesion: 1.0
Nodes (0): 

### Community 1063 - "Community 1063"
Cohesion: 1.0
Nodes (0): 

### Community 1064 - "Community 1064"
Cohesion: 1.0
Nodes (0): 

### Community 1065 - "Community 1065"
Cohesion: 1.0
Nodes (0): 

### Community 1066 - "Community 1066"
Cohesion: 1.0
Nodes (0): 

### Community 1067 - "Community 1067"
Cohesion: 1.0
Nodes (0): 

### Community 1068 - "Community 1068"
Cohesion: 1.0
Nodes (0): 

### Community 1069 - "Community 1069"
Cohesion: 1.0
Nodes (0): 

### Community 1070 - "Community 1070"
Cohesion: 1.0
Nodes (0): 

### Community 1071 - "Community 1071"
Cohesion: 1.0
Nodes (0): 

### Community 1072 - "Community 1072"
Cohesion: 1.0
Nodes (0): 

### Community 1073 - "Community 1073"
Cohesion: 1.0
Nodes (0): 

### Community 1074 - "Community 1074"
Cohesion: 1.0
Nodes (0): 

### Community 1075 - "Community 1075"
Cohesion: 1.0
Nodes (0): 

### Community 1076 - "Community 1076"
Cohesion: 1.0
Nodes (0): 

### Community 1077 - "Community 1077"
Cohesion: 1.0
Nodes (0): 

### Community 1078 - "Community 1078"
Cohesion: 1.0
Nodes (0): 

### Community 1079 - "Community 1079"
Cohesion: 1.0
Nodes (0): 

### Community 1080 - "Community 1080"
Cohesion: 1.0
Nodes (0): 

### Community 1081 - "Community 1081"
Cohesion: 1.0
Nodes (0): 

### Community 1082 - "Community 1082"
Cohesion: 1.0
Nodes (0): 

### Community 1083 - "Community 1083"
Cohesion: 1.0
Nodes (0): 

### Community 1084 - "Community 1084"
Cohesion: 1.0
Nodes (0): 

### Community 1085 - "Community 1085"
Cohesion: 1.0
Nodes (0): 

### Community 1086 - "Community 1086"
Cohesion: 1.0
Nodes (0): 

### Community 1087 - "Community 1087"
Cohesion: 1.0
Nodes (0): 

### Community 1088 - "Community 1088"
Cohesion: 1.0
Nodes (0): 

### Community 1089 - "Community 1089"
Cohesion: 1.0
Nodes (0): 

### Community 1090 - "Community 1090"
Cohesion: 1.0
Nodes (0): 

### Community 1091 - "Community 1091"
Cohesion: 1.0
Nodes (0): 

### Community 1092 - "Community 1092"
Cohesion: 1.0
Nodes (0): 

### Community 1093 - "Community 1093"
Cohesion: 1.0
Nodes (0): 

### Community 1094 - "Community 1094"
Cohesion: 1.0
Nodes (0): 

### Community 1095 - "Community 1095"
Cohesion: 1.0
Nodes (0): 

### Community 1096 - "Community 1096"
Cohesion: 1.0
Nodes (0): 

### Community 1097 - "Community 1097"
Cohesion: 1.0
Nodes (0): 

### Community 1098 - "Community 1098"
Cohesion: 1.0
Nodes (0): 

### Community 1099 - "Community 1099"
Cohesion: 1.0
Nodes (0): 

### Community 1100 - "Community 1100"
Cohesion: 1.0
Nodes (0): 

### Community 1101 - "Community 1101"
Cohesion: 1.0
Nodes (0): 

### Community 1102 - "Community 1102"
Cohesion: 1.0
Nodes (0): 

### Community 1103 - "Community 1103"
Cohesion: 1.0
Nodes (0): 

### Community 1104 - "Community 1104"
Cohesion: 1.0
Nodes (0): 

### Community 1105 - "Community 1105"
Cohesion: 1.0
Nodes (0): 

### Community 1106 - "Community 1106"
Cohesion: 1.0
Nodes (0): 

### Community 1107 - "Community 1107"
Cohesion: 1.0
Nodes (0): 

### Community 1108 - "Community 1108"
Cohesion: 1.0
Nodes (0): 

### Community 1109 - "Community 1109"
Cohesion: 1.0
Nodes (1): Return the number of predictions in this assignment

### Community 1110 - "Community 1110"
Cohesion: 1.0
Nodes (1): Returns a dictionary of info about the object

### Community 1111 - "Community 1111"
Cohesion: 1.0
Nodes (1): Create random AssignResult for tests or debugging.          Kwargs:

### Community 1112 - "Community 1112"
Cohesion: 1.0
Nodes (1): Returns a dictionary of info about the object.

### Community 1113 - "Community 1113"
Cohesion: 1.0
Nodes (1): Args:             rng (None | int | numpy.random.RandomState): seed or state

### Community 1114 - "Community 1114"
Cohesion: 1.0
Nodes (1): Dictionary mapper.         Renames keys according to keymap provided.          A

### Community 1115 - "Community 1115"
Cohesion: 1.0
Nodes (1): Transform network output for a batch into labeled boxes.          Args:

### Community 1116 - "Community 1116"
Cohesion: 1.0
Nodes (1): int: Input feature map levels.

### Community 1117 - "Community 1117"
Cohesion: 1.0
Nodes (1): Async test only det bboxes without augmentation.

### Community 1118 - "Community 1118"
Cohesion: 1.0
Nodes (1): Compute full log-likelihood of a string, with no truncation, for perplexity comp

### Community 1119 - "Community 1119"
Cohesion: 1.0
Nodes (1): Calls either forward_train or forward_test depending on whether         return_l

### Community 1120 - "Community 1120"
Cohesion: 1.0
Nodes (1): Compute target of mask IoU.          Mask IoU target is the IoU of the predicted

### Community 1121 - "Community 1121"
Cohesion: 1.0
Nodes (1): Get the mask scores.          mask_score = bbox_score * mask_iou

### Community 1122 - "Community 1122"
Cohesion: 1.0
Nodes (0): 

### Community 1123 - "Community 1123"
Cohesion: 1.0
Nodes (0): 

### Community 1124 - "Community 1124"
Cohesion: 1.0
Nodes (0): 

### Community 1125 - "Community 1125"
Cohesion: 1.0
Nodes (0): 

### Community 1126 - "Community 1126"
Cohesion: 1.0
Nodes (0): 

### Community 1127 - "Community 1127"
Cohesion: 1.0
Nodes (0): 

### Community 1128 - "Community 1128"
Cohesion: 1.0
Nodes (0): 

### Community 1129 - "Community 1129"
Cohesion: 1.0
Nodes (0): 

### Community 1130 - "Community 1130"
Cohesion: 1.0
Nodes (0): 

### Community 1131 - "Community 1131"
Cohesion: 1.0
Nodes (0): 

### Community 1132 - "Community 1132"
Cohesion: 1.0
Nodes (0): 

### Community 1133 - "Community 1133"
Cohesion: 1.0
Nodes (0): 

### Community 1134 - "Community 1134"
Cohesion: 1.0
Nodes (0): 

### Community 1135 - "Community 1135"
Cohesion: 1.0
Nodes (0): 

### Community 1136 - "Community 1136"
Cohesion: 1.0
Nodes (0): 

### Community 1137 - "Community 1137"
Cohesion: 1.0
Nodes (0): 

### Community 1138 - "Community 1138"
Cohesion: 1.0
Nodes (0): 

### Community 1139 - "Community 1139"
Cohesion: 1.0
Nodes (0): 

### Community 1140 - "Community 1140"
Cohesion: 1.0
Nodes (0): 

### Community 1141 - "Community 1141"
Cohesion: 1.0
Nodes (0): 

### Community 1142 - "Community 1142"
Cohesion: 1.0
Nodes (0): 

### Community 1143 - "Community 1143"
Cohesion: 1.0
Nodes (0): 

### Community 1144 - "Community 1144"
Cohesion: 1.0
Nodes (0): 

### Community 1145 - "Community 1145"
Cohesion: 1.0
Nodes (0): 

### Community 1146 - "Community 1146"
Cohesion: 1.0
Nodes (0): 

### Community 1147 - "Community 1147"
Cohesion: 1.0
Nodes (0): 

### Community 1148 - "Community 1148"
Cohesion: 1.0
Nodes (0): 

### Community 1149 - "Community 1149"
Cohesion: 1.0
Nodes (0): 

### Community 1150 - "Community 1150"
Cohesion: 1.0
Nodes (0): 

### Community 1151 - "Community 1151"
Cohesion: 1.0
Nodes (0): 

### Community 1152 - "Community 1152"
Cohesion: 1.0
Nodes (0): 

### Community 1153 - "Community 1153"
Cohesion: 1.0
Nodes (0): 

### Community 1154 - "Community 1154"
Cohesion: 1.0
Nodes (0): 

### Community 1155 - "Community 1155"
Cohesion: 1.0
Nodes (0): 

### Community 1156 - "Community 1156"
Cohesion: 1.0
Nodes (0): 

### Community 1157 - "Community 1157"
Cohesion: 1.0
Nodes (0): 

### Community 1158 - "Community 1158"
Cohesion: 1.0
Nodes (0): 

### Community 1159 - "Community 1159"
Cohesion: 1.0
Nodes (0): 

### Community 1160 - "Community 1160"
Cohesion: 1.0
Nodes (0): 

### Community 1161 - "Community 1161"
Cohesion: 1.0
Nodes (0): 

### Community 1162 - "Community 1162"
Cohesion: 1.0
Nodes (0): 

### Community 1163 - "Community 1163"
Cohesion: 1.0
Nodes (0): 

### Community 1164 - "Community 1164"
Cohesion: 1.0
Nodes (0): 

### Community 1165 - "Community 1165"
Cohesion: 1.0
Nodes (0): 

### Community 1166 - "Community 1166"
Cohesion: 1.0
Nodes (0): 

### Community 1167 - "Community 1167"
Cohesion: 1.0
Nodes (0): 

### Community 1168 - "Community 1168"
Cohesion: 1.0
Nodes (0): 

### Community 1169 - "Community 1169"
Cohesion: 1.0
Nodes (0): 

### Community 1170 - "Community 1170"
Cohesion: 1.0
Nodes (0): 

### Community 1171 - "Community 1171"
Cohesion: 1.0
Nodes (0): 

### Community 1172 - "Community 1172"
Cohesion: 1.0
Nodes (0): 

### Community 1173 - "Community 1173"
Cohesion: 1.0
Nodes (0): 

### Community 1174 - "Community 1174"
Cohesion: 1.0
Nodes (0): 

### Community 1175 - "Community 1175"
Cohesion: 1.0
Nodes (0): 

### Community 1176 - "Community 1176"
Cohesion: 1.0
Nodes (0): 

### Community 1177 - "Community 1177"
Cohesion: 1.0
Nodes (0): 

### Community 1178 - "Community 1178"
Cohesion: 1.0
Nodes (0): 

### Community 1179 - "Community 1179"
Cohesion: 1.0
Nodes (0): 

### Community 1180 - "Community 1180"
Cohesion: 1.0
Nodes (0): 

### Community 1181 - "Community 1181"
Cohesion: 1.0
Nodes (0): 

### Community 1182 - "Community 1182"
Cohesion: 1.0
Nodes (0): 

### Community 1183 - "Community 1183"
Cohesion: 1.0
Nodes (0): 

### Community 1184 - "Community 1184"
Cohesion: 1.0
Nodes (0): 

### Community 1185 - "Community 1185"
Cohesion: 1.0
Nodes (0): 

### Community 1186 - "Community 1186"
Cohesion: 1.0
Nodes (0): 

### Community 1187 - "Community 1187"
Cohesion: 1.0
Nodes (0): 

### Community 1188 - "Community 1188"
Cohesion: 1.0
Nodes (0): 

### Community 1189 - "Community 1189"
Cohesion: 1.0
Nodes (0): 

### Community 1190 - "Community 1190"
Cohesion: 1.0
Nodes (0): 

### Community 1191 - "Community 1191"
Cohesion: 1.0
Nodes (0): 

### Community 1192 - "Community 1192"
Cohesion: 1.0
Nodes (0): 

### Community 1193 - "Community 1193"
Cohesion: 1.0
Nodes (0): 

### Community 1194 - "Community 1194"
Cohesion: 1.0
Nodes (0): 

### Community 1195 - "Community 1195"
Cohesion: 1.0
Nodes (0): 

### Community 1196 - "Community 1196"
Cohesion: 1.0
Nodes (0): 

### Community 1197 - "Community 1197"
Cohesion: 1.0
Nodes (0): 

### Community 1198 - "Community 1198"
Cohesion: 1.0
Nodes (0): 

### Community 1199 - "Community 1199"
Cohesion: 1.0
Nodes (0): 

### Community 1200 - "Community 1200"
Cohesion: 1.0
Nodes (0): 

### Community 1201 - "Community 1201"
Cohesion: 1.0
Nodes (0): 

### Community 1202 - "Community 1202"
Cohesion: 1.0
Nodes (0): 

### Community 1203 - "Community 1203"
Cohesion: 1.0
Nodes (0): 

### Community 1204 - "Community 1204"
Cohesion: 1.0
Nodes (0): 

### Community 1205 - "Community 1205"
Cohesion: 1.0
Nodes (0): 

### Community 1206 - "Community 1206"
Cohesion: 1.0
Nodes (0): 

### Community 1207 - "Community 1207"
Cohesion: 1.0
Nodes (0): 

### Community 1208 - "Community 1208"
Cohesion: 1.0
Nodes (0): 

### Community 1209 - "Community 1209"
Cohesion: 1.0
Nodes (0): 

### Community 1210 - "Community 1210"
Cohesion: 1.0
Nodes (0): 

### Community 1211 - "Community 1211"
Cohesion: 1.0
Nodes (0): 

### Community 1212 - "Community 1212"
Cohesion: 1.0
Nodes (0): 

### Community 1213 - "Community 1213"
Cohesion: 1.0
Nodes (0): 

### Community 1214 - "Community 1214"
Cohesion: 1.0
Nodes (0): 

### Community 1215 - "Community 1215"
Cohesion: 1.0
Nodes (0): 

### Community 1216 - "Community 1216"
Cohesion: 1.0
Nodes (0): 

### Community 1217 - "Community 1217"
Cohesion: 1.0
Nodes (0): 

### Community 1218 - "Community 1218"
Cohesion: 1.0
Nodes (0): 

### Community 1219 - "Community 1219"
Cohesion: 1.0
Nodes (0): 

### Community 1220 - "Community 1220"
Cohesion: 1.0
Nodes (0): 

### Community 1221 - "Community 1221"
Cohesion: 1.0
Nodes (0): 

### Community 1222 - "Community 1222"
Cohesion: 1.0
Nodes (0): 

### Community 1223 - "Community 1223"
Cohesion: 1.0
Nodes (0): 

### Community 1224 - "Community 1224"
Cohesion: 1.0
Nodes (0): 

### Community 1225 - "Community 1225"
Cohesion: 1.0
Nodes (0): 

### Community 1226 - "Community 1226"
Cohesion: 1.0
Nodes (0): 

### Community 1227 - "Community 1227"
Cohesion: 1.0
Nodes (0): 

### Community 1228 - "Community 1228"
Cohesion: 1.0
Nodes (0): 

### Community 1229 - "Community 1229"
Cohesion: 1.0
Nodes (0): 

### Community 1230 - "Community 1230"
Cohesion: 1.0
Nodes (0): 

### Community 1231 - "Community 1231"
Cohesion: 1.0
Nodes (0): 

### Community 1232 - "Community 1232"
Cohesion: 1.0
Nodes (0): 

### Community 1233 - "Community 1233"
Cohesion: 1.0
Nodes (0): 

### Community 1234 - "Community 1234"
Cohesion: 1.0
Nodes (0): 

### Community 1235 - "Community 1235"
Cohesion: 1.0
Nodes (0): 

### Community 1236 - "Community 1236"
Cohesion: 1.0
Nodes (0): 

### Community 1237 - "Community 1237"
Cohesion: 1.0
Nodes (0): 

### Community 1238 - "Community 1238"
Cohesion: 1.0
Nodes (0): 

### Community 1239 - "Community 1239"
Cohesion: 1.0
Nodes (0): 

### Community 1240 - "Community 1240"
Cohesion: 1.0
Nodes (0): 

### Community 1241 - "Community 1241"
Cohesion: 1.0
Nodes (0): 

### Community 1242 - "Community 1242"
Cohesion: 1.0
Nodes (0): 

### Community 1243 - "Community 1243"
Cohesion: 1.0
Nodes (0): 

### Community 1244 - "Community 1244"
Cohesion: 1.0
Nodes (0): 

### Community 1245 - "Community 1245"
Cohesion: 1.0
Nodes (0): 

### Community 1246 - "Community 1246"
Cohesion: 1.0
Nodes (0): 

### Community 1247 - "Community 1247"
Cohesion: 1.0
Nodes (0): 

### Community 1248 - "Community 1248"
Cohesion: 1.0
Nodes (0): 

### Community 1249 - "Community 1249"
Cohesion: 1.0
Nodes (1): Computation of the ARI clustering metric.      NOTE: This implementation does no

### Community 1250 - "Community 1250"
Cohesion: 1.0
Nodes (1): See `Ari` docstring for allowed keyword arguments.

### Community 1251 - "Community 1251"
Cohesion: 1.0
Nodes (1): Returns the transformed tensor.      Args:       tensor: Any of a set of differe

### Community 1252 - "Community 1252"
Cohesion: 1.0
Nodes (1): Returns the transformed tensor.      Args:       tensor: Any of a set of differe

### Community 1253 - "Community 1253"
Cohesion: 1.0
Nodes (1): Slot Attention module forward pass.

### Community 1254 - "Community 1254"
Cohesion: 1.0
Nodes (1): Computes inverted dot-product attention.      Args:       query: Queries with sh

### Community 1255 - "Community 1255"
Cohesion: 1.0
Nodes (1): Computes multi-head dot-product attention given query, key, and value.      Args

### Community 1256 - "Community 1256"
Cohesion: 1.0
Nodes (1): Apply the ResNet to the inputs `x`.      Args:       x: Inputs.       train: Whe

### Community 1257 - "Community 1257"
Cohesion: 1.0
Nodes (1): Performs a forward pass on a video.      Args:       video: Video of shape `[bat

### Community 1258 - "Community 1258"
Cohesion: 1.0
Nodes (0): 

### Community 1259 - "Community 1259"
Cohesion: 1.0
Nodes (1): Get parameters for ``crop`` for a random sized crop.         Args:             i

### Community 1260 - "Community 1260"
Cohesion: 1.0
Nodes (1): Get parameters for ``crop`` for a random sized crop.         Args:             i

### Community 1261 - "Community 1261"
Cohesion: 1.0
Nodes (1): Update center used for teacher output.

### Community 1262 - "Community 1262"
Cohesion: 1.0
Nodes (1): image_encoder returns the VAE Encoder with pretrained weights.      Usage:     `

### Community 1263 - "Community 1263"
Cohesion: 1.0
Nodes (1): decoder returns the diffusion image decoder model with pretrained      weights.

### Community 1264 - "Community 1264"
Cohesion: 1.0
Nodes (1): tokenizer returns the tokenizer used for text inputs.      Can be overriden for

### Community 1265 - "Community 1265"
Cohesion: 1.0
Nodes (1): text_encoder returns the text encoder with pretrained weights.      Can be overr

### Community 1266 - "Community 1266"
Cohesion: 1.0
Nodes (1): diffusion_model returns the diffusion model with pretrained weights.      Can be

### Community 1267 - "Community 1267"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1268 - "Community 1268"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 1269 - "Community 1269"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1270 - "Community 1270"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_pooler (ROI

### Community 1271 - "Community 1271"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage.         L

### Community 1272 - "Community 1272"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1273 - "Community 1273"
Cohesion: 1.0
Nodes (1): Args:             short_edge_length (list[int]): If ``sample_style=="range"``,

### Community 1274 - "Community 1274"
Cohesion: 1.0
Nodes (1): Compute the output size given input size and target short edge length.

### Community 1275 - "Community 1275"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 1276 - "Community 1276"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads.         It performs bo

### Community 1277 - "Community 1277"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_features (li

### Community 1278 - "Community 1278"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 1279 - "Community 1279"
Cohesion: 1.0
Nodes (1): return features of all patches at the last ViT block         net: the model

### Community 1280 - "Community 1280"
Cohesion: 1.0
Nodes (1): calculate squared distance matrix of each point         params:             feat

### Community 1281 - "Community 1281"
Cohesion: 1.0
Nodes (1): transfer indices of matrix to array         indices: np.array([[i,j],...])

### Community 1282 - "Community 1282"
Cohesion: 1.0
Nodes (1): transfer indices of array to matrix         indices: np.array([i,....])

### Community 1283 - "Community 1283"
Cohesion: 1.0
Nodes (1): visualize each component with different color         :return:

### Community 1284 - "Community 1284"
Cohesion: 1.0
Nodes (1): transfer indices of matrix to array         indices: np.array([[i,j],...])

### Community 1285 - "Community 1285"
Cohesion: 1.0
Nodes (1): transfer indices of array to matrix         indices: np.array([i,....])

### Community 1286 - "Community 1286"
Cohesion: 1.0
Nodes (1): return features of all patches at the last ViT block         net: the model

### Community 1287 - "Community 1287"
Cohesion: 1.0
Nodes (1): calculate squared distance matrix of each point         params:             feat

### Community 1288 - "Community 1288"
Cohesion: 1.0
Nodes (1): transfer indices of matrix to array         indices: np.array([[i,j],...])

### Community 1289 - "Community 1289"
Cohesion: 1.0
Nodes (1): transfer indices of array to matrix         indices: np.array([i,....])

### Community 1290 - "Community 1290"
Cohesion: 1.0
Nodes (1): visualize each component with different color         :return:

### Community 1291 - "Community 1291"
Cohesion: 1.0
Nodes (1): preprocess the cv2 image         mode: fixed -> 224 * 224; flexible -> ratio doe

### Community 1292 - "Community 1292"
Cohesion: 1.0
Nodes (1): load pretrained UnionSeg model         path: the path of the model

### Community 1293 - "Community 1293"
Cohesion: 1.0
Nodes (1): calculate squared distance matrix of each point         params:             feat

### Community 1294 - "Community 1294"
Cohesion: 1.0
Nodes (1): transfer indices of array to matrix         indices: np.array([i,....])

### Community 1295 - "Community 1295"
Cohesion: 1.0
Nodes (1): visualize each component with different color         :return:

### Community 1296 - "Community 1296"
Cohesion: 1.0
Nodes (1): check if previous errors happen again

### Community 1297 - "Community 1297"
Cohesion: 1.0
Nodes (1): Find the bounding box of the largest object in a binary image.         :param bi

### Community 1298 - "Community 1298"
Cohesion: 1.0
Nodes (1): preprocess the cv2 image         mode: fixed -> 224 * 224; flexible -> ratio doe

### Community 1299 - "Community 1299"
Cohesion: 1.0
Nodes (1): load pretrained UnionSeg model         path: the path of the model

### Community 1300 - "Community 1300"
Cohesion: 1.0
Nodes (1): transfer indices of array to matrix         indices: np.array([i,....])

### Community 1301 - "Community 1301"
Cohesion: 1.0
Nodes (1): visualize each component with different color         :return:

### Community 1302 - "Community 1302"
Cohesion: 1.0
Nodes (1): check if previous errors happen again

### Community 1303 - "Community 1303"
Cohesion: 1.0
Nodes (1): Find the bounding box of the largest object in a binary image.         :param bi

### Community 1304 - "Community 1304"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1305 - "Community 1305"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1306 - "Community 1306"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1307 - "Community 1307"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1308 - "Community 1308"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 1309 - "Community 1309"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1310 - "Community 1310"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             in_channels: cha

### Community 1311 - "Community 1311"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1312 - "Community 1312"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1313 - "Community 1313"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1314 - "Community 1314"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1315 - "Community 1315"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1316 - "Community 1316"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 1317 - "Community 1317"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1318 - "Community 1318"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1319 - "Community 1319"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1320 - "Community 1320"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1321 - "Community 1321"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1322 - "Community 1322"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1323 - "Community 1323"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 1324 - "Community 1324"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1325 - "Community 1325"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 1326 - "Community 1326"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 1327 - "Community 1327"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             in_channels: cha

### Community 1328 - "Community 1328"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1329 - "Community 1329"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1330 - "Community 1330"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1331 - "Community 1331"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1332 - "Community 1332"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1333 - "Community 1333"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 1334 - "Community 1334"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1335 - "Community 1335"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1336 - "Community 1336"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1337 - "Community 1337"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 1338 - "Community 1338"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 1339 - "Community 1339"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 1340 - "Community 1340"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1341 - "Community 1341"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 1342 - "Community 1342"
Cohesion: 1.0
Nodes (1): Returns:             torch.optim.Optimizer:          It now calls :func:`detectr

### Community 1343 - "Community 1343"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1344 - "Community 1344"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 1345 - "Community 1345"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 1346 - "Community 1346"
Cohesion: 1.0
Nodes (1): Returns:             DatasetEvaluator or None          It is not implemented by

### Community 1347 - "Community 1347"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1348 - "Community 1348"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 1349 - "Community 1349"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 1350 - "Community 1350"
Cohesion: 1.0
Nodes (1): Returns:             DatasetEvaluator or None          It is not implemented by

### Community 1351 - "Community 1351"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1352 - "Community 1352"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 1353 - "Community 1353"
Cohesion: 1.0
Nodes (0): 

### Community 1354 - "Community 1354"
Cohesion: 1.0
Nodes (0): 

### Community 1355 - "Community 1355"
Cohesion: 1.0
Nodes (0): 

### Community 1356 - "Community 1356"
Cohesion: 1.0
Nodes (0): 

### Community 1357 - "Community 1357"
Cohesion: 1.0
Nodes (1): Computation of the ARI clustering metric.      NOTE: This implementation does no

### Community 1358 - "Community 1358"
Cohesion: 1.0
Nodes (1): See `Ari` docstring for allowed keyword arguments.

### Community 1359 - "Community 1359"
Cohesion: 1.0
Nodes (1): Returns the transformed tensor.      Args:       tensor: Any of a set of differe

### Community 1360 - "Community 1360"
Cohesion: 1.0
Nodes (1): Returns the transformed tensor.      Args:       tensor: Any of a set of differe

### Community 1361 - "Community 1361"
Cohesion: 1.0
Nodes (1): Slot Attention module forward pass.

### Community 1362 - "Community 1362"
Cohesion: 1.0
Nodes (1): Computes inverted dot-product attention.      Args:       query: Queries with sh

### Community 1363 - "Community 1363"
Cohesion: 1.0
Nodes (1): Computes multi-head dot-product attention given query, key, and value.      Args

### Community 1364 - "Community 1364"
Cohesion: 1.0
Nodes (1): Apply the ResNet to the inputs `x`.      Args:       x: Inputs.       train: Whe

### Community 1365 - "Community 1365"
Cohesion: 1.0
Nodes (1): Computes inverted dot-product attention with key per query.      Args:       que

### Community 1366 - "Community 1366"
Cohesion: 1.0
Nodes (1): Slot Attention with explicit slot statistics module forward pass.

### Community 1367 - "Community 1367"
Cohesion: 1.0
Nodes (1): Slot Attention with explicit slot statistics module forward pass.

### Community 1368 - "Community 1368"
Cohesion: 1.0
Nodes (1): Slot Attention translation equiv. module forward pass.

### Community 1369 - "Community 1369"
Cohesion: 1.0
Nodes (1): Slot Attention translation and scale equiv. module forward pass.

### Community 1370 - "Community 1370"
Cohesion: 1.0
Nodes (1): Slot Attention translation and scale equiv. module forward pass.

### Community 1371 - "Community 1371"
Cohesion: 1.0
Nodes (1): Performs a forward pass on a video.      Args:       video: Video of shape `[bat

### Community 1372 - "Community 1372"
Cohesion: 1.0
Nodes (1): Args:             values: tensor of shape (batch, n_true_classes, n_pred_classes

### Community 1373 - "Community 1373"
Cohesion: 1.0
Nodes (1): Compute auxilliary outputs only needed for metrics and visualisations.

### Community 1374 - "Community 1374"
Cohesion: 1.0
Nodes (1): Try to infer same padding for convolutions.

### Community 1375 - "Community 1375"
Cohesion: 1.0
Nodes (1): Try to infer same padding for transposed convolutions.

### Community 1376 - "Community 1376"
Cohesion: 1.0
Nodes (1): Iterates dataset, then adds dummy samples until reaching the specified number of

### Community 1377 - "Community 1377"
Cohesion: 1.0
Nodes (1): Construct padding for property.

### Community 1378 - "Community 1378"
Cohesion: 1.0
Nodes (1): Create pipeline object serving same function as wds.WebDataset.          We do t

### Community 1379 - "Community 1379"
Cohesion: 1.0
Nodes (1): Keys of properties to keep in dataset after filtering.

### Community 1380 - "Community 1380"
Cohesion: 1.0
Nodes (1): Number of samples after pipeline is applied, given original number of samples in

### Community 1381 - "Community 1381"
Cohesion: 1.0
Nodes (1): Apply pipeline to dataset.          Input dataset contains dicts of samples afte

### Community 1382 - "Community 1382"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1383 - "Community 1383"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 1384 - "Community 1384"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1385 - "Community 1385"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1386 - "Community 1386"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_pooler (ROI

### Community 1387 - "Community 1387"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage.         L

### Community 1388 - "Community 1388"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1389 - "Community 1389"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 1390 - "Community 1390"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads.         It performs bo

### Community 1391 - "Community 1391"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_features (li

### Community 1392 - "Community 1392"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 1393 - "Community 1393"
Cohesion: 1.0
Nodes (1): Args:             short_edge_length (list[int]): If ``sample_style=="range"``,

### Community 1394 - "Community 1394"
Cohesion: 1.0
Nodes (1): Compute the output size given input size and target short edge length.

### Community 1395 - "Community 1395"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 1396 - "Community 1396"
Cohesion: 1.0
Nodes (1): Returns:             torch.optim.Optimizer:          It now calls :func:`detectr

### Community 1397 - "Community 1397"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1398 - "Community 1398"
Cohesion: 1.0
Nodes (1): Returns:             torch.optim.Optimizer:          It now calls :func:`detectr

### Community 1399 - "Community 1399"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1400 - "Community 1400"
Cohesion: 1.0
Nodes (1): Returns:             DatasetEvaluator or None          It is not implemented by

### Community 1401 - "Community 1401"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1402 - "Community 1402"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 1403 - "Community 1403"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 1404 - "Community 1404"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 1405 - "Community 1405"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 1406 - "Community 1406"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 1407 - "Community 1407"
Cohesion: 1.0
Nodes (1): Yield successive n-sized chunks from lst.

### Community 1408 - "Community 1408"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 1409 - "Community 1409"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 1410 - "Community 1410"
Cohesion: 1.0
Nodes (0): 

### Community 1411 - "Community 1411"
Cohesion: 1.0
Nodes (1): Build the 3x3 camera matrix K using the given intrinsics.          Equation 6.10

### Community 1412 - "Community 1412"
Cohesion: 1.0
Nodes (1): Convert extrinsics matrix to separate rotation matrix R and translation vector T

### Community 1413 - "Community 1413"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)

### Community 1414 - "Community 1414"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 1415 - "Community 1415"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 1416 - "Community 1416"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 1417 - "Community 1417"
Cohesion: 1.0
Nodes (1): input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output

### Community 1418 - "Community 1418"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1419 - "Community 1419"
Cohesion: 1.0
Nodes (1): input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output

### Community 1420 - "Community 1420"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1421 - "Community 1421"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o

### Community 1422 - "Community 1422"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1423 - "Community 1423"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o

### Community 1424 - "Community 1424"
Cohesion: 1.0
Nodes (1): input: grad_output: (L, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1425 - "Community 1425"
Cohesion: 1.0
Nodes (1): input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)

### Community 1426 - "Community 1426"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 1427 - "Community 1427"
Cohesion: 1.0
Nodes (1): input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L

### Community 1428 - "Community 1428"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 1429 - "Community 1429"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 1430 - "Community 1430"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 1431 - "Community 1431"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hd

### Community 1432 - "Community 1432"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1433 - "Community 1433"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (

### Community 1434 - "Community 1434"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 1435 - "Community 1435"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 1436 - "Community 1436"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 1437 - "Community 1437"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 1438 - "Community 1438"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n

### Community 1439 - "Community 1439"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1440 - "Community 1440"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1441 - "Community 1441"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)

### Community 1442 - "Community 1442"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 1443 - "Community 1443"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 1444 - "Community 1444"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 1445 - "Community 1445"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 1446 - "Community 1446"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 1447 - "Community 1447"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 1448 - "Community 1448"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n

### Community 1449 - "Community 1449"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1450 - "Community 1450"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1451 - "Community 1451"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)

### Community 1452 - "Community 1452"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 1453 - "Community 1453"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 1454 - "Community 1454"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 1455 - "Community 1455"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 1456 - "Community 1456"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 1457 - "Community 1457"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 1458 - "Community 1458"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 1459 - "Community 1459"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1460 - "Community 1460"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 1461 - "Community 1461"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 1462 - "Community 1462"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1463 - "Community 1463"
Cohesion: 1.0
Nodes (1): r"""         Uses iterative furthest point sampling to select a set of npoint fe

### Community 1464 - "Community 1464"
Cohesion: 1.0
Nodes (1): r"""          Parameters         ----------         features : torch.Tensor

### Community 1465 - "Community 1465"
Cohesion: 1.0
Nodes (1): r"""             Find the three nearest neighbors of unknown in known         Pa

### Community 1466 - "Community 1466"
Cohesion: 1.0
Nodes (1): r"""             Performs weight linear interpolation on 3 features         Para

### Community 1467 - "Community 1467"
Cohesion: 1.0
Nodes (1): r"""         Parameters         ----------         grad_out : torch.Tensor

### Community 1468 - "Community 1468"
Cohesion: 1.0
Nodes (1): r"""          Parameters         ----------         features : torch.Tensor

### Community 1469 - "Community 1469"
Cohesion: 1.0
Nodes (1): r"""          Parameters         ----------         grad_out : torch.Tensor

### Community 1470 - "Community 1470"
Cohesion: 1.0
Nodes (1): r"""          Parameters         ----------         radius : float             r

### Community 1471 - "Community 1471"
Cohesion: 1.0
Nodes (1): :param model_type: a string specifying which model to load. [dino_vits8 | dino_v

### Community 1472 - "Community 1472"
Cohesion: 1.0
Nodes (1): Creates a method for position encoding interpolation.         :param patch_size:

### Community 1473 - "Community 1473"
Cohesion: 1.0
Nodes (1): change resolution of model output by changing the stride of the patch extraction

### Community 1474 - "Community 1474"
Cohesion: 1.0
Nodes (1): r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns P

### Community 1475 - "Community 1475"
Cohesion: 1.0
Nodes (1): r""" Apply mask to the given image.

### Community 1476 - "Community 1476"
Cohesion: 1.0
Nodes (0): 

### Community 1477 - "Community 1477"
Cohesion: 1.0
Nodes (0): 

### Community 1478 - "Community 1478"
Cohesion: 1.0
Nodes (1): Convert Stanford3DDataset to PLY format that is compatible with         Synthia

### Community 1479 - "Community 1479"
Cohesion: 1.0
Nodes (1): Args:             io: (str or binary file-like object): input file to load data

### Community 1480 - "Community 1480"
Cohesion: 1.0
Nodes (1): Convert DensePose predictor outputs to BitMasks using some registered         co

### Community 1481 - "Community 1481"
Cohesion: 1.0
Nodes (1): Convert DensePose predictor outputs to DensePoseResult using some registered

### Community 1482 - "Community 1482"
Cohesion: 1.0
Nodes (1): Convert DensePose predictor outputs to DensePoseResult with confidences

### Community 1483 - "Community 1483"
Cohesion: 1.0
Nodes (1): Performs an horizontal flip on DensePose predictor outputs.         Does recursi

### Community 1484 - "Community 1484"
Cohesion: 1.0
Nodes (1): Perform recursive lookup for the given type         to find registered converter

### Community 1485 - "Community 1485"
Cohesion: 1.0
Nodes (1): Convert an instance to the destination type using some registered         conver

### Community 1486 - "Community 1486"
Cohesion: 1.0
Nodes (1): Filters proposals with targets to keep only the ones relevant for         DenseP

### Community 1487 - "Community 1487"
Cohesion: 1.0
Nodes (1): Accumulate instances data for one image          Args:             instances_one

### Community 1488 - "Community 1488"
Cohesion: 1.0
Nodes (1): Pack data into tensors

### Community 1489 - "Community 1489"
Cohesion: 1.0
Nodes (1): Reset embeddings to random values

### Community 1490 - "Community 1490"
Cohesion: 1.0
Nodes (1): Load data from a file          Args:             fpath (str): file path to load

### Community 1491 - "Community 1491"
Cohesion: 1.0
Nodes (1): Load data from a file          Args:             fpath (str): file path to load

### Community 1492 - "Community 1492"
Cohesion: 1.0
Nodes (1): Args:             cfg (CfgNode):             model (nn.Module):             eval

### Community 1493 - "Community 1493"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1494 - "Community 1494"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 1495 - "Community 1495"
Cohesion: 1.0
Nodes (1): Build an optimizer from config.

### Community 1496 - "Community 1496"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             augmentations:

### Community 1497 - "Community 1497"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1498 - "Community 1498"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1499 - "Community 1499"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1500 - "Community 1500"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             is_train: wheth

### Community 1501 - "Community 1501"
Cohesion: 1.0
Nodes (1): Args:             anchors (list[list[Boxes]]): a list of N=#image elements. Each

### Community 1502 - "Community 1502"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 1503 - "Community 1503"
Cohesion: 1.0
Nodes (1): Args:             conv_dim: the output dimension of the conv layers

### Community 1504 - "Community 1504"
Cohesion: 1.0
Nodes (0): 

### Community 1505 - "Community 1505"
Cohesion: 1.0
Nodes (1): Compute gradients for ROIAlignRotated with multiple bounding boxes on the GPU,

### Community 1506 - "Community 1506"
Cohesion: 1.0
Nodes (0): 

### Community 1507 - "Community 1507"
Cohesion: 1.0
Nodes (0): 

### Community 1508 - "Community 1508"
Cohesion: 1.0
Nodes (0): 

### Community 1509 - "Community 1509"
Cohesion: 1.0
Nodes (0): 

### Community 1510 - "Community 1510"
Cohesion: 1.0
Nodes (0): 

### Community 1511 - "Community 1511"
Cohesion: 1.0
Nodes (0): 

### Community 1512 - "Community 1512"
Cohesion: 1.0
Nodes (0): 

### Community 1513 - "Community 1513"
Cohesion: 1.0
Nodes (0): 

### Community 1514 - "Community 1514"
Cohesion: 1.0
Nodes (0): 

### Community 1515 - "Community 1515"
Cohesion: 1.0
Nodes (0): 

### Community 1516 - "Community 1516"
Cohesion: 1.0
Nodes (0): 

### Community 1517 - "Community 1517"
Cohesion: 1.0
Nodes (0): 

### Community 1518 - "Community 1518"
Cohesion: 1.0
Nodes (0): 

### Community 1519 - "Community 1519"
Cohesion: 1.0
Nodes (0): 

### Community 1520 - "Community 1520"
Cohesion: 1.0
Nodes (0): 

### Community 1521 - "Community 1521"
Cohesion: 1.0
Nodes (0): 

### Community 1522 - "Community 1522"
Cohesion: 1.0
Nodes (1): Calculate proper im2col step size, which should be divisible by input_size and n

### Community 1523 - "Community 1523"
Cohesion: 1.0
Nodes (1): Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg

### Community 1524 - "Community 1524"
Cohesion: 1.0
Nodes (1): Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (

### Community 1525 - "Community 1525"
Cohesion: 1.0
Nodes (1): Returns:             tuple: height, width

### Community 1526 - "Community 1526"
Cohesion: 1.0
Nodes (1): Args:             instance_lists (list[Instances])          Returns:

### Community 1527 - "Community 1527"
Cohesion: 1.0
Nodes (1): Args:             box: can be a k-tuple, k-list or an Nxk array/tensor, where k

### Community 1528 - "Community 1528"
Cohesion: 1.0
Nodes (1): Concatenates a list of Boxes into a single Boxes          Arguments:

### Community 1529 - "Community 1529"
Cohesion: 1.0
Nodes (1): Yield a box as a Tensor of shape (4,) at a time.

### Community 1530 - "Community 1530"
Cohesion: 1.0
Nodes (1): Concatenates a list of Keypoints into a single Keypoints          Arguments:

### Community 1531 - "Community 1531"
Cohesion: 1.0
Nodes (1): Returns:             BitMasks: Create a new :class:`BitMasks` by indexing.

### Community 1532 - "Community 1532"
Cohesion: 1.0
Nodes (1): Args:             polygon_masks (list[list[ndarray]] or PolygonMasks)

### Community 1533 - "Community 1533"
Cohesion: 1.0
Nodes (1): Args:             roi_masks:             height, width (int):

### Community 1534 - "Community 1534"
Cohesion: 1.0
Nodes (1): Concatenates a list of BitMasks into a single BitMasks          Arguments:

### Community 1535 - "Community 1535"
Cohesion: 1.0
Nodes (1): Concatenates a list of PolygonMasks into a single PolygonMasks          Argument

### Community 1536 - "Community 1536"
Cohesion: 1.0
Nodes (1): Args: see documentation of :func:`paste_masks_in_image`.

### Community 1537 - "Community 1537"
Cohesion: 1.0
Nodes (1): Args:             tensors: a tuple or list of `torch.Tensor`, each of shape (Hi,

### Community 1538 - "Community 1538"
Cohesion: 1.0
Nodes (1): Concatenates a list of RotatedBoxes into a single RotatedBoxes          Argument

### Community 1539 - "Community 1539"
Cohesion: 1.0
Nodes (1): Yield a box as a Tensor of shape (5,) at a time.

### Community 1540 - "Community 1540"
Cohesion: 1.0
Nodes (1): Similar to :meth:`load()`, but load path relative to the caller's         source

### Community 1541 - "Community 1541"
Cohesion: 1.0
Nodes (1): Load a config file.          Args:             filename: absolute path or relati

### Community 1542 - "Community 1542"
Cohesion: 1.0
Nodes (1): Save a config object to a yaml file.         Note that when the config dictionar

### Community 1543 - "Community 1543"
Cohesion: 1.0
Nodes (1): In-place override contents of cfg.          Args:             cfg: an omegaconf

### Community 1544 - "Community 1544"
Cohesion: 1.0
Nodes (1): Try to convert a config object into Python-like psuedo code.          Note that

### Community 1545 - "Community 1545"
Cohesion: 1.0
Nodes (1): Returns:             int: The current iteration number. When used together with

### Community 1546 - "Community 1546"
Cohesion: 1.0
Nodes (1): Yields:             A context within which all the events added to this storage

### Community 1547 - "Community 1547"
Cohesion: 1.0
Nodes (1): Args:             config_path: relative config filename

### Community 1548 - "Community 1548"
Cohesion: 1.0
Nodes (1): Open a context where some heads in `model.roi_heads` are temporarily turned off.

### Community 1549 - "Community 1549"
Cohesion: 1.0
Nodes (1): This interface is experimental.          Args:             sizes (list[list[floa

### Community 1550 - "Community 1550"
Cohesion: 1.0
Nodes (1): Alias of `num_anchors`.

### Community 1551 - "Community 1551"
Cohesion: 1.0
Nodes (1): Returns:             list[int]: Each int is the number of anchors at every pixel

### Community 1552 - "Community 1552"
Cohesion: 1.0
Nodes (1): This interface is experimental.          Args:             sizes (list[list[floa

### Community 1553 - "Community 1553"
Cohesion: 1.0
Nodes (1): Alias of `num_anchors`.

### Community 1554 - "Community 1554"
Cohesion: 1.0
Nodes (1): Returns:             list[int]: Each int is the number of anchors at every pixel

### Community 1555 - "Community 1555"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1556 - "Community 1556"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 1557 - "Community 1557"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1558 - "Community 1558"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             sem_seg_head: a

### Community 1559 - "Community 1559"
Cohesion: 1.0
Nodes (1): Match ground-truth boxes to a set of multi-level anchors.          Args:

### Community 1560 - "Community 1560"
Cohesion: 1.0
Nodes (1): Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS

### Community 1561 - "Community 1561"
Cohesion: 1.0
Nodes (1): Args:             anchors (list[Boxes]): A list of #feature level Boxes.

### Community 1562 - "Community 1562"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Li

### Community 1563 - "Community 1563"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 1564 - "Community 1564"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape: sh

### Community 1565 - "Community 1565"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_channels (in

### Community 1566 - "Community 1566"
Cohesion: 1.0
Nodes (1): Args:             anchors (list[Boxes]): anchors for each feature map.

### Community 1567 - "Community 1567"
Cohesion: 1.0
Nodes (1): Return the losses from a set of RPN predictions and their associated ground-trut

### Community 1568 - "Community 1568"
Cohesion: 1.0
Nodes (1): Args:             anchors (list[RotatedBoxes]): anchors for each feature map.

### Community 1569 - "Community 1569"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             loss_weight (fl

### Community 1570 - "Community 1570"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1571 - "Community 1571"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1572 - "Community 1572"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1573 - "Community 1573"
Cohesion: 1.0
Nodes (1): Returns:             ShapeSpec: the output feature shape

### Community 1574 - "Community 1574"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_keypoints (

### Community 1575 - "Community 1575"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 1576 - "Community 1576"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.

### Community 1577 - "Community 1577"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the RROI heads.         It performs b

### Community 1578 - "Community 1578"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_pooler (ROI

### Community 1579 - "Community 1579"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage.         L

### Community 1580 - "Community 1580"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 1581 - "Community 1581"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads.         It performs bo

### Community 1582 - "Community 1582"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 1583 - "Community 1583"
Cohesion: 1.0
Nodes (1): This property is a generalization of size_divisibility. Some backbones and train

### Community 1584 - "Community 1584"
Cohesion: 1.0
Nodes (1): Create a list of blocks of the same type that forms one ResNet stage.          A

### Community 1585 - "Community 1585"
Cohesion: 1.0
Nodes (1): Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 15

### Community 1586 - "Community 1586"
Cohesion: 1.0
Nodes (1): Args:         video_height: height the video frame         video_width: width of

### Community 1587 - "Community 1587"
Cohesion: 1.0
Nodes (1): Old style initialization using CfgNode          Args:             cfg: D2 CfgNod

### Community 1588 - "Community 1588"
Cohesion: 1.0
Nodes (1): Args:         video_height: height the video frame         video_width: width of

### Community 1589 - "Community 1589"
Cohesion: 1.0
Nodes (1): Old style initialization using CfgNode          Args:             cfg: D2 CfgNod

### Community 1590 - "Community 1590"
Cohesion: 1.0
Nodes (1): Args:         video_height: height the video frame         video_width: width of

### Community 1591 - "Community 1591"
Cohesion: 1.0
Nodes (1): Args:         video_height: height the video frame         video_width: width of

### Community 1592 - "Community 1592"
Cohesion: 1.0
Nodes (1): Old style initialization using CfgNode          Args:             cfg: D2 CfgNod

### Community 1593 - "Community 1593"
Cohesion: 1.0
Nodes (1): Convert InstancesList to List[Instances]. The input `instances_list` can

### Community 1594 - "Community 1594"
Cohesion: 1.0
Nodes (1): Patching several inference functions inside ROIHeads and its subclasses

### Community 1595 - "Community 1595"
Cohesion: 1.0
Nodes (1): Creates a function that converts outputs of the caffe2 model to         detectro

### Community 1596 - "Community 1596"
Cohesion: 1.0
Nodes (1): caffe2.core.Net: the underlying caffe2 predict net

### Community 1597 - "Community 1597"
Cohesion: 1.0
Nodes (1): caffe2.core.Net: the underlying caffe2 init net

### Community 1598 - "Community 1598"
Cohesion: 1.0
Nodes (1): Args:             dir (str): a directory used to save Caffe2Model with

### Community 1599 - "Community 1599"
Cohesion: 1.0
Nodes (1): Args:             short_edge_length (list[int]): If ``sample_style=="range"``,

### Community 1600 - "Community 1600"
Cohesion: 1.0
Nodes (1): Compute the output size given input size and target short edge length.

### Community 1601 - "Community 1601"
Cohesion: 1.0
Nodes (1): Compute (fractional) per-image repeat factors based on category frequency.

### Community 1602 - "Community 1602"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 1603 - "Community 1603"
Cohesion: 1.0
Nodes (1): Returns:             torch.optim.Optimizer:          It now calls :func:`detectr

### Community 1604 - "Community 1604"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 1605 - "Community 1605"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 1606 - "Community 1606"
Cohesion: 1.0
Nodes (0): 

### Community 1607 - "Community 1607"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 1608 - "Community 1608"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1609 - "Community 1609"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 1610 - "Community 1610"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 1611 - "Community 1611"
Cohesion: 1.0
Nodes (1): Uses iterative furthest point sampling to select a set of npoint features that h

### Community 1612 - "Community 1612"
Cohesion: 1.0
Nodes (1): :param ctx:         :param features: (B, C, N)         :param idx: (B, npoint) i

### Community 1613 - "Community 1613"
Cohesion: 1.0
Nodes (1): Find the three nearest neighbors of unknown in known         :param ctx:

### Community 1614 - "Community 1614"
Cohesion: 1.0
Nodes (1): Find the three nearest neighbors of unknown in known         :param ctx:

### Community 1615 - "Community 1615"
Cohesion: 1.0
Nodes (1): Performs weight linear interpolation on 3 features         :param ctx:         :

### Community 1616 - "Community 1616"
Cohesion: 1.0
Nodes (1): :param ctx:         :param grad_out: (B, C, N) tensor with gradients of outputs

### Community 1617 - "Community 1617"
Cohesion: 1.0
Nodes (1): :param ctx:         :param features: (B, C, N) tensor of features to group

### Community 1618 - "Community 1618"
Cohesion: 1.0
Nodes (1): :param ctx:         :param grad_out: (B, C, npoint, nsample) tensor of the gradi

### Community 1619 - "Community 1619"
Cohesion: 1.0
Nodes (1): :param ctx:         :param radius: float, radius of the balls         :param nsa

### Community 1620 - "Community 1620"
Cohesion: 1.0
Nodes (1): Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`.

### Community 1621 - "Community 1621"
Cohesion: 1.0
Nodes (1): Set beta (or alpha as makes sense for given optimizer).

### Community 1622 - "Community 1622"
Cohesion: 1.0
Nodes (1): Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`.

### Community 1623 - "Community 1623"
Cohesion: 1.0
Nodes (1): To support a custom dataset, implement this function to receive the predicted re

### Community 1624 - "Community 1624"
Cohesion: 1.0
Nodes (1): Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,

### Community 1625 - "Community 1625"
Cohesion: 1.0
Nodes (1): Args:             pts_rect:             img_shape:             calib:          R

### Community 1626 - "Community 1626"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 frame_id:             pred_dicts:

### Community 1627 - "Community 1627"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 frame_id:             pred_dicts:

### Community 1628 - "Community 1628"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 frame_id:             pred_dicts:

### Community 1629 - "Community 1629"
Cohesion: 1.0
Nodes (1): Args:             pts_rect:             img_shape:             cam_intrinsic:

### Community 1630 - "Community 1630"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 frame_id:             pred_dicts:

### Community 1631 - "Community 1631"
Cohesion: 1.0
Nodes (1): Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu

### Community 1632 - "Community 1632"
Cohesion: 1.0
Nodes (1): Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu

### Community 1633 - "Community 1633"
Cohesion: 1.0
Nodes (1): PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:

### Community 1634 - "Community 1634"
Cohesion: 1.0
Nodes (1): Args:             x: x.features (N, C1)             out_channels: C2          Re

### Community 1635 - "Community 1635"
Cohesion: 1.0
Nodes (1): Args:             cls_scores: (N)             iou_scores: (N)             num_po

### Community 1636 - "Community 1636"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 batch_size:                 batch_

### Community 1637 - "Community 1637"
Cohesion: 1.0
Nodes (1): Args:             rois: (N, 7)             roi_labels: (N)             gt_boxes:

### Community 1638 - "Community 1638"
Cohesion: 1.0
Nodes (1): Args:             ctx:             radius: float, radius of the balls

### Community 1639 - "Community 1639"
Cohesion: 1.0
Nodes (1): :param ctx:         :param features: (B, C, N)         :param idx: (B, npoint) i

### Community 1640 - "Community 1640"
Cohesion: 1.0
Nodes (1): Find the three nearest neighbors of unknown in known         :param ctx:

### Community 1641 - "Community 1641"
Cohesion: 1.0
Nodes (1): Performs weight linear interpolation on 3 features         :param ctx:         :

### Community 1642 - "Community 1642"
Cohesion: 1.0
Nodes (1): :param ctx:         :param grad_out: (B, C, N) tensor with gradients of outputs

### Community 1643 - "Community 1643"
Cohesion: 1.0
Nodes (1): :param ctx:         :param features: (B, C, N) tensor of features to group

### Community 1644 - "Community 1644"
Cohesion: 1.0
Nodes (1): :param ctx:         :param radius: float, radius of the balls         :param nsa

### Community 1645 - "Community 1645"
Cohesion: 1.0
Nodes (1): Args:             ctx:             max_range: int, max range of voxels to be gro

### Community 1646 - "Community 1646"
Cohesion: 1.0
Nodes (1): Args:             ctx:             features: (N1 + N2 ..., C) tensor of features

### Community 1647 - "Community 1647"
Cohesion: 1.0
Nodes (1): Args:             ctx:             xyz: (B, N, 3) where N > npoint             n

### Community 1648 - "Community 1648"
Cohesion: 1.0
Nodes (1): Args:             ctx:             xyz: (N1 + N2 + ..., 3) where N > npoint

### Community 1649 - "Community 1649"
Cohesion: 1.0
Nodes (1): Args:             ctx:             unknown: (N1 + N2..., 3)             unknown_

### Community 1650 - "Community 1650"
Cohesion: 1.0
Nodes (1): Args:             ctx:             grad_out: (N1 + N2 ..., C)          Returns:

### Community 1651 - "Community 1651"
Cohesion: 1.0
Nodes (1): Args:             ctx:             points: (B, N, 3)             point_features:

### Community 1652 - "Community 1652"
Cohesion: 1.0
Nodes (1): Args:             ctx:             rois: (N, 7) [x, y, z, dx, dy, dz, heading] (

### Community 1653 - "Community 1653"
Cohesion: 1.0
Nodes (1): :param grad_out: (N, out_x, out_y, out_z, C)         :return:             grad_i

### Community 1654 - "Community 1654"
Cohesion: 1.0
Nodes (0): 

### Community 1655 - "Community 1655"
Cohesion: 1.0
Nodes (1): Logs the global norm of all parameters and of their gradients.

### Community 1656 - "Community 1656"
Cohesion: 1.0
Nodes (1): Logs the global norm of parameters and their gradients, by group.

### Community 1657 - "Community 1657"
Cohesion: 1.0
Nodes (0): 

### Community 1658 - "Community 1658"
Cohesion: 1.0
Nodes (1): List of scalar model parameters that should be logged.          They must be in

### Community 1659 - "Community 1659"
Cohesion: 1.0
Nodes (1): Parameter groups whose norm and gradient norm will be logged separately to tenso

### Community 1660 - "Community 1660"
Cohesion: 1.0
Nodes (1): Number of slots used for representation.          By default, it is equal to the

### Community 1661 - "Community 1661"
Cohesion: 1.0
Nodes (1): Representation size per slot.          This does not apply to models that are no

### Community 1662 - "Community 1662"
Cohesion: 1.0
Nodes (1): Stick breaking process to produce masks         :param: masks (B, K, 1, H, W). I

### Community 1663 - "Community 1663"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 1664 - "Community 1664"
Cohesion: 1.0
Nodes (1): Parses filter string into the corresponding parsing tree.

### Community 1665 - "Community 1665"
Cohesion: 1.0
Nodes (0): 

### Community 1666 - "Community 1666"
Cohesion: 1.0
Nodes (0): 

### Community 1667 - "Community 1667"
Cohesion: 1.0
Nodes (1): list[float]: Size of a single voxel.

### Community 1668 - "Community 1668"
Cohesion: 1.0
Nodes (1): int: Maximum number of points per voxel.

### Community 1669 - "Community 1669"
Cohesion: 1.0
Nodes (1): list[float]: Range of point cloud.

### Community 1670 - "Community 1670"
Cohesion: 1.0
Nodes (1): np.ndarray: The size of grids.

### Community 1671 - "Community 1671"
Cohesion: 1.0
Nodes (1): torch.Tensor: Coordinates of each point in shape (N, 3).

### Community 1672 - "Community 1672"
Cohesion: 1.0
Nodes (1): Set the coordinates of each point.

### Community 1673 - "Community 1673"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with height of each point in shape (N, 1), or

### Community 1674 - "Community 1674"
Cohesion: 1.0
Nodes (1): Set the height of each point.

### Community 1675 - "Community 1675"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with color of each point in shape (N, 3), or

### Community 1676 - "Community 1676"
Cohesion: 1.0
Nodes (1): Set the color of each point.

### Community 1677 - "Community 1677"
Cohesion: 1.0
Nodes (1): torch.Shape: Shape of points.

### Community 1678 - "Community 1678"
Cohesion: 1.0
Nodes (1): Flip the points along given BEV direction.          Args:             bev_direct

### Community 1679 - "Community 1679"
Cohesion: 1.0
Nodes (1): torch.Tensor: BEV of the points in shape (N, 2).

### Community 1680 - "Community 1680"
Cohesion: 1.0
Nodes (1): Convert self to ``dst`` mode.          Args:             dst (:obj:`CoordMode`):

### Community 1681 - "Community 1681"
Cohesion: 1.0
Nodes (1): Concatenate a list of Points into a single Points.          Args:             po

### Community 1682 - "Community 1682"
Cohesion: 1.0
Nodes (1): str: The device of the points are on.

### Community 1683 - "Community 1683"
Cohesion: 1.0
Nodes (1): torch.Tensor: BEV of the points in shape (N, 2).

### Community 1684 - "Community 1684"
Cohesion: 1.0
Nodes (1): list[int]: Total number of base anchors in a feature grid.

### Community 1685 - "Community 1685"
Cohesion: 1.0
Nodes (1): int: Number of feature levels that the generator is applied to.

### Community 1686 - "Community 1686"
Cohesion: 1.0
Nodes (1): Get box regression transformation deltas (dx, dy, dz, dx_size,         dy_size,

### Community 1687 - "Community 1687"
Cohesion: 1.0
Nodes (1): Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,         dz_size, dr

### Community 1688 - "Community 1688"
Cohesion: 1.0
Nodes (1): Decode yaw angle and change it from local to global.i.          Args:

### Community 1689 - "Community 1689"
Cohesion: 1.0
Nodes (1): Convert boxes from `src` mode to `dst` mode.          Args:             box (tup

### Community 1690 - "Community 1690"
Cohesion: 1.0
Nodes (1): torch.Tensor: A vector with height of each box in shape (N, ).

### Community 1691 - "Community 1691"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with the top height of each box in shape (N,

### Community 1692 - "Community 1692"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with bottom's height of each box in shape (N,

### Community 1693 - "Community 1693"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with local yaw of each box in shape (N, ).

### Community 1694 - "Community 1694"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor with center of each box in shape (N, 3).

### Community 1695 - "Community 1695"
Cohesion: 1.0
Nodes (1): torch.Tensor: Coordinates of corners of all the boxes in

### Community 1696 - "Community 1696"
Cohesion: 1.0
Nodes (1): torch.Tensor: 2D BEV box of each box with rotation             in XYWHR format,

### Community 1697 - "Community 1697"
Cohesion: 1.0
Nodes (1): Calculate height overlaps of two boxes.          This function calculates the he

### Community 1698 - "Community 1698"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor with center of each box in shape (N, 3).

### Community 1699 - "Community 1699"
Cohesion: 1.0
Nodes (1): torch.Tensor: Coordinates of corners of all the boxes         in shape (N, 8, 3)

### Community 1700 - "Community 1700"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor with center of each box in shape (N, 3).

### Community 1701 - "Community 1701"
Cohesion: 1.0
Nodes (1): torch.Tensor: Coordinates of corners of all the boxes         in shape (N, 8, 3)

### Community 1702 - "Community 1702"
Cohesion: 1.0
Nodes (1): Convert boxes or points from `src` mode to `dst` mode.          Args:

### Community 1703 - "Community 1703"
Cohesion: 1.0
Nodes (1): Convert boxes from `src` mode to `dst` mode.          Args:             box (tup

### Community 1704 - "Community 1704"
Cohesion: 1.0
Nodes (1): Convert points from `src` mode to `dst` mode.          Args:             point (

### Community 1705 - "Community 1705"
Cohesion: 1.0
Nodes (1): torch.Tensor: A vector with volume of each box.

### Community 1706 - "Community 1706"
Cohesion: 1.0
Nodes (1): torch.Tensor: Size dimensions of each box in shape (N, 3).

### Community 1707 - "Community 1707"
Cohesion: 1.0
Nodes (1): torch.Tensor: A vector with yaw of each box in shape (N, ).

### Community 1708 - "Community 1708"
Cohesion: 1.0
Nodes (1): torch.Tensor: A vector with height of each box in shape (N, ).

### Community 1709 - "Community 1709"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with the top height of each box in shape (N,

### Community 1710 - "Community 1710"
Cohesion: 1.0
Nodes (1): torch.Tensor:             A vector with bottom's height of each box in shape (N,

### Community 1711 - "Community 1711"
Cohesion: 1.0
Nodes (1): Calculate the center of all the boxes.          Note:             In MMDetection

### Community 1712 - "Community 1712"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor with center of each box in shape (N, 3).

### Community 1713 - "Community 1713"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor with center of each box in shape (N, 3).

### Community 1714 - "Community 1714"
Cohesion: 1.0
Nodes (1): torch.Tensor:             a tensor with 8 corners of each box in shape (N, 8, 3)

### Community 1715 - "Community 1715"
Cohesion: 1.0
Nodes (1): torch.Tensor: 2D BEV box of each box with rotation             in XYWHR format,

### Community 1716 - "Community 1716"
Cohesion: 1.0
Nodes (1): torch.Tensor: A tensor of 2D BEV box of each box             without rotation.

### Community 1717 - "Community 1717"
Cohesion: 1.0
Nodes (1): Rotate boxes with points (optional) with the given angle or rotation         mat

### Community 1718 - "Community 1718"
Cohesion: 1.0
Nodes (1): Flip the boxes in BEV along given BEV direction.          Args:             bev_

### Community 1719 - "Community 1719"
Cohesion: 1.0
Nodes (1): Convert self to ``dst`` mode.          Args:             dst (:obj:`Box3DMode`):

### Community 1720 - "Community 1720"
Cohesion: 1.0
Nodes (1): Concatenate a list of Boxes into a single Boxes.          Args:             boxe

### Community 1721 - "Community 1721"
Cohesion: 1.0
Nodes (1): str: The device of the boxes are on.

### Community 1722 - "Community 1722"
Cohesion: 1.0
Nodes (1): Calculate height overlaps of two boxes.          Note:             This function

### Community 1723 - "Community 1723"
Cohesion: 1.0
Nodes (1): Calculate 3D overlaps of two boxes.          Note:             This function cal

### Community 1724 - "Community 1724"
Cohesion: 1.0
Nodes (1): Repeat x `num` times to form a list.

### Community 1725 - "Community 1725"
Cohesion: 1.0
Nodes (1): Get class names of current dataset.          Args:             classes (Sequence

### Community 1726 - "Community 1726"
Cohesion: 1.0
Nodes (1): Get axis_align_matrix from info. If not exist, return identity mat.          Arg

### Community 1727 - "Community 1727"
Cohesion: 1.0
Nodes (1): Filter ground truths by difficulties.          Args:             db_infos (dict)

### Community 1728 - "Community 1728"
Cohesion: 1.0
Nodes (1): Filter ground truths by number of points in the bbox.          Args:

### Community 1729 - "Community 1729"
Cohesion: 1.0
Nodes (1): Remove the points in the sampled bounding boxes.          Args:             poin

### Community 1730 - "Community 1730"
Cohesion: 1.0
Nodes (1): Compute loss.          Args:             bbox_preds (dict): Predictions from for

### Community 1731 - "Community 1731"
Cohesion: 1.0
Nodes (1): Convert the rotation difference to difference in sine function.          Args:

### Community 1732 - "Community 1732"
Cohesion: 1.0
Nodes (1): Calculate losses.          Args:             cls_scores (list[torch.Tensor]): Mu

### Community 1733 - "Community 1733"
Cohesion: 1.0
Nodes (1): Compute loss of the head.          Args:             cls_scores (list[Tensor]):

### Community 1734 - "Community 1734"
Cohesion: 1.0
Nodes (1): Transform network output for a batch into bbox predictions.          Args:

### Community 1735 - "Community 1735"
Cohesion: 1.0
Nodes (1): Compute regression, classification and centerss targets for points         in mu

### Community 1736 - "Community 1736"
Cohesion: 1.0
Nodes (1): Construct Conv-Norm-Act block.          Args:             in_channels (int): Num

### Community 1737 - "Community 1737"
Cohesion: 1.0
Nodes (1): Construct DeConv-Norm-Act-Conv-Norm-Act block.          Args:             in_cha

### Community 1738 - "Community 1738"
Cohesion: 1.0
Nodes (1): Transform box to the axis-aligned or rotated iou loss format.          Args:

### Community 1739 - "Community 1739"
Cohesion: 1.0
Nodes (1): Transform predicted bbox parameters to bbox.          Args:             points (

### Community 1740 - "Community 1740"
Cohesion: 1.0
Nodes (1): Calculate distances from point to box faces.          Args:             points (

### Community 1741 - "Community 1741"
Cohesion: 1.0
Nodes (1): Compute point centerness w.r.t containing box.          Args:             face_d

### Community 1742 - "Community 1742"
Cohesion: 1.0
Nodes (1): Compute targets for final locations for a single scene.          Args:

### Community 1743 - "Community 1743"
Cohesion: 1.0
Nodes (1): Loss function for CenterHead.          Args:             gt_bboxes_3d (list[:obj

### Community 1744 - "Community 1744"
Cohesion: 1.0
Nodes (1): Upsample valid mask predictions.          Args:             valid_pred (Tensor):

### Community 1745 - "Community 1745"
Cohesion: 1.0
Nodes (1): Transform predicted bbox parameters to bbox.          Args:             points (

### Community 1746 - "Community 1746"
Cohesion: 1.0
Nodes (1): Calculate distances from point to box faces.          Args:             points (

### Community 1747 - "Community 1747"
Cohesion: 1.0
Nodes (1): Compute point centerness w.r.t containing box.          Args:             face_d

### Community 1748 - "Community 1748"
Cohesion: 1.0
Nodes (1): Compute targets for final locations for a single scene.          Args:

### Community 1749 - "Community 1749"
Cohesion: 1.0
Nodes (1): Calculate loss of FreeAnchor head.          Args:             cls_scores (list[t

### Community 1750 - "Community 1750"
Cohesion: 1.0
Nodes (1): Compute loss.          Args:             bbox_preds (dict): Predictions from for

### Community 1751 - "Community 1751"
Cohesion: 1.0
Nodes (1): Compute loss.          Args:             bbox_preds (dict): Predictions from for

### Community 1752 - "Community 1752"
Cohesion: 1.0
Nodes (1): Calculate losses.          Args:             cls_scores (list[torch.Tensor]): Mu

### Community 1753 - "Community 1753"
Cohesion: 1.0
Nodes (1): Compute loss.          Args:             bbox_preds (dict): Predictions from for

### Community 1754 - "Community 1754"
Cohesion: 1.0
Nodes (1): Compute losses of the head.

### Community 1755 - "Community 1755"
Cohesion: 1.0
Nodes (1): Transform network output for a batch into bbox predictions.

### Community 1756 - "Community 1756"
Cohesion: 1.0
Nodes (1): Compute loss of the head.          Args:             cls_scores (list[Tensor]):

### Community 1757 - "Community 1757"
Cohesion: 1.0
Nodes (1): Transform network output for a batch into bbox predictions.          Args:

### Community 1758 - "Community 1758"
Cohesion: 1.0
Nodes (1): Convert the rotation difference to difference in sine function.          Args:

### Community 1759 - "Community 1759"
Cohesion: 1.0
Nodes (1): Encode direction to 0 ~ num_bins-1.          Args:             reg_targets (torc

### Community 1760 - "Community 1760"
Cohesion: 1.0
Nodes (1): Compute loss of the head.          Args:             cls_scores (list[Tensor]):

### Community 1761 - "Community 1761"
Cohesion: 1.0
Nodes (1): Transform network output for a batch into bbox predictions.          Args:

### Community 1762 - "Community 1762"
Cohesion: 1.0
Nodes (1): Args:             points (torch.Tensor): points in 2D images, [N, 3],

### Community 1763 - "Community 1763"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             x (torch.Tensor): 4D Tensor in (N,

### Community 1764 - "Community 1764"
Cohesion: 1.0
Nodes (1): Forward function.          All inputs should be sorted by the rank of voxels.

### Community 1765 - "Community 1765"
Cohesion: 1.0
Nodes (1): Backward propagation function.          Args:             gradx (torch.tensor):

### Community 1766 - "Community 1766"
Cohesion: 1.0
Nodes (1): Make a layer from several residual blocks.          Args:             stride (in

### Community 1767 - "Community 1767"
Cohesion: 1.0
Nodes (1): Make a convolutional block.          Args:             in_channels (int): Number

### Community 1768 - "Community 1768"
Cohesion: 1.0
Nodes (1): Make upsampling convolutional block.          Args:             in_channels (int

### Community 1769 - "Community 1769"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             features (torch.Tensor): Point feat

### Community 1770 - "Community 1770"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             features (torch.Tensor): Point feat

### Community 1771 - "Community 1771"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             inputs (torch.Tensor): Pillar/Voxel

### Community 1772 - "Community 1772"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             features (torch.Tensor): Point feat

### Community 1773 - "Community 1773"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             features (torch.Tensor): Point feat

### Community 1774 - "Community 1774"
Cohesion: 1.0
Nodes (1): Forward functions.          Args:             features (torch.Tensor): Features

### Community 1775 - "Community 1775"
Cohesion: 1.0
Nodes (1): Forward functions.          Args:             features (torch.Tensor): Features

### Community 1776 - "Community 1776"
Cohesion: 1.0
Nodes (1): Forward pass.          Args:             points (torch.Tensor): point coordinate

### Community 1777 - "Community 1777"
Cohesion: 1.0
Nodes (1): Split coordinates and features of input points.          Args:             point

### Community 1778 - "Community 1778"
Cohesion: 1.0
Nodes (1): Forward pass.          Args:             points (torch.Tensor): point coordinate

### Community 1779 - "Community 1779"
Cohesion: 1.0
Nodes (1): Forward pass.          Args:             points (torch.Tensor): point coordinate

### Community 1780 - "Community 1780"
Cohesion: 1.0
Nodes (1): Forward pass.          Args:             points (torch.Tensor): point coordinate

### Community 1781 - "Community 1781"
Cohesion: 1.0
Nodes (1): Apply dynamic voxelization to points.          Args:             points (list[to

### Community 1782 - "Community 1782"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D image box head.

### Community 1783 - "Community 1783"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D image box head (not roi).

### Community 1784 - "Community 1784"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D image backbone.

### Community 1785 - "Community 1785"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a neck in image branch.

### Community 1786 - "Community 1786"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D RPN in image detector branch.

### Community 1787 - "Community 1787"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a RoI Head in image branch.

### Community 1788 - "Community 1788"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 3D box head.

### Community 1789 - "Community 1789"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 3D backbone.

### Community 1790 - "Community 1790"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a neck in 3D detector branch.

### Community 1791 - "Community 1791"
Cohesion: 1.0
Nodes (1): Extract bounding boxes from 2d detector.          Args:             img (torch.T

### Community 1792 - "Community 1792"
Cohesion: 1.0
Nodes (1): Apply dynamic voxelization to points.          Args:             points (list[to

### Community 1793 - "Community 1793"
Cohesion: 1.0
Nodes (1): Apply hard voxelization to points.

### Community 1794 - "Community 1794"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a shared head in image branch.

### Community 1795 - "Community 1795"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 3D box head.

### Community 1796 - "Community 1796"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D image box head.

### Community 1797 - "Community 1797"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D image backbone.

### Community 1798 - "Community 1798"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 3D backbone.

### Community 1799 - "Community 1799"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a fusion layer.

### Community 1800 - "Community 1800"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a neck in image branch.

### Community 1801 - "Community 1801"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a neck in 3D detector branch.

### Community 1802 - "Community 1802"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a 2D RPN in image detector branch.

### Community 1803 - "Community 1803"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a RoI Head in image branch.

### Community 1804 - "Community 1804"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a voxel encoder.

### Community 1805 - "Community 1805"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a middle encoder.

### Community 1806 - "Community 1806"
Cohesion: 1.0
Nodes (1): Apply dynamic voxelization to points.          Args:             points (list[to

### Community 1807 - "Community 1807"
Cohesion: 1.0
Nodes (1): bool: Whether the head predicts velocity

### Community 1808 - "Community 1808"
Cohesion: 1.0
Nodes (1): Apply hard voxelization to points.

### Community 1809 - "Community 1809"
Cohesion: 1.0
Nodes (1): Apply hard voxelization to points.

### Community 1810 - "Community 1810"
Cohesion: 1.0
Nodes (1): bool: whether the head has semantic branch

### Community 1811 - "Community 1811"
Cohesion: 1.0
Nodes (1): Assign and sample proposals for training.          Args:             proposal_li

### Community 1812 - "Community 1812"
Cohesion: 1.0
Nodes (1): bool: whether the RoIHead has box head

### Community 1813 - "Community 1813"
Cohesion: 1.0
Nodes (1): bool: whether the RoIHead has mask head

### Community 1814 - "Community 1814"
Cohesion: 1.0
Nodes (1): Initialize the box head.

### Community 1815 - "Community 1815"
Cohesion: 1.0
Nodes (1): Initialize maek head.

### Community 1816 - "Community 1816"
Cohesion: 1.0
Nodes (1): Initialize assigner and sampler.

### Community 1817 - "Community 1817"
Cohesion: 1.0
Nodes (1): Forward function during training.          Args:             x (dict): Contains

### Community 1818 - "Community 1818"
Cohesion: 1.0
Nodes (1): Generating model input.          Generate input by subtracting patch center and

### Community 1819 - "Community 1819"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has regularization loss for weight

### Community 1820 - "Community 1820"
Cohesion: 1.0
Nodes (1): Calls either forward_train or forward_test depending on whether         return_l

### Community 1821 - "Community 1821"
Cohesion: 1.0
Nodes (1): Forward of SparseEncoder.          Args:             voxel_features (torch.Tenso

### Community 1822 - "Community 1822"
Cohesion: 1.0
Nodes (1): Forward of SparseEncoder.          Args:             voxel_features (torch.Tenso

### Community 1823 - "Community 1823"
Cohesion: 1.0
Nodes (1): Forward of SparseUNet.          Args:             voxel_features (torch.float32)

### Community 1824 - "Community 1824"
Cohesion: 1.0
Nodes (1): reduce channel for element-wise addition.          Args:             x (:obj:`Sp

### Community 1825 - "Community 1825"
Cohesion: 1.0
Nodes (1): Forward function to scatter features.

### Community 1826 - "Community 1826"
Cohesion: 1.0
Nodes (1): Placeholder of forward function.

### Community 1827 - "Community 1827"
Cohesion: 1.0
Nodes (1): Compute semantic segmentation loss.          Args:             seg_logit (torch.

### Community 1828 - "Community 1828"
Cohesion: 1.0
Nodes (1): Args:             Input (tensor): Feature has shape (N, C, H, W).          Retur

### Community 1829 - "Community 1829"
Cohesion: 1.0
Nodes (1): forward.          Args:             points (Tensor): (B, N, C) tensor of the inp

### Community 1830 - "Community 1830"
Cohesion: 1.0
Nodes (1): forward.          Args:             points (List[Tensor]): tensor of the feature

### Community 1831 - "Community 1831"
Cohesion: 1.0
Nodes (1): forward.          Args:             target (Tensor): (B, n, 3) tensor of the xyz

### Community 1832 - "Community 1832"
Cohesion: 1.0
Nodes (0): 

### Community 1833 - "Community 1833"
Cohesion: 1.0
Nodes (0): 

### Community 1834 - "Community 1834"
Cohesion: 1.0
Nodes (0): 

### Community 1835 - "Community 1835"
Cohesion: 1.0
Nodes (0): 

### Community 1836 - "Community 1836"
Cohesion: 1.0
Nodes (0): 

### Community 1837 - "Community 1837"
Cohesion: 1.0
Nodes (0): 

### Community 1838 - "Community 1838"
Cohesion: 1.0
Nodes (0): 

### Community 1839 - "Community 1839"
Cohesion: 1.0
Nodes (0): 

### Community 1840 - "Community 1840"
Cohesion: 1.0
Nodes (0): 

### Community 1841 - "Community 1841"
Cohesion: 1.0
Nodes (0): 

### Community 1842 - "Community 1842"
Cohesion: 1.0
Nodes (0): 

### Community 1843 - "Community 1843"
Cohesion: 1.0
Nodes (0): 

### Community 1844 - "Community 1844"
Cohesion: 1.0
Nodes (0): 

### Community 1845 - "Community 1845"
Cohesion: 1.0
Nodes (0): 

### Community 1846 - "Community 1846"
Cohesion: 1.0
Nodes (0): 

### Community 1847 - "Community 1847"
Cohesion: 1.0
Nodes (0): 

### Community 1848 - "Community 1848"
Cohesion: 1.0
Nodes (0): 

### Community 1849 - "Community 1849"
Cohesion: 1.0
Nodes (0): 

### Community 1850 - "Community 1850"
Cohesion: 1.0
Nodes (0): 

### Community 1851 - "Community 1851"
Cohesion: 1.0
Nodes (0): 

### Community 1852 - "Community 1852"
Cohesion: 1.0
Nodes (0): 

### Community 1853 - "Community 1853"
Cohesion: 1.0
Nodes (0): 

### Community 1854 - "Community 1854"
Cohesion: 1.0
Nodes (0): 

### Community 1855 - "Community 1855"
Cohesion: 1.0
Nodes (0): 

### Community 1856 - "Community 1856"
Cohesion: 1.0
Nodes (0): 

### Community 1857 - "Community 1857"
Cohesion: 1.0
Nodes (0): 

### Community 1858 - "Community 1858"
Cohesion: 1.0
Nodes (0): 

### Community 1859 - "Community 1859"
Cohesion: 1.0
Nodes (0): 

### Community 1860 - "Community 1860"
Cohesion: 1.0
Nodes (0): 

### Community 1861 - "Community 1861"
Cohesion: 1.0
Nodes (0): 

### Community 1862 - "Community 1862"
Cohesion: 1.0
Nodes (0): 

### Community 1863 - "Community 1863"
Cohesion: 1.0
Nodes (0): 

### Community 1864 - "Community 1864"
Cohesion: 1.0
Nodes (0): 

### Community 1865 - "Community 1865"
Cohesion: 1.0
Nodes (0): 

### Community 1866 - "Community 1866"
Cohesion: 1.0
Nodes (0): 

### Community 1867 - "Community 1867"
Cohesion: 1.0
Nodes (0): 

### Community 1868 - "Community 1868"
Cohesion: 1.0
Nodes (0): 

### Community 1869 - "Community 1869"
Cohesion: 1.0
Nodes (0): 

### Community 1870 - "Community 1870"
Cohesion: 1.0
Nodes (0): 

### Community 1871 - "Community 1871"
Cohesion: 1.0
Nodes (0): 

### Community 1872 - "Community 1872"
Cohesion: 1.0
Nodes (0): 

### Community 1873 - "Community 1873"
Cohesion: 1.0
Nodes (0): 

### Community 1874 - "Community 1874"
Cohesion: 1.0
Nodes (0): 

### Community 1875 - "Community 1875"
Cohesion: 1.0
Nodes (0): 

### Community 1876 - "Community 1876"
Cohesion: 1.0
Nodes (0): 

### Community 1877 - "Community 1877"
Cohesion: 1.0
Nodes (0): 

### Community 1878 - "Community 1878"
Cohesion: 1.0
Nodes (0): 

### Community 1879 - "Community 1879"
Cohesion: 1.0
Nodes (0): 

### Community 1880 - "Community 1880"
Cohesion: 1.0
Nodes (0): 

### Community 1881 - "Community 1881"
Cohesion: 1.0
Nodes (0): 

### Community 1882 - "Community 1882"
Cohesion: 1.0
Nodes (0): 

### Community 1883 - "Community 1883"
Cohesion: 1.0
Nodes (0): 

### Community 1884 - "Community 1884"
Cohesion: 1.0
Nodes (0): 

### Community 1885 - "Community 1885"
Cohesion: 1.0
Nodes (0): 

### Community 1886 - "Community 1886"
Cohesion: 1.0
Nodes (0): 

### Community 1887 - "Community 1887"
Cohesion: 1.0
Nodes (0): 

### Community 1888 - "Community 1888"
Cohesion: 1.0
Nodes (0): 

### Community 1889 - "Community 1889"
Cohesion: 1.0
Nodes (0): 

### Community 1890 - "Community 1890"
Cohesion: 1.0
Nodes (0): 

### Community 1891 - "Community 1891"
Cohesion: 1.0
Nodes (0): 

### Community 1892 - "Community 1892"
Cohesion: 1.0
Nodes (0): 

### Community 1893 - "Community 1893"
Cohesion: 1.0
Nodes (0): 

### Community 1894 - "Community 1894"
Cohesion: 1.0
Nodes (0): 

### Community 1895 - "Community 1895"
Cohesion: 1.0
Nodes (0): 

### Community 1896 - "Community 1896"
Cohesion: 1.0
Nodes (0): 

### Community 1897 - "Community 1897"
Cohesion: 1.0
Nodes (0): 

### Community 1898 - "Community 1898"
Cohesion: 1.0
Nodes (0): 

### Community 1899 - "Community 1899"
Cohesion: 1.0
Nodes (0): 

### Community 1900 - "Community 1900"
Cohesion: 1.0
Nodes (0): 

### Community 1901 - "Community 1901"
Cohesion: 1.0
Nodes (0): 

### Community 1902 - "Community 1902"
Cohesion: 1.0
Nodes (0): 

### Community 1903 - "Community 1903"
Cohesion: 1.0
Nodes (0): 

### Community 1904 - "Community 1904"
Cohesion: 1.0
Nodes (0): 

### Community 1905 - "Community 1905"
Cohesion: 1.0
Nodes (0): 

### Community 1906 - "Community 1906"
Cohesion: 1.0
Nodes (0): 

### Community 1907 - "Community 1907"
Cohesion: 1.0
Nodes (0): 

### Community 1908 - "Community 1908"
Cohesion: 1.0
Nodes (0): 

### Community 1909 - "Community 1909"
Cohesion: 1.0
Nodes (0): 

### Community 1910 - "Community 1910"
Cohesion: 1.0
Nodes (0): 

### Community 1911 - "Community 1911"
Cohesion: 1.0
Nodes (0): 

### Community 1912 - "Community 1912"
Cohesion: 1.0
Nodes (0): 

### Community 1913 - "Community 1913"
Cohesion: 1.0
Nodes (0): 

### Community 1914 - "Community 1914"
Cohesion: 1.0
Nodes (0): 

### Community 1915 - "Community 1915"
Cohesion: 1.0
Nodes (0): 

### Community 1916 - "Community 1916"
Cohesion: 1.0
Nodes (0): 

### Community 1917 - "Community 1917"
Cohesion: 1.0
Nodes (0): 

### Community 1918 - "Community 1918"
Cohesion: 1.0
Nodes (0): 

### Community 1919 - "Community 1919"
Cohesion: 1.0
Nodes (0): 

### Community 1920 - "Community 1920"
Cohesion: 1.0
Nodes (0): 

### Community 1921 - "Community 1921"
Cohesion: 1.0
Nodes (0): 

### Community 1922 - "Community 1922"
Cohesion: 1.0
Nodes (0): 

### Community 1923 - "Community 1923"
Cohesion: 1.0
Nodes (0): 

### Community 1924 - "Community 1924"
Cohesion: 1.0
Nodes (0): 

### Community 1925 - "Community 1925"
Cohesion: 1.0
Nodes (0): 

### Community 1926 - "Community 1926"
Cohesion: 1.0
Nodes (0): 

### Community 1927 - "Community 1927"
Cohesion: 1.0
Nodes (0): 

### Community 1928 - "Community 1928"
Cohesion: 1.0
Nodes (0): 

### Community 1929 - "Community 1929"
Cohesion: 1.0
Nodes (0): 

### Community 1930 - "Community 1930"
Cohesion: 1.0
Nodes (0): 

### Community 1931 - "Community 1931"
Cohesion: 1.0
Nodes (0): 

### Community 1932 - "Community 1932"
Cohesion: 1.0
Nodes (0): 

### Community 1933 - "Community 1933"
Cohesion: 1.0
Nodes (0): 

### Community 1934 - "Community 1934"
Cohesion: 1.0
Nodes (0): 

### Community 1935 - "Community 1935"
Cohesion: 1.0
Nodes (0): 

### Community 1936 - "Community 1936"
Cohesion: 1.0
Nodes (0): 

### Community 1937 - "Community 1937"
Cohesion: 1.0
Nodes (0): 

### Community 1938 - "Community 1938"
Cohesion: 1.0
Nodes (0): 

### Community 1939 - "Community 1939"
Cohesion: 1.0
Nodes (0): 

### Community 1940 - "Community 1940"
Cohesion: 1.0
Nodes (0): 

### Community 1941 - "Community 1941"
Cohesion: 1.0
Nodes (0): 

### Community 1942 - "Community 1942"
Cohesion: 1.0
Nodes (0): 

### Community 1943 - "Community 1943"
Cohesion: 1.0
Nodes (0): 

### Community 1944 - "Community 1944"
Cohesion: 1.0
Nodes (0): 

### Community 1945 - "Community 1945"
Cohesion: 1.0
Nodes (0): 

### Community 1946 - "Community 1946"
Cohesion: 1.0
Nodes (0): 

### Community 1947 - "Community 1947"
Cohesion: 1.0
Nodes (0): 

### Community 1948 - "Community 1948"
Cohesion: 1.0
Nodes (0): 

### Community 1949 - "Community 1949"
Cohesion: 1.0
Nodes (0): 

### Community 1950 - "Community 1950"
Cohesion: 1.0
Nodes (0): 

### Community 1951 - "Community 1951"
Cohesion: 1.0
Nodes (0): 

### Community 1952 - "Community 1952"
Cohesion: 1.0
Nodes (0): 

### Community 1953 - "Community 1953"
Cohesion: 1.0
Nodes (0): 

### Community 1954 - "Community 1954"
Cohesion: 1.0
Nodes (0): 

### Community 1955 - "Community 1955"
Cohesion: 1.0
Nodes (0): 

### Community 1956 - "Community 1956"
Cohesion: 1.0
Nodes (0): 

### Community 1957 - "Community 1957"
Cohesion: 1.0
Nodes (0): 

### Community 1958 - "Community 1958"
Cohesion: 1.0
Nodes (0): 

### Community 1959 - "Community 1959"
Cohesion: 1.0
Nodes (0): 

### Community 1960 - "Community 1960"
Cohesion: 1.0
Nodes (0): 

### Community 1961 - "Community 1961"
Cohesion: 1.0
Nodes (0): 

### Community 1962 - "Community 1962"
Cohesion: 1.0
Nodes (0): 

### Community 1963 - "Community 1963"
Cohesion: 1.0
Nodes (0): 

### Community 1964 - "Community 1964"
Cohesion: 1.0
Nodes (0): 

### Community 1965 - "Community 1965"
Cohesion: 1.0
Nodes (0): 

### Community 1966 - "Community 1966"
Cohesion: 1.0
Nodes (0): 

### Community 1967 - "Community 1967"
Cohesion: 1.0
Nodes (0): 

### Community 1968 - "Community 1968"
Cohesion: 1.0
Nodes (0): 

### Community 1969 - "Community 1969"
Cohesion: 1.0
Nodes (0): 

### Community 1970 - "Community 1970"
Cohesion: 1.0
Nodes (0): 

### Community 1971 - "Community 1971"
Cohesion: 1.0
Nodes (0): 

### Community 1972 - "Community 1972"
Cohesion: 1.0
Nodes (0): 

### Community 1973 - "Community 1973"
Cohesion: 1.0
Nodes (0): 

### Community 1974 - "Community 1974"
Cohesion: 1.0
Nodes (0): 

### Community 1975 - "Community 1975"
Cohesion: 1.0
Nodes (0): 

### Community 1976 - "Community 1976"
Cohesion: 1.0
Nodes (0): 

### Community 1977 - "Community 1977"
Cohesion: 1.0
Nodes (0): 

### Community 1978 - "Community 1978"
Cohesion: 1.0
Nodes (0): 

### Community 1979 - "Community 1979"
Cohesion: 1.0
Nodes (0): 

### Community 1980 - "Community 1980"
Cohesion: 1.0
Nodes (0): 

### Community 1981 - "Community 1981"
Cohesion: 1.0
Nodes (0): 

### Community 1982 - "Community 1982"
Cohesion: 1.0
Nodes (0): 

### Community 1983 - "Community 1983"
Cohesion: 1.0
Nodes (0): 

### Community 1984 - "Community 1984"
Cohesion: 1.0
Nodes (0): 

### Community 1985 - "Community 1985"
Cohesion: 1.0
Nodes (0): 

### Community 1986 - "Community 1986"
Cohesion: 1.0
Nodes (0): 

### Community 1987 - "Community 1987"
Cohesion: 1.0
Nodes (0): 

### Community 1988 - "Community 1988"
Cohesion: 1.0
Nodes (0): 

### Community 1989 - "Community 1989"
Cohesion: 1.0
Nodes (0): 

### Community 1990 - "Community 1990"
Cohesion: 1.0
Nodes (0): 

### Community 1991 - "Community 1991"
Cohesion: 1.0
Nodes (0): 

### Community 1992 - "Community 1992"
Cohesion: 1.0
Nodes (0): 

### Community 1993 - "Community 1993"
Cohesion: 1.0
Nodes (0): 

### Community 1994 - "Community 1994"
Cohesion: 1.0
Nodes (0): 

### Community 1995 - "Community 1995"
Cohesion: 1.0
Nodes (0): 

### Community 1996 - "Community 1996"
Cohesion: 1.0
Nodes (0): 

### Community 1997 - "Community 1997"
Cohesion: 1.0
Nodes (0): 

### Community 1998 - "Community 1998"
Cohesion: 1.0
Nodes (0): 

### Community 1999 - "Community 1999"
Cohesion: 1.0
Nodes (0): 

### Community 2000 - "Community 2000"
Cohesion: 1.0
Nodes (1): Return state of constrain to axis mode.

### Community 2001 - "Community 2001"
Cohesion: 1.0
Nodes (1): Set state of constrain to axis mode.

### Community 2002 - "Community 2002"
Cohesion: 1.0
Nodes (0): 

### Community 2003 - "Community 2003"
Cohesion: 1.0
Nodes (1): Forward function.          Args:             x (torch.Tensor): 4D Tensor in (N,

### Community 2004 - "Community 2004"
Cohesion: 1.0
Nodes (1): bool: Whether the detector has a neck in 3D detector branch.

### Community 2005 - "Community 2005"
Cohesion: 1.0
Nodes (1): Apply dynamic voxelization to points.          Args:             points (list[to

### Community 2006 - "Community 2006"
Cohesion: 1.0
Nodes (1): Loss function for CenterHead.          Args:             gt_bboxes_3d (list[:obj

### Community 2007 - "Community 2007"
Cohesion: 1.0
Nodes (1): In the forward pass we receive a Tensor containing the input and return

### Community 2008 - "Community 2008"
Cohesion: 1.0
Nodes (1): Returns the folder where the tables are stored for the relevant version.

### Community 2009 - "Community 2009"
Cohesion: 1.0
Nodes (1): Returns the folder where the tables are stored for the relevant version.

### Community 2010 - "Community 2010"
Cohesion: 1.0
Nodes (1): Perform clipping on polygons that are partially behind the camera.         This

### Community 2011 - "Community 2011"
Cohesion: 1.0
Nodes (1): Convert a polygon or multipolygon list to an image mask ndarray.         :param

### Community 2012 - "Community 2012"
Cohesion: 1.0
Nodes (1): Convert a Shapely LineString back to an image mask ndarray.         :param lines

### Community 2013 - "Community 2013"
Cohesion: 1.0
Nodes (1): Convert patch_box to shapely Polygon coordinates.         :param patch_box: Patc

### Community 2014 - "Community 2014"
Cohesion: 1.0
Nodes (1): Check if any lanes are disconnected.

### Community 2015 - "Community 2015"
Cohesion: 1.0
Nodes (1): Computes the angle between the last points of the two trajectories.         The

### Community 2016 - "Community 2016"
Cohesion: 1.0
Nodes (1): Compute the average of l2 norms of each row in the tensor.         :param tensor

### Community 2017 - "Community 2017"
Cohesion: 1.0
Nodes (1): Mainly a smoke test since most of the logic is handled under-the-hood         by

### Community 2018 - "Community 2018"
Cohesion: 1.0
Nodes (1): Convert sample token into standard KITTI folder and local filename format.

### Community 2019 - "Community 2019"
Cohesion: 1.0
Nodes (1): Parses single line from label file into a dict. Boxes are in camera frame. See K

### Community 2020 - "Community 2020"
Cohesion: 1.0
Nodes (1): Transform from nuScenes lidar frame to KITTI reference frame.         :param box

### Community 2021 - "Community 2021"
Cohesion: 1.0
Nodes (1): Projects 3D box into KITTI image FOV.         :param box: 3D box in KITTI refere

### Community 2022 - "Community 2022"
Cohesion: 1.0
Nodes (1): For a token and table, get the filepath to the associated data.         :param t

### Community 2023 - "Community 2023"
Cohesion: 1.0
Nodes (1): Returns transforms for the input token.         :param token: KittiDB unique id.

### Community 2024 - "Community 2024"
Cohesion: 1.0
Nodes (1): Load up the pointcloud for a sample.         :param token: KittiDB unique id.

### Community 2025 - "Community 2025"
Cohesion: 1.0
Nodes (1): Convert box in KITTI image frame to official label string fromat.         :param

### Community 2026 - "Community 2026"
Cohesion: 1.0
Nodes (1): Returns the map mask, optionally dilated.         :param dilation: Dilation in m

### Community 2027 - "Community 2027"
Cohesion: 1.0
Nodes (1): Generate transform matrix for this map mask.         :return: <np.array: 4, 4>.

### Community 2028 - "Community 2028"
Cohesion: 1.0
Nodes (1): Returns the original binary mask stored in map png file.         :return: <np.in

### Community 2029 - "Community 2029"
Cohesion: 1.0
Nodes (1): Returns the number of dimensions.         :return: Number of dimensions.

### Community 2030 - "Community 2030"
Cohesion: 1.0
Nodes (1): Loads point cloud from disk.         :param file_name: Path of the pointcloud fi

### Community 2031 - "Community 2031"
Cohesion: 1.0
Nodes (1): Initialize from serialized dictionary.

### Community 2032 - "Community 2032"
Cohesion: 1.0
Nodes (1): Returns the number of dimensions.         :return: Number of dimensions.

### Community 2033 - "Community 2033"
Cohesion: 1.0
Nodes (1): Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity

### Community 2034 - "Community 2034"
Cohesion: 1.0
Nodes (1): Disable all radar filter settings.         Use this method to plot all radar ret

### Community 2035 - "Community 2035"
Cohesion: 1.0
Nodes (1): Set the defaults for all radar filter settings.         Note that this method af

### Community 2036 - "Community 2036"
Cohesion: 1.0
Nodes (1): Initialize from serialized dictionary.

### Community 2037 - "Community 2037"
Cohesion: 1.0
Nodes (1): Loads RADAR data from a Point Cloud Data file. See details below.         :param

### Community 2038 - "Community 2038"
Cohesion: 1.0
Nodes (1): Return a rotation matrix.         :return: <np.float: 3, 3>. The box's rotation

### Community 2039 - "Community 2039"
Cohesion: 1.0
Nodes (1): Loads the polygon representation of the drivable area for each map.         :par

### Community 2040 - "Community 2040"
Cohesion: 1.0
Nodes (1): Interpolate trajectory with a cubic spline if there are enough points.

### Community 2041 - "Community 2041"
Cohesion: 1.0
Nodes (1): Initialize from serialized dictionary.

### Community 2042 - "Community 2042"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2043 - "Community 2043"
Cohesion: 1.0
Nodes (1): Filters the point cloud such that only points which are within a certain radial

### Community 2044 - "Community 2044"
Cohesion: 1.0
Nodes (1): Compute the distance from this box to the ego vehicle in 2D.

### Community 2045 - "Community 2045"
Cohesion: 1.0
Nodes (1): Returns all EvalBoxes in a list.

### Community 2046 - "Community 2046"
Cohesion: 1.0
Nodes (1): Returns a list of all keys.

### Community 2047 - "Community 2047"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.         :param content: A dictionary with th

### Community 2048 - "Community 2048"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2049 - "Community 2049"
Cohesion: 1.0
Nodes (1): Create a new DataFrame filled with data.         This version overwrites the ori

### Community 2050 - "Community 2050"
Cohesion: 1.0
Nodes (1): Create a new DataFrame for event tracking.

### Community 2051 - "Community 2051"
Cohesion: 1.0
Nodes (1): Merge dataframes.          Params         ------         dfs : list of pandas.Da

### Community 2052 - "Community 2052"
Cohesion: 1.0
Nodes (1): Initialize from serialized dictionary.

### Community 2053 - "Community 2053"
Cohesion: 1.0
Nodes (1): Return the distance function corresponding to the dist_fcn string.

### Community 2054 - "Community 2054"
Cohesion: 1.0
Nodes (1): Returns max recall achieved.

### Community 2055 - "Community 2055"
Cohesion: 1.0
Nodes (1): Returns max recall achieved.

### Community 2056 - "Community 2056"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2057 - "Community 2057"
Cohesion: 1.0
Nodes (1): Returns an md instance corresponding to having no predictions.

### Community 2058 - "Community 2058"
Cohesion: 1.0
Nodes (1): Returns an md instance corresponding to a random results.

### Community 2059 - "Community 2059"
Cohesion: 1.0
Nodes (1): Initialize from serialized dictionary.

### Community 2060 - "Community 2060"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2061 - "Community 2061"
Cohesion: 1.0
Nodes (1): Creates "reasonable" submission (results and metadata) by looping through the mi

### Community 2062 - "Community 2062"
Cohesion: 1.0
Nodes (1): Run the evaluation with fixed randomness on the specified subset, with or withou

### Community 2063 - "Community 2063"
Cohesion: 1.0
Nodes (1): This tests runs the evaluation for an arbitrary random set of predictions.

### Community 2064 - "Community 2064"
Cohesion: 1.0
Nodes (1): This tests runs the evaluation with the ground truth used as predictions.

### Community 2065 - "Community 2065"
Cohesion: 1.0
Nodes (1): Return the distance function corresponding to the dist_fcn string.

### Community 2066 - "Community 2066"
Cohesion: 1.0
Nodes (1): Returns index of max recall achieved.

### Community 2067 - "Community 2067"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2068 - "Community 2068"
Cohesion: 1.0
Nodes (1): Returns a md instance corresponding to having no predictions.

### Community 2069 - "Community 2069"
Cohesion: 1.0
Nodes (1): Returns an md instance corresponding to a random results.

### Community 2070 - "Community 2070"
Cohesion: 1.0
Nodes (1): Calculates the mean over distance thresholds for each label.

### Community 2071 - "Community 2071"
Cohesion: 1.0
Nodes (1): Calculates the mean AP by averaging over distance thresholds and classes.

### Community 2072 - "Community 2072"
Cohesion: 1.0
Nodes (1): Calculates the mean true positive error across all classes for each metric.

### Community 2073 - "Community 2073"
Cohesion: 1.0
Nodes (1): Compute the nuScenes detection score (NDS, weighted sum of the individual scores

### Community 2074 - "Community 2074"
Cohesion: 1.0
Nodes (1): Initialize from serialized content.

### Community 2075 - "Community 2075"
Cohesion: 1.0
Nodes (1): Creates "reasonable" submission (results and metadata) by looping through the mi

### Community 2076 - "Community 2076"
Cohesion: 1.0
Nodes (1): Update stats dict with new combo of ids and counts.         :param stat_dict: {c

### Community 2077 - "Community 2077"
Cohesion: 1.0
Nodes (1): Load a '.png' segmentation mask, ignoring any colour map.

### Community 2078 - "Community 2078"
Cohesion: 1.0
Nodes (1): Load a '.mat' segmentation mask of the kind used in the SBD dataset.

### Community 2079 - "Community 2079"
Cohesion: 1.0
Nodes (1): Fields that will be transformed with this transform.

### Community 2080 - "Community 2080"
Cohesion: 1.0
Nodes (1): Comput visualization output.          A visualization method takes some inputs a

### Community 2081 - "Community 2081"
Cohesion: 1.0
Nodes (1): Convert instance to segmentation mask.          Args:             instance_mask:

### Community 2082 - "Community 2082"
Cohesion: 1.0
Nodes (1): Return current value of hyperparameter based on global step.          Returns:

### Community 2083 - "Community 2083"
Cohesion: 1.0
Nodes (1): Computes the Hutchinson approximation of the hessian trace and accumulates it fo

### Community 2084 - "Community 2084"
Cohesion: 1.0
Nodes (1): Performs a single optimization step.         Arguments:             closure (cal

### Community 2085 - "Community 2085"
Cohesion: 1.0
Nodes (1): Get parameters for ``crop`` for a random sized crop.          Args:

### Community 2086 - "Community 2086"
Cohesion: 1.0
Nodes (0): 

### Community 2087 - "Community 2087"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 2088 - "Community 2088"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2089 - "Community 2089"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 2090 - "Community 2090"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 2091 - "Community 2091"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2092 - "Community 2092"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 2093 - "Community 2093"
Cohesion: 1.0
Nodes (1): Create evaluator(s) for a given dataset.         This uses the special metadata

### Community 2094 - "Community 2094"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2095 - "Community 2095"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 2096 - "Community 2096"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 2097 - "Community 2097"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2098 - "Community 2098"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 2099 - "Community 2099"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: whethe

### Community 2100 - "Community 2100"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             in_channels: cha

### Community 2101 - "Community 2101"
Cohesion: 1.0
Nodes (1): Decode the mask annotation         :param anno: The mask annotation         :ret

### Community 2102 - "Community 2102"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 2103 - "Community 2103"
Cohesion: 1.0
Nodes (1): Returns:             torch.optim.Optimizer:          It now calls :func:`detectr

### Community 2104 - "Community 2104"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2105 - "Community 2105"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 2106 - "Community 2106"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 2107 - "Community 2107"
Cohesion: 1.0
Nodes (1): Returns:             DatasetEvaluator or None          It is not implemented by

### Community 2108 - "Community 2108"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 2109 - "Community 2109"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 2110 - "Community 2110"
Cohesion: 1.0
Nodes (0): 

### Community 2111 - "Community 2111"
Cohesion: 1.0
Nodes (1): Preprocess Pascal VOC labels by converting to integer 255-scale and         mark

### Community 2112 - "Community 2112"
Cohesion: 1.0
Nodes (1): To support a custom dataset, implement this function to receive the predicted re

### Community 2113 - "Community 2113"
Cohesion: 1.0
Nodes (1): Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,

### Community 2114 - "Community 2114"
Cohesion: 1.0
Nodes (1): Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,

### Community 2115 - "Community 2115"
Cohesion: 1.0
Nodes (1): Args:             pts_rect:             img_shape:             calib:          R

### Community 2116 - "Community 2116"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 frame_id:             pred_dicts:

### Community 2117 - "Community 2117"
Cohesion: 1.0
Nodes (1): Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu

### Community 2118 - "Community 2118"
Cohesion: 1.0
Nodes (1): Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu

### Community 2119 - "Community 2119"
Cohesion: 1.0
Nodes (1): PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:

### Community 2120 - "Community 2120"
Cohesion: 1.0
Nodes (1): Args:             aggregate_func:             xyz: (N, 3)             xyz_featur

### Community 2121 - "Community 2121"
Cohesion: 1.0
Nodes (1): Args:             batch_dict:                 batch_size:                 batch_

### Community 2122 - "Community 2122"
Cohesion: 1.0
Nodes (1): Args:             rois: (N, 7)             roi_labels: (N)             gt_boxes:

### Community 2123 - "Community 2123"
Cohesion: 1.0
Nodes (1): Args:             ctx:             features: (M1 + M2 ..., C)             idx: [

### Community 2124 - "Community 2124"
Cohesion: 1.0
Nodes (1): Args:             ctx:             grad_out: (N1 + N2 ..., C)          Returns:

### Community 2125 - "Community 2125"
Cohesion: 1.0
Nodes (1): Args:             ctx:             // support_xyz: (N1 + N2 ..., 3) xyz coordina

### Community 2126 - "Community 2126"
Cohesion: 1.0
Nodes (1): Args:             ctx:             support_xyz: (N1 + N2 ..., 3) xyz coordinates

### Community 2127 - "Community 2127"
Cohesion: 1.0
Nodes (1): Args:             ctx:             grad_new_features: (M1 + M2 ..., num_c_out),

### Community 2128 - "Community 2128"
Cohesion: 1.0
Nodes (1): Args:             point_centers: (N, 3)             max_neighbour_distance: floa

### Community 2129 - "Community 2129"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 2130 - "Community 2130"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 2131 - "Community 2131"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 2132 - "Community 2132"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 2133 - "Community 2133"
Cohesion: 1.0
Nodes (1): Yield successive n-sized chunks from lst.

### Community 2134 - "Community 2134"
Cohesion: 1.0
Nodes (1): database file containing information about preproscessed dataset

### Community 2135 - "Community 2135"
Cohesion: 1.0
Nodes (1): database file containing information labels used by dataset

### Community 2136 - "Community 2136"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)

### Community 2137 - "Community 2137"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 2138 - "Community 2138"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 2139 - "Community 2139"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 2140 - "Community 2140"
Cohesion: 1.0
Nodes (1): input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output

### Community 2141 - "Community 2141"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2142 - "Community 2142"
Cohesion: 1.0
Nodes (1): input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output

### Community 2143 - "Community 2143"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2144 - "Community 2144"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o

### Community 2145 - "Community 2145"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2146 - "Community 2146"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o

### Community 2147 - "Community 2147"
Cohesion: 1.0
Nodes (1): input: grad_output: (L, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2148 - "Community 2148"
Cohesion: 1.0
Nodes (1): input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)

### Community 2149 - "Community 2149"
Cohesion: 1.0
Nodes (1): input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L

### Community 2150 - "Community 2150"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 2151 - "Community 2151"
Cohesion: 1.0
Nodes (1): input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L

### Community 2152 - "Community 2152"
Cohesion: 1.0
Nodes (1): input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),

### Community 2153 - "Community 2153"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hd

### Community 2154 - "Community 2154"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2155 - "Community 2155"
Cohesion: 1.0
Nodes (1): input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (

### Community 2156 - "Community 2156"
Cohesion: 1.0
Nodes (1): input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non

### Community 2157 - "Community 2157"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 2158 - "Community 2158"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 2159 - "Community 2159"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 2160 - "Community 2160"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n

### Community 2161 - "Community 2161"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2162 - "Community 2162"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2163 - "Community 2163"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 2164 - "Community 2164"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 2165 - "Community 2165"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 2166 - "Community 2166"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 2167 - "Community 2167"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input

### Community 2168 - "Community 2168"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 2169 - "Community 2169"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n

### Community 2170 - "Community 2170"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2171 - "Community 2171"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2172 - "Community 2172"
Cohesion: 1.0
Nodes (1): input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output

### Community 2173 - "Community 2173"
Cohesion: 1.0
Nodes (1): input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)

### Community 2174 - "Community 2174"
Cohesion: 1.0
Nodes (1): input: grad_out: (m, c, nsample)         output: (n, c), None

### Community 2175 - "Community 2175"
Cohesion: 1.0
Nodes (1): input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns

### Community 2176 - "Community 2176"
Cohesion: 1.0
Nodes (1): input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:

### Community 2177 - "Community 2177"
Cohesion: 1.0
Nodes (1): input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n

### Community 2178 - "Community 2178"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2179 - "Community 2179"
Cohesion: 1.0
Nodes (1): input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)

### Community 2180 - "Community 2180"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 2181 - "Community 2181"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 2182 - "Community 2182"
Cohesion: 1.0
Nodes (1): More memory-friendly matching

### Community 2183 - "Community 2183"
Cohesion: 1.0
Nodes (1): Performs the matching          Params:             outputs: This is a dict that

### Community 2184 - "Community 2184"
Cohesion: 1.0
Nodes (0): 

### Community 2185 - "Community 2185"
Cohesion: 1.0
Nodes (0): 

### Community 2186 - "Community 2186"
Cohesion: 1.0
Nodes (0): 

### Community 2187 - "Community 2187"
Cohesion: 1.0
Nodes (0): 

### Community 2188 - "Community 2188"
Cohesion: 1.0
Nodes (1): Calculates the image embeddings for the provided image, allowing         masks t

### Community 2189 - "Community 2189"
Cohesion: 1.0
Nodes (1): Predict masks for the given input prompts, using the currently set image.

### Community 2190 - "Community 2190"
Cohesion: 1.0
Nodes (1): Generates masks for the given image.          Arguments:           image (np.nda

### Community 2191 - "Community 2191"
Cohesion: 1.0
Nodes (1): Removes small disconnected regions and holes in masks, then reruns         box N

### Community 2192 - "Community 2192"
Cohesion: 1.0
Nodes (1): Predicts masks end-to-end from provided images and prompts.         If prompts a

### Community 2193 - "Community 2193"
Cohesion: 1.0
Nodes (0): 

### Community 2194 - "Community 2194"
Cohesion: 1.0
Nodes (1): Args:             values: tensor of shape (batch, n_true_classes, n_pred_classes

### Community 2195 - "Community 2195"
Cohesion: 1.0
Nodes (1): Compute auxilliary outputs only needed for metrics and visualisations.

### Community 2196 - "Community 2196"
Cohesion: 1.0
Nodes (1): Try to infer same padding for convolutions.

### Community 2197 - "Community 2197"
Cohesion: 1.0
Nodes (1): Try to infer same padding for transposed convolutions.

### Community 2198 - "Community 2198"
Cohesion: 1.0
Nodes (0): 

### Community 2199 - "Community 2199"
Cohesion: 1.0
Nodes (0): 

### Community 2200 - "Community 2200"
Cohesion: 1.0
Nodes (0): 

### Community 2201 - "Community 2201"
Cohesion: 1.0
Nodes (0): 

### Community 2202 - "Community 2202"
Cohesion: 1.0
Nodes (0): 

### Community 2203 - "Community 2203"
Cohesion: 1.0
Nodes (1): Warning: this function is expensive. Only call it when necessary to         visu

### Community 2204 - "Community 2204"
Cohesion: 1.0
Nodes (0): 

### Community 2205 - "Community 2205"
Cohesion: 1.0
Nodes (0): 

### Community 2206 - "Community 2206"
Cohesion: 1.0
Nodes (0): 

### Community 2207 - "Community 2207"
Cohesion: 1.0
Nodes (0): 

### Community 2208 - "Community 2208"
Cohesion: 1.0
Nodes (0): 

### Community 2209 - "Community 2209"
Cohesion: 1.0
Nodes (1): 128x128 -> proposal size in the original image         original_bboxes: [B, 4],

### Community 2210 - "Community 2210"
Cohesion: 1.0
Nodes (1): Separate binary masks into connected components and return their bounding boxes.

### Community 2211 - "Community 2211"
Cohesion: 1.0
Nodes (1): Enlarge bounding boxes by a common ratio, ensuring they stay within the image di

### Community 2212 - "Community 2212"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2213 - "Community 2213"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage.         L

### Community 2214 - "Community 2214"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 2215 - "Community 2215"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads.         It performs bo

### Community 2216 - "Community 2216"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_features (li

### Community 2217 - "Community 2217"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 2218 - "Community 2218"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 2219 - "Community 2219"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 2220 - "Community 2220"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 2221 - "Community 2221"
Cohesion: 1.0
Nodes (1): Returns:             iterable          It now calls :func:`detectron2.data.build

### Community 2222 - "Community 2222"
Cohesion: 1.0
Nodes (1): Returns:             DatasetEvaluator or None          It is not implemented by

### Community 2223 - "Community 2223"
Cohesion: 1.0
Nodes (1): Evaluate the given model. The given model is expected to already contain

### Community 2224 - "Community 2224"
Cohesion: 1.0
Nodes (1): When the config is defined for certain number of workers (according to         `

### Community 2225 - "Community 2225"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2226 - "Community 2226"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2227 - "Community 2227"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 2228 - "Community 2228"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             is_train: for tr

### Community 2229 - "Community 2229"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2230 - "Community 2230"
Cohesion: 1.0
Nodes (1): Rescale the output instances to the target size.

### Community 2231 - "Community 2231"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2232 - "Community 2232"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 2233 - "Community 2233"
Cohesion: 1.0
Nodes (1): Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo

### Community 2234 - "Community 2234"
Cohesion: 1.0
Nodes (1): It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it

### Community 2235 - "Community 2235"
Cohesion: 1.0
Nodes (1): Returns:             CfgNode: a new config. Same as original if ``cfg.SOLVER.REF

### Community 2236 - "Community 2236"
Cohesion: 1.0
Nodes (1): Generates masks for the given image.          Arguments:           image (np.nda

### Community 2237 - "Community 2237"
Cohesion: 1.0
Nodes (1): Removes small disconnected regions and holes in masks, then reruns         box N

### Community 2238 - "Community 2238"
Cohesion: 1.0
Nodes (1): Removes small disconnected regions and holes in a mask. Returns the         mask

### Community 2239 - "Community 2239"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             is_train: wheth

### Community 2240 - "Community 2240"
Cohesion: 1.0
Nodes (1): Args:             input_shape: shapes (channels and stride) of the input feature

### Community 2241 - "Community 2241"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             input_shape: sha

### Community 2242 - "Community 2242"
Cohesion: 1.0
Nodes (1): :param features: multi-scale features from the backbone         :param masks: im

### Community 2243 - "Community 2243"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.         Args:             in_channels: cha

### Community 2244 - "Community 2244"
Cohesion: 1.0
Nodes (1): Input:             - tgt/tgt_query_pos: nq, bs, d_model             -

### Community 2245 - "Community 2245"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2246 - "Community 2246"
Cohesion: 1.0
Nodes (1): Compute auxilliary outputs only needed for metrics and visualisations.

### Community 2247 - "Community 2247"
Cohesion: 1.0
Nodes (1): Try to infer same padding for convolutions.

### Community 2248 - "Community 2248"
Cohesion: 1.0
Nodes (1): Try to infer same padding for transposed convolutions.

### Community 2249 - "Community 2249"
Cohesion: 1.0
Nodes (1): Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B

### Community 2250 - "Community 2250"
Cohesion: 1.0
Nodes (1): Updates the internal state of the metric. In particular, we track update the cos

### Community 2251 - "Community 2251"
Cohesion: 1.0
Nodes (1): Remaps the semantic classes to the target class using the latest assignments.

### Community 2252 - "Community 2252"
Cohesion: 1.0
Nodes (1): Getter method to access things prototypes.          Returns:             things_

### Community 2253 - "Community 2253"
Cohesion: 1.0
Nodes (1): Getter method to access stuffs prototypes.          Returns:             things_

### Community 2254 - "Community 2254"
Cohesion: 1.0
Nodes (1): Setter method to access things prototypes.          Args:             value (Set

### Community 2255 - "Community 2255"
Cohesion: 1.0
Nodes (1): Setter method to access stuffs prototypes.          Args:             value (Set

### Community 2256 - "Community 2256"
Cohesion: 1.0
Nodes (1): Args:             backbone: a backbone module, must follow detectron2's backbone

### Community 2257 - "Community 2257"
Cohesion: 1.0
Nodes (1): Args:             min_sizes: list of short-edge size to resize the image to

### Community 2258 - "Community 2258"
Cohesion: 1.0
Nodes (1): Open a context where some heads in `model.roi_heads` are temporarily turned off.

### Community 2259 - "Community 2259"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_pooler (ROI

### Community 2260 - "Community 2260"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage. Label the

### Community 2261 - "Community 2261"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 2262 - "Community 2262"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 2263 - "Community 2263"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads. It performs box matchi

### Community 2264 - "Community 2264"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_features (li

### Community 2265 - "Community 2265"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 2266 - "Community 2266"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape: sh

### Community 2267 - "Community 2267"
Cohesion: 1.0
Nodes (0): 

### Community 2268 - "Community 2268"
Cohesion: 1.0
Nodes (0): 

### Community 2269 - "Community 2269"
Cohesion: 1.0
Nodes (0): 

### Community 2270 - "Community 2270"
Cohesion: 1.0
Nodes (0): 

### Community 2271 - "Community 2271"
Cohesion: 1.0
Nodes (0): 

### Community 2272 - "Community 2272"
Cohesion: 1.0
Nodes (0): 

### Community 2273 - "Community 2273"
Cohesion: 1.0
Nodes (0): 

### Community 2274 - "Community 2274"
Cohesion: 1.0
Nodes (0): 

### Community 2275 - "Community 2275"
Cohesion: 1.0
Nodes (0): 

### Community 2276 - "Community 2276"
Cohesion: 1.0
Nodes (0): 

### Community 2277 - "Community 2277"
Cohesion: 1.0
Nodes (0): 

### Community 2278 - "Community 2278"
Cohesion: 1.0
Nodes (0): 

### Community 2279 - "Community 2279"
Cohesion: 1.0
Nodes (0): 

### Community 2280 - "Community 2280"
Cohesion: 1.0
Nodes (0): 

### Community 2281 - "Community 2281"
Cohesion: 1.0
Nodes (0): 

### Community 2282 - "Community 2282"
Cohesion: 1.0
Nodes (0): 

### Community 2283 - "Community 2283"
Cohesion: 1.0
Nodes (0): 

### Community 2284 - "Community 2284"
Cohesion: 1.0
Nodes (0): 

### Community 2285 - "Community 2285"
Cohesion: 1.0
Nodes (0): 

### Community 2286 - "Community 2286"
Cohesion: 1.0
Nodes (1): https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metri

### Community 2287 - "Community 2287"
Cohesion: 1.0
Nodes (1): idx_pd: shape=(b,n), dtype=int, indexed segment         idx_gt: shape=(b,n), dty

### Community 2288 - "Community 2288"
Cohesion: 1.0
Nodes (1): idx_pd: shape=(b,n), dtype=uint8, indexed segment         idx_gt: shape=(b,n), d

### Community 2289 - "Community 2289"
Cohesion: 1.0
Nodes (1): https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

### Community 2290 - "Community 2290"
Cohesion: 1.0
Nodes (1): https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

### Community 2291 - "Community 2291"
Cohesion: 1.0
Nodes (1): - source: shape=(b,m,c)         - target: shape=(b,n,c)

### Community 2292 - "Community 2292"
Cohesion: 1.0
Nodes (1): Convert the original folded images into LMDB files.          The code is adapted

### Community 2293 - "Community 2293"
Cohesion: 1.0
Nodes (1): - video: bgr format, shape=(t,h,w,c=3), uint8         - bbox: both side normaliz

### Community 2294 - "Community 2294"
Cohesion: 1.0
Nodes (1): from the last dim to first

### Community 2295 - "Community 2295"
Cohesion: 1.0
Nodes (1): suppose bbox l-t-r-b is normalized; only zero out-crop bboxs, not remove them

### Community 2296 - "Community 2296"
Cohesion: 1.0
Nodes (1): Structure dataset as follows and run it!         - VOC2012  # as training set

### Community 2297 - "Community 2297"
Cohesion: 1.0
Nodes (1): - image: bgr format, shape=(h,w,c=3), uint8         - segment: index format, sha

### Community 2298 - "Community 2298"
Cohesion: 1.0
Nodes (1): Convert the original TFRecord files into one LMDB file, saving 10x storage space

### Community 2299 - "Community 2299"
Cohesion: 1.0
Nodes (1): Adopted from SAVi official implementation VideoFromTfds class.

### Community 2300 - "Community 2300"
Cohesion: 1.0
Nodes (1): Adopted from SAVi official implementation SparseToDenseAnnotation class.

### Community 2301 - "Community 2301"
Cohesion: 1.0
Nodes (1): - video: bgr format, shape=(t,h,w,c=3), uint8         - bbox: both side normaliz

### Community 2302 - "Community 2302"
Cohesion: 1.0
Nodes (1): Structure dataset as follows and run it!         - clevrtex_full  # as training

### Community 2303 - "Community 2303"
Cohesion: 1.0
Nodes (1): - image: bgr format, shape=(h,w,c=3), uint8         - segment: index format, sha

### Community 2304 - "Community 2304"
Cohesion: 1.0
Nodes (1): Download dataset MSCOCO:         - 2017 Train images [118K/18GB] http://images.c

### Community 2305 - "Community 2305"
Cohesion: 1.0
Nodes (1): straight-through gradient approximation          synchronized:         Straighte

### Community 2306 - "Community 2306"
Cohesion: 1.0
Nodes (1): Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe

### Community 2307 - "Community 2307"
Cohesion: 1.0
Nodes (1): Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe

### Community 2308 - "Community 2308"
Cohesion: 1.0
Nodes (1): euclidean kmeans in pytorch         https://github.com/subhadarship/kmeans_pytor

### Community 2309 - "Community 2309"
Cohesion: 1.0
Nodes (1): encode: in shape (b,c,h,w)         templat: in shape (m,c)         zsoft: in sha

### Community 2310 - "Community 2310"
Cohesion: 1.0
Nodes (1): chunked cdist          source: shape=(b,m,c) or (m,c)         target: shape=(b,n

### Community 2311 - "Community 2311"
Cohesion: 1.0
Nodes (1): does not change ``len(layers)``'s value

### Community 2312 - "Community 2312"
Cohesion: 1.0
Nodes (1): Farthest Point Sampling.

### Community 2313 - "Community 2313"
Cohesion: 1.0
Nodes (1): quantz: [QuantiZ,..]         encode: shape=(b,h,w,c)         zsoft: shape=(b,h,w

### Community 2314 - "Community 2314"
Cohesion: 1.0
Nodes (1): quantz: [QuantiZ,..]         zidx: indexes, shape=(b,h,w,g)         output: shap

### Community 2315 - "Community 2315"
Cohesion: 1.0
Nodes (1): Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe

### Community 2316 - "Community 2316"
Cohesion: 1.0
Nodes (1): Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe

### Community 2317 - "Community 2317"
Cohesion: 1.0
Nodes (1): euclidean kmeans in pytorch         https://github.com/subhadarship/kmeans_pytor

### Community 2318 - "Community 2318"
Cohesion: 1.0
Nodes (1): 对每个 encode 向量，从 templat 中找到最匹配的向量索引，并输出软分配概率。                  :param encode: Te

### Community 2319 - "Community 2319"
Cohesion: 1.0
Nodes (1): chunked cdist          source: shape=(b,m,c) or (m,c)         target: shape=(b,n

### Community 2320 - "Community 2320"
Cohesion: 1.0
Nodes (1): Positional Encoding of shape [1, L, D].

### Community 2321 - "Community 2321"
Cohesion: 1.0
Nodes (0): 

### Community 2322 - "Community 2322"
Cohesion: 1.0
Nodes (0): 

### Community 2323 - "Community 2323"
Cohesion: 1.0
Nodes (0): 

### Community 2324 - "Community 2324"
Cohesion: 1.0
Nodes (0): 

### Community 2325 - "Community 2325"
Cohesion: 1.0
Nodes (0): 

### Community 2326 - "Community 2326"
Cohesion: 1.0
Nodes (0): 

### Community 2327 - "Community 2327"
Cohesion: 1.0
Nodes (0): 

### Community 2328 - "Community 2328"
Cohesion: 1.0
Nodes (1): Builds train/eval data transforms for the dataset class.         :param is_train

### Community 2329 - "Community 2329"
Cohesion: 1.0
Nodes (1): Builds train/eval data transforms for the dataset class.         :param is_train

### Community 2330 - "Community 2330"
Cohesion: 1.0
Nodes (0): 

### Community 2331 - "Community 2331"
Cohesion: 1.0
Nodes (0): 

### Community 2332 - "Community 2332"
Cohesion: 1.0
Nodes (0): 

### Community 2333 - "Community 2333"
Cohesion: 1.0
Nodes (1): Loss Name.          This function must be implemented and will return the name o

### Community 2334 - "Community 2334"
Cohesion: 1.0
Nodes (1): Placeholder of forward function.

### Community 2335 - "Community 2335"
Cohesion: 1.0
Nodes (1): Compute segmentation loss.

### Community 2336 - "Community 2336"
Cohesion: 1.0
Nodes (0): 

### Community 2337 - "Community 2337"
Cohesion: 1.0
Nodes (0): 

### Community 2338 - "Community 2338"
Cohesion: 1.0
Nodes (0): 

### Community 2339 - "Community 2339"
Cohesion: 1.0
Nodes (0): 

### Community 2340 - "Community 2340"
Cohesion: 1.0
Nodes (0): 

### Community 2341 - "Community 2341"
Cohesion: 1.0
Nodes (0): 

### Community 2342 - "Community 2342"
Cohesion: 1.0
Nodes (0): 

### Community 2343 - "Community 2343"
Cohesion: 1.0
Nodes (0): 

### Community 2344 - "Community 2344"
Cohesion: 1.0
Nodes (0): 

### Community 2345 - "Community 2345"
Cohesion: 1.0
Nodes (0): 

### Community 2346 - "Community 2346"
Cohesion: 1.0
Nodes (0): 

### Community 2347 - "Community 2347"
Cohesion: 1.0
Nodes (0): 

### Community 2348 - "Community 2348"
Cohesion: 1.0
Nodes (0): 

### Community 2349 - "Community 2349"
Cohesion: 1.0
Nodes (0): 

### Community 2350 - "Community 2350"
Cohesion: 1.0
Nodes (0): 

### Community 2351 - "Community 2351"
Cohesion: 1.0
Nodes (0): 

### Community 2352 - "Community 2352"
Cohesion: 1.0
Nodes (0): 

### Community 2353 - "Community 2353"
Cohesion: 1.0
Nodes (0): 

### Community 2354 - "Community 2354"
Cohesion: 1.0
Nodes (0): 

### Community 2355 - "Community 2355"
Cohesion: 1.0
Nodes (0): 

### Community 2356 - "Community 2356"
Cohesion: 1.0
Nodes (0): 

### Community 2357 - "Community 2357"
Cohesion: 1.0
Nodes (0): 

### Community 2358 - "Community 2358"
Cohesion: 1.0
Nodes (0): 

### Community 2359 - "Community 2359"
Cohesion: 1.0
Nodes (0): 

### Community 2360 - "Community 2360"
Cohesion: 1.0
Nodes (0): 

### Community 2361 - "Community 2361"
Cohesion: 1.0
Nodes (0): 

### Community 2362 - "Community 2362"
Cohesion: 1.0
Nodes (0): 

### Community 2363 - "Community 2363"
Cohesion: 1.0
Nodes (1): r"""Customize every aspect of training via flags.          Args:             acc

### Community 2364 - "Community 2364"
Cohesion: 1.0
Nodes (1): List up files in `dir_path` with `name_key`, then yield maximum suffix number.

### Community 2365 - "Community 2365"
Cohesion: 1.0
Nodes (1): Get path of maximum-epoch checkpoint in the folder.

### Community 2366 - "Community 2366"
Cohesion: 1.0
Nodes (1): Forward the inputs through the network and produce the predictions.          The

### Community 2367 - "Community 2367"
Cohesion: 1.0
Nodes (1): Load a pretrained MegaFlow model from HuggingFace Hub.          Args:

### Community 2368 - "Community 2368"
Cohesion: 1.0
Nodes (1): Performs feature rotation by splitting and recombining feature dimensions.

### Community 2369 - "Community 2369"
Cohesion: 1.0
Nodes (1): Extract features from images.

### Community 2370 - "Community 2370"
Cohesion: 1.0
Nodes (1): Resize pos_embed weights.          Resize pos_embed using bicubic interpolate me

### Community 2371 - "Community 2371"
Cohesion: 1.0
Nodes (0): 

### Community 2372 - "Community 2372"
Cohesion: 1.0
Nodes (0): 

### Community 2373 - "Community 2373"
Cohesion: 1.0
Nodes (1): Momentum update of evaluation model (exponential moving average)

### Community 2374 - "Community 2374"
Cohesion: 1.0
Nodes (0): 

### Community 2375 - "Community 2375"
Cohesion: 1.0
Nodes (1): Inference interface for the model for PIL image         Args:             pil_im

### Community 2376 - "Community 2376"
Cohesion: 1.0
Nodes (0): 

### Community 2377 - "Community 2377"
Cohesion: 1.0
Nodes (0): 

### Community 2378 - "Community 2378"
Cohesion: 1.0
Nodes (0): 

### Community 2379 - "Community 2379"
Cohesion: 1.0
Nodes (0): 

### Community 2380 - "Community 2380"
Cohesion: 1.0
Nodes (1): Kick off a best-effort cpu-basic sandbox for the session.

### Community 2381 - "Community 2381"
Cohesion: 1.0
Nodes (1): Delete the sandbox Space if one was created for this session.          Retries o

### Community 2382 - "Community 2382"
Cohesion: 1.0
Nodes (1): Get count of active sessions.

### Community 2383 - "Community 2383"
Cohesion: 1.0
Nodes (1): Create a new sandbox by duplicating the template Space.          Generates a uni

### Community 2384 - "Community 2384"
Cohesion: 1.0
Nodes (1): Upload embedded sandbox server + Dockerfile to the Space (single commit).

### Community 2385 - "Community 2385"
Cohesion: 1.0
Nodes (1): Connect to an existing running Space.          Does a health check to verify the

### Community 2386 - "Community 2386"
Cohesion: 1.0
Nodes (1): Public URL of the Space.

### Community 2387 - "Community 2387"
Cohesion: 1.0
Nodes (1): Current Space stage (RUNNING, BUILDING, PAUSED, etc.).

### Community 2388 - "Community 2388"
Cohesion: 1.0
Nodes (1): Cancel pending approval tools when the user continues the conversation.

### Community 2389 - "Community 2389"
Cohesion: 1.0
Nodes (1): Handle user input (like user_input_or_turn in codex.rs:1291)         Returns the

### Community 2390 - "Community 2390"
Cohesion: 1.0
Nodes (1): Remove the last complete turn and notify the frontend.

### Community 2391 - "Community 2391"
Cohesion: 1.0
Nodes (1): Start a fresh conversation inside the active runtime.

### Community 2392 - "Community 2392"
Cohesion: 1.0
Nodes (1): Reload context from a saved session log into the active session.

### Community 2393 - "Community 2393"
Cohesion: 1.0
Nodes (1): Handle batch job execution approval

### Community 2394 - "Community 2394"
Cohesion: 1.0
Nodes (1): Handle shutdown (like shutdown in codex.rs:1329)

### Community 2395 - "Community 2395"
Cohesion: 1.0
Nodes (1): Spawn detached subprocess(es) to retry failed/pending uploads         (fire-and-

### Community 2396 - "Community 2396"
Cohesion: 1.0
Nodes (1): Ensure msg.tool_calls contains proper ToolCall objects, not dicts.          lite

### Community 2397 - "Community 2397"
Cohesion: 1.0
Nodes (1): Token count at which `compact()` kicks in.

### Community 2398 - "Community 2398"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has auxiliary head

### Community 2399 - "Community 2399"
Cohesion: 1.0
Nodes (1): Forward pass through full MBPS model.          Args:             image: Input im

### Community 2400 - "Community 2400"
Cohesion: 1.0
Nodes (1): Predict per-token semantic logits.          Args:             features: (B, N, b

### Community 2401 - "Community 2401"
Cohesion: 1.0
Nodes (1): Predict per-token instance embeddings.          Args:             features: (B,

### Community 2402 - "Community 2402"
Cohesion: 1.0
Nodes (1): Apply BiCMS fusion.          Args:             semantic: Semantic tokens (B, N,

### Community 2403 - "Community 2403"
Cohesion: 1.0
Nodes (1): Apply SSD selective scan.          Args:             x: Input sequence of shape

### Community 2404 - "Community 2404"
Cohesion: 1.0
Nodes (1): Apply Mamba2 block.          Args:             x: Input of shape (B, L, D).

### Community 2405 - "Community 2405"
Cohesion: 1.0
Nodes (1): Apply stack of Mamba2 blocks.          Args:             x: Input of shape (B, L

### Community 2406 - "Community 2406"
Cohesion: 1.0
Nodes (1): Project features to bridge dimension.          Args:             semantic_codes:

### Community 2407 - "Community 2407"
Cohesion: 1.0
Nodes (1): Inverse project from bridge dimension.          Args:             x: Fused featu

### Community 2408 - "Community 2408"
Cohesion: 1.0
Nodes (1): Condition features on depth.          Args:             depth: Depth values of s

### Community 2409 - "Community 2409"
Cohesion: 1.0
Nodes (1): Classify clusters as stuff or things.          Args:             cues: Concatena

### Community 2410 - "Community 2410"
Cohesion: 1.0
Nodes (1): Generate instance masks from features.          In inference mode with depth ava

### Community 2411 - "Community 2411"
Cohesion: 1.0
Nodes (1): Refine masks through cascade stages.          Args:             features: Featur

### Community 2412 - "Community 2412"
Cohesion: 1.0
Nodes (1): Generate proposals from features.          Args:             features: Backbone

### Community 2413 - "Community 2413"
Cohesion: 1.0
Nodes (1): Predict class scores and box deltas.          Args:             pooled_features:

### Community 2414 - "Community 2414"
Cohesion: 1.0
Nodes (1): Predict mask features.          Args:             features: Pooled RoI features,

### Community 2415 - "Community 2415"
Cohesion: 1.0
Nodes (1): Run one cascade stage.          Args:             features: Feature map, shape (

### Community 2416 - "Community 2416"
Cohesion: 1.0
Nodes (1): Generate instance masks and scores.          Args:             features: Input f

### Community 2417 - "Community 2417"
Cohesion: 1.0
Nodes (1): Compute semantic codes from DINO features.          Args:             features:

### Community 2418 - "Community 2418"
Cohesion: 1.0
Nodes (1): Compute semantic codes from spatial features.          Args:             feature

### Community 2419 - "Community 2419"
Cohesion: 1.0
Nodes (1): Extract patch embeddings.          Args:             x: Input image of shape (B,

### Community 2420 - "Community 2420"
Cohesion: 1.0
Nodes (1): Apply multi-head self-attention.          Args:             x: Input of shape (B

### Community 2421 - "Community 2421"
Cohesion: 1.0
Nodes (1): Apply MLP.          Args:             x: Input of shape (B, N, D).             d

### Community 2422 - "Community 2422"
Cohesion: 1.0
Nodes (1): Apply Transformer block (pre-norm).          Args:             x: Input of shape

### Community 2423 - "Community 2423"
Cohesion: 1.0
Nodes (1): Extract DINO features from input image.          Args:             x: Input imag

### Community 2424 - "Community 2424"
Cohesion: 1.0
Nodes (1): Extract patch embeddings.          Args:             x: Input image of shape (B,

### Community 2425 - "Community 2425"
Cohesion: 1.0
Nodes (1): Apply multi-head self-attention.          Args:             x: Input of shape (B

### Community 2426 - "Community 2426"
Cohesion: 1.0
Nodes (1): Extract DINOv3 features from input image.          Args:             x: Input im

### Community 2427 - "Community 2427"
Cohesion: 1.0
Nodes (0): 

### Community 2428 - "Community 2428"
Cohesion: 1.0
Nodes (1): Load an OLMo model from a checkpoint.

### Community 2429 - "Community 2429"
Cohesion: 1.0
Nodes (1): Returns the length of the idx-th trajectory.

### Community 2430 - "Community 2430"
Cohesion: 1.0
Nodes (0): 

### Community 2431 - "Community 2431"
Cohesion: 1.0
Nodes (1): One image / label pair for the given index is picked up and pre-processed.

### Community 2432 - "Community 2432"
Cohesion: 1.0
Nodes (1): Dimension that can be used by transforms to set the correct image size, etc.

### Community 2433 - "Community 2433"
Cohesion: 1.0
Nodes (1): Decorator method that needs to be used around the ``__getitem__`` method. |br|

### Community 2434 - "Community 2434"
Cohesion: 1.0
Nodes (1): Constructs a `BertConfig` from a Python dictionary of parameters.

### Community 2435 - "Community 2435"
Cohesion: 1.0
Nodes (1): Constructs a `BertConfig` from a json file of parameters.

### Community 2436 - "Community 2436"
Cohesion: 1.0
Nodes (1): Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch sta

### Community 2437 - "Community 2437"
Cohesion: 1.0
Nodes (1): Get the reference points used in decoder.          Args:             spatial_sha

### Community 2438 - "Community 2438"
Cohesion: 1.0
Nodes (0): 

### Community 2439 - "Community 2439"
Cohesion: 1.0
Nodes (0): 

### Community 2440 - "Community 2440"
Cohesion: 1.0
Nodes (0): 

### Community 2441 - "Community 2441"
Cohesion: 1.0
Nodes (0): 

### Community 2442 - "Community 2442"
Cohesion: 1.0
Nodes (0): 

### Community 2443 - "Community 2443"
Cohesion: 1.0
Nodes (0): 

### Community 2444 - "Community 2444"
Cohesion: 1.0
Nodes (0): 

### Community 2445 - "Community 2445"
Cohesion: 1.0
Nodes (0): 

### Community 2446 - "Community 2446"
Cohesion: 1.0
Nodes (0): 

### Community 2447 - "Community 2447"
Cohesion: 1.0
Nodes (1): PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/d

### Community 2448 - "Community 2448"
Cohesion: 1.0
Nodes (1): Move the image channels to the first dimension of the numpy         multi-dimens

### Community 2449 - "Community 2449"
Cohesion: 1.0
Nodes (1): Move the image channels to the last dimensiion of the numpy         multi-dimens

### Community 2450 - "Community 2450"
Cohesion: 1.0
Nodes (1): Loads an image from file as a numpy multi-dimensional array          :param img_

### Community 2451 - "Community 2451"
Cohesion: 1.0
Nodes (1): Computes the mean squared error between to RGB images represented as multi-dimen

### Community 2452 - "Community 2452"
Cohesion: 1.0
Nodes (1): Computes the PSNR for a batch of input and output images          :param image_b

### Community 2453 - "Community 2453"
Cohesion: 1.0
Nodes (1): Computes the SSIM for a batch of input and output images          :param image_b

### Community 2454 - "Community 2454"
Cohesion: 1.0
Nodes (1): Abstract function for the data loader class          :returns: N/A         :rtyp

### Community 2455 - "Community 2455"
Cohesion: 1.0
Nodes (1): Abstract function for the data loader class          :returns: N/A         :rtyp

### Community 2456 - "Community 2456"
Cohesion: 1.0
Nodes (0): 

### Community 2457 - "Community 2457"
Cohesion: 1.0
Nodes (1): PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/d

### Community 2458 - "Community 2458"
Cohesion: 1.0
Nodes (1): PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/d

### Community 2459 - "Community 2459"
Cohesion: 1.0
Nodes (1): Move the image channels to the first dimension of the numpy         multi-dimens

### Community 2460 - "Community 2460"
Cohesion: 1.0
Nodes (1): Move the image channels to the last dimensiion of the numpy         multi-dimens

### Community 2461 - "Community 2461"
Cohesion: 1.0
Nodes (1): Loads an image from file as a numpy multi-dimensional array          :param img_

### Community 2462 - "Community 2462"
Cohesion: 1.0
Nodes (1): Normalises image data to be a float between 0 and 1          :param img: Image a

### Community 2463 - "Community 2463"
Cohesion: 1.0
Nodes (1): Computes the mean squared error between to RGB images represented as multi-dimen

### Community 2464 - "Community 2464"
Cohesion: 1.0
Nodes (1): Computes the PSNR for a batch of input and output images          :param image_b

### Community 2465 - "Community 2465"
Cohesion: 1.0
Nodes (1): Computes the SSIM for a batch of input and output images          :param image_b

### Community 2466 - "Community 2466"
Cohesion: 1.0
Nodes (1): Converts a HSV image to RGB         PyTorch implementation of RGB to HSV convers

### Community 2467 - "Community 2467"
Cohesion: 1.0
Nodes (1): Converts an RGB image to HSV         PyTorch implementation of RGB to HSV conver

### Community 2468 - "Community 2468"
Cohesion: 1.0
Nodes (1): Applies a peicewise linear curve defined by a set of knot points to         an i

### Community 2469 - "Community 2469"
Cohesion: 1.0
Nodes (1): Adjust the HSV channels of a HSV image using learnt curves          :param img:

### Community 2470 - "Community 2470"
Cohesion: 1.0
Nodes (1): Adjust the RGB channels of a RGB image using learnt curves          :param img:

### Community 2471 - "Community 2471"
Cohesion: 1.0
Nodes (1): Adjusts the image in LAB space using the predicted curves          :param img: I

### Community 2472 - "Community 2472"
Cohesion: 1.0
Nodes (1): Abstract function for the data loader class          :returns: N/A         :rtyp

### Community 2473 - "Community 2473"
Cohesion: 1.0
Nodes (1): Abstract function for the data loader class          :returns: N/A         :rtyp

### Community 2474 - "Community 2474"
Cohesion: 1.0
Nodes (1): Compute log-likelihood of generating a continuation from a context.         Down

### Community 2475 - "Community 2475"
Cohesion: 1.0
Nodes (1): Generate greedily until a stopping sequence          :param requests: list

### Community 2476 - "Community 2476"
Cohesion: 1.0
Nodes (1): Parse the raw outputs (losses) of the network.          Args:             losses

### Community 2477 - "Community 2477"
Cohesion: 1.0
Nodes (1): Whether the task has a training set

### Community 2478 - "Community 2478"
Cohesion: 1.0
Nodes (1): Whether the task has a validation set

### Community 2479 - "Community 2479"
Cohesion: 1.0
Nodes (1): Whether the task has a test set

### Community 2480 - "Community 2480"
Cohesion: 1.0
Nodes (1): Uses RequestFactory to construct Requests and returns an iterable of         Req

### Community 2481 - "Community 2481"
Cohesion: 1.0
Nodes (1): Take a single document and the LM results and evaluates, returning a         dic

### Community 2482 - "Community 2482"
Cohesion: 1.0
Nodes (1): :returns: {str: [metric_score] -> float}             A dictionary where keys are

### Community 2483 - "Community 2483"
Cohesion: 1.0
Nodes (1): :returns: {str: bool}             A dictionary where keys are the names of subme

### Community 2484 - "Community 2484"
Cohesion: 1.0
Nodes (1): Returns a fewshot context string that is made up of a prepended description

### Community 2485 - "Community 2485"
Cohesion: 1.0
Nodes (1): Downstream tasks with custom word boundaries should override this!

### Community 2486 - "Community 2486"
Cohesion: 1.0
Nodes (1): Whether to include special tokens in encoded text. This should be         determ

### Community 2487 - "Community 2487"
Cohesion: 1.0
Nodes (1): Return the maximum sequence length of the model.         NOTE: Different model c

### Community 2488 - "Community 2488"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`

### Community 2489 - "Community 2489"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

### Community 2490 - "Community 2490"
Cohesion: 1.0
Nodes (1): r"""         start_positions (`torch.LongTensor` of shape `(batch_size,)`, *opti

### Community 2491 - "Community 2491"
Cohesion: 1.0
Nodes (1): r"""         Args:             input_ids (`torch.LongTensor` of shape `(batch_si

### Community 2492 - "Community 2492"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`

### Community 2493 - "Community 2493"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`

### Community 2494 - "Community 2494"
Cohesion: 1.0
Nodes (1): This function is used to re-order the `past_key_values` cache if [`~PretrainedMo

### Community 2495 - "Community 2495"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

### Community 2496 - "Community 2496"
Cohesion: 1.0
Nodes (1): This function is used to re-order the `past_key_values` cache if         [`~PreT

### Community 2497 - "Community 2497"
Cohesion: 1.0
Nodes (1): r""" 		Generates sequences of token ids for models with a language modeling head

### Community 2498 - "Community 2498"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`

### Community 2499 - "Community 2499"
Cohesion: 1.0
Nodes (1): This function is used to re-order the `past_key_values` cache if [`~PreTrainedMo

### Community 2500 - "Community 2500"
Cohesion: 1.0
Nodes (1): r"""         mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices

### Community 2501 - "Community 2501"
Cohesion: 1.0
Nodes (1): This function is used to re-order the `past_key_values` cache if [`~PreTrainedMo

### Community 2502 - "Community 2502"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

### Community 2503 - "Community 2503"
Cohesion: 1.0
Nodes (1): r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):

### Community 2504 - "Community 2504"
Cohesion: 1.0
Nodes (1): Extract entities from tokens.          Returns:             list: list of Entity

### Community 2505 - "Community 2505"
Cohesion: 1.0
Nodes (0): 

### Community 2506 - "Community 2506"
Cohesion: 1.0
Nodes (1): Root function         Args:             step: Current step             c0: In

### Community 2507 - "Community 2507"
Cohesion: 1.0
Nodes (0): 

### Community 2508 - "Community 2508"
Cohesion: 1.0
Nodes (1): Save training dynamics to a .json file         Each line contains a dictionary

### Community 2509 - "Community 2509"
Cohesion: 1.0
Nodes (1): Save training dynamics to a .json file         Each line contains a dictionary

### Community 2510 - "Community 2510"
Cohesion: 1.0
Nodes (0): 

### Community 2511 - "Community 2511"
Cohesion: 1.0
Nodes (1): Reads a tab separated value file.

### Community 2512 - "Community 2512"
Cohesion: 1.0
Nodes (1): 将trie树的查询结果转换为匹配向量          sort=true，将按照起始位置和长度进行排序

### Community 2513 - "Community 2513"
Cohesion: 1.0
Nodes (1): 将trie树的查询结果转换为匹配向量         sort=true，将按照起始位置和长度进行排序

### Community 2514 - "Community 2514"
Cohesion: 1.0
Nodes (0): 

### Community 2515 - "Community 2515"
Cohesion: 1.0
Nodes (0): 

### Community 2516 - "Community 2516"
Cohesion: 1.0
Nodes (0): 

### Community 2517 - "Community 2517"
Cohesion: 1.0
Nodes (0): 

### Community 2518 - "Community 2518"
Cohesion: 1.0
Nodes (1): model return (reconstructed_x, *)

### Community 2519 - "Community 2519"
Cohesion: 1.0
Nodes (1): sample new images from model

### Community 2520 - "Community 2520"
Cohesion: 1.0
Nodes (0): 

### Community 2521 - "Community 2521"
Cohesion: 1.0
Nodes (1): returns the latest losses in a dictionary. Useful for logging.

### Community 2522 - "Community 2522"
Cohesion: 1.0
Nodes (1): model return (reconstructed_x, *)

### Community 2523 - "Community 2523"
Cohesion: 1.0
Nodes (1): sample new images from model

### Community 2524 - "Community 2524"
Cohesion: 1.0
Nodes (0): 

### Community 2525 - "Community 2525"
Cohesion: 1.0
Nodes (1): returns the latest losses in a dictionary. Useful for logging.

### Community 2526 - "Community 2526"
Cohesion: 1.0
Nodes (0): 

### Community 2527 - "Community 2527"
Cohesion: 1.0
Nodes (0): 

### Community 2528 - "Community 2528"
Cohesion: 1.0
Nodes (1): During Pydantic model instantiation, only validate that `entry_point_agent` is i

### Community 2529 - "Community 2529"
Cohesion: 1.0
Nodes (1): Streaming mode to generate LLM response

### Community 2530 - "Community 2530"
Cohesion: 1.0
Nodes (1): Generate LLM response

### Community 2531 - "Community 2531"
Cohesion: 1.0
Nodes (1): Convert state msg list into openai format

### Community 2532 - "Community 2532"
Cohesion: 1.0
Nodes (1): Generate LLM response using streaming mode

### Community 2533 - "Community 2533"
Cohesion: 1.0
Nodes (1): load a built graph_builder from file.

### Community 2534 - "Community 2534"
Cohesion: 1.0
Nodes (1): load a built graph_builder from a config dict

### Community 2535 - "Community 2535"
Cohesion: 1.0
Nodes (1): Put streaming msg into stream writer and trigger the msg handler

### Community 2536 - "Community 2536"
Cohesion: 1.0
Nodes (1): Return next nodes due to current state.

### Community 2537 - "Community 2537"
Cohesion: 1.0
Nodes (1): Return all possible targets.         This is used to static analyze the graph st

### Community 2538 - "Community 2538"
Cohesion: 1.0
Nodes (1): if element is tuple[str, callable]

### Community 2539 - "Community 2539"
Cohesion: 1.0
Nodes (1): Output_msg_format cannot be none when output_schema is none

### Community 2540 - "Community 2540"
Cohesion: 1.0
Nodes (1): Get descriptions for pydantic fields

### Community 2541 - "Community 2541"
Cohesion: 1.0
Nodes (1): Create a class instance using the given name and kwargs.

### Community 2542 - "Community 2542"
Cohesion: 1.0
Nodes (1): Register a class into the factory

### Community 2543 - "Community 2543"
Cohesion: 1.0
Nodes (1): Returns the current active client session (read-only)

### Community 2544 - "Community 2544"
Cohesion: 1.0
Nodes (1): Returns connection status (True/False) for monitoring purposes

### Community 2545 - "Community 2545"
Cohesion: 1.0
Nodes (1): Return a list of tool schemas in OpenAI function format          Returns:

### Community 2546 - "Community 2546"
Cohesion: 1.0
Nodes (1): Execute tool calls with provided tasks          Args:             tasks: List of

### Community 2547 - "Community 2547"
Cohesion: 1.0
Nodes (1): Set the tool controller that manages tool activation rules          Args:

### Community 2548 - "Community 2548"
Cohesion: 1.0
Nodes (1): Get the current tool controller instance          Returns:             The curre

### Community 2549 - "Community 2549"
Cohesion: 1.0
Nodes (1): clear the vector store

### Community 2550 - "Community 2550"
Cohesion: 1.0
Nodes (1): the retrieval entrance

### Community 2551 - "Community 2551"
Cohesion: 1.0
Nodes (1): add new db item to vectorstore

### Community 2552 - "Community 2552"
Cohesion: 1.0
Nodes (1): delete item by it original ids         :param ids:         :return:

### Community 2553 - "Community 2553"
Cohesion: 1.0
Nodes (1): Get vector count          Returns:             Number of vectors

### Community 2554 - "Community 2554"
Cohesion: 1.0
Nodes (1): Get collection information          Returns:             Collection information

### Community 2555 - "Community 2555"
Cohesion: 1.0
Nodes (1): retrival memory and update context messages.         :param messages: context me

### Community 2556 - "Community 2556"
Cohesion: 1.0
Nodes (1): add context messages to memory vectorstore         :param messages: context mess

### Community 2557 - "Community 2557"
Cohesion: 1.0
Nodes (1): clear all memories         :return:

### Community 2558 - "Community 2558"
Cohesion: 1.0
Nodes (1): Extract features from contexts

### Community 2559 - "Community 2559"
Cohesion: 1.0
Nodes (1): Merge current info with existing memories

### Community 2560 - "Community 2560"
Cohesion: 1.0
Nodes (1): Summary the recalled memories

### Community 2561 - "Community 2561"
Cohesion: 1.0
Nodes (1): Based on mem summary info to update basic messages

### Community 2562 - "Community 2562"
Cohesion: 1.0
Nodes (1): Calls the configured LLM model with a given query.          Args:             qu

### Community 2563 - "Community 2563"
Cohesion: 1.0
Nodes (1): Fetches raw text content from the specified URL.          Args:             url

### Community 2564 - "Community 2564"
Cohesion: 1.0
Nodes (1): Processes raw text content to extract relevant information based on a query.

### Community 2565 - "Community 2565"
Cohesion: 1.0
Nodes (1): Retrieves the Content-Type header of a URL via a HEAD request.          Args:

### Community 2566 - "Community 2566"
Cohesion: 1.0
Nodes (1): Instantiates a parser suitable for the given URL.          Selection Logic:

### Community 2567 - "Community 2567"
Cohesion: 1.0
Nodes (1): Evaluate a single data item.          Subclasses must implement this method to d

### Community 2568 - "Community 2568"
Cohesion: 1.0
Nodes (1): Simple, reliable and slow implementation of batch by size

### Community 2569 - "Community 2569"
Cohesion: 1.0
Nodes (1): Do forward, backward and parameter update.

### Community 2570 - "Community 2570"
Cohesion: 1.0
Nodes (1): Do forward pass in evaluation mode.

### Community 2571 - "Community 2571"
Cohesion: 1.0
Nodes (1): Generate a batch of translations.          Args:             sample (dict): batc

### Community 2572 - "Community 2572"
Cohesion: 1.0
Nodes (1): Reorder encoder output according to *new_order*.          Args:             enco

### Community 2573 - "Community 2573"
Cohesion: 1.0
Nodes (1): Do forward, backward and parameter update.

### Community 2574 - "Community 2574"
Cohesion: 1.0
Nodes (1): Do forward pass in evaluation mode.

### Community 2575 - "Community 2575"
Cohesion: 1.0
Nodes (1): Score a batch of translations.

### Community 2576 - "Community 2576"
Cohesion: 1.0
Nodes (1): Initialize constraint states for constrained decoding (if supported).          A

### Community 2577 - "Community 2577"
Cohesion: 1.0
Nodes (1): A constrained step builds a large candidates list from the following:         -

### Community 2578 - "Community 2578"
Cohesion: 1.0
Nodes (1): Does per-sentence processing. Adds all constraints for each         hypothesis t

### Community 2579 - "Community 2579"
Cohesion: 1.0
Nodes (1): Do we require PathManager to access given path?

### Community 2580 - "Community 2580"
Cohesion: 1.0
Nodes (1): Do forward, backward and parameter update.

### Community 2581 - "Community 2581"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2582 - "Community 2582"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2583 - "Community 2583"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2584 - "Community 2584"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary`.

### Community 2585 - "Community 2585"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary`.

### Community 2586 - "Community 2586"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2587 - "Community 2587"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `train_step` and `valid_step` can

### Community 2588 - "Community 2588"
Cohesion: 1.0
Nodes (1): Load the dictionary from the filename          Args:             filename (str):

### Community 2589 - "Community 2589"
Cohesion: 1.0
Nodes (1): Build the dictionary          Args:             filenames (list): list of filena

### Community 2590 - "Community 2590"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             cfg (omegac

### Community 2591 - "Community 2591"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary` (if applicable         for t

### Community 2592 - "Community 2592"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary` (if applicable         for t

### Community 2593 - "Community 2593"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2594 - "Community 2594"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2595 - "Community 2595"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2596 - "Community 2596"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2597 - "Community 2597"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary`.

### Community 2598 - "Community 2598"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary`.

### Community 2599 - "Community 2599"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2600 - "Community 2600"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2601 - "Community 2601"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2602 - "Community 2602"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2603 - "Community 2603"
Cohesion: 1.0
Nodes (1): Return the :class:`~fairseq.data.Dictionary` for the language         model.

### Community 2604 - "Community 2604"
Cohesion: 1.0
Nodes (1): Return the :class:`~fairseq.data.Dictionary` for the language         model.

### Community 2605 - "Community 2605"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2606 - "Community 2606"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             cfg (AudioP

### Community 2607 - "Community 2607"
Cohesion: 1.0
Nodes (1): Return the :class:`~fairseq.data.Dictionary` for the language         model.

### Community 2608 - "Community 2608"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2609 - "Community 2609"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary`.

### Community 2610 - "Community 2610"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary`.

### Community 2611 - "Community 2611"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2612 - "Community 2612"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary`.

### Community 2613 - "Community 2613"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary`.

### Community 2614 - "Community 2614"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2615 - "Community 2615"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).          Args:             args (argpa

### Community 2616 - "Community 2616"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary`.

### Community 2617 - "Community 2617"
Cohesion: 1.0
Nodes (1): Return the target :class:`~fairseq.data.Dictionary`.

### Community 2618 - "Community 2618"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2619 - "Community 2619"
Cohesion: 1.0
Nodes (1): Load the dictionary from the filename          Args:             filename (str):

### Community 2620 - "Community 2620"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2621 - "Community 2621"
Cohesion: 1.0
Nodes (1): Load the masked LM dictionary from the filename          Args:             filen

### Community 2622 - "Community 2622"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2623 - "Community 2623"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2624 - "Community 2624"
Cohesion: 1.0
Nodes (1): Load the dictionary from the filename          Args:             filename (str):

### Community 2625 - "Community 2625"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2626 - "Community 2626"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2627 - "Community 2627"
Cohesion: 1.0
Nodes (1): A context manager to disable gradient synchronization.

### Community 2628 - "Community 2628"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2629 - "Community 2629"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2630 - "Community 2630"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2631 - "Community 2631"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2632 - "Community 2632"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2633 - "Community 2633"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2634 - "Community 2634"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2635 - "Community 2635"
Cohesion: 1.0
Nodes (1): Return a torch.optim.optimizer.Optimizer instance.

### Community 2636 - "Community 2636"
Cohesion: 1.0
Nodes (1): Reset optimizer instance.

### Community 2637 - "Community 2637"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2638 - "Community 2638"
Cohesion: 1.0
Nodes (1): Return an iterable of the parameters held by the optimizer.

### Community 2639 - "Community 2639"
Cohesion: 1.0
Nodes (1): Whether the optimizer supports collapsing of the model         parameters/gradie

### Community 2640 - "Community 2640"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2641 - "Community 2641"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2642 - "Community 2642"
Cohesion: 1.0
Nodes (1): Args:             cfg (omegaconf.DictConfig): fairseq args             params (i

### Community 2643 - "Community 2643"
Cohesion: 1.0
Nodes (1): Args:             args (argparse.Namespace): fairseq args             params (it

### Community 2644 - "Community 2644"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2645 - "Community 2645"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2646 - "Community 2646"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2647 - "Community 2647"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2648 - "Community 2648"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2649 - "Community 2649"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2650 - "Community 2650"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2651 - "Community 2651"
Cohesion: 1.0
Nodes (1): Add arguments to the parser for this LR scheduler.

### Community 2652 - "Community 2652"
Cohesion: 1.0
Nodes (1): Add arguments to the parser for this LR scheduler.

### Community 2653 - "Community 2653"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2654 - "Community 2654"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2655 - "Community 2655"
Cohesion: 1.0
Nodes (1): Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model         fi

### Community 2656 - "Community 2656"
Cohesion: 1.0
Nodes (1): Helper function to build shared embeddings for a set of languages after

### Community 2657 - "Community 2657"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2658 - "Community 2658"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2659 - "Community 2659"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2660 - "Community 2660"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2661 - "Community 2661"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2662 - "Community 2662"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2663 - "Community 2663"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2664 - "Community 2664"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2665 - "Community 2665"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2666 - "Community 2666"
Cohesion: 1.0
Nodes (1): Get normalized probabilities (or log probs) from a net's output.

### Community 2667 - "Community 2667"
Cohesion: 1.0
Nodes (1): Reorder encoder output according to *new_order*.          Args:             enco

### Community 2668 - "Community 2668"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2669 - "Community 2669"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2670 - "Community 2670"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2671 - "Community 2671"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2672 - "Community 2672"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2673 - "Community 2673"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2674 - "Community 2674"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2675 - "Community 2675"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2676 - "Community 2676"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2677 - "Community 2677"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2678 - "Community 2678"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2679 - "Community 2679"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2680 - "Community 2680"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2681 - "Community 2681"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2682 - "Community 2682"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2683 - "Community 2683"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2684 - "Community 2684"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2685 - "Community 2685"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2686 - "Community 2686"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2687 - "Community 2687"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2688 - "Community 2688"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2689 - "Community 2689"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2690 - "Community 2690"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2691 - "Community 2691"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2692 - "Community 2692"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2693 - "Community 2693"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2694 - "Community 2694"
Cohesion: 1.0
Nodes (1): Reorder buffered internal state (for incremental generation).

### Community 2695 - "Community 2695"
Cohesion: 1.0
Nodes (1): Args:             incremental_state: Used to buffer signal; if not None, then in

### Community 2696 - "Community 2696"
Cohesion: 1.0
Nodes (1): Build sinusoidal embeddings.          This matches the implementation in tensor2

### Community 2697 - "Community 2697"
Cohesion: 1.0
Nodes (1): Whether this dataset supports prefetching.

### Community 2698 - "Community 2698"
Cohesion: 1.0
Nodes (1): The number of consumed batches in the current epoch.

### Community 2699 - "Community 2699"
Cohesion: 1.0
Nodes (1): Return the epoch index after *next_epoch_itr* is called.

### Community 2700 - "Community 2700"
Cohesion: 1.0
Nodes (1): Return the epoch index after *next_epoch_itr* is called.

### Community 2701 - "Community 2701"
Cohesion: 1.0
Nodes (1): The number of consumed batches in the current epoch.

### Community 2702 - "Community 2702"
Cohesion: 1.0
Nodes (1): Loads the dictionary from a text file with the format:          ```         <sym

### Community 2703 - "Community 2703"
Cohesion: 1.0
Nodes (1): Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for         th

### Community 2704 - "Community 2704"
Cohesion: 1.0
Nodes (1): Whether this dataset supports prefetching.

### Community 2705 - "Community 2705"
Cohesion: 1.0
Nodes (1): Whether this dataset supports fetching outside the workers of the dataloader.

### Community 2706 - "Community 2706"
Cohesion: 1.0
Nodes (1): fairseq vocabulary file under data root

### Community 2707 - "Community 2707"
Cohesion: 1.0
Nodes (1): Shuffle dataset samples before batching

### Community 2708 - "Community 2708"
Cohesion: 1.0
Nodes (1): Pre-tokenizer to apply before subword tokenization. Returning         a dictiona

### Community 2709 - "Community 2709"
Cohesion: 1.0
Nodes (1): Subword tokenizer to apply after pre-tokenization. Returning         a dictionar

### Community 2710 - "Community 2710"
Cohesion: 1.0
Nodes (1): Prepend target lang ID token as the target BOS (e.g. for to-many         multili

### Community 2711 - "Community 2711"
Cohesion: 1.0
Nodes (1): The dimension of input features (per audio channel)

### Community 2712 - "Community 2712"
Cohesion: 1.0
Nodes (1): The number of channels in the input audio

### Community 2713 - "Community 2713"
Cohesion: 1.0
Nodes (1): Hyper-parameter alpha = 1/T for temperature-based resampling.         (alpha = 1

### Community 2714 - "Community 2714"
Cohesion: 1.0
Nodes (1): Needed by the dataset loader to see if the model requires         raw audio as i

### Community 2715 - "Community 2715"
Cohesion: 1.0
Nodes (1): Audio paths in the manifest TSV can be relative and this provides         the ro

### Community 2716 - "Community 2716"
Cohesion: 1.0
Nodes (1): Size ratios for temperature-based sampling         (https://arxiv.org/abs/1907.0

### Community 2717 - "Community 2717"
Cohesion: 1.0
Nodes (1): Smoothed value used for logging.

### Community 2718 - "Community 2718"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2719 - "Community 2719"
Cohesion: 1.0
Nodes (1): Construct a criterion from command-line args.

### Community 2720 - "Community 2720"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2721 - "Community 2721"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2722 - "Community 2722"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2723 - "Community 2723"
Cohesion: 1.0
Nodes (1): Construct a criterion from command-line args.

### Community 2724 - "Community 2724"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2725 - "Community 2725"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2726 - "Community 2726"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2727 - "Community 2727"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2728 - "Community 2728"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2729 - "Community 2729"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2730 - "Community 2730"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2731 - "Community 2731"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2732 - "Community 2732"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2733 - "Community 2733"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2734 - "Community 2734"
Cohesion: 1.0
Nodes (1): Args for MaskedLM Loss

### Community 2735 - "Community 2735"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2736 - "Community 2736"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2737 - "Community 2737"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2738 - "Community 2738"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2739 - "Community 2739"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2740 - "Community 2740"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2741 - "Community 2741"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2742 - "Community 2742"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2743 - "Community 2743"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2744 - "Community 2744"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2745 - "Community 2745"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2746 - "Community 2746"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2747 - "Community 2747"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2748 - "Community 2748"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2749 - "Community 2749"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2750 - "Community 2750"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2751 - "Community 2751"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2752 - "Community 2752"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2753 - "Community 2753"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2754 - "Community 2754"
Cohesion: 1.0
Nodes (1): Expected sizes:         delays: tgt_len, batch_size         src_lens: 1, batch_s

### Community 2755 - "Community 2755"
Cohesion: 1.0
Nodes (1): delays : bsz, num_heads_x_layers, tgt_len         src_lens : bsz, 1         targ

### Community 2756 - "Community 2756"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2757 - "Community 2757"
Cohesion: 1.0
Nodes (1): Reorder buffered internal state (for incremental generation).

### Community 2758 - "Community 2758"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2759 - "Community 2759"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2760 - "Community 2760"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2761 - "Community 2761"
Cohesion: 1.0
Nodes (1): Setup the task (e.g., load dictionaries).

### Community 2762 - "Community 2762"
Cohesion: 1.0
Nodes (1): Return the :class:`~fairseq.data.Dictionary` for the language         model.

### Community 2763 - "Community 2763"
Cohesion: 1.0
Nodes (1): Return the source :class:`~fairseq.data.Dictionary` (if applicable         for t

### Community 2764 - "Community 2764"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2765 - "Community 2765"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2766 - "Community 2766"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2767 - "Community 2767"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2768 - "Community 2768"
Cohesion: 1.0
Nodes (1): Add model-specific arguments to the parser.

### Community 2769 - "Community 2769"
Cohesion: 1.0
Nodes (1): Build a new model instance.

### Community 2770 - "Community 2770"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2771 - "Community 2771"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2772 - "Community 2772"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2773 - "Community 2773"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2774 - "Community 2774"
Cohesion: 1.0
Nodes (1): Add optimizer-specific arguments to the parser.

### Community 2775 - "Community 2775"
Cohesion: 1.0
Nodes (1): Return a kwarg dictionary that will be used to override optimizer         args s

### Community 2776 - "Community 2776"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2777 - "Community 2777"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2778 - "Community 2778"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2779 - "Community 2779"
Cohesion: 1.0
Nodes (1): Whether the logging outputs returned by `forward` can be summed         across w

### Community 2780 - "Community 2780"
Cohesion: 1.0
Nodes (1): Add criterion-specific arguments to the parser.

### Community 2781 - "Community 2781"
Cohesion: 1.0
Nodes (1): Aggregate logging outputs from data parallel training.

### Community 2782 - "Community 2782"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2783 - "Community 2783"
Cohesion: 1.0
Nodes (1): Load the dictionary from the filename          Args:             filename (str):

### Community 2784 - "Community 2784"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2785 - "Community 2785"
Cohesion: 1.0
Nodes (1): Load the dictionary from the filename          Args:             filename (str):

### Community 2786 - "Community 2786"
Cohesion: 1.0
Nodes (1): Generate a batch of translations.         Args:             models (List[~fairse

### Community 2787 - "Community 2787"
Cohesion: 1.0
Nodes (1): Add task-specific arguments to the parser.

### Community 2788 - "Community 2788"
Cohesion: 1.0
Nodes (0): 

### Community 2789 - "Community 2789"
Cohesion: 1.0
Nodes (0): 

### Community 2790 - "Community 2790"
Cohesion: 1.0
Nodes (0): 

### Community 2791 - "Community 2791"
Cohesion: 1.0
Nodes (1): Create a sentence embedder from a pretrained model.

### Community 2792 - "Community 2792"
Cohesion: 1.0
Nodes (1): Register memory parameters

### Community 2793 - "Community 2793"
Cohesion: 1.0
Nodes (1): Check and initialize memory parameters.

### Community 2794 - "Community 2794"
Cohesion: 1.0
Nodes (1): Create a dictionary from a vocabulary file.

### Community 2795 - "Community 2795"
Cohesion: 1.0
Nodes (1): Index sentences with a dictionary.

### Community 2796 - "Community 2796"
Cohesion: 1.0
Nodes (1): int: Input feature map levels.

### Community 2797 - "Community 2797"
Cohesion: 1.0
Nodes (0): 

### Community 2798 - "Community 2798"
Cohesion: 1.0
Nodes (0): 

### Community 2799 - "Community 2799"
Cohesion: 1.0
Nodes (0): 

### Community 2800 - "Community 2800"
Cohesion: 1.0
Nodes (0): 

### Community 2801 - "Community 2801"
Cohesion: 1.0
Nodes (1): Inference method. Switch model to `eval` mode,          call `.forward(x)` with

### Community 2802 - "Community 2802"
Cohesion: 1.0
Nodes (1): Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B

### Community 2803 - "Community 2803"
Cohesion: 1.0
Nodes (1): Convert ImageNet-normalized [C,H,W] tensor to uint8 [H,W,3] BGR for OpenCV.

### Community 2804 - "Community 2804"
Cohesion: 1.0
Nodes (1): Morphological opening (remove small protrusions) then closing (fill holes).

### Community 2805 - "Community 2805"
Cohesion: 1.0
Nodes (1): Fast bilateral solver for mask refinement.          Simplified version using Ope

### Community 2806 - "Community 2806"
Cohesion: 1.0
Nodes (1): Compute bounding boxes from binary masks. masks: [N, H, W] bool.

### Community 2807 - "Community 2807"
Cohesion: 1.0
Nodes (1): Updates the internal state of the metric. In particular, we track update the cos

### Community 2808 - "Community 2808"
Cohesion: 1.0
Nodes (1): Remaps the semantic classes to the target class using the latest assignments.

### Community 2809 - "Community 2809"
Cohesion: 1.0
Nodes (1): Getter method to access things prototypes.          Returns:             things_

### Community 2810 - "Community 2810"
Cohesion: 1.0
Nodes (1): Getter method to access stuffs prototypes.          Returns:             things_

### Community 2811 - "Community 2811"
Cohesion: 1.0
Nodes (1): Setter method to access stuffs prototypes.          Args:             value (Set

### Community 2812 - "Community 2812"
Cohesion: 1.0
Nodes (0): 

### Community 2813 - "Community 2813"
Cohesion: 1.0
Nodes (0): 

### Community 2814 - "Community 2814"
Cohesion: 1.0
Nodes (0): 

### Community 2815 - "Community 2815"
Cohesion: 1.0
Nodes (1): Build resume command using latest GCS checkpoint.

### Community 2816 - "Community 2816"
Cohesion: 1.0
Nodes (1): Attach SAM masks to samples before geometric augmentation.

### Community 2817 - "Community 2817"
Cohesion: 1.0
Nodes (1): Attach teacher logits before augmentation for aligned teacher gating.

### Community 2818 - "Community 2818"
Cohesion: 1.0
Nodes (1): Walk model hierarchy to find the ViT with .blocks attribute.

### Community 2819 - "Community 2819"
Cohesion: 1.0
Nodes (1): Rebuild optimizer (+ optional scheduler) for new LoRA parameters.

### Community 2820 - "Community 2820"
Cohesion: 1.0
Nodes (1): Unfreeze all LoRA adapter parameters. Returns count.

### Community 2821 - "Community 2821"
Cohesion: 1.0
Nodes (1): Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B

### Community 2822 - "Community 2822"
Cohesion: 1.0
Nodes (1): Updates the internal state of the metric. In particular, we track update the cos

### Community 2823 - "Community 2823"
Cohesion: 1.0
Nodes (1): Remaps the semantic classes to the target class using the latest assignments.

### Community 2824 - "Community 2824"
Cohesion: 1.0
Nodes (1): Getter method to access things prototypes.          Returns:             things_

### Community 2825 - "Community 2825"
Cohesion: 1.0
Nodes (1): Getter method to access stuffs prototypes.          Returns:             things_

### Community 2826 - "Community 2826"
Cohesion: 1.0
Nodes (1): Setter method to access things prototypes.          Args:             value (Set

### Community 2827 - "Community 2827"
Cohesion: 1.0
Nodes (1): Setter method to access stuffs prototypes.          Args:             value (Set

### Community 2828 - "Community 2828"
Cohesion: 1.0
Nodes (1): Construct from a yacs CfgNode (MODEL.LORA.MITIGATIONS).

### Community 2829 - "Community 2829"
Cohesion: 1.0
Nodes (1): Write averaged params into model. Returns count of params updated.

### Community 2830 - "Community 2830"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             sem_seg_head: a

### Community 2831 - "Community 2831"
Cohesion: 1.0
Nodes (1): Match proposals with groundtruth using the matcher at the given stage. Label the

### Community 2832 - "Community 2832"
Cohesion: 1.0
Nodes (1): Promote relaxed-IoU proposals for rare classes in later cascade stages.

### Community 2833 - "Community 2833"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape (Sh

### Community 2834 - "Community 2834"
Cohesion: 1.0
Nodes (1): Initialize DepthFiLMSemSegHead.          Args:             input_shape: shapes (

### Community 2835 - "Community 2835"
Cohesion: 1.0
Nodes (1): Build config dict from detectron2 config.          Args:             cfg: detect

### Community 2836 - "Community 2836"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             num_classes (in

### Community 2837 - "Community 2837"
Cohesion: 1.0
Nodes (1): Prepare some proposals to be used to train the ROI heads. It performs box matchi

### Community 2838 - "Community 2838"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             in_features (li

### Community 2839 - "Community 2839"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             box_in_features

### Community 2840 - "Community 2840"
Cohesion: 1.0
Nodes (1): NOTE: this interface is experimental.          Args:             input_shape: sh

### Community 2841 - "Community 2841"
Cohesion: 1.0
Nodes (1): Return sorted unique foreground values from a semantic PNG.          NOTE: For p

### Community 2842 - "Community 2842"
Cohesion: 1.0
Nodes (1): Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg

### Community 2843 - "Community 2843"
Cohesion: 1.0
Nodes (1): Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (

### Community 2844 - "Community 2844"
Cohesion: 1.0
Nodes (1): Perform the computation         Parameters:             outputs: raw outputs of

### Community 2845 - "Community 2845"
Cohesion: 1.0
Nodes (1): Files that match these patterns are not deleted by cleanup

### Community 2846 - "Community 2846"
Cohesion: 1.0
Nodes (1): maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and n

### Community 2847 - "Community 2847"
Cohesion: 1.0
Nodes (1): Explicit Test-Time Training adaptation.          For each image, perform K gradi

### Community 2848 - "Community 2848"
Cohesion: 1.0
Nodes (1): CRF-inspired pairwise consistency loss (differentiable CRF energy).          For

### Community 2849 - "Community 2849"
Cohesion: 1.0
Nodes (1): Fuse DINOv3 and SSD-1B features via learned cross-attention.          Args:

### Community 2850 - "Community 2850"
Cohesion: 1.0
Nodes (1): Compute Sobel gradients. depth_2d: (H, W) → (2, H, W).

### Community 2851 - "Community 2851"
Cohesion: 1.0
Nodes (1): Confidence mask: 1.0 if >= threshold of 8 neighbors share same class.          R

### Community 2852 - "Community 2852"
Cohesion: 1.0
Nodes (1): Get instance masks from the model.          Args:             features: (B, N, 7

### Community 2853 - "Community 2853"
Cohesion: 1.0
Nodes (1): Generate pseudo-labels from EMA teacher predictions.          Returns semantic l

### Community 2854 - "Community 2854"
Cohesion: 1.0
Nodes (1): Generate pseudo-labels with optional TTA.          Returns:             labels:

### Community 2855 - "Community 2855"
Cohesion: 1.0
Nodes (1): Extract CLS attention map as (H_patches, W_patches) numpy array.          Args:

### Community 2856 - "Community 2856"
Cohesion: 1.0
Nodes (1): Args:             img: (1, 3, H, W) tensor, normalized         Returns:

### Community 2857 - "Community 2857"
Cohesion: 1.0
Nodes (1): Compute Sobel gradients. depth_2d: (H, W) → (2, H, W).

### Community 2858 - "Community 2858"
Cohesion: 1.0
Nodes (1): Extract self-attention affinity matrix from last layer.          Args:

### Community 2859 - "Community 2859"
Cohesion: 1.0
Nodes (1): Apply graph diffusion to affinity matrix W (N, N).

### Community 2860 - "Community 2860"
Cohesion: 1.0
Nodes (1): Propagate features through the initial affinity graph.          Implements lazy

### Community 2861 - "Community 2861"
Cohesion: 1.0
Nodes (1): Nonlinear activation: φ(x) = x + 1.5·ELU(x).          For x > 0: φ(x) = 2.5x  (a

### Community 2862 - "Community 2862"
Cohesion: 1.0
Nodes (1): Refine a discrete segmentation map via NAMR.          Args:             pred:  (

### Community 2863 - "Community 2863"
Cohesion: 1.0
Nodes (1): Extract SD self-attention features for a single image.          Args:

### Community 2864 - "Community 2864"
Cohesion: 1.0
Nodes (1): Extract SSD-1B self-attention features for a single image.          Args:

### Community 2865 - "Community 2865"
Cohesion: 1.0
Nodes (1): Extract patch tokens from images.          Args:             images: (B, 3, H, W

### Community 2866 - "Community 2866"
Cohesion: 1.0
Nodes (1): Bipartite matching between predictions and targets.          Returns:

### Community 2867 - "Community 2867"
Cohesion: 1.0
Nodes (1): (B, 3, H, W) -> (B, N, 768) — CLS+registers already stripped by DINOv3ViTB.

### Community 2868 - "Community 2868"
Cohesion: 1.0
Nodes (1): 27-class Cityscapes mIoU with Hungarian matching.

### Community 2869 - "Community 2869"
Cohesion: 1.0
Nodes (1): Post-process a batch of predictions.          Args:             pred_logits: (B,

### Community 2870 - "Community 2870"
Cohesion: 1.0
Nodes (1): Extract multi-layer features as 2D spatial maps.          Args:             pixe

### Community 2871 - "Community 2871"
Cohesion: 1.0
Nodes (1): Extract DINO features from input image.          Args:             x: Input imag

### Community 2872 - "Community 2872"
Cohesion: 1.0
Nodes (1): Extract DINOv3 features.          Args:             x: Input image (B, 3, H, W),

### Community 2873 - "Community 2873"
Cohesion: 1.0
Nodes (1): Load pretrained DINOv3 weights from HuggingFace.          Args:             mode

### Community 2874 - "Community 2874"
Cohesion: 1.0
Nodes (1): Forward pass should work on MPS.

### Community 2875 - "Community 2875"
Cohesion: 1.0
Nodes (1): Backward pass should work on MPS.

### Community 2876 - "Community 2876"
Cohesion: 1.0
Nodes (1): 4D image input → same shape output.

### Community 2877 - "Community 2877"
Cohesion: 1.0
Nodes (1): 3D sequence input → same shape output.

### Community 2878 - "Community 2878"
Cohesion: 1.0
Nodes (1): All scan modes should support backward pass on images.

### Community 2879 - "Community 2879"
Cohesion: 1.0
Nodes (1): Output should not contain NaN.

### Community 2880 - "Community 2880"
Cohesion: 1.0
Nodes (1): 4D image inputs → same shape outputs.

### Community 2881 - "Community 2881"
Cohesion: 1.0
Nodes (1): 3D sequence inputs → same shape outputs.

### Community 2882 - "Community 2882"
Cohesion: 1.0
Nodes (1): Cross-modal should support backward pass.

### Community 2883 - "Community 2883"
Cohesion: 1.0
Nodes (1): Output should not contain NaN.

### Community 2884 - "Community 2884"
Cohesion: 1.0
Nodes (1): VisionMamba2 forward on MPS for all scan modes.

### Community 2885 - "Community 2885"
Cohesion: 1.0
Nodes (1): VisionMamba2 backward on MPS.

### Community 2886 - "Community 2886"
Cohesion: 1.0
Nodes (1): CrossModalMamba2 on MPS.

### Community 2887 - "Community 2887"
Cohesion: 1.0
Nodes (1): GatedDeltaNet forward pass on MPS.

### Community 2888 - "Community 2888"
Cohesion: 1.0
Nodes (1): GatedDeltaNet backward pass on MPS.

### Community 2889 - "Community 2889"
Cohesion: 1.0
Nodes (1): 4D image input with GDN layer → same shape output.

### Community 2890 - "Community 2890"
Cohesion: 1.0
Nodes (1): All scan modes with GDN should support backward pass.

### Community 2891 - "Community 2891"
Cohesion: 1.0
Nodes (1): GDN output should not contain NaN.

### Community 2892 - "Community 2892"
Cohesion: 1.0
Nodes (1): 4D image inputs with GDN → same shape outputs.

### Community 2893 - "Community 2893"
Cohesion: 1.0
Nodes (1): Cross-modal with GDN should support backward pass.

### Community 2894 - "Community 2894"
Cohesion: 1.0
Nodes (1): GDN cross-modal output should not contain NaN.

### Community 2895 - "Community 2895"
Cohesion: 1.0
Nodes (1): VisionMamba2 + GDN forward on MPS.

### Community 2896 - "Community 2896"
Cohesion: 1.0
Nodes (1): VisionMamba2 + GDN backward on MPS.

### Community 2897 - "Community 2897"
Cohesion: 1.0
Nodes (1): CrossModalMamba2 + GDN on MPS.

### Community 2898 - "Community 2898"
Cohesion: 1.0
Nodes (1): torch.Tensor: concatenated positive and negative boxes

### Community 2899 - "Community 2899"
Cohesion: 1.0
Nodes (1): Returns a dictionary of info about the object.

### Community 2900 - "Community 2900"
Cohesion: 1.0
Nodes (1): Sample positive samples.

### Community 2901 - "Community 2901"
Cohesion: 1.0
Nodes (1): Sample negative samples.

### Community 2902 - "Community 2902"
Cohesion: 1.0
Nodes (1): torch.Tensor: concatenated positive and negative boxes

### Community 2903 - "Community 2903"
Cohesion: 1.0
Nodes (1): Args:             rng (None | int | numpy.random.RandomState): seed or state.

### Community 2904 - "Community 2904"
Cohesion: 1.0
Nodes (1): int: number of feature levels that the generator will be applied

### Community 2905 - "Community 2905"
Cohesion: 1.0
Nodes (1): list[int]: The number of priors (points) at a point         on the feature grid

### Community 2906 - "Community 2906"
Cohesion: 1.0
Nodes (1): Placeholder for sample function.

### Community 2907 - "Community 2907"
Cohesion: 1.0
Nodes (1): Randomly select an img_scale from given candidates.          Args:             i

### Community 2908 - "Community 2908"
Cohesion: 1.0
Nodes (1): Randomly sample an img_scale when ``multiscale_mode=='range'``.          Args:

### Community 2909 - "Community 2909"
Cohesion: 1.0
Nodes (1): Randomly sample an img_scale when ``ratio_range`` is specified.          A ratio

### Community 2910 - "Community 2910"
Cohesion: 1.0
Nodes (1): Loss Name.          This function must be implemented and will return the name o

### Community 2911 - "Community 2911"
Cohesion: 1.0
Nodes (1): Forward function for `MultiheadAttention`.          **kwargs allow passing a mor

### Community 2912 - "Community 2912"
Cohesion: 1.0
Nodes (1): Forward function for `FFN`.          The function would add x to the output tens

### Community 2913 - "Community 2913"
Cohesion: 1.0
Nodes (1): Forward function for `FFN`.         The function would add x to the output tenso

### Community 2914 - "Community 2914"
Cohesion: 1.0
Nodes (1): Get the reference points used in decoder.          Args:             spatial_sha

### Community 2915 - "Community 2915"
Cohesion: 1.0
Nodes (1): Assign boxes to either a ground truth boxes or a negative boxes.

### Community 2916 - "Community 2916"
Cohesion: 1.0
Nodes (1): nn.Module: the normalization layer named "norm0"

### Community 2917 - "Community 2917"
Cohesion: 1.0
Nodes (1): nn.Module: the normalization layer named "norm1"

### Community 2918 - "Community 2918"
Cohesion: 1.0
Nodes (1): Resize pos_embed weights.          Resize pos_embed using bicubic interpolate me

### Community 2919 - "Community 2919"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the first convolution layer

### Community 2920 - "Community 2920"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the second convolution layer

### Community 2921 - "Community 2921"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the first convolution layer

### Community 2922 - "Community 2922"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the second convolution layer

### Community 2923 - "Community 2923"
Cohesion: 1.0
Nodes (1): nn.Module: normalization layer after the third convolution layer

### Community 2924 - "Community 2924"
Cohesion: 1.0
Nodes (1): nn.Module: the normalization layer named "norm1"

### Community 2925 - "Community 2925"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has neck

### Community 2926 - "Community 2926"
Cohesion: 1.0
Nodes (1): bool: whether the segmentor has decode head

### Community 2927 - "Community 2927"
Cohesion: 1.0
Nodes (1): Placeholder for extract features from images.

### Community 2928 - "Community 2928"
Cohesion: 1.0
Nodes (1): Placeholder for encode images with backbone and decode into a         semantic s

### Community 2929 - "Community 2929"
Cohesion: 1.0
Nodes (1): Placeholder for Forward function for training.

### Community 2930 - "Community 2930"
Cohesion: 1.0
Nodes (1): Placeholder for single image test.

### Community 2931 - "Community 2931"
Cohesion: 1.0
Nodes (1): Placeholder for augmentation test.

### Community 2932 - "Community 2932"
Cohesion: 1.0
Nodes (1): Calls either :func:`forward_train` or :func:`forward_test` depending         on

### Community 2933 - "Community 2933"
Cohesion: 1.0
Nodes (1): Loss function.          Args:             all_cls_scores (Tensor): Classificatio

### Community 2934 - "Community 2934"
Cohesion: 1.0
Nodes (1): Loss function.          Args:             all_cls_scores (Tensor): Classificatio

### Community 2935 - "Community 2935"
Cohesion: 1.0
Nodes (1): Loss function.          Args:             all_cls_scores (Tensor): Classificatio

### Community 2936 - "Community 2936"
Cohesion: 1.0
Nodes (1): Placeholder of forward function.

### Community 2937 - "Community 2937"
Cohesion: 1.0
Nodes (1): Compute segmentation loss.

### Community 2938 - "Community 2938"
Cohesion: 1.0
Nodes (1): r"""         Instantiate a [`SiglipConfig`] (or a derived class) from siglip tex

### Community 2939 - "Community 2939"
Cohesion: 1.0
Nodes (1): Make causal mask used for bi-directional self-attention.

### Community 2940 - "Community 2940"
Cohesion: 1.0
Nodes (1): Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_l

### Community 2941 - "Community 2941"
Cohesion: 1.0
Nodes (1): Detects whether the optional user-specified attention_mask & the automatically c

### Community 2942 - "Community 2942"
Cohesion: 1.0
Nodes (1): Preprocess an image or batch of images.          Args:             images (`Imag

### Community 2943 - "Community 2943"
Cohesion: 1.0
Nodes (0): 

### Community 2944 - "Community 2944"
Cohesion: 1.0
Nodes (0): 

### Community 2945 - "Community 2945"
Cohesion: 1.0
Nodes (0): 

### Community 2946 - "Community 2946"
Cohesion: 1.0
Nodes (0): 

### Community 2947 - "Community 2947"
Cohesion: 1.0
Nodes (0): 

### Community 2948 - "Community 2948"
Cohesion: 1.0
Nodes (0): 

### Community 2949 - "Community 2949"
Cohesion: 1.0
Nodes (0): 

### Community 2950 - "Community 2950"
Cohesion: 1.0
Nodes (0): 

### Community 2951 - "Community 2951"
Cohesion: 1.0
Nodes (0): 

### Community 2952 - "Community 2952"
Cohesion: 1.0
Nodes (0): 

### Community 2953 - "Community 2953"
Cohesion: 1.0
Nodes (0): 

### Community 2954 - "Community 2954"
Cohesion: 1.0
Nodes (0): 

### Community 2955 - "Community 2955"
Cohesion: 1.0
Nodes (0): 

### Community 2956 - "Community 2956"
Cohesion: 1.0
Nodes (0): 

### Community 2957 - "Community 2957"
Cohesion: 1.0
Nodes (0): 

### Community 2958 - "Community 2958"
Cohesion: 1.0
Nodes (0): 

### Community 2959 - "Community 2959"
Cohesion: 1.0
Nodes (0): 

### Community 2960 - "Community 2960"
Cohesion: 1.0
Nodes (0): 

### Community 2961 - "Community 2961"
Cohesion: 1.0
Nodes (0): 

### Community 2962 - "Community 2962"
Cohesion: 1.0
Nodes (0): 

### Community 2963 - "Community 2963"
Cohesion: 1.0
Nodes (0): 

### Community 2964 - "Community 2964"
Cohesion: 1.0
Nodes (0): 

### Community 2965 - "Community 2965"
Cohesion: 1.0
Nodes (0): 

### Community 2966 - "Community 2966"
Cohesion: 1.0
Nodes (0): 

### Community 2967 - "Community 2967"
Cohesion: 1.0
Nodes (0): 

### Community 2968 - "Community 2968"
Cohesion: 1.0
Nodes (0): 

### Community 2969 - "Community 2969"
Cohesion: 1.0
Nodes (0): 

### Community 2970 - "Community 2970"
Cohesion: 1.0
Nodes (0): 

### Community 2971 - "Community 2971"
Cohesion: 1.0
Nodes (0): 

### Community 2972 - "Community 2972"
Cohesion: 1.0
Nodes (0): 

### Community 2973 - "Community 2973"
Cohesion: 1.0
Nodes (0): 

### Community 2974 - "Community 2974"
Cohesion: 1.0
Nodes (0): 

### Community 2975 - "Community 2975"
Cohesion: 1.0
Nodes (1): x: (batch_size, seqlen, nheads, headdim)             cos, sin: (seqlen, rotary_d

### Community 2976 - "Community 2976"
Cohesion: 1.0
Nodes (1): logits: (batch, vocab_size)         labels: (batch,)         If process_group is

### Community 2977 - "Community 2977"
Cohesion: 1.0
Nodes (1): Apply one EMA step from the student's current parameters.

### Community 2978 - "Community 2978"
Cohesion: 1.0
Nodes (1): Per-query boolean mask: True => exclude this query from no-object CE.          S

### Community 2979 - "Community 2979"
Cohesion: 1.0
Nodes (1): Multi-scale + flip TTA: average final-block logits per query.          Returns (

### Community 2980 - "Community 2980"
Cohesion: 1.0
Nodes (1): Compute loss masks for each of standard reprojection and depth hint         repr

### Community 2981 - "Community 2981"
Cohesion: 1.0
Nodes (1): Compute proxy supervised loss (depth hint loss) for prediction.              - v

### Community 2982 - "Community 2982"
Cohesion: 1.0
Nodes (1): Compute loss masks for each of standard reprojection and depth hint         repr

### Community 2983 - "Community 2983"
Cohesion: 1.0
Nodes (1): Compute proxy supervised loss (depth hint loss) for prediction.              - v

### Community 2984 - "Community 2984"
Cohesion: 1.0
Nodes (1): Compute loss masks for each of standard reprojection and depth hint         repr

### Community 2985 - "Community 2985"
Cohesion: 1.0
Nodes (1): The best solution from the solver         Returns         -------         x : nd

### Community 2986 - "Community 2986"
Cohesion: 1.0
Nodes (1): The standard deviation of the population energies divided by their         mean.

### Community 2987 - "Community 2987"
Cohesion: 1.0
Nodes (0): 

### Community 2988 - "Community 2988"
Cohesion: 1.0
Nodes (1): If process_group is not None and sequence_parallel=True, we're doing Tensor Para

### Community 2989 - "Community 2989"
Cohesion: 1.0
Nodes (1): xz: (batch, dim, seqlen)

### Community 2990 - "Community 2990"
Cohesion: 1.0
Nodes (1): If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil

### Community 2991 - "Community 2991"
Cohesion: 1.0
Nodes (1): Update center used for teacher output.

### Community 2992 - "Community 2992"
Cohesion: 1.0
Nodes (1): Walk model hierarchy to find the ViT with .blocks attribute.

### Community 2993 - "Community 2993"
Cohesion: 1.0
Nodes (1): Rebuild optimizer (+ optional scheduler) for new LoRA parameters.

### Community 2994 - "Community 2994"
Cohesion: 1.0
Nodes (1): Unfreeze all LoRA adapter parameters. Returns count.

### Community 2995 - "Community 2995"
Cohesion: 1.0
Nodes (1): Convert one image's final-layer logits into the D2 panoptic format.          Pan

### Community 2996 - "Community 2996"
Cohesion: 1.0
Nodes (1): Forward pass through the model.          Args:             image: Input batch wi

### Community 2997 - "Community 2997"
Cohesion: 1.0
Nodes (1): Get scene data including images, camera parameters, and auxiliary info.

### Community 2998 - "Community 2998"
Cohesion: 1.0
Nodes (1): Evaluate 3D reconstruction quality against ground truth.          Args:

### Community 2999 - "Community 2999"
Cohesion: 1.0
Nodes (1): Fuse per-view depth maps into a single point cloud.          Args:             s

### Community 3000 - "Community 3000"
Cohesion: 1.0
Nodes (1): Directory for storing metric JSON files.

### Community 3001 - "Community 3001"
Cohesion: 1.0
Nodes (1): Convert numpy scalars to plain Python floats for JSON safety.

### Community 3002 - "Community 3002"
Cohesion: 1.0
Nodes (1): Compute elementwise mean across a list of homogeneous metric dicts.

### Community 3003 - "Community 3003"
Cohesion: 1.0
Nodes (1): Write JSON with UTF-8 and pretty indentation.

### Community 3004 - "Community 3004"
Cohesion: 1.0
Nodes (1): Fit global mean/V3 and initialize percentiles from a reference set.         fram

### Community 3005 - "Community 3005"
Cohesion: 1.0
Nodes (1): X: (N,D) where N = H*W         Returns PCs_raw: (N,3) using stable basis (fixed

### Community 3006 - "Community 3006"
Cohesion: 1.0
Nodes (1): frame: (H,W,D) -> (H,W,3)

### Community 3007 - "Community 3007"
Cohesion: 1.0
Nodes (1): frames: (T,H,W,D) or list of (H,W,D)         returns: (T,H,W,3)

### Community 3008 - "Community 3008"
Cohesion: 1.0
Nodes (1): Performs feature rotation by splitting and recombining feature dimensions.

### Community 3009 - "Community 3009"
Cohesion: 1.0
Nodes (1): Handle export directory

### Community 3010 - "Community 3010"
Cohesion: 1.0
Nodes (1): Process image directory

### Community 3011 - "Community 3011"
Cohesion: 1.0
Nodes (1): Process video, extract frames

### Community 3012 - "Community 3012"
Cohesion: 1.0
Nodes (0): 

### Community 3013 - "Community 3013"
Cohesion: 1.0
Nodes (0): 

### Community 3014 - "Community 3014"
Cohesion: 1.0
Nodes (1): Extract SD self-attention features.          Args:             image: (1, 3, H,

### Community 3015 - "Community 3015"
Cohesion: 1.0
Nodes (0): 

### Community 3016 - "Community 3016"
Cohesion: 1.0
Nodes (1): (h, w, C) -> bilinear -> (hw[0]*hw[1], C).

### Community 3017 - "Community 3017"
Cohesion: 1.0
Nodes (1): Align feature sequence length via interpolation.          Args:             feat

### Community 3018 - "Community 3018"
Cohesion: 1.0
Nodes (1): Transform DINOv3 features using depth conditioning.          Args:             f

## Knowledge Gaps
- **19703 isolated node(s):** `setup_notebooklm.py — Bootstrap a NotebookLM notebook for MBPS BMVC 2026.  Creat`, `Run notebooklm CLI with given args.`, `Run CLI with --json flag and parse output.`, `Return notebook ID if a notebook named NOTEBOOK_NAME already exists.`, `Create the notebook and return its ID.` (+19698 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 840`** (2 nodes): `masks_to_coco.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 841`** (2 nodes): `check_official_runtime.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 842`** (2 nodes): `run_rama_official.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 843`** (2 nodes): `run_mcg_official.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 844`** (2 nodes): `run_solo_official.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 845`** (2 nodes): `gpuMSTdpk.h`, `MST_kruskal()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 846`** (2 nodes): `gpuMST.h`, `MST_boruvka()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 847`** (2 nodes): `coco_eval.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 848`** (2 nodes): `mex_intersect_hierarchies.cpp`, `mexFunction()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 849`** (2 nodes): `smoke_model.py`, `build_smoke_query_model()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 850`** (2 nodes): `evaluate_coco_boundary_ap.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 851`** (2 nodes): `split_coco_train_sup10_usemask.py`, `split_json()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 852`** (2 nodes): `split_coco_train_sup10.py`, `split_json()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 853`** (2 nodes): `gdrive_downloader.py`, `download()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 854`** (2 nodes): `coco_loader.py`, `coco_loader_lsj.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 855`** (2 nodes): `prepare_coco_point_annotations_without_masks.py`, `get_point_annotations()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 856`** (2 nodes): `gen_install_table.py`, `gen_header()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 857`** (2 nodes): `check_data.py`, `Utility script to check dataset integrity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 858`** (2 nodes): `sweep_object_centric.py`, `get_sweep()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 859`** (2 nodes): `multi_modality_demo.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 860`** (2 nodes): `pcd_demo.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 861`** (2 nodes): `mono_det_demo.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 862`** (2 nodes): `pc_seg_demo.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 863`** (2 nodes): `lyft_data_fixer.py`, `fix_lyft()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 864`** (2 nodes): `test_fpn.py`, `test_secfpn()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 865`** (2 nodes): `test_voxel_generator.py`, `test_voxel_generator()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 866`** (2 nodes): `test_setup_env.py`, `test_setup_multi_processes()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 867`** (2 nodes): `test_semantickitti_dataset.py`, `test_getitem()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 868`** (2 nodes): `test_load_points_from_multi_sweeps.py`, `test_load_points_from_multi_sweeps()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 869`** (2 nodes): `ground_segmentation.py`, `transpose_split_nusc_pcl()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 870`** (2 nodes): `trafo_conversion.py`, `nusc_vehicle_pcl_to_kitti_lidar()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 871`** (2 nodes): `timing_utils.py`, `timeit()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 872`** (2 nodes): `mined_box_db_utils.py`, `load_mined_boxes_db()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 873`** (2 nodes): `generate_api_docs.py`, `Generate the code reference pages and navigation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 874`** (2 nodes): `google_drive.py`, `download_file_from_google_drive()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 875`** (2 nodes): `load_data.py`, `load_data()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 876`** (2 nodes): `download_syn.py`, `download_data()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 877`** (2 nodes): `download_movi_data.py`, `_download()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 878`** (2 nodes): `boxes3d.py`, `fit()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 879`** (2 nodes): `pt3d_oflow.py`, `fit_se3_to_pt3d_oflow_and_masks()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 880`** (2 nodes): `_3d.py`, `visualize_se3s()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 881`** (2 nodes): `preprocess_waymo.py`, `preprocess()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 882`** (2 nodes): `generate_2d_anno.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 883`** (2 nodes): `preprocess_sf_waymo.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 884`** (2 nodes): `get_model_size.py`, `get_model_size()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 885`** (2 nodes): `things.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 886`** (2 nodes): `multiframes_sintel_submission.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 887`** (2 nodes): `things_multiframes.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 888`** (2 nodes): `kitti_multiframes.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 889`** (2 nodes): `sintel.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 890`** (2 nodes): `sintel_submission.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 891`** (2 nodes): `sintel_multiframes.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 892`** (2 nodes): `test_flow_utils.py`, `test_read_write_pfm()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 893`** (2 nodes): `get_loss.py`, `get_loss()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 894`** (2 nodes): `aug_params.py`, `get_params()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 895`** (2 nodes): `pretrain_config.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 896`** (2 nodes): `submissions.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 897`** (2 nodes): `pretrained_download.py`, `@article{hamilton2022unsupervised,   title={Unsupervised Semantic Segmentation b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 898`** (2 nodes): `test_trackio_space_ids.py`, `test_trackio_space_examples_use_hyphenated_ml_intern_prefix()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 899`** (2 nodes): `data_preparation.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 900`** (2 nodes): `_run_likelihood_ratio.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 901`** (2 nodes): `img_transforms.py`, `default_transform()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 902`** (2 nodes): `skeleton_drawer.py`, `draw_skeleton()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 903`** (2 nodes): `apply_pseudo_labels.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 904`** (2 nodes): `tag_generate_idx.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 905`** (2 nodes): `replace_labels.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 906`** (2 nodes): `extract_trans.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 907`** (2 nodes): `create_wikidata_db.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 908`** (2 nodes): `evaluate-ckpt-multilingual.py`, `read_score()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 909`** (2 nodes): `evaluate-ckpt-multidomain.py`, `read_score()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 910`** (2 nodes): `clean_histogram.py`, `read_hist()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 911`** (2 nodes): `aggregate_scores.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 912`** (2 nodes): `normalize.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 913`** (2 nodes): `dedup_all.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 914`** (2 nodes): `libri_labels.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 915`** (2 nodes): `detok.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 916`** (2 nodes): `compare_namespaces.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 917`** (2 nodes): `build_sym_alignment.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 918`** (2 nodes): `spm_decode.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 919`** (2 nodes): `count_docs.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 920`** (2 nodes): `spm_encode.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 921`** (2 nodes): `shard_docs.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 922`** (2 nodes): `remap_cause27_to_trainid.py`, `remap_split()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 923`** (2 nodes): `validate_cups_stage3.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 924`** (2 nodes): `upload_hf_remaining.py`, `upload_with_retry()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 925`** (2 nodes): `diagnose_masks.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 926`** (2 nodes): `test_rare_pool_builder.py`, `test_rare_pool_builder_extracts_synthetic_instances()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 927`** (2 nodes): `mumford_shah_phase_b.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 928`** (2 nodes): `config_mamba.py`, `MambaConfig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 929`** (2 nodes): `softplus.py`, `softplus()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 930`** (2 nodes): `prepare_unmore_coco20k_symlink_mirror.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 931`** (2 nodes): `prepare_unmore_kitti_symlink_mirror.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 932`** (2 nodes): `validate_methodology_uis_bank.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 933`** (2 nodes): `test_mine_unsup_object_evidence.py`, `test_mine_unsup_object_evidence_writes_candidate_bank()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 934`** (2 nodes): `test_fuse_cache_stage0_candidate_bank.py`, `test_fuse_preserves_cache_anchor_and_appends_stage0()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 935`** (2 nodes): `make_qualitative.py`, `load()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 936`** (1 nodes): `sample.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 937`** (1 nodes): `loadvar.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 938`** (1 nodes): `solo_r50_fpn_class_agnostic_pseudo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 939`** (1 nodes): `solo_r101_fpn_class_agnostic_ranker_pseudo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 940`** (1 nodes): `solo_r101_fpn_class_agnostic_selftrain.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 941`** (1 nodes): `solo_r101_fpn_class_agnostic_pseudo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 942`** (1 nodes): `faithful_status.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 943`** (1 nodes): `munster_000169_000019_mcg.run_mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 944`** (1 nodes): `frankfurt_000000_003920_mcg.run_mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 945`** (1 nodes): `frankfurt_000000_002196_mcg.run_mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 946`** (1 nodes): `000000000049_mcg.run_mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 947`** (1 nodes): `ECLgraph.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 948`** (1 nodes): `verify_correctness.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 949`** (1 nodes): `inference_demo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 950`** (1 nodes): `retinanet_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 951`** (1 nodes): `retinanet_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 952`** (1 nodes): `faster_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 953`** (1 nodes): `ssd512_coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 954`** (1 nodes): `rpn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 955`** (1 nodes): `mask_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 956`** (1 nodes): `cascade_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 957`** (1 nodes): `mask_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 958`** (1 nodes): `faster_rcnn_ohem_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 959`** (1 nodes): `cascade_mask_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 960`** (1 nodes): `ssd300_coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 961`** (1 nodes): `cascade_rcnn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 962`** (1 nodes): `faster_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 963`** (1 nodes): `rpn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 964`** (1 nodes): `mask_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 965`** (1 nodes): `retinanet_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 966`** (1 nodes): `cascade_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 967`** (1 nodes): `cascade_mask_rcnn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 968`** (1 nodes): `faster_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 969`** (1 nodes): `fast_mask_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 970`** (1 nodes): `cascade_mask_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 971`** (1 nodes): `faster_rcnn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 972`** (1 nodes): `fast_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 973`** (1 nodes): `cascade_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 974`** (1 nodes): `fast_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 975`** (1 nodes): `rpn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 976`** (1 nodes): `retinanet_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 977`** (1 nodes): `cascade_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 978`** (1 nodes): `mask_rcnn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 979`** (1 nodes): `fast_mask_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 980`** (1 nodes): `rpn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 981`** (1 nodes): `cascade_mask_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 982`** (1 nodes): `rpn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 983`** (1 nodes): `fast_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 984`** (1 nodes): `faster_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 985`** (1 nodes): `fast_mask_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 986`** (1 nodes): `cascade_mask_rcnn_r50_caffe_c4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 987`** (1 nodes): `mask_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 988`** (1 nodes): `libra_faster_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 989`** (1 nodes): `libra_fast_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 990`** (1 nodes): `libra_faster_rcnn_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 991`** (1 nodes): `libra_retinanet_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 992`** (1 nodes): `libra_faster_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 993`** (1 nodes): `ms_rcnn_r101_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 994`** (1 nodes): `ms_rcnn_x101_64x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 995`** (1 nodes): `ms_rcnn_r50_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 996`** (1 nodes): `ssd300_coco_instaboost_4x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 997`** (1 nodes): `cascade_mask_rcnn_r50_fpn_instaboost_4x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 998`** (1 nodes): `mask_rcnn_r50_fpn_instaboost_4x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 999`** (1 nodes): `htc_r50_fpn_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1000`** (1 nodes): `htc_r101_fpn_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1001`** (1 nodes): `htc_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1002`** (1 nodes): `htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1003`** (1 nodes): `htc_without_semantic_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1004`** (1 nodes): `htc_x101_64x4d_fpn_20e_16gpu.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1005`** (1 nodes): `htc_x101_32x4d_fpn_20e_16gpu.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1006`** (1 nodes): `ga_faster_r50_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1007`** (1 nodes): `ga_fast_r50_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1008`** (1 nodes): `ga_retinanet_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1009`** (1 nodes): `ga_faster_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1010`** (1 nodes): `ga_rpn_r101_caffe_rpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1011`** (1 nodes): `ga_retinanet_r50_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1012`** (1 nodes): `ga_rpn_r50_caffe_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1013`** (1 nodes): `ga_rpn_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1014`** (1 nodes): `mask_rcnn_r50_fpn_gn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1015`** (1 nodes): `mask_rcnn_r50_fpn_gn_contrib_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1016`** (1 nodes): `mask_rcnn_r101_fpn_gn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1017`** (1 nodes): `cascade_rcnn_hrnetv2p_w32_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1018`** (1 nodes): `mask_rcnn_hrnetv2p_w32_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1019`** (1 nodes): `faster_rcnn_hrnetv2p_w40_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1020`** (1 nodes): `faster_rcnn_hrnetv2p_w18_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1021`** (1 nodes): `faster_rcnn_hrnetv2p_w32_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1022`** (1 nodes): `cascade_mask_rcnn_hrnetv2p_w32_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1023`** (1 nodes): `mask_rcnn_hrnetv2p_w18_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1024`** (1 nodes): `fcos_hrnetv2p_w32_gn_1x_4gpu.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1025`** (1 nodes): `htc_hrnetv2p_w32_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1026`** (1 nodes): `reppoints_partial_minmax_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1027`** (1 nodes): `bbox_r50_grid_center_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1028`** (1 nodes): `reppoints_moment_r101_fpn_2x_mt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1029`** (1 nodes): `bbox_r50_grid_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1030`** (1 nodes): `reppoints_minmax_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1031`** (1 nodes): `reppoints_moment_r101_dcn_fpn_2x_mt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1032`** (1 nodes): `reppoints_moment_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1033`** (1 nodes): `reppoints_moment_x101_dcn_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1034`** (1 nodes): `reppoints_moment_r50_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1035`** (1 nodes): `reppoints_moment_r50_no_gn_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1036`** (1 nodes): `reppoints_moment_r50_fpn_2x_mt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1037`** (1 nodes): `reppoints_moment_x101_dcn_fpn_2x_mt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1038`** (1 nodes): `reppoints_moment_r101_dcn_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1039`** (1 nodes): `reppoints_moment_r101_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1040`** (1 nodes): `faster_rcnn_r50_fpn_attention_1111_dcn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1041`** (1 nodes): `faster_rcnn_r50_fpn_attention_0010_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1042`** (1 nodes): `faster_rcnn_r50_fpn_attention_1111_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1043`** (1 nodes): `faster_rcnn_r50_fpn_attention_0010_dcn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1044`** (1 nodes): `faster_rcnn_r50_fpn_1x_voc0712.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1045`** (1 nodes): `ssd512_voc.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1046`** (1 nodes): `ssd300_voc.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1047`** (1 nodes): `retinanet_crop640_r50_nasfpn_50e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1048`** (1 nodes): `retinanet_crop640_r50_fpn_50e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1049`** (1 nodes): `mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1050`** (1 nodes): `mask_rcnn_r50_fpn_sbn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1051`** (1 nodes): `mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1052`** (1 nodes): `mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1053`** (1 nodes): `mask_rcnn_r16_gcb_c3-c5_r50_fpn_syncbn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1054`** (1 nodes): `scratch_faster_rcnn_r50_fpn_gn_6x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1055`** (1 nodes): `scratch_mask_rcnn_r50_fpn_gn_6x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1056`** (1 nodes): `retinanet_ghm_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1057`** (1 nodes): `decoupled_solo_r50_fpn_8gpu_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1058`** (1 nodes): `decoupled_solo_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1059`** (1 nodes): `decoupled_solo_light_dcn_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1060`** (1 nodes): `solo_r50_fpn_8gpu_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1061`** (1 nodes): `solo_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1062`** (1 nodes): `decoupled_solo_r101_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1063`** (1 nodes): `solo_r101_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1064`** (1 nodes): `decoupled_solo_light_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1065`** (1 nodes): `fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1066`** (1 nodes): `fcos_r50_caffe_fpn_gn_1x_4gpu.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1067`** (1 nodes): `fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1068`** (1 nodes): `dh_faster_rcnn_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1069`** (1 nodes): `mask_rcnn_mdconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1070`** (1 nodes): `faster_rcnn_dpool_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1071`** (1 nodes): `mask_rcnn_dconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1072`** (1 nodes): `cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1073`** (1 nodes): `faster_rcnn_mdpool_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1074`** (1 nodes): `faster_rcnn_mdconv_c3-c5_group4_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1075`** (1 nodes): `cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1076`** (1 nodes): `faster_rcnn_mdconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1077`** (1 nodes): `faster_rcnn_dconv_c3-c5_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1078`** (1 nodes): `faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1079`** (1 nodes): `grid_rcnn_gn_head_x101_32x4d_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1080`** (1 nodes): `grid_rcnn_gn_head_r50_fpn_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1081`** (1 nodes): `mask_rcnn_r50_fpn_gn_ws_20_23_24e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1082`** (1 nodes): `mask_rcnn_r50_fpn_gn_ws_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1083`** (1 nodes): `mask_rcnn_x101_32x4d_fpn_gn_ws_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1084`** (1 nodes): `faster_rcnn_r50_fpn_gn_ws_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1085`** (1 nodes): `solov2_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1086`** (1 nodes): `solov2_r50_fpn_8gpu_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1087`** (1 nodes): `solov2_light_448_r18_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1088`** (1 nodes): `solov2_r101_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1089`** (1 nodes): `solov2_light_512_dcn_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1090`** (1 nodes): `solov2_r101_dcn_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1091`** (1 nodes): `solov2_x101_dcn_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1092`** (1 nodes): `solov2_light_448_r50_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1093`** (1 nodes): `solov2_light_448_r34_fpn_8gpu_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1094`** (1 nodes): `retinanet_free_anchor_x101-32x4d_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1095`** (1 nodes): `retinanet_free_anchor_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1096`** (1 nodes): `retinanet_free_anchor_r101_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1097`** (1 nodes): `fovea_align_gn_r101_fpn_4gpu_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1098`** (1 nodes): `fovea_align_gn_ms_r101_fpn_4gpu_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1099`** (1 nodes): `fovea_r50_fpn_4gpu_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1100`** (1 nodes): `fovea_align_gn_ms_r50_fpn_4gpu_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1101`** (1 nodes): `fovea_align_gn_r50_fpn_4gpu_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1102`** (1 nodes): `faster_rcnn_r50_fpn_1x_cityscapes.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1103`** (1 nodes): `mask_rcnn_r50_fpn_1x_cityscapes.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1104`** (1 nodes): `retinanet_r50_fpn_fp16_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1105`** (1 nodes): `mask_rcnn_r50_fpn_fp16_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1106`** (1 nodes): `faster_rcnn_r50_fpn_fp16_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1107`** (1 nodes): `ssd300_wider_face.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1108`** (1 nodes): `atss_r50_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1109`** (1 nodes): `Return the number of predictions in this assignment`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1110`** (1 nodes): `Returns a dictionary of info about the object`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1111`** (1 nodes): `Create random AssignResult for tests or debugging.          Kwargs:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1112`** (1 nodes): `Returns a dictionary of info about the object.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1113`** (1 nodes): `Args:             rng (None | int | numpy.random.RandomState): seed or state`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1114`** (1 nodes): `Dictionary mapper.         Renames keys according to keymap provided.          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1115`** (1 nodes): `Transform network output for a batch into labeled boxes.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1116`** (1 nodes): `int: Input feature map levels.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1117`** (1 nodes): `Async test only det bboxes without augmentation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1118`** (1 nodes): `Compute full log-likelihood of a string, with no truncation, for perplexity comp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1119`** (1 nodes): `Calls either forward_train or forward_test depending on whether         return_l`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1120`** (1 nodes): `Compute target of mask IoU.          Mask IoU target is the IoU of the predicted`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1121`** (1 nodes): `Get the mask scores.          mask_score = bbox_score * mask_iou`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1122`** (1 nodes): `root_dir.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1123`** (1 nodes): `check_dbs.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1124`** (1 nodes): `results_box_per_class.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1125`** (1 nodes): `results_segm_proposals.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1126`** (1 nodes): `results_box_proposals.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1127`** (1 nodes): `demo_eval.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1128`** (1 nodes): `db_ids.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1129`** (1 nodes): `gt_wrappers_root.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1130`** (1 nodes): `db_im.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1131`** (1 nodes): `demo_script.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1132`** (1 nodes): `db_gt.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1133`** (1 nodes): `db_show_sseg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1134`** (1 nodes): `db_root_dir.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1135`** (1 nodes): `pascal_colormap.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1136`** (1 nodes): `eval_list.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1137`** (1 nodes): `eval_proposals.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1138`** (1 nodes): `eval_parallel.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1139`** (1 nodes): `eval_one.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1140`** (1 nodes): `show_per_class_table.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1141`** (1 nodes): `write_to_file.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1142`** (1 nodes): `recallatoverlap.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1143`** (1 nodes): `jaccard.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1144`** (1 nodes): `write_boxes_to_file.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1145`** (1 nodes): `average_recall.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1146`** (1 nodes): `seg2bmap.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1147`** (1 nodes): `plot_one_soa.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1148`** (1 nodes): `mask_image.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1149`** (1 nodes): `write_boxes_per_class_to_file.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1150`** (1 nodes): `overlay_contour.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1151`** (1 nodes): `mask2box.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1152`** (1 nodes): `eval_boxes_parallel.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1153`** (1 nodes): `box_area.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1154`** (1 nodes): `boxes_iou.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1155`** (1 nodes): `boxes_intersection.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1156`** (1 nodes): `eval_boxes_list.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1157`** (1 nodes): `labels2boxes.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1158`** (1 nodes): `eval_boxes.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1159`** (1 nodes): `im2mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1160`** (1 nodes): `im2ucm.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1161`** (1 nodes): `get_ground_truth.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1162`** (1 nodes): `database_ids.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1163`** (1 nodes): `database_root_dir.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1164`** (1 nodes): `get_image.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1165`** (1 nodes): `rank_training.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1166`** (1 nodes): `compute_all_ucms.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1167`** (1 nodes): `pareto_choose_point.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1168`** (1 nodes): `compute_mcg_cands.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1169`** (1 nodes): `benchmark_results.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1170`** (1 nodes): `im2mcg_all.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1171`** (1 nodes): `im2ucm_all.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1172`** (1 nodes): `demo_im2mcg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1173`** (1 nodes): `demo_im2ucm.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1174`** (1 nodes): `eval_masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1175`** (1 nodes): `eval_labels.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1176`** (1 nodes): `eval_cands.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1177`** (1 nodes): `eval_and_save_masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1178`** (1 nodes): `eval_and_save_labels.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1179`** (1 nodes): `whiten.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1180`** (1 nodes): `resample_ucm2_sp.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1181`** (1 nodes): `contours2OWT.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1182`** (1 nodes): `seg2bdry.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1183`** (1 nodes): `project_ucms_wrap.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1184`** (1 nodes): `img2ucms.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1185`** (1 nodes): `resample_ucm2.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1186`** (1 nodes): `seg2bdry_wt.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1187`** (1 nodes): `apply_sigmoid.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1188`** (1 nodes): `spectralPb_fast.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1189`** (1 nodes): `empty_ucm.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1190`** (1 nodes): `test_hole_filling.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1191`** (1 nodes): `test_mex_get_tree_cands.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1192`** (1 nodes): `check_hier_correctness.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1193`** (1 nodes): `test_base_perimeters.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1194`** (1 nodes): `reproducibility.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1195`** (1 nodes): `test_ucm2hier.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1196`** (1 nodes): `test_jaccard.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1197`** (1 nodes): `test_cands2masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1198`** (1 nodes): `hole_filling.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1199`** (1 nodes): `compute_full_features.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1200`** (1 nodes): `compute_base_features.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1201`** (1 nodes): `fuse_bpts.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1202`** (1 nodes): `full_cands_from_hiers.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1203`** (1 nodes): `combine_masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1204`** (1 nodes): `get_single_hier_stats.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1205`** (1 nodes): `extract_one_from_all.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1206`** (1 nodes): `pareto_learning.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1207`** (1 nodes): `join_masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1208`** (1 nodes): `pareto_combination.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1209`** (1 nodes): `get_params.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1210`** (1 nodes): `sf_mUCM_multi_3sc_u_4r_12k_params.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1211`** (1 nodes): `cands2masks.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1212`** (1 nodes): `ms_matrix2struct.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1213`** (1 nodes): `cands2labels.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1214`** (1 nodes): `gridbmap2seg.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1215`** (1 nodes): `create_train_samples.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1216`** (1 nodes): `ucm2hier.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1217`** (1 nodes): `write_jaccard_to_file.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1218`** (1 nodes): `seg2gridbmap.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1219`** (1 nodes): `tutorial_RegRF.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1220`** (1 nodes): `regRF_predict.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1221`** (1 nodes): `compile_linux.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1222`** (1 nodes): `regRF_train.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1223`** (1 nodes): `rfImpute.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1224`** (1 nodes): `compile_windows.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1225`** (1 nodes): `test_RegRF_extensively.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1226`** (1 nodes): `edgesDemo.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1227`** (1 nodes): `edgesSweeps.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1228`** (1 nodes): `edgesTrain.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1229`** (1 nodes): `edgesChns.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1230`** (1 nodes): `Contents.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1231`** (1 nodes): `edgesDetect.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1232`** (1 nodes): `edgesEval.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1233`** (1 nodes): `gradientMag.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1234`** (1 nodes): `imPad.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1235`** (1 nodes): `imResample.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1236`** (1 nodes): `gradientHist.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1237`** (1 nodes): `convTri.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1238`** (1 nodes): `rgbConvert.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1239`** (1 nodes): `paretoGroup.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1240`** (1 nodes): `ictest.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1241`** (1 nodes): `box2mask.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1242`** (1 nodes): `mcg_root.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1243`** (1 nodes): `getGaussianAffinity.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1244`** (1 nodes): `ncuts_downsample.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1245`** (1 nodes): `go.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1246`** (1 nodes): `betterjet.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1247`** (1 nodes): `ncuts.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1248`** (1 nodes): `dncuts.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1249`** (1 nodes): `Computation of the ARI clustering metric.      NOTE: This implementation does no`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1250`** (1 nodes): `See `Ari` docstring for allowed keyword arguments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1251`** (1 nodes): `Returns the transformed tensor.      Args:       tensor: Any of a set of differe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1252`** (1 nodes): `Returns the transformed tensor.      Args:       tensor: Any of a set of differe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1253`** (1 nodes): `Slot Attention module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1254`** (1 nodes): `Computes inverted dot-product attention.      Args:       query: Queries with sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1255`** (1 nodes): `Computes multi-head dot-product attention given query, key, and value.      Args`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1256`** (1 nodes): `Apply the ResNet to the inputs `x`.      Args:       x: Inputs.       train: Whe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1257`** (1 nodes): `Performs a forward pass on a video.      Args:       video: Video of shape `[bat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1258`** (1 nodes): `link_coco20k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1259`** (1 nodes): `Get parameters for ``crop`` for a random sized crop.         Args:             i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1260`** (1 nodes): `Get parameters for ``crop`` for a random sized crop.         Args:             i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1261`** (1 nodes): `Update center used for teacher output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1262`** (1 nodes): `image_encoder returns the VAE Encoder with pretrained weights.      Usage:     ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1263`** (1 nodes): `decoder returns the diffusion image decoder model with pretrained      weights.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1264`** (1 nodes): `tokenizer returns the tokenizer used for text inputs.      Can be overriden for`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1265`** (1 nodes): `text_encoder returns the text encoder with pretrained weights.      Can be overr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1266`** (1 nodes): `diffusion_model returns the diffusion model with pretrained weights.      Can be`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1267`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1268`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1269`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1270`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_pooler (ROI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1271`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage.         L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1272`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1273`** (1 nodes): `Args:             short_edge_length (list[int]): If ``sample_style=="range"``,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1274`** (1 nodes): `Compute the output size given input size and target short edge length.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1275`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1276`** (1 nodes): `Prepare some proposals to be used to train the ROI heads.         It performs bo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1277`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_features (li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1278`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1279`** (1 nodes): `return features of all patches at the last ViT block         net: the model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1280`** (1 nodes): `calculate squared distance matrix of each point         params:             feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1281`** (1 nodes): `transfer indices of matrix to array         indices: np.array([[i,j],...])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1282`** (1 nodes): `transfer indices of array to matrix         indices: np.array([i,....])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1283`** (1 nodes): `visualize each component with different color         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1284`** (1 nodes): `transfer indices of matrix to array         indices: np.array([[i,j],...])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1285`** (1 nodes): `transfer indices of array to matrix         indices: np.array([i,....])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1286`** (1 nodes): `return features of all patches at the last ViT block         net: the model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1287`** (1 nodes): `calculate squared distance matrix of each point         params:             feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1288`** (1 nodes): `transfer indices of matrix to array         indices: np.array([[i,j],...])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1289`** (1 nodes): `transfer indices of array to matrix         indices: np.array([i,....])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1290`** (1 nodes): `visualize each component with different color         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1291`** (1 nodes): `preprocess the cv2 image         mode: fixed -> 224 * 224; flexible -> ratio doe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1292`** (1 nodes): `load pretrained UnionSeg model         path: the path of the model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1293`** (1 nodes): `calculate squared distance matrix of each point         params:             feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1294`** (1 nodes): `transfer indices of array to matrix         indices: np.array([i,....])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1295`** (1 nodes): `visualize each component with different color         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1296`** (1 nodes): `check if previous errors happen again`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1297`** (1 nodes): `Find the bounding box of the largest object in a binary image.         :param bi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1298`** (1 nodes): `preprocess the cv2 image         mode: fixed -> 224 * 224; flexible -> ratio doe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1299`** (1 nodes): `load pretrained UnionSeg model         path: the path of the model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1300`** (1 nodes): `transfer indices of array to matrix         indices: np.array([i,....])`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1301`** (1 nodes): `visualize each component with different color         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1302`** (1 nodes): `check if previous errors happen again`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1303`** (1 nodes): `Find the bounding box of the largest object in a binary image.         :param bi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1304`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1305`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1306`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1307`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1308`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1309`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1310`** (1 nodes): `NOTE: this interface is experimental.         Args:             in_channels: cha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1311`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1312`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1313`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1314`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1315`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1316`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1317`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1318`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1319`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1320`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1321`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1322`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1323`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1324`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1325`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1326`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1327`** (1 nodes): `NOTE: this interface is experimental.         Args:             in_channels: cha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1328`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1329`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1330`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1331`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1332`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1333`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1334`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1335`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1336`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1337`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1338`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1339`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1340`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1341`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1342`** (1 nodes): `Returns:             torch.optim.Optimizer:          It now calls :func:`detectr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1343`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1344`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1345`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1346`** (1 nodes): `Returns:             DatasetEvaluator or None          It is not implemented by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1347`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1348`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1349`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1350`** (1 nodes): `Returns:             DatasetEvaluator or None          It is not implemented by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1351`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1352`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1353`** (1 nodes): `merge_jsons.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1354`** (1 nodes): `1_download_pseudo_labels.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1355`** (1 nodes): `0_download_ckpts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1356`** (1 nodes): `eval_cocoapi.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1357`** (1 nodes): `Computation of the ARI clustering metric.      NOTE: This implementation does no`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1358`** (1 nodes): `See `Ari` docstring for allowed keyword arguments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1359`** (1 nodes): `Returns the transformed tensor.      Args:       tensor: Any of a set of differe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1360`** (1 nodes): `Returns the transformed tensor.      Args:       tensor: Any of a set of differe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1361`** (1 nodes): `Slot Attention module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1362`** (1 nodes): `Computes inverted dot-product attention.      Args:       query: Queries with sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1363`** (1 nodes): `Computes multi-head dot-product attention given query, key, and value.      Args`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1364`** (1 nodes): `Apply the ResNet to the inputs `x`.      Args:       x: Inputs.       train: Whe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1365`** (1 nodes): `Computes inverted dot-product attention with key per query.      Args:       que`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1366`** (1 nodes): `Slot Attention with explicit slot statistics module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1367`** (1 nodes): `Slot Attention with explicit slot statistics module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1368`** (1 nodes): `Slot Attention translation equiv. module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1369`** (1 nodes): `Slot Attention translation and scale equiv. module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1370`** (1 nodes): `Slot Attention translation and scale equiv. module forward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1371`** (1 nodes): `Performs a forward pass on a video.      Args:       video: Video of shape `[bat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1372`** (1 nodes): `Args:             values: tensor of shape (batch, n_true_classes, n_pred_classes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1373`** (1 nodes): `Compute auxilliary outputs only needed for metrics and visualisations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1374`** (1 nodes): `Try to infer same padding for convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1375`** (1 nodes): `Try to infer same padding for transposed convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1376`** (1 nodes): `Iterates dataset, then adds dummy samples until reaching the specified number of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1377`** (1 nodes): `Construct padding for property.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1378`** (1 nodes): `Create pipeline object serving same function as wds.WebDataset.          We do t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1379`** (1 nodes): `Keys of properties to keep in dataset after filtering.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1380`** (1 nodes): `Number of samples after pipeline is applied, given original number of samples in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1381`** (1 nodes): `Apply pipeline to dataset.          Input dataset contains dicts of samples afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1382`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1383`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1384`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1385`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1386`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_pooler (ROI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1387`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage.         L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1388`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1389`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1390`** (1 nodes): `Prepare some proposals to be used to train the ROI heads.         It performs bo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1391`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_features (li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1392`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1393`** (1 nodes): `Args:             short_edge_length (list[int]): If ``sample_style=="range"``,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1394`** (1 nodes): `Compute the output size given input size and target short edge length.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1395`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1396`** (1 nodes): `Returns:             torch.optim.Optimizer:          It now calls :func:`detectr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1397`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1398`** (1 nodes): `Returns:             torch.optim.Optimizer:          It now calls :func:`detectr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1399`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1400`** (1 nodes): `Returns:             DatasetEvaluator or None          It is not implemented by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1401`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1402`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1403`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1404`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1405`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1406`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1407`** (1 nodes): `Yield successive n-sized chunks from lst.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1408`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1409`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1410`** (1 nodes): `scannet200_splits.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1411`** (1 nodes): `Build the 3x3 camera matrix K using the given intrinsics.          Equation 6.10`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1412`** (1 nodes): `Convert extrinsics matrix to separate rotation matrix R and translation vector T`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1413`** (1 nodes): `input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1414`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1415`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1416`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1417`** (1 nodes): `input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1418`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1419`** (1 nodes): `input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1420`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1421`** (1 nodes): `input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1422`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1423`** (1 nodes): `input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1424`** (1 nodes): `input: grad_output: (L, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1425`** (1 nodes): `input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1426`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1427`** (1 nodes): `input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1428`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1429`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1430`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1431`** (1 nodes): `input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hd`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1432`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1433`** (1 nodes): `input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1434`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1435`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1436`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1437`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1438`** (1 nodes): `input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1439`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1440`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1441`** (1 nodes): `input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1442`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1443`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1444`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1445`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1446`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1447`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1448`** (1 nodes): `input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1449`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1450`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1451`** (1 nodes): `input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1452`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1453`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1454`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1455`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1456`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1457`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1458`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1459`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1460`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1461`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1462`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1463`** (1 nodes): `r"""         Uses iterative furthest point sampling to select a set of npoint fe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1464`** (1 nodes): `r"""          Parameters         ----------         features : torch.Tensor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1465`** (1 nodes): `r"""             Find the three nearest neighbors of unknown in known         Pa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1466`** (1 nodes): `r"""             Performs weight linear interpolation on 3 features         Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1467`** (1 nodes): `r"""         Parameters         ----------         grad_out : torch.Tensor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1468`** (1 nodes): `r"""          Parameters         ----------         features : torch.Tensor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1469`** (1 nodes): `r"""          Parameters         ----------         grad_out : torch.Tensor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1470`** (1 nodes): `r"""          Parameters         ----------         radius : float             r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1471`** (1 nodes): `:param model_type: a string specifying which model to load. [dino_vits8 | dino_v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1472`** (1 nodes): `Creates a method for position encoding interpolation.         :param patch_size:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1473`** (1 nodes): `change resolution of model output by changing the stride of the patch extraction`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1474`** (1 nodes): `r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns P`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1475`** (1 nodes): `r""" Apply mask to the given image.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1476`** (1 nodes): `dataset_sets.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1477`** (1 nodes): `scannet_constants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1478`** (1 nodes): `Convert Stanford3DDataset to PLY format that is compatible with         Synthia`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1479`** (1 nodes): `Args:             io: (str or binary file-like object): input file to load data`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1480`** (1 nodes): `Convert DensePose predictor outputs to BitMasks using some registered         co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1481`** (1 nodes): `Convert DensePose predictor outputs to DensePoseResult using some registered`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1482`** (1 nodes): `Convert DensePose predictor outputs to DensePoseResult with confidences`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1483`** (1 nodes): `Performs an horizontal flip on DensePose predictor outputs.         Does recursi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1484`** (1 nodes): `Perform recursive lookup for the given type         to find registered converter`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1485`** (1 nodes): `Convert an instance to the destination type using some registered         conver`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1486`** (1 nodes): `Filters proposals with targets to keep only the ones relevant for         DenseP`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1487`** (1 nodes): `Accumulate instances data for one image          Args:             instances_one`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1488`** (1 nodes): `Pack data into tensors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1489`** (1 nodes): `Reset embeddings to random values`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1490`** (1 nodes): `Load data from a file          Args:             fpath (str): file path to load`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1491`** (1 nodes): `Load data from a file          Args:             fpath (str): file path to load`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1492`** (1 nodes): `Args:             cfg (CfgNode):             model (nn.Module):             eval`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1493`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1494`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1495`** (1 nodes): `Build an optimizer from config.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1496`** (1 nodes): `NOTE: this interface is experimental.          Args:             augmentations:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1497`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1498`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1499`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1500`** (1 nodes): `NOTE: this interface is experimental.          Args:             is_train: wheth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1501`** (1 nodes): `Args:             anchors (list[list[Boxes]]): a list of N=#image elements. Each`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1502`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1503`** (1 nodes): `Args:             conv_dim: the output dimension of the conv layers`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1504`** (1 nodes): `get_panoptic_anns_supercategory.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1505`** (1 nodes): `Compute gradients for ROIAlignRotated with multiple bounding boxes on the GPU,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1506`** (1 nodes): `root_cfg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1507`** (1 nodes): `bad_import2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1508`** (1 nodes): `dir1_a.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1509`** (1 nodes): `load_rel.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1510`** (1 nodes): `bad_import.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1511`** (1 nodes): `dir1_b.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1512`** (1 nodes): `mmdet_mask_rcnn_R_50_FPN_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1513`** (1 nodes): `keypoint_rcnn_R_50_FPN_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1514`** (1 nodes): `fcos_R_50_FPN_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1515`** (1 nodes): `retinanet_R_50_FPN_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1516`** (1 nodes): `mask_rcnn_c4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1517`** (1 nodes): `panoptic_fpn_R_50_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1518`** (1 nodes): `mask_rcnn_regnetx_4gf_dds_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1519`** (1 nodes): `mask_rcnn_regnety_4gf_dds_fpn_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1520`** (1 nodes): `mask_rcnn_R_50_FPN_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1521`** (1 nodes): `mask_rcnn_R_50_C4_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1522`** (1 nodes): `Calculate proper im2col step size, which should be divisible by input_size and n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1523`** (1 nodes): `Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1524`** (1 nodes): `Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1525`** (1 nodes): `Returns:             tuple: height, width`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1526`** (1 nodes): `Args:             instance_lists (list[Instances])          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1527`** (1 nodes): `Args:             box: can be a k-tuple, k-list or an Nxk array/tensor, where k`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1528`** (1 nodes): `Concatenates a list of Boxes into a single Boxes          Arguments:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1529`** (1 nodes): `Yield a box as a Tensor of shape (4,) at a time.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1530`** (1 nodes): `Concatenates a list of Keypoints into a single Keypoints          Arguments:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1531`** (1 nodes): `Returns:             BitMasks: Create a new :class:`BitMasks` by indexing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1532`** (1 nodes): `Args:             polygon_masks (list[list[ndarray]] or PolygonMasks)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1533`** (1 nodes): `Args:             roi_masks:             height, width (int):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1534`** (1 nodes): `Concatenates a list of BitMasks into a single BitMasks          Arguments:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1535`** (1 nodes): `Concatenates a list of PolygonMasks into a single PolygonMasks          Argument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1536`** (1 nodes): `Args: see documentation of :func:`paste_masks_in_image`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1537`** (1 nodes): `Args:             tensors: a tuple or list of `torch.Tensor`, each of shape (Hi,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1538`** (1 nodes): `Concatenates a list of RotatedBoxes into a single RotatedBoxes          Argument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1539`** (1 nodes): `Yield a box as a Tensor of shape (5,) at a time.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1540`** (1 nodes): `Similar to :meth:`load()`, but load path relative to the caller's         source`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1541`** (1 nodes): `Load a config file.          Args:             filename: absolute path or relati`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1542`** (1 nodes): `Save a config object to a yaml file.         Note that when the config dictionar`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1543`** (1 nodes): `In-place override contents of cfg.          Args:             cfg: an omegaconf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1544`** (1 nodes): `Try to convert a config object into Python-like psuedo code.          Note that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1545`** (1 nodes): `Returns:             int: The current iteration number. When used together with`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1546`** (1 nodes): `Yields:             A context within which all the events added to this storage`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1547`** (1 nodes): `Args:             config_path: relative config filename`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1548`** (1 nodes): `Open a context where some heads in `model.roi_heads` are temporarily turned off.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1549`** (1 nodes): `This interface is experimental.          Args:             sizes (list[list[floa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1550`** (1 nodes): `Alias of `num_anchors`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1551`** (1 nodes): `Returns:             list[int]: Each int is the number of anchors at every pixel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1552`** (1 nodes): `This interface is experimental.          Args:             sizes (list[list[floa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1553`** (1 nodes): `Alias of `num_anchors`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1554`** (1 nodes): `Returns:             list[int]: Each int is the number of anchors at every pixel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1555`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1556`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1557`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1558`** (1 nodes): `NOTE: this interface is experimental.          Args:             sem_seg_head: a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1559`** (1 nodes): `Match ground-truth boxes to a set of multi-level anchors.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1560`** (1 nodes): `Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1561`** (1 nodes): `Args:             anchors (list[Boxes]): A list of #feature level Boxes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1562`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1563`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1564`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape: sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1565`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_channels (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1566`** (1 nodes): `Args:             anchors (list[Boxes]): anchors for each feature map.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1567`** (1 nodes): `Return the losses from a set of RPN predictions and their associated ground-trut`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1568`** (1 nodes): `Args:             anchors (list[RotatedBoxes]): anchors for each feature map.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1569`** (1 nodes): `NOTE: this interface is experimental.          Args:             loss_weight (fl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1570`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1571`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1572`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1573`** (1 nodes): `Returns:             ShapeSpec: the output feature shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1574`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_keypoints (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1575`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1576`** (1 nodes): `NOTE: this interface is experimental.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1577`** (1 nodes): `Prepare some proposals to be used to train the RROI heads.         It performs b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1578`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_pooler (ROI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1579`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage.         L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1580`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1581`** (1 nodes): `Prepare some proposals to be used to train the ROI heads.         It performs bo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1582`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1583`** (1 nodes): `This property is a generalization of size_divisibility. Some backbones and train`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1584`** (1 nodes): `Create a list of blocks of the same type that forms one ResNet stage.          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1585`** (1 nodes): `Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 15`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1586`** (1 nodes): `Args:         video_height: height the video frame         video_width: width of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1587`** (1 nodes): `Old style initialization using CfgNode          Args:             cfg: D2 CfgNod`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1588`** (1 nodes): `Args:         video_height: height the video frame         video_width: width of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1589`** (1 nodes): `Old style initialization using CfgNode          Args:             cfg: D2 CfgNod`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1590`** (1 nodes): `Args:         video_height: height the video frame         video_width: width of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1591`** (1 nodes): `Args:         video_height: height the video frame         video_width: width of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1592`** (1 nodes): `Old style initialization using CfgNode          Args:             cfg: D2 CfgNod`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1593`** (1 nodes): `Convert InstancesList to List[Instances]. The input `instances_list` can`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1594`** (1 nodes): `Patching several inference functions inside ROIHeads and its subclasses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1595`** (1 nodes): `Creates a function that converts outputs of the caffe2 model to         detectro`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1596`** (1 nodes): `caffe2.core.Net: the underlying caffe2 predict net`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1597`** (1 nodes): `caffe2.core.Net: the underlying caffe2 init net`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1598`** (1 nodes): `Args:             dir (str): a directory used to save Caffe2Model with`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1599`** (1 nodes): `Args:             short_edge_length (list[int]): If ``sample_style=="range"``,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1600`** (1 nodes): `Compute the output size given input size and target short edge length.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1601`** (1 nodes): `Compute (fractional) per-image repeat factors based on category frequency.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1602`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1603`** (1 nodes): `Returns:             torch.optim.Optimizer:          It now calls :func:`detectr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1604`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1605`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1606`** (1 nodes): `create_aug_data.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1607`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1608`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1609`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1610`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1611`** (1 nodes): `Uses iterative furthest point sampling to select a set of npoint features that h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1612`** (1 nodes): `:param ctx:         :param features: (B, C, N)         :param idx: (B, npoint) i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1613`** (1 nodes): `Find the three nearest neighbors of unknown in known         :param ctx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1614`** (1 nodes): `Find the three nearest neighbors of unknown in known         :param ctx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1615`** (1 nodes): `Performs weight linear interpolation on 3 features         :param ctx:         :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1616`** (1 nodes): `:param ctx:         :param grad_out: (B, C, N) tensor with gradients of outputs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1617`** (1 nodes): `:param ctx:         :param features: (B, C, N) tensor of features to group`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1618`** (1 nodes): `:param ctx:         :param grad_out: (B, C, npoint, nsample) tensor of the gradi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1619`** (1 nodes): `:param ctx:         :param radius: float, radius of the balls         :param nsa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1620`** (1 nodes): `Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1621`** (1 nodes): `Set beta (or alpha as makes sense for given optimizer).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1622`** (1 nodes): `Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1623`** (1 nodes): `To support a custom dataset, implement this function to receive the predicted re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1624`** (1 nodes): `Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1625`** (1 nodes): `Args:             pts_rect:             img_shape:             calib:          R`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1626`** (1 nodes): `Args:             batch_dict:                 frame_id:             pred_dicts:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1627`** (1 nodes): `Args:             batch_dict:                 frame_id:             pred_dicts:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1628`** (1 nodes): `Args:             batch_dict:                 frame_id:             pred_dicts:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1629`** (1 nodes): `Args:             pts_rect:             img_shape:             cam_intrinsic:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1630`** (1 nodes): `Args:             batch_dict:                 frame_id:             pred_dicts:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1631`** (1 nodes): `Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1632`** (1 nodes): `Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1633`** (1 nodes): `PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1634`** (1 nodes): `Args:             x: x.features (N, C1)             out_channels: C2          Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1635`** (1 nodes): `Args:             cls_scores: (N)             iou_scores: (N)             num_po`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1636`** (1 nodes): `Args:             batch_dict:                 batch_size:                 batch_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1637`** (1 nodes): `Args:             rois: (N, 7)             roi_labels: (N)             gt_boxes:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1638`** (1 nodes): `Args:             ctx:             radius: float, radius of the balls`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1639`** (1 nodes): `:param ctx:         :param features: (B, C, N)         :param idx: (B, npoint) i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1640`** (1 nodes): `Find the three nearest neighbors of unknown in known         :param ctx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1641`** (1 nodes): `Performs weight linear interpolation on 3 features         :param ctx:         :`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1642`** (1 nodes): `:param ctx:         :param grad_out: (B, C, N) tensor with gradients of outputs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1643`** (1 nodes): `:param ctx:         :param features: (B, C, N) tensor of features to group`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1644`** (1 nodes): `:param ctx:         :param radius: float, radius of the balls         :param nsa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1645`** (1 nodes): `Args:             ctx:             max_range: int, max range of voxels to be gro`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1646`** (1 nodes): `Args:             ctx:             features: (N1 + N2 ..., C) tensor of features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1647`** (1 nodes): `Args:             ctx:             xyz: (B, N, 3) where N > npoint             n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1648`** (1 nodes): `Args:             ctx:             xyz: (N1 + N2 + ..., 3) where N > npoint`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1649`** (1 nodes): `Args:             ctx:             unknown: (N1 + N2..., 3)             unknown_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1650`** (1 nodes): `Args:             ctx:             grad_out: (N1 + N2 ..., C)          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1651`** (1 nodes): `Args:             ctx:             points: (B, N, 3)             point_features:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1652`** (1 nodes): `Args:             ctx:             rois: (N, 7) [x, y, z, dx, dy, dz, heading] (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1653`** (1 nodes): `:param grad_out: (N, out_x, out_y, out_z, C)         :return:             grad_i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1654`** (1 nodes): `sweep_train.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1655`** (1 nodes): `Logs the global norm of all parameters and of their gradients.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1656`** (1 nodes): `Logs the global norm of parameters and their gradients, by group.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1657`** (1 nodes): `paths.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1658`** (1 nodes): `List of scalar model parameters that should be logged.          They must be in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1659`** (1 nodes): `Parameter groups whose norm and gradient norm will be logged separately to tenso`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1660`** (1 nodes): `Number of slots used for representation.          By default, it is equal to the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1661`** (1 nodes): `Representation size per slot.          This does not apply to models that are no`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1662`** (1 nodes): `Stick breaking process to produce masks         :param: masks (B, K, 1, H, W). I`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1663`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1664`** (1 nodes): `Parses filter string into the corresponding parsing tree.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1665`** (1 nodes): `fcos3d_dummy-resnet_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1666`** (1 nodes): `stat.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1667`** (1 nodes): `list[float]: Size of a single voxel.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1668`** (1 nodes): `int: Maximum number of points per voxel.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1669`** (1 nodes): `list[float]: Range of point cloud.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1670`** (1 nodes): `np.ndarray: The size of grids.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1671`** (1 nodes): `torch.Tensor: Coordinates of each point in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1672`** (1 nodes): `Set the coordinates of each point.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1673`** (1 nodes): `torch.Tensor:             A vector with height of each point in shape (N, 1), or`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1674`** (1 nodes): `Set the height of each point.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1675`** (1 nodes): `torch.Tensor:             A vector with color of each point in shape (N, 3), or`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1676`** (1 nodes): `Set the color of each point.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1677`** (1 nodes): `torch.Shape: Shape of points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1678`** (1 nodes): `Flip the points along given BEV direction.          Args:             bev_direct`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1679`** (1 nodes): `torch.Tensor: BEV of the points in shape (N, 2).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1680`** (1 nodes): `Convert self to ``dst`` mode.          Args:             dst (:obj:`CoordMode`):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1681`** (1 nodes): `Concatenate a list of Points into a single Points.          Args:             po`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1682`** (1 nodes): `str: The device of the points are on.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1683`** (1 nodes): `torch.Tensor: BEV of the points in shape (N, 2).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1684`** (1 nodes): `list[int]: Total number of base anchors in a feature grid.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1685`** (1 nodes): `int: Number of feature levels that the generator is applied to.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1686`** (1 nodes): `Get box regression transformation deltas (dx, dy, dz, dx_size,         dy_size,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1687`** (1 nodes): `Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,         dz_size, dr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1688`** (1 nodes): `Decode yaw angle and change it from local to global.i.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1689`** (1 nodes): `Convert boxes from `src` mode to `dst` mode.          Args:             box (tup`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1690`** (1 nodes): `torch.Tensor: A vector with height of each box in shape (N, ).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1691`** (1 nodes): `torch.Tensor:             A vector with the top height of each box in shape (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1692`** (1 nodes): `torch.Tensor:             A vector with bottom's height of each box in shape (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1693`** (1 nodes): `torch.Tensor:             A vector with local yaw of each box in shape (N, ).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1694`** (1 nodes): `torch.Tensor: A tensor with center of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1695`** (1 nodes): `torch.Tensor: Coordinates of corners of all the boxes in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1696`** (1 nodes): `torch.Tensor: 2D BEV box of each box with rotation             in XYWHR format,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1697`** (1 nodes): `Calculate height overlaps of two boxes.          This function calculates the he`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1698`** (1 nodes): `torch.Tensor: A tensor with center of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1699`** (1 nodes): `torch.Tensor: Coordinates of corners of all the boxes         in shape (N, 8, 3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1700`** (1 nodes): `torch.Tensor: A tensor with center of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1701`** (1 nodes): `torch.Tensor: Coordinates of corners of all the boxes         in shape (N, 8, 3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1702`** (1 nodes): `Convert boxes or points from `src` mode to `dst` mode.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1703`** (1 nodes): `Convert boxes from `src` mode to `dst` mode.          Args:             box (tup`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1704`** (1 nodes): `Convert points from `src` mode to `dst` mode.          Args:             point (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1705`** (1 nodes): `torch.Tensor: A vector with volume of each box.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1706`** (1 nodes): `torch.Tensor: Size dimensions of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1707`** (1 nodes): `torch.Tensor: A vector with yaw of each box in shape (N, ).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1708`** (1 nodes): `torch.Tensor: A vector with height of each box in shape (N, ).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1709`** (1 nodes): `torch.Tensor:             A vector with the top height of each box in shape (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1710`** (1 nodes): `torch.Tensor:             A vector with bottom's height of each box in shape (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1711`** (1 nodes): `Calculate the center of all the boxes.          Note:             In MMDetection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1712`** (1 nodes): `torch.Tensor: A tensor with center of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1713`** (1 nodes): `torch.Tensor: A tensor with center of each box in shape (N, 3).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1714`** (1 nodes): `torch.Tensor:             a tensor with 8 corners of each box in shape (N, 8, 3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1715`** (1 nodes): `torch.Tensor: 2D BEV box of each box with rotation             in XYWHR format,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1716`** (1 nodes): `torch.Tensor: A tensor of 2D BEV box of each box             without rotation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1717`** (1 nodes): `Rotate boxes with points (optional) with the given angle or rotation         mat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1718`** (1 nodes): `Flip the boxes in BEV along given BEV direction.          Args:             bev_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1719`** (1 nodes): `Convert self to ``dst`` mode.          Args:             dst (:obj:`Box3DMode`):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1720`** (1 nodes): `Concatenate a list of Boxes into a single Boxes.          Args:             boxe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1721`** (1 nodes): `str: The device of the boxes are on.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1722`** (1 nodes): `Calculate height overlaps of two boxes.          Note:             This function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1723`** (1 nodes): `Calculate 3D overlaps of two boxes.          Note:             This function cal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1724`** (1 nodes): `Repeat x `num` times to form a list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1725`** (1 nodes): `Get class names of current dataset.          Args:             classes (Sequence`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1726`** (1 nodes): `Get axis_align_matrix from info. If not exist, return identity mat.          Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1727`** (1 nodes): `Filter ground truths by difficulties.          Args:             db_infos (dict)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1728`** (1 nodes): `Filter ground truths by number of points in the bbox.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1729`** (1 nodes): `Remove the points in the sampled bounding boxes.          Args:             poin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1730`** (1 nodes): `Compute loss.          Args:             bbox_preds (dict): Predictions from for`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1731`** (1 nodes): `Convert the rotation difference to difference in sine function.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1732`** (1 nodes): `Calculate losses.          Args:             cls_scores (list[torch.Tensor]): Mu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1733`** (1 nodes): `Compute loss of the head.          Args:             cls_scores (list[Tensor]):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1734`** (1 nodes): `Transform network output for a batch into bbox predictions.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1735`** (1 nodes): `Compute regression, classification and centerss targets for points         in mu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1736`** (1 nodes): `Construct Conv-Norm-Act block.          Args:             in_channels (int): Num`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1737`** (1 nodes): `Construct DeConv-Norm-Act-Conv-Norm-Act block.          Args:             in_cha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1738`** (1 nodes): `Transform box to the axis-aligned or rotated iou loss format.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1739`** (1 nodes): `Transform predicted bbox parameters to bbox.          Args:             points (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1740`** (1 nodes): `Calculate distances from point to box faces.          Args:             points (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1741`** (1 nodes): `Compute point centerness w.r.t containing box.          Args:             face_d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1742`** (1 nodes): `Compute targets for final locations for a single scene.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1743`** (1 nodes): `Loss function for CenterHead.          Args:             gt_bboxes_3d (list[:obj`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1744`** (1 nodes): `Upsample valid mask predictions.          Args:             valid_pred (Tensor):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1745`** (1 nodes): `Transform predicted bbox parameters to bbox.          Args:             points (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1746`** (1 nodes): `Calculate distances from point to box faces.          Args:             points (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1747`** (1 nodes): `Compute point centerness w.r.t containing box.          Args:             face_d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1748`** (1 nodes): `Compute targets for final locations for a single scene.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1749`** (1 nodes): `Calculate loss of FreeAnchor head.          Args:             cls_scores (list[t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1750`** (1 nodes): `Compute loss.          Args:             bbox_preds (dict): Predictions from for`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1751`** (1 nodes): `Compute loss.          Args:             bbox_preds (dict): Predictions from for`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1752`** (1 nodes): `Calculate losses.          Args:             cls_scores (list[torch.Tensor]): Mu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1753`** (1 nodes): `Compute loss.          Args:             bbox_preds (dict): Predictions from for`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1754`** (1 nodes): `Compute losses of the head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1755`** (1 nodes): `Transform network output for a batch into bbox predictions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1756`** (1 nodes): `Compute loss of the head.          Args:             cls_scores (list[Tensor]):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1757`** (1 nodes): `Transform network output for a batch into bbox predictions.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1758`** (1 nodes): `Convert the rotation difference to difference in sine function.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1759`** (1 nodes): `Encode direction to 0 ~ num_bins-1.          Args:             reg_targets (torc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1760`** (1 nodes): `Compute loss of the head.          Args:             cls_scores (list[Tensor]):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1761`** (1 nodes): `Transform network output for a batch into bbox predictions.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1762`** (1 nodes): `Args:             points (torch.Tensor): points in 2D images, [N, 3],`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1763`** (1 nodes): `Forward function.          Args:             x (torch.Tensor): 4D Tensor in (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1764`** (1 nodes): `Forward function.          All inputs should be sorted by the rank of voxels.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1765`** (1 nodes): `Backward propagation function.          Args:             gradx (torch.tensor):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1766`** (1 nodes): `Make a layer from several residual blocks.          Args:             stride (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1767`** (1 nodes): `Make a convolutional block.          Args:             in_channels (int): Number`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1768`** (1 nodes): `Make upsampling convolutional block.          Args:             in_channels (int`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1769`** (1 nodes): `Forward function.          Args:             features (torch.Tensor): Point feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1770`** (1 nodes): `Forward function.          Args:             features (torch.Tensor): Point feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1771`** (1 nodes): `Forward function.          Args:             inputs (torch.Tensor): Pillar/Voxel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1772`** (1 nodes): `Forward function.          Args:             features (torch.Tensor): Point feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1773`** (1 nodes): `Forward function.          Args:             features (torch.Tensor): Point feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1774`** (1 nodes): `Forward functions.          Args:             features (torch.Tensor): Features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1775`** (1 nodes): `Forward functions.          Args:             features (torch.Tensor): Features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1776`** (1 nodes): `Forward pass.          Args:             points (torch.Tensor): point coordinate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1777`** (1 nodes): `Split coordinates and features of input points.          Args:             point`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1778`** (1 nodes): `Forward pass.          Args:             points (torch.Tensor): point coordinate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1779`** (1 nodes): `Forward pass.          Args:             points (torch.Tensor): point coordinate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1780`** (1 nodes): `Forward pass.          Args:             points (torch.Tensor): point coordinate`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1781`** (1 nodes): `Apply dynamic voxelization to points.          Args:             points (list[to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1782`** (1 nodes): `bool: Whether the detector has a 2D image box head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1783`** (1 nodes): `bool: Whether the detector has a 2D image box head (not roi).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1784`** (1 nodes): `bool: Whether the detector has a 2D image backbone.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1785`** (1 nodes): `bool: Whether the detector has a neck in image branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1786`** (1 nodes): `bool: Whether the detector has a 2D RPN in image detector branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1787`** (1 nodes): `bool: Whether the detector has a RoI Head in image branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1788`** (1 nodes): `bool: Whether the detector has a 3D box head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1789`** (1 nodes): `bool: Whether the detector has a 3D backbone.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1790`** (1 nodes): `bool: Whether the detector has a neck in 3D detector branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1791`** (1 nodes): `Extract bounding boxes from 2d detector.          Args:             img (torch.T`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1792`** (1 nodes): `Apply dynamic voxelization to points.          Args:             points (list[to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1793`** (1 nodes): `Apply hard voxelization to points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1794`** (1 nodes): `bool: Whether the detector has a shared head in image branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1795`** (1 nodes): `bool: Whether the detector has a 3D box head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1796`** (1 nodes): `bool: Whether the detector has a 2D image box head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1797`** (1 nodes): `bool: Whether the detector has a 2D image backbone.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1798`** (1 nodes): `bool: Whether the detector has a 3D backbone.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1799`** (1 nodes): `bool: Whether the detector has a fusion layer.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1800`** (1 nodes): `bool: Whether the detector has a neck in image branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1801`** (1 nodes): `bool: Whether the detector has a neck in 3D detector branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1802`** (1 nodes): `bool: Whether the detector has a 2D RPN in image detector branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1803`** (1 nodes): `bool: Whether the detector has a RoI Head in image branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1804`** (1 nodes): `bool: Whether the detector has a voxel encoder.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1805`** (1 nodes): `bool: Whether the detector has a middle encoder.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1806`** (1 nodes): `Apply dynamic voxelization to points.          Args:             points (list[to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1807`** (1 nodes): `bool: Whether the head predicts velocity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1808`** (1 nodes): `Apply hard voxelization to points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1809`** (1 nodes): `Apply hard voxelization to points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1810`** (1 nodes): `bool: whether the head has semantic branch`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1811`** (1 nodes): `Assign and sample proposals for training.          Args:             proposal_li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1812`** (1 nodes): `bool: whether the RoIHead has box head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1813`** (1 nodes): `bool: whether the RoIHead has mask head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1814`** (1 nodes): `Initialize the box head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1815`** (1 nodes): `Initialize maek head.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1816`** (1 nodes): `Initialize assigner and sampler.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1817`** (1 nodes): `Forward function during training.          Args:             x (dict): Contains`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1818`** (1 nodes): `Generating model input.          Generate input by subtracting patch center and`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1819`** (1 nodes): `bool: whether the segmentor has regularization loss for weight`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1820`** (1 nodes): `Calls either forward_train or forward_test depending on whether         return_l`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1821`** (1 nodes): `Forward of SparseEncoder.          Args:             voxel_features (torch.Tenso`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1822`** (1 nodes): `Forward of SparseEncoder.          Args:             voxel_features (torch.Tenso`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1823`** (1 nodes): `Forward of SparseUNet.          Args:             voxel_features (torch.float32)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1824`** (1 nodes): `reduce channel for element-wise addition.          Args:             x (:obj:`Sp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1825`** (1 nodes): `Forward function to scatter features.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1826`** (1 nodes): `Placeholder of forward function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1827`** (1 nodes): `Compute semantic segmentation loss.          Args:             seg_logit (torch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1828`** (1 nodes): `Args:             Input (tensor): Feature has shape (N, C, H, W).          Retur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1829`** (1 nodes): `forward.          Args:             points (Tensor): (B, N, C) tensor of the inp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1830`** (1 nodes): `forward.          Args:             points (List[Tensor]): tensor of the feature`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1831`** (1 nodes): `forward.          Args:             target (Tensor): (B, n, 3) tensor of the xyz`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1832`** (1 nodes): `hv_pointpillars_secfpn_4x8_80e_pcdet_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1833`** (1 nodes): `hv_second_secfpn_4x8_80e_pcdet_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1834`** (1 nodes): `hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1835`** (1 nodes): `hv_PartA2_secfpn_4x8_cyclic_80e_pcdet_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1836`** (1 nodes): `default_runtime.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1837`** (1 nodes): `kitti-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1838`** (1 nodes): `range100_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1839`** (1 nodes): `waymoD5-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1840`** (1 nodes): `s3dis-3d-5class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1841`** (1 nodes): `nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1842`** (1 nodes): `nus-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1843`** (1 nodes): `scannet-3d-18class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1844`** (1 nodes): `scannet_seg-3d-20class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1845`** (1 nodes): `kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1846`** (1 nodes): `lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1847`** (1 nodes): `sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1848`** (1 nodes): `waymoD5-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1849`** (1 nodes): `kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1850`** (1 nodes): `nuim_instance.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1851`** (1 nodes): `s3dis_seg-3d-13class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1852`** (1 nodes): `centerpoint_01voxel_second_secfpn_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1853`** (1 nodes): `paconv_cuda_ssg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1854`** (1 nodes): `imvotenet_image.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1855`** (1 nodes): `hv_second_secfpn_kitti.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1856`** (1 nodes): `3dssd.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1857`** (1 nodes): `fcaf3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1858`** (1 nodes): `hv_pointpillars_fpn_lyft.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1859`** (1 nodes): `hv_pointpillars_secfpn_waymo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1860`** (1 nodes): `hv_second_secfpn_waymo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1861`** (1 nodes): `hv_pointpillars_fpn_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1862`** (1 nodes): `fcos3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1863`** (1 nodes): `groupfree3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1864`** (1 nodes): `paconv_ssg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1865`** (1 nodes): `centerpoint_02pillar_second_secfpn_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1866`** (1 nodes): `pointnet2_msg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1867`** (1 nodes): `hv_pointpillars_fpn_range100_lyft.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1868`** (1 nodes): `pointnet2_ssg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1869`** (1 nodes): `hv_pointpillars_secfpn_kitti.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1870`** (1 nodes): `cascade_mask_rcnn_r50_fpn.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1871`** (1 nodes): `smoke.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1872`** (1 nodes): `mask_rcnn_r50_fpn.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1873`** (1 nodes): `cosine.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1874`** (1 nodes): `seg_cosine_200e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1875`** (1 nodes): `mmdet_schedule_1x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1876`** (1 nodes): `schedule_2x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1877`** (1 nodes): `schedule_3x.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1878`** (1 nodes): `seg_cosine_100e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1879`** (1 nodes): `cyclic_40e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1880`** (1 nodes): `seg_cosine_50e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1881`** (1 nodes): `seg_cosine_150e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1882`** (1 nodes): `cyclic_20e.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1883`** (1 nodes): `smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1884`** (1 nodes): `pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1885`** (1 nodes): `pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1886`** (1 nodes): `pointnet2_ssg_xyz-only_16x2_cosine_200e_scannet_seg-3d-20class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1887`** (1 nodes): `pointnet2_msg_xyz-only_16x2_cosine_250e_scannet_seg-3d-20class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1888`** (1 nodes): `pointnet2_ssg_16x2_cosine_50e_s3dis_seg-3d-13class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1889`** (1 nodes): `pointnet2_msg_16x2_cosine_250e_scannet_seg-3d-20class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1890`** (1 nodes): `mask_rcnn_x101_32x4d_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1891`** (1 nodes): `mask_rcnn_r50_caffe_fpn_coco-3x_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1892`** (1 nodes): `mask_rcnn_r50_fpn_coco-2x_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1893`** (1 nodes): `cascade_mask_rcnn_r101_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1894`** (1 nodes): `htc_without_semantic_r50_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1895`** (1 nodes): `mask_rcnn_r101_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1896`** (1 nodes): `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1897`** (1 nodes): `mask_rcnn_r50_fpn_coco-2x_1x_nus-2d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1898`** (1 nodes): `cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1899`** (1 nodes): `mask_rcnn_r50_caffe_fpn_coco-3x_20e_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1900`** (1 nodes): `cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1901`** (1 nodes): `mask_rcnn_r50_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1902`** (1 nodes): `mask_rcnn_r50_caffe_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1903`** (1 nodes): `cascade_mask_rcnn_r50_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1904`** (1 nodes): `htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1905`** (1 nodes): `htc_r50_fpn_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1906`** (1 nodes): `htc_r50_fpn_coco-20e_1x_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1907`** (1 nodes): `htc_r50_fpn_coco-20e_20e_nuim.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1908`** (1 nodes): `hv_second_secfpn_fp16_6x8_80e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1909`** (1 nodes): `hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1910`** (1 nodes): `hv_second_secfpn_6x8_80e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1911`** (1 nodes): `hv_second_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1912`** (1 nodes): `hv_second_secfpn_6x8_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1913`** (1 nodes): `fcaf3d_8x2_s3dis-3d-5class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1914`** (1 nodes): `fcaf3d_8x2_sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1915`** (1 nodes): `fcaf3d_8x2_scannet-3d-18class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1916`** (1 nodes): `dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1917`** (1 nodes): `centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_tta_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1918`** (1 nodes): `centerpoint_02pillar_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1919`** (1 nodes): `centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1920`** (1 nodes): `centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_novelo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1921`** (1 nodes): `centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1922`** (1 nodes): `centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1923`** (1 nodes): `centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_flip-tta_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1924`** (1 nodes): `centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1925`** (1 nodes): `centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1926`** (1 nodes): `centerpoint_01voxel_second_secfpn_dcn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1927`** (1 nodes): `centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1928`** (1 nodes): `centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1929`** (1 nodes): `centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1930`** (1 nodes): `centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1931`** (1 nodes): `centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1932`** (1 nodes): `centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1933`** (1 nodes): `fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1934`** (1 nodes): `fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1935`** (1 nodes): `hv_ssn_secfpn_sbn-all_2x16_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1936`** (1 nodes): `hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1937`** (1 nodes): `hv_ssn_regnet-400mf_secfpn_sbn-all_1x16_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1938`** (1 nodes): `hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1939`** (1 nodes): `hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1940`** (1 nodes): `hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1941`** (1 nodes): `hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1942`** (1 nodes): `hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1943`** (1 nodes): `hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1944`** (1 nodes): `hv_pointpillars_secfpn_sbn-all_range100_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1945`** (1 nodes): `hv_pointpillars_fpn_sbn-all_range100_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1946`** (1 nodes): `hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1947`** (1 nodes): `hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1948`** (1 nodes): `hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1949`** (1 nodes): `hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1950`** (1 nodes): `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1951`** (1 nodes): `hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1952`** (1 nodes): `hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1953`** (1 nodes): `sassd_6x8_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1954`** (1 nodes): `paconv_ssg_8x8_cosine_150e_s3dis_seg-3d-13class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1955`** (1 nodes): `paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1956`** (1 nodes): `point_rcnn_2x8_kitti-3d-3classes.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1957`** (1 nodes): `3dssd_4x4_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1958`** (1 nodes): `hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1959`** (1 nodes): `hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1960`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area3.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1961`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area6.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1962`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1963`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area5.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1964`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area1.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1965`** (1 nodes): `dgcnn_32x4_cosine_100e_s3dis_seg-3d-13class-area4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1966`** (1 nodes): `dv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1967`** (1 nodes): `dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1968`** (1 nodes): `dv_second_secfpn_6x8_80e_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1969`** (1 nodes): `hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1970`** (1 nodes): `hv_pointpillars_regnet-400mf_fpn_sbn-all_fp16_2x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1971`** (1 nodes): `hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1972`** (1 nodes): `hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1973`** (1 nodes): `hv_pointpillars_regnet-400mf_secfpn_sbn-all_range100_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1974`** (1 nodes): `hv_pointpillars_regnet-400mf_fpn_sbn-all_range100_2x8_2x_lyft-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1975`** (1 nodes): `hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1976`** (1 nodes): `hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1977`** (1 nodes): `hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1978`** (1 nodes): `hv_pointpillars_regnet-400mf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1979`** (1 nodes): `hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1980`** (1 nodes): `hv_pointpillars_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1981`** (1 nodes): `hv_pointpillars_regnet-3.2gf_fpn_sbn-all_free-anchor_strong-aug_4x8_3x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1982`** (1 nodes): `hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1983`** (1 nodes): `h3dnet_3x8_scannet-3d-18class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1984`** (1 nodes): `votenet_16x8_sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1985`** (1 nodes): `votenet_iouloss_8x8_scannet-3d-18class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1986`** (1 nodes): `votenet_8x8_scannet-3d-18class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1987`** (1 nodes): `imvotenet_stage2_16x8_sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1988`** (1 nodes): `imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1989`** (1 nodes): `pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d_finetune.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1990`** (1 nodes): `pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1991`** (1 nodes): `pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1992`** (1 nodes): `pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1993`** (1 nodes): `pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1994`** (1 nodes): `groupfree3d_8x4_scannet-3d-18class-L6-O256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1995`** (1 nodes): `groupfree3d_8x4_scannet-3d-18class-w2x-L12-O256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1996`** (1 nodes): `groupfree3d_8x4_scannet-3d-18class-L12-O256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1997`** (1 nodes): `groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1998`** (1 nodes): `imvoxelnet_4x2_sunrgbd-3d-10class.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 1999`** (1 nodes): `imvoxelnet_4x8_kitti-3d-car.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2000`** (1 nodes): `Return state of constrain to axis mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2001`** (1 nodes): `Set state of constrain to axis mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2002`** (1 nodes): `av2_classes.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2003`** (1 nodes): `Forward function.          Args:             x (torch.Tensor): 4D Tensor in (N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2004`** (1 nodes): `bool: Whether the detector has a neck in 3D detector branch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2005`** (1 nodes): `Apply dynamic voxelization to points.          Args:             points (list[to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2006`** (1 nodes): `Loss function for CenterHead.          Args:             gt_bboxes_3d (list[:obj`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2007`** (1 nodes): `In the forward pass we receive a Tensor containing the input and return`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2008`** (1 nodes): `Returns the folder where the tables are stored for the relevant version.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2009`** (1 nodes): `Returns the folder where the tables are stored for the relevant version.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2010`** (1 nodes): `Perform clipping on polygons that are partially behind the camera.         This`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2011`** (1 nodes): `Convert a polygon or multipolygon list to an image mask ndarray.         :param`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2012`** (1 nodes): `Convert a Shapely LineString back to an image mask ndarray.         :param lines`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2013`** (1 nodes): `Convert patch_box to shapely Polygon coordinates.         :param patch_box: Patc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2014`** (1 nodes): `Check if any lanes are disconnected.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2015`** (1 nodes): `Computes the angle between the last points of the two trajectories.         The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2016`** (1 nodes): `Compute the average of l2 norms of each row in the tensor.         :param tensor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2017`** (1 nodes): `Mainly a smoke test since most of the logic is handled under-the-hood         by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2018`** (1 nodes): `Convert sample token into standard KITTI folder and local filename format.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2019`** (1 nodes): `Parses single line from label file into a dict. Boxes are in camera frame. See K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2020`** (1 nodes): `Transform from nuScenes lidar frame to KITTI reference frame.         :param box`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2021`** (1 nodes): `Projects 3D box into KITTI image FOV.         :param box: 3D box in KITTI refere`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2022`** (1 nodes): `For a token and table, get the filepath to the associated data.         :param t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2023`** (1 nodes): `Returns transforms for the input token.         :param token: KittiDB unique id.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2024`** (1 nodes): `Load up the pointcloud for a sample.         :param token: KittiDB unique id.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2025`** (1 nodes): `Convert box in KITTI image frame to official label string fromat.         :param`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2026`** (1 nodes): `Returns the map mask, optionally dilated.         :param dilation: Dilation in m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2027`** (1 nodes): `Generate transform matrix for this map mask.         :return: <np.array: 4, 4>.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2028`** (1 nodes): `Returns the original binary mask stored in map png file.         :return: <np.in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2029`** (1 nodes): `Returns the number of dimensions.         :return: Number of dimensions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2030`** (1 nodes): `Loads point cloud from disk.         :param file_name: Path of the pointcloud fi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2031`** (1 nodes): `Initialize from serialized dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2032`** (1 nodes): `Returns the number of dimensions.         :return: Number of dimensions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2033`** (1 nodes): `Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2034`** (1 nodes): `Disable all radar filter settings.         Use this method to plot all radar ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2035`** (1 nodes): `Set the defaults for all radar filter settings.         Note that this method af`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2036`** (1 nodes): `Initialize from serialized dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2037`** (1 nodes): `Loads RADAR data from a Point Cloud Data file. See details below.         :param`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2038`** (1 nodes): `Return a rotation matrix.         :return: <np.float: 3, 3>. The box's rotation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2039`** (1 nodes): `Loads the polygon representation of the drivable area for each map.         :par`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2040`** (1 nodes): `Interpolate trajectory with a cubic spline if there are enough points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2041`** (1 nodes): `Initialize from serialized dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2042`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2043`** (1 nodes): `Filters the point cloud such that only points which are within a certain radial`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2044`** (1 nodes): `Compute the distance from this box to the ego vehicle in 2D.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2045`** (1 nodes): `Returns all EvalBoxes in a list.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2046`** (1 nodes): `Returns a list of all keys.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2047`** (1 nodes): `Initialize from serialized content.         :param content: A dictionary with th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2048`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2049`** (1 nodes): `Create a new DataFrame filled with data.         This version overwrites the ori`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2050`** (1 nodes): `Create a new DataFrame for event tracking.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2051`** (1 nodes): `Merge dataframes.          Params         ------         dfs : list of pandas.Da`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2052`** (1 nodes): `Initialize from serialized dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2053`** (1 nodes): `Return the distance function corresponding to the dist_fcn string.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2054`** (1 nodes): `Returns max recall achieved.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2055`** (1 nodes): `Returns max recall achieved.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2056`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2057`** (1 nodes): `Returns an md instance corresponding to having no predictions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2058`** (1 nodes): `Returns an md instance corresponding to a random results.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2059`** (1 nodes): `Initialize from serialized dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2060`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2061`** (1 nodes): `Creates "reasonable" submission (results and metadata) by looping through the mi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2062`** (1 nodes): `Run the evaluation with fixed randomness on the specified subset, with or withou`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2063`** (1 nodes): `This tests runs the evaluation for an arbitrary random set of predictions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2064`** (1 nodes): `This tests runs the evaluation with the ground truth used as predictions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2065`** (1 nodes): `Return the distance function corresponding to the dist_fcn string.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2066`** (1 nodes): `Returns index of max recall achieved.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2067`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2068`** (1 nodes): `Returns a md instance corresponding to having no predictions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2069`** (1 nodes): `Returns an md instance corresponding to a random results.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2070`** (1 nodes): `Calculates the mean over distance thresholds for each label.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2071`** (1 nodes): `Calculates the mean AP by averaging over distance thresholds and classes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2072`** (1 nodes): `Calculates the mean true positive error across all classes for each metric.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2073`** (1 nodes): `Compute the nuScenes detection score (NDS, weighted sum of the individual scores`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2074`** (1 nodes): `Initialize from serialized content.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2075`** (1 nodes): `Creates "reasonable" submission (results and metadata) by looping through the mi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2076`** (1 nodes): `Update stats dict with new combo of ids and counts.         :param stat_dict: {c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2077`** (1 nodes): `Load a '.png' segmentation mask, ignoring any colour map.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2078`** (1 nodes): `Load a '.mat' segmentation mask of the kind used in the SBD dataset.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2079`** (1 nodes): `Fields that will be transformed with this transform.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2080`** (1 nodes): `Comput visualization output.          A visualization method takes some inputs a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2081`** (1 nodes): `Convert instance to segmentation mask.          Args:             instance_mask:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2082`** (1 nodes): `Return current value of hyperparameter based on global step.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2083`** (1 nodes): `Computes the Hutchinson approximation of the hessian trace and accumulates it fo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2084`** (1 nodes): `Performs a single optimization step.         Arguments:             closure (cal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2085`** (1 nodes): `Get parameters for ``crop`` for a random sized crop.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2086`** (1 nodes): `prepare_ytvis.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2087`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2088`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2089`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2090`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2091`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2092`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2093`** (1 nodes): `Create evaluator(s) for a given dataset.         This uses the special metadata`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2094`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2095`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2096`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2097`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2098`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2099`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: whethe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2100`** (1 nodes): `NOTE: this interface is experimental.         Args:             in_channels: cha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2101`** (1 nodes): `Decode the mask annotation         :param anno: The mask annotation         :ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2102`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2103`** (1 nodes): `Returns:             torch.optim.Optimizer:          It now calls :func:`detectr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2104`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2105`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2106`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2107`** (1 nodes): `Returns:             DatasetEvaluator or None          It is not implemented by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2108`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2109`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2110`** (1 nodes): `patchconv.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2111`** (1 nodes): `Preprocess Pascal VOC labels by converting to integer 255-scale and         mark`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2112`** (1 nodes): `To support a custom dataset, implement this function to receive the predicted re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2113`** (1 nodes): `Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2114`** (1 nodes): `Only validate in KITTIDataset         Args:             gt_boxes: (N, 7 + C) [x,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2115`** (1 nodes): `Args:             pts_rect:             img_shape:             calib:          R`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2116`** (1 nodes): `Args:             batch_dict:                 frame_id:             pred_dicts:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2117`** (1 nodes): `Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2118`** (1 nodes): `Args:             box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom valu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2119`** (1 nodes): `PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2120`** (1 nodes): `Args:             aggregate_func:             xyz: (N, 3)             xyz_featur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2121`** (1 nodes): `Args:             batch_dict:                 batch_size:                 batch_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2122`** (1 nodes): `Args:             rois: (N, 7)             roi_labels: (N)             gt_boxes:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2123`** (1 nodes): `Args:             ctx:             features: (M1 + M2 ..., C)             idx: [`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2124`** (1 nodes): `Args:             ctx:             grad_out: (N1 + N2 ..., C)          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2125`** (1 nodes): `Args:             ctx:             // support_xyz: (N1 + N2 ..., 3) xyz coordina`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2126`** (1 nodes): `Args:             ctx:             support_xyz: (N1 + N2 ..., 3) xyz coordinates`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2127`** (1 nodes): `Args:             ctx:             grad_new_features: (M1 + M2 ..., num_c_out),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2128`** (1 nodes): `Args:             point_centers: (N, 3)             max_neighbour_distance: floa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2129`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2130`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2131`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2132`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2133`** (1 nodes): `Yield successive n-sized chunks from lst.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2134`** (1 nodes): `database file containing information about preproscessed dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2135`** (1 nodes): `database file containing information labels used by dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2136`** (1 nodes): `input: xyz: (n, 3), offset: (b), new_offset: (b)         output: idx: (m)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2137`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2138`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2139`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2140`** (1 nodes): `input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2141`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2142`** (1 nodes): `input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2143`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2144`** (1 nodes): `input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2145`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2146`** (1 nodes): `input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)         output: o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2147`** (1 nodes): `input: grad_output: (L, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2148`** (1 nodes): `input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2149`** (1 nodes): `input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2150`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2151`** (1 nodes): `input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2152`** (1 nodes): `input: grad_output: [M, h]         output: (N, h, hdim), None, (L, h, hdim, 3),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2153`** (1 nodes): `input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hd`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2154`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2155`** (1 nodes): `input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2156`** (1 nodes): `input: grad_output: (N, h, C//h)         output: (M, h), (N, h, C//h), None, Non`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2157`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2158`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2159`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2160`** (1 nodes): `input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2161`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2162`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2163`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2164`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2165`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2166`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2167`** (1 nodes): `input: grad_out: (n, nsample, c)         output: grad_input1: (n, c), grad_input`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2168`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2169`** (1 nodes): `input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2170`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2171`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2172`** (1 nodes): `input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)         output`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2173`** (1 nodes): `input: input: (n, c), idx : (m, nsample)         output: (m, nsample, c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2174`** (1 nodes): `input: grad_out: (m, c, nsample)         output: (n, c), None`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2175`** (1 nodes): `input: input1: (n, c), input2: (n, c), idx: (n, nsample)         output:  (n, ns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2176`** (1 nodes): `input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2177`** (1 nodes): `input: grad_out: (n, c)         output: grad_input: (n, c), grad_position: (n, n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2178`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2179`** (1 nodes): `input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2180`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2181`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2182`** (1 nodes): `More memory-friendly matching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2183`** (1 nodes): `Performs the matching          Params:             outputs: This is a dict that`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2184`** (1 nodes): `tailwind.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2185`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2186`** (1 nodes): `prod.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2187`** (1 nodes): `dev.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2188`** (1 nodes): `Calculates the image embeddings for the provided image, allowing         masks t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2189`** (1 nodes): `Predict masks for the given input prompts, using the currently set image.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2190`** (1 nodes): `Generates masks for the given image.          Arguments:           image (np.nda`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2191`** (1 nodes): `Removes small disconnected regions and holes in masks, then reruns         box N`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2192`** (1 nodes): `Predicts masks end-to-end from provided images and prompts.         If prompts a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2193`** (1 nodes): `dataset_experiment.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2194`** (1 nodes): `Args:             values: tensor of shape (batch, n_true_classes, n_pred_classes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2195`** (1 nodes): `Compute auxilliary outputs only needed for metrics and visualisations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2196`** (1 nodes): `Try to infer same padding for convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2197`** (1 nodes): `Try to infer same padding for transposed convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2198`** (1 nodes): `convert_selected_MICCAI.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2199`** (1 nodes): `convert_back_toMP4_MICCAI.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2200`** (1 nodes): `convert_back_toMP4_MICCAI_step2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2201`** (1 nodes): `cholec_mp4_2_frames.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2202`** (1 nodes): `mp4_reencode.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2203`** (1 nodes): `Warning: this function is expensive. Only call it when necessary to         visu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2204`** (1 nodes): `movi_d_exp_conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2205`** (1 nodes): `movi_a_exp_conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2206`** (1 nodes): `movi_b_exp_conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2207`** (1 nodes): `movi_e_exp_conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2208`** (1 nodes): `movi_c_exp_conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2209`** (1 nodes): `128x128 -> proposal size in the original image         original_bboxes: [B, 4],`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2210`** (1 nodes): `Separate binary masks into connected components and return their bounding boxes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2211`** (1 nodes): `Enlarge bounding boxes by a common ratio, ensuring they stay within the image di`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2212`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2213`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage.         L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2214`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2215`** (1 nodes): `Prepare some proposals to be used to train the ROI heads.         It performs bo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2216`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_features (li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2217`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2218`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2219`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2220`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2221`** (1 nodes): `Returns:             iterable          It now calls :func:`detectron2.data.build`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2222`** (1 nodes): `Returns:             DatasetEvaluator or None          It is not implemented by`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2223`** (1 nodes): `Evaluate the given model. The given model is expected to already contain`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2224`** (1 nodes): `When the config is defined for certain number of workers (according to         ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2225`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2226`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2227`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2228`** (1 nodes): `NOTE: this interface is experimental.         Args:             is_train: for tr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2229`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2230`** (1 nodes): `Rescale the output instances to the target size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2231`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2232`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2233`** (1 nodes): `Returns:             torch.nn.Module:          It now calls :func:`detectron2.mo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2234`** (1 nodes): `It now calls :func:`detectron2.solver.build_lr_scheduler`.         Overwrite it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2235`** (1 nodes): `Returns:             CfgNode: a new config. Same as original if ``cfg.SOLVER.REF`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2236`** (1 nodes): `Generates masks for the given image.          Arguments:           image (np.nda`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2237`** (1 nodes): `Removes small disconnected regions and holes in masks, then reruns         box N`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2238`** (1 nodes): `Removes small disconnected regions and holes in a mask. Returns the         mask`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2239`** (1 nodes): `NOTE: this interface is experimental.          Args:             is_train: wheth`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2240`** (1 nodes): `Args:             input_shape: shapes (channels and stride) of the input feature`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2241`** (1 nodes): `NOTE: this interface is experimental.         Args:             input_shape: sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2242`** (1 nodes): `:param features: multi-scale features from the backbone         :param masks: im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2243`** (1 nodes): `NOTE: this interface is experimental.         Args:             in_channels: cha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2244`** (1 nodes): `Input:             - tgt/tgt_query_pos: nq, bs, d_model             -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2245`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2246`** (1 nodes): `Compute auxilliary outputs only needed for metrics and visualisations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2247`** (1 nodes): `Try to infer same padding for convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2248`** (1 nodes): `Try to infer same padding for transposed convolutions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2249`** (1 nodes): `Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2250`** (1 nodes): `Updates the internal state of the metric. In particular, we track update the cos`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2251`** (1 nodes): `Remaps the semantic classes to the target class using the latest assignments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2252`** (1 nodes): `Getter method to access things prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2253`** (1 nodes): `Getter method to access stuffs prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2254`** (1 nodes): `Setter method to access things prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2255`** (1 nodes): `Setter method to access stuffs prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2256`** (1 nodes): `Args:             backbone: a backbone module, must follow detectron2's backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2257`** (1 nodes): `Args:             min_sizes: list of short-edge size to resize the image to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2258`** (1 nodes): `Open a context where some heads in `model.roi_heads` are temporarily turned off.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2259`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_pooler (ROI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2260`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage. Label the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2261`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2262`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2263`** (1 nodes): `Prepare some proposals to be used to train the ROI heads. It performs box matchi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2264`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_features (li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2265`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2266`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape: sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2267`** (1 nodes): `gmm.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2268`** (1 nodes): `spectral_clustering.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2269`** (1 nodes): `dinosaur_r-clevrtex.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2270`** (1 nodes): `slotdiffusion_r_vqvae-clevrtex.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2271`** (1 nodes): `dinosaur_r-voc.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2272`** (1 nodes): `slate_r_vqvae-clevrtex.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2273`** (1 nodes): `vqvae-voc-c4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2274`** (1 nodes): `vqvae-clevrtex-c256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2275`** (1 nodes): `vqvae-coco-c256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2276`** (1 nodes): `slate_r_vqvae-coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2277`** (1 nodes): `vqvae-coco-c4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2278`** (1 nodes): `spot_r-coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2279`** (1 nodes): `slate_r_vqvae-voc.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2280`** (1 nodes): `vqvae-voc-c256.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2281`** (1 nodes): `slotdiffusion_r_vqvae-coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2282`** (1 nodes): `dinosaur_r-coco.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2283`** (1 nodes): `vqvae-clevrtex-c4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2284`** (1 nodes): `dinosaur_r-coco-ViT_B16.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2285`** (1 nodes): `slotdiffusion_r_vqvae-voc.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2286`** (1 nodes): `https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metri`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2287`** (1 nodes): `idx_pd: shape=(b,n), dtype=int, indexed segment         idx_gt: shape=(b,n), dty`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2288`** (1 nodes): `idx_pd: shape=(b,n), dtype=uint8, indexed segment         idx_gt: shape=(b,n), d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2289`** (1 nodes): `https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2290`** (1 nodes): `https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2291`** (1 nodes): `- source: shape=(b,m,c)         - target: shape=(b,n,c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2292`** (1 nodes): `Convert the original folded images into LMDB files.          The code is adapted`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2293`** (1 nodes): `- video: bgr format, shape=(t,h,w,c=3), uint8         - bbox: both side normaliz`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2294`** (1 nodes): `from the last dim to first`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2295`** (1 nodes): `suppose bbox l-t-r-b is normalized; only zero out-crop bboxs, not remove them`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2296`** (1 nodes): `Structure dataset as follows and run it!         - VOC2012  # as training set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2297`** (1 nodes): `- image: bgr format, shape=(h,w,c=3), uint8         - segment: index format, sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2298`** (1 nodes): `Convert the original TFRecord files into one LMDB file, saving 10x storage space`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2299`** (1 nodes): `Adopted from SAVi official implementation VideoFromTfds class.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2300`** (1 nodes): `Adopted from SAVi official implementation SparseToDenseAnnotation class.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2301`** (1 nodes): `- video: bgr format, shape=(t,h,w,c=3), uint8         - bbox: both side normaliz`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2302`** (1 nodes): `Structure dataset as follows and run it!         - clevrtex_full  # as training`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2303`** (1 nodes): `- image: bgr format, shape=(h,w,c=3), uint8         - segment: index format, sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2304`** (1 nodes): `Download dataset MSCOCO:         - 2017 Train images [118K/18GB] http://images.c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2305`** (1 nodes): `straight-through gradient approximation          synchronized:         Straighte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2306`** (1 nodes): `Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2307`** (1 nodes): `Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2308`** (1 nodes): `euclidean kmeans in pytorch         https://github.com/subhadarship/kmeans_pytor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2309`** (1 nodes): `encode: in shape (b,c,h,w)         templat: in shape (m,c)         zsoft: in sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2310`** (1 nodes): `chunked cdist          source: shape=(b,m,c) or (m,c)         target: shape=(b,n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2311`** (1 nodes): `does not change ``len(layers)``'s value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2312`** (1 nodes): `Farthest Point Sampling.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2313`** (1 nodes): `quantz: [QuantiZ,..]         encode: shape=(b,h,w,c)         zsoft: shape=(b,h,w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2314`** (1 nodes): `quantz: [QuantiZ,..]         zidx: indexes, shape=(b,h,w,g)         output: shap`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2315`** (1 nodes): `Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2316`** (1 nodes): `Straightening Out the Straight-Through Estimator: Overcoming Optimization Challe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2317`** (1 nodes): `euclidean kmeans in pytorch         https://github.com/subhadarship/kmeans_pytor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2318`** (1 nodes): `对每个 encode 向量，从 templat 中找到最匹配的向量索引，并输出软分配概率。                  :param encode: Te`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2319`** (1 nodes): `chunked cdist          source: shape=(b,m,c) or (m,c)         target: shape=(b,n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2320`** (1 nodes): `Positional Encoding of shape [1, L, D].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2321`** (1 nodes): `proposals2mask.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2322`** (1 nodes): `proposals2bbox_pred_flow.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2323`** (1 nodes): `3D_step.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2324`** (1 nodes): `2D_step.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2325`** (1 nodes): `lost.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2326`** (1 nodes): `train_all.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2327`** (1 nodes): `train_cluster_3d.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2328`** (1 nodes): `Builds train/eval data transforms for the dataset class.         :param is_train`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2329`** (1 nodes): `Builds train/eval data transforms for the dataset class.         :param is_train`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2330`** (1 nodes): `imagenet_variant_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2331`** (1 nodes): `imagenet_config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2332`** (1 nodes): `create_benge_few_shot_train_splits.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2333`** (1 nodes): `Loss Name.          This function must be implemented and will return the name o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2334`** (1 nodes): `Placeholder of forward function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2335`** (1 nodes): `Compute segmentation loss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2336`** (1 nodes): `cityscapes_half_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2337`** (1 nodes): `synthia_aug.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2338`** (1 nodes): `daformer_swin_prompt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2339`** (1 nodes): `daformer_mit-b5_prompt.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2340`** (1 nodes): `daformer_mit-b5.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2341`** (1 nodes): `daformer_swin.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2342`** (1 nodes): `daformer_mit-b5_aug.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2343`** (1 nodes): `daformer_swin_aug.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2344`** (1 nodes): `schedule_200k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2345`** (1 nodes): `schedule_400k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2346`** (1 nodes): `schedule_40k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2347`** (1 nodes): `schedule.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2348`** (1 nodes): `X_to_cityscapes_mit_b5_daformer_prompt_multiscale.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2349`** (1 nodes): `generate_pseudo_label_X_to_cityscapes_daformer_mit_b5.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2350`** (1 nodes): `generate_pseudo_label_X_to_cityscapes_daformer_swin.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2351`** (1 nodes): `X_to_cityscapes_swin_daformer_prompt_multiscale.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2352`** (1 nodes): `daformer_swin_base_patch4_window7_512x512_400k_2e-6_synthia.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2353`** (1 nodes): `GtA_daformer_swin_base_patch4_window7_512x512_40k_2e-6_synthia.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2354`** (1 nodes): `daformer_swin_base_patch4_window7_512x512_400k_6e-6_gta.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2355`** (1 nodes): `GtA_daformer_swin_base_patch4_window7_512x512_40k_1e-5_gta.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2356`** (1 nodes): `daformer_mit-b5_512x512_400k_2e-5_gta.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2357`** (1 nodes): `daformer_mit-b5_512x512_400k_2e-6_synthia.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2358`** (1 nodes): `GtA_daformer_mit-b5_512x512_40k_4e-6_synthia.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2359`** (1 nodes): `GtA_daformer_mit-b5_512x512_40k_8e-6_gta.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2360`** (1 nodes): `convert_HD1K.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2361`** (1 nodes): `convert_things.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2362`** (1 nodes): `convert_sintel.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2363`** (1 nodes): `r"""Customize every aspect of training via flags.          Args:             acc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2364`** (1 nodes): `List up files in `dir_path` with `name_key`, then yield maximum suffix number.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2365`** (1 nodes): `Get path of maximum-epoch checkpoint in the folder.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2366`** (1 nodes): `Forward the inputs through the network and produce the predictions.          The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2367`** (1 nodes): `Load a pretrained MegaFlow model from HuggingFace Hub.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2368`** (1 nodes): `Performs feature rotation by splitting and recombining feature dimensions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2369`** (1 nodes): `Extract features from images.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2370`** (1 nodes): `Resize pos_embed weights.          Resize pos_embed using bicubic interpolate me`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2371`** (1 nodes): `evaluation_method.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2372`** (1 nodes): `evaluation_codalab.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2373`** (1 nodes): `Momentum update of evaluation model (exponential moving average)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2374`** (1 nodes): `download_models.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2375`** (1 nodes): `Inference interface for the model for PIL image         Args:             pil_im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2376`** (1 nodes): `eslint.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2377`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2378`** (1 nodes): `vite-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2379`** (1 nodes): `ThinkingIndicator.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2380`** (1 nodes): `Kick off a best-effort cpu-basic sandbox for the session.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2381`** (1 nodes): `Delete the sandbox Space if one was created for this session.          Retries o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2382`** (1 nodes): `Get count of active sessions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2383`** (1 nodes): `Create a new sandbox by duplicating the template Space.          Generates a uni`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2384`** (1 nodes): `Upload embedded sandbox server + Dockerfile to the Space (single commit).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2385`** (1 nodes): `Connect to an existing running Space.          Does a health check to verify the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2386`** (1 nodes): `Public URL of the Space.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2387`** (1 nodes): `Current Space stage (RUNNING, BUILDING, PAUSED, etc.).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2388`** (1 nodes): `Cancel pending approval tools when the user continues the conversation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2389`** (1 nodes): `Handle user input (like user_input_or_turn in codex.rs:1291)         Returns the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2390`** (1 nodes): `Remove the last complete turn and notify the frontend.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2391`** (1 nodes): `Start a fresh conversation inside the active runtime.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2392`** (1 nodes): `Reload context from a saved session log into the active session.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2393`** (1 nodes): `Handle batch job execution approval`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2394`** (1 nodes): `Handle shutdown (like shutdown in codex.rs:1329)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2395`** (1 nodes): `Spawn detached subprocess(es) to retry failed/pending uploads         (fire-and-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2396`** (1 nodes): `Ensure msg.tool_calls contains proper ToolCall objects, not dicts.          lite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2397`** (1 nodes): `Token count at which `compact()` kicks in.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2398`** (1 nodes): `bool: whether the segmentor has auxiliary head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2399`** (1 nodes): `Forward pass through full MBPS model.          Args:             image: Input im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2400`** (1 nodes): `Predict per-token semantic logits.          Args:             features: (B, N, b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2401`** (1 nodes): `Predict per-token instance embeddings.          Args:             features: (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2402`** (1 nodes): `Apply BiCMS fusion.          Args:             semantic: Semantic tokens (B, N,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2403`** (1 nodes): `Apply SSD selective scan.          Args:             x: Input sequence of shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2404`** (1 nodes): `Apply Mamba2 block.          Args:             x: Input of shape (B, L, D).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2405`** (1 nodes): `Apply stack of Mamba2 blocks.          Args:             x: Input of shape (B, L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2406`** (1 nodes): `Project features to bridge dimension.          Args:             semantic_codes:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2407`** (1 nodes): `Inverse project from bridge dimension.          Args:             x: Fused featu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2408`** (1 nodes): `Condition features on depth.          Args:             depth: Depth values of s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2409`** (1 nodes): `Classify clusters as stuff or things.          Args:             cues: Concatena`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2410`** (1 nodes): `Generate instance masks from features.          In inference mode with depth ava`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2411`** (1 nodes): `Refine masks through cascade stages.          Args:             features: Featur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2412`** (1 nodes): `Generate proposals from features.          Args:             features: Backbone`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2413`** (1 nodes): `Predict class scores and box deltas.          Args:             pooled_features:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2414`** (1 nodes): `Predict mask features.          Args:             features: Pooled RoI features,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2415`** (1 nodes): `Run one cascade stage.          Args:             features: Feature map, shape (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2416`** (1 nodes): `Generate instance masks and scores.          Args:             features: Input f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2417`** (1 nodes): `Compute semantic codes from DINO features.          Args:             features:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2418`** (1 nodes): `Compute semantic codes from spatial features.          Args:             feature`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2419`** (1 nodes): `Extract patch embeddings.          Args:             x: Input image of shape (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2420`** (1 nodes): `Apply multi-head self-attention.          Args:             x: Input of shape (B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2421`** (1 nodes): `Apply MLP.          Args:             x: Input of shape (B, N, D).             d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2422`** (1 nodes): `Apply Transformer block (pre-norm).          Args:             x: Input of shape`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2423`** (1 nodes): `Extract DINO features from input image.          Args:             x: Input imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2424`** (1 nodes): `Extract patch embeddings.          Args:             x: Input image of shape (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2425`** (1 nodes): `Apply multi-head self-attention.          Args:             x: Input of shape (B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2426`** (1 nodes): `Extract DINOv3 features from input image.          Args:             x: Input im`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2427`** (1 nodes): `download_spidepth.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2428`** (1 nodes): `Load an OLMo model from a checkpoint.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2429`** (1 nodes): `Returns the length of the idx-th trajectory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2430`** (1 nodes): `coco_classes.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2431`** (1 nodes): `One image / label pair for the given index is picked up and pre-processed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2432`** (1 nodes): `Dimension that can be used by transforms to set the correct image size, etc.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2433`** (1 nodes): `Decorator method that needs to be used around the ``__getitem__`` method. |br|`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2434`** (1 nodes): `Constructs a `BertConfig` from a Python dictionary of parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2435`** (1 nodes): `Constructs a `BertConfig` from a json file of parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2436`** (1 nodes): `Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch sta`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2437`** (1 nodes): `Get the reference points used in decoder.          Args:             spatial_sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2438`** (1 nodes): `torchvision_example.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2439`** (1 nodes): `focus_detr_swin_tiny_224_4scale_12ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2440`** (1 nodes): `focus_detr_swin_tiny_224_4scale_24ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2441`** (1 nodes): `focus_detr_swin_tiny_224_4scale_22k_12ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2442`** (1 nodes): `focus_detr_swin_base_384_4scale_36ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2443`** (1 nodes): `focus_detr_swin_tiny_224_4scale_36ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2444`** (1 nodes): `focus_detr_swin_tiny_224_4scale_22k_36ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2445`** (1 nodes): `focus_detr_swin_base_224_4scale_36ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2446`** (1 nodes): `focus_detr_swin_large_384_4scale_36ep.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2447`** (1 nodes): `PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2448`** (1 nodes): `Move the image channels to the first dimension of the numpy         multi-dimens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2449`** (1 nodes): `Move the image channels to the last dimensiion of the numpy         multi-dimens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2450`** (1 nodes): `Loads an image from file as a numpy multi-dimensional array          :param img_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2451`** (1 nodes): `Computes the mean squared error between to RGB images represented as multi-dimen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2452`** (1 nodes): `Computes the PSNR for a batch of input and output images          :param image_b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2453`** (1 nodes): `Computes the SSIM for a batch of input and output images          :param image_b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2454`** (1 nodes): `Abstract function for the data loader class          :returns: N/A         :rtyp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2455`** (1 nodes): `Abstract function for the data loader class          :returns: N/A         :rtyp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2456`** (1 nodes): `json_parser.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2457`** (1 nodes): `PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2458`** (1 nodes): `PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2459`** (1 nodes): `Move the image channels to the first dimension of the numpy         multi-dimens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2460`** (1 nodes): `Move the image channels to the last dimensiion of the numpy         multi-dimens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2461`** (1 nodes): `Loads an image from file as a numpy multi-dimensional array          :param img_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2462`** (1 nodes): `Normalises image data to be a float between 0 and 1          :param img: Image a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2463`** (1 nodes): `Computes the mean squared error between to RGB images represented as multi-dimen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2464`** (1 nodes): `Computes the PSNR for a batch of input and output images          :param image_b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2465`** (1 nodes): `Computes the SSIM for a batch of input and output images          :param image_b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2466`** (1 nodes): `Converts a HSV image to RGB         PyTorch implementation of RGB to HSV convers`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2467`** (1 nodes): `Converts an RGB image to HSV         PyTorch implementation of RGB to HSV conver`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2468`** (1 nodes): `Applies a peicewise linear curve defined by a set of knot points to         an i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2469`** (1 nodes): `Adjust the HSV channels of a HSV image using learnt curves          :param img:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2470`** (1 nodes): `Adjust the RGB channels of a RGB image using learnt curves          :param img:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2471`** (1 nodes): `Adjusts the image in LAB space using the predicted curves          :param img: I`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2472`** (1 nodes): `Abstract function for the data loader class          :returns: N/A         :rtyp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2473`** (1 nodes): `Abstract function for the data loader class          :returns: N/A         :rtyp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2474`** (1 nodes): `Compute log-likelihood of generating a continuation from a context.         Down`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2475`** (1 nodes): `Generate greedily until a stopping sequence          :param requests: list`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2476`** (1 nodes): `Parse the raw outputs (losses) of the network.          Args:             losses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2477`** (1 nodes): `Whether the task has a training set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2478`** (1 nodes): `Whether the task has a validation set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2479`** (1 nodes): `Whether the task has a test set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2480`** (1 nodes): `Uses RequestFactory to construct Requests and returns an iterable of         Req`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2481`** (1 nodes): `Take a single document and the LM results and evaluates, returning a         dic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2482`** (1 nodes): `:returns: {str: [metric_score] -> float}             A dictionary where keys are`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2483`** (1 nodes): `:returns: {str: bool}             A dictionary where keys are the names of subme`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2484`** (1 nodes): `Returns a fewshot context string that is made up of a prepended description`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2485`** (1 nodes): `Downstream tasks with custom word boundaries should override this!`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2486`** (1 nodes): `Whether to include special tokens in encoded text. This should be         determ`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2487`** (1 nodes): `Return the maximum sequence length of the model.         NOTE: Different model c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2488`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2489`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2490`** (1 nodes): `r"""         start_positions (`torch.LongTensor` of shape `(batch_size,)`, *opti`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2491`** (1 nodes): `r"""         Args:             input_ids (`torch.LongTensor` of shape `(batch_si`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2492`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2493`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2494`** (1 nodes): `This function is used to re-order the `past_key_values` cache if [`~PretrainedMo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2495`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2496`** (1 nodes): `This function is used to re-order the `past_key_values` cache if         [`~PreT`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2497`** (1 nodes): `r""" 		Generates sequences of token ids for models with a language modeling head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2498`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2499`** (1 nodes): `This function is used to re-order the `past_key_values` cache if [`~PreTrainedMo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2500`** (1 nodes): `r"""         mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2501`** (1 nodes): `This function is used to re-order the `past_key_values` cache if [`~PreTrainedMo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2502`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2503`** (1 nodes): `r"""         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2504`** (1 nodes): `Extract entities from tokens.          Returns:             list: list of Entity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2505`** (1 nodes): `evaluate_all.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2506`** (1 nodes): `Root function         Args:             step: Current step             c0: In`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2507`** (1 nodes): `make_copa.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2508`** (1 nodes): `Save training dynamics to a .json file         Each line contains a dictionary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2509`** (1 nodes): `Save training dynamics to a .json file         Each line contains a dictionary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2510`** (1 nodes): `run_sig_tests.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2511`** (1 nodes): `Reads a tab separated value file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2512`** (1 nodes): `将trie树的查询结果转换为匹配向量          sort=true，将按照起始位置和长度进行排序`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2513`** (1 nodes): `将trie树的查询结果转换为匹配向量         sort=true，将按照起始位置和长度进行排序`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2514`** (1 nodes): `collect_predictions.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2515`** (1 nodes): `collect_joint_predictions.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2516`** (1 nodes): `ploter.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2517`** (1 nodes): `reading.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2518`** (1 nodes): `model return (reconstructed_x, *)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2519`** (1 nodes): `sample new images from model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2520`** (1 nodes): `accepts (original images, *) where * is the same as returned from forward()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2521`** (1 nodes): `returns the latest losses in a dictionary. Useful for logging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2522`** (1 nodes): `model return (reconstructed_x, *)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2523`** (1 nodes): `sample new images from model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2524`** (1 nodes): `accepts (original images, *) where * is the same as returned from forward()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2525`** (1 nodes): `returns the latest losses in a dictionary. Useful for logging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2526`** (1 nodes): `train_cyclegan_b2c.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2527`** (1 nodes): `train_cyclegan_a2b.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2528`** (1 nodes): `During Pydantic model instantiation, only validate that `entry_point_agent` is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2529`** (1 nodes): `Streaming mode to generate LLM response`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2530`** (1 nodes): `Generate LLM response`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2531`** (1 nodes): `Convert state msg list into openai format`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2532`** (1 nodes): `Generate LLM response using streaming mode`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2533`** (1 nodes): `load a built graph_builder from file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2534`** (1 nodes): `load a built graph_builder from a config dict`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2535`** (1 nodes): `Put streaming msg into stream writer and trigger the msg handler`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2536`** (1 nodes): `Return next nodes due to current state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2537`** (1 nodes): `Return all possible targets.         This is used to static analyze the graph st`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2538`** (1 nodes): `if element is tuple[str, callable]`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2539`** (1 nodes): `Output_msg_format cannot be none when output_schema is none`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2540`** (1 nodes): `Get descriptions for pydantic fields`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2541`** (1 nodes): `Create a class instance using the given name and kwargs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2542`** (1 nodes): `Register a class into the factory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2543`** (1 nodes): `Returns the current active client session (read-only)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2544`** (1 nodes): `Returns connection status (True/False) for monitoring purposes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2545`** (1 nodes): `Return a list of tool schemas in OpenAI function format          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2546`** (1 nodes): `Execute tool calls with provided tasks          Args:             tasks: List of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2547`** (1 nodes): `Set the tool controller that manages tool activation rules          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2548`** (1 nodes): `Get the current tool controller instance          Returns:             The curre`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2549`** (1 nodes): `clear the vector store`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2550`** (1 nodes): `the retrieval entrance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2551`** (1 nodes): `add new db item to vectorstore`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2552`** (1 nodes): `delete item by it original ids         :param ids:         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2553`** (1 nodes): `Get vector count          Returns:             Number of vectors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2554`** (1 nodes): `Get collection information          Returns:             Collection information`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2555`** (1 nodes): `retrival memory and update context messages.         :param messages: context me`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2556`** (1 nodes): `add context messages to memory vectorstore         :param messages: context mess`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2557`** (1 nodes): `clear all memories         :return:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2558`** (1 nodes): `Extract features from contexts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2559`** (1 nodes): `Merge current info with existing memories`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2560`** (1 nodes): `Summary the recalled memories`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2561`** (1 nodes): `Based on mem summary info to update basic messages`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2562`** (1 nodes): `Calls the configured LLM model with a given query.          Args:             qu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2563`** (1 nodes): `Fetches raw text content from the specified URL.          Args:             url`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2564`** (1 nodes): `Processes raw text content to extract relevant information based on a query.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2565`** (1 nodes): `Retrieves the Content-Type header of a URL via a HEAD request.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2566`** (1 nodes): `Instantiates a parser suitable for the given URL.          Selection Logic:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2567`** (1 nodes): `Evaluate a single data item.          Subclasses must implement this method to d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2568`** (1 nodes): `Simple, reliable and slow implementation of batch by size`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2569`** (1 nodes): `Do forward, backward and parameter update.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2570`** (1 nodes): `Do forward pass in evaluation mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2571`** (1 nodes): `Generate a batch of translations.          Args:             sample (dict): batc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2572`** (1 nodes): `Reorder encoder output according to *new_order*.          Args:             enco`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2573`** (1 nodes): `Do forward, backward and parameter update.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2574`** (1 nodes): `Do forward pass in evaluation mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2575`** (1 nodes): `Score a batch of translations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2576`** (1 nodes): `Initialize constraint states for constrained decoding (if supported).          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2577`** (1 nodes): `A constrained step builds a large candidates list from the following:         -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2578`** (1 nodes): `Does per-sentence processing. Adds all constraints for each         hypothesis t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2579`** (1 nodes): `Do we require PathManager to access given path?`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2580`** (1 nodes): `Do forward, backward and parameter update.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2581`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2582`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2583`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2584`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2585`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2586`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2587`** (1 nodes): `Whether the logging outputs returned by `train_step` and `valid_step` can`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2588`** (1 nodes): `Load the dictionary from the filename          Args:             filename (str):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2589`** (1 nodes): `Build the dictionary          Args:             filenames (list): list of filena`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2590`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             cfg (omegac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2591`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary` (if applicable         for t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2592`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary` (if applicable         for t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2593`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2594`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2595`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2596`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2597`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2598`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2599`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2600`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2601`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2602`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2603`** (1 nodes): `Return the :class:`~fairseq.data.Dictionary` for the language         model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2604`** (1 nodes): `Return the :class:`~fairseq.data.Dictionary` for the language         model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2605`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2606`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             cfg (AudioP`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2607`** (1 nodes): `Return the :class:`~fairseq.data.Dictionary` for the language         model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2608`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2609`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2610`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2611`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2612`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2613`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2614`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2615`** (1 nodes): `Setup the task (e.g., load dictionaries).          Args:             args (argpa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2616`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2617`** (1 nodes): `Return the target :class:`~fairseq.data.Dictionary`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2618`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2619`** (1 nodes): `Load the dictionary from the filename          Args:             filename (str):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2620`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2621`** (1 nodes): `Load the masked LM dictionary from the filename          Args:             filen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2622`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2623`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2624`** (1 nodes): `Load the dictionary from the filename          Args:             filename (str):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2625`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2626`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2627`** (1 nodes): `A context manager to disable gradient synchronization.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2628`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2629`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2630`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2631`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2632`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2633`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2634`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2635`** (1 nodes): `Return a torch.optim.optimizer.Optimizer instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2636`** (1 nodes): `Reset optimizer instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2637`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2638`** (1 nodes): `Return an iterable of the parameters held by the optimizer.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2639`** (1 nodes): `Whether the optimizer supports collapsing of the model         parameters/gradie`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2640`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2641`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2642`** (1 nodes): `Args:             cfg (omegaconf.DictConfig): fairseq args             params (i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2643`** (1 nodes): `Args:             args (argparse.Namespace): fairseq args             params (it`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2644`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2645`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2646`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2647`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2648`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2649`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2650`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2651`** (1 nodes): `Add arguments to the parser for this LR scheduler.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2652`** (1 nodes): `Add arguments to the parser for this LR scheduler.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2653`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2654`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2655`** (1 nodes): `Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model         fi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2656`** (1 nodes): `Helper function to build shared embeddings for a set of languages after`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2657`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2658`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2659`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2660`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2661`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2662`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2663`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2664`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2665`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2666`** (1 nodes): `Get normalized probabilities (or log probs) from a net's output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2667`** (1 nodes): `Reorder encoder output according to *new_order*.          Args:             enco`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2668`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2669`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2670`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2671`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2672`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2673`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2674`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2675`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2676`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2677`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2678`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2679`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2680`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2681`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2682`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2683`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2684`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2685`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2686`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2687`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2688`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2689`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2690`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2691`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2692`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2693`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2694`** (1 nodes): `Reorder buffered internal state (for incremental generation).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2695`** (1 nodes): `Args:             incremental_state: Used to buffer signal; if not None, then in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2696`** (1 nodes): `Build sinusoidal embeddings.          This matches the implementation in tensor2`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2697`** (1 nodes): `Whether this dataset supports prefetching.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2698`** (1 nodes): `The number of consumed batches in the current epoch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2699`** (1 nodes): `Return the epoch index after *next_epoch_itr* is called.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2700`** (1 nodes): `Return the epoch index after *next_epoch_itr* is called.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2701`** (1 nodes): `The number of consumed batches in the current epoch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2702`** (1 nodes): `Loads the dictionary from a text file with the format:          ```         <sym`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2703`** (1 nodes): `Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for         th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2704`** (1 nodes): `Whether this dataset supports prefetching.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2705`** (1 nodes): `Whether this dataset supports fetching outside the workers of the dataloader.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2706`** (1 nodes): `fairseq vocabulary file under data root`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2707`** (1 nodes): `Shuffle dataset samples before batching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2708`** (1 nodes): `Pre-tokenizer to apply before subword tokenization. Returning         a dictiona`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2709`** (1 nodes): `Subword tokenizer to apply after pre-tokenization. Returning         a dictionar`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2710`** (1 nodes): `Prepend target lang ID token as the target BOS (e.g. for to-many         multili`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2711`** (1 nodes): `The dimension of input features (per audio channel)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2712`** (1 nodes): `The number of channels in the input audio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2713`** (1 nodes): `Hyper-parameter alpha = 1/T for temperature-based resampling.         (alpha = 1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2714`** (1 nodes): `Needed by the dataset loader to see if the model requires         raw audio as i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2715`** (1 nodes): `Audio paths in the manifest TSV can be relative and this provides         the ro`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2716`** (1 nodes): `Size ratios for temperature-based sampling         (https://arxiv.org/abs/1907.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2717`** (1 nodes): `Smoothed value used for logging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2718`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2719`** (1 nodes): `Construct a criterion from command-line args.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2720`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2721`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2722`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2723`** (1 nodes): `Construct a criterion from command-line args.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2724`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2725`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2726`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2727`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2728`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2729`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2730`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2731`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2732`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2733`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2734`** (1 nodes): `Args for MaskedLM Loss`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2735`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2736`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2737`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2738`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2739`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2740`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2741`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2742`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2743`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2744`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2745`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2746`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2747`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2748`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2749`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2750`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2751`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2752`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2753`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2754`** (1 nodes): `Expected sizes:         delays: tgt_len, batch_size         src_lens: 1, batch_s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2755`** (1 nodes): `delays : bsz, num_heads_x_layers, tgt_len         src_lens : bsz, 1         targ`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2756`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2757`** (1 nodes): `Reorder buffered internal state (for incremental generation).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2758`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2759`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2760`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2761`** (1 nodes): `Setup the task (e.g., load dictionaries).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2762`** (1 nodes): `Return the :class:`~fairseq.data.Dictionary` for the language         model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2763`** (1 nodes): `Return the source :class:`~fairseq.data.Dictionary` (if applicable         for t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2764`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2765`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2766`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2767`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2768`** (1 nodes): `Add model-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2769`** (1 nodes): `Build a new model instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2770`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2771`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2772`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2773`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2774`** (1 nodes): `Add optimizer-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2775`** (1 nodes): `Return a kwarg dictionary that will be used to override optimizer         args s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2776`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2777`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2778`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2779`** (1 nodes): `Whether the logging outputs returned by `forward` can be summed         across w`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2780`** (1 nodes): `Add criterion-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2781`** (1 nodes): `Aggregate logging outputs from data parallel training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2782`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2783`** (1 nodes): `Load the dictionary from the filename          Args:             filename (str):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2784`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2785`** (1 nodes): `Load the dictionary from the filename          Args:             filename (str):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2786`** (1 nodes): `Generate a batch of translations.         Args:             models (List[~fairse`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2787`** (1 nodes): `Add task-specific arguments to the parser.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2788`** (1 nodes): `spm_train.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2789`** (1 nodes): `convert_dictionary.lua`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2790`** (1 nodes): `segment_th.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2791`** (1 nodes): `Create a sentence embedder from a pretrained model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2792`** (1 nodes): `Register memory parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2793`** (1 nodes): `Check and initialize memory parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2794`** (1 nodes): `Create a dictionary from a vocabulary file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2795`** (1 nodes): `Index sentences with a dictionary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2796`** (1 nodes): `int: Input feature map levels.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2797`** (1 nodes): `cocoDemo.lua`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2798`** (1 nodes): `evalDemo.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2799`** (1 nodes): `CocoUtils.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2800`** (1 nodes): `getPrmDflt.m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2801`** (1 nodes): `Inference method. Switch model to `eval` mode,          call `.forward(x)` with`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2802`** (1 nodes): `Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2803`** (1 nodes): `Convert ImageNet-normalized [C,H,W] tensor to uint8 [H,W,3] BGR for OpenCV.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2804`** (1 nodes): `Morphological opening (remove small protrusions) then closing (fill holes).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2805`** (1 nodes): `Fast bilateral solver for mask refinement.          Simplified version using Ope`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2806`** (1 nodes): `Compute bounding boxes from binary masks. masks: [N, H, W] bool.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2807`** (1 nodes): `Updates the internal state of the metric. In particular, we track update the cos`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2808`** (1 nodes): `Remaps the semantic classes to the target class using the latest assignments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2809`** (1 nodes): `Getter method to access things prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2810`** (1 nodes): `Getter method to access stuffs prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2811`** (1 nodes): `Setter method to access stuffs prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2812`** (1 nodes): `combine_train_results.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2813`** (1 nodes): `combine_results.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2814`** (1 nodes): `check_k80.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2815`** (1 nodes): `Build resume command using latest GCS checkpoint.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2816`** (1 nodes): `Attach SAM masks to samples before geometric augmentation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2817`** (1 nodes): `Attach teacher logits before augmentation for aligned teacher gating.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2818`** (1 nodes): `Walk model hierarchy to find the ViT with .blocks attribute.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2819`** (1 nodes): `Rebuild optimizer (+ optional scheduler) for new LoRA parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2820`** (1 nodes): `Unfreeze all LoRA adapter parameters. Returns count.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2821`** (1 nodes): `Logs visualizations.          Args:             batch (List[Dict[str, Any]])): B`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2822`** (1 nodes): `Updates the internal state of the metric. In particular, we track update the cos`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2823`** (1 nodes): `Remaps the semantic classes to the target class using the latest assignments.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2824`** (1 nodes): `Getter method to access things prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2825`** (1 nodes): `Getter method to access stuffs prototypes.          Returns:             things_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2826`** (1 nodes): `Setter method to access things prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2827`** (1 nodes): `Setter method to access stuffs prototypes.          Args:             value (Set`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2828`** (1 nodes): `Construct from a yacs CfgNode (MODEL.LORA.MITIGATIONS).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2829`** (1 nodes): `Write averaged params into model. Returns count of params updated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2830`** (1 nodes): `NOTE: this interface is experimental.          Args:             sem_seg_head: a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2831`** (1 nodes): `Match proposals with groundtruth using the matcher at the given stage. Label the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2832`** (1 nodes): `Promote relaxed-IoU proposals for rare classes in later cascade stages.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2833`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape (Sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2834`** (1 nodes): `Initialize DepthFiLMSemSegHead.          Args:             input_shape: shapes (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2835`** (1 nodes): `Build config dict from detectron2 config.          Args:             cfg: detect`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2836`** (1 nodes): `NOTE: this interface is experimental.          Args:             num_classes (in`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2837`** (1 nodes): `Prepare some proposals to be used to train the ROI heads. It performs box matchi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2838`** (1 nodes): `NOTE: this interface is experimental.          Args:             in_features (li`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2839`** (1 nodes): `NOTE: this interface is experimental.          Args:             box_in_features`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2840`** (1 nodes): `NOTE: this interface is experimental.          Args:             input_shape: sh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2841`** (1 nodes): `Return sorted unique foreground values from a semantic PNG.          NOTE: For p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2842`** (1 nodes): `Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.          Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2843`** (1 nodes): `Convert all FrozenBatchNorm2d to BatchNorm2d          Args:             module (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2844`** (1 nodes): `Perform the computation         Parameters:             outputs: raw outputs of`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2845`** (1 nodes): `Files that match these patterns are not deleted by cleanup`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2846`** (1 nodes): `maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2847`** (1 nodes): `Explicit Test-Time Training adaptation.          For each image, perform K gradi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2848`** (1 nodes): `CRF-inspired pairwise consistency loss (differentiable CRF energy).          For`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2849`** (1 nodes): `Fuse DINOv3 and SSD-1B features via learned cross-attention.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2850`** (1 nodes): `Compute Sobel gradients. depth_2d: (H, W) → (2, H, W).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2851`** (1 nodes): `Confidence mask: 1.0 if >= threshold of 8 neighbors share same class.          R`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2852`** (1 nodes): `Get instance masks from the model.          Args:             features: (B, N, 7`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2853`** (1 nodes): `Generate pseudo-labels from EMA teacher predictions.          Returns semantic l`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2854`** (1 nodes): `Generate pseudo-labels with optional TTA.          Returns:             labels:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2855`** (1 nodes): `Extract CLS attention map as (H_patches, W_patches) numpy array.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2856`** (1 nodes): `Args:             img: (1, 3, H, W) tensor, normalized         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2857`** (1 nodes): `Compute Sobel gradients. depth_2d: (H, W) → (2, H, W).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2858`** (1 nodes): `Extract self-attention affinity matrix from last layer.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2859`** (1 nodes): `Apply graph diffusion to affinity matrix W (N, N).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2860`** (1 nodes): `Propagate features through the initial affinity graph.          Implements lazy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2861`** (1 nodes): `Nonlinear activation: φ(x) = x + 1.5·ELU(x).          For x > 0: φ(x) = 2.5x  (a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2862`** (1 nodes): `Refine a discrete segmentation map via NAMR.          Args:             pred:  (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2863`** (1 nodes): `Extract SD self-attention features for a single image.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2864`** (1 nodes): `Extract SSD-1B self-attention features for a single image.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2865`** (1 nodes): `Extract patch tokens from images.          Args:             images: (B, 3, H, W`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2866`** (1 nodes): `Bipartite matching between predictions and targets.          Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2867`** (1 nodes): `(B, 3, H, W) -> (B, N, 768) — CLS+registers already stripped by DINOv3ViTB.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2868`** (1 nodes): `27-class Cityscapes mIoU with Hungarian matching.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2869`** (1 nodes): `Post-process a batch of predictions.          Args:             pred_logits: (B,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2870`** (1 nodes): `Extract multi-layer features as 2D spatial maps.          Args:             pixe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2871`** (1 nodes): `Extract DINO features from input image.          Args:             x: Input imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2872`** (1 nodes): `Extract DINOv3 features.          Args:             x: Input image (B, 3, H, W),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2873`** (1 nodes): `Load pretrained DINOv3 weights from HuggingFace.          Args:             mode`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2874`** (1 nodes): `Forward pass should work on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2875`** (1 nodes): `Backward pass should work on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2876`** (1 nodes): `4D image input → same shape output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2877`** (1 nodes): `3D sequence input → same shape output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2878`** (1 nodes): `All scan modes should support backward pass on images.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2879`** (1 nodes): `Output should not contain NaN.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2880`** (1 nodes): `4D image inputs → same shape outputs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2881`** (1 nodes): `3D sequence inputs → same shape outputs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2882`** (1 nodes): `Cross-modal should support backward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2883`** (1 nodes): `Output should not contain NaN.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2884`** (1 nodes): `VisionMamba2 forward on MPS for all scan modes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2885`** (1 nodes): `VisionMamba2 backward on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2886`** (1 nodes): `CrossModalMamba2 on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2887`** (1 nodes): `GatedDeltaNet forward pass on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2888`** (1 nodes): `GatedDeltaNet backward pass on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2889`** (1 nodes): `4D image input with GDN layer → same shape output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2890`** (1 nodes): `All scan modes with GDN should support backward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2891`** (1 nodes): `GDN output should not contain NaN.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2892`** (1 nodes): `4D image inputs with GDN → same shape outputs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2893`** (1 nodes): `Cross-modal with GDN should support backward pass.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2894`** (1 nodes): `GDN cross-modal output should not contain NaN.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2895`** (1 nodes): `VisionMamba2 + GDN forward on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2896`** (1 nodes): `VisionMamba2 + GDN backward on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2897`** (1 nodes): `CrossModalMamba2 + GDN on MPS.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2898`** (1 nodes): `torch.Tensor: concatenated positive and negative boxes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2899`** (1 nodes): `Returns a dictionary of info about the object.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2900`** (1 nodes): `Sample positive samples.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2901`** (1 nodes): `Sample negative samples.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2902`** (1 nodes): `torch.Tensor: concatenated positive and negative boxes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2903`** (1 nodes): `Args:             rng (None | int | numpy.random.RandomState): seed or state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2904`** (1 nodes): `int: number of feature levels that the generator will be applied`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2905`** (1 nodes): `list[int]: The number of priors (points) at a point         on the feature grid`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2906`** (1 nodes): `Placeholder for sample function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2907`** (1 nodes): `Randomly select an img_scale from given candidates.          Args:             i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2908`** (1 nodes): `Randomly sample an img_scale when ``multiscale_mode=='range'``.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2909`** (1 nodes): `Randomly sample an img_scale when ``ratio_range`` is specified.          A ratio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2910`** (1 nodes): `Loss Name.          This function must be implemented and will return the name o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2911`** (1 nodes): `Forward function for `MultiheadAttention`.          **kwargs allow passing a mor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2912`** (1 nodes): `Forward function for `FFN`.          The function would add x to the output tens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2913`** (1 nodes): `Forward function for `FFN`.         The function would add x to the output tenso`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2914`** (1 nodes): `Get the reference points used in decoder.          Args:             spatial_sha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2915`** (1 nodes): `Assign boxes to either a ground truth boxes or a negative boxes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2916`** (1 nodes): `nn.Module: the normalization layer named "norm0"`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2917`** (1 nodes): `nn.Module: the normalization layer named "norm1"`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2918`** (1 nodes): `Resize pos_embed weights.          Resize pos_embed using bicubic interpolate me`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2919`** (1 nodes): `nn.Module: normalization layer after the first convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2920`** (1 nodes): `nn.Module: normalization layer after the second convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2921`** (1 nodes): `nn.Module: normalization layer after the first convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2922`** (1 nodes): `nn.Module: normalization layer after the second convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2923`** (1 nodes): `nn.Module: normalization layer after the third convolution layer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2924`** (1 nodes): `nn.Module: the normalization layer named "norm1"`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2925`** (1 nodes): `bool: whether the segmentor has neck`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2926`** (1 nodes): `bool: whether the segmentor has decode head`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2927`** (1 nodes): `Placeholder for extract features from images.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2928`** (1 nodes): `Placeholder for encode images with backbone and decode into a         semantic s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2929`** (1 nodes): `Placeholder for Forward function for training.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2930`** (1 nodes): `Placeholder for single image test.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2931`** (1 nodes): `Placeholder for augmentation test.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2932`** (1 nodes): `Calls either :func:`forward_train` or :func:`forward_test` depending         on`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2933`** (1 nodes): `Loss function.          Args:             all_cls_scores (Tensor): Classificatio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2934`** (1 nodes): `Loss function.          Args:             all_cls_scores (Tensor): Classificatio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2935`** (1 nodes): `Loss function.          Args:             all_cls_scores (Tensor): Classificatio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2936`** (1 nodes): `Placeholder of forward function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2937`** (1 nodes): `Compute segmentation loss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2938`** (1 nodes): `r"""         Instantiate a [`SiglipConfig`] (or a derived class) from siglip tex`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2939`** (1 nodes): `Make causal mask used for bi-directional self-attention.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2940`** (1 nodes): `Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_l`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2941`** (1 nodes): `Detects whether the optional user-specified attention_mask & the automatically c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2942`** (1 nodes): `Preprocess an image or batch of images.          Args:             images (`Imag`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2943`** (1 nodes): `cityscapes_1024x1024.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2944`** (1 nodes): `dg_gta_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2945`** (1 nodes): `gta2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2946`** (1 nodes): `syn2city.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2947`** (1 nodes): `gta2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2948`** (1 nodes): `city2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2949`** (1 nodes): `city2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2950`** (1 nodes): `cityscapes_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2951`** (1 nodes): `syn2map-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2952`** (1 nodes): `bdd100k_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2953`** (1 nodes): `mapillary_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2954`** (1 nodes): `city2bdd-1024.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2955`** (1 nodes): `gta_512x512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2956`** (1 nodes): `syn2bdd-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2957`** (1 nodes): `gta2city-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2958`** (1 nodes): `schedule_80k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2959`** (1 nodes): `schedule_20k.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2960`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2961`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2962`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2963`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2964`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2965`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2966`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2c-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2967`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2968`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-c2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2969`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2970`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2971`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2972`** (1 nodes): `mfuser_clip_vit-l_1e-4_20k-c2m-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2973`** (1 nodes): `mfuser_siglip_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2974`** (1 nodes): `mfuser_eva_vit-l_1e-4_20k-g2b-512.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2975`** (1 nodes): `x: (batch_size, seqlen, nheads, headdim)             cos, sin: (seqlen, rotary_d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2976`** (1 nodes): `logits: (batch, vocab_size)         labels: (batch,)         If process_group is`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2977`** (1 nodes): `Apply one EMA step from the student's current parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2978`** (1 nodes): `Per-query boolean mask: True => exclude this query from no-object CE.          S`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2979`** (1 nodes): `Multi-scale + flip TTA: average final-block logits per query.          Returns (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2980`** (1 nodes): `Compute loss masks for each of standard reprojection and depth hint         repr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2981`** (1 nodes): `Compute proxy supervised loss (depth hint loss) for prediction.              - v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2982`** (1 nodes): `Compute loss masks for each of standard reprojection and depth hint         repr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2983`** (1 nodes): `Compute proxy supervised loss (depth hint loss) for prediction.              - v`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2984`** (1 nodes): `Compute loss masks for each of standard reprojection and depth hint         repr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2985`** (1 nodes): `The best solution from the solver         Returns         -------         x : nd`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2986`** (1 nodes): `The standard deviation of the population energies divided by their         mean.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2987`** (1 nodes): `static_switch.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2988`** (1 nodes): `If process_group is not None and sequence_parallel=True, we're doing Tensor Para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2989`** (1 nodes): `xz: (batch, dim, seqlen)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2990`** (1 nodes): `If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * sil`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2991`** (1 nodes): `Update center used for teacher output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2992`** (1 nodes): `Walk model hierarchy to find the ViT with .blocks attribute.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2993`** (1 nodes): `Rebuild optimizer (+ optional scheduler) for new LoRA parameters.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2994`** (1 nodes): `Unfreeze all LoRA adapter parameters. Returns count.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2995`** (1 nodes): `Convert one image's final-layer logits into the D2 panoptic format.          Pan`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2996`** (1 nodes): `Forward pass through the model.          Args:             image: Input batch wi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2997`** (1 nodes): `Get scene data including images, camera parameters, and auxiliary info.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2998`** (1 nodes): `Evaluate 3D reconstruction quality against ground truth.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2999`** (1 nodes): `Fuse per-view depth maps into a single point cloud.          Args:             s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3000`** (1 nodes): `Directory for storing metric JSON files.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3001`** (1 nodes): `Convert numpy scalars to plain Python floats for JSON safety.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3002`** (1 nodes): `Compute elementwise mean across a list of homogeneous metric dicts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3003`** (1 nodes): `Write JSON with UTF-8 and pretty indentation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3004`** (1 nodes): `Fit global mean/V3 and initialize percentiles from a reference set.         fram`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3005`** (1 nodes): `X: (N,D) where N = H*W         Returns PCs_raw: (N,3) using stable basis (fixed`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3006`** (1 nodes): `frame: (H,W,D) -> (H,W,3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3007`** (1 nodes): `frames: (T,H,W,D) or list of (H,W,D)         returns: (T,H,W,3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3008`** (1 nodes): `Performs feature rotation by splitting and recombining feature dimensions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3009`** (1 nodes): `Handle export directory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3010`** (1 nodes): `Process image directory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3011`** (1 nodes): `Process video, extract frames`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3012`** (1 nodes): `quick_assign_shiftavg.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3013`** (1 nodes): `test_one_newline.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3014`** (1 nodes): `Extract SD self-attention features.          Args:             image: (1, 3, H,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3015`** (1 nodes): `test_no_newline.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3016`** (1 nodes): `(h, w, C) -> bilinear -> (hw[0]*hw[1], C).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3017`** (1 nodes): `Align feature sequence length via interpolation.          Args:             feat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 3018`** (1 nodes): `Transform DINOv3 features using depth conditioning.          Args:             f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Trainer` connect `Community 22` to `Community 34`, `Community 2`, `Community 15`, `Community 56`, `Community 24`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **Why does `MetricLogger` connect `Community 5` to `Community 2`?**
  _High betweenness centrality (0.003) - this node is a cross-community bridge._
- **Why does `TqdmFile` connect `Community 6` to `Community 2`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **What connects `setup_notebooklm.py — Bootstrap a NotebookLM notebook for MBPS BMVC 2026.  Creat`, `Run notebooklm CLI with given args.`, `Run CLI with --json flag and parse output.` to the rest of the system?**
  _19703 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._