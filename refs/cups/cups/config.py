import logging
import os
from typing import Any, List, Optional, Tuple

from yacs.config import CfgNode

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_C: CfgNode = CfgNode()

# General system configurations
_C.SYSTEM = CfgNode()
# Set accelerator
_C.SYSTEM.ACCELERATOR = "gpu"
# Number of GPUs to be utilized
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for data loading
_C.SYSTEM.NUM_WORKERS = 16
# Number of nodes to be utilized
_C.SYSTEM.NUM_NODES = 1
# Type of distributed backend to be used
_C.SYSTEM.DISTRIBUTED_BACKEND = "auto"
# Set logging path
_C.SYSTEM.LOG_PATH = "experiments"
# Set seed
_C.SYSTEM.SEED = 1996
# Set run name
_C.SYSTEM.RUN_NAME = None

# Model configurations
_C.MODEL = CfgNode()
# Set if DINO backbone should be used
_C.MODEL.USE_DINO = True
# Set backbone type: "resnet50" (DINO ResNet-50) or "dinov2_vitb" (DINOv2 ViT-B/14)
_C.MODEL.BACKBONE_TYPE = "resnet50"
# Freeze DINOv2 backbone (only used when BACKBONE_TYPE="dinov2_vitb")
_C.MODEL.DINOV2_FREEZE = True
# EoMT-only: freeze the ENTIRE encoder backbone (all blocks + norm), training
# only queries + class/mask heads + upscale. Anti-collapse for self-training.
_C.MODEL.EOMT_FREEZE_ALL_BLOCKS = False
# Set model checkpoint
_C.MODEL.CHECKPOINT = None
# Set inference confidence threshold
_C.MODEL.INFERENCE_CONFIDENCE_THRESHOLD = 0.5
# Set TTA object detection threshold
_C.MODEL.TTA_INFERENCE_CONFIDENCE_THRESHOLD = 0.5
# Set TTA scales
_C.MODEL.TTA_SCALES = (0.5, 0.75, 1.0)

# ROI box-head long-tail losses. EQLv2 and Seesaw are mutually exclusive.
_C.MODEL.ROI_BOX_HEAD = CfgNode()
_C.MODEL.ROI_BOX_HEAD.USE_EQLV2 = False
_C.MODEL.ROI_BOX_HEAD.EQLV2_GAMMA = 12.0
_C.MODEL.ROI_BOX_HEAD.EQLV2_MU = 0.8
_C.MODEL.ROI_BOX_HEAD.EQLV2_ALPHA = 4.0
_C.MODEL.ROI_BOX_HEAD.USE_SEESAW_LOSS = False
_C.MODEL.ROI_BOX_HEAD.SEESAW_P = 0.8
_C.MODEL.ROI_BOX_HEAD.SEESAW_Q = 2.0
# SAM3 Mask-Adapter: cross-attention between ROI features and SAM3 mask embeddings (Ablation 3).
_C.MODEL.ROI_BOX_HEAD.SAM3_MASK_ADAPTER = False
_C.MODEL.ROI_BOX_HEAD.SAM3_ADAPTER_DIM = 256
_C.MODEL.ROI_BOX_HEAD.SAM3_N_MAX_MASKS = 20
_C.MODEL.ROI_BOX_HEAD.SAM3_MASKS_DIR = ""  # absolute path to sam_fine_masks_sam3/train/
# Box-regression loss type for FastRCNNOutputLayers: "smooth_l1" (default) or "giou".
# Plumbed to detectron2 cfg in panoptic_cascade_mask_r_cnn_dinov3 before model build.
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"

# Cascade Mask R-CNN per-stage matching IoU thresholds.
# Default 0.5/0.6/0.7. Set [0.5, 0.55, 0.6] to give noisy pseudo-boxes a chance
# to match at the refinement stages s1/s2.
_C.MODEL.ROI_BOX_CASCADE_HEAD = CfgNode()
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)

# Depth-aware proposal consistency loss for mask head (0.0 = disabled).
# Penalizes fg proposals whose bbox region spans high depth variance.
_C.MODEL.ROI_MASK_HEAD = CfgNode()
_C.MODEL.ROI_MASK_HEAD.DEPTH_DICE_WEIGHT = 0.0

# LoRA/DoRA/Conv-DoRA backbone adaptation
_C.MODEL.LORA = CfgNode()
_C.MODEL.LORA.ENABLED = False
_C.MODEL.LORA.VARIANT = "dora"  # "dora", "conv_dora", or "lora"
_C.MODEL.LORA.RANK = 4
_C.MODEL.LORA.ALPHA = 4.0
_C.MODEL.LORA.DROPOUT = 0.05
_C.MODEL.LORA.LATE_BLOCK_START = 6
_C.MODEL.LORA.LR_A = 1e-5
_C.MODEL.LORA.LR_B = 5e-5
_C.MODEL.LORA.MAGNITUDE_WD = 1e-3
_C.MODEL.LORA.DELAYED_START_STEPS = 500
# Progressive LoRA for Stage-3 self-training (Filatov & Kindulov, 2023)
_C.MODEL.LORA.PROGRESSIVE = CfgNode()
_C.MODEL.LORA.PROGRESSIVE.ENABLED = False
_C.MODEL.LORA.PROGRESSIVE.RANKS = (2, 4, 8)
_C.MODEL.LORA.PROGRESSIVE.ALPHAS = (2.0, 4.0, 8.0)
_C.MODEL.LORA.PROGRESSIVE.COVERAGES = (6, 6, 0)  # late_block_start per round

# Noise-robustness mitigations (ablation experiments 15-21)
_C.MODEL.LORA.MITIGATIONS = CfgNode()
# M1: Cosine LR warmup for LoRA param groups
_C.MODEL.LORA.MITIGATIONS.COSINE_WARMUP = CfgNode()
_C.MODEL.LORA.MITIGATIONS.COSINE_WARMUP.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.COSINE_WARMUP.WARMUP_STEPS = 500
# M2: Magnitude warmup — freeze m for N steps per round
_C.MODEL.LORA.MITIGATIONS.MAGNITUDE_WARMUP = CfgNode()
_C.MODEL.LORA.MITIGATIONS.MAGNITUDE_WARMUP.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.MAGNITUDE_WARMUP.FREEZE_STEPS = 200
# M3: Spectral norm ball constraint on magnitude vector m
_C.MODEL.LORA.MITIGATIONS.SPECTRAL_NORM_BALL = CfgNode()
_C.MODEL.LORA.MITIGATIONS.SPECTRAL_NORM_BALL.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.SPECTRAL_NORM_BALL.DELTA = 0.1
# M4: SWA over last fraction of each self-training round
_C.MODEL.LORA.MITIGATIONS.SWA = CfgNode()
_C.MODEL.LORA.MITIGATIONS.SWA.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.SWA.FRACTION = 0.3
# M5: Confidence-weighted semantic loss from teacher softmax
_C.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS = CfgNode()
_C.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.TEMPERATURE = 1.0
_C.MODEL.LORA.MITIGATIONS.CONFIDENCE_WEIGHTED_LOSS.MIN_WEIGHT = 0.1
# M6: Adaptive delayed start — activate LoRA when head loss converges
_C.MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START = CfgNode()
_C.MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START.ENABLED = False
_C.MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START.TAU = 0.7
_C.MODEL.LORA.MITIGATIONS.ADAPTIVE_DELAYED_START.MAX_WAIT_STEPS = 1000

# Approach B: Stuff-preservation KD loss + Depth FiLM semantic head
_C.MODEL.SEM_SEG_HEAD = CfgNode()
# Weight for stuff-preservation KD loss (0.0 = disabled)
_C.MODEL.SEM_SEG_HEAD.STUFF_KD_WEIGHT = 0.0
# Temperature for softening pseudo-label targets in KD
_C.MODEL.SEM_SEG_HEAD.KD_TEMPERATURE = 2.0
# Enable depth FiLM conditioning (swap head to DepthFiLMSemSegHead)
_C.MODEL.SEM_SEG_HEAD.USE_DEPTH_FILM = False
# Number of depth encoder output channels (sinusoidal + Sobel + raw)
_C.MODEL.SEM_SEG_HEAD.DEPTH_CHANNELS = 15
# LDAM semantic-head loss for long-tail preservation.
_C.MODEL.SEM_SEG_HEAD.LDAM_ENABLED = False
_C.MODEL.SEM_SEG_HEAD.LDAM_MAX_MARGIN = 0.5
_C.MODEL.SEM_SEG_HEAD.LDAM_S = 30.0
_C.MODEL.SEM_SEG_HEAD.LDAM_CLASS_FREQ = ()

# Stage-2 semantic head auxiliary losses (P1-P4 of loss augmentation plan).
# All aux weights default to 0.0 = disabled; enable per-pass via YAML overrides.
# P1 -- LoCE (Lovasz-Softmax + boundary-weighted CE)
_C.MODEL.SEM_SEG_HEAD.LOVASZ_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_DILATE_PX = 3
_C.MODEL.SEM_SEG_HEAD.BOUNDARY_CE_MULT = 2.0
# P2 -- FeatMirror (STEGO correspondence on DINOv3 features)
_C.MODEL.SEM_SEG_HEAD.STEGO_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.STEGO_TEMPERATURE = 0.1
_C.MODEL.SEM_SEG_HEAD.STEGO_KNN_K = 7
_C.MODEL.SEM_SEG_HEAD.STEGO_FEATURE_SOURCE = "fpn_p2"  # "fpn_p2" or "vit_patch"
# P3 -- DGLR (depth-guided logit regularizer)
_C.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.DEPTH_SMOOTH_ALPHA = 10.0
# P4 -- DAff (Gated-CRF + NeCo dense affinity)
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_KERNEL = 5
_C.MODEL.SEM_SEG_HEAD.GATED_CRF_RGB_SIGMA = 0.1
_C.MODEL.SEM_SEG_HEAD.NECO_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.NECO_K = 5

# Stage-4 dead-class recovery semantic-head losses.
# Class ids here are Detectron2 semantic-head target ids (0=thing region,
# 1..S=stuff classes); they are filled dynamically from STAGE4 when possible.
_C.MODEL.SEM_SEG_HEAD.RARE_STUFF_CLASSES = ()
_C.MODEL.SEM_SEG_HEAD.RARE_FOCAL_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.RARE_FOCAL_GAMMA = 2.0
_C.MODEL.SEM_SEG_HEAD.RARE_BOUNDARY_WEIGHT = 0.0
_C.MODEL.SEM_SEG_HEAD.RARE_BOUNDARY_WIDTH = 3

# Dataset configurations
_C.DATA = CfgNode()
# Subdirectory under DATA.ROOT containing precomputed depth maps (empty = disabled)
_C.DATA.DEPTH_SUBDIR = ""
# Dataset to be used (currently cityscapes or kitti)
_C.DATA.DATASET = "cityscapes"
# Dataset path
_C.DATA.ROOT = "datasets/Cityscapes"
# Dataset path validation set 1 (used in training for validation)
_C.DATA.ROOT_VAL = "datasets/Cityscapes"
# Pseudo label dataset
_C.DATA.ROOT_PSEUDO = "pseudo_labels"
# Number data splits
_C.DATA.NUM_PREPROCESSING_SUBSPLITS = 2
# Number data splits
_C.DATA.PREPROCESSING_SUBSPLIT = 1
# Dataset path
_C.DATA.PSEUDO_ROOT = "pseudo_labels"
# Number of semantic pseudo classes
_C.DATA.NUM_PSEUDO_CLASSES = 27
# Set crop resolution
_C.DATA.CROP_RESOLUTION = (640, 1280)
# Set thing stuff threshold
_C.DATA.THING_STUFF_THRESHOLD = 0.08
# Set number of classes to be used (27, 19, or 7)
_C.DATA.NUM_CLASSES = 27
# Set training scale
_C.DATA.SCALE = 0.625
# Set validation scale
_C.DATA.VAL_SCALE = 0.625
# Set if thing regions not occupied by an object proposal should be ignored
_C.DATA.IGNORE_UNKNOWN_THING_REGIONS = False
# Repeat-factor sampling for long-tail pseudo-label training.
_C.DATA.USE_REPEAT_FACTOR_SAMPLER = False
_C.DATA.RFS_THRESHOLD_T = 0.001
_C.DATA.RFS_PRECOMPUTE_CACHE = ""

# Training specific config
_C.TRAINING = CfgNode()
# Set number of epochs to be performed
_C.TRAINING.STEPS = 4000
# Define batch size
_C.TRAINING.BATCH_SIZE = 16
# Set training precision to be used
_C.TRAINING.PRECISION = "bf16"
# Set gradient clipping approach
_C.TRAINING.GRADIENT_CLIP_ALGORITHM = "norm"
# Set gradient clipping value
_C.TRAINING.GRADIENT_CLIP_VAL = 1.0
# Set log frequency
_C.TRAINING.LOG_EVERT_N_STEPS = 1
# Set media log frequency
_C.TRAINING.LOG_MEDIA_N_STEPS = 100
# Set validation frequency (in batch steps, matches val_check_interval)
_C.TRAINING.VAL_EVERY_N_STEPS = 200
# Checkpoint save frequency in optimizer steps (set to VAL_EVERY_N_STEPS // ACCUMULATE_GRAD_BATCHES
# so a checkpoint is saved after every validation; default None = same as VAL_EVERY_N_STEPS)
_C.TRAINING.CKPT_EVERY_N_STEPS = None
# Set if class weighting should be used
_C.TRAINING.CLASS_WEIGHTING = False
# Set type of optimizer to be used
_C.TRAINING.OPTIMIZER = "adamw"
# SGD specific config
_C.TRAINING.SGD = CfgNode()
# Set learning rate
_C.TRAINING.SGD.LEARNING_RATE = 0.005
# Set weight decay
_C.TRAINING.SGD.WEIGHT_DECAY = 0.00005
# Set Adam decays
_C.TRAINING.SGD.MOMENTUM = 0.9
# ADAMW specific config
_C.TRAINING.ADAMW = CfgNode()
# Set learning rate
_C.TRAINING.ADAMW.LEARNING_RATE = 0.0001
# Set weight decay
_C.TRAINING.ADAMW.WEIGHT_DECAY = 0.00001
# Set Adam decays
_C.TRAINING.ADAMW.BETAS = (0.9, 0.999)
# Set drop loss IoU threshold
_C.TRAINING.DROP_LOSS_IOU_THRESHOLD = 0.4
# Set binary flag if drop loss should be used
_C.TRAINING.DROP_LOSS = True
# Gradient accumulation steps (effective batch = batch_size * num_gpus * accumulate)
_C.TRAINING.ACCUMULATE_GRAD_BATCHES = 1
# Per-cascade-stage multipliers applied to loss_box_reg_stage{0,1,2}.
# Empty tuple = no rescaling. Consumed by UnsupervisedModelLossOnly to combat
# noisy-box whiplash on s1/s2 from depth-split pseudo-labels.
_C.TRAINING.CASCADE_BOX_REG_WEIGHTS = ()
# Optional LR schedule. TYPE in {"none", "cosine_warmup"}. When set to
# "cosine_warmup", LR linearly warms up over WARMUP_STEPS then cosine-decays
# from ADAMW.LEARNING_RATE to MIN_LR by TRAINING.STEPS.
_C.TRAINING.LR_SCHEDULER = CfgNode()
_C.TRAINING.LR_SCHEDULER.TYPE = "none"
_C.TRAINING.LR_SCHEDULER.WARMUP_STEPS = 500
_C.TRAINING.LR_SCHEDULER.MIN_LR = 0.000001

# Self-training specific config
_C.SELF_TRAINING = CfgNode()
# Round length in training steps
_C.SELF_TRAINING.ROUND_STEPS = 500
# Self-training rounds
_C.SELF_TRAINING.ROUNDS = 3
# Set if drop-loss should be used for self-training
_C.SELF_TRAINING.USE_DROP_LOSS = False
# Set semantic segmentation threshold
_C.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD = 0.5
# Optional class-frequency-aware thresholding parameters for stage-3 configs.
_C.SELF_TRAINING.CLASS_THRESHOLD_ALPHA = 0.0
_C.SELF_TRAINING.CLASS_FREQUENCIES = ()
# Set confidence step for each stage (base confidence + stage * confidence step)
_C.SELF_TRAINING.CONFIDENCE_STEP = 0.05
# Disable EMA teacher updates (Exp 13: test LoRA implicit smoothing)
_C.SELF_TRAINING.DISABLE_EMA = False
# EMA teacher decay. Default 0.999 (CUPS). Raise to 0.9999 for high-capacity
# decoders (EoMT) to slow teacher drift ~10x and prevent self-distillation
# collapse.
_C.SELF_TRAINING.EMA_DECAY = 0.999

# Fine-object SAM supervision (Stage-4 fine-tuning).
# Uses pre-computed SAM masks as region priors to recover dead/rare classes
# (bicycle, motorcycle, rider, traffic sign, etc.) that depth pseudo-labels miss.
# Enable by setting ENABLED=True and pointing SAM_MASKS_DIR to the output of
# scripts/generate_sam_fine_masks.py (the split-level directory, e.g. .../train/).
_C.SELF_TRAINING.FINE_OBJECT = CfgNode()
_C.SELF_TRAINING.FINE_OBJECT.ENABLED = False
# Scalar weight: total_loss += WEIGHT * L_fine
_C.SELF_TRAINING.FINE_OBJECT.WEIGHT = 0.1
# Directory containing per-image .npz files (output of generate_sam_fine_masks.py)
_C.SELF_TRAINING.FINE_OBJECT.SAM_MASKS_DIR = ""
# Loss mode: "entropy" (class-agnostic, forces confidence inside mask)
#            "thing_coverage" (pushes toward thing-region class; requires correct thing_class_idx)
#            "thing_focal_stuff_entropy" (focal CE on thing masks, entropy on stuff)
#            "thing_focal_only" (focal CE on thing masks only, stuff masks SKIPPED)
_C.SELF_TRAINING.FINE_OBJECT.MODE = "entropy"
# Minimum SAM IoU/confidence score to use a mask (discard noisy masks)
_C.SELF_TRAINING.FINE_OBJECT.MIN_IOU_SCORE = 0.78
# Maximum number of SAM masks to use per image (avoid memory spikes)
_C.SELF_TRAINING.FINE_OBJECT.MAX_MASKS_PER_IMAGE = 30
# Thing-region class index in the semantic head vocabulary.
# In CUPS: channel 0 (the FIRST channel, not the last).  things_classes → 0,
# stuff_classes → 1..S in the dataloader; panoptic_fpn.py skips semantic_label==0.
# Use -1 for auto-detect (resolves to 0).
_C.SELF_TRAINING.FINE_OBJECT.THING_CLASS_IDX = -1
# Focal-loss gamma for thing-class CE in "thing_focal_stuff_entropy" mode.
_C.SELF_TRAINING.FINE_OBJECT.FOCAL_GAMMA = 2.0
# Weight each mask loss by its SAM IoU score (soft weighting instead of hard threshold).
_C.SELF_TRAINING.FINE_OBJECT.USE_IOU_WEIGHTING = True
# Hard-discard masks with IoU below this (SAM3 scores bounded ~0.7 max, so 0.10 keeps nearly all).
_C.SELF_TRAINING.FINE_OBJECT.MIN_HARD_IOU = 0.10
# Maps SAM3 stuff class index → semantic head channel (1-indexed; 0 = thing).
# Encoded as a list of [sam3_idx, head_channel] pairs (yacs does not support dict).
# E.g. [[4, 54], [13, 24]].  Empty list/tuple disables stuff supervision.
_C.SELF_TRAINING.FINE_OBJECT.STUFF_CHANNEL_MAP = ()
# ── Exp 1 (thing_mc_panda): MC-PanDA teacher gating ──────────────────────
_C.SELF_TRAINING.FINE_OBJECT.COMMON_THING_CHANNEL_INDICES = ()
_C.SELF_TRAINING.FINE_OBJECT.TEACHER_LOGIT_WEIGHT = 1.0
# ── Exp 2 (thing_focal_per_class_freq): Equalized focal gamma ────────────
_C.SELF_TRAINING.FINE_OBJECT.CLASS_FREQUENCIES_SAM3 = ()
_C.SELF_TRAINING.FINE_OBJECT.GAMMA_SCALE_FACTOR = 1.0
# ── Exp 3 (thing_focal_stuff_kd): Incrementer-style KD anchor ────────────
_C.SELF_TRAINING.FINE_OBJECT.STUFF_KD_LAMBDA = 0.1
_C.SELF_TRAINING.FINE_OBJECT.STUFF_CHANNEL_START = 1
# Whitelist of SAM3 class indices to load (empty = load all labels).
# Set to [5, 13] for stuff-only injection (traffic_light, pole) to prevent
# thing focal loss from firing on thing-class masks and regressing bicycle/rider.
_C.SELF_TRAINING.FINE_OBJECT.LOAD_CLASS_LABELS = ()

# Stage-4: Dead-Class Recovery (DCR). Disabled by default.
_C.STAGE4 = CfgNode()
_C.STAGE4.ENABLED = False
_C.STAGE4.RARE_CLASSES = ("guard rail", "tunnel", "polegroup", "caravan", "trailer")
# Optional explicit CUPS-space ids. If empty, ids are resolved by matching
# RARE_CLASSES against Cityscapes-27 names and the current thing/stuff split.
_C.STAGE4.RARE_STUFF_PSEUDO_CLASSES = ()
_C.STAGE4.RARE_THING_PSEUDO_CLASSES = ()
_C.STAGE4.TAU_RARE = 0.25
_C.STAGE4.TAU_COMMON = 0.70
_C.STAGE4.MIN_TTA_AGREEMENT = 0.55
_C.STAGE4.REPLAY_WEIGHT = 0.0
_C.STAGE4.FREEZE_BACKBONE = True
_C.STAGE4.FREEZE_RPN = True
_C.STAGE4.USE_EQLV2 = False
_C.STAGE4.EQLV2_GAMMA = 12.0
_C.STAGE4.EQLV2_MU = 0.8
_C.STAGE4.EQLV2_ALPHA = 4.0
_C.STAGE4.USE_SEESAW_LOSS = False
_C.STAGE4.SEESAW_P = 0.8
_C.STAGE4.SEESAW_Q = 2.0
_C.STAGE4.RARE_LOSS_WEIGHT = 1.0
_C.STAGE4.OHEM_ENABLED = False
_C.STAGE4.OHEM_FRACTION = 0.25
_C.STAGE4.OHEM_MIN_KEPT = 16
_C.STAGE4.OHEM_RARE_MIN_KEPT = 2
_C.STAGE4.RARE_RETAIN_ENABLED = False
_C.STAGE4.RARE_RETAIN_MIN_ROIS = 2
_C.STAGE4.RARE_RETAIN_RELAXED_IOU = 0.35
_C.STAGE4.RARE_FOCAL_WEIGHT = 0.0
_C.STAGE4.RARE_FOCAL_GAMMA = 2.0
_C.STAGE4.RARE_BOUNDARY_WEIGHT = 0.0
_C.STAGE4.RARE_BOUNDARY_WIDTH = 3
_C.STAGE4.BANDIT_ENABLED = False

# Validation specific config
_C.VALIDATION = CfgNode()
# Set if PQ metric should adhere to thing stuff split
_C.VALIDATION.ADHERE_THING_STUFF = True
# Set device to cache labels and prediction on
_C.VALIDATION.CACHE_DEVICE = None
# Set if TTA should be used in validation script
_C.VALIDATION.USE_TTA = False
# Set if CRF and center crop should be used in validation script
_C.VALIDATION.USE_CRF = False
# Smaller image side for center crop
_C.VALIDATION.SEMSEG_CENTER_CROP_SIZE = None

# Augmentation specific config
_C.AUGMENTATION = CfgNode()
# Set if copy-paste augmentation should be used
_C.AUGMENTATION.COPY_PASTE = True
# Set number of pasted objects per image
_C.AUGMENTATION.MAX_NUM_PASTED_OBJECTS = 8
# Set number of epochs copy-paste augmentation should be performed using pseudo labels (else pred. will be used)
_C.AUGMENTATION.NUM_STEPS_STARTUP = 500
# Set confidence for copy-paste predictions
_C.AUGMENTATION.CONFIDENCE = 0.75
# Set resolutions for resolution jitter augmentation
_C.AUGMENTATION.RESOLUTIONS = (
    (384, 768),
    (416, 832),
    (448, 896),
    (480, 960),
    (512, 1024),
    (544, 1088),
    (576, 1152),
    (608, 1216),
    (640, 1280),
    (672, 1344),
    (704, 1408),
)
# Persistent rare-instance pool copy-paste.
_C.AUGMENTATION.USE_RARE_POOL = False
_C.AUGMENTATION.RARE_POOL_PATH = ""
_C.AUGMENTATION.RARE_POOL_PASTES_PER_IMAGE = (1, 3)
_C.AUGMENTATION.RARE_POOL_USE_DEPTH_PLACEMENT = True
_C.AUGMENTATION.RARE_POOL_CLASS_REPEAT_OVERRIDES = ((11, 4), (12, 4), (18, 4), (14, 8), (15, 8), (16, 8))

# Pseudo label generation specific config
_C.PSEUDOS = CfgNode()
# Which type of semantic pseudo label generation
_C.PSEUDOS.SEMANTIC_TYPE = ["vanilla", "dguided"][-1]
# Which type of instance pseudo label generation
_C.PSEUDOS.INSTANCE_TYPE = ["vanilla_se3", "ours_se3"][-1]
# Which type of motion for pseudo label generation
_C.PSEUDOS.MOTION_TYPE = ["raft", "smurf"][-1]
# Align semantic class to object proposal
_C.PSEUDOS.NOT_ALIGN_SEMANTIC_TO_INSTANCE_MASK = False

# Label efficient learning specific config
_C.SUPERVISED = CfgNode()
# Set dataset to be used
_C.SUPERVISED.DATASET = "cityscapes_5_0"
# Align semantic class to object proposal
_C.SUPERVISED.ONLY_TRAIN_HEADS = True

# Mask refinement config (for self-training Stage-3)
_C.MASK_REFINEMENT = CfgNode()
# Enable mask refinement after pseudo-label generation
_C.MASK_REFINEMENT.ENABLE = False
# Morphological cleanup (opening + closing)
_C.MASK_REFINEMENT.MORPHOLOGICAL = True
# Guided filter (edge-aware smoothing)
_C.MASK_REFINEMENT.GUIDED_FILTER = True
# Guided filter radius
_C.MASK_REFINEMENT.GUIDED_FILTER_RADIUS = 8
# Guided filter regularization
_C.MASK_REFINEMENT.GUIDED_FILTER_EPS = 0.01
# Bilateral solver
_C.MASK_REFINEMENT.BILATERAL_SOLVER = False
# Minimum instance area (pixels) after refinement
_C.MASK_REFINEMENT.MIN_AREA = 100

__all__: Tuple[str, ...] = ("get_default_config",)


def get_default_config(
    experiment_config_file: Optional[str] = None,
    command_line_arguments: Optional[List[Any]] = None,
) -> CfgNode:
    """Loads config object.

    Args:
        experiment_config_file (Optional[str]): If given config file is used to overwrite the default config.
        command_line_arguments Optional[List[Any]]: Optional command line arguments, overwrites experiment config.

    Returns:
        config (CfgNode): Config object.
    """
    # Get default config
    config = _C.clone()
    # Load and merge experiment config from file
    if experiment_config_file is not None:
        assert isinstance(
            experiment_config_file, str
        ), f"Experiment config file must be a string but {type(experiment_config_file)} given."
        assert os.path.exists(experiment_config_file), f"File {experiment_config_file} does not exists."
        config.merge_from_file(experiment_config_file)
        log.info(f"Experiment config file {experiment_config_file} loaded.")
    if command_line_arguments is not None:
        assert isinstance(
            command_line_arguments, list
        ), f"Command line arguments must be a list [keys, values, ...] but type {type(command_line_arguments)} given."
        config.merge_from_list(command_line_arguments)
        log.info("Command line arguments loaded.")
    # Freeze config
    config.freeze()
    return config
