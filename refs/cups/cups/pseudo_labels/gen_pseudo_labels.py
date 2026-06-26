import gc
import logging
import os
import sys
from argparse import ArgumentParser
from contextlib import nullcontext
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import pathlib

sys.path.append(os.getcwd())
import cups
import functools
from cups.crf import batched_crf
from cups.data import CityscapesStereoVideo, KITTIRaw
from cups.optical_flow import raft_smurf
import cups.scene_flow_2_se3 as _sf_mod
from cups.scene_flow_2_se3 import default_arguments as _sf_default_args
from cups.scene_flow_2_se3.halfres import sf2se3_halfres as get_object_proposals
from cups.semantics.model import DepthG
from cups.thingstuff_split import ThingStuffSplitter
from cups.utils import align_semantic_to_instance, normalize, set_seed_everywhere


# ---------------------------------------------------------------------------
# Production tuning (validated 2026-06-09 on M4; see notebooks/sf2se3_halfres_sanity.ipynb)
#   * L1 (loose inlier thresholds): +25% proposals per non-empty frame
#   * Relaxed NMS filter_thr 0.6 -> 0.3: more weak proposals survive
#   * min_object_size 0.65 -> 0.10: smaller objects pass the 3D-surface gate
# ---------------------------------------------------------------------------
_sf_default_args.sflow2se3_model_inlier_hard_threshold = 0.025      # was 0.0455
_sf_default_args.sflow2se3_se3filter_prob_gain_min = 0.0005          # was 0.0015
_sf_default_args.sflow2se3_se3filter_prob_same_mask_max = 0.30       # was 0.15
_sf_default_args.sflow2se3_min_object_size = 0.10                    # was 0.65

_ORIG_NMS = _sf_mod.mask_matrix_nms

@functools.wraps(_ORIG_NMS)
def _nms_filter_thr_03(masks, labels, scores, filter_thr=0.3, **kw):
    return _ORIG_NMS(masks, labels, scores, filter_thr=filter_thr, **kw)

_sf_mod.mask_matrix_nms = _nms_filter_thr_03

# SF2SE3 generation resolution: scale of the CityscapesStereoVideo 640x1280 frames.
# 0.5 -> 320x640 (the old half-res default — DROPS distant/small moving objects);
# 1.0 -> 640x1280 (CUPS-native — recovers motorcycle/person/rider, 6 thing classes
# vs 3, see reports/sf2se3_fullres_vs_halfres_confirmation_2026-06-14.md).
# Configurable via env so the default stays backward-compatible.
_SF2SE3_SCALE = float(os.environ.get("SF2SE3_SCALE", "0.5"))


def configure() -> Tuple[CfgNode, str]:
    """Function loads default config, experiment config, and parses command line arguments.

    Returns:
        config (CfgNode): Config object.
    """
    # Manage command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--SYSTEM.EXPERIMENT_NAME",
        default=None,
        type=str,
        help="Experiment name.",
    )
    parser.add_argument(
        "--DATA.PSEUDO_ROOT",
        default=None,
        type=str,
        help="Root for pseudo labels.",
    )
    parser.add_argument(
        "--DATA.DATASET",
        default=None,
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "--DATA.ROOT",
        default=None,
        type=str,
        help="Parent dataset root. Cityscapes is read from DATA.ROOT/Cityscapes.",
    )
    parser.add_argument(
        "--SYSTEM.NUM_WORKERS",
        default=None,
        type=int,
        help="Number of workers to be used.",
    )
    parser.add_argument(
        "--DATA.NUM_PREPROCESSING_SUBSPLITS",
        default=None,
        type=int,
        help="Number of dataset subsplits.",
    )
    parser.add_argument(
        "--DATA.PREPROCESSING_SUBSPLIT",
        default=None,
        type=int,
        help="ID of scpecific subsplit.",
    )
    parser.add_argument(
        "--MODEL.CHECKPOINT",
        default="/path_to_checkpoint/depthg.ckpt",
        type=str,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        type=str,
        help="Sets the visible cuda devices.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        choices=("cpu", "cuda", "cuda:0", "mps"),
        help="Torch device for pseudo-label generation. Defaults to cuda:0 when CUDA is available, "
             "mps on Apple Silicon, otherwise cpu.",
    )
    parser.add_argument(
        "--experiment_config_file",
        default=str(pathlib.Path(__file__).resolve().parent / "config_pseudo_labels.yaml"),
        type=str,
        help="Path to experiment config file.",
    )
    parser.add_argument(
        "--clip_sample",
        action="store_true",
        help=(
            "If set, subsample the dataset to keep one (t, t+1) pair per 30-frame clip "
            "(middle of each clip by default). Matches CUPS-official sampling rate; reduces ~83k pairs to ~3k."
        ),
    )
    parser.add_argument(
        "--clip_target_frame",
        default=None,
        type=int,
        help=(
            "When --clip_sample is on, pick the pair whose t-frame index equals this value. "
            "Cityscapes gtFine annotates frame 19 per clip; set to 19 to make the cache aligned with GT. "
            "Default: middle pair of each clip."
        ),
    )
    parser.add_argument(
        "--all_annotated",
        action="store_true",
        help=(
            "Bypass CityscapesStereoVideo's 30-frame-chunk enumeration and instead build one "
            "(t=anno, t+1=anno+1) pair per gtFine annotated frame. Recovers all 2975 train frames "
            "(vs 1885 reachable via the default chunking). Overrides --clip_sample."
        ),
    )
    # Get arguments
    args = parser.parse_args()
    # Arguments to dict
    args_dict: Dict[str, Any] = vars(args)
    # Set cuda devices
    if (cuda_visible_devices := args_dict.pop("cuda_visible_devices")) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    device = args_dict.pop("device")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda":
        device = "cuda:0"
    # Get path to experiment config file
    experiment_config_file = args_dict.pop("experiment_config_file")
    # Pop our non-yacs args before yacs config merge.
    clip_sample = bool(args_dict.pop("clip_sample", False))
    clip_target_frame = args_dict.pop("clip_target_frame", None)
    all_annotated = bool(args_dict.pop("all_annotated", False))
    # To list and remove all None entries from argument dict
    args_list: List[Union[str, Any]] = []
    for key, value in args_dict.items():
        if value is not None:
            args_list.extend((key, value))
    # Load config
    config: CfgNode = cups.get_default_config(
        experiment_config_file=experiment_config_file, command_line_arguments=args_list
    )
    return config, device, clip_sample, clip_target_frame, all_annotated


def _build_clip_sample_indices(
    sample_path: List[Dict[str, str]],
    target_frame: Union[int, None] = None,
) -> List[int]:
    """Group dataset entries by (city, clip_id) parsed from the left_0 filename
    and keep one representative pair per clip.

    Args:
        sample_path: list of {'left_0': ..., 'left_1': ..., ...} dicts.
        target_frame: if not None, pick the pair whose t-frame index equals this
            value (e.g. 19 to align with Cityscapes gtFine). Falls back to the
            middle pair of the clip when the target frame is not present.

    Cityscapes sequence filenames are `<city>_<clipid>_<frameid>_leftImg8bit.png`.
    With temporal_stride=1, every clip has 29 pairs in `sample_path`.
    """
    from collections import defaultdict
    clip_to_entries: Dict[Any, List[Any]] = defaultdict(list)
    for i, sp in enumerate(sample_path):
        stem = os.path.basename(sp["left_0"])
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        clip_key = (parts[0], parts[1])
        try:
            frame_idx = int(parts[2])
        except ValueError:
            frame_idx = -1
        clip_to_entries[clip_key].append((frame_idx, i))

    chosen: List[int] = []
    for clip_key, lst in clip_to_entries.items():
        if target_frame is not None:
            match = [i for fi, i in lst if fi == target_frame]
            if match:
                chosen.append(match[0])
                continue
        # Middle-of-clip fallback.
        lst_sorted = sorted(lst)
        chosen.append(lst_sorted[len(lst_sorted) // 2][1])
    return sorted(chosen)


def _build_all_annotated_sample_paths(cs_root: str, split: str = "train") -> List[Dict[str, str]]:
    """Enumerate one (t=anno, t+1=anno+1) stereo pair per gtFine annotated frame.

    Bypasses CityscapesStereoVideo's 30-frame-chunk slicing — which only catches
    1885 of 2975 train frames because the annotated frame's per-clip index varies.
    Falls back to (t=anno-1, t+1=anno) if anno+1 doesn't exist in the sequence
    (rare; lets the proposals stay wrt the annotated frame).
    """
    from pathlib import Path
    cs = Path(cs_root)
    gt_root = cs / "gtFine" / split
    seq_l = cs / "leftImg8bit_sequence" / split
    seq_r = cs / "rightImg8bit_sequence" / split
    cam_root = cs / "camera" / split

    out: List[Dict[str, str]] = []
    skipped = 0
    for gt_file in sorted(gt_root.glob("*/*_gtFine_labelIds.png")):
        city = gt_file.parent.name
        stem = gt_file.name.replace("_gtFine_labelIds.png", "")
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        clip_id, frame_str = parts[1], parts[2]
        try:
            f = int(frame_str)
        except ValueError:
            continue
        anno_l = seq_l / city / f"{stem}_leftImg8bit.png"
        anno_r = seq_r / city / f"{stem}_rightImg8bit.png"

        # Prefer forward pair (t=anno, t+1=anno+1).
        adj_f = f + 1
        adj_l = seq_l / city / f"{city}_{clip_id}_{adj_f:06d}_leftImg8bit.png"
        adj_r = seq_r / city / f"{city}_{clip_id}_{adj_f:06d}_rightImg8bit.png"
        if not (adj_l.is_file() and adj_r.is_file()):
            # Fall back to backward pair, but keep t=anno (so use t=anno-1 as image_0).
            adj_b = f - 1
            adj_l = seq_l / city / f"{city}_{clip_id}_{adj_b:06d}_leftImg8bit.png"
            adj_r = seq_r / city / f"{city}_{clip_id}_{adj_b:06d}_rightImg8bit.png"
            if not (adj_l.is_file() and adj_r.is_file()):
                skipped += 1
                continue
            # Swap so image_0 is still the annotated frame and image_1 is the back-adj.
            # (sf2se3 returns proposals wrt image_0 → we want annotated.)
            left_0, right_0 = anno_l, anno_r
            left_1, right_1 = adj_l, adj_r
        else:
            left_0, right_0 = anno_l, anno_r
            left_1, right_1 = adj_l, adj_r

        if not (anno_l.is_file() and anno_r.is_file()):
            skipped += 1
            continue

        # Calibration: exact frame match, else clip-level fallback.
        calib_path = cam_root / city / f"{stem}_camera.json"
        if not calib_path.is_file():
            cands = sorted((cam_root / city).glob(f"{city}_{clip_id}_*_camera.json"))
            if not cands:
                skipped += 1
                continue
            calib_path = cands[0]

        out.append({
            "left_0": str(left_0),
            "right_0": str(right_0),
            "left_1": str(left_1),
            "right_1": str(right_1),
            "calibration": str(calib_path),
        })
    return out, skipped


@torch.inference_mode()
def main() -> None:
    # get and log config
    config, device, clip_sample, clip_target_frame, all_annotated = configure()
    log.info(config)
    log.info("Using torch device: %s", device)
    log.info("clip_sample: %s  clip_target_frame: %s  all_annotated: %s",
             clip_sample, clip_target_frame, all_annotated)
    # set all seeds
    set_seed_everywhere(config.SYSTEM.SEED)
    # create directories
    pathlib.Path(config.DATA.PSEUDO_ROOT).mkdir(parents=True, exist_ok=True)

    # Init dataset
    if config.DATA.DATASET == "cityscapes":
        data = CityscapesStereoVideo(
            root=os.path.join(config.DATA.ROOT, "Cityscapes"),
            split="train",
        )
        img_shape = np.array([640, 1280])

    elif config.DATA.DATASET == "kitti":
        data = KITTIRaw(  # type: ignore
            root=os.path.join(config.DATA.ROOT, "KITTI-raw"),
            resize_scale=1.0,
            crop_resolution=(368, 1104),
        )
        img_shape = np.array([368, 1104])
    else:
        raise ValueError("Unknown dataset.")

    if all_annotated and config.DATA.DATASET == "cityscapes":
        cs_root = os.path.join(config.DATA.ROOT, "Cityscapes")
        new_paths, n_skipped = _build_all_annotated_sample_paths(cs_root, split="train")
        # data is the unwrapped CityscapesStereoVideo at this point.
        data.sample_path = new_paths  # type: ignore
        log.info(
            "ALL_ANNOTATED on: %d pairs enumerated directly from gtFine (skipped %d)",
            len(new_paths), n_skipped,
        )
    elif clip_sample and config.DATA.DATASET == "cityscapes":
        clip_indices = _build_clip_sample_indices(data.sample_path, target_frame=clip_target_frame)
        log.info(
            "CLIP_SAMPLE on: keeping %d pairs (1 per clip, target_frame=%s) out of %d stride=1 pairs",
            len(clip_indices), clip_target_frame, len(data),
        )
        data = Subset(data, clip_indices)  # type: ignore

    # split dataset to run generation in parallel
    splitsize = len(data) // config.DATA.NUM_PREPROCESSING_SUBSPLITS
    if len(data) % splitsize == 0:
        splitsize -= 1
    all_ranges = torch.arange(0, len(data), splitsize).int().tolist()
    all_ranges[-1] = len(data)
    data = Subset(  # type: ignore
        data,
        range(all_ranges[config.DATA.PREPROCESSING_SUBSPLIT - 1], all_ranges[config.DATA.PREPROCESSING_SUBSPLIT]),
    )
    print("Each subset has", len(data), "images.")

    pin_memory = device.startswith("cuda")
    data_loader_kwargs: Dict[str, Any] = {
        "collate_fn": lambda x: x[0],
        "num_workers": config.SYSTEM.NUM_WORKERS,
        "pin_memory": pin_memory,
        "shuffle": False,
        "batch_size": 1,
    }
    if pin_memory:
        data_loader_kwargs["pin_memory_device"] = device
    data_loader = DataLoader(data, **data_loader_kwargs)

    # semantic segmentation model
    model = DepthG(
        device=device,
        checkpoint_root=config.MODEL.CHECKPOINT,
        img_shape=img_shape,  # type: ignore
        stride=(int(img_shape[0] // 4), int(img_shape[0] // 4)),
        crop=(int(img_shape[0] // 2), int(img_shape[0] // 2)),
    )

    # thing stuff splitter
    thingstuff_split = ThingStuffSplitter(num_classes_all=model.model.cluster_probe.n_classes)

    # optical flow model
    raft = raft_smurf()
    raft.to(device, torch.float32)  # type: ignore
    raft.eval()

    # generate path
    pathlib.Path(config.DATA.PSEUDO_ROOT).mkdir(parents=True, exist_ok=True)

    failed_images = []
    pool_size = int(config.SYSTEM.NUM_WORKERS) * 2
    pool_context = nullcontext(None) if pool_size <= 0 else Pool(pool_size)
    with pool_context as pool:
        for data in tqdm(data_loader):
            # generate name and save path
            img_name = os.path.split(data["image_0_l_path"])[-1][:-4]  # type: ignore
            if config.DATA.DATASET == "kitti":
                img_name = data["image_0_l_path"].split(os.path.sep)[-4] + "_" + img_name  # type: ignore
            semgt_save_path = os.path.join(config.DATA.PSEUDO_ROOT, img_name + "_" + "semantic" + ".png")
            instgt_save_path = os.path.join(config.DATA.PSEUDO_ROOT, img_name + "_" + "instance" + ".png")
            # check if peudo label already exists
            if os.path.isfile(semgt_save_path) and os.path.isfile(instgt_save_path):
                tqdm.write("Already processed: " + str(img_name))
                # skip pseudo label generation but update thingstuff splitter
                sem_pseudo = T.ToTensor()(Image.open(semgt_save_path)).squeeze()
                inst_pseudo = T.ToTensor()(Image.open(instgt_save_path)).squeeze()
                panoptic_pred = torch.stack([sem_pseudo, inst_pseudo], dim=-1).long()
                thingstuff_split.update(panoptic_pred)
                continue

            # Get instance data
            image_0_l = data["image_0_l"].to(device, torch.float32)  # type: ignore
            image_0_r = data["image_0_r"].to(device, torch.float32)  # type: ignore
            image_1_l = data["image_1_l"].to(device, torch.float32)  # type: ignore
            image_1_r = data["image_1_r"].to(device, torch.float32)  # type: ignore
            valid_pixels = data["valid_pixels"].to(device)  # type: ignore
            baseline = data["baseline"].to(device, torch.float32)  # type: ignore
            intrinsics = data["intrinsics"].to(device, torch.float32)  # type: ignore

            # Make forward passes
            optical_flow_l_forward = raft(image_0_l, image_1_l)
            optical_flow_l_backward = raft(image_1_l, image_0_l)
            disparity_1_forward = raft(image_0_l, image_0_r, disparity=True)
            disparity_2_forward = raft(image_1_l, image_1_r, disparity=True)
            disparity_1_backward = raft(image_0_r, image_0_l, disparity=True, forward=False)
            disparity_2_backward = raft(image_1_r, image_1_l, disparity=True, forward=False)

            try:
                # Get SE(3) object proposals
                object_proposals = get_object_proposals(
                    image_1_l=image_1_l,
                    optical_flow_l_forward=optical_flow_l_forward,
                    optical_flow_l_backward=optical_flow_l_backward,
                    disparity_1_forward=disparity_1_forward,
                    disparity_2_forward=disparity_2_forward,
                    disparity_1_backward=disparity_1_backward,
                    disparity_2_backward=disparity_2_backward,
                    intrinsics=intrinsics,
                    baseline=baseline,
                    valid_pixels=valid_pixels,
                    scale=_SF2SE3_SCALE,
                )

            except Exception:
                print(Exception)
                failed_images.append(img_name)
                tqdm.write("Failed for: " + str(img_name))
                object_proposals = torch.zeros(image_0_l.shape[-2], image_0_l.shape[-1]).long()

            img = normalize(image_0_l)
            disp = disparity_1_forward
            fB = data["intrinsics"][0, 0, 0] * data["baseline"][0]  # type: ignore
            depth = fB / (disp.abs() + 1e-10) * disp.sign()
            depth_weight = 1 / (depth + 1)
            out = model.depth_guided_sliding_window(img, depth_weight)

            # apply CRF to semantic segmentation
            cluster_pred = batched_crf(pool, img, out).argmax(1).long()

            # skip if no object proposals
            if object_proposals.max().item() == 0:
                object_proposals = torch.zeros(image_0_l.shape[-2], image_0_l.shape[-1]).long()
            object_proposals[:, :32] = 0
            object_proposals[:, -32:] = 0

            # merge predictions
            panoptic_pred = torch.stack([cluster_pred.squeeze(), object_proposals.cpu()], dim=-1)
            # align semantic class to object proposal
            panoptic_pred[..., 0] = align_semantic_to_instance(
                panoptic_pred[..., 0], panoptic_pred[..., 1].unsqueeze(0)
            )["aligned_semantics"]
            # update thingstuff splitter
            thingstuff_split.update(panoptic_pred)

            # safe pseudo labels and images
            semantic_label = Image.fromarray(np.array(panoptic_pred[..., 0].cpu(), dtype=np.uint8))
            instance_label = Image.fromarray(np.array(panoptic_pred[..., 1].cpu(), dtype=np.uint8))

            # write image and label files
            semantic_label.save(semgt_save_path)
            instance_label.save(instgt_save_path)

            # Per-frame memory hygiene — required for full-res (scale=1.0) on M4 MPS,
            # which otherwise OOM-SIGKILLs after ~7 frames (no traceback). See
            # reports/sf2se3_fullres_vs_halfres_confirmation_2026-06-14.md.
            del object_proposals, cluster_pred, panoptic_pred, out, depth, depth_weight
            del optical_flow_l_forward, optical_flow_l_backward
            del disparity_1_forward, disparity_2_forward, disparity_1_backward, disparity_2_backward
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    instances_distribution_pixel, instances_distribution_mask, pseudo_class_distribution = thingstuff_split.compute()
    save_data = {
        "distribution all pixels": pseudo_class_distribution,
        "distribution inside object proposals": instances_distribution_pixel,
        "distribution per object proposal": instances_distribution_mask,
    }
    torch.save(
        save_data,
        os.path.join(
            config.DATA.PSEUDO_ROOT, "pseudo_classes_split_" + str(config.DATA.PREPROCESSING_SUBSPLIT) + ".pt"
        ),
    )

    print("Failed images:", failed_images)


if __name__ == "__main__":
    main()
