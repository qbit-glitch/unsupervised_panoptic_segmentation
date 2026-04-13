"""Generate CUPS pseudo-labels on remote GPU using the original pipeline.

Two-pass approach to fit on 11GB GPU (1080 Ti):
  Pass 1: RAFT-SMURF + SF2SE3 → instance proposals (saved as .pt)
  Pass 2: DepthG + CRF → semantics, merged with instances

Handles missing rightImg8bit_sequence by reusing the static right key frame
as an approximation for the temporal right frame (~60ms gap at 17Hz).

Only generates pseudo-labels for the 2,975 Cityscapes training key frames.

Usage (on remote GPU machine):
    cd /home/santosh/mbps_panoptic_segmentation/refs/cups
    conda activate cups
    python ../../mbps_pytorch/gen_cups_pseudo_labels_remote.py \
        --data_root /home/santosh/datasets/cityscapes \
        --depthg_ckpt /home/santosh/datasets/cityscapes/depthg.ckpt \
        --output_dir /home/santosh/datasets/cityscapes/cups_pseudo_labels_official \
        --gpu 0
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from glob import glob
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Add CUPS and DepthG to path (DepthG/src MUST come before script dir
# to prevent mbps_pytorch/data/ from shadowing external/depthg/src/data/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CUPS_ROOT = os.path.join(_PROJECT_ROOT, "refs", "cups")
_DEPTHG_ROOT = os.path.join(_CUPS_ROOT, "external", "depthg")
_DEPTHG_SRC = os.path.join(_DEPTHG_ROOT, "src")
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
for _p in [_DEPTHG_SRC, _DEPTHG_ROOT, _CUPS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(_SCRIPT_DIR)

from cups.crf import dense_crf
from cups.data.utils import CITYSCAPES_TRAINING_FILES, load_image
from cups.optical_flow import raft_smurf
from cups.scene_flow_2_se3 import sf2se3 as get_object_proposals
from cups.semantics.model import DepthG
from cups.thingstuff_split import ThingStuffSplitter
from cups.utils import align_semantic_to_instance, normalize

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CUPS pseudo-labels (remote GPU)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--depthg_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize_scale", type=float, default=0.625)
    return parser.parse_args()


def read_calibration_file(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read Cityscapes camera calibration JSON."""
    with open(path, "r") as f:
        calibration = json.load(f)
    baseline = torch.tensor(calibration["extrinsic"]["baseline"])
    intrinsics = torch.tensor(
        [
            [calibration["intrinsic"]["fx"], 0.0, calibration["intrinsic"]["u0"]],
            [0.0, calibration["intrinsic"]["fy"], calibration["intrinsic"]["v0"]],
            [0.0, 0.0, 1.0],
        ]
    )
    return baseline, intrinsics


def find_temporal_neighbor(
    seq_dir: str, city: str, clip: str, frame: int, stride: int = 1
) -> str:
    """Find the next sequential frame for optical flow computation."""
    next_frame = frame + stride
    candidate = os.path.join(
        seq_dir, city, f"{city}_{clip}_{next_frame:06d}_leftImg8bit.png"
    )
    if os.path.isfile(candidate):
        return candidate
    prev_frame = frame - stride
    candidate = os.path.join(
        seq_dir, city, f"{city}_{clip}_{prev_frame:06d}_leftImg8bit.png"
    )
    if os.path.isfile(candidate):
        return candidate
    logger.warning("No temporal neighbor for %s_%s_%06d, using self", city, clip, frame)
    return os.path.join(seq_dir, city, f"{city}_{clip}_{frame:06d}_leftImg8bit.png")


def find_calibration(cam_dir: str, city: str, clip: str, frame: int) -> str:
    """Find camera calibration file for this clip."""
    cam_file = os.path.join(
        cam_dir, city, f"{city}_{clip}_{frame:06d}_camera.json"
    )
    if os.path.isfile(cam_file):
        return cam_file
    pattern = os.path.join(cam_dir, city, f"{city}_{clip}_*_camera.json")
    candidates = sorted(glob(pattern))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No calibration for {city}_{clip}")


@torch.inference_mode()
def pass1_instances(args: argparse.Namespace, training_files: list) -> None:
    """Pass 1: RAFT-SMURF + SF2SE3 → instance proposals + disparity.

    Saves per-image .pt files with instance proposals and disparity.
    Only RAFT on GPU (~1GB peak). Fits easily on 11GB.
    """
    device = f"cuda:{args.gpu}"
    scale = args.resize_scale
    instance_dir = os.path.join(args.output_dir, "_instances")
    pathlib.Path(instance_dir).mkdir(parents=True, exist_ok=True)

    left_seq_dir = os.path.join(args.data_root, "leftImg8bit_sequence", "train")
    right_static_dir = os.path.join(args.data_root, "rightImg8bit", "train")
    cam_dir = os.path.join(args.data_root, "camera", "train")

    logger.info("Pass 1: Loading RAFT-SMURF...")
    raft = raft_smurf()
    raft.to(device, torch.float32)
    raft.eval()
    logger.info("RAFT GPU mem: %.0f MB", torch.cuda.memory_allocated(args.gpu) / 1e6)

    failed = []
    skipped = 0

    for img_id in tqdm(training_files, desc="Pass 1: instances"):
        pt_path = os.path.join(instance_dir, f"{img_id}.pt")
        if args.resume and os.path.isfile(pt_path):
            skipped += 1
            continue

        parts = img_id.split("_")
        city, clip, frame = parts[0], parts[1], int(parts[2])

        try:
            # Load images
            left_0_path = os.path.join(
                left_seq_dir, city, f"{img_id}_leftImg8bit.png"
            )
            if not os.path.isfile(left_0_path):
                left_0_path = os.path.join(
                    args.data_root, "leftImg8bit", "train", city,
                    f"{img_id}_leftImg8bit.png"
                )
            right_0_path = os.path.join(
                right_static_dir, city, f"{img_id}_rightImg8bit.png"
            )
            left_1_path = find_temporal_neighbor(left_seq_dir, city, clip, frame)
            right_1_path = right_0_path

            image_0_l = F.interpolate(load_image(left_0_path)[None], scale_factor=scale, mode="bilinear")
            image_0_r = F.interpolate(load_image(right_0_path)[None], scale_factor=scale, mode="bilinear")
            image_1_l = F.interpolate(load_image(left_1_path)[None], scale_factor=scale, mode="bilinear")
            image_1_r = F.interpolate(load_image(right_1_path)[None], scale_factor=scale, mode="bilinear")

            # Valid pixels mask (at original resolution, then scale once)
            orig_h, orig_w = 1024, 2048
            valid_pixels = torch.ones(1, 1, orig_h, orig_w)
            valid_pixels[..., :96] = 0.0
            valid_pixels[..., -96:] = 0.0
            valid_pixels[:, :, :round(orig_h * 0.05)] = 0.0
            valid_pixels[:, :, -round(orig_h * 0.2):] = 0.0
            valid_pixels = F.interpolate(valid_pixels, scale_factor=scale, mode="nearest")

            # Calibration
            cam_path = find_calibration(cam_dir, city, clip, frame)
            baseline, intrinsics = read_calibration_file(cam_path)
            intrinsics[0, 0] *= scale
            intrinsics[0, 2] *= scale
            intrinsics[1, 1] *= scale
            intrinsics[1, 2] *= scale

            # Move to GPU
            image_0_l = image_0_l.to(device, torch.float32)
            image_0_r = image_0_r.to(device, torch.float32)
            image_1_l = image_1_l.to(device, torch.float32)
            image_1_r = image_1_r.to(device, torch.float32)
            valid_pixels = valid_pixels.to(device)
            baseline = baseline.reshape(1).to(device, torch.float32)
            intrinsics = intrinsics.reshape(1, 3, 3).to(device, torch.float32)

            # RAFT-SMURF forward passes
            of_fwd = raft(image_0_l, image_1_l)
            of_bwd = raft(image_1_l, image_0_l)
            d1_fwd = raft(image_0_l, image_0_r, disparity=True)
            d1_bwd = raft(image_0_r, image_0_l, disparity=True, forward=False)
            d2_fwd = raft(image_1_l, image_1_r, disparity=True)
            d2_bwd = raft(image_1_r, image_1_l, disparity=True, forward=False)

            # SF2SE3: instance proposals
            try:
                obj_proposals = get_object_proposals(
                    image_1_l=image_1_l,
                    optical_flow_l_forward=of_fwd,
                    optical_flow_l_backward=of_bwd,
                    disparity_1_forward=d1_fwd,
                    disparity_2_forward=d2_fwd,
                    disparity_1_backward=d1_bwd,
                    disparity_2_backward=d2_bwd,
                    intrinsics=intrinsics,
                    baseline=baseline,
                    valid_pixels=valid_pixels,
                )
            except Exception as e:
                tqdm.write(f"SF2SE3 failed for {img_id}: {e}")
                obj_proposals = torch.zeros(
                    image_0_l.shape[-2], image_0_l.shape[-1]
                ).long()

            # Save instance proposals + disparity (CPU) for Pass 2
            torch.save({
                "instances": obj_proposals.cpu(),
                "disparity": d1_fwd.cpu(),
                "baseline": baseline.cpu(),
                "intrinsics": intrinsics.cpu(),
            }, pt_path)

        except Exception as e:
            failed.append(img_id)
            tqdm.write(f"Pass 1 failed for {img_id}: {e}")
            # Save empty data
            h, w = int(1024 * scale), int(2048 * scale)
            torch.save({
                "instances": torch.zeros(h, w).long(),
                "disparity": torch.zeros(1, 2, h, w),
                "baseline": torch.zeros(1),
                "intrinsics": torch.zeros(1, 3, 3),
            }, pt_path)

        # Free GPU memory between images
        torch.cuda.empty_cache()

    # Free RAFT completely
    del raft
    torch.cuda.empty_cache()
    logger.info(
        "Pass 1 done: %d processed, %d skipped, %d failed",
        len(training_files) - skipped - len(failed), skipped, len(failed),
    )


def _slide_segment_batched(depthg: "DepthG", img: torch.Tensor, batch_size: int = 4):
    """Memory-efficient slide_segment: process crops in small batches."""
    from einops import rearrange

    unfolded = F.unfold(img, depthg.crop, stride=depthg.stride,
                        padding=(depthg.bottom_pad, depthg.right_pad))
    unfolded = rearrange(unfolded, "B (C H W) N -> (B N) C H W",
                         H=depthg.crop[0], W=depthg.crop[1])
    n_crops = unfolded.shape[0]
    all_logits = []
    for i in range(0, n_crops, batch_size):
        chunk = unfolded[i:i + batch_size]
        with torch.cuda.amp.autocast():
            _, code = depthg.model.net(chunk)
            code = F.interpolate(code, depthg.crop, mode="bilinear", align_corners=False)
            logits = depthg.model.cluster_probe(code, 2, log_probs=True)
        all_logits.append(logits.float())
    crop_seg_logits = torch.cat(all_logits, dim=0)
    c = crop_seg_logits.size(1)
    crop_seg_logits = rearrange(crop_seg_logits, "(B N) C H W -> B (C H W) N", B=img.size(0))
    preds = F.fold(crop_seg_logits, (img.size(-2), img.size(-1)),
                   depthg.crop, stride=depthg.stride,
                   padding=(depthg.bottom_pad, depthg.right_pad))
    count_mat = F.fold(
        torch.ones(crop_seg_logits.size(0), crop_seg_logits.size(1) // c,
                    crop_seg_logits.size(2), device=img.device),
        (img.size(-2), img.size(-1)), depthg.crop, stride=depthg.stride,
        padding=(depthg.bottom_pad, depthg.right_pad))
    return preds / count_mat


def _depth_guided_sliding_window_lowmem(depthg: "DepthG", img, depth_weight):
    """Memory-efficient version of depth_guided_sliding_window."""
    out_slidingw = _slide_segment_batched(depthg, img, batch_size=4)
    img_small = F.interpolate(
        img, (img.shape[-2] // 2, img.shape[-1] // 2),
        mode="bilinear", align_corners=False,
    ).float()
    with torch.cuda.amp.autocast():
        code = depthg.model.net(img_small)[-1]
        code2 = depthg.model.net(img_small.flip(dims=[3]))[-1]
    code = ((code + code2.flip(dims=[3])) / 2).float()
    code = F.interpolate(code, (img.shape[-2], img.shape[-1]),
                         mode="bilinear", align_corners=False)
    out_singleimg = depthg.model.cluster_probe(code, 2, log_probs=True)
    weight = depth_weight.expand_as(out_slidingw)
    return out_singleimg * weight + out_slidingw * (1 - weight)


@torch.inference_mode()
def pass2_semantics(args: argparse.Namespace, training_files: list) -> None:
    """Pass 2: DepthG + CRF → semantics, merge with saved instances.

    Uses autocast + batched sliding window to fit on 11GB GPU.
    No multiprocessing — CRF runs sequentially (1 image at a time).
    """
    device = f"cuda:{args.gpu}"
    scale = args.resize_scale
    img_shape = np.array([640, 1280])
    instance_dir = os.path.join(args.output_dir, "_instances")

    left_seq_dir = os.path.join(args.data_root, "leftImg8bit_sequence", "train")

    logger.info("Pass 2: Loading DepthG from %s...", args.depthg_ckpt)
    model = DepthG(
        device=device,
        checkpoint_root=args.depthg_ckpt,
        img_shape=img_shape,
        stride=(int(img_shape[0] // 4), int(img_shape[0] // 4)),
        crop=(int(img_shape[0] // 2), int(img_shape[0] // 2)),
    )
    torch.cuda.empty_cache()
    logger.info("DepthG GPU mem: %.0f MB", torch.cuda.memory_allocated(args.gpu) / 1e6)

    thingstuff_split = ThingStuffSplitter(
        num_classes_all=model.model.cluster_probe.n_classes
    )

    failed = []
    skipped = 0
    processed = 0

    for img_id in tqdm(training_files, desc="Pass 2: semantics"):
        img_name = f"{img_id}_leftImg8bit"
        sem_path = os.path.join(args.output_dir, f"{img_name}_semantic.png")
        inst_path = os.path.join(args.output_dir, f"{img_name}_instance.png")

        # Resume support
        if args.resume and os.path.isfile(sem_path) and os.path.isfile(inst_path):
            sem_pseudo = T.ToTensor()(Image.open(sem_path)).squeeze()
            inst_pseudo = T.ToTensor()(Image.open(inst_path)).squeeze()
            panoptic_pred = torch.stack([sem_pseudo, inst_pseudo], dim=-1).long()
            thingstuff_split.update(panoptic_pred)
            skipped += 1
            continue

        parts = img_id.split("_")
        city = parts[0]

        try:
            # Load Pass 1 results
            pt_path = os.path.join(instance_dir, f"{img_id}.pt")
            p1_data = torch.load(pt_path, map_location="cpu", weights_only=True)
            object_proposals = p1_data["instances"]
            disparity = p1_data["disparity"].to(device)
            baseline_val = p1_data["baseline"].to(device)
            intrinsics_val = p1_data["intrinsics"].to(device)

            # Load left image
            left_0_path = os.path.join(
                left_seq_dir, city, f"{img_id}_leftImg8bit.png"
            )
            if not os.path.isfile(left_0_path):
                left_0_path = os.path.join(
                    args.data_root, "leftImg8bit", "train", city,
                    f"{img_id}_leftImg8bit.png"
                )
            image_0_l = F.interpolate(
                load_image(left_0_path)[None], scale_factor=scale, mode="bilinear"
            ).to(device, torch.float32)

            # DepthG: semantic segmentation (low-memory batched + autocast)
            img = normalize(image_0_l)
            fB = intrinsics_val[0, 0, 0] * baseline_val[0]
            # Disparity is [1, 2, H, W] from RAFT — take horizontal component only
            disp = disparity[:, :1, :, :]
            depth = fB / (disp.abs() + 1e-10) * disp.sign()
            depth_weight = 1 / (depth + 1)
            out = _depth_guided_sliding_window_lowmem(model, img, depth_weight)

            # CRF (CPU, sequential — no multiprocessing to avoid CUDA+spawn hang)
            crf_result = dense_crf(img[0].detach().cpu(), out[0].detach().cpu())
            cluster_pred = torch.from_numpy(crf_result).unsqueeze(0).argmax(1).long()

            # Clean instance proposals
            object_proposals[:, :32] = 0
            object_proposals[:, -32:] = 0

            # Merge predictions
            panoptic_pred = torch.stack(
                [cluster_pred.squeeze(), object_proposals], dim=-1
            )
            panoptic_pred[..., 0] = align_semantic_to_instance(
                panoptic_pred[..., 0], panoptic_pred[..., 1].unsqueeze(0)
            )["aligned_semantics"]

            # Update stats
            thingstuff_split.update(panoptic_pred)

            # Save
            Image.fromarray(
                np.array(panoptic_pred[..., 0].cpu(), dtype=np.uint8)
            ).save(sem_path)
            Image.fromarray(
                np.array(panoptic_pred[..., 1].cpu(), dtype=np.uint8)
            ).save(inst_path)
            processed += 1

        except Exception as e:
            failed.append(img_id)
            tqdm.write(f"Pass 2 failed for {img_id}: {e}")
            Image.fromarray(np.zeros((1024, 2048), dtype=np.uint8)).save(sem_path)
            Image.fromarray(np.zeros((1024, 2048), dtype=np.uint8)).save(inst_path)

        # Free GPU between images
        torch.cuda.empty_cache()

        if (processed + skipped) % 100 == 0 and processed > 0:
            logger.info(
                "Progress: %d processed, %d skipped, %d failed",
                processed, skipped, len(failed),
            )

    # Save thing/stuff split statistics
    (
        instances_distribution_pixel,
        instances_distribution_mask,
        pseudo_class_distribution,
    ) = thingstuff_split.compute()
    save_data = {
        "distribution all pixels": pseudo_class_distribution,
        "distribution inside object proposals": instances_distribution_pixel,
        "distribution per object proposal": instances_distribution_mask,
    }
    split_path = os.path.join(args.output_dir, "pseudo_classes_split_1.pt")
    torch.save(save_data, split_path)
    logger.info("Saved thing/stuff split to %s", split_path)

    logger.info(
        "Pass 2 done: %d processed, %d skipped, %d failed / %d total",
        processed, skipped, len(failed), len(training_files),
    )
    if failed:
        logger.info("Failed images: %s", failed[:20])


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Verify directories exist
    for d, name in [
        (os.path.join(args.data_root, "leftImg8bit_sequence", "train"), "leftImg8bit_sequence/train"),
        (os.path.join(args.data_root, "rightImg8bit", "train"), "rightImg8bit/train"),
        (os.path.join(args.data_root, "camera", "train"), "camera/train"),
    ]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing: {d}")

    training_files = sorted(CITYSCAPES_TRAINING_FILES)
    logger.info("Processing %d key frames (two-pass pipeline)", len(training_files))

    # Pass 1: RAFT + SF2SE3 → instances (only RAFT on GPU)
    pass1_instances(args, training_files)

    # Pass 2: DepthG + CRF → semantics (only DepthG on GPU)
    pass2_semantics(args, training_files)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
