"""CUPS-protocol panoptic eval for DepthGuidedUNet (GT-free DINOv3 k27 training;
GT only at Hungarian matching). Reuses CUPS PanopticQualitySemanticMatching so
the number is directly comparable to CUPS PQ=27.8.

Usage:
  PYTHONPATH=refs/cups python mbps_pytorch/eval_cups_protocol.py \
    --checkpoint checkpoints/gtfree_k27/checkpoint_epoch_0008.pth \
    --cityscapes_root /Volumes/code_files/datasets/cityscapes \
    --stuff_things /Volumes/code_files/datasets/cityscapes/stuff_things_k27.json --device cpu
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from mbps_pytorch.train_refine_net import (PseudoLabelDataset, set_seed,
                                           _CS_ID_TO_TRAIN, _THING_IDS)
from mbps_pytorch.refine_net import DepthGuidedUNet

NUM_CLUSTERS = 27
FEATURE_DIM = 1024            # DINOv3 ViT-L/16
FEATURE_SUBDIR = "dinov3_features_vitl16"
DEPTH_SUBDIR = "depth_depthpro"
SEM_SUBDIR = "pseudo_semantic_raw_dinov3_k27_spherical_kmeans_vitl16"
H, W = 512, 1024


def build_pred_panoptic(sem_clusters: np.ndarray, thing_clusters: set,
                        min_area: int = 50) -> np.ndarray:
    """(H,W) cluster ids -> (H,W,2) [cluster_id, instance_id]."""
    pred = np.zeros((*sem_clusters.shape, 2), dtype=np.int32)
    pred[..., 0] = sem_clusters
    nid = 1
    for c in np.unique(sem_clusters):
        if int(c) not in thing_clusters:
            continue  # stuff keeps instance id 0
        lab, n = ndimage.label(sem_clusters == c)
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() >= min_area:
                pred[comp, 1] = nid
                nid += 1
    return pred


def build_gt_panoptic(city: str, stem: str, gt_dir: str) -> np.ndarray:
    """gtFine -> (H,W,2) [trainID(0..18/255), instance_id]."""
    sem_raw = np.array(Image.open(os.path.join(gt_dir, city, f"{stem}_gtFine_labelIds.png")))
    sem = np.full(sem_raw.shape, 255, dtype=np.int32)
    for rid, tid in _CS_ID_TO_TRAIN.items():
        sem[sem_raw == rid] = tid
    inst_raw = np.array(Image.open(os.path.join(gt_dir, city, f"{stem}_gtFine_instanceIds.png")))
    inst = np.zeros_like(sem, dtype=np.int32)
    nid = 1
    for uid in np.unique(inst_raw):
        if uid < 1000:
            continue
        if _CS_ID_TO_TRAIN.get(int(uid) // 1000, 255) in _THING_IDS:
            inst[inst_raw == uid] = nid
            nid += 1
    gt = np.stack([sem, inst], axis=-1)
    if gt.shape[:2] != (H, W):
        gt = np.stack([
            np.array(Image.fromarray(gt[..., 0].astype(np.int32)).resize((W, H), Image.NEAREST)),
            np.array(Image.fromarray(gt[..., 1].astype(np.int32)).resize((W, H), Image.NEAREST)),
        ], axis=-1)
    return gt


def run(args) -> None:
    from cups.metrics.panoptic_quality import PanopticQualitySemanticMatching

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))
    things = set(json.load(open(args.stuff_things))["thing_clusters"])

    model = DepthGuidedUNet(num_classes=NUM_CLUSTERS, feature_dim=FEATURE_DIM,
                            block_type="attention", num_decoder_stages=2,
                            num_bottleneck_blocks=2,
                            gradient_checkpointing=False).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)

    ds = PseudoLabelDataset(args.cityscapes_root, split="val",
                            semantic_subdir=SEM_SUBDIR, feature_subdir=FEATURE_SUBDIR,
                            depth_subdir=DEPTH_SUBDIR, num_classes=NUM_CLUSTERS,
                            target_h=128, target_w=256, return_depth_full=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    gt_dir = os.path.join(args.cityscapes_root, "gtFine", "val")

    for one_to_one in (True, False):
        metric = PanopticQualitySemanticMatching(
            things=set(range(11, 19)), stuffs=set(range(0, 11)),
            num_clusters=NUM_CLUSTERS, perform_one_to_one_matching=one_to_one)
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["dinov2_features"].to(device),  # legacy key name
                               batch["depth"].to(device),
                               batch["depth_grads"].to(device),
                               depth_full=batch["depth_full"].to(device))
                sem = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)        # (128,256)
                sem = np.array(Image.fromarray(sem).resize((W, H), Image.NEAREST)).astype(np.int32)
                pred = build_pred_panoptic(sem, things, min_area=args.min_area)
                gt = build_gt_panoptic(batch["city"][0], batch["stem"][0], gt_dir)
                metric.update(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])
        out = metric.compute()
        pq, sq, rq, pq_t, pq_s, miou = out[0], out[1], out[2], out[6], out[9], out[12]
        mode = "one-to-one" if one_to_one else "many-to-one (CUPS default)"
        print(f"\n=== CUPS protocol [{mode}] : {os.path.basename(args.checkpoint)} ===")
        for name, v in [("PQ", pq), ("PQ_things", pq_t), ("PQ_stuff", pq_s),
                        ("SQ", sq), ("RQ", rq), ("mIoU", miou)]:
            print(f"  {name:10s} = {float(v) * 100:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cityscapes_root", required=True)
    p.add_argument("--stuff_things", required=True)
    p.add_argument("--min_area", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    run(p.parse_args())


if __name__ == "__main__":
    main()
